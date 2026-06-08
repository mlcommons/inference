# Copyright 2025 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
System Under Test (SUT) for E2E DocGrader workload.
Wraps multi_shot_retrieval logic and processes loadgen queries.
"""

import os
import sys
import array
import time
import json
import logging
import threading
import queue
from datetime import datetime
from typing import Dict, Any

import mlperf_loadgen as lg

from QSL import E2EQSLInMemory
from multi_shot_retrieval import multi_shot_retrieval
from retrieve import VectorDB, BM25DB
from utils import setup_llm_config, get_device_config
from llm_logger import LLMLogger

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("E2ESUT")


class E2ESUT:
    """
    System Under Test for E2E DocGrader workload.
    Handles loadgen queries and runs multi-shot retrieval.
    """

    def __init__(
        self,
        dataset_path: str,
        db_path: str,
        retrieval_method: str = "vector",
        max_sub_queries: int = 3,
        top_k_retriever: int = 10,
        top_k_reranking: int = 10,
        max_iterations: int = 10,
        no_rerank: bool = False,
        retrieval_strategy: str = "fixed_k",
        reasoning_effort: str = "medium",
        perf_count: int = None,
        device: str = "auto",
        temperature: float = 1.0,
        max_retries: int = 5,
        output_dir: str = ".",
        args: Any = None,  # Full args object for additional params
    ):
        """
        Initialize SUT with RAG database and configuration.

        Args:
            dataset_path: Path to frames_dataset.tsv
            db_path: Path to database file
            retrieval_method: 'vector' or 'bm25'
            max_sub_queries: Max sub-queries per iteration
            top_k_retriever: Number of docs to retrieve per query
            top_k_reranking: Number of docs after reranking
            max_iterations: Max retrieval iterations
            no_rerank: Skip reranking
            retrieval_strategy: Retrieval strategy
            reasoning_effort: LLM reasoning level
            perf_count: Number of queries for performance testing
            device: Device to use
            temperature: LLM temperature
            max_retries: Max LLM call retries
            output_dir: Output directory for logs
            args: Full args namespace for additional parameters
        """
        self.dataset_path = dataset_path
        self.db_path = db_path
        self.retrieval_method = retrieval_method
        self.max_sub_queries = max_sub_queries
        self.top_k_retriever = top_k_retriever
        self.top_k_reranking = top_k_reranking
        self.max_iterations = max_iterations
        self.no_rerank = no_rerank
        self.retrieval_strategy = retrieval_strategy
        self.reasoning_effort = reasoning_effort
        self.perf_count = perf_count
        self.device = device
        self.temperature = temperature
        self.max_retries = max_retries
        self.output_dir = output_dir
        self.args = args

        # Setup LLM configuration
        self.llm_config = setup_llm_config(args)
        self.llm_config['temperature'] = temperature
        self.llm_config['max_retries'] = max_retries

        # Performance test mode
        if hasattr(args, 'perf_test_mode') and args.perf_test_mode:
            from perf_test_cache import PerfTestCache
            log.info(f"Loading performance test cache from {args.perf_test_mode}")
            self.llm_config['perf_test_cache'] = PerfTestCache(args.perf_test_mode)
        else:
            self.llm_config['perf_test_cache'] = None

        log.info(f"LLM Config: {self.llm_config}")

        # Setup device
        device_config = get_device_config()
        log.info(f"Device Config: {device_config}")

        # Initialize QSL
        log.info("Initializing Query Sample Library...")
        self.qsl = E2EQSLInMemory(dataset_path, perf_count)

        # Initialize database
        log.info("Initializing RAG database...")
        if retrieval_method == "bm25":
            db_class = BM25DB
        else:
            db_class = VectorDB

        # Extract database parameters from args
        self.rag_db = db_class(
            retriever_model=args.retriever_model if hasattr(args, 'retriever_model') else 'BAAI/bge-base-en-v1.5',
            reranker_model=args.reranker_model if hasattr(args, 'reranker_model') else 'BAAI/bge-reranker-base',
            device=device,
            k1=args.bm25_k1 if hasattr(args, 'bm25_k1') else 1.5,
            b=args.bm25_b if hasattr(args, 'bm25_b') else 0.75,
            method=args.bm25_method if hasattr(args, 'bm25_method') else 'lucene',
            database=db_path.replace('.db', ''),
            vector_index_method=args.vector_index_method if hasattr(args, 'vector_index_method') else 'hnsw',
            num_embedding_devices=args.num_embedding_devices if hasattr(args, 'num_embedding_devices') else 1,
            benchmark=args.benchmark if hasattr(args, 'benchmark') else False
        )

        # Load database
        if os.path.exists(db_path):
            log.info(f"Loading database from {db_path}")
            self.rag_db.from_serialized(db_path)
        else:
            raise ValueError(f"Database not found: {db_path}")

        # Build strategy parameters
        self.strategy_params = {}
        if hasattr(args, 'max_results'):
            self.strategy_params["max_results"] = args.max_results
        if retrieval_strategy == "top_p" and hasattr(args, 'top_p'):
            self.strategy_params["p"] = args.top_p
        elif retrieval_strategy == "relative" and hasattr(args, 'relative_ratio'):
            self.strategy_params["ratio"] = args.relative_ratio

        # Initialize LLM logger
        os.makedirs(output_dir, exist_ok=True)
        experiment_start_time = datetime.now()
        log_filename = os.path.join(
            output_dir,
            f"llm_logs_loadgen_{experiment_start_time.strftime('%Y%m%d_%H%M%S')}.json"
        )

        # Determine chunk size from database name
        chunk_size = 768  # default
        db_base_name = os.path.basename(db_path).replace('.db', '')
        if 'len' in db_base_name:
            import re
            match = re.search(r'len(\d+)', db_base_name)
            if match:
                chunk_size = int(match.group(1))

        self.llm_logger = LLMLogger(
            output_file=log_filename,
            experiment_metadata={
                "experiment_name": f"loadgen_{db_base_name}",
                "timestamp_start": experiment_start_time.isoformat(),
                "retrieval_method": retrieval_method,
                "retrieval_mode": "multi_shot",
                "max_iterations": max_iterations,
                "max_sub_queries": max_sub_queries,
                "top_k_retriever": top_k_retriever,
                "chunk_size": chunk_size,
                "device": device,
                "grader_model": self.llm_config.get('grader_model_name', 'unknown'),
                "sufficiency_checker_model": self.llm_config.get('sufficiency_model_name', 'unknown'),
                "query_model": self.llm_config.get('query_model_name', 'unknown'),
            }
        )
        log.info(f"LLM logs will be written to: {log_filename}")

        # Results storage
        self.results = {}
        self.results_lock = threading.Lock()

        # Construct loadgen SUT
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        log.info("SUT initialized successfully")

    def issue_queries(self, query_samples):
        """
        Issue queries to the SUT.
        Called by loadgen for each query batch.
        """
        for query_sample in query_samples:
            sample_id = query_sample.index
            query_id = query_sample.id

            # Get query data from QSL
            sample = self.qsl[sample_id]
            query_text = sample['query']
            expected_urls = sample['expected_urls']
            ground_truth_answer = sample['ground_truth']

            log.info(f"Processing query {sample_id} (QID: {query_id})")

            # Start logging for this query
            # Note: Use sample_id for perf_test_cache consistency (stable across runs)
            # Use query_id for loadgen tracking (unique per run)
            cache_key = str(sample_id)  # Stable dataset index for cache lookup
            self.llm_logger.start_query(cache_key, query_text)

            try:
                # Run multi-shot retrieval
                result = multi_shot_retrieval(
                    self.rag_db,
                    query_text,
                    expected_urls=expected_urls,
                    expected_answer=ground_truth_answer,
                    max_sub_queries=self.max_sub_queries,
                    top_k_retriever=self.top_k_retriever,
                    top_k_reranking=self.top_k_reranking,
                    max_iterations=self.max_iterations,
                    no_rerank=self.no_rerank,
                    retrieval_strategy=self.retrieval_strategy,
                    verbose=False,  # Reduce verbosity for loadgen
                    reasoning_effort=self.reasoning_effort,
                    llm_config=self.llm_config,
                    logger=self.llm_logger,
                    query_id=cache_key,  # Use stable sample_id for cache consistency
                    **self.strategy_params
                )

                # Extract answer
                answer = result.get('llm_answer', 'Unknown')

                # End logging for this query
                self.llm_logger.end_query(
                    retrieval_results={
                        "retrieved_urls": result.get('retrieved_urls', []),
                        "correct_urls": expected_urls,
                        "precision": result.get('precision', 0),
                        "recall": result.get('recall', 0),
                        "f1": result.get('f1', 0)
                    },
                    answer_results={
                        "llm_answer": answer,
                        "ground_truth_answer": ground_truth_answer
                    }
                )

                # Store result
                with self.results_lock:
                    self.results[query_id] = {
                        'query': query_text,
                        'answer': answer,
                        'ground_truth': ground_truth_answer,
                        'retrieved_urls': result.get('retrieved_urls', []),
                        'expected_urls': expected_urls,
                        'metrics': {
                            'precision': result.get('precision', 0),
                            'recall': result.get('recall', 0),
                            'f1': result.get('f1', 0)
                        }
                    }

            except Exception as e:
                log.error(f"Error processing query {sample_id}: {e}")
                import traceback
                traceback.print_exc()
                answer = "Error"

                # Store error result
                with self.results_lock:
                    self.results[query_id] = {
                        'query': query_text,
                        'answer': answer,
                        'error': str(e)
                    }

            # Convert answer to byte array for loadgen
            answer_bytes = answer.encode('utf-8')
            response_array = array.array('B', answer_bytes)
            bi = response_array.buffer_info()

            # Send response to loadgen
            response = lg.QuerySampleResponse(
                query_id,
                bi[0],
                bi[1] * response_array.itemsize,
                len(answer_bytes)
            )
            lg.QuerySamplesComplete([response])

            log.info(f"Completed query {sample_id} (QID: {query_id})")

    def flush_queries(self):
        """Flush any pending queries."""
        pass

    def save_results(self, output_path: str):
        """Save results to JSON file."""
        with self.results_lock:
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
        log.info(f"Results saved to {output_path}")

    def finalize(self):
        """Finalize logging and cleanup."""
        experiment_end_time = datetime.now()
        self.llm_logger.experiment_metadata.update({
            "timestamp_end": experiment_end_time.isoformat(),
            "total_queries": len(self.results)
        })
        self.llm_logger.save()
        log.info("LLM logs finalized")

        # Cleanup reranker
        self.rag_db.shutdown_reranker()
        log.info("Reranker shutdown complete")

    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'sut') and self.sut is not None:
            lg.DestroySUT(self.sut)
            log.info("Finished destroying SUT.")
