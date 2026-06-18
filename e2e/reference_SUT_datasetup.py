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
System Under Test (SUT) for E2E-RAG-Datasetup workload.
Processes HTML documents, chunks them, generates embeddings, and indexes them.
"""

import os
import sys
import array
import time
import json
import logging
import threading
from datetime import datetime
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlperf_loadgen as lg

from QSL_datasetup import DatasetupQSLInMemory
from retrieve import VectorDB
from text_splitter import split_into_fixed_passages
from utils import get_device_config

# Import HTML extractor
try:
    from bs4 import BeautifulSoup
    from read_docs import HTMLExtractor
    HAVE_HTML = True
except ImportError:
    HAVE_HTML = False
    log.warning("BeautifulSoup not available - HTML processing disabled")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("DatasetupSUT")


class DatasetupSUT:
    """
    System Under Test for E2E-RAG-Datasetup workload.
    Handles loadgen document samples and indexes them one-by-one.
    """

    def __init__(
        self,
        documents_dir: str,
        database: str,
        chunk_size: int = 768,
        chunk_overlap: int = 32,
        text_boundary: str = "word",
        retriever_model: str = "BAAI/bge-base-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
        device: str = "auto",
        num_embedding_devices: int = 1,
        vector_index_method: str = "hnsw",
        output_dir: str = ".",
        max_workers: int = 4,  # Parallel processing (default: 4)
        benchmark: bool = False,
        args: Any = None,
    ):
        """
        Initialize SUT for datasetup workload.

        Args:
            documents_dir: Directory containing HTML documents
            database: Database name/path prefix
            chunk_size: Chunk size in characters
            chunk_overlap: Chunk overlap in characters
            text_boundary: Text boundary optimization (word, sentence, none)
            retriever_model: Path to retriever/embedding model
            reranker_model: Path to reranker model
            device: Device to use (auto, cuda, xpu, hpu, cpu)
            num_embedding_devices: Number of devices for parallel embedding
            vector_index_method: FAISS index method (hnsw, flat, ivf)
            output_dir: Output directory for logs
            max_workers: Maximum number of worker threads (default: 4 for parallel processing)
            benchmark: Enable performance benchmarking
            args: Full args namespace for additional parameters
        """
        self.documents_dir = documents_dir
        self.database = database
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_boundary = text_boundary
        self.retriever_model = retriever_model
        self.reranker_model = reranker_model
        self.device = device
        self.num_embedding_devices = num_embedding_devices
        self.vector_index_method = vector_index_method
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.benchmark = benchmark
        self.args = args

        # Setup device
        device_config = get_device_config()
        log.info(f"Device Config: {device_config}")

        # Initialize QSL
        log.info("Initializing Datasetup Query Sample Library...")
        self.qsl = DatasetupQSLInMemory(documents_dir)

        # Initialize HTML extractor
        if not HAVE_HTML:
            raise RuntimeError("BeautifulSoup required for HTML processing. Install with: pip install beautifulsoup4")

        log.info("Initializing HTML extractor...")
        self.html_extractor = HTMLExtractor(
            preserve_tables=True,
            preserve_lists=True,
            text_boundary=text_boundary
        )

        # Initialize vector database
        log.info("Initializing vector database...")
        self.rag_db = VectorDB(
            retriever_model=retriever_model,
            reranker_model=reranker_model,
            device=device,
            database=database,
            num_embedding_devices=num_embedding_devices,
            benchmark=benchmark,
            vector_index_method=vector_index_method
        )

        # Performance tracking
        self.processing_times = []
        self.processing_lock = threading.Lock()
        self.total_passages_indexed = 0
        self.failed_documents = []
        self.start_time = None

        # Results storage for each document
        self.results = {}
        self.results_lock = threading.Lock()

        # Lock for thread-safe database access during incremental indexing
        self.db_lock = threading.Lock()

        # Track completion for final file
        self.completed_count = 0
        self.completion_lock = threading.Lock()
        self.db_saved = False
        self.db_md5 = None

        # Thread pool for parallel document processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        log.info(f"Thread pool initialized with {self.max_workers} workers")

        # Create MLPerf loadgen SUT
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        log.info("SUT construction complete")

    def issue_queries(self, query_samples):
        """
        Issue queries to SUT. Called by loadgen to process documents.

        Args:
            query_samples: List of query samples to process
        """
        if self.start_time is None:
            self.start_time = time.time()

        futures = []
        for sample in query_samples:
            future = self.thread_pool.submit(self._process_document, sample)
            futures.append(future)

        # Wait for all submissions to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                log.error(f"Document processing failed: {e}")

    def _process_document(self, query_sample):
        """
        Process a single document: parse, chunk, embed, and index.

        Args:
            query_sample: Query sample from loadgen containing document info
        """
        sample_id = query_sample.index
        query_id = query_sample.id  # Unique ID for this query instance
        document_info = self.qsl[sample_id]

        file_path = document_info['file_path']
        file_name = document_info['file_name']

        log.info(f"Processing document {sample_id} (QID: {query_id}): {file_name}")

        start_time = time.time()
        success = 0  # 0 = failure, 1 = success
        passages = []  # Initialize empty list

        try:
            # Step 1: Extract text from HTML
            text = self.html_extractor.extract_text(file_path)

            if not text or len(text.strip()) == 0:
                # Empty file - create minimal passage with file name
                log.warning(f"No text extracted from {file_name}, creating minimal passage")
                text = f"Document: {file_name}"

            # Continue with text (even if minimal)
            # Step 2: Split text into fixed-size chunks
            passages = split_into_fixed_passages(
                text,
                fixed_length=self.chunk_size,
                overlap=self.chunk_overlap
            )

            if not passages:
                # Even after splitting, no passages - this shouldn't happen now
                # but create one minimal passage as fallback
                log.warning(f"No passages created from {file_name}, using text as-is")
                passages = [text]

            # Step 3: Generate embeddings (parallel, outside lock)
            passage_metadata = [{'source': file_name, 'passage_id': i} for i in range(len(passages))]

            # Generate embeddings WITHOUT holding db_lock (allows parallel embedding generation)
            log.info(f"Generating embeddings for {len(passages)} passages from {file_name}")
            if self.rag_db._num_embedding_devices > 1:
                embeddings = self.rag_db._embed_documents_parallel(passages)
            else:
                embeddings = self.rag_db._embedding_model.embed_documents(passages)

            log.info(f"Embeddings generated for {file_name}, adding to index")

            # Step 4: Add to index (thread-safe, holds lock only for index update)
            with self.db_lock:
                # Add embeddings and documents to vector store
                ids = self.rag_db._vector_store.add_embeddings(
                    text_embeddings=list(zip(passages, embeddings)),
                    metadatas=passage_metadata
                )

            # Update statistics
            with self.processing_lock:
                self.total_passages_indexed += len(passages)

            success = 1
            log.info(f"Successfully indexed {len(passages)} passages from {file_name}")

            # Check if this is the last file to complete
            with self.completion_lock:
                self.completed_count += 1
                is_last_file = (self.completed_count == len(self.qsl))

            if is_last_file:
                log.info(f"Last file completed! Saving database and computing MD5...")
                # Save database
                db_path = f"{self.database}.db"
                save_start = time.time()
                self.rag_db.serialize(db_path)
                save_end = time.time()
                save_duration = save_end - save_start
                log.info(f"Database saved in {save_duration:.2f}s")

                # Compute MD5 of database file
                import hashlib
                md5_start = time.time()
                md5_hash = hashlib.md5()
                with open(db_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        md5_hash.update(chunk)
                self.db_md5 = md5_hash.hexdigest()
                md5_end = time.time()
                md5_duration = md5_end - md5_start
                log.info(f"Database MD5: {self.db_md5} (computed in {md5_duration:.2f}s)")

                self.db_saved = True

        except Exception as e:
            log.error(f"Error processing {file_name}: {e}")
            success = 0
            with self.processing_lock:
                self.failed_documents.append({
                    'file_name': file_name,
                    'error': str(e)
                })

        end_time = time.time()
        elapsed = end_time - start_time

        # Track performance
        with self.processing_lock:
            self.processing_times.append(elapsed)

        # Store result
        with self.results_lock:
            self.results[sample_id] = {
                'file_name': file_name,
                'success': success,
                'time_seconds': elapsed,
                'passages_count': len(passages) if success else 0
            }

        # Create response for loadgen
        # For the last file, include MD5 hash; otherwise just success/failure byte
        if success and self.db_saved:
            # Last file: return MD5 hash as response
            response_bytes = self.db_md5.encode('utf-8')
            log.info(f"Returning MD5 hash for last file: {self.db_md5}")
        else:
            # Regular file: return single byte (0 or 1)
            response_bytes = bytes([success])

        response_array = array.array('B', response_bytes)
        bi = response_array.buffer_info()

        # Send response to loadgen
        # Parameters: query_id, data_ptr, size_in_bytes, n_tokens (optional)
        response = lg.QuerySampleResponse(
            query_id,  # Use query_id, not sample_id
            bi[0],     # Data pointer
            bi[1] * response_array.itemsize,  # Size in bytes
            len(response_bytes)  # Number of bytes
        )
        lg.QuerySamplesComplete([response])

        log.info(f"Completed document {sample_id} (QID: {query_id}): {file_name}")

    def flush_queries(self):
        """
        Flush any remaining queries. Called by loadgen before test completes.
        Database has already been saved by the last file, so nothing to do here.
        """
        log.info("Flushing queries...")

        if self.db_saved:
            log.info(f"Database already saved by last file. MD5: {self.db_md5}")
        else:
            # Fallback: save database if somehow not done yet
            log.warning("Database not saved by last file - saving now as fallback")
            db_path = f"{self.database}.db"
            save_start = time.time()
            self.rag_db.serialize(db_path)
            save_end = time.time()
            save_duration = save_end - save_start
            log.info(f"Database saved in {save_duration:.2f}s (fallback)")

    def finalize(self):
        """
        Finalize SUT: compute metrics, cleanup.
        Note: Batch indexing and database save happen in flush_queries().
        """
        log.info("Finalizing SUT...")

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)

        # Compute final metrics
        end_time = time.time()
        total_time = end_time - self.start_time if self.start_time else 0

        with self.processing_lock:
            total_docs = len(self.processing_times)
            failed_count = len(self.failed_documents)
            success_count = total_docs - failed_count

            avg_time_per_doc = sum(self.processing_times) / total_docs if total_docs > 0 else 0
            throughput_passages = self.total_passages_indexed / total_time if total_time > 0 else 0
            throughput_docs = total_docs / total_time if total_time > 0 else 0

        log.info("="*80)
        log.info("Datasetup Complete")
        log.info("="*80)
        log.info(f"Total documents processed: {total_docs}")
        log.info(f"Successful: {success_count}")
        log.info(f"Failed: {failed_count}")
        log.info(f"Total passages indexed: {self.total_passages_indexed}")
        log.info(f"Total time: {total_time:.2f}s")
        log.info(f"Throughput: {throughput_passages:.2f} passages/sec")
        log.info(f"Throughput: {throughput_docs:.2f} docs/sec")
        log.info(f"Average time per document: {avg_time_per_doc:.2f}s")
        log.info("="*80)

        # Cleanup reranker queue if it exists
        if hasattr(self.rag_db, '_reranker_queue') and self.rag_db._reranker_queue is not None:
            log.info("Shutting down reranker queue...")
            self.rag_db._reranker_queue.stop()

        # Destroy SUT
        lg.DestroySUT(self.sut)

    def save_results(self, output_path):
        """
        Save detailed results to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        end_time = time.time()
        total_time = end_time - self.start_time if self.start_time else 0

        with self.processing_lock:
            total_docs = len(self.processing_times)
            failed_count = len(self.failed_documents)
            success_count = total_docs - failed_count

            throughput_passages = self.total_passages_indexed / total_time if total_time > 0 else 0
            throughput_docs = total_docs / total_time if total_time > 0 else 0

        # Get vector count from database
        vector_count = 0
        if hasattr(self.rag_db, '_vector_store') and hasattr(self.rag_db._vector_store, 'index'):
            vector_count = self.rag_db._vector_store.index.ntotal

        output = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "documents_dir": self.documents_dir,
                "database": self.database,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "text_boundary": self.text_boundary,
                "retriever_model": self.retriever_model,
                "device": self.device,
                "vector_index_method": self.vector_index_method,
                "num_embedding_devices": self.num_embedding_devices,
            },
            "performance": {
                "total_documents": total_docs,
                "successful_documents": success_count,
                "failed_documents": failed_count,
                "total_passages_indexed": self.total_passages_indexed,
                "vector_count": vector_count,
                "data_setup_time_seconds": total_time,
                "throughput_passages_per_second": throughput_passages,
                "throughput_documents_per_second": throughput_docs,
            },
            "database": {
                "path": f"{self.database}.db",
                "md5": self.db_md5,
                "saved": self.db_saved,
            },
            "per_document_results": self.results,
            "failed_documents": self.failed_documents,
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        log.info(f"Results saved to {output_path}")
