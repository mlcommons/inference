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


import abc
import os
from typing import List, Dict, Any


class RagDB(abc.ABC):
    """Base class for retrieval-augmented generation databases."""

    def __init__(self, reranker_model: str = None, device: str = "auto",
                 benchmark: bool = False, reranker_device: str = None):
        self._reranker_model_name = reranker_model
        self._device = self._determine_device(device)
        # Reranker device defaults to inheriting from --device.
        self._reranker_device = self._determine_device(
            reranker_device) if reranker_device else self._device
        self._reranker_queue = None
        self._benchmark = benchmark
        self._monitor = None

        # Initialize monitoring if benchmark mode enabled
        if self._benchmark:
            from ingestion_monitor import IngestionMonitor
            self._monitor = IngestionMonitor()

        # Initialize out-of-process reranker if specified
        if self._reranker_model_name:
            self._init_reranker()

    def _determine_device(self, device: str) -> str:
        """Determine the best device to use.

        Delegates to utils.detect_device() for auto detection so device-selection
        logic lives in one place. ROCm maps to "cuda"
        """
        if device == "rocm":
            return "cuda"
        if device == "auto":
            from utils import detect_device
            return detect_device()
        return device

    @staticmethod
    def get_data_dir(db_name: str) -> str:
        """Get data directory based on database name."""
        from pathlib import Path
        base_name = Path(db_name).stem  # Remove .db extension if present
        return f"{base_name}_data"

    @staticmethod
    def get_db_path(db_name: str) -> str:
        """Get database file path based on database name."""
        from pathlib import Path
        base_name = Path(db_name).stem  # Remove .db extension if present
        return f"{base_name}.db"

    def _init_reranker(self):
        """Spawn the reranker in its own process."""
        from utils import resolve_gpu_device
        from reranker_worker import RerankerQueue

        device = resolve_gpu_device(
            self._reranker_device, name="reranker",
            override_env="INFERENCE_RERANKER_GPU_DEVICES",
        )
        self._reranker_device = device

        numa_node = os.environ.get("INFERENCE_RERANKER_NUMA_NODE")
        omp_threads = os.environ.get("INFERENCE_RERANKER_OMP_NUM_THREADS")
        print(f"Using {device} for reranker"
              f"{f' (NUMA node {numa_node})' if numa_node else ''}"
              f"{f' OMP={omp_threads}' if omp_threads else ''}")

        self._reranker_queue = RerankerQueue(
            self._reranker_model_name, device=device,
            numa_node=int(numa_node) if numa_node is not None else None,
            omp_threads=int(omp_threads) if omp_threads else None,
        )
        self._reranker_queue.start()

    def _track_component(self, name: str, total_chars: int, item_count: int, func,
                         is_pipeline_input: bool = False, is_pipeline_output: bool = False):
        """Execute function with optional component tracking.

        Args:
            name: Component name
            total_chars: Input size in bytes
            item_count: Number of items processed
            func: Function to execute
            is_pipeline_input: Mark as pipeline input for aggregation
            is_pipeline_output: Mark as pipeline output for aggregation
        """
        if self._benchmark and self._monitor:
            with self._monitor.track_component(name, input_size_bytes=total_chars,
                                               items_count=item_count, text_only=True,
                                               is_pipeline_input=is_pipeline_input,
                                               is_pipeline_output=is_pipeline_output) as ctx:
                result = func()
                ctx.add_text_bytes(total_chars)
                return result
        else:
            return func()

    def _start_ingestion_timer(self):
        """Start the ingestion timer. Works for both benchmark and non-benchmark modes."""
        import time
        if self._benchmark and self._monitor:
            self._monitor.start_ingestion()
        return time.perf_counter()

    def _report_performance(self, ingestion_start_time: float,
                            item_count: int, total_chars: int, db_type: str):
        """Report performance metrics with optional detailed breakdown.

        Args:
            ingestion_start_time: Start time from _start_ingestion_timer() (used only in non-benchmark mode)
            item_count: Number of items processed
            total_chars: Total characters processed
            db_type: Database type string for display
        """
        import time

        if self._benchmark and self._monitor:
            with self._monitor.track_ingestion() as ingestion_ctx:
                ingestion_ctx.set_item_count(item_count)
            print(f"\n=== {db_type} Performance ===")
            self._monitor.print_summary()
        else:
            end_time = time.perf_counter()
            duration = end_time - ingestion_start_time
            docs_per_sec = item_count / duration if duration > 0 else 0
            chars_per_sec = total_chars / duration if duration > 0 else 0
            print(
                f"{db_type} ingestion: {item_count} docs, {total_chars:,} chars in {duration:.2f}s")
            print(
                f"  Performance: {docs_per_sec:.1f} docs/sec, {chars_per_sec/1024:.1f} KB/sec")

    def enable_threading(self):
        """Enable thread-safe access. Override in subclasses that need locks."""
        pass

    @abc.abstractmethod
    def ingest(self, passages: List[str], metadatas: List[Dict[str, Any]]):
        """Ingest passages and their metadata into the database."""
        pass

    @abc.abstractmethod
    def lookup(self, query: str, k: int) -> List[Any]:
        """Retrieve top-k relevant passages for a query."""
        pass

    @abc.abstractmethod
    def serialize(self, path: str):
        """Serialize the database to disk."""
        pass

    @abc.abstractmethod
    def from_serialized(self, path: str):
        """Load the database from disk."""
        pass

    def ingest_from_folder(self, folder_path: str, **kwargs):
        """Ingest data from a folder. Default implementation raises NotImplementedError."""
        raise NotImplementedError(
            f"Folder ingestion not supported for {self.__class__.__name__}")

    def ingest_from_file(self, file_path: str, **kwargs):
        """Ingest data from a JSON file. Default implementation for JSON files.

        Supports both flat and hierarchical passage formats:
        - Flat: entry['passage'] → passage text
        - Hierarchical: entry['child_passage'] → child text (for embedding)
                       entry['parent_passage'] → parent text (stored in metadata)
        """
        import json

        with open(file_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        passage_data = payload.get('passages', [])
        config = payload.get('config', {})

        # Detect hierarchical format
        is_hierarchical = (
            config.get('strategy') == 'hierarchical' or
            (len(passage_data) > 0 and 'child_passage' in passage_data[0])
        )

        doc_list = []
        passage_metadata = []

        if is_hierarchical:
            print(f"Detected hierarchical format")
            for entry in passage_data:
                # Use child for embedding
                doc_list.append(entry['child_passage'])
                # Store all metadata including parent
                metadata = {k: v for k, v in entry.items() if k !=
                            'child_passage'}
                passage_metadata.append(metadata)
        else:
            # Flat format
            for entry in passage_data:
                doc_list.append(entry['passage'])
                passage_metadata.append(
                    {k: v for k, v in entry.items() if k != 'passage'})

        print(f"Ingesting {len(doc_list)} passages from JSON file {file_path}")
        return self.ingest(doc_list, passage_metadata,
                           passages_path=file_path, **kwargs)

    def ingest_from_path(self, source_path: str, **kwargs):
        """Handle both file and folder ingestion.

        Default implementation that delegates to appropriate methods:
        - Folders: calls ingest_from_folder() (may raise NotImplementedError if not overridden)
        - Files: calls ingest_from_file() (default JSON implementation)
        """
        from pathlib import Path

        source_path = Path(source_path)

        if source_path.is_dir():
            print(f"Ingesting documents from folder {source_path}")
            return self.ingest_from_folder(source_path, **kwargs)
        elif source_path.is_file():
            return self.ingest_from_file(source_path, **kwargs)
        else:
            raise ValueError(
                f"Source path {source_path} is neither a file nor a directory")

    def shutdown_reranker(self):
        """Tear down the reranker child process. Safe to call multiple times."""
        if self._reranker_queue is not None:
            self._reranker_queue.stop()
            self._reranker_queue = None

    def rerank(self, query: str, passages: List[str]):
        """Score passages via the reranker; returns (passage, score) in input order."""
        if self._reranker_queue:
            return self._reranker_queue.submit(query, passages)
        return [(p, 0.0) for p in passages]

    def lookup_with_rerank(self, query: str, k: int,
                           rerank_k: int = None) -> List[Any]:
        """Retrieve and rerank passages."""
        if rerank_k is None:
            rerank_k = k

        # Get initial results
        results = self.lookup(query, k=rerank_k)

        # If no reranker or fewer results than requested, return as-is
        if self._reranker_queue is None or len(results) <= k:
            return results[:k]

        # Extract passages for reranking
        passages = [result.page_content for result in results]

        # Rerank
        reranked_passages = self.rerank(query, passages)

        # Map back to original results and return top-k
        reranked_results = []
        for passage, score in reranked_passages[:k]:
            for result in results:
                if result.page_content == passage:
                    reranked_results.append(result)
                    break

        return reranked_results

    @property
    def device(self) -> str:
        """Get the device being used."""
        return self._device
