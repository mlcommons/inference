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
Query Sample Library for RAG DB workload.
Loads HTML document file paths and provides them to MLPerf Loadgen for indexing.
"""

import os
from pathlib import Path
import mlperf_loadgen as lg


class DatasetupQSL:
    """Query Sample Library for RAG DB benchmark."""

    def __init__(self, documents_dir, skip_qsl=False):
        """
        Initialize QSL by loading HTML document file paths.

        Args:
            documents_dir: Path to directory containing HTML documents
            skip_qsl: If True, skip constructing the actual loadgen QSL object
        """
        # Find all HTML files
        print(f"Scanning for HTML documents in {documents_dir}...")
        self.documents_dir = Path(documents_dir)

        if not self.documents_dir.exists():
            raise ValueError(f"Documents directory not found: {documents_dir}")

        # Get all HTML files
        self.html_files = sorted([
            str(f.relative_to(self.documents_dir))
            for f in self.documents_dir.glob("*.html")
        ])

        self.count = len(self.html_files)

        if self.count == 0:
            raise ValueError(f"No HTML files found in {documents_dir}")

        # Sample ID to sample data mapping
        self.sample_id_to_sample = {}

        # Construct loadgen QSL
        if skip_qsl:
            self.qsl = None
        else:
            self.qsl = lg.ConstructQSL(
                self.count,
                self.count,  # perf_count = total count
                self.load_query_samples,
                self.unload_query_samples
            )

        print(f"Dataset loaded: {self.count} HTML files")

    def load_query_samples(self, sample_list):
        """
        Load document samples into memory.
        Called by loadgen before issuing queries.
        """
        for sample_id in sample_list:
            if sample_id < self.count:
                self.sample_id_to_sample[sample_id] = {
                    'file_path': str(self.documents_dir / self.html_files[sample_id]),
                    'file_name': self.html_files[sample_id]
                }

    def unload_query_samples(self, sample_list):
        """
        Unload document samples from memory.
        Called by loadgen after processing is complete.
        """
        for sample_id in sample_list:
            if sample_id in self.sample_id_to_sample:
                del self.sample_id_to_sample[sample_id]

    def __getitem__(self, index):
        """Get sample by index."""
        if index in self.sample_id_to_sample:
            return self.sample_id_to_sample[index]
        else:
            # Fallback: construct on-the-fly if not loaded
            return {
                'file_path': str(self.documents_dir / self.html_files[index]),
                'file_name': self.html_files[index]
            }

    def __len__(self):
        """Return number of samples."""
        return self.count

    def __del__(self):
        """Cleanup."""
        if self.qsl is not None:
            lg.DestroyQSL(self.qsl)
            print("Finished destroying QSL.")


class DatasetupQSLInMemory(DatasetupQSL):
    """
    In-memory version of DatasetupQSL that pre-loads all samples.
    For workloads where all document paths fit in memory.
    """

    def __init__(self, documents_dir):
        # Initialize parent with skip_qsl=False (we need the loadgen QSL!)
        super().__init__(documents_dir, skip_qsl=False)

        # Pre-load all samples
        self.load_query_samples(range(self.count))

        print(f"All {self.count} document paths loaded into memory")
