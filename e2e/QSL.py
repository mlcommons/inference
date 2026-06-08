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
Query Sample Library for E2E DocGrader workload.
Loads queries from frames_dataset.tsv and provides them to MLPerf Loadgen.
"""

import os
import pandas as pd
import mlperf_loadgen as lg


class E2EQSL:
    """Query Sample Library for E2E DocGrader multi-hop RAG benchmark."""

    def __init__(self, dataset_path, perf_count=None, skip_qsl=False):
        """
        Initialize QSL by loading queries from frames_dataset.tsv.

        Args:
            dataset_path: Path to frames_dataset.tsv file
            perf_count: Number of queries to use (None = all queries)
            skip_qsl: If True, skip constructing the actual loadgen QSL object
        """
        # Load dataset
        print(f"Loading dataset from {dataset_path}...")
        self.df = pd.read_csv(dataset_path, sep='\t')

        # Extract queries and ground truth
        self.queries = []
        self.ground_truth = []
        self.expected_urls = []

        for idx, row in self.df.iterrows():
            query = row['Prompt']
            answer = row['Answer']

            # Extract expected Wikipedia URLs
            urls = []
            for col in self.df.columns:
                if col.startswith('wikipedia_link_'):
                    url = row[col]
                    if pd.notna(url) and url != '':
                        urls.append(url)

            self.queries.append(query)
            self.ground_truth.append(answer)
            self.expected_urls.append(urls)

        self.count = len(self.queries)

        # Limit to perf_count if specified
        if perf_count is not None:
            self.count = min(self.count, perf_count)
            self.queries = self.queries[:self.count]
            self.ground_truth = self.ground_truth[:self.count]
            self.expected_urls = self.expected_urls[:self.count]

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

        print(f"Dataset loaded: {self.count} queries")
        if perf_count is not None:
            print(f"  (limited to first {perf_count} queries for performance testing)")

    def load_query_samples(self, sample_list):
        """
        Load query samples into memory.
        Called by loadgen before issuing queries.
        """
        for sample_id in sample_list:
            if sample_id < self.count:
                self.sample_id_to_sample[sample_id] = {
                    'query': self.queries[sample_id],
                    'ground_truth': self.ground_truth[sample_id],
                    'expected_urls': self.expected_urls[sample_id]
                }

    def unload_query_samples(self, sample_list):
        """
        Unload query samples from memory.
        Called by loadgen after queries are complete.
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
                'query': self.queries[index],
                'ground_truth': self.ground_truth[index],
                'expected_urls': self.expected_urls[index]
            }

    def __len__(self):
        """Return number of samples."""
        return self.count

    def __del__(self):
        """Cleanup."""
        if self.qsl is not None:
            lg.DestroyQSL(self.qsl)
            print("Finished destroying QSL.")


class E2EQSLInMemory(E2EQSL):
    """
    In-memory version of E2EQSL that pre-loads all samples.
    For workloads where all queries fit in memory.
    """

    def __init__(self, dataset_path, perf_count=None):
        # Initialize parent with skip_qsl=False (we need the loadgen QSL!)
        super().__init__(dataset_path, perf_count, skip_qsl=False)

        # Pre-load all samples
        self.load_query_samples(range(self.count))

        print(f"All {self.count} queries loaded into memory")
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
Query Sample Library for E2E DocGrader workload.
Loads queries from frames_dataset.tsv and provides them to MLPerf Loadgen.
"""

import os
import pandas as pd
import mlperf_loadgen as lg


class E2EQSL:
    """Query Sample Library for E2E DocGrader multi-hop RAG benchmark."""

    def __init__(self, dataset_path, perf_count=None, skip_qsl=False):
        """
        Initialize QSL by loading queries from frames_dataset.tsv.

        Args:
            dataset_path: Path to frames_dataset.tsv file
            perf_count: Number of queries to use (None = all queries)
            skip_qsl: If True, skip constructing the actual loadgen QSL object
        """
        # Load dataset
        print(f"Loading dataset from {dataset_path}...")
        self.df = pd.read_csv(dataset_path, sep='\t')

        # Extract queries and ground truth
        self.queries = []
        self.ground_truth = []
        self.expected_urls = []

        for idx, row in self.df.iterrows():
            query = row['Prompt']
            answer = row['Answer']

            # Extract expected Wikipedia URLs
            urls = []
            for col in self.df.columns:
                if col.startswith('wikipedia_link_'):
                    url = row[col]
                    if pd.notna(url) and url != '':
                        urls.append(url)

            self.queries.append(query)
            self.ground_truth.append(answer)
            self.expected_urls.append(urls)

        self.count = len(self.queries)

        # Limit to perf_count if specified
        if perf_count is not None:
            self.count = min(self.count, perf_count)
            self.queries = self.queries[:self.count]
            self.ground_truth = self.ground_truth[:self.count]
            self.expected_urls = self.expected_urls[:self.count]

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

        print(f"Dataset loaded: {self.count} queries")
        if perf_count is not None:
            print(f"  (limited to first {perf_count} queries for performance testing)")

    def load_query_samples(self, sample_list):
        """
        Load query samples into memory.
        Called by loadgen before issuing queries.
        """
        for sample_id in sample_list:
            if sample_id < self.count:
                self.sample_id_to_sample[sample_id] = {
                    'query': self.queries[sample_id],
                    'ground_truth': self.ground_truth[sample_id],
                    'expected_urls': self.expected_urls[sample_id]
                }

    def unload_query_samples(self, sample_list):
        """
        Unload query samples from memory.
        Called by loadgen after queries are complete.
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
                'query': self.queries[index],
                'ground_truth': self.ground_truth[index],
                'expected_urls': self.expected_urls[index]
            }

    def __len__(self):
        """Return number of samples."""
        return self.count

    def __del__(self):
        """Cleanup."""
        if self.qsl is not None:
            lg.DestroyQSL(self.qsl)
            print("Finished destroying QSL.")


class E2EQSLInMemory(E2EQSL):
    """
    In-memory version of E2EQSL that pre-loads all samples.
    For workloads where all queries fit in memory.
    """

    def __init__(self, dataset_path, perf_count=None):
        # Initialize parent with skip_qsl=False (we need the loadgen QSL!)
        super().__init__(dataset_path, perf_count, skip_qsl=False)

        # Pre-load all samples
        self.load_query_samples(range(self.count))

        print(f"All {self.count} queries loaded into memory")
