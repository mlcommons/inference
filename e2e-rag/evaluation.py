# Copyright 2025 The MLPerf Authors. All Rights Reserved.
# Copyright 2026 Arm Ltd. and affiliates
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


"""Retrieval metrics (precision / recall / F1 at k) for RAG-QnA results."""

from typing import List, Dict


def calculate_retrieval_metrics(expected_urls: List[str], retrieved_urls: List[str], k_values: List[int] = [
                                1, 3, 5, 10]) -> Dict[str, float]:
    """
    Calculate comprehensive retrieval metrics.

    Args:
        expected_urls: List of expected/ground truth URLs
        retrieved_urls: List of retrieved URLs in ranking order
        k_values: List of k values for Precision@k, Recall@k, F1@k

    Returns:
        Dictionary containing all calculated metrics
    """
    expected_set = set(url for url in expected_urls if url and url.strip())

    # Handle edge cases
    if not expected_set:
        return {f'precision@{k}': 1.0 if len(retrieved_urls) == 0 else 0.0 for k in k_values} | \
               {f'recall@{k}': 1.0 for k in k_values} | \
               {f'f1@{k}': 1.0 if len(retrieved_urls) == 0 else 0.0 for k in k_values} | \
               {'average_precision': 1.0 if len(retrieved_urls) == 0 else 0.0}

    metrics = {}

    # Calculate metrics for different k values, including @N (actual retrieved
    # count)
    num_retrieved = len(retrieved_urls)
    num_expected = len(expected_set)
    k_values_with_n = k_values + [num_retrieved]  # Add N to k_values

    for k in k_values_with_n:
        # Determine the label (use 'N' for the actual retrieved count)
        k_label = 'N' if k == num_retrieved else str(k)

        # Get top k documents
        top_k = retrieved_urls[:k]
        top_k_set = set(top_k)
        relevant_retrieved = len(expected_set.intersection(top_k_set))

        # Precision@k: fraction of retrieved documents that are relevant
        precision_k = relevant_retrieved / k if k > 0 else 0.0
        metrics[f'precision@{k_label}'] = precision_k

        # Recall@k: fraction of relevant documents that are retrieved
        recall_k = relevant_retrieved / num_expected if num_expected > 0 else 0.0
        metrics[f'recall@{k_label}'] = recall_k

        # F1@k: harmonic mean of precision and recall
        if precision_k + recall_k > 0:
            f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
        else:
            f1_k = 0.0
        metrics[f'f1@{k_label}'] = f1_k

    # Mean Average Precision (MAP) - considers ranking order
    ap_sum = 0.0
    relevant_found = 0

    for i, url in enumerate(retrieved_urls):
        if url in expected_set:
            relevant_found += 1
            precision_at_i = relevant_found / (i + 1)
            ap_sum += precision_at_i

    average_precision = ap_sum / \
        len(expected_set) if len(expected_set) > 0 else 0.0
    metrics['average_precision'] = average_precision

    return metrics
