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


"""
Retrieval Evaluation Metrics Module

This module provides comprehensive retrieval evaluation metrics including:
- Precision@k, Recall@k, F1@k
- Mean Average Precision (MAP)
- Comprehensive retrieval metrics
- Detailed dataset analysis by reasoning type and answer link count

Designed for reuse across different retrieval systems including multi-hop QA.
"""

import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from collections import defaultdict
from utils import filter_dataset_by_difficulty


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


def evaluate_retrieval_query(rag_db, query: str, expected_urls: List[str],
                             top_k_retriever: int = 50, top_k_reranking: int = 10,
                             verbose: bool = True, no_rerank: bool = False,
                             retrieval_strategy: str = "fixed_k", print_results: bool = False,
                             return_results: bool = False, **strategy_params) -> Union[Dict[str, Any], Tuple[Dict[str, Any], List[Any]]]:
    """
    Evaluate a single retrieval query and return comprehensive retrieval metrics.

    Args:
        rag_db: RAG database instance
        query: Query string
        expected_urls: List of expected URLs
        top_k_retriever: Number of documents to retrieve initially
        top_k_reranking: Number of documents after reranking
        verbose: Whether to print detailed results
        no_rerank: Skip reranking step for fair comparison between retrieval methods
        retrieval_strategy: Strategy for retrieval ("fixed_k", "top_p", "relative")
        **strategy_params: Parameters for adaptive retrieval strategies

    Returns:
        Dictionary containing all metrics. When return_results=True, returns a tuple of (metrics_dict, retrieved_results).
    """
    import time

    # Step 1: Time the initial retrieval
    retrieval_start = time.perf_counter()
    if retrieval_strategy == "fixed_k":
        results = rag_db.lookup(query, k=top_k_retriever)
    else:
        from retrieve.filter import filter
        max_results = strategy_params.pop("max_results", 20)
        results = filter(rag_db, query, method=retrieval_strategy,
                         max_results=max_results, **strategy_params)
    retrieval_time = time.perf_counter() - retrieval_start

    # Step 2: Apply reranking if enabled and reranker is available
    reranking_time = 0.0
    has_reranker = getattr(rag_db, '_reranker_queue', None) is not None
    if not no_rerank and has_reranker:
        # Safety check: If no results retrieved, skip reranking
        if not results:
            if verbose:
                print(
                    f"Warning: No documents retrieved for query: {query[:50]}")
        else:
            reranking_start = time.perf_counter()
            # Rerank Documents by score; fixed_k takes top-k, adaptive keeps all.
            reranked_results = rag_db.rerank_documents(query, results)
            if retrieval_strategy == "fixed_k":
                results = reranked_results[:top_k_reranking]
            else:
                results = reranked_results
            reranking_time = time.perf_counter() - reranking_start

    # Extract URLs from results in order (maintaining ranking)
    retrieved_urls = []
    for result in results:
        if 'original_url' in result.metadata and result.metadata['original_url']:
            retrieved_urls.append(result.metadata['original_url'])

    # Deduplicate URLs preserving first appearance order (for accurate MAP
    # calculation)
    # Preserves order, removes duplicates
    deduplicated_urls = list(dict.fromkeys(retrieved_urls))

    # Calculate comprehensive metrics using deduplicated URLs (accurate MAP)
    expected_set = set(url for url in expected_urls if url and url.strip())
    metrics = calculate_retrieval_metrics(
        list(expected_set), deduplicated_urls)

    # Track both passages and unique documents
    num_passages = len(results)
    num_unique_docs = len(deduplicated_urls)

    if verbose:
        print(f"Query: {query:50}")
        matches = len(expected_set.intersection(set(deduplicated_urls)))
        print(
            f"Expected ({len(expected_set)}): {sorted(list(expected_set)[:3])}{'...' if len(expected_set) > 3 else ''}")
        print(
            f"Retrieved ({num_passages} passages, {num_unique_docs} unique docs): {deduplicated_urls[:3]}{'...' if num_unique_docs > 3 else ''}")
        print(f"Matches: {matches}")

        metric_categories = [
            ("P", "precision"),
            ("R", "recall"),
            ("F1", "f1")
        ]

        for label, metric_prefix in metric_categories:
            parts = [f"{label}@N: {metrics[f'{metric_prefix}@N']:.3f}"]
            for k in [3, 5, 10]:
                key = f"{metric_prefix}@{k}"
                if key in metrics:
                    parts.append(f"{label}@{k}: {metrics[key]:.3f}")
            print(", ".join(parts))

        print(f"MAP: {metrics['average_precision']:.3f}")
        print("-" * 80)

    # Print detailed results for single query mode
    if print_results:
        print(
            f"\n{retrieval_strategy.upper()} lookup took time. {len(results)} results found:")

        # Display which PDFs the passages are from
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.metadata}")
            print("-" * 50)

        # Show reranked results if reranker is available and reranking was used
        if not no_rerank and has_reranker:
            print(f"\nReranking to top-{top_k_reranking}")
            print(f"Reranking results:")

            for i, result in enumerate(results, 1):
                print(f"{i}. {result.metadata}")
                print(f"   {result.page_content[:200]}...")
                print("-" * 50)
        else:
            if no_rerank:
                print("No reranker used (--no-rerank specified)")
            else:
                print("No reranker available - showing retrieval results only")

    # Calculate retrieval performance metrics
    total_time = retrieval_time + reranking_time
    docs_per_second = len(results) / total_time if total_time > 0 else 0

    # Add retrieval performance to metrics
    retrieval_metrics = {
        'retrieval_time': retrieval_time,
        'reranking_time': reranking_time,
        'total_retrieval_time': total_time,
        'retrieved_passages_count': len(results),
        'retrieved_docs_count': num_unique_docs,
        'docs_per_second': docs_per_second
    }

    # Print retrieval performance if in benchmark mode and single query mode
    # (not evaluation)
    if hasattr(rag_db, '_benchmark') and rag_db._benchmark and print_results:
        print(f"\n🔍 RETRIEVAL PERFORMANCE METRICS")
        print("=" * 50)
        print(f"📊 Query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        print(f"⏱️  Retrieval time: {retrieval_time*1000:.2f}ms")
        if reranking_time > 0:
            print(f"🔄 Reranking time: {reranking_time*1000:.2f}ms")
        print(f"🕐 Total time: {total_time*1000:.2f}ms")
        print(f"📦 Documents retrieved: {len(results)}")
        print(f"🚀 Retrieval speed: {docs_per_second:.1f} docs/sec")
        print(f"💾 Time per query: {total_time:.4f}s")
        print()

    # Return metrics dict with retrieval performance
    merged_metrics = {**metrics, **retrieval_metrics}
    if return_results:
        return merged_metrics, results
    return merged_metrics


def run_evaluation(rag_db, dataset_path: str,
                   top_k_retriever: int = 50, top_k_reranking: int = 10,
                   max_queries: Optional[int] = None, no_rerank: bool = False,
                   retrieval_strategy: str = "fixed_k", detailed_analysis: bool = False,
                   difficulty: int = 0, collect_results: bool = False,
                   result_handler: Optional[Callable[[
                       str, List[Any], Dict[str, Any]], Optional[Any]]] = None,
                   **strategy_params) -> Union[Dict[str, float], Tuple[Dict[str, float], List[Dict[str, Any]]]]:
    """
    Run comprehensive evaluation on a dataset with detailed metrics reporting.

    Args:
        rag_db: RAG database instance
        dataset_path: Path to the dataset TSV file
        top_k_retriever: Number of documents to retrieve initially
        top_k_reranking: Number of documents after reranking
        max_queries: Maximum number of queries to evaluate (None = all)
        no_rerank: Skip reranking step for fair comparison between retrieval methods
        retrieval_strategy: Strategy for retrieval ("fixed_k", "top_p", "relative")
        detailed_analysis: If True, print detailed breakdown by reasoning types and link counts
        difficulty: Minimum number of answer links required (0 = no filtering)
        collect_results: If True, also collect retrieval outputs for each query
        result_handler: Optional callback invoked per query with (prompt, retrieved_docs, metrics)
        **strategy_params: Parameters for adaptive retrieval strategies

    Returns:
        Dictionary of averaged metrics across all queries. When collect_results=True, returns a tuple of (metrics_dict, collected_results).
    """
    df = pd.read_csv(dataset_path, sep='\t')

    # Filter by difficulty if specified
    df = filter_dataset_by_difficulty(df, difficulty)

    # Limit number of queries if specified
    if isinstance(max_queries, int) and max_queries > 0:
        df = df.head(max_queries)
    else:
        max_queries = len(df)

    print(f"\nRunning evaluation on {max_queries} queries from dataset")

    # Aggregate metrics collection
    total_metrics = {}
    all_query_metrics = []  # Store individual query metrics for detailed analysis
    retrieval_times = []
    reranking_times = []
    total_times = []
    docs_per_sec_list = []
    collected_queries = [] if collect_results else None
    valid_queries = 0

    for idx, row in df.iterrows():
        # Extract expected Wikipedia links
        expected_urls = []
        for col in df.columns:
            if col.startswith('wikipedia_link_') and pd.notna(row[col]):
                expected_urls.append(row[col].strip())

        if expected_urls:
            # Get comprehensive metrics for this query
            need_results = collect_results or (result_handler is not None)
            metrics_output = evaluate_retrieval_query(
                rag_db, row['Prompt'], expected_urls,
                top_k_retriever, top_k_reranking, verbose=True, no_rerank=no_rerank,
                retrieval_strategy=retrieval_strategy, return_results=need_results,
                **strategy_params
            )
            if need_results:
                metrics, retrieved_docs = metrics_output
            else:
                metrics = metrics_output
                retrieved_docs = []

            if collect_results and retrieved_docs:
                doc_entries = []
                seen_urls = set()
                for doc in retrieved_docs:
                    url = None
                    if hasattr(doc, 'metadata'):
                        url = doc.metadata.get(
                            'original_url') or doc.metadata.get('source')
                        content = doc.page_content
                    elif isinstance(doc, dict):
                        url = doc.get('url')
                        content = doc.get('content', "")
                    else:
                        content = ""
                    if url and url in seen_urls:
                        continue
                    entry = {
                        "url": url,
                        "content": content[:2000]
                    }
                    doc_entries.append(entry)
                    if url:
                        seen_urls.add(url)
                collected_queries.append({
                    "prompt": row['Prompt'],
                    "docs": doc_entries
                })

            if result_handler:
                result_handler(row['Prompt'], retrieved_docs, metrics)

            # Store metrics for detailed analysis if requested
            if detailed_analysis:
                all_query_metrics.append(metrics)

            # Collect retrieval performance metrics for statistics
            if 'retrieval_time' in metrics:
                retrieval_times.append(metrics['retrieval_time'])
                reranking_times.append(metrics['reranking_time'])
                total_times.append(metrics['total_retrieval_time'])
                docs_per_sec_list.append(metrics['docs_per_second'])

            # Accumulate metrics
            for metric_name, value in metrics.items():
                if metric_name not in total_metrics:
                    total_metrics[metric_name] = 0.0
                total_metrics[metric_name] += value

            valid_queries += 1

    if valid_queries > 0:
        # Calculate average metrics
        avg_metrics = {
            name: total /
            valid_queries for name,
            total in total_metrics.items()}

        # Display results
        results_title = "OVERALL EVALUATION RESULTS" if detailed_analysis else "EVALUATION RESULTS"
        print(f"\n" + "=" * 60)
        print(f"{results_title} ({valid_queries} queries)")
        print(f"=" * 60)
        print(f"PRECISION METRICS:")
        print(
            f"  Precision@N:                {avg_metrics.get('precision@N', 0.0):.3f}")
        if 'precision@1' in avg_metrics:
            print(
                f"  Precision@1:                {avg_metrics['precision@1']:.3f}")
        if 'precision@3' in avg_metrics:
            print(
                f"  Precision@3:                {avg_metrics['precision@3']:.3f}")
        if 'precision@5' in avg_metrics:
            print(
                f"  Precision@5:                {avg_metrics['precision@5']:.3f}")
        if 'precision@10' in avg_metrics:
            print(
                f"  Precision@10:               {avg_metrics['precision@10']:.3f}")
        print(f"")
        print(f"RECALL METRICS:")
        print(
            f"  Recall@N:                   {avg_metrics.get('recall@N', 0.0):.3f}")
        if 'recall@1' in avg_metrics:
            print(
                f"  Recall@1:                   {avg_metrics['recall@1']:.3f}")
        if 'recall@3' in avg_metrics:
            print(
                f"  Recall@3:                   {avg_metrics['recall@3']:.3f}")
        if 'recall@5' in avg_metrics:
            print(
                f"  Recall@5:                   {avg_metrics['recall@5']:.3f}")
        if 'recall@10' in avg_metrics:
            print(
                f"  Recall@10:                  {avg_metrics['recall@10']:.3f}")
        print(f"")
        print(f"F1 METRICS:")
        print(
            f"  F1@N:                       {avg_metrics.get('f1@N', 0.0):.3f}")
        if 'f1@1' in avg_metrics:
            print(f"  F1@1:                       {avg_metrics['f1@1']:.3f}")
        if 'f1@3' in avg_metrics:
            print(f"  F1@3:                       {avg_metrics['f1@3']:.3f}")
        if 'f1@5' in avg_metrics:
            print(f"  F1@5:                       {avg_metrics['f1@5']:.3f}")
        if 'f1@10' in avg_metrics:
            print(f"  F1@10:                      {avg_metrics['f1@10']:.3f}")
        print(f"")
        print(f"RANKING METRICS:")
        print(
            f"  Mean Average Precision:     {avg_metrics['average_precision']:.3f}")
        print(f"")
        print(f"RETRIEVAL STATISTICS:")
        print(
            f"  Avg Passages Retrieved:     {avg_metrics.get('retrieved_passages_count', 0.0):.1f}")
        print(
            f"  Avg Unique Docs (N):        {avg_metrics.get('retrieved_docs_count', 0.0):.1f}")

        # Add retrieval performance statistics if we have retrieval data
        if retrieval_times and hasattr(
                rag_db, '_benchmark') and rag_db._benchmark:
            import numpy as np

            print(f"")
            print(f"🔍 RETRIEVAL PERFORMANCE STATISTICS:")
            print(f"  Retrieval Time (ms):")
            print(
                f"    Average:                  {np.mean(retrieval_times)*1000:.2f}ms")
            print(
                f"    P50 (Median):             {np.percentile(retrieval_times, 50)*1000:.2f}ms")
            print(
                f"    P99:                      {np.percentile(retrieval_times, 99)*1000:.2f}ms")

            if any(t > 0 for t in reranking_times):
                print(f"  Reranking Time (ms):")
                print(
                    f"    Average:                  {np.mean(reranking_times)*1000:.2f}ms")
                print(
                    f"    P50 (Median):             {np.percentile(reranking_times, 50)*1000:.2f}ms")
                print(
                    f"    P99:                      {np.percentile(reranking_times, 99)*1000:.2f}ms")

            print(f"  Total Query Time (ms):")
            print(
                f"    Average:                  {np.mean(total_times)*1000:.2f}ms")
            print(
                f"    P50 (Median):             {np.percentile(total_times, 50)*1000:.2f}ms")
            print(
                f"    P99:                      {np.percentile(total_times, 99)*1000:.2f}ms")

            print(f"  Retrieval Throughput (docs/sec):")
            print(
                f"    Average:                  {np.mean(docs_per_sec_list):.1f} docs/sec")
            print(
                f"    P50 (Median):             {np.percentile(docs_per_sec_list, 50):.1f} docs/sec")
            print(
                f"    P99:                      {np.percentile(docs_per_sec_list, 99):.1f} docs/sec")

        print(f"=" * 60)

        # Print detailed analysis if requested
        if detailed_analysis:
            _print_detailed_analysis(df, all_query_metrics, valid_queries)

        if collect_results:
            return avg_metrics, collected_queries or []
        return avg_metrics
    else:
        print("No valid queries found!")
        if collect_results:
            return {}, []
        return {}


def _print_detailed_analysis(df: pd.DataFrame, all_query_metrics: List[Dict[str, Any]],
                             valid_queries: int) -> None:
    """
    Print detailed dataset analysis broken down by reasoning types and answer link counts.
    (Internal helper function for run_evaluation)

    Args:
        df: DataFrame with dataset (must have 'reasoning_types' column)
        all_query_metrics: List of metrics dictionaries for each query
        valid_queries: Number of valid queries processed
    """
    if valid_queries == 0:
        return

    print("\n" + "=" * 80)
    print("DETAILED DATASET ANALYSIS")
    print("=" * 80)

    # Prepare data - match metrics with reasoning types and link counts
    analysis_data = []
    for idx, metrics in enumerate(all_query_metrics):
        if idx < len(df):
            row = df.iloc[idx]
            reasoning_types = row.get('reasoning_types', 'Unknown')

            # Count Wikipedia links
            num_links = sum(1 for col in df.columns
                            if col.startswith('wikipedia_link_') and pd.notna(row[col]))

            analysis_data.append({
                'reasoning_types': reasoning_types,
                'num_links': num_links,
                'metrics': metrics
            })

    # === ANALYSIS 1: By Reasoning Classification ===
    print("\n" + "-" * 80)
    print("ANALYSIS BY REASONING CLASSIFICATION")
    print("-" * 80)

    # Group by reasoning types
    reasoning_groups = defaultdict(list)
    for data in analysis_data:
        reasoning_groups[data['reasoning_types']].append(data['metrics'])

    # Calculate averages for each reasoning type
    reasoning_results = []
    for reasoning_type, metrics_list in reasoning_groups.items():
        if not metrics_list:
            continue

        avg_metrics = {}
        for key in ['precision@N', 'recall@N', 'f1@N', 'average_precision']:
            values = [m.get(key, 0.0) for m in metrics_list]
            avg_metrics[key] = sum(values) / len(values)

        reasoning_results.append({
            'type': reasoning_type,
            'count': len(metrics_list),
            **avg_metrics
        })

    # Sort by count (most common first)
    reasoning_results.sort(key=lambda x: x['count'], reverse=True)

    # Print top reasoning types
    print(f"\nTop reasoning type combinations:")
    print(f"{'Reasoning Type':<50} {'Count':>6} {'P@N':>6} {'R@N':>6} {'F1@N':>6} {'MAP':>6}")
    print("-" * 80)

    for i, result in enumerate(reasoning_results):
        rt = result['type'][:48] if len(
            result['type']) > 48 else result['type']
        print(f"{rt:<50} {result['count']:6d} "
              f"{result['precision@N']:6.3f} {result['recall@N']:6.3f} "
              f"{result['f1@N']:6.3f} {result['average_precision']:6.3f}")

    # === ANALYSIS 2: By Individual Reasoning Tags ===
    print(f"\n" + "-" * 80)
    print("ANALYSIS BY INDIVIDUAL REASONING TAGS")
    print("-" * 80)

    # Parse reasoning tags (split by |)
    tag_groups = defaultdict(list)
    for data in analysis_data:
        tags = [tag.strip() for tag in data['reasoning_types'].split('|')]
        for tag in tags:
            tag_groups[tag].append(data['metrics'])

    tag_results = []
    for tag, metrics_list in tag_groups.items():
        if not metrics_list:
            continue

        avg_metrics = {}
        for key in ['precision@N', 'recall@N', 'f1@N', 'average_precision']:
            values = [m.get(key, 0.0) for m in metrics_list]
            avg_metrics[key] = sum(values) / len(values)

        tag_results.append({
            'tag': tag,
            'count': len(metrics_list),
            'percentage': len(metrics_list) / valid_queries * 100,
            **avg_metrics
        })

    # Sort by count
    tag_results.sort(key=lambda x: x['count'], reverse=True)

    print(f"\nPerformance by reasoning tag:")
    print(f"{'Tag':<30} {'Count':>6} {'%':>6} {'P@N':>6} {'R@N':>6} {'F1@N':>6} {'MAP':>6}")
    print("-" * 80)

    for result in tag_results:
        tag = result['tag'][:28] if len(result['tag']) > 28 else result['tag']
        print(f"{tag:<30} {result['count']:6d} {result['percentage']:5.1f}% "
              f"{result['precision@N']:6.3f} {result['recall@N']:6.3f} "
              f"{result['f1@N']:6.3f} {result['average_precision']:6.3f}")

    # === ANALYSIS 3: By Number of Answer Links ===
    print(f"\n" + "-" * 80)
    print("ANALYSIS BY NUMBER OF ANSWER LINKS (Multi-hop Analysis)")
    print("-" * 80)

    # Group by number of links
    link_groups = defaultdict(list)
    for data in analysis_data:
        link_groups[data['num_links']].append(data['metrics'])

    link_results = []
    for num_links, metrics_list in link_groups.items():
        if not metrics_list:
            continue

        avg_metrics = {}
        for key in ['precision@N', 'recall@N', 'f1@N', 'average_precision']:
            values = [m.get(key, 0.0) for m in metrics_list]
            avg_metrics[key] = sum(values) / len(values)

        # Classify complexity
        if num_links <= 2:
            complexity = "Simple"
        elif num_links <= 4:
            complexity = "Multi-hop"
        else:
            complexity = "Complex"

        link_results.append({
            'num_links': num_links,
            'complexity': complexity,
            'count': len(metrics_list),
            'percentage': len(metrics_list) / valid_queries * 100,
            **avg_metrics
        })

    # Sort by number of links
    link_results.sort(key=lambda x: x['num_links'])

    print(f"\nPerformance by number of Wikipedia links (reasoning hops):")
    print(f"{'Links':>5} {'Complexity':<12} {'Count':>6} {'%':>6} {'P@N':>6} {'R@N':>6} {'F1@N':>6} {'MAP':>6}")
    print("-" * 80)

    for result in link_results:
        print(f"{result['num_links']:5d} {result['complexity']:<12} "
              f"{result['count']:6d} {result['percentage']:5.1f}% "
              f"{result['precision@N']:6.3f} {result['recall@N']:6.3f} "
              f"{result['f1@N']:6.3f} {result['average_precision']:6.3f}")

    # Summary by complexity category
    print(f"\n" + "-" * 80)
    print("SUMMARY BY COMPLEXITY LEVEL")
    print("-" * 80)

    complexity_groups = defaultdict(list)
    for result in link_results:
        for _ in range(result['count']):
            # Get original metrics for this group
            complexity_groups[result['complexity']].append({
                'precision@N': result['precision@N'],
                'recall@N': result['recall@N'],
                'f1@N': result['f1@N'],
                'average_precision': result['average_precision']
            })

    # Calculate totals
    complexity_summary = []
    for complexity in ["Simple", "Multi-hop", "Complex"]:
        if complexity not in complexity_groups:
            continue

        metrics_list = complexity_groups[complexity]
        count = len(metrics_list)

        # Recalculate from link_results
        matching_results = [
            r for r in link_results if r['complexity'] == complexity]
        total_count = sum(r['count'] for r in matching_results)

        # Weighted average
        weighted_metrics = {}
        for key in ['precision@N', 'recall@N', 'f1@N', 'average_precision']:
            weighted_sum = sum(r[key] * r['count'] for r in matching_results)
            weighted_metrics[key] = weighted_sum / \
                total_count if total_count > 0 else 0.0

        complexity_summary.append({
            'complexity': complexity,
            'count': total_count,
            'percentage': total_count / valid_queries * 100,
            **weighted_metrics
        })

    print(f"\n{'Complexity':<12} {'Count':>6} {'%':>6} {'P@N':>6} {'R@N':>6} {'F1@N':>6} {'MAP':>6}")
    print("-" * 80)

    for result in complexity_summary:
        print(f"{result['complexity']:<12} {result['count']:6d} {result['percentage']:5.1f}% "
              f"{result['precision@N']:6.3f} {result['recall@N']:6.3f} "
              f"{result['f1@N']:6.3f} {result['average_precision']:6.3f}")

    # === ANALYSIS 4: Correlation Between Complexity and Reasoning Types ===
    print(f"\n" + "-" * 80)
    print("CORRELATION: COMPLEXITY vs REASONING TYPES")
    print("-" * 80)

    # Build correlation matrix: complexity level x reasoning tags
    complexity_reasoning_data = defaultdict(lambda: defaultdict(list))

    for data in analysis_data:
        # Determine complexity
        num_links = data['num_links']
        if num_links <= 2:
            complexity = "Simple"
        elif num_links <= 4:
            complexity = "Multi-hop"
        else:
            complexity = "Complex"

        # Extract individual reasoning tags
        tags = [tag.strip() for tag in data['reasoning_types'].split('|')]
        for tag in tags:
            complexity_reasoning_data[complexity][tag].append(data['metrics'])

    # Calculate statistics for each complexity-reasoning combination
    print(f"\n1. REASONING TAG DISTRIBUTION BY COMPLEXITY:")
    print(f"{'Reasoning Tag':<30} {'Simple':>10} {'Multi-hop':>10} {'Complex':>10} {'Total':>10}")
    print("-" * 80)

    # Get all unique tags
    all_tags = set()
    for complexity_data in complexity_reasoning_data.values():
        all_tags.update(complexity_data.keys())

    tag_distribution = {}
    for tag in sorted(all_tags):
        simple_count = len(
            complexity_reasoning_data.get(
                'Simple',
                {}).get(
                tag,
                []))
        multihop_count = len(
            complexity_reasoning_data.get(
                'Multi-hop',
                {}).get(
                tag,
                []))
        complex_count = len(
            complexity_reasoning_data.get(
                'Complex',
                {}).get(
                tag,
                []))
        total = simple_count + multihop_count + complex_count

        tag_distribution[tag] = {
            'simple': simple_count,
            'multihop': multihop_count,
            'complex': complex_count,
            'total': total
        }

        tag_display = tag[:28] if len(tag) > 28 else tag
        print(
            f"{tag_display:<30} {simple_count:10d} {multihop_count:10d} {complex_count:10d} {total:10d}")

    # Calculate percentage distribution
    print(f"\n2. REASONING TAG PERCENTAGE BY COMPLEXITY:")
    print(f"{'Reasoning Tag':<30} {'Simple %':>10} {'Multi %':>10} {'Complex %':>10}")
    print("-" * 80)

    for tag in sorted(all_tags):
        dist = tag_distribution[tag]
        total = dist['total']
        if total > 0:
            simple_pct = (dist['simple'] / total) * 100
            multihop_pct = (dist['multihop'] / total) * 100
            complex_pct = (dist['complex'] / total) * 100

            tag_display = tag[:28] if len(tag) > 28 else tag
            print(
                f"{tag_display:<30} {simple_pct:9.1f}% {multihop_pct:9.1f}% {complex_pct:9.1f}%")

    # Calculate average number of links per reasoning tag
    print(f"\n3. AVERAGE COMPLEXITY (# LINKS) BY REASONING TAG:")
    print(f"{'Reasoning Tag':<30} {'Avg Links':>10} {'Count':>10}")
    print("-" * 80)

    tag_link_stats = defaultdict(list)
    for data in analysis_data:
        tags = [tag.strip() for tag in data['reasoning_types'].split('|')]
        for tag in tags:
            tag_link_stats[tag].append(data['num_links'])

    tag_avg_links = []
    for tag in sorted(all_tags):
        links = tag_link_stats[tag]
        if links:
            avg_links = sum(links) / len(links)
            tag_avg_links.append((tag, avg_links, len(links)))

    # Sort by average links (descending)
    tag_avg_links.sort(key=lambda x: x[1], reverse=True)

    for tag, avg_links, count in tag_avg_links:
        tag_display = tag[:28] if len(tag) > 28 else tag
        print(f"{tag_display:<30} {avg_links:10.2f} {count:10d}")

    # Performance by complexity x reasoning tag (for top tags only)
    print(f"\n4. PERFORMANCE BY COMPLEXITY x TOP REASONING TAGS:")
    print("-" * 80)

    # Get top 5 most common tags
    top_tags = sorted(
        tag_distribution.items(),
        key=lambda x: x[1]['total'],
        reverse=True)[
        :5]

    for tag, _ in top_tags:
        print(f"\n{tag}:")
        print(
            f"{'Complexity':<12} {'Count':>6} {'P@N':>6} {'R@N':>6} {'F1@N':>6} {'MAP':>6}")
        print("-" * 70)

        for complexity in ["Simple", "Multi-hop", "Complex"]:
            metrics_list = complexity_reasoning_data.get(
                complexity, {}).get(tag, [])
            if metrics_list:
                count = len(metrics_list)
                avg_p = sum(m.get('precision@N', 0.0)
                            for m in metrics_list) / count
                avg_r = sum(m.get('recall@N', 0.0)
                            for m in metrics_list) / count
                avg_f1 = sum(m.get('f1@N', 0.0) for m in metrics_list) / count
                avg_map = sum(m.get('average_precision', 0.0)
                              for m in metrics_list) / count

                print(
                    f"{complexity:<12} {count:6d} {avg_p:6.3f} {avg_r:6.3f} {avg_f1:6.3f} {avg_map:6.3f}")

    print("=" * 80)
