"""
Retrieval Evaluation Metrics Module

This module provides comprehensive retrieval evaluation metrics including:
- Precision@k, Recall@k, F1@k
- Mean Average Precision (MAP)
- Legacy compatibility metrics

Designed for reuse across different retrieval systems including multi-hop QA.
"""

import pandas as pd
from typing import List, Dict, Any, Optional


def calculate_retrieval_metrics(expected_urls: List[str], retrieved_urls: List[str], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
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
    
    # Calculate Precision@k, Recall@k, F1@k for different k values
    for k in k_values:
        top_k = retrieved_urls[:k]
        top_k_set = set(top_k)
        
        # Calculate metrics
        relevant_retrieved = len(expected_set.intersection(top_k_set))
        
        # Precision@k: fraction of retrieved documents that are relevant
        precision_k = relevant_retrieved / k if k > 0 else 0.0
        metrics[f'precision@{k}'] = precision_k
        
        # Recall@k: fraction of relevant documents that are retrieved
        recall_k = relevant_retrieved / len(expected_set) if len(expected_set) > 0 else 0.0
        metrics[f'recall@{k}'] = recall_k
        
        # F1@k: harmonic mean of precision and recall
        if precision_k + recall_k > 0:
            f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
        else:
            f1_k = 0.0
        metrics[f'f1@{k}'] = f1_k
    
    # Mean Average Precision (MAP) - considers ranking order
    ap_sum = 0.0
    relevant_found = 0
    
    for i, url in enumerate(retrieved_urls):
        if url in expected_set:
            relevant_found += 1
            precision_at_i = relevant_found / (i + 1)
            ap_sum += precision_at_i
    
    average_precision = ap_sum / len(expected_set) if len(expected_set) > 0 else 0.0
    metrics['average_precision'] = average_precision
    
    return metrics


def evaluate_query(rag_db, query: str, expected_urls: List[str], 
                         top_k_retriever: int = 50, top_k_reranking: int = 10,
                         verbose: bool = True, no_rerank: bool = False) -> Dict[str, Any]:
    """
    Evaluate a single query and return comprehensive metrics.
    
    Args:
        rag_db: RAG database instance
        query: Query string
        expected_urls: List of expected URLs
        top_k_retriever: Number of documents to retrieve initially
        top_k_reranking: Number of documents after reranking
        verbose: Whether to print detailed results
        no_rerank: Skip reranking step for fair comparison between retrieval methods
        
    Returns:
        Dictionary containing all metrics plus legacy score
    """
    # Get retrieval results (with or without reranking)
    if no_rerank:
        results = rag_db.lookup(query, k=top_k_retriever)
    else:
        results = rag_db.lookup_with_rerank(query, k=top_k_reranking, rerank_k=top_k_retriever)
    
    # Extract URLs from results in order (maintaining ranking)
    retrieved_urls = []
    for result in results:
        if 'original_url' in result.metadata and result.metadata['original_url']:
            retrieved_urls.append(result.metadata['original_url'])

    # Deduplicate URLs preserving first appearance order (for accurate MAP calculation)
    deduplicated_urls = list(dict.fromkeys(retrieved_urls))  # Preserves order, removes duplicates
    
    # Calculate comprehensive metrics using deduplicated URLs (accurate MAP)
    expected_set = set(url for url in expected_urls if url and url.strip())
    metrics = calculate_retrieval_metrics(list(expected_set), deduplicated_urls)    # Legacy score for backward compatibility (recall@10)
    legacy_score = metrics.get('recall@10', 0.0)
    
    if verbose:
        # Print detailed results
        print(f"Query: {query[:100]}...")
        print(f"Expected ({len(expected_set)}): {sorted(list(expected_set)[:3])}{'...' if len(expected_set) > 3 else ''}")
        print(f"Retrieved ({len(retrieved_urls)}): {retrieved_urls[:3]}{'...' if len(retrieved_urls) > 3 else ''}")
        
        # Print key metrics (now calculated on deduplicated URLs)
        matches = len(expected_set.intersection(set(deduplicated_urls)))
        print(f"Matches: {matches}, Legacy Score: {legacy_score:.3f}")
        print(f"P@3: {metrics['precision@3']:.3f}, P@5: {metrics['precision@5']:.3f}, P@10: {metrics['precision@10']:.3f}")
        print(f"R@3: {metrics['recall@3']:.3f}, R@5: {metrics['recall@5']:.3f}, R@10: {metrics['recall@10']:.3f}")
        print(f"F1@10: {metrics['f1@10']:.3f}, MAP: {metrics['average_precision']:.3f}")
        print("-" * 80)
    
    # Return metrics dict with legacy score for compatibility
    return {'legacy_score': legacy_score, **metrics}


def run_evaluation(rag_db, dataset_path: str, 
                               top_k_retriever: int = 50, top_k_reranking: int = 10, 
                               max_queries: Optional[int] = None, no_rerank: bool = False) -> Dict[str, float]:
    """
    Run comprehensive evaluation on a dataset with detailed metrics reporting.
    
    Args:
        rag_db: RAG database instance
        dataset_path: Path to the dataset TSV file
        top_k_retriever: Number of documents to retrieve initially
        top_k_reranking: Number of documents after reranking  
        max_queries: Maximum number of queries to evaluate (None = all)
        no_rerank: Skip reranking step for fair comparison between retrieval methods
        
    Returns:
        Dictionary of averaged metrics across all queries
    """
    df = pd.read_csv(dataset_path, sep='\t')
    
    # Limit number of queries if specified
    if isinstance(max_queries, int) and max_queries > 0:
        df = df.head(max_queries)
    else:
        max_queries = len(df)

    print(f"\nRunning evaluation on {max_queries} queries from dataset")
    
    # Aggregate metrics collection
    total_metrics = {}
    valid_queries = 0
    
    for idx, row in df.iterrows():
        # Extract expected Wikipedia links
        expected_urls = []
        for col in df.columns:
            if col.startswith('wikipedia_link_') and pd.notna(row[col]):
                expected_urls.append(row[col].strip())
        
        if expected_urls:
            # Get comprehensive metrics for this query
            metrics = evaluate_query(
                rag_db, row['Prompt'], expected_urls, 
                top_k_retriever, top_k_reranking, verbose=True, no_rerank=no_rerank
            )
            
            # Accumulate metrics
            for metric_name, value in metrics.items():
                if metric_name not in total_metrics:
                    total_metrics[metric_name] = 0.0
                total_metrics[metric_name] += value
            
            valid_queries += 1
    
    if valid_queries > 0:
        # Calculate average metrics
        avg_metrics = {name: total / valid_queries for name, total in total_metrics.items()}
        
        # Display results
        print(f"\n" + "="*60)
        print(f"EVALUATION RESULTS ({valid_queries} queries)")
        print(f"="*60)
        print(f"Legacy Score (Recall@10):     {avg_metrics['legacy_score']:.3f}")
        print(f"")
        print(f"PRECISION METRICS:")
        print(f"  Precision@1:                {avg_metrics['precision@1']:.3f}")
        print(f"  Precision@3:                {avg_metrics['precision@3']:.3f}")
        print(f"  Precision@5:                {avg_metrics['precision@5']:.3f}")
        print(f"  Precision@10:               {avg_metrics['precision@10']:.3f}")
        print(f"")
        print(f"RECALL METRICS:")
        print(f"  Recall@1:                   {avg_metrics['recall@1']:.3f}")
        print(f"  Recall@3:                   {avg_metrics['recall@3']:.3f}")
        print(f"  Recall@5:                   {avg_metrics['recall@5']:.3f}")
        print(f"  Recall@10:                  {avg_metrics['recall@10']:.3f}")
        print(f"")
        print(f"F1 METRICS:")
        print(f"  F1@1:                       {avg_metrics['f1@1']:.3f}")
        print(f"  F1@3:                       {avg_metrics['f1@3']:.3f}")
        print(f"  F1@5:                       {avg_metrics['f1@5']:.3f}")
        print(f"  F1@10:                      {avg_metrics['f1@10']:.3f}")
        print(f"")
        print(f"RANKING METRICS:")
        print(f"  Mean Average Precision:     {avg_metrics['average_precision']:.3f}")
        print(f"="*60)
        
        return avg_metrics
    else:
        print("No valid queries found!")
        return {}