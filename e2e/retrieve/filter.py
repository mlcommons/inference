#!/usr/bin/env python3
"""
Retrieval Filter: Score-based retrieval methods for dynamic result filtering.

This module provides adaptive retrieval strategies that adjust result count
based on score distributions rather than fixed-k limits.

Implements various thresholding approaches:
- Top-p (nucleus sampling) - popular in NLP
- Score threshold - absolute quality bar
- Relative threshold - adaptive to query difficulty  
- Elbow method - natural breakpoints
- Percentile-based - statistical cutoffs

Usage:
    from retrieve import retrieval_filter
    results = retrieval_filter(rag_db, query, method="relative", ratio=0.75)
"""

import numpy as np
from typing import List, Tuple, Any, Dict
import math


def softmax(scores: List[float]) -> List[float]:
    """Convert scores to probabilities using softmax."""
    exp_scores = [math.exp(s) for s in scores]
    sum_exp = sum(exp_scores)
    return [exp_s / sum_exp for exp_s in exp_scores]


def top_p_filter(results_with_scores: List[Tuple[Any, float]], p: float = 0.9) -> List[Any]:
    """
    Top-p (nucleus) sampling: Take results until cumulative probability >= p
    
    Args:
        results_with_scores: List of (result, score) tuples, sorted by score DESC
        p: Cumulative probability threshold (0.8-0.95 typical)
    
    Returns:
        Filtered results list
    """
    if not results_with_scores:
        return []
    
    scores = [score for _, score in results_with_scores]
    probs = softmax(scores)
    
    cumulative_prob = 0.0
    selected_results = []
    
    for i, ((result, score), prob) in enumerate(zip(results_with_scores, probs)):
        cumulative_prob += prob
        selected_results.append(result)
        
        if cumulative_prob >= p:
            break
    
    return selected_results


def score_threshold_filter(results_with_scores: List[Tuple[Any, float]], 
                          threshold: float, higher_better: bool = True) -> List[Any]:
    """
    Absolute score threshold filtering.
    
    Args:
        results_with_scores: List of (result, score) tuples
        threshold: Absolute score cutoff
        higher_better: If True, keep scores >= threshold, else <= threshold
    
    Returns:
        Filtered results list
    """
    selected_results = []
    
    for result, score in results_with_scores:
        if higher_better and score >= threshold:
            selected_results.append(result)
        elif not higher_better and score <= threshold:
            selected_results.append(result)
    
    return selected_results


def relative_threshold_filter(results_with_scores: List[Tuple[Any, float]], 
                             ratio: float = 0.8) -> List[Any]:
    """
    Relative threshold: Keep results within ratio * max_score.
    
    Args:
        results_with_scores: List of (result, score) tuples
        ratio: Fraction of max score to use as threshold (0.7-0.9 typical)
    
    Returns:
        Filtered results list
    """
    if not results_with_scores:
        return []
    
    max_score = max(score for _, score in results_with_scores)
    threshold = ratio * max_score
    
    return score_threshold_filter(results_with_scores, threshold, higher_better=True)


def elbow_method_filter(results_with_scores: List[Tuple[Any, float]]) -> List[Any]:
    """
    Elbow method: Find largest score gap and cut there.
    
    Args:
        results_with_scores: List of (result, score) tuples, sorted by score DESC
    
    Returns:
        Filtered results list
    """
    if len(results_with_scores) <= 1:
        return [result for result, _ in results_with_scores]
    
    scores = [score for _, score in results_with_scores]
    
    # Calculate gaps between consecutive scores
    gaps = []
    for i in range(len(scores) - 1):
        gap = scores[i] - scores[i + 1]  # Assuming DESC order
        gaps.append(gap)
    
    # Find largest gap
    if not gaps:
        return [result for result, _ in results_with_scores]
    
    max_gap_idx = gaps.index(max(gaps))
    cutoff_point = max_gap_idx + 1  # Include the score before the gap
    
    return [result for result, _ in results_with_scores[:cutoff_point]]


def percentile_filter(results_with_scores: List[Tuple[Any, float]], 
                     percentile: float = 90.0) -> List[Any]:
    """
    Percentile-based filtering: Keep top X percentile of scores.
    
    Args:
        results_with_scores: List of (result, score) tuples
        percentile: Percentile threshold (80-95 typical)
    
    Returns:
        Filtered results list
    """
    if not results_with_scores:
        return []
    
    scores = [score for _, score in results_with_scores]
    threshold = np.percentile(scores, percentile)
    
    return score_threshold_filter(results_with_scores, threshold, higher_better=True)


def filter(rag_db, query: str, method: str = "top_p", 
           max_results: int = 100, **kwargs) -> List[Any]:
    """
    Perform adaptive retrieval using score-based filtering.
    
    Args:
        rag_db: RAG database instance (BM25DB or VectorDB)
        query: Search query
        method: Filtering method ("top_p", "score_threshold", "relative", "elbow", "percentile")
        max_results: Maximum results to retrieve initially
        **kwargs: Method-specific parameters
    
    Returns:
        Filtered results list
    """
    # Get results with scores
    if hasattr(rag_db, 'lookup_with_scores'):
        results_with_scores = rag_db.lookup_with_scores(query, k=max_results)
    else:
        raise ValueError(f"Database {type(rag_db)} doesn't support score-based retrieval")
    
    # Handle different score formats (Vector vs BM25)
    if isinstance(rag_db.__class__.__name__, str) and "Vector" in rag_db.__class__.__name__:
        # Vector scores are (result, distance) - convert to (result, similarity)
        # Assuming L2 distance, convert to similarity: similarity = 1 / (1 + distance)
        results_with_scores = [(result, 1.0 / (1.0 + score)) for result, score in results_with_scores]
    
    # Sort by score (descending - higher is better)
    results_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Apply filtering method
    if method == "top_p":
        p = kwargs.get("p", 0.9)
        return top_p_filter(results_with_scores, p)
    
    elif method == "score_threshold":
        threshold = kwargs.get("threshold", 5.0)  # Default for BM25
        return score_threshold_filter(results_with_scores, threshold)
    
    elif method == "relative":
        ratio = kwargs.get("ratio", 0.8)
        return relative_threshold_filter(results_with_scores, ratio)
    
    elif method == "elbow":
        return elbow_method_filter(results_with_scores)
    
    elif method == "percentile":
        percentile = kwargs.get("percentile", 90.0)
        return percentile_filter(results_with_scores, percentile)
    
    else:
        raise ValueError(f"Unknown filtering method: {method}")


def get_score_statistics(rag_db, query: str, k: int = 100) -> Dict[str, float]:
    """Get score distribution statistics for threshold calibration."""
    results_with_scores = rag_db.lookup_with_scores(query, k=k)
    scores = [score for _, score in results_with_scores]
    
    if not scores:
        return {}
    
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": np.mean(scores),
        "median": np.median(scores),
        "std": np.std(scores),
        "p75": np.percentile(scores, 75),
        "p90": np.percentile(scores, 90),
        "p95": np.percentile(scores, 95),
        "count": len(scores)
    }


# Backward compatibility alias
adaptive_retrieval = filter