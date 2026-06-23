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


def softmax(scores: List[float], temperature: float = 1.0) -> List[float]:
    """Convert scores to probabilities using softmax with temperature scaling.
    
    Args:
        scores: List of scores to convert
        temperature: Temperature parameter (lower = sharper distribution)
                    - temperature = 1.0: standard softmax
                    - temperature < 1.0: sharper (more weight on top scores)
                    - temperature > 1.0: smoother (more uniform)
    
    Returns:
        List of probabilities that sum to 1.0
    """
    # Apply temperature scaling
    scaled_scores = [s / temperature for s in scores]
    exp_scores = [math.exp(s) for s in scaled_scores]
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
    
    #print(f"\n[DEBUG top_p_filter] p={p}, num_candidates={len(scores)}")
    #print(f"[DEBUG] Score range: [{min(scores):.4f}, {max(scores):.4f}]")
    #print(f"[DEBUG] Score mean: {sum(scores)/len(scores):.4f}")
    #print(f"[DEBUG] First 10 scores: {[f'{s:.4f}' for s in scores[:10]]}")
    
    # Use temperature scaling to sharpen the distribution
    # Lower temperature = more discriminative (top docs get higher probability)
    # For vector embeddings with compressed L2 distances, use very low temperature
    # Temperature = 0.01 to 0.05 for L2 distances in range [0.27-0.42]
    temperature = 1
    #temperature = 0.02
    probs = softmax(scores, temperature=temperature)
    
    #print(f"[DEBUG] Temperature: {temperature}")
    #print(f"[DEBUG] Probability range: [{min(probs):.6f}, {max(probs):.6f}]")
    #print(f"[DEBUG] First 10 probs: {[f'{p:.6f}' for p in probs[:10]]}")
    #print(f"[DEBUG] Prob sum: {sum(probs):.6f}")
    
    cumulative_prob = 0.0
    selected_results = []
    
    for i, ((result, score), prob) in enumerate(zip(results_with_scores, probs)):
        cumulative_prob += prob
        selected_results.append(result)
        
        if cumulative_prob >= p:
            print(f"[DEBUG] Selected {i+1} documents (cumulative_prob={cumulative_prob:.4f} >= p={p})")
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
    Relative threshold: Keep top ratio fraction of results based on score range.
    
    For both positive and negative scores:
    - Calculates score range between best and worst
    - Keeps only results within top ratio% of that range
    
    Args:
        results_with_scores: List of (result, score) tuples (sorted desc, best first)
        ratio: Fraction of score range to keep (0.7-0.9 typical)
               e.g., 0.9 means keep top 90% of score range
    
    Returns:
        Filtered results list
    
    Example with negative scores:
        Scores: [-0.42, -0.43, -0.44, ..., -0.49]
        best=-0.42, worst=-0.49, range=0.07
        ratio=0.9 → cutoff_range = 0.07 * 0.9 = 0.063
        threshold = -0.42 - 0.063 = -0.483
        Keeps: scores >= -0.483 (top 90% of range)
    """
    if not results_with_scores:
        return []
    
    # Get best and worst scores
    best_score = results_with_scores[0][1]
    worst_score = results_with_scores[-1][1]
    
    # Calculate the score range
    score_range = best_score - worst_score  # Always positive since sorted desc
    
    # Calculate threshold: start from best, move down by (1-ratio) of range
    cutoff_distance = score_range * (1 - ratio)
    threshold = best_score - cutoff_distance
    
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
        rag_db: RAG database instance (VectorDB)
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
    
    # Results already have proper similarity scores (higher is better) from lookup_with_scores
    # Sort by score (descending - higher is better)
    results_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Apply filtering method
    if method == "top_p":
        p = kwargs.get("p", 0.9)
        return top_p_filter(results_with_scores, p)
    
    elif method == "score_threshold":
        threshold = kwargs.get("threshold", 5.0)
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