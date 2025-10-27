"""
Multi-shot Retrieval System

This module implements multi-shot retrieval with query decomposition:
1. Takes a complex query
2. Uses LLM to rewrite/decompose into multiple sub-queries (max k=3)
3. Retrieves documents for each sub-query
4. Optionally reranks combined results
5. Evaluates performance

Architecture:
    Prompt → Query Rewriter (LLM) → k Sub-queries → Retrieval → Reranking → Evaluation
"""

import argparse
import json
import time
import os
from typing import List, Dict, Any, Optional
import pandas as pd

# Set no_proxy to bypass proxy for localhost/127.0.0.1
original_no_proxy = os.environ.get('no_proxy', '')
os.environ['no_proxy'] = '127.0.0.1,localhost,' + original_no_proxy
os.environ['NO_PROXY'] = '127.0.0.1,localhost,' + original_no_proxy

from retrieve import VectorDB, BM25DB
from evaluation import evaluate_retrieval_query, run_evaluation
from utils import set_deterministic_seeds, filter_dataset_by_difficulty
from params import add_all_args
import requests


# LLM Service Configuration
LLM_SERVICE_URL = "http://127.0.0.1:8123/v1/chat/completions"
LLM_MODEL = "/model/gpt-oss-120b-int4-AutoRound/"  # Full path with trailing slash as shown in server command


def query_rewriter_llm(original_query: str, max_queries: int = 3, reasoning_effort: str = "medium") -> List[str]:
    """
    Use LLM to decompose a complex query into multiple sub-queries.
    
    Args:
        original_query: The original complex query
        max_queries: Maximum number of sub-queries to generate (default: 3)
        reasoning: Reasoning level (kept for compatibility, not used in standard vLLM)
        
    Returns:
        List of sub-queries (including original if appropriate)
    """
    
    system_prompt = """You are an expert at decomposing complex multi-hop questions into simpler sub-questions.

Your task: Given a complex question, break it down into 1-3 simpler sub-questions that, when answered together, would help answer the original question.

Guidelines:
1. Identify the key facts/entities needed to answer the question
2. Create sub-questions that retrieve each piece of information
3. Order sub-questions logically (dependencies first)
4. Keep sub-questions clear and specific
5. If the question is already simple, return just the original question

Output format: Return ONLY a JSON array of sub-questions, nothing else.
Example: ["What year did X happen?", "Who won Y in that year?"]
"""

    user_prompt = f"""Original question: {original_query}

Decompose this into at most {max_queries} sub-questions. Return only the JSON array."""

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,  # Low temperature for consistent decomposition
        "reasoning_effort": reasoning_effort,
        "max_tokens": 500
    }
    
    try:
        #print(f"Calling LLM service at {LLM_SERVICE_URL} with model {LLM_MODEL}...")
        response = requests.post(LLM_SERVICE_URL, json=payload, timeout=60)
        
        #print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error response: {response.text[:500]}")
            print(f"Falling back to original query")
            return [original_query]
        
        result = response.json()
        
        # Handle cases where content might be None (e.g., with reasoning models)
        message = result['choices'][0]['message']
        llm_output = message.get('content')
        
        if llm_output is None:
            print(f"Warning: LLM returned None content. Full response: {json.dumps(result, indent=2)[:500]}")
            print(f"Falling back to original query")
            return [original_query]
        
        llm_output = llm_output.strip()
        print(f"LLM output: {llm_output[:200]}...")
        
        # Parse JSON output
        # Handle cases where LLM wraps in markdown code blocks
        if llm_output.startswith("```json"):
            llm_output = llm_output.replace("```json", "").replace("```", "").strip()
        elif llm_output.startswith("```"):
            llm_output = llm_output.replace("```", "").strip()
        
        sub_queries = json.loads(llm_output)
        
        # Validate output
        if not isinstance(sub_queries, list):
            print(f"Warning: LLM output is not a list, using original query")
            return [original_query]
        
        # Limit to max_queries
        sub_queries = sub_queries[:max_queries]
        
        # Ensure at least the original query is included if list is empty
        if not sub_queries:
            return [original_query]
        
        print(f"\n{'='*80}")
        print(f"QUERY DECOMPOSITION")
        print(f"{'='*80}")
        print(f"Original: {original_query}")
        print(f"Sub-queries ({len(sub_queries)}):")
        for i, sq in enumerate(sub_queries, 1):
            print(f"  {i}. {sq}")
        print(f"{'='*80}\n")
        
        return sub_queries
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling LLM service: {e}")
        print(f"Falling back to original query")
        return [original_query]
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM output as JSON: {e}")
        print(f"LLM output: {llm_output[:200]}")
        print(f"Falling back to original query")
        return [original_query]
    except Exception as e:
        print(f"Unexpected error in query rewriting: {e}")
        import traceback
        traceback.print_exc()
        print(f"Falling back to original query")
        return [original_query]


def multi_shot_retrieval(rag_db, original_query: str, expected_urls: List[str],
                         max_sub_queries: int = 3,
                         top_k_retriever: int = 50, 
                         top_k_reranking: int = 10,
                         no_rerank: bool = False,
                         retrieval_strategy: str = "fixed_k",
                         verbose: bool = True,
                         reasoning_effort: str = "medium",
                         **strategy_params) -> Dict[str, Any]:
    """
    Perform multi-shot retrieval with query decomposition.
    
    Args:
        rag_db: RAG database instance
        original_query: Original complex query
        expected_urls: Expected ground truth URLs for evaluation
        max_sub_queries: Maximum number of sub-queries to generate
        top_k_retriever: Number of documents to retrieve per sub-query
        top_k_reranking: Number of documents after final reranking
        no_rerank: Skip reranking step
        retrieval_strategy: Strategy for retrieval
        verbose: Print detailed information
        reasoning_effort: LLM reasoning level
        **strategy_params: Additional parameters for retrieval strategy
        
    Returns:
        Dictionary containing evaluation metrics
    """
    
    start_time = time.perf_counter()
    
    # Step 1: Query Decomposition
    decomposition_start = time.perf_counter()
    sub_queries = query_rewriter_llm(original_query, max_queries=max_sub_queries, reasoning_effort=reasoning_effort)
    decomposition_time = time.perf_counter() - decomposition_start
    
    # Step 2: Retrieve for each sub-query
    # Strategy: Retrieve N/k docs per sub-query to ensure balanced representation from each sub-query
    # This prevents bias toward first sub-query when taking top_k_reranking from concatenated results
    retrieval_start = time.perf_counter()
    all_results = []
    seen_urls = set()
    
    # Calculate docs per sub-query (ensure at least 1)
    num_sub_queries = len(sub_queries)
    docs_per_subquery = max(1, top_k_retriever // num_sub_queries)
    
    if verbose:
        print(f"\nRetrieval strategy: {num_sub_queries} sub-queries × {docs_per_subquery} docs/query = ~{num_sub_queries * docs_per_subquery} total")
    
    for i, sub_query in enumerate(sub_queries, 1):
        if verbose:
            print(f"\nRetrieving for sub-query {i}: {sub_query[:80]}...")
        
        # Retrieve using adjusted k for balanced representation
        if retrieval_strategy == "fixed_k":
            results = rag_db.lookup(sub_query, k=docs_per_subquery)
        else:
            from retrieve.filter import filter
            # Adjust max_results proportionally for non-fixed strategies
            original_max_results = strategy_params.get("max_results", 20)
            adjusted_max_results = max(1, original_max_results // num_sub_queries)
            strategy_params_copy = strategy_params.copy()
            strategy_params_copy["max_results"] = adjusted_max_results
            results = filter(rag_db, sub_query, method=retrieval_strategy, **strategy_params_copy)
        
        # Deduplicate by URL across sub-queries
        for result in results:
            if 'original_url' in result.metadata and result.metadata['original_url']:
                url = result.metadata['original_url']
                if url not in seen_urls:
                    all_results.append(result)
                    seen_urls.add(url)
        
        if verbose:
            print(f"  Retrieved {len(results)} passages, {len(all_results)} unique docs so far")
    
    retrieval_time = time.perf_counter() - retrieval_start
    
    # Step 3: Optional Reranking
    reranking_time = 0.0
    if not no_rerank and hasattr(rag_db, '_reranker_model') and rag_db._reranker_model is not None:
        if all_results:
            reranking_start = time.perf_counter()
            
            # Rerank combined results using original query
            passages = [result.page_content for result in all_results]
            scored_passages = rag_db.rerank(original_query, passages)
            
            # Reconstruct document objects with reranked order
            reranked_results = []
            for text, score in scored_passages:
                for doc in all_results:
                    if doc.page_content == text:
                        reranked_results.append(doc)
                        break
            
            all_results = reranked_results
            reranking_time = time.perf_counter() - reranking_start
            
            if verbose:
                print(f"\nReranked {len(all_results)} documents")
    
    # Apply top_k_reranking limit (whether reranked or not)
    all_results = all_results[:top_k_reranking]
    
    if verbose and len(all_results) > 0:
        print(f"\nFinal result set: {len(all_results)} documents")
    
    # Step 4: Extract URLs and deduplicate
    retrieved_urls = []
    for result in all_results:
        if 'original_url' in result.metadata and result.metadata['original_url']:
            retrieved_urls.append(result.metadata['original_url'])
    
    deduplicated_urls = list(dict.fromkeys(retrieved_urls))
    
    # Step 5: Calculate Metrics
    from evaluation import calculate_retrieval_metrics
    expected_set = set(url for url in expected_urls if url and url.strip())
    metrics = calculate_retrieval_metrics(list(expected_set), deduplicated_urls)
    
    # Add timing information
    total_time = time.perf_counter() - start_time
    metrics.update({
        'decomposition_time': decomposition_time,
        'retrieval_time': retrieval_time,
        'reranking_time': reranking_time,
        'total_time': total_time,
        'num_sub_queries': len(sub_queries),
        'retrieved_passages_count': len(all_results),
        'retrieved_docs_count': len(deduplicated_urls)
    })
    
    # Print results
    if verbose:
        print(f"\n{'='*80}")
        print(f"MULTI-SHOT RETRIEVAL RESULTS")
        print(f"{'='*80}")
        print(f"Original Query: {original_query[:100]}...")
        print(f"Sub-queries: {len(sub_queries)}")
        print(f"Expected ({len(expected_set)}): {sorted(list(expected_set)[:3])}{'...' if len(expected_set) > 3 else ''}")
        print(f"Retrieved ({len(all_results)} passages, {len(deduplicated_urls)} unique docs): {deduplicated_urls[:3]}{'...' if len(deduplicated_urls) > 3 else ''}")
        matches = len(expected_set.intersection(set(deduplicated_urls)))
        print(f"Matches: {matches}")
        print(f"\nMetrics:")
        print(f"  P@N: {metrics.get('precision@N', 0.0):.3f}")
        print(f"  R@N: {metrics.get('recall@N', 0.0):.3f}")
        print(f"  F1@N: {metrics.get('f1@N', 0.0):.3f}")
        print(f"  MAP: {metrics.get('average_precision', 0.0):.3f}")
        print(f"\nTiming:")
        print(f"  Decomposition: {decomposition_time*1000:.1f}ms")
        print(f"  Retrieval: {retrieval_time*1000:.1f}ms")
        if reranking_time > 0:
            print(f"  Reranking: {reranking_time*1000:.1f}ms")
        print(f"  Total: {total_time*1000:.1f}ms")
        print(f"{'='*80}\n")
    
    return metrics


def run_multi_shot_evaluation(rag_db, dataset_path: str,
                              max_sub_queries: int = 3,
                              top_k_retriever: int = 50,
                              top_k_reranking: int = 10,
                              max_queries: Optional[int] = None,
                              no_rerank: bool = False,
                              retrieval_strategy: str = "fixed_k",
                              reasoning_effort: str = "medium",
                              detailed_analysis: bool = False,
                              difficulty: int = 0,
                              **strategy_params) -> Dict[str, float]:
    """
    Run multi-shot evaluation on a dataset.
    
    Args:
        rag_db: RAG database instance
        dataset_path: Path to dataset TSV file
        max_sub_queries: Maximum number of sub-queries per query
        top_k_retriever: Number of documents to retrieve per sub-query
        top_k_reranking: Number of documents after final reranking
        max_queries: Maximum number of queries to evaluate
        no_rerank: Skip reranking step
        retrieval_strategy: Strategy for retrieval
        reasoning_effort: LLM reasoning level
        detailed_analysis: Enable detailed complexity-based analysis
        difficulty: Minimum number of answer links required (0 = no filtering)
        **strategy_params: Additional parameters for retrieval strategy
        
    Returns:
        Dictionary of averaged metrics
    """
    
    df = pd.read_csv(dataset_path, sep='\t')
    
    # Filter by difficulty if specified
    df = filter_dataset_by_difficulty(df, difficulty)
    
    if isinstance(max_queries, int) and max_queries > 0:
        df = df.head(max_queries)
    else:
        max_queries = len(df)
    
    print(f"\n{'='*80}")
    print(f"MULTI-SHOT EVALUATION")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_path}")
    print(f"Queries: {max_queries}")
    print(f"Max sub-queries: {max_sub_queries}")
    print(f"Retrieval strategy: {retrieval_strategy}")
    print(f"LLM reasoning effort: {reasoning_effort}")
    print(f"Detailed analysis: {detailed_analysis}")
    if difficulty > 0:
        print(f"Difficulty filter: >= {difficulty} answer links")
    print(f"{'='*80}\n")
    
    total_metrics = {}
    valid_queries = 0
    all_query_metrics = []  # For detailed analysis
    
    for idx, row in df.iterrows():
        print(f"\n[Query {idx+1}/{max_queries}]")
        
        # Extract expected URLs
        expected_urls = []
        for col in df.columns:
            if col.startswith('wikipedia_link_') and pd.notna(row[col]):
                expected_urls.append(row[col].strip())
        
        if expected_urls:
            metrics = multi_shot_retrieval(
                rag_db, row['Prompt'], expected_urls,
                max_sub_queries=max_sub_queries,
                top_k_retriever=top_k_retriever,
                top_k_reranking=top_k_reranking,
                no_rerank=no_rerank,
                retrieval_strategy=retrieval_strategy,
                verbose=True,
                reasoning_effort=reasoning_effort,
                **strategy_params
            )
            
            # Accumulate metrics
            for metric_name, value in metrics.items():
                if metric_name not in total_metrics:
                    total_metrics[metric_name] = 0.0
                total_metrics[metric_name] += value
            
            valid_queries += 1
            
            # Collect metrics for detailed analysis
            if detailed_analysis:
                all_query_metrics.append(metrics.copy())
    
    if valid_queries > 0:
        # Calculate averages
        avg_metrics = {name: total / valid_queries for name, total in total_metrics.items()}
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"MULTI-SHOT EVALUATION SUMMARY ({valid_queries} queries)")
        print(f"{'='*80}")
        print(f"\nPRECISION METRICS:")
        print(f"  Precision@N:                {avg_metrics.get('precision@N', 0.0):.3f}")
        print(f"\nRECALL METRICS:")
        print(f"  Recall@N:                   {avg_metrics.get('recall@N', 0.0):.3f}")
        print(f"\nF1 METRICS:")
        print(f"  F1@N:                       {avg_metrics.get('f1@N', 0.0):.3f}")
        print(f"\nRANKING METRICS:")
        print(f"  Mean Average Precision:     {avg_metrics.get('average_precision', 0.0):.3f}")
        print(f"\nRETRIEVAL STATISTICS:")
        print(f"  Avg Sub-queries:            {avg_metrics.get('num_sub_queries', 0.0):.1f}")
        print(f"  Avg Passages Retrieved:     {avg_metrics.get('retrieved_passages_count', 0.0):.1f}")
        print(f"  Avg Unique Docs (N):        {avg_metrics.get('retrieved_docs_count', 0.0):.1f}")
        print(f"\nTIMING:")
        print(f"  Avg Decomposition Time:     {avg_metrics.get('decomposition_time', 0.0)*1000:.1f}ms")
        print(f"  Avg Retrieval Time:         {avg_metrics.get('retrieval_time', 0.0)*1000:.1f}ms")
        if avg_metrics.get('reranking_time', 0.0) > 0:
            print(f"  Avg Reranking Time:         {avg_metrics.get('reranking_time', 0.0)*1000:.1f}ms")
        print(f"  Avg Total Time:             {avg_metrics.get('total_time', 0.0)*1000:.1f}ms")
        print(f"{'='*80}\n")
        
        # Print detailed analysis if requested
        if detailed_analysis and all_query_metrics:
            from evaluation import _print_detailed_analysis
            _print_detailed_analysis(df, all_query_metrics, valid_queries)
        
        return avg_metrics
    else:
        print("No valid queries found!")
        return {}


if __name__ == "__main__":
    args = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                   description="Multi-shot retrieval with query decomposition")
    
    # Add all standard parameters
    add_all_args(args)
    
    # Add multi-shot specific parameters
    args.add_argument('--max-sub-queries', type=int, default=3,
                     help='Maximum number of sub-queries to generate (default: 3)')
    args.add_argument('--reasoning', type=str, default='medium',
                     choices=['low', 'medium', 'high'],
                     help='LLM reasoning level for query decomposition (default: medium)')
    
    # Special handling for --eval argument
    for action in args._actions:
        if '--eval' in action.option_strings:
            action.type = lambda x: int(x) if x.isdigit() else True
            action.const = True
            break
    
    args = args.parse_args()
    
    # Set deterministic seeds
    set_deterministic_seeds(args.seed)
    
    # Initialize database
    if args.retrieval_method == "bm25":
        db_class = BM25DB
    else:
        db_class = VectorDB
    
    if args.database is None:
        args.database = db_class.get_default_db_name()
    
    db_file_path = args.database if args.database.endswith('.db') else f"{args.database}.db"
    db_base_name = args.database.replace('.db', '') if args.database.endswith('.db') else args.database
    
    rag_db = db_class(
        retriever_model=args.retriever_model, 
        reranker_model=args.reranker_model, 
        device=args.device,
        k1=args.bm25_k1, b=args.bm25_b, method=args.bm25_method, 
        database=db_base_name,
        delta=args.bm25_delta, backend=args.bm25_backend, 
        stopwords=args.bm25_stopwords,
        show_progress=args.bm25_show_progress, stemmer=args.bm25_stemmer,
        vector_index_method=args.vector_index_method, 
        ivf_nprobe=args.ivf_nprobe,
        load_embeddings=args.load_embeddings, 
        num_embedding_devices=args.num_embedding_devices,
        benchmark=args.benchmark
    )
    
    # Load database
    if os.path.exists(db_file_path):
        print(f"Loading existing database from {db_file_path}")
        rag_db.from_serialized(db_file_path)
    else:
        raise ValueError(f"Database not found: {db_file_path}. Please create it first using single_shot_retrieval.py")
    
    # Build strategy parameters
    strategy_params = {"max_results": args.max_results}
    if args.retrieval_strategy == "top_p":
        strategy_params["p"] = args.top_p
    elif args.retrieval_strategy == "relative":
        strategy_params["ratio"] = args.relative_ratio
    
    # Run evaluation or single query
    if args.eval:
        max_queries = args.eval if isinstance(args.eval, int) and not isinstance(args.eval, bool) and args.eval > 0 else None
        
        metrics = run_multi_shot_evaluation(
            rag_db, args.dataset,
            max_sub_queries=args.max_sub_queries,
            top_k_retriever=args.top_k_retriever,
            top_k_reranking=args.top_k_reranking,
            max_queries=max_queries,
            no_rerank=args.no_rerank,
            retrieval_strategy=args.retrieval_strategy,
            reasoning_effort=args.reasoning,
            detailed_analysis=True,  # Enable detailed complexity analysis
            difficulty=args.difficulty,
            **strategy_params
        )
        
        # Save results
        results_data = {
            "multi_shot": True,
            "max_sub_queries": args.max_sub_queries,
            "reasoning_effort": args.reasoning,
            "metrics": metrics
        }
        
        with open("multi_shot_results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to multi_shot_results.json")
        
    else:
        # Single query multi-shot retrieval
        if not args.query:
            args.query = "Who won the French Open Mens Singles tournament the year that New York City FC won their first MLS Cup title?"
        
        print(f"\nRunning multi-shot retrieval for single query...")
        multi_shot_retrieval(
            rag_db, args.query, expected_urls=[],
            max_sub_queries=args.max_sub_queries,
            top_k_retriever=args.top_k_retriever,
            top_k_reranking=args.top_k_reranking,
            no_rerank=args.no_rerank,
            retrieval_strategy=args.retrieval_strategy,
            verbose=True,
            reasoning_effort=args.reasoning,
            **strategy_params
        )
