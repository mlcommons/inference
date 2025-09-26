import argparse
import json
import time
import os
import pandas as pd
from retrieve import VectorDB

def evaluate_query(vector_store, query, expected_urls, top_k_retriever=50, top_k_reranking=10):
    """Evaluate a single query and return score (0-1)."""
    # Get retrieval results
    results = vector_store.lookup(query, k=top_k_retriever)
    top_k_passages = [result.page_content for result in results]
    
    # Get reranking results  
    reranked_results = vector_store.rerank(query, top_k_passages)
    
    # Extract URLs from top reranked results
    retrieved_urls = set()
    for reranked_passage, score in reranked_results[:top_k_reranking]:
        # Find the corresponding result metadata
        for result in results:
            if result.page_content == reranked_passage:
                if 'original_url' in result.metadata and result.metadata['original_url']:
                    retrieved_urls.add(result.metadata['original_url'])
                break
    
    # Calculate score
    expected_set = set(url for url in expected_urls if url and url.strip())
    if not expected_set:
        return 1.0 if not retrieved_urls else 0.0
    
    matches = len(expected_set.intersection(retrieved_urls))
    score = matches / len(expected_set)
    
    print(f"Query: {query[:100]}...")
    print(f"Expected ({len(expected_set)}): {sorted(list(expected_set)[:3])}{'...' if len(expected_set) > 3 else ''}")
    print(f"Retrieved ({len(retrieved_urls)}): {sorted(list(retrieved_urls)[:3])}{'...' if len(retrieved_urls) > 3 else ''}")
    print(f"Matches: {matches}, Score: {score:.3f}")
    print("-" * 80)
    
    return score

def run_evaluation(vector_store, dataset_path, top_k_retriever=50, top_k_reranking=10):
    """Run evaluation on all queries in dataset."""
    df = pd.read_csv(dataset_path, sep='\t')
    print(f"\nRunning evaluation on {len(df)} queries from dataset")
    
    total_score = 0.0
    valid_queries = 0
    
    for idx, row in df.iterrows():
        # Extract expected Wikipedia links
        expected_urls = []
        for col in df.columns:
            if col.startswith('wikipedia_link_') and pd.notna(row[col]):
                expected_urls.append(row[col].strip())
        
        if expected_urls:
            score = evaluate_query(vector_store, row['Prompt'], expected_urls, top_k_retriever, top_k_reranking)
            total_score += score
            valid_queries += 1
    
    if valid_queries > 0:
        avg_score = total_score / valid_queries
        print(f"\nEvaluation complete: {avg_score:.3f} average score ({valid_queries} queries)")
    else:
        print("No valid queries found!")

# Taken below from frames: https://huggingface.co/datasets/google/frames-benchmark
DEFAULT_QUERY = "Who won the French Open Mens Singles tournament the year that New York City FC won their first MLS Cup title?"
if __name__ == "__main__":
    args = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    args.add_argument("--passages", type=str, default=None, help="Path to the JSON array file with passages\n"
                        "'passage' will be the passage text\n"
                        "all other keys will be metadata\n"
                        "Example: [{'index': int, 'pdf_filename': str, 'passage': str}]\n"
                        "Ignored if --vector_store is provided")
    args.add_argument("--vector_store", type=str, default="vector.db", help="Path to the vector store file\n"
                        "If provided, --passages will be ignored")
    args.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Query to search for")
    args.add_argument("--dataset", type=str, default="data/frames_dataset.tsv")
    args.add_argument("--device", type=str, default="auto", help="Device to run the models on (e.g., 'cpu', 'cuda', 'xpu', or 'auto')")
    args.add_argument("--eval", action="store_true", help="Run evaluation on dataset")
    args.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2")
    args.add_argument("--reranker_model", type=str, default="colbert-ir/colbertv2.0", help="Model to use for reranking - unused for now")
    args.add_argument("--top_k_retriever", type=int, default=50)
    args.add_argument("--top_k_reranking", type=int, default=10)
    args = args.parse_args()

    vector_store = VectorDB(retriever_model=args.retriever_model, reranker_model=args.reranker_model, device=args.device)

    if args.vector_store and os.path.exists(args.vector_store):
        # TODO: incremental ingestion using existing DB
        assert (args.vector_store is None) != (args.passages is None), "Exactly one of --vector_store or --passages must be provided"
        vector_store.from_serialized(args.vector_store)
    else:
        passage_data = json.load(open(args.passages))
        passage_list = [p.pop('passage') for p in passage_data]
        passage_metadata = [p for p in passage_data] # All keys except 'passage' are metadata

        print(f"Ingesting {len(passage_list)} passages from {args.passages}")
        tic = time.time()
        vector_store.ingest(passage_list, passage_metadata)
        toc = time.time()
        ingestion_speed = len(passage_list)/(toc-tic)
        print(f"Ingestion of {len(passage_list)} passages took {toc - tic} seconds. {ingestion_speed:.2f} docs/sec")
        vector_store.serialize(args.vector_store)

    # Run evaluation if requested
    if args.eval:
        run_evaluation(vector_store, args.dataset, 
                      top_k_retriever=args.top_k_retriever, top_k_reranking=args.top_k_reranking)
        exit(0)  # Exit after evaluation

    print(f"Looking up top-{args.top_k_retriever} passages for query:\n\n{args.query}\n\n")
    tic = time.time()
    results = vector_store.lookup(args.query, k=args.top_k_retriever)
    toc = time.time()
    print(f"Lookup took {toc - tic} seconds. Results are below:")

    # Display which PDFs the top-k passages are from
    for result in results:
        print(result.metadata)
        print("-" * 50)
    breakpoint()

    top_k_passages = [result.page_content for result in results]

    print(f"Reranking {len(results)} passages")
    tic = time.time()
    reranked_results = vector_store.rerank(args.query, top_k_passages)
    toc = time.time()
    print(f"Reranking took {toc - tic} seconds. Results are below:")

    for result in reranked_results[:args.top_k_reranking]:
        # print(result)
        for r in results:
            if r.page_content == result[0]:
                print(f"{r.metadata}, score: {result[1]}")
                break
        print("-" * 50)
    breakpoint()
