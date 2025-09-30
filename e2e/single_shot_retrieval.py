import argparse
import json
import time
import os
import pandas as pd
from retrieve import VectorDB, BM25DB

def evaluate_query(rag_db, query, expected_urls, top_k_retriever=50, top_k_reranking=10):
    """Evaluate a single query and return score (0-1)."""
    # Get retrieval results with reranking
    results = rag_db.lookup_with_rerank(query, k=top_k_reranking, rerank_k=top_k_retriever)
    
    # Extract URLs from results
    retrieved_urls = set()
    for result in results:
        if 'original_url' in result.metadata and result.metadata['original_url']:
            retrieved_urls.add(result.metadata['original_url'])
    
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

def run_evaluation(rag_db, dataset_path, top_k_retriever=50, top_k_reranking=10, max_queries=None):
    """Run evaluation on all queries in dataset."""
    df = pd.read_csv(dataset_path, sep='\t')
    
    # Limit number of queries if specified as a positive integer (not boolean)
    if isinstance(max_queries, int) and not isinstance(max_queries, bool) and max_queries > 0:
        df = df.head(max_queries)
        print(f"\nRunning evaluation on {len(df)} queries (limited from {max_queries} requested)")
    else:
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
            score = evaluate_query(rag_db, row['Prompt'], expected_urls, top_k_retriever, top_k_reranking)
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
    args.add_argument("--ingest", type=str, default=None, help="Path to ingest data from:\n"
                        "  - JSON array file: 'passage' will be the passage text, all other keys will be metadata\n"
                        "    Example: [{'index': int, 'pdf_filename': str, 'passage': str}]\n"
                        "  - Folder: For BM25, ingests all .txt files in the folder\n"
                        "Ignored if --database is provided")
    args.add_argument("--database", "--db", type=str, default=None, help="Path to the database file\n"
                        "If provided, --ingest will be ignored\n"
                        "Default: 'bm25.db' for BM25, 'vector.db' for vector")
    args.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Query to search for")
    args.add_argument("--dataset", type=str, default="data/frames_dataset.tsv")
    args.add_argument("--device", type=str, default="auto", help="Device to run the models on (e.g., 'cpu', 'cuda', 'xpu', or 'auto')")
    args.add_argument("--eval", nargs="?", const=True, type=lambda x: int(x) if x.isdigit() else True, 
                     help="Run evaluation on dataset. Optionally specify number of queries to evaluate (e.g., --eval 100)")
    args.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2")
    args.add_argument("--reranker_model", type=str, default="colbert-ir/colbertv2.0", help="Model to use for reranking - unused for now")
    args.add_argument("--retrieval_method", type=str, default="bm25", choices=["bm25", "vector"], 
                      help="Retrieval method: 'bm25' for BM25 lexical search, 'vector' for dense vector search")
    args.add_argument("--threads", type=int, default=4, help="Number of threads for BM25 retrieval (BM25 only). Indexing is single-threaded by default")
    args.add_argument("--bm25_k1", type=float, default=None, help="BM25 k1 parameter (term frequency saturation). Higher values = more weight on term frequency. Default: 1.5")
    args.add_argument("--bm25_b", type=float, default=None, help="BM25 b parameter (document length normalization). 0=no normalization, 1=full normalization. Default: 0.75")
    args.add_argument("--bm25_method", type=str, default=None, choices=["lucene", "robertson", "bm25+"], 
                      help="BM25 variant: 'lucene' (default), 'robertson' (original), 'bm25+' (improved)")
    args.add_argument("--bm25_delta", type=float, default=None, help="BM25 delta parameter (for BM25L/BM25+). Default: 0.5")
    args.add_argument("--bm25_backend", type=str, default=None, choices=["numpy", "numba", "auto"],
                      help="BM25 backend: 'numpy' (default), 'numba' (faster), 'auto' (detect)")
    args.add_argument("--bm25_stopwords", type=str, default=None, help="Stopwords for BM25 tokenization. Default: 'en'")
    args.add_argument("--bm25_show_progress", action="store_true", help="Show progress bars during BM25 indexing")
    args.add_argument("--no-save", action="store_true", help="Skip saving database to disk (useful for optimization trials)")
    args.add_argument("--top_k_retriever", type=int, default=50)
    args.add_argument("--top_k_reranking", type=int, default=10)
    args = args.parse_args()

    # Initialize the appropriate database class
    if args.retrieval_method == "bm25":
        db_class = BM25DB
    else:
        db_class = VectorDB

    # Set default database path based on database class if not provided
    if args.database is None:
        args.database = db_class.get_default_db_name()
    
    # Normalize database path: ensure .db extension for file operations
    if args.retrieval_method == "bm25":
        db_file_path = BM25DB.get_db_path(args.database)  # Always has .db extension
        db_base_name = args.database.replace('.db', '') if args.database.endswith('.db') else args.database
    else:
        db_file_path = args.database if args.database.endswith('.db') else f"{args.database}.db"
        db_base_name = args.database.replace('.db', '') if args.database.endswith('.db') else args.database

    # Create database instance (pass base name without .db)
    rag_db = db_class(retriever_model=args.retriever_model, reranker_model=args.reranker_model, device=args.device, 
                        k1=args.bm25_k1, b=args.bm25_b, method=args.bm25_method, database=db_base_name,
                        delta=args.bm25_delta, backend=args.bm25_backend, stopwords=args.bm25_stopwords, 
                        show_progress=args.bm25_show_progress)

    if os.path.exists(db_file_path):
        # Load existing database
        print(f"Loading existing database from {db_file_path}")
        rag_db.from_serialized(db_file_path)
    else:
        if not args.ingest:
            raise ValueError("Either --database (existing) or --ingest (to create new) must be provided")
        
        # Ingest data from source (file or folder)
        tic = time.time()
        rag_db.ingest_from_source(args.ingest, num_threads=args.threads)
        
        # Get number of passages for timing calculation
        num_passages = len(rag_db._doc_list)  # This should be available after ingestion
        toc = time.time()
        ingestion_speed = num_passages/(toc-tic)
        print(f"Ingestion of {num_passages} passages took {toc - tic:.2f} seconds. {ingestion_speed:.2f} docs/sec")
        
        # Save the database (unless --no-save is specified)
        if not args.no_save:
            print(f"Saving database to {db_file_path}")
            rag_db.serialize(db_file_path)
        else:
            print("Skipping database save (--no-save specified)")

    # Run evaluation if requested
    if args.eval:
        max_queries = args.eval if isinstance(args.eval, int) and not isinstance(args.eval, bool) and args.eval > 0 else None
        run_evaluation(rag_db, args.dataset, 
                      top_k_retriever=args.top_k_retriever, top_k_reranking=args.top_k_reranking,
                      max_queries=max_queries)
        exit(0)  # Exit after evaluation

    print(f"Looking up top-{args.top_k_retriever} passages for query:\n\n{args.query}\n\n")
    tic = time.time()
    results = rag_db.lookup(args.query, k=args.top_k_retriever)
    toc = time.time()
    print(f"Lookup took {toc - tic} seconds. Results are below:")

    # Display which PDFs the top-k passages are from
    for result in results:
        print(result.metadata)
        print("-" * 50)

    # Show reranked results if reranker is available
    if rag_db._reranker_model is not None:
        print(f"\nReranking to top-{args.top_k_reranking}")
        tic = time.time()
        reranked_results = rag_db.lookup_with_rerank(args.query, k=args.top_k_reranking, rerank_k=args.top_k_retriever)
        toc = time.time()
        print(f"Reranking took {toc - tic} seconds. Results are below:")

        for i, result in enumerate(reranked_results, 1):
            print(f"{i}. {result.metadata}")
            print(f"   {result.page_content[:200]}...")
            print("-" * 50)
    else:
        print("No reranker available - showing retrieval results only")
