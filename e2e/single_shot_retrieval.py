import argparse
import json
import time
import os
import pandas as pd
from retrieve import VectorDB, BM25DB
from evaluation import evaluate_retrieval_query, run_evaluation
from utils import set_deterministic_seeds

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
    args.add_argument("--bm25_stemmer", type=str, default=None, choices=["porter", "snowball", "lancaster", "pystemmer"],
                      help="Stemmer for BM25 tokenization: 'porter' (balanced), 'snowball' (modern), 'lancaster' (aggressive), 'pystemmer' (fast C-based)")
    args.add_argument("--vector_index_method", type=str, default="hnsw", 
                      choices=["flat", "hnsw", "ivf"], 
                      help="Vector index method: 'flat' (exact search, slow), 'hnsw' (approximate, fast, default), 'ivf' (inverted file, memory efficient)")
    args.add_argument("--ivf_nprobe", type=int, default=10,
                      help="IVF nprobe parameter: number of clusters to search per query (1-100). Higher = better accuracy but slower. Default: 10")
    args.add_argument("--no-save", action="store_true", help="Skip saving database to disk (useful for optimization trials)")
    args.add_argument("--no-rerank", action="store_true", help="Skip reranking step for fair comparison between retrieval methods")
    args.add_argument("--retrieval_strategy", type=str, default="fixed_k", 
                      choices=["fixed_k", "top_p", "relative"], 
                      help="Retrieval strategy: 'fixed_k' (traditional), 'top_p' (nucleus sampling), 'relative' (score-based)")
    args.add_argument("--top_k_retriever", type=int, default=25)
    args.add_argument("--top_k_reranking", type=int, default=10)
    args.add_argument("--top_p", type=float, default=0.9, help="Top-p threshold for nucleus sampling (0.8-0.95)")
    args.add_argument("--relative_ratio", type=float, default=0.75, help="Relative threshold ratio (0.7-0.9)")
    args.add_argument("--max_results", type=int, default=100, help="Maximum results to consider for adaptive methods")
    args.add_argument("--seed", type=int, default=42, 
                     help="Random seed for reproducible results (default: 42)")
    args.add_argument("--benchmark", action="store_true",
                     help="Run ingestion performance benchmarking")

    args = args.parse_args()    # Set deterministic seeds for reproducible results
    set_deterministic_seeds(args.seed)

    # Initialize the appropriate database class
    if args.retrieval_method == "bm25":
        db_class = BM25DB
    else:
        db_class = VectorDB

    # Set default database path based on database class if not provided
    if args.database is None:
        args.database = db_class.get_default_db_name()
    
    # Normalize database path: ensure .db extension for file operations
    db_file_path = args.database if args.database.endswith('.db') else f"{args.database}.db"
    db_base_name = args.database.replace('.db', '') if args.database.endswith('.db') else args.database

    # Create database instance (pass base name without .db)
    rag_db = db_class(retriever_model=args.retriever_model, reranker_model=args.reranker_model, device=args.device, 
                        k1=args.bm25_k1, b=args.bm25_b, method=args.bm25_method, database=db_base_name,
                        delta=args.bm25_delta, backend=args.bm25_backend, stopwords=args.bm25_stopwords, 
                        show_progress=args.bm25_show_progress, stemmer=args.bm25_stemmer, 
                        vector_index_method=args.vector_index_method, ivf_nprobe=args.ivf_nprobe, 
                        benchmark=args.benchmark)

    if os.path.exists(db_file_path):
        # Load existing database
        print(f"Loading existing database from {db_file_path}")
        rag_db.from_serialized(db_file_path)
    else:
        if not args.ingest:
            raise ValueError("Either --database (existing) or --ingest (to create new) must be provided")
        
        # Ingest from file or folder
        tic = time.time()
        rag_db.ingest_from_path(args.ingest, num_threads=args.threads)
        
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

    # Run evaluation or single query lookup
    if args.eval:
        import json
        max_queries = args.eval if isinstance(args.eval, int) and not isinstance(args.eval, bool) and args.eval > 0 else None
        metrics = run_evaluation(rag_db, args.dataset, 
                      top_k_retriever=args.top_k_retriever, top_k_reranking=args.top_k_reranking,
                      max_queries=max_queries, no_rerank=args.no_rerank,
                      retrieval_strategy=args.retrieval_strategy, top_p=args.top_p, 
                      relative_ratio=args.relative_ratio, max_results=args.max_results)
        
        # Save results for optimization
        results_data = {
            "accuracy": metrics.get('legacy_score', 0.0),  # Backward compatibility
            "metrics": metrics
        }
        
        with open("results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        exit(0)  # Exit after evaluation
    else:
        # Single query lookup - reuse evaluation code for consistency
        import time
        
        strategy_params = {}
        if args.retrieval_strategy == "top_p":
            strategy_params["p"] = args.top_p
        elif args.retrieval_strategy == "relative":
            strategy_params["ratio"] = args.relative_ratio
        
        # Time the retrieval
        tic = time.time()
        evaluate_retrieval_query(rag_db, args.query, expected_urls=[], 
                                top_k_retriever=args.top_k_retriever, top_k_reranking=args.top_k_reranking,
                                verbose=False, no_rerank=getattr(args, 'no_rerank', False), 
                                retrieval_strategy=args.retrieval_strategy, print_results=True,
                                max_results=args.max_results, **strategy_params)
        toc = time.time()
        
        print(f"\nLookup took {toc - tic:.3f} seconds")
