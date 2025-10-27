import argparse
import json
import time
import os
import pandas as pd
from retrieve import VectorDB, BM25DB
from evaluation import evaluate_retrieval_query, run_evaluation
from utils import set_deterministic_seeds
from params import add_all_args

# Taken below from frames: https://huggingface.co/datasets/google/frames-benchmark
DEFAULT_QUERY = "Who won the French Open Mens Singles tournament the year that New York City FC won their first MLS Cup title?"



if __name__ == "__main__":
    args = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    
    # Add all parameters from centralized definitions
    # This includes: Common, General, BM25, Vector, Strategy, and Reranking parameters
    add_all_args(args)
    
    # Special handling for --eval argument (needs custom type)
    # Override the default eval argument with custom type
    for action in args._actions:
        if '--eval' in action.option_strings:
            action.type = lambda x: int(x) if x.isdigit() else True
            action.const = True
            break

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
                        load_embeddings=args.load_embeddings, num_embedding_devices=args.num_embedding_devices,
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
        
        # Build strategy_params with correct parameter names for filter function
        strategy_params = {"max_results": args.max_results}
        if args.retrieval_strategy == "top_p":
            strategy_params["p"] = args.top_p
        elif args.retrieval_strategy == "relative":
            strategy_params["ratio"] = args.relative_ratio
        
        # Run evaluation with detailed analysis enabled
        metrics = run_evaluation(rag_db, args.dataset, 
                      top_k_retriever=args.top_k_retriever, top_k_reranking=args.top_k_reranking,
                      max_queries=max_queries, no_rerank=args.no_rerank,
                      retrieval_strategy=args.retrieval_strategy, detailed_analysis=True,
                      difficulty=args.difficulty,
                      **strategy_params)
        
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
