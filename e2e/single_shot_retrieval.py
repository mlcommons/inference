import argparse
import json
import time
import os
import requests
from retrieve import VectorDB, BM25DB
from evaluation import evaluate_retrieval_query, run_evaluation
from utils import set_deterministic_seeds, setup_llm_config
from params import add_all_args

# Taken below from frames: https://huggingface.co/datasets/google/frames-benchmark
DEFAULT_QUERY = "Who won the French Open Mens Singles tournament the year that New York City FC won their first MLS Cup title?"


def _serialize_params(args):
    params = {}
    for key, value in vars(args).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            params[key] = value
        else:
            params[key] = str(value)
    return params


def _convert_results_to_entries(results, limit=5):
    entries = []
    seen_urls = set()
    for doc in results:
        if hasattr(doc, "metadata"):
            url = doc.metadata.get("original_url") or doc.metadata.get("source")
            content = doc.page_content
        else:
            url = doc.get("url")
            content = doc.get("content", "")
        if url and url in seen_urls:
            continue
        entries.append({
            "url": url,
            "content": content[:2000]
        })
        if url:
            seen_urls.add(url)
        if limit and len(entries) >= limit:
            break
    return entries


def _generate_llm_answer(query, doc_entries, llm_config):
    context_parts = []
    for idx, doc in enumerate(doc_entries, 1):
        source = doc.get("url") or "Unknown source"
        snippet = doc.get("content", "").strip()
        context_parts.append(f"[{idx}] Source: {source}\n{snippet}")
    evidence_block = "\n\n".join(context_parts) if context_parts else "No supporting documents were retrieved."
    user_prompt = (
        "Answer the question using only the provided evidence."
        " Respond with a single word or short phrase, or 'Unknown' if the evidence is insufficient.\n\n"
        f"Question:\n{query}\n\nEvidence:\n{evidence_block}"
    )
    max_tokens = llm_config["max_tokens"]
    if isinstance(max_tokens, int):
        max_tokens = min(max_tokens, 256)
    else:
        max_tokens = 256
    payload = {
        "model": llm_config["model_name"],
        "messages": [
            {
                "role": "system",
                "content": "You are a concise retrieval QA assistant who trusts the supplied context."
            },
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens
    }
    response = requests.post(llm_config["service_url"], json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()



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
    llm_config = setup_llm_config(args) if args.generate_answer else None

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
        max_queries = args.eval if isinstance(args.eval, int) and not isinstance(args.eval, bool) and args.eval > 0 else None
        
        # Build strategy_params with correct parameter names for filter function
        strategy_params = {"max_results": args.max_results}
        if args.retrieval_strategy == "top_p":
            strategy_params["p"] = args.top_p
        elif args.retrieval_strategy == "relative":
            strategy_params["ratio"] = args.relative_ratio
        
        # Run evaluation with detailed analysis enabled
        metrics_output = run_evaluation(rag_db, args.dataset, 
                      top_k_retriever=args.top_k_retriever, top_k_reranking=args.top_k_reranking,
                      max_queries=max_queries, no_rerank=args.no_rerank,
                      retrieval_strategy=args.retrieval_strategy, detailed_analysis=True,
                      difficulty=args.difficulty, collect_results=args.generate_answer,
                      **strategy_params)
        if args.generate_answer:
            metrics, collected_runs = metrics_output
        else:
            metrics = metrics_output
        
        # Save results for optimization
        results_data = {
            "accuracy": metrics.get('legacy_score', 0.0),  # Backward compatibility
            "metrics": metrics
        }
        
        with open("results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        if args.generate_answer:
            answers = []
            for run in collected_runs:
                docs = run["docs"]
                answer_docs = _convert_results_to_entries(docs, limit=5)
                answer = _generate_llm_answer(run["prompt"], answer_docs, llm_config)
                print(f"LLM Answer: {answer}")
                urls = [doc.get("url") for doc in docs if doc.get("url")]
                urls = list(dict.fromkeys(urls))
                answers.append({
                    "prompt": run["prompt"],
                    "retrieved_urls": urls,
                    "llm_answer": answer
                })
            with open("result_single_shot.json", "w") as f:
                json.dump({
                    "params": _serialize_params(args),
                    "results": answers
                }, f, indent=2)
        exit(0)  # Exit after evaluation
    else:
    # Single query lookup - reuse evaluation code for consistency
        
        strategy_params = {}
        if args.retrieval_strategy == "top_p":
            strategy_params["p"] = args.top_p
        elif args.retrieval_strategy == "relative":
            strategy_params["ratio"] = args.relative_ratio
        
        # Time the retrieval
        tic = time.time()
        eval_output = evaluate_retrieval_query(rag_db, args.query, expected_urls=[], 
                                top_k_retriever=args.top_k_retriever, top_k_reranking=args.top_k_reranking,
                                verbose=False, no_rerank=getattr(args, 'no_rerank', False), 
                                retrieval_strategy=args.retrieval_strategy, print_results=True,
                                return_results=args.generate_answer,
                                max_results=args.max_results, **strategy_params)
        if args.generate_answer:
            _, retrieved_docs = eval_output
            full_entries = _convert_results_to_entries(retrieved_docs, limit=0)
            doc_entries = full_entries[:5]
            answer = _generate_llm_answer(args.query, doc_entries, llm_config)
            print(f"\nLLM Answer: {answer}")
            urls = [doc.get("url") for doc in full_entries if doc.get("url")]
            with open("result_single_shot.json", "w") as f:
                json.dump({
                    "params": _serialize_params(args),
                    "results": [{
                        "prompt": args.query,
                        "retrieved_urls": list(dict.fromkeys(urls)),
                        "llm_answer": answer
                    }]
                }, f, indent=2)
        toc = time.time()
        
        print(f"\nLookup took {toc - tic:.3f} seconds")
