import argparse
import json
import time
import os
from retrieve import VectorDB

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
    args.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2")
    args.add_argument("--reranker_model", type=str, default="colbert-ir/colbertv2.0", help="Model to use for reranking - unused for now")
    args.add_argument("--top_k", type=int, default=10)
    args = args.parse_args()

    vector_store = VectorDB(retriever_model=args.retriever_model, reranker_model=args.reranker_model)

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

    print(f"Looking up top-{args.top_k} passages for query:\n\n{args.query}\n\n")
    tic = time.time()
    results = vector_store.lookup(args.query, k=args.top_k)
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

    for result in reranked_results:
        # print(result)
        for r in results:
            if r.page_content == result[0]:
                print(f"{r.metadata}, score: {result[1]}")
                break
        print("-" * 50)
    breakpoint()
