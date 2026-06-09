#!/usr/bin/env python3
"""
Standalone indexing script for measuring indexing KPIs.

This script:
1. Reads chunked passages from a JSON file
2. Indexes them into a vector database
3. Measures the total time taken (including embedding generation)
4. Queries the database to get the actual number of indexed vectors
5. Logs metrics to a JSON file
"""

import argparse
import json
import time
from pathlib import Path
from retrieve import VectorDB
from params import add_all_args


def get_vector_count_from_db(db):
    """
    Query the database to get the actual number of vectors indexed.

    This verifies indexing correctness by querying the FAISS index directly
    rather than counting documents in the input file.

    Args:
        db: VectorDB instance

    Returns:
        int: Number of vectors in the database
    """
    if hasattr(db, '_vector_store') and hasattr(db._vector_store, 'index'):
        # FAISS index has ntotal property that gives the number of indexed vectors
        return db._vector_store.index.ntotal
    elif hasattr(db, '_doc_list'):
        # Fallback: count documents in the internal doc list
        return len(db._doc_list)
    else:
        raise ValueError("Cannot determine vector count from database")


def main():
    parser = argparse.ArgumentParser(
        description="Measure indexing KPIs for vector database",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Add all standard parameters
    add_all_args(parser)

    # Add specific parameters for this script
    parser.add_argument(
        '--output-metrics',
        type=str,
        default='indexing_kpi_metrics.json',
        help='Output JSON file for indexing metrics (default: indexing_kpi_metrics.json)'
    )

    args = parser.parse_args()

    # Validate required arguments
    if not args.ingest:
        parser.error("--ingest <passages_file> is required")

    if not Path(args.ingest).exists():
        parser.error(f"Input file does not exist: {args.ingest}")

    # Use VectorDB for this script
    if args.retrieval_method and args.retrieval_method != "vector":
        print(f"Warning: This script only supports vector indexing. Ignoring --retrieval_method={args.retrieval_method}")

    # Set default database name if not provided
    if args.database is None:
        args.database = VectorDB.get_default_db_name()

    db_file_path = args.database if args.database.endswith('.db') else f"{args.database}.db"
    db_base_name = args.database.replace('.db', '') if args.database.endswith('.db') else args.database

    print("=" * 80)
    print("INDEXING KPI MEASUREMENT")
    print("=" * 80)
    print(f"Input file: {args.ingest}")
    print(f"Database: {db_file_path}")
    print(f"Vector index method: {args.vector_index_method}")
    print(f"Embedding model: {args.retriever_model}")
    print(f"Device: {args.device}")
    print(f"Num embedding devices: {args.num_embedding_devices}")
    print("=" * 80)

    # Create VectorDB instance
    print("\n[1/4] Initializing VectorDB...")
    init_start = time.time()

    rag_db = VectorDB(
        retriever_model=args.retriever_model,
        reranker_model=args.reranker_model,
        device=args.device,
        database=db_base_name,
        vector_index_method=args.vector_index_method,
        ivf_nprobe=args.ivf_nprobe,
        load_embeddings=args.load_embeddings,
        num_embedding_devices=args.num_embedding_devices,
        hierarchical=args.hierarchical,
        embedding_device=args.embedding_device,
        reranker_device=args.reranker_device,
        benchmark=args.benchmark
    )

    init_time = time.time() - init_start
    print(f"✓ Initialization complete in {init_time:.2f}s")

    # Start indexing timer
    print(f"\n[2/4] Starting indexing from {args.ingest}...")
    indexing_start = time.time()

    # Ingest from file - this includes:
    # - Loading passages from JSON
    # - Generating embeddings (or loading from cache)
    # - Training index (for IVF)
    # - Adding vectors to FAISS index
    rag_db.ingest_from_path(args.ingest, num_threads=args.threads)

    indexing_end = time.time()
    indexing_duration = indexing_end - indexing_start

    print(f"✓ Indexing complete in {indexing_duration:.2f}s")

    # Query the database to get actual vector count
    print("\n[3/4] Verifying indexed vectors...")
    vector_count = get_vector_count_from_db(rag_db)
    print(f"✓ Verified {vector_count} vectors in database")

    # Save the database
    print(f"\n[4/4] Saving database to {db_file_path}...")
    save_start = time.time()
    rag_db.serialize(db_file_path)
    save_time = time.time() - save_start
    print(f"✓ Database saved in {save_time:.2f}s")

    # Calculate total time
    total_time = indexing_duration + save_time

    # Calculate throughput metrics
    docs_per_sec = vector_count / indexing_duration if indexing_duration > 0 else 0

    # Prepare metrics
    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": str(args.ingest),
        "database_file": db_file_path,
        "vector_count": vector_count,
        "indexing_time_seconds": round(indexing_duration, 2),
        "save_time_seconds": round(save_time, 2),
        "total_time_seconds": round(total_time, 2),
        "throughput_docs_per_second": round(docs_per_sec, 2),
        "configuration": {
            "vector_index_method": args.vector_index_method,
            "embedding_model": args.retriever_model,
            "device": args.device,
            "num_embedding_devices": args.num_embedding_devices,
            "hierarchical": args.hierarchical,
            "ivf_nprobe": args.ivf_nprobe if args.vector_index_method == "ivf" else None,
            "load_embeddings_cache": args.load_embeddings
        }
    }

    # Save metrics to JSON
    output_file = args.output_metrics
    print(f"\n{'=' * 80}")
    print("INDEXING KPI RESULTS")
    print(f"{'=' * 80}")
    print(f"Vectors indexed: {vector_count}")
    print(f"Indexing time: {indexing_duration:.2f}s")
    print(f"Save time: {save_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {docs_per_sec:.2f} docs/sec")
    print(f"{'=' * 80}")

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Metrics saved to {output_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
