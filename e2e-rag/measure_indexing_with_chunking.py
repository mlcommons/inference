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
Data Setup KPI Measurement

This script measures the complete data setup pipeline:
1. Document loading and chunking (from raw HTML/PDF)
2. Embedding generation
3. Vector indexing into FAISS
4. Database serialization and validation

Measures end-to-end time from raw documents to indexed vector database.
"""

import argparse
import json
import time
import os
import tempfile
from pathlib import Path
from retrieve import VectorDB
from params import add_all_args
from read_docs import DocumentProcessor


def get_vector_count_from_db(db):
    """
    Query the database to get the actual number of vectors indexed.
    """
    if hasattr(db, '_vector_store') and hasattr(db._vector_store, 'index'):
        return db._vector_store.index.ntotal
    elif hasattr(db, '_doc_list'):
        return len(db._doc_list)
    else:
        raise ValueError("Cannot determine vector count from database")


def validate_database(db, expected_passages=None):
    """
    Comprehensive database validation.
    Returns a dictionary with validation results.
    """
    validation = {
        "status": "success",
        "vector_count": 0,
        "docstore_count": 0,
        "index_type": None,
        "index_dimension": None,
        "validation_passed": True,
        "errors": []
    }

    try:
        # Get vector count
        if hasattr(db, '_vector_store') and hasattr(db._vector_store, 'index'):
            validation["vector_count"] = db._vector_store.index.ntotal
            validation["index_dimension"] = db._vector_store.index.d

            # Get index type - simplify IndexHNSWFlat to FAISS-HNSW
            index = db._vector_store.index
            index_class = type(index).__name__
            if index_class == "IndexHNSWFlat":
                validation["index_type"] = "FAISS-HNSW"
            elif hasattr(index, 'metric_type'):
                validation["index_type"] = f"FAISS-{index_class}"
            else:
                validation["index_type"] = index_class

        # Get docstore count
        if hasattr(db, '_doc_list'):
            validation["docstore_count"] = len(db._doc_list)
        elif hasattr(db, '_vector_store') and hasattr(db._vector_store, 'docstore'):
            validation["docstore_count"] = len(db._vector_store.docstore._dict)

        # Validate vector count matches docstore count
        if validation["vector_count"] != validation["docstore_count"]:
            validation["validation_passed"] = False
            validation["errors"].append(
                f"Mismatch: {validation['vector_count']} vectors but {validation['docstore_count']} documents"
            )

        # Validate against expected passages if provided
        if expected_passages is not None:
            if validation["vector_count"] != expected_passages:
                validation["validation_passed"] = False
                validation["errors"].append(
                    f"Expected {expected_passages} passages but got {validation['vector_count']} vectors"
                )

        if not validation["validation_passed"]:
            validation["status"] = "failed"

    except Exception as e:
        validation["status"] = "error"
        validation["validation_passed"] = False
        validation["errors"].append(str(e))

    return validation


def main():
    parser = argparse.ArgumentParser(
        description="Measure complete indexing KPI including chunking",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Add all standard parameters
    add_all_args(parser)

    # Chunking/Document Processing Parameters
    parser.add_argument(
        '--documents',
        type=str,
        help='Directory containing raw documents (HTML/PDF files) for chunking'
    )

    parser.add_argument(
        '--chunk_size',
        type=int,
        default=768,
        help='Chunk size in characters (default: 768)'
    )

    parser.add_argument(
        '--chunk_overlap',
        type=int,
        default=32,
        help='Chunk overlap in characters (default: 32)'
    )

    parser.add_argument(
        '--text_boundary',
        choices=["sentence", "word", "none"],
        default="word",
        help='Text boundary optimization (default: word)'
    )

    parser.add_argument(
        '--chunking_processes',
        type=int,
        default=4,
        help='Number of parallel processes for chunking (default: 4)'
    )

    # Output parameters
    parser.add_argument(
        '--output_metrics',
        type=str,
        default='data_setup_kpi.json',
        help='Output JSON file for metrics (default: data_setup_kpi.json)'
    )

    parser.add_argument(
        '--save_passages',
        type=str,
        help='Save chunked passages to this JSON file (optional)'
    )

    # Database handling
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing database without warning'
    )

    parser.add_argument(
        '--skip_if_exists',
        action='store_true',
        help='Skip saving if database file already exists'
    )

    args = parser.parse_args()

    # Validate required arguments
    if not args.documents and not args.ingest:
        parser.error("Either --documents (for raw docs) or --ingest (for pre-chunked passages) is required")

    # Set default database name if not provided
    if args.database is None:
        args.database = VectorDB.get_default_db_name()

    db_file_path = args.database if args.database.endswith('.db') else f"{args.database}.db"
    db_base_name = args.database.replace('.db', '') if args.database.endswith('.db') else args.database

    # Check if database already exists
    db_exists = Path(db_file_path).exists()
    if db_exists and not args.overwrite and not args.skip_if_exists:
        print(f"⚠️  WARNING: Database file already exists: {db_file_path}")
        print(f"    This will be OVERWRITTEN after indexing completes.")
        print(f"")
        print(f"    Options:")
        print(f"      1. Use different name: DATABASE=new_name")
        print(f"      2. Force overwrite: --overwrite")
        print(f"      3. Skip saving: --skip_if_exists")
        print()

    print("=" * 80)
    print("DATA SETUP KPI MEASUREMENT")
    print("=" * 80)

    if args.documents:
        print(f"Mode: Complete pipeline (chunking + indexing)")
        print(f"Documents: {args.documents}")
        print(f"Chunk size: {args.chunk_size}")
        print(f"Chunk overlap: {args.chunk_overlap}")
    else:
        print(f"Mode: Indexing only (pre-chunked passages)")
        print(f"Passages file: {args.ingest}")

    print(f"Database: {db_file_path}")
    print(f"Vector index method: HNSW (fixed)")
    print(f"Device: {args.device}")
    print(f"Num embedding devices: {args.num_embedding_devices}")
    print("=" * 80)
    print()

    # Track overall pipeline time
    pipeline_start = time.time()

    # ============================================================
    # STEP 1: DOCUMENT CHUNKING (if raw documents provided)
    # ============================================================
    chunking_time = 0
    num_documents = 0
    passages_file = args.ingest

    if args.documents:
        print("[1/5] Document Chunking and Parsing...")
        print()

        chunking_start = time.time()

        # Create passages file with sensible name if not provided
        if args.save_passages:
            passages_file = args.save_passages
        else:
            # Generate filename based on source directory and chunking parameters
            source_dir_name = os.path.basename(os.path.normpath(args.documents))
            passages_file = f"passages_{source_dir_name}_len{args.chunk_size}_ov{args.chunk_overlap}_{args.text_boundary}.json"
            print(f"  Auto-generated passages filename: {passages_file}")

        # Process documents and chunk - enable benchmark mode to track parsing
        processor = DocumentProcessor(
            preserve_tables=True,
            preserve_lists=True,
            text_boundary=args.text_boundary,
            benchmark=True,  # Always enable to track HTML parsing
            processes=args.chunking_processes
        )

        # Create output directory if needed
        temp_text_dir = tempfile.mkdtemp()

        result = processor.process_documents(
            input_dir=args.documents,
            output_dir=temp_text_dir,
            json_file=passages_file,
            max_passage_length=args.chunk_size,
            passage_overlap=args.chunk_overlap,
            fixed_length=args.chunk_size,  # Use fixed-length chunking
            fixed_overlap=args.chunk_overlap
        )

        num_documents = result.get('documents_processed', 0) if result else 0
        num_passages = result.get('passages_generated', 0) if result else 0

        chunking_end = time.time()
        chunking_time = chunking_end - chunking_start

        print(f"✓ Chunking complete in {chunking_time:.2f}s")
        print(f"  Documents processed: {num_documents}")
        print(f"  Passages generated: {num_passages}")
        print(f"  Passages file: {passages_file}")
        print()
    else:
        print("[1/5] Document Chunking - SKIPPED (using pre-chunked passages)")
        print()
        # Initialize variables for skipped chunking
        num_passages = 0

    # Validate passages file exists
    if not Path(passages_file).exists():
        print(f"ERROR: Passages file not found: {passages_file}")
        return 1

    # ============================================================
    # STEP 2: INITIALIZE VECTORDB
    # ============================================================
    print("[2/5] Initializing VectorDB...")
    init_start = time.time()

    # Only load embedding model during ingestion - reranker not needed
    rag_db = VectorDB(
        retriever_model=args.retriever_model,
        reranker_model=None,
        device=args.device,
        database=db_base_name,
        load_embeddings=args.load_embeddings,
        num_embedding_devices=args.num_embedding_devices,
        hierarchical=args.hierarchical,
        embedding_device=args.embedding_device,
        reranker_device=None,  # No reranker during ingestion
        benchmark=args.benchmark
    )

    init_time = time.time() - init_start
    print(f"✓ Initialization complete in {init_time:.2f}s")
    print()

    # ============================================================
    # STEP 3: INDEXING (Embedding + Vector Indexing)
    # ============================================================
    print(f"[3/5] Indexing passages from {passages_file}...")
    indexing_start = time.time()

    rag_db.ingest_from_path(passages_file)

    indexing_end = time.time()
    indexing_duration = indexing_end - indexing_start

    print(f"✓ Indexing complete in {indexing_duration:.2f}s")
    print()

    # ============================================================
    # STEP 4: SAVE DATABASE (part of performance measurement)
    # ============================================================
    save_time = 0
    if args.skip_if_exists and db_exists:
        print(f"[4/5] Skipping database save (file already exists)")
        print(f"ℹ️  Database file not modified: {db_file_path}")
    else:
        print(f"[4/5] Saving database to {db_file_path}...")
        save_start = time.time()
        rag_db.serialize(db_file_path)
        save_time = time.time() - save_start
        if db_exists:
            print(f"✓ Database overwritten in {save_time:.2f}s")
        else:
            print(f"✓ Database saved in {save_time:.2f}s")
    print()

    # ============================================================
    # STEP 5: VALIDATE DATABASE (after save, not part of perf)
    # ============================================================
    print("[5/5] Validating database...")
    validation_results = validate_database(rag_db, expected_passages=num_passages if args.documents else None)
    vector_count = validation_results["vector_count"]

    if validation_results["validation_passed"]:
        print(f"✓ Validation passed: {vector_count} vectors indexed")
        print(f"  - Vector count: {validation_results['vector_count']}")
        print(f"  - Docstore count: {validation_results['docstore_count']}")
        print(f"  - Index type: {validation_results['index_type']}")
        print(f"  - Index dimension: {validation_results['index_dimension']}")
    else:
        print(f"⚠️  Validation issues detected:")
        for error in validation_results["errors"]:
            print(f"  - {error}")
    print()

    # ============================================================
    # CALCULATE METRICS
    # ============================================================
    pipeline_end = time.time()

    # Data setup time = chunking + indexing + save (excludes validation)
    data_setup_time = chunking_time + indexing_duration + save_time

    # Calculate throughput based on data setup time
    docs_per_sec = vector_count / data_setup_time if data_setup_time > 0 else 0

    # Prepare metrics
    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "complete_pipeline" if args.documents else "indexing_only",
        "input_documents": args.documents if args.documents else None,
        "input_passages": passages_file,
        "database_file": db_file_path,
        "vector_count": vector_count,

        "chunking": {
            "enabled": bool(args.documents),
            "total_time_seconds": round(chunking_time, 2) if args.documents else 0,
            "documents_processed": num_documents if args.documents else None,
            "passages_generated": num_passages if args.documents else None,
            "chunk_size": args.chunk_size if args.documents else None,
            "chunk_overlap": args.chunk_overlap if args.documents else None,
            "text_boundary": args.text_boundary if args.documents else None,
            "processes": args.chunking_processes if args.documents else None
        },

        "data_setup_time_seconds": round(data_setup_time, 2),
        "throughput_passages_per_second": round(docs_per_sec, 2),

        "validation": {
            "status": validation_results["status"],
            "validation_passed": validation_results["validation_passed"],
            "vector_count": validation_results["vector_count"],
            "docstore_count": validation_results["docstore_count"],
            "index_type": validation_results["index_type"],
            "index_dimension": validation_results["index_dimension"],
            "errors": validation_results["errors"]
        },

        "configuration": {
            "vector_index_method": "hnsw",
            "embedding_model": args.retriever_model,
            "device": args.device,
            "num_embedding_devices": args.num_embedding_devices,
            "hierarchical": args.hierarchical,
            "load_embeddings_cache": args.load_embeddings
        }
    }

    # Save metrics to JSON
    output_file = args.output_metrics
    print(f"{'=' * 80}")
    print("DATA SETUP KPI RESULTS")
    print(f"{'=' * 80}")

    if args.documents:
        print(f"Documents processed: {num_documents}")
        print(f"Passages generated: {num_passages}")
        print(f"Chunking time: {chunking_time:.2f}s")
        print()

    print(f"Vectors indexed: {vector_count}")
    print(f"Data setup time: {data_setup_time:.2f}s")
    print(f"Throughput: {docs_per_sec:.2f} passages/sec")
    print()
    print(f"Validation: {validation_results['status'].upper()}")
    if not validation_results["validation_passed"]:
        print(f"  Issues: {', '.join(validation_results['errors'])}")
    print(f"{'=' * 80}")
    print()

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print()
    print("Done!")

    # Cleanup temporary text directory used during processing
    if args.documents:
        import shutil
        try:
            shutil.rmtree(temp_text_dir)
        except:
            pass

    return 0


if __name__ == "__main__":
    exit(main())
