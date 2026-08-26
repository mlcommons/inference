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

"""
MLPerf Loadgen entry point for RAG-DB workload.
Initializes QSL/SUT for datasetup, configures loadgen settings, and runs the test.
"""

import os
import argparse
from pathlib import Path

import mlperf_loadgen as lg
from reference_SUT_datasetup import DatasetupSUT


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MLPerf Loadgen for RAG-DB Benchmark"
    )

    # Loadgen-specific arguments
    parser.add_argument(
        "--scenario",
        choices=["Offline", "Server"],
        default="Offline",
        help="Loadgen scenario (default: Offline)"
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Enable accuracy mode (logs all responses for evaluation)"
    )
    parser.add_argument(
        "--mlperf_conf",
        default="mlperf.conf",
        help="MLPerf rules config file"
    )
    parser.add_argument(
        "--user_conf",
        default="user.conf",
        help="User config for LoadGen settings"
    )
    parser.add_argument(
        "--audit_conf",
        default="audit.conf",
        help="Audit config for compliance runs"
    )
    parser.add_argument(
        "--log_dir",
        default="loadgen_logs_datasetup",
        help="Directory for loadgen output logs"
    )
    parser.add_argument(
        "--output_dir",
        default="output_datasetup",
        help="Directory for SUT output files"
    )

    # Datasetup workload arguments
    parser.add_argument(
        "--documents_dir",
        default="doc_html",
        help="Directory containing HTML documents to index"
    )
    parser.add_argument(
        "--database",
        default="vector_html_hnsw_len768_ov32_word",
        help="Database name/path prefix (without .db extension)"
    )

    # Chunking configuration
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=768,
        help="Chunk size in characters (default: 768)"
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=32,
        help="Chunk overlap in characters (default: 32)"
    )
    parser.add_argument(
        "--text_boundary",
        choices=["sentence", "word", "none"],
        default="word",
        help="Text boundary optimization (default: word)"
    )

    # Model paths
    parser.add_argument(
        "--retriever_model",
        default="intfloat_e5-base-v2/e5-base-v2",
        help="Path to retriever/embedding model"
    )
    parser.add_argument(
        "--reranker_model",
        default="colbert-ir_colbertv2.0/colbertv2.0",
        help="Path to reranker model"
    )

    # Device configuration
    parser.add_argument(
        "--device",
        choices=["auto", "xpu", "cuda", "hpu", "cpu"],
        default="auto",
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--num_embedding_devices",
        type=int,
        default=1,
        help="Number of devices for parallel embedding generation"
    )

    # Vector database configuration
    parser.add_argument(
        "--vector_index_method",
        choices=["flat", "hnsw", "ivf"],
        default="hnsw",
        help="FAISS index method (default: hnsw)"
    )

    # Performance options
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable detailed performance benchmarking"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of worker threads for parallel processing (default: 4)"
    )

    args = parser.parse_args()
    return args


# Scenario mapping
scenario_map = {
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
}


def main():
    """Main entry point."""
    args = get_args()
    print(f"Arguments: {args}")

    # Create log directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize SUT
    print("\n" + "=" * 80)
    print("Initializing RAG-DB SUT...")
    print("=" * 80)

    sut = DatasetupSUT(
        documents_dir=args.documents_dir,
        database=args.database,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        text_boundary=args.text_boundary,
        retriever_model=args.retriever_model,
        reranker_model=args.reranker_model,
        device=args.device,
        num_embedding_devices=args.num_embedding_devices,
        vector_index_method=args.vector_index_method,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        benchmark=args.benchmark,
        args=args,
    )

    print("\n" + "=" * 80)
    print("SUT initialization complete")
    print("=" * 80 + "\n")

    # Configure loadgen settings
    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]

    # Set test mode based on --accuracy flag
    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
        print("Running in ACCURACY mode")
    else:
        settings.mode = lg.TestMode.PerformanceOnly
        print("Running in PERFORMANCE mode")

    # Load config files
    if os.path.exists(args.user_conf):
        settings.FromConfig(args.user_conf, "e2e-rag-db", args.scenario)
        print(f"Loaded user config from {args.user_conf}")
    else:
        print(f"Warning: User config not found: {args.user_conf}")
        print("Using default loadgen settings")

    # Configure log output
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.log_dir
    log_output_settings.copy_summary_to_stdout = True

    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    # Run loadgen test
    print("\n" + "=" * 80)
    print("Running MLPerf Loadgen test...")
    print("=" * 80 + "\n")

    lg.StartTestWithLogSettings(
        sut.sut,
        sut.qsl.qsl,
        settings,
        log_settings,
        args.audit_conf
    )

    print("\n" + "=" * 80)
    print("Loadgen test complete")
    print("=" * 80 + "\n")

    # Finalize SUT (batch index, save database, cleanup)
    sut.finalize()

    # Save results
    results_path = os.path.join(args.output_dir, "datasetup_results.json")
    sut.save_results(results_path)
    print(f"Results saved to {results_path}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
