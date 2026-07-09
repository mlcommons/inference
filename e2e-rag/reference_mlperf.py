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
MLPerf Loadgen entry point for RAG-QnA workload.
Initializes QSL/SUT, configures loadgen settings, and runs the test.
"""

import os
import argparse
import subprocess
from pathlib import Path

import mlperf_loadgen as lg
from reference_SUT import E2ESUT
from params import add_all_args


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MLPerf Loadgen for RAG-QnA Multi-hop RAG Benchmark"
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
        help="Enable accuracy pass (vs. performance)"
    )
    parser.add_argument(
        "--mlperf_conf",
        default="mlperf.conf",
        help="MLPerf rules config file"
    )
    parser.add_argument(
        "--user_conf",
        default="user.conf",
        help="User config for LoadGen settings (e.g., target QPS)"
    )
    parser.add_argument(
        "--audit_conf",
        default="audit.conf",
        help="Audit config for compliance runs"
    )
    parser.add_argument(
        "--log_dir",
        default="loadgen_logs",
        help="Directory for loadgen output logs"
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory for SUT output files (results, LLM logs)"
    )

    # Add all standard e2e parameters from params.py first
    # This includes --database, --device, etc.
    add_all_args(parser)

    # E2E workload arguments (non-conflicting with params.py)
    parser.add_argument(
        "--dataset_path",
        default="data/frames_dataset.tsv",
        help="Path to frames_dataset.tsv"
    )
    parser.add_argument(
        "--perf_count",
        type=int,
        default=None,
        help="Number of queries for performance testing (None = all)"
    )

    # Multi-shot specific parameters (these are unique to
    # multi_shot_retrieval.py)
    parser.add_argument(
        '--max-sub-queries',
        type=int,
        default=3,
        help='Maximum number of sub-queries per iteration (default: 3)'
    )
    parser.add_argument(
        '--reasoning',
        type=str,
        default='medium',
        choices=['low', 'medium', 'high'],
        help='LLM reasoning level (default: medium)'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=10,
        help='Maximum retrieval iterations (default: 10)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='LLM sampling temperature (default: 1.0)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=5,
        help='Max retries for LLM calls (default: 5)'
    )

    # Judge service configuration for accuracy evaluation
    parser.add_argument(
        '--judge_service_url',
        default='http://127.0.0.1:8125/v1/chat/completions',
        help='Judge LLM service URL for accuracy evaluation (default: local vLLM)'
    )
    parser.add_argument(
        '--judge_model',
        default='meta-llama/Llama-3.1-8B-Instruct',
        help='Judge LLM model name (default: Llama-3.1-8B-Instruct)'
    )

    # Query service configuration (separate from main LLM service)
    parser.add_argument(
        '--query_service_url',
        default=None,
        help='Query generation service URL (if different from main LLM service)'
    )

    # Threading configuration for parallel query processing
    parser.add_argument(
        '--max_workers',
        type=int,
        default=10,
        help='Maximum number of worker threads for parallel query processing (default: 10)'
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
    print("Initializing RAG-QnA SUT...")
    print("=" * 80)

    sut = E2ESUT(
        dataset_path=args.dataset_path,
        db_path=args.database,
        max_sub_queries=args.max_sub_queries,
        top_k_retriever=args.top_k_retriever,
        top_k_reranking=args.top_k_reranking,
        max_iterations=args.max_iterations,
        no_rerank=args.no_rerank,
        retrieval_strategy=args.retrieval_strategy,
        reasoning_effort=args.reasoning,
        perf_count=args.perf_count,
        device=args.device,
        temperature=args.temperature,
        max_retries=args.max_retries,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        args=args,  # Pass full args for additional params
    )

    print("\n" + "=" * 80)
    print("SUT initialization complete")
    print("=" * 80 + "\n")

    # Configure loadgen settings
    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]

    # Load config files
    if os.path.exists(args.user_conf):
        settings.FromConfig(args.user_conf, "rag-qna", args.scenario)
        print(f"Loaded user config from {args.user_conf}")
    else:
        print(f"Warning: User config not found: {args.user_conf}")
        print("Using default loadgen settings")

    # Set test mode
    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
        print("Running in ACCURACY mode")
    else:
        settings.mode = lg.TestMode.PerformanceOnly
        print("Running in PERFORMANCE mode")

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

    # Finalize SUT (save logs, cleanup)
    sut.finalize()

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    sut.save_results(results_path)
    print(f"Results saved to {results_path}")

    # Run accuracy evaluation if in accuracy mode
    if args.accuracy:
        print("\n" + "=" * 80)
        print("Running accuracy evaluation...")
        print("=" * 80 + "\n")

        cmd = [
            "python3",
            "accuracy_eval.py",
            "--log_dir", args.log_dir,
            "--results_file", results_path,
            "--dataset_path", args.dataset_path,
            "--judge_service_url", args.judge_service_url,
            "--judge_model", args.judge_model
        ]
        print(f"Command: {' '.join(cmd)}")
        subprocess.check_call(cmd)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
