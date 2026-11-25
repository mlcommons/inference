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
Centralized Parameter Definitions

This module provides a single source of truth for all parameters
used across main_qna.py, rag/multi_shot_retrieval.py,
and other scripts.

Usage:
    from common.params import add_all_args, add_common_args, add_retrieval_args

    parser = argparse.ArgumentParser()
    add_all_args(parser)  # Add all parameters
    # OR
    add_common_args(parser)  # Add common args (ingest, database, query, etc.)
    add_retrieval_args(parser)  # Add retrieval-specific args
    args = parser.parse_args()
"""

from typing import Dict, Any, List, Optional
import argparse

# ============================================================================
# Parameter Definitions
# ============================================================================


class ParamDef:
    """Parameter definition with metadata."""

    def __init__(self,
                 name: str,
                 arg_names: List[str],
                 type: type,
                 default: Any,
                 help: str,
                 choices: Optional[List[Any]] = None,
                 nargs: Optional[str] = None,
                 action: Optional[str] = None,
                 category: str = "general",
                 applies_to: List[str] = None):
        """
        Args:
            name: Internal parameter name
            arg_names: List of CLI argument names (e.g., ['--retrieval_top_p'])
            type: Parameter type (int, float, str, bool)
            default: Default value
            help: Help text
            choices: Valid choices (for categorical parameters)
            nargs: argparse nargs value
            action: argparse action (e.g., 'store_true', 'store_false')
            category: Parameter category (general, vector, strategy, reranking)
            applies_to: List of methods this applies to (['vector', 'both'])
        """
        self.name = name
        self.arg_names = arg_names
        self.type = type
        self.default = default
        self.help = help
        self.choices = choices
        self.nargs = nargs
        self.action = action
        self.category = category
        self.applies_to = applies_to or ["both"]

    def add_to_parser(self, parser: argparse.ArgumentParser):
        """Add this parameter to an argument parser."""
        kwargs = {
            'help': self.help,
            'default': self.default,
        }

        if self.action:
            kwargs['action'] = self.action
        else:
            kwargs['type'] = self.type

        if self.choices:
            kwargs['choices'] = self.choices

        if self.nargs:
            kwargs['nargs'] = self.nargs

        parser.add_argument(*self.arg_names, **kwargs)

# ============================================================================
# Common Script Parameters
# ============================================================================


COMMON_PARAMS = [
    ParamDef(
        name="database",
        arg_names=["--database", "--db"],
        type=str,
        default="vector.db",
        help="Path to the vector database file",
        category="common",
        applies_to=["both"]
    ),
    ParamDef(
        name="dataset",
        arg_names=["--dataset"],
        type=str,
        default="frames-benchmark-dataset/frames_dataset.tsv",
        help="Path to evaluation dataset file",
        category="common",
        applies_to=["both"]
    ),
    ParamDef(
        name="embedding_model",
        arg_names=["--embedding_model"],
        type=str,
        default="intfloat_e5-base-v2/e5-base-v2",
        help="Embedding model for passages and queries (local path).",
        category="common",
        applies_to=["vector"]
    ),
    ParamDef(
        name="reranker_model",
        arg_names=["--reranker_model"],
        type=str,
        default="colbert-ir_colbertv2.0/colbertv2.0",
        help="Reranker model (ColBERTv2): local path or Hugging Face id",
        category="common",
        applies_to=["both"]
    ),
    ParamDef(
        name="no_rerank",
        arg_names=["--no-rerank"],
        type=bool,
        default=False,
        help="Skip reranking step for fair comparison between retrieval methods",
        action="store_true",
        category="common",
        applies_to=["both"]
    ),
    ParamDef(
        name="max_results",
        arg_names=["--max_results"],
        type=int,
        default=20,
        help="Maximum results to consider for adaptive methods",
        category="common",
        applies_to=["both"]
    ),
    ParamDef(
        name="benchmark",
        arg_names=["--benchmark"],
        type=bool,
        default=False,
        help="Run ingestion performance benchmarking",
        action="store_true",
        category="common",
        applies_to=["both"]
    ),
    ParamDef(
        name="perf_test_mode",
        arg_names=["--perf-test-mode"],
        type=str,
        default=None,
        help="Path to LLM log file for performance testing (replays cached LLM responses)",
        category="common",
        applies_to=["both"]
    ),
]

# ============================================================================
# General Parameters
# ============================================================================

GENERAL_PARAMS = [
    ParamDef(
        name="device",
        arg_names=["--device"],
        type=str,
        default="auto",
        help="Device to use (auto/cuda/rocm/xpu/hpu/cpu)",
        choices=["auto", "cuda", "rocm", "xpu", "hpu", "cpu"],
        category="general",
        applies_to=["both"]
    ),
    ParamDef(
        name="num_embedding_devices",
        arg_names=["--num_embedding_devices"],
        type=int,
        default=1,
        help="Number of devices to use for parallel embedding generation (supports XPU, CUDA, CPU)",
        category="general",
        applies_to=["vector"]
    ),
    # LLM endpoints: one URL + model pair per role.
    #   query, sufficiency (+ final answer) -> 120B   grader -> 20B   judge -> evaluation/accuracy_eval_qna.py
    # An unset role falls back along: sufficiency -> query -> grader.
    ParamDef(
        name="query_service_url",
        arg_names=["--query_service_url"],
        type=str,
        default="http://127.0.0.1:8123/v1/chat/completions",
        help="Query-generation endpoint (gpt-oss-120b, generate_search_queries).",
        category="general",
        applies_to=["both"]
    ),
    ParamDef(
        name="query_model",
        arg_names=["--query_model"],
        type=str,
        default="gpt-oss_model/gpt-oss-120b",
        help="Query-generation model id, as served by --query_service_url.",
        category="general",
        applies_to=["both"]
    ),
    ParamDef(
        name="grader_service_url",
        arg_names=["--grader_service_url"],
        type=str,
        default="http://127.0.0.1:8192/v1/chat/completions",
        help="Document-grading endpoint (gpt-oss-20b, evaluate_document_relevance). "
             "Also probed for max_tokens when --max_tokens=auto.",
        category="general",
        applies_to=["both"]
    ),
    ParamDef(
        name="grader_model",
        arg_names=["--grader_model"],
        type=str,
        default="auto",
        help="Document-grading model id ('auto' = ask --grader_service_url).",
        category="general",
        applies_to=["both"]
    ),
    ParamDef(
        name="sufficiency_service_url",
        arg_names=["--sufficiency_service_url"],
        type=str,
        default=None,
        help="Sufficiency-check and final-answer endpoint "
             "(check_sufficiency, generate_answer). Default: --query_service_url.",
        category="general",
        applies_to=["both"]
    ),
    ParamDef(
        name="sufficiency_model",
        arg_names=["--sufficiency_model"],
        type=str,
        default=None,
        help="Sufficiency-check and final-answer model id. Default: --query_model.",
        category="general",
        applies_to=["both"]
    ),
    ParamDef(
        name="max_tokens",
        arg_names=["--max_tokens"],
        type=str,
        default="auto",
        help="Maximum tokens for LLM response (auto to detect from service, or specify number)",
        category="general",
        applies_to=["both"]
    ),
]

# ============================================================================
# Retrieval Strategy Parameters
# ============================================================================

STRATEGY_PARAMS = [
    ParamDef(
        name="retrieval_strategy",
        arg_names=["--retrieval_strategy"],
        type=str,
        default="fixed_k",
        help="Strategy for selecting number of retrieved documents",
        choices=["fixed_k", "top_p", "relative"],
        category="strategy",
        applies_to=["both"]
    ),
    ParamDef(
        name="top_k_retriever",
        arg_names=["--top_k_retriever"],
        type=int,
        default=10,
        help="Number of documents to retrieve (for fixed_k strategy or max for dynamic strategies)",
        category="strategy",
        applies_to=["both"]
    ),
    ParamDef(
        name="retrieval_top_p",
        arg_names=["--retrieval_top_p"],
        type=float,
        default=0.9,
        help="Cumulative score threshold for --retrieval_strategy top_p (0.0-1.0). "
             "Retrieval only, unrelated to LLM sampling top_p.",
        category="strategy",
        applies_to=["both"]
    ),
    ParamDef(
        name="relative_ratio",
        arg_names=["--relative_ratio"],
        type=float,
        default=0.8,
        help="Score ratio threshold for relative strategy (0.0-1.0)",
        category="strategy",
        applies_to=["both"]
    ),
]

# ============================================================================
# Reranking Parameters
# ============================================================================

RERANKING_PARAMS = [
    ParamDef(
        name="top_k_reranking",
        arg_names=["--top_k_reranking"],
        type=int,
        default=10,
        help="Number of documents to return after reranking",
        category="reranking",
        applies_to=["both"]
    ),
]

# ============================================================================
# All Parameters
# ============================================================================

ALL_PARAMS = (
    COMMON_PARAMS +
    GENERAL_PARAMS +
    STRATEGY_PARAMS +
    RERANKING_PARAMS
)

# Create lookup dictionaries
PARAM_BY_NAME = {p.name: p for p in ALL_PARAMS}
PARAM_BY_CLI_NAME = {}
for p in ALL_PARAMS:
    for arg_name in p.arg_names:
        PARAM_BY_CLI_NAME[arg_name] = p

# Category groupings
PARAMS_BY_CATEGORY = {
    "common": COMMON_PARAMS,
    "general": GENERAL_PARAMS,
    "strategy": STRATEGY_PARAMS,
    "reranking": RERANKING_PARAMS,
}

# Method-specific parameters
VECTOR_METHOD_PARAMS = [
    p for p in ALL_PARAMS if "vector" in p.applies_to or "both" in p.applies_to]

# ============================================================================
# Helper Functions
# ============================================================================


def add_common_args(parser: argparse.ArgumentParser):
    """
    Add common script parameters (ingest, database, query, etc.)

    Args:
        parser: ArgumentParser to add arguments to
    """
    for param in COMMON_PARAMS:
        param.add_to_parser(parser)


def add_retrieval_args(parser: argparse.ArgumentParser,
                       method: Optional[str] = None,
                       categories: Optional[List[str]] = None):
    """
    Add retrieval parameters to an argument parser.
    Includes: General, Vector, Strategy, and Reranking parameters.
    Does NOT include common script parameters (use add_common_args for those).

    Args:
        parser: ArgumentParser to add arguments to
        method: Filter by method ('vector', or None for all)
        categories: Filter by categories (e.g., ['general', 'strategy'])
    """
    # Get non-common params
    params_to_add = [p for p in ALL_PARAMS if p.category != "common"]

    # Filter by method
    if method:
        params_to_add = [p for p in params_to_add
                         if method in p.applies_to or "both" in p.applies_to]

    # Filter by category
    if categories:
        params_to_add = [p for p in params_to_add if p.category in categories]

    # Add to parser
    for param in params_to_add:
        param.add_to_parser(parser)


def add_all_args(parser: argparse.ArgumentParser,
                 method: Optional[str] = None):
    """
    Add all parameters (common + retrieval) to an argument parser.

    Args:
        parser: ArgumentParser to add arguments to
        method: Filter by method ('vector', or None for all)
    """
    add_common_args(parser)
    add_retrieval_args(parser, method=method)


def print_param_info(method: Optional[str] = None):
    """Print parameter information grouped by category."""

    if method == "vector":
        params = VECTOR_METHOD_PARAMS
    else:
        params = ALL_PARAMS

    # Group by category
    by_category = {}
    for param in params:
        if param.category not in by_category:
            by_category[param.category] = []
        by_category[param.category].append(param)

    # Print
    for category in ["common", "general", "vector", "strategy", "reranking"]:
        if category not in by_category:
            continue

        print(f"\n{category.upper()} Parameters:")
        print("=" * 60)

        for param in by_category[category]:
            print(f"  {param.name}")
            print(f"    CLI: {', '.join(param.arg_names)}")
            print(f"    Default: {param.default}")
            print(f"    Help: {param.help[:80]}..." if len(
                param.help) > 80 else f"    Help: {param.help}")
            if param.choices:
                print(f"    Choices: {param.choices}")

# ============================================================================
# Main (for testing/documentation)
# ============================================================================


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "list":
        method = sys.argv[2] if len(sys.argv) > 2 else None
        print_param_info(method)
    else:
        print("Usage: python -m common.params list [vector]")
