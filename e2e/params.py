"""
Centralized Parameter Definitions

This module provides a single source of truth for all parameters
used across single_shot_retrieval.py, optimize_retrieval.py, and other scripts.

Usage:
    from params import add_all_args, add_common_args, add_retrieval_args
    
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
                 applies_to: List[str] = None,
                 optuna_suggest: Optional[Dict[str, Any]] = None):
        """
        Args:
            name: Internal parameter name
            arg_names: List of CLI argument names (e.g., ['--top-p', '--top_p'])
            type: Parameter type (int, float, str, bool)
            default: Default value
            help: Help text
            choices: Valid choices (for categorical parameters)
            nargs: argparse nargs value
            action: argparse action (e.g., 'store_true', 'store_false')
            category: Parameter category (general, bm25, vector, strategy, reranking)
            applies_to: List of methods this applies to (['bm25', 'vector', 'both'])
            optuna_suggest: Optuna suggestion config (type, min, max, step, etc.)
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
        self.optuna_suggest = optuna_suggest
    
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
    
    def suggest_value(self, trial):
        """Suggest a value for Optuna trial."""
        if not self.optuna_suggest:
            raise ValueError(f"No optuna_suggest config for parameter {self.name}")
        
        config = self.optuna_suggest
        suggest_type = config['type']
        
        if suggest_type == 'float':
            return trial.suggest_float(
                self.name,
                config['min'],
                config['max'],
                step=config.get('step')
            )
        elif suggest_type == 'int':
            return trial.suggest_int(
                self.name,
                config['min'],
                config['max'],
                step=config.get('step', 1)
            )
        elif suggest_type == 'categorical':
            return trial.suggest_categorical(self.name, config['choices'])
        else:
            raise ValueError(f"Unknown suggest type: {suggest_type}")


# ============================================================================
# Common Script Parameters
# ============================================================================

COMMON_PARAMS = [
    ParamDef(
        name="ingest",
        arg_names=["--ingest"],
        type=str,
        default=None,
        help="Path to ingest data from:\n  - JSON array file: 'passage' will be the passage text, all other keys will be metadata\n    Example: [{'index': int, 'pdf_filename': str, 'passage': str}]\n  - Folder: For BM25, ingests all .txt files in the folder\nIgnored if --database is provided",
        category="common",
        applies_to=["both"]
    ),
    ParamDef(
        name="database",
        arg_names=["--database", "--db"],
        type=str,
        default="vector.db",
        help="Path to the database file\nIf provided, --ingest will be ignored\nDefault: 'bm25.db' for BM25, 'vector.db' for vector",
        category="common",
        applies_to=["both"]
    ),
    ParamDef(
        name="query",
        arg_names=["--query"],
        type=str,
        default="Who won the French Open Mens Singles tournament the year that New York City FC won their first MLS Cup title?",
        help="Query to search for",
        category="common",
        applies_to=["both"]
    ),
    ParamDef(
        name="dataset",
        arg_names=["--dataset"],
        type=str,
        default="data/frames_dataset.tsv",
        help="Path to evaluation dataset file",
        category="common",
        applies_to=["both"]
    ),
    ParamDef(
        name="eval",
        arg_names=["--eval"],
        type=str,  # Special handling needed for this one
        default=None,
        help="Run evaluation on dataset. Optionally specify number of queries to evaluate (e.g., --eval 100)",
        nargs="?",
        category="common",
        applies_to=["both"]
    ),
    ParamDef(
        name="difficulty",
        arg_names=["--difficulty"],
        type=int,
        default=0,
        help="Minimum number of answer links (difficulty level). Filters out queries with fewer answer links. Default is 0 (no filtering)",
        category="common",
        applies_to=["both"]
    ),
    ParamDef(
        name="retriever_model",
        arg_names=["--retriever_model"],
        type=str,
        default="intfloat/e5-base-v2",
        help="Model to use for embedding-based retrieval",
        category="common",
        applies_to=["vector"]
    ),
    ParamDef(
        name="reranker_model",
        arg_names=["--reranker_model"],
        type=str,
        default="colbert-ir/colbertv2.0",
        help="Model to use for reranking (currently unused)",
        category="common",
        applies_to=["both"]
    ),
    ParamDef(
        name="retrieval_method",
        arg_names=["--retrieval_method"],
        type=str,
        default="vector",
        help="Retrieval method: 'bm25' for BM25 lexical search, 'vector' for dense vector search",
        choices=["bm25", "vector"],
        category="common",
        applies_to=["both"]
    ),
    ParamDef(
        name="load_embeddings",
        arg_names=["--load-embeddings"],
        type=bool,
        default=False,
        help="Load embeddings from .emb.pkl cache if available (default: False)",
        action="store_true",
        category="common",
        applies_to=["vector"]
    ),
    ParamDef(
        name="no_save",
        arg_names=["--no-save"],
        type=bool,
        default=False,
        help="Skip saving database to disk (useful for optimization trials)",
        action="store_true",
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
        name="seed",
        arg_names=["--seed"],
        type=int,
        default=42,
        help="Random seed for reproducible results (default: 42)",
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
        name="embedding_device",
        arg_names=["--embedding-device"],
        type=str,
        default=None,
        help="Override device for the embedding model (default: inherit --device)",
        choices=["auto", "cuda", "rocm", "xpu", "hpu", "cpu"],
        category="general",
        applies_to=["vector"]
    ),
    ParamDef(
        name="reranker_device",
        arg_names=["--reranker-device"],
        type=str,
        default=None,
        help="Override device for the reranker model (default: inherit --device)",
        choices=["auto", "cuda", "rocm", "xpu", "hpu", "cpu"],
        category="general",
        applies_to=["both"]
    ),
    ParamDef(
        name="threads",
        arg_names=["--threads"],
        type=int,
        default=None,
        help="Number of threads for BM25 retrieval (default: CPU count)",
        category="general",
        applies_to=["bm25"]
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
    ParamDef(
        name="llm_service_url",
        arg_names=["--llm_service_url"],
        type=str,
        default="http://127.0.0.1:8123/v1/chat/completions",
        help="URL for the LLM service endpoint",
        category="general",
        applies_to=["both"]
    ),
    ParamDef(
        name="llm_model",
        arg_names=["--llm_model"],
        type=str,
        default="auto",
        help="LLM model name/path (auto to detect from service). Used by document grader (evaluate_document_relevance).",
        category="general",
        applies_to=["both"]
    ),
    ParamDef(
        name="query_model",
        arg_names=["--query_model"],
        type=str,
        default=None,
        help="LLM model name/path for query generation (generate_search_queries). Defaults to --llm_model if not set. Example: /model/gpt-oss-120b",
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
    ParamDef(
        name="base_doc_dir",
        arg_names=["--base-doc-dir"],
        type=str,
        default="doc_html",
        help="Directory containing full documents to use when --full-doc-context is enabled",
        category="general",
        applies_to=["both"]
    ),
    ParamDef(
        name="save_results",
        arg_names=["--save-results"],
        type=bool,
        default=True,
        help="Save experiment outputs to result_single_shot.json (use --no-save-results to disable)",
        action=argparse.BooleanOptionalAction,
        category="general",
        applies_to=["both"]
    ),
    ParamDef(
        name="full_doc_context",
        arg_names=["--full-doc-context"],
        type=bool,
        default=False,
        help="Load full source documents for LLM context instead of retrieved passages",
        action="store_true",
        category="general",
        applies_to=["both"]
    ),
    ParamDef(
        name="generate_answer",
        arg_names=["--generate-answer"],
        type=bool,
        default=False,
        help="Generate LLM answer outputs and save them alongside retrieval results",
        action="store_true",
        category="general",
        applies_to=["both"]
    ),
]

# ============================================================================
# BM25 Parameters
# ============================================================================

BM25_PARAMS = [
    ParamDef(
        name="bm25_k1",
        arg_names=["--bm25_k1"],
        type=float,
        default=1.2,
        help="BM25 k1 parameter (term frequency saturation)",
        category="bm25",
        applies_to=["bm25"],
        optuna_suggest={'type': 'float', 'min': 0.5, 'max': 3.0, 'step': 0.1}
    ),
    ParamDef(
        name="bm25_b",
        arg_names=["--bm25_b"],
        type=float,
        default=0.75,
        help="BM25 b parameter (document length normalization)",
        category="bm25",
        applies_to=["bm25"],
        optuna_suggest={'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.1}
    ),
    ParamDef(
        name="bm25_method",
        arg_names=["--bm25_method"],
        type=str,
        default="lucene",
        help="BM25 variant to use",
        choices=["lucene", "robertson", "bm25+"],
        category="bm25",
        applies_to=["bm25"],
        optuna_suggest={'type': 'categorical', 'choices': ["lucene", "robertson", "bm25+"]}
    ),
    ParamDef(
        name="bm25_delta",
        arg_names=["--bm25_delta"],
        type=float,
        default=0.5,
        help="BM25+ delta parameter (lower bound for term weights)",
        category="bm25",
        applies_to=["bm25"],
        optuna_suggest={'type': 'float', 'min': 0.0, 'max': 2.0, 'step': 0.1}
    ),
    ParamDef(
        name="bm25_stemmer",
        arg_names=["--bm25_stemmer"],
        type=str,
        default=None,
        help="Stemmer to use for BM25 (None, porter, snowball, lancaster, pystemmer)",
        choices=[None, "porter", "snowball", "lancaster", "pystemmer"],
        category="bm25",
        applies_to=["bm25"],
        optuna_suggest={'type': 'categorical', 'choices': [None, "porter", "snowball", "lancaster", "pystemmer"]}
    ),
    ParamDef(
        name="bm25_stopwords",
        arg_names=["--bm25_stopwords"],
        type=str,
        default="en",
        help="Stopwords language (en, None for no stopwords)",
        category="bm25",
        applies_to=["bm25"]
    ),
    ParamDef(
        name="no_stopwords",
        arg_names=["--no-stopwords"],
        type=bool,
        default=False,
        help="Disable stopwords filtering",
        action="store_true",
        category="bm25",
        applies_to=["bm25"]
    ),
    ParamDef(
        name="bm25_backend",
        arg_names=["--bm25_backend"],
        type=str,
        default="numba",
        help="BM25 computation backend (numpy, numba)",
        choices=["numpy", "numba"],
        category="bm25",
        applies_to=["bm25"],
        optuna_suggest={'type': 'categorical', 'choices': ["numpy", "numba"]}
    ),
    ParamDef(
        name="bm25_show_progress",
        arg_names=["--bm25_show_progress"],
        type=bool,
        default=False,
        help="Show progress bars during BM25 indexing",
        action="store_true",
        category="bm25",
        applies_to=["bm25"]
    ),
]

# ============================================================================
# Vector Parameters
# ============================================================================

VECTOR_PARAMS = [
    ParamDef(
        name="vector_index_method",
        arg_names=["--vector_index_method"],
        type=str,
        default="flat",
        help="FAISS index method (flat, hnsw, ivf)",
        choices=["flat", "hnsw", "ivf"],
        category="vector",
        applies_to=["vector"],
        optuna_suggest={'type': 'categorical', 'choices': ["flat", "hnsw", "ivf"]}
    ),
    ParamDef(
        name="ivf_nprobe",
        arg_names=["--ivf_nprobe"],
        type=int,
        default=10,
        help="Number of clusters to probe for IVF index",
        category="vector",
        applies_to=["vector"],
        optuna_suggest={'type': 'int', 'min': 1, 'max': 100, 'step': 1}
    ),
    ParamDef(
        name="hierarchical",
        arg_names=["--hierarchical"],
        type=bool,
        default=False,
        action="store_true",
        help="Enable hierarchical retrieval (search small chunks, return large parents)",
        category="vector",
        applies_to=["vector"]
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
        applies_to=["both"],
        optuna_suggest={'type': 'categorical', 'choices': ["fixed_k", "top_p", "relative"]}
    ),
    ParamDef(
        name="top_k_retriever",
        arg_names=["--top_k_retriever"],
        type=int,
        default=10,
        help="Number of documents to retrieve (for fixed_k strategy or max for dynamic strategies)",
        category="strategy",
        applies_to=["both"],
        optuna_suggest={'type': 'int', 'min': 5, 'max': 100, 'step': 5}
    ),
    ParamDef(
        name="top_p",
        arg_names=["--top_p"],
        type=float,
        default=0.9,
        help="Cumulative probability threshold for top_p strategy (0.0-1.0)",
        category="strategy",
        applies_to=["both"],
        optuna_suggest={'type': 'float', 'min': 0.5, 'max': 0.99, 'step': 0.01}
    ),
    ParamDef(
        name="relative_ratio",
        arg_names=["--relative_ratio"],
        type=float,
        default=0.8,
        help="Score ratio threshold for relative strategy (0.0-1.0)",
        category="strategy",
        applies_to=["both"],
        optuna_suggest={'type': 'float', 'min': 0.5, 'max': 0.99, 'step': 0.01}
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
        applies_to=["both"],
        optuna_suggest={'type': 'int', 'min': 1, 'max': 50, 'step': 1}
    ),
]

# ============================================================================
# All Parameters
# ============================================================================

ALL_PARAMS = (
    COMMON_PARAMS +
    GENERAL_PARAMS +
    BM25_PARAMS +
    VECTOR_PARAMS +
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
    "bm25": BM25_PARAMS,
    "vector": VECTOR_PARAMS,
    "strategy": STRATEGY_PARAMS,
    "reranking": RERANKING_PARAMS,
}

# Method-specific parameters
BM25_METHOD_PARAMS = [p for p in ALL_PARAMS if "bm25" in p.applies_to or "both" in p.applies_to]
VECTOR_METHOD_PARAMS = [p for p in ALL_PARAMS if "vector" in p.applies_to or "both" in p.applies_to]

# Optimizable parameters (those with optuna_suggest defined)
OPTIMIZABLE_PARAMS = [p for p in ALL_PARAMS if p.optuna_suggest is not None]
OPTIMIZABLE_PARAM_NAMES = [p.name for p in OPTIMIZABLE_PARAMS]

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
    Includes: General, BM25, Vector, Strategy, and Reranking parameters.
    Does NOT include common script parameters (use add_common_args for those).
    
    Args:
        parser: ArgumentParser to add arguments to
        method: Filter by method ('bm25', 'vector', or None for all)
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

def add_all_args(parser: argparse.ArgumentParser, method: Optional[str] = None):
    """
    Add all parameters (common + retrieval) to an argument parser.
    
    Args:
        parser: ArgumentParser to add arguments to
        method: Filter by method ('bm25', 'vector', or None for all)
    """
    add_common_args(parser)
    add_retrieval_args(parser, method=method)

def get_optimizable_params(method: Optional[str] = None) -> List[ParamDef]:
    """
    Get list of parameters that can be optimized with Optuna.
    
    Args:
        method: Filter by method ('bm25', 'vector', or None for all)
    
    Returns:
        List of ParamDef objects
    """
    params = OPTIMIZABLE_PARAMS
    
    if method:
        params = [p for p in params 
                 if method in p.applies_to or "both" in p.applies_to]
    
    return params

def suggest_param(trial, param_name: str) -> Any:
    """
    Suggest a parameter value for Optuna trial.
    
    Args:
        trial: Optuna trial object
        param_name: Parameter name
    
    Returns:
        Suggested value
    """
    if param_name not in PARAM_BY_NAME:
        raise ValueError(f"Unknown parameter: {param_name}")
    
    param = PARAM_BY_NAME[param_name]
    return param.suggest_value(trial)

def get_default_params(method: str) -> Dict[str, Any]:
    """
    Get default parameter values for a method.
    
    Args:
        method: 'bm25' or 'vector'
    
    Returns:
        Dictionary of parameter name -> default value
    """
    if method == "bm25":
        params = BM25_METHOD_PARAMS
    elif method == "vector":
        params = VECTOR_METHOD_PARAMS
    else:
        params = ALL_PARAMS
    
    return {p.name: p.default for p in params}

def format_params_for_cli(params: Dict[str, Any], skip_defaults: bool = True) -> List[str]:
    """
    Format parameter dictionary as CLI arguments.
    
    Args:
        params: Dictionary of parameter name -> value
        skip_defaults: If True, skip parameters that match their default values
    
    Returns:
        List of CLI argument strings
    """
    args = []
    
    for name, value in params.items():
        if name not in PARAM_BY_NAME:
            continue
        
        param_def = PARAM_BY_NAME[name]
        
        # Skip if value matches default (when skip_defaults=True)
        if skip_defaults and value == param_def.default:
            continue
        
        cli_arg = param_def.arg_names[0]
        
        if param_def.action == "store_true":
            if value:
                args.append(cli_arg)
        elif param_def.action == "store_false":
            if not value:
                args.append(cli_arg)
        elif value is not None:
            args.extend([cli_arg, str(value)])
    
    return args

def print_param_info(method: Optional[str] = None):
    """Print parameter information grouped by category."""
    
    if method == "bm25":
        params = BM25_METHOD_PARAMS
    elif method == "vector":
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
    for category in ["common", "general", "bm25", "vector", "strategy", "reranking"]:
        if category not in by_category:
            continue
        
        print(f"\n{category.upper()} Parameters:")
        print("=" * 60)
        
        for param in by_category[category]:
            opt_marker = " [optimizable]" if param.optuna_suggest else ""
            print(f"  {param.name}{opt_marker}")
            print(f"    CLI: {', '.join(param.arg_names)}")
            print(f"    Default: {param.default}")
            print(f"    Help: {param.help[:80]}..." if len(param.help) > 80 else f"    Help: {param.help}")
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
        print("Usage: python retrieval_params.py list [bm25|vector]")
        print("\nOptimizable parameters:")
        for param in OPTIMIZABLE_PARAMS:
            print(f"  - {param.name} ({param.category})")
