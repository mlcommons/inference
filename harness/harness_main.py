# ============================================================================
# harness_main.py
# ---------------
# Main entry point for MLPerf harness that routes to model-specific modules
# ============================================================================

import os
import sys
import argparse
import logging
import shlex
import importlib
import importlib.util
from pathlib import Path

# Add harness directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import base harness
from harness.base_harness import BaseHarness

# Import common harness arguments
from harness.arg_parser import add_common_harness_args, parse_common_harness_args

# ============================================================================
# Model Category Mappings (Single Source of Truth)
# ============================================================================

# Map model category aliases to module names
MODEL_MODULES = {
    'llama3.1-8b': 'harness_llama3.1_8b',
    'llama3_1-8b': 'harness_llama3.1_8b',
    'llama31_8b': 'harness_llama3.1_8b',
    'llama2-70b': 'harness_llama2_70b',
    'llama2_70b': 'harness_llama2_70b',
    'llama270b': 'harness_llama2_70b',
    'deepseek-r1': 'harness_deepseek_r1',
    'deepseek_r1': 'harness_deepseek_r1',
    'gpt-oss-120b': 'harness_gpt_oss_120b',
    'gpt_oss_120b': 'harness_gpt_oss_120b',
    'gptoss120b': 'harness_gpt_oss_120b',
    'qwen3vl': 'harness_qwen3vl',
    'qwen3-vl': 'harness_qwen3vl',
    'qwen3_vl': 'harness_qwen3vl',
}

# Map model categories to LoadGen model names
MODEL_CATEGORY_TO_LG_NAME = {
    'llama3.1-8b': 'llama3_1-8b',
    'llama3_1-8b': 'llama3_1-8b',
    'llama31_8b': 'llama3_1-8b',
    'llama2-70b': 'llama2_70b',
    'llama2_70b': 'llama2_70b',
    'llama270b': 'llama2_70b',
    'deepseek-r1': 'deepseek_r1',
    'deepseek_r1': 'deepseek_r1',
    'gpt-oss-120b': 'gpt_oss_120b',
    'gpt_oss_120b': 'gpt_oss_120b',
    'gptoss120b': 'gpt_oss_120b',
    'qwen3vl': 'qwen3_vl',
    'qwen3_vl': 'qwen3_vl',
    'qwen3vl-vision': 'qwen3_vl',
}

# Valid model categories (derived from MODEL_MODULES keys)
VALID_CATEGORIES = list(MODEL_MODULES.keys())


# ============================================================================
# Helper Functions
# ============================================================================

def detect_model_category_from_name(name: str) -> str:
    """
    Detect model category from a model name or dataset name.
    
    Args:
        name: Model name or dataset name to analyze
        
    Returns:
        Detected model category string or None if not detected
    """
    if not name:
        return None
    
    name_lower = name.lower()
    
    # Check for common patterns
    if 'llama3.1' in name_lower or 'llama-3.1' in name_lower or 'llama3_1' in name_lower:
        if '8b' in name_lower or '8-b' in name_lower:
            return 'llama3.1-8b'
    elif 'llama2' in name_lower or 'llama-2' in name_lower:
        if '70b' in name_lower or '70-b' in name_lower:
            return 'llama2-70b'
    elif 'deepseek' in name_lower and 'r1' in name_lower:
        return 'deepseek-r1'
    elif 'gpt-oss' in name_lower or 'gpt_oss' in name_lower:
        if '120b' in name_lower or '120-b' in name_lower:
            return 'gpt-oss-120b'
    elif 'qwen3' in name_lower and ('vl' in name_lower or 'vision' in name_lower):
        return 'qwen3vl'
    
    return None


def resolve_lg_model_name(model_category: str = None, model_name: str = None, 
                          dataset_name: str = None) -> str:
    """
    Resolve LoadGen model name from various sources.
    
    Args:
        model_category: Model category (if provided)
        model_name: Model name (if provided)
        dataset_name: Dataset name (if provided)
        
    Returns:
        LoadGen model name string
    """
    # Try to get from model category first
    if model_category:
        lg_name = MODEL_CATEGORY_TO_LG_NAME.get(model_category.lower())
        if lg_name:
            return lg_name
    
    # If still not set, try to detect from model name
    if model_name:
        detected_category = detect_model_category_from_name(model_name)
        if detected_category:
            lg_name = MODEL_CATEGORY_TO_LG_NAME.get(detected_category)
            if lg_name:
                return lg_name
    
    # Fallback to dataset_name
    if dataset_name:
        detected_category = detect_model_category_from_name(dataset_name)
        if detected_category:
            lg_name = MODEL_CATEGORY_TO_LG_NAME.get(detected_category)
            if lg_name:
                return lg_name
        # If no category detected, use dataset name as-is
        return dataset_name
    
    # Final fallback
    if model_name:
        return model_name.lower()
    
    return "default"


def load_model_category_module(model_category: str):
    """
    Dynamically load model-specific harness module based on model category.
    
    Args:
        model_category: Category of the model (e.g., "llama3.1-8b", "llama2-70b", "deepseek-r1")
    
    Returns:
        Model harness class or None if not found
    """
    module_path = MODEL_MODULES.get(model_category.lower())
    if not module_path:
        return None
    
    try:
        # Get the directory where this file is located (harness/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # If module_path has dots, it might be a package path or a filename with dots
        if '.' in module_path:
            # Check if it's a package path (e.g., 'language.deepseek-r1.harness_deepseek_r1')
            try:
                # Try importing as a package path first
                module = importlib.import_module(module_path)
            except ImportError:
                # If that fails, try importing as a file (e.g., 'harness_llama3.1_8b')
                # Look for the file in the harness directory (current_dir)
                file_path = os.path.join(current_dir, module_path + '.py')
                
                if os.path.exists(file_path):
                    # Load the module from file
                    spec = importlib.util.spec_from_file_location(module_path, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        # Add harness directory to sys.path temporarily so imports work
                        if current_dir not in sys.path:
                            sys.path.insert(0, current_dir)
                        spec.loader.exec_module(module)
                    else:
                        raise ImportError(f"Could not create spec for '{module_path}'")
                else:
                    raise ImportError(f"Could not find module file '{file_path}' for '{module_path}'")
        else:
            # Simple module name without dots
            module = importlib.import_module(module_path)
        
        # Find the harness class (usually ends with "Harness")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, BaseHarness) and 
                attr != BaseHarness and
                'Harness' in attr_name):
                return attr
        
        return None
    except ImportError as e:
        logging.warning(f"Could not import model module '{module_path}': {e}")
        return None
    except Exception as e:
        logging.warning(f"Error loading model module '{module_path}': {e}")
        return None


def select_harness_class(args, logger) -> type:
    """
    Select the appropriate harness class based on command line arguments.
    
    Args:
        args: Parsed command line arguments
        logger: Logger instance
        
    Returns:
        Harness class to use
    """
    if args.use_generic:
        logger.info("Using generic harness (BaseHarness)")
        return BaseHarness
    
    if args.model_harness_class:
        # Try to load specific harness class
        logger.info(f"Looking for harness class: {args.model_harness_class}")
        # Search in modules from MODEL_MODULES (get unique module names)
        unique_modules = set(MODEL_MODULES.values())
        for module_name in unique_modules:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, args.model_harness_class):
                    logger.info(f"Found harness class in {module_name}")
                    return getattr(module, args.model_harness_class)
            except ImportError:
                continue
        logger.warning(f"Harness class '{args.model_harness_class}' not found, using BaseHarness")
        return BaseHarness
    
    if args.model_category:
        # Use model_category to load harness
        logger.info(f"Loading harness for model category: {args.model_category}")
        model_class = load_model_category_module(args.model_category)
        if model_class:
            logger.info(f"Using model-specific harness: {model_class.__name__}")
            return model_class
        else:
            logger.warning(f"No harness found for category '{args.model_category}', using BaseHarness")
            return BaseHarness
    
    if args.model:
        # Try to auto-detect model category from model name
        logger.info(f"Auto-detecting model category from model name: {args.model}")
        detected_category = detect_model_category_from_name(args.model)
        
        if detected_category:
            logger.info(f"Detected model category: {detected_category}")
            model_class = load_model_category_module(detected_category)
            if model_class:
                logger.info(f"Using model-specific harness: {model_class.__name__}")
                return model_class
            else:
                logger.info(f"No harness found for detected category '{detected_category}', using BaseHarness")
        else:
            logger.info(f"Could not auto-detect model category from '{args.model}', using BaseHarness")
    
    return BaseHarness


def main():
    """Main entry point for harness."""
    # Use ArgumentDefaultsHelpFormatter to show default values and format groups nicely
    class CategoryHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        """Custom formatter that shows defaults and formats groups nicely."""
        pass
    
    parser = argparse.ArgumentParser(
        description="MLPerf Inference Harness - Main Entry Point",
        formatter_class=CategoryHelpFormatter,
        epilog="""
Examples:
  # Run with model category and model name
  python harness_main.py --model-category llama3.1-8b --model meta-llama/Llama-3.1-8B-Instruct --dataset-path ./dataset.pkl
  
  # Run with llama2-70b category
  python harness_main.py --model-category llama2-70b --model meta-llama/Llama-2-70B-Instruct --dataset-path ./dataset.pkl
  
  # Run with generic harness (auto-detects model)
  python harness_main.py --use-generic --model meta-llama/Llama-3.1-8B-Instruct --dataset-path ./dataset.pkl --dataset-name llama3.1-8b
  
  # Use model-specific harness class
  python harness_main.py --model-harness-class Llama31_8BHarness --model meta-llama/Llama-3.1-8B-Instruct --dataset-path ./dataset.pkl
  
  # GPT-OSS-120B: Performance mode (auto-detects dataset name from path)
  python harness_main.py --model-category gpt-oss-120b --model openai/gpt-oss-120b --dataset-path ./perf_eval_ref.parquet --test-mode performance
  
  # GPT-OSS-120B: Accuracy mode (auto-detects dataset name from path)
  python harness_main.py --model-category gpt-oss-120b --model openai/gpt-oss-120b --dataset-path ./acc_eval_ref.parquet --test-mode accuracy
  
  # GPT-OSS-120B: Explicit dataset name (if auto-detection doesn't work)
  python harness_main.py --model-category gpt-oss-120b --model openai/gpt-oss-120b --dataset-path ./perf_eval_ref.parquet --dataset-name perf_eval_ref --test-mode performance
        """
    )
    
    # Add common harness arguments (includes --model)
    add_common_harness_args(parser)
    
    # Add model selection arguments
    model_group = parser.add_argument_group('Model Selection')
    # Note: --model is already added by add_common_harness_args
    # Update the help text for --model to reflect it's for the actual model name/path
    model_action = [action for action in parser._actions if action.dest == 'model']
    if model_action:
        model_action[0].help = "Model name or path (e.g., meta-llama/Llama-3.1-8B-Instruct, /path/to/model)"
    
    model_group.add_argument("--model-category", type=str, default=None,
                           help="Model category to select harness (e.g., llama3.1-8b, llama2-70b, deepseek-r1). "
                                "If not provided, will try to auto-detect from --model")
    model_group.add_argument("--model-harness-class", type=str, default=None,
                            help="Specific harness class name to use (e.g., Llama31_8BHarness)")
    model_group.add_argument("--use-generic", action="store_true",
                           help="Use generic harness (BaseHarness) instead of model-specific")
    
    # Add dataset configuration
    dataset_group = parser.add_argument_group('Dataset Configuration')
    dataset_group.add_argument("--dataset-name", type=str, default=None,
                              help="Dataset name for config lookup (auto-detected from path if not provided)")
    
    # Add LoadGen configuration
    loadgen_group = parser.add_argument_group('LoadGen Configuration')
    loadgen_group.add_argument("--lg-model-name", type=str, default=None,
                               help="LoadGen model name for config lookup (e.g., llama3_1-8b, llama2_70b, deepseek_r1). "
                                    "If not provided, will be auto-detected from model category")
    
    # Add engine arguments LAST (must be last when using REMAINDER)
    # This allows --engine-args to consume all remaining arguments, including those starting with --
    engine_group = parser.add_argument_group('Engine Arguments')
    engine_group.add_argument("--engine-args", type=str, nargs=argparse.REMAINDER, default=None,
                               help="Engine arguments to override server config (e.g., --engine-args --tensor-parallel-size 2 --gpu-memory-utilization 0.8). "
                                    "These arguments will be passed to the inference server and override any values in the server config file. "
                                    "All arguments after --engine-args are consumed as engine arguments.")
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.model and not args.use_generic and not args.model_harness_class:
        parser.error("One of --model, --use-generic, or --model-harness-class must be provided")
    
    if not args.dataset_path:
        parser.error("--dataset-path is required")
    
    # Configure logging - set to DEBUG if debug_mode is enabled
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    debug_mode = getattr(args, 'debug_mode', False)
    if debug_mode:
        log_level = logging.DEBUG
    
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    
    # Log debug mode status
    if debug_mode:
        logger.debug("Debug mode enabled - detailed logging active")
    
    # Print command line used to run the harness
    command_line = ' '.join(shlex.quote(arg) for arg in sys.argv)
    logger.info("=" * 80)
    logger.info("COMMAND LINE:")
    logger.info(command_line)
    logger.info("=" * 80)
    
    # Validate model_category if provided
    if args.model_category and args.model_category.lower() not in [c.lower() for c in VALID_CATEGORIES]:
        logger.warning(f"Unknown model category: {args.model_category}. Will try to load anyway.")
    
    # Parse common arguments
    harness_config = parse_common_harness_args(args)
    
    # Add dataset_name to config (with auto-detection from path if not provided)
    if args.dataset_name:
        harness_config['dataset_name'] = args.dataset_name
        logger.info(f"Using provided dataset name: {args.dataset_name}")
    elif args.dataset_path:
        # Auto-detect from dataset path (filename without extension)
        dataset_name_from_path = Path(args.dataset_path).stem
        harness_config['dataset_name'] = dataset_name_from_path
        logger.info(f"Auto-detected dataset name from path: '{args.dataset_path}' -> '{dataset_name_from_path}'")
    # Note: If neither dataset_name nor dataset_path is provided, dataset_name will be None
    # and will be auto-detected later in DatasetProcessor
    
    # Determine which harness class to use
    harness_class = select_harness_class(args, logger)
    
    # Create and run harness
    logger.info("=" * 80)
    logger.info(f"INITIALIZING HARNESS: {harness_class.__name__}")
    logger.info("=" * 80)
    
    harness = harness_class(**harness_config)
    
    # Write command line to file in output directory
    cmdline_file = harness.output_dir / "cmdline.txt"
    try:
        with open(cmdline_file, 'w') as f:
            f.write(command_line + '\n')
        logger.info(f"Command line written to: {cmdline_file}")
    except Exception as e:
        logger.warning(f"Failed to write command line to file: {e}")
    
    # Determine user_conf and lg_model_name
    user_conf = args.user_conf if hasattr(args, 'user_conf') else "user.conf"
    lg_model_name = args.lg_model_name if hasattr(args, 'lg_model_name') and args.lg_model_name else None
    
    # If no lg_model_name provided, resolve from various sources
    if not lg_model_name:
        lg_model_name = resolve_lg_model_name(
            model_category=args.model_category,
            model_name=args.model,
            dataset_name=harness_config.get('dataset_name')
        )
        logger.info(f"Auto-detected LoadGen model name: {lg_model_name}")
    else:
        logger.info(f"Using provided LoadGen model name: {lg_model_name}")
    
    # Run harness
    results = harness.run(user_conf=user_conf, lg_model_name=lg_model_name)
    
    if results['status'] == 'success':
        print("\n✓ Harness test completed successfully!")
        if not args.mlflow_experiment_name:
            print(f"  To upload results to MLflow later, run:")
            print(f"  python upload_to_mlflow.py --metadata-file {harness.output_dir}/mlflow_metadata.yaml")
        return 0
    else:
        print(f"\n✗ Harness test failed: {results.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

