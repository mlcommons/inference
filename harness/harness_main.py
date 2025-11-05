# ============================================================================
# harness_main.py
# ---------------
# Main entry point for MLPerf harness that routes to model-specific modules
# ============================================================================

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add harness directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import base harness
from harness.base_harness import BaseHarness

# Import common harness arguments
from harness.arg_parser import add_common_harness_args, parse_common_harness_args


def load_model_category_module(model_category: str):
    """
    Dynamically load model-specific harness module based on model category.
    
    Args:
        model_category: Category of the model (e.g., "llama3.1-8b", "llama2-70b", "deepseek-r1")
    
    Returns:
        Model harness class or None if not found
    """
    # Map model categories to module paths
    model_modules = {
        'llama3.1-8b': 'harness_llama3.1_8b',
        'llama3_1-8b': 'harness_llama3.1_8b',
        'llama31_8b': 'harness_llama3.1_8b',
        'llama2-70b': 'harness_llama2_70b',
        'llama2_70b': 'harness_llama2_70b',
        'llama270b': 'harness_llama2_70b',
        'deepseek-r1': 'harness_deepseek_r1',
        'deepseek_r1': 'harness_deepseek_r1',
    }
    
    module_path = model_modules.get(model_category.lower())
    if not module_path:
        return None
    
    try:
        # Import the module using importlib to handle dots in module names
        import importlib
        import importlib.util
        import sys
        import os
        
        # Get the directory where this file is located (harness/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the parent directory (mlperf-inference-6.0-redhat/)
        parent_dir = os.path.dirname(current_dir)
        
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


def main():
    """Main entry point for harness."""
    parser = argparse.ArgumentParser(
        description="MLPerf Inference Harness - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.model and not args.use_generic and not args.model_harness_class:
        parser.error("One of --model, --use-generic, or --model-harness-class must be provided")
    
    if not args.dataset_path:
        parser.error("--dataset-path is required")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    
    # Validate model_category if provided
    valid_categories = ['llama3.1-8b', 'llama3_1-8b', 'llama31_8b', 
                       'llama2-70b', 'llama2_70b', 'llama270b',
                       'deepseek-r1', 'deepseek_r1']
    if args.model_category and args.model_category.lower() not in [c.lower() for c in valid_categories]:
        logger.warning(f"Unknown model category: {args.model_category}. Will try to load anyway.")
    
    # Parse common arguments
    harness_config = parse_common_harness_args(args)
    
    # Add dataset_name to config
    if args.dataset_name:
        harness_config['dataset_name'] = args.dataset_name
    
    # Determine which harness class to use
    harness_class = BaseHarness
    
    if args.use_generic:
        logger.info("Using generic harness (BaseHarness)")
        harness_class = BaseHarness
    elif args.model_harness_class:
        # Try to load specific harness class
        logger.info(f"Looking for harness class: {args.model_harness_class}")
        # Search in common locations
        import importlib
        for module_name in ['harness_llama3.1_8b', 'harness_llama2_70b', 'harness_deepseek_r1']:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, args.model_harness_class):
                    harness_class = getattr(module, args.model_harness_class)
                    logger.info(f"Found harness class in {module_name}")
                    break
            except ImportError:
                continue
        else:
            logger.warning(f"Harness class '{args.model_harness_class}' not found, using BaseHarness")
    elif args.model_category:
        # Use model_category to load harness
        logger.info(f"Loading harness for model category: {args.model_category}")
        model_class = load_model_category_module(args.model_category)
        if model_class:
            harness_class = model_class
            logger.info(f"Using model-specific harness: {harness_class.__name__}")
        else:
            logger.warning(f"No harness found for category '{args.model_category}', using BaseHarness")
    elif args.model:
        # Try to auto-detect model category from model name (backward compatibility)
        logger.info(f"Auto-detecting model category from model name: {args.model}")
        # Try to extract category from model name
        model_lower = args.model.lower()
        detected_category = None
        
        # Check for common patterns
        if 'llama3.1' in model_lower or 'llama-3.1' in model_lower or 'llama3_1' in model_lower:
            if '8b' in model_lower or '8-b' in model_lower:
                detected_category = 'llama3.1-8b'
        elif 'llama2' in model_lower or 'llama-2' in model_lower:
            if '70b' in model_lower or '70-b' in model_lower:
                detected_category = 'llama2-70b'
        elif 'deepseek' in model_lower and 'r1' in model_lower:
            detected_category = 'deepseek-r1'
        
        if detected_category:
            logger.info(f"Detected model category: {detected_category}")
            model_class = load_model_category_module(detected_category)
            if model_class:
                harness_class = model_class
                logger.info(f"Using model-specific harness: {harness_class.__name__}")
            else:
                logger.info(f"No harness found for detected category '{detected_category}', using BaseHarness")
        else:
            logger.info(f"Could not auto-detect model category from '{args.model}', using BaseHarness")
        
        # Auto-detect dataset name if not provided
        if not args.dataset_name and args.model:
            harness_config['dataset_name'] = args.model.lower()
    
    # Create and run harness
    logger.info("=" * 80)
    logger.info(f"INITIALIZING HARNESS: {harness_class.__name__}")
    logger.info("=" * 80)
    
    harness = harness_class(**harness_config)
    
    # Determine user_conf and lg_model_name
    user_conf = args.user_conf if hasattr(args, 'user_conf') else "user.conf"
    lg_model_name = args.lg_model_name if hasattr(args, 'lg_model_name') and args.lg_model_name else None
    
    # If no lg_model_name provided, map from model category or use defaults
    if not lg_model_name:
        # Map model categories to LoadGen model names
        model_category_to_lg_name = {
            'llama3.1-8b': 'llama3_1-8b',
            'llama3_1-8b': 'llama3_1-8b',
            'llama31_8b': 'llama3_1-8b',
            'llama2-70b': 'llama2_70b',
            'llama2_70b': 'llama2_70b',
            'llama270b': 'llama2_70b',
            'deepseek-r1': 'deepseek_r1',
            'deepseek_r1': 'deepseek_r1',
        }
        
        # Try to get from model category first
        if args.model_category:
            lg_model_name = model_category_to_lg_name.get(args.model_category.lower())
            if lg_model_name:
                logger.info(f"Using LoadGen model name '{lg_model_name}' for model category '{args.model_category}'")
        
        # If still not set, try to detect from model name
        if not lg_model_name and args.model:
            model_lower = args.model.lower()
            if 'llama3.1' in model_lower or 'llama-3.1' in model_lower or 'llama3_1' in model_lower:
                if '8b' in model_lower or '8-b' in model_lower:
                    lg_model_name = 'llama3_1-8b'
            elif 'llama2' in model_lower or 'llama-2' in model_lower:
                if '70b' in model_lower or '70-b' in model_lower:
                    lg_model_name = 'llama2_70b'
            elif 'deepseek' in model_lower and 'r1' in model_lower:
                lg_model_name = 'deepseek_r1'
        
        # Fallback to dataset_name or model name
        if not lg_model_name:
            if hasattr(harness, 'dataset_name') and harness.dataset_name:
                # Try to map dataset_name to lg_model_name
                dataset_lower = harness.dataset_name.lower()
                if 'llama3.1' in dataset_lower or 'llama3_1' in dataset_lower:
                    lg_model_name = 'llama3_1-8b'
                elif 'llama2' in dataset_lower:
                    lg_model_name = 'llama2_70b'
                elif 'deepseek' in dataset_lower:
                    lg_model_name = 'deepseek_r1'
                else:
                    lg_model_name = harness.dataset_name
            elif args.model:
                lg_model_name = args.model.lower()
            else:
                lg_model_name = "default"
        
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

