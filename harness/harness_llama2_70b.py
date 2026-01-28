# ============================================================================
# harness_llama2_70b.py
# ---------------------
# Harness implementation for Llama 2 70B model
# Inherits common functionality from BaseHarness
# ============================================================================

import os
import sys
import time
import logging
import argparse
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

# Import MLPerf Loadgen
try:
    import mlperf_loadgen as lg
except ImportError:
    print("mlperf_loadgen is not installed.")
    print("Please install it from the MLPerf Inference repository.")
    sys.exit(1)

# Add harness directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import base harness (now includes dataset configuration support)
from harness.base_harness import BaseHarness


class Llama2_70BHarness(BaseHarness):
    """
    Harness for Llama 2 70B model with MLPerf Loadgen.
    
    Extends BaseHarness with Llama2_70B-specific functionality.
    Most functionality is handled by BaseHarness using dataset configuration.
    This class can be used to add model-specific customizations if needed.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Llama 2 70B harness.
        
        All arguments are passed to BaseHarness.
        The dataset configuration is automatically loaded from llama2-70b.yaml
        """
        # Ensure dataset_name is set if not provided
        if 'dataset_name' not in kwargs:
            kwargs['dataset_name'] = 'llama2-70b'
        
        # Call parent constructor
        super().__init__(**kwargs)
        
        self.logger.info("Llama 2 70B harness initialized")
    
    def run(self, user_conf: str = "user.conf", lg_model_name: str = "llama2_70b") -> Dict[str, Any]:
        """
        Run the harness test.
        
        Args:
            user_conf: User configuration file for LoadGen
            lg_model_name: Model name for LoadGen
        
        Returns:
            Dictionary with test results
        """
        # Use parent's run method which handles everything
        return super().run(user_conf=user_conf, lg_model_name=lg_model_name)
    
    # Use parent's _get_mlflow_parameters and _upload_to_mlflow methods


def main():
    """Main entry point for harness."""
    from harness.arg_parser import add_common_harness_args, parse_common_harness_args
    
    parser = argparse.ArgumentParser(description="MLPerf Harness for Llama 2 70B")
    
    # Add common harness arguments
    add_common_harness_args(parser)
    
    # Add Llama2_70B-specific arguments
    parser.add_argument("--lg-model-name", type=str, default="llama2_70b", help="LoadGen model name")
    parser.add_argument("--dataset-name", type=str, default="llama2-70b",
                       help="Dataset name for config lookup (default: llama2-70b)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Parse common arguments
    harness_config = parse_common_harness_args(args)
    
    # Add dataset_name to config
    harness_config['dataset_name'] = args.dataset_name
    
    # Create and run harness
    harness = Llama2_70BHarness(**harness_config)
    
    results = harness.run(user_conf=args.user_conf, lg_model_name=args.lg_model_name)
    
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

