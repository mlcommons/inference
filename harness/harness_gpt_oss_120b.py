# ============================================================================
# harness_gpt_oss_120b.py
# ------------------------
# Harness implementation for GPT-OSS-120B model
# Supports SGLang backend with proper data prep and sampling parameters
# ============================================================================

import os
import sys
import json
import logging
import argparse
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

# Import base harness
from harness.base_harness import BaseHarness


class GPTOSS120BHarness(BaseHarness):
    """
    Harness for GPT-OSS-120B model with MLPerf Loadgen.
    
    Extends BaseHarness with GPT-OSS-120B-specific functionality:
    - SGLang backend support
    - Tokenized input data preparation
    - Generation config support (sampling parameters)
    - Different parameters for accuracy vs performance
    """
    
    def __init__(self, 
                 generation_config_path: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 accuracy_temperature: Optional[float] = None,
                 accuracy_top_k: Optional[int] = None,
                 accuracy_top_p: Optional[float] = None,
                 **kwargs):
        """
        Initialize GPT-OSS-120B harness.
        
        Args:
            generation_config_path: Path to generation_config.json file
            max_tokens: Maximum tokens to generate (overrides config)
            temperature: Sampling temperature for performance mode
            top_k: Top-k sampling parameter for performance mode
            top_p: Top-p sampling parameter for performance mode
            accuracy_temperature: Sampling temperature for accuracy mode
            accuracy_top_k: Top-k sampling parameter for accuracy mode
            accuracy_top_p: Top-p sampling parameter for accuracy mode
            **kwargs: Additional arguments passed to BaseHarness
        """
        # Ensure dataset_name is set if not provided
        if 'dataset_name' not in kwargs:
            kwargs['dataset_name'] = 'gpt-oss-120b'
        
        # Ensure backend is set (default to SGLang if not specified, but don't override if already set)
        if 'server_config' not in kwargs:
            kwargs['server_config'] = {}
        backend_set = 'backend' in kwargs['server_config']
        backend_value = kwargs['server_config'].get('backend', 'sglang')
        if not backend_set:
            kwargs['server_config']['backend'] = 'sglang'
        
        # Ensure endpoint_type is completions (SGLang uses completions)
        if 'endpoint_type' not in kwargs['server_config']:
            kwargs['server_config']['endpoint_type'] = 'completions'
        
        # Call parent constructor (this initializes self.logger)
        super().__init__(**kwargs)
        
        # Log backend information after logger is initialized
        if not backend_set:
            self.logger.info("GPT-OSS-120B: defaulting backend to sglang (use --backend to override)")
        else:
            self.logger.info(f"GPT-OSS-120B: using backend: {backend_value}")
        
        # Load generation config
        self.generation_config_path = generation_config_path
        self.generation_config = self._load_generation_config()
        
        # Store sampling parameters
        self.max_tokens = max_tokens or self.generation_config.get('max_new_tokens', 32768)
        self.performance_temperature = temperature or self.generation_config.get('temperature', 1.0)
        self.performance_top_k = top_k if top_k is not None else self.generation_config.get('top_k', -1)
        self.performance_top_p = top_p or self.generation_config.get('top_p', 1.0)
        
        # Accuracy mode parameters (default to deterministic)
        self.accuracy_temperature = accuracy_temperature if accuracy_temperature is not None else 0.001
        self.accuracy_top_k = accuracy_top_k if accuracy_top_k is not None else 1
        self.accuracy_top_p = accuracy_top_p if accuracy_top_p is not None else 1.0
        
        self.logger.info("GPT-OSS-120B harness initialized")
        self.logger.info(f"Max tokens: {self.max_tokens}")
        self.logger.info(f"Performance mode params: temp={self.performance_temperature}, top_k={self.performance_top_k}, top_p={self.performance_top_p}")
        self.logger.info(f"Accuracy mode params: temp={self.accuracy_temperature}, top_k={self.accuracy_top_k}, top_p={self.accuracy_top_p}")
    
    def _load_generation_config(self) -> Dict[str, Any]:
        """Load generation configuration from JSON file."""
        if not self.generation_config_path:
            # Try to find default location
            default_path = Path(__file__).parent.parent / "language" / "gpt-oss-120b" / "generation_config.json"
            if default_path.exists():
                self.generation_config_path = str(default_path)
                self.logger.info(f"Using default generation config: {self.generation_config_path}")
            else:
                self.logger.warning("No generation config found, using defaults")
                return {}
        
        if not Path(self.generation_config_path).exists():
            self.logger.warning(f"Generation config not found: {self.generation_config_path}, using defaults")
            return {}
        
        try:
            with open(self.generation_config_path, 'r') as f:
                config = json.load(f)
            
            # Filter out comment fields (starting with _)
            gen_params = {k: v for k, v in config.items() if not k.startswith('_')}
            self.logger.info(f"Loaded generation config from {self.generation_config_path}")
            return gen_params
        except Exception as e:
            self.logger.warning(f"Failed to load generation config: {e}, using defaults")
            return {}
    
    def _pre_run_setup(self):
        """Pre-run setup: configure client with sampling parameters."""
        # Update server config with SGLang-specific settings
        if not self.server_config:
            self.server_config = {}
        
        # Add SGLang-specific client config
        if 'config' not in self.server_config:
            self.server_config['config'] = {}
        
        # Determine backend (default to sglang if not specified)
        backend = self.server_config.get('backend', 'sglang')
        
        # For vllm backend with gpt-oss-120b, default field should be text_input
        if backend == 'vllm':
            # Set input_column to text_input for vllm (vllm expects text input)
            if 'input_column' not in self.server_config:
                self.server_config['input_column'] = 'text_input'
            self.logger.info("Using vllm backend: defaulting input_column to 'text_input'")
        
        # Set SGLang endpoint and use_input_ids flag (only for SGLang)
        if backend == 'sglang':
            self.server_config['config']['use_input_ids'] = True
            self.server_config['config']['sglang_endpoint'] = '/generate'
        
        # Store sampling parameters for both modes
        self.server_config['config']['temperature'] = self.performance_temperature
        self.server_config['config']['top_k'] = self.performance_top_k
        self.server_config['config']['top_p'] = self.performance_top_p
        self.server_config['config']['max_tokens'] = self.max_tokens
        
        # Store accuracy mode parameters
        self.server_config['config']['accuracy_temperature'] = self.accuracy_temperature
        self.server_config['config']['accuracy_top_k'] = self.accuracy_top_k
        self.server_config['config']['accuracy_top_p'] = self.accuracy_top_p
        
        self.logger.info(f"Configured sampling parameters:")
        self.logger.info(f"  Performance mode: temp={self.performance_temperature}, top_k={self.performance_top_k}, top_p={self.performance_top_p}")
        self.logger.info(f"  Accuracy mode: temp={self.accuracy_temperature}, top_k={self.accuracy_top_k}, top_p={self.accuracy_top_p}")
        self.logger.info(f"  max_tokens: {self.max_tokens}")
    
    def run(self, user_conf: str = "user.conf", lg_model_name: str = "gpt-oss-120b") -> Dict[str, Any]:
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


def main():
    """Main entry point for harness."""
    from harness.arg_parser import add_common_harness_args, parse_common_harness_args
    
    parser = argparse.ArgumentParser(description="MLPerf Harness for GPT-OSS-120B")
    
    # Add common harness arguments
    add_common_harness_args(parser)
    
    # Add GPT-OSS-120B-specific arguments
    parser.add_argument("--lg-model-name", type=str, default="gpt-oss-120b", help="LoadGen model name")
    parser.add_argument("--dataset-name", type=str, default="gpt-oss-120b",
                       help="Dataset name for config lookup (default: gpt-oss-120b)")
    parser.add_argument("--generation-config", type=str, default=None,
                       help="Path to generation_config.json file")
    parser.add_argument("--max-tokens", type=int, default=None,
                       help="Maximum tokens to generate (overrides config)")
    parser.add_argument("--temperature", type=float, default=None,
                       help="Sampling temperature for performance mode")
    parser.add_argument("--top-k", type=int, default=None,
                       help="Top-k sampling parameter for performance mode")
    parser.add_argument("--top-p", type=float, default=None,
                       help="Top-p sampling parameter for performance mode")
    parser.add_argument("--accuracy-temperature", type=float, default=None,
                       help="Sampling temperature for accuracy mode (default: 0.001)")
    parser.add_argument("--accuracy-top-k", type=int, default=None,
                       help="Top-k sampling parameter for accuracy mode (default: 1)")
    parser.add_argument("--accuracy-top-p", type=float, default=None,
                       help="Top-p sampling parameter for accuracy mode (default: 1.0)")
    
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
    
    # Add generation config parameters
    if args.generation_config:
        harness_config['generation_config_path'] = args.generation_config
    if args.max_tokens:
        harness_config['max_tokens'] = args.max_tokens
    if args.temperature is not None:
        harness_config['temperature'] = args.temperature
    if args.top_k is not None:
        harness_config['top_k'] = args.top_k
    if args.top_p is not None:
        harness_config['top_p'] = args.top_p
    if args.accuracy_temperature is not None:
        harness_config['accuracy_temperature'] = args.accuracy_temperature
    if args.accuracy_top_k is not None:
        harness_config['accuracy_top_k'] = args.accuracy_top_k
    if args.accuracy_top_p is not None:
        harness_config['accuracy_top_p'] = args.accuracy_top_p
    
    # Create and run harness
    harness = GPTOSS120BHarness(**harness_config)
    
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
