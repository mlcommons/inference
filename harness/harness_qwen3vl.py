# ============================================================================
# harness_qwen3vl.py
# -------------------
# Harness implementation for Qwen3-VL multimodal model
# Supports multimodal data (text + images) with chat completions API
# 
# Note: Qwen3-VL uses multimodal messages format:
# - messages: list of ChatCompletionMessageParam with content that can be:
#   - Text: {"type": "text", "text": "..."}
#   - Image: {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}
# - response_format: Optional JSON schema for guided decoding
# 
# The dataset should contain pre-formatted messages or fields that can be
# converted to messages format (e.g., product_image, product_title, etc.)
# ============================================================================

import os
import sys
import logging
import argparse
import base64
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from io import BytesIO

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


class Qwen3VLHarness(BaseHarness):
    """
    Harness for Qwen3-VL multimodal model with MLPerf Loadgen.
    
    Extends BaseHarness with Qwen3-VL-specific functionality:
    - Multimodal data support (text + images)
    - Chat completions API with messages format
    - Proper QSL loading for multimodal samples
    - Different sampling parameters for accuracy vs performance
    
    Dataset Requirements:
    - The dataset should contain a 'messages' column with pre-formatted messages
    - Messages format: list of ChatCompletionMessageParam with:
      - System message: {"role": "system", "content": "..."}
      - User message: {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}]}
    - Alternatively, dataset can have fields like 'product_image', 'product_title', 'product_description'
      which will be automatically converted to messages format (see reference implementation)
    """
    
    def __init__(self, 
                 use_guided_decoding: bool = False,
                 accuracy_temperature: Optional[float] = None,
                 accuracy_top_k: Optional[int] = None,
                 accuracy_top_p: Optional[float] = None,
                 **kwargs):
        """
        Initialize Qwen3-VL harness.
        
        Args:
            use_guided_decoding: Whether to use guided decoding for structured outputs
            accuracy_temperature: Sampling temperature for accuracy mode
            accuracy_top_k: Top-k sampling parameter for accuracy mode
            accuracy_top_p: Top-p sampling parameter for accuracy mode
            **kwargs: Additional arguments passed to BaseHarness
        """
        # Ensure dataset_name is set if not provided
        if 'dataset_name' not in kwargs:
            kwargs['dataset_name'] = 'qwen3vl'
        
        # Ensure backend is vllm (Qwen3-VL uses vLLM)
        if 'server_config' not in kwargs:
            kwargs['server_config'] = {}
        if 'backend' not in kwargs['server_config']:
            kwargs['server_config']['backend'] = 'vllm'
        
        # Ensure endpoint_type is chat_completions (multimodal uses chat API)
        if 'endpoint_type' not in kwargs['server_config']:
            kwargs['server_config']['endpoint_type'] = 'chat_completions'
        
        # Call parent constructor
        super().__init__(**kwargs)
        
        # Store multimodal-specific settings
        self.use_guided_decoding = use_guided_decoding
        
        # Default sampling parameters
        self.performance_temperature = 0.0
        self.performance_top_k = None
        self.performance_top_p = 1.0
        
        # Accuracy mode parameters (default to deterministic)
        self.accuracy_temperature = accuracy_temperature if accuracy_temperature is not None else 0.0
        self.accuracy_top_k = accuracy_top_k
        self.accuracy_top_p = accuracy_top_p if accuracy_top_p is not None else 1.0
        
        self.logger.info("Qwen3-VL harness initialized")
        self.logger.info(f"Use guided decoding: {self.use_guided_decoding}")
        self.logger.info(f"Performance mode params: temp={self.performance_temperature}, top_p={self.performance_top_p}")
        self.logger.info(f"Accuracy mode params: temp={self.accuracy_temperature}, top_p={self.accuracy_top_p}")
    
    def _pre_run_setup(self):
        """Pre-run setup: configure client with multimodal settings."""
        # Update server config with multimodal-specific settings
        if not self.server_config:
            self.server_config = {}
        
        # Add multimodal-specific client config
        if 'config' not in self.server_config:
            self.server_config['config'] = {}
        
        # Store sampling parameters for both modes
        self.server_config['config']['temperature'] = self.performance_temperature
        self.server_config['config']['top_p'] = self.performance_top_p
        if self.performance_top_k is not None:
            self.server_config['config']['top_k'] = self.performance_top_k
        
        # Store accuracy mode parameters
        self.server_config['config']['accuracy_temperature'] = self.accuracy_temperature
        self.server_config['config']['accuracy_top_p'] = self.accuracy_top_p
        if self.accuracy_top_k is not None:
            self.server_config['config']['accuracy_top_k'] = self.accuracy_top_k
        
        # Store guided decoding setting
        self.server_config['config']['use_guided_decoding'] = self.use_guided_decoding
        
        # Mark as multimodal to use messages format
        self.server_config['config']['multimodal'] = True
        self.server_config['config']['use_messages'] = True
        
        self.logger.info(f"Configured multimodal settings:")
        self.logger.info(f"  Performance mode: temp={self.performance_temperature}, top_p={self.performance_top_p}")
        self.logger.info(f"  Accuracy mode: temp={self.accuracy_temperature}, top_p={self.accuracy_top_p}")
        self.logger.info(f"  Use guided decoding: {self.use_guided_decoding}")
        self.logger.info(f"  Multimodal mode: messages format with images")
    
    def run(self, user_conf: str = "user.conf", lg_model_name: str = "qwen3-vl-235b-a22b") -> Dict[str, Any]:
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
    
    parser = argparse.ArgumentParser(description="MLPerf Harness for Qwen3-VL")
    
    # Add common harness arguments
    add_common_harness_args(parser)
    
    # Add Qwen3-VL-specific arguments
    parser.add_argument("--lg-model-name", type=str, default="qwen3-vl-235b-a22b", help="LoadGen model name")
    parser.add_argument("--dataset-name", type=str, default="qwen3vl",
                       help="Dataset name for config lookup (default: qwen3vl)")
    parser.add_argument("--use-guided-decoding", action="store_true",
                       help="Use guided decoding for structured outputs")
    parser.add_argument("--accuracy-temperature", type=float, default=None,
                       help="Sampling temperature for accuracy mode (default: 0.0)")
    parser.add_argument("--accuracy-top-k", type=int, default=None,
                       help="Top-k sampling parameter for accuracy mode")
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
    
    # Add multimodal-specific parameters
    if args.use_guided_decoding:
        harness_config['use_guided_decoding'] = True
    if args.accuracy_temperature is not None:
        harness_config['accuracy_temperature'] = args.accuracy_temperature
    if args.accuracy_top_k is not None:
        harness_config['accuracy_top_k'] = args.accuracy_top_k
    if args.accuracy_top_p is not None:
        harness_config['accuracy_top_p'] = args.accuracy_top_p
    
    # Create and run harness
    harness = Qwen3VLHarness(**harness_config)
    
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
