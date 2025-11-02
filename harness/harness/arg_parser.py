# ============================================================================
# arg_parser.py
# ------------
# Common argument parser utilities for harnesses
# ============================================================================

import argparse
from typing import Optional


def str_to_bool(value):
    """Convert string to boolean."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes', 'on'):
        return True
    elif value.lower() in ('false', '0', 'no', 'off'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {value}')


def add_common_harness_args(parser: argparse.ArgumentParser):
    """
    Add common harness arguments to argument parser.
    
    Args:
        parser: ArgumentParser instance to add arguments to
    """
    # Core arguments
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--scenario", type=str, default="Offline", choices=["Offline", "Server"],
                       help="LoadGen scenario")
    parser.add_argument("--test-mode", type=str, default="performance", choices=["performance", "accuracy"],
                       help="Test mode")
    parser.add_argument("--api-server-url", type=str, default=None, help="API server URL (if using existing server)")
    parser.add_argument("--batch-size", type=int, default=13368, help="Batch size")
    parser.add_argument("--num-samples", type=int, default=13368, help="Number of samples")
    parser.add_argument("--output-dir", type=str, default="./harness_output", help="Output directory")
    parser.add_argument("--enable-metrics", action="store_true", help="Enable metrics collection")
    parser.add_argument("--server-config", type=str, default=None, help="Server config YAML file")
    parser.add_argument("--user-conf", type=str, default="user.conf", help="LoadGen user config")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    # MLflow arguments
    parser.add_argument("--mlflow-experiment-name", type=str, default=None,
                       help="MLflow experiment name (enables MLflow tracking)")
    parser.add_argument("--mlflow-output-dir", type=str, default=None,
                       help="Output directory to upload to MLflow (defaults to --output-dir)")
    parser.add_argument("--mlflow-host", type=str, default="localhost",
                       help="MLflow tracking server hostname")
    parser.add_argument("--mlflow-port", type=int, default=5000,
                       help="MLflow tracking server port")
    
    # Server scenario arguments
    parser.add_argument("--server-coalesce-queries", type=str_to_bool,
                       default=None, metavar='BOOL',
                       help="Enable query coalescing for Server scenario (Server only, true/false/1/0/yes/no)")
    parser.add_argument("--server-target-qps", type=float, default=None,
                       help="Target queries per second for Server scenario (Server only)")


def parse_common_harness_args(args):
    """
    Parse common harness arguments and return configuration.
    
    Args:
        args: Parsed arguments from argparse
        
    Returns:
        Dictionary with harness configuration
    """
    # Load server config if provided
    server_config = {}
    if args.server_config:
        from backendserver import load_server_config
        server_config = load_server_config(args.server_config)
        server_config['config_file'] = args.server_config
    
    # Construct MLflow tracking URI if experiment name is provided
    mlflow_tracking_uri = None
    if args.mlflow_experiment_name:
        mlflow_tracking_uri = f"http://{args.mlflow_host}:{args.mlflow_port}"
    
    return {
        'model_name': args.model,
        'dataset_path': args.dataset_path,
        'scenario': args.scenario,
        'test_mode': args.test_mode,
        'server_config': server_config,
        'api_server_url': args.api_server_url,
        'batch_size': args.batch_size,
        'num_samples': args.num_samples,
        'output_dir': args.output_dir,
        'enable_metrics': args.enable_metrics,
        'mlflow_tracking_uri': mlflow_tracking_uri,
        'mlflow_experiment_name': args.mlflow_experiment_name,
        'mlflow_output_dir': args.mlflow_output_dir,
        'server_coalesce_queries': args.server_coalesce_queries,
        'server_target_qps': args.server_target_qps
    }

