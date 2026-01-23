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
    parser.add_argument("--model", type=str, required=False, default=None, help="Model name or path")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--scenario", type=str, default="Offline", choices=["Offline", "Server"],
                       help="LoadGen scenario")
    parser.add_argument("--test-mode", type=str, default="performance", choices=["performance", "accuracy"],
                       help="Test mode")
    parser.add_argument("--api-server-url", type=str, default=None, 
                       help="API server URL (if using existing server). For load balancing, specify multiple URLs separated by commas or use --api-server-url multiple times")
    parser.add_argument("--api-server-urls", type=str, nargs='+', default=None,
                       help="Multiple API server URLs for load balancing (alternative to comma-separated --api-server-url)")
    parser.add_argument("--host", type=str, default=None,
                       help="API server host (overrides YAML config). Can specify multiple hosts separated by commas for load balancing")
    parser.add_argument("--batch-size", type=int, default=13368, help="Batch size")
    parser.add_argument("--num-samples", type=int, default=13368, help="Number of samples")
    parser.add_argument("--output-dir", type=str, default="./harness_output", help="Output directory")
    parser.add_argument("--enable-metrics", action="store_true", help="Enable metrics collection")
    parser.add_argument("--server-config", type=str, default=None, help="Server config YAML file")
    parser.add_argument("--user-conf", type=str, default="user.conf", help="LoadGen user config")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    # Dataset configuration arguments
    parser.add_argument("--dataset-config-file", type=str, default=None,
                       help="Path to specific dataset config YAML file (overrides auto-detection)")
    parser.add_argument("--input-column", type=str, default=None,
                       help="Override input column name (overrides config)")
    parser.add_argument("--input-ids-column", type=str, default=None,
                       help="Override input_ids column name (overrides config)")
    parser.add_argument("--output-column", type=str, default=None,
                       help="Override output column name (overrides config)")
    
    # Backend configuration arguments
    parser.add_argument("--backend", type=str, default=None,
                       choices=["vllm", "sglang"],
                       help="Backend type to use (vllm or sglang). When using --api-server-url, defaults to vllm if not specified.")
    
    # API endpoint arguments
    parser.add_argument("--endpoint-type", type=str, default="completions",
                       choices=["completions", "chat_completions"],
                       help="API endpoint type: 'completions' or 'chat_completions' (default: completions)")
    
    # MLflow arguments
    parser.add_argument("--mlflow-experiment-name", type=str, default=None,
                       help="MLflow experiment name (enables MLflow tracking)")
    parser.add_argument("--mlflow-output-dir", type=str, default=None,
                       help="Output directory to upload to MLflow (defaults to --output-dir)")
    parser.add_argument("--mlflow-host", type=str, default="localhost",
                       help="MLflow tracking server hostname")
    parser.add_argument("--mlflow-port", type=int, default=5000,
                       help="MLflow tracking server port")
    parser.add_argument("--mlflow-description", type=str, default=None,
                       help="Optional description for MLflow run (overrides auto-generated description)")
    
    # Server scenario arguments
    parser.add_argument("--server-coalesce-queries", type=str_to_bool,
                       default=None, metavar='BOOL',
                       help="Enable query coalescing for Server scenario (Server only, true/false/1/0/yes/no)")
    parser.add_argument("--server-target-qps", type=float, default=None,
                       help="Target queries per second for Server scenario (Server only)")
    parser.add_argument("--target-qps", type=float, default=None,
                       dest='server_target_qps',
                       help="Target queries per second for Server scenario (alias for --server-target-qps, Server only)")
    
    # Offline scenario arguments
    parser.add_argument("--offline-back-to-back", action="store_true",
                       help="Send requests back-to-back instead of batching (Offline scenario only)")
    
    # Debug arguments
    parser.add_argument("--debug-mode", action="store_true",
                       help="Enable debug mode: print text response and total tokens in accuracy mode")
    
    # Note: --engine-args is NOT added here because it must be added last (after all other arguments)
    # when using argparse.REMAINDER. It should be added in harness_main.py after all other arguments.


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
    
    # Add backend to server_config if specified (command line overrides YAML config)
    if args.backend:
        server_config['backend'] = args.backend
    
    # Add endpoint_type to server_config so client can use it
    if args.endpoint_type:
        server_config['endpoint_type'] = args.endpoint_type
    
    # Handle host specification (from YAML or command line)
    # Command line overrides YAML config
    host = None
    hosts = None
    if args.host:
        # Command line host(s) override YAML config
        host_list = [h.strip() for h in args.host.split(',')]
        if len(host_list) == 1:
            host = host_list[0]
        else:
            hosts = host_list
    elif server_config.get('host'):
        # Use host from YAML config
        yaml_host = server_config.get('host')
        if isinstance(yaml_host, list):
            hosts = yaml_host
        else:
            host = yaml_host
    
    # Handle multiple API server URLs for load balancing
    api_server_urls = None
    if args.api_server_urls:
        # Multiple URLs from --api-server-urls flag
        api_server_urls = [url.rstrip('/') for url in args.api_server_urls]
    elif args.api_server_url:
        # Single URL or comma-separated URLs from --api-server-url
        urls = [url.strip().rstrip('/') for url in args.api_server_url.split(',')]
        if len(urls) > 1:
            api_server_urls = urls
        else:
            # Single URL (backward compatible)
            api_server_urls = None
    elif host or hosts:
        # Construct URLs from host(s) and port(s)
        port = server_config.get('port', 8000)
        ports = server_config.get('ports', None)  # Support multiple ports
        
        if hosts:
            # Multiple hosts for load balancing
            if ports and isinstance(ports, list):
                # Multiple hosts and ports
                if len(hosts) != len(ports):
                    raise ValueError(f"Number of hosts ({len(hosts)}) must match number of ports ({len(ports)})")
                api_server_urls = [f"http://{h}:{p}" for h, p in zip(hosts, ports)]
            else:
                # Multiple hosts, single port
                api_server_urls = [f"http://{h}:{port}" for h in hosts]
        elif host:
            # Single host
            if ports and isinstance(ports, list):
                # Single host, multiple ports
                api_server_urls = [f"http://{host}:{p}" for p in ports]
            else:
                # Single host, single port (no load balancing)
                api_server_urls = None
    
    # Add engine_args if provided
    engine_args = None
    if args.engine_args:
        engine_args = list(args.engine_args)
    
    # Add offline_back_to_back to server_config so client can use it
    if hasattr(args, 'offline_back_to_back') and args.offline_back_to_back:
        if 'config' not in server_config:
            server_config['config'] = {}
        server_config['config']['offline_back_to_back'] = True
    
    return {
        'model_name': args.model,
        'dataset_path': args.dataset_path,
        'scenario': args.scenario,
        'test_mode': args.test_mode,
        'server_config': server_config,
        'api_server_url': args.api_server_url if not api_server_urls else None,  # Backward compatible
        'api_server_urls': api_server_urls,  # New: list of URLs for load balancing
        'host': host,  # Single host (if specified)
        'hosts': hosts,  # Multiple hosts for load balancing (if specified)
        'batch_size': args.batch_size,
        'num_samples': args.num_samples,
        'output_dir': args.output_dir,
        'enable_metrics': args.enable_metrics,
        'mlflow_tracking_uri': mlflow_tracking_uri,
        'mlflow_experiment_name': args.mlflow_experiment_name,
        'mlflow_output_dir': args.mlflow_output_dir,
        'mlflow_description': args.mlflow_description,
        'server_coalesce_queries': args.server_coalesce_queries,
        'server_target_qps': args.server_target_qps,
        'dataset_config_file': args.dataset_config_file,
        'input_column': args.input_column,
        'input_ids_column': args.input_ids_column,
        'output_column': args.output_column,
        'engine_args': engine_args,
        'debug_mode': args.debug_mode if hasattr(args, 'debug_mode') else False
    }

