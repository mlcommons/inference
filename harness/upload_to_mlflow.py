#!/usr/bin/env python3
# ============================================================================
# upload_to_mlflow.py
# -------------------
# Standalone script to upload harness results to MLflow
# Reads metadata YAML file created during harness run
# ============================================================================

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

# Add harness directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import MLflow client
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mlflow_tools'))
    from mlflow_client import MLflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("ERROR: MLflow is not available. Please install it with: pip install mlflow")
    sys.exit(1)


def load_metadata(metadata_file: Path) -> Dict[str, Any]:
    """
    Load metadata from YAML file.
    
    Args:
        metadata_file: Path to metadata YAML file
        
    Returns:
        Metadata dictionary
    """
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = yaml.safe_load(f)
    
    return metadata


def upload_results(metadata: Dict[str, Any], 
                   mlflow_tracking_uri: str,
                   mlflow_experiment_name: str) -> None:
    """
    Upload results to MLflow using metadata.
    
    Args:
        metadata: Metadata dictionary loaded from YAML
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_experiment_name: MLflow experiment name
    """
    logger = logging.getLogger(__name__)
    
    # Extract configuration
    harness_config = metadata.get('harness_config', {})
    test_results = metadata.get('test_results', {})
    paths = metadata.get('paths', {})
    mlflow_config = metadata.get('mlflow_config', {})
    
    # Use provided MLflow config or fall back to metadata
    tracking_uri = mlflow_tracking_uri or mlflow_config.get('tracking_uri')
    experiment_name = mlflow_experiment_name or mlflow_config.get('experiment_name')
    output_dir = mlflow_config.get('output_dir', paths.get('output_dir'))
    
    if not tracking_uri or not experiment_name:
        raise ValueError("MLflow tracking URI and experiment name must be provided either via arguments or metadata file")
    
    logger.info("=" * 80)
    logger.info("UPLOADING RESULTS TO MLFLOW")
    logger.info("=" * 80)
    logger.info(f"Tracking URI: {tracking_uri}")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize MLflow client
    try:
        mlflow_client = MLflowClient(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            client_type="loadgen",
            output_dir=output_dir
        )
        logger.info("MLflow client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize MLflow client: {e}")
        raise
    
    # Start MLflow run
    try:
        mlflow_client.start_run()
        logger.info("MLflow run started")
        
        # Log parameters from harness config
        params = {
            'model_name': harness_config.get('model_name'),
            'dataset_name': harness_config.get('dataset_name'),  # Added dataset name
            'framework': harness_config.get('framework', 'vllm'),  # Added framework
            'scenario': harness_config.get('scenario'),
            'test_mode': harness_config.get('test_mode'),
            'batch_size': str(harness_config.get('batch_size', '')),
            'num_samples': str(harness_config.get('num_samples', ''))
        }
        
        # Add Server-specific parameters if present
        if harness_config.get('server_coalesce_queries') is not None:
            params['server_coalesce_queries'] = str(harness_config.get('server_coalesce_queries'))
        if harness_config.get('server_target_qps') is not None:
            params['server_target_qps'] = str(harness_config.get('server_target_qps'))
        
        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None and v != ''}
        
        mlflow_client.log_parameters(params)
        logger.info(f"Logged {len(params)} parameters")
        
        # Log metrics
        mlflow_client.log_client_metrics(test_results)
        
        # Generate and log description
        description = mlflow_client.get_client_description(test_results)
        mlflow_client.log_description(description)
        
        # Upload artifacts
        mlflow_client.upload_artifacts(
            output_dir=output_dir,
            include_subdirs=True
        )
        
        logger.info("Successfully uploaded to MLflow")
        
    except Exception as e:
        logger.error(f"Failed to upload to MLflow: {e}", exc_info=True)
        raise
    finally:
        # End MLflow run
        try:
            mlflow_client.end_run()
            logger.info("MLflow run ended")
        except Exception as e:
            logger.warning(f"Error ending MLflow run: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload harness results to MLflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload using metadata file (uses MLflow config from metadata)
  python upload_to_mlflow.py --metadata-file ./harness_output/mlflow_metadata.yaml
  
  # Upload with custom MLflow server
  python upload_to_mlflow.py --metadata-file ./harness_output/mlflow_metadata.yaml \\
      --mlflow-host localhost --mlflow-port 5000 --mlflow-experiment-name my-experiment
        """
    )
    
    parser.add_argument("--metadata-file", type=str, required=True,
                       help="Path to metadata YAML file (created during harness run)")
    parser.add_argument("--mlflow-host", type=str, default=None,
                       help="MLflow tracking server hostname (overrides metadata)")
    parser.add_argument("--mlflow-port", type=int, default=None,
                       help="MLflow tracking server port (overrides metadata)")
    parser.add_argument("--mlflow-experiment-name", type=str, default=None,
                       help="MLflow experiment name (overrides metadata)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load metadata
        metadata_file = Path(args.metadata_file)
        logger.info(f"Loading metadata from: {metadata_file}")
        metadata = load_metadata(metadata_file)
        logger.info("Metadata loaded successfully")
        
        # Determine MLflow configuration
        mlflow_config = metadata.get('mlflow_config', {})
        
        # Use command-line args if provided, otherwise use metadata
        if args.mlflow_host and args.mlflow_port:
            mlflow_tracking_uri = f"http://{args.mlflow_host}:{args.mlflow_port}"
        else:
            mlflow_tracking_uri = mlflow_config.get('tracking_uri')
            if not mlflow_tracking_uri:
                raise ValueError("MLflow tracking URI must be provided via --mlflow-host and --mlflow-port, or in metadata file")
        
        mlflow_experiment_name = args.mlflow_experiment_name or mlflow_config.get('experiment_name')
        if not mlflow_experiment_name:
            raise ValueError("MLflow experiment name must be provided via --mlflow-experiment-name or in metadata file")
        
        # Upload results
        upload_results(metadata, mlflow_tracking_uri, mlflow_experiment_name)
        
        print("\n✓ Successfully uploaded results to MLflow!")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"Metadata file not found: {e}")
        print(f"\n✗ Error: {e}")
        print("Make sure the harness run completed successfully and created a metadata file.")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\n✗ Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        print(f"\n✗ Upload failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

