# ============================================================================
# mlflow_client.py
# -----------------
# MLflow client for uploading harness output artifacts and metrics
# Supports different client types (LoadGen, etc.) with flexible metrics handling
# ============================================================================

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLflowClient:
    """
    MLflow client for uploading harness output artifacts and metrics.
    
    Supports different client types (LoadGen, custom clients) with flexible
    handling of descriptions and metrics based on client type.
    
    Features:
    - Automatic artifact upload from output directories
    - Client-specific metrics and descriptions
    - Experiment tracking
    - Run management
    """
    
    def __init__(self,
                 tracking_uri: str,
                 experiment_name: str,
                 client_type: str = "loadgen",
                 output_dir: Optional[str] = None):
        """
        Initialize MLflow client.
        
        Args:
            tracking_uri: MLflow tracking server URI (e.g., http://localhost:5000)
            experiment_name: Name of the MLflow experiment
            client_type: Type of client ("loadgen", "custom", etc.)
            output_dir: Output directory to upload (optional, set later)
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is not installed. Please install it with: pip install mlflow")
        
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.client_type = client_type.lower()
        self.output_dir = Path(output_dir) if output_dir else None
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(tracking_uri)
        
        # Get or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                self.logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            else:
                self.logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
        except Exception as e:
            self.logger.error(f"Failed to set up experiment: {e}")
            raise
        
        # Current run (set when start_run is called)
        self.current_run = None
        self.run_id = None
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional dictionary of tags to attach to the run
            
        Returns:
            Run ID
        """
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{self.client_type}_run_{timestamp}"
        
        mlflow.set_experiment(self.experiment_name)
        
        # Start run with tags
        run_tags = tags or {}
        run_tags['client_type'] = self.client_type
        
        self.current_run = mlflow.start_run(run_name=run_name, tags=run_tags)
        self.run_id = self.current_run.info.run_id
        
        self.logger.info(f"Started MLflow run: {run_name} (ID: {self.run_id})")
        return self.run_id
    
    def end_run(self):
        """End the current MLflow run."""
        if self.current_run:
            mlflow.end_run()
            self.current_run = None
            self.run_id = None
            self.logger.info("Ended MLflow run")
    
    def log_parameters(self, params: Dict[str, Any]):
        """
        Log parameters to the current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        if not self.current_run:
            self.logger.warning("No active run. Parameters not logged.")
            return
        
        try:
            mlflow.log_params(params)
            self.logger.info(f"Logged {len(params)} parameters")
        except Exception as e:
            self.logger.error(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to the current run.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number for the metrics
        """
        if not self.current_run:
            self.logger.warning("No active run. Metrics not logged.")
            return
        
        try:
            if step is not None:
                for name, value in metrics.items():
                    mlflow.log_metric(name, value, step=step)
            else:
                mlflow.log_metrics(metrics)
            self.logger.info(f"Logged {len(metrics)} metrics")
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
    
    def log_client_metrics(self, test_results: Dict[str, Any], additional_metrics: Optional[Dict[str, float]] = None):
        """
        Log metrics specific to the client type.
        
        For LoadGen clients, extracts LoadGen-specific metrics.
        For custom clients, uses provided metrics.
        
        Args:
            test_results: Test results dictionary (from harness run)
            additional_metrics: Additional metrics to log (optional)
        """
        if not self.current_run:
            self.logger.warning("No active run. Metrics not logged.")
            return
        
        metrics = {}
        
        if self.client_type == "loadgen":
            # Extract LoadGen-specific metrics from test results
            if 'status' in test_results:
                metrics['test_status'] = 1.0 if test_results['status'] == 'success' else 0.0
            if 'duration' in test_results:
                metrics['test_duration_seconds'] = float(test_results['duration'])
            if 'scenario' in test_results:
                # Log scenario as tag (already done in start_run, but can log as metric too)
                pass
            if 'num_samples' in test_results:
                metrics['num_samples'] = float(test_results['num_samples'])
            
            # Try to extract LoadGen metrics from summary file if available
            if self.output_dir:
                summary_file = self.output_dir / "mlperf" / "mlperf_log_summary.txt"
                if summary_file.exists():
                    try:
                        mlgen_metrics = self._extract_loadgen_metrics(summary_file)
                        metrics.update(mlgen_metrics)
                    except Exception as e:
                        self.logger.warning(f"Failed to extract LoadGen metrics from summary: {e}")
        else:
            # For custom clients, use provided metrics from test_results
            for key, value in test_results.items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
        
        # Add additional metrics if provided
        if additional_metrics:
            metrics.update(additional_metrics)
        
        if metrics:
            self.log_metrics(metrics)
    
    def _extract_loadgen_metrics(self, summary_file: Path) -> Dict[str, float]:
        """
        Extract LoadGen metrics from summary file.
        
        Args:
            summary_file: Path to mlperf_log_summary.txt
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        try:
            with open(summary_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Look for common LoadGen metrics
                    if 'Throughput' in line or 'throughput' in line.lower():
                        # Extract throughput value
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.lower() in ['throughput', 'qps', 'queries/sec']:
                                if i + 1 < len(parts):
                                    try:
                                        value = float(parts[i + 1])
                                        metrics['loadgen_throughput'] = value
                                    except (ValueError, IndexError):
                                        pass
                    # Add more metric extraction logic as needed
                    # For now, return what we found
        except Exception as e:
            self.logger.warning(f"Error extracting LoadGen metrics: {e}")
        
        return metrics
    
    def log_description(self, description: str):
        """
        Log a description for the run.
        
        Args:
            description: Description text
        """
        if not self.current_run:
            self.logger.warning("No active run. Description not logged.")
            return
        
        try:
            mlflow.log_text(description, "description.txt")
            # Also log as a tag for easier searching
            mlflow.set_tag("description", description[:255])  # MLflow tags have length limits
            self.logger.info("Logged description")
        except Exception as e:
            self.logger.error(f"Failed to log description: {e}")
    
    def get_client_description(self, test_results: Dict[str, Any], additional_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a description based on client type.
        
        Args:
            test_results: Test results dictionary
            additional_info: Additional information to include
            
        Returns:
            Description string
        """
        desc_parts = []
        
        if self.client_type == "loadgen":
            desc_parts.append("MLPerf LoadGen Test Run")
            desc_parts.append(f"Scenario: {test_results.get('scenario', 'Unknown')}")
            desc_parts.append(f"Test Mode: {test_results.get('test_mode', 'Unknown')}")
            desc_parts.append(f"Status: {test_results.get('status', 'Unknown')}")
            desc_parts.append(f"Duration: {test_results.get('duration', 0):.2f} seconds")
            desc_parts.append(f"Number of Samples: {test_results.get('num_samples', 'Unknown')}")
        else:
            desc_parts.append(f"Custom Client Test Run ({self.client_type})")
            desc_parts.append(f"Status: {test_results.get('status', 'Unknown')}")
            if 'duration' in test_results:
                desc_parts.append(f"Duration: {test_results.get('duration', 0):.2f} seconds")
        
        if additional_info:
            desc_parts.append("\nAdditional Information:")
            for key, value in additional_info.items():
                desc_parts.append(f"  {key}: {value}")
        
        return "\n".join(desc_parts)
    
    def upload_artifacts(self, output_dir: Optional[str] = None, include_subdirs: bool = True):
        """
        Upload output directory and subdirectories as artifacts.
        
        Args:
            output_dir: Output directory to upload (uses self.output_dir if None)
            include_subdirs: Whether to include subdirectories
        """
        if not self.current_run:
            self.logger.warning("No active run. Artifacts not uploaded.")
            return
        
        upload_dir = Path(output_dir) if output_dir else self.output_dir
        if not upload_dir:
            self.logger.warning("No output directory specified. Artifacts not uploaded.")
            return
        
        if not upload_dir.exists():
            self.logger.warning(f"Output directory does not exist: {upload_dir}. Artifacts not uploaded.")
            return
        
        try:
            self.logger.info(f"Uploading artifacts from: {upload_dir}")
            
            if include_subdirs:
                # Upload entire directory structure
                mlflow.log_artifacts(str(upload_dir), "output")
            else:
                # Upload only files in the root directory
                for file_path in upload_dir.iterdir():
                    if file_path.is_file():
                        mlflow.log_artifact(str(file_path), "output")
            
            self.logger.info("Artifacts uploaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to upload artifacts: {e}")
            raise
    
    def upload_specific_subdirectories(self, subdirs: List[str], output_dir: Optional[str] = None):
        """
        Upload specific subdirectories as separate artifact directories.
        
        Args:
            subdirs: List of subdirectory names to upload
            output_dir: Output directory to upload from (uses self.output_dir if None)
        """
        if not self.current_run:
            self.logger.warning("No active run. Artifacts not uploaded.")
            return
        
        base_dir = Path(output_dir) if output_dir else self.output_dir
        if not base_dir:
            self.logger.warning("No output directory specified. Artifacts not uploaded.")
            return
        
        if not base_dir.exists():
            self.logger.warning(f"Output directory does not exist: {base_dir}. Artifacts not uploaded.")
            return
        
        try:
            self.logger.info(f"Uploading specific subdirectories: {subdirs}")
            
            for subdir_name in subdirs:
                subdir_path = base_dir / subdir_name
                if subdir_path.exists() and subdir_path.is_dir():
                    # Upload as artifact directory with the subdir name
                    mlflow.log_artifacts(str(subdir_path), f"output/{subdir_name}")
                    self.logger.info(f"Uploaded subdirectory: {subdir_name}")
                else:
                    self.logger.warning(f"Subdirectory not found: {subdir_name}")
            
            self.logger.info("Subdirectories uploaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to upload subdirectories: {e}")
            raise

