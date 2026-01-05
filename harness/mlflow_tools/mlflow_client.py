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
        
        if not metrics:
            self.logger.warning("No metrics to log.")
            return
        
        try:
            # Filter out None and invalid values
            valid_metrics = {}
            for name, value in metrics.items():
                if value is not None and isinstance(value, (int, float)):
                    # Check for NaN and Inf
                    import math
                    if math.isfinite(value):
                        valid_metrics[name] = float(value)
                    else:
                        self.logger.warning(f"Skipping metric {name} with non-finite value: {value}")
                else:
                    self.logger.warning(f"Skipping metric {name} with invalid value: {value} (type: {type(value)})")
            
            if not valid_metrics:
                self.logger.warning("No valid metrics to log after filtering.")
                return
            
            self.logger.info(f"Logging {len(valid_metrics)} valid metrics to MLflow (from {len(metrics)} total)")
            
            if step is not None:
                for name, value in valid_metrics.items():
                    mlflow.log_metric(name, value, step=step)
            else:
                mlflow.log_metrics(valid_metrics)
            
            self.logger.info(f"Successfully logged {len(valid_metrics)} metrics to MLflow")
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}", exc_info=True)
            raise
    
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
                self.logger.debug(f"Looking for summary file at: {summary_file}")
                if summary_file.exists():
                    try:
                        self.logger.info(f"Found summary file: {summary_file}")
                        mlgen_metrics = self._extract_loadgen_metrics(summary_file)
                        self.logger.info(f"Extracted {len(mlgen_metrics)} LoadGen metrics from summary file")
                        metrics.update(mlgen_metrics)
                    except Exception as e:
                        self.logger.warning(f"Failed to extract LoadGen metrics from summary: {e}", exc_info=True)
                else:
                    self.logger.warning(f"Summary file not found at: {summary_file}")
        else:
            # For custom clients, use provided metrics from test_results
            for key, value in test_results.items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
        
        # Add additional metrics if provided
        if additional_metrics:
            metrics.update(additional_metrics)
        
        if metrics:
            # Log all metrics to MLflow
            self.logger.info(f"Logging {len(metrics)} total metrics to MLflow")
            self.log_metrics(metrics)
            
            # Log metric names for debugging
            self.logger.debug(f"Metric names: {sorted(metrics.keys())}")
    
    def _extract_loadgen_metrics(self, summary_file: Path) -> Dict[str, float]:
        """
        Extract LoadGen metrics from MLPerf log summary file.
        
        Parses mlperf_log_summary.txt and extracts:
        - Throughput metrics (samples/sec, tokens/sec)
        - Latency metrics (min, max, mean, percentiles)
        - First Token latency metrics (Server scenario)
        - Time to Output Token metrics (Server scenario)
        - Test parameters
        - Result status
        
        Args:
            summary_file: Path to mlperf_log_summary.txt
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        scenario = None  # Will be detected from file
        
        try:
            with open(summary_file, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            in_additional_stats = False
            in_test_params = False
            
            for line in lines:
                orig_line = line  # Keep original for debugging
                line = line.strip()
                
                # Detect scenario early from "Scenario : Server" or "Scenario : Offline"
                if scenario is None and 'Scenario :' in line:
                    scenario_str = line.split(':', 1)[1].strip() if ':' in line else ''
                    scenario = 'Server' if 'Server' in scenario_str else 'Offline'
                    metrics['loadgen_scenario_server'] = 1.0 if scenario == 'Server' else 0.0
                    self.logger.info(f"Detected scenario from summary file: {scenario}")
                
                # Check for section headers - they appear before separator lines
                if 'Additional Stats' in line:
                    in_additional_stats = True
                    in_test_params = False
                    # Skip separator lines
                    continue
                elif 'Test Parameters Used' in line:
                    in_additional_stats = False
                    in_test_params = True
                    # Skip separator lines
                    continue
                
                # Skip empty lines and separator lines (all '=' or all '-')
                if not line or (line.startswith('=') and all(c in '=-' for c in line)):
                    continue
                
                # Extract Result status
                if line.startswith('Result is :'):
                    result = line.split(':', 1)[1].strip()
                    metrics['loadgen_result_valid'] = 1.0 if result == 'VALID' else 0.0
                
                # Extract throughput metrics (main section) - check BEFORE Additional Stats
                if not in_additional_stats and not in_test_params:
                    # Format: "Samples per second: 43.6595" or "Completed samples per second    : 9.36"
                    if 'samples per second' in line.lower() and ':' in line:
                        # Split on colon, handling multiple spaces
                        parts = line.split(':', 1)  # Split only on first colon
                        if len(parts) == 2:
                            try:
                                value = float(parts[1].strip())
                                # Store both specific and generic names
                                if 'completed' in line.lower():
                                    metrics['loadgen_completed_samples_per_second'] = value
                                    metrics['loadgen_samples_per_second'] = value  # Generic alias
                                else:
                                    metrics['loadgen_samples_per_second'] = value
                            except ValueError:
                                pass
                    
                    # Format: "Tokens per second: 5588.41" or "Completed tokens per second: 1207.14"
                    if 'tokens per second' in line.lower() and ':' in line:
                        # Split on colon, handling multiple spaces
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            try:
                                value = float(parts[1].strip())
                                # Store both specific and generic names
                                if 'completed' in line.lower():
                                    metrics['loadgen_completed_tokens_per_second'] = value
                                    metrics['loadgen_tokens_per_second'] = value  # Generic alias
                                else:
                                    metrics['loadgen_tokens_per_second'] = value
                            except ValueError:
                                pass
                
                # Extract Scheduled samples per second (Server scenario only) - in Additional Stats
                if in_additional_stats and 'Scheduled samples per second' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        try:
                            value = float(parts[1].strip())
                            metrics['loadgen_scheduled_samples_per_second'] = value
                        except ValueError:
                            pass
                
                # Extract latency metrics (Additional Stats section)
                # Note: Server scenario also has "Completed tokens per second" repeated here
                if in_additional_stats:
                    # Min/Max/Mean latency
                    if 'Min latency (ns)' in line and ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            try:
                                value = float(parts[1].strip())
                                metrics['loadgen_min_latency_ns'] = value
                                metrics['loadgen_min_latency_ms'] = value / 1_000_000  # Convert to ms
                            except ValueError:
                                pass
                    
                    if 'Max latency (ns)' in line and ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            try:
                                value = float(parts[1].strip())
                                metrics['loadgen_max_latency_ns'] = value
                                metrics['loadgen_max_latency_ms'] = value / 1_000_000
                            except ValueError:
                                pass
                    
                    if 'Mean latency (ns)' in line and ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            try:
                                value = float(parts[1].strip())
                                metrics['loadgen_mean_latency_ns'] = value
                                metrics['loadgen_mean_latency_ms'] = value / 1_000_000
                            except ValueError:
                                pass
                    
                    # Percentile latencies
                    percentile_patterns = [
                        ('50.00 percentile latency (ns)', 'loadgen_p50_latency_ns', 'loadgen_p50_latency_ms'),
                        ('90.00 percentile latency (ns)', 'loadgen_p90_latency_ns', 'loadgen_p90_latency_ms'),
                        ('95.00 percentile latency (ns)', 'loadgen_p95_latency_ns', 'loadgen_p95_latency_ms'),
                        ('97.00 percentile latency (ns)', 'loadgen_p97_latency_ns', 'loadgen_p97_latency_ms'),
                        ('99.00 percentile latency (ns)', 'loadgen_p99_latency_ns', 'loadgen_p99_latency_ms'),
                        ('99.90 percentile latency (ns)', 'loadgen_p999_latency_ns', 'loadgen_p999_latency_ms'),
                    ]
                    
                    for pattern, ns_key, ms_key in percentile_patterns:
                        if pattern in line and ':' in line:
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                try:
                                    value = float(parts[1].strip())
                                    metrics[ns_key] = value
                                    metrics[ms_key] = value / 1_000_000
                                except ValueError:
                                    pass
                    
                    # First Token latency metrics (Server scenario only)
                    # These only appear in Server scenario summaries
                    if 'Min First Token latency (ns)' in line and ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            try:
                                value = float(parts[1].strip())
                                metrics['loadgen_min_ttft_ns'] = value
                                metrics['loadgen_min_ttft_ms'] = value / 1_000_000
                            except ValueError:
                                pass
                    
                    if 'Max First Token latency (ns)' in line and ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            try:
                                value = float(parts[1].strip())
                                metrics['loadgen_max_ttft_ns'] = value
                                metrics['loadgen_max_ttft_ms'] = value / 1_000_000
                            except ValueError:
                                pass
                    
                    if 'Mean First Token latency (ns)' in line and ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            try:
                                value = float(parts[1].strip())
                                metrics['loadgen_mean_ttft_ns'] = value
                                metrics['loadgen_mean_ttft_ms'] = value / 1_000_000
                            except ValueError:
                                pass
                    
                    # First Token percentile latencies
                    ttft_percentile_patterns = [
                        ('50.00 percentile first token latency (ns)', 'loadgen_p50_ttft_ns', 'loadgen_p50_ttft_ms'),
                        ('90.00 percentile first token latency (ns)', 'loadgen_p90_ttft_ns', 'loadgen_p90_ttft_ms'),
                        ('95.00 percentile first token latency (ns)', 'loadgen_p95_ttft_ns', 'loadgen_p95_ttft_ms'),
                        ('97.00 percentile first token latency (ns)', 'loadgen_p97_ttft_ns', 'loadgen_p97_ttft_ms'),
                        ('99.00 percentile first token latency (ns)', 'loadgen_p99_ttft_ns', 'loadgen_p99_ttft_ms'),
                        ('99.90 percentile first token latency (ns)', 'loadgen_p999_ttft_ns', 'loadgen_p999_ttft_ms'),
                    ]
                    
                    for pattern, ns_key, ms_key in ttft_percentile_patterns:
                        if pattern in line and ':' in line:
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                try:
                                    value = float(parts[1].strip())
                                    metrics[ns_key] = value
                                    metrics[ms_key] = value / 1_000_000
                                except ValueError:
                                    pass
                    
                    # Time to Output Token metrics (Server scenario only)
                    # These only appear in Server scenario summaries
                    if 'Min Time to Output Token (ns)' in line and ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            try:
                                value = float(parts[1].strip())
                                metrics['loadgen_min_tpot_ns'] = value
                                metrics['loadgen_min_tpot_ms'] = value / 1_000_000
                            except ValueError:
                                pass
                    
                    if 'Max Time to Output Token (ns)' in line and ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            try:
                                value = float(parts[1].strip())
                                metrics['loadgen_max_tpot_ns'] = value
                                metrics['loadgen_max_tpot_ms'] = value / 1_000_000
                            except ValueError:
                                pass
                    
                    if 'Mean Time to Output Token (ns)' in line and ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            try:
                                value = float(parts[1].strip())
                                metrics['loadgen_mean_tpot_ns'] = value
                                metrics['loadgen_mean_tpot_ms'] = value / 1_000_000
                            except ValueError:
                                pass
                    
                    # Time to Output Token percentile latencies
                    tpot_percentile_patterns = [
                        ('50.00 percentile time to output token (ns)', 'loadgen_p50_tpot_ns', 'loadgen_p50_tpot_ms'),
                        ('90.00 percentile time to output token (ns)', 'loadgen_p90_tpot_ns', 'loadgen_p90_tpot_ms'),
                        ('95.00 percentile time to output token (ns)', 'loadgen_p95_tpot_ns', 'loadgen_p95_tpot_ms'),
                        ('97.00 percentile time to output token (ns)', 'loadgen_p97_tpot_ns', 'loadgen_p97_tpot_ms'),
                        ('99.00 percentile time to output token (ns)', 'loadgen_p99_tpot_ns', 'loadgen_p99_tpot_ms'),
                        ('99.90 percentile time to output token (ns)', 'loadgen_p999_tpot_ns', 'loadgen_p999_tpot_ms'),
                    ]
                    
                    for pattern, ns_key, ms_key in tpot_percentile_patterns:
                        if pattern in line and ':' in line:
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                try:
                                    value = float(parts[1].strip())
                                    metrics[ns_key] = value
                                    metrics[ms_key] = value / 1_000_000
                                except ValueError:
                                    pass
                
                # Extract test parameters
                if in_test_params:
                    if ':' in line:
                        parts = line.split(':', 1)
                        key = parts[0].strip()
                        try:
                            value = float(parts[1].strip())
                            # Convert parameter names to metric names
                            # Normalize key by removing spaces and converting to lowercase
                            normalized_key = key.lower().replace(' ', '_').replace('(', '').replace(')', '')
                            
                            param_mapping = {
                                'samples_per_query': 'loadgen_samples_per_query',
                                'target_qps': 'loadgen_target_qps',
                                'ttft_latency_ns': 'loadgen_ttft_latency_ns',
                                'tpot_latency_ns': 'loadgen_tpot_latency_ns',
                                'max_async_queries': 'loadgen_max_async_queries',
                                'min_duration_ms': 'loadgen_min_duration_ms',
                                'max_duration_ms': 'loadgen_max_duration_ms',
                                'min_query_count': 'loadgen_min_query_count',
                                'max_query_count': 'loadgen_max_query_count',
                                'performance_sample_count': 'loadgen_performance_sample_count',
                            }
                            
                            metric_key = param_mapping.get(normalized_key)
                            if metric_key:
                                metrics[metric_key] = value
                            # Also log raw parameter with prefix for parameters not in mapping
                            elif normalized_key not in ['qsl_rng_seed', 'sample_index_rng_seed', 'schedule_rng_seed',
                                                         'accuracy_log_rng_seed', 'accuracy_log_probability',
                                                         'accuracy_log_sampling_target', 'print_timestamps',
                                                         'performance_issue_unique', 'performance_issue_same',
                                                         'performance_issue_same_index']:
                                metrics[f'loadgen_param_{normalized_key}'] = value
                        except (ValueError, IndexError):
                            pass
            
            # Log scenario-specific metrics summary
            if metrics:
                scenario_detected = scenario or "Unknown"
                # Count Server-specific metrics
                server_metrics = [k for k in metrics.keys() if any(term in k.lower() for term in ['ttft', 'tpot', 'scheduled', 'completed_samples', 'completed_tokens'])]
                self.logger.info(f"Extracted {len(metrics)} LoadGen metrics from summary file (Scenario: {scenario_detected})")
                if server_metrics and scenario_detected == 'Server':
                    self.logger.info(f"Server-specific metrics found: {len(server_metrics)} metrics including TTFT, TPOT, Scheduled, and Completed throughput")
                    # List key Server metrics
                    key_server_metrics = [k for k in server_metrics if any(term in k for term in ['ttft', 'tpot', 'scheduled', 'completed'])]
                    self.logger.debug(f"Server metrics: {sorted(key_server_metrics)}")
            else:
                self.logger.warning("No metrics extracted from summary file")
                
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
            
            # Check for MLPerf log summary and log it explicitly
            summary_file = upload_dir / "mlperf" / "mlperf_log_summary.txt"
            if summary_file.exists():
                self.logger.info(f"Found MLPerf log summary: {summary_file}")
                # Upload summary file explicitly to ensure it's visible
                mlflow.log_artifact(str(summary_file), "output/mlperf")
                self.logger.info(f"Uploaded MLPerf log summary to output/mlperf/mlperf_log_summary.txt")
            else:
                self.logger.warning(f"MLPerf log summary not found at: {summary_file}")
            
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

