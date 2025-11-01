# ============================================================================
# harness_llama3.1_8b.py
# ----------------------
# Main harness implementation for Llama 3.1 8B model
# Demonstrates integration of inference server, loadgen client, and metrics
# ============================================================================

import os
import sys
import time
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Matplotlib imports - set backend early for headless environments
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# Add harness directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import harness components
try:
    from backendserver import VLLMServer, create_server, start_server_from_config, load_server_config
    from Client import LoadGenOfflineClient, LoadGenServerClient, create_loadgen_client
    from data.dataset_processor import DatasetProcessor
except ImportError:
    # Try relative imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backendserver import VLLMServer, create_server, start_server_from_config, load_server_config
    from Client import LoadGenOfflineClient, LoadGenServerClient, create_loadgen_client
    from data.dataset_processor import DatasetProcessor

# Import MLPerf Loadgen
try:
    import mlperf_loadgen as lg
except ImportError:
    print("mlperf_loadgen is not installed.")
    print("Please install it from the MLPerf Inference repository.")
    sys.exit(1)

# Import metrics collection if available
try:
    # Metrics folder is now within harness directory
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'metrics'))
    from vllm_metrics_collector import VLLMMetricsCollector, JSONStorage
    from vllm_metrics_visualizer import VLLMMetricsVisualizer
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logging.warning("Metrics collection not available")

# Import environment info collector
try:
    from environment.environment_info import EnvironmentInfoCollector
    ENVIRONMENT_INFO_AVAILABLE = True
except ImportError:
    ENVIRONMENT_INFO_AVAILABLE = False

# Import metrics CSVStorage if available
try:
    from metrics.vllm_metrics_collector import CSVStorage
    CSV_STORAGE_AVAILABLE = True
except ImportError:
    CSV_STORAGE_AVAILABLE = False

# Import MLflow client if available
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mlflow_tools'))
    from mlflow_client import MLflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class Llama31_8BHarness:
    """
    Harness for Llama 3.1 8B model with MLPerf Loadgen.
    
    Integrates:
    - Inference server management (vLLM, SGLang)
    - LoadGen client (Offline/Server scenarios)
    - Dataset processing
    - Metrics collection and visualization
    - MLflow tracking and artifact upload
    """
    
    def __init__(self,
                 model_name: str,
                 dataset_path: str,
                 scenario: str = "Offline",
                 test_mode: str = "performance",
                 server_config: Optional[Dict[str, Any]] = None,
                 api_server_url: Optional[str] = None,
                 batch_size: int = 13368,
                 num_samples: int = 13368,
                 output_dir: str = "./harness_output",
                 enable_metrics: bool = False,
                 metrics_interval: int = 10,
                 mlflow_tracking_uri: Optional[str] = None,
                 mlflow_experiment_name: Optional[str] = None,
                 mlflow_output_dir: Optional[str] = None):
        """
        Initialize harness.
        
        Args:
            model_name: Model name or path
            dataset_path: Path to dataset file
            scenario: LoadGen scenario ("Offline" or "Server")
            test_mode: Test mode ("performance" or "accuracy")
            server_config: Server configuration (if starting server)
            api_server_url: API server URL (if using existing server)
            batch_size: Batch size for processing
            num_samples: Number of samples for testing
            output_dir: Output directory for logs and results
            enable_metrics: Enable metrics collection
            metrics_interval: Metrics collection interval (seconds)
            mlflow_tracking_uri: MLflow tracking server URI (e.g., http://localhost:5000)
            mlflow_experiment_name: MLflow experiment name
            mlflow_output_dir: Output directory to upload to MLflow (defaults to output_dir)
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.scenario = scenario
        self.test_mode = test_mode
        self.server_config = server_config or {}
        self.api_server_url = api_server_url
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.output_dir = Path(output_dir)
        self.enable_metrics = enable_metrics
        self.metrics_interval = metrics_interval
        
        # MLflow configuration
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.mlflow_output_dir = Path(mlflow_output_dir) if mlflow_output_dir else self.output_dir
        self.mlflow_client = None
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized output
        self.harness_output_dir = self.output_dir / "harness_output"
        self.server_output_dir = self.output_dir / "server"
        self.metrics_output_dir = self.output_dir / "metrics"
        self.visualizations_output_dir = self.output_dir / "visualizations"
        self.mlperf_output_dir = self.output_dir / "mlperf"
        self.environment_output_dir = self.output_dir / "environment"
        
        # Create all subdirectories
        self.harness_output_dir.mkdir(parents=True, exist_ok=True)
        self.server_output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_output_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_output_dir.mkdir(parents=True, exist_ok=True)
        self.mlperf_output_dir.mkdir(parents=True, exist_ok=True)
        self.environment_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directory structure created at: {self.output_dir}")
        self.logger.info(f"  - Harness output: {self.harness_output_dir}")
        self.logger.info(f"  - Server logs: {self.server_output_dir}")
        self.logger.info(f"  - Metrics: {self.metrics_output_dir}")
        self.logger.info(f"  - Visualizations: {self.visualizations_output_dir}")
        self.logger.info(f"  - MLPerf logs: {self.mlperf_output_dir}")
        self.logger.info(f"  - Environment info: {self.environment_output_dir}")
        
        # Initialize MLflow if configured
        if self.mlflow_tracking_uri and self.mlflow_experiment_name:
            self._initialize_mlflow()
        
        # Setup stdout redirection to harness_output
        self._setup_stdout_redirection()
        
        # Collect environment information
        self._collect_environment_info()
        
        # Components
        self.server = None
        self.client = None
        self.metrics_collector = None
        self.metrics_visualizer = None
        
        # State
        self.server_started = False
        
        # Stdout redirection
        self.stdout_file = None
        self.stderr_file = None
        self.original_stdout = None
        self.original_stderr = None
    
    def _initialize_mlflow(self):
        """Initialize MLflow client if configured."""
        if not MLFLOW_AVAILABLE:
            self.logger.warning("MLflow is not available. MLflow tracking will be disabled.")
            return
        
        try:
            self.mlflow_client = MLflowClient(
                tracking_uri=self.mlflow_tracking_uri,
                experiment_name=self.mlflow_experiment_name,
                client_type="loadgen",
                output_dir=str(self.mlflow_output_dir)
            )
            self.logger.info(f"MLflow client initialized: {self.mlflow_tracking_uri}")
            self.logger.info(f"MLflow experiment: {self.mlflow_experiment_name}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize MLflow client: {e}")
            self.mlflow_client = None
    
    def _setup_stdout_redirection(self):
        """Setup stdout and stderr redirection to harness_output directory."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stdout_file_path = self.harness_output_dir / f"harness_stdout_{timestamp}.log"
            stderr_file_path = self.harness_output_dir / f"harness_stderr_{timestamp}.log"
            
            # Open files for stdout and stderr
            self.stdout_file = open(stdout_file_path, 'w', buffering=1)
            self.stderr_file = open(stderr_file_path, 'w', buffering=1)
            
            # Save original stdout/stderr
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            
            # Redirect stdout and stderr
            sys.stdout = self.stdout_file
            sys.stderr = self.stderr_file
            
            self.logger.info(f"Stdout redirected to: {stdout_file_path}")
            self.logger.info(f"Stderr redirected to: {stderr_file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to setup stdout redirection: {e}")
    
    def _restore_stdout_redirection(self):
        """Restore original stdout and stderr."""
        if self.original_stdout:
            sys.stdout = self.original_stdout
            sys.stdout.flush()
        if self.original_stderr:
            sys.stderr = self.original_stderr
            sys.stderr.flush()
        
        if self.stdout_file:
            self.stdout_file.close()
            self.stdout_file = None
        if self.stderr_file:
            self.stderr_file.close()
            self.stderr_file = None
    
    def _collect_environment_info(self):
        """Collect environment information."""
        if not ENVIRONMENT_INFO_AVAILABLE:
            self.logger.debug("Environment info collector not available")
            return
        
        try:
            collector = EnvironmentInfoCollector(self.environment_output_dir)
            results = collector.collect_all()
            
            self.logger.info("Environment information collected:")
            self.logger.info(f"  - Successfully collected: {list(results.get('success', {}).keys())}")
            if results.get('errors'):
                self.logger.warning(f"  - Errors: {list(results['errors'].keys())}")
        except Exception as e:
            self.logger.warning(f"Error collecting environment info: {e}")
    
    def start_server(self):
        """Start inference server if not using external API."""
        if self.api_server_url:
            self.logger.info(f"Using external API server at: {self.api_server_url}")
            return
        
        self.logger.info("Starting inference server...")
        
        # Use server config if provided
        if 'config_file' in self.server_config:
            config_file = self.server_config['config_file']
            
            # Prepare overrides - override output_dir with harness's server subdirectory
            # Also override model if provided in harness initialization
            overrides = {
                'output_dir': str(self.server_output_dir)
            }
            
            # Override model if provided in harness but not explicitly set to be ignored
            if self.model_name:
                overrides['model'] = self.model_name
            
            self.logger.info(f"Starting server from config file: {config_file}")
            self.logger.info(f"Applying overrides: {overrides}")
            
            # Start server with overrides
            self.server = start_server_from_config(config_file, overrides=overrides)
        else:
            # Create server with configuration
            backend = self.server_config.get('backend', 'vllm')
            # Use server subdirectory if not explicitly set in config
            server_output_dir = self.server_config.get('output_dir')
            if not server_output_dir:
                server_output_dir = str(self.server_output_dir)
            port = self.server_config.get('port', 8000)
            env_vars = self.server_config.get('env_vars', {})
            server_env_vars = self.server_config.get('env_vars', {})
            server_config = self.server_config.get('config', {})
            debug_mode = self.server_config.get('debug_mode', False)
            
            self.logger.info(f"Starting server with output directory: {server_output_dir}")
            
            self.server = create_server(
                backend=backend,
                model=self.model_name,
                output_dir=server_output_dir,
                port=port,
                env_vars=server_env_vars,
                config=server_config,
                debug_mode=debug_mode
            )
            
            self.server.start()
        
        self.server_started = True
        self.api_server_url = f"http://localhost:{self.server.port}"
        self.logger.info(f"Inference server started at: {self.api_server_url}")
    
    def stop_server(self):
        """Stop inference server if we started it."""
        if self.server and self.server_started:
            self.logger.info("Stopping inference server...")
            self.server.stop()
            self.server_started = False
            self.logger.info("Inference server stopped")
    
    def initialize_client(self):
        """Initialize LoadGen client."""
        self.logger.info(f"Initializing LoadGen client (scenario: {self.scenario})")
        
        self.client = create_loadgen_client(
            scenario=self.scenario,
            model_name=self.model_name,
            dataset_path=self.dataset_path,
            test_mode=self.test_mode,
            api_server_url=self.api_server_url,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
            config=self.server_config
        )
        
        self.client.initialize()
        self.logger.info("LoadGen client initialized")
        
        # Start workers for server scenario
        if self.scenario == "Server" and isinstance(self.client, LoadGenServerClient):
            self.client.start_workers()
    
    def initialize_metrics(self):
        """Initialize metrics collection if enabled."""
        if not self.enable_metrics or not METRICS_AVAILABLE:
            return
        
        if not self.api_server_url:
            self.logger.warning("Metrics collection requires API server URL")
            return
        
        try:
            # Use metrics subdirectory - default to CSV format
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_file = self.metrics_output_dir / f"metrics_{timestamp}.csv"
            
            if CSV_STORAGE_AVAILABLE:
                storage = CSVStorage(str(metrics_file))
            else:
                # Fallback to JSON storage
                storage = JSONStorage(str(metrics_file).replace('.csv', '.json'))
            
            self.metrics_collector = VLLMMetricsCollector(
                metrics_endpoint=f"{self.api_server_url}/metrics",
                storage=storage,
                metrics_to_collect=[
                    'vllm:num_requests_running',
                    'vllm:generation_tokens_total',
                    'vllm:request_success_total',
                    'vllm:request_failure_total',
                    'vllm:request_latency',
                    'vllm:gpu_utilization',
                    'vllm:gpu_memory_used',
                    'vllm:kv_cache_usage_ratio'
                ],
                collection_interval=self.metrics_interval,
                timeout=30,
                auto_postprocess=True,  # Enable auto-postprocessing to create processed file
                debug_mode=True
            )
            
            self.metrics_visualizer = VLLMMetricsVisualizer()
            self.logger.info(f"Metrics collection initialized: {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize metrics: {e}")
            self.enable_metrics = False
    
    def run(self, user_conf: str = "user.conf", lg_model_name: str = "llama3_1-8b") -> Dict[str, Any]:
        """
        Run the harness test.
        
        Args:
            user_conf: User configuration file for LoadGen
            lg_model_name: Model name for LoadGen
        
        Returns:
            Dictionary with test results
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING MLPERF HARNESS TEST")
        self.logger.info("=" * 80)
        
        server_started_here = False
        client_initialized = False
        metrics_started = False
        mlflow_run_started = False
        
        try:
            # Start MLflow run if configured
            if self.mlflow_client:
                try:
                    self.mlflow_client.start_run()
                    mlflow_run_started = True
                    
                    # Log parameters
                    params = {
                        'model_name': self.model_name,
                        'scenario': self.scenario,
                        'test_mode': self.test_mode,
                        'batch_size': str(self.batch_size),
                        'num_samples': str(self.num_samples)
                    }
                    self.mlflow_client.log_parameters(params)
                except Exception as e:
                    self.logger.warning(f"Failed to start MLflow run: {e}")
                    mlflow_run_started = False
            
            # Start server if needed
            if not self.api_server_url:
                self.start_server()
                server_started_here = True
            
            # Initialize client
            try:
                self.initialize_client()
                client_initialized = True
            except Exception as e:
                self.logger.error(f"Failed to initialize client: {e}", exc_info=True)
                # Cleanup server if we started it
                if server_started_here:
                    self.stop_server()
                raise
            
            # Initialize metrics
            if self.enable_metrics:
                try:
                    self.initialize_metrics()
                    if self.metrics_collector:
                        self.metrics_collector.start()
                        metrics_started = True
                except Exception as e:
                    self.logger.warning(f"Failed to initialize metrics: {e}")
                    # Metrics failure shouldn't stop the test
                    self.enable_metrics = False
            
            # Configure LoadGen
            settings = lg.TestSettings()
            if self.scenario == "Server":
                settings.scenario = lg.TestScenario.Server
                settings.sample_concatenate_permutation = True
            else:
                settings.scenario = lg.TestScenario.Offline
            
            if self.test_mode == "accuracy":
                settings.mode = lg.TestMode.AccuracyOnly
            else:
                settings.mode = lg.TestMode.PerformanceOnly
            
            settings.use_token_latencies = True
            settings.FromConfig(user_conf, lg_model_name, self.scenario, 1)
            
            # Configure logging - use mlperf subdirectory
            log_output_settings = lg.LogOutputSettings()
            log_output_settings.outdir = str(self.mlperf_output_dir)
            log_output_settings.copy_summary_to_stdout = True
            
            log_settings = lg.LogSettings()
            log_settings.log_output = log_output_settings
            log_settings.enable_trace = False
            
            self.logger.info(f"MLPerf LoadGen output directory: {self.mlperf_output_dir}")
            
            # Create QSL
            total_samples = len(self.client.dataset.input_ids)
            qsl = lg.ConstructQSL(
                total_samples,
                self.num_samples,
                self._load_samples_to_ram,
                self._unload_samples_from_ram
            )
            
            # Create SUT
            sut = lg.ConstructSUT(self.client.issue_query, self.client.flush_queries)
            
            # Run test
            self.logger.info("Starting LoadGen test...")
            test_start = time.time()
            
            try:
                lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)
            except Exception as e:
                self.logger.error(f"LoadGen test execution failed: {e}", exc_info=True)
                raise
            
            test_end = time.time()
            test_duration = test_end - test_start
            
            self.logger.info("=" * 80)
            self.logger.info("TEST COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"Test duration: {test_duration:.2f} seconds")
            
            # Stop metrics collector before generating visualizations
            # (metrics file is finalized when collector stops)
            if metrics_started and self.metrics_collector:
                try:
                    self.logger.info("Stopping metrics collector...")
                    self.metrics_collector.stop()
                    # Wait for file to be written and processed (if auto_postprocess enabled)
                    # Auto-postprocessing happens in stop() method, so wait a bit for processed file
                    time.sleep(1.0)  # Increased delay to ensure processed file is created
                except Exception as e:
                    self.logger.warning(f"Error stopping metrics collector: {e}")
            
            # Generate metrics visualizations after collector is stopped
            if self.enable_metrics and self.metrics_collector:
                self._generate_metrics_visualizations()
            
            test_results = {
                'status': 'success',
                'duration': test_duration,
                'scenario': self.scenario,
                'test_mode': self.test_mode,
                'num_samples': self.num_samples
            }
            
            # Upload to MLflow if configured
            if mlflow_run_started and self.mlflow_client:
                try:
                    self._upload_to_mlflow(test_results)
                except Exception as e:
                    self.logger.warning(f"Failed to upload to MLflow: {e}")
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"Test failed: {e}", exc_info=True)
            test_results = {
                'status': 'failed',
                'error': str(e)
            }
            
            # Upload failure to MLflow if configured
            if mlflow_run_started and self.mlflow_client:
                try:
                    self._upload_to_mlflow(test_results)
                except Exception as e:
                    self.logger.warning(f"Failed to upload failure to MLflow: {e}")
            
            return test_results
        
        finally:
            # Cleanup in reverse order of initialization
            self.logger.info("Performing cleanup...")
            
            # End MLflow run if started
            if mlflow_run_started and self.mlflow_client:
                try:
                    self.mlflow_client.end_run()
                except Exception as e:
                    self.logger.warning(f"Error ending MLflow run: {e}")
            
            # Stop metrics collector if not already stopped (in case of exception)
            if metrics_started and self.metrics_collector:
                try:
                    # Check if already stopped
                    if hasattr(self.metrics_collector, 'running') and self.metrics_collector.running:
                        self.logger.info("Stopping metrics collector in finally block...")
                        self.metrics_collector.stop()
                        time.sleep(0.5)
                except Exception as e:
                    self.logger.warning(f"Error stopping metrics collector in finally: {e}")
            
            # Cleanup client
            if client_initialized and self.client:
                try:
                    self.logger.info("Cleaning up client...")
                    self.client.cleanup()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up client: {e}")
            
            # Stop server (only if we started it)
            if server_started_here:
                try:
                    self.logger.info("Stopping server...")
                    self.stop_server()
                except Exception as e:
                    self.logger.warning(f"Error stopping server: {e}")
            
            # Restore stdout/stderr redirection
            self._restore_stdout_redirection()
            
            self.logger.info("Cleanup completed")
    
    def _upload_to_mlflow(self, test_results: Dict[str, Any]):
        """Upload test results and artifacts to MLflow."""
        if not self.mlflow_client:
            return
        
        try:
            # Log client-specific metrics
            self.mlflow_client.log_client_metrics(test_results)
            
            # Generate and log description
            description = self.mlflow_client.get_client_description(test_results)
            self.mlflow_client.log_description(description)
            
            # Upload artifacts - upload entire output directory
            self.mlflow_client.upload_artifacts(
                output_dir=str(self.mlflow_output_dir),
                include_subdirs=True
            )
            
            self.logger.info("Successfully uploaded to MLflow")
        except Exception as e:
            self.logger.error(f"Failed to upload to MLflow: {e}", exc_info=True)
            raise
    
    def _load_samples_to_ram(self, query_samples):
        """LoadGen callback - samples are pre-loaded in Dataset."""
        pass
    
    def _unload_samples_from_ram(self, query_samples):
        """LoadGen callback - no action needed."""
        pass
    
    def _generate_metrics_visualizations(self):
        """Generate metrics visualizations after test."""
        if not self.metrics_collector or not self.metrics_visualizer:
            self.logger.warning("Metrics collector or visualizer not available")
            return
        
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available. Visualizations will not be generated.")
            return
        
        try:
            storage_file = self.metrics_collector._get_storage_file_path()
            if not storage_file:
                self.logger.warning("No storage file path available from metrics collector")
                return
            
            if not os.path.exists(storage_file):
                self.logger.warning(f"Metrics file not found: {storage_file}")
                return
            
            # Check if file has content
            if os.path.getsize(storage_file) == 0:
                self.logger.warning(f"Metrics file is empty: {storage_file}")
                return
            
            self.logger.info(f"Generating visualizations from metrics file: {storage_file}")
            
            # Get available metrics from the file
            try:
                available_metrics = self.metrics_visualizer.get_available_metrics(storage_file)
                self.logger.info(f"Available metrics in file: {available_metrics}")
            except Exception as e:
                self.logger.warning(f"Could not get available metrics: {e}", exc_info=True)
                available_metrics = []
            
            # Use visualizations subdirectory (already created in __init__)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate visualizations for metrics that are available
            visualization_configs = [
                {
                    'metric': 'vllm:gpu_utilization',
                    'title': 'GPU Utilization Over Time',
                    'filename': f'gpu_utilization_{timestamp}.png'
                },
                {
                    'metric': 'vllm:num_requests_running',
                    'title': 'Running Requests Over Time',
                    'filename': f'requests_running_{timestamp}.png'
                },
                {
                    'metric': 'vllm:request_latency',
                    'title': 'Request Latency Over Time',
                    'filename': f'request_latency_{timestamp}.png'
                },
                {
                    'metric': 'vllm:gpu_memory_used',
                    'title': 'GPU Memory Usage Over Time',
                    'filename': f'gpu_memory_{timestamp}.png'
                }
            ]
            
            successful_viz = 0
            for viz in visualization_configs:
                # Check if metric is available before trying to plot
                if available_metrics and viz['metric'] not in available_metrics:
                    self.logger.debug(f"Metric {viz['metric']} not available in metrics file, skipping")
                    continue
                
                try:
                    save_path = self.visualizations_output_dir / viz['filename']
                    self.logger.info(f"Generating visualization for {viz['metric']}...")
                    
                    # Try to use processed file if available (handles dict labels better)
                    # Check for processed CSV/JSON file
                    base_name = os.path.splitext(storage_file)[0]
                    processed_csv = f"{base_name}_processed.csv"
                    processed_json = f"{base_name}_processed.json"
                    
                    # Use processed file if it exists, otherwise use original
                    viz_file = storage_file
                    if os.path.exists(processed_csv):
                        viz_file = processed_csv
                        self.logger.info(f"Using processed CSV file for visualization: {viz_file}")
                    elif os.path.exists(processed_json):
                        viz_file = processed_json
                        self.logger.info(f"Using processed JSON file for visualization: {viz_file}")
                    
                    # Use show_labels=False to avoid pandas groupby on dict labels
                    # (This is a workaround for the unhashable dict issue)
                    self.metrics_visualizer.plot_metric(
                        file_path=viz_file,
                        metric_name=viz['metric'],
                        title=viz['title'],
                        save_path=str(save_path),
                        show_labels=False  # Disable label grouping to avoid dict unhashable error
                    )
                    # Close all figures to free memory and ensure clean state for next plot
                    plt.close('all')
                    self.logger.info(f"✓ Generated visualization: {save_path}")
                    successful_viz += 1
                except ValueError as e:
                    # Metric not found in data
                    self.logger.warning(f"Metric {viz['metric']} not found in data: {e}")
                except Exception as e:
                    self.logger.error(f"Failed to generate visualization {viz['filename']}: {e}", exc_info=True)
                    # Close plot even on error
                    try:
                        plt.close('all')
                    except:
                        pass
            
            if successful_viz > 0:
                self.logger.info(f"Successfully generated {successful_viz} visualization(s) in {self.visualizations_output_dir}")
            else:
                self.logger.warning("No visualizations were generated. Check if metrics file contains valid data.")
        
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}", exc_info=True)


def main():
    """Main entry point for harness."""
    parser = argparse.ArgumentParser(description="MLPerf Harness for Llama 3.1 8B")
    
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
    parser.add_argument("--lg-model-name", type=str, default="llama3_1-8b", help="LoadGen model name")
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
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Load server config if provided
    server_config = {}
    if args.server_config:
        server_config = load_server_config(args.server_config)
        server_config['config_file'] = args.server_config
    
    # Construct MLflow tracking URI if experiment name is provided
    mlflow_tracking_uri = None
    if args.mlflow_experiment_name:
        mlflow_tracking_uri = f"http://{args.mlflow_host}:{args.mlflow_port}"
    
    # Create and run harness
    harness = Llama31_8BHarness(
        model_name=args.model,
        dataset_path=args.dataset_path,
        scenario=args.scenario,
        test_mode=args.test_mode,
        server_config=server_config,
        api_server_url=args.api_server_url,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        enable_metrics=args.enable_metrics,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
        mlflow_output_dir=args.mlflow_output_dir
    )
    
    results = harness.run(user_conf=args.user_conf, lg_model_name=args.lg_model_name)
    
    if results['status'] == 'success':
        print("\n✓ Harness test completed successfully!")
        return 0
    else:
        print(f"\n✗ Harness test failed: {results.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
