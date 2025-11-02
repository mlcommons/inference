# ============================================================================
# base_harness.py
# ---------------
# Base harness class with common functionality for all MLPerf harnesses
# ============================================================================

import os
import sys
import time
import logging
import signal
import atexit
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Matplotlib imports - set backend early for headless environments
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# Import harness components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backendserver import create_server, start_server_from_config, load_server_config
from Client import create_loadgen_client, LoadGenServerClient

# Import MLPerf Loadgen
try:
    import mlperf_loadgen as lg
except ImportError:
    print("mlperf_loadgen is not installed.")
    sys.exit(1)

# Import metrics collection if available
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'metrics'))
    from vllm_metrics_collector import VLLMMetricsCollector, JSONStorage, CSVStorage
    from vllm_metrics_visualizer import VLLMMetricsVisualizer
    METRICS_AVAILABLE = True
    CSV_STORAGE_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    CSV_STORAGE_AVAILABLE = False

# Import environment info collector
try:
    from environment.environment_info import EnvironmentInfoCollector
    ENVIRONMENT_INFO_AVAILABLE = True
except ImportError:
    ENVIRONMENT_INFO_AVAILABLE = False

# Import MLflow client if available
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mlflow_tools'))
    from mlflow_client import MLflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class BaseHarness:
    """
    Base harness class with common functionality for all MLPerf harnesses.
    
    Provides:
    - Server management (start/stop)
    - LoadGen setup and execution
    - Metrics collection and visualization
    - MLflow integration (optional)
    - Signal handling and cleanup
    - Stdout/stderr redirection
    - YAML metadata storage (for later MLflow upload)
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
                 metrics_interval: int = 15,
                 mlflow_tracking_uri: Optional[str] = None,
                 mlflow_experiment_name: Optional[str] = None,
                 mlflow_output_dir: Optional[str] = None,
                 server_coalesce_queries: Optional[bool] = None,
                 server_target_qps: Optional[float] = None):
        """
        Initialize base harness.
        
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
            server_coalesce_queries: Enable query coalescing for Server scenario (Server only)
            server_target_qps: Target queries per second for Server scenario (Server only)
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
        self.server_coalesce_queries = server_coalesce_queries
        self.server_target_qps = server_target_qps
        
        # MLflow configuration
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.mlflow_output_dir = Path(mlflow_output_dir) if mlflow_output_dir else self.output_dir
        self.mlflow_client = None
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create output directory structure
        self._setup_output_directories()
        
        # Initialize MLflow if configured
        if self.mlflow_tracking_uri and self.mlflow_experiment_name:
            self._initialize_mlflow()
        
        # Setup stdout redirection
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
        
        # Signal handling
        self._cleanup_on_exit = False
        self._setup_signal_handlers()
    
    def _setup_output_directories(self):
        """Setup output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized output
        self.harness_output_dir = self.output_dir / "harness_output"
        self.server_output_dir = self.output_dir / "server"
        self.metrics_output_dir = self.output_dir / "metrics"
        self.visualizations_output_dir = self.output_dir / "visualizations"
        self.mlperf_output_dir = self.output_dir / "mlperf"
        self.environment_output_dir = self.output_dir / "environment"
        
        # Create all subdirectories
        for dir_path in [self.harness_output_dir, self.server_output_dir, 
                        self.metrics_output_dir, self.visualizations_output_dir,
                        self.mlperf_output_dir, self.environment_output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directory structure created at: {self.output_dir}")
    
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
            
            # Update logging handlers to use redirected streams
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler):
                    handler_stream = handler.stream
                    if handler_stream is self.original_stdout or handler_stream is sys.stdout or (hasattr(handler_stream, 'fileno') and handler_stream.fileno() == 1):
                        handler.setStream(self.stdout_file)
                        handler.flush()
                    elif handler_stream is self.original_stderr or handler_stream is sys.stderr or (hasattr(handler_stream, 'fileno') and handler_stream.fileno() == 2):
                        handler.setStream(self.stderr_file)
                        handler.flush()
            
            # Also update handlers for all loggers
            for logger_name in logging.Logger.manager.loggerDict:
                logger = logging.getLogger(logger_name)
                for handler in logger.handlers[:]:
                    if isinstance(handler, logging.StreamHandler):
                        handler_stream = handler.stream
                        if handler_stream is self.original_stdout or handler_stream is sys.stdout or (hasattr(handler_stream, 'fileno') and handler_stream.fileno() == 1):
                            handler.setStream(self.stdout_file)
                            handler.flush()
                        elif handler_stream is self.original_stderr or handler_stream is sys.stderr or (hasattr(handler_stream, 'fileno') and handler_stream.fileno() == 2):
                            handler.setStream(self.stderr_file)
                            handler.flush()
            
            self.stdout_file.flush()
            self.stderr_file.flush()
            
            print(f"Stdout redirected to: {stdout_file_path}", file=self.stdout_file, flush=True)
            print(f"Stderr redirected to: {stderr_file_path}", file=self.stderr_file, flush=True)
            
            self.logger.info(f"Stdout redirected to: {stdout_file_path}")
            self.logger.info(f"Stderr redirected to: {stderr_file_path}")
            
            self.stdout_file.flush()
            self.stderr_file.flush()
        except Exception as e:
            print(f"Failed to setup stdout redirection: {e}", file=self.original_stderr if self.original_stderr else sys.stderr)
            if self.logger:
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
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            """Handle shutdown signals (SIGINT, SIGTERM)."""
            signal_name = signal.Signals(signum).name
            self.logger.warning(f"Received {signal_name} signal. Initiating graceful shutdown...")
            self._cleanup_on_exit = True
            self._emergency_cleanup()
            sys.exit(130 if signum == signal.SIGINT else 143)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Register atexit handler as backup
        atexit.register(self._emergency_cleanup)
    
    def _emergency_cleanup(self):
        """Perform emergency cleanup of server and processes."""
        try:
            self.logger.info("Performing emergency cleanup...")
            
            # Stop metrics collector if running
            if self.metrics_collector and hasattr(self.metrics_collector, 'running'):
                try:
                    if self.metrics_collector.running:
                        self.metrics_collector.stop()
                except Exception as e:
                    self.logger.warning(f"Error stopping metrics collector during cleanup: {e}")
            
            # Stop server if started
            if self.server and self.server_started:
                try:
                    self.logger.info("Stopping server during emergency cleanup...")
                    self.stop_server()
                except Exception as e:
                    self.logger.warning(f"Error stopping server during cleanup: {e}")
            
            # Cleanup client if initialized
            if self.client:
                try:
                    self.logger.info("Cleaning up client during emergency cleanup...")
                    self.client.cleanup()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up client during cleanup: {e}")
            
            # End MLflow run if active
            if self.mlflow_client:
                try:
                    self.mlflow_client.end_run()
                except Exception as e:
                    self.logger.warning(f"Error ending MLflow run during cleanup: {e}")
            
            # Restore stdout/stderr redirection
            self._restore_stdout_redirection()
            
            self.logger.info("Emergency cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during emergency cleanup: {e}")
    
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
            
            # Prepare overrides
            overrides = {
                'output_dir': str(self.server_output_dir)
            }
            
            if self.model_name:
                overrides['model'] = self.model_name
            
            self.logger.info(f"Starting server from config file: {config_file}")
            self.logger.info(f"Applying overrides: {overrides}")
            
            self.server = start_server_from_config(config_file, overrides=overrides)
        else:
            # Create server with configuration
            backend = self.server_config.get('backend', 'vllm')
            server_output_dir = self.server_config.get('output_dir')
            if not server_output_dir:
                server_output_dir = str(self.server_output_dir)
            port = self.server_config.get('port', 8000)
            env_vars = self.server_config.get('env_vars', {})
            server_config = self.server_config.get('config', {})
            debug_mode = self.server_config.get('debug_mode', False)
            
            self.logger.info(f"Starting server with output directory: {server_output_dir}")
            
            self.server = create_server(
                backend=backend,
                model=self.model_name,
                output_dir=server_output_dir,
                port=port,
                env_vars=env_vars,
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
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_file = self.metrics_output_dir / f"metrics_{timestamp}.csv"
            
            if CSV_STORAGE_AVAILABLE:
                storage = CSVStorage(str(metrics_file))
            else:
                storage = JSONStorage(str(metrics_file).replace('.csv', '.json'))
            
            self.metrics_collector = VLLMMetricsCollector(
                metrics_endpoint=f"{self.api_server_url}/metrics",
                storage=storage,
                metrics_to_collect=[
                    'vllm:num_requests_running',
                    'vllm:generation_tokens_total',
                    'vllm:prompt_tokens_total',
                    'vllm:kv_cache_usage_perc',
                    'vllm:time_to_first_token_seconds'
                ],
                collection_interval=self.metrics_interval,
                timeout=30,
                auto_postprocess=True,
                debug_mode=True
            )
            
            self.metrics_visualizer = VLLMMetricsVisualizer()
            self.logger.info(f"Metrics collection initialized: {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize metrics: {e}")
            self.enable_metrics = False
    
    def setup_loadgen_settings(self, user_conf: str, lg_model_name: str):
        """
        Setup LoadGen TestSettings.
        
        Args:
            user_conf: User configuration file for LoadGen
            lg_model_name: Model name for LoadGen
        
        Returns:
            Configured TestSettings object
        """
        settings = lg.TestSettings()
        
        if self.scenario == "Server":
            settings.scenario = lg.TestScenario.Server
            
            # Set Server-specific parameters if provided
            if self.server_coalesce_queries is not None:
                settings.server_coalesce_queries = self.server_coalesce_queries
                self.logger.info(f"Server coalesce queries set to: {self.server_coalesce_queries}")
            
            if self.server_target_qps is not None:
                settings.server_target_qps = self.server_target_qps
                self.logger.info(f"Server target QPS set to: {self.server_target_qps}")
        else:
            settings.scenario = lg.TestScenario.Offline
        
        if self.test_mode == "accuracy":
            settings.mode = lg.TestMode.AccuracyOnly
        else:
            settings.mode = lg.TestMode.PerformanceOnly
        
        settings.use_token_latencies = True
        settings.FromConfig(user_conf, lg_model_name, self.scenario, 1)
        
        return settings
    
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
            if not storage_file or not os.path.exists(storage_file) or os.path.getsize(storage_file) == 0:
                self.logger.warning(f"Metrics file not available: {storage_file}")
                return
            
            self.logger.info(f"Generating visualizations from metrics file: {storage_file}")
            
            try:
                available_metrics = self.metrics_visualizer.get_available_metrics(storage_file)
                self.logger.info(f"Available metrics in file: {available_metrics}")
            except Exception as e:
                self.logger.warning(f"Could not get available metrics: {e}")
                available_metrics = []
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            visualization_configs = [
                {
                    'metric': 'vllm:generation_tokens_total',
                    'title': 'Generation Tokens Total over Time',
                    'filename': f'generation_tokens_total_{timestamp}.png'
                },
                {
                    'metric': 'vllm:num_requests_running',
                    'title': 'Running Requests Over Time',
                    'filename': f'requests_running_{timestamp}.png'
                },
                {
                    'metric': 'vllm:prompt_tokens_total',
                    'title': 'Prompt tokens total over time',
                    'filename': f'prompt_tokens_total_{timestamp}.png'
                },
                {
                    'metric': 'vllm:kv_cache_usage_perc',
                    'title': 'KV cache usage percentage over time',
                    'filename': f'kv_cache_usage_perc_{timestamp}.png'
                }
            ]
            
            successful_viz = 0
            for viz in visualization_configs:
                if available_metrics and viz['metric'] not in available_metrics:
                    self.logger.info(f"Metric {viz['metric']} not available, skipping")
                    continue
                
                try:
                    save_path = self.visualizations_output_dir / viz['filename']
                    
                    # Try to use processed file if available
                    base_name = os.path.splitext(storage_file)[0]
                    processed_csv = f"{base_name}_processed.csv"
                    processed_json = f"{base_name}_processed.json"
                    
                    viz_file = storage_file
                    if os.path.exists(processed_csv):
                        viz_file = processed_csv
                    elif os.path.exists(processed_json):
                        viz_file = processed_json
                    
                    self.metrics_visualizer.plot_metric(
                        file_path=viz_file,
                        metric_name=viz['metric'],
                        title=viz['title'],
                        save_path=str(save_path),
                        show_labels=False
                    )
                    plt.close('all')
                    self.logger.info(f"✓ Generated visualization: {save_path}")
                    successful_viz += 1
                except Exception as e:
                    self.logger.error(f"Failed to generate visualization {viz['filename']}: {e}")
                    try:
                        plt.close('all')
                    except:
                        pass
            
            if successful_viz > 0:
                self.logger.info(f"Successfully generated {successful_viz} visualization(s)")
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
    
    def save_metadata(self, test_results: Dict[str, Any]):
        """
        Save metadata to YAML file for later MLflow upload.
        
        Args:
            test_results: Test results dictionary
        """
        # Get dataset name from path
        dataset_name = Path(self.dataset_path).stem
        
        # Get framework from server config
        framework = self.server_config.get('backend', 'vllm')
        if not framework and self.api_server_url:
            framework = 'vllm'  # Default
        
        metadata = {
            'test_results': test_results,
            'harness_config': {
                'model_name': self.model_name,
                'dataset_path': str(self.dataset_path),
                'dataset_name': dataset_name,
                'framework': framework,
                'scenario': self.scenario,
                'test_mode': self.test_mode,
                'batch_size': self.batch_size,
                'num_samples': self.num_samples,
                'server_coalesce_queries': self.server_coalesce_queries,
                'server_target_qps': self.server_target_qps
            },
            'paths': {
                'output_dir': str(self.output_dir),
                'mlperf_output_dir': str(self.mlperf_output_dir),
                'metrics_output_dir': str(self.metrics_output_dir),
                'visualizations_output_dir': str(self.visualizations_output_dir)
            },
            'mlflow_config': {
                'tracking_uri': self.mlflow_tracking_uri,
                'experiment_name': self.mlflow_experiment_name,
                'output_dir': str(self.mlflow_output_dir)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = self.output_dir / "mlflow_metadata.yaml"
        try:
            with open(metadata_file, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
            self.logger.info(f"Metadata saved to: {metadata_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save metadata: {e}")
    
    def _load_samples_to_ram(self, query_samples):
        """LoadGen callback - samples are pre-loaded in Dataset."""
        pass
    
    def _unload_samples_from_ram(self, query_samples):
        """LoadGen callback - no action needed."""
        pass

