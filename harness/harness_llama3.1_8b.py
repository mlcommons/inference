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
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add harness directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import harness components
try:
    from backendserver import VLLMServer, create_server, start_server_from_config
    from Client import LoadGenOfflineClient, LoadGenServerClient, create_loadgen_client
    from data.dataset_processor import DatasetProcessor
except ImportError:
    # Try relative imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backendserver import VLLMServer, create_server, start_server_from_config
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


class Llama31_8BHarness:
    """
    Harness for Llama 3.1 8B model with MLPerf Loadgen.
    
    Integrates:
    - Inference server management (vLLM, SGLang)
    - LoadGen client (Offline/Server scenarios)
    - Dataset processing
    - Metrics collection and visualization
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
                 metrics_interval: int = 10):
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
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized output
        self.harness_output_dir = self.output_dir / "harness"
        self.server_output_dir = self.output_dir / "server"
        self.metrics_output_dir = self.output_dir / "metrics"
        self.visualizations_output_dir = self.output_dir / "visualizations"
        self.mlperf_output_dir = self.output_dir / "mlperf"
        
        # Create all subdirectories
        self.harness_output_dir.mkdir(parents=True, exist_ok=True)
        self.server_output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_output_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_output_dir.mkdir(parents=True, exist_ok=True)
        self.mlperf_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directory structure created at: {self.output_dir}")
        self.logger.info(f"  - Harness output: {self.harness_output_dir}")
        self.logger.info(f"  - Server logs: {self.server_output_dir}")
        self.logger.info(f"  - Metrics: {self.metrics_output_dir}")
        self.logger.info(f"  - Visualizations: {self.visualizations_output_dir}")
        self.logger.info(f"  - MLPerf logs: {self.mlperf_output_dir}")
        
        # Components
        self.server = None
        self.client = None
        self.metrics_collector = None
        self.metrics_visualizer = None
        
        # State
        self.server_started = False
    
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
            # Use metrics subdirectory
            metrics_file = self.metrics_output_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            storage = JSONStorage(str(metrics_file))
            
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
                auto_postprocess=True,
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
        
        try:
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
            
            # Generate metrics visualizations
            if self.enable_metrics and self.metrics_collector:
                self._generate_metrics_visualizations()
            
            return {
                'status': 'success',
                'duration': test_duration,
                'scenario': self.scenario,
                'test_mode': self.test_mode,
                'num_samples': self.num_samples
            }
            
        except Exception as e:
            self.logger.error(f"Test failed: {e}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e)
            }
        
        finally:
            # Cleanup in reverse order of initialization
            self.logger.info("Performing cleanup...")
            
            # Stop metrics collector
            if metrics_started and self.metrics_collector:
                try:
                    self.logger.info("Stopping metrics collector...")
                    self.metrics_collector.stop()
                except Exception as e:
                    self.logger.warning(f"Error stopping metrics collector: {e}")
            
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
            
            self.logger.info("Cleanup completed")
    
    def _load_samples_to_ram(self, query_samples):
        """LoadGen callback - samples are pre-loaded in Dataset."""
        pass
    
    def _unload_samples_from_ram(self, query_samples):
        """LoadGen callback - no action needed."""
        pass
    
    def _generate_metrics_visualizations(self):
        """Generate metrics visualizations after test."""
        if not self.metrics_collector or not self.metrics_visualizer:
            return
        
        try:
            storage_file = self.metrics_collector._get_storage_file_path()
            if not storage_file or not os.path.exists(storage_file):
                self.logger.warning("No metrics file found for visualization")
                return
            
            # Use visualizations subdirectory (already created in __init__)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate visualizations
            visualizations = [
                {
                    'metric': 'vllm:gpu_utilization',
                    'title': 'GPU Utilization Over Time',
                    'filename': f'gpu_utilization_{timestamp}.png'
                },
                {
                    'metric': 'vllm:num_requests_running',
                    'title': 'Running Requests Over Time',
                    'filename': f'requests_running_{timestamp}.png'
                }
            ]
            
            for viz in visualizations:
                try:
                    save_path = self.visualizations_output_dir / viz['filename']
                    self.metrics_visualizer.plot_metric(
                        file_path=storage_file,
                        metric_name=viz['metric'],
                        title=viz['title'],
                        save_path=str(save_path)
                    )
                    self.logger.info(f"Generated visualization: {save_path}")
                except Exception as e:
                    self.logger.error(f"Failed to generate visualization {viz['filename']}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")


def main():
    """Main entry point for harness."""
    import argparse
    
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
        from backendserver import load_server_config
        server_config = load_server_config(args.server_config)
        server_config['config_file'] = args.server_config
    
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
        enable_metrics=args.enable_metrics
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

