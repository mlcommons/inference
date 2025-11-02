# ============================================================================
# harness_llama3.1_8b.py
# ----------------------
# Harness implementation for Llama 3.1 8B model
# Inherits common functionality from BaseHarness
# ============================================================================

import os
import sys
import time
import logging
import argparse
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

# Add harness directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import base harness
from harness.base_harness import BaseHarness

# Import MLPerf Loadgen
try:
    import mlperf_loadgen as lg
except ImportError:
    print("mlperf_loadgen is not installed.")
    print("Please install it from the MLPerf Inference repository.")
    sys.exit(1)


class Llama31_8BHarness(BaseHarness):
    """
    Harness for Llama 3.1 8B model with MLPerf Loadgen.
    
    Extends BaseHarness with Llama31_8B-specific functionality.
    Most common functionality (server management, metrics, MLflow, etc.)
    is provided by the base class.
    """
    
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
                    
                    # Log parameters (including dataset name and framework)
                    params = self._get_mlflow_parameters()
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
            
            # Setup LoadGen settings using base class method
            settings = self.setup_loadgen_settings(user_conf, lg_model_name)
            
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
            if metrics_started and self.metrics_collector:
                try:
                    self.logger.info("Stopping metrics collector...")
                    self.metrics_collector.stop()
                    time.sleep(1.0)  # Wait for processed file
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
            else:
                # Save metadata for later MLflow upload
                self.save_metadata(test_results)
            
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
            else:
                # Save metadata even on failure
                self.save_metadata(test_results)
            
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
            
            # Stop metrics collector if not already stopped
            if metrics_started and self.metrics_collector:
                try:
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
                    # Small grace period for offline scenario
                    if self.scenario == "Offline":
                        time.sleep(0.5)
                except Exception as e:
                    self.logger.warning(f"Error cleaning up client: {e}")
            
            # Stop server (only if we started it)
            if server_started_here:
                try:
                    self.logger.info("Stopping server...")
                    # Grace period before shutdown
                    time.sleep(1.0)
                    self.stop_server()
                except Exception as e:
                    self.logger.warning(f"Error stopping server: {e}")
            
            # Restore stdout/stderr redirection
            self._restore_stdout_redirection()
            
            self.logger.info("Cleanup completed")
    
    def _get_mlflow_parameters(self) -> Dict[str, str]:
        """
        Get MLflow parameters including dataset name and framework.
        
        Returns:
            Dictionary of parameter names to string values
        """
        params = {
            'model_name': self.model_name,
            'scenario': self.scenario,
            'test_mode': self.test_mode,
            'batch_size': str(self.batch_size),
            'num_samples': str(self.num_samples)
        }
        
        # Add dataset name (extract from path)
        dataset_name = Path(self.dataset_path).stem
        params['dataset_name'] = dataset_name
        
        # Add framework (vllm/sglang)
        framework = self.server_config.get('backend', 'vllm')
        if not framework and self.api_server_url:
            # Try to detect from server URL or infer from context
            framework = 'vllm'  # Default
        params['framework'] = framework
        
        # Add Server-specific parameters if present
        if self.server_coalesce_queries is not None:
            params['server_coalesce_queries'] = str(self.server_coalesce_queries)
        if self.server_target_qps is not None:
            params['server_target_qps'] = str(self.server_target_qps)
        
        return params
    
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


def main():
    """Main entry point for harness."""
    from harness.arg_parser import add_common_harness_args, parse_common_harness_args
    
    parser = argparse.ArgumentParser(description="MLPerf Harness for Llama 3.1 8B")
    
    # Add common harness arguments
    add_common_harness_args(parser)
    
    # Add Llama31_8B-specific arguments
    parser.add_argument("--lg-model-name", type=str, default="llama3_1-8b", help="LoadGen model name")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Parse common arguments
    harness_config = parse_common_harness_args(args)
    
    # Create and run harness
    harness = Llama31_8BHarness(**harness_config)
    
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
