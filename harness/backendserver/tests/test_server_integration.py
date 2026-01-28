#!/usr/bin/env python3
"""
Integration tests for inference_server module.

These tests verify actual server start/stop functionality.
Note: These tests require actual inference server binaries to be installed.
Set SKIP_INTEGRATION_TESTS=1 to skip these tests.
"""

import os
import sys
import time
import unittest
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_server import (
    VLLMServer,
    SGLangServer,
    create_server,
    start_server_from_config,
)


class TestServerIntegration(unittest.TestCase):
    """Integration tests that require actual server binaries."""
    
    @classmethod
    def setUpClass(cls):
        """Check if integration tests should be skipped."""
        if os.environ.get('SKIP_INTEGRATION_TESTS', '0') == '1':
            raise unittest.SkipTest("Integration tests skipped via SKIP_INTEGRATION_TESTS")
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_output_dir = tempfile.mkdtemp(prefix="test_inference_server_integration_")
        # Use high port numbers to avoid conflicts
        self.test_port = 18001
        self.test_model = os.environ.get('TEST_MODEL', 'meta-llama/Llama-2-7b-hf')
        
        # Check if we have a valid test model path
        if not self.test_model:
            raise unittest.SkipTest("TEST_MODEL environment variable not set")
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Make sure any running servers are stopped
        # (This is handled by individual tests, but just in case)
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
    
    def test_vllm_server_start_stop(self):
        """Test starting and stopping a vLLM server."""
        # Skip if vLLM is not available
        try:
            import vllm
        except ImportError:
            raise unittest.SkipTest("vLLM is not installed")
        
        server = VLLMServer(
            model=self.test_model,
            output_dir=self.test_output_dir,
            port=self.test_port,
            startup_timeout=300  # 5 minutes for model loading
        )
        
        try:
            # Start server (may take a while to load model)
            print(f"\nStarting vLLM server with model: {self.test_model}")
            print(f"Output directory: {self.test_output_dir}")
            print(f"Port: {self.test_port}")
            
            start_time = time.time()
            server.start()
            startup_duration = time.time() - start_time
            
            print(f"Server started in {startup_duration:.2f} seconds")
            
            # Verify server is running
            self.assertTrue(server.is_running)
            self.assertIsNotNone(server.process)
            
            # Check health
            self.assertTrue(server.check_health())
            
            # List endpoints
            endpoints = server.list_endpoints()
            print(f"Available endpoints: {list(endpoints.keys())}")
            self.assertIn("health", endpoints)
            
            # Let server run for a few seconds
            time.sleep(5)
            
        finally:
            # Always stop the server
            print("Stopping server...")
            stop_start = time.time()
            server.stop()
            stop_duration = time.time() - stop_start
            print(f"Server stopped in {stop_duration:.2f} seconds")
            
            # Verify server is stopped
            self.assertFalse(server.is_running)
    
    def test_sglang_server_start_stop(self):
        """Test starting and stopping a SGLang server."""
        # Skip if SGLang is not available
        try:
            import sglang
        except ImportError:
            raise unittest.SkipTest("SGLang is not installed")
        
        server = SGLangServer(
            model=self.test_model,
            output_dir=self.test_output_dir,
            port=self.test_port,
            startup_timeout=300
        )
        
        try:
            print(f"\nStarting SGLang server with model: {self.test_model}")
            print(f"Output directory: {self.test_output_dir}")
            print(f"Port: {self.test_port}")
            
            start_time = time.time()
            server.start()
            startup_duration = time.time() - start_time
            
            print(f"Server started in {startup_duration:.2f} seconds")
            
            self.assertTrue(server.is_running)
            self.assertIsNotNone(server.process)
            
            # Check health
            self.assertTrue(server.check_health())
            
            # List endpoints
            endpoints = server.list_endpoints()
            print(f"Available endpoints: {list(endpoints.keys())}")
            
            # Let server run
            time.sleep(5)
            
        finally:
            print("Stopping server...")
            server.stop()
            self.assertFalse(server.is_running)
    
    def test_server_context_manager(self):
        """Test using server as context manager."""
        try:
            import vllm
        except ImportError:
            raise unittest.SkipTest("vLLM is not installed")
        
        with VLLMServer(
            model=self.test_model,
            output_dir=self.test_output_dir,
            port=self.test_port,
            startup_timeout=300
        ) as server:
            self.assertTrue(server.is_running)
            time.sleep(2)
        
        # Server should be stopped after exiting context
        self.assertFalse(server.is_running)
    
    def test_server_from_config(self):
        """Test starting server from YAML configuration."""
        try:
            import vllm
        except ImportError:
            raise unittest.SkipTest("vLLM is not installed")
        
        import yaml
        
        config_content = {
            'backend': 'vllm',
            'model': self.test_model,
            'port': self.test_port,
            'output_dir': self.test_output_dir,
            'startup_timeout': 300,
            'env_vars': {
                'TEST_ENV_VAR': 'test_value'
            },
            'config': {
                'api_server_args': ['--tensor-parallel-size', '1']
            }
        }
        
        config_file = os.path.join(self.test_output_dir, "test_config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        server = None
        try:
            print(f"\nStarting server from config: {config_file}")
            server = start_server_from_config(config_file)
            
            self.assertTrue(server.is_running)
            self.assertTrue(server.check_health())
            
            time.sleep(2)
            
        finally:
            if server:
                server.stop()
                self.assertFalse(server.is_running)


if __name__ == '__main__':
    unittest.main()

