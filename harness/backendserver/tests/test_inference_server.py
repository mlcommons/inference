#!/usr/bin/env python3
"""
Tests for inference_server module.

These tests verify start/stop functionality, heartbeat checks,
endpoint discovery, and configuration loading.
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
    InferenceServer,
    VLLMServer,
    SGLangServer,
    create_server,
    load_server_config,
    start_server_from_config,
    ServerStartupError,
    ServerTimeoutError,
)


class TestInferenceServerBase(unittest.TestCase):
    """Base test class with common setup."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_output_dir = tempfile.mkdtemp(prefix="test_inference_server_")
        self.test_model = "test-model"
        self.test_port = 8001  # Use different port to avoid conflicts
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)


class TestInferenceServer(TestInferenceServerBase):
    """Test InferenceServer base class functionality."""
    
    def test_server_initialization(self):
        """Test server initialization."""
        server = VLLMServer(
            model=self.test_model,
            output_dir=self.test_output_dir,
            port=self.test_port
        )
        
        self.assertEqual(server.model, self.test_model)
        self.assertEqual(server.port, self.test_port)
        self.assertEqual(server.output_dir, Path(self.test_output_dir))
        self.assertFalse(server.is_running)
        self.assertIsNone(server.process)
    
    def test_backend_name(self):
        """Test backend name retrieval."""
        vllm_server = VLLMServer(model=self.test_model)
        self.assertEqual(vllm_server.get_backend_name(), "vllm")
        
        sglang_server = SGLangServer(model=self.test_model)
        self.assertEqual(sglang_server.get_backend_name(), "sglang")
    
    def test_launch_command(self):
        """Test launch command generation."""
        server = VLLMServer(model=self.test_model, port=self.test_port)
        cmd = server.get_launch_command()
        
        self.assertIsInstance(cmd, list)
        self.assertIn("-m", cmd)
        self.assertIn("vllm.entrypoints.openai.api_server", cmd)
        self.assertIn("--model", cmd)
        self.assertIn(self.test_model, cmd)
        self.assertIn("--port", cmd)
        self.assertIn(str(self.test_port), cmd)
    
    def test_endpoints(self):
        """Test endpoint listing."""
        server = VLLMServer(model=self.test_model, port=self.test_port)
        endpoints = server.list_endpoints()
        
        self.assertIsInstance(endpoints, dict)
        self.assertIn("health", endpoints)
        self.assertIn("completions", endpoints)
        self.assertIn("metrics", endpoints)
        
        # Check endpoint URLs
        self.assertEqual(endpoints["health"], f"http://localhost:{self.test_port}/health")
    
    def test_environment_variables(self):
        """Test environment variable handling."""
        env_vars = {
            "TEST_VAR1": "test_value1",
            "TEST_VAR2": "test_value2"
        }
        
        server = VLLMServer(
            model=self.test_model,
            env_vars=env_vars
        )
        
        base_env = server.get_base_env()
        self.assertEqual(base_env["TEST_VAR1"], "test_value1")
        self.assertEqual(base_env["TEST_VAR2"], "test_value2")
    
    def test_log_file_creation(self):
        """Test log file path creation."""
        server = VLLMServer(
            model=self.test_model,
            output_dir=self.test_output_dir
        )
        
        self.assertTrue(server.log_file.parent.exists())
        self.assertEqual(server.log_file.parent, Path(self.test_output_dir))
        self.assertIn("vllm_server", str(server.log_file))
    
    def test_heartbeat_configuration(self):
        """Test heartbeat configuration."""
        server = VLLMServer(
            model=self.test_model,
            heartbeat_interval=10,
            heartbeat_timeout=60
        )
        
        self.assertEqual(server.heartbeat_interval, 10)
        self.assertEqual(server.heartbeat_timeout, 60)


class TestServerFactory(TestInferenceServerBase):
    """Test server factory function."""
    
    def test_create_vllm_server(self):
        """Test creating vLLM server via factory."""
        server = create_server(
            backend="vllm",
            model=self.test_model,
            output_dir=self.test_output_dir,
            port=self.test_port
        )
        
        self.assertIsInstance(server, VLLMServer)
        self.assertEqual(server.model, self.test_model)
    
    def test_create_sglang_server(self):
        """Test creating SGLang server via factory."""
        server = create_server(
            backend="sglang",
            model=self.test_model,
            output_dir=self.test_output_dir,
            port=self.test_port
        )
        
        self.assertIsInstance(server, SGLangServer)
        self.assertEqual(server.model, self.test_model)
    
    def test_create_server_invalid_backend(self):
        """Test creating server with invalid backend."""
        with self.assertRaises(ValueError) as context:
            create_server(backend="invalid", model=self.test_model)
        
        self.assertIn("Unsupported backend", str(context.exception))


class TestConfigurationLoading(TestInferenceServerBase):
    """Test configuration loading from YAML."""
    
    def test_load_config_from_yaml(self):
        """Test loading configuration from YAML file."""
        config_content = """
backend: vllm
model: test-model
port: 8001
output_dir: ./test_logs
heartbeat_interval: 5
heartbeat_timeout: 30
startup_timeout: 600
env_vars:
  TEST_VAR: test_value
config:
  api_server_args:
    - --tensor-parallel-size
    - "1"
"""
        config_file = os.path.join(self.test_output_dir, "test_config.yaml")
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        config = load_server_config(config_file)
        
        self.assertEqual(config["backend"], "vllm")
        self.assertEqual(config["model"], "test-model")
        self.assertEqual(config["port"], 8001)
        self.assertIn("env_vars", config)
        self.assertIn("config", config)
    
    def test_load_config_missing_file(self):
        """Test loading non-existent configuration file."""
        with self.assertRaises(FileNotFoundError):
            load_server_config("nonexistent_config.yaml")
    
    def test_config_with_profile_nsys(self):
        """Test configuration with NSight Systems profiling."""
        config_content = """
backend: vllm
model: test-model
port: 8001
profile:
  enabled: true
  tool: nsys
  output_dir: ./profiles
  args:
    - --trace=cuda,nvtx
"""
        config_file = os.path.join(self.test_output_dir, "test_config_profile.yaml")
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        config = load_server_config(config_file)
        
        self.assertTrue(config.get("profile", {}).get("enabled", False))
        self.assertEqual(config["profile"]["tool"], "nsys")
    
    def test_config_with_profile_pytorch(self):
        """Test configuration with PyTorch profiling."""
        config_content = """
backend: vllm
model: test-model
port: 8001
profile:
  enabled: true
  tool: pytorch
  output_dir: ./profiles
"""
        config_file = os.path.join(self.test_output_dir, "test_config_pytorch.yaml")
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        config = load_server_config(config_file)
        
        profile = config.get("profile", {})
        self.assertTrue(profile.get("enabled", False))
        self.assertEqual(profile["tool"], "pytorch")
    
    def test_config_with_profile_amd(self):
        """Test configuration with AMD profiler."""
        config_content = """
backend: vllm
model: test-model
port: 8001
profile:
  enabled: true
  tool: amd
  output_dir: ./profiles
  args:
    - --stats
"""
        config_file = os.path.join(self.test_output_dir, "test_config_amd.yaml")
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        config = load_server_config(config_file)
        
        profile = config.get("profile", {})
        self.assertTrue(profile.get("enabled", False))
        self.assertEqual(profile["tool"], "amd")
    
    def test_config_with_custom_launch_command(self):
        """Test configuration with custom launch command."""
        config_content = """
backend: vllm
model: test-model
port: 8001
launch_command:
  - python
  - -m
  - custom.server
  - --model
  - test-model
"""
        config_file = os.path.join(self.test_output_dir, "test_config_custom.yaml")
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        config = load_server_config(config_file)
        
        self.assertIn("launch_command", config)
        self.assertIsInstance(config["launch_command"], list)


class TestServerContextManager(TestInferenceServerBase):
    """Test server context manager functionality."""
    
    def test_context_manager(self):
        """Test using server as context manager."""
        # Note: This test doesn't actually start a real server
        # It just verifies the context manager protocol
        server = VLLMServer(
            model=self.test_model,
            output_dir=self.test_output_dir,
            port=self.test_port
        )
        
        # Context manager should not start server without actual start() call
        # We're just testing the protocol exists
        self.assertTrue(hasattr(server, '__enter__'))
        self.assertTrue(hasattr(server, '__exit__'))


class TestServerErrors(TestInferenceServerBase):
    """Test server error handling."""
    
    def test_start_server_already_running(self):
        """Test starting server that's already running."""
        server = VLLMServer(
            model=self.test_model,
            output_dir=self.test_output_dir,
            port=self.test_port
        )
        
        # Manually set running state (simulating already running server)
        server.is_running = True
        
        # Should log warning but not raise exception
        server.start()  # Should not raise exception
    
    def test_stop_server_not_running(self):
        """Test stopping server that's not running."""
        server = VLLMServer(
            model=self.test_model,
            output_dir=self.test_output_dir,
            port=self.test_port
        )
        
        # Should log warning but not raise exception
        server.stop()  # Should not raise exception


if __name__ == '__main__':
    unittest.main()

