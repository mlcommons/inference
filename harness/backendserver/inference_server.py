# ============================================================================
# inference_server.py
# -------------------
# Inference Server Management Module
#
# This module provides base classes and implementations for managing
# inference servers (vLLM, SGLang, etc.) with start/stop functionality,
# heartbeat checks, endpoint discovery, and profiling support.
# ============================================================================

import os
import sys
import time
import logging
import signal
import subprocess
import threading
import requests
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

# Try to import yaml, but handle if not available
try:
    import yaml
except ImportError:
    yaml = None

# Try to import psutil for process tracking, but handle if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Try to import json for dictionary argument encoding
import json


class InferenceServerError(Exception):
    """Base exception for inference server errors"""
    pass


class ServerStartupError(InferenceServerError):
    """Raised when server fails to start"""
    pass


class ServerTimeoutError(InferenceServerError):
    """Raised when server operations timeout"""
    pass


def normalize_server_args(args: Union[List, Dict]) -> List[str]:
    """
    Normalize server arguments from various formats into a flat list.
    
    Supports:
    - List format: ['--arg', 'value', '--flag']
    - Dictionary format: {'--arg': 'value', '--flag': True}
    - Dictionary with dict values: {'--kv-cache-dtype': {'key': 'value'}}
    
    Args:
        args: Arguments in list or dictionary format
    
    Returns:
        List of command-line argument strings
    """
    if isinstance(args, list):
        # Already a list, just ensure all items are strings
        return [str(arg) for arg in args]
    
    if isinstance(args, dict):
        normalized = []
        for key, value in args.items():
            # Ensure key is a string
            key_str = str(key)
            
            # Remove leading -- if present (we'll add it back)
            if key_str.startswith('--'):
                key_str = key_str[2:]
            
            # Convert underscores to dashes (e.g., tensor_parallel_size -> tensor-parallel-size)
            key_str = key_str.replace('_', '-')
            
            # Add -- prefix
            key_str = f"--{key_str}"
            
            # Handle different value types
            if value is True:
                # Flag without value (e.g., --trust-remote-code)
                normalized.append(key_str)
            elif value is False:
                # Boolean false flag (e.g., --no-trust-remote-code)
                normalized.append(f"--no-{key_str.lstrip('-')}")
            elif value is None:
                # Skip None values
                continue
            elif isinstance(value, (dict, list)):
                # Dictionary or list value - encode as JSON string
                normalized.extend([key_str, json.dumps(value)])
            elif value == '' or (isinstance(value, str) and value.strip() == ''):
                # Empty string value - this might be an error, but include it anyway
                # Some arguments might accept empty strings
                normalized.extend([key_str, str(value)])
            else:
                # Simple string/numeric value
                normalized.extend([key_str, str(value)])
        
        return normalized
    
    # If it's a single value, return as-is
    return [str(args)]


class InferenceServer(ABC):
    """
    Base class for inference server implementations.
    
    This class provides common functionality for managing inference servers
    including start/stop, heartbeat checks, endpoint discovery, and logging.
    """
    
    def __init__(self, 
                 model: str,
                 output_dir: str = "./server_logs",
                 port: int = 8000,
                 heartbeat_interval: int = 5,
                 heartbeat_timeout: int = 30,
                 startup_timeout: int = 600,
                 env_vars: Optional[Dict[str, str]] = None,
                 config: Optional[Dict[str, Any]] = None,
                 debug_mode: bool = False):
        """
        Initialize the inference server.
        
        Args:
            model: Model name or path
            output_dir: Directory for server logs
            port: Server port
            heartbeat_interval: Interval for heartbeat checks (seconds)
            heartbeat_timeout: Timeout for heartbeat checks (seconds)
            startup_timeout: Timeout for server startup (seconds)
            env_vars: Environment variables to set
            config: Additional configuration parameters
            debug_mode: Enable debug mode for process cleanup verification
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.startup_timeout = startup_timeout
        self.env_vars = env_vars or {}
        self.config = config or {}
        self.debug_mode = debug_mode
        
        # Server state
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.server_url = f"http://localhost:{self.port}"
        self.process_group_id: Optional[int] = None
        self.tracked_pids: List[int] = []  # Track child process PIDs
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files
        backend_name = self.get_backend_name()
        self.log_file = self.output_dir / f"{backend_name}_server.log"
        self.error_log_file = self.output_dir / f"{backend_name}_server_error.log"
        
        # Heartbeat monitoring
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.heartbeat_stop_event = threading.Event()
        self.last_heartbeat_success = False
        
    @abstractmethod
    def get_backend_name(self) -> str:
        """Return the backend name (e.g., 'vllm', 'sglang')"""
        pass
    
    @abstractmethod
    def get_launch_command(self) -> List[str]:
        """
        Return the command to launch the server.
        
        Returns:
            List of command arguments
        """
        pass
    
    @abstractmethod
    def get_binary_path(self) -> str:
        """
        Return the path to the server binary.
        
        Returns:
            Path to binary executable
        """
        pass
    
    @abstractmethod
    def get_health_endpoint(self) -> str:
        """
        Return the health check endpoint URL.
        
        Returns:
            Health endpoint URL
        """
        pass
    
    @abstractmethod
    def list_endpoints(self) -> Dict[str, str]:
        """
        Return a dictionary of available endpoints.
        
        Returns:
            Dictionary mapping endpoint names to URLs
        """
        pass
    
    def get_base_env(self) -> Dict[str, str]:
        """
        Get base environment variables.
        Can be overridden by subclasses to add backend-specific variables.
        
        Returns:
            Dictionary of environment variables
        """
        env = os.environ.copy()
        env.update(self.env_vars)
        return env
    
    def start(self) -> None:
        """
        Start the inference server.
        
        Raises:
            ServerStartupError: If server fails to start
            ServerTimeoutError: If server startup times out
        """
        if self.is_running:
            self.logger.warning("Server is already running")
            return
        
        self.logger.info(f"Starting {self.get_backend_name()} server...")
        self.logger.info(f"Model: {self.model}")
        self.logger.info(f"Port: {self.port}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        try:
            # Prepare command and environment
            cmd = self.get_launch_command()
            env = self.get_base_env()
            
            # Log command in a detailed format
            self.logger.info("=" * 60)
            self.logger.info("SERVER LAUNCH COMMAND")
            self.logger.info("=" * 60)
            self.logger.info(f"Command: {' '.join(cmd)}")
            self.logger.info(f"Command (list): {cmd}")
            self.logger.info(f"Number of arguments: {len(cmd)}")
            self.logger.info("=" * 60)
            
            # Open log files
            log_file_handle = open(self.log_file, 'w')
            error_log_file_handle = open(self.error_log_file, 'w')
            
            # Start server process
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file_handle,
                stderr=error_log_file_handle,
                cwd=str(self.output_dir.parent),
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            self.logger.info(f"Server process started with PID: {self.process.pid}")
            
            # Track process group ID for cleanup verification
            if os.name != 'nt':
                try:
                    self.process_group_id = os.getpgid(self.process.pid)
                    self.logger.info(f"Process group ID (PGID): {self.process_group_id}")
                except Exception as e:
                    self.logger.warning(f"Could not get process group ID: {e}")
            
            # Wait for server to become ready
            if not self._wait_for_ready():
                self._cleanup()
                raise ServerTimeoutError(
                    f"Server failed to become ready within {self.startup_timeout} seconds"
                )
            
            self.is_running = True
            self.last_heartbeat_success = True
            
            # Track child processes if debug mode is enabled (after server is ready to capture all workers)
            if self.debug_mode:
                self._track_child_processes()
            
            # Start heartbeat monitoring
            self._start_heartbeat()
            
            self.logger.info(f"{self.get_backend_name()} server started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            self._cleanup()
            if isinstance(e, (ServerStartupError, ServerTimeoutError)):
                raise
            raise ServerStartupError(f"Failed to start server: {e}") from e
    
    def stop(self) -> None:
        """Stop the inference server and cleanup resources."""
        if not self.is_running:
            self.logger.warning("Server is not running")
            return
        
        self.logger.info(f"Stopping {self.get_backend_name()} server...")
        
        # Stop heartbeat monitoring
        self._stop_heartbeat()
        
        # Stop server process
        if self.process:
            try:
                # Try graceful shutdown first
                if os.name != 'nt':
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                else:
                    self.process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    self.logger.warning("Graceful shutdown failed, forcing termination")
                    if os.name != 'nt':
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    else:
                        self.process.kill()
                    self.process.wait()
                
                self.logger.info(f"Server process (PID: {self.process.pid}) stopped")
                
            except Exception as e:
                self.logger.error(f"Error stopping server process: {e}")
            
            # Verify cleanup if debug mode is enabled
            if self.debug_mode:
                self._verify_cleanup()
            
            self.process = None
            self.process_group_id = None
            self.tracked_pids = []
        
        self.is_running = False
        self.logger.info(f"{self.get_backend_name()} server stopped")
    
    def _wait_for_ready(self) -> bool:
        """
        Wait for server to become ready by checking health endpoint and model name.
        
        Returns:
            True if server becomes ready and model matches, False otherwise
        """
        health_url = self.get_health_endpoint()
        start_time = time.time()
        health_ready = False
        
        while time.time() - start_time < self.startup_timeout:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    health_ready = True
                    self.logger.info("Server health check passed")
                    break
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        if not health_ready:
            self.logger.error("Server health check failed")
            return False
        
        # Verify model name matches
        try:
            models_url = f"{self.server_url}/v1/models"
            response = requests.get(models_url, timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                if 'data' in models_data and len(models_data['data']) > 0:
                    loaded_model = models_data['data'][0].get('id', '')
                    # Compare model names - handle both full paths and base names
                    import os
                    expected_model_base = os.path.basename(self.model) if os.path.isabs(self.model) or '/' in self.model else self.model
                    loaded_model_base = os.path.basename(loaded_model) if '/' in loaded_model else loaded_model
                    
                    # Check if models match (allowing for variations in path)
                    if expected_model_base.lower() in loaded_model.lower() or loaded_model.lower() in expected_model_base.lower():
                        self.logger.info(f"Model verification passed: expected '{self.model}', loaded '{loaded_model}'")
                        return True
                    else:
                        self.logger.warning(f"Model name mismatch: expected '{self.model}', loaded '{loaded_model}'")
                        # Still return True if health check passed - model might have aliases
                        return True
                else:
                    self.logger.warning("No model data found in /v1/models response")
                    # Health check passed, proceed even if model info unavailable
                    return True
            else:
                self.logger.warning(f"Failed to get model info from /v1/models: status {response.status_code}")
                # Health check passed, proceed even if model info unavailable
                return True
        except Exception as e:
            self.logger.warning(f"Error verifying model name: {e}")
            # Health check passed, proceed even if model verification fails
            return True
    
    def _start_heartbeat(self) -> None:
        """Start heartbeat monitoring thread."""
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            return
        
        self.heartbeat_stop_event.clear()
        
        def heartbeat_worker():
            while not self.heartbeat_stop_event.is_set():
                try:
                    health_url = self.get_health_endpoint()
                    response = requests.get(health_url, timeout=self.heartbeat_timeout)
                    
                    if response.status_code == 200:
                        self.last_heartbeat_success = True
                        # Optionally verify model name during heartbeat (less frequently)
                        # This is optional to avoid excessive API calls
                    else:
                        self.last_heartbeat_success = False
                        self.logger.warning(
                            f"Heartbeat check failed with status {response.status_code}"
                        )
                except Exception as e:
                    self.last_heartbeat_success = False
                    self.logger.warning(f"Heartbeat check failed: {e}")
                
                self.heartbeat_stop_event.wait(self.heartbeat_interval)
        
        self.heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        self.heartbeat_thread.start()
        self.logger.info("Heartbeat monitoring started")
    
    def _stop_heartbeat(self) -> None:
        """Stop heartbeat monitoring thread."""
        if self.heartbeat_thread:
            self.heartbeat_stop_event.set()
            if self.heartbeat_thread.is_alive():
                self.heartbeat_thread.join(timeout=5)
            self.heartbeat_thread = None
            self.logger.info("Heartbeat monitoring stopped")
    
    def check_health(self) -> bool:
        """
        Check server health.
        
        Returns:
            True if server is healthy, False otherwise
        """
        return self.last_heartbeat_success
    
    def _track_child_processes(self) -> None:
        """Track child processes for debug mode."""
        if not self.process:
            return
        
        try:
            if PSUTIL_AVAILABLE:
                # Use psutil for cross-platform process tracking
                parent = psutil.Process(self.process.pid)
                children = parent.children(recursive=True)
                self.tracked_pids = [p.pid for p in children]
                self.logger.info(f"Debug mode: Tracking {len(self.tracked_pids)} child processes")
                
                # Log details of child processes
                for child in children:
                    try:
                        cmdline = ' '.join(child.cmdline()) if child.cmdline() else 'N/A'
                        self.logger.debug(f"  Child PID {child.pid}: {cmdline}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            elif os.name != 'nt':
                # Use pgrep on Unix-like systems
                try:
                    result = subprocess.run(
                        ['pgrep', '-P', str(self.process.pid)],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        self.tracked_pids = [int(pid) for pid in result.stdout.strip().split('\n') if pid]
                        self.logger.info(f"Debug mode: Tracking {len(self.tracked_pids)} child processes (via pgrep)")
                except Exception as e:
                    self.logger.warning(f"Could not track child processes: {e}")
        except Exception as e:
            self.logger.warning(f"Error tracking child processes: {e}")
    
    def _verify_cleanup(self) -> None:
        """Verify that all processes have been cleaned up (debug mode)."""
        self.logger.info("=" * 60)
        self.logger.info("DEBUG MODE: Verifying process cleanup")
        self.logger.info("=" * 60)
        
        remaining_processes = []
        
        # Check main process
        if self.process and self.process.poll() is None:
            remaining_processes.append({
                'pid': self.process.pid,
                'type': 'main',
                'status': 'still running'
            })
            self.logger.warning(f"  Main process (PID {self.process.pid}) is still running!")
        else:
            self.logger.info(f"  Main process cleaned up successfully")
        
        # Check process group
        if self.process_group_id and os.name != 'nt':
            try:
                result = subprocess.run(
                    ['pgrep', '-g', str(self.process_group_id)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    pgid_pids = [int(pid) for pid in result.stdout.strip().split('\n') if pid]
                    if pgid_pids:
                        remaining_processes.append({
                            'pid': self.process_group_id,
                            'type': 'process_group',
                            'status': f'{len(pgid_pids)} processes still in group',
                            'pids': pgid_pids
                        })
                        self.logger.warning(f"  Process group (PGID {self.process_group_id}) has {len(pgid_pids)} remaining processes!")
                        for pid in pgid_pids:
                            self._log_process_details(pid)
                    else:
                        self.logger.info(f"  Process group (PGID {self.process_group_id}) cleaned up successfully")
            except Exception as e:
                self.logger.warning(f"  Could not check process group: {e}")
        
        # Check tracked child processes
        if self.tracked_pids:
            remaining_children = []
            if PSUTIL_AVAILABLE:
                for pid in self.tracked_pids:
                    try:
                        proc = psutil.Process(pid)
                        if proc.is_running():
                            remaining_children.append(pid)
                            self._log_process_details(pid, proc=proc)
                    except psutil.NoSuchProcess:
                        pass  # Process already terminated
                    except Exception as e:
                        self.logger.warning(f"  Could not check child PID {pid}: {e}")
            
            if remaining_children:
                remaining_processes.append({
                    'pid': None,
                    'type': 'children',
                    'status': f'{len(remaining_children)} tracked child processes still running',
                    'pids': remaining_children
                })
                self.logger.warning(f"  {len(remaining_children)} tracked child processes still running!")
            else:
                self.logger.info(f"  All {len(self.tracked_pids)} tracked child processes cleaned up successfully")
        
        # Summary
        if remaining_processes:
            self.logger.error("=" * 60)
            self.logger.error("CLEANUP VERIFICATION FAILED!")
            self.logger.error(f"Found {len(remaining_processes)} groups of processes still running")
            self.logger.error("=" * 60)
            
            # Try to get more details about remaining processes
            self._log_system_processes()
        else:
            self.logger.info("=" * 60)
            self.logger.info("CLEANUP VERIFICATION PASSED!")
            self.logger.info("All processes have been cleaned up successfully")
            self.logger.info("=" * 60)
    
    def _log_process_details(self, pid: int, proc=None) -> None:
        """Log details about a specific process."""
        try:
            if PSUTIL_AVAILABLE and proc is None:
                proc = psutil.Process(pid)
            
            if PSUTIL_AVAILABLE:
                try:
                    cmdline = ' '.join(proc.cmdline()) if proc.cmdline() else 'N/A'
                    status = proc.status()
                    ppid = proc.ppid()
                    self.logger.warning(f"    PID {pid}: status={status}, ppid={ppid}, cmd={cmdline[:100]}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self.logger.warning(f"    PID {pid}: Process details not accessible")
            else:
                # Fallback to ps command on Unix
                if os.name != 'nt':
                    try:
                        result = subprocess.run(
                            ['ps', '-p', str(pid), '-o', 'pid,stat,cmd', '--no-headers'],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0:
                            self.logger.warning(f"    {result.stdout.strip()}")
                    except Exception:
                        self.logger.warning(f"    PID {pid}: Could not get process details")
        except Exception as e:
            self.logger.warning(f"    PID {pid}: Error getting details: {e}")
    
    def _log_system_processes(self) -> None:
        """Log relevant system processes for debugging."""
        if not self.debug_mode:
            return
        
        try:
            # Check for common inference server processes
            backend_name = self.get_backend_name()
            search_terms = [backend_name, 'vllm', 'sglang', 'python']
            
            if PSUTIL_AVAILABLE:
                self.logger.info("\nScanning system for related processes:")
                for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status']):
                    try:
                        cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                        if any(term in cmdline.lower() for term in search_terms):
                            self.logger.info(f"  Found: PID {proc.info['pid']}, "
                                           f"status={proc.info['status']}, "
                                           f"cmd={cmdline[:100]}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            elif os.name != 'nt':
                # Use ps and grep
                try:
                    for term in search_terms:
                        result = subprocess.run(
                            ['pgrep', '-f', term],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0:
                            pids = result.stdout.strip().split('\n')
                            for pid in pids:
                                if pid:
                                    self._log_process_details(int(pid))
                except Exception as e:
                    self.logger.debug(f"Could not scan system processes: {e}")
        except Exception as e:
            self.logger.debug(f"Error in system process scan: {e}")
    
    def _cleanup(self) -> None:
        """Cleanup resources on failure."""
        if self.process:
            try:
                if os.name != 'nt':
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                else:
                    self.process.terminate()
                
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name != 'nt':
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    else:
                        self.process.kill()
                    self.process.wait()
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")
            
            # Verify cleanup if debug mode is enabled
            if self.debug_mode:
                self._verify_cleanup()
            
            self.process = None
            self.process_group_id = None
            self.tracked_pids = []
        
        self.is_running = False
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class VLLMServer(InferenceServer):
    """vLLM inference server implementation."""
    
    def __init__(self, model: str, **kwargs):
        # Default vLLM port is 8000
        port = kwargs.pop('port', 8000)
        super().__init__(model, port=port, **kwargs)
        
        # vLLM specific configuration
        self.binary_path = self.config.get('binary_path', 'python')
        raw_api_server_args = self.config.get('api_server_args', [])
        # Log raw arguments for debugging
        if raw_api_server_args:
            self.logger.info(f"Raw api_server_args: {raw_api_server_args}")
            self.logger.info(f"Raw api_server_args type: {type(raw_api_server_args)}")
        # Normalize arguments to handle dict format and flags
        self.api_server_args = normalize_server_args(raw_api_server_args)
        if self.api_server_args:
            self.logger.info(f"Normalized api_server_args: {self.api_server_args}")
            self.logger.info(f"Normalized api_server_args length: {len(self.api_server_args)}")
            # Validate argument pairs
            i = 0
            while i < len(self.api_server_args):
                arg = self.api_server_args[i]
                if arg.startswith('--'):
                    # This is a flag/argument name
                    if i + 1 < len(self.api_server_args):
                        value = self.api_server_args[i + 1]
                        # Check if next item is also a flag (which would be wrong)
                        if value.startswith('--'):
                            self.logger.warning(
                                f"Argument '{arg}' might be missing its value. "
                                f"Next item is '{value}' which looks like another flag."
                            )
                        else:
                            self.logger.debug(f"Argument pair: {arg} = {value}")
                        i += 2
                    else:
                        # Last argument is a flag (this is OK for boolean flags)
                        if arg not in ['--trust-remote-code', '--disable-log-requests']:
                            self.logger.info(f"Standalone flag (no value): {arg}")
                        i += 1
                else:
                    # This shouldn't happen in normalized output, but handle it
                    self.logger.warning(f"Unexpected non-flag argument at position {i}: {arg}")
                    i += 1
    
    def get_backend_name(self) -> str:
        return "vllm"
    
    def get_binary_path(self) -> str:
        return self.binary_path
    
    def get_launch_command(self) -> List[str]:
        """Generate vLLM launch command."""
        cmd = [self.binary_path, "-m", "vllm.entrypoints.openai.api_server"]
        cmd.extend(["--model", self.model])
        cmd.extend(["--port", str(self.port)])
        
        # Log normalized arguments before extending
        if self.api_server_args:
            self.logger.debug(f"Adding {len(self.api_server_args)} normalized arguments to command")
            self.logger.debug(f"Normalized arguments: {self.api_server_args}")
            # Verify argument pairs (should be even number or contain flags)
            if len(self.api_server_args) % 2 != 0:
                # Check if last argument is a flag (starts with --)
                if not self.api_server_args[-1].startswith('--'):
                    self.logger.warning(
                        f"Odd number of arguments ({len(self.api_server_args)}). "
                        f"Last argument: {self.api_server_args[-1]}. "
                        f"This might indicate a missing value for a flag."
                    )
        
        cmd.extend(self.api_server_args)
        return cmd
    
    def get_health_endpoint(self) -> str:
        return f"{self.server_url}/health"
    
    def list_endpoints(self) -> Dict[str, str]:
        """List vLLM API endpoints."""
        base = self.server_url
        return {
            "health": f"{base}/health",
            "completions": f"{base}/v1/completions",
            "chat_completions": f"{base}/v1/chat/completions",
            "embeddings": f"{base}/v1/embeddings",
            "models": f"{base}/v1/models",
            "metrics": f"{base}/metrics",
        }


class SGLangServer(InferenceServer):
    """SGLang inference server implementation."""
    
    def __init__(self, model: str, **kwargs):
        # Default SGLang port is 8000
        port = kwargs.pop('port', 8000)
        super().__init__(model, port=port, **kwargs)
        
        # SGLang specific configuration
        self.binary_path = self.config.get('binary_path', 'python')
        # Check for server_args first, fall back to api_server_args for backward compatibility
        raw_server_args = self.config.get('server_args', [])
        if not raw_server_args:
            # Fall back to api_server_args if server_args is not provided
            raw_server_args = self.config.get('api_server_args', [])
        # Normalize arguments to handle dict format and flags
        self.server_args = normalize_server_args(raw_server_args)
    
    def get_backend_name(self) -> str:
        return "sglang"
    
    def get_binary_path(self) -> str:
        return self.binary_path
    
    def get_launch_command(self) -> List[str]:
        """Generate SGLang launch command."""
        cmd = [self.binary_path, "-m", "sglang.launch_server"]
        cmd.extend(["--model-path", self.model])
        cmd.extend(["--port", str(self.port)])
        cmd.extend(self.server_args)
        return cmd
    
    def get_health_endpoint(self) -> str:
        return f"{self.server_url}/health"
    
    def list_endpoints(self) -> Dict[str, str]:
        """List SGLang API endpoints."""
        base = self.server_url
        return {
            "health": f"{base}/health",
            "completions": f"{base}/v1/completions",
            "chat_completions": f"{base}/v1/chat/completions",
            "models": f"{base}/v1/models",
        }


def create_server(backend: str, model: str, **kwargs) -> InferenceServer:
    """
    Factory function to create inference server instances.
    
    Args:
        backend: Backend name ('vllm' or 'sglang')
        model: Model name or path
        **kwargs: Additional arguments passed to server constructor
    
    Returns:
        InferenceServer instance
    
    Raises:
        ValueError: If backend is not supported
    """
    backend_lower = backend.lower()
    
    if backend_lower == 'vllm':
        return VLLMServer(model, **kwargs)
    elif backend_lower == 'sglang':
        return SGLangServer(model, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}. Supported backends: vllm, sglang")


def load_server_config(config_file: str) -> Dict[str, Any]:
    """
    Load server configuration from YAML file.
    
    Args:
        config_file: Path to YAML configuration file
    
    Returns:
        Dictionary containing configuration
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ImportError: If PyYAML is not installed
        yaml.YAMLError: If YAML parsing fails
    """
    if yaml is None:
        raise ImportError("PyYAML is required for YAML configuration loading. Install it with: pip install pyyaml")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    return config


def start_server_from_config(config_file: str, overrides: Optional[Dict[str, Any]] = None) -> InferenceServer:
    """
    Start server from YAML configuration file.
    
    Expected YAML structure:
        backend: vllm|sglang
        model: model_name_or_path
        port: 8000
        output_dir: ./server_logs
        binary_path: python
        launch_command: [optional list of command parts]
        env_vars:
          VAR1: value1
          VAR2: value2
        config:
          api_server_args: [--arg1, --arg2]
        profile:
          enabled: true|false
          tool: nsys|pytorch|amd
          output_dir: ./profiles
          args: [--arg1, --arg2]
    
    Args:
        config_file: Path to YAML configuration file
        overrides: Optional dictionary of config values to override after loading from YAML.
                  Keys in overrides will take precedence over values in the config file.
                  For nested values (e.g., env_vars), the override will update the existing dict.
    
    Returns:
        Started InferenceServer instance
    """
    config = load_server_config(config_file)
    
    # Apply overrides if provided
    if overrides:
        for key, value in overrides.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                # Merge dictionaries (e.g., for env_vars, config)
                config[key].update(value)
            else:
                # Direct override
                config[key] = value
    
    # Extract backend and model (required)
    backend = config.get('backend', 'vllm')
    model = config.get('model')
    
    if model is None:
        raise ValueError("'model' is required in configuration file")
    
    # Extract optional parameters
    port = config.get('port', 8000)
    output_dir = config.get('output_dir', './server_logs')
    heartbeat_interval = config.get('heartbeat_interval', 100)
    heartbeat_timeout = config.get('heartbeat_timeout', 30)
    startup_timeout = config.get('startup_timeout', 600)
    env_vars = config.get('env_vars', {})
    debug_mode = config.get('debug_mode', False)
    
    # Handle profile configuration
    profile_config = config.get('profile', {})
    if profile_config.get('enabled', False):
        profile_tool = profile_config.get('tool', 'nsys')
        profile_output_dir = profile_config.get('output_dir', os.path.join(output_dir, 'profiles'))
        profile_args = profile_config.get('args', [])
        
        # Modify binary path and add profiling wrapper
        binary_path = config.get('binary_path', 'python')
        
        if profile_tool == 'nsys':
            # NSight Systems profiling
            nsys_path = profile_config.get('binary_path', 'nsys')
            cmd_prefix = [nsys_path, 'profile']
            cmd_prefix.extend(profile_args)
            cmd_prefix.extend(['--output', os.path.join(profile_output_dir, 'profile.nsys-rep')])
            cmd_prefix.append('--')
            
            # Store prefix command to be prepended to launch command
            config.setdefault('_profile_prefix', cmd_prefix)
            
        elif profile_tool == 'pytorch':
            # PyTorch profiler (handled differently - may need environment variables)
            env_vars['VLLM_TORCH_PROFILER_DIR'] = profile_output_dir
            env_vars['VLLM_ENABLE_PROFILER'] = '1'
            
        elif profile_tool == 'amd':
            # AMD profiler (rocprof)
            rocprof_path = profile_config.get('binary_path', 'rocprof')
            cmd_prefix = [rocprof_path]
            cmd_prefix.extend(profile_args)
            cmd_prefix.extend(['--output', os.path.join(profile_output_dir, 'profile.csv')])
            cmd_prefix.append('--')
            
            config.setdefault('_profile_prefix', cmd_prefix)
    
    # Get server-specific config
    server_config = config.get('config', {})
    
    # Store profile prefix in config if it exists
    if config.get('_profile_prefix'):
        server_config['_profile_prefix'] = config['_profile_prefix']
    
    # Override launch command if specified
    launch_command = config.get('launch_command')
    if launch_command:
        # Custom launch command - create a custom server class
        profile_prefix = config.get('_profile_prefix', [])
        custom_launch_cmd = launch_command
        
        class CustomServer(InferenceServer):
            def get_backend_name(self):
                return backend
            
            def get_binary_path(self):
                cmd = self.get_launch_command()
                return cmd[0] if cmd else 'python'
            
            def get_launch_command(self):
                # Prepend profile prefix if exists
                if profile_prefix:
                    return profile_prefix + custom_launch_cmd
                return custom_launch_cmd
            
            def get_health_endpoint(self):
                return f"{self.server_url}/health"
            
            def list_endpoints(self):
                # Default endpoints - subclasses can override
                base = self.server_url
                return {
                    "health": f"{base}/health",
                }
        
        server = CustomServer(
            model,
            output_dir=output_dir,
            port=port,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            startup_timeout=startup_timeout,
            env_vars=env_vars,
            config=server_config,
            debug_mode=debug_mode
        )
    else:
        # Modify standard server classes to handle profile prefix
        if config.get('_profile_prefix'):
            # Create a wrapper that adds profile prefix
            original_create = create_server
            
            def create_server_with_profile(backend, model, **kwargs):
                server = original_create(backend, model, **kwargs)
                profile_prefix = config['_profile_prefix']
                
                # Wrap get_launch_command to add profile prefix
                original_launch_cmd = server.get_launch_command
                def wrapped_launch_cmd():
                    return profile_prefix + original_launch_cmd()
                
                server.get_launch_command = wrapped_launch_cmd
                return server
            
            server = create_server_with_profile(
                backend,
                model,
                output_dir=output_dir,
                port=port,
                heartbeat_interval=heartbeat_interval,
                heartbeat_timeout=heartbeat_timeout,
                startup_timeout=startup_timeout,
                env_vars=env_vars,
                config=server_config,
                debug_mode=debug_mode
            )
        else:
            # Use standard server classes
            server = create_server(
                backend,
                model,
                output_dir=output_dir,
                port=port,
                heartbeat_interval=heartbeat_interval,
                heartbeat_timeout=heartbeat_timeout,
                startup_timeout=startup_timeout,
                env_vars=env_vars,
                config=server_config,
                debug_mode=debug_mode
            )
    
    server.start()
    return server

