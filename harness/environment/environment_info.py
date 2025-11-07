#!/usr/bin/env python3
"""
Environment Information Collection Module

Collects system information, software versions, and git status for reproducibility.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class EnvironmentInfoCollector:
    """
    Collects environment information including:
    - Python packages (pip list)
    - Git version and status
    - System information
    - GPU information (nvidia-smi)
    - Environment variables
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize environment info collector.
        
        Args:
            output_dir: Directory to save environment information files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def collect_all(self) -> Dict[str, Any]:
        """
        Collect all environment information.
        
        Returns:
            Dictionary with collection results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'success': {},
            'errors': {}
        }
        
        try:
            results['success']['pip_list'] = self.collect_pip_list()
        except Exception as e:
            results['errors']['pip_list'] = str(e)
            self.logger.warning(f"Failed to collect pip list: {e}")
        
        try:
            results['success']['git_info'] = self.collect_git_info()
        except Exception as e:
            results['errors']['git_info'] = str(e)
            self.logger.warning(f"Failed to collect git info: {e}")
        
        try:
            results['success']['system_info'] = self.collect_system_info()
        except Exception as e:
            results['errors']['system_info'] = str(e)
            self.logger.warning(f"Failed to collect system info: {e}")
        
        try:
            results['success']['gpu_info'] = self.collect_gpu_info()
        except Exception as e:
            results['errors']['gpu_info'] = str(e)
            self.logger.warning(f"Failed to collect GPU info: {e}")
        
        try:
            results['success']['environment_vars'] = self.collect_environment_vars()
        except Exception as e:
            results['errors']['environment_vars'] = str(e)
            self.logger.warning(f"Failed to collect environment vars: {e}")
        
        return results
    
    def collect_pip_list(self) -> str:
        """Collect pip list output."""
        self.logger.info("Collecting pip list...")
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list'],
                capture_output=True,
                text=True,
                check=True
            )
            pip_output = result.stdout
            
            # Save to file
            pip_file = self.output_dir / "pip_list.txt"
            with open(pip_file, 'w') as f:
                f.write(pip_output)
            
            self.logger.info(f"Pip list saved to: {pip_file}")
            return pip_output
        except subprocess.CalledProcessError as e:
            self.logger.error(f"pip list command failed: {e.stderr}")
            raise
        except Exception as e:
            self.logger.error(f"Error collecting pip list: {e}")
            raise
    
    def collect_git_info(self) -> Dict[str, Optional[str]]:
        """Collect git version and status."""
        self.logger.info("Collecting git information...")
        git_info = {}
        
        # Try to find git repository root (start from current directory and go up)
        current_dir = Path.cwd()
        git_root = None
        for path in [current_dir] + list(current_dir.parents):
            if (path / '.git').exists():
                git_root = path
                break
        
        if git_root:
            git_info['repository_root'] = str(git_root)
            
            # Git version
            try:
                result = subprocess.run(
                    ['git', '--version'],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=git_root
                )
                git_info['git_version'] = result.stdout.strip()
            except Exception as e:
                self.logger.warning(f"Could not get git version: {e}")
                git_info['git_version'] = None
            
            # Git current commit
            try:
                result = subprocess.run(
                    ['git', 'rev-parse', 'HEAD'],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=git_root
                )
                git_info['current_commit'] = result.stdout.strip()
            except Exception as e:
                self.logger.warning(f"Could not get git commit: {e}")
                git_info['current_commit'] = None
            
            # Git branch
            try:
                result = subprocess.run(
                    ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=git_root
                )
                git_info['current_branch'] = result.stdout.strip()
            except Exception as e:
                self.logger.warning(f"Could not get git branch: {e}")
                git_info['current_branch'] = None
            
            # Git status
            try:
                result = subprocess.run(
                    ['git', 'status', '--short'],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=git_root
                )
                git_info['git_status'] = result.stdout.strip()
            except Exception as e:
                self.logger.warning(f"Could not get git status: {e}")
                git_info['git_status'] = None
            
            # Git diff
            try:
                result = subprocess.run(
                    ['git', 'diff'],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=git_root
                )
                git_info['git_diff'] = result.stdout
            except Exception as e:
                self.logger.warning(f"Could not get git diff: {e}")
                git_info['git_diff'] = None
            
            # Save git info to files
            git_info_file = self.output_dir / "git_info.txt"
            with open(git_info_file, 'w') as f:
                f.write(f"Git Repository Root: {git_info.get('repository_root', 'N/A')}\n")
                f.write(f"Git Version: {git_info.get('git_version', 'N/A')}\n")
                f.write(f"Current Commit: {git_info.get('current_commit', 'N/A')}\n")
                f.write(f"Current Branch: {git_info.get('current_branch', 'N/A')}\n")
                f.write("\n=== Git Status ===\n")
                f.write(git_info.get('git_status', 'N/A') or 'No changes')
                f.write("\n\n=== Git Diff ===\n")
                f.write(git_info.get('git_diff', 'N/A') or 'No differences')
            
            self.logger.info(f"Git info saved to: {git_info_file}")
        else:
            self.logger.warning("No git repository found")
            git_info['repository_root'] = None
        
        return git_info
    
    def collect_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        self.logger.info("Collecting system information...")
        import platform
        
        system_info = {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_executable': sys.executable,
            'cpu_count': os.cpu_count(),
        }
        
        # Collect lscpu output
        lscpu_output = None
        try:
            self.logger.info("Collecting lscpu information...")
            result = subprocess.run(
                ['lscpu'],
                capture_output=True,
                text=True,
                check=True
            )
            lscpu_output = result.stdout
            self.logger.info("Successfully collected lscpu information")
        except FileNotFoundError:
            self.logger.warning("lscpu command not found - skipping CPU details")
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"lscpu command failed: {e.stderr}")
        except Exception as e:
            self.logger.warning(f"Error collecting lscpu info: {e}")
        
        # Save to file
        system_info_file = self.output_dir / "system_info.txt"
        with open(system_info_file, 'w') as f:
            f.write("=== System Information ===\n\n")
            for key, value in system_info.items():
                f.write(f"{key}: {value}\n")
            
            # Add lscpu output if available
            if lscpu_output:
                f.write("\n\n=== CPU Information (lscpu) ===\n\n")
                f.write(lscpu_output)
        
        self.logger.info(f"System info saved to: {system_info_file}")
        return system_info
    
    def collect_gpu_info(self) -> Optional[str]:
        """Collect GPU information using nvidia-smi."""
        self.logger.info("Collecting GPU information...")
        try:
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                text=True,
                check=True
            )
            gpu_output = result.stdout
            
            # Also get detailed GPU info
            try:
                result_detailed = subprocess.run(
                    ['nvidia-smi', '-q'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                gpu_detailed = result_detailed.stdout
            except Exception:
                gpu_detailed = None
            
            # Save to file
            gpu_file = self.output_dir / "nvidia_smi.txt"
            with open(gpu_file, 'w') as f:
                f.write("=== nvidia-smi (quick) ===\n\n")
                f.write(gpu_output)
                if gpu_detailed:
                    f.write("\n\n=== nvidia-smi (detailed) ===\n\n")
                    f.write(gpu_detailed)
            
            self.logger.info(f"GPU info saved to: {gpu_file}")
            return gpu_output
        except FileNotFoundError:
            self.logger.warning("nvidia-smi not found - skipping GPU info collection")
            return None
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"nvidia-smi command failed: {e.stderr}")
            return None
        except Exception as e:
            self.logger.warning(f"Error collecting GPU info: {e}")
            return None
    
    def collect_environment_vars(self) -> Dict[str, str]:
        """Collect relevant environment variables."""
        self.logger.info("Collecting environment variables...")
        
        # Filter for relevant environment variables
        relevant_vars = [
            'PATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_PATH',
            'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
            'OMP_NUM_THREADS', 'PYTHONPATH', 'PYTHONUNBUFFERED',
            'TORCH_CUDA_ARCH_LIST', 'VLLM_NO_USAGE_STATS',
            'HOME', 'USER', 'PWD'
        ]
        
        env_vars = {}
        for var in relevant_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]
        
        # Save to file
        env_file = self.output_dir / "environment_vars.txt"
        with open(env_file, 'w') as f:
            f.write("=== Relevant Environment Variables ===\n\n")
            for key, value in sorted(env_vars.items()):
                f.write(f"{key}={value}\n")
        
        self.logger.info(f"Environment variables saved to: {env_file}")
        return env_vars

