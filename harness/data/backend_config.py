# ============================================================================
# backend_config.py
# -----------------
# Backend configuration loader for validating available endpoints
# ============================================================================

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List


class BackendConfigLoader:
    """Loader for backend configuration YAML files."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize backend configuration loader.
        
        Args:
            config_dir: Directory containing backend configuration files.
                       Defaults to harness/configs/backends/
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Default to harness/configs/backends/
            current_file = Path(__file__).resolve()
            # Go up from data/ to harness/, then to configs/backends/
            self.config_dir = current_file.parent.parent / "configs" / "backends"
        
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Backend config directory: {self.config_dir}")
    
    def load_backend_config(self, backend_name: str) -> Dict[str, Any]:
        """
        Load backend configuration by name.
        
        Args:
            backend_name: Name of the backend (e.g., "vllm", "sglang")
        
        Returns:
            Dictionary with backend configuration
        """
        config_path = self.config_dir / f"{backend_name}.yaml"
        
        if config_path.exists():
            return self._load_config_file(config_path)
        else:
            # Return default config if file doesn't exist
            self.logger.warning(f"No config file found for backend '{backend_name}', using defaults")
            return {
                'name': backend_name,
                'endpoints': ['completions', 'chat_completions'],  # Default to both
                'description': f"Default configuration for {backend_name}"
            }
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Validate required fields
            if 'endpoints' not in config_data:
                config_data['endpoints'] = ['completions', 'chat_completions']
            
            self.logger.info(f"Loaded backend config from: {config_path}")
            return config_data
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def validate_endpoint(self, backend_name: str, endpoint_type: str) -> bool:
        """
        Validate that an endpoint exists for a backend.
        
        Args:
            backend_name: Name of the backend
            endpoint_type: Type of endpoint ('completions' or 'chat_completions')
        
        Returns:
            True if endpoint is available, False otherwise
        """
        config = self.load_backend_config(backend_name)
        available_endpoints = config.get('endpoints', [])
        return endpoint_type in available_endpoints

