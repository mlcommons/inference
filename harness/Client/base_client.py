# ============================================================================
# base_client.py
# --------------
# Base client class for different harness clients
# Supports LoadGen, GuideLLM, and extensible for new clients
# ============================================================================

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseClient(ABC):
    """
    Base class for harness clients.
    
    All clients must implement the interface methods to work with
    the harness infrastructure.
    """
    
    def __init__(self, 
                 client_name: str,
                 model_name: str,
                 dataset_path: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize base client.
        
        Args:
            client_name: Name of the client (e.g., 'loadgen', 'guidellm')
            model_name: Model name or path
            dataset_path: Path to dataset file
            config: Additional configuration dictionary
        """
        self.client_name = client_name
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Client state
        self.is_initialized = False
        self.is_running = False
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the client.
        This should set up datasets, models, and any required resources.
        """
        pass
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Run the client's main execution.
        
        Returns:
            Dictionary with results and metrics
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup resources after execution.
        """
        pass
    
    def get_name(self) -> str:
        """Get client name."""
        return self.client_name
    
    def get_config(self) -> Dict[str, Any]:
        """Get client configuration."""
        return self.config
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

