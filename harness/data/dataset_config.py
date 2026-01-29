# ============================================================================
# dataset_config.py
# -----------------
# Dataset configuration loader for handling different dataset formats
# and field mappings through YAML configuration files
# ============================================================================

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class DatasetFieldMapping:
    """Configuration for dataset field mappings."""
    input_column: str = "input"
    input_ids_column: str = "tok_input"
    output_column: str = "output"
    input_lens_column: Optional[str] = None  # If None, will be calculated from input_ids


@dataclass
class DatasetConfig:
    """Complete dataset configuration."""
    name: str
    description: Optional[str] = None
    fields: DatasetFieldMapping = field(default_factory=DatasetFieldMapping)
    total_sample_count: Optional[int] = None
    file_format: str = "auto"  # auto, json, pickle, csv
    model_specific: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelDatasetConfig:
    """Model-specific dataset configuration."""
    model_name: str
    dataset_name: str
    dataset_config: DatasetConfig
    scenario_specific: Dict[str, Any] = field(default_factory=dict)  # Offline/Server specific overrides


class DatasetConfigLoader:
    """Loader for dataset configuration YAML files."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing dataset configuration files.
                       Defaults to harness/data/configs/
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Default to harness/configs/
            current_file = Path(__file__).resolve()
            # Go up from data/ to harness/, then to configs/
            self.config_dir = current_file.parent.parent / "configs"
        
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.datasets_dir = self.config_dir / "datasets"
        self.models_dir = self.config_dir / "models"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Config directory: {self.config_dir}")
        self.logger.info(f"Datasets config directory: {self.datasets_dir}")
        self.logger.info(f"Models config directory: {self.models_dir}")
    
    def load_dataset_config(self, dataset_name: str, model_name: Optional[str] = None, 
                           config_file: Optional[str] = None) -> DatasetConfig:
        """
        Load dataset configuration by name.
        
        Args:
            dataset_name: Name of the dataset (e.g., "llama3.1-8b", "deepseek-r1")
            model_name: Optional model name for model-specific configs
            config_file: Optional path to specific config file (overrides auto-detection)
        
        Returns:
            DatasetConfig object
        """
        # If config_file is specified, use it directly
        if config_file:
            config_path = Path(config_file)
            if not config_path.is_absolute():
                # Try relative to config directories
                for search_dir in [self.datasets_dir, self.models_dir, self.config_dir]:
                    candidate = search_dir / config_file
                    if candidate.exists():
                        config_path = candidate
                        break
                else:
                    # If not found, assume it's relative to current directory
                    config_path = Path(config_file).resolve()
            
            if config_path.exists():
                return self._load_config_file(config_path, dataset_name)
            else:
                raise FileNotFoundError(f"Config file not found: {config_file}")
        
        # First try model-specific config in models/ directory
        if model_name:
            # Try models/{model_name}/{dataset_name}.yaml
            model_dataset_path = self.models_dir / model_name / f"{dataset_name}.yaml"
            if model_dataset_path.exists():
                return self._load_config_file(model_dataset_path, dataset_name)
            
            # Try models/{model_name}.yaml (model general config)
            model_config_path = self.models_dir / f"{model_name}.yaml"
            if model_config_path.exists():
                return self._load_config_file(model_config_path, dataset_name)
            
            # Fallback: try old format in root config_dir
            old_model_config_path = self.config_dir / f"{model_name}_{dataset_name}.yaml"
            if old_model_config_path.exists():
                return self._load_config_file(old_model_config_path, dataset_name)
        
        # Then try dataset-specific config in datasets/ directory
        dataset_config_path = self.datasets_dir / f"{dataset_name}.yaml"
        if dataset_config_path.exists():
            return self._load_config_file(dataset_config_path, dataset_name)
        
        # Fallback: try old format in root config_dir
        old_dataset_config_path = self.config_dir / f"{dataset_name}.yaml"
        if old_dataset_config_path.exists():
            return self._load_config_file(old_dataset_config_path, dataset_name)
        
        # Fall back to default config
        self.logger.warning(f"No config file found for dataset '{dataset_name}', using defaults")
        return DatasetConfig(
            name=dataset_name,
            description=f"Default configuration for {dataset_name}"
        )
    
    def _load_config_file(self, config_path: Path, dataset_name: str) -> DatasetConfig:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Extract field mappings
            fields_data = config_data.get('fields', {})
            fields = DatasetFieldMapping(
                input_column=fields_data.get('input_column', 'input'),
                input_ids_column=fields_data.get('input_ids_column', 'tok_input'),
                output_column=fields_data.get('output_column', 'output'),
                input_lens_column=fields_data.get('input_lens_column')
            )
            
            # Create dataset config
            dataset_config = DatasetConfig(
                name=config_data.get('name', dataset_name),
                description=config_data.get('description'),
                fields=fields,
                total_sample_count=config_data.get('total_sample_count'),
                file_format=config_data.get('file_format', 'auto'),
                model_specific=config_data.get('model_specific', {})
            )
            
            self.logger.info(f"Loaded dataset config from: {config_path}")
            return dataset_config
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def create_default_config(self, dataset_name: str, 
                             input_column: str = "input",
                             input_ids_column: str = "tok_input",
                             output_column: str = "output") -> Path:
        """
        Create a default configuration file for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            input_column: Input text column name
            input_ids_column: Input token IDs column name
            output_column: Output/target column name
        
        Returns:
            Path to created config file
        """
        config_path = self.config_dir / f"{dataset_name}.yaml"
        
        config_data = {
            'name': dataset_name,
            'description': f"Configuration for {dataset_name} dataset",
            'fields': {
                'input_column': input_column,
                'input_ids_column': input_ids_column,
                'output_column': output_column
            },
            'file_format': 'auto',
            'model_specific': {}
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        self.logger.info(f"Created default config file: {config_path}")
        return config_path
    
    def get_model_dataset_config(self, model_name: str, dataset_name: str) -> ModelDatasetConfig:
        """
        Get model-specific dataset configuration.
        
        Args:
            model_name: Model name
            dataset_name: Dataset name
        
        Returns:
            ModelDatasetConfig object
        """
        dataset_config = self.load_dataset_config(dataset_name, model_name)
        
        return ModelDatasetConfig(
            model_name=model_name,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            scenario_specific={}
        )

