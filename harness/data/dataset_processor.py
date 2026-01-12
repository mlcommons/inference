# ============================================================================
# dataset_processor.py
# -------------------
# Generic dataset processor for handling different data formats
# (JSON, pickle, pandas DataFrame) and converting to standardized format
# ============================================================================

import os
import json
import pickle
import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("pandas not available. Some dataset operations may fail.")

# Import dataset configuration
try:
    from .dataset_config import DatasetConfigLoader, DatasetConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    logging.warning("Dataset config not available. Using default field mappings.")


class DatasetProcessor:
    """
    Generic dataset processor that handles multiple data formats
    and converts them to a standardized format for MLPerf LoadGen.
    
    Supports:
    - JSON files (single objects or arrays)
    - Pickle files (DataFrame or dict)
    - Pandas DataFrame objects
    - CSV files (via pandas)
    """
    
    def __init__(self, 
                 dataset_path: str,
                 model_name: Optional[str] = None,
                 input_column: Optional[str] = None,
                 input_ids_column: Optional[str] = None,
                 output_column: Optional[str] = None,
                 total_sample_count: Optional[int] = None,
                 dataset_name: Optional[str] = None,
                 config_dir: Optional[str] = None):
        """
        Initialize dataset processor.
        
        Args:
            dataset_path: Path to dataset file (JSON, pickle, or CSV)
            model_name: Optional model name for logging and config lookup
            input_column: Column name for input text (overrides config if provided)
            input_ids_column: Column name for tokenized input IDs (overrides config if provided)
            output_column: Column name for output/targets (overrides config if provided)
            total_sample_count: Maximum number of samples to load (None = all, overrides config)
            dataset_name: Name of dataset for config lookup (auto-detected from path if None)
            config_dir: Directory containing dataset config files (defaults to harness/data/configs/)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name or "unknown"
        
        # Try to load dataset configuration
        dataset_config = None
        if CONFIG_AVAILABLE:
            try:
                # Auto-detect dataset name from path if not provided
                if dataset_name is None:
                    dataset_name = self.dataset_path.stem
                
                config_loader = DatasetConfigLoader(config_dir=config_dir)
                dataset_config = config_loader.load_dataset_config(dataset_name, model_name)
                self.logger.info(f"Loaded dataset config for '{dataset_name}'")
            except Exception as e:
                self.logger.warning(f"Could not load dataset config: {e}. Using defaults.")
        
        # Use config values if available, otherwise use provided or default values
        if dataset_config:
            self.input_column = input_column or dataset_config.fields.input_column
            self.input_ids_column = input_ids_column or dataset_config.fields.input_ids_column
            self.output_column = output_column or dataset_config.fields.output_column
            self.input_lens_column = dataset_config.fields.input_lens_column
            self.total_sample_count = total_sample_count or dataset_config.total_sample_count
            self.file_format = dataset_config.file_format
            self.model_specific = dataset_config.model_specific
        else:
            # Fallback to defaults or provided values
            self.input_column = input_column or "input"
            self.input_ids_column = input_ids_column or "tok_input"
            self.output_column = output_column or "output"
            self.input_lens_column = None
            self.total_sample_count = total_sample_count
            self.file_format = "auto"
            self.model_specific = {}
        
        # Dataset data (standardized format)
        self.input: List[str] = []
        self.input_ids: List[List[int]] = []
        self.input_lens: List[int] = []
        self.targets: List[Any] = []
        self.messages: List[Any] = []  # For multimodal models (qwen3vl)
        
        # Load and process dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from file based on extension."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        
        ext = self.dataset_path.suffix.lower()
        
        self.logger.info(f"Loading dataset from: {self.dataset_path}")
        self.logger.info(f"File extension: {ext}")
        self.logger.info(f"Using columns: input='{self.input_column}', input_ids='{self.input_ids_column}', output='{self.output_column}'")
        
        # Determine file format
        if hasattr(self, 'file_format') and self.file_format != "auto":
            format_map = {
                'json': self._load_json,
                'pickle': self._load_pickle,
                'pkl': self._load_pickle,
                'csv': self._load_csv
            }
            loader = format_map.get(self.file_format.lower())
            if loader:
                loader()
            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")
        else:
            # Auto-detect format
            if ext == '.json':
                self._load_json()
            elif ext == '.pkl' or ext == '.pickle':
                self._load_pickle()
            elif ext == '.csv':
                self._load_csv()
            else:
                # Try to auto-detect format
                self.logger.warning(f"Unknown extension {ext}, attempting auto-detection")
                try:
                    self._load_json()
                except:
                    try:
                        self._load_pickle()
                    except:
                        raise ValueError(f"Could not auto-detect format for {self.dataset_path}")
        
        # Process and standardize data
        self._process_data()
        
        self.logger.info(f"Dataset loaded: {len(self.input_ids)} samples")
        if self.input_lens:
            self.logger.info(f"Input length range: {min(self.input_lens)} - {max(self.input_lens)} tokens")
    
    def _load_json(self):
        """Load data from JSON file."""
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # Array of objects
            if not PANDAS_AVAILABLE:
                raise ImportError("pandas required for JSON array processing")
            self.processed_data = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Single object or structured dict
            if PANDAS_AVAILABLE:
                # Try to convert to DataFrame
                if any(isinstance(v, list) for v in data.values()):
                    self.processed_data = pd.DataFrame(data)
                else:
                    # Single row - wrap in list
                    self.processed_data = pd.DataFrame([data])
            else:
                # Fallback: convert to list of dicts
                if any(isinstance(v, list) for v in data.values()):
                    # Dict with lists as values
                    self.raw_data = data
                else:
                    # Single dict - wrap
                    self.raw_data = [data]
        else:
            raise ValueError(f"Unexpected JSON structure: {type(data)}")
    
    def _load_pickle(self):
        """Load data from pickle file."""
        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            self.processed_data = data
        elif isinstance(data, dict):
            if PANDAS_AVAILABLE:
                # Try to convert to DataFrame
                if any(isinstance(v, list) for v in data.values()):
                    self.processed_data = pd.DataFrame(data)
                else:
                    self.processed_data = pd.DataFrame([data])
            else:
                self.raw_data = data
        elif isinstance(data, list):
            if PANDAS_AVAILABLE:
                self.processed_data = pd.DataFrame(data)
            else:
                self.raw_data = data
        else:
            raise ValueError(f"Unexpected pickle data type: {type(data)}")
    
    def _load_csv(self):
        """Load data from CSV file."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for CSV files")
        
        self.processed_data = pd.read_csv(self.dataset_path)
    
    def _process_data(self):
        """Process loaded data into standardized format."""
        # If we have a pandas DataFrame, extract columns
        if hasattr(self, 'processed_data') and PANDAS_AVAILABLE:
            df = self.processed_data
            
            # Print dataset column names and types
            self.logger.info("=" * 80)
            self.logger.info("DATASET COLUMNS AND TYPES")
            self.logger.info("=" * 80)
            for col_name, col_type in df.dtypes.items():
                self.logger.info(f"  {col_name}: {col_type}")
            self.logger.info(f"Total columns: {len(df.columns)}")
            self.logger.info(f"Total rows: {len(df)}")
            self.logger.info("=" * 80)
            
            # Extract messages column if available (for multimodal models like qwen3vl)
            # Check this first as it takes precedence over input/input_ids for multimodal
            if 'messages' in df.columns:
                self.messages = df['messages'].tolist()
                self.logger.info(f"Found 'messages' column with {len(self.messages)} multimodal samples")
                # For multimodal, we may not need input_ids, but initialize empty lists
                self.input = []
                self.input_ids = []
            else:
                self.messages = []
                # Extract input column
                if self.input_column in df.columns:
                    self.input = df[self.input_column].tolist()
                else:
                    self.logger.warning(f"Column '{self.input_column}' not found, using empty list")
                    self.input = []
                
                # Extract input_ids column
                if self.input_ids_column in df.columns:
                    self.input_ids = df[self.input_ids_column].tolist()
                else:
                    self.logger.warning(f"Column '{self.input_ids_column}' not found, trying to tokenize")
                    self.input_ids = self._tokenize_inputs()
            
            # Extract targets/output column
            if self.output_column in df.columns:
                self.targets = df[self.output_column].tolist()
            else:
                self.logger.warning(f"Column '{self.output_column}' not found")
                self.targets = []
            
            # Calculate input lengths (use column if available, otherwise calculate)
            if hasattr(self, 'input_lens_column') and self.input_lens_column and self.input_lens_column in df.columns:
                self.input_lens = df[self.input_lens_column].tolist()
            else:
                self.input_lens = [len(ids) if isinstance(ids, list) else 0 for ids in self.input_ids]
            
        elif hasattr(self, 'raw_data'):
            # Handle raw dict/list data
            # Print dataset column names and types for raw data
            self.logger.info("=" * 80)
            self.logger.info("DATASET COLUMNS AND TYPES")
            self.logger.info("=" * 80)
            if isinstance(self.raw_data, dict):
                # Dict with lists as values
                for key, value in self.raw_data.items():
                    if isinstance(value, list) and len(value) > 0:
                        sample_type = type(value[0]).__name__
                        self.logger.info(f"  {key}: list[{sample_type}] (length: {len(value)})")
                    else:
                        self.logger.info(f"  {key}: {type(value).__name__}")
                self.logger.info(f"Total columns: {len(self.raw_data)}")
            elif isinstance(self.raw_data, list):
                # List of dicts
                if len(self.raw_data) > 0:
                    first_item = self.raw_data[0]
                    if isinstance(first_item, dict):
                        for key, value in first_item.items():
                            self.logger.info(f"  {key}: {type(value).__name__}")
                        self.logger.info(f"Total columns: {len(first_item)}")
                        self.logger.info(f"Total rows: {len(self.raw_data)}")
                    else:
                        self.logger.info(f"  (list item type: {type(first_item).__name__})")
                        self.logger.info(f"Total items: {len(self.raw_data)}")
            self.logger.info("=" * 80)
            
            if isinstance(self.raw_data, dict):
                # Dict with lists as values
                self.input = self.raw_data.get(self.input_column, [])
                self.input_ids = self.raw_data.get(self.input_ids_column, [])
                self.targets = self.raw_data.get(self.output_column, [])
            elif isinstance(self.raw_data, list):
                # List of dicts
                self.input = [item.get(self.input_column, "") for item in self.raw_data]
                self.input_ids = [item.get(self.input_ids_column, []) for item in self.raw_data]
                self.targets = [item.get(self.output_column, None) for item in self.raw_data]
            
            self.input_lens = [len(ids) if isinstance(ids, list) else 0 for ids in self.input_ids]
        
        # Limit sample count if specified
        if self.total_sample_count is not None:
            count = min(self.total_sample_count, len(self.input_ids))
            self.input = self.input[:count]
            self.input_ids = self.input_ids[:count]
            self.input_lens = self.input_lens[:count]
            if self.targets:
                self.targets = self.targets[:count]
            self.logger.info(f"Limited to {count} samples")
    
    def _tokenize_inputs(self) -> List[List[int]]:
        """Tokenize input text if input_ids not available."""
        # This is a placeholder - actual implementation would use tokenizer
        self.logger.warning("Tokenization not implemented. Returning empty lists.")
        return [[] for _ in self.input]
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get a single sample by index."""
        if index >= len(self.input_ids):
            raise IndexError(f"Sample index {index} out of range (max: {len(self.input_ids) - 1})")
        
        return {
            'index': index,
            'input': self.input[index] if index < len(self.input) else "",
            'input_ids': self.input_ids[index],
            'input_len': self.input_lens[index],
            'target': self.targets[index] if index < len(self.targets) else None
        }
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.input_ids)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.input_lens:
            return {}
        
        return {
            'total_samples': len(self.input_ids),
            'max_input_len': max(self.input_lens),
            'min_input_len': min(self.input_lens),
            'avg_input_len': sum(self.input_lens) / len(self.input_lens),
            'has_targets': len(self.targets) > 0,
            'has_input_text': len(self.input) > 0
        }

