"""
Utility functions for data handling and tokenization.

Provides common functionality for data handling and tokenization.
"""

from .data_utils import load_dataset, save_results, generate_timestamped_filename
from .data_utils import prepare_output_dataframe, add_standardized_columns, validate_dataset

# Import tokenization utilities
from .tokenization import (
    StandardTokenizer,
    process_inference_results
)

# Import backend registry functions
from .backend_registry import (
    get_backend_config,
    get_backend_class_path,
    uses_text_input,
    uses_chat_template,
    get_supported_backends,
    get_backend_instance,
    is_backend_compatible_with_runner,
    get_backend_env_vars,
    apply_backend_env_vars,
    # Backend detection and validation
    detect_backend,
    validate_backend,
    validate_runner_for_backend,
    # Feature check functions
    supports_streaming,
    supports_async,
    requires_torchrun
)

# Import runner utilities
from .runner_utils import (
    create_base_argument_parser,
    print_runner_header,
    setup_output_paths
)

# Import validation utilities
from .validation import (
    BackendError,
    BackendNotInitializedError,
    ValidationError,
    require_initialized,
    validate_prompts_input,
    validate_dataset_extended,
    validate_runner_args
)

# Import error handling utilities
from .error_handling import (
    handle_backend_error,
    handle_runner_error
)

__all__ = [
    # Data utilities
    'load_dataset',
    'save_results',
    'generate_timestamped_filename',
    'prepare_output_dataframe',
    'add_standardized_columns',
    'validate_dataset',
    # Backend utilities
    'detect_backend',
    'validate_backend',
    'validate_runner_for_backend',
    # Tokenization
    'StandardTokenizer',
    'process_inference_results',
    # Backend registry
    'get_backend_config',
    'get_backend_class_path',
    'uses_text_input',
    'uses_chat_template',
    'get_supported_backends',
    'get_backend_instance',
    'is_backend_compatible_with_runner',
    'get_backend_env_vars',
    'apply_backend_env_vars',
    'supports_streaming',
    'supports_async',
    'requires_torchrun',
    # Runner utilities
    'create_base_argument_parser',
    'print_runner_header',
    'setup_output_paths',
    # Validation
    'BackendError',
    'BackendNotInitializedError',
    'ValidationError',
    'require_initialized',
    'validate_prompts_input',
    'validate_dataset_extended',
    'validate_runner_args',
    # Error handling
    'handle_backend_error',
    'handle_runner_error'
]
