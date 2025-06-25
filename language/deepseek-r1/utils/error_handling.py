"""Centralized error handling utilities."""
import sys
import traceback
from typing import Optional, Type
from .validation import BackendError, ValidationError


def handle_backend_error(e: Exception, backend_name: str,
                         operation: str) -> None:
    """
    Standardized error handling for backend operations.

    Args:
        e: The exception that occurred
        backend_name: Name of the backend
        operation: Description of the operation that failed
    """
    error_msg = f"\n[{backend_name.upper()}] Error during {operation}: {type(e).__name__}: {str(e)}"

    if isinstance(e, (RuntimeError, ValueError)):
        # Known errors - just print the message
        print(error_msg)
    else:
        # Unexpected errors - print full traceback
        print(error_msg)
        traceback.print_exc()


def handle_runner_error(e: Exception, runner_name: str) -> None:
    """
    Standardized error handling for runners.

    Args:
        e: The exception that occurred
        runner_name: Name of the runner
    """
    if isinstance(e, KeyboardInterrupt):
        print(f"\n{runner_name} interrupted by user")
        sys.exit(1)
    elif isinstance(e, ValidationError):
        print(f"\nValidation error: {e}")
        sys.exit(1)
    elif isinstance(e, BackendError):
        print(f"\nBackend error: {e}")
        sys.exit(1)
    else:
        print(f"\n{runner_name} failed: {e}")
        traceback.print_exc()
        sys.exit(1)
