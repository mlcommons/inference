"""
Modular backend system for MLPerf DeepSeek reference implementation.

Supports TensorRT-LLM, SGLang, vLLM, and PyTorch backends with shared API arguments
but independent execution implementations.
"""

from .base_backend import BaseBackend

# Note: Specific backend implementations are imported dynamically as needed
# to avoid dependency issues when only using certain backends
__all__ = [
    'BaseBackend',
]
