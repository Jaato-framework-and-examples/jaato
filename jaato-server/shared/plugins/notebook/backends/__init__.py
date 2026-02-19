"""Notebook execution backends.

Available backends:
- LocalJupyterBackend: Local Jupyter kernel (no GPU, instant)
- KaggleBackend: Kaggle API (free GPU, async execution) - requires kaggle package
"""

from .base import NotebookBackend
from .local import LocalJupyterBackend

# KaggleBackend is optional - only import if kaggle package is available
try:
    from .kaggle import KaggleBackend
    _KAGGLE_AVAILABLE = True
except ImportError:
    KaggleBackend = None  # type: ignore
    _KAGGLE_AVAILABLE = False

__all__ = [
    'NotebookBackend',
    'LocalJupyterBackend',
    'KaggleBackend',
    '_KAGGLE_AVAILABLE',
]
