"""Notebook execution backends.

Available backends:
- LocalJupyterBackend: Local Jupyter kernel (no GPU, instant)
- KaggleBackend: Kaggle API (free GPU, async execution)
"""

from .base import NotebookBackend
from .local import LocalJupyterBackend
from .kaggle import KaggleBackend

__all__ = [
    'NotebookBackend',
    'LocalJupyterBackend',
    'KaggleBackend',
]
