"""Lazy loading for OpenAI SDK.

This module defers importing the openai SDK until it's actually needed,
improving startup time when this provider isn't used.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI

# Cached module reference
_openai_module = None


def get_openai_module() -> Any:
    """Get the openai module, importing it lazily.

    Returns:
        The openai module.

    Raises:
        ImportError: If the openai package is not installed.
    """
    global _openai_module
    if _openai_module is None:
        try:
            import openai
            _openai_module = openai
        except ImportError as e:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            ) from e
    return _openai_module


def get_openai_client_class() -> Any:
    """Get the OpenAI client class, importing it lazily.

    Returns:
        The OpenAI class.

    Raises:
        ImportError: If the openai package is not installed.
    """
    return get_openai_module().OpenAI
