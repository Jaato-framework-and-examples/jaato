"""Lazy loading for Google GenAI SDK.

This module defers importing the heavy google.genai SDK until it's actually
needed, significantly improving startup time when this provider isn't used.
"""

from typing import TYPE_CHECKING, Any

# For type checking only - doesn't trigger import at runtime
if TYPE_CHECKING:
    from google import genai
    from google.genai import types

# Cached module references
_genai = None
_types = None


def get_genai() -> Any:
    """Get the google.genai module, importing it lazily.

    Returns:
        The google.genai module.

    Raises:
        ImportError: If the google-genai package is not installed.
    """
    global _genai
    if _genai is None:
        try:
            from google import genai
            _genai = genai
        except ImportError as e:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            ) from e
    return _genai


def get_types() -> Any:
    """Get the google.genai.types module, importing it lazily.

    Returns:
        The google.genai.types module.

    Raises:
        ImportError: If the google-genai package is not installed.
    """
    global _types
    if _types is None:
        try:
            from google.genai import types
            _types = types
        except ImportError as e:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            ) from e
    return _types
