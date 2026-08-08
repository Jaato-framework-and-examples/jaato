"""Lazy loading for the OpenAI SDK used by the OpenRouter provider.

OpenRouter speaks the OpenAI chat-completions wire format, so we reuse
the ``openai`` SDK as the transport.  Importing it lazily keeps startup
fast when this provider isn't selected.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI  # noqa: F401

_openai_module = None


def get_openai_module() -> Any:
    """Return the ``openai`` module, importing it on first use.

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
                "Install with: pip install 'jaato-server[openrouter]'"
            ) from e
    return _openai_module


def get_openai_client_class() -> Any:
    """Return the ``OpenAI`` client class lazily."""
    return get_openai_module().OpenAI
