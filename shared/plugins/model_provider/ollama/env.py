"""Environment variable resolution for Ollama provider.

Ollama runs as a local server, so configuration focuses on host/port
rather than API keys.
"""

import os
from typing import Optional


# Default Ollama server address
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def resolve_host() -> str:
    """Resolve Ollama server host URL.

    Checks:
    1. OLLAMA_HOST environment variable

    Returns:
        Ollama host URL (default: http://localhost:11434).
    """
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)


def resolve_model() -> Optional[str]:
    """Resolve default model name from environment.

    Checks:
    1. OLLAMA_MODEL environment variable

    Returns:
        Model name if set, None otherwise.
    """
    return os.environ.get("OLLAMA_MODEL")


def resolve_context_length() -> Optional[int]:
    """Resolve custom context length override.

    Checks:
    1. OLLAMA_CONTEXT_LENGTH environment variable

    Returns:
        Context length in tokens if set, None otherwise.
    """
    val = os.environ.get("OLLAMA_CONTEXT_LENGTH")
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return None
