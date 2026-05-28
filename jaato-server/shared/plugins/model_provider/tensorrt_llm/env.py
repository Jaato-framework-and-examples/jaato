"""Environment variable resolution for the TensorRT-LLM provider.

The provider talks to a ``trtllm-serve`` instance over its
OpenAI-compatible HTTP surface (``POST /v1/chat/completions``,
``GET /v1/models``, ``GET /health``).

Environment variables:
    TENSORRT_LLM_HOST: Server URL (default: http://localhost:8000)
    TENSORRT_LLM_MODEL: Default model name
    TENSORRT_LLM_CONTEXT_LENGTH: Override context window size
    TENSORRT_LLM_API_TOKEN: Optional bearer token (only when the server
        is behind a proxy that enforces auth — trtllm-serve itself does
        not document a built-in API key mechanism)
"""

import os
from typing import Optional


ENV_HOST = "TENSORRT_LLM_HOST"
ENV_MODEL = "TENSORRT_LLM_MODEL"
ENV_CONTEXT_LENGTH = "TENSORRT_LLM_CONTEXT_LENGTH"
ENV_API_TOKEN = "TENSORRT_LLM_API_TOKEN"

DEFAULT_HOST = "http://localhost:8000"

# trtllm-serve's GET /v1/models does not surface per-model context length
# (the engine's max_seq_len is fixed at build time and not echoed in the
# OpenAI catalog response). Fall back to a conservative value when no
# override is provided; users running long-context engines should set
# TENSORRT_LLM_CONTEXT_LENGTH or plugin_configs.tensorrt_llm.context_length.
DEFAULT_CONTEXT_LENGTH = 8192


def resolve_host() -> str:
    """Return the trtllm-serve server URL."""
    return os.environ.get(ENV_HOST, DEFAULT_HOST)


def resolve_model() -> Optional[str]:
    """Return the default model name, if configured."""
    return os.environ.get(ENV_MODEL)


def resolve_context_length() -> Optional[int]:
    """Return the context length override, if configured."""
    val = os.environ.get(ENV_CONTEXT_LENGTH)
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return None


def resolve_api_token() -> Optional[str]:
    """Return the optional bearer token.

    Returns None when trtllm-serve is running without an upstream auth
    proxy (the common local-dev case).
    """
    return os.environ.get(ENV_API_TOKEN)
