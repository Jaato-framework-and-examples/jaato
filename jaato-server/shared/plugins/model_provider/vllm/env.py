"""Environment variable resolution for the vLLM provider.

The provider talks to a vLLM OpenAI-compatible API server
(``vllm.entrypoints.openai.api_server``) over its OpenAI surface
(``POST /v1/chat/completions``, ``GET /v1/models``, ``GET /health``).

Environment variables:
    VLLM_HOST: Server URL (default: http://localhost:8000)
    VLLM_MODEL: Default model name
    VLLM_CONTEXT_LENGTH: Override context window size.  Required for
        long-context engines — vLLM's ``GET /v1/models`` response does
        NOT surface ``max_model_len`` (the value passed at server-launch
        via ``--max-model-len``), so the framework cannot auto-discover
        the context window.  Verified against vLLM stable docs
        2026-06-07 via context7: the catalog entry shape is
        ``{id, object, created, owned_by, root, parent, permission}``
        with no length field.
    VLLM_API_TOKEN: Optional bearer token.  Only required when the
        vLLM server was launched with ``--api-key <token>`` (vLLM's
        native bearer auth), or when fronted by a reverse proxy that
        enforces auth.
"""

import os
from typing import Optional


ENV_HOST = "VLLM_HOST"
ENV_MODEL = "VLLM_MODEL"
ENV_CONTEXT_LENGTH = "VLLM_CONTEXT_LENGTH"
ENV_API_TOKEN = "VLLM_API_TOKEN"

DEFAULT_HOST = "http://localhost:8000"

# vLLM's GET /v1/models does not surface max_model_len (the build/launch
# parameter ``--max-model-len``).  Fall back to a conservative value when
# no override is provided; users running long-context engines must set
# VLLM_CONTEXT_LENGTH or plugin_configs.vllm.context_length.  Verified
# against vLLM stable docs 2026-06-07 via context7.
DEFAULT_CONTEXT_LENGTH = 8192


def resolve_host() -> str:
    """Return the vLLM server URL."""
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

    Returns None when vLLM is running without ``--api-key`` and without
    an upstream auth proxy (the common local-dev case — vLLM accepts
    any ``api_key`` value in that mode, ``"EMPTY"`` is the conventional
    placeholder).
    """
    return os.environ.get(ENV_API_TOKEN)
