"""Environment variable resolution for the TensorRT-LLM provider.

The provider talks to a ``trtllm-serve`` instance over its
OpenAI-compatible HTTP surface (``POST /v1/chat/completions``,
``GET /v1/models``, ``GET /health``).

Environment variables (all REQUIRED at session init time — no
hardcoded fallbacks per the project's "no fallback" rule; missing
values fail fast in ``TensorRTLLMProvider.initialize`` with a clear
``ValueError`` naming the env var and profile knob to set):

    TENSORRT_LLM_HOST: Server URL (e.g. ``http://localhost:8000``).
        Override per-session via ``plugin_configs.tensorrt_llm.host``.
    TENSORRT_LLM_MODEL: Default model name.  Override per-session via
        the profile's top-level ``model:`` knob.
    TENSORRT_LLM_CONTEXT_LENGTH: Context window size.  trtllm-serve's
        ``GET /v1/models`` does not surface per-engine context length
        (``max_seq_len`` is fixed at engine build time and not echoed
        in the OpenAI catalog response), so this MUST be set
        explicitly.  Override per-session via
        ``plugin_configs.tensorrt_llm.context_length``.
    TENSORRT_LLM_API_TOKEN: Optional bearer token (only when the server
        is behind a proxy that enforces auth — trtllm-serve itself does
        not document a built-in API key mechanism).
"""

import os
from typing import Optional


ENV_HOST = "TENSORRT_LLM_HOST"
ENV_MODEL = "TENSORRT_LLM_MODEL"
ENV_CONTEXT_LENGTH = "TENSORRT_LLM_CONTEXT_LENGTH"
ENV_API_TOKEN = "TENSORRT_LLM_API_TOKEN"


def resolve_host() -> Optional[str]:
    """Return the trtllm-serve server URL, or ``None`` when unset.

    Caller is responsible for failing fast when this returns ``None``
    and no profile-level override is in place — see
    ``TensorRTLLMProvider.initialize``.
    """
    return os.environ.get(ENV_HOST)  # env: trtllm-serve URL (required — no localhost fallback)


def resolve_model() -> Optional[str]:
    """Return the default model name, if configured."""
    return os.environ.get(ENV_MODEL)  # env: model name matching the engine's id in /v1/models


def resolve_context_length() -> Optional[int]:
    """Return the context length override, if configured."""
    val = os.environ.get(ENV_CONTEXT_LENGTH)  # env: context window (required — trtllm-serve does not surface max_seq_len)
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
    return os.environ.get(ENV_API_TOKEN)  # env: bearer token, only when fronted by an auth proxy
