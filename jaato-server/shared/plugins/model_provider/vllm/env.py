"""Environment variable resolution for the vLLM provider.

The provider talks to a vLLM OpenAI-compatible API server
(``vllm.entrypoints.openai.api_server``) over its OpenAI surface
(``POST /v1/chat/completions``, ``GET /v1/models``, ``GET /health``).

Environment variables (all REQUIRED at session init time — no
hardcoded fallbacks per the project's "no fallback" rule; missing
values fail fast in ``VLLMProvider.initialize`` with a clear
``ValueError`` naming the env var and profile knob to set):

    VLLM_HOST: Server URL (e.g. ``http://localhost:8000``).  Override
        per-session via ``plugin_configs.vllm.host``.
    VLLM_MODEL: Default model name.  Override per-session via the
        profile's top-level ``model:`` knob.
    VLLM_CONTEXT_LENGTH: Manual context-window override (tier-3 / env
        fallback).  Normally unnecessary — current vLLM versions DO
        surface ``max_model_len`` in each ``GET /v1/models`` entry, which
        the provider auto-detects (tier-1; verified live 2026-06-10
        against a running server: ``data[0].max_model_len`` present).
        Set this (or ``plugin_configs.vllm.context_length``) only for
        older vLLM builds that omit the field, or to pin a value.  See
        ``resolve_context_window``.
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


def resolve_host() -> Optional[str]:
    """Return the vLLM server URL, or ``None`` when unset.

    Caller is responsible for failing fast when this returns ``None``
    and no profile-level override is in place — see
    ``VLLMProvider.initialize``.
    """
    return os.environ.get(ENV_HOST)


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
