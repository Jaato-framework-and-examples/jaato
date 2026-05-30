"""Environment variable resolution for the Triton provider.

Triton's OpenAI-compatible frontend and Triton's native HTTP API run as
separate processes on different ports (defaults: 9000 and 8000).  This
module resolves both URLs independently so users can target deployments
that put them on different hosts (e.g. OpenAI frontend behind a load
balancer, Triton internal on a private network).

Environment variables:
    TRITON_OPENAI_URL: OpenAI frontend URL (default: http://localhost:9000)
    TRITON_CONTROL_URL: Triton native HTTP URL (default: http://localhost:8000)
    TRITON_HOST: Shorthand — sets both URLs to this hostname with default ports
    TRITON_MODEL: Default model name
    TRITON_CONTEXT_LENGTH: Override context window size
    TRITON_API_TOKEN: Optional bearer token
"""

import os
from typing import Optional, Tuple
from urllib.parse import urlparse


ENV_OPENAI_URL = "TRITON_OPENAI_URL"
ENV_CONTROL_URL = "TRITON_CONTROL_URL"
ENV_HOST = "TRITON_HOST"
ENV_MODEL = "TRITON_MODEL"
ENV_CONTEXT_LENGTH = "TRITON_CONTEXT_LENGTH"
ENV_API_TOKEN = "TRITON_API_TOKEN"

DEFAULT_OPENAI_PORT = 9000
DEFAULT_CONTROL_PORT = 8000
DEFAULT_OPENAI_URL = f"http://localhost:{DEFAULT_OPENAI_PORT}"
DEFAULT_CONTROL_URL = f"http://localhost:{DEFAULT_CONTROL_PORT}"

# Triton's model config (max_batch_size, parameters) doesn't carry a
# standard "context length" field — that's a backend-specific concept
# (TRT-LLM has ``max_seq_len``, vLLM has ``max_model_len``, etc.).  We
# don't try to auto-discover it; users set this explicitly.
DEFAULT_CONTEXT_LENGTH = 8192


def _origin_for_host(host: str, default_port: int) -> str:
    """Compose a URL from a ``host`` shorthand and a default port.

    Accepts a bare hostname (``gpu-box``), a ``scheme://host`` (``http://gpu-box``),
    or a full URL (``http://gpu-box:9000``).  When no port is present,
    appends ``default_port``.  Bare hostnames default to the ``http`` scheme.
    """
    host = host.strip().rstrip("/")
    if "://" not in host:
        host = f"http://{host}"
    parsed = urlparse(host)
    if parsed.port is not None:
        return host
    # Re-emit with the default port appended to netloc.
    netloc = f"{parsed.hostname}:{default_port}"
    return f"{parsed.scheme}://{netloc}"


def resolve_urls() -> Tuple[str, str]:
    """Return ``(openai_url, control_url)``.

    Priority:
        1. Both ``TRITON_OPENAI_URL`` and ``TRITON_CONTROL_URL`` set →
           use them as-is (each can be overridden independently).
        2. ``TRITON_HOST`` set, URL env vars absent → derive both URLs
           from the host with default ports.
        3. Neither URL env var nor ``TRITON_HOST`` set → defaults
           (``http://localhost:9000``, ``http://localhost:8000``).

    A single URL env var (only ``TRITON_OPENAI_URL`` or only
    ``TRITON_CONTROL_URL``) is also honored — the other gets its
    default or its ``TRITON_HOST``-derived value.
    """
    openai = os.environ.get(ENV_OPENAI_URL)
    control = os.environ.get(ENV_CONTROL_URL)
    host = os.environ.get(ENV_HOST)

    if openai is None:
        openai = (
            _origin_for_host(host, DEFAULT_OPENAI_PORT)
            if host
            else DEFAULT_OPENAI_URL
        )
    if control is None:
        control = (
            _origin_for_host(host, DEFAULT_CONTROL_PORT)
            if host
            else DEFAULT_CONTROL_URL
        )
    return openai.rstrip("/"), control.rstrip("/")


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

    Returns None when Triton is running without an upstream auth proxy.
    """
    return os.environ.get(ENV_API_TOKEN)
