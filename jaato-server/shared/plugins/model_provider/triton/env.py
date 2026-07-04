"""Environment variable resolution for the Triton provider.

Triton's OpenAI-compatible frontend and Triton's native HTTP API run as
separate processes on different ports (canonical: 9000 and 8000).
This module resolves both URLs independently so users can target
deployments that put them on different hosts (e.g. OpenAI frontend
behind a load balancer, Triton internal on a private network).

Environment variables (URL+host knobs REQUIRED at session init time
— no hardcoded localhost fallbacks per the project's "no fallback"
rule; missing values fail fast in ``TritonProvider.initialize`` with
a clear ``ValueError``):

    TRITON_OPENAI_URL: OpenAI frontend URL (e.g.
        ``http://gpu-box:9000``).
    TRITON_CONTROL_URL: Triton native HTTP URL (e.g.
        ``http://gpu-box:8000``).
    TRITON_HOST: Shorthand — sets both URLs to this hostname with the
        canonical Triton ports.  Use ONE of {URL pair, HOST shorthand}.
    TRITON_MODEL: Default model name.
    TRITON_CONTEXT_LENGTH: Context window size.  Triton's model config
        does not carry a standard "context length" field — backend-
        specific (TRT-LLM has ``max_seq_len``, vLLM has
        ``max_model_len``).  REQUIRED at session init time.
    TRITON_API_TOKEN: Optional bearer token.

The two port constants (``DEFAULT_OPENAI_PORT = 9000``,
``DEFAULT_CONTROL_PORT = 8000``) are NOT fallback values — they are
the canonical Triton-server port assignments, used solely to fill in
the port number when the user provides a bare hostname via
``TRITON_HOST``.  When neither URL nor HOST is set, the provider
fails fast rather than assuming localhost.
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

# Canonical Triton port assignments (NOT fallback values — only used
# to compose a URL from the TRITON_HOST shorthand).  These match the
# Triton example launchers; override the full URL via TRITON_OPENAI_URL
# / TRITON_CONTROL_URL when running on non-default ports.
DEFAULT_OPENAI_PORT = 9000
DEFAULT_CONTROL_PORT = 8000


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


def resolve_urls() -> Tuple[Optional[str], Optional[str]]:
    """Return ``(openai_url, control_url)`` — either may be ``None``.

    Resolution:
        1. ``TRITON_OPENAI_URL`` / ``TRITON_CONTROL_URL`` if set →
           use them as-is.
        2. Otherwise, if ``TRITON_HOST`` is set → derive the missing
           URL by composing the host shorthand with the canonical
           Triton port (9000 for OpenAI, 8000 for control).
        3. Otherwise → ``None`` for that URL.  Caller is responsible
           for failing fast — there is no localhost fallback.

    A single URL env var (only ``TRITON_OPENAI_URL`` or only
    ``TRITON_CONTROL_URL``) is honored — the other gets its
    ``TRITON_HOST``-derived value if HOST is set, else ``None``.
    """
    openai = os.environ.get(ENV_OPENAI_URL)  # env: Triton OpenAI-frontend URL (e.g. http://gpu-box:9000); required unless TRITON_HOST is set
    control = os.environ.get(ENV_CONTROL_URL)  # env: Triton native HTTP URL (e.g. http://gpu-box:8000); required unless TRITON_HOST is set
    host = os.environ.get(ENV_HOST)  # env: shorthand: derives both URLs from this host with canonical ports 9000/8000

    if openai is None and host:
        openai = _origin_for_host(host, DEFAULT_OPENAI_PORT)
    if control is None and host:
        control = _origin_for_host(host, DEFAULT_CONTROL_PORT)

    return (
        openai.rstrip("/") if openai else None,
        control.rstrip("/") if control else None,
    )


def resolve_model() -> Optional[str]:
    """Return the default model name, if configured."""
    return os.environ.get(ENV_MODEL)  # env: default model name


def resolve_context_length() -> Optional[int]:
    """Return the context length override, if configured."""
    val = os.environ.get(ENV_CONTEXT_LENGTH)  # env: context window (required — Triton's model config has no standard context field)
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
    return os.environ.get(ENV_API_TOKEN)  # env: bearer token, only when fronted by an auth proxy
