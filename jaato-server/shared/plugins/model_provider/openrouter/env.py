"""Environment variable resolution for the OpenRouter provider.

Resolution priority:
1. Explicit ``ProviderConfig`` values supplied in code.
2. ``JAATO_OPENROUTER_*`` environment variables.
3. Stored credentials saved by ``openrouter-auth``.
4. Sensible defaults.
"""

import os
from typing import List, Optional

# ============================================================
# Environment Variable Names
# ============================================================

ENV_OPENROUTER_API_KEY = "JAATO_OPENROUTER_API_KEY"
ENV_OPENROUTER_BASE_URL = "JAATO_OPENROUTER_BASE_URL"
ENV_OPENROUTER_MODEL = "JAATO_OPENROUTER_MODEL"
ENV_OPENROUTER_CONTEXT_LENGTH = "JAATO_OPENROUTER_CONTEXT_LENGTH"
ENV_OPENROUTER_HTTP_REFERER = "JAATO_OPENROUTER_HTTP_REFERER"
ENV_OPENROUTER_APP_TITLE = "JAATO_OPENROUTER_APP_TITLE"
ENV_OPENROUTER_APP_CATEGORIES = "JAATO_OPENROUTER_APP_CATEGORIES"
ENV_OPENROUTER_REQUEST_TIMEOUT = "JAATO_OPENROUTER_REQUEST_TIMEOUT"
ENV_OPENROUTER_STREAM_IDLE_TIMEOUT = "JAATO_OPENROUTER_STREAM_IDLE_TIMEOUT"

# OpenRouter uses these header names for app attribution / rankings.
# See https://openrouter.ai/docs/app-attribution.
HEADER_HTTP_REFERER = "HTTP-Referer"
HEADER_APP_TITLE = "X-OpenRouter-Title"
HEADER_APP_CATEGORIES = "X-OpenRouter-Categories"

# Default OpenRouter endpoint.  The same key works for all upstream models.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Default attribution headers — OpenRouter uses these for app rankings
# when an integrator opts in.  Users can override via env vars.
DEFAULT_HTTP_REFERER = "https://github.com/Jaato-framework-and-examples/jaato"
DEFAULT_APP_TITLE = "jaato"

# Marketplace category for jaato.  ``cli-agent`` is the closest fit in
# OpenRouter's taxonomy ("Terminal-based coding assistants") — jaato is
# a terminal-driven multi-provider agentic tool orchestrator.  Per
# https://openrouter.ai/docs/app-attribution, OpenRouter silently
# drops unrecognized categories, so the worst case for the default is
# that future taxonomy changes turn it into a no-op until we update.
DEFAULT_APP_CATEGORIES = ("cli-agent",)

# OpenRouter's documented limits on the X-OpenRouter-Categories header.
MAX_CATEGORIES_PER_REQUEST = 5
MAX_CATEGORY_LENGTH = 30

# ============================================================
# Request deadlines (#732)
# ============================================================
#
# Nothing inside the provider used to bound a single request: whether the
# upstream stalled or the connection silently died, the provider waited
# forever and delegated the timeout to whoever happened to be above it
# (a harness arm-timeout, a budget ceiling).  These three defaults are
# what a request is now bounded by.  Each is overridable per-profile
# (``plugin_configs.openrouter.framework_overrides``) or by env var,
# because a legitimate long generation and a dead socket look identical
# from here.

# Connect deadline — TCP + TLS to openrouter.ai.  Deliberately short:
# a connect that hasn't completed in 15s is not going to.
DEFAULT_CONNECT_TIMEOUT = 15.0

# BYTE-level deadline, handed to httpx as read / write / pool.  Bounds a
# socket that has gone silent entirely (the SDK's own default is 600s, so
# this only makes the value explicit and configurable).
DEFAULT_REQUEST_TIMEOUT = 600.0

# PAYLOAD-level deadline for streaming turns, enforced by
# :class:`~.stall.StreamStallGuard`.  The read timeout above cannot see
# this failure: OpenRouter keeps a stalled stream fed with
# ``: OPENROUTER PROCESSING`` SSE comments, which reset the byte clock
# while the caller's chunk loop never ticks.  300s is set above any
# realistic time-to-first-token (reasoning models can think for minutes)
# and far below the 20+ minute hangs #732 measured.
DEFAULT_STREAM_IDLE_TIMEOUT = 300.0


def resolve_api_key() -> Optional[str]:
    """Resolve the OpenRouter API key from env or stored credentials."""
    env_key = os.environ.get(ENV_OPENROUTER_API_KEY)  # env: OpenRouter API key (sk-or-... from openrouter.ai/settings/keys)
    if env_key:
        return env_key
    try:
        from .auth import get_stored_api_key
        return get_stored_api_key()
    except ImportError:
        return None


def resolve_base_url() -> str:
    """Resolve the OpenRouter base URL from env."""
    return os.environ.get(ENV_OPENROUTER_BASE_URL, DEFAULT_BASE_URL)  # env: endpoint (default https://openrouter.ai/api/v1)


def resolve_model() -> Optional[str]:
    """Resolve a default model name from env, if set."""
    return os.environ.get(ENV_OPENROUTER_MODEL)  # env: default model (vendor/model form; openrouter/auto lets OpenRouter pick)


def resolve_context_length() -> Optional[int]:
    """Resolve the manual context-window override from env, or ``None``.

    Returns ``None`` when unset — the provider auto-detects the per-model
    context length from the OpenRouter catalog (tier-1); this env var is
    only a fallback (no hardcoded default per the project's no-fallback
    rule).
    """
    value = os.environ.get(ENV_OPENROUTER_CONTEXT_LENGTH)  # env: override the catalog-reported context window
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return None


def _resolve_timeout(env_var: str, default: float) -> float:
    """Read a non-negative float deadline from ``env_var``.

    ``0`` is meaningful — it disables the deadline — so it is accepted and
    returned as-is.  A negative or unparseable value falls back to
    ``default`` rather than failing: an env-var typo should not make the
    provider unusable, and the deadline it lands on is still bounded.
    """
    raw = os.environ.get(env_var)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def resolve_request_timeout() -> float:
    """Resolve the byte-level request deadline (httpx read/write/pool)."""
    return _resolve_timeout(  # env: per-request byte deadline in seconds (0 disables)
        ENV_OPENROUTER_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT,
    )


def resolve_stream_idle_timeout() -> float:
    """Resolve the payload-level idle deadline for streaming turns."""
    return _resolve_timeout(  # env: streaming payload idle deadline in seconds (0 disables)
        ENV_OPENROUTER_STREAM_IDLE_TIMEOUT, DEFAULT_STREAM_IDLE_TIMEOUT,
    )


def resolve_http_referer() -> str:
    """Resolve the HTTP-Referer header for OpenRouter app rankings."""
    return os.environ.get(ENV_OPENROUTER_HTTP_REFERER, DEFAULT_HTTP_REFERER)  # env: app-attribution HTTP-Referer header (required for OpenRouter app rankings)


def resolve_app_title() -> str:
    """Resolve the X-Title header for OpenRouter app rankings."""
    return os.environ.get(ENV_OPENROUTER_APP_TITLE, DEFAULT_APP_TITLE)  # env: app-attribution X-OpenRouter-Title header (display name)


def resolve_app_categories() -> List[str]:
    """Resolve the X-OpenRouter-Categories header value as a list.

    The env var is read as a comma-separated string (the same form
    that becomes the wire header value); whitespace around each entry
    is stripped, empty entries are dropped.  Returns the
    :data:`DEFAULT_APP_CATEGORIES` tuple as a list when unset.

    Format validation is the caller's job — this function trusts that
    the user knows OpenRouter's taxonomy.  Unrecognized categories are
    silently ignored by OpenRouter per
    https://openrouter.ai/docs/app-attribution, so the failure mode is
    "no category attached" rather than a request error.
    """
    raw = os.environ.get(ENV_OPENROUTER_APP_CATEGORIES)  # env: comma-separated X-OpenRouter-Categories header (default cli-agent)
    if raw is None:
        return list(DEFAULT_APP_CATEGORIES)
    parts = [c.strip() for c in raw.split(",")]
    return [c for c in parts if c]


def get_checked_credential_locations(config=None) -> List[str]:
    """Describe which credential locations were checked.

    Returned by errors so users see exactly what the provider looked at
    before giving up on credentials.

    ``config`` (optional ``ProviderConfig``) surfaces the highest-precedence
    source — the profile ``plugin_configs.openrouter.api_key`` knob — which
    the env-only checks below cannot see.
    """
    from ..base import profile_api_key_location

    locations: List[str] = [profile_api_key_location(config, "openrouter")]

    api_key = os.environ.get(ENV_OPENROUTER_API_KEY)
    if api_key:
        masked = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        locations.append(f"{ENV_OPENROUTER_API_KEY}: set ({masked})")
    else:
        locations.append(f"{ENV_OPENROUTER_API_KEY}: not set")

    try:
        from .auth import get_stored_api_key, get_credential_file_path
        stored_key = get_stored_api_key()
        if stored_key:
            cred_path = get_credential_file_path() or "openrouter_auth.json"
            masked = f"{stored_key[:8]}...{stored_key[-4:]}" if len(stored_key) > 12 else "***"
            locations.append(f"Stored credentials ({cred_path}): set ({masked})")
        else:
            locations.append(
                "Stored credentials: not configured (use 'openrouter-auth login')"
            )
    except ImportError:
        locations.append("Stored credentials: auth module not available")

    base_url = os.environ.get(ENV_OPENROUTER_BASE_URL)
    if base_url:
        locations.append(f"{ENV_OPENROUTER_BASE_URL}: {base_url}")
    else:
        locations.append(f"Endpoint: {DEFAULT_BASE_URL} (default)")

    model = os.environ.get(ENV_OPENROUTER_MODEL)
    if model:
        locations.append(f"{ENV_OPENROUTER_MODEL}: {model}")

    return locations
