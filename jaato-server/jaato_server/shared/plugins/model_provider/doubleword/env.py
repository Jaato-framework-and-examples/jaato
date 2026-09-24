"""Environment variable resolution for the Doubleword provider.

Configuration is resolved from environment variables:
- JAATO_DOUBLEWORD_API_KEY for the API key
- JAATO_DOUBLEWORD_BASE_URL for the endpoint (default: the hosted
  OpenAI-compatible inference API)
- JAATO_DOUBLEWORD_MODEL for the default model name
- JAATO_DOUBLEWORD_CONTEXT_LENGTH to override the catalog-detected context
  window
- JAATO_DOUBLEWORD_SERVICE_TIER to select the inference tier
  ("flex" = the discounted async tier, "priority" = realtime) without
  editing the profile

Resolution priority:
1. Explicit config passed in code (ProviderConfig)
2. JAATO_DOUBLEWORD_* environment variables
3. Stored credentials (doubleword-auth)
"""

import os
from typing import List, Optional

# ============================================================
# Environment Variable Names
# ============================================================

ENV_DOUBLEWORD_API_KEY = "JAATO_DOUBLEWORD_API_KEY"
ENV_DOUBLEWORD_BASE_URL = "JAATO_DOUBLEWORD_BASE_URL"
ENV_DOUBLEWORD_MODEL = "JAATO_DOUBLEWORD_MODEL"
ENV_DOUBLEWORD_CONTEXT_LENGTH = "JAATO_DOUBLEWORD_CONTEXT_LENGTH"
ENV_DOUBLEWORD_SERVICE_TIER = "JAATO_DOUBLEWORD_SERVICE_TIER"

# Default OpenAI-compatible endpoint for the hosted Doubleword Inference API
# (https://docs.doubleword.ai/inference-api).  Every catalog model is
# addressable by name through this single gateway.
DEFAULT_BASE_URL = "https://api.doubleword.ai/v1"


def resolve_api_key(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """Resolve the Doubleword API key from environment or stored credentials.

    Resolution priority:
    1. ``JAATO_DOUBLEWORD_API_KEY``.
    2. Stored credentials from the doubleword-auth flow (resolves under
       ``config_root`` then workspace then ``~/.jaato/`` per
       :func:`shared.config_resolver.resolve_config_search_path`).

    Args:
        workspace_path: Optional explicit workspace path passed through to
            the credential lookup.
        config_root: Optional read-only-config root override.

    Returns:
        API key if found, None otherwise.
    """
    env_key = os.environ.get(ENV_DOUBLEWORD_API_KEY)
    if env_key:
        return env_key
    try:
        from .auth import get_stored_api_key
        return get_stored_api_key(
            workspace_path=workspace_path, config_root=config_root,
        )
    except ImportError:
        return None


def resolve_base_url() -> str:
    """Resolve the Doubleword base URL from environment.

    Returns:
        The API base URL (the hosted gateway unless overridden).
    """
    return os.environ.get(ENV_DOUBLEWORD_BASE_URL, DEFAULT_BASE_URL)  # env: endpoint (default https://api.doubleword.ai/v1)


def resolve_model() -> Optional[str]:
    """Resolve the default model name from environment.

    Returns:
        Model name if found, None otherwise.
    """
    return os.environ.get(ENV_DOUBLEWORD_MODEL)  # env: default model name (e.g. deepseek-ai/DeepSeek-V4-Pro)


def resolve_context_length() -> Optional[int]:
    """Resolve a context-window override from the environment.

    Returns the ``JAATO_DOUBLEWORD_CONTEXT_LENGTH`` override as an int, or
    ``None`` when unset/invalid.  This is only the manual override tier:
    the provider's PRIMARY tier is catalog auto-detect at connect time
    (when Doubleword's ``GET /v1/models`` reports a per-model context
    length).  No hardcoded fallback is substituted (project no-fallback
    rule).
    """
    value = os.environ.get(ENV_DOUBLEWORD_CONTEXT_LENGTH)  # env: override the catalog-detected context window
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return None


def resolve_service_tier() -> Optional[str]:
    """Resolve the inference tier from the environment.

    Doubleword prices by delivery window on the *same* chat endpoint: the
    request-body field ``service_tier`` selects between ``"flex"`` (the
    discounted async tier — work is queued and guaranteed to start within
    ~1 minute) and ``"priority"`` (realtime).  This env var is the
    profile-free way to flip a session onto the cheap tier; the profile
    knob ``plugin_configs.doubleword.api_params.service_tier`` takes
    precedence when both are set.

    Returns:
        The tier string (stripped), or None when unset/blank.  The value
        is forwarded verbatim — Doubleword validates it server-side, so a
        future tier name works without a provider release.
    """
    value = os.environ.get(ENV_DOUBLEWORD_SERVICE_TIER, "")  # env: inference tier ("flex" = discounted async, "priority" = realtime)
    value = value.strip()
    return value or None


def is_self_hosted(base_url: str) -> bool:
    """Check if the base URL points to a local/self-hosted instance.

    Doubleword is a hosted service, so this is normally False; it stays
    for the rare case of a user fronting the API through a local proxy
    (``JAATO_DOUBLEWORD_BASE_URL=http://localhost:...``), where an API
    key may not be required.

    Args:
        base_url: The configured endpoint URL.

    Returns:
        True if the URL appears to be a local instance.
    """
    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    return (
        host in ("localhost", "127.0.0.1", "0.0.0.0")
        or host.startswith("192.168.")
        or host.startswith("10.")
    )


def get_checked_credential_locations(config=None) -> List[str]:
    """Get the list of locations checked for credentials.

    Used for error messages to help users understand what was checked.

    ``config`` (optional ``ProviderConfig``) surfaces the highest-precedence
    source — the profile ``plugin_configs.doubleword.api_key`` knob — which
    the env-only checks below cannot see.

    Returns:
        List of location descriptions.
    """
    from ..base import profile_api_key_location

    locations = [profile_api_key_location(config, "doubleword")]

    jaato_key = os.environ.get(ENV_DOUBLEWORD_API_KEY)  # env: Doubleword API key
    if jaato_key:
        masked = (
            f"{jaato_key[:6]}...{jaato_key[-4:]}" if len(jaato_key) > 12 else "***"
        )
        locations.append(f"{ENV_DOUBLEWORD_API_KEY}: set ({masked})")
    else:
        locations.append(f"{ENV_DOUBLEWORD_API_KEY}: not set")

    try:
        from .auth import get_stored_api_key, get_credential_file_path
        stored_key = get_stored_api_key()
        if stored_key:
            cred_path = get_credential_file_path() or "doubleword_auth.json"
            masked = (
                f"{stored_key[:6]}...{stored_key[-4:]}"
                if len(stored_key) > 12 else "***"
            )
            locations.append(f"Stored credentials ({cred_path}): set ({masked})")
        else:
            locations.append(
                "Stored credentials: not configured (use 'doubleword-auth login')"
            )
    except ImportError:
        locations.append("Stored credentials: auth module not available")

    base_url = os.environ.get(ENV_DOUBLEWORD_BASE_URL)
    if base_url:
        locations.append(f"{ENV_DOUBLEWORD_BASE_URL}: {base_url}")
    else:
        locations.append(f"Endpoint: {DEFAULT_BASE_URL} (default)")

    model = os.environ.get(ENV_DOUBLEWORD_MODEL)
    if model:
        locations.append(f"{ENV_DOUBLEWORD_MODEL}: {model}")

    return locations
