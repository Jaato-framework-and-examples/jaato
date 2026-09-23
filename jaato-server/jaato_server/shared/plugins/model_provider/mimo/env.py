"""Environment variable resolution for the Xiaomi MiMo provider.

Configuration is resolved from environment variables:
- JAATO_MIMO_API_KEY for the API key (then the vendor's own MIMO_API_KEY)
- JAATO_MIMO_BASE_URL for the endpoint (default: https://api.xiaomimimo.com/v1)
- JAATO_MIMO_MODEL for the default model name
- JAATO_MIMO_CONTEXT_LENGTH to override the context window

Resolution priority:
1. Explicit config passed in code (ProviderConfig)
2. JAATO_MIMO_* environment variables
3. MIMO_API_KEY (the vendor's own documented variable)
4. Stored credentials (mimo-auth)

Not available in the EU, the UK or Korea (the API answers 403).
Token Plan keys (tp-...) only work against the plan hosts (https://token-plan-{cn,sgp,ams}.xiaomimimo.com/v1) — set JAATO_MIMO_BASE_URL accordingly.
"""

import os
from typing import List, Optional

# ============================================================
# Environment Variable Names
# ============================================================

ENV_MIMO_API_KEY = "JAATO_MIMO_API_KEY"
ENV_MIMO_VENDOR_API_KEY = "MIMO_API_KEY"
ENV_MIMO_BASE_URL = "JAATO_MIMO_BASE_URL"
ENV_MIMO_MODEL = "JAATO_MIMO_MODEL"
ENV_MIMO_CONTEXT_LENGTH = "JAATO_MIMO_CONTEXT_LENGTH"

# Default OpenAI-compatible endpoint.  One host serves every region; keys and balances are per region.
DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"


def resolve_api_key(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """Resolve the Xiaomi MiMo API key from environment or stored credentials.

    Resolution priority:
    1. ``JAATO_MIMO_API_KEY``.
    2. ``MIMO_API_KEY`` — the vendor's own variable, honoured so a user
       who already exported it for the vendor SDK works with no extra
       configuration.
    3. Stored credentials from the mimo-auth flow (resolves under
       ``config_root`` then workspace then ``~/.jaato/`` per
       :func:`shared.config_resolver.resolve_config_search_path`).

    Returns:
        API key if found, None otherwise.
    """
    env_key = os.environ.get(ENV_MIMO_API_KEY) or os.environ.get(ENV_MIMO_VENDOR_API_KEY)  # env: Xiaomi MiMo API key (jaato namespace, then the vendor's own)
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
    """Resolve the Xiaomi MiMo base URL from environment.

    Returns:
        The API base URL (the hosted endpoint unless overridden).
    """
    return os.environ.get(ENV_MIMO_BASE_URL, DEFAULT_BASE_URL)  # env: endpoint (default https://api.xiaomimimo.com/v1)


def resolve_model() -> Optional[str]:
    """Resolve the default model name from environment.

    Returns:
        Model name if found, None otherwise.
    """
    return os.environ.get(ENV_MIMO_MODEL)  # env: default model name (e.g. mimo-v2.5-pro)


def resolve_context_length() -> Optional[int]:
    """Resolve a context-window override from the environment.

    Returns the ``JAATO_MIMO_CONTEXT_LENGTH`` override as an int, or
    ``None`` when unset/invalid.  No hardcoded fallback is substituted
    (project no-fallback rule).
    """
    value = os.environ.get(ENV_MIMO_CONTEXT_LENGTH)  # env: override the context window
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return None


def is_self_hosted(base_url: str) -> bool:
    """Check if the base URL points to a local/self-hosted instance.

    Xiaomi MiMo is a hosted service, so this is normally False; it stays for
    the case of a user fronting the API through a local proxy
    (``JAATO_MIMO_BASE_URL=http://localhost:...``), where an API key may
    not be required.

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


def _masked(key: str) -> str:
    return f"{key[:6]}...{key[-4:]}" if len(key) > 12 else "***"


def get_checked_credential_locations(config=None) -> List[str]:
    """Get the list of locations checked for credentials.

    Used for error messages to help users understand what was checked.

    ``config`` (optional ``ProviderConfig``) surfaces the highest-precedence
    source — the profile ``plugin_configs.mimo.api_key`` knob — which
    the env-only checks below cannot see.

    Returns:
        List of location descriptions.
    """
    from ..base import profile_api_key_location

    locations = [profile_api_key_location(config, "mimo")]

    for var in (ENV_MIMO_API_KEY, ENV_MIMO_VENDOR_API_KEY):
        value = os.environ.get(var)
        locations.append(f"{var}: set ({_masked(value)})" if value else f"{var}: not set")

    try:
        from .auth import get_stored_api_key, get_credential_file_path
        stored_key = get_stored_api_key()
        if stored_key:
            cred_path = get_credential_file_path() or "mimo_auth.json"
            locations.append(f"Stored credentials ({cred_path}): set ({_masked(stored_key)})")
        else:
            locations.append(
                "Stored credentials: not configured (use 'mimo-auth login')"
            )
    except ImportError:
        locations.append("Stored credentials: auth module not available")

    base_url = os.environ.get(ENV_MIMO_BASE_URL)
    if base_url:
        locations.append(f"{ENV_MIMO_BASE_URL}: {base_url}")
    else:
        locations.append(f"Endpoint: {DEFAULT_BASE_URL} (default)")

    model = os.environ.get(ENV_MIMO_MODEL)
    if model:
        locations.append(f"{ENV_MIMO_MODEL}: {model}")

    return locations
