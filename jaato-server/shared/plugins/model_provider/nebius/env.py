"""Environment variable resolution for the Nebius Token Factory provider.

Configuration is resolved from environment variables:
- JAATO_NEBIUS_API_KEY (jaato namespace, highest priority) or the vendor's
  own NEBIUS_API_KEY for the API key
- JAATO_NEBIUS_BASE_URL for the endpoint (default: the serverless API)
- JAATO_NEBIUS_MODEL for the default model name
- JAATO_NEBIUS_CONTEXT_LENGTH to override the catalog-detected context window

Resolution priority:
1. Explicit config passed in code (ProviderConfig)
2. JAATO_NEBIUS_* / NEBIUS_API_KEY environment variables
3. Stored credentials (nebius-auth)
"""

import os
from typing import List, Optional

# ============================================================
# Environment Variable Names
# ============================================================

# Primary (jaato namespace) and the vendor's documented variable.  Both are
# legitimate documented sources, consulted in priority order — not a guessed
# fallback (project no-fallback rule).
ENV_NEBIUS_API_KEY = "JAATO_NEBIUS_API_KEY"
ENV_NEBIUS_API_KEY_VENDOR = "NEBIUS_API_KEY"
ENV_NEBIUS_BASE_URL = "JAATO_NEBIUS_BASE_URL"
ENV_NEBIUS_MODEL = "JAATO_NEBIUS_MODEL"
ENV_NEBIUS_CONTEXT_LENGTH = "JAATO_NEBIUS_CONTEXT_LENGTH"

# Default serverless inference endpoint for Nebius Token Factory.
# https://docs.tokenfactory.nebius.com/quickstart
DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"


def resolve_api_key(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """Resolve the Nebius API key from environment or stored credentials.

    Resolution priority (each a documented source, in order):
    1. ``JAATO_NEBIUS_API_KEY`` (jaato namespace).
    2. ``NEBIUS_API_KEY`` (the vendor's own documented variable, so users
       who already set it for the Nebius/OpenAI SDK work with no extra
       configuration).
    3. Stored credentials from the nebius-auth plugin (resolves under
       ``config_root`` then workspace then ``~/.jaato/`` per
       :func:`shared.config_resolver.resolve_config_search_path`).

    Args:
        workspace_path: Optional explicit workspace path passed through to
            the credential lookup.
        config_root: Optional read-only-config root override.

    Returns:
        API key if found, None otherwise.
    """
    env_key = os.environ.get(ENV_NEBIUS_API_KEY) or os.environ.get(
        ENV_NEBIUS_API_KEY_VENDOR
    )
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
    """Resolve the Nebius API base URL from environment.

    Returns:
        The API base URL (the fixed serverless endpoint unless overridden).
    """
    return os.environ.get(ENV_NEBIUS_BASE_URL, DEFAULT_BASE_URL)


def resolve_model() -> Optional[str]:
    """Resolve the default model name from environment.

    Returns:
        Model name if found, None otherwise.
    """
    return os.environ.get(ENV_NEBIUS_MODEL)


def resolve_context_length() -> Optional[int]:
    """Resolve a context-window override from the environment.

    Returns the ``JAATO_NEBIUS_CONTEXT_LENGTH`` override as an int, or
    ``None`` when unset/invalid.  This is only the manual override tier:
    Nebius's ``GET /v1/models`` reports a per-model ``context_length``
    (RichModel), so the provider's PRIMARY tier is catalog auto-detect at
    connect time.  No hardcoded fallback is substituted (project
    no-fallback rule).
    """
    value = os.environ.get(ENV_NEBIUS_CONTEXT_LENGTH)
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return None


def is_self_hosted(base_url: str) -> bool:
    """Check if the base URL points to a local/self-hosted instance.

    Nebius Token Factory is a hosted service, so this is normally False;
    it stays for the rare case of a user fronting the API through a local
    proxy (``JAATO_NEBIUS_BASE_URL=http://localhost:...``), where an API
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


def get_checked_credential_locations() -> List[str]:
    """Get the list of locations checked for credentials.

    Used for error messages to help users understand what was checked.

    Returns:
        List of location descriptions.
    """
    locations = []

    jaato_key = os.environ.get(ENV_NEBIUS_API_KEY)
    if jaato_key:
        masked = (
            f"{jaato_key[:6]}...{jaato_key[-4:]}" if len(jaato_key) > 12 else "***"
        )
        locations.append(f"{ENV_NEBIUS_API_KEY}: set ({masked})")
    else:
        locations.append(f"{ENV_NEBIUS_API_KEY}: not set")

    vendor_key = os.environ.get(ENV_NEBIUS_API_KEY_VENDOR)
    if vendor_key:
        masked = (
            f"{vendor_key[:6]}...{vendor_key[-4:]}" if len(vendor_key) > 12 else "***"
        )
        locations.append(f"{ENV_NEBIUS_API_KEY_VENDOR}: set ({masked})")
    else:
        locations.append(f"{ENV_NEBIUS_API_KEY_VENDOR}: not set")

    try:
        from .auth import get_stored_api_key, get_credential_file_path
        stored_key = get_stored_api_key()
        if stored_key:
            cred_path = get_credential_file_path() or "nebius_auth.json"
            masked = (
                f"{stored_key[:6]}...{stored_key[-4:]}"
                if len(stored_key) > 12 else "***"
            )
            locations.append(f"Stored credentials ({cred_path}): set ({masked})")
        else:
            locations.append(
                "Stored credentials: not configured (use 'nebius-auth login')"
            )
    except ImportError:
        locations.append("Stored credentials: auth module not available")

    base_url = os.environ.get(ENV_NEBIUS_BASE_URL)
    if base_url:
        locations.append(f"{ENV_NEBIUS_BASE_URL}: {base_url}")
    else:
        locations.append(f"Endpoint: {DEFAULT_BASE_URL} (default)")

    model = os.environ.get(ENV_NEBIUS_MODEL)
    if model:
        locations.append(f"{ENV_NEBIUS_MODEL}: {model}")

    return locations
