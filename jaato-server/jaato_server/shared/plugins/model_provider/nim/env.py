"""Environment variable resolution for NVIDIA NIM provider.

Configuration is resolved from environment variables:
- JAATO_NIM_API_KEY for the API key (required for hosted, optional for self-hosted)
- JAATO_NIM_BASE_URL for the endpoint (default: NVIDIA hosted API)
- JAATO_NIM_MODEL for the default model name
- JAATO_NIM_CONTEXT_LENGTH for overriding context window size

Resolution priority:
1. Explicit config passed in code (ProviderConfig)
2. JAATO_NIM_* environment variables
3. Defaults
"""

import os
from typing import List, Optional

# ============================================================
# Environment Variable Names
# ============================================================

ENV_NIM_API_KEY = "JAATO_NIM_API_KEY"
ENV_NIM_BASE_URL = "JAATO_NIM_BASE_URL"
ENV_NIM_MODEL = "JAATO_NIM_MODEL"
ENV_NIM_CONTEXT_LENGTH = "JAATO_NIM_CONTEXT_LENGTH"

# Default endpoint for NVIDIA hosted NIM API
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


def resolve_api_key(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """Resolve NIM API key from environment or stored credentials.

    Resolution priority:
    1. JAATO_NIM_API_KEY environment variable
    2. Stored credentials from nim-auth plugin (resolves under
       ``config_root`` then workspace then ``~/.jaato/`` per
       :func:`shared.config_resolver.resolve_config_search_path`).

    Args:
        workspace_path: Optional explicit workspace path passed through to
            the credential lookup.
        config_root: Optional read-only-config root override.  When set,
            stored credentials are looked up under
            ``<config_root>/nim_auth.json`` instead of
            ``<workspace>/.jaato/nim_auth.json``.

    Returns:
        API key if found, None otherwise.
    """
    env_key = os.environ.get(ENV_NIM_API_KEY)  # env: API key for hosted NIM (nvapi-... from build.nvidia.com)
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
    """Resolve the NIM API base URL from environment.

    Returns:
        The API base URL.
    """
    return os.environ.get(ENV_NIM_BASE_URL, DEFAULT_BASE_URL)  # env: endpoint (default https://integrate.api.nvidia.com/v1); point at self-hosted NIM


def resolve_model() -> Optional[str]:
    """Resolve default model name from environment.

    Returns:
        Model name if found, None otherwise.
    """
    return os.environ.get(ENV_NIM_MODEL)  # env: default model name


def resolve_context_length() -> Optional[int]:
    """Resolve context window size from the environment.

    Returns the ``JAATO_NIM_CONTEXT_LENGTH`` override as an int, or ``None``
    when unset/invalid.  No hardcoded fallback is substituted (project
    no-fallback rule); the caller routes this through ``resolve_context_window``
    and raises a "not configured" error when no tier resolves.  NIM's
    OpenAI-compatible ``/v1/models`` does not surface a per-model context
    window, so there is no auto-detect tier for this provider.
    """
    value = os.environ.get(ENV_NIM_CONTEXT_LENGTH)  # env: override context window size
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return None


def is_self_hosted(base_url: str) -> bool:
    """Check if the base URL points to a self-hosted NIM instance.

    Self-hosted instances (localhost, private networks) typically don't
    require API key authentication.

    Args:
        base_url: The NIM API endpoint URL.

    Returns:
        True if the URL appears to be a self-hosted instance.
    """
    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    return host in ("localhost", "127.0.0.1", "0.0.0.0") or host.startswith("192.168.") or host.startswith("10.")


def get_checked_credential_locations(config=None) -> List[str]:
    """Get list of locations checked for credentials.

    Used for error messages to help users understand what was checked.

    ``config`` (optional ``ProviderConfig``) surfaces the highest-precedence
    source — the profile ``plugin_configs.nim.api_key`` knob — which the
    env-only checks below cannot see.

    Returns:
        List of location descriptions.
    """
    from ..base import profile_api_key_location

    locations = [profile_api_key_location(config, "nim")]

    api_key = os.environ.get(ENV_NIM_API_KEY)
    if api_key:
        masked = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        locations.append(f"{ENV_NIM_API_KEY}: set ({masked})")
    else:
        locations.append(f"{ENV_NIM_API_KEY}: not set")

    # Check stored credentials
    try:
        from .auth import get_stored_api_key, get_credential_file_path
        stored_key = get_stored_api_key()
        if stored_key:
            cred_path = get_credential_file_path() or "nim_auth.json"
            masked = f"{stored_key[:8]}...{stored_key[-4:]}" if len(stored_key) > 12 else "***"
            locations.append(f"Stored credentials ({cred_path}): set ({masked})")
        else:
            locations.append("Stored credentials: not configured (use 'nim-auth login')")
    except ImportError:
        locations.append("Stored credentials: auth module not available")

    base_url = os.environ.get(ENV_NIM_BASE_URL)
    if base_url:
        locations.append(f"{ENV_NIM_BASE_URL}: {base_url}")
    else:
        locations.append(f"Endpoint: {DEFAULT_BASE_URL} (default)")

    model = os.environ.get(ENV_NIM_MODEL)
    if model:
        locations.append(f"{ENV_NIM_MODEL}: {model}")

    return locations
