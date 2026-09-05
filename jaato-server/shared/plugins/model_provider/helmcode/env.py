"""Environment variable resolution for the Helmcode provider.

Configuration is resolved from environment variables:
- JAATO_HELMCODE_API_KEY (jaato namespace, highest priority) or the vendor's
  own HELMCODE_API_KEY for the API key
- JAATO_HELMCODE_BASE_URL for the endpoint (default: the hosted
  OpenAI-compatible inference API)
- JAATO_HELMCODE_MODEL for the default model name
- JAATO_HELMCODE_CONTEXT_LENGTH to override the catalog-detected context
  window

Resolution priority:
1. Explicit config passed in code (ProviderConfig)
2. JAATO_HELMCODE_API_KEY / HELMCODE_API_KEY environment variables
3. Stored credentials (helmcode-auth)
"""

import os
from typing import List, Optional

# ============================================================
# Environment Variable Names
# ============================================================

# Primary (jaato namespace) and the vendor's documented variable.  Both are
# legitimate documented sources, consulted in priority order — not a guessed
# fallback (project no-fallback rule).  Helmcode's own quickstart and
# authentication docs export HELMCODE_API_KEY.
ENV_HELMCODE_API_KEY = "JAATO_HELMCODE_API_KEY"
ENV_HELMCODE_API_KEY_VENDOR = "HELMCODE_API_KEY"
ENV_HELMCODE_BASE_URL = "JAATO_HELMCODE_BASE_URL"
ENV_HELMCODE_MODEL = "JAATO_HELMCODE_MODEL"
ENV_HELMCODE_CONTEXT_LENGTH = "JAATO_HELMCODE_CONTEXT_LENGTH"

# Default OpenAI-compatible endpoint for the hosted Helmcode API
# (https://helmcode.com/docs/authentication).  Every catalog model — the
# open-weight models Helmcode runs on its own EU hardware and the resold
# frontier models alike — is addressable by name through this single
# gateway; only the ``model`` id changes.
DEFAULT_BASE_URL = "https://api.helmcode.com/v1"


def resolve_api_key(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """Resolve the Helmcode API key from environment or stored credentials.

    Resolution priority (each a documented source, in order):
    1. ``JAATO_HELMCODE_API_KEY`` (jaato namespace).
    2. ``HELMCODE_API_KEY`` (the vendor's own documented variable, so users
       who already set it for Helmcode's OpenAI SDK examples work with no
       extra configuration).
    3. Stored credentials from the helmcode-auth flow (resolves under
       ``config_root`` then workspace then ``~/.jaato/`` per
       :func:`shared.config_resolver.resolve_config_search_path`).

    Args:
        workspace_path: Optional explicit workspace path passed through to
            the credential lookup.
        config_root: Optional read-only-config root override.

    Returns:
        API key if found, None otherwise.
    """
    env_key = os.environ.get(ENV_HELMCODE_API_KEY) or os.environ.get(
        ENV_HELMCODE_API_KEY_VENDOR
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
    """Resolve the Helmcode base URL from environment.

    Returns:
        The API base URL (the hosted gateway unless overridden).
    """
    return os.environ.get(ENV_HELMCODE_BASE_URL, DEFAULT_BASE_URL)  # env: endpoint (default https://api.helmcode.com/v1)


def resolve_model() -> Optional[str]:
    """Resolve the default model name from environment.

    Returns:
        Model name if found, None otherwise.
    """
    return os.environ.get(ENV_HELMCODE_MODEL)  # env: default model name (e.g. qwen3.6)


def resolve_context_length() -> Optional[int]:
    """Resolve a context-window override from the environment.

    Returns the ``JAATO_HELMCODE_CONTEXT_LENGTH`` override as an int, or
    ``None`` when unset/invalid.  This is the manual override tier: the
    provider's PRIMARY tier is catalog auto-detect at connect time (when
    Helmcode's ``GET /v1/models`` reports a per-model context length).  No
    hardcoded fallback is substituted (project no-fallback rule) — the
    per-model windows are published at https://helmcode.com/docs/models.
    """
    value = os.environ.get(ENV_HELMCODE_CONTEXT_LENGTH)  # env: override the catalog-detected context window
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return None


def is_self_hosted(base_url: str) -> bool:
    """Check if the base URL points to a local/self-hosted instance.

    Helmcode is a hosted service, so this is normally False; it stays for
    the case of a user fronting the API through a local proxy
    (``JAATO_HELMCODE_BASE_URL=http://localhost:...``), where an API key
    may not be required.  Helmcode's On-premise plan is the same API
    served from the customer's own hardware, which is reached the same
    way — through ``JAATO_HELMCODE_BASE_URL``.  Note that an on-premise
    deployment on a routable corporate host is *not* matched here and
    still requires a key, which is the safe default: only unmistakably
    local addresses waive the credential check.

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
    source — the profile ``plugin_configs.helmcode.api_key`` knob — which the
    env-only checks below cannot see.

    Returns:
        List of location descriptions.
    """
    from ..base import profile_api_key_location

    locations = [profile_api_key_location(config, "helmcode")]

    jaato_key = os.environ.get(ENV_HELMCODE_API_KEY)  # env: Helmcode API key (jaato namespace, highest priority)
    if jaato_key:
        masked = (
            f"{jaato_key[:6]}...{jaato_key[-4:]}" if len(jaato_key) > 12 else "***"
        )
        locations.append(f"{ENV_HELMCODE_API_KEY}: set ({masked})")
    else:
        locations.append(f"{ENV_HELMCODE_API_KEY}: not set")

    vendor_key = os.environ.get(ENV_HELMCODE_API_KEY_VENDOR)  # env: the vendor's own key var; honored when JAATO_HELMCODE_API_KEY is unset
    if vendor_key:
        masked = (
            f"{vendor_key[:6]}...{vendor_key[-4:]}" if len(vendor_key) > 12 else "***"
        )
        locations.append(f"{ENV_HELMCODE_API_KEY_VENDOR}: set ({masked})")
    else:
        locations.append(f"{ENV_HELMCODE_API_KEY_VENDOR}: not set")

    try:
        from .auth import get_stored_api_key, get_credential_file_path
        stored_key = get_stored_api_key()
        if stored_key:
            cred_path = get_credential_file_path() or "helmcode_auth.json"
            masked = (
                f"{stored_key[:6]}...{stored_key[-4:]}"
                if len(stored_key) > 12 else "***"
            )
            locations.append(f"Stored credentials ({cred_path}): set ({masked})")
        else:
            locations.append(
                "Stored credentials: not configured (use 'helmcode-auth login')"
            )
    except ImportError:
        locations.append("Stored credentials: auth module not available")

    base_url = os.environ.get(ENV_HELMCODE_BASE_URL)
    if base_url:
        locations.append(f"{ENV_HELMCODE_BASE_URL}: {base_url}")
    else:
        locations.append(f"Endpoint: {DEFAULT_BASE_URL} (default)")

    model = os.environ.get(ENV_HELMCODE_MODEL)
    if model:
        locations.append(f"{ENV_HELMCODE_MODEL}: {model}")

    return locations
