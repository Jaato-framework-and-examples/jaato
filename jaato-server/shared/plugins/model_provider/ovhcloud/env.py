"""Environment variable resolution for the OVHcloud AI Endpoints provider.

Configuration is resolved from environment variables:
- JAATO_OVHCLOUD_API_KEY (jaato namespace, highest priority) or the vendor's
  own OVH_AI_ENDPOINTS_ACCESS_TOKEN for the API key
- JAATO_OVHCLOUD_BASE_URL for the endpoint (default: the unified
  OpenAI-compatible AI Endpoints gateway)
- JAATO_OVHCLOUD_MODEL for the default model name
- JAATO_OVHCLOUD_CONTEXT_LENGTH to override the catalog-detected context window
- JAATO_OVHCLOUD_ALLOW_ANONYMOUS to opt into OVHcloud's keyless free tier
  (heavily rate-limited; intended for evaluation only)

Resolution priority:
1. Explicit config passed in code (ProviderConfig)
2. JAATO_OVHCLOUD_* / OVH_AI_ENDPOINTS_ACCESS_TOKEN environment variables
3. Stored credentials (ovhcloud-auth)
"""

import os
from typing import List, Optional

# ============================================================
# Environment Variable Names
# ============================================================

# Primary (jaato namespace) and the vendor's documented variable.  Both are
# legitimate documented sources, consulted in priority order — not a guessed
# fallback (project no-fallback rule).  OVHcloud's own AI Endpoints examples
# read OVH_AI_ENDPOINTS_ACCESS_TOKEN.
ENV_OVHCLOUD_API_KEY = "JAATO_OVHCLOUD_API_KEY"
ENV_OVHCLOUD_API_KEY_VENDOR = "OVH_AI_ENDPOINTS_ACCESS_TOKEN"
ENV_OVHCLOUD_BASE_URL = "JAATO_OVHCLOUD_BASE_URL"
ENV_OVHCLOUD_MODEL = "JAATO_OVHCLOUD_MODEL"
ENV_OVHCLOUD_CONTEXT_LENGTH = "JAATO_OVHCLOUD_CONTEXT_LENGTH"
ENV_OVHCLOUD_ALLOW_ANONYMOUS = "JAATO_OVHCLOUD_ALLOW_ANONYMOUS"

# Default unified OpenAI-compatible endpoint for OVHcloud AI Endpoints.
# Every catalog model is addressable by name through this single gateway
# (the older per-model *.endpoints.kepler.ai.cloud.ovh.net URLs still work,
# via JAATO_OVHCLOUD_BASE_URL, but are not the default).
# https://help.ovhcloud.com/csm/en-public-cloud-ai-endpoints-getting-started
DEFAULT_BASE_URL = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"


def resolve_api_key(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """Resolve the OVHcloud API key from environment or stored credentials.

    Resolution priority (each a documented source, in order):
    1. ``JAATO_OVHCLOUD_API_KEY`` (jaato namespace).
    2. ``OVH_AI_ENDPOINTS_ACCESS_TOKEN`` (the vendor's own documented
       variable, so users who already set it for OVHcloud's OpenAI SDK
       examples work with no extra configuration).
    3. Stored credentials from the ovhcloud-auth flow (resolves under
       ``config_root`` then workspace then ``~/.jaato/`` per
       :func:`shared.config_resolver.resolve_config_search_path`).

    Args:
        workspace_path: Optional explicit workspace path passed through to
            the credential lookup.
        config_root: Optional read-only-config root override.

    Returns:
        API key if found, None otherwise.
    """
    env_key = os.environ.get(ENV_OVHCLOUD_API_KEY) or os.environ.get(
        ENV_OVHCLOUD_API_KEY_VENDOR
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
    """Resolve the OVHcloud AI Endpoints base URL from environment.

    Returns:
        The API base URL (the unified gateway unless overridden).
    """
    return os.environ.get(ENV_OVHCLOUD_BASE_URL, DEFAULT_BASE_URL)  # env: endpoint (default https://oai.endpoints.kepler.ai.cloud.ovh.net/v1)


def resolve_model() -> Optional[str]:
    """Resolve the default model name from environment.

    Returns:
        Model name if found, None otherwise.
    """
    return os.environ.get(ENV_OVHCLOUD_MODEL)  # env: default model name (e.g. gpt-oss-120b)


def resolve_context_length() -> Optional[int]:
    """Resolve a context-window override from the environment.

    Returns the ``JAATO_OVHCLOUD_CONTEXT_LENGTH`` override as an int, or
    ``None`` when unset/invalid.  This is only the manual override tier:
    the provider's PRIMARY tier is catalog auto-detect at connect time
    (when OVHcloud's ``GET /v1/models`` reports a per-model context
    length).  No hardcoded fallback is substituted (project no-fallback
    rule).
    """
    value = os.environ.get(ENV_OVHCLOUD_CONTEXT_LENGTH)  # env: override the catalog-detected context window
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return None


def resolve_allow_anonymous() -> bool:
    """Whether the keyless free tier is enabled via the environment.

    OVHcloud AI Endpoints officially serves unauthenticated requests on a
    heavily rate-limited free tier (intended for evaluation).  jaato's
    agentic workloads would trip that limit constantly, so keyless access
    is an explicit opt-in — never a silent fallback for a missing key.
    """
    value = os.environ.get(ENV_OVHCLOUD_ALLOW_ANONYMOUS, "")  # env: opt into the keyless rate-limited free tier
    return value.strip().lower() in ("1", "true", "yes", "on")


def is_self_hosted(base_url: str) -> bool:
    """Check if the base URL points to a local/self-hosted instance.

    OVHcloud AI Endpoints is a hosted service, so this is normally False;
    it stays for the rare case of a user fronting the API through a local
    proxy (``JAATO_OVHCLOUD_BASE_URL=http://localhost:...``), where an API
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
    source — the profile ``plugin_configs.ovhcloud.api_key`` knob — which the
    env-only checks below cannot see.

    Returns:
        List of location descriptions.
    """
    from ..base import profile_api_key_location

    locations = [profile_api_key_location(config, "ovhcloud")]

    jaato_key = os.environ.get(ENV_OVHCLOUD_API_KEY)  # env: OVHcloud AI Endpoints API key (jaato namespace, highest priority)
    if jaato_key:
        masked = (
            f"{jaato_key[:6]}...{jaato_key[-4:]}" if len(jaato_key) > 12 else "***"
        )
        locations.append(f"{ENV_OVHCLOUD_API_KEY}: set ({masked})")
    else:
        locations.append(f"{ENV_OVHCLOUD_API_KEY}: not set")

    vendor_key = os.environ.get(ENV_OVHCLOUD_API_KEY_VENDOR)  # env: the vendor's own access-token var; honored when JAATO_OVHCLOUD_API_KEY is unset
    if vendor_key:
        masked = (
            f"{vendor_key[:6]}...{vendor_key[-4:]}" if len(vendor_key) > 12 else "***"
        )
        locations.append(f"{ENV_OVHCLOUD_API_KEY_VENDOR}: set ({masked})")
    else:
        locations.append(f"{ENV_OVHCLOUD_API_KEY_VENDOR}: not set")

    try:
        from .auth import get_stored_api_key, get_credential_file_path
        stored_key = get_stored_api_key()
        if stored_key:
            cred_path = get_credential_file_path() or "ovhcloud_auth.json"
            masked = (
                f"{stored_key[:6]}...{stored_key[-4:]}"
                if len(stored_key) > 12 else "***"
            )
            locations.append(f"Stored credentials ({cred_path}): set ({masked})")
        else:
            locations.append(
                "Stored credentials: not configured (use 'ovhcloud-auth login')"
            )
    except ImportError:
        locations.append("Stored credentials: auth module not available")

    base_url = os.environ.get(ENV_OVHCLOUD_BASE_URL)
    if base_url:
        locations.append(f"{ENV_OVHCLOUD_BASE_URL}: {base_url}")
    else:
        locations.append(f"Endpoint: {DEFAULT_BASE_URL} (default)")

    model = os.environ.get(ENV_OVHCLOUD_MODEL)
    if model:
        locations.append(f"{ENV_OVHCLOUD_MODEL}: {model}")

    return locations
