"""Environment variable resolution for GitHub Models provider.

This module handles loading configuration from environment variables,
following a naming convention that mirrors the Google GenAI provider:
- GITHUB_TOKEN for the access token (standard GitHub convention)
- JAATO_GITHUB_* for framework-specific settings

Resolution priority:
1. Explicit config passed in code
2. Device Code OAuth tokens (stored via github-auth login)
3. JAATO_GITHUB_* environment variables
4. Standard GitHub environment variables (GITHUB_TOKEN)
5. Defaults
"""

import os
import urllib.parse
import urllib.request
from typing import Literal, Optional, List

# Auth method type
AuthMethod = Literal["pat", "app_token", "oauth", "auto"]

# ============================================================
# Environment Variable Names
# ============================================================

# GitHub standard (industry convention)
ENV_GITHUB_TOKEN = "GITHUB_TOKEN"

# GitHub Models specific (JAATO namespace)
ENV_GITHUB_AUTH_METHOD = "JAATO_GITHUB_AUTH_METHOD"
ENV_GITHUB_ORGANIZATION = "JAATO_GITHUB_ORGANIZATION"
ENV_GITHUB_ENTERPRISE = "JAATO_GITHUB_ENTERPRISE"
ENV_GITHUB_ENDPOINT = "JAATO_GITHUB_ENDPOINT"

# Proxy configuration (JAATO namespace)
# JAATO_NO_PROXY uses exact host matching (unlike standard NO_PROXY which does suffix matching)
ENV_JAATO_NO_PROXY = "JAATO_NO_PROXY"

# Default endpoint for GitHub Models (new API as of May 2025)
# The old Azure endpoint (models.inference.ai.azure.com) was deprecated July 2025
DEFAULT_ENDPOINT = "https://models.github.ai/inference"


def _get_env_with_fallback(*names: str, default: Optional[str] = None) -> Optional[str]:
    """Get the first defined environment variable from a list of names."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def resolve_auth_method() -> AuthMethod:
    """Resolve the authentication method from environment.

    Priority:
    1. JAATO_GITHUB_AUTH_METHOD if set
    2. Default to "auto" which uses GITHUB_TOKEN

    Returns:
        The resolved authentication method.
    """
    explicit = os.environ.get(ENV_GITHUB_AUTH_METHOD, "").lower()
    if explicit in ("pat", "app_token", "auto"):
        return explicit  # type: ignore
    return "auto"


def resolve_token() -> Optional[str]:
    """Resolve GitHub token from environment or stored OAuth.

    Priority:
    1. Device Code OAuth tokens (stored via github-auth login)
    2. GITHUB_TOKEN environment variable

    Returns:
        Token if found, None otherwise.
    """
    # First, try stored OAuth token from device code flow
    # Note: We return the OAuth token here for validation purposes.
    # The provider will exchange it for a Copilot token when making API calls.
    try:
        from .oauth import get_stored_oauth_token
        stored_token = get_stored_oauth_token()
        if stored_token:
            return stored_token
    except ImportError:
        pass  # oauth module not available

    # Fall back to environment variable
    return os.environ.get(ENV_GITHUB_TOKEN)


def resolve_token_source() -> Optional[str]:
    """Determine the source of the resolved token.

    Returns:
        "oauth" if using stored OAuth token,
        "env" if using GITHUB_TOKEN env var,
        None if no token found.
    """
    # Check stored OAuth token first
    try:
        from .oauth import get_stored_oauth_token
        if get_stored_oauth_token():
            return "oauth"
    except ImportError:
        pass

    # Check environment variable
    if os.environ.get(ENV_GITHUB_TOKEN):
        return "env"

    return None


def resolve_organization() -> Optional[str]:
    """Resolve GitHub organization for billing attribution.

    When set, API requests use the org-scoped endpoint for billing attribution.

    Returns:
        Organization name if found, None otherwise.
    """
    return os.environ.get(ENV_GITHUB_ORGANIZATION)


def resolve_enterprise() -> Optional[str]:
    """Resolve GitHub enterprise name.

    Used for enterprise context and policy compliance.

    Returns:
        Enterprise name if found, None otherwise.
    """
    return os.environ.get(ENV_GITHUB_ENTERPRISE)


def resolve_endpoint() -> str:
    """Resolve the GitHub Models API endpoint.

    Priority:
    1. JAATO_GITHUB_ENDPOINT if set
    2. Default GitHub Models endpoint

    Returns:
        The API endpoint URL.
    """
    return os.environ.get(ENV_GITHUB_ENDPOINT, DEFAULT_ENDPOINT)


def get_checked_credential_locations(auth_method: AuthMethod = "auto") -> List[str]:
    """Get list of locations checked for credentials.

    Used for error messages to help users understand what was checked.

    Args:
        auth_method: The authentication method being used.

    Returns:
        List of location descriptions.
    """
    locations = []

    # Check stored OAuth token
    try:
        from .oauth import get_stored_oauth_token
        oauth_token = get_stored_oauth_token()
        if oauth_token:
            masked = f"{oauth_token[:10]}...{oauth_token[-4:]}"
            locations.append(f"Device Code OAuth: set ({masked})")
        else:
            locations.append("Device Code OAuth: not configured (run 'github-auth login')")
    except ImportError:
        locations.append("Device Code OAuth: not available")

    # Check environment variable
    token = os.environ.get(ENV_GITHUB_TOKEN)
    if token:
        # Mask the token for security
        masked = f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "***"
        locations.append(f"{ENV_GITHUB_TOKEN}: set ({masked})")
    else:
        locations.append(f"{ENV_GITHUB_TOKEN}: not set")

    org = os.environ.get(ENV_GITHUB_ORGANIZATION)
    if org:
        locations.append(f"{ENV_GITHUB_ORGANIZATION}: {org}")

    enterprise = os.environ.get(ENV_GITHUB_ENTERPRISE)
    if enterprise:
        locations.append(f"{ENV_GITHUB_ENTERPRISE}: {enterprise}")

    endpoint = os.environ.get(ENV_GITHUB_ENDPOINT)
    if endpoint:
        locations.append(f"{ENV_GITHUB_ENDPOINT}: {endpoint}")
    else:
        locations.append(f"Endpoint: {DEFAULT_ENDPOINT} (default)")

    return locations


# ============================================================
# Proxy Configuration
# ============================================================

# Environment variable to enable/disable Kerberos proxy auth
ENV_JAATO_KERBEROS_PROXY = "JAATO_KERBEROS_PROXY"


def _get_jaato_no_proxy_hosts() -> List[str]:
    """Get list of hosts from JAATO_NO_PROXY.

    Returns:
        List of hostnames (lowercase) that should bypass proxy.
    """
    value = os.environ.get(ENV_JAATO_NO_PROXY, "")
    if not value:
        return []
    return [h.strip().lower() for h in value.split(",") if h.strip()]


def should_bypass_proxy(url: str) -> bool:
    """Check if a URL should bypass proxy based on JAATO_NO_PROXY.

    Uses exact host matching (case-insensitive), unlike standard NO_PROXY
    which does suffix matching.

    Args:
        url: The URL to check.

    Returns:
        True if the host matches exactly an entry in JAATO_NO_PROXY.
    """
    no_proxy_hosts = _get_jaato_no_proxy_hosts()
    if not no_proxy_hosts:
        return False

    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    if not host:
        return False

    host = host.lower()
    return host in no_proxy_hosts


def is_kerberos_proxy_enabled() -> bool:
    """Check if Kerberos proxy authentication is enabled.

    Enabled when JAATO_KERBEROS_PROXY is set to 'true', '1', or 'yes'.

    Returns:
        True if Kerberos proxy auth should be used.
    """
    value = os.environ.get(ENV_JAATO_KERBEROS_PROXY, "").lower()
    return value in ("true", "1", "yes")


def get_url_opener(url: str) -> urllib.request.OpenerDirector:
    """Get an appropriate urllib opener for a URL.

    Priority:
    1. If JAATO_NO_PROXY matches (exact), bypass proxy entirely
    2. If JAATO_KERBEROS_PROXY=true, use Kerberos proxy authentication
    3. Otherwise use default opener (standard proxy env vars)

    Args:
        url: The URL that will be requested.

    Returns:
        An OpenerDirector configured appropriately for the URL.
    """
    if should_bypass_proxy(url):
        # Create opener with empty ProxyHandler to disable proxy
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))

    if is_kerberos_proxy_enabled():
        # Use Kerberos proxy authentication
        try:
            from .proxy_auth import create_kerberos_proxy_opener
            return create_kerberos_proxy_opener()
        except ImportError:
            # pyspnego not installed, fall back to default
            pass

    # Use default opener (includes ProxyHandler with env vars)
    return urllib.request.build_opener()
