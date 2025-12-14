"""Environment variable resolution for GitHub Models provider.

This module handles loading configuration from environment variables,
following a naming convention that mirrors the Google GenAI provider:
- GITHUB_TOKEN for the access token (standard GitHub convention)
- JAATO_GITHUB_* for framework-specific settings

Resolution priority:
1. Explicit config passed in code
2. JAATO_GITHUB_* environment variables
3. Standard GitHub environment variables (GITHUB_TOKEN)
4. Defaults
"""

import os
from typing import Literal, Optional, List

# Auth method type
AuthMethod = Literal["pat", "app_token", "auto"]

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

# Default endpoint for GitHub Models
DEFAULT_ENDPOINT = "https://models.inference.ai.azure.com"


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
    """Resolve GitHub token from environment.

    Looks for GITHUB_TOKEN which is the standard GitHub environment variable.

    Returns:
        Token if found, None otherwise.
    """
    return os.environ.get(ENV_GITHUB_TOKEN)


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


def get_checked_credential_locations(auth_method: AuthMethod) -> List[str]:
    """Get list of locations checked for credentials.

    Used for error messages to help users understand what was checked.

    Args:
        auth_method: The authentication method being used.

    Returns:
        List of location descriptions.
    """
    locations = []

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
