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

# Default context window when no override is provided
DEFAULT_CONTEXT_LENGTH = 32768


def resolve_api_key() -> Optional[str]:
    """Resolve NIM API key from environment.

    Returns:
        API key if found, None otherwise.
    """
    return os.environ.get(ENV_NIM_API_KEY)


def resolve_base_url() -> str:
    """Resolve the NIM API base URL from environment.

    Returns:
        The API base URL.
    """
    return os.environ.get(ENV_NIM_BASE_URL, DEFAULT_BASE_URL)


def resolve_model() -> Optional[str]:
    """Resolve default model name from environment.

    Returns:
        Model name if found, None otherwise.
    """
    return os.environ.get(ENV_NIM_MODEL)


def resolve_context_length() -> int:
    """Resolve context window size from environment.

    Returns:
        Context window size in tokens.
    """
    value = os.environ.get(ENV_NIM_CONTEXT_LENGTH)
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return DEFAULT_CONTEXT_LENGTH


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


def get_checked_credential_locations() -> List[str]:
    """Get list of locations checked for credentials.

    Used for error messages to help users understand what was checked.

    Returns:
        List of location descriptions.
    """
    locations = []

    api_key = os.environ.get(ENV_NIM_API_KEY)
    if api_key:
        masked = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        locations.append(f"{ENV_NIM_API_KEY}: set ({masked})")
    else:
        locations.append(f"{ENV_NIM_API_KEY}: not set")

    base_url = os.environ.get(ENV_NIM_BASE_URL)
    if base_url:
        locations.append(f"{ENV_NIM_BASE_URL}: {base_url}")
    else:
        locations.append(f"Endpoint: {DEFAULT_BASE_URL} (default)")

    model = os.environ.get(ENV_NIM_MODEL)
    if model:
        locations.append(f"{ENV_NIM_MODEL}: {model}")

    return locations
