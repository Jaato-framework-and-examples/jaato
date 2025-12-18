"""Environment variable resolution for Anthropic provider.

Provides a simple API key-based authentication (simpler than other providers).
"""

import os
from typing import Optional


def resolve_api_key() -> Optional[str]:
    """Resolve Anthropic API key from environment.

    Checks:
    1. ANTHROPIC_API_KEY environment variable

    Returns:
        API key if found, None otherwise.
    """
    return os.environ.get("ANTHROPIC_API_KEY")


def get_checked_credential_locations() -> list[str]:
    """Get list of locations checked for credentials.

    Returns:
        List of environment variable names checked.
    """
    return ["ANTHROPIC_API_KEY"]
