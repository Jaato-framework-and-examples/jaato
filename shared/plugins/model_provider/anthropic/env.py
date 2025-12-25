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


def resolve_enable_thinking() -> bool:
    """Resolve whether extended thinking is enabled.

    Checks:
    1. JAATO_ANTHROPIC_ENABLE_THINKING environment variable

    Returns:
        True if enabled, False otherwise.
    """
    val = os.environ.get("JAATO_ANTHROPIC_ENABLE_THINKING", "").lower()
    return val in ("1", "true", "yes", "on")


def resolve_thinking_budget() -> int:
    """Resolve thinking budget (max thinking tokens).

    Checks:
    1. JAATO_ANTHROPIC_THINKING_BUDGET environment variable

    Returns:
        Thinking budget in tokens (default: 10000).
    """
    val = os.environ.get("JAATO_ANTHROPIC_THINKING_BUDGET", "10000")
    try:
        return int(val)
    except ValueError:
        return 10000


def resolve_enable_caching() -> bool:
    """Resolve whether prompt caching is enabled.

    Checks:
    1. JAATO_ANTHROPIC_ENABLE_CACHING environment variable

    Returns:
        True if enabled, False otherwise.
    """
    val = os.environ.get("JAATO_ANTHROPIC_ENABLE_CACHING", "").lower()
    return val in ("1", "true", "yes", "on")
