"""Environment variable resolution for Antigravity provider.

Supports OAuth token authentication for accessing AI models through
Google's Antigravity backend.
"""

import os
from typing import List, Optional


def resolve_project_id() -> Optional[str]:
    """Resolve Antigravity project ID from environment.

    Checks:
    1. JAATO_ANTIGRAVITY_PROJECT_ID environment variable

    Returns:
        Project ID if found, None otherwise.
    """
    return os.environ.get("JAATO_ANTIGRAVITY_PROJECT_ID")


def resolve_endpoint() -> Optional[str]:
    """Resolve custom Antigravity endpoint from environment.

    Checks:
    1. JAATO_ANTIGRAVITY_ENDPOINT environment variable

    Returns:
        Endpoint URL if found, None otherwise.
    """
    return os.environ.get("JAATO_ANTIGRAVITY_ENDPOINT")


def resolve_quota_type() -> str:
    """Resolve preferred quota type from environment.

    Checks:
    1. JAATO_ANTIGRAVITY_QUOTA environment variable

    Returns:
        Quota type: "antigravity" (default) or "gemini-cli"
    """
    val = os.environ.get("JAATO_ANTIGRAVITY_QUOTA", "").lower()
    if val in ("gemini-cli", "gemini_cli", "cli"):
        return "gemini-cli"
    return "antigravity"


def resolve_thinking_level() -> Optional[str]:
    """Resolve default thinking level for Gemini 3 models.

    Checks:
    1. JAATO_ANTIGRAVITY_THINKING_LEVEL environment variable

    Returns:
        Thinking level ("minimal", "low", "medium", "high") if set.
    """
    val = os.environ.get("JAATO_ANTIGRAVITY_THINKING_LEVEL", "").lower()
    if val in ("minimal", "low", "medium", "high"):
        return val
    return None


def resolve_thinking_budget() -> int:
    """Resolve thinking budget for Claude thinking models.

    Checks:
    1. JAATO_ANTIGRAVITY_THINKING_BUDGET environment variable

    Returns:
        Thinking budget in tokens (default: 8192).
    """
    val = os.environ.get("JAATO_ANTIGRAVITY_THINKING_BUDGET", "8192")
    try:
        budget = int(val)
        # Clamp to valid range
        if budget < 8192:
            return 8192
        if budget > 32768:
            return 32768
        return budget
    except ValueError:
        return 8192


def resolve_auto_rotate() -> bool:
    """Resolve whether to auto-rotate accounts on rate limit.

    Checks:
    1. JAATO_ANTIGRAVITY_AUTO_ROTATE environment variable

    Returns:
        True if enabled (default), False otherwise.
    """
    val = os.environ.get("JAATO_ANTIGRAVITY_AUTO_ROTATE", "true").lower()
    return val in ("1", "true", "yes", "on")


def resolve_retry_empty() -> bool:
    """Resolve whether to retry on empty responses.

    Checks:
    1. JAATO_ANTIGRAVITY_RETRY_EMPTY environment variable

    Returns:
        True if enabled (default), False otherwise.
    """
    val = os.environ.get("JAATO_ANTIGRAVITY_RETRY_EMPTY", "true").lower()
    return val in ("1", "true", "yes", "on")


def resolve_session_recovery() -> bool:
    """Resolve whether to enable session recovery on tool_result_missing errors.

    Checks:
    1. JAATO_ANTIGRAVITY_SESSION_RECOVERY environment variable

    Returns:
        True if enabled (default), False otherwise.
    """
    val = os.environ.get("JAATO_ANTIGRAVITY_SESSION_RECOVERY", "true").lower()
    return val in ("1", "true", "yes", "on")


def get_checked_credential_locations() -> List[str]:
    """Get list of locations checked for credentials.

    Returns:
        List of locations/methods checked for authentication.
    """
    return [
        "OAuth tokens (via antigravity-auth login)",
        "~/.config/jaato/antigravity_accounts.json",
        "JAATO_ANTIGRAVITY_PROJECT_ID environment variable",
    ]
