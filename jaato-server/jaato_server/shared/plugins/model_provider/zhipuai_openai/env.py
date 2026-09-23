"""Environment variable resolution for Zhipu AI OpenAI-compatible provider.

This provider uses Z.AI's OpenAI-compatible API endpoint (``/api/paas/v4``)
instead of the Anthropic-compatible one.  The API key is the same Z.AI
credential used by the Anthropic-compatible provider.

Environment variables:
    ZHIPUAI_API_KEY: Z.AI API key (shared with the zhipuai provider)
    ZHIPUAI_OPENAI_BASE_URL: OpenAI-compatible endpoint
        (default: https://api.z.ai/api/paas/v4)
    ZHIPUAI_OPENAI_MODEL: Default model name
    ZHIPUAI_OPENAI_CONTEXT_LENGTH: Override context window size
    ZHIPUAI_ENABLE_THINKING: Enable extended thinking (default: false)
    ZHIPUAI_THINKING_BUDGET: Max thinking tokens (default: 10000)
"""

import os
from typing import List, Optional


# ============================================================
# Environment Variable Names
# ============================================================

# Shared with the zhipuai (Anthropic-compat) provider — same Z.AI account.
ENV_API_KEY = "ZHIPUAI_API_KEY"

ENV_BASE_URL = "ZHIPUAI_OPENAI_BASE_URL"
ENV_MODEL = "ZHIPUAI_OPENAI_MODEL"
ENV_CONTEXT_LENGTH = "ZHIPUAI_OPENAI_CONTEXT_LENGTH"
ENV_ENABLE_THINKING = "ZHIPUAI_ENABLE_THINKING"
ENV_THINKING_BUDGET = "ZHIPUAI_THINKING_BUDGET"

# Z.AI OpenAI-compatible API endpoint.
# The OpenAI SDK appends /chat/completions internally.
DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"

# Default model
DEFAULT_MODEL = "glm-4.7"

# Default context window when no override is provided (matches GLM-5/4.7 generation)
DEFAULT_CONTEXT_LENGTH = 204800


def resolve_api_key() -> Optional[str]:
    """Resolve Z.AI API key from environment or stored credentials.

    Resolution priority:
    1. ZHIPUAI_API_KEY environment variable
    2. Stored credentials from zhipuai-auth plugin (shared credential file)

    Returns:
        API key if found, None otherwise.
    """
    env_key = os.environ.get(ENV_API_KEY)
    if env_key:
        return env_key
    # Fall back to zhipuai's stored credentials (same API key)
    try:
        from ..zhipuai.auth import get_stored_api_key
        return get_stored_api_key()
    except ImportError:
        return None


def resolve_base_url() -> str:
    """Resolve the OpenAI-compatible API base URL.

    Returns:
        The API base URL.
    """
    return os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL)  # env: endpoint for Zhipu's OpenAI-compatible API (default https://api.z.ai/api/paas/v4)


def resolve_model() -> Optional[str]:
    """Resolve default model name from environment.

    Returns:
        Model name if found, None otherwise.
    """
    return os.environ.get(ENV_MODEL)  # env: default model name


def resolve_context_length() -> Optional[int]:
    """Resolve custom context length override.

    Returns:
        Context length in tokens if set, None otherwise.
    """
    val = os.environ.get(ENV_CONTEXT_LENGTH)  # env: override context window size
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return None


def resolve_enable_thinking() -> bool:
    """Resolve whether extended thinking is enabled.

    Returns:
        True if thinking is enabled, False by default.
    """
    val = os.environ.get(ENV_ENABLE_THINKING, "").lower()
    return val in ("1", "true", "yes")


def resolve_thinking_budget() -> int:
    """Resolve thinking budget (max thinking tokens).

    Returns:
        Thinking budget in tokens (default: 10000).
    """
    val = os.environ.get(ENV_THINKING_BUDGET)
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return 10000


def get_checked_credential_locations() -> List[str]:
    """Get list of locations checked for credentials.

    Used for error messages to help users understand what was checked.

    Returns:
        List of location descriptions.
    """
    locations = []

    api_key = os.environ.get(ENV_API_KEY)
    if api_key:
        masked = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        locations.append(f"{ENV_API_KEY}: set ({masked})")
    else:
        locations.append(f"{ENV_API_KEY}: not set")

    # Check zhipuai stored credentials
    try:
        from ..zhipuai.auth import get_stored_api_key, get_credential_file_path
        stored_key = get_stored_api_key()
        if stored_key:
            cred_path = get_credential_file_path() or "zhipuai_auth.json"
            masked = f"{stored_key[:8]}...{stored_key[-4:]}" if len(stored_key) > 12 else "***"
            locations.append(f"Stored credentials ({cred_path}): set ({masked})")
        else:
            locations.append("Stored credentials: not configured (use 'zhipuai-auth login')")
    except ImportError:
        locations.append("Stored credentials: zhipuai auth module not available")

    base_url = os.environ.get(ENV_BASE_URL)
    if base_url:
        locations.append(f"{ENV_BASE_URL}: {base_url}")
    else:
        locations.append(f"Endpoint: {DEFAULT_BASE_URL} (default)")

    return locations
