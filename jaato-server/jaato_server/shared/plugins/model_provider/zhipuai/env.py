"""Environment variable resolution for Zhipu AI provider.

Zhipu AI (Z.AI) is a Chinese AI company offering the GLM family of models
(GLM-5, GLM-4.7, GLM-4.6, GLM-4.5, etc.).  This provider uses the
Anthropic-compatible API for GLM Coding Plan users.
"""

import os
from typing import Optional


# Default Z.AI Anthropic-compatible API endpoint
# NOTE: The Anthropic SDK appends /v1/messages to this base URL internally.
# Do NOT include /v1 here or requests will hit /v1/v1/messages (404).
DEFAULT_ZHIPUAI_BASE_URL = "https://api.z.ai/api/anthropic"

# Default model for Zhipu AI
DEFAULT_ZHIPUAI_MODEL = "glm-4.7"


def resolve_api_key() -> Optional[str]:
    """Resolve Zhipu AI API key from environment.

    Resolution order:

    1. ``JAATO_ZHIPUAI_API_KEY`` — jaato-namespaced (recommended).
       Mirrors the convention used by other jaato providers
       (``JAATO_OPENROUTER_API_KEY``, ``JAATO_NIM_API_KEY``, etc.).
       Workspace ``.env`` values flow through ``_resolve_secret_uri``
       daemon-side, so ``JAATO_ZHIPUAI_API_KEY=pass://path/to/secret``
       resolves before reaching the runner via the session_env
       channel.
    2. ``ZHIPUAI_API_KEY`` — upstream SDK convention, kept for
       backward compat with existing ``.env`` files / direct shell
       exports.

    Returns:
        API key if set, None otherwise.  First non-empty value wins.
    """
    return (
        os.environ.get("JAATO_ZHIPUAI_API_KEY")  # env: Zhipu AI API key (jaato namespace, highest priority)
        or os.environ.get("ZHIPUAI_API_KEY")  # env: the vendor's own API key var; honored when JAATO_ZHIPUAI_API_KEY is unset
    )


def resolve_base_url() -> str:
    """Resolve Zhipu AI API base URL.

    Checks:
    1. ZHIPUAI_BASE_URL environment variable

    Returns:
        Base URL (default: https://api.z.ai/api/anthropic).
    """
    return os.environ.get("ZHIPUAI_BASE_URL", DEFAULT_ZHIPUAI_BASE_URL)  # env: endpoint for Zhipu's Anthropic-compatible API (default https://api.z.ai/api/anthropic)


def resolve_model() -> Optional[str]:
    """Resolve default model name from environment.

    Checks:
    1. ZHIPUAI_MODEL environment variable

    Returns:
        Model name if set, None otherwise.
    """
    return os.environ.get("ZHIPUAI_MODEL")  # env: default model name


def resolve_context_length() -> Optional[int]:
    """Resolve custom context length override.

    Checks:
    1. ZHIPUAI_CONTEXT_LENGTH environment variable

    Returns:
        Context length in tokens if set, None otherwise.
    """
    val = os.environ.get("ZHIPUAI_CONTEXT_LENGTH")  # env: override context window size
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return None


def resolve_enable_thinking() -> bool:
    """Resolve whether extended thinking is enabled.

    Checks:
    1. ZHIPUAI_ENABLE_THINKING environment variable

    Returns:
        True if thinking is enabled, False by default.
    """
    val = os.environ.get("ZHIPUAI_ENABLE_THINKING", "").lower()  # env: enable thinking mode on supported GLM models
    return val in ("1", "true", "yes")


def resolve_thinking_budget() -> int:
    """Resolve thinking budget (max thinking tokens).

    Checks:
    1. ZHIPUAI_THINKING_BUDGET environment variable

    Returns:
        Thinking budget in tokens (default: 10000).
    """
    val = os.environ.get("ZHIPUAI_THINKING_BUDGET")  # env: max thinking tokens when thinking mode is enabled
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return 10000
