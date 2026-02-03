"""Zhipu AI (Z.AI) Model Provider Plugin.

This provider uses Zhipu AI's Anthropic-compatible API to access GLM models,
primarily targeting GLM Coding Plan subscribers.

Available Models:
- GLM-4.7: Latest model with native chain-of-thought reasoning (128K context)
- GLM-4.7-Flash: Fast inference variant
- GLM-4: General purpose model
- GLM-4V: Vision-enabled multimodal model
- GLM-4-Assistant: Optimized for agentic tasks

Requirements:
- Zhipu AI API key from https://open.bigmodel.cn/
- anthropic package (for Anthropic-compatible API)

Configuration:
    Environment variables:
        ZHIPUAI_API_KEY: API key (required)
        ZHIPUAI_BASE_URL: API base URL (default: https://api.z.ai/api/anthropic/v1)
        ZHIPUAI_MODEL: Default model name
        ZHIPUAI_CONTEXT_LENGTH: Override context length

    ProviderConfig.extra:
        base_url: Override ZHIPUAI_BASE_URL
        context_length: Override context length

Example:
    from shared.plugins.model_provider.zhipuai import ZhipuAIProvider

    provider = ZhipuAIProvider()
    provider.initialize(ProviderConfig(api_key="your-key"))
    provider.connect('glm-4.7')
    response = provider.send_message("Hello!")

    # Or with environment variable:
    # export ZHIPUAI_API_KEY=your-key
    provider = ZhipuAIProvider()
    provider.initialize()
    provider.connect('glm-4.7')
"""

from .auth import (
    ZhipuAICredentials,
    clear_credentials,
    get_stored_api_key,
    get_stored_base_url,
    load_credentials,
    login_interactive,
    login_with_key,
    logout,
    save_credentials,
    status as auth_status,
    validate_api_key,
)
from .env import (
    DEFAULT_ZHIPUAI_BASE_URL,
    DEFAULT_ZHIPUAI_MODEL,
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
    resolve_model,
)
from .provider import (
    DEFAULT_CONTEXT_LIMIT,
    KNOWN_MODELS,
    ZhipuAIAPIKeyNotFoundError,
    ZhipuAIConnectionError,
    ZhipuAIProvider,
    create_provider,
)

__all__ = [
    # Main provider
    "ZhipuAIProvider",
    "create_provider",
    # Errors
    "ZhipuAIAPIKeyNotFoundError",
    "ZhipuAIConnectionError",
    # Constants
    "DEFAULT_CONTEXT_LIMIT",
    "KNOWN_MODELS",
    # Environment
    "resolve_api_key",
    "resolve_base_url",
    "resolve_model",
    "resolve_context_length",
    "DEFAULT_ZHIPUAI_BASE_URL",
    "DEFAULT_ZHIPUAI_MODEL",
    # Authentication
    "ZhipuAICredentials",
    "login_interactive",
    "login_with_key",
    "logout",
    "auth_status",
    "validate_api_key",
    "save_credentials",
    "load_credentials",
    "clear_credentials",
    "get_stored_api_key",
    "get_stored_base_url",
]
