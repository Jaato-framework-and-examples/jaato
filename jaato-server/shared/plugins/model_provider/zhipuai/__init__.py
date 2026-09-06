"""Zhipu AI (Z.AI) Model Provider Plugin.

This provider uses Zhipu AI's Anthropic-compatible API to access GLM models,
primarily targeting GLM Coding Plan subscribers.

Available Models:
- GLM-4.7: Latest flagship with native chain-of-thought reasoning (200K context)
- GLM-4.7-Flash/Flashx: Fast inference variants (200K context)
- GLM-4.6: Previous flagship, strong coding (200K context)
- GLM-4.5/Air/Flash: Balanced and lightweight models (128K context)

Requirements:
- Zhipu AI API key from https://open.bigmodel.cn/
- anthropic package (for Anthropic-compatible API)

Configuration:
    Environment variables:
        ZHIPUAI_API_KEY: API key (required)
        ZHIPUAI_BASE_URL: API base URL (default: https://api.z.ai/api/anthropic)
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
    KNOWN_MODELS,
    MODEL_CONTEXT_LIMITS,
    ZHIPUAI_MODELS_URL,
    ZhipuAIAPIKeyNotFoundError,
    ZhipuAIConnectionError,
    ZhipuAIRateLimitError,
    ZhipuAIProvider,
    create_provider,
    fetch_zhipuai_models,
)

__all__ = [
    # Main provider
    "ZhipuAIProvider",
    "create_provider",
    # Errors
    "ZhipuAIAPIKeyNotFoundError",
    "ZhipuAIConnectionError",
    "ZhipuAIRateLimitError",
    # Constants
    "KNOWN_MODELS",
    "MODEL_CONTEXT_LIMITS",
    # Environment
    "resolve_api_key",
    "resolve_base_url",
    "resolve_model",
    "resolve_context_length",
    "DEFAULT_ZHIPUAI_BASE_URL",
    "DEFAULT_ZHIPUAI_MODEL",
    "ZHIPUAI_MODELS_URL",
    "fetch_zhipuai_models",
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


# --- Provider capability contract (see docs/model-provider-capabilities.md) ---
from ..base import (  # noqa: E402
    ProviderCapabilities, ProviderKnobs, KnobLayer, KnobSpec, AuthSource,
)

PROVIDER_CAPABILITIES = ProviderCapabilities(
    user_message_images=True,
    tool_result_images=True,
    pdf_input=False,
    tool_choice_forwarding=False,
    thinking=True,
    prompt_caching=False,
    streaming=True,
    cancellation=True,
    output_media=False,
)

# --- Provider config-knob contract (authored from provider.py read sites) ---
# Anthropic-compatible endpoint (subclasses AnthropicProvider): layered
# _knob api_params + framework_overrides, no routing layer.
PROVIDER_KNOBS = ProviderKnobs(layers=(
    KnobLayer("api_params", (
        KnobSpec("temperature", "float"),
        KnobSpec("top_p", "float"),
        KnobSpec("top_k", "int"),
        KnobSpec("max_tokens", "int"),
        KnobSpec("enable_thinking", "bool"),
        KnobSpec("thinking_budget", "int"),
    ), description="Anthropic Messages API request-body fields"),
    KnobLayer("framework_overrides", (
        KnobSpec("base_url", "str"),
        KnobSpec("context_length", "int"),
    ), description="endpoint / context overrides"),
))
PROVIDER_QUIRKS = frozenset()

# --- Provider credential-resolution contract (from verify_auth/resolve_*) ---
PROVIDER_AUTH_RESOLUTION = (
    AuthSource("api_key_param", "api_key", "plugin_configs.zhipuai.api_key"),
    AuthSource("env", "JAATO_ZHIPUAI_API_KEY"),
    AuthSource("env", "ZHIPUAI_API_KEY", "vendor SDK var"),
    AuthSource("stored", "zhipuai-auth",
               "zhipuai_auth.json (config_root → workspace → ~/.jaato)"),
)
