"""Zhipu AI (Z.AI) Model Provider Plugin — OpenAI-compatible API.

This provider uses Z.AI's OpenAI-compatible API endpoint (``/api/paas/v4``)
to access GLM models.  Unlike the ``zhipuai`` provider (which uses the
Anthropic-compatible endpoint), this variant supports prompt caching.

The same Z.AI API key works for both providers.  Users who already have
credentials configured via ``zhipuai-auth`` can use this provider without
any additional setup — just switch the provider name.

Available Models:
- GLM-5: Flagship MoE model, agentic engineering (200K context)
- GLM-4.7: Flagship with native chain-of-thought reasoning (200K context)
- GLM-4.7-Flash/Flashx: Fast inference variants (200K context)
- GLM-4.6: Previous flagship, strong coding (200K context)
- GLM-4.5/Air/Flash: Balanced and lightweight models (128K context)

Requirements:
- Zhipu AI API key from https://z.ai/model-api or https://open.bigmodel.cn/
- openai package (for OpenAI-compatible API)

Configuration:
    Environment variables:
        ZHIPUAI_API_KEY: API key (shared with zhipuai provider)
        ZHIPUAI_OPENAI_BASE_URL: API base URL
            (default: https://api.z.ai/api/paas/v4)
        ZHIPUAI_OPENAI_MODEL: Default model name
        ZHIPUAI_OPENAI_CONTEXT_LENGTH: Override context length

    ProviderConfig.extra:
        base_url: Override ZHIPUAI_OPENAI_BASE_URL
        context_length: Override context length
        enable_thinking: Enable extended thinking (default: False)
        thinking_budget: Max thinking tokens (default: 10000)

Example:
    from shared.plugins.model_provider.zhipuai_openai import ZhipuAIOpenAIProvider

    provider = ZhipuAIOpenAIProvider()
    provider.initialize(ProviderConfig(api_key="your-key"))
    provider.connect('glm-5')
"""

from .provider import (
    ZhipuAIOpenAIProvider,
    create_provider,
    KNOWN_MODELS,
    MODEL_CONTEXT_LIMITS,
)

__all__ = [
    "ZhipuAIOpenAIProvider",
    "create_provider",
    "KNOWN_MODELS",
    "MODEL_CONTEXT_LIMITS",
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
PROVIDER_KNOBS = ProviderKnobs(layers=(
    KnobLayer("top_level", (
        KnobSpec("base_url", "str", None, "ZHIPUAI_OPENAI_BASE_URL override"),
        KnobSpec("context_length", "int"),
        KnobSpec("enable_thinking", "bool"),
        KnobSpec("thinking_budget", "int"),
        KnobSpec("output_modalities", "list", None,
                 "assert what the model can EMIT; no catalog reports output "
                 "modalities, so without this the floor is text and the "
                 "startup check refuses an outbound tier role"),
    ), description="connection / generation"),
))
PROVIDER_QUIRKS = frozenset({
    # Opt-in prose-emulated tool calling for upstream models that cannot
    # emit native tool calls (schemas prompt-injected with hashed wire
    # ids; fenced tool_call blocks parsed from the response text).  See
    # shared/plugins/model_provider/_prose_tools.py.
    "prose_tool_calls",
})

# --- Provider credential-resolution contract (from verify_auth/resolve_*) ---
PROVIDER_AUTH_RESOLUTION = (
    AuthSource("api_key_param", "api_key",
               "plugin_configs.zhipuai_openai.api_key"),
    AuthSource("env", "ZHIPUAI_API_KEY", "shared with zhipuai provider"),
    AuthSource("stored", "zhipuai-auth", "shared zhipuai_auth.json"),
)
