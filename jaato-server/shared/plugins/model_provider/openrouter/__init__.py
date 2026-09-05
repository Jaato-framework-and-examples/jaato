"""OpenRouter model provider plugin.

OpenRouter (https://openrouter.ai) is a unified gateway that exposes
hundreds of models from many vendors (OpenAI, Anthropic, Google, Meta,
Mistral, DeepSeek, ...) behind a single OpenAI-compatible API.

Authentication:
- Set ``JAATO_OPENROUTER_API_KEY`` (sk-or-...) from
  https://openrouter.ai/settings/keys
- Or run ``openrouter-auth key <api_key>`` to store it persistently.
"""

from .provider import OpenRouterProvider, create_provider

__all__ = ["OpenRouterProvider", "create_provider"]


# --- Provider capability contract (see docs/model-provider-capabilities.md) ---
from ..base import (  # noqa: E402
    ProviderCapabilities, ProviderKnobs, KnobLayer, KnobSpec, AuthSource,
)

PROVIDER_CAPABILITIES = ProviderCapabilities(
    user_message_images=True,
    tool_result_images=True,
    pdf_input=True,
    tool_choice_forwarding=False,
    thinking=True,
    prompt_caching=True,
    streaming=True,
    cancellation=True,
    output_media=False,
)

# --- Provider config-knob contract (authored from provider.py read sites) ---
PROVIDER_KNOBS = ProviderKnobs(layers=(
    KnobLayer("top_level", (
        KnobSpec("api_key", "str", None, "OpenRouter API key (sk-or-...)"),
        KnobSpec("http_referer", "str", None, "HTTP-Referer attribution header"),
        KnobSpec("app_title", "str", None, "X-OpenRouter-Title header"),
        KnobSpec("app_categories", "list", None, "X-OpenRouter-Categories header"),
        KnobSpec("extra_headers", "dict", None,
                 "additional headers (e.g. x-anthropic-beta)"),
    ), description="auth / app-attribution identity"),
    KnobLayer("api_params", (
        KnobSpec("models", "list", None, "cross-model fallback list"),
        KnobSpec("temperature", "float"),
        KnobSpec("top_p", "float"),
        KnobSpec("top_k", "int"),
        KnobSpec("max_tokens", "int"),
        KnobSpec("parallel_tool_calls", "bool"),
        KnobSpec("service_tier", "str", None,
                 "auto|default|flex|priority|scale — OpenAI-style "
                 "processing tier forwarded to tier-supporting upstreams"),
        KnobSpec("strict_tools", "bool", None, "emit strict:true on tool defs"),
        KnobSpec("enable_thinking", "bool"),
        KnobSpec("thinking_budget", "int", None, "→ reasoning.max_tokens"),
        KnobSpec("thinking_level", "str", None, "low|medium|high → reasoning.effort"),
        KnobSpec("cache_prompt", "str", "auto", "auto|true|false"),
        KnobSpec("cache_ttl", "str", "5m", "5m|1h"),
    ), description="OpenAI Chat Completions request-body fields"),
    KnobLayer("routing", opaque=True,
              description="OpenRouter provider-routing extension — any key "
                          "forwarded verbatim (order, sort, only, ignore, "
                          "quantizations, max_price, ...)"),
    KnobLayer("framework_overrides", (
        KnobSpec("context_length", "int"),
        KnobSpec("base_url", "str"),
        KnobSpec("modalities", "list"),
        KnobSpec("connect_timeout", "float", 15.0,
                 "TCP + TLS handshake deadline, seconds"),
        KnobSpec("request_timeout", "float", 600.0,
                 "byte-level deadline (httpx read/write/pool), seconds; "
                 "0 disables"),
        KnobSpec("stream_idle_timeout", "float", 300.0,
                 "payload idle deadline for streaming turns, seconds; "
                 "0 disables.  Raises StallTimeoutError — httpx cannot "
                 "see this failure because OpenRouter's keep-alive SSE "
                 "comments reset the byte clock (#732)"),
    ), description="rare escape hatches"),
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
    AuthSource("api_key_param", "api_key", "plugin_configs.openrouter.api_key"),
    AuthSource("env", "JAATO_OPENROUTER_API_KEY"),
    AuthSource("stored", "openrouter-auth",
               "openrouter_auth.json (workspace → ~/.jaato)"),
)
