"""OVHcloud AI Endpoints model provider plugin.

Access to the model catalog served by OVHcloud AI Endpoints — a hosted,
serverless inference API (OpenAI-compatible) for open models (Llama,
Mistral, Qwen, gpt-oss, DeepSeek, ...) behind a single unified gateway.
The provider bootstraps the active model's context window and input
modalities from the ``GET /v1/models`` catalog at connect time, with
manual override knobs when the catalog doesn't report them.

Authentication (API key, Bearer token):
- Set ``JAATO_OVHCLOUD_API_KEY`` (jaato namespace) or
  ``OVH_AI_ENDPOINTS_ACCESS_TOKEN`` (the vendor's own documented variable),
  or store credentials via ``ovhcloud-auth``.  Keys are created in the
  OVHcloud Manager (Public Cloud → AI & Machine Learning → AI Endpoints →
  API keys).  ``JAATO_OVHCLOUD_ALLOW_ANONYMOUS=true`` opts into the keyless
  rate-limited free tier (evaluation only — never a silent fallback).
"""

from .provider import OVHcloudProvider, create_provider

__all__ = ["OVHcloudProvider", "create_provider"]


# --- Provider capability contract (see docs/model-provider-capabilities.md) ---
from ..base import (  # noqa: E402
    ProviderCapabilities, ProviderKnobs, KnobLayer, KnobSpec, AuthSource,
)

PROVIDER_CAPABILITIES = ProviderCapabilities(
    user_message_images=True,
    tool_result_images=True,
    pdf_input=False,
    tool_choice_forwarding=True,
    thinking=True,
    prompt_caching=False,
    streaming=True,
    cancellation=True,
    output_media=False,
)

# --- Provider config-knob contract (authored from provider.py read sites) ---
PROVIDER_KNOBS = ProviderKnobs(layers=(
    KnobLayer("top_level", (
        KnobSpec("api_key", "str", None, "OVHcloud AI Endpoints API key"),
        KnobSpec("base_url", "str", None, "JAATO_OVHCLOUD_BASE_URL override"),
        KnobSpec("context_length", "int"),
        KnobSpec("modalities", "list"),
        KnobSpec("allow_anonymous", "bool", False,
                 "opt into the keyless rate-limited free tier"),
        KnobSpec("extra_body", "dict", None,
                 "opaque passthrough to OpenAI create() extra_body"),
    ), description="connection / identity"),
    KnobLayer("api_params", (
        KnobSpec("temperature", "float"),
        KnobSpec("top_p", "float"),
        KnobSpec("max_tokens", "int"),
        KnobSpec("tool_choice", "str"),
        KnobSpec("parallel_tool_calls", "bool"),
        KnobSpec("frequency_penalty", "float"),
        KnobSpec("presence_penalty", "float"),
        KnobSpec("seed", "int"),
        KnobSpec("stop", "list"),
    ), description="OpenAI Chat Completions params (filtered allow-list)"),
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
               "plugin_configs.ovhcloud.api_key (pass:// ok)"),
    AuthSource("env", "JAATO_OVHCLOUD_API_KEY"),
    AuthSource("env", "OVH_AI_ENDPOINTS_ACCESS_TOKEN", "vendor var"),
    AuthSource("stored", "ovhcloud-auth",
               "ovhcloud_auth.json (config_root → workspace → ~/.jaato)"),
    AuthSource("none", "",
               "anonymous free tier — explicit opt-in via "
               "JAATO_OVHCLOUD_ALLOW_ANONYMOUS / allow_anonymous knob"),
)
