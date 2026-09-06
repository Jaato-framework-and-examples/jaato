"""Doubleword model provider plugin.

Access to the model catalog served by Doubleword (https://doubleword.ai) —
a hosted, serverless inference API (OpenAI-compatible) for open models
(DeepSeek, Qwen, GLM, Kimi, gpt-oss, Nemotron, ...) that prices by
delivery window: the same chat endpoint serves realtime requests and, via
the ``service_tier: "flex"`` body field, the discounted async tier
(queued work guaranteed to start within ~1 minute).  The provider looks
the active model's context window and input modalities up in the
``GET /v1/models`` catalog at connect time, falling through to manual
override knobs.  **In practice the knobs are required**: Doubleword's
catalog serves bare OpenAI-shaped entries with no context-length or
modality field (verified live 2026-07-19), so set
``plugin_configs.doubleword.context_length`` or
``JAATO_DOUBLEWORD_CONTEXT_LENGTH`` — ``connect()`` fails loud without
one rather than guessing.

Authentication (API key, Bearer token):
- Set ``JAATO_DOUBLEWORD_API_KEY``, or store credentials via
  ``doubleword-auth``.  Keys are generated at
  https://app.doubleword.ai/api-keys.
"""

from .provider import DoublewordProvider, create_provider

__all__ = ["DoublewordProvider", "create_provider"]


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
    output_media=True,   # shares _openai_compat's wired streaming loop.
)

# --- Provider config-knob contract (authored from provider.py read sites) ---
PROVIDER_KNOBS = ProviderKnobs(layers=(
    KnobLayer("top_level", (
        KnobSpec("api_key", "str", None, "Doubleword API key"),
        KnobSpec("base_url", "str", None, "JAATO_DOUBLEWORD_BASE_URL override"),
        KnobSpec("context_length", "int"),
        KnobSpec("modalities", "list"),
        KnobSpec("extra_body", "dict", None,
                 "opaque passthrough to OpenAI create() extra_body"),
        KnobSpec("output_modalities", "list", None,
                 "assert what the model can EMIT; no catalog reports output "
                 "modalities, so without this the floor is text and the "
                 "startup check refuses an outbound tier role"),
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
        KnobSpec("service_tier", "str", None,
                 "inference tier: \"flex\" = discounted async (minutes-level "
                 "latency, ~1 min to first token), \"priority\" = realtime; "
                 "JAATO_DOUBLEWORD_SERVICE_TIER is the env fallback"),
        KnobSpec("modalities", "list", None,
                 "OUTPUT selector [\"text\",\"audio\"] — OpenAI's field, the "
                 "opposite direction from the tier key of the same name"),
        KnobSpec("audio", "dict", None,
                 "voice/format companion of api_params.modalities"),
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
               "plugin_configs.doubleword.api_key (pass:// ok)"),
    AuthSource("env", "JAATO_DOUBLEWORD_API_KEY"),
    AuthSource("stored", "doubleword-auth",
               "doubleword_auth.json (config_root → workspace → ~/.jaato)"),
)
