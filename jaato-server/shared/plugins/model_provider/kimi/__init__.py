"""Moonshot AI Kimi model provider plugin.

Access to Moonshot AI's Kimi models (``kimi-k3``, ``kimi-k2.7-code``,
``kimi-k2.6``) through the hosted OpenAI-compatible API at
https://api.moonshot.ai/v1 (China: https://api.moonshot.cn/v1; the Kimi
Code plan at https://api.kimi.com/coding/v1 with its own model ids).
``GET /v1/models`` reports each model's context window and input
modalities, so both are detected at connect time.

The provider opts into **reasoning replay** (K3 requires the assistant
message back as-is, ``reasoning_content`` included), maps the framework
thinking knobs onto K3's ``reasoning_effort`` and K2.x's ``thinking``
object, forwards only the parameters Kimi's request schema names
(sampling parameters are fixed per model and a 400 when sent), stamps
``strict: false`` on tool definitions, and stops the retry loop on the
non-transient quota-exhausted 429.

Authentication (API key, Bearer token):
- Set ``JAATO_KIMI_API_KEY`` (or the vendor's ``MOONSHOT_API_KEY``), or
  store credentials via ``kimi-auth``.  Keys are generated at
  https://platform.kimi.ai/console/api-keys; K3 is unlocked after a
  first top-up.
"""

from .provider import KimiProvider, create_provider

__all__ = ["KimiProvider", "create_provider"]


# --- Provider capability contract (see docs/model-provider-capabilities.md) ---
from ..base import (  # noqa: E402
    ProviderCapabilities, ProviderKnobs, KnobLayer, KnobSpec, AuthSource,
)

PROVIDER_CAPABILITIES = ProviderCapabilities(
    user_message_images=True,     # every current model; catalog supports_image_in
    tool_result_images=True,
    pdf_input=False,
    audio_input=False,
    tool_choice_forwarding=True,  # full set on K3; auto/none on K2.x
    thinking=True,
    prompt_caching=False,         # automatic on the vendor side; nothing emitted
    streaming=True,
    cancellation=True,
    output_media=True,            # shares _openai_compat's wired streaming loop.
    reasoning_replay=True,        # K3: the assistant message goes back as-is
)

# --- Provider config-knob contract (authored from provider.py read sites) ---
PROVIDER_KNOBS = ProviderKnobs(layers=(
    KnobLayer("top_level", (
        KnobSpec("api_key", "str", None, "Kimi API key"),
        KnobSpec("base_url", "str", None,
                 "JAATO_KIMI_BASE_URL override (.cn platform, Kimi Code plan, a proxy)"),
        KnobSpec("context_length", "int", None,
                 "manual override when the catalog lacks the model"),
        KnobSpec("modalities", "list", None,
                 "assert input modalities for a model the catalog does not classify"),
        KnobSpec("extra_body", "dict", None,
                 "opaque passthrough to OpenAI create() extra_body"),
        KnobSpec("output_modalities", "list", None,
                 "assert what the model can EMIT; no catalog reports output "
                 "modalities, so without this the floor is text and the "
                 "startup check refuses an outbound tier role"),
    ), description="connection / identity"),
    KnobLayer("api_params", (
        KnobSpec("max_tokens", "int", None,
                 "sent as max_completion_tokens; >= 16k advised for thinking models"),
        KnobSpec("tool_choice", "str", None,
                 "auto|none|required|named on K3; auto|none on K2.x (else folded to auto)"),
        KnobSpec("stop", "list", None, "up to 5 strings of up to 32 bytes"),
        KnobSpec("response_format", "dict", None,
                 "{type: text|json_object|json_schema}"),
        KnobSpec("prompt_cache_key", "str", None,
                 "stable per-task key that raises the automatic cache hit rate "
                 "(required on the Kimi Code plan)"),
        KnobSpec("thinking_level", "str", None,
                 "K3 → reasoning_effort: low|high|max (vendor default max)"),
        KnobSpec("enable_thinking", "bool", None,
                 "K2.6 → thinking.type; ignored on K3 (always on) and K2.7-code (forced on)"),
        KnobSpec("thinking_keep", "str", None,
                 "K2.6 → thinking.keep: \"all\" replays historical reasoning "
                 "server-side; unset = the vendor default (null)"),
        KnobSpec("strict_tools", "bool", None,
                 "stamp strict: true on tool definitions (default false — Kimi "
                 "defaults strict on, and the framework's schemas are not "
                 "authored for strict mode)"),
        KnobSpec("modalities", "list", None,
                 "OUTPUT selector [\"text\",\"audio\"] — OpenAI's field, the "
                 "opposite direction from the tier key of the same name"),
        KnobSpec("audio", "dict", None,
                 "voice/format companion of api_params.modalities"),
    ), description="Kimi Chat Completions params (filtered allow-list; no "
                   "sampling parameters — they are fixed per model)"),
))
PROVIDER_QUIRKS = frozenset({
    # Opt-in prose-emulated tool calling for upstream models that cannot
    # emit native tool calls.  See shared/plugins/model_provider/_prose_tools.py.
    "prose_tool_calls",
})

# --- Provider credential-resolution contract (from verify_auth/resolve_*) ---
PROVIDER_AUTH_RESOLUTION = (
    AuthSource("api_key_param", "api_key",
               "plugin_configs.kimi.api_key (pass:// ok)"),
    AuthSource("env", "JAATO_KIMI_API_KEY"),
    AuthSource("env", "MOONSHOT_API_KEY", "vendor SDK var"),
    AuthSource("stored", "kimi-auth",
               "kimi_auth.json (config_root → workspace → ~/.jaato)"),
)
