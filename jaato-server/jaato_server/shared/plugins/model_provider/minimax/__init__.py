"""MiniMax model provider plugin.

Access to MiniMax's models (``MiniMax-M3``, ``MiniMax-M2.7`` and the
``-highspeed`` variants, plus the legacy M2 / M2.1 / M2.5) through the
hosted OpenAI-compatible API at https://api.minimax.io/v1 (China:
https://api.minimax.cn/v1 — keys are region-bound).  M3 has a 1M-token
window, an adaptive thinking toggle and image input; the M2 family is
200K, text-only and always thinking.  The catalog reports no per-model
metadata, so the window comes from a built-in table beneath the
``context_length`` knob.

The provider always requests ``reasoning_split`` (so reasoning does not
arrive as ``<think>`` inside the text), opts into **reasoning replay**
echoing both ``reasoning_content`` and ``reasoning_details``, folds
``tool_choice`` to the vendor's ``none`` / ``auto``, and always sends
``max_completion_tokens`` (the vendor default truncates tool calls).

Authentication (API key, Bearer token):
- Set ``JAATO_MINIMAX_API_KEY`` (or the vendor's ``MINIMAX_API_KEY``), or
  store credentials via ``minimax-auth``.  A Token Plan subscription key
  (``sk-cp-...``) works on ``/v1`` unchanged.
"""

from .provider import MiniMaxProvider, create_provider

__all__ = ["MiniMaxProvider", "create_provider"]


# --- Provider capability contract (see docs/model-provider-capabilities.md) ---
from ..base import (  # noqa: E402
    ProviderCapabilities, ProviderKnobs, KnobLayer, KnobSpec, AuthSource,
)

PROVIDER_CAPABILITIES = ProviderCapabilities(
    user_message_images=True,     # MiniMax-M3
    tool_result_images=True,
    pdf_input=False,
    audio_input=False,
    tool_choice_forwarding=True,  # forwarded, folded to the vendor's none/auto
    thinking=True,
    prompt_caching=False,         # automatic on the vendor side; nothing emitted
    streaming=True,
    cancellation=True,
    output_media=True,            # shares _openai_compat's wired streaming loop.
    reasoning_replay=True,        # reasoning_content + reasoning_details echoed
)

# --- Provider config-knob contract (authored from provider.py read sites) ---
PROVIDER_KNOBS = ProviderKnobs(layers=(
    KnobLayer("top_level", (
        KnobSpec("api_key", "str", None, "MiniMax API key or Token Plan key"),
        KnobSpec("base_url", "str", None,
                 "JAATO_MINIMAX_BASE_URL override (the .cn platform, a proxy)"),
        KnobSpec("context_length", "int", None,
                 "override the built-in per-model window"),
        KnobSpec("modalities", "list", None,
                 "assert input modalities for a model the table does not name"),
        KnobSpec("extra_body", "dict", None,
                 "opaque passthrough to OpenAI create() extra_body"),
        KnobSpec("output_modalities", "list", None,
                 "assert what the model can EMIT; no catalog reports output "
                 "modalities, so without this the floor is text and the "
                 "startup check refuses an outbound tier role"),
    ), description="connection / identity"),
    KnobLayer("api_params", (
        KnobSpec("temperature", "float", None, "0-2"),
        KnobSpec("top_p", "float"),
        KnobSpec("max_tokens", "int", None,
                 "sent as max_completion_tokens; default = the model's recommended cap"),
        KnobSpec("tool_choice", "str", None,
                 "none|auto; anything else is folded to auto with a warning"),
        KnobSpec("parallel_tool_calls", "bool"),
        KnobSpec("stop", "list"),
        KnobSpec("service_tier", "str", None, "\"priority\" = 1.5x price, faster"),
        KnobSpec("enable_thinking", "bool", None,
                 "M3 → thinking: {type: adaptive|disabled}; ignored on M2.x (always on)"),
        KnobSpec("modalities", "list", None,
                 "OUTPUT selector [\"text\",\"audio\"] — OpenAI's field, the "
                 "opposite direction from the tier key of the same name"),
        KnobSpec("audio", "dict", None,
                 "voice/format companion of api_params.modalities"),
    ), description="OpenAI Chat Completions params (filtered allow-list)"),
))
PROVIDER_QUIRKS = frozenset({
    # Opt-in prose-emulated tool calling for upstream models that cannot
    # emit native tool calls.  See shared/plugins/model_provider/_prose_tools.py.
    "prose_tool_calls",
})

# --- Provider credential-resolution contract (from verify_auth/resolve_*) ---
PROVIDER_AUTH_RESOLUTION = (
    AuthSource("api_key_param", "api_key",
               "plugin_configs.minimax.api_key (pass:// ok)"),
    AuthSource("env", "JAATO_MINIMAX_API_KEY"),
    AuthSource("env", "MINIMAX_API_KEY", "vendor SDK var"),
    AuthSource("stored", "minimax-auth",
               "minimax_auth.json (config_root → workspace → ~/.jaato)"),
)
