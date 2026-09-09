"""Xiaomi MiMo model provider plugin.

Access to Xiaomi's MiMo models (``mimo-v2.5-pro``, ``mimo-v2.5``) through
the hosted OpenAI-compatible API at https://api.xiaomimimo.com/v1.  Both
current models have a 1M-token context window and thinking on by default;
``mimo-v2.5`` also accepts images.  The catalog reports no per-model
metadata, so the window comes from a built-in table beneath the
``context_length`` knob.

The provider opts into **reasoning replay** — in thinking mode MiMo
refuses (400) the next request of a tool-call loop unless the assistant
message carries the ``reasoning_content`` it produced — and folds
``tool_choice`` to the one value the vendor accepts (``auto``).

Authentication (API key, Bearer token):
- Set ``JAATO_MIMO_API_KEY`` (or the vendor's ``MIMO_API_KEY``), or store
  credentials via ``mimo-auth``.  Keys are generated at
  https://platform.xiaomimimo.com/#/console/api-keys.
- Not available in the EU, the UK or Korea.
"""

from .provider import MiMoProvider, create_provider

__all__ = ["MiMoProvider", "create_provider"]


# --- Provider capability contract (see docs/model-provider-capabilities.md) ---
from ..base import (  # noqa: E402
    ProviderCapabilities, ProviderKnobs, KnobLayer, KnobSpec, AuthSource,
)

PROVIDER_CAPABILITIES = ProviderCapabilities(
    user_message_images=True,     # mimo-v2.5 (the omnimodal model)
    tool_result_images=True,
    pdf_input=False,
    audio_input=False,            # the model listens; the wire is not probed yet
    tool_choice_forwarding=True,  # forwarded, folded to the vendor's "auto"
    thinking=True,
    prompt_caching=False,         # automatic on the vendor side; nothing emitted
    streaming=True,
    cancellation=True,
    output_media=True,            # shares _openai_compat's wired streaming loop.
    reasoning_replay=True,        # mandatory: 400 without it in thinking mode
)

# --- Provider config-knob contract (authored from provider.py read sites) ---
PROVIDER_KNOBS = ProviderKnobs(layers=(
    KnobLayer("top_level", (
        KnobSpec("api_key", "str", None, "MiMo API key"),
        KnobSpec("base_url", "str", None,
                 "JAATO_MIMO_BASE_URL override (Token Plan hosts, a local proxy)"),
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
        KnobSpec("temperature", "float", None,
                 "forced to 1.0 by the vendor while thinking is enabled"),
        KnobSpec("top_p", "float", None,
                 "forced to 0.95 by the vendor while thinking is enabled"),
        KnobSpec("max_tokens", "int", None,
                 "sent as max_completion_tokens (up to 131072)"),
        KnobSpec("tool_choice", "str", None,
                 "only \"auto\" is accepted; anything else is folded to it with a warning"),
        KnobSpec("parallel_tool_calls", "bool"),
        KnobSpec("stop", "list"),
        KnobSpec("response_format", "dict", None,
                 "{type: json_object} works on both models; no json_schema"),
        KnobSpec("enable_thinking", "bool", None,
                 "→ thinking: {type: enabled|disabled} (vendor default: enabled)"),
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
               "plugin_configs.mimo.api_key (pass:// ok)"),
    AuthSource("env", "JAATO_MIMO_API_KEY"),
    AuthSource("env", "MIMO_API_KEY", "vendor SDK var"),
    AuthSource("stored", "mimo-auth",
               "mimo_auth.json (config_root → workspace → ~/.jaato)"),
)
