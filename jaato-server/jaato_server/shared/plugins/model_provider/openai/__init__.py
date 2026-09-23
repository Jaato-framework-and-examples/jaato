"""Native OpenAI model provider plugin.

First-party access to ``api.openai.com`` — as opposed to reaching OpenAI's
models through OpenRouter or an OpenAI-*compatible* gateway.  Two wires,
one plugin, selected by ``plugin_configs.openai.api``:

- ``chat`` (default) — Chat Completions, on the shared ``_openai_compat``
  transport, with PDF (``file``) and audio (``input_audio``) content
  blocks enabled because OpenAI's own endpoint carries them.
- ``responses`` — the Responses API: a flat ``input`` item list, typed SSE
  events, reasoning summaries, and ``input_tokens``/``output_tokens``
  usage.  This is the surface OpenAI ships to first.

**The context window must be configured.**  OpenAI's ``GET /v1/models``
reports no capacity for any model, so ``plugin_configs.openai.context_length``
(or ``JAATO_OPENAI_CONTEXT_LENGTH``) is required — ``connect()`` fails
loud without one rather than guessing, per the project's no-fallback rule.

Authentication (API key, Bearer token):
- Set ``JAATO_OPENAI_API_KEY`` (the vendor's own ``OPENAI_API_KEY`` is
  honored too), put the key in ``plugin_configs.openai.api_key`` as a
  ``pass://`` URI, or write ``~/.jaato/openai_auth.json``.  Keys are
  created at https://platform.openai.com/api-keys.
"""

from .provider import OpenAIProvider, create_provider

__all__ = ["OpenAIProvider", "create_provider"]


# --- Provider capability contract (see docs/model-provider-capabilities.md) ---
from ..base import (  # noqa: E402
    ProviderCapabilities, ProviderKnobs, KnobLayer, KnobSpec, AuthSource,
)

PROVIDER_CAPABILITIES = ProviderCapabilities(
    user_message_images=True,
    tool_result_images=True,
    # OpenAI's own wire carries what the gateways sharing the transport
    # do not: ``file`` blocks for PDFs and ``input_audio`` for audio.
    pdf_input=True,
    audio_input=True,
    tool_choice_forwarding=True,
    # Reasoning summaries, on the Responses wire.  ``supports_thinking()``
    # is False on the chat wire, where the reasoning models think, are
    # billed for it, and return none of it.
    thinking=True,
    # NOT declared, and the reason is the column's meaning rather than
    # the vendor's behaviour.  OpenAI does cache long prefixes -- both
    # wires read the resulting ``cached_tokens`` back out of the prompt
    # total -- but it does so UNCONDITIONALLY, with no client annotation
    # and no TTL to choose.  ``prompt_caching=True`` in this tree means
    # "the common ``cache:`` profile field has something to deliver here"
    # (``test_cache_profile_field`` derives its expectation from this
    # flag), and for an automatic cache it has nothing: accepting
    # ``cache: {ttl: 1h}`` and dropping it is the silently-inert knob
    # that whole area exists to have stopped.
    prompt_caching=False,
    streaming=True,
    cancellation=True,
    output_media=True,
    # OpenAI's reasoning models are not asked to hand their thinking back:
    # the Responses API keeps that continuity server-side (an `include`
    # of `reasoning.encrypted_content` is its mechanism, opaque to us),
    # and Chat Completions returns no reasoning text at all, so there is
    # nothing to replay on either wire.
    reasoning_replay=False,
)

# --- Provider config-knob contract (authored from provider.py read sites) ---
PROVIDER_KNOBS = ProviderKnobs(layers=(
    KnobLayer("top_level", (
        KnobSpec("api", "str", "chat",
                 "which wire to speak: \"chat\" (Chat Completions) or "
                 "\"responses\" (the Responses API); JAATO_OPENAI_API is "
                 "the env fallback"),
        KnobSpec("api_key", "str", None, "OpenAI API key"),
        KnobSpec("base_url", "str", None, "JAATO_OPENAI_BASE_URL override"),
        KnobSpec("organization", "str", None,
                 "OpenAI-Organization header (billing attribution)"),
        KnobSpec("project", "str", None,
                 "OpenAI-Project header; a project-scoped key needs it"),
        KnobSpec("context_length", "int", None,
                 "REQUIRED — OpenAI's catalog reports no context window, "
                 "so connect() fails loud without this or "
                 "JAATO_OPENAI_CONTEXT_LENGTH"),
        KnobSpec("modalities", "list", None,
                 "assert/correct INPUT modalities for a model the built-in "
                 "table has not caught up with"),
        KnobSpec("extra_body", "dict", None,
                 "opaque passthrough to OpenAI create() extra_body"),
        KnobSpec("output_modalities", "list", None,
                 "assert what the model can EMIT; no catalog reports output "
                 "modalities, so without this the floor is text and the "
                 "startup check refuses an outbound tier role"),
    ), description="connection / identity / wire selection"),
    KnobLayer("api_params", (
        KnobSpec("temperature", "float"),
        KnobSpec("top_p", "float"),
        KnobSpec("max_tokens", "int", None, "chat wire only"),
        KnobSpec("max_output_tokens", "int", None, "responses wire only"),
        KnobSpec("tool_choice", "str"),
        KnobSpec("parallel_tool_calls", "bool"),
        KnobSpec("frequency_penalty", "float", None, "chat wire only"),
        KnobSpec("presence_penalty", "float", None, "chat wire only"),
        KnobSpec("seed", "int", None, "chat wire only"),
        KnobSpec("stop", "list", None, "chat wire only"),
        KnobSpec("reasoning", "dict", None,
                 "responses wire only: {effort, summary}"),
        KnobSpec("text", "dict", None,
                 "responses wire only: {format} — structured output"),
        KnobSpec("store", "bool", False,
                 "responses wire only; jaato owns the history, so server-"
                 "side threading is off unless deliberately enabled"),
        KnobSpec("include", "list", None,
                 "responses wire only, e.g. "
                 "[\"reasoning.encrypted_content\"]"),
        KnobSpec("truncation", "str", None, "responses wire only"),
        KnobSpec("metadata", "dict", None, "responses wire only"),
        KnobSpec("prompt_cache_key", "str", None, "responses wire only"),
        KnobSpec("safety_identifier", "str", None, "responses wire only"),
        KnobSpec("service_tier", "str", None,
                 "auto|default|flex|priority — processing tier"),
        KnobSpec("modalities", "list", None,
                 "OUTPUT selector [\"text\",\"audio\"] — OpenAI's field, the "
                 "opposite direction from the tier key of the same name"),
        KnobSpec("audio", "dict", None,
                 "voice/format companion of api_params.modalities"),
    ), description=(
        "request-body fields; the allow-list applied depends on the wire "
        "(a chat-only key on the responses wire is dropped with a warning, "
        "and vice versa)"
    )),
))
PROVIDER_QUIRKS = frozenset({
    # Inherited from the shared OpenAI-compat transport.  Meaningless on
    # OpenAI's own models — they all emit native tool calls — but the
    # ``base_url`` knob makes this provider usable against a local proxy
    # fronting something that does not, and the quirk is what that costs.
    "prose_tool_calls",
})

# --- Provider credential-resolution contract (from verify_auth/resolve_*) ---
PROVIDER_AUTH_RESOLUTION = (
    AuthSource("api_key_param", "api_key",
               "plugin_configs.openai.api_key (pass:// ok)"),
    AuthSource("env", "JAATO_OPENAI_API_KEY"),
    AuthSource("env", "OPENAI_API_KEY", "the vendor's own variable"),
    AuthSource("stored", "openai_auth.json",
               "config_root → workspace/.jaato → ~/.jaato; {\"api_key\": ...}"),
)
