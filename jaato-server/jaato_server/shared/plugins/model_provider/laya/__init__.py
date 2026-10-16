"""Laya Model Provider Plugin: a decision model, served by ``decide()``.

Laya (ConvAI Innovations, Apache 2.0; ``convaiinnovations/laya`` on
Hugging Face) answers typed questions (``noul`` / ``choice`` /
``score``) about a state with probabilities, in one encoder forward
pass.  It writes no text: ``complete()`` raises, and the model is bound
in a profile's ``decision_models:`` block, never as a tier
(docs/design/decision-models.md §7).

Two modes (``plugin_configs.laya.mode``):

- ``http`` (default): ``POST {host}/v1/systemone`` on a ``laya-serve``
  process (``pip install "laya[serve]"; laya-serve``), the same body as
  TypeSafe Jev.
- ``local``: the ``laya`` package in the runner process.  The pool slot
  is then retired at ``session.end`` (#1565).

Profile example::

    decision_models:
      local:
        provider: laya
        model: convaiinnovations/laya
    plugin_configs:
      laya:
        host: http://gpu-box:8000      # or JAATO_LAYA_HOST

Environment: ``JAATO_LAYA_HOST``, ``JAATO_LAYA_API_KEY`` (then
``LAYA_API_KEY``), ``JAATO_LAYA_MODE``, ``JAATO_LAYA_CONTEXT_LENGTH``.
"""

from .provider import DEFAULT_MODEL, LayaProvider, create_provider

__all__ = ["LayaProvider", "create_provider", "DEFAULT_MODEL"]


# --- Provider capability contract (see docs/model-provider-capabilities.md) ---
from ..base import (  # noqa: E402
    AuthSource, KnobLayer, KnobSpec, ProviderCapabilities, ProviderKnobs,
)

PROVIDER_CAPABILITIES = ProviderCapabilities(
    chat=False,             # a decision model: complete() refuses every turn
    user_message_images=False,
    tool_result_images=False,
    pdf_input=False,
    audio_input=False,
    tool_choice_forwarding=False,
    thinking=False,
    prompt_caching=False,
    streaming=False,
    cancellation=True,
    output_media=False,
    reasoning_replay=False,
    decisions=True,
    decision_images=False,   # laya answers about text only
)

# --- Provider config-knob contract (authored from provider.py read sites) ---
PROVIDER_KNOBS = ProviderKnobs(layers=(
    KnobLayer("top_level", (
        KnobSpec("mode", "str", "http",
                 "http (laya-serve) | local (the laya package in the runner; "
                 "the pool slot is retired)"),
        KnobSpec("host", "str", "http://localhost:8000",
                 "laya-serve URL (JAATO_LAYA_HOST)"),
        KnobSpec("api_key", "str", None,
                 "bearer token, when laya-serve was started with LAYA_API_KEY"),
        KnobSpec("subfolder", "str", None,
                 "checkpoint subfolder for mode local: multilingual | "
                 "typed-decisions"),
        KnobSpec("context_length", "int", None,
                 "window override; the published checkpoints are known "
                 "(512 English, 1024 multilingual / typed-decisions)"),
    ), description="server, mode and checkpoint"),
    KnobLayer("framework_overrides", (
        KnobSpec("connect_timeout", "float", 10.0,
                 "TCP connect deadline, seconds; 0 disables"),
        KnobSpec("request_timeout", "float", 60.0,
                 "request deadline, seconds; 0 disables"),
    ), description="deadlines"),
))
PROVIDER_QUIRKS = frozenset()

# --- What explain says about this provider beyond the contract fields ---
PROVIDER_NOTES = (
    "A decision model: it answers typed questions through decide() and "
    "writes no text. Bind it in decision_models:, never as model: or a tier.",
    "mode: http (default) needs a running laya-serve "
    "(pip install \"laya[serve]\"; laya-serve). mode: local loads the laya "
    "package and torch into the runner, which retires that pool slot.",
    "model: convaiinnovations/laya lets laya-serve's router pick the "
    "English or multilingual checkpoint; convaiinnovations/laya-multilingual "
    "and -typed-decisions pin one.",
)
PROVIDER_CONTEXT_RESOLUTION = (
    "profile knob 'context_length' → env JAATO_LAYA_CONTEXT_LENGTH → the "
    "published checkpoints (512 English, 1024 multilingual / typed-decisions) "
    "→ else unknown and not checked; an answer Laya reports as truncated is "
    "refused either way"
)

# --- Provider credential-resolution contract (from verify_auth/resolve_*) ---
PROVIDER_AUTH_RESOLUTION = (
    AuthSource("api_key_param", "api_key",
               "plugin_configs.laya.api_key (optional)"),
    AuthSource("env", "JAATO_LAYA_API_KEY", "optional"),
    AuthSource("env", "LAYA_API_KEY", "laya-serve's own variable (optional)"),
)
