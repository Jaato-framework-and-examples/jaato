"""TypeSafe Model Provider Plugin: Jev, a decision model, on TypeSafe's API.

Jev answers typed questions (``noul`` / ``choice`` / ``score``, and
``box`` on Jev 1.14) about a state, which may include still images, with
calibrated probabilities and writes no text.  ``decide()``
posts to ``POST https://api.typesafe.ai/v1/systemone``; ``complete()``
raises.  Bind it in a profile's ``decision_models:`` block, never as a
tier (docs/design/decision-models.md §6.2, §7).

Profile example::

    decision_models:
      triage:
        provider: typesafe
        model: jev-latest
    plugin_configs:
      typesafe:
        api_key: pass://typesafe/key

The key: ``plugin_configs.typesafe.api_key`` first, then
``JAATO_TYPESAFE_API_KEY``, ``TYPESAFE_API_KEY`` and ``typesafe-auth key``.
"""

from .provider import DEFAULT_MODEL, TypeSafeProvider, create_provider

__all__ = ["TypeSafeProvider", "create_provider", "DEFAULT_MODEL"]


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
    decision_images=True,   # assets + TypeSafe-Danger-Zone-Images (§12)
)

# --- Provider config-knob contract (authored from provider.py read sites) ---
PROVIDER_KNOBS = ProviderKnobs(layers=(
    KnobLayer("top_level", (
        KnobSpec("api_key", "str", None,
                 "TypeSafe API key; read first, before JAATO_TYPESAFE_API_KEY, "
                 "TYPESAFE_API_KEY and typesafe-auth (pass:// ok)"),
        KnobSpec("base_url", "str", "https://api.typesafe.ai",
                 "API root (JAATO_TYPESAFE_BASE_URL); /v1/systemone is under it"),
        KnobSpec("context_length", "int", None,
                 "window the pre-send estimate checks "
                 "(JAATO_TYPESAFE_CONTEXT_LENGTH); unset = not checked"),
    ), description="key, API root and window"),
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
    "The same Jev is reachable through openrouter's decide(); this is "
    "TypeSafe's own API, keyed by a TypeSafe key "
    "(https://console.typesafe.ai/keys).",
    "model: jev-latest is TypeSafe's alias for the current release; "
    "jev-1.13.0, jev-1.14.0 and jev-preview pin one. Ids are passed through.",
    "decide(images=...) sends still PNG/JPEG images (280-1568 px a side, "
    "checked before sending, never resized) and allows box questions. Images "
    "and box are 'Dreamer-ready' at TypeSafe; box needs Jev 1.14 "
    "(jev-1.14.0 / jev-preview).",
)
PROVIDER_CONTEXT_RESOLUTION = (
    "profile knob 'context_length' → env JAATO_TYPESAFE_CONTEXT_LENGTH → "
    "else unknown and not checked (TypeSafe publishes no catalog)"
)

# --- Provider credential-resolution contract (from resolve_key) ---
PROVIDER_AUTH_RESOLUTION = (
    AuthSource("api_key_param", "api_key",
               "plugin_configs.typesafe.api_key (pass:// ok)"),
    AuthSource("env", "JAATO_TYPESAFE_API_KEY"),
    AuthSource("env", "TYPESAFE_API_KEY", "vendor SDK var"),
    AuthSource("stored", "typesafe-auth",
               "typesafe_auth.json (config_root → workspace → ~/.jaato)"),
)
