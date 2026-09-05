"""Helmcode model provider plugin.

Access to the model catalogue served by Helmcode — private AI inference
for European teams.  Open-weight models (GLM, DeepSeek, Qwen, Gemma) run
on hardware Helmcode operates in the EU with zero prompt retention, at a
flat monthly rate rather than metered per token; the same
OpenAI-compatible API and the same key also reach nine resold frontier
models from Anthropic, OpenAI and Google, which run on those providers'
US infrastructure and are billed per token from prepaid credit.  The
provider bootstraps the active model's context window and input
modalities from the ``GET /v1/models`` catalog at connect time, with
manual override knobs when the catalog doesn't report them.

Authentication (API key, Bearer token):
- Set ``JAATO_HELMCODE_API_KEY`` (jaato namespace) or ``HELMCODE_API_KEY``
  (the vendor's own documented variable), or store credentials via
  ``helmcode-auth``.  Keys are issued per workspace from the Helmcode
  dashboard (API Keys -> Create key) and belong to the organisation
  rather than to an individual.
"""

from .provider import HelmcodeProvider, create_provider

__all__ = ["HelmcodeProvider", "create_provider"]


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
    # Helmcode's own models are served without an explicit prompt-cache
    # surface, and the resold frontier models cache upstream (the console
    # bills Anthropic cache writes) without exposing cache_control
    # breakpoints through this API — so there is nothing for a cache
    # plugin to place.  Usage-side cached-token reporting, where an
    # upstream sends it, is parsed by the shared OpenAI-compat layer
    # regardless of this flag.
    prompt_caching=False,
    streaming=True,
    cancellation=True,
)

# --- Provider config-knob contract (authored from provider.py read sites) ---
PROVIDER_KNOBS = ProviderKnobs(layers=(
    KnobLayer("top_level", (
        KnobSpec("api_key", "str", None, "Helmcode API key"),
        KnobSpec("base_url", "str", None, "JAATO_HELMCODE_BASE_URL override"),
        KnobSpec("context_length", "int"),
        KnobSpec("modalities", "list"),
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
               "plugin_configs.helmcode.api_key (pass:// ok)"),
    AuthSource("env", "JAATO_HELMCODE_API_KEY"),
    AuthSource("env", "HELMCODE_API_KEY", "vendor var"),
    AuthSource("stored", "helmcode-auth",
               "helmcode_auth.json (config_root → workspace → ~/.jaato)"),
)
