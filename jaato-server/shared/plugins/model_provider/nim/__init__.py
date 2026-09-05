"""NVIDIA NIM model provider plugin.

This provider enables access to AI models through NVIDIA NIM (Inference
Microservices), supporting both NVIDIA's hosted API and self-hosted
NIM containers.

Authentication:
- Hosted API: Set JAATO_NIM_API_KEY with an nvapi-... key
- Self-hosted: Set JAATO_NIM_BASE_URL to your NIM container endpoint
"""

from .provider import NIMProvider, create_provider

__all__ = ["NIMProvider", "create_provider"]


# --- Provider capability contract (see docs/model-provider-capabilities.md) ---
from ..base import (  # noqa: E402
    ProviderCapabilities, ProviderKnobs, KnobLayer, KnobSpec, AuthSource,
)

PROVIDER_CAPABILITIES = ProviderCapabilities(
    user_message_images=True,
    tool_result_images=True,
    pdf_input=False,
    tool_choice_forwarding=False,
    thinking=True,
    prompt_caching=False,
    streaming=True,
    cancellation=True,
    output_media=False,
)

# --- Provider config-knob contract (authored from provider.py read sites) ---
PROVIDER_KNOBS = ProviderKnobs(layers=(
    KnobLayer("top_level", (
        KnobSpec("api_key", "str", None, "NIM API key (nvapi-...)"),
        KnobSpec("base_url", "str", None,
                 "JAATO_NIM_BASE_URL (self-hosted endpoint)"),
        KnobSpec("context_length", "int"),
        KnobSpec("output_modalities", "list", None,
                 "assert what the model can EMIT; no catalog reports output "
                 "modalities, so without this the floor is text and the "
                 "startup check refuses an outbound tier role"),
    ), description="connection / identity"),
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
    AuthSource("api_key_param", "api_key", "plugin_configs.nim.api_key"),
    AuthSource("env", "JAATO_NIM_API_KEY"),
    AuthSource("stored", "nim-auth",
               "nim_auth.json (config_root → workspace → ~/.jaato); "
               "self-hosted endpoints may need none"),
)
