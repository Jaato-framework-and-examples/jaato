"""Google GenAI (Vertex AI / Gemini) model provider.

This package provides a ModelProviderPlugin implementation for Google's
GenAI SDK, supporting Vertex AI and Gemini models.

Usage:
    from shared.plugins.model_provider.google_genai import GoogleGenAIProvider

    provider = GoogleGenAIProvider()
    provider.initialize(ProviderConfig(project='my-project', location='us-central1'))
    provider.connect('gemini-2.5-flash')
"""

from .provider import GoogleGenAIProvider, create_provider

__all__ = [
    "GoogleGenAIProvider",
    "create_provider",
]


# --- Provider capability contract (see docs/model-provider-capabilities.md) ---
from ..base import (  # noqa: E402
    ProviderCapabilities, ProviderKnobs, KnobLayer, KnobSpec, AuthSource,
)

PROVIDER_CAPABILITIES = ProviderCapabilities(
    user_message_images=True,
    tool_result_images=True,
    pdf_input=True,
    tool_choice_forwarding=False,
    thinking=False,
    prompt_caching=True,
    streaming=True,
    cancellation=True,
    output_media=False,
)

# --- Provider config-knob contract (authored from provider.py read sites) ---
PROVIDER_KNOBS = ProviderKnobs(layers=(
    KnobLayer("top_level", (
        KnobSpec("context_length", "int"),
        KnobSpec("modalities", "list"),
        # Read flat off ``ProviderConfig.extra`` by
        # ``GoogleGenAICachePlugin.initialize`` — declared here because that
        # is the read site's actual position.  Unlike Anthropic's breakpoint
        # scheme this creates a server-side ``CachedContent`` object bound to
        # the active model, billed on creation and deleted at shutdown.
        KnobSpec("enable_caching", "bool", False,
                 "create a server-side CachedContent for system+tools "
                 "(needs ~32k tokens of stable prefix to engage)"),
        KnobSpec("cache_ttl", "str", "3600s",
                 "CachedContent TTL, Google duration format (e.g. 3600s)"),
    ), description="context / modality overrides + prompt-cache control"),
))
PROVIDER_QUIRKS = frozenset()

# --- Provider credential-resolution contract (from verify_auth/resolve_*) ---
PROVIDER_AUTH_RESOLUTION = (
    AuthSource("api_key_param", "api_key", "plugin_configs.google_genai.api_key"),
    AuthSource("env", "GOOGLE_GENAI_API_KEY", "AI Studio api-key mode"),
    AuthSource("stored", "GOOGLE_APPLICATION_CREDENTIALS", "service-account file"),
    AuthSource("adc", "", "Application Default Credentials (gcloud / GCE metadata)"),
)
