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
from ..base import ProviderCapabilities  # noqa: E402

PROVIDER_CAPABILITIES = ProviderCapabilities(
    user_message_images=True,
    tool_result_images=True,
    pdf_input=True,
    tool_choice_forwarding=False,
    thinking=False,
    prompt_caching=True,
    streaming=True,
    cancellation=True,
)
