"""Nebius Token Factory model provider plugin.

Access to the open-model catalog served by Nebius Token Factory's
serverless inference API (OpenAI-compatible). The provider bootstraps the
active model's context window and input modalities from the
``GET /v1/models`` catalog at connect time.

Authentication (API key, Bearer token):
- Set ``JAATO_NEBIUS_API_KEY`` (jaato namespace) or ``NEBIUS_API_KEY``
  (the vendor's own documented variable), or store credentials via
  ``nebius-auth``. Keys are issued from https://tokenfactory.nebius.com.
"""

from .provider import NebiusProvider, create_provider

__all__ = ["NebiusProvider", "create_provider"]


# --- Provider capability contract (see docs/model-provider-capabilities.md) ---
from ..base import ProviderCapabilities  # noqa: E402

PROVIDER_CAPABILITIES = ProviderCapabilities(
    user_message_images=True,
    tool_result_images=True,
    tool_choice_forwarding=True,
    thinking=True,
    prompt_caching=False,
    streaming=True,
    cancellation=True,
)
