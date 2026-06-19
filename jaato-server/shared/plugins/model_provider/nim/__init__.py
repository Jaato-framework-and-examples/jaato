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
from ..base import ProviderCapabilities  # noqa: E402

PROVIDER_CAPABILITIES = ProviderCapabilities(
    user_message_images=False,
    tool_result_images=False,
    tool_choice_forwarding=False,
    thinking=True,
    prompt_caching=False,
    streaming=True,
    cancellation=True,
)
