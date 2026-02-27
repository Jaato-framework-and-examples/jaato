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
