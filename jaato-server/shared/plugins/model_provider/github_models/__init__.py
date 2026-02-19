"""GitHub Models provider plugin.

This provider enables access to AI models through the GitHub Models API,
supporting GPT, Claude, Gemini, and other models available on GitHub.

Authentication methods:
- Personal Access Token (PAT) with `models: read` scope
- GitHub App token with `models: read` permission
- Fine-grained PAT (recommended for enterprise SSO)

Enterprise features:
- Organization-attributed billing
- Enterprise policy compliance
- SSO support (fine-grained PATs auto-authorized)
"""

from .provider import GitHubModelsProvider, create_provider

__all__ = ["GitHubModelsProvider", "create_provider"]
