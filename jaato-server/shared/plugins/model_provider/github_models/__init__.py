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


# --- Provider capability contract (see docs/model-provider-capabilities.md) ---
from ..base import (  # noqa: E402
    ProviderCapabilities, ProviderKnobs, KnobLayer, KnobSpec, AuthSource,
)

PROVIDER_CAPABILITIES = ProviderCapabilities(
    user_message_images=False,
    tool_result_images=False,
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
        KnobSpec("organization", "str", None, "billing-attribution org"),
        KnobSpec("enterprise", "str"),
        KnobSpec("endpoint", "str"),
        KnobSpec("context_length", "int"),
    ), description="connection / billing"),
))
PROVIDER_QUIRKS = frozenset()

# --- Provider credential-resolution contract (from verify_auth/resolve_*) ---
PROVIDER_AUTH_RESOLUTION = (
    AuthSource("oauth", "github_oauth.json",
               "device-code token (.jaato/ → ~/.jaato)"),
    AuthSource("env", "GITHUB_TOKEN"),
)
