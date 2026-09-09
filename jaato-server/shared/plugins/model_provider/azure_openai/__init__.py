"""Azure OpenAI model provider plugin.

OpenAI's models served from a resource in your own Azure subscription:
same request and response shapes as the native provider (so the shared
``_openai_compat`` transport is reused unchanged), different everything
around them —

- **deployment-name routing** — ``model:`` carries the DEPLOYMENT name the
  subscription chose, not a model id, and the SDK turns it into
  ``/openai/deployments/<name>/...``;
- **a pinned ``api-version``** — required, no default, because the date
  decides which request fields exist;
- **key or Microsoft Entra ID auth** — ``auth: aad`` mints a bearer token
  per request from the host's identity (managed identity, workload
  identity, ``az login``) and needs no stored secret at all.

``plugin_configs.azure_openai.context_length`` is required in practice:
Azure's data plane lists deployments, not capacities, and a deployment
name does not identify the model version behind it, so ``connect()`` fails
loud rather than guessing.

Entra auth needs the optional ``azure-identity`` package:
``pip install 'jaato-server[azure-openai]'``.
"""

from .provider import AzureOpenAIProvider, create_provider

__all__ = ["AzureOpenAIProvider", "create_provider"]


# --- Provider capability contract (see docs/model-provider-capabilities.md) ---
from ..base import (  # noqa: E402
    ProviderCapabilities, ProviderKnobs, KnobLayer, KnobSpec, AuthSource,
)

PROVIDER_CAPABILITIES = ProviderCapabilities(
    user_message_images=True,
    tool_result_images=True,
    # NOT declared, and not an oversight: the ``file`` and ``input_audio``
    # content blocks are gated on Azure by api-version and by what the
    # deployment points at, so carrying them would be a guess about
    # somebody else's resource.  The native ``openai`` provider declares
    # both because its single endpoint carries them unconditionally.
    pdf_input=False,
    audio_input=False,
    tool_choice_forwarding=True,
    # Azure returns no reasoning text on the chat-completions surface.
    thinking=False,
    # Azure caches automatically for the models that support it, with no
    # client annotation and no TTL to choose, and the shared transport
    # reads the resulting ``cached_tokens`` back out of the prompt total.
    # Declared False for the same reason as the native provider: this
    # column means "the common ``cache:`` field has something to deliver",
    # and an unconfigurable cache gives it nothing.
    prompt_caching=False,
    streaming=True,
    cancellation=True,
    # Shares _openai_compat's wired streaming loop, which decodes
    # model-emitted audio deltas.
    output_media=True,
)

# --- Provider config-knob contract (authored from provider.py read sites) ---
PROVIDER_KNOBS = ProviderKnobs(layers=(
    KnobLayer("top_level", (
        KnobSpec("endpoint", "str", None,
                 "REQUIRED — https://<resource>.openai.azure.com"),
        KnobSpec("api_version", "str", None,
                 "REQUIRED — the api-version date, e.g. 2024-10-21; no "
                 "default, because it decides which request fields exist"),
        KnobSpec("auth", "str", "key",
                 "\"key\" (resource key) or \"aad\" (Microsoft Entra ID, "
                 "needs the azure-identity package)"),
        KnobSpec("api_key", "str", None, "Azure OpenAI resource key"),
        KnobSpec("context_length", "int", None,
                 "REQUIRED — Azure reports no per-deployment window, and a "
                 "deployment name does not identify the model behind it"),
        KnobSpec("model_name", "str", None,
                 "the MODEL id this deployment serves, e.g. gpt-4o; the only "
                 "thing that makes the input-modality table applicable, "
                 "since the deployment name is arbitrary"),
        KnobSpec("modalities", "list", None,
                 "assert INPUT modalities directly, e.g. [\"text\",\"image\"]"),
        KnobSpec("extra_body", "dict", None,
                 "opaque passthrough to OpenAI create() extra_body"),
        KnobSpec("output_modalities", "list", None,
                 "assert what the model can EMIT; no catalog reports output "
                 "modalities, so without this the floor is text and the "
                 "startup check refuses an outbound tier role"),
    ), description="resource / deployment / credential"),
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
        KnobSpec("modalities", "list", None,
                 "OUTPUT selector [\"text\",\"audio\"] — OpenAI's field, the "
                 "opposite direction from the tier key of the same name"),
        KnobSpec("audio", "dict", None,
                 "voice/format companion of api_params.modalities"),
    ), description="OpenAI Chat Completions params (filtered allow-list)"),
))
PROVIDER_QUIRKS = frozenset({
    # Inherited from the shared OpenAI-compat transport.
    "prose_tool_calls",
})

# --- Provider credential-resolution contract (from verify_auth/resolve_*) ---
PROVIDER_AUTH_RESOLUTION = (
    AuthSource("api_key_param", "api_key",
               "plugin_configs.azure_openai.api_key (pass:// ok)"),
    AuthSource("env", "JAATO_AZURE_OPENAI_API_KEY"),
    AuthSource("env", "AZURE_OPENAI_API_KEY", "the vendor's own variable"),
    AuthSource("stored", "azure_openai_auth.json",
               "config_root → workspace/.jaato → ~/.jaato; the key and its "
               "endpoint are stored together, because a key from another "
               "resource authenticates against nothing here"),
    AuthSource("adc", "DefaultAzureCredential",
               "auth: aad — managed identity / workload identity / az login, "
               "via azure-identity; no stored secret"),
)
