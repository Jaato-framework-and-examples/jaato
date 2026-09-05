"""LM Studio Model Provider Plugin — OpenAI-compatible chat + native load control.

LM Studio exposes a local inference server with two relevant surfaces:

- ``POST /v1/chat/completions`` — OpenAI-compatible chat (used for all completions)
- ``POST /api/v1/models/load`` — native load endpoint where the model is
  loaded into memory with a specific configuration (context length, GPU
  offload, KV-cache offload, eval batch size, flash attention, number of
  experts for MoE models, etc.).  This provider invokes the load endpoint
  automatically when ``config.extra['load']`` is populated from the
  session profile.

Profile example (``plugin_configs`` keyed by provider name):

    {
      "provider": "lmstudio",
      "model": "openai/gpt-oss-20b",
      "plugin_configs": {
        "lmstudio": {
          "host": "http://localhost:1234",
          "context_length": 16384,
          "load": {
            "context_length": 16384,
            "eval_batch_size": 512,
            "flash_attention": true,
            "offload_kv_cache_to_gpu": true,
            "num_experts": 8
          }
        }
      }
    }

Profile knobs reach ``ProviderConfig.extra`` through the plugin_configs
wiring in ``jaato_runtime.create_provider()`` — providers are plugins,
so profile-level overrides live under ``plugin_configs[<provider_name>]``.

Environment variables:
    LMSTUDIO_HOST: Server URL (default: http://localhost:1234)
    LMSTUDIO_MODEL: Default model name
    LMSTUDIO_CONTEXT_LENGTH: Override context window size
    LMSTUDIO_API_TOKEN: Optional bearer token (only when LM Studio requires it)
"""

from .provider import LMStudioProvider, create_provider

__all__ = ["LMStudioProvider", "create_provider"]


# --- Provider capability contract (see docs/model-provider-capabilities.md) ---
from ..base import (  # noqa: E402
    ProviderCapabilities, ProviderKnobs, KnobLayer, KnobSpec, AuthSource,
)

PROVIDER_CAPABILITIES = ProviderCapabilities(
    user_message_images=True,
    tool_result_images=True,
    pdf_input=False,
    tool_choice_forwarding=False,
    thinking=False,
    prompt_caching=False,
    streaming=True,
    cancellation=True,
    output_media=False,
)

# --- Provider config-knob contract (authored from provider.py read sites) ---
PROVIDER_KNOBS = ProviderKnobs(layers=(
    KnobLayer("top_level", (
        KnobSpec("host", "str", None, "LM Studio server URL (LMSTUDIO_HOST)"),
        KnobSpec("api_token", "str", None, "bearer token (only if required)"),
        KnobSpec("context_length", "int"),
    ), description="LM Studio server connection"),
    KnobLayer("load", opaque=True,
              description="native /api/v1/models/load body — any key "
                          "forwarded verbatim (context_length, "
                          "flash_attention, offload_kv_cache_to_gpu, "
                          "eval_batch_size, num_experts, ...)"),
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
    AuthSource("api_key_param", "api_token",
               "plugin_configs.lmstudio.api_token (optional)"),
    AuthSource("env", "LMSTUDIO_API_TOKEN",
               "optional — only if the server requires a token"),
)
