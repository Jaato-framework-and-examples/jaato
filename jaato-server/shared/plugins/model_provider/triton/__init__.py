"""NVIDIA Triton Inference Server provider — OpenAI-compatible chat
+ KServe v2 model lifecycle control.

Triton serves models via two cooperating processes:

- ``openai_frontend`` exposes ``POST /v1/chat/completions`` (and the rest
  of the OpenAI Chat API surface) on its own port (default 9000).
- ``tritonserver`` exposes the native KServe v2 protocol on a separate
  port (default 8000), including the model repository extension at
  ``POST /v2/repository/models/<name>/load`` and ``/unload``.

This provider drives both: chat goes through the OpenAI URL, model
lifecycle control goes through the Triton control URL. When
``plugin_configs.triton.load`` is supplied, ``connect()`` loads (or
reloads) the model with the requested parameters before the first
chat — the same active/passive pattern as the LM Studio provider.

Compared to the bare ``tensorrt_llm`` provider, Triton adds:

- **Real in-band model loading** — no process supervision needed
- **Multi-model concurrent serving** on a single Triton instance
- **Backend-agnostic** — TRT-LLM, vLLM, PyTorch, ONNX, Python, etc.
  all behind the same wire format

Profile example (``plugin_configs`` keyed by provider name):

    {
      "provider": "triton",
      "model": "llama-3.1-8b-instruct",
      "plugin_configs": {
        "triton": {
          "openai_url": "http://gpu-box:9000",
          "control_url": "http://gpu-box:8000",
          "context_length": 131072,
          "load": {
            "parameters": {
              "config": "{\\"max_batch_size\\": 8}"
            }
          }
        }
      }
    }

Environment variables:
    TRITON_OPENAI_URL: OpenAI frontend URL (default: http://localhost:9000)
    TRITON_CONTROL_URL: Triton native HTTP URL for model control
        (default: http://localhost:8000)
    TRITON_HOST: Shorthand — when set without explicit URLs, both
        OpenAI and control URLs use this host with the default ports
    TRITON_MODEL: Default model name
    TRITON_CONTEXT_LENGTH: Override context window size (Triton's
        per-model config is too backend-specific for reliable
        auto-discovery)
    TRITON_API_TOKEN: Optional bearer token (only when behind an auth
        proxy — Triton itself relies on reverse-proxy auth)
"""

from .provider import TritonProvider, create_provider

__all__ = ["TritonProvider", "create_provider"]


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
        KnobSpec("openai_url", "str", None, "OpenAI frontend URL"),
        KnobSpec("control_url", "str", None, "Triton KServe control URL"),
        KnobSpec("host", "str", None, "shorthand host (derives both URLs)"),
        KnobSpec("api_token", "str", None, "bearer token (auth proxy)"),
        KnobSpec("context_length", "int"),
    ), description="server connection"),
    KnobLayer("load", opaque=True,
              description="KServe model-load body — forwarded verbatim "
                          "(parameters.config, ...)"),
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
               "plugin_configs.triton.api_token (optional)"),
    AuthSource("env", "TRITON_API_TOKEN",
               "optional — only behind an auth proxy"),
)
