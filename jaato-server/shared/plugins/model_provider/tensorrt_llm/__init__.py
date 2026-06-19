"""NVIDIA TensorRT-LLM Model Provider Plugin — OpenAI-compatible chat.

``trtllm-serve`` is the HTTP front-end shipped with NVIDIA TensorRT-LLM.
It compiles models into optimized TensorRT engines (custom CUDA kernels
for attention/GEMM/MoE, in-flight batching, KV-cache reuse, FP8/INT4
quantization, speculative decoding) and exposes them through an
OpenAI-compatible REST surface:

- ``POST /v1/chat/completions`` — chat (used for all completions)
- ``POST /v1/completions``      — text completion (unused here)
- ``GET  /v1/models``           — model catalog
- ``GET  /health``              — liveness probe
- ``GET  /version``             — server version
- ``GET  /metrics``             — runtime statistics

The provider is intentionally **passive**: it talks to a `trtllm-serve`
instance the user has already started. Engine build (``trtllm-build``),
model selection (``--model``), and per-engine knobs (tensor parallel,
KV-cache fraction, max batch size, …) all live at the server-launch
boundary — there is no in-band load endpoint analogous to LM Studio's
``/api/v1/models/load``.

Profile example (``plugin_configs`` keyed by provider name):

    {
      "provider": "tensorrt_llm",
      "model": "meta-llama/Llama-3.1-8B-Instruct",
      "plugin_configs": {
        "tensorrt_llm": {
          "host": "http://gpu-box:8000",
          "context_length": 131072,
          "api_token": "..."
        }
      }
    }

Profile knobs reach ``ProviderConfig.extra`` through the plugin_configs
wiring in ``jaato_runtime.create_provider()``.

Environment variables:
    TENSORRT_LLM_HOST: Server URL (default: http://localhost:8000)
    TENSORRT_LLM_MODEL: Default model name
    TENSORRT_LLM_CONTEXT_LENGTH: Override context window size
    TENSORRT_LLM_API_TOKEN: Optional bearer token (only when the server
        is fronted by a proxy that enforces auth — trtllm-serve itself
        does not document a built-in API key mechanism)
"""

from .provider import TensorRTLLMProvider, create_provider

__all__ = ["TensorRTLLMProvider", "create_provider"]


# --- Provider capability contract (see docs/model-provider-capabilities.md) ---
from ..base import ProviderCapabilities  # noqa: E402

PROVIDER_CAPABILITIES = ProviderCapabilities(
    user_message_images=True,
    tool_result_images=True,
    tool_choice_forwarding=False,
    thinking=False,
    prompt_caching=False,
    streaming=True,
    cancellation=True,
)
