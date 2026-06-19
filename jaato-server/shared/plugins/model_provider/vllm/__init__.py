"""vLLM Model Provider Plugin — OpenAI-compatible chat.

vLLM is a high-throughput, memory-efficient LLM serving engine.  Its
OpenAI-compatible API server (``vllm.entrypoints.openai.api_server``)
exposes a faithful subset of the OpenAI REST surface:

- ``POST /v1/chat/completions`` — chat (used for all completions)
- ``POST /v1/completions``      — text completion (unused here)
- ``POST /v1/embeddings``       — embeddings (unused here)
- ``GET  /v1/models``           — model catalog
- ``GET  /health``              — liveness probe

The provider is intentionally **passive**: it talks to a vLLM server
the user has already started.  Model selection (``--model``), context
length (``--max-model-len``), tool-call parser
(``--enable-auto-tool-choice --tool-call-parser <name>``), and engine
knobs (tensor parallel, GPU memory utilization, quantization, etc.)
all live at the server-launch boundary.  There is no in-band load
endpoint analogous to LM Studio's ``/api/v1/models/load``.

Researched against vLLM stable docs (https://docs.vllm.ai/en/stable/)
on 2026-06-07 via the context7 MCP:

- **``/v1/models`` response shape**: standard OpenAI list with entries
  ``{id, object, created, owned_by, root, parent, permission}``.
  ``max_model_len`` is **NOT** included — operators must communicate
  the context window out-of-band (env var / profile knob).  Mirrors
  trtllm-serve's behavior.
- **``--api-key <token>``**: native vLLM server flag.  When set,
  clients must present ``Authorization: Bearer <token>`` on every
  request; otherwise vLLM accepts any value (the common default is
  ``"EMPTY"``).
- **Tool-calling**: enabled per-server via ``--enable-auto-tool-choice``
  + ``--tool-call-parser <name>``.  Parser names are per-model-family
  (``hermes``, ``mistral``, ``llama3_json``, ``pythonic``, ``granite``,
  ``phi4_mini_json``, ``apertus``, ``xlam``, ...).  These flags shape
  the server's PARSING of model output back into OpenAI tool-call
  shape — they do not change the request body.  Client-side this
  provider just emits standard OpenAI ``tools`` / ``tool_choice``.
- **Streaming**: faithful OpenAI SSE format, including
  ``stream_options={"include_usage": true}`` so usage arrives in the
  trailing chunk.
- **Structured outputs**: native ``response_format={"type": "json_object"}``
  and ``response_format={"type": "json_schema", "json_schema": {...}}``
  both work.  Guided decoding (``guided_json``, ``guided_choice``,
  ``guided_regex``, ``guided_grammar``) is reachable via the OpenAI
  SDK's ``extra_body`` kwarg but is currently NOT exposed through
  this provider's profile knobs — operators wanting strict guided
  decoding compose ``extra_body`` themselves at the harness layer.
  See "Sub-decisions" in ``smoke/README.md``.
- **Mid-stream errors**: when vLLM raises after HTTP 200 has been
  committed (most commonly: prompt exceeds ``--max-model-len``, KV
  cache exhaustion), the connection drops mid-stream and the OpenAI
  SDK wraps it as ``openai.APIConnectionError`` with
  ``httpx.RemoteProtocolError`` inside.  This is the same class of
  failure as ``TensorRTLLMMidStreamError`` and ``TritonMidStreamError``;
  the provider classifies it explicitly so the user is pointed at
  engine-config debugging, not network debugging.  Per the rule in
  ``feedback_no_hardcoded_likely_cause_in_error_messages.md``, the
  classifier surfaces a generic "what we observed" message with a
  pointer to server-side logs — not a guess at the cause.

Profile example (``plugin_configs`` keyed by provider name):

    {
      "provider": "vllm",
      "model": "Qwen/Qwen2.5-7B-Instruct",
      "plugin_configs": {
        "vllm": {
          "host": "http://gpu-box:8000",
          "context_length": 32768,
          "api_token": "..."
        }
      }
    }

Profile knobs reach ``ProviderConfig.extra`` through the plugin_configs
wiring in ``jaato_runtime.create_provider()``.

Environment variables:
    VLLM_HOST: Server URL (default: http://localhost:8000)
    VLLM_MODEL: Default model name
    VLLM_CONTEXT_LENGTH: Override context window size (required for
        long-context engines — vLLM's /v1/models does not surface
        max_model_len, the build-time --max-model-len value)
    VLLM_API_TOKEN: Optional bearer token (only when the server was
        launched with --api-key, or when fronted by an auth proxy)
"""

from .provider import VLLMProvider, create_provider

__all__ = ["VLLMProvider", "create_provider"]


# --- Provider capability contract (see docs/model-provider-capabilities.md) ---
from ..base import ProviderCapabilities  # noqa: E402

PROVIDER_CAPABILITIES = ProviderCapabilities(
    user_message_images=False,
    tool_result_images=False,
    tool_choice_forwarding=True,
    thinking=False,
    prompt_caching=False,
    streaming=True,
    cancellation=True,
)
