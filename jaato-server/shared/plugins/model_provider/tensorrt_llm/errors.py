"""Error types for the TensorRT-LLM provider."""

from typing import List, Optional


class TensorRTLLMError(Exception):
    """Base class for TensorRT-LLM provider errors."""
    pass


class TensorRTLLMConnectionError(TensorRTLLMError):
    """trtllm-serve is not reachable at the configured host."""

    def __init__(self, host: str, message: str = ""):
        self.host = host
        detail = f": {message}" if message else ""
        super().__init__(
            f"Cannot connect to trtllm-serve at {host}{detail}\n"
            f"Make sure the server is running, e.g.:\n"
            f"  trtllm-serve <model> --host 0.0.0.0 --port 8000"
        )


class TensorRTLLMModelNotFoundError(TensorRTLLMError):
    """Requested model is not served by this trtllm-serve instance.

    A ``trtllm-serve`` process exposes exactly one engine (the model it
    was launched with), so this fires when ``connect()`` is called with a
    name that doesn't match the running engine's ``id`` in ``/v1/models``.
    """

    def __init__(self, model: str, available: Optional[List[str]] = None):
        self.model = model
        self.available = available or []
        if self.available:
            avail_str = ", ".join(self.available[:5])
            if len(self.available) > 5:
                avail_str += f", ... ({len(self.available)} total)"
            super().__init__(
                f"Model '{model}' is not served by this trtllm-serve instance.\n"
                f"Available: {avail_str}\n"
                f"trtllm-serve hosts one engine per process; restart it with the "
                f"engine you want, or point TENSORRT_LLM_HOST at a different server."
            )
        else:
            super().__init__(
                f"Model '{model}' is not served by this trtllm-serve instance.\n"
                f"trtllm-serve hosts one engine per process; check the server's "
                f"launch arguments or query GET /v1/models."
            )


class TensorRTLLMMidStreamError(TensorRTLLMError):
    """trtllm-serve dropped the connection mid-stream.

    Fires when ``chat.completions.create`` returns HTTP 200, the server
    starts streaming, and then the connection closes before the body is
    complete (``httpx.RemoteProtocolError: peer closed connection
    without sending complete message body`` / ``incomplete chunked
    read``).  Server-side this is almost always one of:

    1. **Prompt exceeds engine ``max_input_length``.**  trtllm-build
       takes ``--max_input_len`` as a build-time parameter SEPARATE from
       ``max_seq_len``; engines default to ``max_num_tokens`` when
       ``max_input_len`` is unset.  Prompts over the cap trigger a
       server-side ``tensorrt_llm.executor.utils.RequestError: Prompt
       length (XXXX) exceeds maximum input length (YYYY)`` which
       propagates AS a mid-stream connection drop because the HTTP 200
       headers are already committed by the time the engine validates
       (trtllm-serve's ASGI middleware sends headers before the
       scheduler accepts the request).
    2. **KV-cache exhaustion** mid-generation (sustained generation
       past ``max_num_tokens`` scheduler-step budget).
    3. **Engine-internal exception** (rare; usually surfaces as a CUDA
       OOM in the server log).

    Empirically demonstrated 2026-06-06 by the kb-orchestrator cascade
    against a WSL2 trtllm-serve endpoint built with
    ``max_input_len=16384`` (silent default from
    ``max_num_tokens=16384``) — discovery prompt at 19,977 tokens
    triggered the mid-stream drop.  Pre-this-class the error surfaced
    as a generic ``TensorRTLLMConnectionError: Cannot connect to
    trtllm-serve at ...`` which directed the user toward firewall /
    host-reachability debugging — the wrong tree entirely.
    """

    def __init__(self, host: str, original_error: Optional[str] = None):
        self.host = host
        self.original_error = original_error
        lines = [
            f"trtllm-serve at {host} dropped the connection mid-stream.",
        ]
        if original_error:
            lines.append(f"Underlying error: {original_error}")
        lines.extend([
            "",
            "This is almost always a server-side engine error AFTER the",
            "HTTP 200 has been committed.  Common causes:",
            "",
            "  1. Prompt exceeds the engine's max_input_length (a",
            "     trtllm-build parameter, separate from max_seq_len).",
            "     The engine silently caps at max_num_tokens when",
            "     max_input_len is unset at build time.  Default 16384.",
            "",
            "  2. KV-cache exhaustion under sustained generation past",
            "     max_num_tokens scheduler-step budget.",
            "",
            "  3. Engine-internal exception (rare; usually CUDA OOM).",
            "",
            "Decisive diagnostic: check the trtllm-serve log on the",
            "server host for an entry around the time of this error.",
            "Look for 'Prompt length (XXXX) exceeds maximum input length'",
            "or 'CUDA out of memory'.  The log file path depends on how",
            "the unit is configured; common locations:",
            "  - /opt/trtllm/logs/trtllm-serve.log (systemd file append)",
            "  - journalctl -u trtllm-serve.service --since '5 minutes ago'",
            "",
            "Fixes by cause:",
            "  - max_input_length too low → rebuild engine with",
            "    'trtllm-build --max_input_len <new>' matching max_seq_len.",
            "  - Prompt too large for the engine → shrink the prompt",
            "    via 'suppress_base_instructions: true' in the profile",
            "    (drops ~/.jaato/instructions) and/or reduce preloaded",
            "    plugin schemas.",
            "  - KV-cache OOM → reduce max_batch_size or",
            "    kv_cache_free_gpu_mem_fraction at engine build.",
        ])
        super().__init__("\n".join(lines))


class TensorRTLLMAuthenticationError(TensorRTLLMError):
    """trtllm-serve (or its fronting proxy) rejected the bearer token.

    trtllm-serve does not document a built-in API-key mechanism — this
    typically means an upstream reverse proxy (nginx, an API gateway,
    Triton's auth layer) enforced auth and the supplied
    ``TENSORRT_LLM_API_TOKEN`` was wrong or missing.
    """

    def __init__(self, original_error: Optional[str] = None):
        self.original_error = original_error
        lines = ["trtllm-serve rejected the bearer token."]
        if original_error:
            lines.append(f"Error: {original_error}")
        lines.extend([
            "",
            "If the server is behind an auth proxy:",
            "  1. Obtain the token from your proxy / gateway",
            "  2. Set TENSORRT_LLM_API_TOKEN=<token> or "
            "plugin_configs.tensorrt_llm.api_token in your profile",
        ])
        super().__init__("\n".join(lines))
