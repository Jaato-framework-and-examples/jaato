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
    read``).

    The HTTP 200 + chunked headers are committed by trtllm-serve's ASGI
    middleware BEFORE the executor validates the request, so any
    server-side failure (prompt validation, KV-cache pressure, engine
    exception, OOM, shutdown) surfaces as the same wire-level
    connection drop.  The framework cannot extract the failure reason
    from the wire — by the time httpx sees the drop, the SSE chunk
    that would carry it has been silently discarded.

    Per the rule in
    ``feedback_no_hardcoded_likely_cause_in_error_messages.md``, this
    error surfaces "what we observed" (HTTP 200 committed, then
    connection drop) and points at the server log for the actual
    reason.  It does NOT enumerate likely causes — every previous
    attempt to do so anchored operators on the wrong hypothesis when
    the real cause was a different one.

    Symmetric to ``VLLMMidStreamError`` / ``TritonMidStreamError`` —
    every OpenAI-compatible self-hosted inference server exposes the
    same envelope-commit-before-validate failure mode.
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
            "The HTTP 200 was already committed when the connection",
            "dropped, so the framework cannot see the failure reason",
            "from the wire.  Check the trtllm-serve log on the host",
            "running the engine for an entry around the time of this",
            "error — it will name the actual cause.",
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
