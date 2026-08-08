"""Error types for the vLLM provider."""

from typing import List, Optional


class VLLMError(Exception):
    """Base class for vLLM provider errors."""
    pass


class VLLMConnectionError(VLLMError):
    """vLLM server is not reachable at the configured host."""

    def __init__(self, host: str, message: str = ""):
        self.host = host
        detail = f": {message}" if message else ""
        super().__init__(
            f"Cannot connect to vLLM at {host}{detail}\n"
            f"Make sure the server is running, e.g.:\n"
            f"  vllm serve <model> --host 0.0.0.0 --port 8000"
        )


class VLLMModelNotFoundError(VLLMError):
    """Requested model is not served by this vLLM instance.

    A vLLM server process exposes the model (or LoRA-adapted models)
    listed in ``GET /v1/models``.  This fires when ``connect()`` is
    called with a name that doesn't match any entry in the catalog.
    """

    def __init__(self, model: str, available: Optional[List[str]] = None):
        self.model = model
        self.available = available or []
        if self.available:
            avail_str = ", ".join(self.available[:5])
            if len(self.available) > 5:
                avail_str += f", ... ({len(self.available)} total)"
            super().__init__(
                f"Model '{model}' is not served by this vLLM instance.\n"
                f"Available: {avail_str}\n"
                f"vLLM serves the model passed to 'vllm serve <model>' at launch "
                f"(plus any LoRA adapters); restart it with the engine you want, "
                f"or point VLLM_HOST at a different server."
            )
        else:
            super().__init__(
                f"Model '{model}' is not served by this vLLM instance.\n"
                f"Check the server's launch arguments or query GET /v1/models."
            )


class VLLMMidStreamError(VLLMError):
    """vLLM dropped the connection mid-stream.

    Fires when ``chat.completions.create`` returns HTTP 200, the server
    starts streaming, and then the connection closes before the body is
    complete (``httpx.RemoteProtocolError: peer closed connection
    without sending complete message body`` / ``incomplete chunked
    read``).

    Per the rule in
    ``feedback_no_hardcoded_likely_cause_in_error_messages.md``, this
    message does NOT guess at the cause — the framework didn't actually
    parse the upstream failure (the connection drop is by definition
    pre-parse).  We surface "what we observed" (connection drop after
    HTTP 200) and point the operator at the vLLM server log for the
    actual diagnostic line.

    Symmetric to ``TensorRTLLMMidStreamError`` /
    ``TritonMidStreamError`` — both providers wrap a different engine
    behind the same OpenAI-compatible frontend and surface the same
    failure mode.
    """

    def __init__(self, host: str, original_error: Optional[str] = None):
        self.host = host
        self.original_error = original_error
        lines = [
            f"vLLM at {host} dropped the connection mid-stream.",
        ]
        if original_error:
            lines.append(f"Underlying error: {original_error}")
        lines.extend([
            "",
            "The HTTP 200 was already committed when the connection",
            "dropped, so the framework cannot see the failure reason",
            "from the wire.  Check the vLLM server log on the host",
            "running the engine for an entry around the time of this",
            "error — it will name the actual cause.",
        ])
        super().__init__("\n".join(lines))


class VLLMAuthenticationError(VLLMError):
    """vLLM (or its fronting proxy) rejected the bearer token.

    vLLM has a native ``--api-key <token>`` server flag.  Requests
    without a matching ``Authorization: Bearer <token>`` header are
    rejected with HTTP 401.  Production deployments commonly also sit
    behind a reverse proxy that enforces auth; the failure mode is
    identical from the client's perspective.
    """

    def __init__(self, original_error: Optional[str] = None):
        self.original_error = original_error
        lines = ["vLLM rejected the bearer token."]
        if original_error:
            lines.append(f"Error: {original_error}")
        lines.extend([
            "",
            "If the server was launched with --api-key <token> (vLLM's",
            "native bearer auth) or sits behind an auth proxy:",
            "  1. Obtain the token from the operator / proxy / gateway",
            "  2. Set VLLM_API_TOKEN=<token> or "
            "plugin_configs.vllm.api_token in your profile",
        ])
        super().__init__("\n".join(lines))
