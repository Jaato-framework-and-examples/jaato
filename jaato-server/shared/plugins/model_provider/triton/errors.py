"""Error types for the Triton provider."""

from typing import List, Optional


class TritonError(Exception):
    """Base class for Triton provider errors."""
    pass


class TritonConnectionError(TritonError):
    """Triton (or its OpenAI frontend) is not reachable.

    Carries the offending URL — Triton exposes two separate HTTP
    surfaces (chat on the OpenAI frontend port, model control on the
    KServe v2 port), so the error message must identify which one
    failed.
    """

    def __init__(self, url: str, message: str = "", *, surface: str = ""):
        self.url = url
        self.surface = surface
        detail = f": {message}" if message else ""
        surface_label = f" ({surface})" if surface else ""
        super().__init__(
            f"Cannot connect to Triton at {url}{surface_label}{detail}\n"
            f"Make sure tritonserver and (if used) the OpenAI frontend are running:\n"
            f"  tritonserver --model-repository=/models --model-control-mode=explicit\n"
            f"  python3 openai_frontend/main.py --model-repository=/models ..."
        )


class TritonModelNotFoundError(TritonError):
    """Requested model is not present in Triton's model repository."""

    def __init__(self, model: str, available: Optional[List[str]] = None):
        self.model = model
        self.available = available or []
        if self.available:
            avail_str = ", ".join(self.available[:5])
            if len(self.available) > 5:
                avail_str += f", ... ({len(self.available)} total)"
            super().__init__(
                f"Model '{model}' not found in Triton's model repository.\n"
                f"Available: {avail_str}\n"
                f"Add the model under --model-repository or pick one of the above."
            )
        else:
            super().__init__(
                f"Model '{model}' not found in Triton's model repository.\n"
                f"Check --model-repository or query POST /v2/repository/index."
            )


class TritonLoadError(TritonError):
    """Triton refused to load the model with the supplied configuration.

    Raised when ``POST /v2/repository/models/<name>/load`` returns a
    non-2xx — typically indicates a malformed ``parameters`` override,
    a model whose backend isn't available, or insufficient resources.
    """

    def __init__(
        self,
        model: str,
        status_code: int,
        body: str,
        load_config: Optional[dict] = None,
    ):
        self.model = model
        self.status_code = status_code
        self.body = body
        self.load_config = load_config or {}
        config_summary = (
            ", ".join(f"{k}={v}" for k, v in self.load_config.items())
            if self.load_config
            else "(default — no parameters override)"
        )
        super().__init__(
            f"Triton failed to load '{model}' (HTTP {status_code}).\n"
            f"Requested load body: {config_summary}\n"
            f"Response: {body[:500]}"
        )


class TritonMidStreamError(TritonError):
    """Triton dropped the connection mid-stream on the OpenAI frontend.

    Symmetric to ``TensorRTLLMMidStreamError`` — both providers wrap the
    same trtllm engine and surface the same class of bug via different
    OpenAI-compatible frontends.  Triton's ``openai_frontend`` (or the
    builtin ``--openai-port`` server on 25.x+) commits HTTP 200 headers
    before the engine's scheduler validates the request, so engine-level
    errors (prompt-too-long, KV-cache OOM, internal exceptions) propagate
    as ``httpx.RemoteProtocolError: peer closed connection without
    sending complete message body`` rather than a structured HTTP 4xx/5xx.

    Almost always one of:

    1. **Prompt exceeds engine ``max_input_length``.**  The engine
       cached at ``<model_repo>/tensorrt_llm/1/`` was built with
       ``trtllm-build --max_input_len <N>``; prompts over N trigger
       a server-side ``RequestError: Prompt length (XXXX) exceeds
       maximum input length (N)``.  The default when ``max_input_len``
       is unset at build time is ``max_num_tokens`` (typically 16384).
    2. **KV-cache exhaustion** under sustained generation past
       ``max_num_tokens`` scheduler-step budget.
    3. **Engine-internal exception** (rare; usually surfaces as CUDA
       OOM in Triton's log).

    Pre-this-class the error surfaced as a generic
    ``TritonConnectionError`` which directed the user toward Triton-
    reachability debugging — the wrong tree.
    """

    def __init__(self, openai_url: str, original_error: Optional[str] = None):
        self.openai_url = openai_url
        self.original_error = original_error
        lines = [
            f"Triton (OpenAI frontend) at {openai_url} dropped the "
            "connection mid-stream.",
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
            "Decisive diagnostic: check Triton's log on the server host",
            "for an entry around the time of this error.  Look for",
            "'Prompt length (XXXX) exceeds maximum input length' or",
            "'CUDA out of memory'.  Common locations:",
            "  - docker logs <triton-container> (when run as a container)",
            "  - /opt/triton/logs/triton.log (when run via systemd unit",
            "    with file logging)",
            "  - journalctl -u triton.service --since '5 minutes ago'",
            "",
            "Fixes by cause:",
            "  - max_input_length too low → rebuild engine with",
            "    'trtllm-build --max_input_len <new>' and re-place in",
            "    the model_repository.  Restart Triton.",
            "  - Prompt too large for the engine → shrink via",
            "    'suppress_base_instructions: true' in the profile and/or",
            "    reduce preloaded plugin schemas.",
            "  - KV-cache OOM → reduce max_batch_size or",
            "    kv_cache_free_gpu_mem_fraction at engine build.",
        ])
        super().__init__("\n".join(lines))


class TritonAuthenticationError(TritonError):
    """Triton (or its fronting proxy) rejected the bearer token.

    Triton itself does not implement bearer-token auth — this typically
    means an upstream reverse proxy or Triton's ``--openai-restricted-api``
    header gate rejected the request.
    """

    def __init__(self, original_error: Optional[str] = None):
        self.original_error = original_error
        lines = ["Triton rejected the bearer token."]
        if original_error:
            lines.append(f"Error: {original_error}")
        lines.extend([
            "",
            "If Triton is behind an auth proxy or --openai-restricted-api:",
            "  1. Obtain the token / header value from your proxy / gateway",
            "  2. Set TRITON_API_TOKEN=<token> or "
            "plugin_configs.triton.api_token in your profile",
        ])
        super().__init__("\n".join(lines))
