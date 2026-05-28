"""TensorRT-LLM provider using ``trtllm-serve``'s OpenAI-compatible API.

``trtllm-serve`` is NVIDIA's HTTP front-end for TensorRT-LLM engines.
Its surface is plain OpenAI:

- Chat:    ``POST /v1/chat/completions``
- List:    ``GET  /v1/models``
- Health:  ``GET  /health``

Chat completions run through the OpenAI SDK exactly like the ``lmstudio``
and ``nim`` providers.  The provider is **passive** — it talks to a
``trtllm-serve`` instance the user has already launched.  Engine build
(``trtllm-build``) and per-engine knobs (tensor parallel, KV-cache
fraction, max batch size, ...) all live at the server-launch boundary;
there is no in-band load endpoint analogous to LM Studio's
``/api/v1/models/load``.

Authentication is optional: ``trtllm-serve`` itself does not document a
built-in API key mechanism, but production deployments are commonly
fronted by a reverse proxy that enforces auth.  When
``TENSORRT_LLM_API_TOKEN`` (or ``plugin_configs.tensorrt_llm.api_token``)
is set, it is sent as ``Authorization: Bearer <token>``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import httpx

# Reuse NIM's OpenAI lazy imports and converters — identical SDK,
# identical wire format.
from ..nim._lazy import get_openai_client_class, get_openai_module

if TYPE_CHECKING:
    from openai import OpenAI

from ..base import (
    FunctionCallDetectedCallback,
    ProviderConfig,
    StreamingCallback,
    ThinkingCallback,
    UsageUpdateCallback,
)
from jaato_sdk.plugins.model_provider.types import (
    CancelToken,
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    ThinkingConfig,
    ToolSchema,
    TokenUsage,
    TurnResult,
)
from ..nim.converters import (
    clear_tool_name_mapping,
    get_original_tool_name,
    history_to_openai,
    map_finish_reason,
    response_from_openai,
    tool_schemas_to_openai,
)
from .env import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_HOST,
    resolve_api_token,
    resolve_context_length,
    resolve_host,
    resolve_model,
)
from .errors import (
    TensorRTLLMAuthenticationError,
    TensorRTLLMConnectionError,
    TensorRTLLMModelNotFoundError,
)


logger = logging.getLogger(__name__)


def _extract_cache_tokens(usage_obj) -> Optional[int]:
    """Extract cached_tokens from an OpenAI usage object, if present.

    trtllm-serve surfaces KV-cache reuse stats in the OpenAI-standard
    shape ``usage.prompt_tokens_details.cached_tokens`` when the engine
    has prompt caching enabled.  Returns ``None`` when the server omits
    the field.
    """
    if not usage_obj:
        return None
    details = getattr(usage_obj, "prompt_tokens_details", None)
    if not details:
        return None
    cached = getattr(details, "cached_tokens", None)
    if cached is not None and isinstance(cached, int) and cached > 0:
        return cached
    return None


class TensorRTLLMProvider:
    """TensorRT-LLM provider talking to ``trtllm-serve``'s /v1 endpoint.

    Lifecycle:
        1. ``__init__()``              — no state yet
        2. ``initialize(config)``      — resolve host / token / context override,
                                         create the OpenAI client, probe ``/health``
        3. ``connect(model)``          — verify the engine in ``/v1/models``
                                         matches the requested model name
        4. ``complete(messages, ...)`` — stateless chat
        5. ``shutdown()``              — release the OpenAI client
    """

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        self._client: Optional["OpenAI"] = None
        self._model_name: Optional[str] = None

        self._host: str = DEFAULT_HOST
        self._api_token: Optional[str] = None
        self._auth_info: str = ""

        self._last_usage: TokenUsage = TokenUsage()
        self._context_length_override: Optional[int] = None

        self._agent_type: str = "main"
        self._agent_name: Optional[str] = None
        self._agent_id: str = "main"

    def set_agent_context(
        self,
        agent_type: str = "main",
        agent_name: Optional[str] = None,
        agent_id: str = "main",
    ) -> None:
        """Record agent context so trace messages are attributable."""
        self._agent_type = agent_type
        self._agent_name = agent_name
        self._agent_id = agent_id

    def _trace(self, msg: str) -> None:
        """Write a trace message to the provider trace log."""
        from shared.trace import provider_trace
        if self._agent_type == "main":
            prefix = "tensorrt_llm:main"
        elif self._agent_name:
            prefix = f"tensorrt_llm:subagent:{self._agent_name}"
        else:
            prefix = f"tensorrt_llm:subagent:{self._agent_id}"
        provider_trace(prefix, msg)

    @property
    def name(self) -> str:
        """Provider identifier — used as the key in ``plugin_configs``."""
        return "tensorrt_llm"

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider.

        Reads host / optional token / context override from ``config.extra``
        (populated from the session profile's
        ``plugin_configs['tensorrt_llm']``), then verifies the server is
        reachable via ``GET /health``.

        Args:
            config: ``ProviderConfig`` whose ``extra`` dict may contain:

                - ``host`` (str): Override ``TENSORRT_LLM_HOST``.
                - ``context_length`` (int): Override context window size.
                - ``api_token`` (str): Bearer token when an auth proxy
                  fronts ``trtllm-serve``.

        Raises:
            TensorRTLLMConnectionError: Server not reachable.
        """
        self._trace("[INIT] Starting initialization")

        if config is None:
            config = ProviderConfig()

        self._host = (config.extra.get("host") or resolve_host()).rstrip("/")
        self._api_token = config.extra.get("api_token") or resolve_api_token()

        context_extra = config.extra.get("context_length")
        if context_extra:
            self._context_length_override = int(context_extra)
        else:
            self._context_length_override = resolve_context_length()

        self._auth_info = (
            f"local ({self._host}, bearer)"
            if self._api_token
            else f"local ({self._host})"
        )
        self._trace(f"[INIT] host={self._host}")

        self._client = self._create_client()
        self._verify_connectivity()
        self._trace("[INIT] Initialization complete")

    def _create_client(self) -> "OpenAI":
        """Build the OpenAI client pointing at trtllm-serve's /v1 endpoint.

        trtllm-serve accepts any non-empty API key (its OpenAI shim
        ignores the value), but we forward the real bearer token when one
        is configured so fronting proxies that enforce auth also work.
        """
        client_class = get_openai_client_class()
        return client_class(
            base_url=f"{self._host}/v1",
            api_key=self._api_token or "trtllm",
        )

    def _auth_headers(self) -> Dict[str, str]:
        """Return auth headers for direct ``httpx`` calls."""
        headers = {"Content-Type": "application/json"}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"
        return headers

    def _verify_connectivity(self) -> None:
        """Confirm trtllm-serve is reachable via ``GET /health``.

        ``/health`` is the cheapest liveness probe trtllm-serve exposes;
        ``/v1/models`` would also work but does extra work and isn't
        guaranteed to be ready instantly after process start.
        """
        try:
            response = httpx.get(
                f"{self._host}/health",
                headers=self._auth_headers(),
                timeout=5,
            )
            response.raise_for_status()
        except httpx.ConnectError:
            raise TensorRTLLMConnectionError(self._host)
        except httpx.TimeoutException:
            raise TensorRTLLMConnectionError(self._host, "Connection timed out")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise TensorRTLLMAuthenticationError(original_error=str(exc))
            raise TensorRTLLMConnectionError(self._host, str(exc))
        except httpx.HTTPError as exc:
            raise TensorRTLLMConnectionError(self._host, str(exc))

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None,
        config: Optional[ProviderConfig] = None,
    ) -> bool:
        """Credentials-only check — trtllm-serve has no built-in auth.

        Per the provider-plugin contract (``shared/plugins/CLAUDE.md``),
        ``verify_auth`` runs on a *fresh, uninitialized* instance and must
        only check whether credentials are **available** — not whether
        they are valid, and not whether the remote service is reachable.
        Reachability and engine validity are the job of ``initialize()``
        and ``connect()``.

        For TensorRT-LLM that means: there is nothing to authenticate in
        the default setup, so we always return ``True``.  When the
        upstream is behind an auth proxy, the bearer is read either from
        ``plugin_configs['tensorrt_llm']['api_token']`` (via
        ``config.extra`` when supplied by the runtime) or from
        ``TENSORRT_LLM_API_TOKEN``.  Its presence is reported for
        operator visibility but is never a hard requirement.
        """
        token: Optional[str] = None
        if config is not None and config.extra:
            token = config.extra.get("api_token")
        if not token:
            token = resolve_api_token()

        if on_message:
            if token:
                masked = (
                    f"{token[:4]}…{token[-4:]}" if len(token) > 8 else "***"
                )
                on_message(f"TensorRT-LLM bearer token configured ({masked})")
            else:
                on_message(
                    "TensorRT-LLM: no authentication required "
                    "(reachability validated at session start)"
                )
        return True

    def shutdown(self) -> None:
        """Close the OpenAI client."""
        if self._client:
            self._client.close()
        self._client = None
        self._model_name = None

    def get_auth_info(self) -> str:
        """Short human-readable description of auth state."""
        return self._auth_info or "TensorRT-LLM (local)"

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Select the model for subsequent ``complete()`` calls.

        Verifies the requested model matches the engine ``trtllm-serve``
        is hosting.  A single ``trtllm-serve`` process exposes exactly
        one engine, so the catalog returned by ``GET /v1/models`` will
        contain one entry — the model name passed to ``trtllm-serve``'s
        ``--model`` argument (or the HuggingFace repo id of the source
        weights).

        Args:
            model: Model identifier as ``trtllm-serve`` reports it in
                ``/v1/models`` (e.g. ``meta-llama/Llama-3.1-8B-Instruct``).
            skip_model_test: Skip the GET-models validation call.

        Raises:
            TensorRTLLMModelNotFoundError: Server is hosting a different engine.
        """
        if not skip_model_test:
            catalog = self._fetch_catalog()
            if catalog and model not in {entry["id"] for entry in catalog}:
                raise TensorRTLLMModelNotFoundError(
                    model, available=[entry["id"] for entry in catalog],
                )

        self._model_name = model

        logger.info(
            "Connected to trtllm-serve model: %s (context=%d)",
            model, self.get_context_limit(),
        )

    def _fetch_catalog(self) -> List[Dict[str, Any]]:
        """Query ``GET /v1/models`` and return the raw ``data`` array.

        Standard OpenAI shape: ``{"object": "list", "data": [{"id": ...}]}``.
        Returns an empty list when the server is unreachable so callers
        can degrade gracefully (the connect-time test treats an empty
        catalog as "skip validation" rather than "model not found", which
        avoids spurious failures on transient blips).
        """
        try:
            response = httpx.get(
                f"{self._host}/v1/models",
                headers=self._auth_headers(),
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            return payload.get("data", [])
        except httpx.HTTPError as exc:
            logger.warning("Failed to list trtllm-serve models: %s", exc)
            return []

    @property
    def is_connected(self) -> bool:
        """True when both client and model are set."""
        return self._client is not None and self._model_name is not None

    @property
    def model_name(self) -> Optional[str]:
        """Currently selected model name, or ``None`` before ``connect()``."""
        return self._model_name

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List models served by this trtllm-serve instance.

        A single ``trtllm-serve`` process hosts one engine, so this
        typically returns a single name — the model passed to
        ``--model`` at server launch.
        """
        catalog = self._fetch_catalog()
        names = [entry["id"] for entry in catalog]
        if prefix:
            names = [n for n in names if n.startswith(prefix)]
        return sorted(names)

    # ==================== Stateless Completion ====================

    def complete(
        self,
        messages: List[Message],
        system_instruction: Optional[str] = None,
        tools: Optional[List[ToolSchema]] = None,
        *,
        response_schema: Optional[Dict[str, Any]] = None,
        cancel_token: Optional[CancelToken] = None,
        on_chunk: Optional[StreamingCallback] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
        on_thinking: Optional[ThinkingCallback] = None,
    ) -> TurnResult:
        """Run one stateless chat completion through trtllm-serve's /v1.

        Streaming and non-streaming paths mirror ``lmstudio``'s
        implementation — trtllm-serve speaks OpenAI's wire format
        faithfully, including ``tools`` / ``tool_choice`` for models
        whose tokenizer chat template supports function calling.
        """
        if not self._client or not self._model_name:
            raise RuntimeError(
                "Provider not connected. Call initialize() and connect() first."
            )

        clear_tool_name_mapping()

        openai_messages: List[Dict[str, Any]] = []
        if system_instruction:
            openai_messages.append({"role": "system", "content": system_instruction})
        openai_messages.extend(history_to_openai(list(messages)))

        kwargs: Dict[str, Any] = {}
        if tools:
            openai_tools = tool_schemas_to_openai(tools)
            if openai_tools:
                kwargs["tools"] = openai_tools
        if response_schema:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            if on_chunk:
                provider_response = self._stream_response(
                    messages=openai_messages,
                    kwargs=kwargs,
                    on_chunk=on_chunk,
                    cancel_token=cancel_token,
                    on_usage_update=on_usage_update,
                    on_thinking=on_thinking,
                    trace_prefix="COMPLETE_STREAM",
                )
            else:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=openai_messages,
                    **kwargs,
                )
                provider_response = response_from_openai(response)
                if response and response.usage:
                    cached = _extract_cache_tokens(response.usage)
                    if cached is not None:
                        provider_response.usage.cache_read_tokens = cached

            self._last_usage = provider_response.usage

            text = provider_response.get_text()
            if response_schema and text:
                try:
                    provider_response.structured_output = json.loads(text)
                except json.JSONDecodeError:
                    pass

            return TurnResult.from_provider_response(provider_response)
        except Exception as exc:
            self._handle_api_error(exc)
            raise

    def _stream_response(
        self,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_thinking: Optional[ThinkingCallback] = None,
        trace_prefix: str = "STREAM",
    ) -> ProviderResponse:
        """Accumulate text, tool calls, and usage from a streaming response.

        trtllm-serve emits the same OpenAI delta shape as LM Studio /
        vLLM / NIM, including ``stream_options={include_usage: true}``
        support so usage arrives in the trailing chunk.
        """
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        accumulated_text: List[str] = []
        parts: List[Part] = []
        finish_reason = FinishReason.UNKNOWN
        function_calls: List[FunctionCall] = []
        usage = TokenUsage()
        was_cancelled = False

        tool_call_accumulators: Dict[int, Dict[str, Any]] = {}

        def flush_text_block():
            nonlocal accumulated_text
            if accumulated_text:
                text = "".join(accumulated_text)
                parts.append(Part.from_text(text))
                accumulated_text = []

        def flush_tool_calls():
            nonlocal tool_call_accumulators
            for idx in sorted(tool_call_accumulators.keys()):
                tc = tool_call_accumulators[idx]
                func_name = tc.get("function", {}).get("name")
                if func_name:
                    try:
                        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    tool_id = tc.get("id")
                    original_name = get_original_tool_name(func_name)
                    fc = FunctionCall(id=tool_id, name=original_name, args=args)
                    parts.append(Part.from_function_call(fc))
                    function_calls.append(fc)
            tool_call_accumulators.clear()

        try:
            self._trace(f"{trace_prefix}_START")
            chunk_count = 0
            response_stream = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                **kwargs,
            )

            for chunk in response_stream:
                if cancel_token and cancel_token.is_cancelled:
                    was_cancelled = True
                    finish_reason = FinishReason.CANCELLED
                    break

                if not chunk.choices:
                    if chunk.usage:
                        usage = TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens or 0,
                            output_tokens=chunk.usage.completion_tokens or 0,
                            total_tokens=chunk.usage.total_tokens or 0,
                            cache_read_tokens=_extract_cache_tokens(chunk.usage),
                        )
                        if on_usage_update and usage.total_tokens > 0:
                            on_usage_update(usage)
                    continue

                for choice in chunk.choices:
                    delta = choice.delta
                    if not delta:
                        if choice.finish_reason:
                            finish_reason = map_finish_reason(choice.finish_reason)
                        continue

                    if delta.content:
                        chunk_count += 1
                        accumulated_text.append(delta.content)
                        on_chunk(delta.content)

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_call_accumulators:
                                tool_call_accumulators[idx] = {
                                    "id": tc_delta.id,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            acc = tool_call_accumulators[idx]
                            if tc_delta.id:
                                acc["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    acc["function"]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    acc["function"]["arguments"] += tc_delta.function.arguments

                    if choice.finish_reason:
                        finish_reason = map_finish_reason(choice.finish_reason)

                if chunk.usage:
                    usage = TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                        total_tokens=chunk.usage.total_tokens or 0,
                        cache_read_tokens=_extract_cache_tokens(chunk.usage),
                    )
                    if on_usage_update and usage.total_tokens > 0:
                        on_usage_update(usage)

            self._trace(
                f"{trace_prefix}_END chunks={chunk_count} finish_reason={finish_reason}"
            )

        except Exception as exc:
            if cancel_token and cancel_token.is_cancelled:
                was_cancelled = True
                finish_reason = FinishReason.CANCELLED
            else:
                raise

        flush_text_block()
        flush_tool_calls()

        if function_calls and not was_cancelled:
            finish_reason = FinishReason.TOOL_USE

        return ProviderResponse(
            parts=parts,
            usage=usage,
            finish_reason=finish_reason,
            raw=None,
            thinking=None,
        )

    # ==================== Error Handling ====================

    def _handle_api_error(self, error: Exception) -> None:
        """Map OpenAI SDK exceptions to TensorRT-LLM-specific error types.

        Distinguishes connection errors from auth/model errors so the
        reliability layer can classify retryability correctly.
        """
        openai = get_openai_module()

        if isinstance(error, openai.AuthenticationError):
            raise TensorRTLLMAuthenticationError(original_error=str(error)) from error

        if isinstance(error, openai.NotFoundError):
            raise TensorRTLLMModelNotFoundError(
                model=self._model_name or "unknown"
            ) from error

        if isinstance(error, openai.APIConnectionError):
            raise TensorRTLLMConnectionError(self._host, str(error)) from error

        # Other APIStatusError subclasses fall through — the caller sees
        # the original exception, which already carries status/body.

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """Heuristic token count (~4 chars/token).

        trtllm-serve does not expose a tokenization endpoint via its
        OpenAI-compatible surface.
        """
        return len(content) // 4

    def get_context_limit(self) -> int:
        """Return the context window size for the currently connected model.

        Priority:
            1. Explicit ``context_length`` override from profile/env
            2. Conservative default (8192)

        Note: trtllm-serve's ``GET /v1/models`` does not surface
        per-engine context length (``max_seq_len`` is fixed at engine
        build time and is not echoed in the OpenAI catalog response).
        Long-context engines should set ``TENSORRT_LLM_CONTEXT_LENGTH``
        or ``plugin_configs.tensorrt_llm.context_length``.
        """
        if self._context_length_override:
            return self._context_length_override
        return DEFAULT_CONTEXT_LENGTH

    def get_token_usage(self) -> TokenUsage:
        """Token usage from the most recent completion."""
        return self._last_usage

    # ==================== Capabilities ====================

    def supports_structured_output(self) -> bool:
        """trtllm-serve accepts ``response_format={'type': 'json_object'}``."""
        return True

    def supports_streaming(self) -> bool:
        """Streaming works through the OpenAI-compatible endpoint."""
        return True

    def supports_stop(self) -> bool:
        """Streaming responses can be interrupted via ``cancel_token``."""
        return True

    def supports_thinking(self) -> bool:
        """trtllm-serve doesn't expose thinking/reasoning content on /v1."""
        return False

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """No-op — thinking is not exposed through trtllm-serve's /v1 API."""
        pass

    # ==================== Error Classification ====================

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """Classify exceptions for the reliability layer's retry policy."""
        if isinstance(exc, TensorRTLLMConnectionError):
            return {"transient": True, "rate_limit": False, "infra": True}
        return None

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        """trtllm-serve does not emit retry-after hints."""
        return None

    # ==================== Static Auth Helpers ====================

    @staticmethod
    def login(on_message=None) -> None:
        """No-op — TensorRT-LLM does not use interactive auth."""
        if on_message:
            on_message(
                "trtllm-serve has no built-in auth. If your deployment is "
                "fronted by an auth proxy, set TENSORRT_LLM_API_TOKEN."
            )


def create_provider() -> TensorRTLLMProvider:
    """Factory function consumed by the provider discovery machinery."""
    return TensorRTLLMProvider()
