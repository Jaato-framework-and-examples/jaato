"""LM Studio provider using the OpenAI-compatible API + native load endpoint.

LM Studio's local server exposes three relevant surfaces:

- Chat:   ``POST /v1/chat/completions``   (OpenAI-compatible)
- List:   ``GET  /api/v0/models``          (native, includes ``max_context_length``)
- Load:   ``POST /api/v1/models/load``     (native, load-time configuration)

Chat completions run through the OpenAI SDK exactly like ``zhipuai_openai``
and ``nim``.  The load endpoint is invoked during ``connect()`` when the
session profile supplies ``config.extra['load']`` — the dict is POSTed as
the request body verbatim (LM Studio's load params already use snake_case),
so adding new LM Studio parameters requires no provider change.

When ``load`` is absent, the provider operates in **passive mode** — it
relies on whatever model the user has already loaded in LM Studio.  This
matches the Ollama provider's behaviour.

Authentication is optional: LM Studio typically runs unauthenticated on
localhost, but when ``Require API Token`` is enabled the token is read
from ``LMSTUDIO_API_TOKEN`` and sent as a bearer.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import httpx

# Reuse NIM's OpenAI lazy imports and converters — identical SDK, identical wire format.
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
    LMStudioAuthenticationError,
    LMStudioConnectionError,
    LMStudioLoadError,
    LMStudioModelNotFoundError,
)


logger = logging.getLogger(__name__)


def _extract_cache_tokens(usage_obj) -> Optional[int]:
    """Extract cached_tokens from an OpenAI usage object, if present.

    LM Studio reports cached tokens in the same shape as OpenAI:
    ``usage.prompt_tokens_details.cached_tokens``.  Returns ``None`` when
    the model/server doesn't surface caching stats.
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


class LMStudioProvider:
    """LM Studio provider: OpenAI-compatible chat + native load-control.

    Lifecycle:
        1. ``__init__()``              — no state yet
        2. ``initialize(config)``      — resolve host, auth, create OpenAI client,
                                         verify server reachability
        3. ``connect(model)``          — verify model exists; if
                                         ``config.extra['load']`` is set, POST it
                                         to ``/api/v1/models/load`` to reconfigure
                                         the in-memory model
        4. ``complete(messages, ...)`` — stateless chat
        5. ``shutdown()``              — release the OpenAI client

    Load parameters pass through as a raw dict.  LM Studio accepts
    snake_case keys natively (``context_length``, ``eval_batch_size``,
    ``flash_attention``, ``num_experts``, ``offload_kv_cache_to_gpu``,
    ``echo_load_config``), so no translation table is maintained —
    adding new knobs as LM Studio ships them requires no code change here.
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

        # Per-model context length learned from /api/v0/models during connect().
        self._discovered_context_length: Optional[int] = None

        # Load configuration (passthrough dict).  None means "passive mode":
        # do not invoke /api/v1/models/load during connect().
        self._load_config: Optional[Dict[str, Any]] = None

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
            prefix = "lmstudio:main"
        elif self._agent_name:
            prefix = f"lmstudio:subagent:{self._agent_name}"
        else:
            prefix = f"lmstudio:subagent:{self._agent_id}"
        provider_trace(prefix, msg)

    @property
    def name(self) -> str:
        """Provider identifier — used as the key in ``plugin_configs``."""
        return "lmstudio"

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider.

        Reads host / optional token / context override / load dict from
        ``config.extra`` (populated from the session profile's
        ``plugin_configs['lmstudio']``), then verifies the LM Studio
        server is reachable.

        Args:
            config: ``ProviderConfig`` whose ``extra`` dict may contain:

                - ``host`` (str): Override ``LMSTUDIO_HOST``.
                - ``context_length`` (int): Override context window size.
                - ``api_token`` (str): Bearer token when LM Studio requires it.
                - ``load`` (dict): Passthrough payload for
                  ``POST /api/v1/models/load``.  When set, the endpoint is
                  invoked during ``connect()``; when absent, the provider
                  uses whatever model is already loaded (passive mode).

        Raises:
            LMStudioConnectionError: Server not reachable.
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

        load = config.extra.get("load")
        if load is not None and not isinstance(load, dict):
            raise ValueError(
                f"lmstudio load config must be a dict, got {type(load).__name__}"
            )
        self._load_config = load

        self._auth_info = (
            f"local ({self._host}, bearer)"
            if self._api_token
            else f"local ({self._host})"
        )
        self._trace(f"[INIT] host={self._host} load_params={'yes' if load else 'no'}")

        self._client = self._create_client()
        self._verify_connectivity()
        self._trace("[INIT] Initialization complete")

    def _create_client(self) -> "OpenAI":
        """Build the OpenAI client pointing at LM Studio's /v1 endpoint.

        LM Studio accepts any non-empty API key, but we forward the real
        bearer token when one is configured so servers with ``Require API
        Token`` enabled also work.
        """
        client_class = get_openai_client_class()
        return client_class(
            base_url=f"{self._host}/v1",
            api_key=self._api_token or "lm-studio",
        )

    def _auth_headers(self) -> Dict[str, str]:
        """Return auth headers for direct ``httpx`` calls to the native API."""
        headers = {"Content-Type": "application/json"}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"
        return headers

    def _verify_connectivity(self) -> None:
        """Confirm the LM Studio server is reachable before any real call."""
        try:
            response = httpx.get(
                f"{self._host}/api/v0/models",
                headers=self._auth_headers(),
                timeout=5,
            )
            response.raise_for_status()
        except httpx.ConnectError:
            raise LMStudioConnectionError(self._host)
        except httpx.TimeoutException:
            raise LMStudioConnectionError(self._host, "Connection timed out")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise LMStudioAuthenticationError(original_error=str(exc))
            raise LMStudioConnectionError(self._host, str(exc))
        except httpx.HTTPError as exc:
            raise LMStudioConnectionError(self._host, str(exc))

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None,
    ) -> bool:
        """Check that LM Studio is reachable (and authorised if token is set).

        Must work before ``initialize()``: it does not touch
        ``self._client``.
        """
        host = resolve_host()
        try:
            response = httpx.get(
                f"{host}/api/v0/models",
                headers={
                    "Authorization": f"Bearer {resolve_api_token()}",
                } if resolve_api_token() else {},
                timeout=5,
            )
            response.raise_for_status()
            if on_message:
                on_message(f"Connected to LM Studio at {host}")
            return True
        except httpx.HTTPError as exc:
            if on_message:
                on_message(f"Cannot connect to LM Studio at {host}: {exc}")
            return False

    def shutdown(self) -> None:
        """Close the OpenAI client."""
        if self._client:
            self._client.close()
        self._client = None
        self._model_name = None

    def get_auth_info(self) -> str:
        """Short human-readable description of auth state."""
        return self._auth_info or "LM Studio (local)"

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Select the model for subsequent ``complete()`` calls.

        Verifies the model exists in LM Studio's catalog, then — if the
        profile supplied ``config.extra['load']`` — POSTs that dict to
        ``/api/v1/models/load`` to load/reconfigure the model with the
        requested parameters.  Without a ``load`` dict the provider is
        passive: it assumes the user has already loaded the model via
        the LM Studio UI or ``lms load``.

        Args:
            model: Model identifier as it appears in LM Studio
                (e.g. ``openai/gpt-oss-20b``, ``qwen/qwen2.5-coder-14b``).
            skip_model_test: Skip the GET-models validation call.  Load
                control is still invoked when requested.

        Raises:
            LMStudioModelNotFoundError: Model is not present in LM Studio.
            LMStudioLoadError: ``/api/v1/models/load`` returned a non-2xx.
        """
        if not skip_model_test:
            catalog = self._fetch_catalog()
            if catalog and model not in {entry["id"] for entry in catalog}:
                raise LMStudioModelNotFoundError(
                    model, available=[entry["id"] for entry in catalog],
                )
            # Remember the native max_context_length for get_context_limit().
            for entry in catalog or []:
                if entry["id"] == model:
                    max_ctx = entry.get("max_context_length")
                    if isinstance(max_ctx, int) and max_ctx > 0:
                        self._discovered_context_length = max_ctx
                    break

        self._model_name = model

        if self._load_config is not None:
            self._load_model(model, self._load_config)

        logger.info(
            "Connected to LM Studio model: %s (context=%d, load_applied=%s)",
            model, self.get_context_limit(), bool(self._load_config),
        )

    def _load_model(self, model: str, load_params: Dict[str, Any]) -> None:
        """POST the profile's load config to ``/api/v1/models/load``.

        The body is the load_params dict with ``model`` injected; any
        user-supplied ``model`` key is ignored to avoid inconsistency
        with ``connect()``'s argument.
        """
        body = {**load_params, "model": model}
        self._trace(f"[LOAD] POST {self._host}/api/v1/models/load body_keys={sorted(body.keys())}")
        try:
            response = httpx.post(
                f"{self._host}/api/v1/models/load",
                json=body,
                headers=self._auth_headers(),
                # Loading a large model can legitimately take a while
                # (disk read + KV-cache init + GPU transfer).
                timeout=600.0,
            )
        except httpx.HTTPError as exc:
            raise LMStudioConnectionError(self._host, f"load failed: {exc}") from exc

        if response.status_code >= 400:
            raise LMStudioLoadError(
                model=model,
                status_code=response.status_code,
                body=response.text,
                load_config=load_params,
            )
        self._trace(f"[LOAD] status={response.status_code}")

    def _fetch_catalog(self) -> List[Dict[str, Any]]:
        """Query ``GET /api/v0/models`` and return the raw ``data`` array.

        Each entry carries ``id``, ``type``, ``state``, and
        ``max_context_length`` among other fields.
        """
        try:
            response = httpx.get(
                f"{self._host}/api/v0/models",
                headers=self._auth_headers(),
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            return payload.get("data", [])
        except httpx.HTTPError as exc:
            logger.warning("Failed to list LM Studio models: %s", exc)
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
        """List models available in LM Studio.

        Queries the native ``/api/v0/models`` endpoint.  Returns an empty
        list if the server is unreachable — callers can surface that as a
        clear error instead of a fake catalog.
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
        """Run one stateless chat completion through LM Studio's /v1 endpoint.

        Streaming and non-streaming paths mirror ``zhipuai_openai``'s
        implementation — LM Studio speaks OpenAI's wire format faithfully.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("Provider not connected. Call initialize() and connect() first.")

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

        Mirrors ``zhipuai_openai``'s streaming loop: LM Studio emits the
        same OpenAI delta shape.
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

            self._trace(f"{trace_prefix}_END chunks={chunk_count} finish_reason={finish_reason}")

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
        """Map OpenAI SDK exceptions to LM Studio-specific error types.

        Distinguishes connection errors from auth/model errors so the
        reliability layer can classify retryability correctly.
        """
        openai = get_openai_module()

        if isinstance(error, openai.AuthenticationError):
            raise LMStudioAuthenticationError(original_error=str(error)) from error

        if isinstance(error, openai.NotFoundError):
            raise LMStudioModelNotFoundError(model=self._model_name or "unknown") from error

        if isinstance(error, openai.APIConnectionError):
            raise LMStudioConnectionError(self._host, str(error)) from error

        # Other APIStatusError subclasses fall through — the caller sees
        # the original exception, which already carries status/body.

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """Heuristic token count (~4 chars/token).

        LM Studio does not expose a tokenization endpoint via its
        OpenAI-compatible surface.
        """
        return len(content) // 4

    def get_context_limit(self) -> int:
        """Return the context window size for the currently connected model.

        Priority:
            1. Explicit ``context_length`` override from profile/env
            2. ``max_context_length`` discovered from ``/api/v0/models``
            3. Conservative default (8192)
        """
        if self._context_length_override:
            return self._context_length_override
        if self._discovered_context_length:
            return self._discovered_context_length
        return DEFAULT_CONTEXT_LENGTH

    def get_token_usage(self) -> TokenUsage:
        """Token usage from the most recent completion."""
        return self._last_usage

    # ==================== Capabilities ====================

    def supports_structured_output(self) -> bool:
        """LM Studio accepts ``response_format={'type': 'json_object'}``."""
        return True

    def supports_streaming(self) -> bool:
        """Streaming works through the OpenAI-compatible endpoint."""
        return True

    def supports_stop(self) -> bool:
        """Streaming responses can be interrupted via ``cancel_token``."""
        return True

    def supports_thinking(self) -> bool:
        """LM Studio doesn't expose thinking/reasoning content on /v1."""
        return False

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """No-op — thinking is not exposed through LM Studio's /v1 API."""
        pass

    # ==================== Error Classification ====================

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """Classify exceptions for the reliability layer's retry policy."""
        if isinstance(exc, LMStudioConnectionError):
            return {"transient": True, "rate_limit": False, "infra": True}
        if isinstance(exc, LMStudioLoadError):
            # A load failure is deterministic given the config — don't retry.
            return {"transient": False, "rate_limit": False, "infra": False}
        return None

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        """LM Studio does not emit retry-after hints."""
        return None

    # ==================== Static Auth Helpers ====================

    @staticmethod
    def login(on_message=None) -> None:
        """No-op — LM Studio does not use interactive auth."""
        if on_message:
            on_message(
                "LM Studio runs locally; no login required. "
                "If 'Require API Token' is enabled, set LMSTUDIO_API_TOKEN."
            )


def create_provider() -> LMStudioProvider:
    """Factory function consumed by the provider discovery machinery."""
    return LMStudioProvider()
