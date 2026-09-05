"""Shared base for OpenAI-compatible model providers.

``OpenAICompatProvider`` owns the transport-level machinery that the providers
fronting an OpenAI-compatible chat-completions API (nim, nebius, vllm, lmstudio,
tensorrt_llm, triton, zhipuai_openai) otherwise copy-paste: the streaming loop,
the completion skeleton, OpenAI client construction, error mapping, and the
capability / token / trace boilerplate.

Provider-specific concerns are hooks:

- **credentials / base_url** — ``_resolve_credentials(config)`` sets
  ``self._base_url`` + ``self._api_key`` (the canonical attributes
  ``_create_client`` reads) and validates auth.  Subclasses with env-var names
  like ``host``/``token`` normalise into ``_base_url``/``_api_key`` here.
- **context window** — ``_detect_context(config) -> Optional[int]`` (the
  provider's resolution tier); ``_context_error_message()`` for the fail-loud
  "no hardcoded fallback" message.
- **error taxonomy** — the ``_ERR_*`` class attributes name the provider's
  exception classes; the nim-family shares the parameterized
  ``_handle_api_error`` / ``classify_error`` / ``get_retry_after``, while a
  family with a different taxonomy (the local-host providers) overrides them.
- **identity** — ``name`` (property), plus ``verify_auth`` / ``get_auth_info``
  / ``list_models``.
- ``REASONING_CAPABLE_MODELS`` — models exposing ``reasoning_content``.

Subclasses are stateless w.r.t. conversation history (the session owns the
message list and passes it to ``complete()`` each call).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from base64 import b64decode as _b64decode
from binascii import Error as BinasciiError

from ._lazy import get_openai_client_class, get_openai_module

if TYPE_CHECKING:
    from openai import OpenAI

from ..base import (
    ModalityCapabilityMixin,
    FunctionCallDetectedCallback,
    MediaDelta,
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
    ToolSchema,
    TokenUsage,
    ThinkingConfig,
    TurnResult,
    normalize_inclusive_usage,
    parse_tool_call_arguments,
    require_terminated_stream,
    resolve_tool_use_finish,
)
from .converters import (
    get_original_tool_name,
    history_to_openai,
    map_finish_reason,
    response_from_openai,
    tool_schemas_to_openai,
)
from .._prose_tools import (
    augment_system_with_tools,
    messages_to_prose_chat,
    read_prose_tool_calls_quirk,
    rewrite_prose_tool_calls,
)
from shared.tool_id_map import tool_choice_to_wire

logger = logging.getLogger(__name__)


def _extract_audio_delta(delta: Any) -> Optional[Tuple[bytes, str]]:
    """Pull ``(raw_bytes, transcript)`` out of a streaming ``delta.audio``.

    Returns ``None`` when the delta carries no audio -- the case for every
    text-only provider and every text chunk, so this is the cheap common
    path.

    Defensive by necessity.  ``audio`` is absent from OpenAI's published
    OpenAPI schema for the streaming delta, and so from the generated SDK
    types too, meaning it may surface as an attribute, inside
    ``model_extra``, or as a plain dict depending on the client.  It is
    therefore probed rather than accessed, and a malformed or undecodable
    payload yields ``None`` instead of raising: one bad audio chunk must
    not abort a turn that is otherwise streaming fine.
    """
    audio = getattr(delta, "audio", None)
    if audio is None and isinstance(delta, dict):
        audio = delta.get("audio")
    if audio is None:
        extra = getattr(delta, "model_extra", None)
        if isinstance(extra, dict):
            audio = extra.get("audio")
    if audio is None:
        return None

    if isinstance(audio, dict):
        encoded = audio.get("data")
        transcript = audio.get("transcript") or ""
    else:
        encoded = getattr(audio, "data", None)
        transcript = getattr(audio, "transcript", None) or ""

    if not encoded:
        return None
    try:
        raw = _b64decode(encoded)
    except (BinasciiError, ValueError, TypeError):
        logger.warning("Discarding an undecodable audio delta")
        return None
    if not raw:
        return None
    return raw, transcript


class OpenAICompatProvider(ModalityCapabilityMixin):
    """Base class for OpenAI-compatible chat-completions providers.

    See the module docstring for the hook contract.  Lifecycle:

        1. ``__init__()`` — create instance (no connections yet)
        2. ``initialize(config)`` — resolve credentials, create OpenAI client
        3. ``connect(model)`` — set the active model
        4. ``complete(messages, ...)`` — stateless completion
        5. ``shutdown()`` — release resources
    """

    # --- subclass-provided exception classes (parameterize the shared error
    # mapping for the nim-family; local-host providers override the handlers).
    _ERR_AUTHENTICATION: type
    _ERR_RATE_LIMIT: type
    _ERR_MODEL_NOT_FOUND: type
    _ERR_CONTEXT_LIMIT: type
    _ERR_INFRASTRUCTURE: type

    # OpenAI Chat Completions body fields forwarded from
    # ``plugin_configs.<provider>.api_params``.  Allowlisted (not blind
    # passthrough) so a typo'd / unsupported key surfaces as a profile warning
    # rather than an opaque OpenAI 400.  Subclasses may override.
    # ``modalities`` here is OpenAI's OUTPUT selector (``["text","audio"]``)
    # and is NOT the jaato tier key of the same name, which declares INPUT
    # roles.  They coexist in one profile -- ``api_params.modalities`` vs
    # ``model_tiers.<tier>.modalities`` -- so the layer a key sits under is
    # what disambiguates them.  ``audio`` is its companion
    # (``{"voice": ..., "format": ...}``); OpenAI requires both together,
    # and both were previously dropped here with a warning, which made
    # audio output unrequestable through any OpenAI-compatible provider.
    # OpenAI streams audio as headerless pcm16 -- 24 kHz mono signed
    # 16-bit little-endian -- and ONLY pcm16: requesting wav/mp3 together
    # with ``stream=true`` is rejected upstream.  Spelled out in the mime
    # type because a headerless payload carries no way to recover the rate
    # or channel count, and a consumer that guesses wrong plays noise.
    STREAM_AUDIO_MIME = "audio/pcm;rate=24000;channels=1;encoding=s16le"

    _FORWARDED_API_PARAMS = frozenset({
        "temperature", "top_p", "max_tokens", "tool_choice",
        "parallel_tool_calls", "frequency_penalty", "presence_penalty",
        "seed", "stop", "modalities", "audio",
    })

    # Models known to expose reasoning/thinking via ``reasoning_content``.
    REASONING_CAPABLE_MODELS: List[str] = []

    def __init__(self) -> None:
        """Initialize the provider (not yet connected)."""
        self._client: Optional[OpenAI] = None
        self._model_name: Optional[str] = None

        # Credentials (canonical attrs; ``_resolve_credentials`` populates them)
        self._api_key: Optional[str] = None
        self._base_url: str = ""

        # Per-call accounting (NOT conversation state)
        self._last_usage: TokenUsage = TokenUsage()
        self._context_length: int = 0

        # OpenAI Chat Completions body fields from
        # plugin_configs.<provider>.api_params (filtered to
        # _FORWARDED_API_PARAMS) + opaque extra_body, forwarded on each call.
        self._api_params: Dict[str, Any] = {}
        self._extra_body: Optional[Dict[str, Any]] = None

        # Quirk: prose_tool_calls (opt-in via profile.quirks).  When set,
        # the model is assumed unable to emit native tool calls: the tools
        # array is withheld, schemas are prompt-injected (hashed wire ids),
        # tool traffic in history is replayed as text, and fenced tool_call
        # blocks in the response are parsed back into FunctionCall parts.
        # See shared/plugins/model_provider/_prose_tools.py.
        self._prose_tool_calls: bool = False

        # Thinking/reasoning configuration
        self._enable_thinking: bool = True

        # Agent context for trace identification
        self._agent_type: str = "main"
        self._agent_name: Optional[str] = None
        self._agent_id: str = "main"

        # Stashed config (post-init helpers reuse workspace_path/config_root)
        self._config: Optional[ProviderConfig] = None

    # ==================== Identity / tracing ====================

    @property
    def name(self) -> str:
        """Provider identifier (e.g. ``"nim"``).  Subclasses must override."""
        raise NotImplementedError

    def set_agent_context(
        self,
        agent_type: str = "main",
        agent_name: Optional[str] = None,
        agent_id: str = "main",
    ) -> None:
        """Set agent context for trace identification.

        Args:
            agent_type: Type of agent ("main" or "subagent").
            agent_name: Optional name for the agent.
            agent_id: Unique identifier for the agent instance.
        """
        self._agent_type = agent_type
        self._agent_name = agent_name
        self._agent_id = agent_id

    def _trace(self, msg: str) -> None:
        """Write a trace message to the per-agent provider log."""
        from shared.trace import provider_trace
        slug = self.name
        if self._agent_type == "main":
            prefix = f"{slug}:main"
        elif self._agent_name:
            prefix = f"{slug}:subagent:{self._agent_name}"
        else:
            prefix = f"{slug}:subagent:{self._agent_id}"
        provider_trace(prefix, msg)

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Resolve credentials + context window and create the OpenAI client.

        Skeleton shared by all OpenAI-compat providers; the provider-specific
        parts are the ``_resolve_credentials`` / ``_detect_context`` /
        ``_context_error_message`` hooks.

        Raises:
            Whatever ``_resolve_credentials`` raises on missing auth, or
            ``ValueError`` (``_context_error_message``) if the context window
            cannot be resolved (no hardcoded fallback).
        """
        if config is None:
            config = ProviderConfig()
        self._config = config

        # Hook: sets self._base_url + self._api_key and validates auth.
        self._resolve_credentials(config)

        # Common: api_params (allowlisted) + extra_body from the profile.
        self._read_api_params(config)

        # Common: the prose_tool_calls model quirk (profile.quirks).
        self._prose_tool_calls = read_prose_tool_calls_quirk(
            config.extra, provider=self.name)

        # Hook: provider's context-window resolution.  Runs after the auth
        # check so an auth failure surfaces first.
        self._resolve_context(config)

        self._client = self._create_client()
        self._trace(f"[INIT] client created, base_url={self._base_url}")

    def _resolve_credentials(self, config: ProviderConfig) -> None:
        """Hook: populate ``self._base_url`` + ``self._api_key``, validate auth.

        Subclasses normalise their env-var / profile credentials into the two
        canonical attributes that ``_create_client`` reads, and raise on
        missing-and-required auth.
        """
        raise NotImplementedError

    def _detect_context(self, config: ProviderConfig) -> Optional[int]:
        """Hook: resolve the context window (detect → profile → env → None)."""
        raise NotImplementedError

    def _context_error_message(self) -> str:
        """Hook: the fail-loud message when the context window is unresolved."""
        raise NotImplementedError

    def _resolve_context(self, config: ProviderConfig) -> None:
        """Resolve + validate the context window (default: at-init, fail-loud).

        Default (nim-family): ``_detect_context`` then fail-loud via
        ``_context_error_message``.  Providers that bootstrap the window LATER
        — nebius reads it from the catalog at ``connect()`` once the model is
        known — override this to stash their knobs instead.
        """
        self._context_length = self._detect_context(config)
        if not self._context_length:
            raise ValueError(self._context_error_message())

    def _read_api_params(self, config: ProviderConfig) -> None:
        """Read ``api_params`` (allowlisted) + ``extra_body`` from the profile.

        Lifted so every OpenAI-compat provider uniformly honors
        ``plugin_configs.<provider>.api_params`` (temperature / tool_choice /
        max_tokens / ...) and ``.extra_body`` (opaque passthrough for
        endpoint-specific request-body extensions the OpenAI ``create()``
        signature doesn't name — e.g. guided decoding, cache_salt).
        """
        api_params = config.extra.get("api_params") or {}
        if api_params:
            if not isinstance(api_params, dict):
                raise TypeError(
                    f"{self.name} 'api_params' config must be a dict of OpenAI "
                    f"Chat Completions fields, got {type(api_params).__name__}"
                )
            self._api_params = {
                k: v for k, v in api_params.items()
                if k in self._FORWARDED_API_PARAMS
            }
            dropped = set(api_params) - self._FORWARDED_API_PARAMS
            if dropped:
                logger.warning(
                    "%s api_params: ignoring unsupported key(s) %s; forwarded "
                    "fields are %s",
                    self.name, sorted(dropped), sorted(self._FORWARDED_API_PARAMS),
                )
        extra_body = config.extra.get("extra_body")
        if extra_body is not None:
            if not isinstance(extra_body, dict):
                raise TypeError(
                    f"{self.name} 'extra_body' config must be a dict, got "
                    f"{type(extra_body).__name__}"
                )
            self._extra_body = extra_body

    def _create_client(self) -> "OpenAI":
        """Create the OpenAI client for ``self._base_url`` / ``self._api_key``.

        The OpenAI SDK requires a key string even for keyless self-hosted
        endpoints, so a placeholder is substituted when no key is set.
        """
        client_class = get_openai_client_class()
        api_key = self._api_key or "not-needed"
        return client_class(base_url=self._base_url, api_key=api_key)

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None,
        config: Optional["ProviderConfig"] = None,
    ) -> bool:
        """Hook: verify credentials exist WITHOUT touching ``self._client``.

        Called on a fresh, uninitialized instance before a session is created
        (see the plugin guide's ``verify_auth`` contract).  Subclasses must
        implement.
        """
        raise NotImplementedError

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._client:
            self._client.close()
        self._client = None
        self._model_name = None

    def get_auth_info(self) -> str:
        """Hook: short human-readable description of the credential source."""
        raise NotImplementedError

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Set the model to use.  Validation is deferred to the first API call.

        Args:
            model: Model ID.
            skip_model_test: Accepted for protocol compatibility; this provider
                already defers validation to the first API call.
        """
        self._model_name = model

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected and ready."""
        return self._client is not None and self._model_name is not None

    @property
    def model_name(self) -> Optional[str]:
        """Get the current model name."""
        return self._model_name

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """Hook: list available models (provider-specific).  Default: none."""
        return []

    # ==================== Stateless Completion ====================

    def _apply_api_params(
        self, kwargs: Dict[str, Any], tool_choice: Optional[Any],
    ) -> None:
        """Forward profile ``api_params`` (+ per-call ``tool_choice``) into the
        ``chat.completions.create`` kwargs.

        Profile fields (already filtered to :data:`_FORWARDED_API_PARAMS` at
        init) apply to every call; a per-call ``tool_choice`` overrides the
        profile's.  ``tool_choice`` is dropped when no tools are present this
        turn (OpenAI rejects ``tool_choice`` without ``tools``).  A name-bearing
        ``tool_choice`` is mapped through :func:`tool_choice_to_wire` — tool
        names are hashed on the wire, so forcing a tool by its human name would
        otherwise be rejected ("Tool X not found in tools list"); string forms
        ("required"/"auto") pass through.
        """
        for key, value in self._api_params.items():
            kwargs[key] = value
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if "tool_choice" in kwargs and "tools" not in kwargs:
            kwargs.pop("tool_choice")
        if "tool_choice" in kwargs:
            kwargs["tool_choice"] = tool_choice_to_wire(kwargs["tool_choice"])

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
        tool_choice: Optional[Dict[str, Any]] = None,
    ) -> TurnResult:
        """Stateless completion: convert messages, call the API, return the result.

        The caller (session) owns the message list and passes it in full each
        call — this method holds no conversation state.  Returns
        ``TurnResult.from_provider_response(r)`` on success and **raises**
        transient errors for ``with_retry``.

        Raises:
            RuntimeError: If the provider is not initialized/connected.
        """
        if not self._client or not self._model_name:
            raise RuntimeError(
                "Provider not connected. Call initialize() and connect() first."
            )

        # Quirk path: models that cannot emit native tool calls get the
        # text-encoded protocol — schemas prompt-injected, tool traffic in
        # history replayed as prose, no tools array on the wire.
        prose_mode = bool(self._prose_tool_calls and tools)

        # Build OpenAI-format messages from explicit parameters
        openai_messages: List[Dict[str, Any]] = []
        if prose_mode:
            system_text = augment_system_with_tools(system_instruction, tools)
            if system_text:
                openai_messages.append({"role": "system",
                                        "content": system_text})
            openai_messages.extend(messages_to_prose_chat(list(messages)))
        else:
            if system_instruction:
                openai_messages.append({"role": "system",
                                        "content": system_instruction})
            openai_messages.extend(history_to_openai(list(messages)))

        # Build kwargs
        kwargs: Dict[str, Any] = {}
        if tools and not prose_mode:
            openai_tools = tool_schemas_to_openai(tools)
            if openai_tools:
                kwargs["tools"] = openai_tools
        if response_schema:
            kwargs["response_format"] = {"type": "json_object"}
        # Profile api_params (+ per-call tool_choice override) into the shared
        # kwargs so BOTH the streaming and batch paths forward them.
        self._apply_api_params(kwargs, tool_choice)

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
                # Non-streaming cache-hit count (the streaming path sets this
                # per-chunk); response_from_openai doesn't carry it.
                cached = self._extract_cache_tokens(getattr(response, "usage", None))
                if cached is not None and provider_response.usage is not None:
                    provider_response.usage.cache_read_tokens = cached
                    # ...and then take it back OUT of prompt_tokens, which
                    # on this wire counted it.  See ``TokenUsage``.
                    normalize_inclusive_usage(provider_response.usage)

            # Prose-mode counterpart of the native tool-call flush: parse
            # fenced tool_call blocks out of the text into FunctionCall
            # parts (no-op on cancelled / call-free responses).
            if prose_mode:
                provider_response = rewrite_prose_tool_calls(
                    provider_response, call_id_prefix=self.name)

            # Per-call accounting (NOT conversation state)
            self._last_usage = provider_response.usage

            # Parse structured output if schema was requested
            text = provider_response.get_text()
            if response_schema and text:
                try:
                    provider_response.structured_output = json.loads(text)
                except json.JSONDecodeError:
                    pass

            return TurnResult.from_provider_response(provider_response)
        except Exception as e:
            self._handle_api_error(e)
            raise

    @staticmethod
    def _extract_cache_tokens(usage: Any) -> Optional[int]:
        """OpenAI-compatible cache-hit count (``usage.prompt_tokens_details
        .cached_tokens``), or None when absent / zero (no cache hit).

        Lets cache hit-rate and $ savings be measured uniformly across the
        fleet — previously a per-provider copy (and missing entirely on nim).

        This count is a SUBSET of the same usage object's
        ``prompt_tokens``.  Callers must therefore pair it with
        :func:`normalize_inclusive_usage`, or the tokens get counted on
        both sides of every downstream sum (issue #758).
        """
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details is not None else None
        # ``isinstance`` and not merely truthiness: the count is now
        # ARITHMETIC (it comes out of ``prompt_tokens``), so a field an
        # upstream sent as a string — or a test double left as a mock —
        # must read as "not reported" rather than reach the subtraction.
        return cached if isinstance(cached, int) and cached else None

    def _emit_audio_delta(
        self,
        delta: Any,
        on_chunk: StreamingCallback,
        media_sequence: int,
    ) -> int:
        """Emit one model-generated audio chunk; return the new sequence.

        Returns ``media_sequence`` unchanged when the delta carries no
        audio -- the case for every text-only provider and every text
        chunk, so the common path costs one call and a ``None`` check.

        Extracted from the streaming loop rather than inlined so that
        already-oversized function does not grow further.
        """
        audio_delta = _extract_audio_delta(delta)
        if audio_delta is None:
            return media_sequence
        raw, transcript = audio_delta
        media_sequence += 1
        on_chunk(MediaDelta(
            mime_type=self.STREAM_AUDIO_MIME,
            data=raw,
            sequence=media_sequence,
            transcript=transcript,
        ))
        return media_sequence

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
        """Core streaming loop.

        Accumulates text chunks, tool-call deltas, reasoning content, and usage
        (including cache-read tokens) from the streaming response.  Handles
        cancellation via ``cancel_token`` and closes the HTTP connection +
        httpx pool on cancel so the upstream stops generating immediately.
        """
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        accumulated_text: List[str] = []
        accumulated_thinking: List[str] = []
        parts: List[Part] = []
        finish_reason = FinishReason.UNKNOWN
        # Whether the wire ever said the turn ended.  Tracked apart from
        # ``finish_reason`` because ``map_finish_reason`` answers UNKNOWN
        # for a label it does not recognise, which is still an upstream
        # that terminated the turn (#687).
        terminal_seen = False
        function_calls: List[FunctionCall] = []
        usage = TokenUsage()
        was_cancelled = False

        # Track tool call accumulation (streaming sends tool calls in pieces)
        tool_call_accumulators: Dict[int, Dict[str, Any]] = {}

        # Monotonic index over model-generated media chunks, so a consumer
        # can spot a gap left by backpressure.  Separate from the text
        # chunk counter: they are different streams.
        media_sequence = -1

        def flush_text_block():
            """Flush accumulated text as a single Part."""
            nonlocal accumulated_text
            if accumulated_text:
                text = "".join(accumulated_text)
                parts.append(Part.from_text(text))
                accumulated_text = []

        def flush_tool_calls():
            """Flush accumulated tool calls as Parts."""
            nonlocal tool_call_accumulators
            for idx in sorted(tool_call_accumulators.keys()):
                tc = tool_call_accumulators[idx]
                func_name = tc.get("function", {}).get("name")
                if func_name:
                    # Unreadable arguments stay unreadable: the session
                    # refuses the call and tells the model, rather than
                    # executing a zero-argument call it never made (#750).
                    args, unreadable_args = parse_tool_call_arguments(
                        tc.get("function", {}).get("arguments")
                    )
                    tool_id = tc.get("id")
                    original_name = get_original_tool_name(func_name)
                    if not tool_id:
                        self._trace(f"ERROR: Missing tool call ID for {func_name}")
                    if unreadable_args is not None:
                        self._trace(
                            f"UNREADABLE_TOOL_ARGS name={original_name} "
                            f"chars={len(unreadable_args)}"
                        )
                    fc = FunctionCall(
                        id=tool_id,
                        name=original_name,
                        args=args,
                        unreadable_args=unreadable_args,
                    )
                    parts.append(Part.from_function_call(fc))
                    function_calls.append(fc)
            tool_call_accumulators.clear()

        response_stream = None
        chunk_count = 0
        try:
            self._trace(f"{trace_prefix}_START")
            response_stream = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                **kwargs,
            )

            for chunk in response_stream:
                # Check for cancellation
                if cancel_token and cancel_token.is_cancelled:
                    self._trace(f"{trace_prefix}_CANCELLED after {chunk_count} chunks")
                    was_cancelled = True
                    finish_reason = FinishReason.CANCELLED
                    break

                if not chunk.choices:
                    # Final chunk may have only usage
                    if chunk.usage:
                        usage = normalize_inclusive_usage(TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens or 0,
                            output_tokens=chunk.usage.completion_tokens or 0,
                            total_tokens=chunk.usage.total_tokens or 0,
                            cache_read_tokens=self._extract_cache_tokens(chunk.usage),
                        ))
                        self._trace(f"{trace_prefix}_USAGE prompt={usage.prompt_tokens} output={usage.output_tokens}")
                        if on_usage_update and usage.total_tokens > 0:
                            on_usage_update(usage)
                    continue

                for choice in chunk.choices:
                    delta = choice.delta
                    if not delta:
                        if choice.finish_reason:
                            terminal_seen = True
                            finish_reason = map_finish_reason(choice.finish_reason)
                        continue

                    # Extract reasoning/thinking (e.g. DeepSeek-R1)
                    if self._enable_thinking:
                        reasoning = getattr(delta, "reasoning_content", None)
                        if reasoning and isinstance(reasoning, str):
                            self._trace(f"{trace_prefix}_THINKING len={len(reasoning)}")
                            accumulated_thinking.append(reasoning)
                            if on_thinking:
                                on_thinking(reasoning)

                    # Accumulate text
                    if delta.content:
                        chunk_count += 1
                        accumulated_text.append(delta.content)
                        on_chunk(delta.content)

                    # Model-generated audio.  The OpenAI streaming shape is
                    # ``delta.audio.data`` (base64) with an optional running
                    # ``transcript``.  It is NOT in the published OpenAPI
                    # schema for the streaming delta -- and so not in the
                    # generated SDK types either -- so it arrives as an
                    # untyped extra and is read defensively rather than by
                    # attribute access.  While streaming, OpenAI emits only
                    # pcm16 (24 kHz mono s16le, headerless), which is why
                    # the mime type spells the parameters out: the payload
                    # carries no header to recover them from.
                    media_sequence = self._emit_audio_delta(
                        delta, on_chunk, media_sequence
                    )

                    # Accumulate tool calls (they come in pieces)
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_call_accumulators:
                                self._trace(f"TOOL_CALL_START idx={idx} id={tc_delta.id!r} name={getattr(tc_delta.function, 'name', '')!r}")
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

                    # Extract finish reason
                    if choice.finish_reason:
                        terminal_seen = True
                        finish_reason = map_finish_reason(choice.finish_reason)

                # Extract usage from chunk (some providers include it per-chunk)
                if chunk.usage:
                    usage = normalize_inclusive_usage(TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                        total_tokens=chunk.usage.total_tokens or 0,
                        cache_read_tokens=self._extract_cache_tokens(chunk.usage),
                    ))
                    if on_usage_update and usage.total_tokens > 0:
                        on_usage_update(usage)

            self._trace(f"{trace_prefix}_END chunks={chunk_count} finish_reason={finish_reason}")

        except Exception as e:
            self._trace(f"{trace_prefix}_ERROR {type(e).__name__}: {e}")
            if cancel_token and cancel_token.is_cancelled:
                was_cancelled = True
                finish_reason = FinishReason.CANCELLED
            else:
                raise
        finally:
            # Close the underlying HTTP connection so the upstream stops
            # generating immediately on cancel. The SDK ``Stream`` only sends
            # TCP-close at GC time, which on a cancelled turn keeps the upstream
            # billing the entire response.
            if response_stream is not None:
                try:
                    response_stream.close()
                except Exception as close_exc:  # pragma: no cover - best effort
                    self._trace(
                        f"{trace_prefix}_CLOSE_ERROR "
                        f"{type(close_exc).__name__}: {close_exc}"
                    )

            # SHAPE B (cancel-leak fix, 2026-06-09): close the openai client's
            # httpx pool when cancelled.  ``response_stream.close()`` alone does
            # NOT propagate TCP-FIN to the upstream.
            if was_cancelled and self._client is not None:
                try:
                    self._client.close()
                except Exception:  # pragma: no cover - best effort
                    pass

        # Flush remaining text and tool calls
        flush_text_block()
        flush_tool_calls()

        # TOOL_USE fills in an unreported or merely-``stop`` finish; it
        # must not displace a terminal one.  A turn that hit the output
        # cap mid-``arguments`` carries fragments, not a request — see
        # ``resolve_tool_use_finish`` and issue #745.
        finish_reason = resolve_tool_use_finish(
            finish_reason,
            has_function_calls=bool(function_calls) and not was_cancelled,
        )

        thinking = "".join(accumulated_thinking) if accumulated_thinking else None

        # A stream that stopped arriving is not a turn that finished
        # (#687).  Raises rather than returning the fragment.
        return require_terminated_stream(
            ProviderResponse(
                parts=parts,
                usage=usage,
                finish_reason=finish_reason,
                raw=None,
                thinking=thinking,
            ),
            terminal_seen=terminal_seen,
            was_cancelled=was_cancelled,
            provider=self.name,
            model=self._model_name,
            chunks=chunk_count,
        )

    # ==================== Error Handling ====================

    def _rebuild_client_after_connection_error(self) -> None:
        """Discard the HTTP client so the next attempt gets a fresh one.

        An ``APIConnectionError`` can leave the underlying ``httpx``
        transport permanently unusable.  ``with_retry`` retries the CALL,
        not the client, so without this every backoff re-runs against the
        same dead object: the ladder is guaranteed to exhaust, and the
        session stays broken for the rest of its life even after the
        network recovers.  Measured (jaato #705): a session died at 19:57
        and still failed at 20:01 while ``curl`` answered in 30ms, a direct
        provider call in another process succeeded, and a NEW session in
        the SAME daemon completed normally — the only difference being a
        freshly built client.

        Best-effort by design.  A failure to close or rebuild must not
        replace the caller's original error with a second one: the
        connection error is what the caller needs to see and classify, and
        a failed rebuild simply leaves the ladder no worse off than before
        this method existed.
        """
        try:
            if self._client is not None:
                self._client.close()
        except Exception:  # noqa: BLE001 - closing a dead transport may throw
            pass
        try:
            self._client = self._create_client()
            self._trace("[RECOVERY] client rebuilt after APIConnectionError")
        except Exception as exc:  # noqa: BLE001
            self._trace(
                f"[RECOVERY] client rebuild FAILED after APIConnectionError: "
                f"{type(exc).__name__}: {exc}"
            )

    def _handle_api_error(self, error: Exception) -> None:
        """Map OpenAI SDK exceptions to the provider's error taxonomy.

        Parameterized by the ``_ERR_*`` class attributes so the nim-family
        (identical mapping, different classes) shares this one implementation.
        Providers with a different taxonomy override it.
        """
        openai = get_openai_module()

        if isinstance(error, openai.AuthenticationError):
            raise self._ERR_AUTHENTICATION(
                original_error=str(error),
            ) from error

        if isinstance(error, openai.RateLimitError):
            retry_after = None
            response = getattr(error, "response", None)
            if response:
                retry_header = getattr(response.headers, "get", lambda *a: None)("retry-after")
                if retry_header:
                    try:
                        retry_after = float(retry_header)
                    except ValueError:
                        pass
            raise self._ERR_RATE_LIMIT(
                retry_after=retry_after,
                original_error=str(error),
            ) from error

        if isinstance(error, openai.NotFoundError):
            raise self._ERR_MODEL_NOT_FOUND(
                model=self._model_name or "unknown",
                original_error=str(error),
            ) from error

        if isinstance(error, openai.APIConnectionError):
            # Rebuild BEFORE raising: with_retry retries the call, not
            # the client, so a poisoned transport would make every
            # remaining backoff a guaranteed failure (#705).
            self._rebuild_client_after_connection_error()
            raise self._ERR_INFRASTRUCTURE(
                status_code=0,
                original_error=str(error),
            ) from error

        if isinstance(error, openai.InternalServerError):
            status_code = getattr(error, "status_code", 500)
            raise self._ERR_INFRASTRUCTURE(
                status_code=status_code,
                original_error=str(error),
            ) from error

        if isinstance(error, openai.APIStatusError):
            status_code = getattr(error, "status_code", 0)
            error_str = str(error).lower()

            # Context limit errors
            if any(x in error_str for x in ("context_length", "too large", "max size", "tokens_limit")):
                max_tokens = None
                match = re.search(r'max (?:size|tokens)[:\s]+(\d+)', error_str)
                if match:
                    max_tokens = int(match.group(1))
                raise self._ERR_CONTEXT_LIMIT(
                    model=self._model_name or "unknown",
                    max_tokens=max_tokens,
                    original_error=str(error),
                ) from error

            # 5xx infrastructure errors
            if 500 <= status_code < 600:
                raise self._ERR_INFRASTRUCTURE(
                    status_code=status_code,
                    original_error=str(error),
                ) from error

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """Estimate tokens (~4 chars/token); OpenAI-compat APIs expose no
        token-count endpoint."""
        return len(content) // 4

    def get_context_limit(self) -> int:
        """Context window size resolved at ``initialize()``."""
        return self._context_length

    def get_token_usage(self) -> TokenUsage:
        """Token usage from the last response."""
        return self._last_usage

    # ==================== Capabilities ====================

    def supports_structured_output(self) -> bool:
        """OpenAI-compatible ``response_format`` is supported."""
        return True

    def supports_streaming(self) -> bool:
        """Streaming is supported via the OpenAI-compatible API."""
        return True

    def supports_stop(self) -> bool:
        """Streaming responses can be cancelled via ``cancel_token``."""
        return True

    def supports_thinking(self) -> bool:
        """True for models known to expose ``reasoning_content``."""
        return self._is_reasoning_capable()

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """Enable/disable extraction of ``reasoning_content``."""
        self._enable_thinking = config.enabled

    def _is_reasoning_capable(self) -> bool:
        """Check if the current model exposes reasoning content."""
        if not self._model_name:
            return False
        name_lower = self._model_name.lower()
        for prefix in self.REASONING_CAPABLE_MODELS:
            if name_lower.startswith(prefix) or name_lower.endswith(prefix):
                return True
        return False

    # ==================== Error Classification for Retry ====================

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """Classify an exception for retry (parameterized by ``_ERR_*``)."""
        if isinstance(exc, self._ERR_RATE_LIMIT):
            return {"transient": True, "rate_limit": True, "infra": False}

        if isinstance(exc, self._ERR_INFRASTRUCTURE):
            return {"transient": True, "rate_limit": False, "infra": True}

        return None

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        """Extract a retry-after hint from a rate-limit exception."""
        if isinstance(exc, self._ERR_RATE_LIMIT) and getattr(exc, "retry_after", None):
            return float(exc.retry_after)

        return None
