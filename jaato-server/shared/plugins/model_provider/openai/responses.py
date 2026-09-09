"""The Responses-API transport for the native OpenAI provider.

``_openai_compat`` owns the Chat Completions transport that nine gateways
share.  The Responses API is a different wire — a flat ``input`` item list,
top-level ``function_call`` items keyed by ``call_id``, typed SSE events
instead of choice deltas, ``input_tokens``/``output_tokens`` usage — so it
gets its own transport rather than a flag on that one.  The shapes are
tabulated in :mod:`responses_converters`.

**Why jaato speaks it at all.**  It is where OpenAI ships first (the
reasoning controls, the built-in tool types, encrypted reasoning
continuity), and Chat Completions is explicitly the *legacy* surface.  A
"native OpenAI" provider that spoke only the legacy wire would be a
gateway with a different base URL.

**Stream for the UX, parse the terminal object for the truth.**  Text and
reasoning deltas are forwarded to the caller as they arrive, because that
is what a person watches; but the parts that become history are read from
the ``response`` object carried by the terminal event, which is the
API's own authoritative account of what it produced.  The accumulated
deltas are the fallback for the one case with no terminal object — a
cancelled turn — where the fragments are all there is.

**Cancellation closes the transport.**  As on the chat path: the SDK's
``Stream`` only sends TCP-close at GC time, which on a cancelled turn
leaves the upstream generating (and billing) the whole response.

**Session state stays ours.**  Every request is sent with ``store=False``
and the full ``input`` — jaato owns the conversation, its GC decides what
the model sees, and a server-side thread keyed by ``previous_response_id``
would silently diverge from the history the rest of the framework
reasons about.  An author who wants server-side storage sets
``api_params.store: true`` deliberately.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from jaato_sdk.plugins.model_provider.types import (
    CancelToken,
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    ToolSchema,
    parse_tool_call_arguments,
    require_terminated_stream,
    resolve_tool_use_finish,
)

from shared.tool_id_map import tool_choice_to_wire, wire_name_trace_fields
from .._media_deltas import (
    NO_MEDIA_YET,
    ensure_spoken_part,
    media_chunk_count,
)

from .responses_converters import (
    _get,
    finish_reason_from_response,
    get_original_tool_name,
    history_to_responses,
    parts_from_responses_output,
    thinking_from_responses_output,
    tool_schemas_to_responses,
    usage_from_responses,
)

#: Request-body fields forwarded from ``plugin_configs.openai.api_params``
#: on the Responses wire.  Deliberately NOT the chat allow-list: the two
#: APIs disagree on the name of the single most-used knob
#: (``max_output_tokens`` vs ``max_tokens``), so sharing one set would
#: forward a field the endpoint 400s on and hide the typo'd-key warning
#: that makes such a mistake visible.
RESPONSES_API_PARAMS = frozenset({
    "temperature", "top_p", "max_output_tokens", "tool_choice",
    "parallel_tool_calls", "reasoning", "text", "service_tier",
    "truncation", "metadata", "store", "include", "prompt_cache_key",
    "safety_identifier", "modalities", "audio",
})

#: Reasoning-summary deltas.  Two spellings because the API renamed the
#: event and still emits the older one for some models; both are the same
#: channel to us.
_THINKING_EVENTS = frozenset({
    "response.reasoning_summary_text.delta",
    "response.reasoning_text.delta",
})

#: Model-emitted audio.  Two events, because the bytes and the words
#: arrive separately on this wire exactly as they do on the chat one —
#: which is why the shared decoder is handed a chat-shaped shim rather
#: than reimplemented here.
_AUDIO_DATA_EVENT = "response.output_audio.delta"
_AUDIO_TRANSCRIPT_EVENT = "response.output_audio_transcript.delta"

#: Events that carry the terminal ``response`` object.
_TERMINAL_EVENTS = frozenset({
    "response.completed", "response.incomplete", "response.failed",
})


class _ResponsesAccumulator:
    """Ordered accumulation of one Responses stream.

    Keyed by ``output_index`` — the API's own ordering of the items it is
    producing — so text and tool calls come back interleaved as the model
    emitted them, rather than text-then-calls regardless of order.

    Each slot is ``{"kind": "text"|"call", ...}``.  A ``call`` slot holds
    the ``call_id`` / ``name`` announced by ``response.output_item.added``
    and the ``arguments`` string assembled from the argument deltas; the
    ``.done`` event replaces that string with the API's own copy, which is
    authoritative and cannot have lost a delta.

    Lifetime: one stream.  Its output is used only when the stream carried
    no terminal ``response`` object (a cancelled turn) — otherwise
    :func:`parts_from_responses_output` reads that object instead.
    """

    def __init__(self) -> None:
        self._slots: Dict[int, Dict[str, Any]] = {}
        self._thinking: List[str] = []
        self.text_chunks = 0
        #: Monotonic index over model-generated media chunks, so a
        #: consumer can spot a gap left by backpressure.  Separate from
        #: ``text_chunks``: they are different streams.
        self.media_sequence = NO_MEDIA_YET
        #: One-slot buffer: a chunk is held until the next arrives, so
        #: the end of the utterance can flag the one before it ``final``.
        self.media_pending: List[Any] = []
        #: What the model SAID.  The bytes and the words arrive in
        #: separate events, so the transcript cannot be read off the
        #: chunks that were emitted.
        self.media_transcript: List[str] = []

    def add_text(self, index: int, delta: str) -> None:
        """Append a text delta to the slot at ``index``."""
        slot = self._slots.setdefault(index, {"kind": "text", "text": []})
        if slot["kind"] != "text":  # pragma: no cover - API would not do this
            return
        slot["text"].append(delta)
        self.text_chunks += 1

    def open_item(self, index: int, item: Any) -> None:
        """Record a ``function_call`` item announced at ``index``."""
        if _get(item, "type") != "function_call":
            return
        self._slots[index] = {
            "kind": "call",
            "call_id": _get(item, "call_id") or _get(item, "id"),
            "name": _get(item, "name") or "",
            "arguments": _get(item, "arguments") or "",
        }

    def add_arguments(self, index: int, delta: str) -> None:
        """Append an argument delta to the call slot at ``index``."""
        slot = self._slots.get(index)
        if slot is not None and slot["kind"] == "call":
            slot["arguments"] += delta

    def set_arguments(self, index: int, arguments: Optional[str]) -> None:
        """Replace a call slot's arguments with the API's own final copy."""
        slot = self._slots.get(index)
        if slot is not None and slot["kind"] == "call" and arguments is not None:
            slot["arguments"] = arguments

    def add_thinking(self, delta: str) -> None:
        """Append a reasoning-summary delta."""
        self._thinking.append(delta)

    @property
    def has_text(self) -> bool:
        """Did the model write text of its own this turn?

        The question :func:`ensure_spoken_part` asks: a spoken turn's
        transcript stands in for the words only when there are no words
        already (#869).
        """
        return any(
            slot["kind"] == "text" and any(slot["text"])
            for slot in self._slots.values()
        )

    @property
    def thinking(self) -> Optional[str]:
        """The reasoning summary streamed so far, or ``None``."""
        return "".join(self._thinking) or None

    def parts(self) -> List[Part]:
        """The accumulated parts, in ``output_index`` order."""
        parts: List[Part] = []
        for index in sorted(self._slots):
            slot = self._slots[index]
            if slot["kind"] == "text":
                text = "".join(slot["text"])
                if text:
                    parts.append(Part.from_text(text))
                continue
            args, unreadable = parse_tool_call_arguments(slot["arguments"])
            parts.append(Part.from_function_call(FunctionCall(
                id=slot["call_id"],
                name=get_original_tool_name(slot["name"]),
                args=args,
                unreadable_args=unreadable,
            )))
        return parts


class ResponsesTransport:
    """Mixin holding the Responses-API request/response path.

    Mixed into :class:`~.provider.OpenAIProvider`, which supplies
    ``self._client`` (an ``openai.OpenAI``), ``self._model_name``,
    ``self._api_params``, ``self._extra_body``, ``self._enable_thinking``,
    ``self._trace`` and ``self._handle_api_error`` — the same attributes
    the Chat Completions base owns, reused rather than duplicated so the
    two wires share one client, one error taxonomy and one trace log.
    """

    # ==================== Request assembly ====================

    def _responses_kwargs(
        self,
        tools: Optional[List[ToolSchema]],
        tool_choice: Optional[Any],
        response_schema: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assemble the non-``input`` half of a ``responses.create`` call.

        Profile ``api_params`` (already filtered to
        :data:`RESPONSES_API_PARAMS`) apply to every call; a per-call
        ``tool_choice`` overrides the profile's.  ``tool_choice`` is
        dropped when this turn sends no tools — the API rejects it
        without them — and a name-bearing choice is mapped through
        :func:`tool_choice_to_wire`, because tool names are hashed.

        ``store`` defaults to ``False`` (see the module docstring) but is
        a forwardable param, so a profile that sets it wins.
        """
        kwargs: Dict[str, Any] = {"store": False}
        kwargs.update(self._api_params)
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body

        wire_tools = tool_schemas_to_responses(tools)
        if wire_tools:
            kwargs["tools"] = wire_tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if "tool_choice" in kwargs and "tools" not in kwargs:
            kwargs.pop("tool_choice")
        if "tool_choice" in kwargs:
            kwargs["tool_choice"] = tool_choice_to_wire(kwargs["tool_choice"])

        if response_schema:
            kwargs["text"] = {"format": {"type": "json_object"}}
        return kwargs

    # ==================== Streaming ====================

    def _apply_content_event(
        self,
        etype: str,
        event: Any,
        acc: _ResponsesAccumulator,
        on_chunk: Callable[[Any], None],
        on_thinking: Optional[Callable[[str], None]],
    ) -> None:
        """Route one non-terminal event into the accumulator.

        Unknown event types are ignored on purpose: the Responses event
        vocabulary grows, and a provider that raised on an event it had
        not been taught would break on an API addition that costs it
        nothing to skip.
        """
        if self._apply_audio_event(etype, event, acc, on_chunk):
            return
        if self._apply_call_event(etype, event, acc):
            return
        if etype == "response.output_text.delta":
            delta = _get(event, "delta") or ""
            if delta:
                acc.add_text(_get(event, "output_index") or 0, delta)
                on_chunk(delta)
        elif etype in _THINKING_EVENTS and self._enable_thinking:
            delta = _get(event, "delta") or ""
            if delta:
                acc.add_thinking(delta)
                if on_thinking:
                    on_thinking(delta)

    def _apply_call_event(
        self, etype: str, event: Any, acc: _ResponsesAccumulator,
    ) -> bool:
        """Route a tool-call event into the accumulator.

        Three events build one call: the item announcement carries the
        ``call_id`` and name, the argument deltas build the JSON string,
        and the ``.done`` event replaces it with the API's own final copy
        — authoritative, because it cannot have lost a delta.

        Returns:
            True when the event was a tool-call event and was handled.
        """
        if etype == "response.output_item.added":
            item = _get(event, "item")
            index = _get(event, "output_index") or 0
            acc.open_item(index, item)
            if _get(item, "type") == "function_call":
                self._trace(
                    f"TOOL_CALL_START idx={index} "
                    f"id={_get(item, 'call_id')!r} "
                    + wire_name_trace_fields(_get(item, "name") or "")
                )
            return True
        if etype == "response.function_call_arguments.delta":
            acc.add_arguments(
                _get(event, "output_index") or 0, _get(event, "delta") or "")
            return True
        if etype == "response.function_call_arguments.done":
            acc.set_arguments(
                _get(event, "output_index") or 0, _get(event, "arguments"))
            return True
        return False

    def _apply_audio_event(
        self,
        etype: str,
        event: Any,
        acc: _ResponsesAccumulator,
        on_chunk: Callable[[Any], None],
    ) -> bool:
        """Decode a model-audio event, or report that this was not one.

        The bytes (``response.output_audio.delta``) and the words
        (``response.output_audio_transcript.delta``) are separate events
        here, and separate *deltas* on the chat wire — the same split,
        differently spelled.  So each is wrapped in the chat-shaped shim
        the shared decoder already reads (``delta.audio.data`` /
        ``.transcript``) rather than given a second decoder to drift
        from: one place decides the mime, the sequence numbering, the
        one-slot buffer and the ``final`` marker.

        Returns:
            True when the event was audio and has been handled.
        """
        if etype == _AUDIO_DATA_EVENT:
            shim = {"audio": {"data": _get(event, "delta") or ""}}
        elif etype == _AUDIO_TRANSCRIPT_EVENT:
            shim = {"audio": {"transcript": _get(event, "delta") or ""}}
        else:
            return False
        acc.media_sequence = self.emit_media_delta(
            shim, on_chunk, acc.media_sequence, acc.media_transcript,
            acc.media_pending, lambda: acc.has_text,
        )
        return True

    def _stream_responses(
        self,
        input_items: List[Dict[str, Any]],
        instructions: Optional[str],
        kwargs: Dict[str, Any],
        on_chunk: Callable[[Any], None],
        cancel_token: Optional[CancelToken] = None,
        on_usage_update: Optional[Callable[[Any], None]] = None,
        on_thinking: Optional[Callable[[str], None]] = None,
    ) -> ProviderResponse:
        """Stream one Responses turn and return the assembled response.

        Returns the response built from the terminal ``response`` object
        when the stream delivered one, and from the accumulated deltas
        when it did not (a cancelled turn).  A stream that simply STOPS —
        no terminal event, not cancelled — raises through
        :func:`require_terminated_stream` rather than handing back its
        fragments as a finished turn (#687).
        """
        acc = _ResponsesAccumulator()
        terminal_seen = False
        was_cancelled = False
        terminal_response: Any = None
        usage = usage_from_responses(None)
        stream = None

        try:
            self._trace("RESPONSES_STREAM_START")
            stream = self._client.responses.create(
                model=self._model_name,
                input=input_items,
                instructions=instructions,
                stream=True,
                **kwargs,
            )
            for event in stream:
                if cancel_token and cancel_token.is_cancelled:
                    self._trace("RESPONSES_STREAM_CANCELLED")
                    was_cancelled = True
                    break
                etype = _get(event, "type") or ""
                if etype in _TERMINAL_EVENTS:
                    terminal_seen = True
                    terminal_response = _get(event, "response")
                    usage = usage_from_responses(
                        _get(terminal_response, "usage"))
                    if on_usage_update and usage.total_tokens > 0:
                        on_usage_update(usage)
                    continue
                self._apply_content_event(
                    etype, event, acc, on_chunk, on_thinking)
            # The upstream's end-of-audio marker normally released the
            # last chunk already; the stream ending is the backstop for a
            # turn that sent none, and is conclusive.
            self.flush_media_stream(
                on_chunk, acc.media_pending, acc.media_transcript,
                lambda: acc.has_text,
            )
            self._trace(
                f"RESPONSES_STREAM_END chunks={acc.text_chunks} "
                f"terminal_seen={terminal_seen}"
            )
        except Exception as exc:
            self._trace(f"RESPONSES_STREAM_ERROR {type(exc).__name__}: {exc}")
            if cancel_token and cancel_token.is_cancelled:
                was_cancelled = True
            else:
                raise
        finally:
            self._close_stream(stream, was_cancelled)

        return self._assemble(
            acc, terminal_response, usage,
            terminal_seen=terminal_seen,
            was_cancelled=was_cancelled,
        )

    def _close_stream(self, stream: Any, was_cancelled: bool) -> None:
        """Release the HTTP stream, and the pool too when cancelled.

        ``stream.close()`` alone does not propagate TCP-FIN to the
        upstream, so a cancelled turn keeps being generated (and billed)
        until GC gets round to the client.  Best-effort throughout: a
        failure to close must not replace the caller's error.
        """
        if stream is not None:
            try:
                stream.close()
            except Exception as exc:  # pragma: no cover - best effort
                self._trace(
                    f"RESPONSES_STREAM_CLOSE_ERROR {type(exc).__name__}: {exc}")
        if was_cancelled and self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover - best effort
                pass

    def _assemble(
        self,
        acc: _ResponsesAccumulator,
        terminal_response: Any,
        usage: Any,
        *,
        terminal_seen: bool,
        was_cancelled: bool,
    ) -> ProviderResponse:
        """Build the ``ProviderResponse`` for a finished (or cut) stream."""
        if terminal_response is not None:
            parts = parts_from_responses_output(
                _get(terminal_response, "output"))
            finish_reason = finish_reason_from_response(terminal_response)
            thinking = (
                thinking_from_responses_output(_get(terminal_response, "output"))
                or acc.thinking
            )
        else:
            parts = acc.parts()
            finish_reason = (
                FinishReason.CANCELLED if was_cancelled else FinishReason.UNKNOWN
            )
            thinking = acc.thinking

        finish_reason = resolve_tool_use_finish(
            finish_reason,
            has_function_calls=(
                any(p.function_call for p in parts) and not was_cancelled
            ),
        )
        # A turn that only spoke has no Part of its own — without this the
        # session sees an empty response and nudges a good answer.
        ensure_spoken_part(parts, "".join(acc.media_transcript))
        return require_terminated_stream(
            ProviderResponse(
                parts=parts,
                media_chunks=media_chunk_count(acc.media_sequence),
                usage=usage,
                finish_reason=finish_reason,
                raw=terminal_response,
                thinking=thinking,
            ),
            terminal_seen=terminal_seen,
            was_cancelled=was_cancelled,
            provider=self.name,
            model=self._model_name,
            chunks=acc.text_chunks + media_chunk_count(acc.media_sequence),
        )

    # ==================== Non-streaming ====================

    def _batch_responses(
        self,
        input_items: List[Dict[str, Any]],
        instructions: Optional[str],
        kwargs: Dict[str, Any],
    ) -> ProviderResponse:
        """One non-streamed Responses turn.

        There is no termination question here: a returned object IS the
        terminal object, so this path never consults
        ``require_terminated_stream``.
        """
        self._trace("RESPONSES_BATCH_START")
        response = self._client.responses.create(
            model=self._model_name,
            input=input_items,
            instructions=instructions,
            **kwargs,
        )
        output = _get(response, "output")
        parts = parts_from_responses_output(output)
        return ProviderResponse(
            parts=parts,
            usage=usage_from_responses(_get(response, "usage")),
            finish_reason=resolve_tool_use_finish(
                finish_reason_from_response(response),
                has_function_calls=any(p.function_call for p in parts),
            ),
            raw=response,
            thinking=thinking_from_responses_output(output),
        )

    # ==================== Entry point ====================

    def _complete_via_responses(
        self,
        messages: List[Message],
        system_instruction: Optional[str],
        tools: Optional[List[ToolSchema]],
        *,
        response_schema: Optional[Dict[str, Any]] = None,
        cancel_token: Optional[CancelToken] = None,
        on_chunk: Optional[Callable[[Any], None]] = None,
        on_usage_update: Optional[Callable[[Any], None]] = None,
        on_thinking: Optional[Callable[[str], None]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        """Run one turn on the Responses wire.

        The system prompt becomes ``instructions`` rather than a leading
        message — the API's own place for it, and the one it re-applies
        across a stored thread.
        """
        input_items = history_to_responses(list(messages))
        kwargs = self._responses_kwargs(tools, tool_choice, response_schema)
        if on_chunk:
            return self._stream_responses(
                input_items, system_instruction, kwargs, on_chunk,
                cancel_token=cancel_token,
                on_usage_update=on_usage_update,
                on_thinking=on_thinking,
            )
        return self._batch_responses(input_items, system_instruction, kwargs)
