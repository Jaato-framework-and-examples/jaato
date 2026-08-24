"""Typed RPC envelope for the daemon ↔ runner channel.

Defines the wire-level frame schema described in
``docs/design/per_session_confined_runner.md`` §4.1 + §4.8:

- Every frame is a JSON object with ``id`` (int) and ``kind`` (str).
- ``kind`` is one of: ``"request"``, ``"response"``, ``"stream"``,
  ``"cancel"``, ``"event"``.
- Request/response carry an envelope shape with ``ok``, ``result``,
  ``error``, ``warnings``, ``telemetry``.

These dataclasses are NOT the wire format — they're the in-memory
shape produced/consumed by callers.  ``to_dict`` / ``from_dict``
serialize them to JSON-friendly dicts that ``shared.framing``
encodes as length-prefixed UTF-8.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------
# Frame kinds
# ----------------------------------------------------------------------

KIND_REQUEST = "request"
KIND_RESPONSE = "response"
KIND_STREAM = "stream"
KIND_CANCEL = "cancel"
KIND_EVENT = "event"

# Stream channel discriminators (§4.1.1).  ``"display"`` is for
# user-visible chunks routed through ``on_output``; future channels
# (e.g. ``"progress"``) reserved.
STREAM_CHANNEL_DISPLAY = "display"


# ----------------------------------------------------------------------
# Error payload
# ----------------------------------------------------------------------


@dataclass
class ErrorPayload:
    """Structured error inside a response envelope.

    Mirrors the in-process ``{"error": ..., "traceback": ...}`` shape
    today's ``_execute_impl`` returns, lifted out of ``result`` into a
    distinct envelope field.

    ``traceback`` is optional — populated when the executor raised an
    exception, omitted when the executor returned ``(False, dict)``
    intentionally.

    Cross-tenant info-leak caveat (§4.8): tracebacks may carry
    workspace-absolute paths.  The runner SHOULD path-sanitize before
    sending; sanitization is a Phase 3 task and lives in the
    error-construction site, not in this dataclass.
    """

    type: str
    message: str
    traceback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type, "message": self.message}
        if self.traceback is not None:
            d["traceback"] = self.traceback
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ErrorPayload":
        return cls(
            type=d["type"],
            message=d["message"],
            traceback=d.get("traceback"),
        )


# ----------------------------------------------------------------------
# Executor outcome (the ``(ok, payload)`` contract, made transportable)
# ----------------------------------------------------------------------

# ``ErrorPayload.type`` value marking a DOMAIN failure — an executor that
# returned ``(False, payload)`` without raising — as opposed to a crash,
# where ``type`` is the real exception class name.  The runner-side
# dispatcher has always stamped this (``runner/rpc.py``); naming it here
# lets the daemon-side dispatcher and the RPC client agree on the same
# discriminator instead of matching the string in three places.
TOOL_ERROR_TYPE = "ToolError"


@dataclass
class ExecutorOutcome:
    """An executor's ``(ok, payload)`` result, as an explicit TYPE.

    **Why this exists.**  Plugin executors signal domain failure by
    returning a ``(False, payload)`` 2-tuple (``split_executor_result``).
    That contract is carried by a Python ``tuple`` — and JSON has no
    tuple type, so ``json.dumps((False, {...}))`` yields ``[False, {...}]``
    and the receiving ``isinstance(x, tuple)`` check fails.  The flag then
    demotes from FLAG to DATA: ``split_executor_result`` falls through to
    ``return True, executor_result``, the call reports success, and the
    payload arrives wrapped as ``{"result": [False, {...}]}``.  Both the
    ``is_error`` flag and the deeper ``tool_result_is_error`` body check
    read ``False``, so a failing forwarded tool is invisible to every
    consumer-side error check.

    **Why a type and not a shape.**  The tempting fix is to accept any
    2-element list whose head is a bool.  That makes a list that IS a
    payload indistinguishable from an ``(ok, payload)`` pair — the same
    absent-vs-empty ambiguity, moved rather than removed.  A handler
    returning ``ExecutorOutcome`` states the contract; no other handler
    can be mistaken for one.

    The dispatcher unpacks this into the envelope's own halves —
    ``ok`` + ``result`` + an ``error`` stamped :data:`TOOL_ERROR_TYPE` —
    which is exactly what :class:`ResponseEnvelope` documents ``ok`` for
    ("kept explicit because some plugins legitimately return
    ``(False, dict)`` without raising").  The field existed for this case
    and the daemon path routed around it.

    Attributes:
        ok: The executor's success flag.
        payload: The executor's result value, unchanged.
    """

    ok: bool
    payload: Any = None


# ----------------------------------------------------------------------
# Response envelope (§4.8)
# ----------------------------------------------------------------------


@dataclass
class ResponseEnvelope:
    """Typed envelope for ``kind: "response"`` frames.

    Wire shape (after ``to_dict``)::

        {
          "id": <int>,
          "kind": "response",
          "ok": <bool>,
          "result": <any>,
          "error": null | {"type": ..., "message": ..., "traceback": ...?},
          "warnings": [<str>, ...],
          "telemetry": {<str>: <any>, ...}
        }

    ``ok`` is the boolean half of today's in-process
    ``Tuple[bool, Any]`` return.  Redundant with ``error is None`` for
    most callers, but kept explicit because some plugins legitimately
    return ``(False, dict)`` without raising.
    """

    id: int
    ok: bool
    result: Any = None
    error: Optional[ErrorPayload] = None
    warnings: List[str] = field(default_factory=list)
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": KIND_RESPONSE,
            "ok": self.ok,
            "result": self.result,
            "error": self.error.to_dict() if self.error is not None else None,
            "warnings": list(self.warnings),
            "telemetry": dict(self.telemetry),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResponseEnvelope":
        if d.get("kind") != KIND_RESPONSE:
            raise ValueError(
                f"ResponseEnvelope.from_dict: kind != 'response' "
                f"(got {d.get('kind')!r})"
            )
        err_dict = d.get("error")
        return cls(
            id=int(d["id"]),
            ok=bool(d["ok"]),
            result=d.get("result"),
            error=ErrorPayload.from_dict(err_dict) if err_dict else None,
            warnings=list(d.get("warnings") or []),
            telemetry=dict(d.get("telemetry") or {}),
        )


# ----------------------------------------------------------------------
# Request envelope
# ----------------------------------------------------------------------


@dataclass
class RequestEnvelope:
    """Typed envelope for ``kind: "request"`` frames.

    Wire shape::

        {
          "id": <int>,
          "kind": "request",
          "method": <str>,
          "args": <dict>
        }

    Methods Phase 2 implements:
    - ``"echo"`` — echoes args back (RPC-overhead measurement).
    - ``"tool.execute"`` — args = ``{"name": <str>, "args": <dict>}``.
    """

    id: int
    method: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": KIND_REQUEST,
            "method": self.method,
            "args": dict(self.args),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RequestEnvelope":
        if d.get("kind") != KIND_REQUEST:
            raise ValueError(
                f"RequestEnvelope.from_dict: kind != 'request' "
                f"(got {d.get('kind')!r})"
            )
        return cls(
            id=int(d["id"]),
            method=str(d["method"]),
            args=dict(d.get("args") or {}),
        )


# ----------------------------------------------------------------------
# Stream frame (§4.1.1)
# ----------------------------------------------------------------------


@dataclass
class StreamFrame:
    """``kind: "stream"`` frame for in-flight streaming output.

    Wire shape::

        {
          "id": <int>,                 # request_id of the in-flight call
          "kind": "stream",
          "channel": "display",        # only channel in Phase 2
          "source": <str>,             # "stdout" | "stderr" | plugin-defined
          "text": <str>,
          "mode": <str> | null         # plugin-defined display mode hint
        }

    Display chunks are forwarded to the session's ``on_output`` callback
    for user display ONLY — they are NOT part of the structured tool
    result.  Plugins MUST NOT smuggle structured data through this
    channel; the model only sees the terminating ``ResponseEnvelope``.
    """

    id: int
    source: str
    text: str
    mode: Optional[str] = None
    channel: str = STREAM_CHANNEL_DISPLAY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": KIND_STREAM,
            "channel": self.channel,
            "source": self.source,
            "text": self.text,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StreamFrame":
        if d.get("kind") != KIND_STREAM:
            raise ValueError(
                f"StreamFrame.from_dict: kind != 'stream' (got {d.get('kind')!r})"
            )
        return cls(
            id=int(d["id"]),
            source=str(d["source"]),
            text=str(d["text"]),
            mode=d.get("mode"),
            channel=str(d.get("channel") or STREAM_CHANNEL_DISPLAY),
        )


# ----------------------------------------------------------------------
# Notification frame (§7c step 6.6.4.1)
# ----------------------------------------------------------------------


@dataclass
class NotificationFrame:
    """``kind: "event"`` frame for runner→daemon notifications
    streamed alongside output frames during long-running RPCs
    (currently :meth:`session.send_message`).

    Phase 3 §7c step 6.6.4.1.  Per the §7c step 6.6.2 audit
    (commit 9f28f96d), this is the wire-format extension the
    audit's "stream-channel multiplex" rationale called for.
    The existing :class:`StreamFrame` carries display output
    chunks; this frame carries structured event notifications
    that the daemon-side per-call handler demuxes by
    ``event_type`` and routes to the appropriate
    ``server.emit(<Event>)`` or other action.

    Wire shape::

        {
          "id": <int>,                  # request_id of the in-flight call
          "kind": "event",
          "event_type": <str>,          # discriminator, e.g.
                                        # "instruction_budget_updated"
          "payload": <dict>             # event-type-specific data
        }

    The ``KIND_EVENT`` scaffolding has been in
    ``envelope.py:32`` since Phase 2; the daemon's read loop
    has dispatched ``kind == "event"`` frames as a placeholder
    debug-log no-op (server/runner_rpc_client.py pre-§7c-
    step-6.6.4.1).  This commit promotes that scaffolding into
    a typed wire surface + handler-registration mechanism on
    :class:`RunnerRPCClient`.

    Used by §7c step 6.6.4.2's 7-callback collapse — the
    runner-side ``JaatoSession`` callback hooks emit
    notification frames; the daemon-side
    ``session_send_message`` wrapper demuxes them.

    Defensive contract: ``event_type`` must be a non-empty
    string (no implicit defaulting); ``payload`` must be a
    dict (may be empty for parameter-less notifications).
    Per-event-type validation happens at the daemon-side
    consumer, NOT at the wire boundary — keeps the protocol
    extensible without renegotiating the wire shape per new
    event type.
    """

    id: int
    event_type: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": KIND_EVENT,
            "event_type": self.event_type,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NotificationFrame":
        if d.get("kind") != KIND_EVENT:
            raise ValueError(
                f"NotificationFrame.from_dict: kind != 'event' "
                f"(got {d.get('kind')!r})"
            )
        event_type = d.get("event_type")
        if not isinstance(event_type, str) or not event_type:
            raise ValueError(
                f"NotificationFrame.from_dict: 'event_type' must be a "
                f"non-empty str; got {event_type!r}"
            )
        payload = d.get("payload") or {}
        if not isinstance(payload, dict):
            raise ValueError(
                f"NotificationFrame.from_dict: 'payload' must be a dict "
                f"or omitted; got {type(payload).__name__}"
            )
        return cls(
            id=int(d["id"]),
            event_type=event_type,
            payload=dict(payload),
        )


# ----------------------------------------------------------------------
# Cancel frame
# ----------------------------------------------------------------------


@dataclass
class CancelFrame:
    """``kind: "cancel"`` frame instructing the peer to abort call ``id``.

    Wire shape::

        { "id": <int>, "kind": "cancel" }

    The runner trips the per-call ``CancelToken`` on receipt; the tool
    detects via ``get_current_cancel_token()`` and returns early or
    raises ``CancelledException``.  See §4.1.1.
    """

    id: int

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": KIND_CANCEL}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CancelFrame":
        if d.get("kind") != KIND_CANCEL:
            raise ValueError(
                f"CancelFrame.from_dict: kind != 'cancel' (got {d.get('kind')!r})"
            )
        return cls(id=int(d["id"]))


# ----------------------------------------------------------------------
# Session-bootstrap envelope re-export (Phase 3 §3.3a)
# ----------------------------------------------------------------------
#
# Defined in ``shared/session_envelope.py`` so both daemon and runner
# import from a single canonical path; re-exported here so the
# runner's import surface stays consolidated under
# ``server.runner.envelope`` (callers don't need to know whether a
# given envelope type is RPC-wire or session-bootstrap).
from shared.session_envelope import (  # noqa: E402 — re-export at end
    SESSION_ENVELOPE_VERSION,
    SessionInitEnvelope,
)
