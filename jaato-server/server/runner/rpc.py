"""Runner-side RPC dispatcher.

Implements the runner end of the daemon ↔ runner channel
described in ``docs/design/per_session_confined_runner.md`` §4.1.

Wire format: length-prefixed JSON frames (see :mod:`shared.framing`).
Frame schema: see :mod:`server.runner.envelope`.

The dispatcher serves on a blocking Unix socket (the runner inherits
the socketpair as fd 3, per §4.1 + §4.6).  On the runner side we use
the synchronous framing helpers and a worker-thread-per-call pool —
asyncio is daemon-side only.

Key responsibilities:

1. **Decode frames.**  ``kind: "request"`` triggers a tool execution;
   ``kind: "cancel"`` trips the per-call ``CancelToken``; other kinds
   are reserved for Phase 3 (runner→daemon RPCs, events).

2. **Dispatch ``tool.execute``** by name to a registered method on the
   :class:`server.runner.tool_executor.ToolExecutor`.

3. **Forward streaming output.**  The executor is given a per-call
   ``on_output`` callback; each invocation emits a
   :class:`server.runner.envelope.StreamFrame` over the same socket
   as the in-flight request, tagged with the call's ``request_id``.

4. **Honour cancellation.**  A daemon-side cancel frame trips the
   per-call ``CancelToken``; the tool detects via
   ``get_current_cancel_token()`` and either returns early or raises
   ``CancelledException``.  Either way the runner emits a terminating
   response with ``ok=False, error.type="CancelledException"``.

5. **Surface failures via the typed envelope (§4.8).**  Any executor
   exception is caught here and serialized into the response's
   ``error`` payload so the daemon side decoder doesn't have to be
   ready for partial frames or transport-level oddities.

Phase 2 scope:

- Single multiplexed channel.  Streaming back-pressure mitigation
  (§6.6) lives in the cli runner (output cap), not here.
- One thread per in-flight call.  Bounded worker pool (default 8
  concurrent tools — same as the daemon's parallel-tools cap).
- No runner→daemon RPCs — those are Phase 3.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol

from shared.framing import (
    FrameTooLargeError,
    read_frame_sync,
    write_frame_sync,
)

from jaato_sdk.plugins.model_provider.types import CancelToken

from .envelope import (
    KIND_CANCEL,
    KIND_REQUEST,
    KIND_RESPONSE,
    KIND_STREAM,
    STREAM_CHANNEL_DISPLAY,
    CancelFrame,
    ErrorPayload,
    RequestEnvelope,
    ResponseEnvelope,
    StreamFrame,
)


logger = logging.getLogger(__name__)


# We deliberately re-use :class:`jaato_sdk.plugins.model_provider.types.CancelToken`
# rather than defining a runner-private type: ``run_command`` and the
# rest of the in-process plugin contract already poll
# ``token.is_cancelled`` / call ``token.cancel(reason)``, and the
# runner-tier code path migrates plugin code unchanged.  Re-exported
# for callers that want to type-hint with the runner module.
__all__ = [
    "CancelToken",
    "RunnerRPC",
    "get_current_cancel_token",
    "get_current_output_callback",
]


# Thread-local channels for plugins to access per-call state without
# threading them through executor signatures.  Same pattern as the
# in-process ``shared.ai_tool_runner._thread_local``.
_thread_local = threading.local()


def get_current_cancel_token() -> Optional[CancelToken]:
    """Return the cancel token for the currently-executing tool call.

    Plugins poll this to decide when to abort long-running operations.
    Returns ``None`` outside a tool execution.
    """
    return getattr(_thread_local, "cancel_token", None)


def get_current_output_callback() -> Optional[Callable[[str, str, Optional[str]], None]]:
    """Return the streaming-output callback for the current tool call.

    Plugins call ``cb(source, text, mode)`` to emit chunks for user
    display; the dispatcher translates the call into a
    :class:`StreamFrame` over the wire.  Returns ``None`` outside a
    tool execution.
    """
    return getattr(_thread_local, "on_output", None)


# ----------------------------------------------------------------------
# Method registry contract
# ----------------------------------------------------------------------


class ExecuteFn(Protocol):
    """Shape of a runner-tier executor callable.

    Args mirror today's in-process executor: name → args dict → result
    dict.  The dispatcher provides the streaming callback + cancel
    token through thread-local state (set by :class:`RunnerRPC` BEFORE
    invoking the executor).

    The executor returns ``Tuple[bool, Any]`` matching the in-process
    ``ToolExecutor.execute`` contract today; the dispatcher wraps the
    return into a :class:`ResponseEnvelope`.
    """

    def __call__(self, name: str, args: Dict[str, Any]) -> "tuple[bool, Any]":
        ...


# ----------------------------------------------------------------------
# RunnerRPC dispatcher
# ----------------------------------------------------------------------


@dataclass
class _ActiveCall:
    """Runner-side bookkeeping for an in-flight tool call.

    ``cancel_token`` is tripped when a ``CancelFrame`` arrives.
    """

    cancel_token: CancelToken


class RunnerRPC:
    """Bidirectional dispatcher serving on a blocking Unix socket.

    Lifecycle:

    1. Construct with the inherited socketpair fd (typically fd 3) and
       a callable that executes ``method`` requests.
    2. ``serve()`` blocks the calling thread reading frames; per-call
       work runs in the worker pool.
    3. On peer-EOF (clean disconnect or daemon shutdown — §6.7), the
       loop exits cleanly.

    Thread model:

    - One reader thread (the caller of ``serve()``).
    - Worker pool for tool executions (default 8 concurrent — matches
      the daemon's parallel-tools cap).
    - All writes serialized via a per-instance write lock so partial
      frames can never interleave on the wire.
    """

    def __init__(
        self,
        sock: socket.socket,
        execute_fn: ExecuteFn,
        *,
        max_workers: int = 8,
        workspace_root: Optional[str] = None,
    ) -> None:
        """Construct the dispatcher.

        Args:
            sock: The inherited socketpair fd (typically fd 3).
            execute_fn: Tool-execution callable.
            max_workers: Concurrent tool-call cap.
            workspace_root: Per-session workspace root, for traceback
                sanitization (Phase 3 §3.1).  When set, captured
                tracebacks have ``<workspace_root>/...`` paths
                redacted to ``<WORKSPACE>/...`` before crossing the
                RPC boundary.  ``None`` skips workspace-pass; the
                home-jaato pass still runs.
        """
        self._sock = sock
        self._execute_fn = execute_fn
        self._workspace_root = workspace_root
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="runner-rpc",
        )
        self._write_lock = threading.Lock()
        self._active_calls: Dict[int, _ActiveCall] = {}
        self._active_lock = threading.Lock()
        self._closed = False

    # --------------------------- write paths ---------------------------

    def _write(self, payload: Dict[str, Any]) -> None:
        """Serialize *payload* to the wire under the write lock.

        The lock prevents partial-frame interleave when stream frames
        and the terminating response race.  Failures are logged and
        swallowed — a write failure usually means the daemon went
        away (§6.7), which the reader loop will surface cleanly via
        EOF on its next read.
        """
        encoded = json.dumps(payload, default=str)
        try:
            with self._write_lock:
                if not self._closed:
                    write_frame_sync(self._sock, encoded)
        except (OSError, BrokenPipeError) as exc:
            # Peer gone; mark closed so subsequent attempts no-op.
            logger.info(
                "runner RPC: write failed (%s) — peer disconnected, "
                "shutting down writer", exc,
            )
            self._closed = True

    def _emit_stream(
        self,
        request_id: int,
        source: str,
        text: str,
        mode: Optional[str],
    ) -> None:
        """Emit a streaming-output chunk for the given in-flight call."""
        frame = StreamFrame(
            id=request_id,
            source=source,
            text=text,
            mode=mode,
            channel=STREAM_CHANNEL_DISPLAY,
        )
        self._write(frame.to_dict())

    def _emit_response(
        self,
        request_id: int,
        ok: bool,
        result: Any,
        error: Optional[ErrorPayload] = None,
    ) -> None:
        env = ResponseEnvelope(
            id=request_id,
            ok=ok,
            result=result,
            error=error,
        )
        self._write(env.to_dict())

    # --------------------------- request paths -------------------------

    def _make_on_output(
        self, request_id: int
    ) -> Callable[[str, str, Optional[str]], None]:
        """Build the per-call ``on_output(source, text, mode)`` callback.

        Captures ``request_id`` so the resulting stream frames carry
        the right call id.
        """

        def _cb(source: str, text: str, mode: Optional[str] = None) -> None:
            self._emit_stream(request_id, source, text, mode)

        return _cb

    def _handle_request(self, env: RequestEnvelope) -> None:
        """Worker-thread entrypoint for one in-flight request.

        Sets thread-local cancel token + on_output, calls the executor,
        emits the terminating response.  Any exception from the
        executor is caught and serialized into the error payload — the
        wire never sees a half-open call.
        """
        token = CancelToken()
        with self._active_lock:
            self._active_calls[env.id] = _ActiveCall(cancel_token=token)

        _thread_local.cancel_token = token
        _thread_local.on_output = self._make_on_output(env.id)
        try:
            try:
                ok, result = self._dispatch_method(env)
            except Exception as exc:  # noqa: BLE001 — boundary of executor
                tb = traceback.format_exc()
                logger.exception(
                    "runner RPC: executor for method=%r raised", env.method,
                )
                # Phase 3 §3.1: redact tenant + operator absolute paths
                # before the traceback crosses the RPC boundary.  The
                # daemon-side log + any forwarded events are
                # potentially cross-tenant visibility surfaces.
                from .sanitize import sanitize_traceback
                tb = sanitize_traceback(tb, self._workspace_root)
                err = ErrorPayload(
                    type=type(exc).__name__,
                    message=sanitize_traceback(str(exc), self._workspace_root),
                    traceback=tb,
                )
                self._emit_response(env.id, ok=False, result=None, error=err)
                return

            # Successful execution — but the executor may have returned
            # ``(False, dict)`` to indicate a domain failure (e.g.
            # permission denied), which is serialized into the error
            # payload too so the daemon decoder is symmetric.
            if ok:
                self._emit_response(env.id, ok=True, result=result)
            else:
                err = ErrorPayload(
                    type="ToolError",
                    message=_extract_error_message(result),
                )
                self._emit_response(
                    env.id, ok=False, result=result, error=err,
                )
        finally:
            _thread_local.cancel_token = None
            _thread_local.on_output = None
            with self._active_lock:
                self._active_calls.pop(env.id, None)

    def _dispatch_method(
        self, env: RequestEnvelope,
    ) -> "tuple[bool, Any]":
        """Route ``method`` to the executor.

        Phase 2 supports two methods:
        - ``"echo"`` — return args verbatim (RPC-overhead probe).
        - ``"tool.execute"`` — args = ``{"name": ..., "args": ...}``;
          delegate to the executor.
        """
        if env.method == "echo":
            return True, dict(env.args)

        if env.method == "tool.execute":
            tool_name = str(env.args.get("name") or "")
            tool_args = dict(env.args.get("args") or {})
            if not tool_name:
                return False, {"error": "tool.execute: missing 'name' arg"}
            return self._execute_fn(tool_name, tool_args)

        return False, {"error": f"unknown method: {env.method!r}"}

    def _handle_cancel(self, frame: CancelFrame) -> None:
        with self._active_lock:
            active = self._active_calls.get(frame.id)
        if active is None:
            logger.debug(
                "runner RPC: cancel for unknown call id=%d — already finished?",
                frame.id,
            )
            return
        active.cancel_token.cancel()
        logger.info("runner RPC: cancel tripped for call id=%d", frame.id)

    # --------------------------- main loop -----------------------------

    def serve(self) -> None:
        """Blocking serve loop — reads frames until peer EOF.

        Returns cleanly on EOF (graceful shutdown — §6.7).  Exceptions
        from the worker pool surface via the executor's failure path
        and never propagate here.
        """
        try:
            while True:
                try:
                    raw = read_frame_sync(self._sock)
                except FrameTooLargeError as exc:
                    logger.error(
                        "runner RPC: peer sent oversized frame: %s — closing",
                        exc,
                    )
                    return
                if raw is None:
                    logger.info("runner RPC: peer closed; serve loop exiting")
                    return

                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.error(
                        "runner RPC: malformed JSON frame: %s — closing", exc,
                    )
                    return

                kind = payload.get("kind")
                if kind == KIND_REQUEST:
                    try:
                        env = RequestEnvelope.from_dict(payload)
                    except (KeyError, ValueError) as exc:
                        logger.error(
                            "runner RPC: malformed request frame: %s", exc,
                        )
                        continue
                    # Dispatch to a worker thread.
                    self._pool.submit(self._handle_request, env)
                elif kind == KIND_CANCEL:
                    try:
                        frame = CancelFrame.from_dict(payload)
                    except (KeyError, ValueError) as exc:
                        logger.error(
                            "runner RPC: malformed cancel frame: %s", exc,
                        )
                        continue
                    self._handle_cancel(frame)
                else:
                    logger.warning(
                        "runner RPC: ignoring unknown frame kind=%r", kind,
                    )
        finally:
            # Mark closed FIRST so concurrent writers (worker threads
            # racing to emit a stream/response on a now-dead socket)
            # short-circuit instead of touching the closed fd.
            self._closed = True
            # Half-close the write side so the peer sees EOF and can
            # exit its own read loop cleanly (§6.7 bidirectional
            # benign-EOF rule).  ``shutdown(SHUT_WR)`` rather than
            # full close because in-flight worker writes still need
            # the fd briefly; the eventual ``shutdown()`` call from
            # the owner closes it fully.
            try:
                self._sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            self._pool.shutdown(wait=False, cancel_futures=True)

    def shutdown(self) -> None:
        """Initiate shutdown from outside the serve loop."""
        self._closed = True
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _extract_error_message(result: Any) -> str:
    """Best-effort error string from a domain-failure result dict.

    Plugins return ``(False, {"error": "...", ...})`` today; mirror
    that shape here so the typed envelope's error.message field is
    populated even for non-exception failures.
    """
    if isinstance(result, dict):
        for key in ("error", "message"):
            v = result.get(key)
            if isinstance(v, str) and v:
                return v
    return "tool execution failed"
