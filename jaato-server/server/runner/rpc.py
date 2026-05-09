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
import concurrent.futures as _concurrent_futures
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

        # Phase 3 §3.2: runner → daemon outgoing-call bookkeeping.
        # ``_outgoing_calls`` is keyed by request-id and holds a
        # ``concurrent.futures.Future`` (NOT asyncio.Future — the
        # runner side is synchronous, so callers block on
        # ``fut.result(timeout)``).  ``_next_outgoing_id`` is a
        # counter for outgoing-from-runner request IDs.
        #
        # ID-space note: the runner's outgoing-call IDs and the
        # daemon's incoming-call IDs share the wire but are scoped
        # by direction in the dispatcher — incoming ``response``
        # frames are always for outgoing-from-runner calls (the
        # daemon never sends ``response`` for its own request),
        # so look-up against ``_outgoing_calls`` is unambiguous.
        self._outgoing_calls: Dict[int, "_concurrent_futures.Future"] = {}
        self._outgoing_lock = threading.Lock()
        self._next_outgoing_id = 1

        # Phase 3 §3.3c: runner-side session host (constructed when
        # the daemon sends ``session.bootstrap``).  ``None`` until
        # then.  Plugin migrations §3.4-§3.10 route runner-tier
        # dispatch through this host's session executor.
        self._session_host = None  # type: Optional[Any]
        self._session_lock = threading.Lock()

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
        # Phase 3 §3.15: lift ``_telemetry`` off the result dict (if
        # present) and move it to ``envelope.telemetry`` — the
        # canonical wire location for tool-call telemetry.
        #
        # Plugins keep writing ``_telemetry`` into the result they
        # return (in-process API unchanged); the runner-side
        # dispatcher strips it on serialization so the wire form has
        # ``result`` clean of telemetry side-channels.  The daemon-
        # side ``_forward_via_runner`` is symmetric: it re-injects
        # ``envelope.telemetry`` back into ``result["_telemetry"]``
        # for transitional compatibility with consumers that still
        # read from the result dict (jaato_session.py's OTel
        # forwarder).  Post-seat-flip cleanup will retire the
        # re-injection once consumers read ``envelope.telemetry``
        # directly.
        telemetry: Dict[str, Any] = {}
        if isinstance(result, dict) and "_telemetry" in result:
            lifted = result.pop("_telemetry")
            if isinstance(lifted, dict):
                telemetry = dict(lifted)

        env = ResponseEnvelope(
            id=request_id,
            ok=ok,
            result=result,
            error=error,
            telemetry=telemetry,
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
            # Phase 3 §3.3c part 3a: when a session host has been
            # bootstrapped, route through its executor so the full
            # runner-tier plugin set is reachable (not just the
            # cli-only Phase 2 ``execute_fn``).  Falls through to the
            # Phase 2 surface when no host is set — preserves
            # cli-only runners + tests.
            with self._session_lock:
                host = self._session_host
            if host is not None and host.session is not None:
                executor = getattr(host.session, "_executor", None)
                if executor is not None:
                    return self._dispatch_via_session_executor(
                        executor, tool_name, tool_args, env.id,
                    )
            return self._execute_fn(tool_name, tool_args)

        if env.method == "session.bootstrap":
            # Phase 3 §3.3c: daemon hands the runner a
            # SessionInitEnvelope; the runner constructs the live
            # JaatoSession host and stashes it for downstream
            # dispatch (Phase 4+ removes the daemon-side seat).
            return self._handle_session_bootstrap(env.args)

        if env.method == "session.health_check":
            # Phase 3 §3.3c precursor: read-only probe of the
            # runner-side session host's status.  Daemon uses this
            # to verify the bidirectional session-method dispatch
            # surface works (independent of the tool.execute path)
            # before §3.3c's full daemon-shell rewrite migrates
            # send_message / get_history / etc. through the same
            # surface.  Always returns ok=True with a status dict;
            # callers branch on ``ready`` / ``has_host``.
            return self._handle_session_health_check()

        if env.method == "session.get_session_state":
            # Phase 3 §3.3c precursor: read a single session-state
            # key from the runner-side JaatoSession.  args =
            # ``{"key": str, "default": Any}``.  Mirrors the
            # JaatoSession.get_session_state(key, default) shape.
            return self._handle_session_get_state(env.args)

        if env.method == "session.set_session_state":
            # Phase 3 §3.3c precursor: write a single session-state
            # key on the runner-side JaatoSession.  args =
            # ``{"key": str, "value": Any}``.  ``value`` must be
            # JSON-serializable per the JaatoSession contract;
            # daemon-side serialization already enforces this so
            # we just propagate.
            return self._handle_session_set_state(env.args)

        if env.method == "session.is_running":
            # Phase 3 §3.3c precursor: read-only probe — is a
            # message currently being processed?  Daemon's
            # ``client.is_processing`` check delegates here once
            # the seat-flip lands.
            return self._handle_session_is_running()

        if env.method == "session.request_stop":
            # Phase 3 §3.3c precursor: signal cancellation to the
            # runner-side JaatoSession's in-flight message.  args
            # = ``{"reason": str}``.  Returns whether a cancellation
            # was actually issued (False if no message was
            # running).
            return self._handle_session_request_stop(env.args)

        if env.method == "session.get_history":
            # Phase 3 §3.3c precursor: read the runner-side
            # JaatoSession's conversation history.  args = ``{}`` or
            # ``{"raw": bool}`` — when raw=True returns the
            # un-transformed view (premium pseudonymization
            # consumers); default returns the transformed view that
            # lives in the canonical container.
            return self._handle_session_get_history(env.args)

        if env.method == "session.get_context_usage":
            # Phase 3 §3.3c precursor: read-only snapshot of
            # context-window usage stats (model, total_tokens,
            # context_limit, percent_used, etc.).  Daemon-side
            # ``ContextUpdatedEvent`` emission delegates here once
            # the seat-flip lands.
            return self._handle_session_get_context_usage()

        if env.method == "session.get_context_limit":
            # Phase 3 §7b.1 precursor: read-only context-window
            # size in tokens.  Daemon-side falls back to this
            # when ``get_context_usage`` returns 0 / missing —
            # split out as its own RPC for that fallback path
            # (the alternative was extending the usage dict, but
            # callers want the limit independently of the usage
            # snapshot).
            return self._handle_session_get_context_limit()

        if env.method == "session.send_message":
            # Phase 3 §7b.2: the big one — runner-side
            # JaatoSession.send_message dispatched via runner-RPC.
            # Streams output chunks back through the existing
            # stream-frame channel (mirrors tool.execute's
            # _make_on_output bridge); cancellation propagates via
            # the existing cancel-frame mechanism + an on_cancel
            # hook into the session's request_stop.
            #
            # Long-running: returns when the model loop closes.
            # Wire shape: args = {"prompt": str}; result =
            # {"response": str}.
            return self._handle_session_send_message(env.args, env.id)

        if env.method == "session.shutdown":
            # Phase 3 §3.3c precursor: graceful runner-side session
            # teardown.  Calls the bootstrapped JaatoSession's
            # ``close_session`` (firing on_session_end hooks) and
            # drops the host reference.  Daemon-side
            # ``JaatoServer.shutdown`` will call this BEFORE
            # closing the RPC transport so plugins get a clean
            # teardown signal rather than being abruptly killed by
            # the runner-process exit.
            return self._handle_session_shutdown()

        if env.method == "session.set_terminal_width":
            # Phase 3 §3.3c precursor: write-only config — push
            # the daemon's terminal width to the runner so
            # enrichment notifications format correctly.  args =
            # ``{"width": int}``.
            return self._handle_session_set_terminal_width(env.args)

        if env.method == "session.set_streaming_enabled":
            # Phase 3 §3.3c precursor: write-only config — toggle
            # the session's streaming mode.  args =
            # ``{"enabled": bool}``.  Daemon's
            # ``client.set_streaming_enabled`` will delegate here.
            return self._handle_session_set_streaming_enabled(env.args)

        if env.method == "session.get_all_session_state":
            # Phase 3 §3.3c precursor: bulk-snapshot all session-
            # attached state.  Mirrors
            # ``JaatoSession.get_all_session_state()`` which
            # invokes every registered provider once + merges with
            # set-state values; provider values win on collision.
            # Used by the daemon at journal-save / waypoint-snapshot
            # / fork-snapshot time once the seat-flip migrates
            # those code paths.
            return self._handle_session_get_all_state()

        if env.method == "session.set_presentation_context":
            # Phase 3 §3.3c precursor: push the daemon's
            # PresentationContext (client display capabilities) to
            # the runner-side JaatoSession so its system-prompt
            # display-context block matches.  args =
            # ``{"context": <serialized PresentationContext dict>}``.
            return self._handle_session_set_presentation_context(env.args)

        if env.method == "session.reset":
            # Phase 3 §3.3c precursor: clear the runner-side
            # JaatoSession's conversation history.  Today only
            # supports the no-history "fresh reset" path —
            # restoring a saved history requires Message
            # round-trip serialization which is its own design
            # task.  args = ``{}``.
            return self._handle_session_reset()

        if env.method == "session.get_turn_accounting":
            # Phase 3 §3.3c precursor: read the runner-side
            # token-usage / timing per-turn list.  Daemon's
            # session-info / persistence paths use this for
            # telemetry attribution + journal save.
            return self._handle_session_get_turn_accounting()

        if env.method == "session.set_reference_authorizer":
            # Phase 3 §7c step 6.1: forward the daemon's
            # ReferenceAuthorizer state as a bool flag (the Python
            # object can't cross RPC; the runner-side references
            # plugin reads the flag + uses the existing
            # ``apparmor.add_reference_fragment`` runner→daemon
            # RPC to authorize paths).  args = ``{"enabled": bool}``.
            return self._handle_session_set_reference_authorizer(env.args)

        if env.method == "session.snapshot_instruction_budget":
            # Phase 3 §7c step 6.1: read the runner-side
            # JaatoSession's InstructionBudget snapshot for the
            # daemon's ``emit_current_state`` call site.  Returns
            # ``{"snapshot": <dict|None>}``.  None means the session
            # has no budget yet (pre-configure).  args = ``{}``.
            return self._handle_session_snapshot_instruction_budget()

        return False, {"error": f"unknown method: {env.method!r}"}

    def _dispatch_via_session_executor(
        self,
        executor: Any,
        tool_name: str,
        tool_args: Dict[str, Any],
        request_id: int,
    ) -> "tuple[bool, Any]":
        """Run *tool_name* through the bootstrapped session's executor.

        Phase 3 §3.3c part 3a.  The session's ``ToolExecutor`` (from
        ``shared/ai_tool_runner.py``) takes additional parameters
        beyond ``(name, args)`` — a streaming-output callback and a
        cancel token — that the runner-side dispatcher already
        manages via thread-local state.  This shim threads them
        through so the in-process plugin contract works
        runner-side without changes.

        The shared module's ``ToolExecutor.execute`` reads the
        cancel token + output callback from
        ``shared.ai_tool_runner._thread_local`` when not passed
        explicitly, so we install both via the same thread-local
        the session-side executor expects (mirroring what
        cli_runner does when forwarding through to ``run_command``).
        """
        # Build the per-call streaming adapter (adapts the runner's
        # ``on_output(source, text, mode)`` thread-local protocol to
        # whatever in-process executor expects).  The session's
        # executor reads ``_thread_local.tool_output_callback`` if
        # set.  Wire that to our own thread-local so chunks flow
        # through the runner's stream-frame channel.
        on_output = self._make_on_output(request_id)
        active_token = None
        with self._active_lock:
            active = self._active_calls.get(request_id)
        if active is not None:
            active_token = active.cancel_token

        # Bridge into the in-process thread-local that
        # ToolExecutor.execute reads from when callbacks are not
        # passed explicitly.
        try:
            from shared.ai_tool_runner import _thread_local as _ai_tl
        except ImportError:
            _ai_tl = None

        prior_cb = None
        prior_token = None
        if _ai_tl is not None:
            prior_cb = getattr(_ai_tl, "tool_output_callback", None)
            prior_token = getattr(_ai_tl, "cancel_token", None)
            _ai_tl.tool_output_callback = on_output
            if active_token is not None:
                _ai_tl.cancel_token = active_token

        try:
            ok, result = executor.execute(
                tool_name,
                tool_args,
                tool_output_callback=on_output,
                cancel_token=active_token,
            )
        finally:
            if _ai_tl is not None:
                _ai_tl.tool_output_callback = prior_cb
                _ai_tl.cancel_token = prior_token

        return ok, result

    def _handle_session_bootstrap(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Run the runner-side session bootstrap from a daemon envelope.

        Constructs a :class:`server.runner.session.RunnerSessionHost`
        from the supplied :class:`SessionInitEnvelope` and stashes it
        on the dispatcher.  Returns a small status dict the daemon
        can log + branch on (``ready``, ``session_id``, ``stage`` on
        failure).

        Idempotency: re-bootstrap with the same envelope returns
        ``ok`` without re-constructing.  Re-bootstrap with a
        different envelope is a hard failure — the daemon's spawn
        should issue exactly one bootstrap per runner.
        """
        from .envelope import SessionInitEnvelope
        from .session import (
            BootstrapError,
            RunnerSessionHost,
            bootstrap_session,
        )

        try:
            envelope = SessionInitEnvelope.from_dict(args)
        except (KeyError, ValueError) as exc:
            return False, {
                "error": f"session.bootstrap: invalid envelope: {exc}",
                "stage": "decode",
            }

        with self._session_lock:
            existing = self._session_host
            if existing is not None:
                if existing.envelope == envelope:
                    return True, {
                        "ok": True,
                        "ready": existing.is_ready,
                        "session_id": existing.session_id,
                        "note": "already bootstrapped (idempotent re-call)",
                    }
                return False, {
                    "error": (
                        f"session.bootstrap: runner already hosting "
                        f"session_id={existing.session_id!r}; refusing to "
                        f"re-bootstrap with different envelope"
                    ),
                    "stage": "duplicate",
                }

        try:
            host: RunnerSessionHost = bootstrap_session(envelope)
        except BootstrapError as exc:
            return False, {
                "error": f"session.bootstrap: {exc.message}",
                "stage": exc.stage,
            }

        with self._session_lock:
            self._session_host = host

        return True, {
            "ok": True,
            "ready": host.is_ready,
            "session_id": host.session_id,
        }

    def _handle_session_health_check(self) -> "tuple[bool, Any]":
        """Read-only probe of the runner-side session host's status.

        Phase 3 §3.3c precursor.  Returns a status dict the daemon
        can use to verify that the runner has bootstrapped a
        session AND that the bidirectional session-method dispatch
        surface is reachable (independent of the ``tool.execute``
        path).  This is the smallest non-bootstrap session-method
        RPC handler — proves the wiring before §3.3c's full
        daemon-shell rewrite adds ``session.send_message`` / etc.
        through the same surface.

        Returns:
            ``ok=True`` with a status dict carrying:

            - ``has_host`` (bool): True iff
              ``session.bootstrap`` has been called.
            - ``ready`` (bool): True iff the host has a configured
              JaatoSession (False during construction, after
              shutdown, or in test-stub mode).
            - ``session_id`` (str): The bootstrapped envelope's
              session_id; empty string when no host is set.
            - ``tool_count`` (int): Count of tool schemas the
              session's plugin registry exposes.  ``-1`` when the
              session is None / not yet ready / can't enumerate.
              Useful as a sanity check that the runner-side plugin
              set actually loaded.
        """
        with self._session_lock:
            host = self._session_host

        if host is None:
            return True, {
                "has_host": False,
                "ready": False,
                "session_id": "",
                "tool_count": -1,
            }

        tool_count = -1
        session = host.session
        if session is not None:
            try:
                runtime = getattr(session, "_runtime", None)
                registry = getattr(runtime, "registry", None) if runtime else None
                if registry is not None:
                    schemas = registry.get_exposed_tool_schemas()
                    tool_count = len(schemas)
            except Exception:  # noqa: BLE001 — probe must not raise
                tool_count = -1

        return True, {
            "has_host": True,
            "ready": host.is_ready,
            "session_id": host.session_id,
            "tool_count": tool_count,
        }

    def _require_ready_session(
        self,
    ) -> "tuple[bool, Any, Any]":
        """Common precondition for the session-state RPC handlers.

        Phase 3 §3.3c precursor.  Resolves the bootstrapped
        runner-side JaatoSession; returns a ``(ready, error_or_None,
        session_or_None)`` tuple.  ``ready=True`` means the session
        is bootstrapped + configured and the caller can use
        ``session`` directly; ``ready=False`` means an error tuple
        ready to return from the dispatcher.
        """
        with self._session_lock:
            host = self._session_host
        if host is None:
            return False, (False, {
                "error": "session not bootstrapped on this runner",
                "stage": "no_host",
            }), None
        session = host.session
        if session is None:
            return False, (False, {
                "error": (
                    "session host bootstrapped but JaatoSession is None "
                    "(test-stub mode or configure() failed)"
                ),
                "stage": "no_session",
            }), None
        return True, None, session

    def _handle_session_get_state(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Read a single session-state key from the runner-side
        JaatoSession.

        Args (over the wire): ``{"key": str, "default": Any}``.

        Returns:
            ``(True, {"value": Any})`` on success.  The value is
            whatever ``JaatoSession.get_session_state(key,
            default)`` returns — possibly ``None``, a primitive,
            or a JSON-serializable container.

            ``(False, {"error": ..., "stage": ...})`` when the
            session isn't bootstrapped or configure failed.
        """
        key = args.get("key")
        if not isinstance(key, str) or not key:
            return False, {
                "error": "session.get_session_state: missing 'key' arg",
                "stage": "decode",
            }
        default = args.get("default")
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        try:
            value = session.get_session_state(key, default)
        except Exception as exc:  # noqa: BLE001 — provider may raise
            return False, {
                "error": (
                    f"session.get_session_state: provider for {key!r} "
                    f"raised {type(exc).__name__}: {exc}"
                ),
                "stage": "provider",
            }
        return True, {"value": value}

    def _handle_session_set_state(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Write a single session-state key on the runner-side
        JaatoSession.

        Args: ``{"key": str, "value": Any}``.  ``value`` must be
        JSON-serializable per the
        :meth:`JaatoSession.set_session_state` contract — the
        JSON wire format already enforces this on the daemon
        side, but we re-check on receipt to surface a clean
        error if a non-serialisable value somehow crosses
        (e.g., a future binary-frame channel).
        """
        key = args.get("key")
        if not isinstance(key, str) or not key:
            return False, {
                "error": "session.set_session_state: missing 'key' arg",
                "stage": "decode",
            }
        if "value" not in args:
            return False, {
                "error": "session.set_session_state: missing 'value' arg",
                "stage": "decode",
            }
        value = args["value"]
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        try:
            session.set_session_state(key, value)
        except TypeError as exc:
            # JaatoSession raises TypeError when value isn't JSON-
            # serialisable.  Surface the underlying message so the
            # daemon-side caller can attribute the failure.
            return False, {
                "error": f"session.set_session_state: {exc}",
                "stage": "validate",
            }
        return True, {"ok": True}

    def _handle_session_is_running(self) -> "tuple[bool, Any]":
        """Read-only: is a message currently being processed?

        Mirrors :meth:`JaatoSession.is_running`.  Returns
        ``{"running": bool}`` on success, error envelope when no
        session is bootstrapped (so the daemon can distinguish
        "no session" from "session present but idle").
        """
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        return True, {"running": bool(session.is_running())}

    def _handle_session_get_history(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Read the runner-side JaatoSession's conversation history.

        Args: ``{"raw": bool}`` — optional; defaults to False.
        When True returns the un-transformed view via
        ``get_history_raw()`` (used by premium consumers that need
        the pre-pseudonymization form for user display); when
        False returns the canonical transformed view.

        Returns:
            ``(True, {"history": [<Message dict>, ...]})``.  Each
            message is serialized via ``to_dict()`` so the wire
            form is JSON-friendly.  History order is preserved
            (oldest first).

        Phase 3 §3.3c precursor.  Daemon-side
        ``client.get_history()`` will delegate here once the
        seat-flip migrates that surface.
        """
        raw = bool(args.get("raw", False))
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        try:
            messages = (
                session.get_history_raw() if raw
                else session.get_history()
            )
        except Exception as exc:  # noqa: BLE001 — read boundary
            return False, {
                "error": (
                    f"session.get_history: read failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "read",
            }

        history_dicts: list = []
        for msg in messages:
            try:
                history_dicts.append(_serialize_message_for_wire(msg))
            except Exception:  # noqa: BLE001 — boundary
                # Single-message serialization failure must not
                # drop the whole history — substitute a
                # placeholder so the count stays accurate and
                # the daemon can log the issue.
                history_dicts.append(
                    {"role": "system", "content": "<unserialisable>"},
                )
        return True, {"history": history_dicts}

    def _handle_session_get_context_limit(self) -> "tuple[bool, Any]":
        """Read-only context-window size in tokens (Phase 3 §7b.1
        precursor).

        Returns ``{"context_limit": int}`` on success.  The
        underlying :meth:`JaatoSession.get_context_limit` returns
        an int directly; we wrap it in a dict for symmetry with
        the other read handlers.

        Daemon-side callers use this as a fallback when
        ``session.get_context_usage()['context_limit']`` is 0 /
        missing (provider not yet initialized; usage dict
        contracts).
        """
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        try:
            limit = session.get_context_limit()
        except Exception as exc:  # noqa: BLE001 — read boundary
            return False, {
                "error": (
                    f"session.get_context_limit: read failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "read",
            }
        if not isinstance(limit, int) or limit < 0:
            return False, {
                "error": (
                    f"session.get_context_limit: expected non-negative "
                    f"int, got {limit!r}"
                ),
                "stage": "read",
            }
        return True, {"context_limit": limit}

    def _handle_session_get_context_usage(self) -> "tuple[bool, Any]":
        """Read-only snapshot of context-window usage stats.

        Returns the dict :meth:`JaatoSession.get_context_usage`
        produces (model, context_limit, total_tokens,
        prompt_tokens, output_tokens, turns, percent_used,
        tokens_remaining).  Wrapped in ``{"usage": <dict>}`` so
        the wire shape is symmetric with the other read handlers.
        """
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        try:
            usage = session.get_context_usage()
        except Exception as exc:  # noqa: BLE001 — read boundary
            return False, {
                "error": (
                    f"session.get_context_usage: read failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "read",
            }
        # Defensive: ensure the result is a dict even if a custom
        # session subclass returns something else.
        if not isinstance(usage, dict):
            return False, {
                "error": (
                    f"session.get_context_usage: expected dict, got "
                    f"{type(usage).__name__}"
                ),
                "stage": "read",
            }
        return True, {"usage": dict(usage)}

    def _handle_session_set_reference_authorizer(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Forward the daemon's ``ReferenceAuthorizer`` state to the
        runner-side ``JaatoSession`` as a bool flag.

        Phase 3 §7c step 6.1.  The actual ``ReferenceAuthorizer``
        Python object can't cross the RPC boundary (it holds a
        daemon-side ``AppArmorManager`` reference); the daemon
        translates ``authorizer is not None`` into a bool flag,
        and the runner-side session stores it via
        :meth:`JaatoSession.set_reference_authorization_enabled`.

        When the references plugin migrates runner-side, it reads
        the flag via :meth:`is_reference_authorization_enabled`
        and uses the existing ``apparmor.add_reference_fragment``
        runner→daemon RPC (Phase 3 §3.2.2) to authorize paths.
        The session_id for the RPC call is already known runner-
        side via the bootstrap envelope.

        Args: ``{"enabled": bool}``.  Returns ``{"ok": True}`` on
        success.  Coerces truthy non-bool values to bool — daemon
        callers pass a real bool but the coercion avoids spurious
        decode failures.

        Defensive contract: a missing ``enabled`` key surfaces as
        ``stage="decode"`` (not silently treated as ``False``); a
        spelling slip in the daemon-side wrapper would otherwise
        silently disable authorization.
        """
        if "enabled" not in args:
            return False, {
                "error": (
                    "session.set_reference_authorizer: 'enabled' key required"
                ),
                "stage": "decode",
            }
        enabled = bool(args["enabled"])
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        setter = getattr(session, "set_reference_authorization_enabled", None)
        if not callable(setter):
            return False, {
                "error": (
                    "session.set_reference_authorizer: session has no "
                    "set_reference_authorization_enabled method "
                    "(rolling-upgrade gap?)"
                ),
                "stage": "missing_method",
            }
        try:
            setter(enabled)
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.set_reference_authorizer: setter raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "set",
            }
        return True, {"ok": True}

    def _handle_session_snapshot_instruction_budget(
        self,
    ) -> "tuple[bool, Any]":
        """Read the runner-side JaatoSession's InstructionBudget
        snapshot for the daemon's ``emit_current_state`` call site.

        Phase 3 §7c step 6.1.  Pre-§7c the daemon read
        ``session.instruction_budget.snapshot()`` directly from
        the in-process JaatoSession (core.py:1091).  Post-§7c
        step 6.2 that read migrates to this RPC.

        Returns ``{"snapshot": <dict>}`` when the runner-side
        session has an instruction_budget configured (i.e.
        post-:meth:`JaatoSession.configure`).  Returns
        ``{"snapshot": None}`` when no budget exists yet — the
        daemon's caller treats None as "skip the
        InstructionBudgetEvent emit", matching pre-§7c behavior
        (the ``if session.instruction_budget:`` guard).

        The snapshot dict already includes ``session_id`` /
        ``agent_id`` / ``agent_type`` / ``context_limit`` /
        ``total_tokens`` / ``utilization_percent`` etc. — see
        :meth:`InstructionBudget.snapshot` for the full schema.
        Caller pulls ``agent_id`` from the returned dict directly;
        no separate RPC needed.

        Defensive contract: the snapshot may include nested dicts
        (``entries``); we ``copy.deepcopy`` to isolate daemon-side
        mutation from runner-side state.

        On read failure (e.g. ``snapshot()`` raises), returns a
        clean ``stage="read"`` error rather than crashing the
        runner.
        """
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        # Read the budget — None when session.configure() hasn't
        # populated it yet.  ``getattr`` so a JaatoSession variant
        # without the property surfaces as snapshot=None rather
        # than AttributeError (forward-compat).
        budget = getattr(session, "instruction_budget", None)
        if budget is None:
            return True, {"snapshot": None}
        snapshot_fn = getattr(budget, "snapshot", None)
        if not callable(snapshot_fn):
            return False, {
                "error": (
                    "session.snapshot_instruction_budget: instruction_budget "
                    "has no snapshot() method"
                ),
                "stage": "missing_method",
            }
        try:
            raw = snapshot_fn()
        except Exception as exc:  # noqa: BLE001 — read boundary
            return False, {
                "error": (
                    f"session.snapshot_instruction_budget: snapshot() raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "read",
            }
        if not isinstance(raw, dict):
            return False, {
                "error": (
                    f"session.snapshot_instruction_budget: expected dict, "
                    f"got {type(raw).__name__}"
                ),
                "stage": "read",
            }
        # Deep-copy to isolate daemon-side mutation; the snapshot
        # contains a nested ``entries`` dict whose values may be
        # further nested.
        import copy
        return True, {"snapshot": copy.deepcopy(raw)}

    def _handle_session_get_turn_accounting(self) -> "tuple[bool, Any]":
        """Read the runner-side per-turn token usage / timing list.

        Phase 3 §3.3c precursor.  Returns
        ``{"turns": [<dict>, ...]}``.  Each entry is the dict
        :meth:`JaatoSession.get_turn_accounting` produces (a
        list of per-turn account dicts).  Empty list when no
        turns recorded.
        """
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        try:
            turns = session.get_turn_accounting()
        except Exception as exc:  # noqa: BLE001 — read boundary
            return False, {
                "error": (
                    f"session.get_turn_accounting: read failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "read",
            }
        if not isinstance(turns, list):
            return False, {
                "error": (
                    f"session.get_turn_accounting: expected list, "
                    f"got {type(turns).__name__}"
                ),
                "stage": "read",
            }
        # Defensive: each entry should be a dict; copy to isolate
        # daemon-side mutation.
        return True, {"turns": [dict(t) if isinstance(t, dict) else t for t in turns]}

    def _handle_session_reset(self) -> "tuple[bool, Any]":
        """Clear the runner-side JaatoSession's conversation history.

        Phase 3 §3.3c precursor.  Calls
        ``JaatoSession.reset_session()`` with no history — fresh
        reset.  Restoring a saved history requires Message
        round-trip serialization (Message lacks ``from_dict``
        today) and lands as a separate handler when that design
        completes.

        Returns ``{"ok": True}`` on success.
        """
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        try:
            session.reset_session()
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.reset: reset_session raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "reset",
            }
        return True, {"ok": True}

    def _handle_session_set_presentation_context(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Push the daemon's PresentationContext to the runner-
        side JaatoSession (Phase 3 §3.3c precursor).

        Args: ``{"context": <dict>}`` — the serialized
        PresentationContext (Pydantic model).  Reconstructs the
        model on the runner side via ``model_validate`` /
        ``parse_obj`` so the JaatoSession setter receives the
        correct type.

        Returns ``{"ok": True}`` on success.

        Defensive: schema validation failures surface as
        ``stage="decode"`` errors so daemon-side callers can
        attribute the failure (vs the generic transport boundary).
        """
        ctx_dict = args.get("context")
        if not isinstance(ctx_dict, dict):
            return False, {
                "error": (
                    f"session.set_presentation_context: 'context' must "
                    f"be a dict; got {type(ctx_dict).__name__}"
                ),
                "stage": "decode",
            }
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        try:
            from jaato_sdk.events import PresentationContext
            # Pydantic v2 uses model_validate; v1 uses parse_obj.
            # Try v2 first, fall back to v1.
            ctor = (
                getattr(PresentationContext, "model_validate", None)
                or getattr(PresentationContext, "parse_obj", None)
            )
            if ctor is None:
                return False, {
                    "error": (
                        "session.set_presentation_context: cannot "
                        "reconstruct PresentationContext (no "
                        "model_validate / parse_obj)"
                    ),
                    "stage": "decode",
                }
            ctx = ctor(ctx_dict)
        except Exception as exc:  # noqa: BLE001 — schema validation
            return False, {
                "error": (
                    f"session.set_presentation_context: schema "
                    f"validation failed: {type(exc).__name__}: {exc}"
                ),
                "stage": "decode",
            }
        try:
            session.set_presentation_context(ctx)
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.set_presentation_context: setter raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "set",
            }
        return True, {"ok": True}

    def _handle_session_get_all_state(self) -> "tuple[bool, Any]":
        """Bulk-snapshot all session-attached state.

        Phase 3 §3.3c precursor.  Daemon-side journal-save /
        waypoint-snapshot / fork-snapshot paths will delegate
        here once the seat-flip migrates them.

        Returns ``{"state": <dict>}`` — a JSON-friendly snapshot
        merging set-state values + provider returns.  Provider
        values win on key collision (matches the underlying
        ``JaatoSession.get_all_session_state`` contract).
        Returned dict is a copy — daemon-side mutation doesn't
        propagate back into session state.

        On read failure (e.g. a provider raises), returns a clean
        ``stage="read"`` error rather than crashing the runner.
        """
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        getter = getattr(session, "get_all_session_state", None)
        if not callable(getter):
            return False, {
                "error": (
                    "session.get_all_session_state: session has no "
                    "get_all_session_state method"
                ),
                "stage": "missing_method",
            }
        try:
            snapshot = getter()
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.get_all_session_state: read failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "read",
            }
        if not isinstance(snapshot, dict):
            return False, {
                "error": (
                    f"session.get_all_session_state: expected dict, "
                    f"got {type(snapshot).__name__}"
                ),
                "stage": "read",
            }
        return True, {"state": dict(snapshot)}

    def _handle_session_set_terminal_width(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Push the daemon's terminal width to the runner-side
        JaatoSession (Phase 3 §3.3c precursor).

        Args: ``{"width": int}``.  Returns ``{"ok": True}`` on
        success.  Validates that width is a positive integer —
        terminal-width zero / negative is nonsensical and a
        common bug-detection signal.
        """
        width = args.get("width")
        if not isinstance(width, int) or width <= 0:
            return False, {
                "error": (
                    f"session.set_terminal_width: 'width' must be a "
                    f"positive int; got {width!r}"
                ),
                "stage": "decode",
            }
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        try:
            session.set_terminal_width(width)
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.set_terminal_width: setter raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "set",
            }
        return True, {"ok": True}

    def _handle_session_set_streaming_enabled(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Toggle the runner-side JaatoSession's streaming mode
        (Phase 3 §3.3c precursor).

        Args: ``{"enabled": bool}``.  Returns ``{"ok": True}`` on
        success.  Coerces truthy non-bool values to bool — daemon
        callers should pass actual booleans, but the coercion
        avoids spurious failures for ``enabled: 1`` etc. that
        commonly cross the JSON wire.
        """
        enabled = args.get("enabled")
        if enabled is None:
            return False, {
                "error": "session.set_streaming_enabled: missing 'enabled' arg",
                "stage": "decode",
            }
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        try:
            session.set_streaming_enabled(bool(enabled))
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.set_streaming_enabled: setter raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "set",
            }
        return True, {"ok": True}

    def _handle_session_send_message(
        self, args: Dict[str, Any], request_id: int,
    ) -> "tuple[bool, Any]":
        """Phase 3 §7b.2: the big one.

        Runs ``JaatoSession.send_message(prompt, on_output=...)`` on
        the bootstrapped runner-side session.  Streams output
        chunks back through the existing stream-frame channel
        (the daemon-side wrapper picks them up via the
        ``on_output`` callback the daemon-side caller provided).
        Cancellation propagates via the existing cancel-frame
        mechanism: when the daemon sends a ``cancel`` frame for
        this request_id, the dispatcher's
        :meth:`_handle_cancel` cancels
        ``_active_calls[request_id].cancel_token``; we hook
        ``on_cancel`` to call ``session.request_stop`` so the
        in-flight message wakes up and exits.

        Args (over the wire): ``{"prompt": str}``.

        Returns:
            ``(True, {"response": str})`` — the final model
            response text on success.
            ``(False, {"error": ..., "stage": ...})`` on validation
            failure / no-host / cancellation / exception.

        This handler is long-running.  The dispatcher's worker
        thread blocks on ``session.send_message`` for the
        duration of the model's function-calling loop; cancel
        frames during that window propagate cleanly via the
        on_cancel hook.

        Wire-cost note: model API calls happen runner-side via
        the runner's own JaatoRuntime + provider plugin (the
        §3.3b factory constructs a full runtime including
        provider).  Post-§7c the design intent (§4.2) is for
        providers to stay daemon-side and the runner to invoke
        them via a future ``client.complete`` callback primitive
        (§7b.3).  Until that lands, the runner-side provider is
        a transitional duplicate of the daemon's — duplicate
        cost but functionally correct.
        """
        prompt = args.get("prompt")
        if not isinstance(prompt, str):
            return False, {
                "error": "session.send_message: missing 'prompt' arg (str)",
                "stage": "decode",
            }
        ready, err, session = self._require_ready_session()
        if not ready:
            return err

        # Streaming bridge.  ``_make_on_output(request_id)`` returns
        # a callable that pumps each chunk onto the stream-frame
        # channel for the daemon-side wrapper to receive.  Same
        # mechanism tool.execute uses.
        on_output = self._make_on_output(request_id)

        # Cancel-frame hook.  When the daemon sends a cancel for
        # this request_id, _handle_cancel triggers
        # ``_active_calls[request_id].cancel_token.cancel()``.  We
        # install an on_cancel callback that propagates that into
        # the session's own request_stop so the in-flight model
        # loop exits at the next turn boundary.
        with self._active_lock:
            active = self._active_calls.get(request_id)

        def _on_dispatcher_cancel() -> None:
            try:
                session.request_stop(reason="rpc_cancel")
            except Exception:  # noqa: BLE001 — cancel best-effort
                logger.exception(
                    "session.send_message: request_stop raised during "
                    "cancel propagation",
                )

        if active is not None and active.cancel_token is not None:
            try:
                active.cancel_token.on_cancel(_on_dispatcher_cancel)
            except Exception:  # noqa: BLE001 — on_cancel optional
                logger.debug(
                    "session.send_message: cancel_token.on_cancel "
                    "registration raised; cancellation may not "
                    "propagate to session",
                )

        # Run the message loop.  Model API calls happen
        # synchronously here; output streams via on_output.
        try:
            response = session.send_message(prompt, on_output=on_output)
        except Exception as exc:  # noqa: BLE001 — boundary
            # Cancellation surfaces as a typed exception today
            # (CancelledException from the model_provider types).
            # Translate the cancel case to the dispatcher's
            # CancelledException error envelope so the daemon-side
            # wrapper sees the same shape as a tool.execute cancel.
            from jaato_sdk.plugins.model_provider.types import (
                CancelledException,
            )
            if isinstance(exc, CancelledException):
                return False, {
                    "error": str(exc) or "Cancelled",
                    "stage": "cancelled",
                }
            return False, {
                "error": (
                    f"session.send_message: model loop raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "send",
            }

        # Defensive: send_message returns str by contract; coerce
        # if a custom session subclass returns something else.
        if response is None:
            response = ""
        elif not isinstance(response, str):
            response = str(response)

        return True, {"response": response}

    def _handle_session_shutdown(self) -> "tuple[bool, Any]":
        """Graceful runner-side session teardown.

        Phase 3 §3.3c precursor.  Drops the bootstrapped
        :class:`RunnerSessionHost` (calling its session's
        ``close_session()`` if available so on_session_end hooks
        fire) and clears the dispatcher's ``_session_host``
        reference.  The runner process itself stays alive — the
        daemon's runner-RPC close ladder owns process termination.
        This handler is just the session-level lifecycle bookend
        that mirrors the bootstrap-then-shutdown cycle.

        Returns:
            ``(True, {"shutdown_session_id": str})`` on success.
            ``"shutdown_session_id"`` is the id of the session that
            was torn down, or empty string when no session was
            bootstrapped (no-op).

            ``(False, {"error": ..., "stage": ...})`` when
            ``close_session`` raised.

        Idempotent: re-calling after teardown returns success
        with empty session_id (mirrors the no-host case).
        """
        with self._session_lock:
            host = self._session_host
            self._session_host = None  # drop ref BEFORE close so
                                        # parallel handlers see the
                                        # post-shutdown state

        if host is None:
            return True, {"shutdown_session_id": ""}

        session_id = host.session_id
        session = host.session
        if session is not None:
            close = getattr(session, "close_session", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001 — boundary
                    logger.warning(
                        "session.shutdown: close_session for %s "
                        "raised %s — host already dropped, "
                        "surfacing error to daemon",
                        session_id, exc, exc_info=True,
                    )
                    return False, {
                        "error": (
                            f"session.shutdown: close_session raised "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        "stage": "close",
                    }

        return True, {"shutdown_session_id": session_id}

    def _handle_session_request_stop(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Signal cancellation to the runner-side JaatoSession's
        in-flight message.

        Args: ``{"reason": str}`` — optional; defaults to
        ``"user_cancelled"`` per the JaatoSession contract.

        Returns ``{"cancelled": bool}`` — True if a cancellation
        was actually issued (a message was running), False if no
        message was running (no-op).  Mirrors the boolean
        :meth:`JaatoSession.request_stop` returns.
        """
        reason = args.get("reason", "")
        if not isinstance(reason, str):
            reason = ""
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        cancelled = bool(session.request_stop(reason=reason))
        return True, {"cancelled": cancelled}

    @property
    def session_host(self):
        """Read accessor for the currently-bootstrapped session host.

        Returns ``None`` until ``session.bootstrap`` has been called
        successfully.  Phase 3 §3.4-§3.10 will route runner-tier
        plugin dispatch through ``host.session._executor`` instead
        of the cli-only Phase 2 ``execute_fn``; this property is
        the read seat for that future.
        """
        with self._session_lock:
            return self._session_host

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
                elif kind == KIND_RESPONSE:
                    # Phase 3 §3.2: response to an outgoing-from-runner
                    # call (e.g. ``client.prompt_operator``).  Look up
                    # the matching future and resolve.
                    try:
                        env = ResponseEnvelope.from_dict(payload)
                    except (KeyError, ValueError) as exc:
                        logger.error(
                            "runner RPC: malformed response frame: %s", exc,
                        )
                        continue
                    fut: Optional["_concurrent_futures.Future"]
                    with self._outgoing_lock:
                        fut = self._outgoing_calls.pop(env.id, None)
                    if fut is None:
                        logger.debug(
                            "runner RPC: response for unknown outgoing id=%d "
                            "— already cancelled?", env.id,
                        )
                        continue
                    if not fut.done():
                        fut.set_result(env)
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
            # Fail any outgoing calls awaiting daemon responses
            # (Phase 3 §3.2).  The daemon is gone or the loop hit
            # an unrecoverable error; callers blocked in
            # ``outgoing_call(...).result(timeout=...)`` need to
            # see a clear failure rather than wait out their timeout.
            with self._outgoing_lock:
                for fut in self._outgoing_calls.values():
                    if not fut.done():
                        fut.set_exception(
                            RuntimeError(
                                "runner RPC channel closed before response"
                            )
                        )
                self._outgoing_calls.clear()
            self._pool.shutdown(wait=False, cancel_futures=True)

    # ---------------------- runner → daemon outgoing -------------------

    def outgoing_call(
        self,
        method: str,
        args: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> ResponseEnvelope:
        """Send an outgoing request to the daemon; block until response.

        Phase 3 §3.2.  Used by runner-side plugins that need a
        daemon-tier capability — concretely:

        - ``client.prompt_operator`` from the permission plugin's
          ASK path (§3.2.1).
        - ``apparmor.add_reference_fragment`` from references'
          ``selectReferences`` admit path (§3.2.2).
        - ``telemetry.publish`` from the telemetry adapter (§3.15).

        Synchronous because the runner side runs plugin code in
        worker threads (not asyncio); callers block on
        ``Future.result(timeout)``.

        Args:
            method: Daemon-side handler name (e.g.
                ``"client.prompt_operator"``).
            args: Args dict passed to the handler.
            timeout: Wall-clock cap; ``None`` means wait
                indefinitely.  Callers SHOULD set a finite
                timeout in production to avoid wedging on a
                half-dead daemon.

        Returns:
            The :class:`ResponseEnvelope` parsed from the daemon's
            response.  Caller inspects ``ok`` / ``result`` /
            ``error`` per the typed-envelope contract (§4.8).

        Raises:
            RuntimeError: when the channel is closed.
            concurrent.futures.TimeoutError: when *timeout* fires
                before the response arrives.
        """
        if self._closed:
            raise RuntimeError("runner RPC channel is closed")

        with self._outgoing_lock:
            request_id = self._next_outgoing_id
            self._next_outgoing_id += 1

        fut: "_concurrent_futures.Future" = _concurrent_futures.Future()
        with self._outgoing_lock:
            self._outgoing_calls[request_id] = fut

        env = RequestEnvelope(id=request_id, method=method, args=args or {})
        self._write(env.to_dict())

        try:
            return fut.result(timeout=timeout)
        finally:
            with self._outgoing_lock:
                self._outgoing_calls.pop(request_id, None)

    def shutdown(self) -> None:
        """Initiate shutdown from outside the serve loop."""
        self._closed = True
        # Fail any in-flight outgoing calls with a clean error.
        with self._outgoing_lock:
            for fut in self._outgoing_calls.values():
                if not fut.done():
                    fut.set_exception(
                        RuntimeError(
                            "runner RPC channel closed before response"
                        )
                    )
            self._outgoing_calls.clear()
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _serialize_message_for_wire(msg: Any) -> Any:
    """JSON-friendly serialization of a conversation Message
    (Phase 3 §3.3c precursor).

    Used by the runner-side ``session.get_history`` handler.  Tries
    in order:

    1. ``msg.to_dict()`` if defined — custom message types (test
       doubles, future opt-in serializers) win.
    2. :func:`dataclasses.asdict` for plain dataclasses (the real
       ``shared.plugins.model_provider.types.Message`` shape).  Enum
       values are coerced to their ``.value`` form recursively so
       JSON encoders don't choke.
    3. Pass-through (last resort — caller's try/except will catch).

    The wire form must round-trip cleanly to ``json.dumps`` — the
    runner-side framing uses JSON throughout.
    """
    to_dict = getattr(msg, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    import dataclasses
    if dataclasses.is_dataclass(msg) and not isinstance(msg, type):
        return _coerce_for_json(dataclasses.asdict(msg))
    return msg


def _coerce_for_json(value: Any) -> Any:
    """Recursively coerce dataclass-derived dicts to JSON-friendly
    primitives.  Enums → their ``.value``; nested dicts/lists
    recursed; everything else passed through."""
    import enum
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _coerce_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [_coerce_for_json(v) for v in value]
    return value


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
