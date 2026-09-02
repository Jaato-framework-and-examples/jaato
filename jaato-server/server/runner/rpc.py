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
from typing import Any, Callable, Dict, List, Optional, Protocol

from shared.framing import (
    FrameTooLargeError,
    read_frame_sync,
    write_frame_sync,
)

from jaato_sdk.plugins.model_provider.types import (
    CancelToken,
    DISCOVERABILITY_EAGER,
    DISCOVERABILITY_DEFERRED,
)

from .envelope import (
    KIND_CANCEL,
    KIND_REQUEST,
    KIND_RESPONSE,
    KIND_STREAM,
    STREAM_CHANNEL_DISPLAY,
    CancelFrame,
    ErrorPayload,
    NotificationFrame,
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


#: RPC methods that RUN MODEL OR USER CODE, and are therefore unbounded in
#: duration.  These get the WORK lane.
#:
#: **The criterion is "does this handler run model or user code?"**, not "is
#: it slow today".  That matters because the second question cannot be
#: answered by whoever adds the next verb and the first one can: a handler
#: that calls the provider, replays the model loop, or invokes a tool or a
#: user command belongs here; one that reads or sets session state does not.
#:
#: WHY THE SPLIT EXISTS.  Every method except ``session.bootstrap`` used to
#: share one 8-worker pool.  ``session.send_message`` holds a worker for an
#: ENTIRE TURN and ``tool.execute`` is called with ``timeout=None``, while
#: ``session.offer_message`` is a lock, a bool read and a list append.  A
#: control-plane operation's latency was bounded by the slowest work in the
#: pool -- so a delivery could be reported ``unreachable`` on a 2s timeout
#: because an unrelated tool was still running.
#:
#: ``session.bootstrap`` is in NEITHER set: it runs synchronously on the main
#: thread so ``aa_change_profile`` confines the thread that later spawns the
#: workers (per-thread in the kernel apparmor module).  See the dispatch site.
WORK_LANE_METHODS = frozenset({
    "tool.execute",              # runs the tool
    "session.send_message",      # runs an entire turn
    "session.replay_messages",   # re-runs the model loop
    "session.execute_user_command",   # runs a user command
    "echo",                      # §8.3 RPC-overhead benchmark; deliberately
                                 # in the work lane so a benchmark cannot
                                 # measure the control lane's latency
})

#: Runs on the main thread, in neither pool.  Kept as an explicit name so the
#: "every method is classified" guard can account for it.
MAIN_THREAD_METHODS = frozenset({"session.bootstrap"})


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
        control_workers: int = 4,
        workspace_root: Optional[str] = None,
    ) -> None:
        """Construct the dispatcher.

        Args:
            sock: The inherited socketpair fd (typically fd 3).
            execute_fn: Tool-execution callable.
            max_workers: Concurrent cap for the WORK lane -- the methods
                in :data:`WORK_LANE_METHODS`, which run model or user code
                and are unbounded in duration.
            control_workers: Concurrent cap for the CONTROL lane, which
                serves everything else.  Small on purpose: the work there is
                a lock and a dict lookup, so this is about never QUEUEING
                behind a turn, not about throughput.
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
            thread_name_prefix="runner-rpc-work",
        )
        # Separate lane so a control-plane RPC never waits behind a turn or a
        # tool.  Distinct thread_name_prefix so a stack dump says which lane
        # a wedged thread is in.
        self._control_pool = ThreadPoolExecutor(
            max_workers=control_workers,
            thread_name_prefix="runner-rpc-ctl",
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

    def emit_notification(
        self,
        request_id: int,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a notification frame for the given in-flight call.

        Phase 3 §7c step 6.6.4.1.  Runner-side session callbacks
        (instruction-budget updates, retry notifications, etc.)
        call this to surface events back to the daemon during
        long-running RPCs (currently :meth:`session.send_message`).
        The daemon-side per-call notification handler (registered
        via :meth:`RunnerRPCClient.call`'s ``on_notification``
        kwarg) demuxes by ``event_type`` and routes to the
        appropriate ``server.emit(<Event>)`` or other action.

        Per the §7c step 6.6.2 audit (commit 9f28f96d): this is
        the wire-format extension the audit's "stream-channel
        multiplex" rationale called for.  Same wire socket as
        :class:`StreamFrame` (output chunks); different ``kind``
        discriminator (``"event"`` vs ``"stream"``).

        Used by §7c step 6.6.4.2's 7-callback collapse.

        Args:
            request_id: The in-flight call's id (must match the
                outer ``RequestEnvelope.id``).
            event_type: Discriminator for the daemon-side demux.
                Caller and consumer agree on the set; the protocol
                doesn't validate per-event-type contracts.
            payload: Event-type-specific dict.  Defaults to empty
                dict for parameter-less notifications.
        """
        frame = NotificationFrame(
            id=request_id,
            event_type=event_type,
            payload=dict(payload or {}),
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
        import threading as _thr   # [RPC_DIAG] register-stall trace — DIAG BRANCH
        logger.info(
            "[RPC_DIAG] _handle_request ENTER method=%s id=%s tid=%s",
            env.method, env.id, _thr.get_ident())
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
                # A domain-failure dict may carry its own frames -- see the
                # model-loop catch in ``_handle_session_send_message``.  It is
                # the only source of a traceback on this branch, because
                # nothing raised: without reading it, ``ErrorPayload.traceback``
                # is None and every consumer downstream gets the summary line
                # and nothing else.
                err = ErrorPayload(
                    type="ToolError",
                    message=_extract_error_message(result),
                    traceback=_extract_error_traceback(result),
                )
                self._emit_response(
                    env.id, ok=False, result=result, error=err,
                )
        finally:
            logger.info(   # [RPC_DIAG] register-stall trace — DIAG BRANCH
                "[RPC_DIAG] _handle_request EXIT method=%s id=%s", env.method, env.id)
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

        if env.method == "session.end":
            # Phase 2 cascade-sharing: daemon signals end of one
            # cascade-session.  Runner calls plugin.reset_for_next_session()
            # on every initialized plugin so the slot's plugin state
            # is clean before the next session of the same cascade
            # claims this slot.  See docs/design/runner-cascade-sharing.md.
            return self._handle_session_end()

        if env.method == "subagent.forward_event":
            # Phase 4 §4.3.6b: daemon forwards an event from an
            # isolated sub-runner back to this parent runner.  The
            # handler looks up the SubagentPlugin via the live
            # JaatoRuntime's registry and routes to its
            # ``receive_forwarded_event`` method, which mirrors the
            # default-share path's ``inject_prompt`` contract.
            return self._handle_subagent_forward_event(env.args)

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

        if env.method == "session.try_completion_nudge":
            # Phase 3 §7c step 6.6.4.3a: atomic check-and-increment
            # for the completion-nudge guard.  Collapses 3 daemon-
            # side private-attr reaches (read
            # ``_signal_completion_called`` + read/inc
            # ``_completion_nudges_fired``) into one round-trip.
            # args = ``{"max_nudges": int}``.  Returns
            # ``{"should_nudge": bool, "nudges_fired": int}``.
            return self._handle_session_try_completion_nudge(env.args)

        if env.method == "session.try_drain_pending_user":
            # Multi-turn deadlock fix: after a turn ends, atomically pop a
            # pending high-priority (USER/PARENT/SYSTEM) message that raced
            # into the turn wind-down and was queued with no active turn to
            # drain it.  args = ``{}``.  Returns ``{"text": str | None}``.
            return self._handle_session_try_drain_pending_user()

        if env.method == "session.get_auth_info":
            # Phase 3 §7c step 6.6.4.5c.1: read provider-credential
            # source string from the runner-side session.  Replaces
            # the daemon-side ``self._jaato.auth_info`` reach.
            # args = ``{}``.  Returns ``{"auth_info": str}``.
            return self._handle_session_get_auth_info()

        if env.method == "session.get_user_commands":
            # Phase 3 §7c step 6.6.4.5c.2: read the runner-side
            # session's user-command catalog.  Replaces 2 daemon-side
            # reaches into ``self._jaato.get_user_commands()``.
            # args = ``{}``.  Returns
            # ``{"commands": {<name>: <UserCommand-as-dict>, ...}}``.
            # Wire shape per the 5c.2 audit decision: dict-shape-only
            # (Path B) — UserCommand + CommandParameter are NamedTuples
            # with primitive fields, no callables to strip.
            return self._handle_session_get_user_commands()

        if env.method == "session.execute_user_command":
            # Phase 3 §7c step 6.6.4.5c.3: invoke a user command on
            # the runner-side session.  Replaces the daemon-side reach
            # into ``self._jaato.execute_user_command(name, args)``.
            # args = ``{"name": str, "args": dict}``.  Returns
            # ``{"result": <tagged-dict>, "shared": bool}`` where the
            # tagged dict is one of:
            #   {"_kind": "HelpLines", "lines": [[text, style], ...]}
            #   {"_kind": "dict", "value": <json-dict>}
            #   {"_kind": "str", "value": <str>}  (other types coerced)
            # Wire shape per the 5c.3 audit (Path A bounded to 3 cases):
            # daemon does structured access on HelpLines.lines and
            # dict keys for "model" / IPC return; everything else is
            # display-only and stringifies safely.
            return self._handle_session_execute_user_command(env.args)

        if env.method == "session.get_model_completions":
            # Phase 3 §7c step 6.6.4.5c.4: get completion candidates
            # for the "model" command's subcommand arguments.  Replaces
            # 2 daemon-side reaches: core.py:4285 (model-name list)
            # and command_router.py:1149 (model-subcommand expansion).
            # args = ``{"args": List[str]}``.  Returns
            # ``{"completions": [{"value": str, "description": str}, ...]}``.
            # Wire shape per the 5c.4 audit decision: dict-shape-only
            # (Path A) — CommandCompletion is a NamedTuple with
            # primitive fields (value, description), no callables.
            return self._handle_session_get_model_completions(env.args)

        if env.method == "session.register_client_tools":
            return self._handle_session_register_client_tools(env.args)

        if env.method == "session.get_tool_schemas":
            # Phase 3 §7c step 6.6.4.5c.5: read the runner-side
            # session's resolved tool schemas (preloaded plugins +
            # on-demand activations).  Replaces 2 daemon-side
            # reaches: core.py:1407 (tool-ID registry) and
            # core.py:3759 (signal_completion_in_surface filter).
            # args = ``{}``.  Returns ``{"schemas": [<dict>, ...]}``
            # with each entry mapping ToolSchema fields directly
            # except for ``traits: FrozenSet[str]`` which becomes
            # ``traits: List[str]`` on the wire.
            # Wire shape per the 5c.5 audit decision: dict-shape-only
            # (Path A) — pre-impl grep verified all 7 ToolSchema
            # fields + the nested EditableContent fields are JSON-
            # encodable.  Daemon callsites read only ``.name`` and
            # ``.category`` (primitive str), so the migration is
            # behavior-preserving.
            return self._handle_session_get_tool_schemas()

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

        if env.method == "session.apply_budget_degrade":
            # Mid-flight cascade degrade: the shared pool crossed a rung and
            # this still-running child must degrade too, rather than keeping
            # the ceiling it was handed at spawn.
            return self._handle_session_apply_budget_degrade(env.args or {})

        if env.method == "session.get_budget_usage":
            # The session's ABSOLUTE budget consumption per dimension, as
            # the per-session BudgetTracker accumulated it (per RESPONSE).
            # A cascade pool reconciles against this rather than summing an
            # event stream, which has proven both duplicable and droppable.
            return self._handle_session_get_budget_usage(env.args)

        if env.method == "session.restore_budget_usage":
            # Counterpart to get_budget_usage.  Without it a reloaded session
            # ran with a zeroed tracker and no cross-turn ceiling could fire.
            return self._handle_session_restore_budget_usage(env.args)

        if env.method == "session.get_budget_exhausted":
            # The enforcement latch, read at save time so it can travel with
            # the usage snapshot.
            return self._handle_session_get_budget_exhausted()

        if env.method == "session.get_context_limit":
            # Phase 3 §7b.1 precursor: read-only context-window
            # size in tokens.  Daemon-side falls back to this
            # when ``get_context_usage`` returns 0 / missing —
            # split out as its own RPC for that fallback path
            # (the alternative was extending the usage dict, but
            # callers want the limit independently of the usage
            # snapshot).
            return self._handle_session_get_context_limit()

        if env.method == "session.get_rendered_system_instruction":
            # Issue #787: the system instruction as it stood at the end
            # of ``configure()`` — after prefetch expansion, before the
            # runtime additions.  The daemon persists it so a revive
            # RESTORES the prompt instead of re-deriving it (which
            # re-ran mandatory prefetch scripts against an empty
            # ``agent_params`` and made such sessions unwakeable).
            return self._handle_session_get_rendered_system_instruction()

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

        if env.method == "session.offer_message":
            # Step 2: ATOMIC queue-or-report.  The session is the authority on
            # whether it is mid-turn; the daemon holds a replica that clears
            # LATER (only once ``session.send_message`` returns and its model
            # thread unwinds), so a delivery decided daemon-side can be
            # decided on stale state -- and a message queued into a turn that
            # has already ended is drained by nothing.  args = ``{"text":
            # str, "source_id": str?, "source_type": str?}``; returns
            # ``{"outcome": "queued"|"needs_turn"}``.
            return self._handle_session_offer_message(env.args)

        if env.method == "session.inject_prompt":
            # Phase 3 §7c step 6.1: inject a prompt into the
            # runner-side session's message queue (mid-turn or
            # idle-routed based on source_type).  Replaces the
            # daemon-side direct call at core.py:3238.  args =
            # ``{"text": str, "source_id": str?, "source_type":
            # str?}`` where source_type is the SourceType enum's
            # .value (e.g. "user", "parent", "child", "system",
            # "event").
            return self._handle_session_inject_prompt(env.args)

        if env.method == "session.set_initial_history":
            # Phase 3 §7c step 6.6.1.1: seed an empty session
            # with replayed conversation history from a SessionState
            # snapshot.  Replaces the daemon-side direct call at
            # session_manager.py:2130.  args = ``{"messages":
            # [<serialized message dict>, ...]}`` — the daemon
            # serializes via shared/plugins/session/serializer.py
            # and the runner deserializes there too, so the wire
            # carries the same JSON-compatible shape disk
            # persistence already uses.
            return self._handle_session_set_initial_history(env.args)

        if env.method == "session.restore_turn_accounting":
            # Phase 3 §7c step 6.6.1.2: replace the runner-side
            # session's per-turn token-usage / timing list from a
            # SessionState snapshot.  Replaces the daemon-side
            # private-attr write at session_manager.py:2558-2559
            # (now public ``JaatoSession.restore_turn_accounting``
            # since §7c step 6.6.1.0).  args = ``{"turns":
            # [<dict>, ...]}`` — turns are already JSON-native
            # dicts in the persistence serializer (no special
            # wrapper needed; per the same wire-shape-reuse
            # rationale as 6.6.1.1).
            return self._handle_session_restore_turn_accounting(env.args)

        if env.method == "session.restore_conversation_budget":
            # Phase 3 §7c step 6.6.1.3: restore the runner-side
            # session's CONVERSATION instruction-budget entry from
            # a SessionState snapshot.  Replaces the daemon-side
            # reach at session_manager.py:2592-2593 through
            # ``session.instruction_budget.restore_conversation_from_snapshot``
            # (now public
            # ``JaatoSession.restore_conversation_budget`` since
            # §7c step 6.6.1.0).  args = ``{"snapshot": <dict>}``
            # — the snapshot is a JSON-native dict produced by
            # ``InstructionBudget.get_conversation_snapshot()`` /
            # ``SourceEntry.to_dict()`` (no special wrapper).
            return self._handle_session_restore_conversation_budget(env.args)

        if env.method == "session.append_history_message":
            # Phase 3 §7c step 6.6.3.1: append a single message
            # to the runner-side session's history.  Replaces the
            # daemon-side get-modify-reset dance at
            # session_manager.py:2855 (interrupted-tool-call
            # recovery path).  args = ``{"message": <serialized
            # message dict>}`` — wire shape reuses
            # shared/plugins/session/serializer.py's
            # serialize_message / deserialize_message round-trip
            # (same wire-shape-reuse rationale as 6.6.1.1's
            # set_initial_history).
            return self._handle_session_append_history_message(env.args)

        if env.method == "session.snapshot_conversation_budget":
            # Phase 3 §7c step 6.6.3.2: return the runner-side
            # session's CONVERSATION instruction-budget snapshot
            # for persistence-save.  Inverse of
            # ``session.restore_conversation_budget`` (6.6.1.3).
            # Replaces the daemon-side reach at
            # session_manager.py:2986 through
            # ``session.instruction_budget.get_conversation_snapshot``.
            # args = ``{}``.  Returns ``{"snapshot": <dict|None>}``
            # — None when no budget configured (pre-configure).
            return self._handle_session_snapshot_conversation_budget()

        if env.method == "session.set_parallel_tools_override":
            # Phase 3 §7c step 6.6.3.3: stash a per-turn
            # override for parallel-tool execution on the
            # runner-side session.  Replaces the daemon-side
            # private-attr write at session_manager.py:4096
            # (now public ``JaatoSession.set_parallel_tools_override``
            # since §7c step 6.6.3.0).  args = ``{"enabled": bool}``.
            # Override is consumed once and cleared after the
            # next turn boundary; this RPC's lifecycle is per-
            # turn-pre-send_message.
            return self._handle_session_set_parallel_tools_override(env.args)

        if env.method == "session.replay_messages":
            # Phase 3 §7c step 6.6.3.4: run a one-shot completion
            # against an arbitrary message list (capability
            # primitive for session-manipulation tools — fork /
            # interrogate / replay).  Replaces the daemon-side
            # call at session_manager.py:4338.  args =
            # ``{"messages": [<dict>, ...], "timeout": float?}``
            # — messages serialized via the existing
            # serialize_history (same wire-shape-reuse rationale
            # as 6.6.1.1's set_initial_history + 6.6.3.1's
            # append_history_message).  Returns
            # ``{"response_text": str}``.
            #
            # Blocking (the underlying ``replay_messages`` waits
            # for exclusive provider access) — daemon caller
            # already runs in a worker thread per the pre-§7c
            # pattern at session_manager.py:4336.
            return self._handle_session_replay_messages(env.args)

        if env.method == "session.resolve_fork_point":
            # Phase 3 §7c step 6.6.3.5: resolve a fork-point
            # specifier (after_message / after_tool_call /
            # after_timestamp) to a message index in the
            # session's history.  Replaces the daemon-side call
            # at session_manager.py:4362 (ResolveForkPointRequest
            # SDK handler).  args = ``{"after_message": int?,
            # "after_tool_call": str?, "after_timestamp": str?,
            # "history": [<dict>, ...]?}`` — history is optional;
            # runner defaults to ``session.get_history()`` (the
            # daemon caller's existing pattern at line 4363).
            # Returns ``{"fork_index": int}``.  Pure read; no
            # cancel surface, no streaming.
            return self._handle_session_resolve_fork_point(env.args)

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

    def _handle_subagent_forward_event(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Route a daemon-forwarded sub-runner event to the
        SubagentPlugin (Phase 4 §4.3.6b).

        Looks up the SubagentPlugin from the bootstrapped JaatoSession
        host's runtime registry, then dispatches the event to its
        ``receive_forwarded_event`` method.  The plugin translates
        the event into ``inject_prompt`` on the parent session,
        mirroring the default-share path so the parent model sees
        isolated + in-runner subagent events identically.

        Expected args:
            subagent_id (str): The subagent id from the spawn-time
                response.
            event_kind (str): "output" | "status" | "error" (open
                for forward-compat).
            event_payload (dict): Event-kind-specific payload.

        Returns:
            ``(True, {"ok": True})`` on success.
            ``(False, {"error": "..."})`` when no host bootstrapped,
            no SubagentPlugin available, or the plugin rejected the
            event.  Daemon-side caller logs but doesn't retry.
        """
        # Validate args.
        subagent_id = args.get("subagent_id")
        event_kind = args.get("event_kind")
        event_payload = args.get("event_payload")
        if not isinstance(subagent_id, str) or not subagent_id:
            return False, {
                "error": "subagent.forward_event: subagent_id required",
            }
        if not isinstance(event_kind, str) or not event_kind:
            return False, {
                "error": "subagent.forward_event: event_kind required",
            }
        if not isinstance(event_payload, dict):
            return False, {
                "error": (
                    "subagent.forward_event: event_payload must be a dict"
                ),
            }

        # Find the runtime's plugin registry via the bootstrapped
        # host.  Without a host, there's no plugin to dispatch to —
        # log + fail.
        with self._session_lock:
            host = self._session_host
        if host is None or host.session is None:
            return False, {
                "error": (
                    "subagent.forward_event: no session host bootstrapped"
                ),
            }

        runtime = getattr(host, "runtime", None)
        registry = getattr(runtime, "_registry", None) if runtime else None
        if registry is None:
            return False, {
                "error": (
                    "subagent.forward_event: no plugin registry on runtime"
                ),
            }

        # Plugin lookup — by canonical name 'subagent'.  Test fakes
        # may have different registry shapes; tolerate get_plugin
        # absence + dict-style access.
        subagent_plugin = None
        if hasattr(registry, "get_plugin"):
            try:
                subagent_plugin = registry.get_plugin("subagent")
            except Exception:  # noqa: BLE001
                pass
        if subagent_plugin is None and hasattr(registry, "_plugins"):
            subagent_plugin = registry._plugins.get("subagent")  # type: ignore[attr-defined]
        if subagent_plugin is None:
            return False, {
                "error": (
                    "subagent.forward_event: SubagentPlugin not loaded"
                ),
            }

        if not hasattr(subagent_plugin, "receive_forwarded_event"):
            return False, {
                "error": (
                    "subagent.forward_event: plugin lacks "
                    "receive_forwarded_event method (older version)"
                ),
            }

        try:
            result = subagent_plugin.receive_forwarded_event(
                subagent_id=subagent_id,
                event_kind=event_kind,
                event_payload=event_payload,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "subagent.forward_event: plugin raised for "
                "subagent_id=%s event_kind=%s",
                subagent_id, event_kind,
            )
            return False, {
                "error": (
                    f"plugin raised: {type(exc).__name__}: {exc}"
                ),
            }

        # Plugin returned {ok, error?}.  Propagate as the RPC response.
        if not isinstance(result, dict):
            return False, {
                "error": (
                    f"plugin returned non-dict: {type(result).__name__}"
                ),
            }
        if result.get("ok"):
            return True, result
        return False, result

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

        # Phase 3 §7c Step 7.2: attach the runner-internal
        # ``RunnerRPCClient`` to the runner-side registry as
        # ``registry.runner_rpc_client``.  Runner-side plugins (the
        # permission plugin in §3.7) detect runner-side execution
        # context via this attribute and route ASKs through the
        # daemon's ``client.prompt_operator`` RPC (registered
        # daemon-side in §7c Step 7.1).  Pre-§7.2 the attribute
        # was never assigned in production; the permission
        # plugin's ``_get_runner_rpc_channel`` lookup always
        # returned None and silently fell back to the in-process
        # channel (orphaned post-seat-flip).
        try:
            if host.runtime is not None and getattr(
                host.runtime, "_registry", None,
            ) is not None:
                from .rpc_client import RunnerRPCClient
                runner_rpc_client = RunnerRPCClient(self)
                setattr(
                    host.runtime._registry,
                    "runner_rpc_client",
                    runner_rpc_client,
                )
        except Exception:  # noqa: BLE001 — best-effort wiring
            logger.exception(
                "session.bootstrap: failed to wire "
                "registry.runner_rpc_client (Step 7.2)",
            )

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

    def _handle_session_end(self) -> "tuple[bool, Any]":
        """Cascade-sharing session boundary — reset per-session plugin state.

        Phase 2.  Called once by the daemon at session teardown for
        sessions served from the pool slot.  Iterates every plugin
        registered in the runner-side session's plugin registry and
        invokes ``reset_for_next_session()`` on it.  Per-plugin
        decisions about what to reset / preserve are owned by the
        plugin (Phase 1 audit shipped in PR #160 + #161).

        On any per-plugin reset exception, the plugin's name + the
        exception text are appended to the ``errors`` list and the
        sweep continues.  Caller (daemon) treats a non-empty
        ``errors`` list as a slot-poisoning signal: the slot must
        NOT be returned to the pool because its plugin state is
        partially-reset (undefined).

        Returns:
            ``(True, {"plugins_reset": int, "errors": List[str]})`` —
            ``ok`` stays True even when individual plugin resets
            fail; daemon branches on ``errors``.  ``(False, error)``
            only on the structural "no session host" case (which is
            a programmer error: daemon shouldn't call session.end
            when no session was bootstrapped).
        """
        ready, err, session = self._require_ready_session()
        if not ready:
            return err

        runtime = getattr(session, "_runtime", None)
        registry = getattr(runtime, "registry", None) if runtime else None
        if registry is None:
            return False, {
                "error": (
                    "session.end: runner-side session has no plugin "
                    "registry — cannot fire reset_for_next_session"
                ),
                "stage": "no_registry",
            }

        plugins_reset = 0
        errors: List[str] = []
        # Fire on_session_end hooks BEFORE clearing plugin state.
        # ``close_session`` snapshots the current session state +
        # notifies the session_plugin's ``on_session_end`` hook.
        # Doing this before ``reset_for_next_session`` means hooks
        # see the live, pre-reset state of every plugin (matches
        # ``session.shutdown`` semantics at L3450).
        close = getattr(session, "close_session", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001 — boundary
                # Best-effort: close_session failing must not block
                # the slot return.  Append to errors so the daemon
                # branches on it; the slot won't be returned to the
                # pool (cascade_returned stays False on errors !=
                # []).  See JaatoServer.shutdown cascade-return path.
                errors.append(
                    f"close_session: {type(exc).__name__}: {exc}"
                )

        for name in registry.list_available():
            plugin = registry.get_plugin(name)
            if plugin is None:
                continue
            reset = getattr(plugin, "reset_for_next_session", None)
            if reset is None:
                # Plugin is on an older base class — default no-op
                # behavior (skip silently).  This is rare: Phase 1
                # added reset_for_next_session to both ToolPlugin
                # and EnrichmentPlugin protocols.
                continue
            try:
                reset()
                plugins_reset += 1
            except Exception as exc:  # noqa: BLE001 — per-plugin boundary
                errors.append(f"{name}: {type(exc).__name__}: {exc}")

        # PR #174 hotfix (server 0.6.151+): clear the runner-side
        # session host so the next ``session.bootstrap`` on this slot
        # (cascade reuse path) is NOT rejected by the
        # already-bootstrapped guard at line 1077.  Pre-fix: cascade
        # reuse succeeded at the daemon-side slot routing layer but
        # the runner's bootstrap handler refused with
        # ``ToolError: runner already hosting session_id=...; refusing
        # to re-bootstrap with different envelope`` (because the old
        # session_host stayed installed).  The old session's model
        # loop kept running on the runner instead of being replaced;
        # surfaced by peer's kb-orchestrator v152-retry-7 cascade
        # 2026-05-21 as a duplicate ``cascade_after_discovery`` fire
        # 30s after the supposed reuse.
        #
        # Mirrors ``_handle_session_shutdown`` at L3439 — drop the
        # session_host reference under the session_lock so parallel
        # handlers see the post-end state.  Done AFTER plugin reset
        # so any reset() that reads ``self._session_host`` (e.g. for
        # logging context) still sees the right value.
        with self._session_lock:
            self._session_host = None

        return True, {
            "plugins_reset": plugins_reset,
            "errors": errors,
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

    def _handle_session_get_rendered_system_instruction(
        self,
    ) -> "tuple[bool, Any]":
        """Read-only snapshot of the configure-time system instruction.

        Returns ``{"rendered_system_instruction": str | None}``.  ``None``
        is a legitimate answer (the session has not been configured yet),
        so it is NOT an error — the daemon persists nothing in that case
        and the revive falls back to re-rendering, which is exactly the
        pre-#787 behaviour.

        See :meth:`JaatoSession.get_rendered_system_instruction` for why
        this is the frozen render rather than the live attribute.
        """
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        getter = getattr(session, "get_rendered_system_instruction", None)
        if not callable(getter):
            # Older runner-side session object (mixed-build test stubs).
            # Absent is the same answer as "nothing rendered yet".
            return True, {"rendered_system_instruction": None}
        try:
            rendered = getter()
        except Exception as exc:  # noqa: BLE001 — read boundary
            return False, {
                "error": (
                    f"session.get_rendered_system_instruction: read "
                    f"failed: {type(exc).__name__}: {exc}"
                ),
                "stage": "read",
            }
        if rendered is not None and not isinstance(rendered, str):
            return False, {
                "error": (
                    f"session.get_rendered_system_instruction: expected "
                    f"str or None, got {type(rendered).__name__}"
                ),
                "stage": "read",
            }
        return True, {"rendered_system_instruction": rendered}

    def _handle_session_apply_budget_degrade(
        self, params: "dict",
    ) -> "tuple[bool, Any]":
        """Apply cascade-pushed degrade rungs to this running session."""
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        apply = getattr(session, "apply_cascade_degrade", None)
        if not callable(apply):
            return False, {"error": "session has no apply_cascade_degrade"}
        return True, apply(
            params.get("rungs") or [], params.get("pool_pressure"))

    def _handle_session_get_budget_exhausted(self) -> "tuple[bool, Any]":
        """Why a budget ceiling stopped this session, or ``None``."""
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        reader = getattr(session, "budget_exhausted_reason", None)
        return True, {"reason": reader() if callable(reader) else None}

    def _handle_session_restore_budget_usage(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Re-seed the runner session's budget usage from a snapshot.

        Args (over the wire): ``{"usage": {"turns": 2.0, ...}}``.

        Returns ``(True, {"restored": bool})`` -- False when the session runs
        unbudgeted, which is not an error.
        """
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        args = args or {}
        usage = args.get("usage") or {}
        reason = args.get("exhausted_reason")
        restorer = getattr(session, "restore_budget_usage", None)
        if not callable(restorer):
            return True, {"restored": False}
        restorer(usage)
        # The ENFORCEMENT latch travels with the usage: restoring one without
        # the other leaves a session at its ceiling that still serves a turn.
        latch = getattr(session, "restore_budget_exhausted", None)
        if callable(latch):
            latch(reason)
        return True, {"restored": bool(usage), "exhausted": bool(reason)}

    def _handle_session_get_budget_usage(
        self, args: Optional[Dict[str, Any]] = None,
    ) -> "tuple[bool, Any]":
        """Read-only snapshot of the session's absolute budget consumption.

        Wrapped as ``{"usage": {...}}`` for symmetry with the other read
        handlers.  Empty dict when the session tracks no budget.

        ``args["tracker_only"]`` forwards to
        :meth:`JaatoSession.get_budget_usage` and suppresses the unbudgeted
        ``{"tokens": N}`` fallback -- see that method for why persisting the
        fallback destroys the snapshot it overwrites.
        """
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        tracker_only = bool((args or {}).get("tracker_only", False))
        try:
            getter = getattr(session, "get_budget_usage", None)
            usage = getter(tracker_only=tracker_only) if callable(getter) else {}
        except Exception as exc:  # noqa: BLE001
            return False, {"error": f"session.get_budget_usage: {exc}"}
        return True, {"usage": dict(usage or {})}

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

    def _handle_session_offer_message(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Atomically enqueue a message, or report that a turn is needed.

        The runner-side half of step 2.  Delegates to
        :meth:`JaatoSession.offer_message`, which holds ``_delivery_lock``
        across the check-and-enqueue so it cannot interleave with the turn's
        ``_is_running = False`` flip.

        Distinct from ``session.inject_prompt`` on purpose: inject QUEUES
        unconditionally and answers only "did that raise", which is the same
        answer whether a drain is coming or the message will sit forever.
        This verb answers the question the caller actually has.

        Wire shape mirrors ``session.inject_prompt`` so the two are decodable
        by the same client code.  Returns ``{"outcome": "queued"}`` or
        ``{"outcome": "needs_turn"}``; the daemon starts the turn on the
        latter, since a session cannot start its own.
        """
        from shared.message_queue import SourceType

        text = args.get("text")
        if not isinstance(text, str):
            return False, {
                "error": (
                    f"session.offer_message: 'text' must be a str; "
                    f"got {type(text).__name__}"
                ),
                "stage": "decode",
            }
        source_id = args.get("source_id")
        if source_id is not None and not isinstance(source_id, str):
            return False, {
                "error": (
                    f"session.offer_message: 'source_id' must be a str "
                    f"or omitted; got {type(source_id).__name__}"
                ),
                "stage": "decode",
            }
        source_type_str = args.get("source_type")
        source_type_enum: Any = None
        if source_type_str is not None:
            if not isinstance(source_type_str, str):
                return False, {
                    "error": (
                        f"session.offer_message: 'source_type' must be a str "
                        f"or omitted; got {type(source_type_str).__name__}"
                    ),
                    "stage": "decode",
                }
            try:
                source_type_enum = SourceType(source_type_str)
            except ValueError:
                valid = sorted(s.value for s in SourceType)
                return False, {
                    "error": (
                        f"session.offer_message: 'source_type' must be one "
                        f"of {valid}; got {source_type_str!r}"
                    ),
                    "stage": "decode",
                }
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        offer = getattr(session, "offer_message", None)
        if not callable(offer):
            return False, {
                "error": (
                    "session.offer_message: session has no offer_message "
                    "method (rolling-upgrade gap?)"
                ),
                "stage": "missing_method",
            }
        require_idle = args.get("require_idle", False)
        if not isinstance(require_idle, bool):
            return False, {
                "error": (
                    f"session.offer_message: 'require_idle' must be a bool "
                    f"or omitted; got {type(require_idle).__name__}"
                ),
                "stage": "decode",
            }
        try:
            outcome = offer(
                text, source_id=source_id, source_type=source_type_enum,
                require_idle=require_idle,
            )
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.offer_message: offer raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "offer",
            }
        return True, {"outcome": outcome}

    def _handle_session_inject_prompt(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Inject a prompt into the runner-side session's message queue.

        Phase 3 §7c step 6.1.  Replaces the pre-§7c daemon-side
        call ``self._jaato.get_session().inject_prompt(text, ...)``
        at server/core.py:3238.

        ``JaatoSession.inject_prompt`` accepts a ``SourceType`` enum
        instance; the wire carries its string value (e.g. "user")
        which the handler maps back to the enum.

        Args:
            text: required str — the prompt text.
            source_id: optional str — sender identifier (defaults
                to "unknown" inside JaatoSession).
            source_type: optional str — ANY
                :class:`shared.message_queue.SourceType` value.
                Deliberately NOT re-listed here: the enum owns the
                set, and a prose copy drifts the moment one is added
                (``sibling`` shipped and this list kept saying five).
                The runtime check below reads the enum, so it was
                only the DOCUMENTATION that was wrong — which is its
                own hazard, because a caller trusts it.
                Tier semantics — who may interrupt a turn in progress
                — live in ``HIGH_PRIORITY_SOURCES`` /
                ``IDLE_ONLY_SOURCES``.  Defaults to "user" inside
                JaatoSession.

        Returns ``{"ok": True}`` on success.

        Defensive contract: ``text`` is REQUIRED — missing key
        surfaces as ``stage="decode"`` rather than silently
        injecting an empty prompt.  Unknown ``source_type`` value
        surfaces as ``stage="decode"`` rather than collapsing to
        the framework default — protects against typos that would
        misroute message priority.
        """
        from shared.message_queue import SourceType

        text = args.get("text")
        if not isinstance(text, str):
            return False, {
                "error": (
                    f"session.inject_prompt: 'text' must be a str; "
                    f"got {type(text).__name__}"
                ),
                "stage": "decode",
            }
        source_id = args.get("source_id")
        if source_id is not None and not isinstance(source_id, str):
            return False, {
                "error": (
                    f"session.inject_prompt: 'source_id' must be a str "
                    f"or omitted; got {type(source_id).__name__}"
                ),
                "stage": "decode",
            }
        source_type_str = args.get("source_type")
        source_type_enum: Any = None
        if source_type_str is not None:
            if not isinstance(source_type_str, str):
                return False, {
                    "error": (
                        f"session.inject_prompt: 'source_type' must be a str "
                        f"or omitted; got {type(source_type_str).__name__}"
                    ),
                    "stage": "decode",
                }
            try:
                source_type_enum = SourceType(source_type_str)
            except ValueError:
                valid = sorted(s.value for s in SourceType)
                return False, {
                    "error": (
                        f"session.inject_prompt: 'source_type' must be one "
                        f"of {valid}; got {source_type_str!r}"
                    ),
                    "stage": "decode",
                }
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        injector = getattr(session, "inject_prompt", None)
        if not callable(injector):
            return False, {
                "error": (
                    "session.inject_prompt: session has no inject_prompt "
                    "method (rolling-upgrade gap?)"
                ),
                "stage": "missing_method",
            }
        try:
            injector(text, source_id=source_id, source_type=source_type_enum)
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.inject_prompt: inject raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "inject",
            }
        return True, {"ok": True}

    def _handle_session_set_initial_history(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Seed the runner-side session with replayed conversation history.

        Phase 3 §7c step 6.6.1.1.  Replaces the pre-§7c daemon-
        side call ``jaato_session.set_initial_history(initial_history)``
        in ``server/session_manager.py:2130`` (the
        ``create_headless_session`` path that seeds
        ``initial_history`` before the first ``send_message``).

        Wire shape: ``{"messages": [<dict>, ...]}`` where each
        dict is a serialized :class:`Message` from
        ``shared.plugins.session.serializer.serialize_message``.
        The runner deserializes via
        :func:`shared.plugins.session.serializer.deserialize_history`
        — the same code disk persistence uses, so the wire-shape
        is the round-trippable JSON format already proven in
        ``test_serializer.py``.

        Defensive contract:

          - ``messages`` must be a list (missing key or non-list
            → ``stage="decode"``).
          - Per-element decode failures (malformed Part type,
            missing role) surface as ``stage="decode"`` with
            the underlying serializer error, not as a partial
            seed.
          - :meth:`JaatoSession.set_initial_history` itself
            raises ``RuntimeError`` if the session is not idle
            or its history is non-empty — the daemon-side
            persistence-restore path never violates this, but a
            defensive ``stage="set"`` wraps it to surface
            misuse cleanly.

        Args: ``{"messages": List[Dict[str, Any]]}``.
        Returns: ``{"ok": True}`` on success.
        """
        messages_data = args.get("messages")
        if messages_data is None:
            return False, {
                "error": (
                    "session.set_initial_history: 'messages' key required"
                ),
                "stage": "decode",
            }
        if not isinstance(messages_data, list):
            return False, {
                "error": (
                    f"session.set_initial_history: 'messages' must be a list; "
                    f"got {type(messages_data).__name__}"
                ),
                "stage": "decode",
            }

        try:
            from shared.plugins.session.serializer import deserialize_history
            messages = deserialize_history(messages_data)
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.set_initial_history: deserialize failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "decode",
            }

        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        setter = getattr(session, "set_initial_history", None)
        if not callable(setter):
            return False, {
                "error": (
                    "session.set_initial_history: session has no "
                    "set_initial_history method (rolling-upgrade gap?)"
                ),
                "stage": "missing_method",
            }
        try:
            setter(messages)
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.set_initial_history: set_initial_history raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "set",
            }
        return True, {"ok": True}

    def _handle_session_restore_turn_accounting(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Replace the runner-side session's per-turn token-usage /
        timing list from a SessionState snapshot.

        Phase 3 §7c step 6.6.1.2.  Replaces the pre-§7c daemon-
        side private-attr write at
        ``server/session_manager.py:2558-2559``:

            jaato_session._turn_accounting = list(state.turn_accounting)

        Now wraps the public method
        :meth:`JaatoSession.restore_turn_accounting` added in
        §7c step 6.6.1.0 (commit 13ce5939).

        Wire shape: ``{"turns": [<dict>, ...]}`` — turn entries
        are already JSON-native dicts in the persistence
        serializer (see
        ``shared/plugins/session/serializer.py:215`` which stores
        them verbatim under the ``turn_accounting`` key).  No
        special encode/decode needed.

        Defensive contract:

          - ``turns`` must be a list (missing key or non-list
            → ``stage="decode"``).
          - Each element should be a dict; we don't validate
            element schema (turn-accounting entry shape evolves
            with provider integrations) but reject non-dict
            elements at the boundary to surface wire-corruption
            issues cleanly.
          - The session-side method takes a copy via ``list(turns)``
            so caller-side mutation can't propagate; the handler
            doesn't need to copy again.

        Args: ``{"turns": List[Dict[str, Any]]}``.
        Returns: ``{"ok": True}`` on success.
        """
        turns = args.get("turns")
        if turns is None:
            return False, {
                "error": (
                    "session.restore_turn_accounting: 'turns' key required"
                ),
                "stage": "decode",
            }
        if not isinstance(turns, list):
            return False, {
                "error": (
                    f"session.restore_turn_accounting: 'turns' must be a "
                    f"list; got {type(turns).__name__}"
                ),
                "stage": "decode",
            }
        for i, entry in enumerate(turns):
            if not isinstance(entry, dict):
                return False, {
                    "error": (
                        f"session.restore_turn_accounting: 'turns[{i}]' "
                        f"must be a dict; got {type(entry).__name__}"
                    ),
                    "stage": "decode",
                }
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        setter = getattr(session, "restore_turn_accounting", None)
        if not callable(setter):
            return False, {
                "error": (
                    "session.restore_turn_accounting: session has no "
                    "restore_turn_accounting method (rolling-upgrade gap?)"
                ),
                "stage": "missing_method",
            }
        try:
            setter(turns)
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.restore_turn_accounting: setter raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "set",
            }
        return True, {"ok": True}

    def _handle_session_restore_conversation_budget(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Restore the runner-side session's CONVERSATION instruction-
        budget entry from a SessionState snapshot.

        Phase 3 §7c step 6.6.1.3.  Replaces the pre-§7c daemon-
        side reach at ``server/session_manager.py:2592-2593``:

            jaato_session.instruction_budget.restore_conversation_from_snapshot(
                state.budget_state)

        Now wraps the public method
        :meth:`JaatoSession.restore_conversation_budget` added
        in §7c step 6.6.1.0 (commit 13ce5939).  The public
        method is no-op when ``self._instruction_budget`` is
        None (pre-:meth:`configure`); this handler preserves
        that semantic — a "successful" no-op rather than an
        error.

        Wire shape: ``{"snapshot": <dict>}`` — the snapshot is
        a JSON-native dict produced by
        :meth:`InstructionBudget.get_conversation_snapshot` /
        :meth:`SourceEntry.to_dict` (see
        instruction_budget.py:399).  No special wrapper needed;
        the wire shape IS the persistence shape.

        Defensive contract:

          - ``snapshot`` must be a dict (missing key or non-
            dict → ``stage="decode"``).
          - Empty / falsy snapshot is permitted: the underlying
            method is documented as no-op when ``snapshot`` is
            empty (instruction_budget.py:407 ``if not snapshot:
            return``).  Treat ``{"snapshot": {}}`` as success.
          - Per-key schema NOT validated — accepts any dict
            shape.  The underlying ``restore_conversation_from_snapshot``
            handles its own structural checks (gc_policy enum,
            children recursion); a malformed snapshot surfaces
            as ``stage="set"``.

        Args: ``{"snapshot": Dict[str, Any]}``.
        Returns: ``{"ok": True}`` on success.
        """
        if "snapshot" not in args:
            return False, {
                "error": (
                    "session.restore_conversation_budget: 'snapshot' key required"
                ),
                "stage": "decode",
            }
        snapshot = args["snapshot"]
        if not isinstance(snapshot, dict):
            return False, {
                "error": (
                    f"session.restore_conversation_budget: 'snapshot' must "
                    f"be a dict; got {type(snapshot).__name__}"
                ),
                "stage": "decode",
            }
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        setter = getattr(session, "restore_conversation_budget", None)
        if not callable(setter):
            return False, {
                "error": (
                    "session.restore_conversation_budget: session has no "
                    "restore_conversation_budget method (rolling-upgrade "
                    "gap?)"
                ),
                "stage": "missing_method",
            }
        try:
            setter(snapshot)
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.restore_conversation_budget: setter raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "set",
            }
        return True, {"ok": True}

    def _handle_session_append_history_message(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Append a single message to the runner-side session's
        history.

        Phase 3 §7c step 6.6.3.1.  Replaces the pre-§7c daemon-
        side get-modify-reset dance at
        ``server/session_manager.py:2855`` (interrupted-tool-
        call recovery path).  Wraps the public method
        :meth:`JaatoSession.append_history_message` added in
        §7c step 6.6.3.0.

        Wire shape: ``{"message": <dict>}`` where the dict is a
        serialized :class:`Message` from
        ``shared.plugins.session.serializer.serialize_message``.
        Same wire-shape-reuse rationale as 6.6.1.1's
        ``set_initial_history``: no new wire format invented.

        Defensive contract:

          - ``message`` must be a dict (missing key or non-dict
            → ``stage="decode"``).
          - Per-element decode failures (missing role, unknown
            Part type) surface as ``stage="decode"`` with the
            underlying serializer error.
          - Underlying ``append_history_message`` calls
            ``reset_session(modified_history)``, which clears
            ``_turn_accounting`` as a side effect — the daemon
            caller's existing semantic is preserved exactly.

        Args: ``{"message": Dict[str, Any]}``.
        Returns: ``{"ok": True}`` on success.
        """
        message_data = args.get("message")
        if message_data is None:
            return False, {
                "error": (
                    "session.append_history_message: 'message' key required"
                ),
                "stage": "decode",
            }
        if not isinstance(message_data, dict):
            return False, {
                "error": (
                    f"session.append_history_message: 'message' must be a "
                    f"dict; got {type(message_data).__name__}"
                ),
                "stage": "decode",
            }
        try:
            from shared.plugins.session.serializer import deserialize_message
            message = deserialize_message(message_data)
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.append_history_message: deserialize failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "decode",
            }
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        appender = getattr(session, "append_history_message", None)
        if not callable(appender):
            return False, {
                "error": (
                    "session.append_history_message: session has no "
                    "append_history_message method (rolling-upgrade gap?)"
                ),
                "stage": "missing_method",
            }
        try:
            appender(message)
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.append_history_message: appender raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "set",
            }
        return True, {"ok": True}

    def _handle_session_resolve_fork_point(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Resolve a fork-point specifier to a message index in
        the session's history.

        Phase 3 §7c step 6.6.3.5.  Replaces the pre-§7c daemon-
        side call at ``server/session_manager.py:4362``.  Wraps
        the existing public method
        :meth:`JaatoSession.resolve_fork_point` (already at
        jaato_session.py:8455; no missing-method gap — this is
        the second of the 5 prerequisites without an
        encapsulation cleanup).

        Wire shape:

          - ``after_message`` (optional int): direct message
            index.
          - ``after_tool_call`` (optional str): tool-call id;
            returns the index of the message containing the
            FunctionCall.id or its ToolResult.
          - ``after_timestamp`` (optional str): HH:MM:SS or ISO
            timestamp; returns the index of the last message at
            or before this time.
          - ``history`` (optional list): the history to search.
            When omitted, the runner uses
            ``session.get_history()`` (the daemon caller's
            existing pattern at line 4363).

          Exactly one of the 3 specifiers SHOULD be set.  The
          underlying method documents that "if none are given,
          returns the last message index" — the handler
          preserves that semantic.

        Returns: ``{"fork_index": int}``.

        Defensive contract:

          - All 3 specifiers optional; underlying method handles
            the all-None case gracefully.
          - 'history' (when provided) must be a list of dicts.
            Per-element decode failures surface as
            stage="decode".
          - When 'history' is omitted, the runner reads
            ``session.get_history()``.
          - resolve_fork_point may raise on malformed timestamp
            strings; defensively wrapped as stage="resolve".

        Args: ``{"after_message": int?, "after_tool_call": str?,
                 "after_timestamp": str?, "history": List[Dict]?}``.
        Returns: ``{"fork_index": int}``.
        """
        after_message = args.get("after_message")
        if after_message is not None and not isinstance(after_message, int):
            return False, {
                "error": (
                    f"session.resolve_fork_point: 'after_message' must be "
                    f"int or omitted; got {type(after_message).__name__}"
                ),
                "stage": "decode",
            }
        after_tool_call = args.get("after_tool_call")
        if after_tool_call is not None and not isinstance(after_tool_call, str):
            return False, {
                "error": (
                    f"session.resolve_fork_point: 'after_tool_call' must "
                    f"be str or omitted; got {type(after_tool_call).__name__}"
                ),
                "stage": "decode",
            }
        after_timestamp = args.get("after_timestamp")
        if after_timestamp is not None and not isinstance(after_timestamp, str):
            return False, {
                "error": (
                    f"session.resolve_fork_point: 'after_timestamp' must "
                    f"be str or omitted; got {type(after_timestamp).__name__}"
                ),
                "stage": "decode",
            }

        history_data = args.get("history")
        history: Any = None
        if history_data is not None:
            if not isinstance(history_data, list):
                return False, {
                    "error": (
                        f"session.resolve_fork_point: 'history' must be a "
                        f"list or omitted; got {type(history_data).__name__}"
                    ),
                    "stage": "decode",
                }
            try:
                from shared.plugins.session.serializer import deserialize_history
                history = deserialize_history(history_data)
            except Exception as exc:  # noqa: BLE001 — boundary
                return False, {
                    "error": (
                        f"session.resolve_fork_point: deserialize failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "stage": "decode",
                }

        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        resolver = getattr(session, "resolve_fork_point", None)
        if not callable(resolver):
            return False, {
                "error": (
                    "session.resolve_fork_point: session has no "
                    "resolve_fork_point method (rolling-upgrade gap?)"
                ),
                "stage": "missing_method",
            }
        # Default history to session.get_history() when omitted —
        # matches the daemon caller's existing pattern at
        # session_manager.py:4363.
        if history is None:
            history_getter = getattr(session, "get_history", None)
            if not callable(history_getter):
                return False, {
                    "error": (
                        "session.resolve_fork_point: session has no "
                        "get_history method (cannot default 'history' arg)"
                    ),
                    "stage": "missing_method",
                }
            try:
                history = history_getter()
            except Exception as exc:  # noqa: BLE001 — boundary
                return False, {
                    "error": (
                        f"session.resolve_fork_point: get_history raised "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "stage": "read",
                }
        try:
            fork_index = resolver(
                history=history,
                after_message=after_message,
                after_tool_call=after_tool_call,
                after_timestamp=after_timestamp,
            )
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.resolve_fork_point: resolver raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "resolve",
            }
        if not isinstance(fork_index, int):
            return False, {
                "error": (
                    f"session.resolve_fork_point: expected int fork_index; "
                    f"got {type(fork_index).__name__}"
                ),
                "stage": "resolve",
            }
        return True, {"fork_index": fork_index}

    def _handle_session_replay_messages(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Run a one-shot completion against an arbitrary message
        list and return the model's text response.

        Phase 3 §7c step 6.6.3.4.  Replaces the pre-§7c daemon-
        side call at ``server/session_manager.py:4338``.  Wraps
        the existing public method
        :meth:`JaatoSession.replay_messages` (already at
        jaato_session.py:8252; no missing-method gap).

        Wire shape:
          - ``messages``: List of serialized Message dicts (from
            ``shared.plugins.session.serializer.serialize_message``).
            Same wire-shape-reuse rationale as 6.6.1.1's
            ``set_initial_history`` + 6.6.3.1's
            ``append_history_message``.
          - ``timeout`` (optional): float seconds.  Defaults to
            120.0 (matching the underlying method's default).
          - Returns ``{"response_text": str}`` on success.

        Blocking semantic:

          The underlying ``replay_messages`` waits for exclusive
          provider access (so concurrent in-flight turn calls
          serialize).  This handler runs synchronously inside
          the runner's RPC dispatcher.  Daemon caller's pre-§7c
          pattern was to invoke from a worker thread — that
          pattern is preserved post-seat-flip via the daemon-
          side wrapper's normal awaitable path (the dispatcher
          loop on the daemon side won't block a single RPC for
          minutes).

        Defensive contract:

          - 'messages' must be a list of dicts (missing key,
            non-list, or non-dict element → ``stage="decode"``).
          - 'timeout' (optional) must be a number > 0 if
            provided.
          - Per-element decode failures (malformed Part type,
            missing role) surface as ``stage="decode"`` with
            the underlying serializer error.
          - The session-side method may raise on provider
            errors; defensively wrapped as ``stage="replay"``
            (distinct from the standard ``stage="set"`` since
            this is an active completion, not a pure setter).

        Args: ``{"messages": List[Dict], "timeout": float?}``.
        Returns: ``{"response_text": str}`` on success.
        """
        messages_data = args.get("messages")
        if messages_data is None:
            return False, {
                "error": (
                    "session.replay_messages: 'messages' key required"
                ),
                "stage": "decode",
            }
        if not isinstance(messages_data, list):
            return False, {
                "error": (
                    f"session.replay_messages: 'messages' must be a list; "
                    f"got {type(messages_data).__name__}"
                ),
                "stage": "decode",
            }

        timeout = args.get("timeout", 120.0)
        if timeout is not None:
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                return False, {
                    "error": (
                        f"session.replay_messages: 'timeout' must be a "
                        f"positive number; got {timeout!r}"
                    ),
                    "stage": "decode",
                }

        try:
            from shared.plugins.session.serializer import deserialize_history
            messages = deserialize_history(messages_data)
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.replay_messages: deserialize failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "decode",
            }

        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        replayer = getattr(session, "replay_messages", None)
        if not callable(replayer):
            return False, {
                "error": (
                    "session.replay_messages: session has no "
                    "replay_messages method (rolling-upgrade gap?)"
                ),
                "stage": "missing_method",
            }
        try:
            response_text = replayer(messages, timeout=float(timeout))
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.replay_messages: replay raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "replay",
            }
        # Coerce to str — defensive against a custom session
        # subclass returning non-str (matches §7c step 6.1 (3/3)
        # send_message's response coercion pattern).
        return True, {"response_text": str(response_text or "")}

    def _handle_session_set_parallel_tools_override(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Stash a per-turn parallel-tools override on the runner-
        side session.

        Phase 3 §7c step 6.6.3.3.  Replaces the pre-§7c daemon-
        side private-attr write at
        ``server/session_manager.py:4096``:

            jaato_session._parallel_tools_override = event.parallel_tools

        Now wraps the public method
        :meth:`JaatoSession.set_parallel_tools_override` added
        in §7c step 6.6.3.0.

        Wire shape: ``{"enabled": bool}``.  Coerces truthy /
        falsy non-bool values to bool (matches §7c step 6.1
        (1/3)'s ``set_reference_authorizer`` pattern).

        Defensive contract:

          - 'enabled' key REQUIRED — missing surfaces as
            stage="decode" rather than silently treating as
            False (which would silently disable parallel-tool
            execution; opposite of operator intent).
          - Setter never raises in practice (it's a single
            attribute write), but defensively wraps in
            stage="set".

        Args: ``{"enabled": bool}``.
        Returns: ``{"ok": True}`` on success.
        """
        if "enabled" not in args:
            return False, {
                "error": (
                    "session.set_parallel_tools_override: 'enabled' key required"
                ),
                "stage": "decode",
            }
        enabled = bool(args["enabled"])
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        setter = getattr(session, "set_parallel_tools_override", None)
        if not callable(setter):
            return False, {
                "error": (
                    "session.set_parallel_tools_override: session has no "
                    "set_parallel_tools_override method (rolling-upgrade gap?)"
                ),
                "stage": "missing_method",
            }
        try:
            setter(enabled)
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.set_parallel_tools_override: setter raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "set",
            }
        return True, {"ok": True}

    def _handle_session_snapshot_conversation_budget(
        self,
    ) -> "tuple[bool, Any]":
        """Return the runner-side session's CONVERSATION
        instruction-budget snapshot for persistence-save.

        Phase 3 §7c step 6.6.3.2.  Inverse of
        :meth:`_handle_session_restore_conversation_budget`
        (6.6.1.3).  Replaces the pre-§7c daemon-side reach at
        ``server/session_manager.py:2986``:

            jaato_session.instruction_budget.get_conversation_snapshot()

        Now wraps the public method
        :meth:`JaatoSession.snapshot_conversation_budget` added
        in §7c step 6.6.3.0.

        Returns ``{"snapshot": <dict>}`` when the runner-side
        session has a budget with a conversation entry; returns
        ``{"snapshot": None}`` when no budget exists yet
        (pre-:meth:`configure`) or when the budget has no
        conversation source entry.

        Wire shape: same JSON-native dict the persistence
        serializer already exercises (`SourceEntry.to_dict()`).
        Same wire-shape-reuse rationale as the §7c step 6.1
        trio + 6.6.1 trio.

        Defensive contract:

          - The snapshot may be a nested dict (recursive
            children); we ``copy.deepcopy`` to isolate daemon-
            side mutation from runner-side state — same shape
            as ``session.snapshot_instruction_budget`` (§7c
            step 6.1 (2/3) at commit 1043bfde).
        """
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        getter = getattr(session, "snapshot_conversation_budget", None)
        if not callable(getter):
            return False, {
                "error": (
                    "session.snapshot_conversation_budget: session has no "
                    "snapshot_conversation_budget method (rolling-upgrade "
                    "gap?)"
                ),
                "stage": "missing_method",
            }
        try:
            raw = getter()
        except Exception as exc:  # noqa: BLE001 — read boundary
            return False, {
                "error": (
                    f"session.snapshot_conversation_budget: getter raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "read",
            }
        if raw is None:
            return True, {"snapshot": None}
        if not isinstance(raw, dict):
            return False, {
                "error": (
                    f"session.snapshot_conversation_budget: expected dict "
                    f"or None, got {type(raw).__name__}"
                ),
                "stage": "read",
            }
        # Deep-copy to isolate daemon-side mutation; the snapshot
        # may contain nested ``children`` sub-dicts.
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
        # Optional user-message multimodal attachments (wire form:
        # ``[{mime_type, data: base64-str, display_name}, ...]``).  Forwarded to
        # the session's multimodal path; absent/empty → text-only (unchanged).
        attachments = args.get("attachments") or None
        if attachments is not None and not isinstance(attachments, list):
            return False, {
                "error": "session.send_message: 'attachments' must be a list",
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

        # Phase 3 §7c step 6.6.4.2: install notification-emitting
        # callbacks on the runner-side session for the duration of
        # this send_message call.  Per the §7c step 6.6.2 audit
        # (commit 9f28f96d): the 7 daemon-side callback wirings
        # at sites 1996/2011/3391/3415/3430/3440/4291 collapse
        # into runner-side notification emissions via the §7c
        # step 6.6.4.1 NotificationFrame protocol (commit
        # 6e31d375).  Each session callback emits a frame with a
        # well-known event_type; the daemon-side wrapper's
        # ``on_notification`` handler (installed in §7c step
        # 6.6.4.3b) demuxes by event_type and invokes
        # ``server.emit(<Event>)`` or ``server._start_model_thread(...)``.
        #
        # Save the original callbacks so we can restore on exit —
        # other callers (e.g. the daemon's pre-§7c-step-6.6.4.3b
        # ``_start_model_thread``) may have wired their own.
        original_callbacks = self._install_session_notification_callbacks(
            session, request_id,
        )

        # Phase 3 §7c step 6.6.4.3b: per-call notification shims
        # for ``on_usage_update`` + ``on_gc_threshold`` kwargs.
        # Audit Finding 1: these aren't ``set_*_callback`` setters
        # — they're send_message kwargs, so they live only for
        # this one call (no install/restore needed).  Closes the
        # 7→9-callback miss caught by the §7c step 6.6.4.3
        # implementation-review audit.
        usage_shim = self._make_usage_update_notification_shim(request_id)
        gc_shim = self._make_gc_threshold_notification_shim(request_id)
        gc_phase_shim = self._make_gc_phase_notification_shim(request_id)

        # Run the message loop.  Model API calls happen
        # synchronously here; output streams via on_output;
        # usage + gc-threshold events stream via notification frames.
        # Turn-count snapshot: the post-turn forwarding below must fire iff a
        # NEW turn actually landed in turn_accounting.  This is the mechanical
        # guarantee behind the refused-turn suppression (a refused turn
        # appends nothing, so the event would re-emit the PREVIOUS turn's
        # numbers) AND what makes it safe to forward on the cancelled path.
        try:
            _turns_before = len(session.get_turn_accounting() or ())
        except Exception:  # noqa: BLE001
            _turns_before = None

        try:
            try:
                response = session.send_message(
                    prompt,
                    on_output=on_output,
                    on_usage_update=usage_shim,
                    on_gc_threshold=gc_shim,
                    on_gc_phase=gc_phase_shim,
                    attachments=attachments,
                )
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
                    # A CANCELLED turn RAN and SPENT — it is not a REFUSED
                    # turn (which never started).  Returning early here
                    # skipped the post-turn forwarding entirely, so that
                    # turn's tokens reached no TurnCompletedEvent and were
                    # invisible to anything accumulating from that event —
                    # including the cascade pool, which then believed a
                    # cascade had ~1.6x more headroom than it did.  Every
                    # child that exhausts its own budget ends exactly this
                    # way, so the leak hit precisely the children that
                    # mattered.  Forward first, then return the cancel
                    # envelope unchanged.
                    self._forward_post_turn_hooks(session, _turns_before)
                    return False, {
                        "error": str(exc) or "Cancelled",
                        "stage": "cancelled",
                    }
                # CAPTURE THE FRAMES HERE.  This path RETURNS a dict rather
                # than raising, so the envelope's ErrorPayload is built by
                # ``_extract_error_message`` from that dict -- there is no
                # exception left for the dispatcher to read a traceback off.
                # Stringifying the exception produced a message that LOOKS
                # like a complete report ("model loop raised AttributeError:
                # <text>") while being a summary, so the discarded frames
                # were invisible in its own output.
                #
                # Sanitized like every other traceback crossing this boundary
                # (§3.1): the daemon log and any forwarded event are
                # potentially cross-tenant surfaces.
                from .sanitize import sanitize_traceback
                return False, {
                    "error": (
                        f"session.send_message: model loop raised "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "traceback": sanitize_traceback(
                        traceback.format_exc(), self._workspace_root,
                    ),
                    "stage": "send",
                }

            # Path G (2026-06-07): post-turn ``AgentUIHooks`` forwarding.
            #
            # Pre-§7c-seat-flip the daemon-side
            # ``JaatoClient.send_message`` wrapper (see
            # ``shared/jaato_client.py:662-696``) fired three hooks after
            # ``session.send_message`` returned:
            # ``on_agent_turn_completed`` (→ ``TurnCompletedEvent``),
            # ``on_agent_context_updated`` (→ ``ContextUpdatedEvent``),
            # ``on_agent_history_updated`` (→ persistence).
            #
            # Post-§7c the daemon's ``_start_model_thread`` calls
            # ``runner_rpc.session_send_message_threadsafe`` (this
            # handler) directly instead of going through
            # ``JaatoClient.send_message``.  PR #82 / commit 631678e1
            # wired the runner-side ``_AgentUIHooksNotificationShim`` to
            # FORWARD these three calls — but no caller in the root IPC
            # path actually invoked the shim.  Result: root sessions
            # never emitted ``TurnCompletedEvent``; SDK clients waiting
            # for it (every provider smoke harness, every external IPC
            # consumer) hung until disconnect timeout.
            #
            # This block restores the JaatoClient-wrapper semantics for
            # the root IPC path: fetch turn accounting + context + history
            # from the runner-side session and fan out through the shim
            # (still installed at this point — restore runs in the
            # ``finally`` below).  Subagent sessions are unaffected:
            # they run their model loop inside
            # ``subagent/plugin.py`` and fire the same hooks themselves
            # at ``_run_subagent_async`` (line ~3408).  Best-effort —
            # forwarding failure must not corrupt the send_message
            # response.
            self._forward_post_turn_hooks(session, _turns_before)
        finally:
            # Phase 3 §7c step 6.6.4.2: restore any pre-existing
            # callbacks the session had so other callers (e.g. a
            # daemon-side _start_model_thread that wired its own)
            # see the same shape after this RPC completes.
            self._restore_session_notification_callbacks(
                session, original_callbacks,
            )

        # Defensive: send_message returns str by contract; coerce
        # if a custom session subclass returns something else.
        if response is None:
            response = ""
        elif not isinstance(response, str):
            response = str(response)

        # Typed budget-exhaustion signal.  A ceiling that only announces
        # itself in prose ("[Generation cancelled (...)]" plus a system line)
        # cannot be told from a normal finish by a driver -- it would have to
        # substring-match, which is the parse-the-log shape budgets exist to
        # replace.  Reported by a suspend/resume cascade whose operator needs
        # "stopped at the ceiling" to exit non-zero rather than look achieved.
        result = {"response": response}
        # getattr with a default, NOT try/except: a bare except here would
        # swallow a renamed accessor and silently emit no signal at all --
        # the failure mode is a ceiling that stops working invisibly.
        _reason_fn = getattr(session, "budget_exhausted_reason", None)
        reason = _reason_fn() if callable(_reason_fn) else None
        if reason:
            result["budget_exhausted"] = True
            result["budget_exhausted_reason"] = reason
            try:
                result["budget_usage"] = session.get_budget_usage() or {}
            except Exception:  # noqa: BLE001
                pass
        return True, result

    # ----------------------------------------------------------------------
    # §7c step 6.6.4.2: notification-emitting callback wiring for the
    # runner-side session during ``session.send_message`` handling.
    #
    # Maps each session callback to a NotificationFrame ``event_type`` +
    # payload via the §7c step 6.6.4.1 protocol (commit 6e31d375).
    # The daemon-side wrapper's ``on_notification`` handler (installed
    # in §7c step 6.6.4.3) demuxes by event_type and invokes
    # ``server.emit(<Event>)`` or ``server._start_model_thread(...)``.
    #
    # Until 6.6.4.3 lands the daemon-side leg drop, the runner-side
    # session never processes a turn (the daemon's
    # ``_jaato.send_message()`` does), so these callbacks are dormant
    # — installation is behavior-preserving.
    # ----------------------------------------------------------------------

    # Event-type constants — daemon-side demuxer matches on these.
    _NOTIF_INSTRUCTION_BUDGET_UPDATED = "instruction_budget_updated"
    _NOTIF_PROMPT_INJECTED = "prompt_injected"
    _NOTIF_CONTINUATION_NEEDED = "continuation_needed"
    _NOTIF_RETRY = "retry"
    _NOTIF_MID_TURN_INTERRUPT = "mid_turn_interrupt"
    _NOTIF_EVENTS_SUBSCRIBED = "events_subscribed"
    # §7c step 6.6.4.3b additions — the audit-caught per-call kwargs.
    _NOTIF_USAGE_UPDATE = "usage_update"
    _NOTIF_GC_THRESHOLD = "gc_threshold"
    _NOTIF_GC_PHASE = "gc_phase"

    # Path F (cycle 7): AgentUIHooks methods that the runner-side
    # session calls but `_ui_hooks` is None post-§7c — see backlog
    # `project_backlog_runner_ui_hooks_gap.md` (Finding 3).  Each
    # event_type mirrors one ui_hooks method; daemon-side demuxer
    # re-emits via ServerAgentHooks.
    _NOTIF_TOOL_CALL_START = "tool_call_start"
    _NOTIF_TOOL_CALL_END = "tool_call_end"
    _NOTIF_TOOL_OUTPUT = "tool_output"
    _NOTIF_TURN_PROGRESS = "turn_progress"

    # Path F regression fix (2026-05-12): runner-side
    # ``lifecycle_tools._execute_signal_completion`` calls
    # ``hooks.on_agent_completed`` and ``JaatoSession`` calls
    # ``hooks.on_session_quiescent`` on the runner-side shim — both
    # were no-op ``pass`` pre-fix, so neither event reached the
    # daemon-side reactor engine.  The Path F audit at
    # ``_AgentUIHooksNotificationShim`` (below) incorrectly assumed
    # these methods were "covered daemon-side" — true pre-§7c when
    # ``lifecycle_tools.py`` lived in the daemon process, untrue
    # post-§7c when the module moved into the runner subprocess.
    _NOTIF_AGENT_COMPLETED = "agent_completed"
    _NOTIF_SESSION_QUIESCENT = "session_quiescent"

    # Path F sweep (2026-05-12 follow-up to the
    # ``agent_completed`` / ``session_quiescent`` fix): close the
    # remaining 6 runner-side ``ServerAgentHooks`` methods that the
    # original Path F audit incorrectly classified as
    # "covered daemon-side".  All 6 are called from runner-tier
    # code paths (the subagent plugin lives in runner-tier per
    # ``PLUGIN_TIER = "runner"`` and JaatoClient lives runner-side
    # post-§7c) so their ``self._ui_hooks.on_X`` calls were dropping
    # on the shim's no-op stubs pre-fix.
    _NOTIF_AGENT_CREATED = "agent_created"
    _NOTIF_AGENT_STATUS_CHANGED = "agent_status_changed"
    _NOTIF_AGENT_TURN_COMPLETED = "agent_turn_completed"
    _NOTIF_AGENT_CONTEXT_UPDATED = "agent_context_updated"
    _NOTIF_AGENT_GC_CONFIG = "agent_gc_config"
    _NOTIF_AGENT_HISTORY_UPDATED = "agent_history_updated"

    # Phase 4 §4.4 (Finding 2 closure): session-plugin description-
    # callback bridges runner → daemon as a notification frame.
    # Runner-side session plugin (now runner-tier per §4.4 sub-action
    # A) fires ``_on_description_changed(session_id, description)``
    # when the model invokes ``session_describe``; the shim emits
    # this event_type; daemon-side demuxer re-emits as
    # ``SessionDescriptionUpdatedEvent`` for the TUI's session-picker
    # to refresh.
    _NOTIF_DESCRIPTION_UPDATED = "description_updated"

    def _forward_post_turn_hooks(self, session, turns_before) -> None:
        """Fire the post-turn ``AgentUIHooks`` fan-out for a turn that RAN.

        Path G (2026-06-07) restored the JaatoClient-wrapper semantics for
        the root IPC path: ``on_agent_turn_completed`` ->
        ``TurnCompletedEvent``, plus context + history updates.

        Called on BOTH the normal and the CANCELLED return paths.  A
        cancelled turn ran and spent tokens; skipping it (as the early
        ``return`` used to) left that spend in no TurnCompletedEvent at all,
        invisible to every consumer accumulating from that event — the
        cascade pool most consequentially, since a child that exhausts its
        own budget ends by cancellation, so the leak hit exactly the
        children whose spend mattered most.

        Gated on a NEW turn having landed in ``turn_accounting``
        (``turns_before`` snapshot).  That is the mechanical guarantee: the
        payload is sourced from ``turn_accounting[-1]``, so firing when
        nothing was appended re-emits the PREVIOUS turn's tokens and
        duration — which is what a REFUSED turn would do, and why refused
        turns must stay suppressed.  The count check subsumes the
        refused-flag check and covers every other no-op path too.

        Best-effort throughout: forwarding failure must not corrupt the
        send_message response.
        """
        ui_hooks = getattr(session, "_ui_hooks", None)
        if ui_hooks is None:
            return
        try:
            turn_accounting = session.get_turn_accounting() or []
        except Exception:  # noqa: BLE001
            return
        # No new turn => nothing completed => do not re-emit the last one.
        if turns_before is not None and len(turn_accounting) <= turns_before:
            return
        if not turn_accounting:
            return
        try:
            agent_id = getattr(session, "_agent_id", None) or "main"
            last_turn = turn_accounting[-1]
            ui_hooks.on_agent_turn_completed(
                agent_id=agent_id,
                turn_number=max(0, len(turn_accounting) - 1),
                prompt_tokens=last_turn.get("prompt", 0),
                output_tokens=last_turn.get("output", 0),
                total_tokens=last_turn.get("total", 0),
                duration_seconds=last_turn.get("duration_seconds", 0),
                function_calls=last_turn.get("function_calls", []),
                cache_read_tokens=last_turn.get("cache_read"),
                cache_creation_tokens=last_turn.get("cache_creation"),
                spend_total_tokens=last_turn.get("spend_total"),
                spend_cache_read_tokens=last_turn.get("spend_cache_read"),
                spend_cache_creation_tokens=last_turn.get(
                    "spend_cache_creation"),
                cost_usd=last_turn.get("cost_usd"),
                finish_reason=last_turn.get("finish_reason", "stop"),
            )
            usage = session.get_context_usage()
            ui_hooks.on_agent_context_updated(
                agent_id=agent_id,
                total_tokens=usage.get("total_tokens", 0),
                prompt_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                turns=usage.get("turns", 0),
                percent_used=usage.get("percent_used", 0),
            )
            ui_hooks.on_agent_history_updated(
                agent_id=agent_id, history=session.get_history(),
            )
            # LAST, and last for a reason: SessionTerminatedEvent is terminal
            # by contract, so nothing may follow it.  The session detects
            # quiescence inside ``send_message`` and records it; flushing it
            # here puts the terminal event after this turn's own events
            # instead of before them.
            try:
                session.flush_session_quiescent()
            except Exception:  # noqa: BLE001 — never break the post-turn path
                logger.warning(
                    "flush_session_quiescent raised", exc_info=True)
        except Exception:  # noqa: BLE001 — best-effort forwarding
            logger.exception(
                "post-turn AgentUIHooks forwarding raised — "
                "send_message response still returned",
            )

    def _make_usage_update_notification_shim(
        self, request_id: int,
    ) -> Any:
        """Build a per-call ``on_usage_update`` shim that emits a
        ``usage_update`` NotificationFrame.

        Phase 3 §7c step 6.6.4.3b + Path E (cycle 6).  The shim
        bundles the serialized ``TokenUsage`` dict TOGETHER with
        ``context_limit`` + ``turns`` count read locally from the
        runner-side session.  Pre-Path-E the daemon-side handler
        called BACK into the runner via 2 blocking RPCs
        (``session_get_context_limit`` + ``session_get_turn_accounting``)
        to compute these values — that path raced against the
        runner's still-active ``send_message`` call and timed out.
        Batching the values into the notification payload eliminates
        the in-band RPCs; daemon-side handler reads from payload.

        Defensive: a ``TokenUsage`` instance is a dataclass with
        well-known field names (prompt_tokens, output_tokens,
        total_tokens, plus optional cache/reasoning/thinking/
        cost_usd).  ``getattr`` keeps the shim resilient to
        provider-side subclasses adding fields.

        Context-limit lookup is also defensive: failures fall back
        to ``0`` so the daemon-side handler can detect the missing-
        value case (and either consult its cached value or omit the
        ``context_limit`` field of ``ContextUpdatedEvent``).
        """
        rpc = self

        def _shim(usage: Any) -> None:
            try:
                # Path E (cycle 6): compute context_limit + turns
                # locally from the runner-side session so the daemon
                # handler doesn't need to call back into the runner.
                context_limit = 0
                turns = 0
                try:
                    host = rpc._session_host
                    sess = getattr(host, "session", None) if host else None
                    if sess is not None:
                        getter = getattr(sess, "get_context_limit", None)
                        if callable(getter):
                            context_limit = int(getter() or 0)
                        accounting = getattr(sess, "_turn_accounting", None)
                        if accounting is None:
                            accounting_getter = getattr(
                                sess, "get_turn_accounting", None,
                            )
                            if callable(accounting_getter):
                                accounting = accounting_getter()
                        if accounting is not None:
                            turns = len(accounting)
                except Exception:  # noqa: BLE001 — never fail the
                    # notification shim because of an accessor crash
                    logger.exception(
                        "usage_update shim: context_limit/turns lookup "
                        "crashed; emitting payload with defaults"
                    )

                payload = {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                    "cache_read_tokens": getattr(usage, "cache_read_tokens", None),
                    "cache_creation_tokens": getattr(
                        usage, "cache_creation_tokens", None,
                    ),
                    "reasoning_tokens": getattr(usage, "reasoning_tokens", None),
                    "thinking_tokens": getattr(usage, "thinking_tokens", None),
                    "cost_usd": getattr(usage, "cost_usd", None),
                    # Path E batched values:
                    "context_limit": context_limit,
                    "turns": turns,
                }
                rpc.emit_notification(
                    request_id=request_id,
                    event_type=rpc._NOTIF_USAGE_UPDATE,
                    payload=payload,
                )
            except Exception:  # noqa: BLE001
                logger.exception("usage_update notify raised")

        return _shim

    def _make_gc_phase_notification_shim(
        self, request_id: int,
    ) -> Any:
        """Build a per-call ``on_gc_phase`` shim emitting ``gc_phase`` frames.

        Signature: ``(phase: str, payload: dict) -> None``.  The daemon-side
        handler re-emits a typed ``GCEvent``.  Sibling of the ``gc_threshold``
        shim below, which carries only the threshold crossing and whose
        handler renders PROSE -- this one carries the whole lifecycle
        (about_to_run / started / completed) as branchable values.

        The payload is forwarded verbatim: ``gc_support.run_gc`` owns its
        shape, so a new field there reaches clients without a change here.
        """
        rpc = self

        def _shim(phase: str, payload: Dict[str, Any]) -> None:
            try:
                rpc.emit_notification(
                    request_id=request_id,
                    event_type=rpc._NOTIF_GC_PHASE,
                    payload={"phase": str(phase), **(payload or {})},
                )
            except Exception:  # noqa: BLE001
                logger.exception("gc_phase notify raised")

        return _shim

    def _make_gc_threshold_notification_shim(
        self, request_id: int,
    ) -> Any:
        """Build a per-call ``on_gc_threshold`` shim that emits a
        ``gc_threshold`` NotificationFrame.

        Phase 3 §7c step 6.6.4.3b.  Signature: ``(percent_used:
        float, threshold: float) -> None``.  Daemon-side handler
        re-emits ``SystemMessageEvent`` with the audit-text the
        pre-migration daemon callback used.
        """
        rpc = self

        def _shim(percent_used: float, threshold: float) -> None:
            try:
                rpc.emit_notification(
                    request_id=request_id,
                    event_type=rpc._NOTIF_GC_THRESHOLD,
                    payload={
                        "percent_used": float(percent_used),
                        "threshold": float(threshold),
                    },
                )
            except Exception:  # noqa: BLE001
                logger.exception("gc_threshold notify raised")

        return _shim

    def _install_session_notification_callbacks(
        self, session: Any, request_id: int,
    ) -> Dict[str, Any]:
        """Install notification-emitting callbacks on the session.

        Phase 3 §7c step 6.6.4.2.  Each session callback emits a
        NotificationFrame with a well-known event_type; daemon-
        side handler demuxes.

        Returns a dict of original callbacks (for restoration in
        ``_restore_session_notification_callbacks``).  Best-effort:
        sessions without a particular setter (rolling-upgrade
        scenario, or test stubs) are silently skipped — the
        corresponding event type just won't fire.

        Args:
            session: The runner-side JaatoSession.
            request_id: The in-flight call's id; threaded into
                each NotificationFrame.
        """
        rpc = self
        originals: Dict[str, Any] = {}

        # instruction_budget_callback(snapshot: dict) -> None
        if hasattr(session, "set_instruction_budget_callback"):
            originals["instruction_budget"] = getattr(
                session, "_on_instruction_budget_updated", None,
            )

            def _ib_cb(snapshot: dict) -> None:
                try:
                    rpc.emit_notification(
                        request_id=request_id,
                        event_type=rpc._NOTIF_INSTRUCTION_BUDGET_UPDATED,
                        payload={"snapshot": dict(snapshot or {})},
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("instruction_budget notify raised")

            try:
                session.set_instruction_budget_callback(_ib_cb)
            except Exception:  # noqa: BLE001
                logger.debug("set_instruction_budget_callback raised")

        # prompt_injected_callback(text: str) -> None
        if hasattr(session, "set_prompt_injected_callback"):
            originals["prompt_injected"] = getattr(
                session, "_on_prompt_injected", None,
            )

            def _pi_cb(text: str) -> None:
                try:
                    rpc.emit_notification(
                        request_id=request_id,
                        event_type=rpc._NOTIF_PROMPT_INJECTED,
                        payload={"text": str(text or "")},
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("prompt_injected notify raised")

            try:
                session.set_prompt_injected_callback(_pi_cb)
            except Exception:  # noqa: BLE001
                logger.debug("set_prompt_injected_callback raised")

        # continuation_callback(child_messages: str) -> None
        if hasattr(session, "set_continuation_callback"):
            originals["continuation"] = getattr(
                session, "_on_continuation_needed", None,
            )

            def _cont_cb(child_messages: str) -> None:
                try:
                    rpc.emit_notification(
                        request_id=request_id,
                        event_type=rpc._NOTIF_CONTINUATION_NEEDED,
                        payload={"child_messages": str(child_messages or "")},
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("continuation notify raised")

            try:
                session.set_continuation_callback(_cont_cb)
            except Exception:  # noqa: BLE001
                logger.debug("set_continuation_callback raised")

        # retry_callback(message: str, attempt: int, max_attempts: int, delay: float)
        if hasattr(session, "set_retry_callback"):
            originals["retry"] = getattr(session, "_on_retry", None)

            def _retry_cb(
                message: str, attempt: int, max_attempts: int, delay: float,
            ) -> None:
                try:
                    rpc.emit_notification(
                        request_id=request_id,
                        event_type=rpc._NOTIF_RETRY,
                        payload={
                            "message": str(message or ""),
                            "attempt": int(attempt),
                            "max_attempts": int(max_attempts),
                            "delay": float(delay),
                        },
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("retry notify raised")

            try:
                session.set_retry_callback(_retry_cb)
            except Exception:  # noqa: BLE001
                logger.debug("set_retry_callback raised")

        # mid_turn_interrupt_callback(partial_chars: int, prompt_preview: str)
        if hasattr(session, "set_mid_turn_interrupt_callback"):
            originals["mid_turn_interrupt"] = getattr(
                session, "_on_mid_turn_interrupt", None,
            )

            def _mti_cb(partial_chars: int, prompt_preview: str) -> None:
                try:
                    rpc.emit_notification(
                        request_id=request_id,
                        event_type=rpc._NOTIF_MID_TURN_INTERRUPT,
                        payload={
                            "partial_chars": int(partial_chars),
                            "prompt_preview": str(prompt_preview or ""),
                        },
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("mid_turn_interrupt notify raised")

            try:
                session.set_mid_turn_interrupt_callback(_mti_cb)
            except Exception:  # noqa: BLE001
                logger.debug("set_mid_turn_interrupt_callback raised")

        # _event_bus_tools._on_subscribed(agent_id: str, event_names: list)
        # Direct private-attr write — JaatoSession exposes
        # _event_bus_tools as a private attr without a public setter
        # for the on_subscribed slot.  Mirrors the daemon-side pattern
        # at core.py:1996 (pre-§7c-step-6.6.4.3).
        ebt = getattr(session, "_event_bus_tools", None)
        if ebt is not None and hasattr(ebt, "_on_subscribed"):
            originals["events_subscribed"] = ebt._on_subscribed

            def _ebt_cb(agent_id: str, event_names: list) -> None:
                try:
                    rpc.emit_notification(
                        request_id=request_id,
                        event_type=rpc._NOTIF_EVENTS_SUBSCRIBED,
                        payload={
                            "agent_id": str(agent_id or ""),
                            "event_names": list(event_names or []),
                        },
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("events_subscribed notify raised")

            ebt._on_subscribed = _ebt_cb

        # Path F (cycle 7): install an AgentUIHooks shim that emits
        # notification frames for the 4 ui_hooks methods the runner-
        # side session calls (on_tool_call_start, on_tool_call_end,
        # on_tool_output, on_turn_progress).  See backlog
        # project_backlog_runner_ui_hooks_gap.md.  Pre-Path-F the
        # runner-side ``session._ui_hooks`` was None and all calls
        # silently dropped — TUI never saw tool events or turn
        # progress.  Direct attribute assignment (vs ``set_ui_hooks``)
        # because ``set_ui_hooks`` also overwrites ``_agent_id``;
        # the shim only needs the hooks slot.  Stored under key
        # ``ui_hooks`` so restore-on-exit can put the original
        # (typically None) back.
        if hasattr(session, "_ui_hooks"):
            originals["ui_hooks"] = session._ui_hooks
            try:
                session._ui_hooks = _AgentUIHooksNotificationShim(
                    rpc, request_id,
                )
            except Exception:  # noqa: BLE001
                logger.debug("ui_hooks shim install raised")

        # Phase 4 §4.4 (Finding 2 closure): install a description-
        # callback shim on the runner-side session plugin so model-
        # invoked ``session_describe`` calls bridge to the daemon as
        # a ``description_updated`` NotificationFrame.  Pre-§4.4 the
        # session plugin was daemon-tier and unreachable from the
        # runner-side model loop; sub-action A flipped it to runner-
        # tier so it's now loaded in the runner registry.
        try:
            runtime = getattr(session, "_runtime", None)
            registry = getattr(runtime, "registry", None) if runtime else None
            session_plugin = (
                registry.get_plugin("session") if registry else None
            )
            if (
                session_plugin is not None
                and hasattr(session_plugin, "set_description_callback")
            ):
                originals["description_callback"] = getattr(
                    session_plugin, "_on_description_changed", None,
                )

                def _desc_cb(session_id: str, description: str) -> None:
                    try:
                        rpc.emit_notification(
                            request_id=request_id,
                            event_type=rpc._NOTIF_DESCRIPTION_UPDATED,
                            payload={
                                "session_id": str(session_id or ""),
                                "description": str(description or ""),
                            },
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception("description_updated notify raised")

                session_plugin.set_description_callback(_desc_cb)
        except Exception:  # noqa: BLE001
            logger.debug("description_callback shim install raised")

        return originals

    def _restore_session_notification_callbacks(
        self, session: Any, originals: Dict[str, Any],
    ) -> None:
        """Restore the session's pre-installation callbacks.

        Phase 3 §7c step 6.6.4.2.  Counterpart to
        :meth:`_install_session_notification_callbacks`.  Each
        restoration is best-effort — if a setter raises, log
        and continue (don't mask the original send_message
        result with a teardown error).
        """
        if "instruction_budget" in originals and hasattr(
            session, "set_instruction_budget_callback",
        ):
            try:
                session.set_instruction_budget_callback(
                    originals["instruction_budget"],
                )
            except Exception:  # noqa: BLE001
                logger.debug("restore instruction_budget callback raised")
        if "prompt_injected" in originals and hasattr(
            session, "set_prompt_injected_callback",
        ):
            try:
                session.set_prompt_injected_callback(
                    originals["prompt_injected"],
                )
            except Exception:  # noqa: BLE001
                logger.debug("restore prompt_injected callback raised")
        if "continuation" in originals and hasattr(
            session, "set_continuation_callback",
        ):
            try:
                session.set_continuation_callback(originals["continuation"])
            except Exception:  # noqa: BLE001
                logger.debug("restore continuation callback raised")
        if "retry" in originals and hasattr(session, "set_retry_callback"):
            try:
                session.set_retry_callback(originals["retry"])
            except Exception:  # noqa: BLE001
                logger.debug("restore retry callback raised")
        if "mid_turn_interrupt" in originals and hasattr(
            session, "set_mid_turn_interrupt_callback",
        ):
            try:
                session.set_mid_turn_interrupt_callback(
                    originals["mid_turn_interrupt"],
                )
            except Exception:  # noqa: BLE001
                logger.debug("restore mid_turn_interrupt callback raised")
        if "events_subscribed" in originals:
            ebt = getattr(session, "_event_bus_tools", None)
            if ebt is not None and hasattr(ebt, "_on_subscribed"):
                try:
                    ebt._on_subscribed = originals["events_subscribed"]
                except Exception:  # noqa: BLE001
                    logger.debug("restore events_subscribed slot raised")
        # Path F (cycle 7): restore the pre-shim ui_hooks (typically
        # None for the runner-side session — see backlog
        # project_backlog_runner_ui_hooks_gap.md).
        if "ui_hooks" in originals and hasattr(session, "_ui_hooks"):
            try:
                session._ui_hooks = originals["ui_hooks"]
            except Exception:  # noqa: BLE001
                logger.debug("restore ui_hooks raised")
        # Phase 4 §4.4: restore the pre-shim description callback
        # on the runner-side session plugin (typically None — no
        # callback wired pre-§4.4 since the daemon-side wiring at
        # core.py:2487 was on a different instance).
        if "description_callback" in originals:
            try:
                runtime = getattr(session, "_runtime", None)
                registry = getattr(runtime, "registry", None) if runtime else None
                session_plugin = (
                    registry.get_plugin("session") if registry else None
                )
                if (
                    session_plugin is not None
                    and hasattr(session_plugin, "set_description_callback")
                ):
                    session_plugin.set_description_callback(
                        originals["description_callback"],
                    )
            except Exception:  # noqa: BLE001
                logger.debug("restore description_callback raised")

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

    def _handle_session_try_completion_nudge(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Atomic check-and-increment for the completion-nudge guard.

        Phase 3 §7c step 6.6.4.3a.  Collapses 3 daemon-side reaches
        (``_signal_completion_called`` read,
        ``_completion_nudges_fired`` read, increment) into one
        round-trip — required for §7c step 6.6.4.3b's seat-flip
        where the JaatoSession lives in this runner process.

        Args: ``{"max_nudges": int}`` — caller's nudge-budget knob
        (the existing daemon-side site uses
        ``MAX_COMPLETION_NUDGES = 2``).

        Returns:
            ``(True, {"should_nudge": bool, "nudges_fired": int})``.
            ``nudges_fired`` is the post-increment value when
            ``should_nudge`` is True; the unchanged current count
            otherwise.
            ``(False, {"error": ..., "stage": "decode"})`` when
            ``max_nudges`` is missing or not an int.
            ``(False, {"error": ..., "stage": "no_host" | "no_session"})``
            when the session host isn't bootstrapped.
            ``(False, {"error": ..., "stage": "missing_method"})``
            when the bootstrapped session lacks the public method
            (rolling-upgrade scenario where the runner is newer
            than the daemon's session class).
        """
        max_nudges = args.get("max_nudges")
        if not isinstance(max_nudges, int) or isinstance(max_nudges, bool):
            return False, {
                "error": (
                    "session.try_completion_nudge: missing or non-int "
                    "'max_nudges' arg"
                ),
                "stage": "decode",
            }
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        try_method = getattr(session, "try_completion_nudge", None)
        if not callable(try_method):
            return False, {
                "error": (
                    "session.try_completion_nudge: session class lacks "
                    "public try_completion_nudge() method"
                ),
                "stage": "missing_method",
            }
        try:
            should_nudge, nudges_fired = try_method(max_nudges)
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.try_completion_nudge: try_completion_nudge "
                    f"raised {type(exc).__name__}: {exc}"
                ),
                "stage": "call",
            }
        return True, {
            "should_nudge": bool(should_nudge),
            "nudges_fired": int(nudges_fired),
        }

    def _handle_session_try_drain_pending_user(self) -> "tuple[bool, Any]":
        """Atomically pop a pending high-priority message for the daemon's
        post-turn drain (multi-turn deadlock fix).

        Delegates to ``JaatoSession.try_drain_pending_user`` on the runner-
        side session.  Returns ``{"text": str | None}`` — the message text to
        run as a fresh turn, or ``None`` when nothing is queued.
        """
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        try_method = getattr(session, "try_drain_pending_user", None)
        if not callable(try_method):
            return False, {
                "error": (
                    "session.try_drain_pending_user: session class lacks "
                    "public try_drain_pending_user() method"
                ),
                "stage": "missing_method",
            }
        try:
            text = try_method()
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.try_drain_pending_user: try_drain_pending_user "
                    f"raised {type(exc).__name__}: {exc}"
                ),
                "stage": "call",
            }
        return True, {"text": text}

    def _handle_session_get_auth_info(self) -> "tuple[bool, Any]":
        """Read the credential-source description string from the
        runner-side session's provider.

        Phase 3 §7c step 6.6.4.5c.1.  Replaces 2 daemon-side reaches
        into ``self._jaato.auth_info`` (core.py:2073, 4481) — the
        property reads ``_session._provider.get_auth_info()`` daemon-
        side, which post-seat-flip is the wrong (dead) session.

        Returns:
            ``(True, {"auth_info": str})`` on success.  Empty string
            when no provider is attached or the provider doesn't
            implement ``get_auth_info`` — same defensive shape the
            old daemon-side property had.

            ``(False, {"error": ..., "stage": ...})`` on
            ``no_host`` / ``no_session`` / ``missing_method``.

        ``missing_method`` surfaces when the session class lacks the
        ``get_auth_info`` public method — covers the rolling-upgrade
        scenario where the runner is newer than the daemon's
        JaatoSession class.
        """
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        get_method = getattr(session, "get_auth_info", None)
        if not callable(get_method):
            return False, {
                "error": (
                    "session.get_auth_info: session class lacks public "
                    "get_auth_info() method"
                ),
                "stage": "missing_method",
            }
        try:
            auth_info = get_method()
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.get_auth_info: get_auth_info raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "call",
            }
        return True, {"auth_info": str(auth_info or "")}

    def _handle_session_get_user_commands(self) -> "tuple[bool, Any]":
        """Read the runner-side session's user-command catalog.

        Phase 3 §7c step 6.6.4.5c.2.  Replaces 2 daemon-side reaches
        into ``self._jaato.get_user_commands()``.  Wire shape per
        the 5c.2 audit decision: **dict-shape-only** (Path B).
        ``UserCommand`` and ``CommandParameter`` are NamedTuples
        with primitive fields (str/bool) — no callables, no class
        refs, no Type[X] — so straight dict serialization is
        sufficient.

        Daemon callers reconstruct ``UserCommand`` instances on
        receipt; the handler callable itself stays runner-side and
        gets invoked via ``session.execute_user_command`` (5c.3).

        Returns:
            ``(True, {"commands": {<name>: <UserCommand-as-dict>, ...}})``
            on success.  Per-command dict shape::

                {
                    "name": str,
                    "description": str,
                    "share_with_model": bool,
                    "parameters": [
                        {"name": str, "description": str,
                         "required": bool, "capture_rest": bool},
                        ...
                    ] | null
                }

            ``(False, {"error": ..., "stage": ...})`` on
            ``no_host`` / ``no_session`` / ``missing_method`` /
            ``call``.
        """
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        get_method = getattr(session, "get_user_commands", None)
        if not callable(get_method):
            return False, {
                "error": (
                    "session.get_user_commands: session class lacks public "
                    "get_user_commands() method"
                ),
                "stage": "missing_method",
            }
        try:
            commands = get_method()
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.get_user_commands: get_user_commands raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "call",
            }
        # Serialize each UserCommand to dict.  Per the audit, fields
        # are all primitives + Optional[List[CommandParameter]] which
        # is itself a primitive-only NamedTuple.
        serialized: Dict[str, Any] = {}
        for name, cmd in (commands or {}).items():
            params_serialized = None
            cmd_params = getattr(cmd, "parameters", None)
            if cmd_params:
                params_serialized = [
                    {
                        "name": str(getattr(p, "name", "")),
                        "description": str(getattr(p, "description", "")),
                        "required": bool(getattr(p, "required", False)),
                        "capture_rest": bool(getattr(p, "capture_rest", False)),
                    }
                    for p in cmd_params
                ]
            serialized[str(name)] = {
                "name": str(getattr(cmd, "name", name)),
                "description": str(getattr(cmd, "description", "")),
                "share_with_model": bool(getattr(cmd, "share_with_model", False)),
                "parameters": params_serialized,
            }
        return True, {"commands": serialized}

    def _handle_session_execute_user_command(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Invoke a user command on the runner-side session.

        Phase 3 §7c step 6.6.4.5c.3.  Replaces the daemon-side reach
        into ``self._jaato.execute_user_command(name, args)``
        (core.py:4044).  Wire shape per the 5c.3 audit decision:
        **Path A** (per-type reconstruction) bounded to 3 result
        shapes — pre-implementation grep verified the daemon does
        structured access on ``HelpLines.lines`` and dict keys for
        the "model" command + IPC-return shape.

        Args: ``{"name": str, "args": dict}``.

        Returns:
            ``(True, {"result": <tagged-dict>, "shared": bool})``
            on success.  ``shared`` is the second half of the
            JaatoSession.execute_user_command return tuple
            (``share_with_model`` flag).  ``result`` is one of:

            - ``{"_kind": "HelpLines", "lines": [[text, style], ...]}``
              when result is a :class:`HelpLines`.
            - ``{"_kind": "dict", "value": <json-dict>}`` when the
              result is a dict (covers the "model" command's
              ``{"success": ..., "current_model": ...}`` shape).
            - ``{"_kind": "str", "value": <str>}`` otherwise
              (everything-else coerced — display-only).

            ``(False, {"error": ..., "stage": ...})`` on
            ``decode`` (missing/malformed args) / ``no_host`` /
            ``no_session`` / ``missing_method`` / ``call``.
        """
        name = args.get("name")
        if not isinstance(name, str) or not name:
            return False, {
                "error": (
                    "session.execute_user_command: missing or non-str "
                    "'name' arg"
                ),
                "stage": "decode",
            }
        cmd_args = args.get("args", {})
        if cmd_args is None:
            cmd_args = {}
        if not isinstance(cmd_args, dict):
            return False, {
                "error": (
                    f"session.execute_user_command: 'args' must be a "
                    f"dict (got {type(cmd_args).__name__})"
                ),
                "stage": "decode",
            }
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        exec_method = getattr(session, "execute_user_command", None)
        if not callable(exec_method):
            return False, {
                "error": (
                    "session.execute_user_command: session class lacks "
                    "public execute_user_command() method"
                ),
                "stage": "missing_method",
            }
        try:
            result, shared = exec_method(name, cmd_args)
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.execute_user_command: execute_user_command "
                    f"raised {type(exc).__name__}: {exc}"
                ),
                "stage": "call",
            }

        # Per-type serialization (Path A bounded to 3 shapes).
        from jaato_sdk.plugins.base import HelpLines
        if isinstance(result, HelpLines):
            # ``lines`` is List[tuple]; serialize as list-of-lists for
            # JSON safety (tuples become lists on the wire anyway).
            tagged = {
                "_kind": "HelpLines",
                "lines": [list(t) for t in (result.lines or [])],
            }
        elif isinstance(result, dict):
            tagged = {"_kind": "dict", "value": result}
        else:
            # str-or-other-coerced.  Matches the daemon-side
            # pre-§7c-step-6.6.4.5c.3 IPC-return fallback at
            # core.py:4099 (``{"result": str(result)}``).
            tagged = {"_kind": "str", "value": str(result) if result is not None else ""}

        return True, {"result": tagged, "shared": bool(shared)}

    def _handle_session_get_model_completions(
        self, args: Dict[str, Any],
    ) -> "tuple[bool, Any]":
        """Get completion candidates for the "model" command's
        subcommand arguments.

        Phase 3 §7c step 6.6.4.5c.4.  Replaces 2 daemon-side
        reaches: ``core.py:4285`` (model-name list for the toolbar)
        and ``command_router.py:1149`` (model-subcommand expansion
        for the IPC completion catalog).  Wire shape per the
        5c.4 audit decision: dict-shape-only (Path A) — mirror
        of 5c.2's UserCommand serialization since CommandCompletion
        is also a NamedTuple with primitive fields only
        (``value: str``, ``description: str``).

        Args: ``{"args": List[str]}`` — the arguments typed so
        far.  Empty list returns subcommands (``list`` / ``select``
        / ``help``); ``["select"]`` returns model names; etc.

        Returns:
            ``(True, {"completions": [{"value": str, "description": str}, ...]})``
            on success.  Empty list when no completions match.

            ``(False, {"error": ..., "stage": ...})`` on
            ``decode`` (non-list args) / ``no_host`` /
            ``no_session`` / ``missing_method`` / ``call``.
        """
        raw_args = args.get("args", [])
        if raw_args is None:
            raw_args = []
        if not isinstance(raw_args, list):
            return False, {
                "error": (
                    f"session.get_model_completions: 'args' must be a "
                    f"list (got {type(raw_args).__name__})"
                ),
                "stage": "decode",
            }
        # Coerce each entry to str — wire safety for callers that
        # might send non-str types in the list.
        str_args = [str(a) for a in raw_args]
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        get_method = getattr(session, "get_model_completions", None)
        if not callable(get_method):
            return False, {
                "error": (
                    "session.get_model_completions: session class lacks "
                    "public get_model_completions() method"
                ),
                "stage": "missing_method",
            }
        try:
            completions = get_method(str_args)
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.get_model_completions: get_model_completions "
                    f"raised {type(exc).__name__}: {exc}"
                ),
                "stage": "call",
            }
        # Serialize each CommandCompletion (NamedTuple with
        # primitive fields, mirror of UserCommand serialization
        # in 5c.2).
        serialized = [
            {
                "value": str(getattr(c, "value", "")),
                "description": str(getattr(c, "description", "") or ""),
            }
            for c in (completions or [])
        ]
        return True, {"completions": serialized}

    def _handle_session_register_client_tools(self, args) -> "tuple[bool, Any]":
        """Mid-session glue of client-provided ("host") tool SCHEMAS onto the
        LIVE runner registry, so the runner-tier model sees a tool the client
        registered AFTER session.new — without a session restart.

        Mirrors the bootstrap-time ``_register_client_tools_on_runner`` (which
        only ran from ``envelope.client_tools`` at spawn); the model's next
        ``get_tool_schemas`` (live-read) surfaces the new tool.  Execution is
        unchanged — the runner-side forwarding executor proxies daemon-side via
        the ``__client_tools__`` sentinel → the existing ToolExecuteRequestEvent
        → the client runs the handler.

        Args (over the wire): ``{"client_tools": [schema_dict, ...]}``.
        Returns ``{"registered": [names]}``.
        """
        client_tools = args.get("client_tools")
        if not isinstance(client_tools, list):
            return False, {
                "error": "session.register_client_tools: 'client_tools' must "
                         "be a list",
                "stage": "decode",
            }
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        runtime = getattr(session, "_runtime", None)
        registry = getattr(runtime, "registry", None) if runtime else None
        if registry is None:
            return False, {
                "error": "session.register_client_tools: no runner registry",
                "stage": "no_registry",
            }
        from server.runner.session import _register_client_tools_on_runner
        _register_client_tools_on_runner(registry, client_tools)
        # ``_register_client_tools_on_runner`` records each tool as
        # ``auto_approved=True`` in ``registry._core_auto_approved`` — but
        # ``check_permission`` gates on the permission POLICY whitelist, not
        # that registry set.  The bridge (``add_whitelist_tools``) runs ONCE at
        # bootstrap (``jaato_runtime`` after ``envelope.client_tools``
        # registration), so tools registered HERE — mid-session, after that
        # one-time sync — are auto-approved in the registry yet absent from the
        # policy whitelist.  A cold-revived session driven headlessly (e.g. a
        # ``session.wake``) always registers its client tools mid-session (the
        # client attaches after revive), so the tool would prompt for operator
        # permission and block forever (no operator on a headless turn).  Sync
        # the newly-auto-approved names into the runner permission whitelist
        # now — mirrors the bootstrap sync and the daemon-side handler for
        # ``PermissionAddWhitelistRequest``.
        permission_plugin = getattr(
            getattr(session, "_runtime", None), "permission_plugin", None)
        if permission_plugin is not None:
            names = [ct.get("name") for ct in client_tools if ct.get("name")]
            if names:
                permission_plugin.add_whitelist_tools(names)
        # The registry registration above wires the forwarding EXECUTOR, but the
        # model's per-turn tool list is the cached ``session._tools`` (built at
        # configure()).  Append the new schemas so the model both SEES and can
        # CALL them — mirrors the refresh path at jaato_session.py:~1971.  The
        # next provider call / get_tool_schemas surfaces them.
        from jaato_sdk.plugins.model_provider.types import ToolSchema
        if getattr(session, "_tools", None) is not None:
            existing = {s.name for s in session._tools}
            for ct in client_tools:
                nm = ct.get("name")
                if nm and nm not in existing:
                    session._tools.append(ToolSchema(
                        name=nm,
                        description=ct.get("description", ""),
                        parameters=ct.get("parameters", {}),
                        category=ct.get("category") or None,
                        # Default EAGER (see _register_client_tools_on_runner);
                        # honor an explicit discoverability from the client.
                        discoverability=ct.get("discoverability", DISCOVERABILITY_EAGER),
                    ))
        return True, {
            "registered": [
                ct.get("name") for ct in client_tools if ct.get("name")
            ],
        }

    def _handle_session_get_tool_schemas(self) -> "tuple[bool, Any]":
        """Read the runner-side session's resolved tool schemas.

        Phase 3 §7c step 6.6.4.5c.5.  Replaces 2 daemon-side
        reaches into ``self._jaato.get_tool_schemas()``:
        ``core.py:1407`` (tool-ID registry build) and
        ``core.py:3759`` (``signal_completion_in_surface`` filter
        at the completion-nudge guard).

        Returns the session-resolved subset (preloaded plugins +
        on-demand activations) — NOT the registry's full set.
        This is why a daemon-side ``_runtime`` cache wouldn't work
        as initially planned in §7c step 6.6.4.5's Refinement 2;
        ``JaatoRuntime.get_tool_schemas()`` returns the full set
        and would over-include tools the session has filtered out.

        Wire shape per the 5c.5 audit decision: dict-shape-only
        (Path A).  All 7 ToolSchema fields + nested EditableContent
        fields are JSON-encodable; ``traits: FrozenSet[str]``
        converts to ``traits: List[str]`` on the wire and back to
        FrozenSet on receipt.

        Per-schema dict shape::

            {
                "name": str,
                "description": str,
                "parameters": <json-dict>,  # JSON Schema
                "category": str | null,
                "discoverability": str,
                "editable": {
                    "parameters": List[str],
                    "format": str,
                    "template": str | null,
                } | null,
                "traits": List[str],
            }

        Returns:
            ``(True, {"schemas": [<dict>, ...]})`` on success.
            Empty list when the session has no tools configured.

            ``(False, {"error": ..., "stage": ...})`` on
            ``no_host`` / ``no_session`` / ``missing_method`` /
            ``call``.
        """
        ready, err, session = self._require_ready_session()
        if not ready:
            return err
        get_method = getattr(session, "get_tool_schemas", None)
        if not callable(get_method):
            return False, {
                "error": (
                    "session.get_tool_schemas: session class lacks public "
                    "get_tool_schemas() method"
                ),
                "stage": "missing_method",
            }
        try:
            schemas = get_method()
        except Exception as exc:  # noqa: BLE001 — boundary
            return False, {
                "error": (
                    f"session.get_tool_schemas: get_tool_schemas raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                "stage": "call",
            }
        serialized: List[Dict[str, Any]] = []
        for s in (schemas or []):
            editable = getattr(s, "editable", None)
            editable_serialized = None
            if editable is not None:
                editable_serialized = {
                    "parameters": list(getattr(editable, "parameters", []) or []),
                    "format": str(getattr(editable, "format", "yaml") or "yaml"),
                    "template": (
                        str(editable.template)
                        if getattr(editable, "template", None) is not None
                        else None
                    ),
                }
            serialized.append({
                "name": str(getattr(s, "name", "")),
                "description": str(getattr(s, "description", "") or ""),
                "parameters": dict(getattr(s, "parameters", {}) or {}),
                "category": (
                    str(s.category)
                    if getattr(s, "category", None) is not None
                    else None
                ),
                "discoverability": str(
                    getattr(s, "discoverability", DISCOVERABILITY_DEFERRED)
                    or DISCOVERABILITY_DEFERRED,
                ),
                "editable": editable_serialized,
                # FrozenSet → list for wire safety (JSON has no set type).
                "traits": list(getattr(s, "traits", frozenset()) or []),
            })
        return True, {"schemas": serialized}

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
                    logger.info(   # [RPC_DIAG] register-stall trace — DIAG BRANCH
                        "[RPC_DIAG] serve recv method=%s id=%s", env.method, env.id)
                    if env.method == "session.bootstrap":
                        # Pool PR 5a-fix: ``session.bootstrap`` runs
                        # synchronously on the main thread (NOT via
                        # ``self._pool.submit``).  Reason: step 1c
                        # of ``bootstrap_session`` calls
                        # ``aa_change_profile`` which is per-thread
                        # in the Linux apparmor kernel module —
                        # only the CALLING thread gets confined.  If
                        # bootstrap ran in a worker thread, only that
                        # worker would be confined; subsequently-
                        # spawned workers (for tool.execute etc.)
                        # would inherit the MAIN thread's
                        # ``unconfined`` cred via pthread_create.
                        # Running synchronously on main thread means
                        # main confines BEFORE any worker spawns;
                        # later worker threads inherit the confined
                        # cred cleanly.  See the v67 cascade smoke
                        # debug for the empirical evidence (worker
                        # thread aa_change_profile silent-no-ops at
                        # the verification step because /proc/self/
                        # attr/current returns process-level state,
                        # i.e. main thread's profile).  Same
                        # synchronous-on-main-thread pattern that
                        # cold-spawn uses in ``__main__.py`` step 2
                        # — pool slot now mirrors it.
                        self._handle_request(env)
                    elif env.method in WORK_LANE_METHODS:
                        # Unbounded: runs model or user code.
                        self._pool.submit(self._handle_request, env)
                    else:
                        # Control plane -- bounded work, its own lane, so it
                        # cannot queue behind a turn or a tool.  Unclassified
                        # methods land here by falling through; the guard in
                        # ``test_every_rpc_method_has_a_lane`` makes that a
                        # test failure rather than a silent latency cliff.
                        self._control_pool.submit(self._handle_request, env)
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
            self._control_pool.shutdown(wait=False, cancel_futures=True)

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
# Path F (cycle 7): AgentUIHooks → NotificationFrame shim
# ----------------------------------------------------------------------


class _AgentUIHooksNotificationShim:
    """Runner-side ``AgentUIHooks`` shim that emits NotificationFrames.

    Path F (cycle 7).  Bridges the 4 ``AgentUIHooks`` methods the
    runner-side ``JaatoSession`` calls (``on_tool_call_start``,
    ``on_tool_call_end``, ``on_tool_output``, ``on_turn_progress``)
    to the §7c step 6.6.4.1 NotificationFrame protocol.  Daemon-side
    ``_build_send_message_notification_handler`` demuxes by
    ``event_type`` and re-emits via ``ServerAgentHooks``.

    All eight ``AgentUIHooks`` methods that have runner-side callers
    (``on_agent_completed`` / ``on_session_quiescent`` /
    ``on_agent_created`` / ``on_agent_status_changed`` /
    ``on_agent_turn_completed`` / ``on_agent_context_updated`` /
    ``on_agent_gc_config`` / ``on_agent_history_updated``) emit
    NotificationFrames here.

    Audit history: pre-§7c these methods were "covered daemon-side"
    because their callers (``lifecycle_tools._execute_signal_completion``,
    ``subagent`` plugin, ``JaatoClient`` wrapper) all ran in the
    daemon process where ``ServerAgentHooks`` lives.  Post-§7c the
    runner subprocess hosts the subagent plugin (PLUGIN_TIER =
    "runner") and JaatoClient, so every ``self._ui_hooks.on_X`` call
    on the runner side hits this shim — pre-fix the shim was ``pass``
    no-op for all 8 methods, silently dropping the events before they
    could reach the daemon-side reactor engine + event-bus subscribers.

    The remaining methods (``on_agent_output``) use a different path
    (``on_output`` kwarg threading through the stream callback chain,
    not the ui_hooks slot) and are intentionally no-ops here.
    - ``on_agent_instruction_budget_updated``: already covered by
      the §7c step 6.6.4.2 ``instruction_budget_updated``
      notification frame.

    Pre-Path-F the runner-side session's ``_ui_hooks`` was None and
    these methods silently dropped — see
    ``docs/design/project_backlog_runner_ui_hooks_gap.md``
    (Finding 3) for the audit-of-record back-reference.
    """

    def __init__(self, rpc: Any, request_id: int) -> None:
        self._rpc = rpc
        self._request_id = request_id

    def on_tool_call_start(
        self,
        agent_id: str,
        tool_name: str,
        tool_args: "Dict[str, Any]",
        call_id: Optional[str] = None,
    ) -> None:
        try:
            self._rpc.emit_notification(
                request_id=self._request_id,
                event_type=self._rpc._NOTIF_TOOL_CALL_START,
                payload={
                    "agent_id": str(agent_id or ""),
                    "tool_name": str(tool_name or ""),
                    "tool_args": dict(tool_args or {}),
                    "call_id": call_id,
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("tool_call_start notify raised")

    def on_tool_call_end(
        self,
        agent_id: str,
        tool_name: str,
        success: bool,
        duration_seconds: float,
        error_message: Optional[str] = None,
        call_id: Optional[str] = None,
        backgrounded: bool = False,
        continuation_id: Optional[str] = None,
        show_output: Optional[bool] = None,
        show_popup: Optional[bool] = None,
        is_error_result: bool = False,
        result_status: Optional[str] = None,
    ) -> None:
        try:
            self._rpc.emit_notification(
                request_id=self._request_id,
                event_type=self._rpc._NOTIF_TOOL_CALL_END,
                payload={
                    "agent_id": str(agent_id or ""),
                    "tool_name": str(tool_name or ""),
                    "success": bool(success),
                    "is_error_result": bool(is_error_result),
                    "result_status": result_status,
                    "duration_seconds": float(duration_seconds),
                    "error_message": error_message,
                    "call_id": call_id,
                    "backgrounded": bool(backgrounded),
                    "continuation_id": continuation_id,
                    "show_output": show_output,
                    "show_popup": show_popup,
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("tool_call_end notify raised")

    def on_tool_output(
        self,
        agent_id: str,
        call_id: str,
        chunk: str,
    ) -> None:
        try:
            self._rpc.emit_notification(
                request_id=self._request_id,
                event_type=self._rpc._NOTIF_TOOL_OUTPUT,
                payload={
                    "agent_id": str(agent_id or ""),
                    "call_id": str(call_id or ""),
                    "chunk": str(chunk or ""),
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("tool_output notify raised")

    def on_turn_progress(
        self,
        agent_id: str,
        total_tokens: int,
        prompt_tokens: int,
        output_tokens: int,
        percent_used: float,
        pending_tool_calls: int,
        cache_read_tokens: Optional[int] = None,
        cache_creation_tokens: Optional[int] = None,
    ) -> None:
        try:
            self._rpc.emit_notification(
                request_id=self._request_id,
                event_type=self._rpc._NOTIF_TURN_PROGRESS,
                payload={
                    "agent_id": str(agent_id or ""),
                    "total_tokens": int(total_tokens or 0),
                    "prompt_tokens": int(prompt_tokens or 0),
                    "output_tokens": int(output_tokens or 0),
                    "percent_used": float(percent_used or 0.0),
                    "pending_tool_calls": int(pending_tool_calls or 0),
                    "cache_read_tokens": cache_read_tokens,
                    "cache_creation_tokens": cache_creation_tokens,
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("turn_progress notify raised")

    # ---- no-op fillers for the rest of the AgentUIHooks protocol ----
    # The runner-side session doesn't call these (per the
    # backlog doc's callsite enumeration), but defining them keeps
    # the shim duck-type-compatible with AgentUIHooks consumers
    # that might iterate or hasattr-check.

    def on_agent_created(
        self,
        agent_id: str,
        agent_name: str = "",
        agent_type: str = "",
        profile_name: Optional[str] = None,
        parent_agent_id: Optional[str] = None,
        created_at: Any = None,
        **_kwargs: Any,
    ) -> None:
        """Forward ``on_agent_created`` across the wire.

        Called by the subagent plugin (PLUGIN_TIER="runner") when a
        new subagent session is provisioned.  Daemon-side
        ``ServerAgentHooks.on_agent_created`` populates
        ``server._agents`` and fires ``AgentCreatedEvent``.
        """
        try:
            created_at_str: Optional[str]
            if created_at is None:
                created_at_str = None
            elif hasattr(created_at, "isoformat"):
                created_at_str = created_at.isoformat()
            else:
                created_at_str = str(created_at)
            self._rpc.emit_notification(
                request_id=self._request_id,
                event_type=self._rpc._NOTIF_AGENT_CREATED,
                payload={
                    "agent_id": str(agent_id or ""),
                    "agent_name": str(agent_name or ""),
                    "agent_type": str(agent_type or ""),
                    "profile_name": profile_name,
                    "parent_agent_id": parent_agent_id,
                    "created_at": created_at_str,
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("agent_created notify raised")

    def on_agent_output(self, *args: Any, **kwargs: Any) -> None:
        # Runner-side session uses the ``on_output`` kwarg path
        # (stream frames), not _ui_hooks.on_agent_output — covered
        # daemon-side at _start_model_thread's output_callback.
        pass

    def on_agent_status_changed(
        self,
        agent_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Forward ``on_agent_status_changed`` across the wire.

        Called from runner-side subagent plugin status transitions
        (idle / active / cancelled / errored).  Daemon-side
        ``ServerAgentHooks.on_agent_status_changed`` mutates
        ``server._agents[agent_id].status`` and fires
        ``AgentStatusChangedEvent``.
        """
        try:
            self._rpc.emit_notification(
                request_id=self._request_id,
                event_type=self._rpc._NOTIF_AGENT_STATUS_CHANGED,
                payload={
                    "agent_id": str(agent_id or ""),
                    "status": str(status or ""),
                    "error": error,
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("agent_status_changed notify raised")

    def on_agent_completed(
        self,
        agent_id: str,
        completed_at: Any = None,
        success: bool = True,
        token_usage: Optional[Dict[str, int]] = None,
        turns_used: Optional[int] = None,
        error: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Forward ``on_agent_completed`` across the wire.

        Pre-fix (2026-05-12) this was a ``pass`` no-op that
        dropped the event entirely.  Runner-side
        ``lifecycle_tools._execute_signal_completion`` calls this
        method after validating the agent's typed payload; the
        daemon-side ``ServerAgentHooks.on_agent_completed``
        receives the demuxed frame and fires
        ``AgentCompletedEvent`` into the reactor engine.
        """
        try:
            completed_at_str: Optional[str]
            if completed_at is None:
                completed_at_str = None
            elif hasattr(completed_at, "isoformat"):
                completed_at_str = completed_at.isoformat()
            else:
                completed_at_str = str(completed_at)
            self._rpc.emit_notification(
                request_id=self._request_id,
                event_type=self._rpc._NOTIF_AGENT_COMPLETED,
                payload={
                    "agent_id": str(agent_id or ""),
                    "completed_at": completed_at_str,
                    "success": bool(success),
                    "token_usage": (
                        dict(token_usage) if token_usage else None
                    ),
                    "turns_used": (
                        int(turns_used) if turns_used is not None else None
                    ),
                    "error": str(error or ""),
                    "payload": payload,
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("agent_completed notify raised")

    def on_session_quiescent(
        self,
        agent_id: str,
        reason: str = "natural",
    ) -> None:
        """Forward ``on_session_quiescent`` across the wire.

        Pre-fix (2026-05-12) this was a ``pass`` no-op.  Runner-
        side ``JaatoSession``'s quiescence hook (jaato_session.py
        line ~4906) calls this after the
        ``signal_completion``-bearing turn has fully wrapped up;
        the daemon-side ``ServerAgentHooks.on_session_quiescent``
        receives the demuxed frame and fires
        ``SessionTerminatedEvent`` to attached clients.
        """
        try:
            self._rpc.emit_notification(
                request_id=self._request_id,
                event_type=self._rpc._NOTIF_SESSION_QUIESCENT,
                payload={
                    "agent_id": str(agent_id or ""),
                    "reason": str(reason or "natural"),
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("session_quiescent notify raised")

    def on_agent_turn_completed(
        self,
        agent_id: str,
        turn_number: int,
        prompt_tokens: int,
        output_tokens: int,
        total_tokens: int,
        duration_seconds: float,
        function_calls: List[Dict[str, Any]],
        cache_read_tokens: Optional[int] = None,
        cache_creation_tokens: Optional[int] = None,
        spend_total_tokens: Optional[int] = None,
        spend_cache_read_tokens: Optional[int] = None,
        spend_cache_creation_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        finish_reason: str = "stop",
    ) -> None:
        """Forward ``on_agent_turn_completed`` across the wire.

        Daemon-side ``ServerAgentHooks.on_agent_turn_completed``
        appends turn-accounting + fires ``TurnCompletedEvent`` for
        TUI per-turn timing / token-cost display.

        ``function_calls`` is the per-call timing list captured in
        ``turn_data['function_calls']`` (each entry: ``{name, start_time,
        end_time, duration_seconds}``).  ``TurnCompletedEvent.function_calls``
        is typed ``List[Dict[str, Any]]`` — pass the list through verbatim.
        """
        try:
            self._rpc.emit_notification(
                request_id=self._request_id,
                event_type=self._rpc._NOTIF_AGENT_TURN_COMPLETED,
                payload={
                    "agent_id": str(agent_id or ""),
                    "turn_number": int(turn_number or 0),
                    "spend_total_tokens": (
                        int(spend_total_tokens)
                        if spend_total_tokens is not None else None
                    ),
                    # None stays None here too — "no cache usage reported"
                    # is not "zero cache traffic".
                    "spend_cache_read_tokens": (
                        int(spend_cache_read_tokens)
                        if spend_cache_read_tokens is not None else None
                    ),
                    "spend_cache_creation_tokens": (
                        int(spend_cache_creation_tokens)
                        if spend_cache_creation_tokens is not None else None
                    ),
                    "prompt_tokens": int(prompt_tokens or 0),
                    "output_tokens": int(output_tokens or 0),
                    "total_tokens": int(total_tokens or 0),
                    "duration_seconds": float(duration_seconds or 0.0),
                    # None stays None across the wire: "the provider
                    # reported no cost" and "the cost was zero" are
                    # different answers, and ``float(x or 0.0)`` would
                    # collapse them.
                    "cost_usd": (
                        float(cost_usd) if cost_usd is not None else None
                    ),
                    "function_calls": list(function_calls or []),
                    "cache_read_tokens": cache_read_tokens,
                    "cache_creation_tokens": cache_creation_tokens,
                    "finish_reason": str(finish_reason or "stop"),
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("agent_turn_completed notify raised")

    def on_agent_context_updated(
        self,
        agent_id: str,
        total_tokens: int,
        prompt_tokens: int,
        output_tokens: int,
        turns: int,
        percent_used: float,
    ) -> None:
        """Forward ``on_agent_context_updated`` across the wire.

        Daemon-side ``ServerAgentHooks.on_agent_context_updated``
        mutates context_usage on ``server._agents`` + fires
        ``ContextUpdatedEvent`` (the event TUI uses to render the
        usage bar).
        """
        try:
            self._rpc.emit_notification(
                request_id=self._request_id,
                event_type=self._rpc._NOTIF_AGENT_CONTEXT_UPDATED,
                payload={
                    "agent_id": str(agent_id or ""),
                    "total_tokens": int(total_tokens or 0),
                    "prompt_tokens": int(prompt_tokens or 0),
                    "output_tokens": int(output_tokens or 0),
                    "turns": int(turns or 0),
                    "percent_used": float(percent_used or 0.0),
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("agent_context_updated notify raised")

    def on_agent_gc_config(
        self,
        agent_id: str,
        threshold: float,
        strategy: str,
        target_percent: Optional[float] = None,
        continuous_mode: bool = False,
    ) -> None:
        """Forward ``on_agent_gc_config`` across the wire.

        Daemon-side ``ServerAgentHooks.on_agent_gc_config`` stores
        GC config on agent state + fires ``GCConfigEvent``.
        """
        try:
            self._rpc.emit_notification(
                request_id=self._request_id,
                event_type=self._rpc._NOTIF_AGENT_GC_CONFIG,
                payload={
                    "agent_id": str(agent_id or ""),
                    "threshold": float(threshold or 0.0),
                    "strategy": str(strategy or ""),
                    "target_percent": target_percent,
                    "continuous_mode": bool(continuous_mode),
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("agent_gc_config notify raised")

    def on_agent_history_updated(
        self,
        agent_id: str,
        history: Any,
    ) -> None:
        """Forward ``on_agent_history_updated`` across the wire.

        Daemon-side ``ServerAgentHooks.on_agent_history_updated``
        stores the snapshot under ``server._agents[agent_id].history``
        for the persist/restore + session-inspector paths.

        ``history`` is an opaque snapshot — passed through to the
        daemon verbatim.  Pydantic JSON serialization happens at
        ``emit_notification`` time; the daemon-side demuxer just
        forwards the deserialized payload.
        """
        try:
            self._rpc.emit_notification(
                request_id=self._request_id,
                event_type=self._rpc._NOTIF_AGENT_HISTORY_UPDATED,
                payload={
                    "agent_id": str(agent_id or ""),
                    "history": history,
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("agent_history_updated notify raised")

    def on_agent_instruction_budget_updated(self, *args: Any, **kwargs: Any) -> None:
        # Covered by §7c step 6.6.4.2 instruction_budget_updated
        # notification — invoked via set_instruction_budget_callback,
        # not via _ui_hooks.
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
    2. **Real ``Message`` dataclass** → canonical session
       serializer (``shared.plugins.session.serializer.serialize_message``).
       Path E (cycle 6).  Pre-Path-E this path used
       ``dataclasses.asdict`` which produced a different wire shape
       (parts as raw dataclass dump) than the canonical session
       serializer expects (parts as tagged-union).  The mismatch
       crashed both ``session_manager._save_session`` and the
       replay path with ``'dict' object has no attribute 'role'``.
       Switching to the canonical serializer makes the wire format
       round-trip cleanly through ``serialize_history`` /
       ``deserialize_history``.
    3. :func:`dataclasses.asdict` for OTHER dataclasses (test fakes
       that don't implement ``to_dict`` but aren't real Messages).
       Preserved for backward compat with existing test stubs.
       Enum values coerced via :func:`_coerce_for_json`.
    4. Pass-through (last resort — caller's try/except will catch).

    The wire form must round-trip cleanly to ``json.dumps`` — the
    runner-side framing uses JSON throughout.
    """
    to_dict = getattr(msg, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    # Path E (cycle 6): real Message dataclass → canonical
    # session-serializer wire format.
    try:
        from jaato_sdk.plugins.model_provider.types import Message
        from shared.plugins.session.serializer import serialize_message
        if isinstance(msg, Message):
            return serialize_message(msg)
    except ImportError:
        pass
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


def _extract_error_traceback(result: Any) -> "Optional[str]":
    """Frames a domain-failure result dict chose to carry, or ``None``.

    Sibling of :func:`_extract_error_message`.  A path that RETURNS a failure
    rather than raising has no exception for the dispatcher to introspect, so
    the frames can only come from the dict itself.  Absent stays absent -- a
    placeholder here would read like evidence to whoever is debugging.
    """
    if isinstance(result, dict):
        tb = result.get("traceback")
        if isinstance(tb, str) and tb:
            return tb
    return None


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
