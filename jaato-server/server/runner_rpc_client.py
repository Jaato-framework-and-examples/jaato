"""Daemon-side RPC client for the runner channel.

Pairs with :class:`server.runner.rpc.RunnerRPC`.  The daemon side
runs in asyncio (the rest of the daemon is asyncio-based), so the
parent end of the socketpair is exposed as a
``(StreamReader, StreamWriter)`` pair and the read loop is a
background asyncio task.

Public surface:

- :class:`RunnerRPCClient` — owns the parent socket, dispatches frames,
  resolves request futures, forwards stream chunks to per-call
  ``on_output`` callbacks, propagates cancel-token trips as cancel
  frames.
- :func:`call` (method on RunnerRPCClient) — the load-bearing entry
  point.  Returns a :class:`server.runner.envelope.ResponseEnvelope`.

Design notes:

- **Bridging asyncio ↔ blocking-tool world.**  ``JaatoSession`` runs
  tool execution from a worker thread (today's ``ToolExecutor.execute``
  is synchronous).  The daemon-side cli plugin's stub calls
  ``runner_rpc.call(...)`` from that thread; we expose
  :meth:`call_threadsafe` which wraps ``asyncio.run_coroutine_threadsafe``
  + waits on the resulting concurrent.futures.Future, so the calling
  thread blocks until the runner responds.
- **Cancel token integration.**  The plugin passes today's
  ``CancelToken`` (from ``shared.ai_tool_runner``).  We register a
  cancel-callback that schedules the cancel frame onto our event loop.
- **Streaming.**  Stream frames are dispatched to the per-call
  ``on_output`` callback synchronously from the read loop; if the
  callback is slow the loop slows too.  Phase 2 plugins (cli) have
  cheap callbacks (just append to a daemon-side history); Phase 3
  may revisit if a slow callback becomes a bottleneck.
- **Lifecycle.**  ``close()`` closes the parent socket (runner sees
  EOF and exits via §6.7 benign-EOF), waits up to 5s for runner exit,
  SIGTERMs after that, SIGKILLs after another 2s grace (§4.6 "Death
  — graceful").
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Tuple

if TYPE_CHECKING:  # pragma: no cover — types only
    from shared.session_envelope import SessionInitEnvelope

from shared.framing import (
    FrameTooLargeError,
    read_frame,
    write_frame,
)

from server.runner.envelope import (
    KIND_EVENT,
    KIND_REQUEST,
    KIND_RESPONSE,
    KIND_STREAM,
    CancelFrame,
    NotificationFrame,
    RequestEnvelope,
    ResponseEnvelope,
    StreamFrame,
)


logger = logging.getLogger(__name__)


# Type alias for the per-call streaming callback (mirrors
# server.runner.rpc.get_current_output_callback).
OnOutputCb = Callable[[str, str, Optional[str]], None]

# Type alias for the per-call notification callback (Phase 3
# §7c step 6.6.4.1).  Receives ``(event_type, payload)`` for
# each NotificationFrame the runner emits during the in-flight
# call.  Synchronous — runs inside the daemon's read loop, so
# the callback should be fast (queue work elsewhere if needed).
OnNotificationCb = Callable[[str, Dict[str, Any]], None]


class RunnerCallError(RuntimeError):
    """Raised when a runner RPC fails for transport/protocol reasons.

    Distinct from a tool-level failure (which surfaces in the
    response envelope's ``error`` payload, not as an exception).

    Carries the runner-side ``traceback`` when the envelope had one.  The
    runner sanitizes and ships frames in ``ErrorPayload.traceback``
    (``runner/rpc.py`` on every executor exception; ``envelope.py`` puts it
    on the wire) and NOTHING daemon-side read it.  So a crash inside a model
    loop reached every consumer as one sanitized line — exception type and
    message intact, frames gone — and the line reads like a finished error,
    which is worse than an obviously truncated one: a reader assumes they
    have the wrong log rather than that the frames were dropped.
    """

    def __init__(self, message: str, *, traceback_text: Optional[str] = None):
        super().__init__(message)
        #: Runner-side frames, or ``None`` when the envelope carried none.
        self.traceback_text = traceback_text


class RunnerRPCTimeout(TimeoutError):
    """A timeout raised by THIS transport, not by a model or a provider.

    **Subclasses ``TimeoutError`` deliberately.**  ``ipc.py``,
    ``command_router.py``, ``session_manager.py`` and ``apparmor.py`` all
    catch ``TimeoutError``; a fresh exception type would stop being caught by
    every one of them.  What the subclass adds is the ability to tell OUR
    plumbing failing from the agent failing -- which the daemon's model-thread
    handler needs and could not previously do.

    Why that matters: the model thread ends on a bare ``except Exception`` and
    terminates the session for anything it catches.  A daemon-side timeout --
    our own event loop not scheduling a coroutine -- was therefore treated
    exactly like a provider rejecting us permanently, and killed a healthy
    cascade half.  Observed twice on two builds: a stalled loop took out a
    session mid-run, the cascade policy unloaded it, and its sibling spent the
    rest of the run sending into a session that could no longer be woken.
    """


class DaemonLoopTimeout(RunnerRPCTimeout):
    """The daemon's event loop did not run a scheduled coroutine in time."""


class RunnerAnswerTimeout(RunnerRPCTimeout):
    """The runner did not answer an RPC that its loop DID run."""


async def _await_runner(coro: Any, timeout: Optional[float], method: str) -> Any:
    """Await *coro*, and if it times out say WHICH timeout that was.

    Two timeouts are stacked on every threadsafe call: this one (the RUNNER
    did not answer) and ``_run_threadsafe``'s ``future.result`` (the DAEMON
    LOOP never delivered the result).  In this Python
    ``asyncio.TimeoutError``, ``concurrent.futures.TimeoutError`` and
    ``TimeoutError`` are THE SAME CLASS, so a caller logging
    ``type(exc).__name__`` gets ``TimeoutError`` for both and cannot tell a
    busy runner from a saturated daemon loop.  Reported live: an
    ``unreachable`` delivery whose log line named the type and still did not
    identify the layer.

    The TYPE is deliberately unchanged -- several call sites catch
    ``TimeoutError`` (ipc.py, command_router.py, session_manager.py:4786,
    apparmor.py) and would stop catching a new one.  What changes is that the
    exception carries a MESSAGE: ``str(TimeoutError())`` is the empty string,
    which is why these rendered blank wherever they were logged.
    """
    if timeout is None:
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout)
    except TimeoutError as exc:
        raise RunnerAnswerTimeout(
            f"runner did not answer {method} within {timeout}s "
            f"(runner-side: the RPC was sent and no response came back)"
        ) from exc


def _result_from_loop(future: Any, timeout: Optional[float], method: str) -> Any:
    """Block on *future*, and if THAT times out say it was the daemon loop.

    The companion to :func:`_await_runner` -- see its docstring for why both
    exist and why the type stays ``TimeoutError``.

    **Reaching this branch means the loop was not running the coroutine.**
    That is not a hedge, it is implied by which timeout fired.  Every caller
    arms an inner timeout that is strictly SMALLER than this outer one
    (get_history 30/35, get_context_usage 5/10, shutdown 10/15, end 10/15),
    and the inner timer is armed only once the coroutine actually RUNS.  So if
    the outer fires and the inner did not, the coroutine either never ran or
    the loop stopped servicing timers after it started -- both of which are
    "the loop was not running it".

    An earlier version of this docstring said the inner "normally fires
    first".  It does not: measured 18 stalls in 28 minutes across two
    independent harnesses, the inner fired ZERO times.  The sentence was an
    expectation about this system that the system was contradicting in front
    of whoever read the logs, and it took a reader who trusted the docstring
    less than the evidence to notice.

    Whether the loop is BLOCKED (something synchronous holding it) or merely
    saturated is not decided here; see the daemon-loop-stall investigation.
    """
    try:
        return future.result(timeout=timeout)
    except TimeoutError as exc:
        raise DaemonLoopTimeout(
            f"daemon loop did not deliver {method} within {timeout}s "
            f"(daemon-side: the coroutine was scheduled onto the loop and "
            f"did not complete)"
        ) from exc


def _call_error(prefix: str, response: "ResponseEnvelope") -> RunnerCallError:
    """Build a :class:`RunnerCallError` that KEEPS the runner's frames.

    One helper rather than three copies of the same two lines: the frames
    were dropped identically at every site, and a fourth site would have
    dropped them again.  See :class:`RunnerCallError` for what was lost.
    """
    err = response.error
    err_type = err.type if err else "UnknownError"
    err_msg = err.message if err else "no error message"
    return RunnerCallError(
        f"{prefix} failed: {err_type}: {err_msg}",
        traceback_text=(err.traceback if err else None),
    )


class RunnerRPCClient:
    """Asyncio client wrapping the parent end of the runner socketpair.

    Constructed by the SessionManager after :meth:`RunnerSpawner.spawn`
    returns; one instance per session.  The instance is stashed on
    the session's :class:`JaatoServer` and consumed by the cli plugin
    stub.
    """

    def __init__(
        self,
        sock: socket.socket,
        *,
        runner_pid: int,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        rpc_server: Optional["RunnerRPCServer"] = None,
    ) -> None:
        """Construct the client.

        Args:
            sock: The parent end of the runner socketpair.
            runner_pid: Spawned runner PID — for waitpid + signal
                escalation in ``close``.
            loop: Optional event-loop override (defaults to current).
            rpc_server: Optional :class:`RunnerRPCServer` for handling
                runner → daemon RPCs (Phase 3 §3.2).  When ``None``,
                incoming ``request`` frames from the runner are
                rejected with an ``UnknownMethod`` error response.
                Phase 2 callers passing no server keep working —
                the runner only originated requests in Phase 3+.
        """
        self._raw_sock = sock
        self._runner_pid = runner_pid
        self._loop = loop or asyncio.get_event_loop()

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._read_task: Optional[asyncio.Task] = None
        self._closed = False
        self._next_id = 1

        # In-flight calls — resolved by the read loop when the
        # terminating ``response`` frame arrives.
        self._in_flight: Dict[int, "asyncio.Future[ResponseEnvelope]"] = {}
        # Per-call streaming callback — looked up on every stream frame.
        self._stream_cbs: Dict[int, OnOutputCb] = {}
        # Phase 3 §7c step 6.6.4.1: per-call notification callback —
        # looked up on every NotificationFrame.  Registered via
        # :meth:`call`'s ``on_notification`` kwarg.
        self._notification_cbs: Dict[int, OnNotificationCb] = {}

        # Phase 3 §3.2: handler registry for runner → daemon RPCs.
        # Lazy-init to an empty server so callers that don't use the
        # incoming direction don't have to construct one explicitly.
        if rpc_server is None:
            from server.runner_rpc_server import RunnerRPCServer
            rpc_server = RunnerRPCServer()
        self._rpc_server: "RunnerRPCServer" = rpc_server
        # Tasks dispatching incoming request frames; tracked so we
        # can cancel on close().
        self._dispatch_tasks: Dict[int, asyncio.Task] = {}

    @property
    def runner_pid(self) -> int:
        return self._runner_pid

    # --------------------------- lifecycle -----------------------------

    async def start(self) -> None:
        """Adopt the parent socket onto the asyncio event loop and
        begin the background read loop.

        Must be called from inside the event loop.  After this returns,
        :meth:`call` and :meth:`call_threadsafe` are usable.
        """
        loop = asyncio.get_running_loop()
        self._raw_sock.setblocking(False)

        reader = asyncio.StreamReader(loop=loop)
        protocol = asyncio.StreamReaderProtocol(reader, loop=loop)
        transport, _ = await loop.connect_accepted_socket(
            lambda: protocol, self._raw_sock,
        )
        writer = asyncio.StreamWriter(transport, protocol, reader, loop)

        self._reader = reader
        self._writer = writer
        self._loop = loop
        self._read_task = loop.create_task(
            self._read_loop(),
            name=f"runner-rpc-read-{self._runner_pid}",
        )

    def reset_for_slot_reuse(self) -> None:
        """Phase 2 cascade-sharing: clear per-session RPC state while
        keeping the transport (socket + read task + writer) alive.

        Called by the daemon's cascade-teardown path when a pool slot
        is returned to the pool.  The slot's socket lives on (the
        runner subprocess stays alive, ready for the next session),
        so the asyncio transport that adopted the socket via
        :meth:`start`'s ``connect_accepted_socket`` must NOT be torn
        down — doing so would close the socket and EOF the runner.

        What this clears (per-session state safe to drop between
        cascade sessions):

          - ``_in_flight``: in-flight RPC futures fail with
            ``RunnerCallError`` (no in-flight calls should exist
            at this point — the session_end RPC has already
            resolved — but defensive cleanup catches any stragglers).
          - ``_dispatch_tasks``: any in-flight daemon-side handlers
            for runner→daemon requests are cancelled.
          - ``_stream_cbs``, ``_notification_cbs``: per-session
            output/notification subscribers are dropped — the next
            session installs fresh callbacks via session_send_message.

        What this preserves (transport state tied to the socket):

          - ``_reader``, ``_writer``, ``_read_task``: the asyncio
            transport adopted by :meth:`start` stays running.  When
            the next session's ``session.bootstrap`` arrives, the
            same reader/writer carry the bytes.

        Symmetric to :meth:`close`'s "tear down per-session state"
        block but stops short of closing the socket / killing the
        runner.  Call this between cascade sessions; call
        :meth:`close` at cascade-end (slot teardown).
        """
        # In-flight futures fail loudly so any awaiter sees a clean
        # error rather than hanging.  (At this point session_end has
        # already resolved, so this should be a no-op in practice.)
        for fut in list(self._in_flight.values()):
            if not fut.done():
                fut.set_exception(
                    RunnerCallError(
                        "runner RPC reset for slot reuse "
                        "(in-flight call dropped between cascade sessions)"
                    )
                )
        self._in_flight.clear()
        for task in list(self._dispatch_tasks.values()):
            if not task.done():
                task.cancel()
        self._dispatch_tasks.clear()
        self._stream_cbs.clear()
        self._notification_cbs.clear()
        logger.debug(
            "RunnerRPCClient.reset_for_slot_reuse: cleared per-session "
            "state; transport (sock fd=%s pid=%d) stays alive",
            self._raw_sock.fileno() if self._raw_sock else "?",
            self._runner_pid,
        )

    async def close(self, *, timeout: float = 5.0) -> None:
        """Close the parent socket and wait for the runner to exit.

        §4.6 "Death — graceful" ladder:
        1. Close parent socket → runner sees EOF, exits.
        2. Wait up to *timeout* seconds for ``waitpid`` to reap.
        3. SIGTERM, wait 2s.
        4. SIGKILL, reap.
        """
        if self._closed:
            return
        self._closed = True

        # #284/#280: capture the slot's process group BEFORE teardown
        # begins.  The slot leads its own session (os.setsid at fork),
        # so its whole subtree — jdtls language servers, mcp servers,
        # PTY shells — shares this pgid.  We SIGKILL the group after the
        # slot leader is reaped (step 3) so nothing orphans to the
        # daemon subreaper and leaks memory (the OOM root cause).
        # Captured now because getpgid() fails once the slot is reaped.
        slot_pgid = self._capture_slot_pgid()

        # 1. Close socket → runner sees EOF.
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception as exc:  # noqa: BLE001
                logger.debug("RunnerRPCClient: writer close: %s", exc)

        # Cancel the read loop.
        if self._read_task is not None and not self._read_task.done():
            self._read_task.cancel()
            try:
                await self._read_task
            except (asyncio.CancelledError, Exception):
                pass

        # Cancel any in-flight runner → daemon dispatch tasks
        # (Phase 3 §3.2).  A handler awaiting operator interaction
        # gets a clean cancellation; the runner sees its outgoing
        # request fail when the read loop catches the channel-closed
        # state on the next read attempt.
        for task in list(self._dispatch_tasks.values()):
            if not task.done():
                task.cancel()
        self._dispatch_tasks.clear()

        # Fail any still-pending callers with a clean error.
        for fut in list(self._in_flight.values()):
            if not fut.done():
                fut.set_exception(
                    RunnerCallError("runner RPC closed before response arrived")
                )
        self._in_flight.clear()
        self._stream_cbs.clear()
        self._notification_cbs.clear()

        # 2. Wait for runner to exit.
        await self._wait_runner_exit(timeout)

        # 3. #284/#280: sweep the slot's process group — SIGKILL any
        # surviving subprocesses (jdtls et al.) the slot spawned.  This
        # is the daemon-side guarantee the slot-side backstops (atexit /
        # PDEATHSIG) could not provide: it runs on EVERY teardown path
        # (clean EOF, SIGTERM, SIGKILL).  No-op when the slot already
        # reaped its children cleanly.
        self._sweep_slot_group(slot_pgid)

    def _capture_slot_pgid(self) -> Optional[int]:
        """Return the slot's process-group id for teardown sweeping.

        Returns ``None`` — skipping the sweep — when the slot is already
        gone, OR (defensively) when the slot shares the **daemon's**
        process group.  A shared group means ``os.setsid`` regressed at
        slot-fork; ``os.killpg`` on it would SIGKILL the daemon itself,
        so we refuse and log loudly rather than self-destruct.  See
        #284/#280.
        """
        try:
            pgid = os.getpgid(self._runner_pid)
        except (ProcessLookupError, OSError):
            return None
        if pgid == os.getpgrp():
            logger.error(
                "RunnerRPCClient: slot pid=%d shares the daemon process "
                "group (pgid=%d) — slot setsid regressed; SKIPPING subtree "
                "sweep to avoid killing the daemon. jdtls may leak (#284). "
                "Fix slot-fork os.setsid().",
                self._runner_pid, pgid,
            )
            return None
        return pgid

    def _sweep_slot_group(self, pgid: Optional[int]) -> None:
        """SIGKILL every surviving process in the slot's group (#284).

        Closes the orphan-on-teardown leak: subprocesses the slot
        spawned (jdtls, mcp, PTY shells) inherit its pgid and would
        otherwise re-parent to the daemon subreaper and accumulate until
        OOM.  Logs the swept members so the daemon log proves the leak
        is closed.  No-op when the group is already empty (the common
        case once the slot exits cleanly).
        """
        if pgid is None:
            return
        survivors = self._list_group_members(pgid)
        if not survivors:
            return  # slot reaped its children cleanly — nothing to do
        logger.info(
            "RunnerRPCClient: slot pid=%d teardown — sweeping process group "
            "pgid=%d: SIGKILL %d survivor(s): %s",
            self._runner_pid, pgid, len(survivors),
            ", ".join(f"{p}({c})" for p, c in survivors),
        )
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            return
        # Reap survivors that re-parent to us (the subreaper) as zombies.
        for pid, _comm in survivors:
            try:
                os.waitpid(pid, os.WNOHANG)
            except (ChildProcessError, OSError):
                pass

    @staticmethod
    def _list_group_members(pgid: int) -> List[Tuple[int, str]]:
        """Enumerate live ``(pid, comm)`` in process group *pgid* via /proc.

        Best-effort instrumentation for the teardown sweep (#284): the
        returned list is logged (proving which subprocesses leaked) and
        used to reap zombies.  Races with process exit are benign — the
        subsequent ``killpg`` is the authority.  ``/proc/<pid>/stat``
        field 5 is the process group; field 2 (``comm``) may contain
        spaces/parens, so we slice between the first ``(`` and last
        ``)``.
        """
        members: List[Tuple[int, str]] = []
        try:
            entries = os.listdir("/proc")
        except OSError:
            return members
        for entry in entries:
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat", "rb") as fh:
                    data = fh.read()
            except (FileNotFoundError, ProcessLookupError, OSError):
                continue
            rparen = data.rfind(b")")
            lparen = data.find(b"(")
            if rparen < 0 or lparen < 0 or rparen < lparen:
                continue
            comm = data[lparen + 1:rparen].decode("utf-8", "replace")
            # After "...) ": fields[0]=state, [1]=ppid, [2]=pgrp
            fields = data[rparen + 2:].split()
            if len(fields) >= 3:
                try:
                    if int(fields[2]) == pgid:
                        members.append((int(entry), comm))
                except ValueError:
                    continue
        return members

    async def _wait_runner_exit(self, timeout: float) -> None:
        """Reap the runner with a SIGTERM/SIGKILL ladder."""
        deadline = self._loop.time() + timeout
        while self._loop.time() < deadline:
            try:
                pid, status = os.waitpid(self._runner_pid, os.WNOHANG)
            except ChildProcessError:
                return  # already reaped (e.g. SIGCHLD watcher beat us)
            if pid != 0:
                logger.info(
                    "RunnerRPCClient: runner pid=%d exited (status=%d)",
                    self._runner_pid, status,
                )
                return
            await asyncio.sleep(0.05)

        # SIGTERM ladder.
        logger.warning(
            "RunnerRPCClient: runner pid=%d did not exit within %.1fs of "
            "EOF; sending SIGTERM",
            self._runner_pid, timeout,
        )
        try:
            os.kill(self._runner_pid, signal.SIGTERM)
        except ProcessLookupError:
            return

        deadline = self._loop.time() + 2.0
        while self._loop.time() < deadline:
            try:
                pid, _ = os.waitpid(self._runner_pid, os.WNOHANG)
            except ChildProcessError:
                return
            if pid != 0:
                return
            await asyncio.sleep(0.05)

        logger.warning(
            "RunnerRPCClient: runner pid=%d still alive after SIGTERM; "
            "escalating to SIGKILL", self._runner_pid,
        )
        try:
            os.kill(self._runner_pid, signal.SIGKILL)
            os.waitpid(self._runner_pid, 0)
        except (ProcessLookupError, ChildProcessError):
            pass

    # --------------------------- read loop -----------------------------

    async def _read_loop(self) -> None:
        """Decode frames until EOF or unrecoverable error.

        On peer-EOF (graceful) or any transport error, fail every
        in-flight future and return — the caller's ``close()`` cleans
        up the rest.
        """
        try:
            while True:
                try:
                    raw = await read_frame(self._reader)
                except FrameTooLargeError as exc:
                    logger.error(
                        "RunnerRPCClient: peer sent oversized frame: %s",
                        exc,
                    )
                    break
                if raw is None:
                    logger.info("RunnerRPCClient: runner closed connection")
                    break

                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.error(
                        "RunnerRPCClient: malformed JSON frame: %s", exc,
                    )
                    break

                kind = payload.get("kind")
                if kind == KIND_RESPONSE:
                    try:
                        env = ResponseEnvelope.from_dict(payload)
                    except (KeyError, ValueError) as exc:
                        logger.error(
                            "RunnerRPCClient: bad response frame: %s", exc,
                        )
                        continue
                    fut = self._in_flight.pop(env.id, None)
                    logger.info(   # [RPC_DIAG] DIAG BRANCH — daemon reply match
                        "[RPC_DIAG] daemon read_loop RESPONSE id=%s in_flight_had=%s client=%s",
                        env.id, fut is not None, id(self))
                    self._stream_cbs.pop(env.id, None)
                    self._notification_cbs.pop(env.id, None)
                    if fut is not None and not fut.done():
                        fut.set_result(env)
                    else:
                        logger.debug(
                            "RunnerRPCClient: response for unknown id=%d",
                            env.id,
                        )
                elif kind == KIND_STREAM:
                    try:
                        sf = StreamFrame.from_dict(payload)
                    except (KeyError, ValueError) as exc:
                        logger.error(
                            "RunnerRPCClient: bad stream frame: %s", exc,
                        )
                        continue
                    cb = self._stream_cbs.get(sf.id)
                    if cb is not None:
                        try:
                            cb(sf.source, sf.text, sf.mode)
                        except Exception:  # noqa: BLE001
                            logger.exception(
                                "RunnerRPCClient: on_output for id=%d raised",
                                sf.id,
                            )
                elif kind == KIND_REQUEST:
                    # Phase 3 §3.2: runner → daemon RPC.  Dispatch in
                    # a child task so a slow handler (e.g. a
                    # ``client.prompt_operator`` waiting on operator
                    # input) doesn't block the read loop.
                    try:
                        env = RequestEnvelope.from_dict(payload)
                    except (KeyError, ValueError) as exc:
                        logger.error(
                            "RunnerRPCClient: bad runner request frame: %s",
                            exc,
                        )
                        continue
                    task = self._loop.create_task(
                        self._handle_runner_request(env),
                        name=f"runner-rpc-dispatch-{env.id}",
                    )
                    self._dispatch_tasks[env.id] = task
                    task.add_done_callback(
                        lambda t, fid=env.id: self._dispatch_tasks.pop(fid, None)
                    )
                elif kind == KIND_EVENT:
                    # Phase 3 §7c step 6.6.4.1: dispatch
                    # notification frames to the per-call
                    # notification callback registered via
                    # :meth:`call`'s ``on_notification`` kwarg.
                    # Used by §7c step 6.6.4.2's 7-callback
                    # collapse — runner-side session callbacks
                    # emit NotificationFrame; daemon-side
                    # consumer demuxes by ``event_type``.
                    try:
                        nf = NotificationFrame.from_dict(payload)
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "RunnerRPCClient: malformed notification frame: %r",
                            payload,
                        )
                        continue
                    cb = self._notification_cbs.get(nf.id)
                    if cb is None:
                        # No registered handler for this in-flight
                        # call — debug-log + drop.  Matches the
                        # pre-§7c-step-6.6.4.1 unhandled-event
                        # log behavior.
                        logger.debug(
                            "RunnerRPCClient: unhandled notification "
                            "frame for id=%d (event_type=%r)",
                            nf.id, nf.event_type,
                        )
                        continue
                    try:
                        cb(nf.event_type, nf.payload)
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "RunnerRPCClient: on_notification for "
                            "id=%d (event_type=%r) raised",
                            nf.id, nf.event_type,
                        )
                else:
                    logger.warning(
                        "RunnerRPCClient: unknown frame kind=%r", kind,
                    )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("RunnerRPCClient: read loop crashed")
        finally:
            # Mark the channel closed so subsequent ``call()`` attempts
            # raise RunnerCallError up front instead of writing to a
            # dead transport (which would surface as ConnectionResetError
            # at a confusing layer).
            self._closed = True
            for fid, fut in list(self._in_flight.items()):
                if not fut.done():
                    fut.set_exception(
                        RunnerCallError(
                            f"runner RPC closed before id={fid} responded"
                        )
                    )
            self._in_flight.clear()
            self._stream_cbs.clear()
            self._notification_cbs.clear()

    # --------------------- runner → daemon dispatch --------------------

    def register_handler(
        self,
        method: str,
        handler: Callable[[Dict[str, Any]], "Awaitable[Any]"],
    ) -> None:
        """Bind *method* to *handler* on the underlying RunnerRPCServer.

        Convenience wrapper that forwards to ``self._rpc_server.register``.
        Handlers register via the client because the client owns the
        socket; the server is just the dispatch table.
        """
        self._rpc_server.register(method, handler)

    @property
    def rpc_server(self) -> "RunnerRPCServer":
        """Read accessor for the underlying handler registry.

        Useful for callers that want to register many handlers in
        one place (e.g. a session-init flow registering all of
        ``client.prompt_operator``, ``apparmor.add_reference_fragment``,
        ``telemetry.publish``).
        """
        return self._rpc_server

    async def _handle_runner_request(self, env: RequestEnvelope) -> None:
        """Dispatch one runner → daemon request and write the response.

        Runs in a child task spawned by the read loop so handlers
        that await I/O don't block subsequent frames.  Any exception
        in the handler is caught by :meth:`RunnerRPCServer.dispatch`
        and serialized into the response envelope's ``error`` field;
        the only thing that can fail HERE is the response-write
        itself, which we handle by logging + giving up (the runner
        will see the read-loop's EOF on the next attempt).
        """
        response = await self._rpc_server.dispatch(env)
        if self._closed or self._writer is None:
            logger.debug(
                "RunnerRPCClient: skipping response for id=%d — channel closed",
                env.id,
            )
            return
        try:
            await write_frame(self._writer, json.dumps(response.to_dict()))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RunnerRPCClient: failed to write response for id=%d: %s",
                env.id, exc,
            )

    # ----------------------------- call --------------------------------

    async def call(
        self,
        method: str,
        args: Optional[Dict[str, Any]] = None,
        *,
        on_output: Optional[OnOutputCb] = None,
        on_notification: Optional[OnNotificationCb] = None,
        cancel_token: Optional[Any] = None,
    ) -> ResponseEnvelope:
        """Send a request, await the terminating response.

        Args:
            method: ``"echo"`` or ``"tool.execute"``.
            args: Method args (e.g. ``{"name": "cli_based_tool",
                "args": {"command": "echo hi"}}``).
            on_output: Optional callback invoked for each stream frame
                ``(source, text, mode)``.
            on_notification: Optional callback invoked for each
                NotificationFrame (Phase 3 §7c step 6.6.4.1)
                ``(event_type, payload)``.  Used by long-running
                RPCs (currently :meth:`session.send_message`)
                for runner-side session callbacks that surface
                events back to the daemon.
            cancel_token: Optional :class:`CancelToken` (the SDK type).
                If set and tripped, sends a cancel frame to the runner.

        Returns:
            The :class:`ResponseEnvelope` parsed from the runner's
            terminating response.

        Raises:
            RunnerCallError: transport-level failure (peer
                disconnect, malformed frame, etc).
        """
        if self._closed:
            raise RunnerCallError("RunnerRPCClient is closed")
        if self._writer is None:
            raise RunnerCallError(
                "RunnerRPCClient.start() must be called before call()"
            )

        request_id = self._next_id
        self._next_id += 1

        env = RequestEnvelope(id=request_id, method=method, args=args or {})

        fut: "asyncio.Future[ResponseEnvelope]" = self._loop.create_future()
        self._in_flight[request_id] = fut
        logger.info(   # [RPC_DIAG] DIAG BRANCH — daemon future registered
            "[RPC_DIAG] daemon _in_flight SET id=%s client=%s", request_id, id(self))
        if on_output is not None:
            self._stream_cbs[request_id] = on_output
        if on_notification is not None:
            self._notification_cbs[request_id] = on_notification

        # Wire cancel-token → cancel frame.  We register a callback
        # that schedules ``_send_cancel`` onto our loop.  The token
        # may be tripped from any thread, so we use call_soon_threadsafe.
        cancel_handle: Optional[Callable[[], None]] = None
        if cancel_token is not None and hasattr(cancel_token, "register_callback"):
            def _on_cancel() -> None:
                self._loop.call_soon_threadsafe(
                    lambda: self._loop.create_task(self._send_cancel(request_id))
                )
            cancel_token.register_callback(_on_cancel)
        elif cancel_token is not None and hasattr(cancel_token, "wait"):
            # Fallback: poll-via-callback isn't available; spawn a
            # waiter task.  The wait completes when the token's
            # internal Event is set.
            async def _waiter() -> None:
                while not getattr(cancel_token, "is_cancelled", False):
                    await asyncio.sleep(0.05)
                    if fut.done():
                        return
                if not fut.done():
                    await self._send_cancel(request_id)
            self._loop.create_task(_waiter(), name=f"cancel-waiter-{request_id}")

        # Send the request.
        await write_frame(self._writer, json.dumps(env.to_dict()))

        try:
            return await fut
        finally:
            self._in_flight.pop(request_id, None)
            self._stream_cbs.pop(request_id, None)
            self._notification_cbs.pop(request_id, None)

    async def _send_cancel(self, request_id: int) -> None:
        """Send a cancel frame for *request_id* if the call is still in flight."""
        if self._closed or self._writer is None:
            return
        if request_id not in self._in_flight:
            return  # already finished
        try:
            await write_frame(
                self._writer,
                json.dumps(CancelFrame(id=request_id).to_dict()),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "RunnerRPCClient: cancel-write for id=%d failed: %s",
                request_id, exc,
            )

    # --------- threadsafe wrapper for the cli plugin's sync stub ---------

    def _guard_not_on_loop(self, method: str) -> None:
        """Refuse a blocking threadsafe call made FROM the loop thread.

        Every ``*_threadsafe`` wrapper schedules a coroutine onto
        ``self._loop`` and then BLOCKS on the future.  From a worker thread
        that is correct and is what they exist for.  From the loop thread it
        is a self-deadlock: the thread is waiting for a coroutine that only
        it could run.  It breaks only on timeout — observed live as a
        constant-width 10s stall per streaming-progress notification,
        diagnosed by the loop watchdog in its first ninety seconds (#631):
        seven stalls, seven identical stacks, deepest frame
        ``session_get_context_limit_threadsafe`` under
        ``_read_loop -> cb(...)``.

        Raising HERE, before scheduling, is the load-bearing detail: raising
        after ``run_coroutine_threadsafe`` would leave the coroutine queued
        to run with its side effects once the loop frees up, detached from
        any caller.

        This converts every future instance of the class -- any callback
        anyone ever adds to a loop-side path that makes a threadsafe
        round-trip -- from a silent, recurring, timeout-width stall into a
        loud one-time failure at the exact line.
        """
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            return                       # not on any loop thread: fine
        if running is self._loop:
            raise RuntimeError(
                f"{method} called from the event-loop thread it targets. "
                f"This self-deadlocks: the call blocks the loop waiting for "
                f"a coroutine only the loop can run, and breaks only on "
                f"timeout. Call it from a worker thread, or use the async "
                f"variant and await it."
            )

    def call_threadsafe(
        self,
        method: str,
        args: Optional[Dict[str, Any]] = None,
        *,
        on_output: Optional[OnOutputCb] = None,
        cancel_token: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> ResponseEnvelope:
        """Synchronous wrapper for callers off the event loop.

        Used by the daemon-side cli plugin stub, which runs in the
        ``ToolExecutor``'s worker thread (synchronous contract).
        Wraps :meth:`call` via ``asyncio.run_coroutine_threadsafe``
        and blocks the caller until the response arrives.

        Args:
            timeout: Optional wall-clock cap.  ``None`` means no
                timeout — callers should set this to a finite value
                in production to avoid wedging on a half-dead runner.

        Raises:
            concurrent.futures.TimeoutError: when *timeout* fires.
            RunnerCallError: transport failure.
        """
        self._guard_not_on_loop("call_threadsafe")
        coro = self.call(
            method, args, on_output=on_output, cancel_token=cancel_token,
        )
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return _result_from_loop(future, timeout, method)

    # --------------------- session.bootstrap (Phase 3 §3.3c) -------------

    async def bootstrap_session(
        self,
        envelope: "SessionInitEnvelope",
        *,
        timeout: Optional[float] = 30.0,
    ) -> Dict[str, Any]:
        """Send a ``session.bootstrap`` RPC carrying *envelope*.

        Phase 3 §3.3c part 1: the runner stores the resulting
        :class:`server.runner.session.RunnerSessionHost` and is
        ready to dispatch runner-tier plugin calls against
        ``host.session._executor`` (Phase 3 §3.4-§3.10 wave
        migrations).

        Args:
            envelope: Pre-built :class:`SessionInitEnvelope`.
                Daemon callers build this from the resolved profile
                + plugin set inside ``_create_session_impl`` (or
                §3.12.0's bootstrap helper).
            timeout: Wall-clock cap.  Default 30s — bootstrap may
                involve provider connect + plugin discovery on the
                runner side, which can be slow on cold imports.

        Returns:
            The response envelope's ``result`` dict (typed by the
            runner-side ``_handle_session_bootstrap``):
            ``{"ok": True, "ready": bool, "session_id": str}`` on
            success; ``{"error": "...", "stage": "..."}`` on
            failure.

        Raises:
            RunnerCallError: transport-level failure (peer
                disconnect, malformed frame).
            asyncio.TimeoutError: when *timeout* fires.
        """
        from shared.session_envelope import SessionInitEnvelope as _SE
        if not isinstance(envelope, _SE):
            raise TypeError(
                f"bootstrap_session: expected SessionInitEnvelope, got "
                f"{type(envelope).__name__}"
            )

        coro = self.call("session.bootstrap", envelope.to_dict())
        response = await _await_runner(coro, timeout, "session.bootstrap")

        if not response.ok or response.error is not None:
            raise _call_error("session.bootstrap", response)
        result = response.result if isinstance(response.result, dict) else {}
        return result

    def bootstrap_session_threadsafe(
        self,
        envelope: "SessionInitEnvelope",
        *,
        timeout: Optional[float] = 30.0,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for ``bootstrap_session`` from worker
        threads.

        ``_create_session_impl`` runs synchronously in a worker
        thread (today's hook architecture); this wrapper lets it
        send the bootstrap envelope without bouncing through
        ``asyncio.run_coroutine_threadsafe`` boilerplate at every
        call site.
        """
        self._guard_not_on_loop("bootstrap_session_threadsafe")
        coro = self.bootstrap_session(envelope, timeout=timeout)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return _result_from_loop(
            future,
            (timeout + 5.0 if timeout is not None else None),
            "session.bootstrap",
        )

    # ------------------------------------------------------------------
    # Phase 3 §3.3c precursor — named-method wrappers
    # ------------------------------------------------------------------
    #
    # Each of these wraps a runner-side handler with a typed Python
    # API.  All return the raw result dict the runner produces; the
    # daemon-side caller branches on its keys (``ready``, ``value``,
    # ``cancelled``, etc.).  Errors from the runner side surface as
    # :class:`RunnerCallError` (transport / handler crash) or as
    # error-shaped result dicts (``{"error": ..., "stage": ...}``)
    # for the per-handler defensive cases.

    async def session_health_check(
        self, *, timeout: Optional[float] = 5.0,
    ) -> Dict[str, Any]:
        """Probe the runner-side session host's status.

        Returns ``{"has_host": bool, "ready": bool,
        "session_id": str, "tool_count": int}``.  Always succeeds
        at the protocol level (the handler never returns ok=False
        for missing host — the dict's ``has_host`` field carries
        that signal).
        """
        return await self._call_named(
            "session.health_check", {}, timeout=timeout,
        )

    async def session_end(
        self, *, timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        """Cascade-sharing session boundary — Phase 2.

        Tells the runner-side session host to fire
        ``reset_for_next_session()`` on every initialized plugin.
        Caller (daemon's session-teardown path) uses the returned
        dict to decide whether the slot is safe to return to the
        pool:

          - ``errors == []`` → slot is fully reset, safe to
            ``pool_manager.return_slot_after_session(slot)``.
          - ``errors != []`` → at least one plugin's reset raised;
            slot is in undefined state, MUST be torn down
            (``slot.sock.close()`` + waitpid), not pooled.

        Returns ``{"plugins_reset": int, "errors": List[str]}``.

        Default 10s timeout: plugin reset is per-plugin in-memory
        work (clearing dicts, dropping callbacks).  Allow some
        slack for plugins like ``interactive_shell`` whose reset
        closes PTY sessions (which may flush buffers).
        """
        return await self._call_named(
            "session.end", {}, timeout=timeout,
        )

    def session_end_threadsafe(
        self, *, timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for ``session_end`` from worker
        threads.  Mirrors ``bootstrap_session_threadsafe`` —
        session teardown runs synchronously inside the daemon's
        per-session worker thread.
        """
        return self._run_threadsafe(
            self.session_end(timeout=timeout), timeout=timeout,
        )

    def session_health_check_threadsafe(
        self, *, timeout: Optional[float] = 5.0,
    ) -> Dict[str, Any]:
        return self._run_threadsafe(
            self.session_health_check(timeout=timeout), timeout=timeout,
        )

    async def session_register_client_tools(
        self, client_tools: List[Dict[str, Any]], *,
        timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        """Mid-session: glue newly-registered client-provided ("host") tool
        SCHEMAS to the LIVE runner registry so the runner-tier model sees them
        WITHOUT a session restart (calls ``_register_client_tools_on_runner``).
        Execution still forwards daemon-side → the client via the existing
        host-tool proxy.  Returns ``{"registered": [names]}``.
        """
        return await self._call_named(
            "session.register_client_tools",
            {"client_tools": client_tools}, timeout=timeout,
        )

    def session_register_client_tools_threadsafe(
        self, client_tools: List[Dict[str, Any]], *,
        timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        return self._run_threadsafe(
            self.session_register_client_tools(client_tools, timeout=timeout),
            timeout=timeout,
        )

    async def session_get_state(
        self,
        key: str,
        default: Any = None,
        *,
        timeout: Optional[float] = 5.0,
    ) -> Any:
        """Read a single session-state key from the runner.

        Returns the raw value (whatever the runner-side
        ``JaatoSession.get_session_state(key, default)`` returns —
        possibly None, primitive, or container).
        """
        result = await self._call_named(
            "session.get_session_state",
            {"key": key, "default": default},
            timeout=timeout,
        )
        return result.get("value")

    def session_get_state_threadsafe(
        self, key: str, default: Any = None,
        *, timeout: Optional[float] = 5.0,
    ) -> Any:
        return self._run_threadsafe(
            self.session_get_state(key, default, timeout=timeout),
            timeout=timeout,
        )

    async def session_set_state(
        self,
        key: str,
        value: Any,
        *,
        timeout: Optional[float] = 5.0,
    ) -> None:
        """Write a single session-state key on the runner.

        ``value`` must be JSON-serialisable per the JaatoSession
        contract; non-serialisable values raise
        :class:`RunnerCallError` with the runner's
        ``stage="validate"`` error.
        """
        await self._call_named(
            "session.set_session_state",
            {"key": key, "value": value},
            timeout=timeout,
        )

    def session_set_state_threadsafe(
        self, key: str, value: Any,
        *, timeout: Optional[float] = 5.0,
    ) -> None:
        self._run_threadsafe(
            self.session_set_state(key, value, timeout=timeout),
            timeout=timeout,
        )

    async def session_is_running(
        self, *, timeout: Optional[float] = 5.0,
    ) -> bool:
        result = await self._call_named(
            "session.is_running", {}, timeout=timeout,
        )
        return bool(result.get("running", False))

    def session_is_running_threadsafe(
        self, *, timeout: Optional[float] = 5.0,
    ) -> bool:
        return self._run_threadsafe(
            self.session_is_running(timeout=timeout), timeout=timeout,
        )

    async def session_request_stop(
        self,
        reason: str = "",
        *,
        timeout: Optional[float] = 5.0,
    ) -> bool:
        """Signal cancellation to the runner-side in-flight message.

        Returns True if a cancellation was actually issued (matches
        :meth:`JaatoSession.request_stop`'s contract); False if no
        message was running.
        """
        result = await self._call_named(
            "session.request_stop", {"reason": reason},
            timeout=timeout,
        )
        return bool(result.get("cancelled", False))

    def session_request_stop_threadsafe(
        self, reason: str = "",
        *, timeout: Optional[float] = 5.0,
    ) -> bool:
        return self._run_threadsafe(
            self.session_request_stop(reason, timeout=timeout),
            timeout=timeout,
        )

    async def session_try_completion_nudge(
        self,
        max_nudges: int,
        *,
        timeout: Optional[float] = 5.0,
    ) -> Tuple[bool, int]:
        """Atomic check-and-increment for the completion-nudge guard.

        Phase 3 §7c step 6.6.4.3a.  Replaces 3 daemon-side
        private-attr reaches into the JaatoSession
        (``_signal_completion_called`` read,
        ``_completion_nudges_fired`` read + increment) with one
        RPC round-trip — required for §7c step 6.6.4.3b's
        seat-flip where the JaatoSession lives in a separate
        process.

        Args:
            max_nudges: Caller's nudge-budget knob (the
                ``_start_model_thread`` site uses
                ``MAX_COMPLETION_NUDGES = 2``).

        Returns:
            ``(should_nudge, nudges_fired)`` tuple.
            ``nudges_fired`` is the post-increment value when
            ``should_nudge`` is True; the unchanged current count
            otherwise.

        Raises:
            RunnerCallError on transport failure or runner-side
                exception (decode / no_host / no_session /
                missing_method / call).
        """
        result = await self._call_named(
            "session.try_completion_nudge",
            {"max_nudges": int(max_nudges)},
            timeout=timeout,
        )
        return (
            bool(result.get("should_nudge", False)),
            int(result.get("nudges_fired", 0)),
        )

    def session_try_completion_nudge_threadsafe(
        self,
        max_nudges: int,
        *,
        timeout: Optional[float] = 5.0,
    ) -> Tuple[bool, int]:
        return self._run_threadsafe(
            self.session_try_completion_nudge(max_nudges, timeout=timeout),
            timeout=timeout,
        )

    async def session_try_drain_pending_user(
        self, *, timeout: Optional[float] = 5.0,
    ) -> Optional[str]:
        """Atomically pop a pending high-priority (USER/PARENT/SYSTEM)
        message for the daemon's post-turn drain (multi-turn deadlock fix).

        See :meth:`JaatoSession.try_drain_pending_user`.  Returns the message
        text to run as the next turn, or ``None`` when nothing is queued (or
        a turn is already running runner-side).

        Raises:
            RunnerCallError on transport failure or runner-side exception.
        """
        result = await self._call_named(
            "session.try_drain_pending_user",
            {},
            timeout=timeout,
        )
        return result.get("text")

    def session_try_drain_pending_user_threadsafe(
        self, *, timeout: Optional[float] = 5.0,
    ) -> Optional[str]:
        return self._run_threadsafe(
            self.session_try_drain_pending_user(timeout=timeout),
            timeout=timeout,
        )

    async def session_get_auth_info(
        self, *, timeout: Optional[float] = 5.0,
    ) -> str:
        """Read the credential-source description string from the
        runner-side session's provider.

        Phase 3 §7c step 6.6.4.5c.1.  Replaces 2 daemon-side reaches
        into ``self._jaato.auth_info`` (core.py:2073, 4481) — the
        property reads ``_session._provider.get_auth_info()`` daemon-
        side, which post-seat-flip is the wrong (dead) session.

        Returns a human-readable string like ``"API key from
        ~/.jaato/zhipuai_auth.json"`` or ``"PKCE OAuth"``.  Empty
        string when no provider is attached or the provider doesn't
        implement ``get_auth_info``.

        Raises:
            RunnerCallError on transport failure or runner-side
                exception (no_host / no_session / missing_method /
                call).  Daemon callers using this in display paths
                may want to wrap in try/except and fall back to ""
                on any error.
        """
        result = await self._call_named(
            "session.get_auth_info", {}, timeout=timeout,
        )
        return str(result.get("auth_info", "") or "")

    def session_get_auth_info_threadsafe(
        self, *, timeout: Optional[float] = 5.0,
    ) -> str:
        return self._run_threadsafe(
            self.session_get_auth_info(timeout=timeout),
            timeout=timeout,
        )

    async def session_get_user_commands(
        self, *, timeout: Optional[float] = 10.0,
    ) -> "Dict[str, Any]":
        """Read the runner-side session's user-command catalog.

        Phase 3 §7c step 6.6.4.5c.2.  Replaces 2 daemon-side reaches
        into ``self._jaato.get_user_commands()``.  Wire shape per the
        5c.2 audit decision: dict-shape-only — the wrapper
        reconstructs ``UserCommand`` / ``CommandParameter`` NamedTuple
        instances on receipt so daemon callers can use the existing
        ``parse_command_args(cmd, raw_args)`` helper unmodified.

        Returns a ``Dict[str, UserCommand]`` matching the pre-§7c
        ``JaatoClient.get_user_commands()`` shape.

        Raises:
            RunnerCallError on transport failure or runner-side
                exception (no_host / no_session / missing_method /
                call).
        """
        from jaato_sdk.plugins.base import UserCommand, CommandParameter

        result = await self._call_named(
            "session.get_user_commands", {}, timeout=timeout,
        )
        commands_raw = result.get("commands", {})
        if not isinstance(commands_raw, dict):
            raise RunnerCallError(
                f"session_get_user_commands: expected dict for "
                f"'commands', got {type(commands_raw).__name__}"
            )
        commands: Dict[str, UserCommand] = {}
        for name, cmd_dict in commands_raw.items():
            if not isinstance(cmd_dict, dict):
                continue
            params_raw = cmd_dict.get("parameters")
            params: "Optional[list]" = None
            if isinstance(params_raw, list):
                params = [
                    CommandParameter(
                        name=str(p.get("name", "")),
                        description=str(p.get("description", "")),
                        required=bool(p.get("required", False)),
                        capture_rest=bool(p.get("capture_rest", False)),
                    )
                    for p in params_raw
                    if isinstance(p, dict)
                ]
            commands[str(name)] = UserCommand(
                name=str(cmd_dict.get("name", name)),
                description=str(cmd_dict.get("description", "")),
                share_with_model=bool(cmd_dict.get("share_with_model", False)),
                parameters=params,
            )
        return commands

    def session_get_user_commands_threadsafe(
        self, *, timeout: Optional[float] = 10.0,
    ) -> "Dict[str, Any]":
        return self._run_threadsafe(
            self.session_get_user_commands(timeout=timeout),
            timeout=timeout,
        )

    async def session_execute_user_command(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = 60.0,
    ) -> Tuple[Any, bool]:
        """Invoke a user command on the runner-side session.

        Phase 3 §7c step 6.6.4.5c.3.  Replaces the daemon-side
        reach into ``self._jaato.execute_user_command(name, args)``
        (core.py:4044).

        Default timeout is 60s — user commands run synchronously
        and may include slow network ops (auth flows, model
        catalog fetches).  Daemon callers wanting a hard cap
        pass an explicit value.

        Args:
            name: Command name (e.g. ``"auth"``, ``"model"``).
            args: Optional parsed-args dict from
                ``parse_command_args``.

        Returns:
            ``(result, shared_with_model)`` tuple matching the
            pre-§7c ``JaatoClient.execute_user_command`` shape.
            ``result`` is reconstructed on receipt:
            - tagged ``HelpLines`` → :class:`HelpLines` instance
              with ``lines`` re-tupled (``List[tuple]``).
            - tagged ``dict`` → the dict unchanged.
            - tagged ``str`` → the stringified value.

        Raises:
            RunnerCallError on transport failure or runner-side
                exception (decode / no_host / no_session /
                missing_method / call).  Daemon-side exception
                handler at core.py wraps user-command failures in
                ``{"error": ...}`` returns.
        """
        from jaato_sdk.plugins.base import HelpLines

        result_dict = await self._call_named(
            "session.execute_user_command",
            {"name": str(name), "args": dict(args or {})},
            timeout=timeout,
        )
        tagged = result_dict.get("result", {})
        shared = bool(result_dict.get("shared", False))

        if not isinstance(tagged, dict) or "_kind" not in tagged:
            # Defensive: unrecognized shape — surface as str
            # (matches daemon-side ``str(result)`` fallback).
            return str(tagged) if tagged is not None else "", shared

        kind = tagged.get("_kind")
        if kind == "HelpLines":
            lines_raw = tagged.get("lines", []) or []
            # Re-tuple each line entry — the wire flattens tuples to
            # lists, but the daemon's HelpLines.lines contract is
            # ``List[tuple]``.
            lines = [tuple(entry) for entry in lines_raw if isinstance(entry, list)]
            return HelpLines(lines=lines), shared
        if kind == "dict":
            value = tagged.get("value", {})
            return (value if isinstance(value, dict) else {}), shared
        # kind == "str" or anything else.
        value = tagged.get("value", "")
        return str(value or ""), shared

    def session_execute_user_command_threadsafe(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = 60.0,
    ) -> Tuple[Any, bool]:
        return self._run_threadsafe(
            self.session_execute_user_command(name, args, timeout=timeout),
            timeout=timeout,
        )

    async def session_get_model_completions(
        self,
        args: "Optional[list]" = None,
        *,
        timeout: Optional[float] = 10.0,
    ) -> "list":
        """Get completion candidates for the "model" command's
        subcommand arguments.

        Phase 3 §7c step 6.6.4.5c.4.  Replaces 2 daemon-side
        reaches into ``self._jaato.get_model_completions(args)``
        (core.py:4285 + command_router.py:1149).  Wire shape per
        the 5c.4 audit: dict-shape-only (Path A, mirror of 5c.2's
        UserCommand pattern).

        Args:
            args: Optional list of argument strings typed so far.
                ``None`` or empty list returns subcommands;
                ``["select"]`` returns model names; etc.

        Returns:
            List of :class:`CommandCompletion` NamedTuple
            instances reconstructed on receipt — daemon callers
            can read ``.value`` and ``.description`` unchanged.

        Raises:
            RunnerCallError on transport failure or runner-side
                exception (decode / no_host / no_session /
                missing_method / call).
        """
        from jaato_sdk.plugins.base import CommandCompletion

        result = await self._call_named(
            "session.get_model_completions",
            {"args": list(args or [])},
            timeout=timeout,
        )
        completions_raw = result.get("completions", [])
        if not isinstance(completions_raw, list):
            raise RunnerCallError(
                f"session_get_model_completions: expected list for "
                f"'completions', got {type(completions_raw).__name__}"
            )
        completions = []
        for entry in completions_raw:
            if not isinstance(entry, dict):
                continue
            completions.append(CommandCompletion(
                value=str(entry.get("value", "")),
                description=str(entry.get("description", "") or ""),
            ))
        return completions

    def session_get_model_completions_threadsafe(
        self,
        args: "Optional[list]" = None,
        *,
        timeout: Optional[float] = 10.0,
    ) -> "list":
        return self._run_threadsafe(
            self.session_get_model_completions(args, timeout=timeout),
            timeout=timeout,
        )

    async def session_get_tool_schemas(
        self, *, timeout: Optional[float] = 10.0,
    ) -> "list":
        """Read the runner-side session's resolved tool schemas.

        Phase 3 §7c step 6.6.4.5c.5.  Replaces 2 daemon-side
        reaches into ``self._jaato.get_tool_schemas()`` (core.py:1407
        + core.py:3759).  Returns the session-resolved subset
        (preloaded + on-demand-activated tools), NOT the runtime's
        full registry set.

        Wire shape per the 5c.5 audit: dict-shape-only (Path A);
        ``traits`` field round-trips FrozenSet ↔ list.  Daemon
        wrapper reconstructs ``ToolSchema`` (and nested
        ``EditableContent``) instances on receipt so existing
        daemon callers can read ``.name`` / ``.category`` /
        ``.traits`` / etc. unmodified.

        Raises:
            RunnerCallError on transport failure or runner-side
                exception (no_host / no_session / missing_method /
                call).
        """
        from jaato_sdk.plugins.model_provider.types import (
            EditableContent, ToolSchema, DISCOVERABILITY_DEFERRED,
        )

        result = await self._call_named(
            "session.get_tool_schemas", {}, timeout=timeout,
        )
        schemas_raw = result.get("schemas", [])
        if not isinstance(schemas_raw, list):
            raise RunnerCallError(
                f"session_get_tool_schemas: expected list for 'schemas', "
                f"got {type(schemas_raw).__name__}"
            )
        schemas = []
        for entry in schemas_raw:
            if not isinstance(entry, dict):
                continue
            editable_raw = entry.get("editable")
            editable = None
            if isinstance(editable_raw, dict):
                editable = EditableContent(
                    parameters=list(editable_raw.get("parameters", []) or []),
                    format=str(editable_raw.get("format", "yaml") or "yaml"),
                    template=(
                        str(editable_raw["template"])
                        if editable_raw.get("template") is not None
                        else None
                    ),
                )
            traits_raw = entry.get("traits", []) or []
            traits = frozenset(str(t) for t in traits_raw)
            schemas.append(ToolSchema(
                name=str(entry.get("name", "")),
                description=str(entry.get("description", "") or ""),
                parameters=dict(entry.get("parameters", {}) or {}),
                category=(
                    str(entry["category"])
                    if entry.get("category") is not None
                    else None
                ),
                discoverability=str(
                    entry.get("discoverability", DISCOVERABILITY_DEFERRED)
                    or DISCOVERABILITY_DEFERRED,
                ),
                editable=editable,
                traits=traits,
            ))
        return schemas

    def session_get_tool_schemas_threadsafe(
        self, *, timeout: Optional[float] = 10.0,
    ) -> "list":
        return self._run_threadsafe(
            self.session_get_tool_schemas(timeout=timeout),
            timeout=timeout,
        )

    async def session_get_history(
        self,
        *,
        raw: bool = False,
        timeout: Optional[float] = 30.0,
    ) -> list:
        """Read the runner-side conversation history.

        Returns a list of message dicts (each as serialized by
        :meth:`Message.to_dict`).  When ``raw=True``, returns the
        un-transformed view (premium pseudonymization consumers).
        Default 30s timeout because large histories may take time
        to serialize over the wire.
        """
        result = await self._call_named(
            "session.get_history", {"raw": raw}, timeout=timeout,
        )
        return list(result.get("history", []))

    def session_get_history_threadsafe(
        self, *, raw: bool = False, timeout: Optional[float] = 30.0,
    ) -> list:
        return self._run_threadsafe(
            self.session_get_history(raw=raw, timeout=timeout),
            timeout=timeout,
        )

    async def session_get_context_usage(
        self, *, timeout: Optional[float] = 5.0,
    ) -> Dict[str, Any]:
        """Read context-window usage stats from the runner."""
        result = await self._call_named(
            "session.get_context_usage", {}, timeout=timeout,
        )
        return dict(result.get("usage", {}))

    async def session_get_context_limit(
        self, *, timeout: Optional[float] = 5.0,
    ) -> int:
        """Read the context-window size in tokens from the runner.

        Daemon-side fallback for when ``session_get_context_usage``
        returns a usage dict with a missing / zero ``context_limit``
        (provider not yet initialized; dict contract evolves)."""
        result = await self._call_named(
            "session.get_context_limit", {}, timeout=timeout,
        )
        return int(result.get("context_limit", 0))

    def session_get_context_limit_threadsafe(
        self, *, timeout: Optional[float] = 5.0,
    ) -> int:
        return self._run_threadsafe(
            self.session_get_context_limit(timeout=timeout),
            timeout=timeout,
        )

    def session_get_context_usage_threadsafe(
        self, *, timeout: Optional[float] = 5.0,
    ) -> Dict[str, Any]:
        return self._run_threadsafe(
            self.session_get_context_usage(timeout=timeout),
            timeout=timeout,
        )

    async def session_get_rendered_system_instruction(
        self, *, timeout: Optional[float] = 5.0,
    ) -> Optional[str]:
        """The system instruction as it stood at the end of ``configure()``.

        Issue #787: the daemon persists this so a revive RESTORES the
        prompt (via ``BootstrapEnvelope.system_instruction_override``)
        instead of re-deriving it.  Re-deriving re-runs the profile's
        mandatory ``{{!py:...}}`` prefetch, which on revive was handed an
        empty ``agent_params`` and aborted session-prep — a session that
        ran fine and persisted fine could then not be woken at all.

        Returns:
            The rendered prompt, or ``None`` when the runner-side session
            has not been configured (nothing to snapshot).  ``None`` is a
            valid answer, not an error: the caller persists nothing and
            the revive falls back to re-rendering.
        """
        result = await self._call_named(
            "session.get_rendered_system_instruction", {}, timeout=timeout,
        )
        rendered = result.get("rendered_system_instruction")
        return rendered if isinstance(rendered, str) else None

    def session_get_rendered_system_instruction_threadsafe(
        self, *, timeout: Optional[float] = 5.0,
    ) -> Optional[str]:
        return self._run_threadsafe(
            self.session_get_rendered_system_instruction(timeout=timeout),
            timeout=timeout,
        )

    async def session_apply_budget_degrade(
        self, rungs: list, pool_pressure: Optional[str] = None,
        *, timeout: Optional[float] = 5.0,
    ) -> Dict[str, Any]:
        """Push cascade degrade rungs onto a RUNNING session.

        ``pool_pressure`` is the POOL's rendered pressure — the child would
        otherwise report its own, which reads as a contradiction next to a
        pool-triggered rung.
        """
        return await self._call_named(
            "session.apply_budget_degrade",
            {"rungs": list(rungs), "pool_pressure": pool_pressure},
            timeout=timeout,
        )

    def session_apply_budget_degrade_threadsafe(
        self, rungs: list, pool_pressure: Optional[str] = None,
        *, timeout: Optional[float] = 5.0,
    ) -> Dict[str, Any]:
        return self._run_threadsafe(
            self.session_apply_budget_degrade(
                rungs, pool_pressure, timeout=timeout),
            timeout=timeout,
        )

    async def session_get_budget_usage(
        self, *, tracker_only: bool = False, timeout: Optional[float] = 5.0,
    ) -> Dict[str, Any]:
        """The session's ABSOLUTE budget consumption, per dimension.

        ``tracker_only=True`` is MANDATORY for persistence: without it an
        unbudgeted session answers with the synthetic ``{"tokens": N}``
        fallback, which a save then writes over a real multi-dimension
        snapshot.  See :meth:`JaatoSession.get_budget_usage`.
        """
        result = await self._call_named(
            "session.get_budget_usage", {"tracker_only": tracker_only},
            timeout=timeout,
        )
        return dict(result.get("usage", {}))

    def session_get_budget_usage_threadsafe(
        self, *, tracker_only: bool = False, timeout: Optional[float] = 5.0,
    ) -> Dict[str, Any]:
        return self._run_threadsafe(
            self.session_get_budget_usage(
                tracker_only=tracker_only, timeout=timeout),
            timeout=timeout,
        )

    async def session_get_budget_exhausted(
        self, *, timeout: Optional[float] = 5.0,
    ) -> Optional[str]:
        """Why a budget ceiling stopped this session, or ``None``."""
        result = await self._call_named(
            "session.get_budget_exhausted", {}, timeout=timeout,
        )
        return result.get("reason")

    def session_get_budget_exhausted_threadsafe(
        self, *, timeout: Optional[float] = 5.0,
    ) -> Optional[str]:
        return self._run_threadsafe(
            self.session_get_budget_exhausted(timeout=timeout),
            timeout=timeout,
        )

    async def session_restore_budget_usage(
        self, usage: Dict[str, float], *,
        exhausted_reason: Optional[str] = None,
        timeout: Optional[float] = 5.0,
    ) -> bool:
        """Re-seed the runner session's budget usage after a reload.

        Counterpart to :meth:`session_get_budget_usage`.  Absolute, not a
        delta -- the caller is restoring a snapshot, not observing new spend.
        """
        result = await self._call_named(
            "session.restore_budget_usage",
            {"usage": dict(usage or {}), "exhausted_reason": exhausted_reason},
            timeout=timeout,
        )
        return bool(result.get("restored", False))

    def session_restore_budget_usage_threadsafe(
        self, usage: Dict[str, float], *,
        exhausted_reason: Optional[str] = None,
        timeout: Optional[float] = 5.0,
    ) -> bool:
        return self._run_threadsafe(
            self.session_restore_budget_usage(
                usage, exhausted_reason=exhausted_reason, timeout=timeout),
            timeout=timeout,
        )

    async def session_get_turn_accounting(
        self, *, timeout: Optional[float] = 10.0,
    ) -> list:
        """Read the runner-side per-turn token usage / timing list."""
        result = await self._call_named(
            "session.get_turn_accounting", {}, timeout=timeout,
        )
        return list(result.get("turns", []))

    def session_get_turn_accounting_threadsafe(
        self, *, timeout: Optional[float] = 10.0,
    ) -> list:
        return self._run_threadsafe(
            self.session_get_turn_accounting(timeout=timeout),
            timeout=timeout,
        )

    async def session_reset(
        self, *, timeout: Optional[float] = 5.0,
    ) -> None:
        """Clear the runner-side conversation history (fresh reset).

        Currently only the no-history reset path is supported.
        Restoring a saved history requires Message round-trip
        serialization that's deferred to a future commit.
        """
        await self._call_named("session.reset", {}, timeout=timeout)

    def session_reset_threadsafe(
        self, *, timeout: Optional[float] = 5.0,
    ) -> None:
        self._run_threadsafe(
            self.session_reset(timeout=timeout), timeout=timeout,
        )

    async def session_set_presentation_context(
        self, ctx: Any, *, timeout: Optional[float] = 5.0,
    ) -> None:
        """Push the daemon's PresentationContext to the runner.

        Accepts either a Pydantic model instance (calls
        ``model_dump`` / ``dict`` to serialize) or a pre-serialized
        dict.  The runner-side handler reconstructs the model via
        ``model_validate``.
        """
        if hasattr(ctx, "model_dump"):
            ctx_dict = ctx.model_dump()
        elif hasattr(ctx, "dict"):
            ctx_dict = ctx.dict()
        elif isinstance(ctx, dict):
            ctx_dict = ctx
        else:
            raise TypeError(
                f"session_set_presentation_context: ctx must be a "
                f"Pydantic model or dict; got {type(ctx).__name__}"
            )
        await self._call_named(
            "session.set_presentation_context",
            {"context": ctx_dict},
            timeout=timeout,
        )

    def session_set_presentation_context_threadsafe(
        self, ctx: Any, *, timeout: Optional[float] = 5.0,
    ) -> None:
        self._run_threadsafe(
            self.session_set_presentation_context(ctx, timeout=timeout),
            timeout=timeout,
        )

    async def session_get_all_state(
        self, *, timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        """Bulk-snapshot the runner-side session-attached state.

        Returns the full state dict (set-state + provider values
        merged; provider wins on key collision).  Default 10s
        timeout because providers may take time (encryption,
        external lookups, etc.) and the snapshot is invoked at
        journal-save time when latency tolerance is higher.
        """
        result = await self._call_named(
            "session.get_all_session_state", {}, timeout=timeout,
        )
        return dict(result.get("state", {}))

    def session_get_all_state_threadsafe(
        self, *, timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        return self._run_threadsafe(
            self.session_get_all_state(timeout=timeout),
            timeout=timeout,
        )

    async def session_set_terminal_width(
        self, width: int, *, timeout: Optional[float] = 5.0,
    ) -> None:
        """Push the daemon's terminal width to the runner-side
        JaatoSession.  Validates width client-side too — defensive
        against typos that would only surface at runner-decode."""
        if not isinstance(width, int) or width <= 0:
            raise ValueError(
                f"session_set_terminal_width: width must be positive int; "
                f"got {width!r}"
            )
        await self._call_named(
            "session.set_terminal_width", {"width": width}, timeout=timeout,
        )

    def session_set_terminal_width_threadsafe(
        self, width: int, *, timeout: Optional[float] = 5.0,
    ) -> None:
        self._run_threadsafe(
            self.session_set_terminal_width(width, timeout=timeout),
            timeout=timeout,
        )

    async def session_set_streaming_enabled(
        self, enabled: bool, *, timeout: Optional[float] = 5.0,
    ) -> None:
        """Toggle the runner-side JaatoSession's streaming mode."""
        await self._call_named(
            "session.set_streaming_enabled",
            {"enabled": bool(enabled)},
            timeout=timeout,
        )

    def session_set_streaming_enabled_threadsafe(
        self, enabled: bool, *, timeout: Optional[float] = 5.0,
    ) -> None:
        self._run_threadsafe(
            self.session_set_streaming_enabled(enabled, timeout=timeout),
            timeout=timeout,
        )

    async def session_set_reference_authorizer(
        self, enabled: bool, *, timeout: Optional[float] = 5.0,
    ) -> None:
        """Forward the daemon's ``ReferenceAuthorizer`` state to the
        runner-side ``JaatoSession`` as a bool flag.

        Phase 3 §7c step 6.1.  The actual ``ReferenceAuthorizer``
        Python object can't cross the RPC boundary (it holds a
        daemon-side ``AppArmorManager`` reference); the daemon
        translates ``authorizer is not None`` into the bool flag.

        When the references plugin migrates runner-side, it reads
        :meth:`JaatoSession.is_reference_authorization_enabled` and
        uses the existing ``apparmor.add_reference_fragment``
        runner→daemon RPC to authorize paths.  The session_id for
        the RPC call is already known runner-side via the bootstrap
        envelope.

        Args:
            enabled: ``True`` if the daemon-side
                ``ReferenceAuthorizer`` is non-None (i.e. WS
                provisioned an AppArmor profile for this session).
                ``False`` for unconfined sessions.
        """
        await self._call_named(
            "session.set_reference_authorizer",
            {"enabled": bool(enabled)},
            timeout=timeout,
        )

    def session_set_reference_authorizer_threadsafe(
        self, enabled: bool, *, timeout: Optional[float] = 5.0,
    ) -> None:
        self._run_threadsafe(
            self.session_set_reference_authorizer(enabled, timeout=timeout),
            timeout=timeout,
        )

    async def session_snapshot_instruction_budget(
        self, *, timeout: Optional[float] = 5.0,
    ) -> "Optional[Dict[str, Any]]":
        """Read the runner-side JaatoSession's InstructionBudget
        snapshot.

        Phase 3 §7c step 6.1.  Replaces the pre-§7c daemon-side
        read ``self._jaato.get_session().instruction_budget.snapshot()``
        in ``JaatoServer.emit_current_state`` (core.py:1091).

        Returns the snapshot dict (with ``session_id`` / ``agent_id``
        / ``agent_type`` / ``context_limit`` / ``total_tokens`` /
        ``utilization_percent`` / ``entries`` etc.) when the runner-
        side session has a budget configured, or ``None`` when the
        budget hasn't been populated yet (pre-:meth:`JaatoSession.configure`).

        Caller pulls ``agent_id`` from the returned dict directly;
        no separate RPC needed.
        """
        result = await self._call_named(
            "session.snapshot_instruction_budget", {}, timeout=timeout,
        )
        snapshot = result.get("snapshot")
        if snapshot is None:
            return None
        if not isinstance(snapshot, dict):
            raise RunnerCallError(
                f"session_snapshot_instruction_budget: expected dict, "
                f"got {type(snapshot).__name__}"
            )
        return dict(snapshot)

    def session_snapshot_instruction_budget_threadsafe(
        self, *, timeout: Optional[float] = 5.0,
    ) -> "Optional[Dict[str, Any]]":
        return self._run_threadsafe(
            self.session_snapshot_instruction_budget(timeout=timeout),
            timeout=timeout,
        )

    async def session_offer_message(
        self,
        text: str,
        *,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        require_idle: bool = False,
        timeout: Optional[float] = 5.0,
    ) -> str:
        """Ask the SESSION to queue this message, or to say a turn is needed.

        Step 2.  The daemon-side ``_model_running`` is a REPLICA of the
        session's ``_is_running`` and clears strictly later -- only once
        ``session.send_message`` returns and the model thread unwinds.  Any
        queue-or-drive decision taken against that replica can therefore be
        taken against stale state, and a message queued into a turn that has
        already ended is drained by nothing.  This verb moves the decision to
        the side that owns the state, and makes it atomic there.

        Prefer this over :meth:`session_inject_prompt` for delivery.  Inject
        queues unconditionally and reports only whether the call raised --
        the same answer whether a drain is coming or the message will sit
        forever.

        Returns:
            ``"queued"``     -- a running turn's drain WILL collect it.
            ``"busy"``       -- only when *require_idle*: a turn is running
            and the caller asked not to add to the queue in that case.
            Nothing was enqueued.

            ``"needs_turn"`` -- nothing would drain it; it was NOT enqueued,
            and the caller must start a turn with this text.  A session
            cannot start its own turn, which is why the answer comes back
            here rather than being acted on runner-side.

        Raises:
            RunnerCallError: on transport failure or a malformed response --
            an undelivered message must never be reported as delivered.
        """
        if not isinstance(text, str):
            raise ValueError(
                f"session_offer_message: text must be str; "
                f"got {type(text).__name__}"
            )
        body: Dict[str, Any] = {"text": text}
        if source_id is not None:
            body["source_id"] = source_id
        if source_type is not None:
            body["source_type"] = source_type
        if require_idle:
            body["require_idle"] = True
        result = await self._call_named(
            "session.offer_message", body, timeout=timeout,
        )
        outcome = (result or {}).get("outcome")
        if outcome not in ("queued", "needs_turn", "busy"):
            # A shape we cannot read is NOT a delivery.  Reporting it as one
            # is the failure this whole change exists to remove.
            raise RunnerCallError(
                f"session.offer_message returned an unreadable outcome "
                f"{outcome!r}; expected 'queued', 'needs_turn' or 'busy'"
            )
        return outcome

    def session_offer_message_threadsafe(
        self,
        text: str,
        *,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        require_idle: bool = False,
        timeout: Optional[float] = 5.0,
    ) -> str:
        return self._run_threadsafe(
            self.session_offer_message(
                text,
                source_id=source_id,
                source_type=source_type,
                require_idle=require_idle,
                timeout=timeout,
            ),
            timeout=timeout,
        )

    async def session_inject_prompt(
        self,
        text: str,
        *,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        timeout: Optional[float] = 5.0,
    ) -> None:
        """Inject a prompt into the runner-side session's message
        queue (mid-turn or idle-routed based on source_type).

        Phase 3 §7c step 6.1.  Replaces the pre-§7c daemon-side
        call ``self._jaato.get_session().inject_prompt(text, ...)``
        in ``JaatoServer.execute_command``-side code paths
        (core.py:3238).

        Args:
            text: The prompt text to inject.
            source_id: Optional sender id (defaults to "unknown"
                runner-side).
            source_type: Optional ``SourceType`` enum value as
                string — any member of the enum (defaults to "user"
                runner-side).  Not re-listed here: the enum owns the
                set and a prose copy drifts, which is exactly what
                happened when ``sibling`` was added.  Daemon-side
                callers typically have a
                :class:`shared.message_queue.SourceType` enum
                instance — pass ``source_type=enum.value`` to
                serialize.

        The wire carries ``source_type`` as the enum's string
        value to keep the wire schema JSON-compatible; the runner-
        side handler maps it back to the enum.
        """
        if not isinstance(text, str):
            raise ValueError(
                f"session_inject_prompt: text must be str; "
                f"got {type(text).__name__}"
            )
        body: Dict[str, Any] = {"text": text}
        if source_id is not None:
            body["source_id"] = source_id
        if source_type is not None:
            body["source_type"] = source_type
        await self._call_named(
            "session.inject_prompt", body, timeout=timeout,
        )

    def session_inject_prompt_threadsafe(
        self,
        text: str,
        *,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        timeout: Optional[float] = 5.0,
    ) -> None:
        self._run_threadsafe(
            self.session_inject_prompt(
                text,
                source_id=source_id,
                source_type=source_type,
                timeout=timeout,
            ),
            timeout=timeout,
        )

    async def session_set_initial_history(
        self,
        messages: "List[Any]",
        *,
        timeout: Optional[float] = 10.0,
    ) -> None:
        """Seed the runner-side session with replayed conversation
        history from a SessionState snapshot.

        Phase 3 §7c step 6.6.1.1.  Replaces the pre-§7c daemon-
        side call ``jaato_session.set_initial_history(initial_history)``
        at ``server/session_manager.py:2130``.

        Daemon callers pass a list of :class:`Message` instances;
        the wrapper serializes via
        ``shared.plugins.session.serializer.serialize_history``
        before sending.  Wire shape is the same JSON-compatible
        dict-list disk persistence uses.

        Default timeout is 10s — set_initial_history is normally
        cheap (just appends to an empty history) but the
        deserialize step on the runner side iterates every
        message + every part, so we give some headroom for
        large-history seeds (waypoint forks can carry hundreds
        of messages).

        Args:
            messages: List of :class:`Message` instances.
                Caller owns the list; the wrapper takes a copy
                via ``serialize_history``.
        """
        from shared.plugins.session.serializer import serialize_history

        body = {"messages": serialize_history(messages)}
        await self._call_named(
            "session.set_initial_history", body, timeout=timeout,
        )

    def session_set_initial_history_threadsafe(
        self,
        messages: "List[Any]",
        *,
        timeout: Optional[float] = 10.0,
    ) -> None:
        self._run_threadsafe(
            self.session_set_initial_history(messages, timeout=timeout),
            timeout=timeout,
        )

    async def session_restore_turn_accounting(
        self,
        turns: "List[Dict[str, Any]]",
        *,
        timeout: Optional[float] = 5.0,
    ) -> None:
        """Replace the runner-side session's per-turn token-usage /
        timing list from a SessionState snapshot.

        Phase 3 §7c step 6.6.1.2.  Replaces the pre-§7c daemon-
        side private-attr write at
        ``server/session_manager.py:2558-2559``.  Wraps the
        public method ``JaatoSession.restore_turn_accounting``
        added in §7c step 6.6.1.0.

        Wire shape: ``{"turns": [<dict>, ...]}`` — turn entries
        are already JSON-native dicts in the persistence
        serializer (no special encode needed; same wire-shape-
        reuse rationale as 6.6.1.1's `set_initial_history` and
        the existing `get_turn_accounting` round-trip).

        Args:
            turns: List of per-turn dicts from a
                :class:`SessionState` snapshot.  Each entry
                contains token counts (``prompt_tokens``,
                ``output_tokens``, ``total_tokens``, ...) and
                timing fields.  Caller owns the list; the wire
                serializer copies it.
        """
        await self._call_named(
            "session.restore_turn_accounting",
            {"turns": list(turns)},
            timeout=timeout,
        )

    def session_restore_turn_accounting_threadsafe(
        self,
        turns: "List[Dict[str, Any]]",
        *,
        timeout: Optional[float] = 5.0,
    ) -> None:
        self._run_threadsafe(
            self.session_restore_turn_accounting(turns, timeout=timeout),
            timeout=timeout,
        )

    async def session_restore_conversation_budget(
        self,
        snapshot: "Dict[str, Any]",
        *,
        timeout: Optional[float] = 5.0,
    ) -> None:
        """Restore the runner-side session's CONVERSATION instruction-
        budget entry from a SessionState snapshot.

        Phase 3 §7c step 6.6.1.3.  Replaces the pre-§7c daemon-
        side reach at ``server/session_manager.py:2592-2593``.
        Wraps the public method
        ``JaatoSession.restore_conversation_budget`` added in
        §7c step 6.6.1.0.

        Wire shape: ``{"snapshot": <dict>}`` — the snapshot is
        a JSON-native dict produced by
        ``InstructionBudget.get_conversation_snapshot()`` /
        ``SourceEntry.to_dict()``.  Same wire-shape-reuse
        rationale as 6.6.1.1's `set_initial_history` and
        6.6.1.2's `restore_turn_accounting`: no special wrapper
        needed, the persistence shape IS the wire shape.

        Args:
            snapshot: Conversation-source snapshot dict from a
                :class:`SessionState`'s ``budget_state``.  Empty
                / falsy snapshots are permitted (the underlying
                method is documented as no-op for empty input).
        """
        await self._call_named(
            "session.restore_conversation_budget",
            {"snapshot": dict(snapshot)},
            timeout=timeout,
        )

    def session_restore_conversation_budget_threadsafe(
        self,
        snapshot: "Dict[str, Any]",
        *,
        timeout: Optional[float] = 5.0,
    ) -> None:
        self._run_threadsafe(
            self.session_restore_conversation_budget(snapshot, timeout=timeout),
            timeout=timeout,
        )

    async def session_append_history_message(
        self,
        message: "Any",
        *,
        timeout: Optional[float] = 5.0,
    ) -> None:
        """Append a single message to the runner-side session's history.

        Phase 3 §7c step 6.6.3.1.  Replaces the pre-§7c daemon-
        side get-modify-reset dance at
        ``server/session_manager.py:2855`` (interrupted-tool-
        call recovery).  Wraps the public method
        ``JaatoSession.append_history_message`` added in §7c
        step 6.6.3.0.

        Wire shape: ``{"message": <dict>}`` — serialized
        :class:`Message` per
        ``shared.plugins.session.serializer.serialize_message``.
        Same wire-shape-reuse rationale as 6.6.1.1's
        set_initial_history.

        Args:
            message: A :class:`Message` instance.  Daemon
                callers pass the Message directly; the wrapper
                serializes via ``serialize_message`` before
                sending.
        """
        from shared.plugins.session.serializer import serialize_message

        body = {"message": serialize_message(message)}
        await self._call_named(
            "session.append_history_message", body, timeout=timeout,
        )

    def session_append_history_message_threadsafe(
        self,
        message: "Any",
        *,
        timeout: Optional[float] = 5.0,
    ) -> None:
        self._run_threadsafe(
            self.session_append_history_message(message, timeout=timeout),
            timeout=timeout,
        )

    async def session_snapshot_conversation_budget(
        self, *, timeout: Optional[float] = 5.0,
    ) -> "Optional[Dict[str, Any]]":
        """Return the runner-side session's CONVERSATION
        instruction-budget snapshot for persistence-save.

        Phase 3 §7c step 6.6.3.2.  Inverse of
        ``session_restore_conversation_budget`` (6.6.1.3).
        Replaces the pre-§7c daemon-side reach at
        ``server/session_manager.py:2986``.  Wraps the public
        method ``JaatoSession.snapshot_conversation_budget``
        added in §7c step 6.6.3.0.

        Returns the snapshot dict (JSON-native; same shape disk
        persistence already exercises) when the budget has a
        conversation entry; returns ``None`` when budget
        unavailable / no conversation entry.
        """
        result = await self._call_named(
            "session.snapshot_conversation_budget", {}, timeout=timeout,
        )
        snapshot = result.get("snapshot")
        if snapshot is None:
            return None
        if not isinstance(snapshot, dict):
            raise RunnerCallError(
                f"session_snapshot_conversation_budget: expected dict, "
                f"got {type(snapshot).__name__}"
            )
        return dict(snapshot)

    def session_snapshot_conversation_budget_threadsafe(
        self, *, timeout: Optional[float] = 5.0,
    ) -> "Optional[Dict[str, Any]]":
        return self._run_threadsafe(
            self.session_snapshot_conversation_budget(timeout=timeout),
            timeout=timeout,
        )

    async def session_set_parallel_tools_override(
        self, enabled: bool, *, timeout: Optional[float] = 5.0,
    ) -> None:
        """Stash a per-turn parallel-tools override on the runner-
        side session.

        Phase 3 §7c step 6.6.3.3.  Replaces the pre-§7c daemon-
        side private-attr write at
        ``server/session_manager.py:4096``.  Wraps the public
        method ``JaatoSession.set_parallel_tools_override``
        added in §7c step 6.6.3.0.

        The override is consumed once and cleared after the
        next turn boundary — caller's lifecycle is per-turn-
        pre-send_message.  Daemon's existing ``if event.parallel_tools
        is not None:`` guard at session_manager.py:4095 prevents
        passing None through this wrapper.

        Args:
            enabled: True to force parallel-tool execution for
                the next turn; False to disable.
        """
        await self._call_named(
            "session.set_parallel_tools_override",
            {"enabled": bool(enabled)},
            timeout=timeout,
        )

    def session_set_parallel_tools_override_threadsafe(
        self, enabled: bool, *, timeout: Optional[float] = 5.0,
    ) -> None:
        self._run_threadsafe(
            self.session_set_parallel_tools_override(enabled, timeout=timeout),
            timeout=timeout,
        )

    async def session_replay_messages(
        self,
        messages: "List[Any]",
        *,
        replay_timeout: float = 120.0,
        timeout: Optional[float] = 180.0,
    ) -> str:
        """Run a one-shot completion against an arbitrary message
        list and return the model's text response.

        Phase 3 §7c step 6.6.3.4.  Replaces the pre-§7c daemon-
        side call at ``server/session_manager.py:4338``.  Wraps
        the existing public method
        ``JaatoSession.replay_messages`` (no missing-method gap).

        Wire shape: ``{"messages": [<dict>, ...], "timeout":
        float}`` — messages serialized via the existing
        ``serialize_history`` (same wire-shape-reuse rationale as
        6.6.1.1 + 6.6.3.1).  Returns the model's text response
        directly (the wrapper unwraps ``{"response_text": str}``).

        Args:
            messages: The full message list to send to the
                provider.  Caller owns construction; the wrapper
                serializes via ``serialize_history``.
            replay_timeout: Maximum seconds to wait for exclusive
                provider access on the runner side.  Default
                120s (matches ``JaatoSession.replay_messages``'s
                default).
            timeout: Optional wall-clock cap on the RPC itself.
                Default 180s — wider than ``replay_timeout`` so
                the runner's domain-level timeout fires before
                the transport-level one.
        """
        from shared.plugins.session.serializer import serialize_history

        body = {
            "messages": serialize_history(messages),
            "timeout": float(replay_timeout),
        }
        result = await self._call_named(
            "session.replay_messages", body, timeout=timeout,
        )
        response_text = result.get("response_text")
        if not isinstance(response_text, str):
            raise RunnerCallError(
                f"session_replay_messages: expected str response_text; "
                f"got {type(response_text).__name__}"
            )
        return response_text

    def session_replay_messages_threadsafe(
        self,
        messages: "List[Any]",
        *,
        replay_timeout: float = 120.0,
        timeout: Optional[float] = 180.0,
    ) -> str:
        return self._run_threadsafe(
            self.session_replay_messages(
                messages,
                replay_timeout=replay_timeout,
                timeout=timeout,
            ),
            timeout=timeout,
        )

    async def session_resolve_fork_point(
        self,
        *,
        after_message: Optional[int] = None,
        after_tool_call: Optional[str] = None,
        after_timestamp: Optional[str] = None,
        history: "Optional[List[Any]]" = None,
        timeout: Optional[float] = 5.0,
    ) -> int:
        """Resolve a fork-point specifier to a message index in
        the runner-side session's history.

        Phase 3 §7c step 6.6.3.5.  Replaces the pre-§7c daemon-
        side call at ``server/session_manager.py:4362``.  Wraps
        the existing public method
        ``JaatoSession.resolve_fork_point`` (no missing-method
        gap).

        Wire shape: ``{"after_message": int?, "after_tool_call":
        str?, "after_timestamp": str?, "history": [<dict>...]?}``.
        Returns the int fork_index.

        When ``history`` is None (default), the runner uses
        ``session.get_history()`` — matches the daemon caller's
        existing pattern at session_manager.py:4363 which
        always passed the session's current history.

        Args:
            after_message: Direct message index specifier.
            after_tool_call: Tool-call id specifier.
            after_timestamp: HH:MM:SS or ISO timestamp specifier.
            history: Optional pre-snapshotted history.  When
                None, the runner reads its own current history.
            timeout: RPC wall-clock cap.
        """
        body: Dict[str, Any] = {}
        if after_message is not None:
            body["after_message"] = int(after_message)
        if after_tool_call is not None:
            body["after_tool_call"] = str(after_tool_call)
        if after_timestamp is not None:
            body["after_timestamp"] = str(after_timestamp)
        if history is not None:
            from shared.plugins.session.serializer import serialize_history
            body["history"] = serialize_history(history)
        result = await self._call_named(
            "session.resolve_fork_point", body, timeout=timeout,
        )
        fork_index = result.get("fork_index")
        if not isinstance(fork_index, int):
            raise RunnerCallError(
                f"session_resolve_fork_point: expected int fork_index; "
                f"got {type(fork_index).__name__}"
            )
        return fork_index

    def session_resolve_fork_point_threadsafe(
        self,
        *,
        after_message: Optional[int] = None,
        after_tool_call: Optional[str] = None,
        after_timestamp: Optional[str] = None,
        history: "Optional[List[Any]]" = None,
        timeout: Optional[float] = 5.0,
    ) -> int:
        return self._run_threadsafe(
            self.session_resolve_fork_point(
                after_message=after_message,
                after_tool_call=after_tool_call,
                after_timestamp=after_timestamp,
                history=history,
                timeout=timeout,
            ),
            timeout=timeout,
        )

    async def session_send_message(
        self,
        prompt: str,
        *,
        on_output: Optional[Any] = None,
        on_notification: Optional[OnNotificationCb] = None,
        cancel_token: Optional[Any] = None,
        timeout: Optional[float] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        on_result: Optional[Any] = None,
    ) -> str:
        """Phase 3 §7b.2: dispatch ``session.send_message`` to the
        runner.  Long-running — streams output via ``on_output``
        and returns the final response text.

        ``on_result`` receives the FULL result dict before the text is
        extracted.  The return stays ``str`` because every caller depends on
        it, but the runner puts more than text there -- notably the typed
        budget signal (``budget_exhausted`` / ``budget_exhausted_reason`` /
        ``budget_usage``, rpc.py).  Without this hook those fields died at
        this boundary: the daemon had the ceiling in hand and dropped it, so
        a refusal reached clients only as prose and a wake-driven driver
        waited out its whole timeout.

        Args:
            prompt: User message to send to the model.
            on_output: Optional callback ``(source, text, mode)``
                invoked for each output chunk as the model
                streams.  Same shape ``call()`` already supports
                via the stream-frame channel.
            on_notification: Optional callback
                ``(event_type, payload)`` invoked for each
                NotificationFrame the runner emits during the
                in-flight call.  Phase 3 §7c step 6.6.4.1 wire-
                format extension.  Used by the §7c step 6.6.4.2
                7-callback collapse for runner-side session
                callbacks (instruction-budget updates, retry
                notifications, mid-turn-interrupt events,
                continuation triggers, etc.) that surface back
                to the daemon during streaming.
            cancel_token: Optional CancelToken — when tripped,
                the daemon-side ``call`` sends a cancel frame to
                the runner; the runner-side handler propagates
                via the session's ``request_stop``.
            timeout: Wall-clock cap.  ``None`` (default) means no
                timeout — model API calls + multi-turn loops can
                legitimately take minutes.  Daemon-side callers
                wanting a hard cap pass an explicit value.

        Returns:
            The final model response text.

        Raises:
            RunnerCallError: on transport failure, cancellation
                (``stage="cancelled"``), or send-loop crash
                (``stage="send"``).
        """
        # Text-only sends stay wire-identical (no attachments key); user-message
        # multimodal adds the base64 attachment list for the runner session.
        args: Dict[str, Any] = {"prompt": prompt}
        if attachments:
            args["attachments"] = attachments
        coro = self.call(
            "session.send_message",
            args,
            on_output=on_output,
            on_notification=on_notification,
            cancel_token=cancel_token,
        )
        response = await _await_runner(coro, timeout, "session.send_message")
        if not response.ok or response.error is not None:
            raise _call_error("session.send_message", response)
        if not isinstance(response.result, dict):
            raise RunnerCallError(
                f"session.send_message: unexpected result type "
                f"{type(response.result).__name__}; expected dict"
            )
        if on_result is not None:
            on_result(response.result)
        return str(response.result.get("response", ""))

    def session_send_message_threadsafe(
        self,
        prompt: str,
        *,
        on_output: Optional[Any] = None,
        on_notification: Optional[OnNotificationCb] = None,
        cancel_token: Optional[Any] = None,
        timeout: Optional[float] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        on_result: Optional[Any] = None,
    ) -> str:
        """Synchronous wrapper for ``session_send_message`` from
        worker threads.  Note: long-running; the future may block
        the calling thread for the duration of the model loop.
        Daemon-side callers in worker pools should set
        ``timeout`` if a hard cap is desired.

        Phase 3 §7c step 6.6.4.3b: ``on_notification`` plumbs
        through to the underlying async wrapper for the 9-callback
        collapse demuxer used by ``_start_model_thread``.
        """
        self._guard_not_on_loop("session_send_message_threadsafe")
        coro = self.session_send_message(
            prompt,
            on_output=on_output,
            on_notification=on_notification,
            cancel_token=cancel_token,
            timeout=timeout,
            attachments=attachments,
            on_result=on_result,
        )
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return _result_from_loop(
            future,
            (timeout + 5.0 if timeout is not None else None),
            "session.send_message",
        )

    async def session_shutdown(
        self, *, timeout: Optional[float] = 10.0,
    ) -> str:
        """Graceful runner-side session teardown.

        Tells the runner to call ``close_session`` on the
        bootstrapped JaatoSession (firing on_session_end hooks)
        and drop the host reference.  The runner process itself
        stays alive — the daemon's ``RunnerRPCClient.close``
        ladder owns process termination.

        Returns the id of the session that was torn down, or
        empty string when no session was bootstrapped (no-op).

        Default 10s timeout — close_session may invoke plugin
        shutdown hooks that take time (file flushes, network
        connections, etc.).  Daemon callers wanting fail-fast
        cleanup pass timeout=2.0 explicitly.
        """
        result = await self._call_named(
            "session.shutdown", {}, timeout=timeout,
        )
        return str(result.get("shutdown_session_id", ""))

    def session_shutdown_threadsafe(
        self, *, timeout: Optional[float] = 10.0,
    ) -> str:
        return self._run_threadsafe(
            self.session_shutdown(timeout=timeout),
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_named(
        self,
        method: str,
        args: Dict[str, Any],
        *,
        timeout: Optional[float],
    ) -> Dict[str, Any]:
        """Common dispatch: call *method* with *args*; raise on
        protocol failure; return the raw result dict.

        Used by the named-method wrappers above so each one is a
        2-line pass-through to ``call(method, args)`` with the
        same error handling shape (mirrors the
        ``bootstrap_session`` template).
        """
        coro = self.call(method, args)
        response = await _await_runner(coro, timeout, method)
        if not response.ok or response.error is not None:
            raise _call_error(method, response)
        if not isinstance(response.result, dict):
            raise RunnerCallError(
                f"{method}: unexpected result type "
                f"{type(response.result).__name__}; expected dict"
            )
        return response.result

    def _run_threadsafe(
        self,
        coro: Any,
        *,
        timeout: Optional[float],
    ) -> Any:
        """Synchronous wrapper for the named-method coroutines from
        worker threads.  Mirrors ``bootstrap_session_threadsafe``."""
        self._guard_not_on_loop("_run_threadsafe")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return _result_from_loop(
            future,
            (timeout + 5.0 if timeout is not None else None),
            getattr(coro, "__qualname__", None) or "a runner RPC",
        )
