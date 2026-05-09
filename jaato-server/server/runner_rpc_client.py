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
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING, Tuple

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
    RequestEnvelope,
    ResponseEnvelope,
    StreamFrame,
)


logger = logging.getLogger(__name__)


# Type alias for the per-call streaming callback (mirrors
# server.runner.rpc.get_current_output_callback).
OnOutputCb = Callable[[str, str, Optional[str]], None]


class RunnerCallError(RuntimeError):
    """Raised when a runner RPC fails for transport/protocol reasons.

    Distinct from a tool-level failure (which surfaces in the
    response envelope's ``error`` payload, not as an exception).
    """


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

        # 2. Wait for runner to exit.
        await self._wait_runner_exit(timeout)

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
                    self._stream_cbs.pop(env.id, None)
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
                    # Phase 3 wires this into the daemon's EventBus
                    # per §4.4.  For Phase 2 we drop them with a
                    # debug log.
                    logger.debug(
                        "RunnerRPCClient: unhandled event frame: %r",
                        payload,
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
        cancel_token: Optional[Any] = None,
    ) -> ResponseEnvelope:
        """Send a request, await the terminating response.

        Args:
            method: ``"echo"`` or ``"tool.execute"``.
            args: Method args (e.g. ``{"name": "cli_based_tool",
                "args": {"command": "echo hi"}}``).
            on_output: Optional callback invoked for each stream frame
                ``(source, text, mode)``.
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
        if on_output is not None:
            self._stream_cbs[request_id] = on_output

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
        coro = self.call(
            method, args, on_output=on_output, cancel_token=cancel_token,
        )
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

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
        if timeout is not None:
            response = await asyncio.wait_for(coro, timeout)
        else:
            response = await coro

        if not response.ok or response.error is not None:
            err_type = response.error.type if response.error else "UnknownError"
            err_msg = response.error.message if response.error else "no message"
            raise RunnerCallError(
                f"session.bootstrap failed: {err_type}: {err_msg}"
            )
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
        coro = self.bootstrap_session(envelope, timeout=timeout)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(
            timeout=(timeout + 5.0 if timeout is not None else None),
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

    def session_health_check_threadsafe(
        self, *, timeout: Optional[float] = 5.0,
    ) -> Dict[str, Any]:
        return self._run_threadsafe(
            self.session_health_check(timeout=timeout), timeout=timeout,
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

    def session_get_context_usage_threadsafe(
        self, *, timeout: Optional[float] = 5.0,
    ) -> Dict[str, Any]:
        return self._run_threadsafe(
            self.session_get_context_usage(timeout=timeout),
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
        if timeout is not None:
            response = await asyncio.wait_for(coro, timeout)
        else:
            response = await coro
        if not response.ok or response.error is not None:
            err_type = response.error.type if response.error else "UnknownError"
            err_msg = response.error.message if response.error else "no error message"
            raise RunnerCallError(
                f"{method} failed: {err_type}: {err_msg}"
            )
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
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(
            timeout=(timeout + 5.0 if timeout is not None else None),
        )
