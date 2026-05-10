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
                string (one of "parent", "child", "user", "system",
                "event"; defaults to "user" runner-side).  Daemon-
                side callers typically have a
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

    async def session_send_message(
        self,
        prompt: str,
        *,
        on_output: Optional[Any] = None,
        cancel_token: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Phase 3 §7b.2: dispatch ``session.send_message`` to the
        runner.  Long-running — streams output via ``on_output``
        and returns the final response text.

        Args:
            prompt: User message to send to the model.
            on_output: Optional callback ``(source, text, mode)``
                invoked for each output chunk as the model
                streams.  Same shape ``call()`` already supports
                via the stream-frame channel.
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
        coro = self.call(
            "session.send_message",
            {"prompt": prompt},
            on_output=on_output,
            cancel_token=cancel_token,
        )
        if timeout is not None:
            response = await asyncio.wait_for(coro, timeout)
        else:
            response = await coro
        if not response.ok or response.error is not None:
            err_type = response.error.type if response.error else "UnknownError"
            err_msg = response.error.message if response.error else "no error message"
            raise RunnerCallError(
                f"session.send_message failed: {err_type}: {err_msg}"
            )
        if not isinstance(response.result, dict):
            raise RunnerCallError(
                f"session.send_message: unexpected result type "
                f"{type(response.result).__name__}; expected dict"
            )
        return str(response.result.get("response", ""))

    def session_send_message_threadsafe(
        self,
        prompt: str,
        *,
        on_output: Optional[Any] = None,
        cancel_token: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Synchronous wrapper for ``session_send_message`` from
        worker threads.  Note: long-running; the future may block
        the calling thread for the duration of the model loop.
        Daemon-side callers in worker pools should set
        ``timeout`` if a hard cap is desired."""
        coro = self.session_send_message(
            prompt,
            on_output=on_output,
            cancel_token=cancel_token,
            timeout=timeout,
        )
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(
            timeout=(timeout + 5.0 if timeout is not None else None),
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
