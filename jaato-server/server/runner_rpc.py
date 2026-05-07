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
from typing import Any, Callable, Dict, Optional, Tuple

from shared.framing import (
    FrameTooLargeError,
    read_frame,
    write_frame,
)

from server.runner.envelope import (
    KIND_EVENT,
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
    ) -> None:
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
            for fid, fut in list(self._in_flight.items()):
                if not fut.done():
                    fut.set_exception(
                        RunnerCallError(
                            f"runner RPC closed before id={fid} responded"
                        )
                    )
            self._in_flight.clear()
            self._stream_cbs.clear()

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
