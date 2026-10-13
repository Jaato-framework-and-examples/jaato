"""A daemon message over 1 MiB closed the SDK client, and the caller timed out.

jaato #1279, then #1586 in production.  The daemon accepts and sends messages
up to 16 MiB (``--ws-max-message-size``), and the SDK's ``WSClient`` passed no
``max_size`` to ``websockets.connect``, so it kept the library's 1 MiB for
what it RECEIVED.  A ``session.list`` reply of 1,203,006 bytes, or the user
echo of a long voice note, made the CLIENT close the connection with 1009;
``_read_message`` swallowed the ``ConnectionClosed``; the recovery client
reconnected; and the call waiting for the reply raised ``TimeoutError``
fifteen seconds later with nothing saying a frame had been refused.  The
daemon's own log blamed its 16 MiB limit, because it reported the 1009 it
ECHOED as the one it sent.

Every case drives a real :class:`JaatoWSServer` (``start()`` on an ephemeral
port) and a real :class:`WSClient`.  Where a case needs the daemon to answer
with a large message, the test wraps the server's dispatch for that one verb
and answers with a >1 MiB event through the server's own ``_send_to_client``
-- the size of the reply is what is simulated, never the transport.

* the default receive ceiling is the daemon's, so a 2 MiB reply arrives;
* a client configured below a reply gets :class:`MessageTooLarge` -- raised
  to a correlated call, and announced on the event stream so the facade's
  ``ask`` fails with it rather than waiting out its timeout;
* a message over the daemon's advertised limit is refused BEFORE it is sent,
  and the connection survives;
* the recovery client builds every inner client with the same ceiling;
* the daemon's log names the CLIENT when the client refused.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional, Tuple

import pytest

from jaato_sdk.client.convenience import Session, TurnTimeout
from jaato_sdk.client.errors import MessageTooLarge
from jaato_sdk.client.ws import WSClient, WSRecoveryClient
from jaato_sdk.events import (
    AgentOutputEvent,
    PoolStatusEvent,
    TurnCompletedEvent,
    serialize_event,
)
from jaato_sdk.ws_limits import (
    DEFAULT_WS_MAX_MESSAGE_SIZE as SDK_DEFAULT,
    LEGACY_WS_MAX_MESSAGE_SIZE,
)
from jaato_server.server.websocket import (
    DEFAULT_WS_MAX_MESSAGE_SIZE,
    HAS_WEBSOCKETS,
    JaatoWSServer,
)
from jaato_server.shared.tests.reversion import Reversion

pytestmark = pytest.mark.skipif(not HAS_WEBSOCKETS, reason="websockets not installed")

_WS = "jaato-sdk/jaato_sdk/client/ws.py"
_IPC = "jaato-sdk/jaato_sdk/client/ipc.py"
_FACADE = "jaato-sdk/jaato_sdk/client/convenience.py"
_SERVER = "jaato-server/jaato_server/server/websocket.py"

REVERSIONS = [
    Reversion(
        target=_WS,
        find='''        connect_kwargs: dict = {"max_size": self._max_message_size}''',
        replace='''        connect_kwargs: dict = {}''',
        test="test_a_two_mebibyte_reply_reaches_the_client",
        because="without max_size the client keeps websockets' 1 MiB and closes the connection with 1009 on any larger daemon message (#1279, #1586)",
    ),
    Reversion(
        target=_WS,
        find='''                self._close_cause = cause
            return None''',
        replace='''            return None''',
        test="test_a_refused_reply_raises_message_too_large_to_the_waiting_call",
        because="a 1009 the client did not record reaches the waiting call as a bare ConnectionError, indistinguishable from a network blip that retrying fixes",
    ),
    Reversion(
        target=_IPC,
        find='''                    self._connected = False
                    self._announce_close_cause()
                    break''',
        replace='''                    self._connected = False
                    break''',
        test="test_ask_fails_with_message_too_large_instead_of_timing_out",
        because="event-driven waiters see no exception; without the announced event the facade waited out its timeout (#1586)",
    ),
    Reversion(
        target=_FACADE,
        find='''        if getattr(ev, "error_type", None) == MESSAGE_TOO_LARGE_ERROR_TYPE:''',
        replace='''        if False:''',
        test="test_ask_fails_with_message_too_large_instead_of_timing_out",
        because="the facade must settle on the client's MessageTooLarge event; the reply it waits for was lost with the connection",
    ),
    Reversion(
        target=_WS,
        find='''            if size > limit:
                raise MessageTooLarge(''',
        replace='''            if False:
                raise MessageTooLarge(''',
        test="test_a_message_over_the_advertised_limit_is_refused_before_sending",
        because="sent, an over-limit message is answered with a 1009 that ends the connection and loses everything in flight on it",
    ),
    Reversion(
        target=_WS,
        find='''            presentation=self._presentation,
            max_message_size=self._max_message_size,
        )''',
        replace='''            presentation=self._presentation,
        )''',
        test="test_the_recovery_client_builds_every_inner_client_with_the_ceiling",
        because="a reconnect that reverted to the default would undo the configured ceiling on the first network blip",
    ),
    Reversion(
        target=_SERVER,
        find='''    if text.startswith("received"):''',
        replace='''    if False:''',
        test="test_the_daemon_log_names_the_client_when_the_client_refused",
        because="the daemon's log blamed its own 16 MiB limit for a frame the client's 1 MiB refused (#1586)",
    ),
]

#: Over the legacy 1 MiB, far under the daemon's 16 MiB -- the reported range.
_BIG = 2 * 1024 * 1024


async def _started(server: JaatoWSServer) -> Tuple[asyncio.Task, int]:
    task = asyncio.create_task(server.start())
    for _ in range(300):
        if getattr(server, "_server", None) is not None:
            break
        await asyncio.sleep(0.01)
    assert server._server is not None, "server did not bind"
    return task, server._server.sockets[0].getsockname()[1]


async def _stop(server: JaatoWSServer, task: asyncio.Task) -> None:
    await server.stop()
    try:
        await asyncio.wait_for(task, timeout=5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()


def _answer_with(server: JaatoWSServer, wire_type: str, make_reply) -> None:
    """Have the server answer frames of ``wire_type`` with ``make_reply``.

    Everything else reaches the server's real dispatch unchanged.
    ``make_reply(frame) -> [events]`` -- each sent through the server's own
    ``_send_to_client``, so the size of the reply is all that is simulated.
    """
    real = server._dispatch_client_message

    async def dispatch(client_id: str, message: Any) -> None:
        try:
            frame = json.loads(message)
        except (TypeError, ValueError):
            frame = {}
        if isinstance(frame, dict) and frame.get("type") == wire_type:
            for ev in make_reply(frame):
                await server._send_to_client(client_id, ev)
            return
        await real(client_id, message)

    server._dispatch_client_message = dispatch  # type: ignore[assignment]


def _big_pool_answer(frame: dict) -> list:
    return [PoolStatusEvent(request_id=frame.get("request_id", ""),
                            ok=False, error="x" * _BIG)]


def _pool_wire_type() -> str:
    from jaato_sdk.events import PoolStatusRequest
    return json.loads(serialize_event(PoolStatusRequest()))["type"]


def _send_wire_type() -> str:
    from jaato_sdk.events import SendMessageRequest
    return json.loads(serialize_event(SendMessageRequest(text="")))["type"]


async def _with_client(server: JaatoWSServer, body, **client_kwargs) -> Any:
    task, port = await _started(server)
    client = WSClient(f"ws://127.0.0.1:{port}", **client_kwargs)
    try:
        await client.connect(timeout=10)
        return await body(client)
    finally:
        await client.disconnect()
        await _stop(server, task)


def _server(max_message_size: Optional[int] = None) -> JaatoWSServer:
    return JaatoWSServer(host="127.0.0.1", port=0, required_token=None,
                         max_message_size=max_message_size)


def test_the_two_ends_share_one_default():
    """The daemon's default IS the SDK's, not a copy that can drift."""
    assert DEFAULT_WS_MAX_MESSAGE_SIZE == SDK_DEFAULT
    assert WSClient("ws://unused").max_message_size == DEFAULT_WS_MAX_MESSAGE_SIZE


def test_a_two_mebibyte_reply_reaches_the_client():
    server = _server()
    _answer_with(server, _pool_wire_type(), _big_pool_answer)

    async def body(client: WSClient):
        return await client.pool_status(timeout=10)

    answer = asyncio.run(_with_client(server, body))
    assert len(answer.error) == _BIG


def test_a_refused_reply_raises_message_too_large_to_the_waiting_call():
    """A client configured below the reply: told why, at once, not timed out."""
    server = _server()
    _answer_with(server, _pool_wire_type(), _big_pool_answer)

    async def body(client: WSClient):
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(MessageTooLarge) as info:
            await client.pool_status(timeout=20)
        return info.value, loop.time() - started, client.close_cause

    exc, elapsed, cause = asyncio.run(_with_client(
        server, body, max_message_size=LEGACY_WS_MAX_MESSAGE_SIZE))
    assert exc.direction == "inbound"
    assert exc.limit == LEGACY_WS_MAX_MESSAGE_SIZE
    assert exc.close_code == 1009
    assert cause is exc
    assert elapsed < 10, f"waited {elapsed:.1f}s for a reply that was refused"


def test_ask_fails_with_message_too_large_instead_of_timing_out():
    """The #1279 shape: the daemon's user echo of a large input exceeds the
    client's ceiling.  The facade learns it from the event stream."""
    server = _server()

    def echo_then_finish(frame: dict) -> list:
        return [
            AgentOutputEvent(agent_id="main", source="user", text="y" * _BIG,
                             mode="write"),
            TurnCompletedEvent(agent_id="main"),
        ]

    _answer_with(server, _send_wire_type(), echo_then_finish)

    async def body(client: WSClient):
        with pytest.raises(MessageTooLarge) as info:
            await Session(client, "s1").ask("a long voice note", timeout=8)
        return info.value

    exc = asyncio.run(_with_client(
        server, body, max_message_size=LEGACY_WS_MAX_MESSAGE_SIZE))
    assert exc.direction == "inbound"


def test_ask_still_completes_when_the_echo_fits():
    """Control: the same echo under the default ceiling is just an echo."""
    server = _server()

    def echo_then_finish(frame: dict) -> list:
        return [
            AgentOutputEvent(agent_id="main", source="user", text="y" * _BIG,
                             mode="write"),
            AgentOutputEvent(agent_id="main", source="model", text="ok",
                             mode="write"),
            TurnCompletedEvent(agent_id="main"),
        ]

    _answer_with(server, _send_wire_type(), echo_then_finish)

    async def body(client: WSClient):
        return await Session(client, "s1").ask("a long voice note", timeout=8)

    assert asyncio.run(_with_client(server, body)) == "ok"


def test_a_message_over_the_advertised_limit_is_refused_before_sending():
    server = _server(max_message_size=LEGACY_WS_MAX_MESSAGE_SIZE)
    seen = []
    _answer_with(server, _send_wire_type(), lambda f: seen.append(f) or [])

    async def body(client: WSClient):
        assert client.server_max_message_size == LEGACY_WS_MAX_MESSAGE_SIZE
        with pytest.raises(MessageTooLarge) as info:
            await client.send_message("hi", attachments=[{
                "mime_type": "audio/wav", "data": b"\x00" * _BIG,
                "display_name": "note.wav"}])
        # The connection survived: a small message still goes through.
        await client.send_message("small")
        for _ in range(200):
            if seen:
                break
            await asyncio.sleep(0.01)
        return info.value, client.is_connected

    exc, connected = asyncio.run(_with_client(server, body))
    assert exc.direction == "outbound"
    assert exc.limit == LEGACY_WS_MAX_MESSAGE_SIZE
    assert exc.size and exc.size > LEGACY_WS_MAX_MESSAGE_SIZE
    assert connected
    assert [f.get("text") for f in seen] == ["small"]


def test_the_recovery_client_builds_every_inner_client_with_the_ceiling():
    rc = WSRecoveryClient("ws://unused", max_message_size=3 * 1024 * 1024)
    inner = rc._make_client(auto_start=False)
    assert inner.max_message_size == 3 * 1024 * 1024


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "16M"])
def test_a_malformed_ceiling_is_refused_at_construction(bad):
    with pytest.raises(ValueError):
        WSClient("ws://unused", max_message_size=bad)
    with pytest.raises(ValueError):
        WSRecoveryClient("ws://unused", max_message_size=bad)


def test_the_daemon_log_names_the_client_when_the_client_refused(caplog):
    server = _server()
    _answer_with(server, _pool_wire_type(), _big_pool_answer)

    async def body(client: WSClient):
        with pytest.raises(MessageTooLarge):
            await client.pool_status(timeout=20)
        for _ in range(200):
            if not server._clients:
                break
            await asyncio.sleep(0.01)

    with caplog.at_level(logging.DEBUG, logger="jaato_server.server.websocket"):
        asyncio.run(_with_client(
            server, body, max_message_size=LEGACY_WS_MAX_MESSAGE_SIZE))
    lines = [r.getMessage() for r in caplog.records
             if r.name == "jaato_server.server.websocket"
             and r.levelno == logging.WARNING and "1009" in r.getMessage()]
    assert lines, [r.getMessage() for r in caplog.records]
    assert "received close 1009" in lines[0], lines[0]
    assert "CLIENT refused" in lines[0], lines[0]
    assert "--ws-max-message-size" not in lines[0], lines[0]


def test_a_recovery_client_subscriber_sees_the_refusal_not_a_timeout():
    """The #1586 driver: a WSRecoveryClient whose subscriber awaits a reply.

    The recovery client reconnects after the 1009 ("Reconnection
    successful!"), so the waiter used to see nothing at all.  It now gets
    the client's ``MessageTooLarge`` event, and ``close_cause`` survives the
    reconnect that replaced the inner client.
    """
    from jaato_sdk.client.config import RecoveryConfig
    from jaato_sdk.events import EventType

    server = _server()
    _answer_with(server, _pool_wire_type(), _big_pool_answer)

    async def run():
        task, port = await _started(server)
        rc = WSRecoveryClient(
            f"ws://127.0.0.1:{port}",
            max_message_size=LEGACY_WS_MAX_MESSAGE_SIZE,
            config=RecoveryConfig(base_delay=0.1, max_delay=0.2))
        try:
            await rc.connect(timeout=10)
            seen: asyncio.Future = asyncio.get_running_loop().create_future()
            # The standalone daemon also answers the handshake's own verbs
            # with unrelated ErrorEvents; only the client's own one counts.
            rc.subscribe(EventType.ERROR, lambda ev: (
                ev.error_type == "MessageTooLarge" and not seen.done()
                and seen.set_result(ev)))
            with pytest.raises(MessageTooLarge):
                await rc.pool_status(timeout=20)
            ev = await asyncio.wait_for(seen, timeout=10)
            return ev, rc.close_cause
        finally:
            await rc.close()
            await _stop(server, task)

    ev, cause = asyncio.run(run())
    assert ev.error_type == "MessageTooLarge"
    assert ev.details["direction"] == "inbound"
    assert isinstance(cause, MessageTooLarge)
