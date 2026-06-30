"""Tests for :class:`jaato_sdk.client.ws.WSClient` — the WebSocket transport.

A fake in-memory request/response WebSocket stands in for a remote daemon, so no
network / server is needed (CI-safe). These prove:

* the convenience facade (``Session.ask``) rides UNCHANGED on ``WSClient`` over
  the WS transport (same command/event protocol as IPC);
* the transport seam — bearer token appended to the Upgrade URL, ``recv`` /
  ``send`` frame handling (no length prefix), and ``disconnect`` closing the WS.

The real end-to-end fidelity against a live ``--web-socket`` daemon is the
example suite's job; here we isolate the transport + facade wiring.
"""

import asyncio

from jaato_sdk import ClientType, WSClient, WSRecoveryClient
import jaato_sdk.client.ws as ws_mod
from jaato_sdk.events import (
    AgentOutputEvent,
    CommandRequest,
    ConnectedEvent,
    SendMessageRequest,
    SessionInfoEvent,
    TurnCompletedEvent,
    deserialize_event,
    serialize_event,
)


class _FakeWS:
    """In-memory request/response WebSocket simulating a remote daemon.

    ``recv`` yields queued frames; ``send`` records the frame and, for the two
    commands the facade drives, queues the server's response (the daemon's
    command-router behavior, simplified). The server greets with a
    ``ConnectedEvent`` unprompted on open, exactly like ``websocket.py``.
    """

    def __init__(self) -> None:
        self._outbox: "asyncio.Queue[str]" = asyncio.Queue()
        self.sent: list[str] = []
        self.closed = False
        self._outbox.put_nowait(
            serialize_event(
                ConnectedEvent(
                    protocol_version="1.0",
                    server_info={
                        "client_id": "ws-client-1",
                        "server_version": "9.9.9",
                    },
                )
            )
        )

    async def recv(self) -> str:
        # Blocks until a frame is available; cancelled by disconnect's drain
        # cancellation (CancelledError), matching the real transport.
        return await self._outbox.get()

    async def send(self, msg: str) -> None:
        self.sent.append(msg)
        ev = deserialize_event(msg)
        if isinstance(ev, CommandRequest) and ev.command == "session.new":
            self._outbox.put_nowait(
                serialize_event(SessionInfoEvent(session_id="ws-sess-1"))
            )
        elif isinstance(ev, SendMessageRequest):
            self._outbox.put_nowait(
                serialize_event(AgentOutputEvent(source="model", text="Hi from WS"))
            )
            self._outbox.put_nowait(serialize_event(TurnCompletedEvent()))
        # set_workspace / client-config commands: no response (fire-and-forget).

    async def close(self) -> None:
        self.closed = True


class _FakeWSModule:
    """Stand-in for the ``websockets`` module — records the connect URL."""

    def __init__(self) -> None:
        self.last_url: str | None = None
        self.last_ws: _FakeWS | None = None

    async def connect(self, url: str, *a, **k) -> _FakeWS:
        self.last_url = url
        self.last_ws = _FakeWS()
        return self.last_ws


def test_facade_ask_rides_on_ws(monkeypatch):
    fake_mod = _FakeWSModule()
    monkeypatch.setattr(ws_mod, "_ws_module", lambda: fake_mod)

    async def _run():
        async with WSClient.session(
            "ws://fake:8080", token="secret", client_type=ClientType.API
        ) as s:
            answer = await s.ask("hello")
        assert answer == "Hi from WS"

        # bearer token appended as a query parameter on the Upgrade URL
        assert "token=secret" in fake_mod.last_url

        # the facade drove session.new + message.send over the WS
        cmds = [deserialize_event(m) for m in fake_mod.last_ws.sent]
        assert any(getattr(c, "command", None) == "session.new" for c in cmds)
        assert any(isinstance(c, SendMessageRequest) for c in cmds)

        # WS closed on context exit
        assert fake_mod.last_ws.closed

    asyncio.run(_run())


def test_token_omitted_when_absent(monkeypatch):
    fake_mod = _FakeWSModule()
    monkeypatch.setattr(ws_mod, "_ws_module", lambda: fake_mod)

    async def _run():
        async with WSClient.session("ws://fake:8080", client_type=ClientType.API):
            pass
        assert "token=" not in fake_mod.last_url
        assert fake_mod.last_url == "ws://fake:8080"

    asyncio.run(_run())


def test_token_appended_with_ampersand_when_url_has_query(monkeypatch):
    fake_mod = _FakeWSModule()
    monkeypatch.setattr(ws_mod, "_ws_module", lambda: fake_mod)

    async def _run():
        async with WSClient.session(
            "ws://fake:8080/?v=1", token="t", client_type=ClientType.API
        ):
            pass
        assert fake_mod.last_url == "ws://fake:8080/?v=1&token=t"

    asyncio.run(_run())


def test_read_write_message_transport_seam():
    """_read_message / _write_message ride directly on the WS (no length
    prefix). bytes frames decode to str; str frames pass through."""

    class _Echo:
        def __init__(self):
            self.sent = []
            self._frames = ["plain-str", b"bytes-frame"]

        async def recv(self):
            return self._frames.pop(0)

        async def send(self, msg):
            self.sent.append(msg)

    async def _run():
        c = WSClient("ws://x", client_type=ClientType.API)
        c._ws = _Echo()
        assert await c._read_message() == "plain-str"
        assert await c._read_message() == "bytes-frame"  # decoded from bytes
        await c._write_message("outgoing")
        assert c._ws.sent == ["outgoing"]

    asyncio.run(_run())


def test_read_message_returns_none_on_recv_error():
    """A closed/broken WS -> _read_message returns None (drain exits cleanly)."""

    class _Broken:
        async def recv(self):
            raise ConnectionError("ws gone")

    async def _run():
        c = WSClient("ws://x", client_type=ClientType.API)
        c._ws = _Broken()
        assert await c._read_message() is None

    asyncio.run(_run())


# --- WSRecoveryClient: IPCRecoveryClient machinery over the WS transport ---

def test_ws_recovery_make_client_builds_wsclient():
    """WSRecoveryClient reuses the IPCRecoveryClient recovery machinery, but its
    `_make_client` factory builds a `WSClient` (url/token) — the only transport
    seam. `auto_start` is meaningless for WS and is ignored."""
    from jaato_sdk import IPCRecoveryClient

    c = WSRecoveryClient("wss://host:8080", token="secret", config_root="/cfg")
    assert isinstance(c, IPCRecoveryClient)
    inner = c._make_client(auto_start=True)  # auto_start ignored for WS
    assert isinstance(inner, WSClient)
    assert inner._url == "wss://host:8080"
    assert inner._token == "secret"
    assert inner.config_root == "/cfg"


def test_facade_ask_rides_on_ws_recovery(monkeypatch):
    """The convenience facade (`Session.ask`) rides on `WSRecoveryClient` over
    the WS transport exactly as on `WSClient` — now with auto-reconnect
    underneath (the recovery state machine + event pump are inherited; only the
    transport differs)."""
    fake_mod = _FakeWSModule()
    monkeypatch.setattr(ws_mod, "_ws_module", lambda: fake_mod)

    async def _run():
        async with WSRecoveryClient.session(
            "ws://fake:8080", token="secret", client_type=ClientType.API
        ) as s:
            answer = await s.ask("hello")
        assert answer == "Hi from WS"
        # bearer token rode the Upgrade URL through the recovery wrapper
        assert "token=secret" in fake_mod.last_url
        # the facade drove session.new + message.send over the WS
        cmds = [deserialize_event(m) for m in fake_mod.last_ws.sent]
        assert any(getattr(c, "command", None) == "session.new" for c in cmds)
        assert any(isinstance(c, SendMessageRequest) for c in cmds)

    asyncio.run(_run())
