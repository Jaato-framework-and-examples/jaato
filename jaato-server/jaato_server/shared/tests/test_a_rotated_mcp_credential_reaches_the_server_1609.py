"""A rotated credential reaches an open remote MCP connection (#1609).

A remote MCP server (``.mcp.json`` ``"type": "http"`` / ``"sse"``) is sent
headers resolved ONCE, when its connection opens.  An application that
authenticates its users in an identity provider puts the user's bearer token
in the session environment (``"Authorization": "Bearer ${MCP_TOKEN}"``), and
that token expires (Keycloak's default: 5 minutes).  Before #1609:

* a **401** from the server was final -- only a 404 / dead transport
  reconnected, though a reconnect is exactly what re-resolves the header;
  on SSE the refused POST killed the transport and the call never answered;
* ``session.reload_env`` re-applied the environment and rebuilt the provider
  and left every MCP connection on the old token.

Pinned here, against a REAL MCP server (``mcp_http_fixture_server.py`` with a
token file, so the test rotates what the server accepts):

* a 401 reconnects once with headers re-resolved and the call succeeds, on
  streamable HTTP and on SSE;
* a server that keeps refusing fails the call, bounded, instead of looping;
* ``session.reload_env`` reopens the remote servers whose header templates
  read a changed variable, and leaves the others alone.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("uvicorn")

from jaato_server.shared.mcp_context_manager import (
    MCPClientManager,
    MCPServerUnavailableError,
)
from jaato_server.shared.mcp_remote import header_env_names
from jaato_server.shared.tests.reversion import Reversion

_MANAGER = "jaato-server/jaato_server/shared/mcp_context_manager.py"
_REMOTE = "jaato-server/jaato_server/shared/mcp_remote.py"
_PLUGIN = "jaato-server/jaato_server/shared/plugins/mcp/plugin.py"
_RPC = "jaato-server/jaato_server/server/runner/rpc.py"

REVERSIONS = [
    Reversion(
        target=_MANAGER,
        find="                        or connection.credential_refused\n",
        replace="",
        test="test_a_401_reconnects_once_with_the_header_re_resolved[http]",
        because="a 401 would be final although a reconnect re-resolves the "
                "header",
    ),
    Reversion(
        target=_MANAGER,
        find="        if watch is None:\n            return await call\n",
        replace="        if True:\n            return await call\n",
        test="test_a_401_reconnects_once_with_the_header_re_resolved[sse]",
        because="on SSE the refused POST ends the transport and the call "
                "would never answer",
    ),
    Reversion(
        target=_REMOTE,
        find="        elif response.status_code == 401:\n",
        replace="        elif False:\n",
        test="test_a_401_reconnects_once_with_the_header_re_resolved[sse]",
        because="nothing would record that the server refused the credential",
    ),
    Reversion(
        target=_PLUGIN,
        find="            if header_env_names(config.headers or {}) & changed:\n",
        replace="            if True:\n",
        test="test_reload_env_reopens_only_the_servers_whose_headers_changed",
        because="a reload would reconnect servers whose credential did not "
                "change",
    ),
    Reversion(
        target=_RPC,
        find="            changed_env_names(environ_before, dict(os.environ)),\n",
        replace="            [],\n",
        test="test_reload_env_reopens_only_the_servers_whose_headers_changed",
        because="session.reload_env would leave MCP connections on the old "
                "token",
    ),
]

_FIXTURE = Path(__file__).with_name("mcp_http_fixture_server.py")
_VAR = "TEST_MCP_TOKEN_1609"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Server:
    """A fixture MCP server that accepts exactly the token in a file."""

    def __init__(self, tmp_path: Path, transport: str, tag: str,
                 token: str | None):
        self.port = _free_port()
        self.log = tmp_path / f"headers-{tag}.log"
        self.token_file = None
        if token is not None:
            self.token_file = tmp_path / f"token-{tag}"
            self.accept(token)
        path = "sse" if transport == "sse" else "mcp"
        self.url = f"http://127.0.0.1:{self.port}/{path}"
        argv = [sys.executable, str(_FIXTURE), str(self.port), str(self.log),
                transport]
        if self.token_file is not None:
            argv.append(str(self.token_file))
        self.proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", self.port), 0.2).close()
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("fixture MCP server did not start")

    def accept(self, token: str) -> None:
        self.token_file.write_text(f"Bearer {token}")

    def stop(self) -> None:
        self.proc.terminate()
        self.proc.wait(10)

    def authorizations(self) -> list:
        return self.log.read_text().splitlines() if self.log.exists() else []


@pytest.fixture
def servers(tmp_path):
    started = []

    def start(transport="http", tag="a", token="old"):
        srv = _Server(tmp_path, transport, tag, token)
        started.append(srv)
        return srv

    yield start
    for srv in started:
        srv.stop()


def _call(mgr, text):
    # Bounded: before #1609 an SSE call refused with 401 never answered.
    return asyncio.wait_for(mgr.call_tool("remote", "echo", {"text": text}), 20)


@pytest.mark.parametrize("transport", ["http", "sse"])
def test_a_401_reconnects_once_with_the_header_re_resolved(
        servers, transport, monkeypatch):
    srv = servers(transport)
    monkeypatch.setenv(_VAR, "old")

    async def body():
        async with MCPClientManager() as mgr:
            await mgr.connect_remote("remote", transport, srv.url,
                                     {"Authorization": "Bearer ${%s}" % _VAR})
            first = await _call(mgr, "one")
            # The identity provider rotated the token; the application wrote
            # the new one into the session env.
            srv.accept("new")
            os.environ[_VAR] = "new"
            second = await _call(mgr, "two")
            return first, second

    first, second = asyncio.run(body())
    assert "echo: one" in first.content[0].text
    assert "echo: two" in second.content[0].text
    seen = srv.authorizations()
    assert "Bearer old" in seen and "Bearer new" in seen


def test_a_server_that_keeps_refusing_fails_the_call(servers, monkeypatch):
    srv = servers("http")
    monkeypatch.setenv(_VAR, "old")

    async def body():
        async with MCPClientManager() as mgr:
            await mgr.connect_remote("remote", "http", srv.url,
                                     {"Authorization": "Bearer ${%s}" % _VAR})
            await _call(mgr, "one")
            srv.accept("something-we-never-get")
            with pytest.raises(MCPServerUnavailableError):
                await _call(mgr, "two")
            # failed fast afterwards, no further attempt
            with pytest.raises(MCPServerUnavailableError):
                await _call(mgr, "three")

    asyncio.run(body())
    refused = [a for a in srv.authorizations() if a == "Bearer old"]
    # one connect + one call before the rotation, then the refused call and
    # ONE reconnect attempt: bounded, not a loop.
    assert len(refused) < 20


def test_header_env_names_reads_the_variables_a_template_names():
    assert header_env_names({
        "Authorization": "Bearer ${TOKEN}",
        "X-Tenant": "${ TENANT }-${REGION}",
        "X-Fixed": "constant",
        "X-Secret": "pass://mcp/token",
    }) == {"TOKEN", "TENANT", "REGION"}


class _Registry:
    def __init__(self, plugins):
        self._plugins = plugins

    def list_available(self):
        return list(self._plugins)

    def get_plugin(self, name):
        return self._plugins[name]


class _Session:
    is_running = False

    def __init__(self, registry):
        self._runtime = type("R", (), {"registry": registry})()
        self._session_env = {}

    def reload_provider(self):
        return {"provider": "echo", "model": "m", "auth_info": "none"}


def test_reload_env_reopens_only_the_servers_whose_headers_changed(
        servers, tmp_path, monkeypatch):
    from jaato_server.server.runner import session as runner_session
    from jaato_server.server.runner.rpc import RunnerRPC
    from jaato_server.server.runner.session import RunnerSessionHost
    from jaato_server.shared.plugins.mcp.plugin import MCPToolPlugin
    from jaato_server.shared.session_envelope import SessionInitEnvelope

    rotating = servers("http", "rotating", token="old")
    steady = servers("http", "steady", token=None)
    cfg = tmp_path / ".mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "rotating": {"type": "http", "url": rotating.url,
                     "headers": {"Authorization": "Bearer ${%s}" % _VAR}},
        "steady": {"type": "http", "url": steady.url,
                   "headers": {"X-Fixed": "constant"}},
    }}))
    monkeypatch.setattr(runner_session, "_PRISTINE_ENVIRON", None)
    monkeypatch.delenv(_VAR, raising=False)
    runner_session.apply_session_env({_VAR: "old"})

    plugin = MCPToolPlugin()
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()
    try:
        plugin.initialize({"config_path": str(cfg), "scrub_secret_env": "default"})
        deadline = time.time() + 20
        while time.time() < deadline and len(plugin._connected_servers) < 2:
            time.sleep(0.1)
        assert plugin._connected_servers == {"rotating", "steady"}
        steady_requests = len(steady.authorizations())

        rpc = RunnerRPC(a, lambda name, args: (False, {"error": "none"}))
        rpc._session_host = RunnerSessionHost(
            envelope=SessionInitEnvelope(
                session_id="s-1609", workspace_path=str(tmp_path),
                profile_name="p", provider_name="echo", model_name="m",
                plugins=["mcp"],
            ),
            runtime=None, session=_Session(_Registry({"mcp": plugin})),
        )
        rotating.accept("new")
        ok, answer = rpc._handle_session_reload_env(
            {"session_env": {_VAR: "new"}})
        assert ok is True
        assert answer["refreshed"] == {"mcp": ["rotating"]}

        # The reopened connection carries the new token before any call.
        deadline = time.time() + 20
        while time.time() < deadline and "Bearer new" not in rotating.authorizations():
            time.sleep(0.1)
        assert "Bearer new" in rotating.authorizations()
        time.sleep(0.5)
        assert len(steady.authorizations()) == steady_requests
        assert plugin._failed_servers == {}
    finally:
        plugin.shutdown()
        monkeypatch.setattr(runner_session, "_PRISTINE_ENVIRON", None)
