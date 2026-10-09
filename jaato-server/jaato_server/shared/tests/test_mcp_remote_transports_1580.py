"""Remote MCP servers: streamable HTTP and SSE in ``.mcp.json`` (#1580).

Before #1580 ``MCPClientManager`` imported ``stdio_client`` and nothing
else, ``ServerConfig`` had no url, and the ``"type"`` key was read by
nobody, so a hosted server (Context7 at ``https://mcp.context7.com/mcp``)
needed a local ``npx mcp-remote`` in front of it.

These tests drive a REAL streamable-HTTP MCP server (the installed SDK's
own, under uvicorn on 127.0.0.1, ``mcp_http_fixture_server.py``) and pin:

* ``list_tools`` + ``call_tool`` over ``type: http``;
* a ``${VAR}`` header is resolved at connect time, reaches the configured
  host, and never the stored config's repr;
* configured headers are removed from a request to another host (a
  redirect), the ``web_fetch`` ``secret_host_bindings`` rule;
* stdio entries are unchanged, an unknown ``type`` is refused (never
  spawned as stdio), a bad url is refused;
* a server that cannot connect leaves the plugin up without its tools,
  reported once at WARNING naming it;
* a server restart (stale ``Mcp-Session-Id``) reconnects once on the next
  call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("uvicorn")

from jaato_server.shared import mcp_remote
from jaato_server.shared.mcp_context_manager import MCPClientManager, ServerConfig
from jaato_server.shared.mcp_remote import MCPConfigError, server_transport
from jaato_server.shared.tests.reversion import Reversion

_REMOTE = "jaato-server/jaato_server/shared/mcp_remote.py"
_MANAGER = "jaato-server/jaato_server/shared/mcp_context_manager.py"
_PLUGIN = "jaato-server/jaato_server/shared/plugins/mcp/plugin.py"

REVERSIONS = [
    Reversion(
        target=_REMOTE,
        find=("    if kind != TRANSPORT_STDIO and kind not in REMOTE_TRANSPORTS:\n"),
        replace="    if False:\n",
        test="test_an_unknown_type_is_refused_not_spawned",
        because="an unknown type would fall through as a transport nobody "
                "opens instead of being refused for that server",
    ),
    Reversion(
        target=_REMOTE,
        find="        if host_matches(request.url.host, bound_host):\n            return\n",
        replace="        return\n",
        test="test_configured_headers_never_follow_a_redirect_to_another_host",
        because="a redirect to another host would carry the secret header",
    ),
    Reversion(
        target=_MANAGER,
        find="        if config.is_remote:\n            return await self._do_connect_remote(config)\n",
        replace="",
        test="test_http_server_lists_and_calls_tools_with_a_resolved_header",
        because="a remote entry would be spawned as a stdio subprocess",
    ),
    Reversion(
        target=_REMOTE,
        find="        if not isinstance(value, str) or _unresolved(value):\n",
        replace="        if not isinstance(value, str):\n",
        test="test_a_header_that_does_not_resolve_is_refused",
        because="an unset ${VAR} would be sent literally as the credential",
    ),
    Reversion(
        target=_MANAGER,
        find=("                if not (connection.session_stale or is_stale_session(exc)\n"
              "                        or connection.credential_refused\n"
              "                        or connection.transport_closed):\n"),
        replace="                if True:\n",
        test="test_a_restarted_server_reconnects_once_on_the_next_call",
        because="a stale Mcp-Session-Id would fail every later call",
    ),
    Reversion(
        target=_PLUGIN,
        find=('                    self._log_event(LOG_WARN, "Connection failed", '
              'server=name, details=error_msg)\n'),
        replace=('                    self._log_event(LOG_DEBUG, "Connection failed", '
                 'server=name, details=error_msg)\n'),
        test="test_a_server_that_cannot_connect_leaves_the_plugin_up",
        because="a failed connect would be silent at the default log level",
    ),
]

_FIXTURE = Path(__file__).with_name("mcp_http_fixture_server.py")
_SECRET = "s3cr3t-token-1580"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Server:
    """A fixture MCP server subprocess that can be restarted on its port."""

    def __init__(self, tmp_path: Path, transport: str = "http"):
        self.port = _free_port()
        self.transport = transport
        self.log = tmp_path / f"headers-{transport}.log"
        path = "sse" if transport == "sse" else "mcp"
        self.url = f"http://127.0.0.1:{self.port}/{path}"
        self.proc = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, str(_FIXTURE), str(self.port), str(self.log),
             self.transport],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", self.port), 0.2).close()
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("fixture MCP server did not start")

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait(10)
            self.proc = None

    def authorizations(self) -> list:
        if not self.log.exists():
            return []
        return self.log.read_text().splitlines()


@pytest.fixture
def server(tmp_path):
    srv = _Server(tmp_path)
    srv.start()
    yield srv
    srv.stop()


def _run(coro):
    return asyncio.run(coro)


async def _connect_and(url, headers, body):
    async with MCPClientManager() as mgr:
        await mgr.connect_remote("remote", "http", url, headers)
        return await body(mgr)


def test_http_server_lists_and_calls_tools_with_a_resolved_header(server, monkeypatch):
    monkeypatch.setenv("TEST_MCP_KEY_1580", _SECRET)
    headers = {"Authorization": "Bearer ${TEST_MCP_KEY_1580}"}

    async def body(mgr):
        conn = mgr.get_connection("remote")
        names = [t.name for t in conn.tools]
        result = await mgr.call_tool("remote", "echo", {"text": "hi"})
        return names, result, repr(conn.config)

    names, result, config_repr = _run(_connect_and(server.url, headers, body))
    assert names == ["echo"]
    assert "echo: hi" in result.content[0].text
    assert f"Bearer {_SECRET}" in server.authorizations()
    assert _SECRET not in config_repr  # the template is stored, not the value


def test_sse_server_lists_and_calls_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MCP_KEY_1580", _SECRET)
    srv = _Server(tmp_path, "sse")
    srv.start()

    async def go():
        async with MCPClientManager() as mgr:
            await mgr.connect_remote(
                "remote", "sse", srv.url,
                {"Authorization": "Bearer ${TEST_MCP_KEY_1580}"})
            return await mgr.call_tool("remote", "echo", {"text": "sse"})

    try:
        result = _run(go())
    finally:
        srv.stop()
    assert "echo: sse" in result.content[0].text
    assert f"Bearer {_SECRET}" in srv.authorizations()


def test_a_header_that_does_not_resolve_is_refused(monkeypatch):
    monkeypatch.delenv("UNSET_1580", raising=False)
    with pytest.raises(MCPConfigError) as err:
        mcp_remote.resolve_headers("remote", {"X-Key": "${UNSET_1580}"})
    assert "X-Key" in str(err.value)
    with pytest.raises(MCPConfigError):
        mcp_remote.resolve_headers("remote", {"X-Key": "nosuchscheme://a/b"})
    assert mcp_remote.resolve_headers("remote", {"A": "plain"}) == {"A": "plain"}


def test_configured_headers_never_follow_a_redirect_to_another_host():
    import httpx

    seen = []

    def handler(request):
        seen.append((request.url.host, request.headers.get("X-Key")))
        if request.url.host == "api.example.com":
            return httpx.Response(302, headers={"location": "https://evil.example.net/x"})
        return httpx.Response(200, text="ok")

    def sdk_factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(headers=headers, transport=httpx.MockTransport(handler),
                                 follow_redirects=True)

    factory = mcp_remote.bound_client_factory(
        sdk_factory, ["X-Key"], "https://api.example.com/mcp")

    async def go():
        async with factory(headers={"X-Key": _SECRET}) as client:
            await client.get("https://api.example.com/mcp")

    _run(go())
    assert seen == [("api.example.com", _SECRET), ("evil.example.net", None)]


def test_stdio_entries_are_unchanged():
    assert server_transport("s", {"command": "x"}) == "stdio"
    assert server_transport("s", {"type": "stdio", "command": "x"}) == "stdio"
    cfg = ServerConfig(name="s", command="srv", args=["a"], env={"K": "v"})
    assert not cfg.is_remote
    params = cfg.to_stdio_params()
    assert (params.command, params.args, params.env["K"]) == ("srv", ["a"], "v")


def test_an_unknown_type_is_refused_not_spawned():
    with pytest.raises(MCPConfigError) as err:
        server_transport("s", {"type": "websocket", "url": "wss://x"})
    assert "websocket" in str(err.value)
    with pytest.raises(MCPConfigError):
        server_transport("s", {"url": "https://x/mcp"})  # url, no type
    assert server_transport("s", {"type": "streamable-http", "url": "https://x"}) == "http"
    assert server_transport("s", {"type": "SSE", "url": "https://x"}) == "sse"


@pytest.mark.parametrize("spec", [
    {"type": "http"},
    {"type": "http", "url": "ftp://x/mcp"},
    {"type": "sse", "url": "https://u:p@x/mcp"},
    {"type": "http", "url": "https://x/mcp?k=${KEY}"},
    {"type": "http", "url": "https://x/mcp", "headers": {"Bad Name": "v"}},
])
def test_a_bad_remote_entry_is_refused(spec):
    with pytest.raises(MCPConfigError):
        mcp_remote.remote_endpoint("s", spec)


def test_a_restarted_server_reconnects_once_on_the_next_call(server):
    async def body(mgr):
        first = await mgr.call_tool("remote", "echo", {"text": "one"})
        server.stop()
        server.start()  # same port, new process: our session id is stale
        second = await mgr.call_tool("remote", "echo", {"text": "two"})
        return first, second

    first, second = _run(_connect_and(server.url, {}, body))
    assert "echo: one" in first.content[0].text
    assert "echo: two" in second.content[0].text


def test_a_server_that_cannot_connect_leaves_the_plugin_up(server, tmp_path, caplog):
    from jaato_server.shared.plugins.mcp.plugin import MCPToolPlugin

    dead = f"http://127.0.0.1:{_free_port()}/mcp"
    cfg = tmp_path / ".mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "good": {"type": "http", "url": server.url},
        "dead": {"type": "http", "url": dead},
        "odd": {"type": "carrier-pigeon", "url": dead},
    }}))
    plugin = MCPToolPlugin()
    caplog.set_level(logging.WARNING, logger="jaato_server.shared.plugins.mcp.plugin")
    try:
        plugin.initialize({"config_path": str(cfg), "scrub_secret_env": "default"})
        deadline = time.time() + 20
        while time.time() < deadline and len(plugin._failed_servers) < 2:
            time.sleep(0.1)
        assert plugin._connected_servers == {"good"}
        assert set(plugin._failed_servers) == {"dead", "odd"}
        names = [s.name for s in plugin.get_tool_schemas()]
        assert any("echo" in n for n in names)
        warned = [r for r in caplog.records if r.levelno == logging.WARNING]
        for bad in ("dead", "odd"):
            assert sum(f"[MCP:{bad}]" in r.getMessage() for r in warned) == 1
    finally:
        plugin.shutdown()
