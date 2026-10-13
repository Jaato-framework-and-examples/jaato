"""An MCP tool call has a ceiling an author can set, and abandoning a call
cancels it on the server (#1646).

Before: every call waited a literal 30 s with no knob, a call abandoned on
that ceiling (or on the session's cancel token) kept running on the server
for nobody, and every call shared ONE response queue on a thread that ran
calls one at a time -- so a slow call held every other call, and its late
answer was read by whichever call asked next.

Pinned against a REAL MCP server (``mcp_http_fixture_server.py``, whose
``slow`` tool records ``start`` / ``done`` / ``cancelled``), through
``MCPToolPlugin._execute``:

* the ceiling resolves ``.mcp.json`` ``"timeout"`` > ``plugin_configs.mcp.
  call_timeout`` > 30 s, and the timeout error names the knob;
* a call abandoned on its ceiling, or on the session's cancel token, is
  cancelled on the server (``notifications/cancelled``);
* calls run concurrently, each answering on its own queue.

mcp 2.x sends ``notifications/cancelled`` itself when the request's task is
cancelled; mcp 1.x does not, so ``ServerConnection.call_tool`` sends it.
That half is pinned against a session shaped like 1.x's.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("uvicorn")

from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/mcp/plugin.py"
_TIMEOUT = "jaato-server/jaato_server/shared/plugins/mcp/call_timeout.py"
_MANAGER = "jaato-server/jaato_server/shared/mcp_context_manager.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find=("                    self._abandon_call(\n"
              "                        call_id, f\"timed out after {timeout:g}s\")\n"),
        replace="",
        test="test_a_call_past_its_ceiling_is_cancelled_on_the_server[http]",
        because="the server would keep working on an answer nobody reads",
    ),
    Reversion(
        target=_PLUGIN,
        find="                self._abandon_call(call_id, \"cancelled by the client\")\n",
        replace="",
        test="test_a_session_cancel_cancels_the_call_on_the_server",
        because="a stopped session would leave its MCP call running",
    ),
    Reversion(
        target=_PLUGIN,
        find="                            self._start_call_task(manager, msg_type, data)\n",
        replace="                            await self._run_call(manager, msg_type, data)\n",
        test="test_a_slow_call_does_not_hold_another",
        because="calls would run one at a time again",
    ),
    Reversion(
        target=_TIMEOUT,
        find="        (\"server\", (server_spec or {}).get(SERVER_KEY)\n",
        replace="        (\"server\", None\n",
        test="test_the_server_entry_outranks_the_plugin_config",
        because="the .mcp.json \"timeout\" would be read by nothing",
    ),
    Reversion(
        target=_MANAGER,
        find=("            if request_id is not None:\n"
              "                await _send_cancelled(\n"),
        replace=("            if False:\n"
                 "                await _send_cancelled(\n"),
        test="test_mcp_1x_is_told_the_call_was_cancelled",
        because="on mcp 1.x the server is never told",
    ),
]

_FIXTURE = Path(__file__).with_name("mcp_http_fixture_server.py")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Server:
    def __init__(self, tmp_path: Path, transport: str):
        self.port = _free_port()
        log = tmp_path / f"headers-{transport}.log"
        self.events = Path(str(log) + ".events")
        path = "sse" if transport == "sse" else "mcp"
        self.url = f"http://127.0.0.1:{self.port}/{path}"
        self.proc = subprocess.Popen(
            [sys.executable, str(_FIXTURE), str(self.port), str(log),
             transport],
            env={**os.environ, "MCP_FIXTURE_SLOW_TOOL": "1"},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", self.port), 0.2).close()
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("fixture MCP server did not start")

    def lines(self) -> list:
        return self.events.read_text().split("\n") if self.events.exists() else []

    def wait_for(self, line: str, seconds: float = 10.0) -> bool:
        deadline = time.time() + seconds
        while time.time() < deadline:
            if line in self.lines():
                return True
            time.sleep(0.05)
        return False

    def stop(self) -> None:
        self.proc.terminate()
        self.proc.wait(10)


def _plugin(tmp_path, server: _Server, *, entry=None, plugin_cfg=None):
    from jaato_server.shared.plugins.mcp.plugin import MCPToolPlugin
    spec = {"type": "sse" if server.url.endswith("/sse") else "http",
            "url": server.url, **(entry or {})}
    cfg = tmp_path / ".mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"fixture": spec}}))
    plugin = MCPToolPlugin()
    plugin.initialize({"config_path": str(cfg), **(plugin_cfg or {})})
    deadline = time.time() + 20
    while time.time() < deadline and "fixture" not in plugin._connected_servers:
        time.sleep(0.1)
    assert "fixture" in plugin._connected_servers, plugin._failed_servers
    plugin._load_config_cache()
    return plugin


@pytest.fixture
def server(tmp_path, request):
    srv = _Server(tmp_path, getattr(request, "param", "http"))
    yield srv
    srv.stop()


@pytest.mark.parametrize("server", ["http", "sse"], indirect=True)
def test_a_call_past_its_ceiling_is_cancelled_on_the_server(tmp_path, server):
    plugin = _plugin(tmp_path, server, entry={"timeout": 1})
    try:
        started = time.monotonic()
        ok, out = plugin._execute("slow", {"seconds": 4, "tag": "a"})
        assert ok is False
        assert time.monotonic() - started < 3
        assert "timed out after 1s" in out["error"]
        assert "\"timeout\" on the .mcp.json entry for 'fixture'" in out["error"]
        assert server.wait_for("cancelled a"), server.lines()
        time.sleep(4)
        assert "done a" not in server.lines()
    finally:
        plugin.shutdown()


def test_a_session_cancel_cancels_the_call_on_the_server(tmp_path, server):
    from jaato_sdk.plugins.model_provider.types import CancelledException
    from jaato_server.shared import ai_tool_runner
    from jaato_sdk.plugins.model_provider.types import CancelToken
    plugin = _plugin(tmp_path, server)
    token = CancelToken()
    caught = []

    def run():
        ai_tool_runner._thread_local.cancel_token = token
        try:
            plugin._execute("slow", {"seconds": 6, "tag": "b"})
        except CancelledException as exc:
            caught.append(exc)
        finally:
            ai_tool_runner._thread_local.cancel_token = None

    worker = threading.Thread(target=run)
    try:
        worker.start()
        assert server.wait_for("start b")
        token.cancel()
        worker.join(5)
        assert caught, "the session's cancel did not end the call"
        assert server.wait_for("cancelled b", 5), server.lines()
    finally:
        worker.join(10)
        plugin.shutdown()


def test_a_slow_call_does_not_hold_another(tmp_path, server):
    plugin = _plugin(tmp_path, server)
    slow = threading.Thread(
        target=plugin._execute, args=("slow", {"seconds": 3, "tag": "c"}))
    try:
        slow.start()
        assert server.wait_for("start c")
        started = time.monotonic()
        out = plugin._execute("echo", {"text": "quick"})
        assert time.monotonic() - started < 2
        assert out["result"]["content"] == ["echo: quick"]
    finally:
        slow.join(10)
        plugin.shutdown()


def test_the_plugin_config_sets_the_ceiling(tmp_path, server):
    plugin = _plugin(tmp_path, server, plugin_cfg={"call_timeout": 1})
    try:
        ok, out = plugin._execute("slow", {"seconds": 3, "tag": "d"})
        assert ok is False
        assert "timed out after 1s" in out["error"]
        assert "plugin_configs.mcp.call_timeout" in out["error"]
    finally:
        plugin.shutdown()


def test_the_server_entry_outranks_the_plugin_config(tmp_path, server):
    plugin = _plugin(tmp_path, server, entry={"timeout": 0},
                     plugin_cfg={"call_timeout": 1})
    try:
        out = plugin._execute("slow", {"seconds": 2, "tag": "e"})
        assert out.get("result", {}).get("content") == ["slept 2"], out
    finally:
        plugin.shutdown()


def test_the_ceiling_rule():
    from jaato_server.shared.plugins.mcp.call_timeout import (
        DEFAULT_CALL_TIMEOUT, resolve_call_timeout)
    assert resolve_call_timeout({}, None) == (DEFAULT_CALL_TIMEOUT, "default")
    assert resolve_call_timeout({}, 90) == (90.0, "plugin")
    assert resolve_call_timeout({"timeout": 120}, 90) == (120.0, "server")
    assert resolve_call_timeout({"timeout": 0}, 90) == (None, "server")
    # A value that is not seconds is ignored; it never buys "unbounded".
    for bad in ("60", -1, True, float("nan")):
        assert resolve_call_timeout({"timeout": bad}, 45) == (45.0, "plugin")
        assert resolve_call_timeout({}, bad) == (DEFAULT_CALL_TIMEOUT, "default")


class _Session1x:
    """The shape ``_sdk_1x_next_request_id`` reads on mcp 1.x."""

    def __init__(self):
        self._request_id = 7
        self._response_streams = {}
        self.sent = []

    async def call_tool(self, name, arguments, progress_callback=None):
        self._request_id += 1
        await asyncio.sleep(60)

    async def send_notification(self, note):
        self.sent.append(note)


def test_mcp_1x_is_told_the_call_was_cancelled():
    from jaato_server.shared.mcp_context_manager import (
        ServerConfig, ServerConnection)
    session = _Session1x()
    conn = ServerConnection(config=ServerConfig(name="s", command="x"),
                            session=session)

    async def run():
        task = asyncio.ensure_future(conn.call_tool("slow", {}))
        await asyncio.sleep(0.05)
        task.cancel("timed out after 1s")
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert len(session.sent) == 1
    note = session.sent[0]
    params = getattr(note, "root", note).params.model_dump(by_alias=True)
    assert params["requestId"] == 7
    assert params["reason"] == "timed out after 1s"


def test_mcp_2x_sessions_are_left_to_their_sdk():
    from jaato_server.shared.mcp_context_manager import _sdk_1x_next_request_id

    class Session2x:
        pass

    assert _sdk_1x_next_request_id(Session2x()) is None


def test_explain_renders_the_rule_the_plugin_applies():
    from jaato_server.shared.plugins.mcp.call_timeout import CALL_TIMEOUT_RULES
    from jaato_server.shared.scaffold.explain import _mcp_transport_lines
    lines, data = _mcp_transport_lines()
    for rule in CALL_TIMEOUT_RULES:
        assert f"    - {rule}" in lines
    assert data["call_timeout"]["rules"] == list(CALL_TIMEOUT_RULES)
