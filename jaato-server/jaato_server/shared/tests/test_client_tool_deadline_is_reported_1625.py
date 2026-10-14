"""A client can learn, and set, the deadline its host tools are held to (#1625).

The daemon waits for a ``client_tools=`` handler's result, then hands the
model ``{"error": "Client tool <name> timed out after <N>s"}``.  A driver
whose handlers wrap their own deadline (to return a sentence the model can
read) set it to 45 s against a daemon limit of 30 s, and nothing told it:
the limit was in no ``explain clients`` row, and the per-tool ``timeout``
key that changes it was undocumented and untyped.

The fix, and what each case pins:

* one resolver (:mod:`jaato_sdk.client_tool_timeout`) decides the budget:
  the spec's ``timeout`` ms, else 30 s, refusing a value that is not a
  positive number -- and both transports enforce through it;
* the daemon REPORTS the budgets on ``ToolsRegisterClientResultEvent``
  (protocol 1.39) and ``IPCClient.register_client_tools`` returns them;
* ``explain clients``' TIMEOUTS table lists the clock.

The IPC case drives a real ``JaatoIPCServer`` and a real ``IPCClient``; the
daemon's answer builder is replaced with one reporting 7 s, so a 30 s return
can only mean the SDK computed the value itself instead of asking.
"""

from __future__ import annotations

import ast
import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

from jaato_server.shared.tests.reversion import Reversion

_SDK_TO = "jaato-sdk/jaato_sdk/client_tool_timeout.py"
_SERVER_CT = "jaato-server/jaato_server/server/client_tools.py"
_IPC = "jaato-server/jaato_server/server/ipc.py"
_WS = "jaato-server/jaato_server/server/websocket.py"
_INTROSPECT = "jaato-server/jaato_server/shared/scaffold/introspect.py"

_REPO = Path(__file__).resolve().parents[4]


# ------------------------------------------------------------- the resolver


def test_the_resolver_honours_a_valid_timeout_and_defaults_otherwise() -> None:
    from jaato_sdk.client_tool_timeout import (
        DEFAULT_CLIENT_TOOL_TIMEOUT_MS, resolve_client_tool_timeout,
    )

    assert DEFAULT_CLIENT_TOOL_TIMEOUT_MS == 30000  # the default is unchanged
    absent = resolve_client_tool_timeout({"name": "t"})
    assert (absent.timeout_s, absent.source, absent.problem) == (30.0, "default", None)
    given = resolve_client_tool_timeout({"name": "t", "timeout": 60000})
    assert (given.timeout_s, given.source) == (60.0, "client")
    for bad in ("60000", 0, -5, float("nan"), float("inf"), True):
        got = resolve_client_tool_timeout({"name": "t", "timeout": bad})
        assert got.timeout_s == 30.0, bad
        assert got.source == "default" and got.problem, bad


# -------------------------------------------------- what the daemon enforces


def test_the_daemon_executor_waits_the_resolved_budget(monkeypatch: Any) -> None:
    """``register_client_tools`` builds each executor with the resolver's value."""
    from jaato_server.server import client_tools as ct

    seen: dict = {}
    real = ct.make_client_tool_executor

    def _capture(name: str, timeout_s: float, *a: Any, **k: Any) -> Any:
        seen[name] = timeout_s
        return real(name, timeout_s, *a, **k)

    monkeypatch.setattr(ct, "make_client_tool_executor", _capture)

    class _Reg:
        def register_core_tool(self, *a: Any, **k: Any) -> None:
            pass

    server = type("S", (), {"client_tool_schemas": {}})()
    ct.register_client_tools(
        _Reg(), server,
        [{"name": "slow", "timeout": 90000}, {"name": "plain"},
         {"name": "typo", "timeout": "90s"}],
        send_request=lambda e: True, waiters={})
    assert seen == {"slow": 90.0, "plain": 30.0, "typo": 30.0}


def test_both_transports_resolve_the_budget_through_one_function() -> None:
    """No transport reads ``timeout`` itself: the WS path inlines its executor."""
    for rel in (_SERVER_CT, _WS):
        tree = ast.parse((_REPO / rel).read_text())
        calls = {n.func.id if isinstance(n.func, ast.Name) else
                 getattr(n.func, "attr", "")
                 for n in ast.walk(tree) if isinstance(n, ast.Call)}
        assert "client_tool_timeout_seconds" in calls, rel
        literals = [n.value for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and n.value == 30000]
        assert not literals, f"{rel} restates the default"


def test_the_timeout_message_names_the_budget() -> None:
    from jaato_server.server.client_tools import make_client_tool_executor

    ex = make_client_tool_executor("t", 0.05, lambda e: True, {})
    started = time.monotonic()
    assert ex({}) == {"error": "Client tool t timed out after 0.05s"}
    assert time.monotonic() - started < 2


# ----------------------------------------------- what the client is told


class _NoSessionManager:
    """No session exists, so the tools are BUFFERED (the registered-first path)."""

    def get_session(self, session_id: str) -> Any:
        return None

    def __getattr__(self, name: str) -> Callable[..., None]:
        return lambda *a, **k: None


def _drive(body: Callable[[Any], Any]) -> None:
    from jaato_sdk.client.ipc import IPCClient
    from jaato_sdk.events import ClientType
    from jaato_server.server.ipc import JaatoIPCServer

    tmp = tempfile.mkdtemp(prefix="jaato1625-", dir="/tmp")
    sock = os.path.join(tmp, "d.sock")

    async def main() -> None:
        server = JaatoIPCServer(socket_path=sock)
        server._session_manager = _NoSessionManager()
        task = asyncio.create_task(server.start())
        try:
            deadline = time.monotonic() + 5
            while not os.path.exists(sock):
                assert time.monotonic() < deadline, "IPC server did not bind"
                await asyncio.sleep(0.02)
            client = IPCClient(sock, auto_start=False, client_type=ClientType.API)
            assert await client.connect(timeout=5)
            try:
                await body(client)
            finally:
                await client.disconnect()
        finally:
            await server.stop()
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    try:
        asyncio.run(main())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_register_client_tools_returns_the_daemons_reported_budget(
        monkeypatch: Any) -> None:
    from jaato_server.server import client_tools as ct

    real = ct.registration_answer

    def _seven(tools: Any, request_id: str, *, buffered: bool) -> Any:
        ev = real(tools, request_id, buffered=buffered)
        ev.tools = [dict(r, timeout_ms=7000, timeout_s=7.0) for r in ev.tools]
        return ev

    monkeypatch.setattr(ct, "registration_answer", _seven)
    got: dict = {}

    async def body(client: Any) -> None:
        got["ret"] = await client.register_client_tools(
            [{"name": "fetch", "description": "d", "parameters": {},
              "handler": lambda a: "ok"}], timeout=3.0)
        got["attr"] = dict(client.client_tool_timeouts)

    _drive(body)
    assert got["ret"] == {"fetch": 7.0}, got
    assert got["attr"] == {"fetch": 7.0}


def test_an_older_client_is_sent_no_answer() -> None:
    """A request with no ``request_id`` (a pre-1.39 SDK) gets nothing new."""
    from jaato_sdk.events import EventType, ToolsRegisterClientRequest

    seen: list = []

    async def body(client: Any) -> None:
        client.subscribe(EventType.TOOLS_REGISTER_CLIENT_RESULT, seen.append)
        await client._send_event(ToolsRegisterClientRequest(
            tools=[{"name": "fetch"}]))
        await asyncio.sleep(0.5)

    _drive(body)
    assert seen == []


# ----------------------------------------------------- what introspection says


def test_explain_clients_lists_the_host_tool_clock() -> None:
    from jaato_server.shared.scaffold import introspect

    rows = [t for t in introspect.client_timeouts()
            if t.where.startswith("client_tools=")]
    assert len(rows) == 1, introspect.client_timeouts()
    assert rows[0].default == 30.0
    assert "timed out after" in rows[0].on_expiry


REVERSIONS = [
    Reversion(
        target=_SDK_TO,
        find="    if isinstance(raw, bool) or not isinstance(raw, (int, float)):",
        replace="    if not isinstance(raw, (int, float)):",
        test="test_the_resolver_honours_a_valid_timeout_and_defaults_otherwise",
        because="timeout: True is honoured as a one-millisecond deadline",
    ),
    Reversion(
        target=_SERVER_CT,
        find="        timeout_s = client_tool_timeout_seconds(tool_def)\n",
        replace="        timeout_s = 30.0\n",
        test="test_the_daemon_executor_waits_the_resolved_budget",
        because="the IPC transport ignores the client's declared deadline",
    ),
    Reversion(
        target=_WS,
        find="            timeout = client_tool_timeout_seconds(tool_def)\n",
        replace="            timeout = tool_def.get('timeout', 30000) / 1000.0\n",
        test="test_both_transports_resolve_the_budget_through_one_function",
        because="the WS transport keeps its own reading of the deadline",
    ),
    Reversion(
        target=_IPC,
        find=("            await self._answer_client_tool_registration(\n"
              "                client_id, event, buffered)\n"),
        replace="",
        test="test_register_client_tools_returns_the_daemons_reported_budget",
        because="the daemon never reports the budget, so the SDK guesses",
    ),
    Reversion(
        target=_IPC,
        find='        if not request_id:\n            return\n'
             '        from jaato_server.server.client_tools import registration_answer',
        replace='        from jaato_server.server.client_tools import registration_answer',
        test="test_an_older_client_is_sent_no_answer",
        because="an older client is sent an event type it does not know",
    ),
    Reversion(
        target=_INTROSPECT,
        find="    rows += _client_tool_timeout_rows()\n",
        replace="",
        test="test_explain_clients_lists_the_host_tool_clock",
        because="the TIMEOUTS table again omits the host-tool clock",
    ),
]
