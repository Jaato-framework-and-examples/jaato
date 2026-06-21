"""Client-provided ("host") tools reach the runner-tier model.

Runner-side: ``_register_client_tools_on_runner`` registers each schema as a
core tool (so ``list_tools`` shows it) with a daemon-forwarding executor;
``_make_client_tool_forwarder`` routes execution to the daemon via
``daemon.plugin_execute`` under the ``__client_tools__`` sentinel.  Daemon-side:
the ``daemon.plugin_execute`` handler routes that sentinel to the session
registry's proxy executor (which sends ``ToolExecuteRequestEvent`` to the ws
client).  Pre-fix the schema registered only on the daemon registry → the runner
model was blind (the #344-sibling split).
"""

import asyncio
from types import SimpleNamespace

from server.runner.session import (
    _register_client_tools_on_runner,
    _make_client_tool_forwarder,
    _CLIENT_TOOL_PLUGIN,
)
from server.runner_rpc_handlers.daemon_plugin_execute import DaemonPluginExecuteHandler


class _FakeRegistry:
    def __init__(self, rpc=None):
        self.runner_rpc_client = rpc
        self.core = {}

    def register_core_tool(self, schema, executor, auto_approved=False):
        self.core[schema.name] = (schema, executor, auto_approved)


def test_register_registers_schema_and_forwarder():
    reg = _FakeRegistry()
    _register_client_tools_on_runner(reg, [
        {"name": "send_to_telegram", "description": "d",
         "parameters": {"type": "object"}},
        {"description": "no name"},            # skipped
    ])
    assert list(reg.core) == ["send_to_telegram"]
    schema, executor, auto = reg.core["send_to_telegram"]
    assert schema.name == "send_to_telegram" and auto is True and callable(executor)


def test_mid_session_handler_registers_and_appends_to_session_tools():
    # The session.register_client_tools RPC handler glues a tool registered
    # AFTER session.new onto the LIVE runner: registers it on the registry
    # (forwarding executor) AND appends the schema to session._tools (the
    # cached per-turn list the model is actually sent) so it can be CALLED.
    from server.runner.rpc import RunnerRPC
    reg = _FakeRegistry()
    session = SimpleNamespace(_runtime=SimpleNamespace(registry=reg), _tools=[])
    rpc = SimpleNamespace(_require_ready_session=lambda: (True, None, session))
    ok, payload = RunnerRPC._handle_session_register_client_tools(
        rpc, {"client_tools": [
            {"name": "new_tool", "description": "d", "parameters": {}}]})
    assert ok and payload["registered"] == ["new_tool"]
    assert "new_tool" in reg.core                              # forwarding executor
    assert any(s.name == "new_tool" for s in session._tools)   # model can call it


def test_mid_session_handler_rejects_non_list():
    from server.runner.rpc import RunnerRPC
    ok, payload = RunnerRPC._handle_session_register_client_tools(
        SimpleNamespace(), {"client_tools": "nope"})
    assert ok is False and payload["stage"] == "decode"


def test_forwarder_routes_via_sentinel():
    calls = []
    # KEYWORD-ONLY mock matching the real RunnerRPCClient.daemon_plugin_execute
    # signature (``def ...(self, *, plugin_name, tool_name, args, timeout=None)``)
    # — a positional call raises TypeError, so this catches the regression where
    # the forwarder passed args positionally.
    def _fake(*, plugin_name, tool_name, args, timeout=None):
        calls.append((plugin_name, tool_name, args))
        return {"ok": 1}
    rpc = SimpleNamespace(daemon_plugin_execute=_fake)
    ex = _make_client_tool_forwarder(_FakeRegistry(rpc=rpc), "send_to_telegram")
    assert ex({"text": "hi"}) == {"ok": 1}
    assert calls == [(_CLIENT_TOOL_PLUGIN, "send_to_telegram", {"text": "hi"})]


def test_forwarder_errors_without_channel():
    ex = _make_client_tool_forwarder(_FakeRegistry(rpc=None), "t")
    assert "error" in ex({})


def test_daemon_handler_routes_sentinel_to_proxy_executor():
    captured = {}

    def proxy(args):
        captured["args"] = args
        return {"result": "sent"}

    server = SimpleNamespace(registry=SimpleNamespace(
        get_exposed_executors=lambda: {"send_to_telegram": proxy},
        get_plugin=lambda n: None))           # would fail if the sentinel fell through
    h = DaemonPluginExecuteHandler(server)
    out = asyncio.run(h.handle({
        "plugin_name": _CLIENT_TOOL_PLUGIN,
        "tool_name": "send_to_telegram",
        "args": {"text": "hi"}}))
    assert out == {"result": "sent"}
    assert captured["args"] == {"text": "hi"}


def test_daemon_handler_unknown_client_tool_errors():
    server = SimpleNamespace(registry=SimpleNamespace(
        get_exposed_executors=lambda: {}, get_plugin=lambda n: None))
    h = DaemonPluginExecuteHandler(server)
    try:
        asyncio.run(h.handle({"plugin_name": _CLIENT_TOOL_PLUGIN,
                              "tool_name": "missing", "args": {}}))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "not registered" in str(e)
