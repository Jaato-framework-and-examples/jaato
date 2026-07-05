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
         "parameters": {"type": "object"}},                    # default eager
        {"name": "deep_tool", "description": "d", "parameters": {},
         "discoverability": "discoverable"},                   # explicit opt-out
        {"description": "no name"},            # skipped
    ])
    assert set(reg.core) == {"send_to_telegram", "deep_tool"}
    schema, executor, auto = reg.core["send_to_telegram"]
    assert schema.name == "send_to_telegram" and auto is True and callable(executor)
    # Client tools default to EAGER ('core') so the model uses them on INTENT
    # (not only after list_tools or a persona that names them).
    assert schema.discoverability == "core"
    # An explicit "discoverability" is honored (opt back to deferred).
    assert reg.core["deep_tool"][0].discoverability == "discoverable"


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
    appended = [s for s in session._tools if s.name == "new_tool"]
    assert appended and appended[0].discoverability == "core"  # EAGER → callable on intent


def test_mid_session_handler_syncs_runner_permission_whitelist():
    # A client tool registered mid-session is recorded ``auto_approved=True`` in
    # ``registry._core_auto_approved`` — but ``check_permission`` gates on the
    # permission POLICY whitelist, and the registry→whitelist bridge
    # (``add_whitelist_tools``) runs ONCE at bootstrap.  So the mid-session
    # handler MUST sync the new names into the runner permission whitelist
    # itself, else a headless-driven turn (e.g. session.wake, which always
    # registers its client tools after cold-revive) would raise an operator
    # permission prompt for the tool and block forever with no operator to
    # answer.  Regression for the wake headless-dispatch deadlock.
    from server.runner.rpc import RunnerRPC

    class _FakePermission:
        def __init__(self):
            self.whitelisted = []

        def add_whitelist_tools(self, tools):
            self.whitelisted.extend(tools)

    reg = _FakeRegistry()
    perm = _FakePermission()
    session = SimpleNamespace(
        _runtime=SimpleNamespace(registry=reg, permission_plugin=perm),
        _tools=[])
    rpc = SimpleNamespace(_require_ready_session=lambda: (True, None, session))
    ok, payload = RunnerRPC._handle_session_register_client_tools(
        rpc, {"client_tools": [
            {"name": "record_note", "description": "d", "parameters": {}},
            {"description": "no name"},          # skipped — no name to whitelist
        ]})
    assert ok and payload["registered"] == ["record_note"]
    # The named tool is now on the runner permission whitelist → check_permission
    # short-circuits ALLOW instead of prompting.
    assert perm.whitelisted == ["record_note"]


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
