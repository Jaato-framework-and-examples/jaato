"""One plugin tool call answers as a session would, with no session (#1606).

The runner-side half (``shared/plugin_tool_call.PluginToolHost``) asks a
plugin registry the questions a session's wire asks: who owns the tool, is
it in the profile's surface, what does the wire carry for it -- and runs a
call through a ``ToolExecutor`` wired the way ``JaatoSession.configure``
wires one.  Each test drives a real ``PluginRegistry`` holding the real
``template`` plugin, a real ``PermissionPlugin`` and a real executor, and no
``JaatoSession`` anywhere: the property under test is that the answer is the
profile's without a session to ask.

The daemon half (``server/plugin_tool_calls.py``) is checked where it decides
something of its own: which call a permission answer belongs to, and the tag
that lets a caller with several calls tell their ASKs apart.  The whole path
-- a real daemon, a runner bootstrapped as a plugin host, the SDK -- is
``jaato_sdk/conformance/test_plugin_tool_calls_1606.py``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("jinja2")

from jaato_sdk.events import (  # noqa: E402
    PermissionRequestedEvent,
    PermissionResponseRequest,
    PluginToolInvokeRequest,
)

from jaato_server.server.plugin_tool_calls import (  # noqa: E402
    PluginToolCalls,
    _CallSink,
    _InFlight,
    request_problem,
)
from jaato_server.shared.jaato_runtime import JaatoRuntime  # noqa: E402
from jaato_server.shared.plugin_tool_call import build_plugin_tool_host  # noqa: E402
from jaato_server.shared.plugins.permission.plugin import PermissionPlugin  # noqa: E402
from jaato_server.shared.plugins.registry import PluginRegistry  # noqa: E402
from jaato_server.shared.plugins.template.plugin import TemplatePlugin  # noqa: E402
from jaato_server.shared.tests.reversion import Reversion  # noqa: E402

_CALL = "jaato-server/jaato_server/shared/plugin_tool_call.py"
_CALLS = "jaato-server/jaato_server/server/plugin_tool_calls.py"
_BOOT = "jaato-server/jaato_server/server/runner/session.py"

REVERSIONS = [
    Reversion(
        target=_BOOT,
        find=("    if envelope.plugin_host:\n"
              "        return _build_plugin_host(envelope, runtime, child_cb)\n"),
        replace="",
        test="test_the_runner_bootstraps_a_plugin_host_and_no_session",
        because="a plugin tool call's runner builds a session after all",
    ),
    Reversion(
        target=_CALL,
        find="        if self._plugins is None or plugin_name in self._plugins:\n",
        replace="        if True:\n",
        test="test_a_plugin_the_profile_does_not_enable_has_no_tools",
        because="every registered plugin's tools are reachable, whatever the "
                "profile's plugins: list says (#1590)",
    ),
    Reversion(
        target=_CALL,
        find="        refusal = self.tool_scope_refusal(tool)\n",
        replace="        refusal = ''\n",
        test="test_describe_says_a_scoped_out_tool_does_not_exist",
        because="describe answers a tool the profile scoped out as present",
    ),
    Reversion(
        target=_CALL,
        find="        visible = filter_visible_tool_schemas(self.registry, [schema])\n",
        replace="        visible = [schema]\n",
        test="test_describe_answers_the_schema_the_profile_narrows",
        because="describe returns the plugin's declared schema, not the one "
                "the profile's narrow_tool_schema puts on the wire",
    ),
    Reversion(
        target=_CALL,
        find='        if owner_name != plugin_name:\n',
        replace="        if False:\n",
        test="test_describe_refuses_a_tool_named_under_another_plugin",
        because="a tool is reachable by naming any plugin",
    ),
    Reversion(
        target=_CALL,
        find="    return executor.execute(tool, args, **kwargs)\n",
        replace=("    _fn = executor._registry.get_plugin_for_tool(tool)"
                 ".get_executors()[tool]\n"
                 "    return True, _fn(args)\n"),
        test="test_invoke_goes_through_the_permission_gate",
        because="the call skips the wired executor, so the profile's "
                "permission policy never runs",
    ),
    Reversion(
        target=_CALLS,
        find=("            if call.server is not None and request_id in "
              "_pending_prompt_ids(call.server):\n"),
        replace="            if False:\n",
        test="test_a_permission_answer_reaches_the_call_waiting_on_it",
        because="the caller's answer is routed to its own session, and the "
                "call waits on nobody until its timeout",
    ),
    Reversion(
        target=_CALLS,
        find=('            event = event.model_copy(\n'
              '                update={"origin_request_id": self.origin_request_id or None})\n'),
        replace="            pass\n",
        test="test_a_forwarded_ask_names_the_call_it_belongs_to",
        because="a caller with several calls cannot tell which one is asking",
    ),
]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


def _host(tmp_path: Path, *, template_config=None, plugins=("template",),
          scopes=None, policy=None):
    """A plugin host over a real registry, as the runner's bootstrap builds it."""
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    runtime = JaatoRuntime(provider_name="echo", workspace_path=ws)
    reg = PluginRegistry()
    reg.set_workspace_path(str(ws))
    reg.register_plugin(TemplatePlugin(), expose=True,
                        config={"workspace_path": str(ws),
                                **(template_config or {})})
    perm = None
    if policy is not None:
        perm = PermissionPlugin()
        perm.initialize({"policy": policy})
    runtime.configure_plugins(reg, permission_plugin=perm)
    return build_plugin_tool_host(
        runtime, list(plugins), scopes,
        permission_context={"agent_type": "main", "session_id": "plugintool_x"})


def _properties(answer) -> set:
    return set(answer["tool_schema"]["parameters"].get("properties", {}))


def test_describe_answers_the_schema_the_profile_narrows(tmp_path):
    narrow = _host(tmp_path, template_config={"allow_inline_template": False}
                   ).describe("template", "renderTemplateToFile")
    assert narrow["exists"], narrow
    assert "template" not in _properties(narrow)
    assert "template_id" in _properties(narrow)


def test_describe_says_a_scoped_out_tool_does_not_exist(tmp_path):
    host = _host(tmp_path, scopes={"template": ["listAvailableTemplates"]})
    answer = host.describe("template", "renderTemplateToFile")
    assert not answer["exists"]
    assert answer["reason"] == "not_in_surface"
    assert answer["tool_schema"] is None
    assert host.describe("template", "listAvailableTemplates")["exists"]


def test_a_plugin_the_profile_does_not_enable_has_no_tools(tmp_path):
    host = _host(tmp_path, plugins=("todo",), policy={"defaultPolicy": "allow"})
    answer = host.describe("template", "listAvailableTemplates")
    assert not answer["exists"]
    assert answer["reason"] == "not_in_surface"
    assert "does not enable" in answer["detail"]
    assert host.invoke("template", "listAvailableTemplates", {})["ran"] is False


def test_describe_refuses_a_tool_named_under_another_plugin(tmp_path):
    answer = _host(tmp_path).describe("memory", "listAvailableTemplates")
    assert not answer["exists"]
    assert answer["reason"] == "wrong_plugin"


def test_invoke_goes_through_the_permission_gate(tmp_path):
    out = tmp_path / "ws" / "never.txt"
    answer = _host(tmp_path, policy={"defaultPolicy": "deny"}).invoke(
        "template", "renderTemplateToFile",
        {"template": "x", "variables": {}, "output_path": str(out)})
    assert answer["ran"]
    assert answer["success"] is False, answer
    assert not out.exists()


def test_invoke_refuses_what_describe_says_does_not_exist(tmp_path):
    host = _host(tmp_path, scopes={"template": ["listAvailableTemplates"]},
                 policy={"defaultPolicy": "allow"})
    answer = host.invoke(
        "template", "renderTemplateToFile",
        {"template": "x", "variables": {}, "output_path": str(tmp_path / "o")})
    assert answer == {"ran": False, "reason": "not_in_surface",
                      "detail": answer["detail"]}


# ------------------------------------------------------- the daemon half


class _Handler:
    def __init__(self, request_id):
        self._event = SimpleNamespace(request_id=request_id)

    def pending_events(self):
        return [self._event]


class _Server:
    def __init__(self, request_id):
        self._prompt_operator_handler = _Handler(request_id)
        self.answers = []

    def respond_to_permission(self, request_id, response, *, edited_arguments=None,
                              user_id=None):
        self.answers.append((request_id, response, user_id))


def test_a_permission_answer_reaches_the_call_waiting_on_it():
    sent = []
    calls = PluginToolCalls(SimpleNamespace(), lambda c, e: sent.append((c, e)),
                            lambda c: {"client-1": "alice"}.get(c))
    server = _Server("perm-1")
    calls._inflight["_plugin_tool:x"] = _InFlight("client-1", "s1", server)
    answer = PermissionResponseRequest(request_id="perm-1", response="y")
    assert calls.route_permission_response("client-1", "", answer)
    assert server.answers == [("perm-1", "y", "alice")]
    assert sent == []
    # Another client's answer to the same id is not this call's.
    assert calls.route_permission_response("client-2", "own", answer) is False


def test_a_forwarded_ask_names_the_call_it_belongs_to():
    sent = []
    sink = _CallSink(caller="client-1", send=lambda c, e: sent.append((c, e)),
                     origin_request_id="pti_abc")
    ask = PermissionRequestedEvent(request_id="perm-1", tool_name="t")
    sink(ask)
    [(client, forwarded)] = sent
    assert client == "client-1"
    assert forwarded.origin_request_id == "pti_abc"
    assert ask.origin_request_id is None  # the relay's copy is untouched


def test_a_request_must_name_one_configuration():
    assert request_problem(PluginToolInvokeRequest(tool="t")) == "plugin is required"
    assert request_problem(PluginToolInvokeRequest(
        plugin="p", tool="t", profile="x", plugin_configs={})).startswith(
            "profile and plugin_configs")
    assert request_problem(PluginToolInvokeRequest(plugin="p", tool="t")) == ""


# ------------------------------------------------------- the runner bootstrap


def test_the_runner_bootstraps_a_plugin_host_and_no_session(tmp_path):
    """``plugin_host`` stops ``bootstrap_session`` before the session: no
    model is named, none is needed, and the host answers for the profile."""
    from jaato_server.server.runner.session import bootstrap_session
    from jaato_server.shared.session_envelope import SessionInitEnvelope

    ws = tmp_path / "ws"
    ws.mkdir()
    host = bootstrap_session(SessionInitEnvelope(
        session_id="plugintool_test", workspace_path=str(ws),
        profile_name="", provider_name="", model_name="",
        plugins=[{"name": "template"}],
        plugin_configs={"template": {"allow_inline_template": False}},
        plugin_host=True,
    ))
    assert host.session is None
    assert host.plugin_host is not None
    answer = host.plugin_host.describe("template", "renderTemplateToFile")
    assert answer["exists"], answer
    assert "template" not in answer["tool_schema"]["parameters"]["properties"]
