"""One plugin tool call with no model turn answers as a session would (#1606).

The runner-side half (``shared/plugin_tool_call.py``) is a handful of
questions asked of a SESSION: who owns the tool, is it in this session's
surface, what does this session's wire carry for it, and -- for an invoke --
what does this session's executor make of the call.  Each test drives a real
``PluginRegistry`` holding the real ``template`` and ``permission`` plugins
and a real ``JaatoSession``, because the property under test is that the
answer is the session's, not a copy of its rules.

The daemon half (``server/plugin_tool_calls.py``) is checked where it decides
something of its own: which call a permission answer belongs to, and the tag
that lets a caller with several calls tell their ASKs apart.  The whole path
-- a real daemon, a real runner, the SDK -- is
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
from jaato_server.shared.jaato_session import JaatoSession  # noqa: E402
from jaato_server.shared.plugin_tool_call import (  # noqa: E402
    describe_plugin_tool,
    invoke_plugin_tool,
)
from jaato_server.shared.plugins.permission.plugin import PermissionPlugin  # noqa: E402
from jaato_server.shared.plugins.registry import PluginRegistry  # noqa: E402
from jaato_server.shared.plugins.template.plugin import TemplatePlugin  # noqa: E402
from jaato_server.shared.session_context import isolated_current_session  # noqa: E402
from jaato_server.shared.tests.reversion import Reversion  # noqa: E402

_CALL = "jaato-server/jaato_server/shared/plugin_tool_call.py"
_CALLS = "jaato-server/jaato_server/server/plugin_tool_calls.py"

REVERSIONS = [
    Reversion(
        target=_CALL,
        find=('    if hasattr(session, "tool_in_surface") and not '
              'session.tool_in_surface(tool):\n'),
        replace="    if False:\n",
        test="test_describe_says_a_scoped_out_tool_does_not_exist",
        because="describe answers a tool the profile scoped out as present",
    ),
    Reversion(
        target=_CALL,
        find="    visible = filter_visible_tool_schemas(registry, [schema], session=session)\n",
        replace="    visible = [schema]\n",
        test="test_describe_answers_the_schema_the_session_narrows",
        because="describe returns the plugin's declared schema, not the one "
                "this session's narrow_tool_schema puts on the wire",
    ),
    Reversion(
        target=_CALL,
        find='    if owner_name != plugin_name:\n',
        replace="    if False:\n",
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
        because="the call skips the session's executor, so its permission "
                "policy never runs",
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
    with isolated_current_session():
        yield


def _session(tmp_path: Path, *, template_config=None, scopes=None,
             policy=None) -> JaatoSession:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    runtime = JaatoRuntime(provider_name="echo", workspace_path=ws)
    reg = PluginRegistry()
    reg.set_workspace_path(str(ws))
    reg.register_plugin(TemplatePlugin(), expose=True,
                        config={"workspace_path": str(ws),
                                **(template_config or {})})
    runtime.configure_plugins(reg)
    session = JaatoSession(runtime, "echo", agent_id="main")
    session.configure(
        skip_provider=True, plugins=["template"], tool_scopes=scopes,
        plugin_configs={"template": dict(template_config or {})},
    )
    session._provider = SimpleNamespace(uses_external_tools=lambda: True)
    if policy is not None:
        perm = PermissionPlugin()
        perm.initialize({"policy": policy})
        session._executor.set_permission_plugin(perm)
    return session


def _properties(answer) -> set:
    return set(answer["tool_schema"]["parameters"].get("properties", {}))


def test_describe_answers_the_schema_the_session_narrows(tmp_path):
    narrow = describe_plugin_tool(
        _session(tmp_path, template_config={"allow_inline_template": False}),
        "template", "renderTemplateToFile")
    assert narrow["exists"], narrow
    assert "template" not in _properties(narrow)
    assert "template_id" in _properties(narrow)


def test_describe_says_a_scoped_out_tool_does_not_exist(tmp_path):
    session = _session(tmp_path, scopes={"template": ["listAvailableTemplates"]})
    answer = describe_plugin_tool(session, "template", "renderTemplateToFile")
    assert not answer["exists"]
    assert answer["reason"] == "not_in_surface"
    assert answer["tool_schema"] is None
    assert describe_plugin_tool(
        session, "template", "listAvailableTemplates")["exists"]


def test_describe_refuses_a_tool_named_under_another_plugin(tmp_path):
    answer = describe_plugin_tool(
        _session(tmp_path), "memory", "listAvailableTemplates")
    assert not answer["exists"]
    assert answer["reason"] == "wrong_plugin"


def test_invoke_goes_through_the_permission_gate(tmp_path):
    out = tmp_path / "ws" / "never.txt"
    session = _session(tmp_path, policy={"defaultPolicy": "deny"})
    answer = invoke_plugin_tool(
        session, "template", "renderTemplateToFile",
        {"template": "x", "variables": {}, "output_path": str(out)})
    assert answer["ran"]
    assert answer["success"] is False, answer
    assert not out.exists()


def test_invoke_refuses_what_describe_says_does_not_exist(tmp_path):
    session = _session(tmp_path, scopes={"template": ["listAvailableTemplates"]},
                       policy={"defaultPolicy": "allow"})
    answer = invoke_plugin_tool(
        session, "template", "renderTemplateToFile",
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
