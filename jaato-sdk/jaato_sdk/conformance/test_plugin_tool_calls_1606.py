"""One plugin tool call with no model turn, against a live daemon (#1606).

``describe_plugin_tool`` / ``invoke_plugin_tool`` are only worth having if
what they answer is what a SESSION with that configuration would do: the
schema its wire carries, the surface its profile allows, the permission gate
its policy runs.  A unit test of the daemon's orchestration cannot show that
-- the answers come from a runner, through a session the daemon builds -- so
every case here drives a real daemon (echo provider, no credentials, no
network) and a real runner through the SDK.

The ``template`` plugin is the issue's first user (kbwiki offers its three
tools over MCP) and it has every property worth checking: a per-session
setting that narrows a schema (``allow_inline_template``), auto-approved
tools, and a tool the permission policy asks about.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from jaato_sdk import IPCClient
from jaato_sdk.conformance.daemon import ConformanceDaemon, echo_workspace
from jaato_sdk.events import ClientType

pytest.importorskip("jinja2")

pytestmark = pytest.mark.conformance

ASK_POLICY = {"permission": {"policy": {"defaultPolicy": "ask"}}}


@pytest.fixture(scope="module")
def tool_daemon():
    """A daemon whose workspace declares three template profiles.

    ``tpl-narrow`` turns inline templates off, which removes ``template``
    from ``renderTemplateToFile``'s schema; ``tpl-scoped`` scopes the plugin
    to ``listAvailableTemplates``; ``tpl-ask`` asks before anything that is
    not auto-approved.
    """
    root = Path(tempfile.mkdtemp(prefix="jaato-plugin-tool-ws-"))
    echo_workspace(root, name="tpl-narrow", plugins=["template"],
                   plugin_configs={"template": {"allow_inline_template": False},
                                   **ASK_POLICY})
    echo_workspace(root, name="tpl-scoped",
                   plugins=["template(tools:[listAvailableTemplates])"],
                   plugin_configs=dict(ASK_POLICY))
    echo_workspace(root, name="tpl-ask", plugins=["template"],
                   plugin_configs=dict(ASK_POLICY))
    d = ConformanceDaemon(root)
    try:
        yield d.start()
    finally:
        d.stop()


def _run(daemon, body):
    async def _main():
        c = IPCClient(socket_path=daemon.socket_path, client_type=ClientType.API,
                      workspace_path=str(daemon.workspace), auto_start=False)
        assert await c.connect(timeout=60), "could not connect to the test daemon"
        try:
            return await body(c)
        finally:
            await c.disconnect()
    return asyncio.run(_main())


def _properties(answer) -> set:
    return set(((answer.tool_schema or {}).get("parameters") or {})
               .get("properties", {}))


def test_describe_answers_the_schema_the_session_narrows(tool_daemon):
    async def body(c):
        narrow = await c.describe_plugin_tool(
            "template", "renderTemplateToFile", profile="tpl-narrow")
        inline = await c.describe_plugin_tool(
            "template", "renderTemplateToFile", plugin_configs={})
        return narrow, inline

    narrow, inline = _run(tool_daemon, body)
    assert narrow.ok and narrow.exists, narrow
    assert "template" not in _properties(narrow)
    assert "template_id" in _properties(narrow)
    assert inline.ok and inline.exists, inline
    assert "template" in _properties(inline)


def test_describe_says_a_scoped_out_tool_does_not_exist(tool_daemon):
    async def body(c):
        scoped = await c.describe_plugin_tool(
            "template", "renderTemplateToFile", profile="tpl-scoped")
        wrong = await c.describe_plugin_tool(
            "memory", "listAvailableTemplates", profile="tpl-ask")
        return scoped, wrong

    scoped, wrong = _run(tool_daemon, body)
    assert scoped.ok and not scoped.exists
    assert scoped.reason == "not_in_surface", scoped
    assert scoped.tool_schema is None
    assert wrong.ok and not wrong.exists
    assert wrong.reason == "wrong_plugin", wrong


def test_invoke_runs_an_auto_approved_tool(tool_daemon):
    async def body(c):
        return await c.invoke_plugin_tool(
            "template", "listAvailableTemplates", {}, profile="tpl-ask")

    answer = _run(tool_daemon, body)
    assert answer.ok, answer
    assert answer.success is True
    assert isinstance(answer.result, dict) and "templates" in answer.result


def test_invoke_hands_the_ask_to_the_caller_and_runs_on_yes(tool_daemon):
    out = tool_daemon.workspace / "rendered-yes.txt"
    asked = []

    async def body(c):
        return await c.invoke_plugin_tool(
            "template", "renderTemplateToFile",
            {"template": "hi {{ who }}", "variables": {"who": "mcp"},
             "output_path": str(out)},
            profile="tpl-ask",
            on_permission=lambda ev: (asked.append(ev.tool_name), "y")[1])

    answer = _run(tool_daemon, body)
    assert asked == ["renderTemplateToFile"]
    assert answer.ok and answer.success is True, answer
    assert out.read_text(encoding="utf-8") == "hi mcp"


def test_invoke_with_no_answerer_is_refused_not_left_waiting(tool_daemon):
    out = tool_daemon.workspace / "rendered-no.txt"

    async def body(c):
        return await c.invoke_plugin_tool(
            "template", "renderTemplateToFile",
            {"template": "x", "variables": {}, "output_path": str(out)},
            profile="tpl-ask", timeout=60)

    answer = _run(tool_daemon, body)
    assert answer.ok, answer
    assert answer.success is False
    assert not out.exists()


def test_invoke_refuses_a_tool_the_profile_scopes_out(tool_daemon):
    out = tool_daemon.workspace / "rendered-scoped.txt"

    async def body(c):
        return await c.invoke_plugin_tool(
            "template", "renderTemplateToFile",
            {"template": "x", "variables": {}, "output_path": str(out)},
            profile="tpl-scoped",
            on_permission=lambda ev: "y")

    answer = _run(tool_daemon, body)
    assert not answer.ok
    assert answer.category == "not_in_surface", answer
    assert not out.exists()


def test_a_bad_request_and_a_missing_profile_say_why(tool_daemon):
    async def body(c):
        both = await c.describe_plugin_tool(
            "template", "listAvailableTemplates", profile="tpl-ask",
            plugin_configs={})
        missing = await c.invoke_plugin_tool(
            "template", "listAvailableTemplates", {}, profile="no-such-profile")
        return both, missing

    both, missing = _run(tool_daemon, body)
    assert not both.ok and both.category == "invalid_request"
    assert not missing.ok and missing.category == "session_failed", missing
    assert "no-such-profile" in missing.error
