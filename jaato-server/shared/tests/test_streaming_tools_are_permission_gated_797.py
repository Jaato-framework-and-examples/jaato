"""A ``-stream`` tool is the same tool, and is judged by the same rules.

WHY THIS EXISTS (#797).  ``PluginRegistry`` auto-generates a
``<tool>-stream`` variant for every tool on a ``StreamingCapable``
plugin and puts it in the model's tool list -- ``notebook_execute-stream``
is core/eager, so it is on the wire from the first turn of any profile
carrying ``notebook``.  ``JaatoSession`` routed those variants to
``_execute_streaming_tool``, which resolves the plugin and calls
``StreamManager.start_stream`` -> ``plugin.execute_streaming`` directly.

That route never touched ``ToolExecutor.execute``, and
``ToolExecutor._execute_impl`` held the only ``check_permission`` call in
the tree.  ``grep -n permission shared/plugins/streaming/*.py`` returned
nothing.  So a tool the operator had **explicitly blacklisted** executed
with zero prompts and zero audit records when the model appended seven
characters to its name.  Not "the prompt was skipped": a standing denial
was defeatable by a suffix the model can type.

WHAT IS PINNED HERE, and each is attached to a way the fix could rot:

  * behavioural -- a denied tool invoked as ``<name>-stream`` does not
    execute, and the refusal reaches the model in the SAME shape a denied
    plain call produces.  A denial the model cannot recognise is a
    different defect wearing the fix as a disguise.
  * the name judged is the BASE name, so the variant inherits every rule
    already written for the tool.  Gating on ``fc.name`` would "pass"
    every allow-path test and re-open the bypass for anyone whose policy
    names the tool.
  * structural -- no function in ``jaato_session.py`` may reach
    ``start_stream`` without passing a gate first, so a future streaming
    entry point cannot reintroduce an unchecked execution path.
  * one gate, not two -- ``_execute_impl`` must reach the permission
    plugin through the shared helper rather than its own
    ``check_permission`` call.  Two copies of a security gate is how this
    class of defect recurs, which is the whole reason the fix factored
    rather than duplicated.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from shared.ai_tool_runner import ToolExecutor
from shared.jaato_session import JaatoSession
from shared.plugins.permission import PermissionPlugin
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion
from jaato_sdk.plugins.model_provider.types import FunctionCall

_SESSION_SRC = Path(__file__).resolve().parents[1] / "jaato_session.py"
_RUNNER_SRC = Path(__file__).resolve().parents[1] / "ai_tool_runner.py"

_SESSION = "jaato-server/shared/jaato_session.py"
_RUNNER = "jaato-server/shared/ai_tool_runner.py"


REVERSIONS = [
    Reversion(
        target=_SESSION,
        find="""        allowed, denial, tool_args = self._gate_streaming_tool(base_name, fc)
        if not allowed:
            return (False, denial)
""",
        replace="""        tool_args = fc.args
""",
        test="test_a_blacklisted_tool_does_not_execute_as_a_stream_variant",
        because="the -stream route executes again with no permission check",
    ),
    Reversion(
        target=_SESSION,
        find="""        gate = self._executor.check_permission_only(
            base_name, fc.args, fc.id,
        )""",
        replace="""        gate = self._executor.check_permission_only(
            fc.name, fc.args, fc.id,
        )""",
        test="test_the_gate_judges_the_base_name_not_the_wire_name",
        because=(
            "the gate judges the -stream name, so a policy written for the "
            "tool no longer binds the variant"
        ),
    ),
    Reversion(
        target=_RUNNER,
        find="""        gate = self.check_permission_only(name, args, call_id, debug)
        if not gate.allowed:
            return False, gate.denial""",
        replace="""        gate = self.check_permission_only(name, args, call_id, debug)
        if not gate.allowed and False:
            return False, gate.denial""",
        test="test_the_executor_still_denies_the_plain_call",
        because="the executor's own gate stopped refusing a denied tool",
    ),
]


# --------------------------------------------------------------- fixtures


class _RecordingStreamingPlugin:
    """A ``StreamingCapable`` stand-in that records what it was asked to run.

    Only ``execute_streaming`` matters here: if the gate works, this is
    never entered, and the assertion that matters is about the *absence*
    of a recorded call rather than about any output it could produce.
    """

    name = "recorder"

    def __init__(self) -> None:
        self.executed: List[Tuple[str, Dict[str, Any]]] = []

    def supports_streaming(self, tool_name: str) -> bool:
        return True

    def execute_streaming(self, tool_name: str, arguments: Dict[str, Any], **kw):
        self.executed.append((tool_name, arguments))
        return iter(())


def _permission_plugin(policy: Dict[str, Any]) -> PermissionPlugin:
    plugin = PermissionPlugin()
    plugin.initialize({"policy": policy, "channel_type": "queue"})
    return plugin


def _executor(plugin: Optional[PermissionPlugin]) -> ToolExecutor:
    """A ``ToolExecutor`` that can resolve ``danger_tool``.

    The executor refuses -- before the permission check -- any tool it
    cannot resolve, so the plain-call comparison below needs a real
    registration for the denial to be a PERMISSION denial rather than a
    missing-executor one.
    """
    executor = ToolExecutor(auto_background_enabled=False)
    executor.register("danger_tool", lambda args: {"ran": True})
    if plugin is not None:
        executor.set_permission_plugin(plugin)
    return executor


def _session(executor: Optional[ToolExecutor], plugin) -> JaatoSession:
    """A ``JaatoSession`` wired with real permission machinery.

    The registry, stream manager and UI hooks are doubles -- the code
    under test is the session's own routing, and a real registry would
    drag in plugin discovery without changing what is asserted.
    ``start_stream`` delegates to the plugin so "did the tool run?" is
    answered by the plugin's own record, not by a mock's call count.
    """
    session = JaatoSession.__new__(JaatoSession)
    session._agent_id = "agent-1"
    session._agent_type = "main"
    session._agent_name = None
    session._ui_hooks = None
    session._executor = executor
    session._runtime = MagicMock()
    session._runtime.registry.get_base_tool_name.side_effect = (
        lambda n: n[: -len("-stream")] if n.endswith("-stream") else n
    )
    session._runtime.registry.get_streaming_plugin.return_value = plugin
    session._runtime.registry.get_plugin_for_tool.return_value = plugin

    handle = MagicMock(stream_id="s1", initial_chunks=[])
    handle.status.value = "running"

    def _start_stream(**kwargs):
        kwargs["plugin"].execute_streaming(
            kwargs["tool_name"], kwargs["arguments"])
        return handle

    session._stream_manager = MagicMock()
    session._stream_manager.start_stream.side_effect = _start_stream
    return session


def _call(name: str = "danger_tool-stream", **args) -> FunctionCall:
    return FunctionCall(id="call_1", name=name,
                        args=args or {"command": "rm -rf /tmp/evidence"})


# ------------------------------------------------------------ behavioural


def test_a_blacklisted_tool_does_not_execute_as_a_stream_variant() -> None:
    """The reproduction from #797, verbatim.

    The operator has explicitly denied ``danger_tool``.  The model calls
    ``danger_tool-stream``.  Before the fix this executed.
    """
    perm = _permission_plugin({"defaultPolicy": "ask"})
    perm._policy.add_session_blacklist("danger_tool")

    # Precondition: the base tool really is denied, by blacklist.
    allowed, info = perm.check_permission("danger_tool", {}, {}, "pre")
    assert allowed is False
    assert info.get("method") == "session_blacklist"

    streaming_plugin = _RecordingStreamingPlugin()
    session = _session(_executor(perm), streaming_plugin)

    ok, result = JaatoSession._execute_streaming_tool(session, _call(), None)

    assert ok is False, "a blacklisted tool was allowed to stream"
    assert streaming_plugin.executed == [], (
        "the tool EXECUTED despite a standing denial -- this is the #797 "
        f"bypass: {streaming_plugin.executed}")
    assert not session._stream_manager.start_stream.called, (
        "the stream was started before the permission gate ran")
    assert "Permission denied" in result["error"]


def test_the_denial_reaches_the_model_in_the_executors_own_shape() -> None:
    """A refusal must be indistinguishable from a plain refusal.

    The model recognises a denial by the ``_permission`` block and the
    ``error`` prefix.  A streaming route that refused in some novel shape
    would leave the model unable to tell a denial from a tool failure,
    and would re-issue the call.
    """
    perm = _permission_plugin({"defaultPolicy": "deny"})
    executor = _executor(perm)
    streaming_plugin = _RecordingStreamingPlugin()
    session = _session(executor, streaming_plugin)

    stream_ok, stream_result = JaatoSession._execute_streaming_tool(
        session, _call(), None)
    plain_ok, plain_result = executor.execute(
        "danger_tool", {"command": "rm -rf /tmp/evidence"}, call_id="call_1")

    assert stream_ok is plain_ok is False
    assert set(stream_result) == set(plain_result) == {"error", "_permission"}
    assert stream_result["error"] == plain_result["error"]
    assert stream_result["_permission"] == plain_result["_permission"]
    assert stream_result["_permission"]["decision"] == "denied"


def test_the_executor_still_denies_the_plain_call() -> None:
    """The refactor moved the gate; it must not have loosened it.

    ``_execute_impl`` now reaches the permission plugin through the
    shared helper.  This is the half of that change that says the
    executor's own behaviour is unchanged.
    """
    perm = _permission_plugin({"defaultPolicy": "deny"})
    executor = _executor(perm)

    ok, result = executor.execute("danger_tool", {}, call_id="c1")

    assert ok is False
    assert result["_permission"]["decision"] == "denied"
    assert result["_permission"]["method"] == "default"


def test_an_allowed_tool_still_streams() -> None:
    """The gate must refuse, not break streaming.

    A fix that denied everything would pass every assertion above.
    """
    perm = _permission_plugin(
        {"defaultPolicy": "deny", "whitelist": {"tools": ["danger_tool"]}})
    streaming_plugin = _RecordingStreamingPlugin()
    session = _session(_executor(perm), streaming_plugin)

    ok, result = JaatoSession._execute_streaming_tool(session, _call(), None)

    assert ok is True, result
    assert streaming_plugin.executed == [
        ("danger_tool", {"command": "rm -rf /tmp/evidence"})]
    assert result["stream_id"] == "s1"


def test_the_gate_judges_the_base_name_not_the_wire_name() -> None:
    """A policy naming the tool must bind its ``-stream`` variant.

    Judging ``fc.name`` would ask the policy about ``danger_tool-stream``
    -- a name no operator writes -- so a blacklist on ``danger_tool``
    would not match and the bypass would be back with a gate in place to
    hide it.
    """
    asked: List[str] = []

    perm = _permission_plugin({"defaultPolicy": "allow"})
    real_check = perm.check_permission

    def _record(tool_name, args, context=None, call_id=None):
        asked.append(tool_name)
        return real_check(tool_name, args, context, call_id)

    perm.check_permission = _record  # type: ignore[method-assign]
    session = _session(_executor(perm), _RecordingStreamingPlugin())

    JaatoSession._execute_streaming_tool(session, _call(), None)

    assert asked == ["danger_tool"], (
        "the permission gate was asked about the wire name; a policy "
        f"written for the tool would not match it: {asked}")


def test_no_executor_means_nothing_to_enforce() -> None:
    """With no executor there is no permission plugin, so the call runs.

    Same answer ``ToolExecutor`` gives when no plugin is set.  Stated
    because it is the one branch that deliberately does not gate, and a
    reader must not mistake it for the hole.
    """
    streaming_plugin = _RecordingStreamingPlugin()
    session = _session(None, streaming_plugin)

    ok, _result = JaatoSession._execute_streaming_tool(session, _call(), None)

    assert ok is True
    assert streaming_plugin.executed


def test_an_edited_approval_reaches_the_stream() -> None:
    """An approval may rewrite the arguments; the stream must use them.

    ``fc.args`` is what the model asked for, ``gate.args`` is what the
    responder approved.  Executing the former would run something nobody
    said yes to.
    """
    perm = _permission_plugin({"defaultPolicy": "allow"})
    perm.check_permission = lambda *a, **k: (  # type: ignore[method-assign]
        True,
        {"reason": "edited", "method": "user_approved", "was_edited": True,
         "modified_args": {"command": "echo safe"}},
    )
    streaming_plugin = _RecordingStreamingPlugin()
    session = _session(_executor(perm), streaming_plugin)

    ok, _result = JaatoSession._execute_streaming_tool(session, _call(), None)

    assert ok is True
    assert streaming_plugin.executed == [("danger_tool", {"command": "echo safe"})]


# --------------------------------------------------------- structural guard


def _functions(src: Path) -> List[ast.FunctionDef]:
    tree = ast.parse(src.read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _calls_named(node: ast.AST, name: str) -> List[ast.Call]:
    """Every call in *node* whose callee attribute/name is *name*."""
    out = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if isinstance(func, ast.Attribute) and func.attr == name:
            out.append(sub)
        elif isinstance(func, ast.Name) and func.id == name:
            out.append(sub)
    return out


_GATE_CALLS = ("check_permission_only", "_gate_streaming_tool")


def test_every_start_stream_caller_passes_a_gate_first() -> None:
    """No streaming entry point may reach execution ungated.

    This is the guard that would have caught #797 when the ``-stream``
    branch was introduced.  It is deliberately about ``start_stream``
    rather than about ``_execute_streaming_tool`` by name: a NEW function
    that starts a stream is exactly the shape the next bypass would take.
    """
    offenders = []
    for fn in _functions(_SESSION_SRC):
        starts = _calls_named(fn, "start_stream")
        if not starts:
            continue
        gates = [c for n in _GATE_CALLS for c in _calls_named(fn, n)]
        if not gates:
            offenders.append(f"{fn.name}: starts a stream and never gates")
            continue
        if min(g.lineno for g in gates) > min(s.lineno for s in starts):
            offenders.append(
                f"{fn.name}: gates at line {min(g.lineno for g in gates)}, "
                f"AFTER start_stream at {min(s.lineno for s in starts)}")

    assert not offenders, (
        "a streaming execution path reaches the tool without a permission "
        "check (#797):\n  " + "\n  ".join(offenders))


def test_both_streaming_routing_sites_funnel_through_the_gated_helper() -> None:
    """The sequential and parallel loops must both reach the gate.

    #797 existed at TWO routing sites.  They are gated because both call
    ``_execute_streaming_tool``, which gates -- so the property to pin is
    that every site still routes there, rather than one of them growing
    its own call to ``StreamManager``.
    """
    src = _SESSION_SRC.read_text(encoding="utf-8")
    routed = src.count("self._execute_streaming_tool(")
    assert routed >= 2, (
        f"expected both routing sites to call _execute_streaming_tool, "
        f"found {routed}")

    gated = [fn for fn in _functions(_SESSION_SRC)
             if fn.name == "_execute_streaming_tool"]
    assert len(gated) == 1
    assert _calls_named(gated[0], "_gate_streaming_tool"), (
        "_execute_streaming_tool no longer gates, so every routing site "
        "that funnels into it is unchecked")


def test_the_permission_gate_has_exactly_one_implementation() -> None:
    """``_execute_impl`` must not grow a second copy of the gate.

    The fix factored the executor's check into ``check_permission_only``
    precisely so the streaming route shares it.  A second
    ``check_permission`` call inside ``_execute_impl`` would be two gates
    that can drift -- which is how a bypass of this class comes back.
    """
    impls = [fn for fn in _functions(_RUNNER_SRC) if fn.name == "_execute_impl"]
    assert len(impls) == 1
    assert not _calls_named(impls[0], "check_permission"), (
        "_execute_impl calls the permission plugin directly again; it must "
        "go through check_permission_only so there is ONE gate")
    assert _calls_named(impls[0], "check_permission_only"), (
        "_execute_impl no longer runs the permission gate at all")

    helpers = [fn for fn in _functions(_RUNNER_SRC)
               if _calls_named(fn, "check_permission")]
    assert [fn.name for fn in helpers] == ["check_permission_only"], (
        "more than one function in ai_tool_runner.py calls the permission "
        f"plugin: {[fn.name for fn in helpers]}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
