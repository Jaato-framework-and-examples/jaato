"""Every permission decision says what it was (issue #951).

Before this, ``check_permission`` traced exactly two things: that a
check had started, and — on the ASK branch only — that it was about to
prompt.  Every terminal decision went to ``_log_decision``, which
appends to an in-memory list that reaches no file and no event.  So an
ALLOW and a DENY produced byte-identical operator-visible output: one
``check_permission: tool=X`` line, then silence.

That made a real failure undiagnosable.  #951 reports a subagent whose
``writeNewFile`` was checked 38 times and never ran, with zero ASK
prompts and no file on disk — and re-running the same profile with
``defaultPolicy: deny`` instead of ``ask`` produced *the same* logs and
the same ~10-token tool result.  A genuine policy DENY and a call
vanishing after the gate were indistinguishable, so the report could
describe the symptom exactly and not name the verdict.

The fix is structural rather than per-branch: ``check_permission`` is a
single-exit wrapper around ``_check_permission_impl``, which keeps its
twenty-odd rule-specific exits.  A branch added later is traced
whether or not its author remembers to trace it — which is the property
a per-branch fix would not have.

These tests cover:

- one DECISION line per terminal decision, naming ``allowed`` + the
  rule kind (``method``), for every decision kind the plugin can reach;
- the two exits that recorded *nothing* at all (``not_initialized``,
  ``unknown``);
- a raising check — a third outcome the old logging could not express;
- WHO asked, taken from the caller's per-session context rather than
  from the registry-shared singleton's own ``_agent_name``;
- the audit entry gaining the rule kind + caller, so
  ``EvalContext.execution_log`` can be reasoned over;
- an AST guard that the wrapper stays single-exit.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Dict, List, Tuple

import pytest

from .. import plugin as permission_plugin_module
from ..plugin import PermissionPlugin, _describe_permission_caller


# ---------------------------------------------------------------- harness


@pytest.fixture
def traced(monkeypatch) -> List[Tuple[str, str]]:
    """Capture ``(component, message)`` pairs instead of writing a file."""
    lines: List[Tuple[str, str]] = []

    def _capture(component: str, msg: str, **_kwargs: Any) -> None:
        lines.append((component, msg))

    monkeypatch.setattr(permission_plugin_module, "_trace_write", _capture)
    return lines


def _decisions(lines: List[Tuple[str, str]]) -> List[str]:
    return [msg for _c, msg in lines if "check_permission: DECISION" in msg]


def _plugin(policy: Dict[str, Any]) -> PermissionPlugin:
    plugin = PermissionPlugin()
    plugin.initialize({"policy": policy, "channel_type": "queue"})
    return plugin


SUBAGENT_CONTEXT = {
    "agent_type": "subagent",
    "agent_name": "documentalista",
    "session_id": "20260910_115239",
}


# ------------------------------------------------- one line per decision


def test_whitelist_allow_is_traced(traced) -> None:
    plugin = _plugin({"defaultPolicy": "deny",
                      "whitelist": {"tools": ["writeNewFile"]}})
    allowed, _info = plugin.check_permission(
        "writeNewFile", {"path": "docs/x.md"}, SUBAGENT_CONTEXT, "call_1")

    assert allowed is True
    line, = _decisions(traced)
    assert "tool=writeNewFile" in line
    assert "call_id=call_1" in line
    assert "allowed=True" in line
    assert "method=whitelist" in line


def test_default_deny_is_traced_and_not_silent(traced) -> None:
    """The exact case #951 could not tell apart from "nothing happened"."""
    plugin = _plugin({"defaultPolicy": "deny", "whitelist": {"tools": []}})
    allowed, _info = plugin.check_permission(
        "writeNewFile", {"path": "docs/x.md"}, SUBAGENT_CONTEXT, "call_2")

    assert allowed is False
    line, = _decisions(traced)
    assert "allowed=False" in line
    assert "method=default" in line
    assert "reason=" in line


def test_allow_and_deny_are_distinguishable(traced) -> None:
    """The regression this file exists to prevent: identical output."""
    allow_plugin = _plugin({"defaultPolicy": "deny",
                            "whitelist": {"tools": ["writeNewFile"]}})
    allow_plugin.check_permission("writeNewFile", {}, None, "c")
    allowed_line, = _decisions(traced)

    traced.clear()
    deny_plugin = _plugin({"defaultPolicy": "deny",
                           "whitelist": {"tools": []}})
    deny_plugin.check_permission("writeNewFile", {}, None, "c")
    denied_line, = _decisions(traced)

    assert allowed_line != denied_line


def test_blacklist_deny_names_the_rule_kind(traced) -> None:
    plugin = _plugin({"defaultPolicy": "allow",
                      "blacklist": {"tools": ["removeFile"]}})
    allowed, _info = plugin.check_permission("removeFile", {}, None, "c")

    assert allowed is False
    assert "method=blacklist" in _decisions(traced)[0]


def test_suspension_allow_is_traced(traced) -> None:
    """An ALLOW nobody decided is not the same as a whitelist ALLOW."""
    plugin = _plugin({"defaultPolicy": "deny", "whitelist": {"tools": []}})
    plugin._idle_suspended = True

    allowed, _info = plugin.check_permission("writeNewFile", {}, None, "c")

    assert allowed is True
    assert "method=idle_suspension" in _decisions(traced)[0]


def test_allow_all_is_traced(traced) -> None:
    plugin = _plugin({"defaultPolicy": "deny", "whitelist": {"tools": []}})
    plugin._allow_all = True

    allowed, _info = plugin.check_permission("writeNewFile", {}, None, "c")

    assert allowed is True
    assert "method=allow_all" in _decisions(traced)[0]


# --------------------------------------- the exits that recorded nothing


def test_uninitialized_plugin_allow_is_traced_and_logged(traced) -> None:
    """A plugin whose ``initialize()`` never landed permits everything.

    It always did — and it left no trace line and no audit entry, so a
    session running unchecked looked exactly like one running on a
    policy that permits.  It is an ALLOW; it must say so.
    """
    plugin = PermissionPlugin()  # deliberately not initialized

    allowed, info = plugin.check_permission("writeNewFile", {}, None, "c")

    assert allowed is True
    assert info["method"] == "not_initialized"
    assert "method=not_initialized" in _decisions(traced)[0]
    assert plugin.get_execution_log()[-1]["decision"] == "allow"


def test_check_that_raises_is_traced_and_reraised(traced) -> None:
    """``ToolExecutor`` converts a raise into a fail-closed denial.

    Name it here, where the cause is still visible, rather than only
    downstream where it reads as a policy decision.
    """
    plugin = _plugin({"defaultPolicy": "allow"})
    boom = RuntimeError("policy engine exploded")
    plugin._policy.check = lambda *a, **k: (_ for _ in ()).throw(boom)

    with pytest.raises(RuntimeError):
        plugin.check_permission("writeNewFile", {}, SUBAGENT_CONTEXT, "c")

    raised = [m for _c, m in traced if "check_permission: RAISED" in m]
    assert len(raised) == 1
    assert "RuntimeError: policy engine exploded" in raised[0]
    assert "agent=subagent:documentalista" in raised[0]


# ------------------------------------------------------------- who asked


def test_decision_names_the_asking_agent(traced) -> None:
    """One shared plugin decides for a parent and every subagent.

    #951's traces show the shared registry labelling the subagent's own
    lines with the PARENT's name (``@escriba``), because that identity
    comes from whatever initialized the singleton last.  The permission
    line takes it from the caller's context instead.
    """
    plugin = _plugin({"defaultPolicy": "deny", "whitelist": {"tools": []}})
    plugin._agent_name = "escriba"  # the parent — what the singleton holds

    plugin.check_permission("writeNewFile", {}, SUBAGENT_CONTEXT, "c")

    line, = _decisions(traced)
    assert "agent=subagent:documentalista" in line
    assert "session=20260910_115239" in line


def test_entry_line_keeps_its_shape(traced) -> None:
    """``check_permission: tool=X call_id=Y`` still leads the line.

    Operators (and #951's own tables) grep for it; identity is appended,
    not substituted.
    """
    plugin = _plugin({"defaultPolicy": "allow"})
    plugin.check_permission("writeNewFile", {}, SUBAGENT_CONTEXT, "call_9")

    entry = [m for _c, m in traced
             if m.startswith("check_permission: tool=writeNewFile "
                             "call_id=call_9")]
    assert len(entry) == 1


def test_describe_caller_tolerates_a_partial_context() -> None:
    assert _describe_permission_caller(None) == ""
    assert _describe_permission_caller({}) == ""
    assert _describe_permission_caller({"agent_type": "main"}) == " agent=main:?"
    assert _describe_permission_caller(
        {"session_id": "s1"}) == " session=s1"


# ---------------------------------------------------------- audit record


def test_audit_entry_gains_rule_kind_and_caller() -> None:
    """``_log_decision`` knows the reason, not the rule kind or the caller.

    Both are known at the single exit, so they are stamped on there —
    the same after-the-fact enrichment approver identity already used
    (#859).  Evaluators read this list as
    ``EvalContext.execution_log``.
    """
    plugin = _plugin({"defaultPolicy": "deny",
                      "whitelist": {"tools": ["writeNewFile"]}})

    plugin.check_permission(
        "writeNewFile", {"path": "docs/x.md"}, SUBAGENT_CONTEXT, "call_7")

    entry = plugin.get_execution_log()[-1]
    assert entry["decision"] == "allow"
    assert entry["method"] == "whitelist"
    assert entry["call_id"] == "call_7"
    assert entry["agent_type"] == "subagent"
    assert entry["agent_name"] == "documentalista"
    assert entry["session_id"] == "20260910_115239"


def test_two_decisions_get_two_stamped_entries() -> None:
    """The stamp must not decorate a previous call's entry."""
    plugin = _plugin({"defaultPolicy": "deny",
                      "whitelist": {"tools": ["writeNewFile"]}})

    plugin.check_permission("writeNewFile", {}, SUBAGENT_CONTEXT, "call_a")
    plugin.check_permission("removeFile", {}, SUBAGENT_CONTEXT, "call_b")

    first, second = plugin.get_execution_log()[-2:]
    assert (first["tool_name"], first["method"], first["call_id"]) == (
        "writeNewFile", "whitelist", "call_a")
    assert (second["tool_name"], second["method"], second["call_id"]) == (
        "removeFile", "default", "call_b")


# ------------------------------------------------------------- AST guard


def _wrapper_ast() -> ast.FunctionDef:
    source = textwrap.dedent(
        inspect.getsource(PermissionPlugin.check_permission))
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def test_wrapper_is_single_exit() -> None:
    """The property that makes a future branch traced by default.

    If ``check_permission`` grows a second ``return``, a decision can
    once again leave the plugin without a DECISION line — which is
    exactly the shape #951 reported.  Tracing per-branch has to be
    re-done for every branch; single-exit cannot drift.
    """
    returns = [n for n in ast.walk(_wrapper_ast()) if isinstance(n, ast.Return)]
    assert len(returns) == 1, (
        "check_permission must stay single-exit so every decision passes "
        "the DECISION trace; move new logic into _check_permission_impl"
    )


def test_wrapper_delegates_to_the_impl() -> None:
    """And the single exit must be the impl's verdict, not a new one."""
    calls = [
        n.func.attr
        for n in ast.walk(_wrapper_ast())
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert calls.count("_check_permission_impl") == 1
    assert "_stamp_last_decision" in calls


def _unlogged_verdict_lines(func: Any, recorders: Tuple[str, ...]) -> List[int]:
    """Verdict-returns with no decision recorded between them.

    Read as: between two consecutive ``return <bool>, <info>`` lines
    there must be at least one call that writes an audit entry, so
    every verdict has its own and no two share one.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func))).body[0]
    verdicts = sorted(
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple)
    )
    records = sorted(
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in recorders
    )
    assert verdicts, f"{func.__name__} must return verdicts"
    unlogged: List[int] = []
    previous = 0
    for line in verdicts:
        if not any(previous < record < line for record in records):
            unlogged.append(line)
        previous = line
    return unlogged


def test_impl_logs_a_decision_before_every_verdict() -> None:
    """No exit of the policy evaluation may return without an audit entry.

    The stamp at the single exit decorates the entry ``_log_decision``
    just wrote, so an exit that logs nothing both loses its own audit
    record and would mislabel the previous call's.  #951 closed the
    last two such exits (``not_initialized`` and ``unknown``); this
    keeps them closed.

    ``_handle_channel_response`` counts as a recorder — it logs on
    every one of its own branches, which the next test asserts.
    """
    unlogged = _unlogged_verdict_lines(
        PermissionPlugin._check_permission_impl,
        ("_log_decision", "_handle_channel_response"),
    )
    assert not unlogged, (
        "verdict returned with no decision recorded before it, at "
        f"_check_permission_impl-relative line(s) {unlogged}"
    )


def test_channel_response_logs_a_decision_before_every_verdict() -> None:
    """The helper the ASK branch delegates its verdict to logs too."""
    unlogged = _unlogged_verdict_lines(
        PermissionPlugin._handle_channel_response, ("_log_decision",))
    assert not unlogged, (
        "channel verdict returned with no _log_decision before it, at "
        f"_handle_channel_response-relative line(s) {unlogged}"
    )
