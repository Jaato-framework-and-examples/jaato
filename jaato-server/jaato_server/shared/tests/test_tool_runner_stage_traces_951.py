"""The stage between the permission gate and the executor is visible (#951).

#951's report ends at a wall: ``[PERMISSION] check_permission`` fires,
``[FILE_EDIT] _resolve_path`` never does, and nothing in between says
anything — so "the policy denied it", "no executor could be resolved"
and "the call was handed to auto-background" were one indistinguishable
silence.  The reporter's own words: *"Whatever consumes the call between
the permission gate and file_edit's executor is the place to look."*
There was no way to look.

``ToolExecutor._execute_impl`` has three exits that end a tool call
without running its body, and each now names itself:

- the pre-permission refusal (no executor resolvable at all),
- the permission denial, with the rule kind that produced it,
- the auto-background handoff.

Plus the two facts that bracket a call that DID run: which callable was
resolved, and what the model was handed back.

Deliberately kept apart from the permission plugin's own DECISION line:
this is the *consumer's* record, so a decision made by a plugin that
traces nothing of its own (a stub, a wrapper, an out-of-tree policy
engine) is still recorded where it takes effect.

The ledger record gains the asking session for the same reason the
trace line does — one permission plugin instance serves a parent and
every subagent sharing its registry, so a row naming only the tool
cannot be attributed to either.
"""

from typing import Any, Dict, List, Tuple

import pytest

from .. import ai_tool_runner
from ..ai_tool_runner import (
    ToolExecutor,
    _describe_executor,
    _permission_caller_fields,
    _summarize_result,
)


# Helpers ─────────────────────────────────────────────────────────────


@pytest.fixture
def traced(monkeypatch) -> List[str]:
    """Collect ``[TOOL_RUNNER]`` messages instead of writing a file."""
    lines: List[str] = []

    def _capture(component: str, msg: str, **_kwargs: Any) -> None:
        if component == "TOOL_RUNNER":
            lines.append(msg)

    monkeypatch.setattr(ai_tool_runner, "_trace_write", _capture)
    return lines


class _Policy:
    """Minimal permission plugin: one verdict, recorded verbatim."""

    def __init__(self, allowed: bool, method: str, reason: str = "because") -> None:
        self._verdict = (allowed, {"method": method, "reason": reason})
        self.calls: List[Tuple[str, Any]] = []

    def check_permission(self, name, args, context=None, call_id=None):
        self.calls.append((name, call_id))
        return self._verdict


class _Raising:
    def check_permission(self, name, args, context=None, call_id=None):
        raise RuntimeError("policy engine exploded")


def _executor(permission=None, context=None) -> ToolExecutor:
    ex = ToolExecutor()
    ex.register("echo", lambda args: {"got": args})
    if permission is not None:
        ex.set_permission_plugin(permission, context=context)
    return ex


def _only(lines: List[str], prefix: str) -> str:
    matching = [line for line in lines if line.startswith(prefix)]
    assert len(matching) == 1, f"expected one {prefix!r} line, got {matching}"
    return matching[0]


# The three ways a call ends without running ──────────────────────────


def test_denied_call_names_the_verdict_and_the_rule(traced) -> None:
    """A DENY that reaches the model as a ~10-token error now says so."""
    ex = _executor(_Policy(False, "default", "Denied by default policy"))

    success, result = ex.execute("echo", {"k": "v"}, call_id="call_1")

    assert success is False
    assert "Permission denied" in result["error"]
    verdict = _only(traced, "permission:")
    assert "verdict=DENY" in verdict
    assert "method=default" in verdict
    assert "call_id=call_1" in verdict
    # and what the model was handed
    assert "ok=False" in _only(traced, "result:")


def test_allowed_call_names_the_verdict_and_the_executor(traced) -> None:
    ex = _executor(_Policy(True, "whitelist"))

    success, _result = ex.execute("echo", {"k": "v"}, call_id="call_2")

    assert success is True
    assert "verdict=ALLOW" in _only(traced, "permission:")
    resolved = _only(traced, "resolve:")
    assert "executor=" in resolved
    assert "MISSING" not in resolved
    assert "ok=True" in _only(traced, "result:")


def test_allow_and_deny_traces_differ(traced) -> None:
    """The distinction the report could not make from any log."""
    _executor(_Policy(True, "whitelist")).execute("echo", {}, call_id="c")
    allowed = _only(traced, "permission:")

    traced.clear()
    _executor(_Policy(False, "default")).execute("echo", {}, call_id="c")
    denied = _only(traced, "permission:")

    assert allowed != denied


def test_unresolvable_tool_says_so_before_the_gate(traced) -> None:
    """Refused before the permission check — and it used to look the same.

    ``No executor registered for X`` is roughly the size of
    ``Permission denied: ...``; #951 measured both as
    ``result_tokens=10``.
    """
    ex = _executor(_Policy(True, "whitelist"))

    success, result = ex.execute("nosuchtool", {}, call_id="call_3")

    assert success is False
    assert result["error"] == "No executor registered for nosuchtool"
    line = _only(traced, "resolve:")
    assert "executor=MISSING" in line
    assert "before the permission check" in line
    # The gate was never consulted, so nothing may claim a verdict.
    assert not [msg for msg in traced if msg.startswith("permission:")]


def test_a_raising_permission_check_is_traced_as_a_denial(traced) -> None:
    """The runner fails closed; the trace names the cause, not just the effect."""
    ex = _executor(_Raising())

    success, result = ex.execute("echo", {}, call_id="call_4")

    assert success is False
    assert "Permission check failed" in result["error"]
    line = _only(traced, "permission:")
    assert "verdict=DENY" in line
    assert "method=check_failed" in line
    assert "RuntimeError: policy engine exploded" in line


def test_auto_background_handoff_is_traced(traced) -> None:
    """The third exit: approved, and its body runs somewhere else."""

    class _Backgroundable:
        name = "slowplugin"

        def get_auto_background_threshold(self, tool_name):
            return 0.01

    ex = _executor(_Policy(True, "whitelist"))
    ex._registry = object()  # only truthiness is read on this path
    ex._get_plugin_for_tool = lambda name: _Backgroundable()
    ex._execute_with_auto_background = (
        lambda name, args, plugin, threshold, meta: (True, {"task_id": "t1"})
    )

    success, result = ex.execute("echo", {}, call_id="call_5")

    assert (success, result) == (True, {"task_id": "t1"})
    line = _only(traced, "auto-background:")
    assert "plugin=slowplugin" in line
    assert "leaves this path" in line


# No payload in the log ───────────────────────────────────────────────


def test_result_trace_reports_shape_not_content(traced) -> None:
    """A trace file is not the place for the file the tool just wrote."""
    ex = ToolExecutor()
    ex.register("write", lambda args: {"success": True, "content": "S3CRET"})

    ex.execute("write", {}, call_id="call_6")

    line = _only(traced, "result:")
    assert "S3CRET" not in line
    assert "keys=content,success" in line


def test_summarize_result_surfaces_the_error_string() -> None:
    """The one value worth quoting: why it failed."""
    assert "error='nope'" in _summarize_result({"error": "nope"})
    assert _summarize_result("plain") == "str"


def test_describe_executor_is_explicit_about_absence() -> None:
    assert _describe_executor(None) == "MISSING"
    assert "test_describe_executor" in _describe_executor(
        test_describe_executor_is_explicit_about_absence)


# Ledger attribution ──────────────────────────────────────────────────


def test_ledger_permission_record_names_the_asking_session() -> None:
    """Which agent asked — the parent, or which subagent under it."""

    class _Ledger:
        def __init__(self) -> None:
            self.records: List[Tuple[str, Dict[str, Any]]] = []

        def _record(self, kind: str, payload: Dict[str, Any]) -> None:
            self.records.append((kind, payload))

    ledger = _Ledger()
    ex = ToolExecutor(ledger=ledger)
    ex.register("echo", lambda args: {"ok": True})
    ex.set_permission_plugin(
        _Policy(True, "whitelist"),
        context={"agent_type": "subagent",
                 "agent_name": "documentalista",
                 "session_id": "20260910_115239"},
    )

    ex.execute("echo", {}, call_id="call_7")

    kind, payload = ledger.records[0]
    assert kind == "permission-check"
    assert payload["agent_type"] == "subagent"
    assert payload["agent_name"] == "documentalista"
    assert payload["session_id"] == "20260910_115239"
    assert payload["method"] == "whitelist"


def test_caller_fields_omit_what_the_context_does_not_say() -> None:
    """Absent identity is no key, not a ``None`` — as for approver (#859)."""
    assert _permission_caller_fields(None) == {}
    assert _permission_caller_fields({"agent_type": "main"}) == {
        "agent_type": "main"}
    assert _permission_caller_fields(
        {"agent_name": None, "session_id": "s"}) == {"session_id": "s"}
