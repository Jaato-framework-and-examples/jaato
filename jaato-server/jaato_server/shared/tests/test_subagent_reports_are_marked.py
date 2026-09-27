"""A subagent's report reaches clients as a report, not as the user's turn.

A subagent reports to its parent with ``inject_prompt(...,
source_type=SourceType.CHILD)``.  The queue knew it was a CHILD, but a
client never did: live, the report became a continuation turn the daemon
started without an echo, so nothing showed; on attach, the replay drew it
as a ``user`` unit, so the web client showed it as the user's own bubble.

``shared.subagent_report`` is the one rule both paths now read:

- the live echo, ``JaatoServer._echo_subagent_report`` (``source="child"``);
- the paged replay, ``history_pages`` (``origin: "subagent"`` on the unit);
- the full replay, ``_emit_replay_unit`` (``source="child"``).

The rule reads the marker, because the stored history carries no source
type.  So the first test here is the one that keeps the rule true: every
CHILD ``inject_prompt`` in the tree must begin with the marker.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

from jaato_sdk.plugins.model_provider.types import Message, Part, Role
from jaato_server.server.core import JaatoServer
from jaato_server.server.history_pages import build_units
from jaato_server.shared.subagent_report import (
    SUBAGENT_REPORT_MARKER, SUBAGENT_REPORT_ORIGIN, SUBAGENT_REPORT_SOURCE,
    is_subagent_report,
)
from jaato_server.shared.tests.reversion import Reversion

_PKG = Path(__file__).resolve().parents[2]  # .../jaato_server
_CORE = "jaato-server/jaato_server/server/core.py"
_PAGES = "jaato-server/jaato_server/server/history_pages.py"

REVERSIONS = [
    Reversion(
        target=_PAGES,
        find="origin=(SUBAGENT_REPORT_ORIGIN\n",
        replace="origin=(\"\"\n",
        test="test_a_report_in_history_is_a_subagent_unit",
        because="the paged replay draws a report as the user's own turn",
    ),
    Reversion(
        target=_CORE,
        find="                    server._echo_subagent_report(child_messages)\n",
        replace="",
        test="test_the_continuation_branch_echoes_the_report",
        because="a report reaches the agent live and no client sees it",
    ),
    Reversion(
        target=_CORE,
        find="            source = (SUBAGENT_REPORT_SOURCE\n",
        replace="            source = (unit.kind\n",
        test="test_the_full_replay_sends_a_report_as_child",
        because="the TUI's full replay draws a report as the user's turn",
    ),
]


# -- the rule's premise: every CHILD report carries the marker -------------

def _is_child_source(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "source_type":
            v = kw.value
            return isinstance(v, ast.Attribute) and v.attr == "CHILD"
    return False


def _leading_text(node: ast.AST, func: Optional[ast.AST]) -> Optional[str]:
    """The literal text a message expression starts with, when knowable."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values:
        first = node.values[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _leading_text(node.left, func)
    if isinstance(node, ast.Name) and func is not None:
        for sub in ast.walk(func):
            if (isinstance(sub, ast.Assign) and len(sub.targets) == 1
                    and isinstance(sub.targets[0], ast.Name)
                    and sub.targets[0].id == node.id):
                return _leading_text(sub.value, None)
    return None


def _child_injections() -> List[tuple]:
    found = []
    for path in sorted(_PKG.rglob("*.py")):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call in ast.walk(func):
                if (isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "inject_prompt"
                        and call.args and _is_child_source(call)):
                    found.append((path, call.lineno,
                                  _leading_text(call.args[0], func)))
    return found


def test_every_child_report_begins_with_the_marker():
    found = _child_injections()
    # Non-vacuous: the subagent plugin, the permission and clarification
    # channels and the session itself all report this way.
    assert len(found) >= 8, found
    bad = [f"{p.relative_to(_PKG)}:{line} starts {text!r}"
           for p, line, text in found
           if not (text or "").startswith(SUBAGENT_REPORT_MARKER)]
    assert not bad, (
        "a CHILD inject_prompt that does not start with "
        f"{SUBAGENT_REPORT_MARKER!r} reaches clients as the user's turn:\n"
        + "\n".join(bad))


def test_the_rule():
    assert is_subagent_report("[SUBAGENT agent_id=s1 event=COMPLETED]\ndone")
    assert is_subagent_report("  \n[SUBAGENT agent_id=s1 source=model]\nhi")
    assert is_subagent_report("<hidden>x</hidden>[SUBAGENT agent_id=s1 event=IDLE]")
    assert not is_subagent_report("please check [SUBAGENT agent_id=s1 event=IDLE]")
    assert not is_subagent_report("")
    assert not is_subagent_report(None)


# -- the three places it is applied -----------------------------------------

def _user(text: str) -> Message:
    return Message(role=Role.USER, parts=[Part(text=text)])


def test_a_report_in_history_is_a_subagent_unit():
    units = build_units([
        _user("fix the bug"),
        _user("[SUBAGENT agent_id=s1 event=COMPLETED]\nfound it"),
    ])
    dicts = [u.to_dict(i) for i, u in enumerate(units)]
    assert [d["kind"] for d in dicts] == ["user", "user"]
    assert "origin" not in dicts[0]
    assert dicts[1]["origin"] == SUBAGENT_REPORT_ORIGIN


def test_the_origin_does_not_move_a_cursor():
    report = "[SUBAGENT agent_id=s1 event=COMPLETED]\nfound it"
    unit = build_units([_user(report)])[0]
    plain = build_units([_user("found it")])[0]
    assert unit.origin and not plain.origin
    unit.origin = ""
    assert unit.digest == build_units([_user(report)])[0].digest


def test_the_full_replay_sends_a_report_as_child():
    events: List[Any] = []
    units = build_units([
        _user("fix the bug"),
        _user("[SUBAGENT agent_id=s1 event=COMPLETED]\nfound it"),
    ])
    for u in units:
        JaatoServer._emit_replay_unit(events.append, "main", u, None)
    assert [e.source for e in events] == ["user", SUBAGENT_REPORT_SOURCE]


def _fake_server() -> SimpleNamespace:
    emitted: List[Any] = []
    return SimpleNamespace(_main_agent_id="main", emit=emitted.append,
                           emitted=emitted)


def test_the_live_echo_marks_a_report_and_leaves_a_person_alone():
    fake = _fake_server()
    JaatoServer._echo_subagent_report(
        fake, "[SUBAGENT agent_id=s1 event=COMPLETED]\ndone")
    JaatoServer._echo_subagent_report(fake, "what the user queued")
    assert [(e.source, e.agent_id, e.mode) for e in fake.emitted] == [
        (SUBAGENT_REPORT_SOURCE, "main", "write")]


def test_the_continuation_branch_echoes_the_report():
    """The echo is only real if the notification handler calls it."""
    tree = ast.parse((_PKG / "server" / "core.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (isinstance(test, ast.Compare)
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "continuation_needed"):
            calls = [c.func.attr for c in ast.walk(node)
                     if isinstance(c, ast.Call)
                     and isinstance(c.func, ast.Attribute)]
            assert "_echo_subagent_report" in calls
            return
    raise AssertionError("no continuation_needed branch in core.py")
