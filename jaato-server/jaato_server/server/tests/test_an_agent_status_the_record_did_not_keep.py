"""A status the wire carried and the record did not keep.

``AgentState.status`` has exactly one reader: the attach replay, which
tells a client arriving mid-session what each agent is doing.  The MAIN
agent's status emits called ``JaatoServer.emit()`` directly and never
touched the field, so it stayed at the ``"idle"`` it is constructed with
for the whole life of the session -- while the subagent path
(``on_agent_status_changed``) stamped it before emitting and was correct.

So a client that attached to a session whose model thread was running was
told the main agent was idle, and stayed told until the turn ended.  A
second browser tab, a reconnect, a re-attach after a detach: the web
client's working indicator never came on, and nothing in the event stream
said why.

``emit_agent_status`` is the one door: it records and then emits, so the
two halves cannot disagree again.  The AST guard is the load-bearing half
-- a behavioural test can only exercise a call site somebody thought of,
and the defect is a call site nobody did.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, List

import pytest

CORE = Path(__file__).resolve().parents[1] / "core.py"

try:  # pragma: no cover - import shape differs per invocation
    from jaato_server.shared.tests.reversion import (
        Reversion,
    )
except Exception:  # pragma: no cover
    Reversion = None  # type: ignore[assignment]


_CORE = "jaato-server/jaato_server/server/core.py"


REVERSIONS = [] if Reversion is None else [
    Reversion(
        target=_CORE,
        find=(
            '        agent = getattr(self, "_agents", {}).get(agent_id)\n'
            "        if agent is not None:\n"
            "            agent.status = status\n"
        ),
        replace="",
        test="test_the_record_follows_the_wire",
        because=(
            "the one door stops recording the status it announces, so the "
            "attach replay is back to reporting a main agent that is always "
            "idle"
        ),
    ),
    Reversion(
        target=_CORE,
        find='        self.emit_agent_status(self._main_agent_id, "active")',
        replace=(
            "        self.emit(AgentStatusChangedEvent(\n"
            "            agent_id=self._main_agent_id,\n"
            '            status="active",\n'
            "        ))"
        ),
        test="test_every_status_report_goes_through_the_door",
        because="a status report goes around the door again",
    ),
]


# ----------------------------------------------------------------------
# The record follows the wire
# ----------------------------------------------------------------------


class _Sink:
    """Collects what would go to clients."""

    def __init__(self) -> None:
        self.events: List[Any] = []


def _server(sink: _Sink) -> Any:
    from jaato_server.server.core import JaatoServer

    srv = JaatoServer(workspace_path=None, session_id="status-record")
    srv.emit = sink.events.append  # type: ignore[method-assign]
    return srv


def test_the_record_follows_the_wire() -> None:
    """A status the main agent reported is the status a later attach sees."""
    from jaato_server.server.core import AgentState

    sink = _Sink()
    srv = _server(sink)
    main = srv._main_agent_id
    srv._agents[main] = AgentState(agent_id=main, name="main", agent_type="main")
    assert srv._agents[main].status == "idle"

    srv.emit_agent_status(main, "active")
    assert srv._agents[main].status == "active", (
        "the attach replay reads this field; leaving it at 'idle' is what "
        "made a mid-turn attach report an idle agent"
    )
    assert sink.events[-1].status == "active"

    srv.emit_agent_status(main, "done")
    assert srv._agents[main].status == "done"


def test_an_unknown_agent_is_still_announced() -> None:
    """No record is not a reason to withhold the event.

    A status for an agent this server never registered is a client-facing
    fact about the session; dropping it would trade a stale indicator for
    a missing one.
    """
    sink = _Sink()
    srv = _server(sink)
    srv.emit_agent_status("sub-nobody-knows", "active")
    assert sink.events[-1].agent_id == "sub-nobody-knows"
    assert sink.events[-1].status == "active"


def test_the_error_rides_along() -> None:
    sink = _Sink()
    srv = _server(sink)
    srv.emit_agent_status("sub-1", "error", "boom")
    assert sink.events[-1].error == "boom"


# ----------------------------------------------------------------------
# The guard: nothing reports a status around the door
# ----------------------------------------------------------------------


def _is_emit_call(node: ast.AST) -> bool:
    """Is this a call to ``.emit(...)`` / ``emit(...)``?"""
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
    return name == "emit"


def _claims_a_new_status(call: ast.Call) -> bool:
    """Does this emit call CONSTRUCT a status rather than re-read one?

    The attach replay passes ``status=agent.status`` -- a read of the record,
    not a new claim about it -- so it is excluded by construction rather than
    by being named.
    """
    for arg in call.args:
        if not isinstance(arg, ast.Call):
            continue
        if getattr(arg.func, "id", "") != "AgentStatusChangedEvent":
            continue
        status = {k.arg: k.value for k in arg.keywords}.get("status")
        if isinstance(status, ast.Attribute) and status.attr == "status":
            continue
        return True
    return False


def _status_emits_outside_the_door(source: str | None = None) -> List[int]:
    """Lines constructing an ``AgentStatusChangedEvent`` inside an emit call.

    ``emit_agent_status``'s own body is the one legitimate site; everything
    else that reports a status must go through it, or the record and the wire
    can disagree about an agent again (#1139).
    """
    tree = ast.parse(source if source is not None else CORE.read_text())
    door = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "emit_agent_status"
    )
    inside = {id(n) for n in ast.walk(door)}
    return [
        node.lineno
        for node in ast.walk(tree)
        if _is_emit_call(node)
        and id(node) not in inside
        and _claims_a_new_status(node)
    ]


def test_every_status_report_goes_through_the_door() -> None:
    offenders = _status_emits_outside_the_door()
    assert not offenders, (
        "server/core.py line(s) "
        + ", ".join(str(n) for n in offenders)
        + " emit an AgentStatusChangedEvent without recording it on "
          "AgentState.status.  Call emit_agent_status() instead -- that "
          "divergence is #1139: the attach replay then reports a status "
          "nobody is in."
    )


def test_the_guard_can_fail() -> None:
    """A guard that cannot fail proves nothing."""
    source = CORE.read_text().replace(
        "        self.emit_agent_status(self._main_agent_id, \"active\")",
        "        self.emit(AgentStatusChangedEvent(\n"
        "            agent_id=self._main_agent_id,\n"
        "            status=\"active\",\n"
        "        ))",
        1,
    )
    assert source != CORE.read_text(), "the reversion did not apply"
    assert _status_emits_outside_the_door(source), (
        "the scan did not flag a status emitted around the door, so the "
        "guard above proves nothing"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
