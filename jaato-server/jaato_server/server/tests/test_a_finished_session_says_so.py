"""A session that finished is listed as finished (record 2.11, protocol 1.29).

Before this, nothing on a record said a session had ended.  The web
client's End deleted the record, so a session a person chose to end could
only be kept as a sleeping one.  ``server.session_finished`` marks a
session when a ``SessionTerminatedEvent`` names a finishing reason (the
person ended it, the agent completed, the budget stopped it) and clears the
mark when a turn starts again; the record and both session listings carry
it.
"""

from __future__ import annotations

import threading
from datetime import datetime
from types import SimpleNamespace

import pytest

from jaato_sdk.events import (
    AgentOutputEvent, AgentStatusChangedEvent, SessionTerminatedEvent,
)
from jaato_server.server.session_finished import (
    FINISHED_REASONS, note_lifecycle,
)
from jaato_server.server.session_manager import (
    RuntimeSessionInfo, SessionManager, session_picker_fields,
)
from jaato_server.shared.plugins.session.base import SessionState
from jaato_server.shared.plugins.session.serializer import (
    deserialize_session_info, deserialize_session_state,
    serialize_session_state,
)
from jaato_server.shared.tests.reversion import Reversion

_FINISHED = "jaato-server/jaato_server/server/session_finished.py"
_MANAGER = "jaato-server/jaato_server/server/session_manager.py"
_SERIALIZER = "jaato-server/jaato_server/shared/plugins/session/serializer.py"

REVERSIONS = [
    Reversion(
        target=_MANAGER,
        find="                note_session_lifecycle(session, event)\n",
        replace="",
        test="test_a_routed_terminal_marks_the_loaded_session",
        because="the daemon never records that a session finished",
    ),
    Reversion(
        target=_FINISHED,
        find='    "client_request", "stopped", "natural", "budget_exhausted",\n',
        replace='    "client_request", "stopped", "natural", "budget_exhausted",\n'
                '    "error",\n',
        test="test_only_the_finishing_reasons_mark_a_session",
        because="a failed session is buried under Finished",
    ),
    Reversion(
        target=_FINISHED,
        find="    if reopens(event, main):\n",
        replace="    if False:\n",
        test="test_a_turn_starting_reopens_a_finished_session",
        because="a session that runs again stays listed as finished",
    ),
    Reversion(
        target=_SERIALIZER,
        find="        ended_at=data.get('ended_at'),  # None on pre-2.11 records\n        end_reason=data.get('end_reason'),\n    )\n\n\ndef serialize_session_info",
        replace="    )\n\n\ndef serialize_session_info",
        test="test_the_mark_survives_the_record",
        because="a finished session comes back unfinished after a restart",
    ),
]


def _record(**kw) -> SimpleNamespace:
    base = dict(ended_at=None, end_reason=None, is_dirty=False,
                server=SimpleNamespace(_main_agent_id="main"))
    base.update(kw)
    return SimpleNamespace(**base)


def _terminal(reason: str) -> SessionTerminatedEvent:
    return SessionTerminatedEvent(session_id="s1", agent_id="main",
                                  reason=reason)


@pytest.mark.parametrize("reason", sorted(FINISHED_REASONS))
def test_a_finishing_terminal_marks_the_session(reason):
    s = _record()
    assert note_lifecycle(s, _terminal(reason)) is True
    assert s.end_reason == reason
    assert datetime.fromisoformat(s.ended_at).tzinfo is not None
    assert s.is_dirty


def test_only_the_finishing_reasons_mark_a_session():
    assert FINISHED_REASONS == {
        "client_request", "stopped", "natural", "budget_exhausted"}
    for reason in ("error", "cascade_cancelled", "operator_request",
                   "max_session_seconds", "max_orphan_seconds"):
        s = _record()
        assert note_lifecycle(s, _terminal(reason)) is False
        assert s.ended_at is None and s.end_reason is None


def test_a_turn_starting_reopens_a_finished_session():
    s = _record(ended_at="2026-09-27T10:00:00+00:00", end_reason="natural")
    # Reading it (output, a subagent going active) keeps it finished.
    note_lifecycle(s, AgentOutputEvent(agent_id="main", source="model",
                                       text="x", mode="write"))
    note_lifecycle(s, AgentStatusChangedEvent(agent_id="sub1",
                                              status="active"))
    assert s.end_reason == "natural"
    # Its own main agent starting a turn does not.
    note_lifecycle(s, AgentStatusChangedEvent(agent_id="main",
                                              status="active"))
    assert s.ended_at is None and s.end_reason is None
    assert s.is_dirty


def test_the_mark_survives_the_record():
    state = SessionState(
        session_id="s1", history=[], created_at=datetime(2026, 9, 27),
        updated_at=datetime(2026, 9, 27),
        ended_at="2026-09-27T10:00:00+00:00", end_reason="budget_exhausted",
    )
    data = serialize_session_state(state)
    assert data["version"] == "2.11"
    back = deserialize_session_state(data)
    assert (back.ended_at, back.end_reason) == (
        "2026-09-27T10:00:00+00:00", "budget_exhausted")
    info = deserialize_session_info(data)
    assert (info.ended_at, info.end_reason) == (
        "2026-09-27T10:00:00+00:00", "budget_exhausted")


def test_an_older_record_is_not_finished():
    data = serialize_session_state(SessionState(
        session_id="s1", history=[], created_at=datetime(2026, 9, 27),
        updated_at=datetime(2026, 9, 27)))
    data.pop("ended_at"), data.pop("end_reason")
    data["version"] = "2.10"
    assert deserialize_session_state(data).ended_at is None
    assert deserialize_session_info(data).end_reason is None


def test_the_listing_row_carries_it():
    row = RuntimeSessionInfo(
        session_id="s1", name="n", description=None, created_at="",
        last_activity="", model_provider="", model_name="",
        is_processing=False, is_loaded=False, client_count=0, turn_count=0,
        ended_at="2026-09-27T10:00:00+00:00", end_reason="client_request")
    fields = session_picker_fields(row)
    assert fields["ended_at"] == "2026-09-27T10:00:00+00:00"
    assert fields["end_reason"] == "client_request"
    open_row = session_picker_fields(SimpleNamespace())
    assert open_row["ended_at"] is None and open_row["end_reason"] is None


def _bare_manager(session) -> SessionManager:
    sm = SessionManager.__new__(SessionManager)
    sm._lock = threading.RLock()
    sm._sessions = {"s1": session}
    sm._workspace_monitors = {}
    sm._cascade_clients = {}
    sm._cascade_clients_lock = threading.Lock()
    sm._handle_turn_tracking_event = lambda *a, **k: None
    sm._accumulate_cascade_budget = lambda *a, **k: None
    sm._emit_to_client = lambda *a, **k: None
    sm._apply_default_cascade_policy = lambda *a, **k: None
    return sm


def test_a_routed_terminal_marks_the_loaded_session():
    session = _record(session_id="s1", attached_clients=set(),
                      cascade_driver_id=None)
    sm = _bare_manager(session)
    sm._emit_to_session("s1", _terminal("client_request"))
    assert session.end_reason == "client_request"
    assert session.is_dirty
