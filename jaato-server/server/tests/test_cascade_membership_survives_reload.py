"""Cascade membership must survive an unload, and a COLD sibling keeps its name.

Three defects, one root.  ``cascade_driver_id`` was never persisted:
``Session.cascade_driver_id`` is set at create and re-supplied by the CALLER of
``wake_session``, so a stage that unloaded on ORPHAN and was later LOADED
(attach / disk-restore rather than an explicit wake) came back with ``None``
and silently left its cascade.

That made ``sibling_name`` (#592) incoherent — the ADDRESS survived a reload
while the MEMBERSHIP did not, so a revived sibling held a name belonging to a
cascade it was no longer in.  And because the uniqueness check read only the
in-memory table, the address could be reissued the moment its owner went cold.

Reachable in practice because a cascade is NOT fixed at creation: a premium
reactor rule matching an agent-caused event can read the cid off it and mint
further cid-stamped sessions later
(``jaato_premium/reactors/action_context.py`` takes ``cascade_driver_id``;
``engine.py`` matches ``event_type == "agent.completed"``).  The operator
authors the rule, so the topology is operator-chosen — but the COUNT and the
TIMING are the agent's.
"""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from server.session_manager import SessionManager, validate_sibling_name
from shared.plugins.session.base import SessionState
from shared.plugins.session import serializer as S


def _state(**kw):
    now = datetime.now(timezone.utc)
    base = dict(session_id="s1", history=[], created_at=now, updated_at=now,
                turn_count=0)
    base.update(kw)
    return SessionState(**base)


# ------------------------------------------------- 1. membership persists

def test_cascade_membership_round_trips():
    st = _state(cascade_driver_id="cid-1", sibling_name="reviewer")
    back = S.deserialize_session_state(S.serialize_session_state(st))
    assert back.cascade_driver_id == "cid-1", (
        "the session came back outside its cascade — its address now names a "
        "cascade it is not in, and cid-routed events no longer reach it")
    assert back.sibling_name == "reviewer"


def test_the_restore_path_puts_the_session_back_in_its_cascade():
    """Structural: _load_session_impl needs a live daemon to call."""
    import inspect
    src = inspect.getsource(SessionManager._load_session_impl)
    assert 'cascade_driver_id=getattr(state, "cascade_driver_id", None)' in src, (
        "disk-restore does not restore cascade membership")


def test_the_save_path_captures_it():
    import inspect
    src = inspect.getsource(SessionManager._save_session)
    assert "cascade_driver_id=session.cascade_driver_id" in src, (
        "membership is never written, so restoring it cannot help")


# --------------------------------------------- 2. the INDEX carries both

def test_the_lightweight_index_carries_membership_and_address():
    """A cold sibling is read from the INDEX, not the full state."""
    st = _state(cascade_driver_id="cid-1", sibling_name="reviewer")
    info = S.deserialize_session_info(S.serialize_session_info(st))
    assert info.sibling_name == "reviewer", (
        "the index omits sibling_name, so a cold sibling is invisible to both "
        "the roster and the uniqueness check")
    assert info.cascade_driver_id == "cid-1"


# ------------------------------------- 3. uniqueness sees COLD siblings

def _mgr(live=(), persisted=()):
    mgr = SimpleNamespace(
        _sessions={
            f"live-{i}": SimpleNamespace(sibling_name=n, cascade_driver_id=c)
            for i, (n, c) in enumerate(live)},
        _get_persisted_sessions=lambda workspace_path=None: [
            SimpleNamespace(session_id=f"cold-{i}", sibling_name=n,
                            cascade_driver_id=c)
            for i, (n, c) in enumerate(persisted)],
    )
    return mgr


def test_a_cold_sibling_still_owns_its_address():
    """THE reactor scenario, and the one that fails on everything merged.

    reviewer runs → unloads on ORPHAN → a reactor mints a new cid-stamped
    session asking for the same name.
    """
    mgr = _mgr(live=[], persisted=[("reviewer", "cid-1")])
    claimed = SessionManager._known_sibling_addresses(mgr, None)
    assert validate_sibling_name("reviewer", "cid-1", claimed) is not None, (
        "a cold sibling's address was reissued — when it revives, one cascade "
        "has two sessions answering to one name")


def test_a_live_sibling_still_blocks_too():
    mgr = _mgr(live=[("reviewer", "cid-1")])
    claimed = SessionManager._known_sibling_addresses(mgr, None)
    assert validate_sibling_name("reviewer", "cid-1", claimed) is not None


def test_a_cold_sibling_in_ANOTHER_cascade_does_not_block():
    mgr = _mgr(persisted=[("reviewer", "cid-other")])
    claimed = SessionManager._known_sibling_addresses(mgr, None)
    assert validate_sibling_name("reviewer", "cid-1", claimed) is None


def test_a_session_counted_twice_is_not_a_false_collision():
    """Loaded sessions appear in BOTH views; the union must dedupe by id."""
    mgr = SimpleNamespace(
        _sessions={"s1": SimpleNamespace(sibling_name="reviewer",
                                         cascade_driver_id="cid-1")},
        _get_persisted_sessions=lambda workspace_path=None: [
            SimpleNamespace(session_id="s1", sibling_name="reviewer",
                            cascade_driver_id="cid-1")],
    )
    claimed = SessionManager._known_sibling_addresses(mgr, None)
    assert len([c for c in claimed if c == ("reviewer", "cid-1")]) == 1


def test_an_unreadable_index_warns_rather_than_silently_narrowing(caplog):
    def _boom(workspace_path=None):
        raise OSError("index unreadable")

    mgr = SimpleNamespace(_sessions={}, _get_persisted_sessions=_boom)
    with caplog.at_level("WARNING"):
        claimed = SessionManager._known_sibling_addresses(mgr, None)
    assert claimed == []
    assert any("COLD sibling" in r.getMessage() for r in caplog.records), (
        "falling back to the in-memory view silently is how a duplicate "
        "address gets issued")
