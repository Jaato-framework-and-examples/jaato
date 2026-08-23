"""Budget usage survives the FULL persistence round trip.

PR #580 wired four seams and I verified each: restore works given a snapshot,
the load path calls it, a restored tracker fires where a zeroed one does not.
All true. The feature still did not work, because a FIFTH seam was never
enumerated -- ``serializer.py`` writes a fixed key list, so ``budget_usage``
was dropped on the way to disk and the persisted JSON carried no such key.
Every half was correct; nothing produced a snapshot for the other half to
restore.

Reported by the cascade author, who ran a real suspend instead of trusting the
merge: zero ``session.restore_budget_usage`` calls, and no ``budget_usage`` key
in the session JSON.

The lesson, in test form: *a test that asserts restore works, given a snapshot,
cannot fail when nothing produces a snapshot.* So this pins the ROUND TRIP --
usage on the clock, through serialization, back into a tracker -- rather than
either half alone.
"""
from datetime import datetime, timezone

import pytest

from shared.budget_control import BudgetControlConfig, BudgetTracker
from shared.jaato_session import JaatoSession
from shared.plugins.session.base import SessionState
from shared.plugins.session.serializer import (
    deserialize_session_state,
    serialize_session_state,
)


def _state(**kw):
    now = datetime.now(timezone.utc)
    return SessionState(session_id="s1", history=[], created_at=now,
                        updated_at=now, **kw)


def _tracker(limits):
    return BudgetTracker(BudgetControlConfig.from_dict(
        {"limits": limits, "degrade": [{"at": "100%", "action": "abort"}]}))


class TestSerializationCarriesIt:
    def test_usage_reaches_the_persisted_form(self):
        """THE bug: the key was absent from the JSON entirely."""
        payload = serialize_session_state(_state(budget_usage={"turns": 2.0}))
        assert "budget_usage" in payload, (
            "serializer writes a fixed key list -- a field added to "
            "SessionState alone never reaches disk"
        )
        assert payload["budget_usage"] == {"turns": 2.0}

    def test_usage_comes_back_out(self):
        payload = serialize_session_state(_state(budget_usage={"turns": 2.0}))
        assert deserialize_session_state(payload).budget_usage == {"turns": 2.0}

    def test_absent_key_deserialises_to_none(self):
        """Sessions persisted before this field existed must still load."""
        payload = serialize_session_state(_state())
        payload.pop("budget_usage", None)
        assert deserialize_session_state(payload).budget_usage is None

    def test_it_does_not_collide_with_the_conversation_budget(self):
        """budget_state is a DIFFERENT subsystem; both must survive."""
        payload = serialize_session_state(
            _state(budget_usage={"turns": 1.0}, budget_state={"tokens": 99}))
        restored = deserialize_session_state(payload)
        assert restored.budget_usage == {"turns": 1.0}
        assert restored.budget_state == {"tokens": 99}


class TestTheWholeChain:
    def test_a_ceiling_crossed_before_a_reload_is_still_crossed_after(self):
        """Usage on the clock -> disk -> tracker -> the rung fires.

        This is the property the two half-tests could not express between
        them: it fails if EITHER the snapshot is dropped on write or the
        restore is never called.
        """
        limits = {"turns": 2}
        live = _tracker(limits)
        live.observe(turns=1)                       # 1 of 2 spent

        # ---- save: snapshot -> SessionState -> persisted form ----
        session = JaatoSession.__new__(JaatoSession)
        session._budget_tracker = live
        session._turn_accounting = []
        snapshot = JaatoSession.get_budget_usage(session)
        payload = serialize_session_state(_state(budget_usage=snapshot))

        # ---- load: persisted form -> SessionState -> fresh tracker ----
        restored_state = deserialize_session_state(payload)
        reloaded = JaatoSession.__new__(JaatoSession)
        reloaded._budget_tracker = _tracker(limits)
        JaatoSession.restore_budget_usage(reloaded, restored_state.budget_usage)

        fired = reloaded._budget_tracker.observe(turns=1)
        assert [r.at_percent for r in fired] == [100.0], (
            "the ceiling did not survive the round trip -- a second turn "
            "after a reload must still cross a turns:2 limit"
        )

    def test_the_same_chain_without_persistence_does_not_fire(self):
        """Control: proves the assertion above is testing the round trip."""
        reloaded = _tracker({"turns": 2})           # no restore at all
        assert not reloaded.observe(turns=1)
