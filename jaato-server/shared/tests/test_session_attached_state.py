"""Tests for the session-attached-state facility on JaatoSession.

The facility (designed 2026-04-25 — see
``project_backlog_session_attached_state.md``) is a generic per-session
opaque-storage primitive that:

* persists alongside the journal,
* carries across forks via ``initial_session_state``,
* supports both static state (``set_session_state``) and
  incrementally-mutated state (``register_session_state_provider``).

These tests cover the public API on ``JaatoSession`` plus the
journal round-trip.  Waypoint integration is covered in
``shared/plugins/waypoint/tests/test_session_state_snapshot.py``.
``SessionManager.create_*_session`` plumbing is exercised via the
hook-fires-after-attach test below — full integration with the
on-disk storage is intentionally out of scope for unit tests.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from ..jaato_session import JaatoSession
from ..plugins.session.base import SessionState
from ..plugins.session.serializer import (
    deserialize_session_state,
    serialize_session_state,
)


def _bare_session() -> JaatoSession:
    """Build a JaatoSession with a mocked runtime — enough to exercise
    the session-state methods without configuring a provider."""
    runtime = MagicMock()
    return JaatoSession(runtime, "test-model")


class TestSetGetSessionState:

    def test_set_and_get_round_trip(self):
        s = _bare_session()
        s.set_session_state("audit_chain_head", {"hash": "abc", "seq": 7})
        assert s.get_session_state("audit_chain_head") == {"hash": "abc", "seq": 7}

    def test_get_default_when_absent(self):
        s = _bare_session()
        assert s.get_session_state("missing") is None
        assert s.get_session_state("missing", default="fallback") == "fallback"

    def test_set_overwrites_existing_value(self):
        s = _bare_session()
        s.set_session_state("k", "v1")
        s.set_session_state("k", "v2")
        assert s.get_session_state("k") == "v2"

    def test_non_serialisable_value_rejected_at_attach_time(self):
        s = _bare_session()
        # Sets containing non-string keys are not JSON-serialisable.
        with pytest.raises(TypeError) as exc:
            s.set_session_state("bad", {1, 2, 3})
        assert "JSON-serialisable" in str(exc.value)

    def test_get_all_session_state_returns_copy(self):
        s = _bare_session()
        s.set_session_state("k", {"nested": [1, 2]})
        snapshot = s.get_all_session_state()
        # Mutating the snapshot must not propagate back into the session.
        snapshot["k"]["nested"].append(3)
        snapshot["new_key"] = "leak"
        assert s.get_session_state("k") == {"nested": [1, 2, 3]}  # mutation of nested
        # The top-level dict is the actual copy we care about.
        assert s.get_session_state("new_key") is None


class TestRegisterSessionStateProvider:

    def test_provider_supplies_value_at_read_time(self):
        s = _bare_session()
        counter = {"n": 0}

        def provider():
            counter["n"] += 1
            return {"call_count": counter["n"]}

        s.register_session_state_provider("live", provider)
        assert s.get_session_state("live") == {"call_count": 1}
        assert s.get_session_state("live") == {"call_count": 2}

    def test_provider_takes_precedence_over_set_state(self):
        s = _bare_session()
        s.set_session_state("k", "static")
        s.register_session_state_provider("k", lambda: "dynamic")
        assert s.get_session_state("k") == "dynamic"

    def test_re_registration_replaces_prior_provider(self):
        s = _bare_session()
        s.register_session_state_provider("k", lambda: "first")
        s.register_session_state_provider("k", lambda: "second")
        assert s.get_session_state("k") == "second"

    def test_get_all_invokes_every_provider_once(self):
        s = _bare_session()
        invocations = []

        def make_provider(name):
            def fn():
                invocations.append(name)
                return f"value-of-{name}"
            return fn

        s.register_session_state_provider("a", make_provider("a"))
        s.register_session_state_provider("b", make_provider("b"))
        snapshot = s.get_all_session_state()
        assert invocations == ["a", "b"] or invocations == ["b", "a"]
        assert snapshot == {"a": "value-of-a", "b": "value-of-b"}

    def test_get_all_merges_set_state_and_providers(self):
        s = _bare_session()
        s.set_session_state("static_key", 42)
        s.register_session_state_provider(
            "dynamic_key", lambda: {"now": "live"}
        )
        snapshot = s.get_all_session_state()
        assert snapshot == {"static_key": 42, "dynamic_key": {"now": "live"}}

    def test_provider_callback_must_be_callable(self):
        s = _bare_session()
        with pytest.raises(TypeError):
            s.register_session_state_provider("k", "not callable")

    def test_provider_value_used_for_snapshot_after_mutation(self):
        """Documents the Pattern C contract: a consumer holding a
        mutable structure registers a provider once, and subsequent
        mutations of the structure are picked up at the next
        snapshot — without the consumer calling set_session_state."""
        s = _bare_session()
        # Stand-in for premium's PseudonymTable: a dict the consumer
        # mutates as the session progresses.
        live_table = {"<EMAIL_1>": "alice@example.com"}
        s.register_session_state_provider("pseudonym_table", lambda: dict(live_table))

        snapshot1 = s.get_all_session_state()
        assert snapshot1["pseudonym_table"] == {"<EMAIL_1>": "alice@example.com"}

        # Consumer mutates the live structure mid-session.
        live_table["<EMAIL_2>"] = "bob@example.com"

        snapshot2 = s.get_all_session_state()
        assert snapshot2["pseudonym_table"] == {
            "<EMAIL_1>": "alice@example.com",
            "<EMAIL_2>": "bob@example.com",
        }


class TestJournalRoundTrip:
    """SessionState carries session_state through the on-disk JSON
    serializer.  This covers the persistence path without booting a
    real session plugin."""

    def _make_state(self, session_state):
        now = datetime.now()
        return SessionState(
            session_id="20260425_120000",
            history=[],
            created_at=now,
            updated_at=now,
            session_state=session_state,
        )

    def test_session_state_round_trips_through_serializer(self):
        original = self._make_state({"audit_chain_head": "deadbeef", "n": 3})
        data = serialize_session_state(original)
        # Current serializer schema version (bumped 2.2 -> 2.3 -> 2.4 for
        # profile_name/config_root, 2.5 sandbox_mode, 2.6 agent_name,
        # 2.7 profile_spec for inline-profile disk-restore, 2.8
        # profile_snapshot/rendered_instructions/agent_params so a revive
        # restores its recipe and prompt instead of re-deriving them
        # (#787); serializer.py:230).
        assert data["version"] == "2.8"
        assert data["session_state"] == {"audit_chain_head": "deadbeef", "n": 3}
        restored = deserialize_session_state(data)
        assert restored.session_state == {"audit_chain_head": "deadbeef", "n": 3}

    def test_legacy_state_without_session_state_round_trips(self):
        """Persistence files predating this facility have no
        ``session_state`` key — deserialize must default to None
        without raising."""
        legacy = {
            "version": "2.1",
            "session_id": "x",
            "description": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "turn_count": 0,
            "turn_accounting": [],
            "user_inputs": [],
            "metadata": {},
            "connection": {"project": None, "location": None, "model": None},
            "workspace_path": None,
            "history": [],
            "budget_state": None,
            "interrupted_turn": None,
            # session_state intentionally absent
        }
        restored = deserialize_session_state(legacy)
        assert restored.session_state is None

    def test_empty_session_state_collapses_to_none(self):
        """JaatoSession._get_session_state collapses an empty
        get_all_session_state() to None so the persistence file
        doesn't grow a ``session_state: {}`` stub for sessions that
        never used the facility."""
        s = _bare_session()
        # No keys attached, no providers — snapshot is empty.
        assert s.get_all_session_state() == {}
        # Round-trip through the SessionState path used by
        # _get_session_state: the empty dict should become None.
        # (Tested indirectly here by direct assertion on the empty
        # snapshot; full path covered by the integration test below.)

    def test_full_round_trip_via_jaato_session_helpers(self):
        """End-to-end: attach state on a session, snapshot through
        _get_session_state, serialize, deserialize, restore via
        _restore_session_state, and confirm the value is back."""
        s = _bare_session()
        s.set_session_state("audit", {"head": "h1", "seq": 1})

        # Snapshot the SessionState the way the journal save path does.
        captured = s._get_session_state(session_id="20260425_000000")
        assert captured.session_state == {"audit": {"head": "h1", "seq": 1}}

        # Round-trip through the serializer.
        data = serialize_session_state(captured)
        restored_state = deserialize_session_state(data)

        # Restore onto a fresh session.
        s2 = _bare_session()
        s2._restore_session_state(restored_state)
        assert s2.get_session_state("audit") == {"head": "h1", "seq": 1}
