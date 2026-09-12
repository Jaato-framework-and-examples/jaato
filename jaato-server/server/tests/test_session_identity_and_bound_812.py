"""A session an operator can SEE must be one they can ACT on (#812).

Three parts of the issue, three groups of tests:

1. **Identity** — ``RunnerIdentity``, its record-2.10 persistence, and the
   workspace-index section, so a session id resolves to a process.
2. **Surface and stop** — ``list_orphan_sessions`` / ``stop_session`` and
   their IPC verbs, so an unattended session can be found and stopped.
3. **The bound** — the daemon-side wall-clock sweep, so an unattended
   session cannot outlive a ceiling merely because the process holding
   that ceiling died.

The load-bearing NEGATIVE tests are in :class:`TestTheBoundDoesNotKillWhatItMustNot`:
the framework deliberately supports sessions that outlive their client, and
a bound that killed those would be worse than the bug.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Optional
from unittest.mock import MagicMock

import pytest

from server.session_identity import (
    RUNNER_IDENTITY_SCHEMA,
    RunnerIdentity,
    identity_from_server,
)
from server.session_lifetime import (
    REASON_MAX_ORPHAN_SECONDS,
    REASON_MAX_SESSION_SECONDS,
    SessionLifetimeObservation,
    describe_armed_bounds,
    evaluate,
    evaluate_one,
    resolve_bounds,
)
from server.session_manager import Session, SessionManager
from server.session_workspace_index import SessionWorkspaceIndex
from shared.runtime_limits import DEFAULT_MAX_ORPHAN_SECONDS, RuntimeLimits


# ----------------------------------------------------------------------
# Fixtures — mirror the production caller shape
# ----------------------------------------------------------------------


def _make_sm() -> SessionManager:
    """A SessionManager skeleton carrying only what #812's methods read.

    Same construction pattern as ``test_cascade_cancel.py::_make_sm`` — the
    full ``__init__`` stands up transports, a workspace index on the real
    ``~/.jaato`` and a plugin registry, none of which these paths touch.
    """
    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {}
    sm._lock = threading.RLock()
    sm._orphan_since = {}
    sm._lifetime_watchdog = None
    sm._lifetime_watchdog_stop = threading.Event()
    sm._lifetime_sweep_interval = 15.0
    sm._session_index = MagicMock()
    sm._session_index.identity.return_value = None
    sm._emit_to_session = MagicMock()
    return sm


def _make_session(
    session_id: str,
    *,
    clients: Optional[set] = None,
    loaded_at: float = 0.0,
    limits: Optional[RuntimeLimits] = None,
    processing: bool = False,
) -> Session:
    """A REAL ``Session`` (the dataclass production builds), mock server.

    Real rather than a MagicMock because the fields under test —
    ``attached_clients``, ``loaded_at``, ``runner_identity`` — are the
    dataclass's own, and a mock would satisfy any assertion about them.
    """
    server = MagicMock()
    server._model_running = processing
    server._main_agent_id = "main"
    server.stop.return_value = processing
    server._profile = MagicMock()
    server._profile.runtime_limits = limits
    server._profile.name = "test-profile"
    session = Session(
        session_id=session_id,
        name=session_id,
        server=server,
        created_at=datetime.now().isoformat(),
        loaded_at=loaded_at,
    )
    session.attached_clients = set(clients or ())
    return session


# ======================================================================
# 1. Identity
# ======================================================================


class TestRunnerIdentity:
    def test_round_trips_through_a_dict(self):
        ident = RunnerIdentity(
            runner_pid=41237, pool_served=True, pool_slot_pid=41237,
            cascade_driver_id="cid-7", apparmor_profile="jaato-ws-abc",
            recorded_at=123.0,
        )
        back = RunnerIdentity.from_dict(ident.to_dict())
        assert back == ident
        assert ident.to_dict()["schema"] == RUNNER_IDENTITY_SCHEMA

    def test_from_dict_tolerates_nothing_and_junk(self):
        """This runs on the session LOAD path — it may never raise."""
        assert RunnerIdentity.from_dict(None) is None
        assert RunnerIdentity.from_dict("not a dict") is None
        assert RunnerIdentity.from_dict([]) is None
        # A record from a newer server carrying an unknown key.
        ident = RunnerIdentity.from_dict(
            {"runner_pid": 5, "some_future_field": "x"})
        assert ident.runner_pid == 5

    def test_a_restored_record_is_stale(self):
        """The pid named belonged to a previous process lifetime."""
        live = RunnerIdentity(runner_pid=99).to_dict()
        assert live["stale"] is False
        restored = RunnerIdentity.from_dict(live, stale=True)
        assert restored.stale is True
        assert "STALE" in restored.describe()

    def test_reads_pid_and_slot_off_a_spawned_runner(self):
        server = MagicMock()
        server._spawned_runner.pid = 4242
        server._spawned_runner.profile_name = "jaato-ws-x"
        server._spawned_runner.pool_slot.pid = 4242
        server._spawned_runner.pool_slot.cascade_id = "cid-1"

        ident = identity_from_server(server)
        assert ident.runner_pid == 4242
        assert ident.pool_served is True
        assert ident.pool_slot_pid == 4242
        assert ident.cascade_driver_id == "cid-1"
        assert ident.stale is False

    def test_a_server_with_no_runner_yields_none_not_an_error(self):
        """An in-process session has no process to identify; that is not a
        failure, and must not raise on the spawn path."""
        server = MagicMock()
        server._spawned_runner = None
        assert identity_from_server(server) is None

        bare = object()
        assert identity_from_server(bare) is None


class TestIdentityPersistence:
    def test_record_2_10_carries_it_and_older_records_still_load(self):
        """The #787 / #859 precedent: a new field, old records unaffected."""
        from shared.plugins.session.serializer import (
            deserialize_session_state, serialize_session_state,
        )

        pre = {
            "version": "2.9", "session_id": "20260903_084517",
            "created_at": "2026-09-03T08:45:17",
            "updated_at": "2026-09-03T08:52:22", "history": [],
        }
        state = deserialize_session_state(pre)
        assert state.runner_identity is None     # absent, not an error

        state.runner_identity = RunnerIdentity(runner_pid=7).to_dict()
        blob = serialize_session_state(state)
        assert blob["version"] == "2.10"
        assert blob["runner_identity"]["runner_pid"] == 7

        # ...and the 2.10 record round-trips.
        again = deserialize_session_state(json.loads(json.dumps(blob)))
        assert again.runner_identity["runner_pid"] == 7

    def test_a_1_x_record_still_loads(self):
        from shared.plugins.session.serializer import deserialize_session_state
        state = deserialize_session_state({
            "version": "1.0", "session_id": "old",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        })
        assert state.session_id == "old"
        assert state.runner_identity is None


class TestWorkspaceIndexIdentity:
    def test_identity_section_round_trips_on_disk(self, tmp_path):
        path = tmp_path / "index.json"
        idx = SessionWorkspaceIndex(path=path)
        idx.record("s1", "/ws/one")
        idx.record_identity("s1", RunnerIdentity(runner_pid=11).to_dict())

        reopened = SessionWorkspaceIndex(path=path)
        assert reopened.resolve("s1") == "/ws/one"
        assert reopened.identity("s1")["runner_pid"] == 11
        assert reopened.identities()["s1"]["runner_pid"] == 11

    def test_an_index_written_before_812_still_loads(self, tmp_path):
        """The section is additive: no ``identity`` key is not a format
        error, it is every file written before this change."""
        path = tmp_path / "index.json"
        path.write_text(json.dumps({"map": {"s1": "/ws"}, "ambiguous": []}))
        idx = SessionWorkspaceIndex(path=path)
        assert idx.resolve("s1") == "/ws"
        assert idx.identity("s1") is None

    def test_an_ambiguous_id_refuses_a_workspace_but_still_names_a_process(
            self, tmp_path):
        """Ambiguity is about WHICH WORKSPACE.  Refusing to say which
        process ran it would withhold exactly the diagnostic #812 lacked."""
        idx = SessionWorkspaceIndex(path=tmp_path / "index.json")
        idx.record("dup", "/ws/a")
        idx.record("dup", "/ws/b")          # collision -> ambiguous
        idx.record_identity("dup", RunnerIdentity(runner_pid=3).to_dict())

        assert idx.resolve("dup") is None
        assert idx.identity("dup")["runner_pid"] == 3

    def test_forget_drops_the_identity_with_the_mapping(self, tmp_path):
        idx = SessionWorkspaceIndex(path=tmp_path / "index.json")
        idx.record("s1", "/ws")
        idx.record_identity("s1", RunnerIdentity(runner_pid=1).to_dict())
        idx.forget("s1")
        assert idx.resolve("s1") is None
        assert idx.identity("s1") is None


class TestIdentityOnTheSessionManager:
    def test_a_loaded_session_answers_from_memory(self):
        sm = _make_sm()
        session = _make_session("s1")
        session.runner_identity = RunnerIdentity(runner_pid=77)
        sm._sessions["s1"] = session
        assert sm.get_runner_identity("s1")["runner_pid"] == 77

    def test_a_cold_session_answers_from_the_index_and_is_stale(self):
        sm = _make_sm()
        sm._session_index.identity.return_value = RunnerIdentity(
            runner_pid=88).to_dict()
        found = sm.get_runner_identity("gone")
        assert found["runner_pid"] == 88
        assert found["stale"] is True      # nothing may act on it

    def test_an_unknown_session_answers_none(self):
        sm = _make_sm()
        assert sm.get_runner_identity("never-existed") is None

    def test_recording_stamps_the_session_and_the_index(self):
        sm = _make_sm()
        sm._sessions["s1"] = _make_session("s1")
        server = MagicMock()
        server._spawned_runner.pid = 909
        server._spawned_runner.profile_name = ""
        server._spawned_runner.pool_slot = None

        ident = sm._record_runner_identity("s1", server, cascade_driver_id="c")
        assert ident.runner_pid == 909
        assert sm._sessions["s1"].runner_identity.runner_pid == 909
        sm._session_index.record_identity.assert_called_once()

    def test_an_index_write_failure_does_not_break_the_spawn(self):
        """Diagnostics never fail a spawn that already succeeded."""
        sm = _make_sm()
        sm._sessions["s1"] = _make_session("s1")
        sm._session_index.record_identity.side_effect = OSError("read-only")
        server = MagicMock()
        server._spawned_runner.pid = 5
        server._spawned_runner.profile_name = ""
        server._spawned_runner.pool_slot = None

        ident = sm._record_runner_identity("s1", server)
        assert ident.runner_pid == 5     # returned despite the index failing

    def test_refresh_never_clears_a_good_record(self):
        """``None`` means "could not read one", never "there is none" —
        clearing on a transient read would destroy the evidence."""
        sm = _make_sm()
        session = _make_session("s1")
        session.runner_identity = RunnerIdentity(runner_pid=31)
        session.server._spawned_runner = None
        sm._refresh_runner_identity(session)
        assert session.runner_identity.runner_pid == 31


# ======================================================================
# 2. Orphan surface and stop-by-id
# ======================================================================


class TestOrphanSurface:
    def test_lists_only_sessions_with_no_client(self):
        sm = _make_sm()
        sm._sessions["attached"] = _make_session(
            "attached", clients={"client_ipc_14"})
        sm._sessions["orphan"] = _make_session("orphan", clients=set())

        rows = sm.list_orphan_sessions()
        assert [r["session_id"] for r in rows] == ["orphan"]

    def test_a_row_says_what_an_operator_needs_to_act(self):
        sm = _make_sm()
        session = _make_session("orphan", processing=True)
        session.runner_identity = RunnerIdentity(runner_pid=4242)
        session.workspace_path = "/ws/arm"
        sm._sessions["orphan"] = session

        row = sm.list_orphan_sessions()[0]
        assert row["is_processing"] is True          # spending, right now
        assert row["runner"]["runner_pid"] == 4242   # which process
        assert row["workspace_path"] == "/ws/arm"
        assert row["max_orphan_seconds"] == DEFAULT_MAX_ORPHAN_SECONDS

    def test_a_headless_marker_is_a_client(self):
        """A woken / reactor-driven session is NEVER an orphan."""
        sm = _make_sm()
        sm._sessions["woken"] = _make_session(
            "woken", clients={SessionManager._HEADLESS_CLIENT_ID})
        assert sm.list_orphan_sessions() == []


class TestStopSession:
    def test_stops_a_running_session_and_emits_a_reason(self):
        sm = _make_sm()
        sm._sessions["s1"] = _make_session("s1", processing=True)

        result = sm.stop_session("s1", reason="operator_request")
        assert result == {
            "session_id": "s1", "found": True, "stopped": True,
            "was_processing": True, "reason": "operator_request",
        }
        sm._sessions["s1"].server.stop.assert_called_once()
        event = sm._emit_to_session.call_args[0][1]
        assert event.reason == "operator_request"

    def test_an_idle_session_is_a_successful_stop_with_nothing_to_cancel(self):
        sm = _make_sm()
        sm._sessions["s1"] = _make_session("s1", processing=False)
        result = sm.stop_session("s1")
        assert result["found"] is True
        assert result["stopped"] is False
        assert result["was_processing"] is False

    def test_an_unknown_session_reports_rather_than_raising(self):
        sm = _make_sm()
        result = sm.stop_session("never-existed")
        assert result["found"] is False
        assert result["stopped"] is False

    def test_a_raising_server_still_emits_the_termination(self):
        sm = _make_sm()
        session = _make_session("s1", processing=True)
        session.server.stop.side_effect = RuntimeError("boom")
        sm._sessions["s1"] = session

        result = sm.stop_session("s1")
        assert result["found"] is True
        sm._emit_to_session.assert_called_once()

    def test_it_never_signals_the_runner_pid(self):
        """Stopping is cancellation, not a kill: killing a pool-served
        runner destroys a slot other cascade stages expect to reuse."""
        import server.session_manager as mod
        source = mod.SessionManager.stop_session.__doc__ or ""
        assert "NOT a kill" in source
        session = _make_session("s1")
        session.runner_identity = RunnerIdentity(runner_pid=1)
        sm = _make_sm()
        sm._sessions["s1"] = session
        with pytest.MonkeyPatch.context() as mp:
            killed = []
            mp.setattr("os.kill", lambda *a, **k: killed.append(a))
            sm.stop_session("s1")
            assert killed == []


# ======================================================================
# 3. The daemon-side wall-clock bound
# ======================================================================


class TestBoundResolution:
    def test_unset_takes_the_framework_defaults(self):
        assert resolve_bounds(None) == (None, DEFAULT_MAX_ORPHAN_SECONDS)
        assert resolve_bounds(RuntimeLimits()) == (
            None, DEFAULT_MAX_ORPHAN_SECONDS)

    def test_zero_is_the_explicit_opt_out_and_outranks_the_default(self):
        assert resolve_bounds(RuntimeLimits(max_orphan_seconds=0)) == (
            None, None)

    def test_a_declared_value_wins(self):
        assert resolve_bounds(RuntimeLimits(
            max_session_seconds=1800, max_orphan_seconds=60,
        )) == (1800.0, 60.0)

    def test_a_limits_object_predating_the_fields_does_not_raise(self):
        """A revived session's snapshot may know nothing of these."""
        class Old:
            memory_max_mb = None
        assert resolve_bounds(Old()) == (None, DEFAULT_MAX_ORPHAN_SECONDS)

    def test_the_armed_line_names_the_effective_values(self):
        """#735: a cap that silently does not apply is worse than no cap."""
        line = describe_armed_bounds(None)
        assert "max_orphan_seconds=900.0s (framework default)" in line
        assert "max_session_seconds=unbounded" in line


class TestBoundEvaluation:
    def test_an_orphan_past_its_grace_is_stopped(self):
        obs = SessionLifetimeObservation(
            session_id="s1", loaded_at=0.0, orphaned_since=0.0,
            limits=RuntimeLimits(max_orphan_seconds=100),
        )
        verdict = evaluate_one(obs, now=101.0)
        assert verdict.reason == REASON_MAX_ORPHAN_SECONDS
        assert verdict.limit_seconds == 100.0

    def test_an_orphan_within_its_grace_is_left_alone(self):
        obs = SessionLifetimeObservation(
            session_id="s1", loaded_at=0.0, orphaned_since=0.0,
            limits=RuntimeLimits(max_orphan_seconds=100),
        )
        assert evaluate_one(obs, now=99.0) is None

    def test_total_lifetime_is_reported_in_preference_to_orphanhood(self):
        """Crossing both, "it ran too long" is the more informative
        verdict."""
        obs = SessionLifetimeObservation(
            session_id="s1", loaded_at=0.0, orphaned_since=0.0,
            limits=RuntimeLimits(
                max_session_seconds=10, max_orphan_seconds=10),
        )
        assert evaluate_one(obs, now=50.0).reason == REASON_MAX_SESSION_SECONDS

    def test_an_attached_session_is_never_judged_on_the_orphan_bound(self):
        obs = SessionLifetimeObservation(
            session_id="s1", loaded_at=0.0, orphaned_since=None,
            limits=RuntimeLimits(max_orphan_seconds=1),
        )
        assert evaluate_one(obs, now=10_000.0) is None

    def test_evaluate_returns_only_the_crossers(self):
        limits = RuntimeLimits(max_orphan_seconds=10)
        verdicts = evaluate([
            SessionLifetimeObservation("ok", 0.0, 0.0, limits),
            SessionLifetimeObservation("late", 0.0, 0.0, limits),
        ], now=5.0)
        assert verdicts == []
        verdicts = evaluate([
            SessionLifetimeObservation("ok", 0.0, None, limits),
            SessionLifetimeObservation("late", 0.0, 0.0, limits),
        ], now=50.0)
        assert [v.session_id for v in verdicts] == ["late"]


class TestTheSweep:
    def test_it_stops_the_orphan_that_crossed_its_bound(self):
        """#812 end to end at the manager: an orphaned, still-processing
        session is stopped daemon-side without any client involvement."""
        sm = _make_sm()
        sm._sessions["runaway"] = _make_session(
            "runaway", clients=set(), loaded_at=0.0, processing=True,
            limits=RuntimeLimits(max_orphan_seconds=10),
        )
        # First sweep observes the orphanhood; nothing has elapsed yet.
        assert sm.sweep_session_lifetimes(now=0.0) == []
        # Later sweep: past the grace.
        verdicts = sm.sweep_session_lifetimes(now=100.0)
        assert [v.session_id for v in verdicts] == ["runaway"]
        sm._sessions["runaway"].server.stop.assert_called_once()
        assert sm._emit_to_session.call_args[0][1].reason == \
            REASON_MAX_ORPHAN_SECONDS

    def test_a_reconnect_renews_the_grace(self):
        """The bound measures CONTINUOUS orphanhood."""
        sm = _make_sm()
        session = _make_session(
            "s1", clients=set(), limits=RuntimeLimits(max_orphan_seconds=10))
        sm._sessions["s1"] = session

        sm.sweep_session_lifetimes(now=0.0)
        session.attached_clients.add("client_ipc_1")     # reconnect
        sm.sweep_session_lifetimes(now=5.0)
        assert "s1" not in sm._orphan_since
        session.attached_clients.clear()                 # gone again
        sm.sweep_session_lifetimes(now=6.0)
        # The clock restarted at 6.0, so 12.0 is only 6s of orphanhood.
        assert sm.sweep_session_lifetimes(now=12.0) == []

    def test_the_orphan_map_does_not_grow(self):
        sm = _make_sm()
        sm._sessions["s1"] = _make_session("s1", clients=set())
        sm.sweep_session_lifetimes(now=1.0)
        assert "s1" in sm._orphan_since
        del sm._sessions["s1"]                            # unloaded
        sm.sweep_session_lifetimes(now=2.0)
        assert sm._orphan_since == {}

    def test_a_crossed_session_is_not_re_stopped_every_sweep(self):
        sm = _make_sm()
        sm._sessions["s1"] = _make_session(
            "s1", clients=set(), limits=RuntimeLimits(max_orphan_seconds=10))
        sm.sweep_session_lifetimes(now=0.0)
        assert len(sm.sweep_session_lifetimes(now=100.0)) == 1
        # Session survived the stop (wedged model thread): re-judged from
        # now, not stopped again immediately.
        assert sm.sweep_session_lifetimes(now=101.0) == []

    def test_the_watchdog_arms_once_and_disarms(self):
        sm = _make_sm()
        assert sm.start_lifetime_watchdog(interval_seconds=0.05) is True
        assert sm.start_lifetime_watchdog() is False      # idempotent
        sm.stop_lifetime_watchdog()
        assert sm._lifetime_watchdog is None


class TestTheBoundDoesNotKillWhatItMustNot:
    """The framework deliberately supports sessions that outlive their
    client.  A bound that killed those would be worse than the bug it
    fixes, so each documented detached shape gets a test."""

    def test_a_woken_session_is_not_stopped(self):
        """``session.wake`` / ``resume_session`` attach the synthetic
        headless client, so a revived session is never an orphan."""
        sm = _make_sm()
        sm._sessions["woken"] = _make_session(
            "woken", clients={SessionManager._HEADLESS_CLIENT_ID},
            limits=RuntimeLimits(max_orphan_seconds=1),
        )
        sm.sweep_session_lifetimes(now=0.0)
        assert sm.sweep_session_lifetimes(now=10_000.0) == []

    def test_a_cascade_stage_holding_its_drivers_client_is_not_stopped(self):
        sm = _make_sm()
        stage = _make_session(
            "stage", clients={"client_ipc_3"},
            limits=RuntimeLimits(max_orphan_seconds=1))
        stage.cascade_driver_id = "cid-9"
        sm._sessions["stage"] = stage
        sm.sweep_session_lifetimes(now=0.0)
        assert sm.sweep_session_lifetimes(now=10_000.0) == []

    def test_a_cold_session_is_invisible_to_the_sweep(self):
        """A completion-gated session that ended is UNLOADED; the resume
        path through ``session.wake`` is untouched because the sweep only
        ever looks at ``_sessions``."""
        sm = _make_sm()
        assert sm.sweep_session_lifetimes(now=10_000.0) == []

    def test_an_explicit_zero_disables_the_default_grace(self):
        sm = _make_sm()
        sm._sessions["forever"] = _make_session(
            "forever", clients=set(), limits=RuntimeLimits(
                max_orphan_seconds=0))
        sm.sweep_session_lifetimes(now=0.0)
        assert sm.sweep_session_lifetimes(now=10_000_000.0) == []

    def test_an_attended_session_is_never_bounded_by_default(self):
        """``max_session_seconds`` is opt-in: an interactive session left
        open over a lunch break is not a defect."""
        sm = _make_sm()
        sm._sessions["tui"] = _make_session("tui", clients={"c1"})
        assert sm.sweep_session_lifetimes(now=10_000_000.0) == []


# ======================================================================
# Inheritance: most-restrictive-wins, like max_parallel_tools
# ======================================================================


class TestInheritance:
    def _merge(self, parent_limits, child_limits):
        from shared.plugins.subagent.config import _merge_runtime_limits

        parent = MagicMock()
        parent.name = "base"
        parent.runtime_limits = parent_limits
        child = MagicMock()
        child.name = "leaf"
        child.runtime_limits = child_limits
        return _merge_runtime_limits([parent], child)

    def test_a_child_may_narrow_a_wall_clock_bound(self):
        merged, conflicts = self._merge(
            RuntimeLimits(max_orphan_seconds=600),
            RuntimeLimits(max_orphan_seconds=60),
        )
        assert merged.max_orphan_seconds == 60
        assert conflicts == []

    def test_a_child_may_not_widen_one(self):
        merged, _ = self._merge(
            RuntimeLimits(max_orphan_seconds=60),
            RuntimeLimits(max_orphan_seconds=6000),
        )
        assert merged.max_orphan_seconds == 60

    def test_a_child_may_not_disable_an_ancestors_bound(self):
        """0 is the LEAST restrictive value, so it cannot win a min()."""
        merged, _ = self._merge(
            RuntimeLimits(max_orphan_seconds=60),
            RuntimeLimits(max_orphan_seconds=0),
        )
        assert merged.max_orphan_seconds == 60

    def test_zero_survives_when_every_layer_declares_it(self):
        merged, _ = self._merge(
            RuntimeLimits(max_orphan_seconds=0),
            RuntimeLimits(max_orphan_seconds=0),
        )
        assert merged.max_orphan_seconds == 0

    def test_layers_differing_only_in_a_min_wins_field_do_not_conflict(self):
        """Two parents agreeing on every cgroup ceiling and differing only
        here must be resolved by min(), not reported."""
        from shared.plugins.subagent.config import _merge_runtime_limits

        def _p(name, limits):
            p = MagicMock()
            p.name = name
            p.runtime_limits = limits
            return p

        child = MagicMock()
        child.name = "leaf"
        child.runtime_limits = None
        merged, conflicts = _merge_runtime_limits(
            [
                _p("a", RuntimeLimits(pids_max=64, max_session_seconds=100)),
                _p("b", RuntimeLimits(pids_max=64, max_session_seconds=200)),
            ],
            child,
        )
        assert conflicts == []
        assert merged.max_session_seconds == 100
        assert merged.pids_max == 64


class TestValidation:
    def test_zero_is_accepted_and_negative_is_not(self):
        RuntimeLimits(max_orphan_seconds=0)          # explicit opt-out
        RuntimeLimits(max_session_seconds=0)
        with pytest.raises(ValueError, match="max_orphan_seconds"):
            RuntimeLimits(max_orphan_seconds=-1)
        with pytest.raises(ValueError, match="max_session_seconds"):
            RuntimeLimits(max_session_seconds=-0.5)

    def test_a_bool_is_not_a_deadline(self):
        with pytest.raises(ValueError):
            RuntimeLimits(max_orphan_seconds=True)

    def test_from_dict_carries_the_fields(self):
        limits = RuntimeLimits.from_dict({
            "max_session_seconds": 1800, "max_orphan_seconds": 120,
        })
        assert limits.max_session_seconds == 1800
        assert limits.max_orphan_seconds == 120
        assert limits.extra == {}       # not parked as unknown keys
