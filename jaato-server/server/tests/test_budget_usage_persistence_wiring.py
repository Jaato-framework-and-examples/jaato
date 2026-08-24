"""A budget ceiling survives a session unload END TO END.

``BudgetTracker.restore_usage`` existed and was tested, but NOTHING CALLED IT
on reload -- the fix existed and did not take effect. This pins the four seams
that make it actually work:

  SessionState.budget_usage           the field (distinct from budget_state,
                                      which is the CONVERSATION budget)
  session.restore_budget_usage        the runner RPC verb
  session_restore_budget_usage_*      the daemon-side client method
  _save / _load_session_impl          snapshot at save, restore at load

Why it matters: sessions unload on ORPHAN, so a suspend/resume driver that
disconnects during a wait is evicted every time. Before this, every resume
rebuilt the tracker zeroed, so `turns` and `seconds` ceilings never fired
however many resumes ran -- and the longer a goal suspends, the more certainly
the ceiling meant to bound it resets.
"""
import inspect

import pytest

from shared.budget_control import BudgetControlConfig, BudgetTracker
from shared.plugins.session.base import SessionState


ABORT = [{"at": "100%", "action": "abort"}]


def _cfg(limits):
    return BudgetControlConfig.from_dict({"limits": limits, "degrade": ABORT})


class TestTheSeamsExist:
    def test_state_carries_usage_separately_from_the_conversation_budget(self):
        fields = SessionState.__dataclass_fields__
        assert "budget_usage" in fields
        assert "budget_state" in fields, (
            "budget_state is the CONVERSATION budget -- the two must not be "
            "merged; conflating them is what hid this gap"
        )

    def test_runner_exposes_a_restore_verb(self):
        from server.runner.rpc import RunnerRPC
        assert hasattr(RunnerRPC, "_handle_session_restore_budget_usage")
        dispatch = inspect.getsource(RunnerRPC._dispatch_method)
        assert "session.restore_budget_usage" in dispatch, (
            "the handler exists but no verb routes to it"
        )

    def test_daemon_client_can_call_it(self):
        from server.runner_rpc_client import RunnerRPCClient
        assert hasattr(RunnerRPCClient, "session_restore_budget_usage")
        assert hasattr(RunnerRPCClient, "session_restore_budget_usage_threadsafe")

    def test_save_snapshots_the_usage(self):
        from server.session_manager import SessionManager
        save_src = "".join(
            inspect.getsource(fn)
            for _, fn in inspect.getmembers(SessionManager, inspect.isfunction)
            if "budget_usage=" in inspect.getsource(fn)
        )
        assert "session_get_budget_usage_threadsafe" in save_src, (
            "nothing snapshots budget usage at save time"
        )

    def test_load_actually_calls_the_restore(self):
        """By CALLING it, not by grepping for the name.

        The first version of this test searched _load_session_impl's source
        for the RPC name -- and passed with the call line deleted, because the
        getattr lookup above it still mentioned the name. Checking a symbol is
        mentioned is not checking it is used.
        """
        from unittest.mock import MagicMock
        from server.session_manager import SessionManager

        rpc = MagicMock()
        server = MagicMock()
        server._runner_rpc = rpc
        state = MagicMock()
        state.budget_usage = {"turns": 2.0}

        applied = SessionManager._restore_budget_usage(
            MagicMock(), server, state.budget_usage,
            state.budget_exhausted_reason, "sess-1")

        assert applied is True
        rpc.session_restore_budget_usage_threadsafe.assert_called_once()
        assert rpc.session_restore_budget_usage_threadsafe.call_args[0][0] == {
            "turns": 2.0}

    @pytest.mark.parametrize("usage", [None, {}])
    def test_load_skips_when_there_is_nothing_to_restore(self, usage):
        """Nothing to restore = neither usage NOR the exhaustion latch.

        Both fields are pinned explicitly: a MagicMock auto-creates
        ``budget_exhausted_reason`` as a truthy Mock, so leaving it unset made
        this assert the opposite of what it reads -- the same auto-attribute
        trap that bit the cascade-clamp and sandbox tests elsewhere.
        """
        from unittest.mock import MagicMock
        from server.session_manager import SessionManager
        rpc = MagicMock()
        server = MagicMock(); server._runner_rpc = rpc
        state = MagicMock()
        state.budget_usage = usage
        state.budget_exhausted_reason = None

        assert SessionManager._restore_budget_usage(
            MagicMock(), server, state.budget_usage,
            state.budget_exhausted_reason, "sess-1") is False
        rpc.session_restore_budget_usage_threadsafe.assert_not_called()

    def test_load_restores_a_latch_even_without_usage(self):
        """A ceiling that stopped the session must be re-asserted.

        Usage could legitimately be absent (older snapshot) while the latch is
        present; refusing to restore then would serve turns past a ceiling.
        """
        from unittest.mock import MagicMock
        from server.session_manager import SessionManager
        rpc = MagicMock()
        server = MagicMock(); server._runner_rpc = rpc
        state = MagicMock()
        state.budget_usage = None
        state.budget_exhausted_reason = "budget_exhausted (turns 100%)"

        assert SessionManager._restore_budget_usage(
            MagicMock(), server, state.budget_usage,
            state.budget_exhausted_reason, "sess-1") is True
        kwargs = rpc.session_restore_budget_usage_threadsafe.call_args.kwargs
        assert kwargs["exhausted_reason"] == "budget_exhausted (turns 100%)"

    def test_a_failing_restore_is_reported_not_swallowed(self, caplog):
        """A ceiling that silently stops applying is the worst outcome."""
        import logging
        from unittest.mock import MagicMock
        from server.session_manager import SessionManager

        rpc = MagicMock()
        rpc.session_restore_budget_usage_threadsafe.side_effect = RuntimeError("boom")
        server = MagicMock(); server._runner_rpc = rpc
        state = MagicMock(); state.budget_usage = {"turns": 2.0}

        with caplog.at_level(logging.WARNING):
            applied = SessionManager._restore_budget_usage(
                MagicMock(), server, state.budget_usage,
            state.budget_exhausted_reason, "sess-1")

        assert applied is False
        assert any(
            "ceilings restart from zero" in record.getMessage()
            for record in caplog.records
        ), caplog.text


class TestTheBehaviourItBuys:
    def test_ceiling_survives_the_round_trip(self):
        """Snapshot -> new tracker -> restore -> the ceiling still fires."""
        cfg = _cfg({"turns": 2})
        before = BudgetTracker(cfg)
        before.observe(turns=1)
        snapshot = before.usage.as_dict()

        after_unload = BudgetTracker(cfg)                 # what a reload builds
        assert not after_unload.observe(turns=1), "precondition: zeroed"

        restored = BudgetTracker(cfg)
        restored.restore_usage(snapshot)
        assert [r.at_percent for r in restored.observe(turns=1)] == [100.0]

    def test_seconds_survives_too(self):
        """Not just turns -- every cross-turn dimension was affected."""
        cfg = _cfg({"seconds": 100})
        before = BudgetTracker(cfg)
        before.observe(seconds=60)

        restored = BudgetTracker(cfg)
        restored.restore_usage(before.usage.as_dict())
        assert [r.at_percent for r in restored.observe(seconds=40)] == [100.0]

    def test_an_unbudgeted_session_round_trips_harmlessly(self):
        from shared.jaato_session import JaatoSession
        s = JaatoSession.__new__(JaatoSession)
        s._budget_tracker = None
        JaatoSession.restore_budget_usage(s, {"turns": 5})   # must not raise
