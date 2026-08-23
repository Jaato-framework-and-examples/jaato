"""_restore_budget_usage must honour the runner's ``restored`` bool.

The daemon-side helper used to call the restore RPC and DISCARD its return
value, then log "Restored budget usage" unconditionally.  The runner answers
``restored: False`` when the reloaded session has no BudgetTracker -- i.e.
it came back unbudgeted and every cross-turn ceiling is gone.  Reporting
that as success is why the suspend/resume budget arc stayed invisible for
six rounds: the observable said "Restored" whether or not anything was.

Calls the helper (a source grep would survive the revert -- the RPC name is
still mentioned by the getattr above the call).
"""
from types import SimpleNamespace

import pytest

from server.session_manager import SessionManager


class _RPC:
    def __init__(self, restored):
        self._restored = restored
        self.calls = []

    def session_restore_budget_usage_threadsafe(self, usage, **kw):
        self.calls.append((usage, kw))
        return self._restored


def _state():
    return SimpleNamespace(budget_usage={"turns": 2.0},
                           budget_exhausted_reason=None)


def _call(restored):
    rpc = _RPC(restored)
    server = SimpleNamespace(_runner_rpc=rpc)
    applied = SessionManager._restore_budget_usage(
        SimpleNamespace(), server, _state(), "s1")
    return applied, rpc


def test_runner_reporting_not_restored_is_a_failure():
    # The regression: RPC returns cleanly, runner applied NOTHING.
    applied, rpc = _call(False)
    assert rpc.calls, "the RPC was never invoked -- test proves nothing"
    assert applied is False, (
        "daemon reported a successful budget restore while the runner said "
        "restored=False; the reloaded session is unbudgeted and its ceiling "
        "is silently gone")


def test_runner_confirming_restore_succeeds():
    applied, rpc = _call(True)
    assert rpc.calls
    assert applied is True


def test_warns_when_snapshot_did_not_land(caplog):
    with caplog.at_level("WARNING"):
        _call(False)
    assert any("did NOT apply" in r.message or "did NOT apply" in r.getMessage()
               for r in caplog.records), \
        "a lost ceiling must be visible at WARNING, not silent"


def test_no_snapshot_is_not_a_failure_path():
    # Unbudgeted session with nothing to restore: returns False, no RPC.
    rpc = _RPC(True)
    server = SimpleNamespace(_runner_rpc=rpc)
    applied = SessionManager._restore_budget_usage(
        SimpleNamespace(), server,
        SimpleNamespace(budget_usage=None, budget_exhausted_reason=None), "s1")
    assert applied is False
    assert rpc.calls == []
