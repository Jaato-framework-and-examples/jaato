"""Tests for ``SessionManager._cascade_teardown_isolated_subagents``
(Phase 4 §4.3.6d).

When a parent session is unloaded, its isolated subagents'
sub-runners + sub-cgroups + sub-AppArmor profiles must be torn
down so kernel state stays consistent and no orphaned resources
leak across session lifecycles.

Test surfaces:
1. Happy path — all three teardown steps fire in order.
2. Filtering — only handles belonging to the named parent are
   torn down (siblings of other parents stay registered).
3. Best-effort — failure in one step doesn't strand the others.
4. Empty registry — no-op return 0.
5. Resources without cgroup_path skip cgroup teardown.
6. Resources without sub_apparmor_profile skip AppArmor teardown.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from server.session_manager import (
    SessionManager,
    SubRunnerHandle,
)


def _make_session_manager(apparmor_manager=None, cgroups_manager=None):
    sm = SessionManager.__new__(SessionManager)
    sm._lock = threading.RLock()
    sm._sessions = {}
    sm._isolated_sub_runners = {}
    if apparmor_manager is not None:
        sm._apparmor_manager = apparmor_manager
    if cgroups_manager is not None:
        sm._cgroups_manager = cgroups_manager
    return sm


def _register_handle(
    sm,
    parent_session_id="sess-A",
    subagent_id="agent-1",
    sub_apparmor_profile="jaato-ws-sess-A//agent-1",
    cgroup_path="/sys/fs/cgroup/jaato/jaato-ws-sess-A__sub_agent-1",
):
    handle = SubRunnerHandle(
        parent_session_id=parent_session_id,
        subagent_id=subagent_id,
        isolated_session_id=f"{parent_session_id}__sub_{subagent_id}",
        rpc=MagicMock(),
        spawned=MagicMock(),
        sub_apparmor_profile=sub_apparmor_profile,
        cgroup_path=cgroup_path,
    )
    sm._isolated_sub_runners[handle.isolated_session_id] = handle
    return handle


class TestHappyPath:
    def test_all_teardown_steps_fire(self):
        apparmor = MagicMock()
        cgroups = MagicMock()
        sm = _make_session_manager(
            apparmor_manager=apparmor, cgroups_manager=cgroups,
        )
        handle = _register_handle(sm)

        count = sm._cascade_teardown_isolated_subagents("sess-A")

        assert count == 1
        # Cgroup teardown invoked with the isolated_session_id.
        cgroups.teardown_cgroup.assert_called_once_with(
            "sess-A__sub_agent-1",
        )
        # AppArmor teardown invoked with parent + subagent ids.
        apparmor.teardown_sub_profile.assert_called_once_with(
            parent_session_id="sess-A",
            subagent_id="agent-1",
        )
        # Handle removed from registry.
        assert "sess-A__sub_agent-1" not in sm._isolated_sub_runners


class TestFiltering:
    def test_only_owned_handles_torn_down(self):
        """Sibling parents' handles stay registered."""
        apparmor = MagicMock()
        cgroups = MagicMock()
        sm = _make_session_manager(
            apparmor_manager=apparmor, cgroups_manager=cgroups,
        )
        _register_handle(sm, parent_session_id="sess-A", subagent_id="a1")
        _register_handle(sm, parent_session_id="sess-A", subagent_id="a2")
        _register_handle(sm, parent_session_id="sess-B", subagent_id="b1")

        count = sm._cascade_teardown_isolated_subagents("sess-A")

        assert count == 2
        # sess-B handle untouched.
        assert "sess-B__sub_b1" in sm._isolated_sub_runners
        # AppArmor teardown was NOT called for sess-B's handle.
        teardown_calls = apparmor.teardown_sub_profile.call_args_list
        parent_ids = [c.kwargs["parent_session_id"] for c in teardown_calls]
        assert "sess-B" not in parent_ids


class TestBestEffortResilience:
    def test_cgroup_teardown_failure_does_not_block_apparmor(self):
        apparmor = MagicMock()
        cgroups = MagicMock()
        cgroups.teardown_cgroup.side_effect = RuntimeError("kill failed")
        sm = _make_session_manager(
            apparmor_manager=apparmor, cgroups_manager=cgroups,
        )
        _register_handle(sm)

        count = sm._cascade_teardown_isolated_subagents("sess-A")

        assert count == 1
        # AppArmor teardown still attempted despite cgroup failure.
        apparmor.teardown_sub_profile.assert_called_once()
        # Handle removed from registry.
        assert "sess-A__sub_agent-1" not in sm._isolated_sub_runners

    def test_apparmor_teardown_failure_does_not_strand_handle(self):
        apparmor = MagicMock()
        apparmor.teardown_sub_profile.side_effect = RuntimeError(
            "parser failed",
        )
        cgroups = MagicMock()
        sm = _make_session_manager(
            apparmor_manager=apparmor, cgroups_manager=cgroups,
        )
        _register_handle(sm)

        count = sm._cascade_teardown_isolated_subagents("sess-A")

        assert count == 1
        # Handle still removed (best-effort: log + continue).
        assert "sess-A__sub_agent-1" not in sm._isolated_sub_runners


class TestEmptyRegistry:
    def test_no_handles_returns_zero(self):
        sm = _make_session_manager()
        count = sm._cascade_teardown_isolated_subagents("sess-A")
        assert count == 0


class TestPartialResources:
    def test_no_cgroup_skips_cgroup_teardown(self):
        """Sub-runner without runtime_limits has no cgroup_path —
        skip cgroup teardown."""
        apparmor = MagicMock()
        cgroups = MagicMock()
        sm = _make_session_manager(
            apparmor_manager=apparmor, cgroups_manager=cgroups,
        )
        _register_handle(sm, cgroup_path="")  # No cgroup.

        sm._cascade_teardown_isolated_subagents("sess-A")

        cgroups.teardown_cgroup.assert_not_called()
        # AppArmor still torn down.
        apparmor.teardown_sub_profile.assert_called_once()

    def test_no_apparmor_skips_apparmor_teardown(self):
        """Sub-runner without sub-profile (shouldn't happen in §4.3.6
        but defensive) skips AppArmor teardown."""
        apparmor = MagicMock()
        cgroups = MagicMock()
        sm = _make_session_manager(
            apparmor_manager=apparmor, cgroups_manager=cgroups,
        )
        _register_handle(sm, sub_apparmor_profile="")  # No profile.

        sm._cascade_teardown_isolated_subagents("sess-A")

        apparmor.teardown_sub_profile.assert_not_called()
        # Cgroup still torn down.
        cgroups.teardown_cgroup.assert_called_once()


class TestOrphanSubCgroupReap:
    """Phase 5 §5.3: after the known-handles loop, the cascade scans
    for sub-cgroups whose directories exist on disk but aren't in the
    just-torn-down set, and reaps them via ``teardown_cgroup``.

    Catches kernel state left behind by rollback failures or
    mid-spawn crashes — the §4.3.9 item 4 gap."""

    def test_orphans_returned_by_scanner_are_torn_down(self):
        """Scanner returns 2 orphans → ``teardown_cgroup`` is called
        for each in addition to the known handle."""
        apparmor = MagicMock()
        cgroups = MagicMock()
        cgroups.list_orphan_sub_cgroups.return_value = [
            "sess-A__sub_orphan1",
            "sess-A__sub_orphan2",
        ]
        sm = _make_session_manager(
            apparmor_manager=apparmor, cgroups_manager=cgroups,
        )
        _register_handle(sm, subagent_id="known")

        count = sm._cascade_teardown_isolated_subagents("sess-A")

        # Return value pins the count of HANDLES, not orphans.
        assert count == 1
        # Scanner consulted with the known-set the cascade just
        # processed (the single known handle's isolated_session_id).
        scan_call = cgroups.list_orphan_sub_cgroups.call_args
        assert scan_call.args[0] == "sess-A"
        assert scan_call.args[1] == {"sess-A__sub_known"}
        # teardown_cgroup called for known + each orphan.
        torn = [c.args[0] for c in cgroups.teardown_cgroup.call_args_list]
        assert sorted(torn) == [
            "sess-A__sub_known",
            "sess-A__sub_orphan1",
            "sess-A__sub_orphan2",
        ]

    def test_scanner_runs_even_when_no_known_handles(self):
        """Empty registry → cascade returns 0; but the orphan scan
        STILL runs.  The ledger contract is "audit pass at every
        parent teardown" — a crash that never registered a handle
        can leave kernel state behind, and that's exactly what the
        scan catches.  Phase 5 §5.3 removed the pre-§5.3 short-
        circuit on empty handles for this reason."""
        cgroups = MagicMock()
        cgroups.list_orphan_sub_cgroups.return_value = []
        sm = _make_session_manager(cgroups_manager=cgroups)

        count = sm._cascade_teardown_isolated_subagents("sess-A")

        assert count == 0
        # Scanner consulted with the empty known-set.
        cgroups.list_orphan_sub_cgroups.assert_called_once()
        scan_call = cgroups.list_orphan_sub_cgroups.call_args
        assert scan_call.args[0] == "sess-A"
        assert scan_call.args[1] == set()

    def test_scanner_finds_and_reaps_orphan_without_any_known_handle(self):
        """Pure orphan scenario: handle registration crashed before
        the cascade was invoked.  Scanner finds the leftover cgroup
        and reaps it; cascade returns 0 (no HANDLES torn down)."""
        cgroups = MagicMock()
        cgroups.list_orphan_sub_cgroups.return_value = [
            "sess-A__sub_crashed",
        ]
        sm = _make_session_manager(cgroups_manager=cgroups)

        count = sm._cascade_teardown_isolated_subagents("sess-A")

        assert count == 0
        cgroups.teardown_cgroup.assert_called_once_with(
            "sess-A__sub_crashed",
        )

    def test_orphan_teardown_failure_does_not_block_cascade_return(self):
        """An orphan reap that raises must not prevent the cascade
        from returning the known-handle count."""
        apparmor = MagicMock()
        cgroups = MagicMock()
        cgroups.list_orphan_sub_cgroups.return_value = [
            "sess-A__sub_orphan",
        ]
        # First call (known handle) succeeds; second (orphan) raises.
        cgroups.teardown_cgroup.side_effect = [
            True, RuntimeError("kill failed"),
        ]
        sm = _make_session_manager(
            apparmor_manager=apparmor, cgroups_manager=cgroups,
        )
        _register_handle(sm, subagent_id="known")

        count = sm._cascade_teardown_isolated_subagents("sess-A")

        assert count == 1
        # Both teardown calls attempted.
        assert cgroups.teardown_cgroup.call_count == 2

    def test_scanner_failure_does_not_block_cascade_return(self):
        """If the scanner itself raises (e.g., FS error), cascade still
        returns the known-handle count."""
        apparmor = MagicMock()
        cgroups = MagicMock()
        cgroups.list_orphan_sub_cgroups.side_effect = OSError(
            "permission denied",
        )
        sm = _make_session_manager(
            apparmor_manager=apparmor, cgroups_manager=cgroups,
        )
        _register_handle(sm, subagent_id="known")

        count = sm._cascade_teardown_isolated_subagents("sess-A")

        assert count == 1
        # Known handle was torn down before the orphan scan.
        cgroups.teardown_cgroup.assert_called_once_with("sess-A__sub_known")

    def test_warning_logged_per_orphan(self, caplog):
        """An operator running with WARNING-level logging sees one
        line per orphan so reliability incidents can be correlated."""
        import logging
        cgroups = MagicMock()
        cgroups.list_orphan_sub_cgroups.return_value = [
            "sess-A__sub_orphan1",
        ]
        sm = _make_session_manager(cgroups_manager=cgroups)
        _register_handle(sm, subagent_id="known")

        with caplog.at_level(logging.WARNING):
            sm._cascade_teardown_isolated_subagents("sess-A")

        warning_messages = [
            r.message for r in caplog.records
            if r.levelno >= logging.WARNING
        ]
        assert any(
            "orphaned sub-cgroup" in m
            and "sess-A__sub_orphan1" in m
            for m in warning_messages
        ), f"missing orphan WARNING; got {warning_messages!r}"
