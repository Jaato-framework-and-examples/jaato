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
