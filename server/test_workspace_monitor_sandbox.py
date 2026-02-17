"""Tests for workspace monitor sandbox path support.

Tests the ability of WorkspaceMonitor to watch additional directories
that the user has granted readwrite access to via the sandbox add command.
"""

import os
import threading
from typing import Dict, List, Set

import pytest

from server.workspace_monitor import WorkspaceMonitor, _ChangeAccumulator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace directory with some files."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "file1.txt").write_text("hello")
    (ws / "subdir").mkdir()
    (ws / "subdir" / "file2.txt").write_text("world")
    return str(ws)


@pytest.fixture
def sandbox_dir(tmp_path):
    """Create a temporary sandbox directory with some files."""
    sb = tmp_path / "sandbox"
    sb.mkdir()
    (sb / "config.json").write_text("{}")
    (sb / "data").mkdir()
    (sb / "data" / "output.csv").write_text("a,b,c")
    return str(sb)


@pytest.fixture
def sandbox_dir2(tmp_path):
    """Create a second sandbox directory."""
    sb = tmp_path / "sandbox2"
    sb.mkdir()
    (sb / "notes.txt").write_text("notes")
    return str(sb)


# ---------------------------------------------------------------------------
# Tests: add_sandbox_path / remove_sandbox_path
# ---------------------------------------------------------------------------

class TestSandboxPathManagement:
    """Tests for adding and removing sandbox paths."""

    def test_add_sandbox_path_seeds_baseline(self, workspace, sandbox_dir):
        """Adding a sandbox path should seed its baseline."""
        changes = []
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: changes.extend(c))
        monitor.start()
        try:
            monitor.add_sandbox_path(sandbox_dir)

            # Baseline should contain the sandbox files
            assert sandbox_dir in monitor._sandbox_baselines
            baseline = monitor._sandbox_baselines[sandbox_dir]
            expected_files = {
                os.path.join(sandbox_dir, "config.json"),
                os.path.join(sandbox_dir, "data", "output.csv"),
            }
            assert baseline == expected_files
        finally:
            monitor.stop()

    def test_add_sandbox_path_skips_workspace(self, workspace):
        """Adding the workspace path itself should be a no-op."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            monitor.add_sandbox_path(workspace)
            assert len(monitor._sandbox_baselines) == 0
        finally:
            monitor.stop()

    def test_add_sandbox_path_idempotent(self, workspace, sandbox_dir):
        """Adding the same sandbox path twice should be a no-op."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            monitor.add_sandbox_path(sandbox_dir)
            monitor.add_sandbox_path(sandbox_dir)
            assert len(monitor._sandbox_watches) == 1
        finally:
            monitor.stop()

    def test_add_nonexistent_sandbox_path(self, workspace, tmp_path):
        """Adding a nonexistent path should log a warning but not crash."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            monitor.add_sandbox_path(str(tmp_path / "nonexistent"))
            assert len(monitor._sandbox_baselines) == 0
        finally:
            monitor.stop()

    def test_remove_sandbox_path(self, workspace, sandbox_dir):
        """Removing a sandbox path should clean up watches and tracked entries."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            monitor.add_sandbox_path(sandbox_dir)
            assert sandbox_dir in monitor._sandbox_baselines

            # Simulate a tracked entry from sandbox
            config_path = os.path.join(sandbox_dir, "config.json")
            monitor.tracked[config_path] = "modified"

            monitor.remove_sandbox_path(sandbox_dir)
            assert sandbox_dir not in monitor._sandbox_baselines
            assert sandbox_dir not in monitor._sandbox_watches
            assert config_path not in monitor.tracked
        finally:
            monitor.stop()

    def test_remove_untracked_sandbox_path(self, workspace):
        """Removing a path that was never added should be a no-op."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            monitor.remove_sandbox_path("/nonexistent/path")
            # Should not raise
        finally:
            monitor.stop()

    def test_get_sandbox_paths(self, workspace, sandbox_dir, sandbox_dir2):
        """get_sandbox_paths should return all currently monitored paths."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            assert monitor.get_sandbox_paths() == []
            monitor.add_sandbox_path(sandbox_dir)
            assert monitor.get_sandbox_paths() == [sandbox_dir]
            monitor.add_sandbox_path(sandbox_dir2)
            paths = set(monitor.get_sandbox_paths())
            assert paths == {sandbox_dir, sandbox_dir2}
        finally:
            monitor.stop()

    def test_update_sandbox_paths_adds_and_removes(self, workspace, sandbox_dir, sandbox_dir2):
        """update_sandbox_paths should sync the set of watched paths."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            # Add first sandbox
            monitor.update_sandbox_paths([sandbox_dir])
            assert set(monitor.get_sandbox_paths()) == {sandbox_dir}

            # Replace with second sandbox
            monitor.update_sandbox_paths([sandbox_dir2])
            assert set(monitor.get_sandbox_paths()) == {sandbox_dir2}

            # Add both
            monitor.update_sandbox_paths([sandbox_dir, sandbox_dir2])
            assert set(monitor.get_sandbox_paths()) == {sandbox_dir, sandbox_dir2}

            # Remove all
            monitor.update_sandbox_paths([])
            assert monitor.get_sandbox_paths() == []
        finally:
            monitor.stop()

    def test_update_sandbox_paths_excludes_workspace(self, workspace, sandbox_dir):
        """update_sandbox_paths should not watch the workspace itself."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            monitor.update_sandbox_paths([workspace, sandbox_dir])
            assert set(monitor.get_sandbox_paths()) == {sandbox_dir}
        finally:
            monitor.stop()


# ---------------------------------------------------------------------------
# Tests: Sandbox file event tracking
# ---------------------------------------------------------------------------

class TestSandboxEventTracking:
    """Tests for tracking file changes in sandbox paths."""

    def test_sandbox_create_event(self, workspace, sandbox_dir):
        """Creating a file in a sandbox path should track it with absolute path."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            monitor.add_sandbox_path(sandbox_dir)

            new_file = os.path.join(sandbox_dir, "new_file.txt")
            # Simulate the filesystem event
            monitor._on_sandbox_fs_event(new_file, "created", sandbox_dir)

            assert new_file in monitor.tracked
            assert monitor.tracked[new_file] == "created"
        finally:
            monitor.stop()

    def test_sandbox_modify_baseline_file(self, workspace, sandbox_dir):
        """Modifying a baseline sandbox file should track as 'modified'."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            monitor.add_sandbox_path(sandbox_dir)

            config_path = os.path.join(sandbox_dir, "config.json")
            monitor._on_sandbox_fs_event(config_path, "modified", sandbox_dir)

            assert config_path in monitor.tracked
            assert monitor.tracked[config_path] == "modified"
        finally:
            monitor.stop()

    def test_sandbox_create_then_delete(self, workspace, sandbox_dir):
        """Creating then deleting a sandbox file should remove from tracked."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            monitor.add_sandbox_path(sandbox_dir)

            new_file = os.path.join(sandbox_dir, "temp.txt")
            monitor._on_sandbox_fs_event(new_file, "created", sandbox_dir)
            assert new_file in monitor.tracked

            monitor._on_sandbox_fs_event(new_file, "deleted", sandbox_dir)
            assert new_file not in monitor.tracked
        finally:
            monitor.stop()

    def test_sandbox_delete_baseline_file(self, workspace, sandbox_dir):
        """Deleting a baseline sandbox file should track as 'deleted'."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            monitor.add_sandbox_path(sandbox_dir)

            config_path = os.path.join(sandbox_dir, "config.json")
            monitor._on_sandbox_fs_event(config_path, "deleted", sandbox_dir)

            assert config_path in monitor.tracked
            assert monitor.tracked[config_path] == "deleted"
        finally:
            monitor.stop()

    def test_sandbox_events_use_absolute_paths(self, workspace, sandbox_dir):
        """Sandbox tracked entries should use absolute paths (not relative)."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            monitor.add_sandbox_path(sandbox_dir)

            new_file = os.path.join(sandbox_dir, "test.txt")
            monitor._on_sandbox_fs_event(new_file, "created", sandbox_dir)

            # Key should be absolute path
            tracked_keys = list(monitor.tracked.keys())
            sandbox_keys = [k for k in tracked_keys if os.path.isabs(k)]
            assert new_file in sandbox_keys
        finally:
            monitor.stop()

    def test_sandbox_hidden_files_skipped(self, workspace, sandbox_dir):
        """Hidden files in sandbox paths should be skipped."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            monitor.add_sandbox_path(sandbox_dir)

            hidden_file = os.path.join(sandbox_dir, ".hidden")
            monitor._on_sandbox_fs_event(hidden_file, "created", sandbox_dir)

            assert hidden_file not in monitor.tracked
        finally:
            monitor.stop()

    def test_sandbox_and_workspace_coexist(self, workspace, sandbox_dir):
        """Sandbox and workspace changes should coexist in tracked dict."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            monitor.add_sandbox_path(sandbox_dir)

            # Workspace event (relative path)
            ws_file = os.path.join(workspace, "new.txt")
            monitor._on_fs_event(ws_file, "created")
            assert "new.txt" in monitor.tracked

            # Sandbox event (absolute path)
            sb_file = os.path.join(sandbox_dir, "sb_new.txt")
            monitor._on_sandbox_fs_event(sb_file, "created", sandbox_dir)
            assert sb_file in monitor.tracked

            # Both should be in snapshot
            snapshot = monitor.get_snapshot()
            paths = {e["path"] for e in snapshot}
            assert "new.txt" in paths
            assert sb_file in paths
        finally:
            monitor.stop()


# ---------------------------------------------------------------------------
# Tests: Snapshot and persistence
# ---------------------------------------------------------------------------

class TestSnapshotAndPersistence:
    """Tests for snapshot, tracked dict, and restore with sandbox paths."""

    def test_get_snapshot_includes_sandbox(self, workspace, sandbox_dir):
        """get_snapshot should include sandbox-tracked files."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            monitor.add_sandbox_path(sandbox_dir)

            sb_file = os.path.join(sandbox_dir, "new.txt")
            monitor._on_sandbox_fs_event(sb_file, "created", sandbox_dir)

            snapshot = monitor.get_snapshot()
            assert any(e["path"] == sb_file for e in snapshot)
        finally:
            monitor.stop()

    def test_get_tracked_dict_includes_sandbox(self, workspace, sandbox_dir):
        """get_tracked_dict should include sandbox entries."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            monitor.add_sandbox_path(sandbox_dir)

            sb_file = os.path.join(sandbox_dir, "new.txt")
            monitor._on_sandbox_fs_event(sb_file, "created", sandbox_dir)

            tracked = monitor.get_tracked_dict()
            assert sb_file in tracked
            assert tracked[sb_file] == "created"
        finally:
            monitor.stop()

    def test_restore_preserves_sandbox_entries(self, workspace, sandbox_dir):
        """restore() should preserve sandbox entries in tracked dict."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            sb_file = os.path.join(sandbox_dir, "restored.txt")
            persisted = {
                "workspace_file.txt": "modified",
                sb_file: "created",
            }

            monitor.restore(persisted)
            assert monitor.tracked["workspace_file.txt"] == "modified"
            assert monitor.tracked[sb_file] == "created"
        finally:
            monitor.stop()

    def test_active_file_count_includes_sandbox(self, workspace, sandbox_dir):
        """active_file_count should count sandbox files too."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            monitor.add_sandbox_path(sandbox_dir)

            # Add workspace file
            ws_file = os.path.join(workspace, "new.txt")
            monitor._on_fs_event(ws_file, "created")

            # Add sandbox file
            sb_file = os.path.join(sandbox_dir, "new.txt")
            monitor._on_sandbox_fs_event(sb_file, "created", sandbox_dir)

            assert monitor.active_file_count == 2
        finally:
            monitor.stop()


# ---------------------------------------------------------------------------
# Tests: Reconcile with sandbox paths
# ---------------------------------------------------------------------------

class TestReconcileWithSandbox:
    """Tests for reconcile() with sandbox path support.

    reconcile() detects changes that happened while the server was down
    (between stop and restart).  To test this, we simulate the restart
    cycle: start → restore persisted state → create/delete files on disk
    "offline" → reconcile.
    """

    def test_reconcile_detects_deleted_sandbox_file(self, workspace, sandbox_dir):
        """reconcile() should detect tracked sandbox files deleted while down.

        Simulates server restart: start monitor, track file as modified,
        persist, stop, delete file on disk, restart, restore, reconcile.
        """
        # Phase 1: original server run — file was modified during session
        monitor1 = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor1.add_sandbox_path(sandbox_dir)
        monitor1.start()
        config_path = os.path.join(sandbox_dir, "config.json")
        monitor1.tracked[config_path] = "modified"
        persisted = monitor1.get_tracked_dict()
        monitor1.stop()

        # Phase 2: while server is down, the file is deleted
        os.remove(config_path)

        # Phase 3: server restarts
        monitor2 = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor2.add_sandbox_path(sandbox_dir)
        monitor2.start()
        try:
            monitor2.restore(persisted)
            changes = monitor2.reconcile()

            del_changes = [c for c in changes if c["path"] == config_path]
            assert len(del_changes) == 1
            assert del_changes[0]["status"] == "deleted"
        finally:
            monitor2.stop()

    def test_reconcile_created_sandbox_file_deleted_while_down(self, workspace, sandbox_dir):
        """reconcile() should handle created-then-gone sandbox files.

        If a sandbox file was tracked as "created" (not in baseline) and
        then deleted while the server was down, it should vanish from
        tracked entirely.
        """
        monitor1 = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor1.add_sandbox_path(sandbox_dir)
        monitor1.start()
        # Track a file as "created" (it was born during the session)
        new_path = os.path.join(sandbox_dir, "ephemeral.txt")
        with open(new_path, "w") as f:
            f.write("temp")
        monitor1.tracked[new_path] = "created"
        persisted = monitor1.get_tracked_dict()
        monitor1.stop()

        # Delete while server is down
        os.remove(new_path)

        # Restart
        monitor2 = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor2.add_sandbox_path(sandbox_dir)
        monitor2.start()
        try:
            monitor2.restore(persisted)
            changes = monitor2.reconcile()

            # Should report deletion
            del_changes = [c for c in changes if c["path"] == new_path]
            assert len(del_changes) == 1
            assert del_changes[0]["status"] == "deleted"
            # Should no longer be in tracked (created-then-gone vanishes)
            assert new_path not in monitor2.tracked
        finally:
            monitor2.stop()

    def test_reconcile_both_workspace_and_sandbox_deletions(self, workspace, sandbox_dir):
        """reconcile() should handle deletions in both workspace and sandbox."""
        monitor1 = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor1.add_sandbox_path(sandbox_dir)
        monitor1.start()

        # Track files as modified in both locations
        ws_tracked = "file1.txt"
        sb_tracked = os.path.join(sandbox_dir, "config.json")
        monitor1.tracked[ws_tracked] = "modified"
        monitor1.tracked[sb_tracked] = "modified"
        persisted = monitor1.get_tracked_dict()
        monitor1.stop()

        # Delete both while server is down
        os.remove(os.path.join(workspace, "file1.txt"))
        os.remove(sb_tracked)

        # Restart
        monitor2 = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor2.add_sandbox_path(sandbox_dir)
        monitor2.start()
        try:
            monitor2.restore(persisted)
            changes = monitor2.reconcile()
            paths = {c["path"] for c in changes}

            assert ws_tracked in paths
            assert sb_tracked in paths
        finally:
            monitor2.stop()


# ---------------------------------------------------------------------------
# Tests: Deferred sandbox path addition (before start)
# ---------------------------------------------------------------------------

class TestDeferredSandboxPaths:
    """Tests for adding sandbox paths before start() is called."""

    def test_add_sandbox_before_start(self, workspace, sandbox_dir):
        """Sandbox paths added before start() should be watched after start()."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)

        # Add before starting
        monitor.add_sandbox_path(sandbox_dir)
        assert sandbox_dir in monitor._sandbox_baselines
        assert sandbox_dir not in monitor._sandbox_watches

        monitor.start()
        try:
            # After start, the watch should be scheduled
            assert sandbox_dir in monitor._sandbox_watches
        finally:
            monitor.stop()


# ---------------------------------------------------------------------------
# Tests: Stop cleans up sandbox state
# ---------------------------------------------------------------------------

class TestStopCleanup:
    """Tests that stop() properly cleans up sandbox state."""

    def test_stop_clears_sandbox_state(self, workspace, sandbox_dir):
        """stop() should clear sandbox baselines and watches."""
        monitor = WorkspaceMonitor(workspace, on_changed=lambda c: None)
        monitor.start()
        try:
            monitor.add_sandbox_path(sandbox_dir)
            assert len(monitor._sandbox_baselines) == 1
            assert len(monitor._sandbox_watches) == 1
        finally:
            monitor.stop()

        assert len(monitor._sandbox_baselines) == 0
        assert len(monitor._sandbox_watches) == 0


# ---------------------------------------------------------------------------
# Tests: Accumulator integration
# ---------------------------------------------------------------------------

class TestAccumulatorIntegration:
    """Tests that sandbox events go through the debounce accumulator."""

    def test_sandbox_event_reaches_callback(self, workspace, sandbox_dir):
        """Sandbox file changes should eventually reach the on_changed callback."""
        changes_received = []
        event = threading.Event()

        def on_changed(changes):
            changes_received.extend(changes)
            event.set()

        # Use very short debounce for testing
        monitor = WorkspaceMonitor(workspace, on_changed=on_changed)
        monitor._accumulator = _ChangeAccumulator(on_flush=monitor._handle_flush, debounce=0.05)
        monitor.start()
        try:
            monitor.add_sandbox_path(sandbox_dir)

            # Trigger a sandbox event
            new_file = os.path.join(sandbox_dir, "callback_test.txt")
            monitor._on_sandbox_fs_event(new_file, "created", sandbox_dir)

            # Wait for debounce
            event.wait(timeout=2.0)

            assert len(changes_received) > 0
            assert any(c["path"] == new_file for c in changes_received)
        finally:
            monitor.stop()
