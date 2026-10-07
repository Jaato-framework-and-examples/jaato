"""The TUI's ``workspace_clear`` survives a reattach (#1189).

``clear()`` emptied the panel and the next ``WorkspaceFilesSnapshotEvent`` --
sent on every reattach -- brought the whole list back, because a snapshot
entry does not say when it changed.  With the daemon's numbering (protocol
1.19) the panel keeps only what changed after the clear, and drops the mark
rather than comparing it when the monitor was rebuilt or the daemon sends no
numbering.
"""
from workspace_panel import WorkspacePanel


def _entries(panel):
    return dict(panel._files)


def test_a_cleared_panel_stays_cleared_across_a_snapshot():
    p = WorkspacePanel()
    p.apply_changes([{"path": "old.py", "status": "created"}], seq=1, epoch="e1")
    p.clear()
    p.apply_changes([{"path": "old.py", "status": "created"}], seq=2, epoch="e1")
    p.apply_snapshot(
        [{"path": "old.py", "status": "created"}, {"path": "gone.py", "status": "modified"}],
        seq=2, epoch="e1", seqs={"old.py": 2, "gone.py": 1},
    )
    # old.py changed again after the clear; gone.py did not.
    assert _entries(p) == {"old.py": "created"}


def test_a_mark_from_a_rebuilt_monitor_is_dropped_not_compared():
    p = WorkspacePanel()
    p.apply_changes([{"path": "a.py", "status": "created"}], seq=500, epoch="e1")
    p.clear()
    # The session was reloaded: the new monitor counts from 0.
    p.apply_snapshot([{"path": "a.py", "status": "created"}], seq=3, epoch="e2", seqs={"a.py": 3})
    assert _entries(p) == {"a.py": "created"}
    assert p._clear_mark is None


def test_without_numbering_a_snapshot_behaves_as_it_always_did():
    """The control: an older daemon -- the snapshot replaces the list."""
    p = WorkspacePanel()
    p.apply_changes([{"path": "a.py", "status": "created"}])
    p.clear()
    p.apply_snapshot([{"path": "a.py", "status": "created"}])
    assert _entries(p) == {"a.py": "created"}
