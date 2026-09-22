"""The Files panel's reset survives a reconnect (#1189).

A client can reset its Files panel to "only what changes from now on" -- the
TUI's ``workspace_clear``.  Kept only on the client, that reset was undone by
the next reconnect: the daemon rebuilds an attaching client's list from a
``WorkspaceFilesSnapshotEvent`` applied wholesale, and ``{path, status}`` does
not say WHEN a file changed.  A browser reconnects routinely, so a naive
reset worked in testing and stopped working in use.

The daemon now numbers every flushed batch (``seq``) and names the monitor
instance that numbered it (``epoch``); the snapshot carries each path's
latest number (``seqs``).  The epoch is the part that keeps this from failing
silently: a monitor rebuilt on session reload counts again from 0, and a
client comparing an old mark against the new counter would hide every new
change.  So restored entries carry no number of their own, and a new
instance has a new epoch.

The web client's half -- keeping the mark, filtering, voiding it out loud --
is ``jaato-web-coder-ui/src/store/workspaceView.ts`` and its tests.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List

import pytest

from server.session_manager import SessionManager
from server.workspace_monitor import ChangeBatch, WorkspaceMonitor
from shared.tests.reversion import Reversion

_MON = "jaato-server/server/workspace_monitor.py"
_SM = "jaato-server/server/session_manager.py"

REVERSIONS = [
    Reversion(
        target=_SM,
        find="""                seq=getattr(changes, "seq", None),
                epoch=getattr(changes, "epoch", None),
""",
        replace="",
        test="test_the_changed_event_carries_the_batch_number",
        because=(
            "the monitor numbers the batch and the emitter drops it, so no "
            "client ever learns when a file changed"
        ),
    ),
    Reversion(
        target=_SM,
        find="""        snapshot = monitor.get_snapshot()
        numbering = monitor.get_sequence_state()
        self._emit_to_client(client_id, WorkspaceFilesSnapshotEvent(""",
        replace="""        snapshot = monitor.get_snapshot()
        numbering = monitor.get_sequence_state()
        if snapshot: self._emit_to_client(client_id, WorkspaceFilesSnapshotEvent(""",
        test="test_an_empty_snapshot_is_still_sent_because_it_carries_the_epoch",
        because=(
            "an empty monitor sent no snapshot, so a reconnecting client kept "
            "its old list and never learned the monitor had been rebuilt"
        ),
    ),
    Reversion(
        target=_MON,
        find="""        self.seq += 1
        for change in changes:""",
        replace="""        self.seq = 1
        for change in changes:""",
        test="test_each_batch_is_numbered_one_more_than_the_last",
        because="a number that does not increase cannot say what changed after a mark",
    ),
]


def _monitor(tmp_path, sink: List[Any]) -> WorkspaceMonitor:
    return WorkspaceMonitor(str(tmp_path), on_changed=sink.append)


def _flush(m: WorkspaceMonitor, *changes: tuple) -> None:
    """Drive one flushed batch the way the accumulator would."""
    with m._lock:
        for path, status in changes:
            if status == "deleted":
                m.tracked.pop(path, None)
            else:
                m.tracked[path] = status
    m._handle_flush([{"path": p, "status": s} for p, s in changes])


def test_each_batch_is_numbered_one_more_than_the_last(tmp_path):
    sink: List[Any] = []
    m = _monitor(tmp_path, sink)
    _flush(m, ("a.py", "created"))
    _flush(m, ("b.py", "created"), ("a.py", "created"))
    assert [b.seq for b in sink] == [1, 2]
    assert m.file_seqs == {"a.py": 2, "b.py": 2}


def test_a_batch_is_still_the_list_every_caller_expects(tmp_path):
    """The control: existing callbacks extend a list and must keep working."""
    sink: List[Any] = []
    _flush(_monitor(tmp_path, sink), ("a.py", "created"))
    batch = sink[0]
    assert isinstance(batch, ChangeBatch) and isinstance(batch, list)
    assert list(batch) == [{"path": "a.py", "status": "created"}]


def test_a_deleted_path_drops_its_number(tmp_path):
    sink: List[Any] = []
    m = _monitor(tmp_path, sink)
    _flush(m, ("a.py", "created"))
    _flush(m, ("a.py", "deleted"))
    assert "a.py" not in m.file_seqs
    assert m.get_sequence_state()["seqs"] == {}


def test_restored_entries_carry_no_number_from_the_previous_monitor(tmp_path):
    first = _monitor(tmp_path, [])
    _flush(first, ("a.py", "created"))
    second = _monitor(tmp_path, [])
    _flush(second, ("x.py", "created"))
    second.restore(first.get_tracked_dict())
    state = second.get_sequence_state()
    assert state["seqs"] == {}
    assert state["epoch"] != first.epoch


def test_every_monitor_instance_has_its_own_epoch(tmp_path):
    assert _monitor(tmp_path, []).epoch != _monitor(tmp_path, []).epoch


def _manager(monitor: WorkspaceMonitor, session_id: str = "s1") -> tuple:
    sm = SessionManager.__new__(SessionManager)
    sm._lock = threading.RLock()
    sm._sessions = {}
    sm._workspace_monitors = {session_id: monitor}
    sent: List[tuple] = []
    sm._emit_to_client = lambda cid, ev: sent.append((cid, ev))
    sm._emit_to_session = lambda sid, ev: sent.append((sid, ev))
    return sm, sent


def test_the_snapshot_carries_each_paths_number(tmp_path):
    m = _monitor(tmp_path, [])
    _flush(m, ("a.py", "created"))
    _flush(m, ("b.py", "modified"))
    sm, sent = _manager(m)
    sm._send_workspace_snapshot("s1", "c1")
    (_cid, ev), = sent
    assert ev.epoch == m.epoch and ev.seq == 2
    assert ev.seqs == {"a.py": 1, "b.py": 2}


def test_an_empty_snapshot_is_still_sent_because_it_carries_the_epoch(tmp_path):
    m = _monitor(tmp_path, [])
    sm, sent = _manager(m)
    sm._send_workspace_snapshot("s1", "c1")
    assert len(sent) == 1
    assert sent[0][1].files == [] and sent[0][1].epoch == m.epoch


def test_the_changed_event_carries_the_batch_number(tmp_path, monkeypatch):
    sm, sent = _manager(_monitor(tmp_path, []))
    sm._workspace_monitors = {}
    monkeypatch.setattr(sm, "_stop_workspace_monitor", lambda sid: None, raising=False)
    monkeypatch.setattr(sm, "_wire_sandbox_to_monitor", lambda *a, **k: None, raising=False)
    sm._start_workspace_monitor("s1", str(tmp_path), server=object())
    monitor = sm._workspace_monitors["s1"]
    try:
        _flush(monitor, ("a.py", "created"))
    finally:
        monitor.stop()
    events: List[Dict[str, Any]] = [ev for _sid, ev in sent]
    assert events and events[-1].seq == 1 and events[-1].epoch == monitor.epoch
    assert events[-1].changes == [{"path": "a.py", "status": "created"}]
