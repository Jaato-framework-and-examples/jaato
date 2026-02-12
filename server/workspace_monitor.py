"""Workspace file monitoring using watchdog.

Tracks filesystem changes within a session's workspace directory, maintaining
a delta of files created, modified, or deleted since the session was started.

Architecture
------------
- On ``start()``, an initial ``os.walk`` seeds the **baseline** set (files that
  existed before the session).
- A watchdog ``Observer`` watches the directory recursively.
- Filesystem events are accumulated into a debounce buffer; after 300 ms of
  quiet the buffer is flushed as a single ``WorkspaceFilesChangedEvent``.
- The ``tracked`` dict maps relative paths to a ``FileStatus`` string
  (``"created"``, ``"modified"``, or ``"deleted"``).
- Only the *delta from baseline* is exposed to clients.

Thread-safety
-------------
All mutable state is guarded by ``threading.Lock``.  The watchdog observer
and the debounce timer run on their own threads, so callers (session manager,
IPC broadcast) do not need additional synchronisation.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from shared.utils.gitignore import GitignoreParser

logger = logging.getLogger(__name__)

# How long (seconds) to wait after the last filesystem event before flushing
# the accumulated changes as a single batched event.
_DEBOUNCE_SECONDS = 0.3


class _ChangeAccumulator:
    """Thread-safe accumulator for batching filesystem changes.

    Collects add/remove operations and flushes them after a configurable
    quiet period.  The ``on_flush`` callback receives a list of
    ``{"path": str, "status": str}`` dicts.
    """

    def __init__(
        self,
        on_flush: Callable[[List[Dict[str, str]]], None],
        debounce: float = _DEBOUNCE_SECONDS,
    ):
        self._on_flush = on_flush
        self._debounce = debounce
        self._lock = threading.Lock()
        # Accumulated changes keyed by path; value is latest status.
        self._pending: Dict[str, str] = {}
        self._timer: Optional[threading.Timer] = None

    def record(self, path: str, status: str) -> None:
        """Record a change; resets the debounce timer.

        Args:
            path: Relative path that changed.
            status: One of ``"created"``, ``"modified"``, ``"deleted"``.
        """
        with self._lock:
            self._pending[path] = status
            # Reset timer
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        """Flush pending changes to the callback."""
        with self._lock:
            if not self._pending:
                return
            changes = [
                {"path": p, "status": s} for p, s in self._pending.items()
            ]
            self._pending.clear()
            self._timer = None

        # Invoke outside the lock.
        try:
            self._on_flush(changes)
        except Exception:
            logger.exception("Error in workspace change flush callback")

    def cancel(self) -> None:
        """Cancel any pending flush."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._pending.clear()


class _EventHandler(FileSystemEventHandler):
    """Watchdog handler that feeds events into the monitor."""

    def __init__(self, monitor: "WorkspaceMonitor"):
        super().__init__()
        self._monitor = monitor

    def on_created(self, event):
        if not event.is_directory:
            self._monitor._on_fs_event(event.src_path, "created")

    def on_modified(self, event):
        if not event.is_directory:
            self._monitor._on_fs_event(event.src_path, "modified")

    def on_deleted(self, event):
        if not event.is_directory:
            self._monitor._on_fs_event(event.src_path, "deleted")

    def on_moved(self, event):
        if not event.is_directory:
            self._monitor._on_fs_event(event.src_path, "deleted")
            self._monitor._on_fs_event(event.dest_path, "created")


class WorkspaceMonitor:
    """Monitors a workspace directory and tracks file changes since session start.

    Lifecycle:
        1. ``__init__(workspace_path, on_changed)`` – creates the monitor.
        2. ``start()`` – seeds baseline via ``os.walk``, starts watchdog.
        3. Filesystem events arrive → debounced → ``on_changed`` callback.
        4. ``stop()`` – tears down watchdog and debounce timer.

    The ``tracked`` dict holds only the *session delta*:
    - ``"created"`` – file did not exist in baseline, now exists.
    - ``"modified"`` – file existed in baseline and was modified.
    - ``"deleted"`` – file existed in baseline and was deleted.

    Files created and then deleted within the same session vanish from
    ``tracked`` entirely.

    Attributes:
        workspace_path: The root directory being monitored.
        tracked: Current delta dict ``{relative_path: status}``.
        baseline: Set of relative paths present when the session started.
    """

    def __init__(
        self,
        workspace_path: str,
        on_changed: Callable[[List[Dict[str, str]]], None],
    ):
        """Initialize the workspace monitor.

        Args:
            workspace_path: Absolute path to the workspace root.
            on_changed: Callback receiving a list of change dicts
                        ``[{"path": str, "status": str}, ...]`` after each
                        debounced batch.
        """
        self.workspace_path = os.path.abspath(workspace_path)
        self._on_changed = on_changed

        self._lock = threading.Lock()
        self.baseline: Set[str] = set()
        self.tracked: Dict[str, str] = {}

        self._gitignore: Optional[GitignoreParser] = None
        self._observer: Optional[Observer] = None
        self._accumulator = _ChangeAccumulator(on_flush=self._handle_flush)
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Seed the baseline and start watching for changes.

        Safe to call multiple times; subsequent calls are no-ops if already
        running.
        """
        if self._running:
            return

        ws = Path(self.workspace_path)
        if not ws.is_dir():
            logger.warning("Workspace path does not exist: %s", self.workspace_path)
            return

        # Build gitignore filter.
        self._gitignore = GitignoreParser(ws, include_defaults=True)

        # Seed baseline via os.walk.
        self._seed_baseline()

        # Start watchdog observer.
        self._observer = Observer()
        handler = _EventHandler(self)
        self._observer.schedule(handler, self.workspace_path, recursive=True)
        self._observer.daemon = True
        self._observer.start()
        self._running = True
        logger.info(
            "Workspace monitor started: %s (%d baseline files)",
            self.workspace_path,
            len(self.baseline),
        )

    def stop(self) -> None:
        """Stop watching and release resources."""
        self._running = False
        self._accumulator.cancel()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None
        logger.info("Workspace monitor stopped: %s", self.workspace_path)

    def get_snapshot(self) -> List[Dict[str, str]]:
        """Return the full current tracked state as a list of change dicts.

        Used when a client reconnects and needs the complete delta since
        session start.

        Returns:
            List of ``{"path": str, "status": str}`` for every tracked file.
        """
        with self._lock:
            return [
                {"path": p, "status": s} for p, s in self.tracked.items()
            ]

    def get_tracked_dict(self) -> Dict[str, str]:
        """Return a copy of the tracked dict for persistence.

        Returns:
            Dict mapping relative path → status string.
        """
        with self._lock:
            return dict(self.tracked)

    def restore(self, tracked: Dict[str, str], baseline: Optional[Set[str]] = None) -> None:
        """Restore previously persisted state.

        Called during session reload to bring the monitor back to its
        pre-shutdown state before starting the watchdog.

        After restoring, call ``reconcile()`` to detect anything that
        changed on disk while the server was down.

        Args:
            tracked: Previously persisted tracked dict.
            baseline: Previously persisted baseline set.  If None the
                      current baseline (from ``start()``) is kept.
        """
        with self._lock:
            self.tracked = dict(tracked)
            if baseline is not None:
                self.baseline = set(baseline)

    def reconcile(self) -> List[Dict[str, str]]:
        """Compare persisted state against actual disk and emit a delta.

        Must be called **after** ``start()`` and ``restore()`` so that
        both the watchdog and the persisted state are available.

        Returns:
            List of change dicts representing what changed while the
            server was down (may be empty).
        """
        ws = Path(self.workspace_path)
        current_files: Set[str] = set()
        for dirpath, dirnames, filenames in os.walk(self.workspace_path):
            # Prune ignored directories in-place.
            dirnames[:] = [
                d for d in dirnames
                if not self._is_ignored(os.path.join(dirpath, d), is_dir=True)
            ]
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                if not self._is_ignored(full, is_dir=False):
                    rel = os.path.relpath(full, self.workspace_path)
                    current_files.add(rel)

        changes: List[Dict[str, str]] = []
        with self._lock:
            # Files that exist now but weren't in baseline → might be new.
            for f in current_files - self.baseline:
                if f not in self.tracked:
                    self.tracked[f] = "created"
                    changes.append({"path": f, "status": "created"})

            # Files tracked as created/modified but no longer on disk → deleted.
            for f, status in list(self.tracked.items()):
                if status in ("created", "modified") and f not in current_files:
                    if f in self.baseline:
                        self.tracked[f] = "deleted"
                        changes.append({"path": f, "status": "deleted"})
                    else:
                        del self.tracked[f]
                        changes.append({"path": f, "status": "deleted"})

        return changes

    @property
    def active_file_count(self) -> int:
        """Number of created/modified files (excludes deleted)."""
        with self._lock:
            return sum(
                1 for s in self.tracked.values() if s != "deleted"
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _seed_baseline(self) -> None:
        """Walk the workspace and populate ``self.baseline``."""
        baseline: Set[str] = set()
        for dirpath, dirnames, filenames in os.walk(self.workspace_path):
            # Prune ignored directories in-place so os.walk skips them.
            dirnames[:] = [
                d for d in dirnames
                if not self._is_ignored(os.path.join(dirpath, d), is_dir=True)
            ]
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                if not self._is_ignored(full, is_dir=False):
                    rel = os.path.relpath(full, self.workspace_path)
                    baseline.add(rel)

        with self._lock:
            self.baseline = baseline

    def _is_ignored(self, abs_path: str, is_dir: bool = False) -> bool:
        """Check if a path should be ignored.

        Args:
            abs_path: Absolute filesystem path.
            is_dir: Whether the path is a directory.

        Returns:
            True if the path matches gitignore or default ignore patterns.
        """
        if self._gitignore is None:
            return False

        p = Path(abs_path)
        # For directories, append a trailing separator so the parser
        # can match directory-only patterns.
        if is_dir:
            # Create a fake child so relative_to works and the parser sees
            # the directory name in the path parts.
            return self._gitignore.is_ignored(p)
        return self._gitignore.is_ignored(p)

    def _on_fs_event(self, abs_path: str, event_type: str) -> None:
        """Handle a raw filesystem event from watchdog.

        Translates the event into the session-delta semantics and feeds
        it into the debounce accumulator.

        Args:
            abs_path: Absolute path of the changed file.
            event_type: One of ``"created"``, ``"modified"``, ``"deleted"``.
        """
        if not self._running:
            return

        # Ignore paths outside workspace (shouldn't happen, but guard).
        try:
            rel = os.path.relpath(abs_path, self.workspace_path)
        except ValueError:
            return

        # Filter ignored paths.
        if self._is_ignored(abs_path, is_dir=False):
            return

        with self._lock:
            in_baseline = rel in self.baseline
            current_status = self.tracked.get(rel)

            if event_type == "created":
                if in_baseline:
                    # Was in baseline → treat as modified (re-appeared).
                    new_status = "modified"
                else:
                    new_status = "created"
                self.tracked[rel] = new_status

            elif event_type == "modified":
                if current_status == "created":
                    # Still "created" – was born this session, modification
                    # doesn't change the status.
                    new_status = "created"
                elif in_baseline:
                    new_status = "modified"
                elif current_status is None and not in_baseline:
                    # Modified event for a file not in baseline and not
                    # tracked → treat as created (watchdog sometimes fires
                    # modified before created on some platforms).
                    new_status = "created"
                else:
                    new_status = current_status or "modified"
                self.tracked[rel] = new_status

            elif event_type == "deleted":
                if current_status == "created":
                    # Created this session then deleted → vanish.
                    del self.tracked[rel]
                    new_status = "deleted"
                elif in_baseline:
                    self.tracked[rel] = "deleted"
                    new_status = "deleted"
                elif current_status is not None:
                    # Was tracked (e.g., modified) but not in baseline.
                    del self.tracked[rel]
                    new_status = "deleted"
                else:
                    # Not tracked, not baseline – ignore.
                    return
            else:
                return

        self._accumulator.record(rel, new_status)

    def _handle_flush(self, changes: List[Dict[str, str]]) -> None:
        """Called by the accumulator after the debounce period.

        Forwards the batched changes to the external callback.

        Args:
            changes: List of ``{"path": str, "status": str}`` dicts.
        """
        if changes:
            self._on_changed(changes)
