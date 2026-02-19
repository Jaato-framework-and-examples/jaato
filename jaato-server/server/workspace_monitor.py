"""Workspace file monitoring using watchdog.

Tracks filesystem changes within a session's workspace directory, maintaining
a delta of files created, modified, or deleted since the session was started.

Also supports monitoring additional *sandbox paths* — directories that the
user has granted readwrite access to via the ``sandbox add readwrite`` command.
Files within sandbox paths are tracked using **absolute paths** in the
``tracked`` dict so they don't collide with workspace-relative entries.

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
- Sandbox paths are tracked with absolute path keys in the same ``tracked``
  dict, distinguishable by their leading ``/``.

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
    """Watchdog handler that feeds events into the monitor.

    Handles events from both the main workspace watch and sandbox path
    watches.  The ``sandbox_root`` attribute, when set, tells the monitor
    that events originate from a sandbox path and should be tracked with
    absolute paths.
    """

    def __init__(self, monitor: "WorkspaceMonitor", sandbox_root: Optional[str] = None):
        super().__init__()
        self._monitor = monitor
        self.sandbox_root = sandbox_root

    def on_created(self, event):
        if not event.is_directory:
            self._monitor._on_fs_event(event.src_path, "created", sandbox_root=self.sandbox_root)

    def on_modified(self, event):
        if not event.is_directory:
            self._monitor._on_fs_event(event.src_path, "modified", sandbox_root=self.sandbox_root)

    def on_deleted(self, event):
        if not event.is_directory:
            self._monitor._on_fs_event(event.src_path, "deleted", sandbox_root=self.sandbox_root)

    def on_moved(self, event):
        if not event.is_directory:
            self._monitor._on_fs_event(event.src_path, "deleted", sandbox_root=self.sandbox_root)
            self._monitor._on_fs_event(event.dest_path, "created", sandbox_root=self.sandbox_root)


class WorkspaceMonitor:
    """Monitors a workspace directory and tracks file changes since session start.

    Also monitors additional *sandbox paths* — directories the user has
    granted readwrite access to.  Files from sandbox paths are tracked in the
    same ``tracked`` dict but keyed by their **absolute path** (which always
    starts with ``/``) so they are distinguishable from workspace-relative
    entries.  Each sandbox path gets its own watchdog schedule on the same
    observer.

    Lifecycle:
        1. ``__init__(workspace_path, on_changed)`` – creates the monitor.
        2. ``start()`` – seeds baseline via ``os.walk``, starts watchdog.
        3. Filesystem events arrive → debounced → ``on_changed`` callback.
        4. ``add_sandbox_path(path)`` / ``remove_sandbox_path(path)`` – manage
           additional watched directories at runtime.
        5. ``stop()`` – tears down watchdog and debounce timer.

    The ``tracked`` dict holds only the *session delta*:
    - ``"created"`` – file did not exist in baseline, now exists.
    - ``"modified"`` – file existed in baseline and was modified.
    - ``"deleted"`` – file existed in baseline and was deleted.

    Files created and then deleted within the same session vanish from
    ``tracked`` entirely.

    Attributes:
        workspace_path: The root directory being monitored.
        tracked: Current delta dict ``{path: status}``.  Workspace files use
            relative paths; sandbox files use absolute paths.
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

        # Sandbox path monitoring: maps absolute sandbox root → baseline set
        self._sandbox_baselines: Dict[str, Set[str]] = {}
        # Maps absolute sandbox root → watchdog ObservedWatch handle
        self._sandbox_watches: Dict[str, object] = {}

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

        # Schedule watches for any sandbox paths that were added before start().
        for sandbox_path, baseline in list(self._sandbox_baselines.items()):
            if sandbox_path not in self._sandbox_watches:
                sb_handler = _EventHandler(self, sandbox_root=sandbox_path)
                try:
                    watch = self._observer.schedule(sb_handler, sandbox_path, recursive=True)
                    self._sandbox_watches[sandbox_path] = watch
                    logger.info(
                        "Sandbox path watch started (deferred): %s (%d baseline files)",
                        sandbox_path, len(baseline),
                    )
                except Exception:
                    logger.exception("Failed to schedule deferred sandbox watch for %s", sandbox_path)

    def stop(self) -> None:
        """Stop watching and release resources.

        Stops the main workspace watch and all sandbox watches.
        """
        self._running = False
        self._accumulator.cancel()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None
        self._sandbox_watches.clear()
        self._sandbox_baselines.clear()
        logger.info("Workspace monitor stopped: %s", self.workspace_path)

    # ------------------------------------------------------------------
    # Sandbox path management
    # ------------------------------------------------------------------

    def add_sandbox_path(self, abs_path: str) -> None:
        """Start watching a sandbox path for file changes.

        Sandbox paths are directories outside the workspace that the user
        has granted readwrite access to.  Files within them are tracked
        using absolute paths in the ``tracked`` dict.

        Safe to call before ``start()`` — the watch will be deferred until
        the observer is running.  Calling with an already-watched path
        is a no-op.

        Args:
            abs_path: Absolute path to the directory to monitor.
        """
        abs_path = os.path.abspath(abs_path)

        # Skip if same as workspace (already watched)
        if os.path.realpath(abs_path) == os.path.realpath(self.workspace_path):
            return

        # Skip if already watched
        if abs_path in self._sandbox_watches:
            return

        if not os.path.isdir(abs_path):
            logger.warning("Sandbox path does not exist or is not a directory: %s", abs_path)
            return

        # Seed baseline for this sandbox path
        sandbox_baseline: Set[str] = set()
        for dirpath, dirnames, filenames in os.walk(abs_path):
            # Skip hidden directories by default for sandbox paths
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]
            for fname in filenames:
                if not fname.startswith('.'):
                    full = os.path.join(dirpath, fname)
                    sandbox_baseline.add(full)

        with self._lock:
            self._sandbox_baselines[abs_path] = sandbox_baseline

        # Schedule watchdog watch if observer is running
        if self._observer is not None and self._running:
            handler = _EventHandler(self, sandbox_root=abs_path)
            try:
                watch = self._observer.schedule(handler, abs_path, recursive=True)
                self._sandbox_watches[abs_path] = watch
                logger.info(
                    "Sandbox path watch started: %s (%d baseline files)",
                    abs_path, len(sandbox_baseline),
                )
            except Exception:
                logger.exception("Failed to schedule sandbox watch for %s", abs_path)
        else:
            # Observer not running yet — store baseline so it can be
            # scheduled when start() is called or observer restarts.
            logger.debug(
                "Sandbox path queued (observer not running): %s (%d baseline files)",
                abs_path, len(sandbox_baseline),
            )

    def remove_sandbox_path(self, abs_path: str) -> None:
        """Stop watching a sandbox path and remove its tracked entries.

        Args:
            abs_path: Absolute path of the sandbox directory to stop monitoring.
        """
        abs_path = os.path.abspath(abs_path)

        # Unschedule watchdog watch
        watch = self._sandbox_watches.pop(abs_path, None)
        if watch is not None and self._observer is not None:
            try:
                self._observer.unschedule(watch)
            except Exception:
                logger.debug("Failed to unschedule sandbox watch for %s (may already be stopped)", abs_path)

        with self._lock:
            self._sandbox_baselines.pop(abs_path, None)

            # Remove tracked entries that belong to this sandbox root
            prefix = abs_path + os.sep
            to_remove = [
                key for key in self.tracked
                if key == abs_path or key.startswith(prefix)
            ]
            for key in to_remove:
                del self.tracked[key]

        if watch is not None:
            logger.info("Sandbox path watch stopped: %s", abs_path)

    def get_sandbox_paths(self) -> List[str]:
        """Return the list of currently monitored sandbox paths.

        Returns:
            List of absolute paths being monitored as sandbox paths.
        """
        return list(self._sandbox_baselines.keys())

    def update_sandbox_paths(self, readwrite_paths: List[str]) -> None:
        """Synchronise the set of watched sandbox paths.

        Adds watches for new paths and removes watches for paths that are
        no longer in the provided list.  The workspace path itself is always
        excluded.

        Args:
            readwrite_paths: Complete list of absolute paths that should be
                monitored.  Paths not already watched will be added; paths
                currently watched but absent from this list will be removed.
        """
        desired = set()
        ws_real = os.path.realpath(self.workspace_path)
        for p in readwrite_paths:
            p = os.path.abspath(p)
            if os.path.realpath(p) != ws_real:
                desired.add(p)

        current = set(self._sandbox_baselines.keys())

        for p in desired - current:
            self.add_sandbox_path(p)

        for p in current - desired:
            self.remove_sandbox_path(p)

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

        Reconciles both workspace files and sandbox-path files.

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

        # Also scan sandbox paths
        sandbox_current: Set[str] = set()
        for sandbox_root in self._sandbox_baselines:
            if os.path.isdir(sandbox_root):
                for dirpath, dirnames, filenames in os.walk(sandbox_root):
                    dirnames[:] = [d for d in dirnames if not d.startswith('.')]
                    for fname in filenames:
                        if not fname.startswith('.'):
                            full = os.path.join(dirpath, fname)
                            sandbox_current.add(full)

        changes: List[Dict[str, str]] = []
        with self._lock:
            # --- Workspace reconciliation ---
            # Files that exist now but weren't in baseline → might be new.
            for f in current_files - self.baseline:
                if f not in self.tracked:
                    self.tracked[f] = "created"
                    changes.append({"path": f, "status": "created"})

            # Files tracked as created/modified but no longer on disk → deleted.
            # Only check workspace-relative paths (not starting with /)
            for f, status in list(self.tracked.items()):
                if os.path.isabs(f):
                    continue  # sandbox entries handled below
                if status in ("created", "modified") and f not in current_files:
                    if f in self.baseline:
                        self.tracked[f] = "deleted"
                        changes.append({"path": f, "status": "deleted"})
                    else:
                        del self.tracked[f]
                        changes.append({"path": f, "status": "deleted"})

            # --- Sandbox reconciliation ---
            all_sandbox_baseline = set()
            for bl in self._sandbox_baselines.values():
                all_sandbox_baseline |= bl

            # New sandbox files not in baseline
            for f in sandbox_current - all_sandbox_baseline:
                if f not in self.tracked:
                    self.tracked[f] = "created"
                    changes.append({"path": f, "status": "created"})

            # Sandbox files tracked but gone from disk
            for f, status in list(self.tracked.items()):
                if not os.path.isabs(f):
                    continue  # workspace entries handled above
                if status in ("created", "modified") and f not in sandbox_current:
                    if f in all_sandbox_baseline:
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

    def _on_fs_event(
        self,
        abs_path: str,
        event_type: str,
        sandbox_root: Optional[str] = None,
    ) -> None:
        """Handle a raw filesystem event from watchdog.

        Translates the event into the session-delta semantics and feeds
        it into the debounce accumulator.

        For workspace events (``sandbox_root is None``), the tracked key is
        a path relative to the workspace.  For sandbox events, the tracked
        key is the **absolute path** of the file.

        Args:
            abs_path: Absolute path of the changed file.
            event_type: One of ``"created"``, ``"modified"``, ``"deleted"``.
            sandbox_root: If set, the event comes from a sandbox path watch
                and this is the absolute root of that sandbox directory.
        """
        if not self._running:
            return

        if sandbox_root is not None:
            self._on_sandbox_fs_event(abs_path, event_type, sandbox_root)
            return

        # --- Workspace event (original logic) ---

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

    def _on_sandbox_fs_event(
        self,
        abs_path: str,
        event_type: str,
        sandbox_root: str,
    ) -> None:
        """Handle a filesystem event from a sandbox path watch.

        Uses the absolute file path as the tracked key.  Baseline for the
        sandbox root is consulted to determine create vs modify semantics.

        Args:
            abs_path: Absolute path of the changed file.
            event_type: One of ``"created"``, ``"modified"``, ``"deleted"``.
            sandbox_root: Absolute root of the sandbox directory.
        """
        # Skip hidden files in sandbox paths
        basename = os.path.basename(abs_path)
        if basename.startswith('.'):
            return

        key = abs_path

        with self._lock:
            sandbox_baseline = self._sandbox_baselines.get(sandbox_root, set())
            in_baseline = abs_path in sandbox_baseline
            current_status = self.tracked.get(key)

            if event_type == "created":
                new_status = "modified" if in_baseline else "created"
                self.tracked[key] = new_status

            elif event_type == "modified":
                if current_status == "created":
                    new_status = "created"
                elif in_baseline:
                    new_status = "modified"
                elif current_status is None and not in_baseline:
                    new_status = "created"
                else:
                    new_status = current_status or "modified"
                self.tracked[key] = new_status

            elif event_type == "deleted":
                if current_status == "created":
                    del self.tracked[key]
                    new_status = "deleted"
                elif in_baseline:
                    self.tracked[key] = "deleted"
                    new_status = "deleted"
                elif current_status is not None:
                    del self.tracked[key]
                    new_status = "deleted"
                else:
                    return
            else:
                return

        self._accumulator.record(key, new_status)

    def _handle_flush(self, changes: List[Dict[str, str]]) -> None:
        """Called by the accumulator after the debounce period.

        Forwards the batched changes to the external callback.

        Args:
            changes: List of ``{"path": str, "status": str}`` dicts.
        """
        if changes:
            self._on_changed(changes)
