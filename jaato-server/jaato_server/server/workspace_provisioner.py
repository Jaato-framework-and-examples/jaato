"""Workspace Provisioner for Jaato Server.

Provisions isolated workspace directories for remote WebSocket sessions.
Each provisioned workspace is a subdirectory under
``{workspace_root}/sessions/{session_id}/``.  Templates from
``{workspace_root}/templates/{name}/`` are copied into new workspaces
to provide initial configuration (``.env``, ``.jaato/``).

Lifecycle:
    provision()       → workspace created, template applied
    get_workspace()   → lookup by session_id
    update_activity() → bump last_activity timestamp
    teardown()        → remove workspace directory
    reap_expired()    → remove workspaces exceeding max_age

The provisioner persists a manifest file at
``{workspace_root}/sessions/manifest.json`` so that workspace metadata
survives server restarts.
"""

import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProvisionedWorkspace:
    """Metadata for a server-provisioned workspace directory.

    Attributes:
        session_id: The session this workspace belongs to.
        path: Absolute filesystem path to the workspace directory.
        template: Name of the template that was copied, or None.
        created_at: ISO-8601 timestamp of creation.
        last_activity: ISO-8601 timestamp of last activity (updated on
            message send, tool execution, etc.).
        client_id: The client that owns this workspace, if any.
    """
    session_id: str
    path: str
    template: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_activity: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    client_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkspaceProvisioner:
    """Provisions isolated workspace directories for remote sessions.

    Each provisioned workspace is a subdirectory under
    ``{workspace_root}/sessions/{session_id}/``.  Templates from
    ``{workspace_root}/templates/{name}/`` are copied into new workspaces
    to provide initial configuration.

    Thread-safety: all public methods are protected by ``_lock``.
    """

    def __init__(
        self,
        workspace_root: str,
        default_template: str = "default",
    ):
        """Initialize the provisioner.

        Args:
            workspace_root: Root directory containing ``templates/`` and
                ``sessions/`` subdirectories.
            default_template: Name of the template to use when none is
                specified.  Resolved to ``{workspace_root}/templates/{name}/``.
        """
        self._root = Path(workspace_root).expanduser().resolve()
        self._sessions_dir = self._root / "sessions"
        self._templates_dir = self._root / "templates"
        self._manifest_path = self._sessions_dir / "manifest.json"
        self._default_template = default_template

        self._workspaces: Dict[str, ProvisionedWorkspace] = {}
        self._lock = threading.Lock()

        self._reaper_thread: Optional[threading.Thread] = None
        self._reaper_stop = threading.Event()

        # Ensure directories exist
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._templates_dir.mkdir(parents=True, exist_ok=True)

        # Load existing manifest
        self._load_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def provision(
        self,
        session_id: str,
        client_id: Optional[str] = None,
        template: Optional[str] = None,
    ) -> ProvisionedWorkspace:
        """Create an isolated workspace directory for a session.

        Steps:
            1. Create ``{sessions_dir}/{session_id}/``
            2. Copy template contents if the template directory exists
            3. Create ``.jaato/`` subdirectory
            4. Persist manifest

        Args:
            session_id: Unique session identifier.
            client_id: Owning client identifier, if any.
            template: Template name to copy from.  Falls back to the
                default template configured at init time.

        Returns:
            A ``ProvisionedWorkspace`` with the absolute path.

        Raises:
            ValueError: If a workspace already exists for this session_id.
        """
        with self._lock:
            if session_id in self._workspaces:
                raise ValueError(
                    f"Workspace already provisioned for session: {session_id}"
                )

            ws_path = self._sessions_dir / session_id
            ws_path.mkdir(parents=True, exist_ok=True)

            # Apply template
            template_name = template or self._default_template
            template_dir = self._templates_dir / template_name
            applied_template: Optional[str] = None

            if template_dir.is_dir():
                self._copy_template(template_dir, ws_path)
                applied_template = template_name
                logger.info(
                    "Applied template '%s' to workspace %s",
                    template_name,
                    session_id,
                )

            # Ensure .jaato directory exists
            (ws_path / ".jaato").mkdir(exist_ok=True)

            # Ensure .env exists (even if empty)
            env_file = ws_path / ".env"
            if not env_file.exists():
                env_file.touch()

            now = datetime.now(timezone.utc).isoformat()
            workspace = ProvisionedWorkspace(
                session_id=session_id,
                path=str(ws_path),
                template=applied_template,
                created_at=now,
                last_activity=now,
                client_id=client_id,
            )

            self._workspaces[session_id] = workspace
            self._save_manifest()

            logger.info("Provisioned workspace for session %s at %s", session_id, ws_path)
            return workspace

    def teardown(self, session_id: str) -> None:
        """Remove a provisioned workspace and its contents.

        Args:
            session_id: Session whose workspace should be removed.
                Silently ignores unknown session_ids.
        """
        with self._lock:
            workspace = self._workspaces.pop(session_id, None)
            if not workspace:
                return

            ws_path = Path(workspace.path)
            if ws_path.exists():
                shutil.rmtree(ws_path, ignore_errors=True)
                logger.info("Removed workspace for session %s", session_id)

            self._save_manifest()

    def reap_expired(self, max_age_seconds: int = 86400) -> List[str]:
        """Remove workspaces not accessed within ``max_age_seconds``.

        Args:
            max_age_seconds: Maximum age in seconds since last activity.
                Defaults to 24 hours.

        Returns:
            List of session_ids that were reaped.
        """
        now = datetime.now(timezone.utc)
        reaped: List[str] = []

        with self._lock:
            for session_id, ws in list(self._workspaces.items()):
                try:
                    last = datetime.fromisoformat(ws.last_activity)
                    age = (now - last).total_seconds()
                    if age > max_age_seconds:
                        reaped.append(session_id)
                except (ValueError, TypeError):
                    # Malformed timestamp — reap it
                    reaped.append(session_id)

            for session_id in reaped:
                ws = self._workspaces.pop(session_id, None)
                if ws:
                    ws_path = Path(ws.path)
                    if ws_path.exists():
                        shutil.rmtree(ws_path, ignore_errors=True)

            if reaped:
                self._save_manifest()

        if reaped:
            logger.info("Reaped %d expired workspaces: %s", len(reaped), reaped)

        return reaped

    def get_workspace(self, session_id: str) -> Optional[ProvisionedWorkspace]:
        """Look up a provisioned workspace by session_id.

        Returns:
            The ``ProvisionedWorkspace`` or None if not found.
        """
        with self._lock:
            return self._workspaces.get(session_id)

    def list_workspaces(self) -> List[ProvisionedWorkspace]:
        """List all provisioned workspaces.

        Returns:
            Snapshot of all tracked workspaces.
        """
        with self._lock:
            return list(self._workspaces.values())

    def update_activity(self, session_id: str) -> None:
        """Bump the ``last_activity`` timestamp for a workspace.

        Called on message send, tool execution, etc. to prevent reaping
        of active workspaces.

        Args:
            session_id: Session to update.  Silently ignores unknown ids.
        """
        with self._lock:
            ws = self._workspaces.get(session_id)
            if ws:
                ws.last_activity = datetime.now(timezone.utc).isoformat()
                # Don't save manifest on every activity bump — the reaper
                # will pick up stale entries on the next pass regardless.

    # ------------------------------------------------------------------
    # Reaper
    # ------------------------------------------------------------------

    def start_reaper(
        self,
        interval_seconds: int = 3600,
        max_age_seconds: int = 86400,
        on_teardown: Optional[Callable[[str], None]] = None,
    ) -> threading.Thread:
        """Start a background thread that periodically reaps expired workspaces.

        Args:
            interval_seconds: How often to check (default: hourly).
            max_age_seconds: Max age before reaping (default: 24h).
            on_teardown: Callback invoked for each reaped session_id,
                e.g. to clean up AppArmor profiles.

        Returns:
            The daemon thread (auto-stops on process exit).
        """
        self._reaper_stop.clear()

        def _reaper_loop():
            while not self._reaper_stop.wait(timeout=interval_seconds):
                try:
                    reaped = self.reap_expired(max_age_seconds)
                    if on_teardown:
                        for session_id in reaped:
                            try:
                                on_teardown(session_id)
                            except Exception:
                                logger.exception(
                                    "on_teardown callback failed for %s",
                                    session_id,
                                )
                except Exception:
                    logger.exception("Workspace reaper error")

        thread = threading.Thread(
            target=_reaper_loop,
            name="workspace-reaper",
            daemon=True,
        )
        thread.start()
        self._reaper_thread = thread
        logger.info(
            "Workspace reaper started (interval=%ds, max_age=%ds)",
            interval_seconds,
            max_age_seconds,
        )
        return thread

    def stop_reaper(self) -> None:
        """Stop the background reaper thread."""
        self._reaper_stop.set()
        if self._reaper_thread:
            self._reaper_thread.join(timeout=5)
            self._reaper_thread = None

    # ------------------------------------------------------------------
    # Template support
    # ------------------------------------------------------------------

    def list_templates(self) -> List[str]:
        """List available template names.

        Returns:
            Names of subdirectories under ``{workspace_root}/templates/``.
        """
        if not self._templates_dir.is_dir():
            return []
        return [
            d.name
            for d in self._templates_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_manifest(self) -> None:
        """Load workspace manifest from disk."""
        if not self._manifest_path.exists():
            return

        try:
            with open(self._manifest_path, "r") as f:
                data = json.load(f)

            for ws_data in data.get("workspaces", []):
                session_id = ws_data.get("session_id")
                if not session_id:
                    continue
                # Only load if the directory still exists
                ws_path = Path(ws_data.get("path", ""))
                if ws_path.is_dir():
                    self._workspaces[session_id] = ProvisionedWorkspace(
                        session_id=session_id,
                        path=str(ws_path),
                        template=ws_data.get("template"),
                        created_at=ws_data.get("created_at", ""),
                        last_activity=ws_data.get("last_activity", ""),
                        client_id=ws_data.get("client_id"),
                    )

            logger.debug(
                "Loaded %d provisioned workspaces from manifest",
                len(self._workspaces),
            )
        except Exception:
            logger.exception("Failed to load workspace manifest")

    def _save_manifest(self) -> None:
        """Save workspace manifest to disk.

        Must be called with ``_lock`` held.
        """
        try:
            data = {
                "workspaces": [ws.to_dict() for ws in self._workspaces.values()],
            }
            tmp_path = self._manifest_path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            tmp_path.replace(self._manifest_path)
        except Exception:
            logger.exception("Failed to save workspace manifest")

    def _copy_template(self, src: Path, dst: Path) -> None:
        """Copy template contents into a workspace directory.

        Copies files and directories from ``src`` into ``dst``, skipping
        any that already exist in the destination.

        Args:
            src: Source template directory.
            dst: Destination workspace directory.
        """
        for item in src.iterdir():
            target = dst / item.name
            if target.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
