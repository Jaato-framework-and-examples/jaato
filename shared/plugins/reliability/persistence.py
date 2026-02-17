"""Persistence layer for reliability plugin.

Implements three-level persistence hierarchy:
- Session: In-memory state restored on reconnect (via IPC)
- Workspace: Project-specific data in .jaato/reliability.json
- User: Cross-workspace defaults in ~/.jaato/reliability.json
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .types import (
    FailureRecord,
    FailureSeverity,
    ToolReliabilityState,
    TrustState,
)

logger = logging.getLogger(__name__)

# Current persistence format version
PERSISTENCE_VERSION = 1


@dataclass
class SessionSettings:
    """Runtime settings for current session. Restored on reconnect."""

    # Nudge settings
    nudge_level: Optional[str] = None  # "gentle", "direct", "off", or None (inherit)
    nudge_enabled: bool = True

    # Recovery settings
    recovery_mode: Optional[str] = None  # "auto", "ask", or None (inherit)

    # Model switching
    model_switch_strategy: Optional[str] = None  # "suggest", "auto", "disabled", or None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nudge_level": self.nudge_level,
            "nudge_enabled": self.nudge_enabled,
            "recovery_mode": self.recovery_mode,
            "model_switch_strategy": self.model_switch_strategy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionSettings":
        return cls(
            nudge_level=data.get("nudge_level"),
            nudge_enabled=data.get("nudge_enabled", True),
            recovery_mode=data.get("recovery_mode"),
            model_switch_strategy=data.get("model_switch_strategy"),
        )


@dataclass
class SessionReliabilityState:
    """In-memory state for current session. Restored on reconnect via IPC."""

    session_id: str
    started_at: datetime = field(default_factory=datetime.now)

    # Current turn tracking
    current_turn_index: int = 0
    current_turn_tools: List[str] = field(default_factory=list)

    # Live escalation overrides (temporary, session-only)
    escalation_overrides: Dict[str, str] = field(default_factory=dict)  # key -> TrustState.value

    # Session-level settings (override workspace/user)
    settings: SessionSettings = field(default_factory=SessionSettings)

    def serialize(self) -> bytes:
        """Serialize for IPC persistence during disconnect."""
        data = {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "current_turn_index": self.current_turn_index,
            "current_turn_tools": self.current_turn_tools,
            "escalation_overrides": self.escalation_overrides,
            "settings": self.settings.to_dict(),
        }
        return json.dumps(data).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "SessionReliabilityState":
        """Restore after client reconnects to server."""
        obj = json.loads(data.decode("utf-8"))
        return cls(
            session_id=obj["session_id"],
            started_at=datetime.fromisoformat(obj["started_at"]),
            current_turn_index=obj.get("current_turn_index", 0),
            current_turn_tools=obj.get("current_turn_tools", []),
            escalation_overrides=obj.get("escalation_overrides", {}),
            settings=SessionSettings.from_dict(obj.get("settings", {})),
        )


@dataclass
class WorkspaceReliabilityData:
    """Workspace-level persistence in .jaato/reliability.json."""

    version: int = PERSISTENCE_VERSION

    # Tool reliability states (only non-trusted need persistence)
    tool_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Recent failure history for this project
    failure_history: List[Dict[str, Any]] = field(default_factory=list)

    # Workspace-specific settings/overrides
    settings: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    last_updated: Optional[str] = None
    created: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "tool_states": self.tool_states,
            "failure_history": self.failure_history,
            "settings": self.settings,
            "last_updated": datetime.now().isoformat(),
            "created": self.created or datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkspaceReliabilityData":
        return cls(
            version=data.get("version", PERSISTENCE_VERSION),
            tool_states=data.get("tool_states", {}),
            failure_history=data.get("failure_history", []),
            settings=data.get("settings", {}),
            last_updated=data.get("last_updated"),
            created=data.get("created"),
        )


@dataclass
class UserReliabilityData:
    """User-level persistence in ~/.jaato/reliability.json."""

    version: int = PERSISTENCE_VERSION

    # Default configuration
    default_settings: Dict[str, Any] = field(default_factory=dict)

    # Global tool patterns (tools that fail across workspaces)
    global_tool_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Timestamps
    last_updated: Optional[str] = None
    created: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "default_settings": self.default_settings,
            "global_tool_patterns": self.global_tool_patterns,
            "last_updated": datetime.now().isoformat(),
            "created": self.created or datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserReliabilityData":
        return cls(
            version=data.get("version", PERSISTENCE_VERSION),
            default_settings=data.get("default_settings", {}),
            global_tool_patterns=data.get("global_tool_patterns", {}),
            last_updated=data.get("last_updated"),
            created=data.get("created"),
        )


class ReliabilityPersistence:
    """Manages persistence across session, workspace, and user levels."""

    def __init__(self, workspace_path: Optional[str] = None):
        self._workspace_path = Path(workspace_path) if workspace_path else None
        self._user_path = Path.home() / ".jaato" / "reliability.json"

        # Cached data
        self._workspace_data: Optional[WorkspaceReliabilityData] = None
        self._user_data: Optional[UserReliabilityData] = None

    @property
    def workspace_file(self) -> Optional[Path]:
        """Get workspace persistence file path."""
        if self._workspace_path:
            return self._workspace_path / ".jaato" / "reliability.json"
        return None

    @property
    def user_file(self) -> Path:
        """Get user persistence file path."""
        return self._user_path

    # -------------------------------------------------------------------------
    # Workspace Persistence
    # -------------------------------------------------------------------------

    def load_workspace(self) -> WorkspaceReliabilityData:
        """Load workspace reliability data."""
        if self._workspace_data is not None:
            return self._workspace_data

        workspace_file = self.workspace_file
        if workspace_file and workspace_file.exists():
            try:
                data = json.loads(workspace_file.read_text(encoding="utf-8"))
                data = self._migrate_if_needed(data)
                self._workspace_data = WorkspaceReliabilityData.from_dict(data)
                logger.debug(f"Loaded workspace reliability data from {workspace_file}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load workspace reliability data: {e}")
                self._workspace_data = WorkspaceReliabilityData()
        else:
            self._workspace_data = WorkspaceReliabilityData()

        return self._workspace_data

    def save_workspace(self, data: Optional[WorkspaceReliabilityData] = None) -> bool:
        """Save workspace reliability data. Returns True on success."""
        workspace_file = self.workspace_file
        if not workspace_file:
            logger.debug("No workspace path set, skipping workspace save")
            return False

        if data is not None:
            self._workspace_data = data

        if self._workspace_data is None:
            return False

        try:
            # Ensure directory exists
            workspace_file.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write
            tmp_path = workspace_file.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(self._workspace_data.to_dict(), indent=2), encoding="utf-8")
            tmp_path.rename(workspace_file)

            logger.debug(f"Saved workspace reliability data to {workspace_file}")
            return True
        except (OSError, IOError) as e:
            logger.error(f"Failed to save workspace reliability data: {e}")
            return False

    # -------------------------------------------------------------------------
    # User Persistence
    # -------------------------------------------------------------------------

    def load_user(self) -> UserReliabilityData:
        """Load user reliability data."""
        if self._user_data is not None:
            return self._user_data

        if self._user_path.exists():
            try:
                data = json.loads(self._user_path.read_text(encoding="utf-8"))
                data = self._migrate_if_needed(data)
                self._user_data = UserReliabilityData.from_dict(data)
                logger.debug(f"Loaded user reliability data from {self._user_path}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load user reliability data: {e}")
                self._user_data = UserReliabilityData()
        else:
            self._user_data = UserReliabilityData()

        return self._user_data

    def save_user(self, data: Optional[UserReliabilityData] = None) -> bool:
        """Save user reliability data. Returns True on success."""
        if data is not None:
            self._user_data = data

        if self._user_data is None:
            return False

        try:
            # Ensure directory exists
            self._user_path.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write
            tmp_path = self._user_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(self._user_data.to_dict(), indent=2), encoding="utf-8")
            tmp_path.rename(self._user_path)

            logger.debug(f"Saved user reliability data to {self._user_path}")
            return True
        except (OSError, IOError) as e:
            logger.error(f"Failed to save user reliability data: {e}")
            return False

    # -------------------------------------------------------------------------
    # Tool State Persistence
    # -------------------------------------------------------------------------

    def save_tool_state(self, state: ToolReliabilityState) -> None:
        """Save a tool reliability state to workspace."""
        workspace = self.load_workspace()

        # Only persist non-trusted states
        if state.state == TrustState.TRUSTED:
            workspace.tool_states.pop(state.failure_key, None)
        else:
            workspace.tool_states[state.failure_key] = state.to_dict()

        self.save_workspace()

    def load_tool_states(self) -> Dict[str, ToolReliabilityState]:
        """Load all tool states from workspace."""
        workspace = self.load_workspace()
        states = {}

        for key, state_dict in workspace.tool_states.items():
            try:
                states[key] = ToolReliabilityState.from_dict(state_dict)
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to load tool state {key}: {e}")

        return states

    # -------------------------------------------------------------------------
    # Failure History Persistence
    # -------------------------------------------------------------------------

    def append_failure(self, record: FailureRecord, max_history: int = 1000) -> None:
        """Append a failure record to workspace history."""
        workspace = self.load_workspace()
        workspace.failure_history.append(record.to_dict())

        # Prune old entries
        if len(workspace.failure_history) > max_history:
            workspace.failure_history = workspace.failure_history[-max_history:]

        self.save_workspace()

    def load_failure_history(self) -> List[FailureRecord]:
        """Load failure history from workspace."""
        workspace = self.load_workspace()
        records = []

        for record_dict in workspace.failure_history:
            try:
                records.append(FailureRecord.from_dict(record_dict))
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to load failure record: {e}")

        return records

    # -------------------------------------------------------------------------
    # Settings Management
    # -------------------------------------------------------------------------

    def get_effective_setting(self, key: str, session_settings: Optional[SessionSettings] = None) -> Any:
        """Get effective setting value using merge hierarchy.

        Priority: session > workspace > user > default
        """
        # Check session override
        if session_settings:
            value = getattr(session_settings, key, None)
            if value is not None:
                return value

        # Check workspace override
        workspace = self.load_workspace()
        if key in workspace.settings:
            return workspace.settings[key]

        # Check user default
        user = self.load_user()
        if key in user.default_settings:
            return user.default_settings[key]

        return None

    def save_setting_to_workspace(self, key: str, value: Any) -> bool:
        """Save a setting to workspace level."""
        workspace = self.load_workspace()
        workspace.settings[key] = value
        return self.save_workspace()

    def save_setting_to_user(self, key: str, value: Any) -> bool:
        """Save a setting to user level."""
        user = self.load_user()
        user.default_settings[key] = value
        return self.save_user()

    def clear_workspace_setting(self, key: str) -> bool:
        """Clear a workspace-level setting (inherit from user)."""
        workspace = self.load_workspace()
        workspace.settings.pop(key, None)
        return self.save_workspace()

    # -------------------------------------------------------------------------
    # Migration
    # -------------------------------------------------------------------------

    def _migrate_if_needed(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate data to current version if needed."""
        version = data.get("version", 1)

        if version < PERSISTENCE_VERSION:
            # Future migrations go here
            data["version"] = PERSISTENCE_VERSION
            logger.info(f"Migrated reliability data from v{version} to v{PERSISTENCE_VERSION}")

        return data

    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        """Clear cached data, forcing reload on next access."""
        self._workspace_data = None
        self._user_data = None

    def set_workspace_path(self, path: Optional[str]) -> None:
        """Update workspace path and invalidate cache."""
        self._workspace_path = Path(path) if path else None
        self._workspace_data = None  # Invalidate workspace cache
