"""Workspace Manager for Jaato Server.

Handles workspace discovery, creation, and configuration management
for web clients that need to select a workspace before starting a session.

Workspaces are directories under a configurable root that contain either:
- A .jaato/ directory
- A .env file

The manager persists workspace metadata to ~/.jaato/workspaces.json
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import dotenv_values

logger = logging.getLogger(__name__)

# Default registry path
DEFAULT_REGISTRY_PATH = Path.home() / ".jaato" / "workspaces.json"

# Known providers and their required env vars
PROVIDER_ENV_VARS = {
    "anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
    "google": ["PROJECT_ID", "GOOGLE_APPLICATION_CREDENTIALS"],
    "github": ["GITHUB_TOKEN"],
    "antigravity": [],  # Uses OAuth, checked differently
    "ollama": ["OLLAMA_HOST", "OLLAMA_MODEL"],
    "claude_cli": [],  # Uses CLI auth
}

# Provider detection order (check these env vars to determine provider)
PROVIDER_DETECTION = [
    ("JAATO_PROVIDER", None),  # Explicit override
    ("ANTHROPIC_API_KEY", "anthropic"),
    ("ANTHROPIC_AUTH_TOKEN", "anthropic"),
    ("GITHUB_TOKEN", "github"),
    ("PROJECT_ID", "google"),
    ("OLLAMA_HOST", "ollama"),
]


@dataclass
class WorkspaceInfo:
    """Information about a workspace."""
    name: str  # Relative path from root
    path: str  # Absolute path
    configured: bool  # Has valid provider config
    provider: Optional[str] = None
    model: Optional[str] = None
    last_accessed: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkspaceManager:
    """Manages workspace discovery, creation, and configuration."""

    def __init__(
        self,
        workspace_root: str,
        registry_path: Optional[Path] = None,
    ):
        """Initialize the workspace manager.

        Args:
            workspace_root: Root directory containing workspaces.
            registry_path: Path to workspace registry file (default: ~/.jaato/workspaces.json).
        """
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.registry_path = registry_path or DEFAULT_REGISTRY_PATH

        # Ensure root exists
        if not self.workspace_root.exists():
            logger.warning(f"Workspace root does not exist: {self.workspace_root}")

        # In-memory cache
        self._workspaces: Dict[str, WorkspaceInfo] = {}

        # Currently selected workspace
        self._selected_workspace: Optional[str] = None

        # Load registry
        self._load_registry()

    def _load_registry(self) -> None:
        """Load workspace registry from disk."""
        if not self.registry_path.exists():
            logger.debug(f"No workspace registry at {self.registry_path}")
            return

        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)

            for ws_data in data.get("workspaces", []):
                name = ws_data.get("name")
                if name:
                    self._workspaces[name] = WorkspaceInfo(
                        name=name,
                        path=ws_data.get("path", ""),
                        configured=ws_data.get("configured", False),
                        provider=ws_data.get("provider"),
                        model=ws_data.get("model"),
                        last_accessed=ws_data.get("last_accessed"),
                    )

            logger.debug(f"Loaded {len(self._workspaces)} workspaces from registry")

        except Exception as e:
            logger.warning(f"Failed to load workspace registry: {e}")

    def _save_registry(self) -> None:
        """Save workspace registry to disk."""
        try:
            # Ensure directory exists
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "root": str(self.workspace_root),
                "workspaces": [ws.to_dict() for ws in self._workspaces.values()],
            }

            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self._workspaces)} workspaces to registry")

        except Exception as e:
            logger.warning(f"Failed to save workspace registry: {e}")

    def discover_workspaces(self) -> List[WorkspaceInfo]:
        """Discover workspaces under the root directory.

        Scans for directories containing .jaato/ or .env files.
        Updates internal cache and persists to registry.

        Returns:
            List of discovered workspaces.
        """
        if not self.workspace_root.exists():
            logger.warning(f"Workspace root does not exist: {self.workspace_root}")
            return []

        discovered = []

        for entry in self.workspace_root.iterdir():
            if not entry.is_dir():
                continue

            # Skip hidden directories (except we look inside for .jaato)
            if entry.name.startswith("."):
                continue

            # Check if this looks like a workspace
            has_jaato = (entry / ".jaato").is_dir()
            has_env = (entry / ".env").is_file()

            if has_jaato or has_env:
                ws_info = self._analyze_workspace(entry)
                discovered.append(ws_info)
                self._workspaces[ws_info.name] = ws_info

        # Save updated registry
        self._save_registry()

        logger.info(f"Discovered {len(discovered)} workspaces under {self.workspace_root}")
        return discovered

    def _analyze_workspace(self, path: Path) -> WorkspaceInfo:
        """Analyze a workspace directory to determine its configuration status.

        Args:
            path: Absolute path to workspace directory.

        Returns:
            WorkspaceInfo with configuration details.
        """
        name = path.name
        env_file = path / ".env"

        provider = None
        model = None
        configured = False

        if env_file.exists():
            env_vars = dotenv_values(env_file)

            # Detect provider
            provider = self._detect_provider(env_vars)

            # Get model if set
            model = env_vars.get("MODEL_NAME") or env_vars.get("JAATO_MODEL")

            # Consider configured if we have a provider
            configured = provider is not None

        # Check existing entry for last_accessed
        existing = self._workspaces.get(name)
        last_accessed = existing.last_accessed if existing else None

        return WorkspaceInfo(
            name=name,
            path=str(path),
            configured=configured,
            provider=provider,
            model=model,
            last_accessed=last_accessed,
        )

    def _detect_provider(self, env_vars: Dict[str, Optional[str]]) -> Optional[str]:
        """Detect the provider from environment variables.

        Args:
            env_vars: Dictionary of environment variables.

        Returns:
            Provider name or None if not detected.
        """
        for env_var, provider in PROVIDER_DETECTION:
            if env_var in env_vars and env_vars[env_var]:
                if provider is None:
                    # JAATO_PROVIDER is explicit
                    return env_vars[env_var]
                return provider

        return None

    def list_workspaces(self) -> List[WorkspaceInfo]:
        """List all known workspaces.

        Combines cached workspaces with fresh discovery.

        Returns:
            List of workspace info.
        """
        # Re-discover to get fresh state
        self.discover_workspaces()
        return list(self._workspaces.values())

    def create_workspace(self, name: str) -> WorkspaceInfo:
        """Create a new workspace.

        Args:
            name: Name for the new workspace (becomes subdirectory name).

        Returns:
            WorkspaceInfo for the created workspace.

        Raises:
            ValueError: If workspace already exists or name is invalid.
        """
        # Validate name
        if not name or "/" in name or "\\" in name:
            raise ValueError(f"Invalid workspace name: {name}")

        path = self.workspace_root / name

        if path.exists():
            raise ValueError(f"Workspace already exists: {name}")

        # Create directory and .jaato subdirectory
        path.mkdir(parents=True)
        (path / ".jaato").mkdir()

        # Create empty .env file
        (path / ".env").touch()

        ws_info = WorkspaceInfo(
            name=name,
            path=str(path),
            configured=False,
            last_accessed=datetime.utcnow().isoformat(),
        )

        self._workspaces[name] = ws_info
        self._save_registry()

        logger.info(f"Created workspace: {name} at {path}")
        return ws_info

    def select_workspace(self, name: str) -> WorkspaceInfo:
        """Select a workspace for the session.

        Args:
            name: Workspace name (relative path from root).

        Returns:
            WorkspaceInfo with current configuration status.

        Raises:
            ValueError: If workspace does not exist.
        """
        path = self.workspace_root / name

        if not path.exists():
            raise ValueError(f"Workspace does not exist: {name}")

        # Re-analyze to get fresh state
        ws_info = self._analyze_workspace(path)
        ws_info.last_accessed = datetime.utcnow().isoformat()

        self._workspaces[name] = ws_info
        self._selected_workspace = name
        self._save_registry()

        logger.info(f"Selected workspace: {name}")
        return ws_info

    def get_selected_workspace(self) -> Optional[WorkspaceInfo]:
        """Get the currently selected workspace.

        Returns:
            WorkspaceInfo or None if no workspace selected.
        """
        if not self._selected_workspace:
            return None
        return self._workspaces.get(self._selected_workspace)

    def get_workspace_path(self, name: Optional[str] = None) -> Optional[Path]:
        """Get the absolute path to a workspace.

        Args:
            name: Workspace name, or None for selected workspace.

        Returns:
            Absolute path or None.
        """
        target = name or self._selected_workspace
        if not target:
            return None

        ws_info = self._workspaces.get(target)
        if ws_info:
            return Path(ws_info.path)

        # Fallback to computed path
        return self.workspace_root / target

    def get_env_file(self, name: Optional[str] = None) -> Optional[Path]:
        """Get the .env file path for a workspace.

        Args:
            name: Workspace name, or None for selected workspace.

        Returns:
            Path to .env file or None.
        """
        ws_path = self.get_workspace_path(name)
        if ws_path:
            return ws_path / ".env"
        return None

    def get_config_status(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration status for a workspace.

        Args:
            name: Workspace name, or None for selected workspace.

        Returns:
            Dictionary with configuration status.
        """
        target = name or self._selected_workspace
        if not target:
            return {
                "workspace": None,
                "configured": False,
                "available_providers": list(PROVIDER_ENV_VARS.keys()),
                "missing_fields": ["workspace"],
            }

        ws_info = self._workspaces.get(target)
        if not ws_info:
            ws_path = self.workspace_root / target
            if ws_path.exists():
                ws_info = self._analyze_workspace(ws_path)
            else:
                return {
                    "workspace": target,
                    "configured": False,
                    "available_providers": list(PROVIDER_ENV_VARS.keys()),
                    "missing_fields": ["workspace does not exist"],
                }

        missing = []
        if not ws_info.configured:
            missing.append("provider")
        if not ws_info.model:
            missing.append("model")

        return {
            "workspace": target,
            "configured": ws_info.configured,
            "provider": ws_info.provider,
            "model": ws_info.model,
            "available_providers": list(PROVIDER_ENV_VARS.keys()),
            "missing_fields": missing,
        }

    def update_config(
        self,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update workspace configuration.

        Writes provider and optional model/API key to the workspace's .env file.

        Args:
            provider: Provider name.
            model: Model name (optional).
            api_key: API key (optional, for non-OAuth providers).
            name: Workspace name, or None for selected workspace.

        Returns:
            Dictionary with update result.

        Raises:
            ValueError: If no workspace selected or invalid provider.
        """
        target = name or self._selected_workspace
        if not target:
            raise ValueError("No workspace selected")

        env_file = self.get_env_file(target)
        if not env_file:
            raise ValueError(f"Cannot find .env for workspace: {target}")

        # Read existing env vars
        existing = {}
        if env_file.exists():
            existing = dict(dotenv_values(env_file))

        # Update with new values
        existing["JAATO_PROVIDER"] = provider

        if model:
            existing["MODEL_NAME"] = model

        if api_key:
            # Determine which env var to use based on provider
            if provider == "anthropic":
                existing["ANTHROPIC_API_KEY"] = api_key
            elif provider == "github":
                existing["GITHUB_TOKEN"] = api_key
            elif provider == "google":
                # For Google, API key isn't typically used, but store it anyway
                existing["GOOGLE_API_KEY"] = api_key

        # Write back to .env file
        self._write_env_file(env_file, existing)

        # Re-analyze and update cache
        ws_path = self.workspace_root / target
        ws_info = self._analyze_workspace(ws_path)
        ws_info.last_accessed = datetime.utcnow().isoformat()
        self._workspaces[target] = ws_info
        self._save_registry()

        logger.info(f"Updated config for workspace {target}: provider={provider}, model={model}")

        return {
            "workspace": target,
            "provider": provider,
            "model": model,
            "success": True,
        }

    def _write_env_file(self, path: Path, env_vars: Dict[str, Optional[str]]) -> None:
        """Write environment variables to a .env file.

        Args:
            path: Path to .env file.
            env_vars: Dictionary of environment variables.
        """
        lines = []
        for key, value in env_vars.items():
            if value is not None:
                # Quote values that contain spaces or special chars
                if " " in value or "=" in value or '"' in value:
                    value = f'"{value}"'
                lines.append(f"{key}={value}")

        path.write_text("\n".join(lines) + "\n")
