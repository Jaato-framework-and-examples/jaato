"""Configuration models for subagent plugin."""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SubagentProfile:
    """Configuration profile for a subagent.

    Defines what tools and capabilities a subagent has access to,
    allowing the parent model to delegate specialized tasks.

    Attributes:
        name: Unique identifier for this subagent profile.
        description: Human-readable description of what this subagent does.
        plugins: List of plugin names to enable for this subagent.
        plugin_configs: Per-plugin configuration overrides.
        system_instructions: Additional system instructions for the subagent.
        model: Optional model override (uses parent's model if not specified).
        max_turns: Maximum conversation turns before returning (default: 10).
        auto_approved: Whether this subagent can be spawned without permission.
        icon: Optional custom ASCII art icon (3 lines) for UI visualization.
        icon_name: Optional name of predefined icon (e.g., "code_assistant").
    """
    name: str
    description: str
    plugins: List[str] = field(default_factory=list)
    plugin_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    system_instructions: Optional[str] = None
    model: Optional[str] = None
    max_turns: int = 10
    auto_approved: bool = False
    icon: Optional[List[str]] = None
    icon_name: Optional[str] = None


def _parse_profile_file(file_path: Path) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Parse a single profile file (JSON or YAML).

    Args:
        file_path: Path to the profile file.

    Returns:
        Tuple of (profile_name, profile_data) or (None, None) on error.
    """
    try:
        content = file_path.read_text(encoding='utf-8')

        if file_path.suffix in ('.yaml', '.yml'):
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                logger.warning(
                    "PyYAML not installed, skipping YAML profile: %s",
                    file_path
                )
                return None, None
        elif file_path.suffix == '.json':
            data = json.loads(content)
        else:
            logger.debug("Skipping non-profile file: %s", file_path)
            return None, None

        if not isinstance(data, dict):
            logger.warning("Profile file must contain a dict: %s", file_path)
            return None, None

        # Profile name is either explicit 'name' field or derived from filename
        name = data.get('name') or file_path.stem

        return name, data

    except json.JSONDecodeError as e:
        logger.warning("Invalid JSON in profile file %s: %s", file_path, e)
        return None, None
    except Exception as e:
        logger.warning("Error reading profile file %s: %s", file_path, e)
        return None, None


def discover_profiles(
    profiles_dir: str,
    base_path: Optional[str] = None
) -> Dict[str, 'SubagentProfile']:
    """Discover subagent profiles from a directory.

    Scans the specified directory for .json and .yaml/.yml files,
    parsing each as a subagent profile definition.

    Args:
        profiles_dir: Directory path to scan (relative or absolute).
        base_path: Base path for resolving relative profiles_dir.
                   Defaults to current working directory.

    Returns:
        Dict mapping profile names to SubagentProfile instances.
    """
    if base_path is None:
        base_path = os.getcwd()

    # Resolve the profiles directory path
    profiles_path = Path(profiles_dir)
    if not profiles_path.is_absolute():
        profiles_path = Path(base_path) / profiles_path

    if not profiles_path.exists():
        logger.debug("Profiles directory does not exist: %s", profiles_path)
        return {}

    if not profiles_path.is_dir():
        logger.warning("Profiles path is not a directory: %s", profiles_path)
        return {}

    profiles: Dict[str, SubagentProfile] = {}

    # Scan for profile files
    for file_path in profiles_path.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix not in ('.json', '.yaml', '.yml'):
            continue

        name, data = _parse_profile_file(file_path)
        if name is None or data is None:
            continue

        # Create SubagentProfile from parsed data
        profile = SubagentProfile(
            name=name,
            description=data.get('description', ''),
            plugins=data.get('plugins', []),
            plugin_configs=data.get('plugin_configs', {}),
            system_instructions=data.get('system_instructions'),
            model=data.get('model'),
            max_turns=data.get('max_turns', 10),
            auto_approved=data.get('auto_approved', False),
            icon=data.get('icon'),
            icon_name=data.get('icon_name'),
        )

        profiles[name] = profile
        logger.debug("Discovered profile '%s' from %s", name, file_path)

    if profiles:
        logger.info(
            "Discovered %d profile(s) from %s: %s",
            len(profiles),
            profiles_path,
            ", ".join(profiles.keys())
        )

    return profiles


@dataclass
class SubagentConfig:
    """Top-level configuration for the subagent plugin.

    Attributes:
        project: GCP project ID for Vertex AI.
        location: Vertex AI region (e.g., 'us-central1').
        default_model: Default model for subagents.
        profiles: Dict of named subagent profiles.
        allow_inline: Whether to allow inline subagent creation.
        inline_allowed_plugins: Plugins allowed for inline subagent creation.
        auto_discover_profiles: Whether to auto-discover profiles from profiles_dir.
        profiles_dir: Directory to scan for profile files (default: .jaato/profiles).
    """
    project: str
    location: str
    default_model: str = "gemini-2.5-flash"
    profiles: Dict[str, SubagentProfile] = field(default_factory=dict)
    allow_inline: bool = True
    inline_allowed_plugins: List[str] = field(default_factory=list)
    auto_discover_profiles: bool = True
    profiles_dir: str = ".jaato/profiles"

    def add_profile(self, profile: SubagentProfile) -> None:
        """Add a subagent profile."""
        self.profiles[profile.name] = profile

    def get_profile(self, name: str) -> Optional[SubagentProfile]:
        """Get a subagent profile by name."""
        return self.profiles.get(name)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubagentConfig':
        """Create SubagentConfig from a dictionary.

        Args:
            data: Configuration dictionary with structure:
                {
                    "project": "...",
                    "location": "...",
                    "default_model": "...",
                    "profiles": {
                        "profile_name": {
                            "description": "...",
                            "plugins": [...],
                            ...
                        }
                    },
                    "allow_inline": true,
                    "inline_allowed_plugins": [...],
                    "auto_discover_profiles": true,
                    "profiles_dir": ".jaato/profiles"
                }

        Returns:
            SubagentConfig instance.
        """
        profiles = {}
        for name, profile_data in data.get('profiles', {}).items():
            profiles[name] = SubagentProfile(
                name=name,
                description=profile_data.get('description', ''),
                plugins=profile_data.get('plugins', []),
                plugin_configs=profile_data.get('plugin_configs', {}),
                system_instructions=profile_data.get('system_instructions'),
                model=profile_data.get('model'),
                max_turns=profile_data.get('max_turns', 10),
                auto_approved=profile_data.get('auto_approved', False),
                icon=profile_data.get('icon'),
                icon_name=profile_data.get('icon_name'),
            )

        return cls(
            project=data.get('project', ''),
            location=data.get('location', ''),
            default_model=data.get('default_model', 'gemini-2.5-flash'),
            profiles=profiles,
            allow_inline=data.get('allow_inline', True),
            inline_allowed_plugins=data.get('inline_allowed_plugins', []),
            auto_discover_profiles=data.get('auto_discover_profiles', True),
            profiles_dir=data.get('profiles_dir', '.jaato/profiles'),
        )


@dataclass
class SubagentResult:
    """Result from a subagent execution.

    Attributes:
        success: Whether the subagent completed successfully.
        response: The subagent's final response text.
        turns_used: Number of conversation turns used.
        error: Error message if success is False.
        token_usage: Token usage statistics if available.
        agent_id: ID of the subagent session (for multi-turn conversations).
    """
    success: bool
    response: str
    turns_used: int = 0
    error: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None
    agent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for tool response."""
        result = {
            'success': self.success,
            'response': self.response,
            'turns_used': self.turns_used,
        }
        if self.error:
            result['error'] = self.error
        if self.token_usage:
            result['token_usage'] = self.token_usage
        if self.agent_id:
            result['agent_id'] = self.agent_id
        return result
