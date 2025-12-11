"""Configuration models for subagent plugin."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    """
    project: str
    location: str
    default_model: str = "gemini-2.5-flash"
    profiles: Dict[str, SubagentProfile] = field(default_factory=dict)
    allow_inline: bool = True
    inline_allowed_plugins: List[str] = field(default_factory=list)

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
                    "inline_allowed_plugins": [...]
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
    """
    success: bool
    response: str
    turns_used: int = 0
    error: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None

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
        return result
