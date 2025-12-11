"""ASCII art icons for agent visualization.

This module provides default ASCII art icons for different agent types
and a registry for custom icons defined in profiles.
"""

from typing import Dict, List, Optional


# Default icons for agent types (3 lines each, medium detail)
DEFAULT_ICONS: Dict[str, List[str]] = {
    # Main agent - robot/AI icon
    "main": [
        "  ╭─┐  ",
        "  │█│  ",
        "  └┬┘  "
    ],

    # Default subagent - gear icon
    "default_subagent": [
        "  ⚙ ⚙  ",
        "   ▀▄▀  ",
        "   ║║   "
    ],

    # Code assistant - code brackets
    "code_assistant": [
        " </>  ",
        "  ▼   ",
        " ╚═╝  "
    ],

    # Research agent - magnifying glass
    "research": [
        " [🔍] ",
        "  ║║║  ",
        "  ╚╩╝  "
    ],

    # File editor
    "file_editor": [
        " ┌─┐  ",
        " │≡│  ",
        " └─┘  "
    ],

    # Data analyzer
    "data_analyst": [
        " ▄▄▄  ",
        " ║█║  ",
        " ╚═╝  "
    ],

    # Test runner
    "test_runner": [
        " ▶║   ",
        " ▶║   ",
        " ▶║   "
    ],

    # Web scraper
    "web_scraper": [
        " ╔╦╗  ",
        " ╠╬╣  ",
        " ╚╩╝  "
    ],

    # Generic task agent
    "task_agent": [
        " ┌▶┐  ",
        " │░│  ",
        " └─┘  "
    ],
}


class AgentIconRegistry:
    """Registry for agent icons with fallback to defaults."""

    def __init__(self):
        """Initialize the icon registry."""
        self._custom_icons: Dict[str, List[str]] = {}

    def register_icon(self, name: str, icon_lines: List[str]) -> None:
        """Register a custom icon.

        Args:
            name: Icon name (typically profile name).
            icon_lines: List of 3 strings representing the icon.

        Raises:
            ValueError: If icon_lines doesn't have exactly 3 lines.
        """
        if len(icon_lines) != 3:
            raise ValueError(f"Icon must have exactly 3 lines, got {len(icon_lines)}")

        self._custom_icons[name] = icon_lines

    def get_icon(
        self,
        agent_type: str,
        profile_name: Optional[str] = None,
        custom_icon: Optional[List[str]] = None
    ) -> List[str]:
        """Get icon for an agent.

        Priority order:
        1. custom_icon (if provided)
        2. Profile-specific custom icon (if registered)
        3. Profile name default icon (if exists in DEFAULT_ICONS)
        4. Agent type default icon
        5. Fallback to default_subagent

        Args:
            agent_type: "main" or "subagent".
            profile_name: Profile name (e.g., "code_assistant"), if subagent.
            custom_icon: Explicit custom icon (3 lines).

        Returns:
            List of 3 strings representing the icon.
        """
        # 1. Explicit custom icon
        if custom_icon:
            if len(custom_icon) == 3:
                return custom_icon
            # Invalid custom icon - continue to fallback

        # 2. Profile-specific custom icon (registered)
        if profile_name and profile_name in self._custom_icons:
            return self._custom_icons[profile_name]

        # 3. Profile name default icon
        if profile_name and profile_name in DEFAULT_ICONS:
            return DEFAULT_ICONS[profile_name]

        # 4. Agent type default icon
        if agent_type in DEFAULT_ICONS:
            return DEFAULT_ICONS[agent_type]

        # 5. Fallback
        return DEFAULT_ICONS["default_subagent"]

    def load_from_config(self, config: Dict[str, List[str]]) -> None:
        """Load custom icons from configuration.

        Args:
            config: Dict mapping icon names to icon_lines.

        Example config:
            {
                "my_custom_agent": [
                    " ┌─┐  ",
                    " │*│  ",
                    " └─┘  "
                ]
            }
        """
        for name, icon_lines in config.items():
            try:
                self.register_icon(name, icon_lines)
            except ValueError as e:
                # Log warning but don't fail - just skip invalid icons
                import logging
                logging.warning(f"Invalid icon for '{name}': {e}")


# Global registry instance
_global_registry = AgentIconRegistry()


def get_icon(
    agent_type: str,
    profile_name: Optional[str] = None,
    custom_icon: Optional[List[str]] = None
) -> List[str]:
    """Convenience function to get icon from global registry.

    Args:
        agent_type: "main" or "subagent".
        profile_name: Profile name, if subagent.
        custom_icon: Explicit custom icon (3 lines).

    Returns:
        List of 3 strings representing the icon.
    """
    return _global_registry.get_icon(agent_type, profile_name, custom_icon)


def register_icon(name: str, icon_lines: List[str]) -> None:
    """Register a custom icon in global registry.

    Args:
        name: Icon name.
        icon_lines: List of 3 strings.
    """
    _global_registry.register_icon(name, icon_lines)


def load_icons_from_config(config: Dict[str, List[str]]) -> None:
    """Load custom icons from configuration into global registry.

    Args:
        config: Dict mapping icon names to icon_lines.
    """
    _global_registry.load_from_config(config)
