"""Theme configuration for the rich client.

Allows users to customize colors via:
1. JSON config file: .jaato/theme.json (project-level)
                     ~/.jaato/theme.json (user-level fallback)
2. Environment variable: JAATO_THEME=dark|light|high-contrast (preset selection)

Themes use a two-tier color system:
1. Base palette: 11 core colors (primary, secondary, success, etc.)
2. Semantic styles: UI elements mapped to palette colors + modifiers

Both prompt_toolkit (for TUI chrome) and Rich (for output formatting)
styles derive from the same theme configuration.

Built-in themes are stored as JSON files in the themes/ directory alongside
this module. Users can inspect and override them by placing custom theme
files in:
- ~/.jaato/themes/<name>.json (user-level override)
- .jaato/themes/<name>.json (project-level override)

Theme file discovery order (first found wins):
1. JAATO_THEME environment variable (preset name)
2. Saved user preference (~/.jaato/preferences.json)
3. Project custom theme (.jaato/theme.json)
4. User custom theme (~/.jaato/theme.json)
5. User override of built-in theme (~/.jaato/themes/<name>.json)
6. Project override of built-in theme (.jaato/themes/<name>.json)
7. Built-in theme from package (rich-client/themes/<name>.json)
8. Hardcoded fallback (if all else fails)
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prompt_toolkit.styles import Style

logger = logging.getLogger(__name__)

# Regex for validating hex colors
HEX_COLOR_PATTERN = re.compile(r'^#[0-9A-Fa-f]{6}$')


def is_hex_color(value: str) -> bool:
    """Check if a string is a valid hex color."""
    return bool(HEX_COLOR_PATTERN.match(value))


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple.

    Args:
        hex_color: Hex color string like "#FF5500".

    Returns:
        Tuple of (R, G, B) integers 0-255.
    """
    hex_color = hex_color.lstrip('#')
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


@dataclass
class StyleSpec:
    """Specification for a single style.

    Combines foreground/background colors with text modifiers.
    Colors can be hex values (#RRGGBB) or palette references (e.g., "primary").
    """
    fg: Optional[str] = None  # Foreground color (hex or palette name)
    bg: Optional[str] = None  # Background color (hex or palette name)
    bold: bool = False
    italic: bool = False
    dim: bool = False
    underline: bool = False

    def to_prompt_toolkit(self, palette: Dict[str, str]) -> str:
        """Convert to prompt_toolkit style string.

        Args:
            palette: Dict mapping color names to hex values.

        Returns:
            Style string like "bg:#333333 #ffffff bold"
        """
        parts = []

        # Background color
        if self.bg:
            color = palette.get(self.bg, self.bg) if not is_hex_color(self.bg) else self.bg
            parts.append(f"bg:{color}")

        # Foreground color
        if self.fg:
            color = palette.get(self.fg, self.fg) if not is_hex_color(self.fg) else self.fg
            parts.append(color)

        # Modifiers
        if self.bold:
            parts.append("bold")
        if self.italic:
            parts.append("italic")
        if self.dim:
            parts.append("dim")  # prompt_toolkit doesn't have dim, but we include it
        if self.underline:
            parts.append("underline")

        return " ".join(parts)

    def to_rich(self, palette: Dict[str, str]) -> str:
        """Convert to Rich style string.

        Args:
            palette: Dict mapping color names to hex values.

        Returns:
            Style string like "bold #ffffff on #333333"
        """
        parts = []

        # Modifiers first in Rich
        if self.bold:
            parts.append("bold")
        if self.italic:
            parts.append("italic")
        if self.dim:
            parts.append("dim")
        if self.underline:
            parts.append("underline")

        # Foreground color
        if self.fg:
            color = palette.get(self.fg, self.fg) if not is_hex_color(self.fg) else self.fg
            parts.append(color)

        # Background color (Rich uses "on #color" syntax)
        if self.bg:
            color = palette.get(self.bg, self.bg) if not is_hex_color(self.bg) else self.bg
            parts.append(f"on {color}")

        return " ".join(parts)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StyleSpec":
        """Create StyleSpec from a dictionary."""
        return cls(
            fg=data.get("fg"),
            bg=data.get("bg"),
            bold=data.get("bold", False),
            italic=data.get("italic", False),
            dim=data.get("dim", False),
            underline=data.get("underline", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary (only non-default values)."""
        result = {}
        if self.fg:
            result["fg"] = self.fg
        if self.bg:
            result["bg"] = self.bg
        if self.bold:
            result["bold"] = True
        if self.italic:
            result["italic"] = True
        if self.dim:
            result["dim"] = True
        if self.underline:
            result["underline"] = True
        return result


def _discover_builtin_themes() -> List[str]:
    """Discover built-in themes by scanning the themes/ directory.

    Returns:
        List of theme names (without .json extension).
    """
    themes_dir = _get_builtin_themes_dir()
    if not themes_dir.exists():
        return ["dark"]  # Minimum fallback

    return sorted([
        path.stem for path in themes_dir.glob("*.json")
    ])


def get_builtin_theme_names() -> List[str]:
    """Get list of built-in theme names.

    Returns:
        List of available built-in theme names.
    """
    return _discover_builtin_themes()

# Fallback palette colors (used when JSON files are unavailable)
# These are kept as a safety net and for backwards compatibility.
_FALLBACK_PALETTE_DARK = {
    "primary": "#5fd7ff",      # Cyan - main accent
    "secondary": "#87d787",    # Light green - secondary accent
    "accent": "#d7af87",       # Tan/orange - tertiary accent
    "success": "#5fd75f",      # Green - success states
    "warning": "#ffff5f",      # Yellow - warnings
    "error": "#ff5f5f",        # Red - errors
    "muted": "#808080",        # Gray - deemphasized
    "background": "#1a1a1a",   # Dark background
    "surface": "#333333",      # Raised surfaces
    "text": "#ffffff",         # Primary text
    "text_muted": "#aaaaaa",   # Secondary text
}

# Backwards compatibility alias
DEFAULT_PALETTE = _FALLBACK_PALETTE_DARK

_FALLBACK_PALETTE_LIGHT = {
    "primary": "#0077cc",
    "secondary": "#006600",
    "accent": "#996600",
    "success": "#006600",
    "warning": "#996600",
    "error": "#cc0000",
    "muted": "#666666",
    "background": "#f5f5f5",
    "surface": "#ffffff",
    "text": "#000000",
    "text_muted": "#666666",
}

# Backwards compatibility alias
LIGHT_PALETTE = _FALLBACK_PALETTE_LIGHT

_FALLBACK_PALETTE_HIGH_CONTRAST = {
    "primary": "#00ffff",
    "secondary": "#00ff00",
    "accent": "#ffaa00",
    "success": "#00ff00",
    "warning": "#ffff00",
    "error": "#ff0000",
    "muted": "#aaaaaa",
    "background": "#000000",
    "surface": "#1a1a1a",
    "text": "#ffffff",
    "text_muted": "#cccccc",
}

# Backwards compatibility alias
HIGH_CONTRAST_PALETTE = _FALLBACK_PALETTE_HIGH_CONTRAST

# Default semantic style mappings
DEFAULT_SEMANTIC_STYLES = {
    # Agent tab bar
    "agent_tab_selected": StyleSpec(fg="primary", bold=True, underline=True),
    "agent_tab_dim": StyleSpec(fg="muted"),
    "agent_tab_separator": StyleSpec(fg="#404040"),
    "agent_tab_hint": StyleSpec(fg="#606060", italic=True),
    "agent_tab_scroll": StyleSpec(fg="primary", bold=True),

    # Agent status symbols
    "agent_processing": StyleSpec(fg="#5f87ff"),  # Blue
    "agent_awaiting": StyleSpec(fg="primary"),
    "agent_finished": StyleSpec(fg="muted"),
    "agent_error": StyleSpec(fg="error"),
    "agent_permission": StyleSpec(fg="warning"),

    # Agent popup
    "agent_popup_border": StyleSpec(fg="primary"),
    "agent_popup_icon": StyleSpec(fg="primary"),
    "agent_popup_name": StyleSpec(fg="text", bold=True),

    # Session bar
    "session_bar_bg": StyleSpec(bg="background"),
    "session_bar_label": StyleSpec(fg="muted"),
    "session_bar_id": StyleSpec(fg="primary"),
    "session_bar_separator": StyleSpec(fg="#404040"),
    "session_bar_description": StyleSpec(fg="secondary"),
    "session_bar_workspace": StyleSpec(fg="accent"),

    # Status bar
    "status_bar_bg": StyleSpec(bg="surface"),
    "status_bar_label": StyleSpec(fg="muted", bg="surface"),
    "status_bar_value": StyleSpec(fg="text", bg="surface", bold=True),
    "status_bar_separator": StyleSpec(fg="#555555", bg="surface"),
    "status_bar_warning": StyleSpec(fg="warning"),

    # Output panel
    "output_panel_bg": StyleSpec(bg="background"),

    # Plan panel - step statuses (bg="surface" to match status bar)
    "plan_pending": StyleSpec(fg="muted", bg="surface"),
    "plan_in_progress": StyleSpec(fg="#5f87ff", bg="surface"),  # Blue
    "plan_completed": StyleSpec(fg="success", bg="surface"),
    "plan_failed": StyleSpec(fg="error", bg="surface"),
    "plan_skipped": StyleSpec(fg="warning", bg="surface"),
    # Plan panel - overall plan statuses (for completeness)
    "plan_active": StyleSpec(fg="primary", bg="surface"),
    "plan_cancelled": StyleSpec(fg="warning", bg="surface", dim=True),
    # Plan popup - additional styles
    "plan_popup_border": StyleSpec(fg="primary"),
    "plan_popup_background": StyleSpec(bg="background"),
    "plan_result": StyleSpec(fg="success", dim=True),
    "plan_error_text": StyleSpec(fg="error", dim=True),
    "plan_popup_empty": StyleSpec(fg="muted", dim=True, italic=True),
    "plan_popup_empty_border": StyleSpec(fg="muted", dim=True),
    "plan_popup_scroll_indicator": StyleSpec(fg="muted", dim=True, italic=True),
    "plan_popup_hint": StyleSpec(fg="muted", dim=True),
    "plan_popup_step_number": StyleSpec(fg="muted", dim=True),
    "plan_popup_step_description": StyleSpec(fg="text"),
    "plan_popup_step_active": StyleSpec(fg="text", bold=True),
    "plan_popup_result_prefix": StyleSpec(fg="muted", dim=True),
    "plan_popup_error_prefix": StyleSpec(fg="muted", dim=True),
    "plan_popup_separator": StyleSpec(fg="muted", dim=True),
    "plan_popup_progress": StyleSpec(fg="text", bold=True),
    "plan_popup_progress_detail": StyleSpec(fg="muted", dim=True),

    # Output - headers
    "user_header": StyleSpec(fg="success", bold=True),
    "user_header_separator": StyleSpec(fg="success", dim=True),
    "model_header": StyleSpec(fg="primary", bold=True),
    "model_header_separator": StyleSpec(fg="primary", dim=True),

    # Output - thinking (extended reasoning)
    "thinking_header": StyleSpec(fg="primary", dim=True),
    "thinking_header_separator": StyleSpec(dim=True),
    "thinking_border": StyleSpec(fg="primary", dim=True),
    "thinking_content": StyleSpec(dim=True, italic=True),
    "thinking_footer": StyleSpec(fg="primary", dim=True),
    "thinking_footer_separator": StyleSpec(dim=True),

    # Output - tool display
    "tool_output": StyleSpec(fg="#87D7D7", italic=True),  # Pale cyan
    "tool_source_label": StyleSpec(fg="#808080", dim=True),  # Dim magenta replaced with muted
    "tool_name": StyleSpec(fg="primary"),  # Tool names in output
    "tool_border": StyleSpec(dim=True),  # Tool output box borders
    "tool_success": StyleSpec(fg="success"),
    "tool_error": StyleSpec(fg="error", dim=True),
    "tool_pending": StyleSpec(fg="warning", bold=True),
    "tool_duration": StyleSpec(dim=True),
    "tool_selected": StyleSpec(fg="text"),  # Reversed in actual rendering
    "tool_unselected": StyleSpec(dim=True),
    "tool_indicator": StyleSpec(fg="primary", dim=True),  # Spinner/progress indicator

    # Output - misc
    "permission_prompt": StyleSpec(fg="warning", bold=True),
    "permission_text": StyleSpec(fg="primary"),
    "permission_denied": StyleSpec(fg="error", dim=True),
    "clarification_label": StyleSpec(fg="primary"),
    "clarification_icon": StyleSpec(fg="primary"),
    "clarification_required": StyleSpec(fg="warning"),
    "clarification_answer": StyleSpec(fg="success", dim=True),
    "clarification_question": StyleSpec(dim=True),

    # System messages
    "system_info": StyleSpec(dim=True),  # Default system message
    "system_highlight": StyleSpec(fg="primary"),  # Informational highlights
    "system_error": StyleSpec(fg="error"),  # Error messages
    "system_error_bold": StyleSpec(fg="error", bold=True),  # Emphasized errors
    "system_warning": StyleSpec(fg="warning"),  # Warnings/interrupts
    "system_success": StyleSpec(fg="success"),  # Success messages
    "system_emphasis": StyleSpec(bold=True),  # Emphasized messages
    "system_version": StyleSpec(fg="primary", bold=True),  # Version/release names
    "system_progress": StyleSpec(dim=True, italic=True),  # Progress indicators
    "system_init_error": StyleSpec(fg="error", dim=True),  # Initialization failures
    "pager_nav": StyleSpec(fg="primary", bold=True),  # Pager navigation hints

    # UI elements
    "spinner": StyleSpec(fg="primary"),
    "separator": StyleSpec(fg="muted", dim=True),
    "hint": StyleSpec(fg="muted", italic=True),
    "scroll_indicator": StyleSpec(fg="muted", dim=True, italic=True),
    "scroll_arrow": StyleSpec(fg="primary", dim=True),
    "muted": StyleSpec(dim=True),
    "emphasis": StyleSpec(bold=True),
    "warning_icon": StyleSpec(fg="warning"),
    "panel_border": StyleSpec(fg="primary"),
    "tree_connector": StyleSpec(dim=True),
    "truncation": StyleSpec(fg="primary", dim=True, italic=True),

    # Completion menu
    "completion_bg": StyleSpec(bg="surface", fg="text"),
    "completion_selected": StyleSpec(bg="success", fg="text"),
    "completion_meta": StyleSpec(bg="surface", fg="muted"),

    # Pending prompts bar
    "pending_prompt": StyleSpec(fg="primary"),
    "pending_prompt_overflow": StyleSpec(fg="muted", italic=True),

    # Input area
    "input_text": StyleSpec(fg="text"),
    "input_prompt": StyleSpec(fg="primary", bold=True),
}


def _get_builtin_themes_dir() -> Path:
    """Get the path to the built-in themes directory.

    Returns:
        Path to the themes/ directory alongside this module.
    """
    return Path(__file__).parent / "themes"


def _get_theme_search_paths(theme_name: str) -> List[Path]:
    """Get ordered list of paths to search for a theme file.

    Args:
        theme_name: Name of the theme (e.g., "dark", "light").

    Returns:
        List of paths in priority order (first found wins):
        1. User override: ~/.jaato/themes/<name>.json
        2. Project override: .jaato/themes/<name>.json
        3. Built-in: rich-client/themes/<name>.json
    """
    filename = f"{theme_name}.json"
    return [
        Path.home() / ".jaato" / "themes" / filename,  # User override
        Path(".jaato") / "themes" / filename,           # Project override
        _get_builtin_themes_dir() / filename,           # Built-in
    ]


def _load_theme_from_json(theme_name: str) -> Optional["ThemeConfig"]:
    """Load a theme from JSON file, checking override locations first.

    Searches for the theme in this order:
    1. ~/.jaato/themes/<name>.json (user override)
    2. .jaato/themes/<name>.json (project override)
    3. rich-client/themes/<name>.json (built-in)

    Args:
        theme_name: Name of the theme to load.

    Returns:
        ThemeConfig if found and loaded, None otherwise.
    """
    for path in _get_theme_search_paths(theme_name):
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)

                # Import here to avoid circular dependency
                config = ThemeConfig.from_dict(data)
                config._source_path = str(path)

                # Check if this is an override or built-in
                if path.parent.name == "themes" and path.parent.parent == Path.home() / ".jaato":
                    logger.info(f"Loaded theme '{theme_name}' from user override: {path}")
                elif path.parent.name == "themes" and path.parent.parent == Path(".jaato"):
                    logger.info(f"Loaded theme '{theme_name}' from project override: {path}")
                else:
                    logger.debug(f"Loaded built-in theme '{theme_name}' from {path}")

                return config

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in theme file {path}: {e}")
            except Exception as e:
                logger.warning(f"Error reading theme file {path}: {e}")

    return None


def _create_fallback_theme(theme_name: str) -> "ThemeConfig":
    """Create a fallback theme from hardcoded values.

    Used when JSON theme files are unavailable.

    Args:
        theme_name: Name of the theme ("dark", "light", "high-contrast").

    Returns:
        ThemeConfig with hardcoded values.
    """
    if theme_name == "light":
        palette = _FALLBACK_PALETTE_LIGHT.copy()
        semantic = {k: StyleSpec(
            fg=v.fg, bg=v.bg, bold=v.bold, italic=v.italic, dim=v.dim, underline=v.underline
        ) for k, v in DEFAULT_SEMANTIC_STYLES.items()}
        # Light theme overrides
        semantic["tool_output"] = StyleSpec(fg="#006688", italic=True)
        semantic["tool_source_label"] = StyleSpec(fg="#555555", dim=True)
        semantic["agent_tab_separator"] = StyleSpec(fg="#aaaaaa")
        semantic["agent_tab_hint"] = StyleSpec(fg="#888888", italic=True)
        semantic["session_bar_separator"] = StyleSpec(fg="#aaaaaa")
        semantic["status_bar_separator"] = StyleSpec(fg="#bbbbbb", bg="surface")
        semantic["plan_in_progress"] = StyleSpec(fg="#0055cc", bg="surface")
        semantic["agent_processing"] = StyleSpec(fg="#0055cc")
        semantic["tool_indicator"] = StyleSpec(fg="#0055cc", dim=True)
        semantic["plan_popup_empty"] = StyleSpec(fg="#666666", italic=True)
        semantic["plan_popup_empty_border"] = StyleSpec(fg="#888888")
        semantic["plan_popup_scroll_indicator"] = StyleSpec(fg="#666666", italic=True)
        semantic["plan_popup_hint"] = StyleSpec(fg="#666666")
        semantic["plan_popup_step_number"] = StyleSpec(fg="#555555")
        semantic["plan_popup_result_prefix"] = StyleSpec(fg="#555555")
        semantic["plan_popup_error_prefix"] = StyleSpec(fg="#555555")
        semantic["plan_popup_separator"] = StyleSpec(fg="#888888")
        semantic["plan_popup_progress_detail"] = StyleSpec(fg="#555555")
        # Thinking styles - use darker colors without dim for light background
        semantic["thinking_header"] = StyleSpec(fg="#555555")
        semantic["thinking_header_separator"] = StyleSpec(fg="#888888")
        semantic["thinking_border"] = StyleSpec(fg="#888888")
        semantic["thinking_content"] = StyleSpec(fg="#555555", italic=True)
        semantic["thinking_footer"] = StyleSpec(fg="#555555")
        semantic["thinking_footer_separator"] = StyleSpec(fg="#888888")
        return ThemeConfig(
            name="light",
            description="Light theme for bright terminals (fallback)",
            colors=palette,
            semantic=semantic,
            _source_path="fallback",
        )
    elif theme_name == "high-contrast":
        palette = _FALLBACK_PALETTE_HIGH_CONTRAST.copy()
        semantic = {k: StyleSpec(
            fg=v.fg, bg=v.bg, bold=v.bold, italic=v.italic, dim=v.dim, underline=v.underline
        ) for k, v in DEFAULT_SEMANTIC_STYLES.items()}
        # High contrast overrides
        semantic["separator"] = StyleSpec(fg="#aaaaaa")
        semantic["hint"] = StyleSpec(fg="#cccccc", italic=True)
        semantic["plan_popup_empty"] = StyleSpec(fg="#aaaaaa", italic=True)
        semantic["plan_popup_empty_border"] = StyleSpec(fg="#aaaaaa")
        semantic["plan_popup_scroll_indicator"] = StyleSpec(fg="#cccccc", italic=True)
        semantic["plan_popup_hint"] = StyleSpec(fg="#cccccc")
        semantic["plan_popup_step_number"] = StyleSpec(fg="#aaaaaa")
        semantic["plan_popup_result_prefix"] = StyleSpec(fg="#aaaaaa")
        semantic["plan_popup_error_prefix"] = StyleSpec(fg="#aaaaaa")
        semantic["plan_popup_separator"] = StyleSpec(fg="#aaaaaa")
        semantic["plan_popup_progress_detail"] = StyleSpec(fg="#cccccc")
        # Thinking styles - use brighter colors without dim for high contrast
        semantic["thinking_header"] = StyleSpec(fg="#cccccc")
        semantic["thinking_header_separator"] = StyleSpec(fg="#aaaaaa")
        semantic["thinking_border"] = StyleSpec(fg="#cccccc")
        semantic["thinking_content"] = StyleSpec(fg="#cccccc", italic=True)
        semantic["thinking_footer"] = StyleSpec(fg="#cccccc")
        semantic["thinking_footer_separator"] = StyleSpec(fg="#aaaaaa")
        return ThemeConfig(
            name="high-contrast",
            description="High contrast theme for accessibility (fallback)",
            colors=palette,
            semantic=semantic,
            _source_path="fallback",
        )
    else:
        # Default to dark theme
        return ThemeConfig(
            name="dark",
            description="Default dark theme (fallback)",
            colors=_FALLBACK_PALETTE_DARK.copy(),
            semantic={k: StyleSpec(
                fg=v.fg, bg=v.bg, bold=v.bold, italic=v.italic, dim=v.dim, underline=v.underline
            ) for k, v in DEFAULT_SEMANTIC_STYLES.items()},
            _source_path="fallback",
        )


def get_builtin_theme(theme_name: str) -> "ThemeConfig":
    """Get a built-in theme by name, loading from JSON with fallback.

    Args:
        theme_name: Name of the theme ("dark", "light", "high-contrast").

    Returns:
        ThemeConfig instance.
    """
    if theme_name not in get_builtin_theme_names():
        logger.warning(f"Unknown theme '{theme_name}', falling back to 'dark'")
        theme_name = "dark"

    # Try loading from JSON first
    config = _load_theme_from_json(theme_name)
    if config:
        return config

    # Fall back to hardcoded values
    logger.warning(f"Could not load theme '{theme_name}' from JSON, using fallback")
    return _create_fallback_theme(theme_name)


def list_available_themes() -> List[str]:
    """List all available theme names.

    Returns:
        List of theme names (built-in + any custom themes found).
    """
    themes = set(get_builtin_theme_names())

    # Check for custom themes in user directory
    user_themes_dir = Path.home() / ".jaato" / "themes"
    if user_themes_dir.exists():
        for path in user_themes_dir.glob("*.json"):
            themes.add(path.stem)

    # Check for custom themes in project directory
    project_themes_dir = Path(".jaato") / "themes"
    if project_themes_dir.exists():
        for path in project_themes_dir.glob("*.json"):
            themes.add(path.stem)

    return sorted(themes)


@dataclass
class ThemeConfig:
    """Configuration for UI theme colors and styles.

    Attributes:
        name: Theme name for display.
        description: Optional description.
        colors: Base palette mapping color names to hex values.
        semantic: UI element styles mapping to palette colors.
    """
    name: str = "dark"
    description: str = ""
    colors: Dict[str, str] = field(default_factory=lambda: DEFAULT_PALETTE.copy())
    semantic: Dict[str, StyleSpec] = field(default_factory=lambda: DEFAULT_SEMANTIC_STYLES.copy())

    # Metadata (not serialized)
    _source_path: str = field(default="builtin")
    _modified: bool = field(default=False)

    def resolve_color(self, name_or_hex: str) -> str:
        """Resolve a color name to its hex value.

        Args:
            name_or_hex: Either a palette name ("primary") or hex color ("#ff0000").

        Returns:
            Hex color string.
        """
        if is_hex_color(name_or_hex):
            return name_or_hex
        return self.colors.get(name_or_hex, name_or_hex)

    def get_prompt_toolkit_style(self) -> Style:
        """Generate prompt_toolkit Style from theme configuration.

        Returns:
            Style object for use with prompt_toolkit Application.
        """
        style_dict = {}

        # Map semantic styles to prompt_toolkit class names
        pt_mapping = {
            # Agent tab bar
            "agent-tab-bar": "session_bar_bg",
            "agent-tab.selected": "agent_tab_selected",
            "agent-tab.dim": "agent_tab_dim",
            "agent-tab.separator": "agent_tab_separator",
            "agent-tab.hint": "agent_tab_hint",
            "agent-tab.scroll": "agent_tab_scroll",
            "agent-tab.symbol.processing": "agent_processing",
            "agent-tab.symbol.awaiting": "agent_awaiting",
            "agent-tab.symbol.finished": "agent_finished",
            "agent-tab.symbol.error": "agent_error",
            "agent-tab.symbol.permission": "agent_permission",
            # Agent popup
            "agent-popup.border": "agent_popup_border",
            "agent-popup.icon": "agent_popup_icon",
            "agent-popup.name": "agent_popup_name",
            "agent-popup.status.processing": "agent_processing",
            "agent-popup.status.awaiting": "agent_awaiting",
            "agent-popup.status.finished": "agent_finished",
            "agent-popup.status.error": "agent_error",
            # Session bar
            "session-bar": "session_bar_bg",
            "session-bar.label": "session_bar_label",
            "session-bar.id": "session_bar_id",
            "session-bar.separator": "session_bar_separator",
            "session-bar.description": "session_bar_description",
            "session-bar.workspace": "session_bar_workspace",
            "session-bar.dim": "agent_tab_dim",
            # Pending prompts
            "pending-prompts-bar": "session_bar_bg",
            "pending-prompt": "pending_prompt",
            "pending-prompt.overflow": "pending_prompt_overflow",
            # Status bar
            "status-bar": "status_bar_bg",
            "status-bar.label": "status_bar_label",
            "status-bar.value": "status_bar_value",
            "status-bar.separator": "status_bar_separator",
            "status-bar.warning": "status_bar_warning",
            # Output panel
            "output-panel": "output_panel_bg",
            # Plan symbols
            "plan.pending": "plan_pending",
            "plan.in-progress": "plan_in_progress",
            "plan.completed": "plan_completed",
            "plan.failed": "plan_failed",
            "plan.skipped": "plan_skipped",
            "plan.active": "plan_active",
            "plan.cancelled": "plan_cancelled",
            # Completion menu
            "completion-menu.completion": "completion_bg",
            "completion-menu.completion.current": "completion_selected",
            "completion-menu.meta.completion": "completion_meta",
            "completion-menu.meta.completion.current": "completion_selected",
            # Input area
            "": "input_text",  # Default text style
            "prompt": "input_prompt",
        }

        for pt_class, semantic_name in pt_mapping.items():
            if semantic_name in self.semantic:
                style_spec = self.semantic[semantic_name]
                style_dict[pt_class] = style_spec.to_prompt_toolkit(self.colors)

        return Style.from_dict(style_dict)

    def get_rich_style(self, semantic_name: str) -> str:
        """Get Rich style string for a semantic style name.

        Args:
            semantic_name: Name of the semantic style (e.g., "tool_output").

        Returns:
            Rich style string (e.g., "bold #5fd7ff").
        """
        if semantic_name not in self.semantic:
            logger.warning(f"Unknown semantic style: {semantic_name}")
            return ""

        return self.semantic[semantic_name].to_rich(self.colors)

    def get_color(self, name: str) -> str:
        """Get a palette color by name.

        Args:
            name: Color name (e.g., "primary", "success").

        Returns:
            Hex color string.
        """
        return self.colors.get(name, "#ffffff")

    def to_terminal_theme(self) -> "TerminalTheme":
        """Convert theme to Rich TerminalTheme for SVG/HTML export.

        Maps the theme's palette colors to a TerminalTheme suitable for
        Rich Console's save_svg() and save_html() methods.

        Returns:
            TerminalTheme instance with colors from this theme's palette.
        """
        from rich.terminal_theme import TerminalTheme

        bg = hex_to_rgb(self.colors.get("background", "#1a1a1a"))
        fg = hex_to_rgb(self.colors.get("text", "#ffffff"))

        # Build ANSI color palette (16 colors: 8 normal + 8 bright)
        # Standard ANSI: black, red, green, yellow, blue, magenta, cyan, white
        ansi_colors = [
            hex_to_rgb(self.colors.get("background", "#1a1a1a")),  # 0: black
            hex_to_rgb(self.colors.get("error", "#ff5f5f")),       # 1: red
            hex_to_rgb(self.colors.get("success", "#5fd75f")),     # 2: green
            hex_to_rgb(self.colors.get("warning", "#ffff5f")),     # 3: yellow
            hex_to_rgb(self.colors.get("primary", "#5fd7ff")),     # 4: blue
            hex_to_rgb(self.colors.get("accent", "#d7af87")),      # 5: magenta
            hex_to_rgb(self.colors.get("secondary", "#87d787")),   # 6: cyan
            hex_to_rgb(self.colors.get("text", "#ffffff")),        # 7: white
            # Bright variants (same colors for simplicity)
            hex_to_rgb(self.colors.get("muted", "#808080")),       # 8: bright black
            hex_to_rgb(self.colors.get("error", "#ff5f5f")),       # 9: bright red
            hex_to_rgb(self.colors.get("success", "#5fd75f")),     # 10: bright green
            hex_to_rgb(self.colors.get("warning", "#ffff5f")),     # 11: bright yellow
            hex_to_rgb(self.colors.get("primary", "#5fd7ff")),     # 12: bright blue
            hex_to_rgb(self.colors.get("accent", "#d7af87")),      # 13: bright magenta
            hex_to_rgb(self.colors.get("secondary", "#87d787")),   # 14: bright cyan
            hex_to_rgb(self.colors.get("text", "#ffffff")),        # 15: bright white
        ]

        return TerminalTheme(
            background=bg,
            foreground=fg,
            normal=ansi_colors[:8],
            bright=ansi_colors[8:],
        )

    def set_color(self, name: str, hex_value: str) -> bool:
        """Set a palette color.

        Args:
            name: Color name.
            hex_value: Hex color value.

        Returns:
            True if valid and set, False otherwise.
        """
        if not is_hex_color(hex_value):
            return False
        if name not in self.colors:
            return False
        self.colors[name] = hex_value
        self._modified = True
        return True

    def set_semantic_style(self, name: str, spec: StyleSpec) -> bool:
        """Set a semantic style.

        Args:
            name: Semantic style name.
            spec: StyleSpec to set.

        Returns:
            True if valid and set, False otherwise.
        """
        if name not in DEFAULT_SEMANTIC_STYLES:
            return False
        self.semantic[name] = spec
        self._modified = True
        return True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThemeConfig":
        """Create ThemeConfig from a dictionary.

        Args:
            data: Theme configuration dictionary.

        Returns:
            ThemeConfig instance.
        """
        name = data.get("name", "custom")
        description = data.get("description", "")

        # Load palette colors
        colors = DEFAULT_PALETTE.copy()
        if "colors" in data:
            for key, value in data["colors"].items():
                if key in colors and is_hex_color(value):
                    colors[key] = value

        # Load semantic styles
        semantic = {k: StyleSpec(
            fg=v.fg, bg=v.bg, bold=v.bold, italic=v.italic, dim=v.dim, underline=v.underline
        ) for k, v in DEFAULT_SEMANTIC_STYLES.items()}

        if "semantic" in data:
            for key, value in data["semantic"].items():
                if key in semantic:
                    if isinstance(value, dict):
                        semantic[key] = StyleSpec.from_dict(value)
                    elif isinstance(value, str):
                        # Simple color string
                        semantic[key] = StyleSpec(fg=value)

        return cls(
            name=name,
            description=description,
            colors=colors,
            semantic=semantic,
        )

    @classmethod
    def from_file(cls, path: Path) -> Optional["ThemeConfig"]:
        """Load theme from a JSON file.

        Args:
            path: Path to theme JSON file.

        Returns:
            ThemeConfig if loaded successfully, None otherwise.
        """
        if not path.exists():
            return None

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            config = cls.from_dict(data)
            config._source_path = str(path)
            logger.info(f"Loaded theme '{config.name}' from {path}")
            return config

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in theme file {path}: {e}")
        except Exception as e:
            logger.warning(f"Error reading theme file {path}: {e}")

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Export theme configuration to dictionary.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": "1.0",
            "colors": self.colors.copy(),
            "semantic": {
                name: spec.to_dict()
                for name, spec in self.semantic.items()
                if spec.to_dict()  # Only include non-default values
            },
        }

    def save(self, path: Path) -> bool:
        """Save theme to a JSON file.

        Args:
            path: Path to save the theme file.

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)

            self._source_path = str(path)
            self._modified = False
            logger.info(f"Saved theme to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save theme to {path}: {e}")
            return False

    def copy(self) -> "ThemeConfig":
        """Create a deep copy of this theme."""
        return ThemeConfig(
            name=self.name,
            description=self.description,
            colors=self.colors.copy(),
            semantic={k: StyleSpec(
                fg=v.fg, bg=v.bg, bold=v.bold, italic=v.italic, dim=v.dim, underline=v.underline
            ) for k, v in self.semantic.items()},
            _source_path=self._source_path,
            _modified=self._modified,
        )

    @property
    def is_modified(self) -> bool:
        """Check if theme has unsaved changes."""
        return self._modified

    @property
    def source_path(self) -> str:
        """Get the path this theme was loaded from."""
        return self._source_path


class _BuiltinThemesDict(dict):
    """Lazy-loading dictionary for built-in themes.

    Loads themes from JSON files on first access, with fallback to
    hardcoded values if JSON files are unavailable.
    """

    def __init__(self):
        super().__init__()
        self._loaded = set()

    def __getitem__(self, key: str) -> ThemeConfig:
        if key not in self._loaded:
            self._load_theme(key)
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        return key in get_builtin_theme_names()

    def get(self, key: str, default=None) -> Optional[ThemeConfig]:
        if key not in get_builtin_theme_names():
            return default
        return self[key]

    def _load_theme(self, name: str) -> None:
        """Load a theme and cache it."""
        if name in get_builtin_theme_names():
            theme = get_builtin_theme(name)
            super().__setitem__(name, theme)
            self._loaded.add(name)

    def keys(self):
        return get_builtin_theme_names()

    def values(self):
        return [self[name] for name in get_builtin_theme_names()]

    def items(self):
        return [(name, self[name]) for name in get_builtin_theme_names()]


# Built-in themes (lazy-loaded from JSON files)
BUILTIN_THEMES: Dict[str, ThemeConfig] = _BuiltinThemesDict()


def save_theme_preference(theme_name: str) -> bool:
    """Save the selected theme to user preferences.

    Args:
        theme_name: Name of the theme (preset name like 'dark', 'light', 'high-contrast',
                    or 'custom' for user-defined themes).

    Returns:
        True if saved successfully, False otherwise.
    """
    from preferences import save_preference
    return save_preference("theme", theme_name)


def load_theme_preference() -> Optional[str]:
    """Load the saved theme preference.

    Returns:
        Theme name if found, None otherwise.
    """
    from preferences import load_preference
    return load_preference("theme")


def load_theme(
    project_path: str = ".jaato/theme.json",
    user_path: Optional[str] = None,
) -> ThemeConfig:
    """Load theme with fallback chain.

    Priority order:
    1. JAATO_THEME environment variable (preset name: dark, light, high-contrast)
    2. Saved user preference (~/.jaato/preferences.json)
    3. Project-level custom theme (.jaato/theme.json)
    4. User-level custom theme (~/.jaato/theme.json)
    5. Default "dark" theme (loaded from JSON with fallback to hardcoded)

    When loading a built-in theme (via env var, preference, or default),
    the system checks for overrides in this order:
    - ~/.jaato/themes/<name>.json (user override)
    - .jaato/themes/<name>.json (project override)
    - rich-client/themes/<name>.json (built-in)
    - Hardcoded fallback (if all JSON files unavailable)

    Args:
        project_path: Project-level theme config path.
        user_path: User-level theme config path.

    Returns:
        ThemeConfig instance.
    """
    if user_path is None:
        user_path = str(Path.home() / ".jaato" / "theme.json")

    # Check for preset selection via environment (highest priority - temporary override)
    preset = os.environ.get("JAATO_THEME", "").strip().lower()
    if preset and preset in get_builtin_theme_names():
        logger.info(f"Using theme preset '{preset}' from JAATO_THEME environment variable")
        return get_builtin_theme(preset).copy()

    # Check for saved user preference
    saved_pref = load_theme_preference()
    if saved_pref:
        if saved_pref in get_builtin_theme_names():
            logger.info(f"Using saved theme preference: {saved_pref}")
            return get_builtin_theme(saved_pref).copy()
        # If preference is "custom" or unknown, fall through to file loading

    # Try project-level config
    project_config = ThemeConfig.from_file(Path(project_path))
    if project_config:
        return project_config

    # Try user-level config
    user_config = ThemeConfig.from_file(Path(user_path))
    if user_config:
        return user_config

    # Fall back to default
    logger.debug("Using default dark theme")
    return get_builtin_theme("dark").copy()


def get_semantic_style_names() -> List[str]:
    """Get list of all semantic style names.

    Returns:
        Sorted list of semantic style names.
    """
    return sorted(DEFAULT_SEMANTIC_STYLES.keys())


def get_palette_color_names() -> List[str]:
    """Get list of all palette color names.

    Returns:
        List of palette color names in logical order.
    """
    return [
        "primary", "secondary", "accent",
        "success", "warning", "error",
        "muted", "background", "surface",
        "text", "text_muted",
    ]


def validate_theme(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate theme configuration data.

    Args:
        data: Theme configuration dictionary to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []

    # Check version
    version = data.get("version")
    if version and version != "1.0":
        errors.append(f"Unsupported theme version: {version}")

    # Validate colors
    if "colors" in data:
        for name, value in data["colors"].items():
            if name not in DEFAULT_PALETTE:
                errors.append(f"Unknown palette color: {name}")
            elif not is_hex_color(value):
                errors.append(f"Invalid hex color for {name}: {value}")

    # Validate semantic styles
    if "semantic" in data:
        for name, spec in data["semantic"].items():
            if name not in DEFAULT_SEMANTIC_STYLES:
                errors.append(f"Unknown semantic style: {name}")
            elif isinstance(spec, dict):
                if "fg" in spec and spec["fg"] and not is_hex_color(spec["fg"]):
                    # Could be palette reference
                    if spec["fg"] not in DEFAULT_PALETTE:
                        errors.append(f"Invalid color reference in {name}.fg: {spec['fg']}")
                if "bg" in spec and spec["bg"] and not is_hex_color(spec["bg"]):
                    if spec["bg"] not in DEFAULT_PALETTE:
                        errors.append(f"Invalid color reference in {name}.bg: {spec['bg']}")

    return len(errors) == 0, errors


def generate_example_theme() -> str:
    """Generate an example theme.json with comments.

    Returns:
        JSON string suitable for creating a template file.
    """
    example = {
        "_comment": "Theme configuration for jaato rich client",
        "_palette_docs": "Base colors that semantic styles reference",
        "_semantic_docs": "UI element styles - use palette names or hex colors",

        "name": "custom",
        "description": "My custom theme",
        "version": "1.0",

        "colors": DEFAULT_PALETTE.copy(),

        "semantic": {
            "agent_tab_selected": {"fg": "primary", "bold": True, "underline": True},
            "tool_output": {"fg": "#87D7D7", "italic": True},
            "model_header": {"fg": "primary", "bold": True},
            "user_header": {"fg": "success", "bold": True},
        },
    }
    return json.dumps(example, indent=2)
