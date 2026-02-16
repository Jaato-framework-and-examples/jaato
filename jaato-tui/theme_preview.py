"""Live preview panel for theme editor.

Renders mock UI elements to preview theme changes in real-time.
"""

from rich.text import Text

from theme import ThemeConfig


class ThemePreview:
    """Renders a live preview of theme changes.

    Displays mock UI elements styled with the current theme configuration,
    allowing users to see changes in real-time.
    """

    def __init__(self, theme: ThemeConfig):
        """Initialize the preview.

        Args:
            theme: ThemeConfig to use for styling.
        """
        self._theme = theme

    def set_theme(self, theme: ThemeConfig) -> None:
        """Update the theme for preview.

        Args:
            theme: New ThemeConfig to use.
        """
        self._theme = theme

    def render(self, width: int = 35) -> Text:
        """Render preview as Rich Text.

        Args:
            width: Width of the preview panel.

        Returns:
            Rich Text object with styled preview content.
        """
        output = Text()
        t = self._theme  # Shorthand

        # === Agent Tab Bar ===
        output.append("AGENT TAB BAR\n", style="bold")
        output.append("\u25d0 main", style=t.get_rich_style("agent_processing"))
        output.append(" \u2502 ", style=t.get_rich_style("separator"))
        output.append("\u23f8 sub-agent", style=t.get_rich_style("agent_awaiting"))
        output.append(" \u2502 ", style=t.get_rich_style("separator"))
        output.append("\u25cf done", style=t.get_rich_style("agent_finished"))
        output.append("\n")

        # === Status Bar ===
        output.append("\nSTATUS BAR\n", style="bold")
        output.append("Provider: ", style=t.get_rich_style("status_bar_label"))
        output.append("google", style=t.get_rich_style("status_bar_value"))
        output.append(" \u2502 ", style=t.get_rich_style("separator"))
        output.append("Model: ", style=t.get_rich_style("status_bar_label"))
        output.append("gemini-2.5", style=t.get_rich_style("status_bar_value"))
        output.append("\n")

        # === Separator ===
        output.append("\u2500" * width + "\n", style=t.get_rich_style("separator"))

        # === Conversation ===
        output.append("\nCONVERSATION\n", style="bold")

        # User message
        output.append("[user] ", style=t.get_rich_style("user_header"))
        output.append("Hello, can you help me?\n")

        # Model response
        output.append("[model] ", style=t.get_rich_style("model_header"))
        output.append("Of course! Let me check...\n")

        # === Tool Execution ===
        output.append("\nTOOL EXECUTION\n", style="bold")

        # Completed tool
        output.append("\u251c\u2500 read_file ", style=t.get_rich_style("separator"))
        output.append("\u2713", style=t.get_rich_style("tool_success"))
        output.append(" (0.3s)\n", style=t.get_rich_style("tool_duration"))
        output.append("\u2502   ", style=t.get_rich_style("separator"))
        output.append("/path/to/file.py\n", style=t.get_rich_style("tool_output"))

        # In-progress tool
        output.append("\u251c\u2500 grep ", style=t.get_rich_style("separator"))
        output.append("\u23f3", style=t.get_rich_style("tool_pending"))
        output.append("\n")

        # Failed tool
        output.append("\u2514\u2500 write_file ", style=t.get_rich_style("separator"))
        output.append("\u2717", style=t.get_rich_style("tool_error"))
        output.append(" Error: Permission denied\n", style=t.get_rich_style("tool_error"))

        # === Plan Progress ===
        output.append("\nPLAN PROGRESS\n", style="bold")
        output.append("\u25cf", style=t.get_rich_style("plan_completed"))
        output.append("\u25cf", style=t.get_rich_style("plan_completed"))
        output.append("\u25cf", style=t.get_rich_style("plan_completed"))
        output.append("\u25d0", style=t.get_rich_style("plan_in_progress"))
        output.append("\u25cb", style=t.get_rich_style("plan_pending"))
        output.append("\u25cb", style=t.get_rich_style("plan_pending"))
        output.append(" 3/6 complete\n", style=t.get_rich_style("hint"))

        # === Permission Prompt ===
        output.append("\nPERMISSION PROMPT\n", style="bold")
        output.append("[askPermission] ", style=t.get_rich_style("permission_prompt"))
        output.append("Execute bash command?\n", style=t.get_rich_style("permission_text"))
        output.append("(y/n/a/d) ", style=t.get_rich_style("hint"))

        # === Hints ===
        output.append("\nHINTS & UI\n", style="bold")
        output.append("\u25b2 5 more lines to scroll\n", style=t.get_rich_style("scroll_indicator"))
        output.append("Press Ctrl+P for plan\n", style=t.get_rich_style("hint"))

        # === Color Palette ===
        output.append("\nCOLOR PALETTE\n", style="bold")
        palette_colors = [
            ("primary", t.get_color("primary")),
            ("secondary", t.get_color("secondary")),
            ("success", t.get_color("success")),
            ("warning", t.get_color("warning")),
            ("error", t.get_color("error")),
            ("muted", t.get_color("muted")),
        ]
        for name, hex_color in palette_colors:
            output.append("\u2588\u2588", style=hex_color)
            output.append(f" {name}\n", style=t.get_rich_style("hint"))

        return output

    def render_compact(self, width: int = 30) -> Text:
        """Render a compact preview suitable for small panels.

        Args:
            width: Width of the preview panel.

        Returns:
            Rich Text object with compact preview.
        """
        output = Text()
        t = self._theme

        # Agent status line
        output.append("\u25d0", style=t.get_rich_style("agent_processing"))
        output.append(" main ", style=t.get_rich_style("agent_tab_selected"))
        output.append("\u2502 ", style=t.get_rich_style("separator"))
        output.append("\u25cf", style=t.get_rich_style("agent_finished"))
        output.append(" done\n", style=t.get_rich_style("agent_finished"))

        # Separator
        output.append("\u2500" * width + "\n", style=t.get_rich_style("separator"))

        # Conversation snippet
        output.append("[user] ", style=t.get_rich_style("user_header"))
        output.append("Hello\n")
        output.append("[model] ", style=t.get_rich_style("model_header"))
        output.append("Response\n")

        # Tool snippet
        output.append("\u2514\u2500 read ", style=t.get_rich_style("separator"))
        output.append("\u2713", style=t.get_rich_style("tool_success"))
        output.append(" file.py\n", style=t.get_rich_style("tool_output"))

        # Plan progress
        output.append("\u25cf\u25cf\u25d0\u25cb\u25cb", style="")
        # Apply individual styles
        progress_text = output._spans[-1] if output._spans else None
        output.append("\n")

        return output

    def render_color_sample(self, hex_color: str, width: int = 20) -> Text:
        """Render a color sample with the given hex color.

        Args:
            hex_color: Hex color to display.
            width: Width of the sample.

        Returns:
            Rich Text showing the color.
        """
        output = Text()
        # Color swatch
        output.append("\u2588" * 4, style=hex_color)
        output.append(f" {hex_color}", style=self._theme.get_rich_style("hint"))
        return output
