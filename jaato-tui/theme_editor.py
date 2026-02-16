"""Theme editor TUI component.

Provides an interactive overlay for editing theme colors and styles.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from color_picker import ColorPicker
from theme import (
    BUILTIN_THEMES,
    StyleSpec,
    ThemeConfig,
    get_palette_color_names,
    get_semantic_style_names,
)
from theme_preview import ThemePreview

logger = logging.getLogger(__name__)


class EditorSection(Enum):
    """Sections in the theme editor."""
    PRESETS = "presets"
    COLORS = "colors"
    SEMANTIC = "semantic"


class ThemeEditor:
    """Interactive theme editor overlay.

    Provides a full-screen overlay for editing theme colors with:
    - Preset theme selection
    - Base color palette editing
    - Semantic style editing
    - Live preview panel
    - Color picker integration

    Usage:
        editor = ThemeEditor(current_theme, on_save=save_callback)

        # In render loop
        panel = editor.render(width, height)

        # Handle keys
        if editor.handle_key(key):
            # Key was consumed by editor
            pass
    """

    def __init__(
        self,
        theme: ThemeConfig,
        on_save: Optional[Callable[[ThemeConfig], None]] = None,
        on_close: Optional[Callable[[], None]] = None,
    ):
        """Initialize theme editor.

        Args:
            theme: Current theme to edit (will be copied).
            on_save: Callback when theme is saved.
            on_close: Callback when editor is closed.
        """
        self._theme = theme.copy()
        self._original_theme = theme.copy()
        self._on_save = on_save
        self._on_close = on_close

        # Navigation state
        self._section = EditorSection.PRESETS
        self._selected_index = 0
        self._scroll_offset = 0

        # Editor state
        self._editing = False
        self._color_picker: Optional[ColorPicker] = None
        self._editing_item: Optional[str] = None

        # Preview
        self._preview = ThemePreview(self._theme)

        # Presets list
        self._preset_names = list(BUILTIN_THEMES.keys())

        # Palette colors
        self._color_names = get_palette_color_names()

        # Semantic styles
        self._semantic_names = get_semantic_style_names()

        # Status message
        self._status_message = ""

    @property
    def theme(self) -> ThemeConfig:
        """Get the current theme being edited."""
        return self._theme

    @property
    def is_modified(self) -> bool:
        """Check if theme has unsaved changes."""
        return self._theme.is_modified

    @property
    def is_editing(self) -> bool:
        """Check if currently editing a color."""
        return self._editing

    def _get_current_list(self) -> List[str]:
        """Get the list for current section."""
        if self._section == EditorSection.PRESETS:
            return self._preset_names
        elif self._section == EditorSection.COLORS:
            return self._color_names
        else:
            return self._semantic_names

    def _clamp_selection(self) -> None:
        """Ensure selection is within bounds."""
        items = self._get_current_list()
        if items:
            self._selected_index = max(0, min(self._selected_index, len(items) - 1))
        else:
            self._selected_index = 0

    def _next_section(self) -> None:
        """Cycle to next section."""
        sections = [EditorSection.PRESETS, EditorSection.COLORS, EditorSection.SEMANTIC]
        idx = sections.index(self._section)
        self._section = sections[(idx + 1) % 3]
        self._selected_index = 0
        self._scroll_offset = 0

    def _prev_section(self) -> None:
        """Cycle to previous section."""
        sections = [EditorSection.PRESETS, EditorSection.COLORS, EditorSection.SEMANTIC]
        idx = sections.index(self._section)
        self._section = sections[(idx - 1) % 3]
        self._selected_index = 0
        self._scroll_offset = 0

    def _start_editing(self) -> None:
        """Start editing the currently selected item."""
        if self._section == EditorSection.PRESETS:
            # Apply preset
            preset_name = self._preset_names[self._selected_index]
            self._apply_preset(preset_name)
            return

        if self._section == EditorSection.COLORS:
            color_name = self._color_names[self._selected_index]
            hex_color = self._theme.get_color(color_name)
            self._color_picker = ColorPicker(hex_color)
            self._editing_item = color_name
            self._editing = True

        elif self._section == EditorSection.SEMANTIC:
            semantic_name = self._semantic_names[self._selected_index]
            style = self._theme.semantic.get(semantic_name)
            if style and style.fg:
                # Get the resolved color
                hex_color = self._theme.resolve_color(style.fg)
                self._color_picker = ColorPicker(hex_color)
                self._editing_item = semantic_name
                self._editing = True

    def _finish_editing(self, confirmed: bool) -> None:
        """Finish editing current item.

        Args:
            confirmed: Whether to apply changes.
        """
        if not self._editing or not self._color_picker:
            return

        if confirmed and self._editing_item:
            new_color = self._color_picker.current_color

            if self._section == EditorSection.COLORS:
                self._theme.set_color(self._editing_item, new_color)
                self._status_message = f"Set {self._editing_item} to {new_color}"

            elif self._section == EditorSection.SEMANTIC:
                # Update semantic style foreground
                old_style = self._theme.semantic.get(self._editing_item)
                if old_style:
                    new_style = StyleSpec(
                        fg=new_color,
                        bg=old_style.bg,
                        bold=old_style.bold,
                        italic=old_style.italic,
                        dim=old_style.dim,
                        underline=old_style.underline,
                    )
                    self._theme.set_semantic_style(self._editing_item, new_style)
                    self._status_message = f"Set {self._editing_item} fg to {new_color}"

            # Update preview
            self._preview.set_theme(self._theme)

        self._editing = False
        self._color_picker = None
        self._editing_item = None

    def _apply_preset(self, preset_name: str) -> None:
        """Apply a built-in preset theme.

        Args:
            preset_name: Name of preset to apply.
        """
        if preset_name in BUILTIN_THEMES:
            preset = BUILTIN_THEMES[preset_name]
            self._theme = preset.copy()
            self._theme._modified = True
            self._preview.set_theme(self._theme)
            self._status_message = f"Applied '{preset_name}' preset"

    def _save_theme(self) -> None:
        """Save the current theme."""
        # Default save path
        save_path = Path(".jaato/theme.json")

        if self._theme.save(save_path):
            self._status_message = f"Saved to {save_path}"
            if self._on_save:
                self._on_save(self._theme)
        else:
            self._status_message = "Failed to save theme"

    def handle_key(self, key: str) -> bool:
        """Handle a key press.

        Args:
            key: Key name from prompt_toolkit.

        Returns:
            True if key was handled, False otherwise.
        """
        key = key.lower()

        # If editing a color, delegate to color picker
        if self._editing and self._color_picker:
            result = self._color_picker.handle_key(key)
            if result is None:
                # Cancelled
                self._finish_editing(False)
                return True
            elif result == "":
                # Handled but not finished
                return True
            elif result.startswith("#"):
                # Confirmed with color
                self._finish_editing(True)
                return True
            # Not handled by picker, fall through

        # Navigation
        if key in ("up", "k"):
            self._selected_index = max(0, self._selected_index - 1)
            self._clamp_selection()
            return True

        elif key in ("down", "j"):
            self._selected_index += 1
            self._clamp_selection()
            return True

        elif key == "tab":
            self._next_section()
            return True

        elif key == "s-tab":
            self._prev_section()
            return True

        elif key == "enter":
            self._start_editing()
            return True

        elif key == "s":
            self._save_theme()
            return True

        elif key == "escape":
            if self._editing:
                self._finish_editing(False)
            elif self._on_close:
                self._on_close()
            return True

        elif key == "q":
            if self._on_close:
                self._on_close()
            return True

        elif key == "r":
            # Reset to original
            self._theme = self._original_theme.copy()
            self._preview.set_theme(self._theme)
            self._status_message = "Reset to original theme"
            return True

        elif key == "pageup":
            self._selected_index = max(0, self._selected_index - 10)
            self._clamp_selection()
            return True

        elif key == "pagedown":
            self._selected_index += 10
            self._clamp_selection()
            return True

        elif key == "home":
            self._selected_index = 0
            return True

        elif key == "end":
            items = self._get_current_list()
            self._selected_index = len(items) - 1
            return True

        return False

    def render(self, width: int, height: int) -> Panel:
        """Render the theme editor as a Rich Panel.

        Args:
            width: Available width.
            height: Available height.

        Returns:
            Rich Panel containing the editor UI.
        """
        # Calculate layout
        left_width = max(30, width // 2 - 2)
        right_width = width - left_width - 5

        # Build content table
        table = Table.grid(padding=(0, 2))
        table.add_column(width=left_width)
        table.add_column(width=right_width)

        # Left side: sections
        left_content = self._render_left_panel(left_width, height - 6)

        # Right side: preview or color picker
        if self._editing and self._color_picker:
            right_content = self._render_color_picker_panel()
        else:
            right_content = self._render_preview_panel(right_width)

        table.add_row(left_content, right_content)

        # Status and help
        footer = self._render_footer()

        # Combine
        full_content = Text()
        full_content.append_text(Text.from_markup(table.__rich_console__(None, None).__next__() if hasattr(table, '__rich_console__') else str(table)))
        full_content.append("\n")
        full_content.append_text(footer)

        # Build panel
        title = "Theme Editor"
        if self._theme.is_modified:
            title += " [modified]"

        return Panel(
            table,
            title=title,
            subtitle=self._render_footer(),
            border_style=self._theme.get_rich_style("agent_popup_border") or "cyan",
        )

    def render_simple(self, width: int, height: int) -> Text:
        """Render a simpler text-based version.

        Args:
            width: Available width.
            height: Available height.

        Returns:
            Rich Text with editor content.
        """
        output = Text()

        # Title
        title = "Theme Editor"
        if self._theme.is_modified:
            title += " *"
        output.append(f"{'=' * ((width - len(title) - 2) // 2)} {title} {'=' * ((width - len(title) - 2) // 2)}\n", style="bold")
        output.append("\n")

        # Calculate heights
        section_height = height - 10  # Leave room for preview and footer
        preview_height = 8

        # Section tabs
        for section in EditorSection:
            if section == self._section:
                output.append(f" [{section.value.upper()}] ", style="bold reverse")
            else:
                output.append(f"  {section.value}  ", style="dim")
        output.append("\n")
        output.append("\u2500" * width + "\n", style="dim")

        # Section content
        items = self._get_current_list()
        visible_items = min(len(items), section_height - 2)
        start_idx = max(0, min(self._selected_index - visible_items // 2, len(items) - visible_items))

        for i in range(start_idx, min(start_idx + visible_items, len(items))):
            item = items[i]
            is_selected = i == self._selected_index

            if is_selected:
                output.append("\u25b6 ", style="bold")
            else:
                output.append("  ")

            if self._section == EditorSection.PRESETS:
                # Show preset name with indicator
                is_active = self._theme.name == item
                output.append(item, style="bold" if is_selected else "")
                if is_active:
                    output.append(" (active)", style="dim")

            elif self._section == EditorSection.COLORS:
                # Show color name and swatch
                hex_color = self._theme.get_color(item)
                output.append(f"{item:12s} ", style="bold" if is_selected else "")
                output.append("\u2588\u2588\u2588\u2588", style=hex_color)
                output.append(f" {hex_color}", style="dim")

            elif self._section == EditorSection.SEMANTIC:
                # Show semantic style name and preview
                style = self._theme.semantic.get(item)
                rich_style = self._theme.get_rich_style(item)
                output.append(f"{item:25s} ", style="bold" if is_selected else "")
                output.append("Sample", style=rich_style)
                if style:
                    modifiers = []
                    if style.bold:
                        modifiers.append("B")
                    if style.italic:
                        modifiers.append("I")
                    if style.dim:
                        modifiers.append("D")
                    if modifiers:
                        output.append(f" [{','.join(modifiers)}]", style="dim")

            output.append("\n")

        # Scroll indicators
        if start_idx > 0:
            output.append(f"\u25b2 {start_idx} more\n", style="dim italic")
        if start_idx + visible_items < len(items):
            output.append(f"\u25bc {len(items) - start_idx - visible_items} more\n", style="dim italic")

        # Separator
        output.append("\n")
        output.append("\u2500" * width + "\n", style="dim")

        # Color picker (if editing)
        if self._editing and self._color_picker:
            output.append(f"Editing: {self._editing_item}\n", style="bold")
            output.append_text(self._color_picker.render())
            output.append("\n")
        else:
            # Compact preview
            output.append("PREVIEW\n", style="bold dim")
            output.append_text(self._preview.render_compact(width - 4))

        # Footer
        output.append("\n")
        output.append("\u2500" * width + "\n", style="dim")
        output.append_text(self._render_footer())

        # Status message
        if self._status_message:
            output.append("\n")
            output.append(self._status_message, style="dim italic")

        return output

    def _render_left_panel(self, width: int, height: int) -> Text:
        """Render the left section panel."""
        output = Text()

        # Section tabs
        for section in EditorSection:
            if section == self._section:
                output.append(f"[{section.value}]", style="bold reverse")
            else:
                output.append(f" {section.value} ", style="dim")
            output.append(" ")
        output.append("\n\n")

        # Section content
        items = self._get_current_list()
        for i, item in enumerate(items[:height-2]):
            is_selected = i == self._selected_index
            self._render_item(output, item, is_selected)

        return output

    def _render_item(self, output: Text, item: str, is_selected: bool) -> None:
        """Render a single list item."""
        if is_selected:
            output.append("\u25b6 ", style="bold")
        else:
            output.append("  ")

        if self._section == EditorSection.PRESETS:
            output.append(item, style="bold" if is_selected else "")
            if self._theme.name == item:
                output.append(" \u2713", style="green")

        elif self._section == EditorSection.COLORS:
            hex_color = self._theme.get_color(item)
            output.append(f"{item:12s}", style="bold" if is_selected else "")
            output.append(" \u2588\u2588", style=hex_color)
            output.append(f" {hex_color}", style="dim")

        elif self._section == EditorSection.SEMANTIC:
            rich_style = self._theme.get_rich_style(item)
            output.append(f"{item:20s}", style="bold" if is_selected else "")
            output.append(" Aa", style=rich_style)

        output.append("\n")

    def _render_preview_panel(self, width: int) -> Text:
        """Render the preview panel."""
        output = Text()
        output.append("LIVE PREVIEW\n", style="bold")
        output.append("\u2500" * (width - 2) + "\n", style="dim")
        output.append_text(self._preview.render(width - 2))
        return output

    def _render_color_picker_panel(self) -> Text:
        """Render the color picker panel."""
        output = Text()
        output.append(f"EDITING: {self._editing_item}\n", style="bold")
        output.append("\n")
        if self._color_picker:
            output.append_text(self._color_picker.render())
        return output

    def _render_footer(self) -> Text:
        """Render the footer with keybindings help."""
        output = Text()

        if self._editing:
            output.append("\u2190\u2192", style="bold")
            output.append(":adjust ")
            output.append("Tab", style="bold")
            output.append(":channel ")
            output.append("Enter", style="bold")
            output.append(":confirm ")
            output.append("Esc", style="bold")
            output.append(":cancel")
        else:
            output.append("\u2191\u2193", style="bold")
            output.append(":nav ")
            output.append("Tab", style="bold")
            output.append(":section ")
            output.append("Enter", style="bold")
            output.append(":edit ")
            output.append("S", style="bold")
            output.append(":save ")
            output.append("R", style="bold")
            output.append(":reset ")
            output.append("Esc/Q", style="bold")
            output.append(":close")

        return output


def run_editor(theme_path: Optional[str] = None) -> None:
    """Run the theme editor as a standalone TUI application.

    Args:
        theme_path: Optional path to load/save theme. Defaults to .jaato/theme.json
    """
    import sys
    import tty
    import termios
    from pathlib import Path
    from rich.console import Console
    from rich.live import Live

    # Determine theme path
    if theme_path:
        save_path = Path(theme_path)
    else:
        save_path = Path(".jaato/theme.json")

    # Load existing theme or start with default
    theme = ThemeConfig.from_file(save_path)
    if theme is None:
        theme = BUILTIN_THEMES["dark"].copy()

    console = Console()
    saved = [False]  # Use list to allow mutation in closure

    def on_save(t: ThemeConfig) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if t.save(save_path):
            saved[0] = True

    editor = ThemeEditor(theme, on_save=on_save)

    # Get terminal size
    width = console.width
    height = console.height - 2  # Leave room for status

    def get_key() -> str:
        """Read a single keypress."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            # Handle escape sequences
            if ch == '\x1b':
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A':
                        return 'up'
                    elif ch3 == 'B':
                        return 'down'
                    elif ch3 == 'C':
                        return 'right'
                    elif ch3 == 'D':
                        return 'left'
                return 'escape'
            elif ch == '\r' or ch == '\n':
                return 'enter'
            elif ch == '\t':
                return 'tab'
            elif ch == '\x03':  # Ctrl-C
                return 'ctrl-c'
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    try:
        with Live(console=console, refresh_per_second=10, screen=True) as live:
            while True:
                # Render editor
                panel = editor.render_simple(width, height)
                live.update(panel)

                # Get input
                key = get_key()

                # Handle quit
                if key == 'ctrl-c':
                    break

                # Map keys to editor
                key_map = {
                    'up': 'up',
                    'down': 'down',
                    'left': 'left',
                    'right': 'right',
                    'enter': 'enter',
                    'tab': 'tab',
                    'escape': 'escape',
                    'q': 'q',
                    'Q': 'q',
                    's': 's',
                    'S': 's',
                    'r': 'r',
                    'R': 'r',
                }

                mapped = key_map.get(key, key)
                result = editor.handle_key(mapped)

                if result == "close":
                    break
                elif result == "saved":
                    # Brief feedback then continue
                    pass

    except KeyboardInterrupt:
        pass

    # Final message
    console.clear()
    if saved[0]:
        console.print(f"[green]Theme saved to: {save_path}[/green]")
    else:
        console.print("[dim]Theme editor closed without saving.[/dim]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interactive theme editor for jaato")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to theme.json file (default: .jaato/theme.json)"
    )
    args = parser.parse_args()

    run_editor(args.path)
