"""Display manager using prompt_toolkit with Rich rendering.

Uses prompt_toolkit for full-screen layout management with Rich content
rendered inside prompt_toolkit windows.

This approach renders Rich content to ANSI strings, then wraps them with
prompt_toolkit's ANSI() for display in FormattedTextControl windows.
"""

import shutil
import sys
from io import StringIO
from typing import Any, Callable, Dict, Optional

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, VSplit, Window, ConditionalContainer
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.filters import Condition

from rich.console import Console

# Type checking import for InputHandler
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from input_handler import InputHandler

from plan_panel import PlanPanel
from output_buffer import OutputBuffer


class RichRenderer:
    """Renders Rich content to ANSI strings for prompt_toolkit."""

    def __init__(self, width: int = 80):
        self._width = width

    def set_width(self, width: int) -> None:
        self._width = width

    @property
    def width(self) -> int:
        return self._width

    def render(self, renderable) -> str:
        """Render a Rich object to ANSI string."""
        buffer = StringIO()
        console = Console(
            file=buffer,
            width=self._width,
            force_terminal=True,
            color_system="truecolor",
        )
        console.print(renderable, end="")
        return buffer.getvalue()


class PTDisplay:
    """Display manager using prompt_toolkit with Rich content.

    Uses prompt_toolkit Application for full-screen layout with:
    - Plan panel at top (conditional, hidden when no plan)
    - Output panel in middle (fills remaining space)
    - Input prompt at bottom

    Rich content is rendered to ANSI strings and displayed in
    prompt_toolkit's FormattedTextControl using ANSI() wrapper.
    """

    def __init__(self, input_handler: Optional["InputHandler"] = None):
        """Initialize the display.

        Args:
            input_handler: Optional InputHandler for completion support.
                          If provided, enables tab completion for commands and files.
        """
        self._width, self._height = shutil.get_terminal_size()

        # Rich components
        self._plan_panel = PlanPanel()
        self._output_buffer = OutputBuffer()
        self._output_buffer.set_width(self._width - 4)

        # Rich renderer
        self._renderer = RichRenderer(self._width)

        # Layout configuration
        self._plan_height = 12

        # Input handling with optional completion
        self._input_handler = input_handler
        self._input_buffer = Buffer(
            completer=input_handler._completer if input_handler else None,
            history=input_handler._pt_history if input_handler else None,
            complete_while_typing=True if input_handler else False,
            enable_history_search=True,  # Enable up/down arrow history navigation
        )
        self._input_callback: Optional[Callable[[str], None]] = None

        # Build prompt_toolkit application
        self._app: Optional[Application] = None
        self._build_app()

    def _has_plan(self) -> bool:
        """Check if plan panel should be visible."""
        return self._plan_panel.has_plan

    def _get_scroll_page_size(self) -> int:
        """Get the number of lines to scroll per page (half the visible height)."""
        available_height = self._height - 1  # minus input row
        if self._plan_panel.has_plan:
            available_height -= self._plan_height
        # Scroll by half the visible content area
        return max(3, (available_height - 4) // 2)

    def _get_plan_content(self):
        """Get rendered plan content as ANSI for prompt_toolkit."""
        if self._plan_panel.has_plan:
            rendered = self._plan_panel.render()
            return ANSI(self._renderer.render(rendered))
        return ANSI("")

    def _get_output_content(self):
        """Get rendered output content as ANSI for prompt_toolkit."""
        self._output_buffer._flush_current_block()

        # Calculate available height for output
        available_height = self._height - 1  # minus input row
        if self._plan_panel.has_plan:
            available_height -= self._plan_height

        # Render output panel
        panel = self._output_buffer.render_panel(
            height=available_height,
            width=self._width,
        )
        return ANSI(self._renderer.render(panel))

    def _build_app(self) -> None:
        """Build the prompt_toolkit application."""
        # Key bindings
        kb = KeyBindings()

        @kb.add("enter")
        def handle_enter(event):
            """Handle enter key - submit input or advance pager."""
            if getattr(self, '_pager_active', False):
                # In pager mode - advance page
                if self._input_callback:
                    self._input_callback("")  # Empty string advances pager
            else:
                # Normal mode - submit input
                text = self._input_buffer.text.strip()
                # Add to history before reset (like PromptSession does)
                if text and self._input_buffer.history:
                    self._input_buffer.history.append_string(text)
                self._input_buffer.reset()
                if self._input_callback:
                    self._input_callback(text)

        @kb.add("q")
        def handle_q(event):
            """Handle 'q' key - quit pager if active, otherwise type 'q'."""
            if getattr(self, '_pager_active', False):
                # In pager mode - quit pager
                if self._input_callback:
                    self._input_callback("q")
            else:
                # Normal mode - insert 'q' character
                event.current_buffer.insert_text("q")

        @kb.add("c-c")
        def handle_ctrl_c(event):
            """Handle Ctrl-C - exit."""
            event.app.exit(exception=KeyboardInterrupt())

        @kb.add("c-d")
        def handle_ctrl_d(event):
            """Handle Ctrl-D - EOF."""
            event.app.exit(exception=EOFError())

        @kb.add("pageup")
        def handle_page_up(event):
            """Handle Page-Up - scroll output up."""
            self._output_buffer.scroll_up(lines=self._get_scroll_page_size())
            self._app.invalidate()

        @kb.add("pagedown")
        def handle_page_down(event):
            """Handle Page-Down - scroll output down."""
            self._output_buffer.scroll_down(lines=self._get_scroll_page_size())
            self._app.invalidate()

        @kb.add("home")
        def handle_home(event):
            """Handle Home - scroll to top of output."""
            # Scroll up by a large amount to reach the top
            total_lines = sum(line.display_lines for line in self._output_buffer._lines)
            self._output_buffer.scroll_up(lines=total_lines)
            self._app.invalidate()

        @kb.add("end")
        def handle_end(event):
            """Handle End - scroll to bottom of output."""
            self._output_buffer.scroll_to_bottom()
            self._app.invalidate()

        @kb.add("up")
        def handle_up(event):
            """Handle Up arrow - history/completion navigation."""
            event.current_buffer.auto_up()

        @kb.add("down")
        def handle_down(event):
            """Handle Down arrow - history/completion navigation."""
            event.current_buffer.auto_down()

        # Plan panel (conditional - hidden when no plan)
        plan_window = ConditionalContainer(
            Window(
                FormattedTextControl(self._get_plan_content),
                height=self._plan_height,
            ),
            filter=Condition(self._has_plan),
        )

        # Output panel (fills remaining space)
        output_window = Window(
            FormattedTextControl(self._get_output_content),
            wrap_lines=False,
        )

        # Input prompt label - changes during pager mode
        def get_prompt_text():
            if getattr(self, '_pager_active', False):
                return [("class:prompt.pager", "── Enter: next, q: quit ──")]
            return [("class:prompt", "You> ")]

        prompt_label = Window(
            FormattedTextControl(get_prompt_text),
            height=1,
            dont_extend_width=True,
        )

        # Input text area - hidden during pager mode
        input_window = ConditionalContainer(
            Window(
                BufferControl(buffer=self._input_buffer),
                height=1,
            ),
            filter=Condition(lambda: not getattr(self, '_pager_active', False)),
        )

        # Input row (label + optional input area)
        input_row = VSplit([prompt_label, input_window])

        # Root layout with completions menu
        from prompt_toolkit.layout.containers import FloatContainer, Float
        root = FloatContainer(
            content=HSplit([
                plan_window,
                output_window,
                input_row,
            ]),
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=8),
                ),
            ],
        )

        layout = Layout(root, focused_element=input_window)

        # Get style from input handler if available
        style = self._input_handler._pt_style if self._input_handler else None

        self._app = Application(
            layout=layout,
            key_bindings=kb,
            full_screen=True,
            mouse_support=False,
            style=style,
        )

    def refresh(self) -> None:
        """Refresh the display.

        Invalidates the prompt_toolkit app to trigger re-render.
        The content getter methods (_get_plan_content, _get_output_content)
        are called automatically during render.
        """
        if self._app and self._app.is_running:
            # Invalidate schedules a redraw
            self._app.invalidate()
            # Force the renderer to redraw immediately
            try:
                self._app.renderer.render(self._app, self._app.layout)
            except Exception:
                # Renderer may not be ready in some edge cases
                pass

    def start(self) -> None:
        """Start the display (non-blocking).

        Just validates we're in a TTY. Actual display starts with run_input_loop().
        """
        if not sys.stdout.isatty():
            sys.exit(
                "Error: rich-client requires an interactive terminal.\n"
                "Use simple-client for non-TTY environments."
            )

    def stop(self) -> None:
        """Stop the display."""
        if self._app and self._app.is_running:
            self._app.exit()

    def run_input_loop(self, on_input: Callable[[str], None]) -> None:
        """Run the input loop.

        This is a blocking call that runs the prompt_toolkit Application.
        The on_input callback is called each time the user presses Enter.

        Args:
            on_input: Callback called with user input text.
        """
        self._input_callback = on_input
        self._app.run()

    # Plan panel methods

    def update_plan(self, plan_data: Dict[str, Any]) -> None:
        """Update the plan panel."""
        self._plan_panel.update_plan(plan_data)
        self.refresh()

    def clear_plan(self) -> None:
        """Clear the plan panel."""
        self._plan_panel.clear()
        self.refresh()

    @property
    def has_plan(self) -> bool:
        """Check if there's an active plan."""
        return self._plan_panel.has_plan

    # Output buffer methods

    def append_output(self, source: str, text: str, mode: str) -> None:
        """Append output to the scrolling panel."""
        self._output_buffer.append(source, text, mode)
        # Auto-scroll to bottom when new output arrives
        self._output_buffer.scroll_to_bottom()
        self.refresh()

    def add_system_message(self, message: str, style: str = "dim") -> None:
        """Add a system message to the output."""
        self._output_buffer.add_system_message(message, style)
        # Auto-scroll to bottom when new output arrives
        self._output_buffer.scroll_to_bottom()
        self.refresh()

    def clear_output(self) -> None:
        """Clear the output buffer."""
        self._output_buffer.clear()
        self.refresh()

    def show_lines(self, lines: list, page_size: int = None) -> None:
        """Show content, automatically paginating if needed.

        Args:
            lines: List of (text, style) tuples to display.
            page_size: Lines per page. If None, uses available height - 4.
        """
        if not lines:
            return

        # Calculate page size based on available height
        if page_size is None:
            available = self._height - 1  # minus input row
            if self._plan_panel.has_plan:
                available -= self._plan_height
            # Account for panel borders
            page_size = max(5, available - 4)

        # Check if pagination is needed
        if len(lines) <= page_size:
            # No pagination needed - just display all lines
            for text, style in lines:
                self._output_buffer.add_system_message(text, style)
            self._output_buffer.add_system_message("", style="dim")  # Trailing blank
            self.refresh()
        else:
            # Start pager mode
            self._start_pager(lines, page_size)

    def _start_pager(self, lines: list, page_size: int) -> None:
        """Start paged display mode (internal).

        Args:
            lines: List of (text, style) tuples to display.
            page_size: Lines per page.
        """
        self._pager_lines = lines
        self._pager_page_size = page_size
        self._pager_current = 0
        self._pager_active = True

        self._show_pager_page()

    def _show_pager_page(self) -> None:
        """Show the current pager page."""
        if not self._pager_active:
            return

        lines = self._pager_lines
        page_size = self._pager_page_size
        current = self._pager_current

        total_lines = len(lines)
        total_pages = (total_lines + page_size - 1) // page_size
        page_num = (current // page_size) + 1

        # Clear output for fresh page
        self._output_buffer.clear()

        # Calculate what to show
        end_line = min(current + page_size, total_lines)
        is_last_page = end_line >= total_lines
        lines_on_page = end_line - current

        # For the last page, if it's not full, backfill from previous content
        # to keep the panel full (content is bottom-aligned)
        if is_last_page and lines_on_page < page_size and current > 0:
            # Calculate how many lines we need to backfill
            backfill_count = page_size - lines_on_page
            # Start from earlier in the content
            start_line = max(0, current - backfill_count)
            for text, style in lines[start_line:current]:
                self._output_buffer.add_system_message(text, style)
            # Add a separator to show where new content starts
            self._output_buffer.add_system_message("─" * 40, style="dim")

        # Show current page content
        for text, style in lines[current:end_line]:
            self._output_buffer.add_system_message(text, style)

        # Show pagination status if not last page
        if not is_last_page:
            self._output_buffer.add_system_message(
                f"── Page {page_num}/{total_pages} ── Press Enter for more, 'q' to quit ──",
                style="bold cyan"
            )
        # Note: pager_active is deactivated in handle_pager_input when user advances past last page

        self.refresh()

    def handle_pager_input(self, text: str) -> bool:
        """Handle input while in pager mode.

        Args:
            text: User input text.

        Returns:
            True if input was handled by pager, False if pager not active.
        """
        if not getattr(self, '_pager_active', False):
            return False

        if text.lower() == 'q':
            # Quit pager - add blank line for separation from next command
            self._pager_active = False
            # Don't clear - keep current page content visible
            self._output_buffer.add_system_message("", style="dim")
            self.refresh()
            return True

        # Empty string or any other input advances to next page
        self._pager_current += self._pager_page_size
        if self._pager_current >= len(self._pager_lines):
            # Reached end - last page already shown, just deactivate
            self._pager_active = False
            self._output_buffer.add_system_message("", style="dim")
            self.refresh()
        else:
            self._show_pager_page()

        return True

    @property
    def pager_active(self) -> bool:
        """Check if pager mode is active."""
        return getattr(self, '_pager_active', False)
