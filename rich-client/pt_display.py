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
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
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
    from agent_registry import AgentRegistry

from plan_panel import PlanPanel
from output_buffer import OutputBuffer
from agent_panel import AgentPanel


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

    def __init__(self, input_handler: Optional["InputHandler"] = None, agent_registry: Optional["AgentRegistry"] = None):
        """Initialize the display.

        Args:
            input_handler: Optional InputHandler for completion support.
                          If provided, enables tab completion for commands and files.
            agent_registry: Optional AgentRegistry for agent visibility panel.
                          If provided, enables the agent panel (30% width).
        """
        self._width, self._height = shutil.get_terminal_size()

        # Agent registry and panel
        self._agent_registry = agent_registry
        self._agent_panel: Optional[AgentPanel] = None
        if agent_registry:
            self._agent_panel = AgentPanel(agent_registry)

        # Calculate output width based on whether agent panel is present
        output_width_ratio = 0.7 if self._agent_panel else 1.0
        output_width = int(self._width * output_width_ratio) - 4

        # Rich components
        self._plan_panel = PlanPanel()
        self._output_buffer = OutputBuffer()
        self._output_buffer.set_width(output_width)

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

        # Spinner animation timer (spinner state is in output_buffer)
        self._spinner_timer_active = False

        # Status bar info
        self._model_provider: str = ""
        self._model_name: str = ""
        self._context_usage: Dict[str, Any] = {}

        # Build prompt_toolkit application
        self._app: Optional[Application] = None
        self._build_app()

    def _update_dimensions(self) -> bool:
        """Check if terminal size changed and update components if so.

        Returns:
            True if dimensions changed, False otherwise.
        """
        new_width, new_height = shutil.get_terminal_size()
        if new_width != self._width or new_height != self._height:
            self._width = new_width
            self._height = new_height

            # Update output buffer width (70% if agent panel present, 100% otherwise)
            output_width_ratio = 0.7 if self._agent_panel else 1.0
            output_width = int(self._width * output_width_ratio) - 4
            self._output_buffer.set_width(output_width)

            # Update agent panel width if present
            if self._agent_panel:
                agent_width = int(self._width * 0.3) - 2
                self._agent_panel.set_width(agent_width)

            self._renderer.set_width(self._width)
            return True
        return False

    def _has_plan(self) -> bool:
        """Check if plan panel should be visible."""
        return self._plan_panel.is_visible

    def _get_status_bar_content(self):
        """Get status bar content as formatted text."""
        # Agent name (FIRST field, from selected agent if registry present)
        if self._agent_registry:
            agent_name = self._agent_registry.get_selected_agent_name()
        else:
            agent_name = "main"

        provider = self._model_provider or "—"
        model = self._model_name or "—"

        # Build context usage display (show percentage available)
        # Use selected agent's context if registry present
        if self._agent_registry:
            usage = self._agent_registry.get_selected_context_usage()
        else:
            usage = self._context_usage

        if usage:
            percent_used = usage.get('percent_used', 0)
            percent_available = 100 - percent_used
            total = usage.get('total_tokens', 0)
            # Format: "88% available (15K used)"
            if total >= 1000:
                context_str = f"{percent_available:.0f}% available ({total // 1000}K used)"
            else:
                context_str = f"{percent_available:.0f}% available ({total} used)"
        else:
            context_str = "100% available"

        # Build formatted text with columns
        # Agent | Provider | Model | Context
        return [
            ("class:status-bar.label", " Agent: "),
            ("class:status-bar.value", agent_name),
            ("class:status-bar.separator", "  │  "),
            ("class:status-bar.label", "Provider: "),
            ("class:status-bar.value", provider),
            ("class:status-bar.separator", "  │  "),
            ("class:status-bar.label", "Model: "),
            ("class:status-bar.value", model),
            ("class:status-bar.separator", "  │  "),
            ("class:status-bar.label", "Context: "),
            ("class:status-bar.value", context_str),
            ("class:status-bar", " "),
        ]

    def _get_scroll_page_size(self) -> int:
        """Get the number of lines to scroll per page (half the visible height)."""
        # Ensure dimensions are current
        self._update_dimensions()
        available_height = self._height - 2  # minus input row and status bar
        if self._plan_panel.has_plan:
            available_height -= self._plan_height
        # Scroll by half the visible content area
        return max(3, (available_height - 4) // 2)

    def _get_plan_content(self):
        """Get rendered plan content as ANSI for prompt_toolkit."""
        if self._plan_panel.has_plan:
            rendered = self._plan_panel.render()
            return to_formatted_text(ANSI(self._renderer.render(rendered)))
        return to_formatted_text(ANSI(""))

    def _get_output_content(self):
        """Get rendered output content as ANSI for prompt_toolkit."""
        # Check for terminal resize and update dimensions if needed
        self._update_dimensions()

        # Use selected agent's buffer if registry present, otherwise use default
        if self._agent_registry:
            output_buffer = self._agent_registry.get_selected_buffer()
            if not output_buffer:
                output_buffer = self._output_buffer
        else:
            output_buffer = self._output_buffer

        output_buffer._flush_current_block()

        # Calculate available height for output
        available_height = self._height - 2  # minus input row and status bar
        if self._plan_panel.has_plan:
            available_height -= self._plan_height

        # Calculate width (70% if agent panel present)
        output_width_ratio = 0.7 if self._agent_panel else 1.0
        panel_width = int(self._width * output_width_ratio)

        # Render output panel
        panel = output_buffer.render_panel(
            height=available_height,
            width=panel_width,
        )
        return to_formatted_text(ANSI(self._renderer.render(panel)))

    def _get_agent_panel_content(self):
        """Get rendered agent panel content as ANSI for prompt_toolkit."""
        if not self._agent_panel:
            return to_formatted_text(ANSI(""))

        # Calculate available height for agent panel
        available_height = self._height - 2  # minus input row and status bar
        if self._plan_panel.has_plan:
            available_height -= self._plan_height

        # Render agent panel
        panel = self._agent_panel.render(available_height)
        return to_formatted_text(ANSI(self._renderer.render(panel)))

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

        @kb.add("f1")
        def handle_f1(event):
            """Handle F1 - toggle plan panel collapse/expand."""
            if self._plan_panel.has_plan:
                self._plan_panel.toggle_collapsed()
                self._app.invalidate()

        @kb.add("c-f1")
        def handle_ctrl_f1(event):
            """Handle Ctrl+F1 - toggle plan panel visibility."""
            if self._plan_panel.has_plan:
                self._plan_panel.toggle_hidden()
                self._app.invalidate()

        @kb.add("f2")
        def handle_f2(event):
            """Handle F2 - cycle through agents."""
            if self._agent_registry:
                self._agent_registry.cycle_selection()
                self._app.invalidate()

        # Status bar at top (always visible, 1 line)
        status_bar = Window(
            FormattedTextControl(self._get_status_bar_content),
            height=1,
            style="class:status-bar",
        )

        # Plan panel (conditional - hidden when no plan)
        # Height is dynamic: smaller when collapsed
        def get_plan_height():
            if self._plan_panel.is_collapsed:
                return 7  # Compact: title + progress bar + prev step + current step + borders
            return self._plan_height

        plan_window = ConditionalContainer(
            Window(
                FormattedTextControl(self._get_plan_content),
                height=get_plan_height,
            ),
            filter=Condition(self._has_plan),
        )

        # Output panel (fills remaining space)
        output_window = Window(
            FormattedTextControl(self._get_output_content),
            wrap_lines=False,
        )

        # Agent panel (conditional - only if agent_registry present)
        agent_window = Window(
            FormattedTextControl(self._get_agent_panel_content),
            wrap_lines=False,
        )

        # Combine output and agent panels (70/30 split if agent panel present)
        if self._agent_panel:
            # VSplit: output (70%) on left, agent panel (30%) on right
            content_area = VSplit([
                output_window,
                agent_window,
            ])
        else:
            # No agent panel - just output
            content_area = output_window

        # Input prompt label - changes based on mode (pager, waiting for channel, normal)
        def get_prompt_text():
            if getattr(self, '_pager_active', False):
                return [("class:prompt.pager", "── Enter: next, q: quit ──")]
            if getattr(self, '_waiting_for_channel_input', False):
                return [("class:prompt.permission", "Answer> ")]
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
                status_bar,
                plan_window,
                content_area,  # Output panel (or VSplit with output + agent panel)
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

        NOTE: We only call invalidate() - do NOT call renderer.render() directly
        as this may be called from background threads and would cause race
        conditions with the main event loop's rendering.
        """
        if self._app and self._app.is_running:
            # Invalidate schedules a redraw in the main event loop
            self._app.invalidate()

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

    def run_input_loop(
        self,
        on_input: Callable[[str], None],
        initial_prompt: Optional[str] = None
    ) -> None:
        """Run the input loop.

        This is a blocking call that runs the prompt_toolkit Application.
        The on_input callback is called each time the user presses Enter.

        Args:
            on_input: Callback called with user input text.
            initial_prompt: Optional prompt to auto-submit once event loop is running.
        """
        self._input_callback = on_input

        if initial_prompt:
            # Pre-fill the input buffer and schedule auto-submit
            self.set_input_text(initial_prompt)

            def auto_submit():
                self.submit_input()

            self._app.pre_run_callables = [auto_submit]

        self._app.run()

    # Status bar methods

    def set_model_info(self, provider: str, model: str) -> None:
        """Set the model provider and name for the status bar.

        Args:
            provider: Model provider name (e.g., "Google GenAI", "Anthropic").
            model: Model name (e.g., "gemini-2.5-flash").
        """
        self._model_provider = provider
        self._model_name = model
        self.refresh()

    def update_context_usage(self, usage: Dict[str, Any]) -> None:
        """Update context usage display in status bar.

        Args:
            usage: Dict with 'total_tokens', 'prompt_tokens', 'output_tokens', etc.
        """
        self._context_usage = usage
        self.refresh()

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

    def add_to_history(self, text: str) -> None:
        """Add text to input history.

        Args:
            text: The text to add to history.
        """
        if text and self._input_buffer and self._input_buffer.history:
            self._input_buffer.history.append_string(text)

    def set_input_text(self, text: str) -> None:
        """Set the input buffer text (pre-fill input).

        Args:
            text: The text to set in the input buffer.
        """
        if self._input_buffer:
            self._input_buffer.text = text
            self._input_buffer.cursor_position = len(text)

    def submit_input(self) -> None:
        """Submit the current input buffer content (simulate Enter key)."""
        if self._input_buffer:
            text = self._input_buffer.text.strip()
            if text and self._input_buffer.history:
                self._input_buffer.history.append_string(text)
            self._input_buffer.reset()
            if self._input_callback:
                self._input_callback(text)

    def show_lines(self, lines: list, page_size: int = None) -> None:
        """Show content, automatically paginating if needed.

        Args:
            lines: List of (text, style) tuples to display.
            page_size: Lines per page. If None, uses available height - 4.
        """
        if not lines:
            return

        # Ensure dimensions are current
        self._update_dimensions()

        # Calculate page size based on available height
        if page_size is None:
            available = self._height - 2  # minus input row and status bar
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

    def start_spinner(self) -> None:
        """Start the spinner animation to show model is thinking.

        Thread-safe: can be called from background threads.
        """
        self._output_buffer.start_spinner()
        self._spinner_timer_active = True
        # Schedule spinner advance in main event loop (thread-safe)
        if self._app and self._app.is_running:
            self._app.loop.call_soon_threadsafe(self._advance_spinner)
        else:
            self._advance_spinner()

    def stop_spinner(self) -> None:
        """Stop the spinner animation."""
        self._spinner_timer_active = False
        self._output_buffer.stop_spinner()
        self.refresh()

    def _advance_spinner(self) -> None:
        """Advance spinner animation frame."""
        if not self._spinner_timer_active:
            return
        self._output_buffer.advance_spinner()
        self.refresh()
        # Schedule next frame using prompt_toolkit's call_later
        if self._app and self._app.is_running:
            self._app.loop.call_later(0.1, self._advance_spinner)

    def set_waiting_for_channel_input(self, waiting: bool) -> None:
        """Set whether we're waiting for channel (permission/clarification) input.

        Args:
            waiting: True if waiting for channel input, False otherwise.
        """
        self._waiting_for_channel_input = waiting
        self.refresh()
