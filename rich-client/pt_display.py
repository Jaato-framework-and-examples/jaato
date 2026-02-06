"""Display manager using prompt_toolkit with Rich rendering.

Uses prompt_toolkit for full-screen layout management with Rich content
rendered inside prompt_toolkit windows.

This approach renders Rich content to ANSI strings, then wraps them with
prompt_toolkit's ANSI() for display in FormattedTextControl windows.
"""

import re
import shutil
import sys
import time
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import HSplit, VSplit, Window, ConditionalContainer
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import Processor, Transformation, TransformationInput
from prompt_toolkit.filters import Condition
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.document import Document

from rich.console import Console

# Type checking import for InputHandler
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from input_handler import InputHandler
    from agent_registry import AgentRegistry

from plan_panel import PlanPanel
from budget_panel import BudgetPanel
from output_buffer import OutputBuffer
from agent_panel import AgentPanel
from agent_tab_bar import AgentTabBar
from clipboard import ClipboardConfig, ClipboardProvider, create_provider
from keybindings import KeybindingConfig, load_keybindings, format_key_for_display
from theme import ThemeConfig, load_theme
from shared.plugins.formatter_pipeline import create_pipeline
from shared.plugins.hidden_content_filter import create_plugin as create_hidden_filter
from shared.plugins.code_block_formatter import create_plugin as create_code_block_formatter
from shared.plugins.diff_formatter import create_plugin as create_diff_formatter
from shared.plugins.table_formatter import create_plugin as create_table_formatter


def consolidate_fragments(fragments):
    """Consolidate consecutive fragments with the same style.

    ANSI parsing produces character-by-character fragments.
    This merges them into larger chunks for efficiency.
    """
    result = []
    current_style = None
    current_text = []
    for style, char in fragments:
        if style == current_style:
            current_text.append(char)
        else:
            if current_text:
                result.append((current_style, ''.join(current_text)))
            current_style = style
            current_text = [char]
    if current_text:
        result.append((current_style, ''.join(current_text)))
    return result


def ansi_to_plain_and_fragments(ansi_str: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Convert ANSI string to plain text and style fragments.

    Returns:
        Tuple of (plain_text, fragments) where fragments is a list of (style, text) tuples.
    """
    fragments = consolidate_fragments(to_formatted_text(ANSI(ansi_str)))
    plain_text = ''.join(text for _, text in fragments)
    return plain_text, fragments


class StyledOutputProcessor(Processor):
    """Input processor that applies stored style fragments to buffer lines.

    This enables BufferControl to display styled text while maintaining
    selection functionality. The styles are looked up from a callback
    that returns fragments for each line number.

    Selection is visualized by applying reverse video to selected characters.
    """

    def __init__(self, get_line_fragments: Callable[[int], Optional[List[Tuple[str, str]]]],
                 buffer: Optional['Buffer'] = None):
        self._get_line_fragments = get_line_fragments
        self._buffer = buffer

    def apply_transformation(self, ti: TransformationInput) -> Transformation:
        """Apply stored styles to existing fragments, with selection highlighting."""
        styled_fragments = self._get_line_fragments(ti.lineno)
        if not styled_fragments:
            # No styling info, return as-is
            return Transformation(ti.fragments)

        # Get total character count from input
        input_text = ''.join(text for _, text in ti.fragments)
        input_len = len(input_text)

        if input_len == 0:
            return Transformation(ti.fragments)

        # Get selection range for this line if buffer has active selection
        selection_start_col = None
        selection_end_col = None
        if self._buffer and self._buffer.selection_state:
            doc = self._buffer.document
            sel_start = min(self._buffer.cursor_position, self._buffer.selection_state.original_cursor_position)
            sel_end = max(self._buffer.cursor_position, self._buffer.selection_state.original_cursor_position)

            # Convert absolute positions to line/column
            sel_start_row, sel_start_col_abs = doc.translate_index_to_position(sel_start)
            sel_end_row, sel_end_col_abs = doc.translate_index_to_position(sel_end)

            # Check if this line is within selection
            if sel_start_row <= ti.lineno <= sel_end_row:
                if sel_start_row == sel_end_row:
                    # Selection on single line
                    selection_start_col = sel_start_col_abs
                    selection_end_col = sel_end_col_abs
                elif ti.lineno == sel_start_row:
                    # First line of multi-line selection
                    selection_start_col = sel_start_col_abs
                    selection_end_col = input_len
                elif ti.lineno == sel_end_row:
                    # Last line of multi-line selection
                    selection_start_col = 0
                    selection_end_col = sel_end_col_abs
                else:
                    # Middle line - fully selected
                    selection_start_col = 0
                    selection_end_col = input_len

        # Build result using our styled fragments, but ensure character count matches
        result = []
        styled_char_count = 0

        for style, text in styled_fragments:
            if styled_char_count >= input_len:
                break
            # Truncate if this fragment would exceed input length
            remaining = input_len - styled_char_count
            if len(text) > remaining:
                text = text[:remaining]
            if text:
                # Apply selection highlighting if this fragment overlaps selection
                if selection_start_col is not None:
                    frag_start = styled_char_count
                    frag_end = styled_char_count + len(text)

                    # Check overlap with selection
                    if frag_end > selection_start_col and frag_start < selection_end_col:
                        # Fragment overlaps with selection - split if needed
                        parts = []

                        # Part before selection
                        if frag_start < selection_start_col:
                            before_len = selection_start_col - frag_start
                            parts.append((style, text[:before_len]))
                            text = text[before_len:]
                            frag_start = selection_start_col

                        # Selected part
                        sel_in_frag_end = min(frag_end, selection_end_col)
                        sel_len = sel_in_frag_end - frag_start
                        if sel_len > 0:
                            # Add reverse video for selected text
                            sel_style = style + ' reverse' if style else 'reverse'
                            parts.append((sel_style, text[:sel_len]))
                            text = text[sel_len:]

                        # Part after selection
                        if text:
                            parts.append((style, text))

                        result.extend(parts)
                        styled_char_count = frag_end
                        continue

                result.append((style, text))
                styled_char_count += len(text)

        # If our styled fragments are shorter than input, pad with remaining input text
        if styled_char_count < input_len:
            remaining_text = input_text[styled_char_count:]
            result.append(('', remaining_text))

        return Transformation(result)


def _clean_panel_borders(text: str) -> str:
    """Remove Rich panel borders from multi-line selection.

    When selecting text across lines in a Rich panel, the box-drawing
    border characters get included. This cleans them up for clipboard.
    """
    lines = text.split('\n')
    if len(lines) <= 1:
        return text

    import re
    cleaned = []
    for line in lines:
        # Remove left border: box-drawing chars (│║┃) + optional space
        line = re.sub(r'^[│║┃]+\s?', '', line)
        # Remove right border: optional space + box-drawing chars
        line = re.sub(r'\s?[│║┃]+$', '', line)
        cleaned.append(line)

    return '\n'.join(cleaned)


class ScrollableBufferControl(BufferControl):
    """BufferControl that handles mouse scroll events.

    Extends BufferControl to intercept scroll events while preserving
    text selection functionality.
    """

    # Double-click threshold in seconds
    DOUBLE_CLICK_THRESHOLD = 0.4

    def __init__(self, on_scroll_up=None, on_scroll_down=None, input_buffer=None,
                 on_selection_complete=None, **kwargs):
        super().__init__(**kwargs)
        self._on_scroll_up = on_scroll_up
        self._on_scroll_down = on_scroll_down
        self._input_buffer = input_buffer  # Reference to input buffer for focus return
        self._mouse_down_cursor_pos = None  # Track cursor position at mouse down
        self._on_selection_complete = on_selection_complete  # Callback with selected text
        self._last_click_time = 0.0  # For double-click detection
        self._last_click_pos = None  # Position of last click

    def _select_word_at_cursor(self) -> str | None:
        """Select the word at the current cursor position and return it."""
        doc = self.buffer.document
        text = doc.text
        pos = self.buffer.cursor_position

        if not text or pos < 0 or pos > len(text):
            return None

        # Find word boundaries (alphanumeric + underscore)
        def is_word_char(c):
            return c.isalnum() or c == '_'

        # Find start of word
        start = pos
        while start > 0 and is_word_char(text[start - 1]):
            start -= 1

        # Find end of word
        end = pos
        while end < len(text) and is_word_char(text[end]):
            end += 1

        if start == end:
            return None

        return text[start:end]

    def mouse_handler(self, mouse_event: MouseEvent):
        """Handle mouse scroll events, delegate others to parent."""
        import time

        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            if self._on_scroll_up:
                self._on_scroll_up()
                return None  # Event handled
        elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            if self._on_scroll_down:
                self._on_scroll_down()
                return None  # Event handled
        # For click/drag events, handle focus and selection
        if mouse_event.event_type == MouseEventType.MOUSE_DOWN:
            from prompt_toolkit.application import get_app
            from prompt_toolkit.selection import SelectionType
            app = get_app()
            if app and app.layout:
                # Focus this control to enable selection
                app.layout.focus(self)
            # Call parent handler to set cursor position
            super().mouse_handler(mouse_event)

            current_time = time.time()
            current_pos = self.buffer.cursor_position

            # Check for double-click (same position within threshold)
            is_double_click = (
                self._last_click_pos is not None and
                current_pos == self._last_click_pos and
                (current_time - self._last_click_time) < self.DOUBLE_CLICK_THRESHOLD
            )

            if is_double_click:
                # Double-click: select word, copy, deselect
                word = self._select_word_at_cursor()
                if word and self._on_selection_complete:
                    self._on_selection_complete(word)
                # Reset double-click tracking
                self._last_click_time = 0.0
                self._last_click_pos = None
                self._mouse_down_cursor_pos = None
                # Return focus to input
                if app and app.layout and self._input_buffer:
                    app.layout.focus(self._input_buffer)
                return None

            # Single click - track for potential double-click
            self._last_click_time = current_time
            self._last_click_pos = current_pos
            # Remember cursor position at mouse down
            self._mouse_down_cursor_pos = self.buffer.cursor_position
            # Start selection from current cursor position
            self.buffer.start_selection(selection_type=SelectionType.CHARACTERS)
            return None
        elif mouse_event.event_type == MouseEventType.MOUSE_MOVE:
            # During drag, update cursor position (extends selection)
            super().mouse_handler(mouse_event)
            return None
        elif mouse_event.event_type == MouseEventType.MOUSE_UP:
            # Update final cursor position
            super().mouse_handler(mouse_event)
            # If cursor didn't move (just a click, no drag), clear selection
            if self._mouse_down_cursor_pos is not None:
                if self.buffer.cursor_position == self._mouse_down_cursor_pos:
                    # Just a click without drag - clear selection
                    self.buffer.exit_selection()
                else:
                    # Selection completed - copy to clipboard if callback provided
                    if self._on_selection_complete and self.buffer.selection_state:
                        # Get selected text by extracting range from document
                        sel_start = min(self.buffer.cursor_position,
                                        self.buffer.selection_state.original_cursor_position)
                        sel_end = max(self.buffer.cursor_position,
                                      self.buffer.selection_state.original_cursor_position)
                        selected_text = self.buffer.document.text[sel_start:sel_end]
                        # Clean up panel borders from multi-line selections
                        selected_text = _clean_panel_borders(selected_text)
                        if selected_text:
                            self._on_selection_complete(selected_text)
                    # Clear selection after copying
                    self.buffer.exit_selection()
            self._mouse_down_cursor_pos = None
            # Return focus to input so user can type
            from prompt_toolkit.application import get_app
            app = get_app()
            if app and app.layout and self._input_buffer:
                app.layout.focus(self._input_buffer)
            return None
        return super().mouse_handler(mouse_event)


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

    # Panel width ratios when agent panel is visible
    OUTPUT_PANEL_WIDTH_RATIO = 0.8  # 80% for output
    AGENT_PANEL_WIDTH_RATIO = 0.2   # 20% for agents

    # Input area height limits (expandable input)
    INPUT_MIN_HEIGHT = 1   # Minimum input height (single line)
    INPUT_MAX_HEIGHT = 10  # Maximum input height before scrolling

    def __init__(
        self,
        input_handler: Optional["InputHandler"] = None,
        agent_registry: Optional["AgentRegistry"] = None,
        clipboard_config: Optional[ClipboardConfig] = None,
        keybinding_config: Optional[KeybindingConfig] = None,
        theme_config: Optional[ThemeConfig] = None,
        server_formatted: bool = False,
    ):
        """Initialize the display.

        Args:
            input_handler: Optional InputHandler for completion support.
                          If provided, enables tab completion for commands and files.
            agent_registry: Optional AgentRegistry for agent visibility panel.
                          If provided, enables the agent panel (AGENT_PANEL_WIDTH_RATIO width).
            clipboard_config: Optional ClipboardConfig for copy/paste operations.
                             If not provided, uses config from environment.
            keybinding_config: Optional KeybindingConfig for custom keybindings.
                              If not provided, loads from config files or uses defaults.
            theme_config: Optional ThemeConfig for UI theming.
                         If not provided, loads from config files or uses default dark theme.
            server_formatted: If True, skip client-side formatting (server already formatted).
                             Used in IPC mode where server handles syntax highlighting.
        """
        self._width, self._height = shutil.get_terminal_size()

        # Keybinding configuration (needed early for panels)
        self._keybinding_config = keybinding_config or load_keybindings()

        # Theme configuration (needed for styling)
        self._theme = theme_config or load_theme()

        # Agent registry and tab bar (horizontal tabs at top)
        self._agent_registry = agent_registry
        self._agent_tab_bar: Optional[AgentTabBar] = None
        self._agent_panel: Optional[AgentPanel] = None  # Keep for compatibility
        if agent_registry:
            self._agent_tab_bar = AgentTabBar(
                agent_registry,
                cycle_key=self._keybinding_config.cycle_agents
            )
            self._agent_tab_bar.set_width(self._width)

        # Calculate output width (now always 100% since tab bar is horizontal)
        output_width = self._width - 4

        # Rich components
        self._plan_panel = PlanPanel(toggle_key=self._keybinding_config.toggle_plan)
        self._plan_panel.set_theme(self._theme)
        self._budget_panel = BudgetPanel(toggle_key=self._keybinding_config.toggle_budget)
        self._budget_panel.set_theme(self._theme)
        self._output_buffer = OutputBuffer()
        self._output_buffer.set_width(output_width)
        self._output_buffer.set_keybinding_config(self._keybinding_config)
        self._output_buffer.set_theme(self._theme)

        # Formatter pipeline for output processing (syntax highlighting, diff coloring)
        # Skip in server_formatted mode - server already handles formatting
        self._formatter_pipeline = None
        self._code_block_formatter = None  # Keep reference for theme updates
        if not server_formatted:
            self._formatter_pipeline = create_pipeline()
            self._formatter_pipeline.register(create_hidden_filter())         # priority 5
            self._formatter_pipeline.register(create_diff_formatter())        # priority 20
            self._formatter_pipeline.register(create_table_formatter())       # priority 25
            # Code block formatter with line numbers enabled
            self._code_block_formatter = create_code_block_formatter()
            self._code_block_formatter.initialize({"line_numbers": True})
            self._code_block_formatter.set_syntax_theme(self._theme.name)  # Match UI theme
            self._formatter_pipeline.register(self._code_block_formatter)     # priority 40
            self._formatter_pipeline.set_console_width(output_width)
            self._output_buffer.set_formatter_pipeline(self._formatter_pipeline)

        # Set keybinding config and theme on agent registry buffers too
        if self._agent_registry:
            self._agent_registry.set_keybinding_config_all(self._keybinding_config)
            self._agent_registry.set_theme_all(self._theme)
            # Also set formatter pipeline on agent buffers (only if not server_formatted)
            if self._formatter_pipeline:
                self._agent_registry.set_formatter_pipeline_all(self._formatter_pipeline)

        # Rich renderer
        self._renderer = RichRenderer(self._width)

        # Output display buffer (prompt_toolkit Buffer for selection support)
        # Plain text goes here, styling is applied via StyledOutputProcessor
        self._output_pt_buffer = Buffer(
            document=Document("", 0),
            read_only=True,
        )
        self._output_line_fragments: Dict[int, List[Tuple[str, str]]] = {}

        # Input handling with optional completion
        self._input_handler = input_handler
        self._input_buffer = Buffer(
            completer=input_handler._completer if input_handler else None,
            history=input_handler._pt_history if input_handler else None,
            complete_while_typing=True if input_handler else False,
            enable_history_search=True,  # Enable up/down arrow history navigation
            on_text_changed=lambda _: self._on_input_changed(),  # Trigger layout update
        )
        self._input_callback: Optional[Callable[[str], None]] = None

        # Spinner animation timer (spinner state is in output_buffer)
        self._spinner_timer_active = False

        # Status bar info
        self._model_provider: str = ""
        self._model_name: str = ""
        self._context_usage: Dict[str, Any] = {}
        self._gc_threshold: Optional[float] = None  # GC threshold percentage (e.g., 80.0)
        self._gc_strategy: Optional[str] = None  # GC strategy name (e.g., "truncate", "hybrid")
        self._gc_target_percent: Optional[float] = None  # Target usage after GC
        self._gc_continuous_mode: bool = False  # True if GC runs after every turn

        # Session bar info
        self._session_id: str = ""
        self._session_description: str = ""
        self._session_workspace: str = ""

        # Permission status for session bar display
        self._permission_default: str = "ask"  # "allow", "deny", or "ask"
        self._permission_suspension: Optional[str] = None  # "turn", "idle", "session", or None

        # Temporary status message (for copy feedback, etc.)
        self._status_message: Optional[str] = None
        self._status_message_expires: float = 0.0

        # Persistent connection status (for IPC reconnection state)
        self._connection_status: Optional[str] = None
        self._connection_status_style: str = "warning"  # "warning" or "error"

        # Initialization progress tracking
        self._init_progress_lines: List[Tuple[str, str]] = []  # [(line, style), ...]

        # Clipboard support
        self._clipboard_config = clipboard_config or ClipboardConfig.from_env()
        self._clipboard: ClipboardProvider = create_provider(self._clipboard_config)

        # Mouse selection tracking (for preemptive copy)
        self._mouse_selection_start: Optional[int] = None  # Start Y coordinate
        self._mouse_selecting: bool = False

        # Permission input filtering state
        self._waiting_for_channel_input: bool = False
        self._valid_input_prefixes: set = set()  # All valid prefixes for permission responses
        self._last_valid_permission_input: str = ""  # Track last valid input for reverting

        # Permission keyboard navigation state
        self._permission_response_options: Optional[list] = None  # Current response options
        self._permission_focus_index: int = 0  # Currently focused option index

        # Stop callback for interrupting model generation
        self._stop_callback: Optional[Callable[[], bool]] = None
        self._is_running_callback: Optional[Callable[[], bool]] = None

        # Pending prompts queue (displayed above input field)
        # Each entry is (prompt_text, timestamp)
        self._pending_prompts: List[Tuple[str, float]] = []

        # Debounced refresh for streaming performance
        self._refresh_pending: bool = False
        self._refresh_interval: float = 0.05  # 50ms debounce window

        # Custom prompt override (for exit confirmation, etc.)
        self._custom_prompt: Optional[str] = None

        # Search mode state
        self._search_mode: bool = False
        self._search_query: str = ""

        # Paste registry for large paste handling
        # Large pastes are stored here and replaced with placeholders
        self._paste_registry: Dict[int, str] = {}
        self._paste_counter: int = 0
        self._paste_threshold_lines: int = 10  # Lines threshold for placeholder
        self._paste_threshold_chars: int = 1000  # Character threshold for placeholder

        # Build prompt_toolkit application
        self._app: Optional[Application] = None
        self._build_app()

    def reload_keybindings(self, config: Optional[KeybindingConfig] = None) -> bool:
        """Reload keybindings from configuration.

        This rebuilds the prompt_toolkit application with new keybindings.

        Args:
            config: Optional new KeybindingConfig. If None, reloads from
                   config files and environment variables.

        Returns:
            True if keybindings were reloaded successfully.
        """
        if config is None:
            config = load_keybindings()

        self._keybinding_config = config
        self._build_app()
        return True

    def _update_dimensions(self) -> bool:
        """Check if terminal size changed and update components if so.

        Returns:
            True if dimensions changed, False otherwise.
        """
        new_width, new_height = shutil.get_terminal_size()
        if new_width != self._width or new_height != self._height:
            self._width = new_width
            self._height = new_height

            # Update output buffer width (now always 100% since tab bar is horizontal)
            output_width = self._width - 4
            self._output_buffer.set_width(output_width)

            # Update agent tab bar width if present
            if self._agent_tab_bar:
                self._agent_tab_bar.set_width(self._width)

            self._renderer.set_width(self._width)
            return True
        return False

    def _on_input_changed(self) -> None:
        """Called when input buffer text changes - invalidates layout for resize.

        In permission mode, this also validates input and reverts invalid characters.
        Also triggers completion for @file and %prompt references on deletion,
        since prompt_toolkit's complete_while_typing only triggers on insertion.
        """
        # In permission mode, validate input and revert if invalid
        if self._waiting_for_channel_input and self._valid_input_prefixes:
            current_text = self._input_buffer.text
            if not self._is_valid_permission_input(current_text):
                # Revert to last valid input
                self._input_buffer.text = self._last_valid_permission_input
                self._input_buffer.cursor_position = len(self._last_valid_permission_input)
            else:
                # Update last valid input
                self._last_valid_permission_input = current_text
        elif self._search_mode:
            # Search mode: update search results as user types
            self._update_search()
        else:
            # Normal mode: retrigger completion for @file and %prompt on deletion
            # prompt_toolkit's complete_while_typing only triggers on insertion,
            # so we manually trigger when user is in a completion context
            self._maybe_retrigger_completion()

        if self._app and self._app.is_running:
            self._app.invalidate()

    def _maybe_retrigger_completion(self) -> None:
        """Retrigger completion if user is in an @file or %prompt context.

        This handles the case where user deletes characters - completion should
        update to show matches for the shorter pattern.
        """
        if not self._input_buffer or not self._app:
            return

        text = self._input_buffer.document.text_before_cursor
        if not text:
            return

        # Check if we're in a completion context (after @ or %)
        # Find the last @ or % that looks like a reference start
        in_completion_context = False

        # Check for @file context
        at_pos = text.rfind('@')
        if at_pos != -1:
            # Valid if @ is at start or preceded by whitespace/punctuation
            if at_pos == 0 or not (text[at_pos - 1].isalnum() or text[at_pos - 1] in '._-'):
                # Check no space immediately after @
                if at_pos == len(text) - 1 or text[at_pos + 1] != ' ':
                    in_completion_context = True

        # Check for %prompt context
        if not in_completion_context:
            pct_pos = text.rfind('%')
            if pct_pos != -1:
                # Valid if % is at start or not preceded by alphanumeric
                if pct_pos == 0 or not text[pct_pos - 1].isalnum():
                    # Check no space immediately after %
                    if pct_pos == len(text) - 1 or text[pct_pos + 1] != ' ':
                        in_completion_context = True

        # Trigger completion if in context and completion not already active
        if in_completion_context and not self._input_buffer.complete_state:
            self._input_buffer.start_completion()

    def _get_input_height(self) -> int:
        """Calculate dynamic height for input area based on content.

        Returns the number of lines needed to display the current input,
        clamped between INPUT_MIN_HEIGHT and INPUT_MAX_HEIGHT.
        """
        if not self._input_buffer:
            return self.INPUT_MIN_HEIGHT

        text = self._input_buffer.text
        if not text:
            return self.INPUT_MIN_HEIGHT

        # Count newlines in the text (each newline = additional line)
        line_count = text.count('\n') + 1

        # Also account for line wrapping based on terminal width
        # Subtract prompt width ("You> " = 5 chars) from available width
        prompt_width = 8  # "You> " or "Answer> " with some padding
        available_width = max(20, self._width - prompt_width)

        # Calculate wrapped lines for each logical line
        wrapped_lines = 0
        for line in text.split('\n'):
            if len(line) == 0:
                wrapped_lines += 1
            else:
                # Ceiling division: how many rows does this line take?
                wrapped_lines += (len(line) + available_width - 1) // available_width

        # Use the larger of line count or wrapped line count
        height = max(line_count, wrapped_lines)

        # Clamp to configured limits
        return max(self.INPUT_MIN_HEIGHT, min(height, self.INPUT_MAX_HEIGHT))

    def _get_current_plan_data(self) -> Optional[Dict[str, Any]]:
        """Get plan data for the currently selected agent.

        Uses agent registry if available (per-agent plans), otherwise
        falls back to the global PlanPanel.
        """
        if self._agent_registry:
            return self._agent_registry.get_selected_plan_data()
        return self._plan_panel._plan_data

    def _get_current_plan_symbols(self) -> List[tuple]:
        """Get plan symbols for the currently selected agent.

        Returns formatted text tuples for prompt_toolkit status bar.
        """
        plan_data = self._get_current_plan_data()
        if not plan_data:
            return []

        steps = plan_data.get("steps", [])
        if not steps:
            return []

        # Map status to prompt_toolkit style classes
        style_map = {
            "pending": "class:plan.pending",
            "in_progress": "class:plan.in-progress",
            "completed": "class:plan.completed",
            "failed": "class:plan.failed",
            "skipped": "class:plan.skipped",
            "active": "class:plan.active",
            "cancelled": "class:plan.cancelled",
        }

        symbol_map = {
            "pending": "○",
            "in_progress": "◐",
            "completed": "●",
            "failed": "✗",
            "skipped": "⊘",
            "active": "▸",
            "cancelled": "⊘",
        }

        # Sort by sequence and build formatted tuples
        sorted_steps = sorted(steps, key=lambda s: s.get("sequence", 0))
        result = []
        for step in sorted_steps:
            status = step.get("status", "pending")
            symbol = symbol_map.get(status, "○")
            style = style_map.get(status, "class:plan.pending")
            result.append((style, symbol))

        return result

    def _current_plan_has_data(self) -> bool:
        """Check if there's plan data for the current agent."""
        return self._get_current_plan_data() is not None

    def _budget_captures_keys(self) -> bool:
        """Check if budget panel should capture navigation keys.

        Budget panel captures keys only when visible AND input buffer is empty
        AND no permission request is pending. Permission requests have priority
        over budget panel navigation.
        """
        return (
            self._budget_panel.is_visible
            and not self._input_buffer.text.strip()
            and not self._waiting_for_channel_input
        )

    def _get_session_bar_content(self):
        """Get session bar content as formatted text."""
        if not self._session_id:
            return [("class:session-bar.dim", " No session")]

        # Build: Session: <id> - <description>  │  Workspace: <path>
        result = [
            ("class:session-bar.label", " Session: "),
            ("class:session-bar.id", self._session_id),
        ]

        if self._session_description:
            result.extend([
                ("class:session-bar.separator", " - "),
                ("class:session-bar.description", self._session_description),
            ])

        if self._session_workspace:
            result.extend([
                ("class:session-bar.separator", "  │  "),
                ("class:session-bar.label", "Workspace: "),
                ("class:session-bar.workspace", self._session_workspace),
            ])

        # Add tools expansion indicator (use active buffer, same as keybinding)
        tools_expanded = self._get_active_buffer().tools_expanded
        tools_indicator = "▼ expanded" if tools_expanded else "▶ collapsed"
        result.extend([
            ("class:session-bar.separator", "  │  "),
            ("class:session-bar.label", "Toolblocks: "),
            ("class:session-bar.value", tools_indicator),
            ("class:session-bar.dim", " [Ctrl+T]"),
        ])

        # Add permission status indicator
        result.extend([
            ("class:session-bar.separator", "  │  "),
            ("class:session-bar.label", "Permissions: "),
        ])

        if self._permission_suspension:
            # Suspended - show "allow" with scope in parentheses
            result.extend([
                ("class:session-bar.value", "allow"),
                ("class:session-bar.dim", f" ({self._permission_suspension})"),
            ])
        else:
            # Normal mode - show the effective default policy
            # Use different style classes for different modes
            policy = self._permission_default
            if policy == "allow":
                result.append(("class:session-bar.value", "allow"))
            elif policy == "deny":
                result.append(("class:session-bar.warning", "deny"))
            else:  # "ask"
                result.append(("class:session-bar.value", "ask"))

        result.append(("class:session-bar", " "))
        return result

    def _get_status_bar_content(self):
        """Get status bar content as formatted text."""
        # Check for temporary status message (e.g., copy feedback)
        if self._status_message and time.time() < self._status_message_expires:
            return [
                ("class:status-bar.label", " "),
                ("class:status-bar.value bold", self._status_message),
                ("class:status-bar", " "),
            ]
        elif self._status_message:
            # Message expired, clear it
            self._status_message = None

        # Check for persistent connection status (IPC reconnection)
        if self._connection_status:
            style_class = "class:status-bar.warning" if self._connection_status_style == "warning" else "class:status-bar.error"
            return [
                ("class:status-bar.label", " "),
                (style_class + " bold", self._connection_status),
                ("class:status-bar", " "),
            ]

        provider = self._model_provider or "—"
        model = self._model_name or "—"

        # Build context usage display (show percentage available)
        # Use selected agent's context and GC config if registry present
        if self._agent_registry:
            usage = self._agent_registry.get_selected_context_usage()
            gc_threshold, gc_strategy, gc_target_percent, gc_continuous_mode = self._agent_registry.get_selected_gc_config()
        else:
            usage = self._context_usage
            gc_threshold = self._gc_threshold
            gc_strategy = self._gc_strategy
            gc_target_percent = self._gc_target_percent
            gc_continuous_mode = self._gc_continuous_mode

        if usage:
            percent_used = usage.get('percent_used', 0)
            percent_available = 100 - percent_used
        else:
            percent_available = 100.0

        # Build context string with token count and optional GC hint
        if usage:
            total = usage.get('total_tokens', 0)
            # Format token count
            if total >= 1000:
                tokens_str = f"{total // 1000}K used"
            else:
                tokens_str = f"{total} used"

            # Add GC hint if configured
            if gc_continuous_mode and gc_target_percent is not None:
                # Continuous mode: show target instead of threshold
                target_available = 100 - gc_target_percent
                context_str = f"{percent_available:.0f}% available ({tokens_str}, continuous →{target_available:.0f}%)"
            elif gc_threshold is not None:
                gc_trigger_available = 100 - gc_threshold
                strategy = gc_strategy or "gc"
                context_str = f"{percent_available:.0f}% available ({tokens_str}, {strategy} at {gc_trigger_available:.0f}%)"
            else:
                context_str = f"{percent_available:.0f}% available ({tokens_str})"
        elif gc_continuous_mode and gc_target_percent is not None:
            # No usage yet but continuous GC is configured
            target_available = 100 - gc_target_percent
            context_str = f"{percent_available:.0f}% available (continuous →{target_available:.0f}%)"
        elif gc_threshold is not None:
            # No usage yet but threshold GC is configured
            gc_trigger_available = 100 - gc_threshold
            strategy = gc_strategy or "gc"
            context_str = f"{percent_available:.0f}% available ({strategy} at {gc_trigger_available:.0f}%)"
        else:
            context_str = "100% available"

        # Build formatted text with columns
        # Plan symbols | Provider | Model | Context
        result = []

        # Add plan symbols if there's an active plan (use selected agent's plan if registry present)
        plan_symbols = self._get_current_plan_symbols()
        if plan_symbols:
            result.append(("class:status-bar.label", " "))
            result.extend(plan_symbols)
            # Add hint to show popup (only when popup is not visible)
            if not self._plan_panel.is_popup_visible:
                result.append(("class:status-bar.label", " [Ctrl+P to expand]"))

        result.extend([
            ("class:status-bar.separator", "  │  " if result else " "),
            ("class:status-bar.label", "Provider: "),
            ("class:status-bar.value", provider),
            ("class:status-bar.separator", "  │  "),
            ("class:status-bar.label", "Model: "),
            ("class:status-bar.value", model),
            ("class:status-bar.separator", "  │  "),
            ("class:status-bar.label", "Context: "),
            ("class:status-bar.value", context_str),
        ])

        # Add budget hint when panel has data and is not visible
        if self._budget_panel.has_data() and not self._budget_panel.is_visible:
            budget_key = format_key_for_display(self._keybinding_config.toggle_budget)
            result.append(("class:status-bar.label", f" [{budget_key} for budget]"))

        result.append(("class:status-bar", " "))

        return result

    def _get_scroll_page_size(self) -> int:
        """Get the number of lines to scroll per page (half the visible height)."""
        # Ensure dimensions are current
        self._update_dimensions()
        input_height = self._get_input_height()
        available_height = self._height - 1 - input_height  # minus status bar and input area
        # Scroll by half the visible content area
        return max(3, (available_height - 4) // 2)

    def _get_budget_popup_content(self):
        """Get rendered budget popup content as ANSI for prompt_toolkit."""
        # Calculate popup dimensions based on terminal size
        popup_width = max(50, min(90, int(self._width * 0.7)))
        popup_height = max(10, min(20, int(self._height * 0.5)))

        rendered = self._budget_panel.render(popup_height, popup_width)
        return to_formatted_text(ANSI(self._renderer.render(rendered)))

    def _get_plan_popup_content(self):
        """Get rendered plan popup content as ANSI for prompt_toolkit."""
        # Calculate popup width (about 60% of terminal, min 40, max 80)
        popup_width = max(40, min(80, int(self._width * 0.6)))

        # Calculate max visible steps based on terminal height
        # Reserve space for: status bar (1), input area (3), popup borders (3), progress bar (2)
        available_height = self._height - 9
        # Each step takes at least 1 line, results/errors take 1 more
        # Aim for reasonable step visibility (at least 3, at most 15)
        max_steps = max(3, min(15, available_height // 2))
        self._plan_panel.set_popup_max_visible_steps(max_steps)

        # Use current agent's plan data (from registry if available)
        plan_data = self._get_current_plan_data()

        # Pass plan_data directly to avoid thread-unsafe state modification
        rendered = self._plan_panel.render_popup(width=popup_width, plan_data=plan_data)

        return to_formatted_text(ANSI(self._renderer.render(rendered)))

    def _get_active_buffer(self):
        """Get the active output buffer (selected agent's or main)."""
        if self._agent_registry:
            buffer = self._agent_registry.get_selected_buffer()
            if buffer:
                return buffer
        return self._output_buffer

    # ─────────────────────────────────────────────────────────────────────────
    # Search mode
    # ─────────────────────────────────────────────────────────────────────────

    def _open_search(self) -> None:
        """Open search mode."""
        self._search_mode = True
        self._search_query = ""
        # Clear any existing search state in output buffer
        buffer = self._get_active_buffer()
        buffer.clear_search()
        # Clear the input buffer for search query entry
        if self._input_buffer:
            self._input_buffer.text = ""

    def _close_search(self) -> None:
        """Close search mode and clear search highlights."""
        self._search_mode = False
        self._search_query = ""
        buffer = self._get_active_buffer()
        buffer.clear_search()
        # Clear the input buffer
        if self._input_buffer:
            self._input_buffer.text = ""
        # Sync display to remove highlights
        self._sync_output_display()

    def _update_search(self) -> None:
        """Update search results based on current input."""
        if not self._search_mode or not self._input_buffer:
            return

        query = self._input_buffer.text
        if query != self._search_query:
            self._search_query = query
            buffer = self._get_active_buffer()
            buffer.search(query)
            # Sync display to show highlighted content
            self._sync_output_display()
            # Invalidate to trigger redraw
            if self._app:
                self._app.invalidate()

    # ─────────────────────────────────────────────────────────────────────────
    # Paste handling
    # ─────────────────────────────────────────────────────────────────────────

    def _register_paste(self, content: str) -> int:
        """Register a large paste and return its ID.

        Args:
            content: The pasted content to store.

        Returns:
            Unique paste ID for the placeholder.
        """
        # Normalize line endings: \r\n -> \n, then \r -> \n
        # Terminal pastes often use \r (carriage return) instead of \n
        normalized = content.replace('\r\n', '\n').replace('\r', '\n')
        self._paste_counter += 1
        self._paste_registry[self._paste_counter] = normalized
        return self._paste_counter

    def _get_paste(self, paste_id: int) -> Optional[str]:
        """Get and remove a registered paste by ID.

        Args:
            paste_id: The paste ID from the placeholder.

        Returns:
            The stored paste content, or None if not found.
        """
        return self._paste_registry.pop(paste_id, None)

    def _expand_paste_placeholders(self, text: str) -> str:
        """Expand paste placeholders with their original content.

        Args:
            text: Input text that may contain paste placeholders.

        Returns:
            Text with placeholders replaced by original paste content.
        """
        pattern = r'\[paste #(\d+): \+\d+ lines\]'

        def replace(m: re.Match) -> str:
            paste_id = int(m.group(1))
            content = self._get_paste(paste_id)
            return content if content is not None else m.group(0)

        return re.sub(pattern, replace, text)

    def _is_large_paste(self, data: str) -> bool:
        """Check if pasted data exceeds thresholds.

        Args:
            data: The pasted text.

        Returns:
            True if paste should use placeholder.
        """
        line_count = len(data.splitlines()) if data else 0
        return line_count > self._paste_threshold_lines or len(data) > self._paste_threshold_chars

    def _get_output_content(self):
        """Get rendered output content as ANSI for prompt_toolkit."""
        # Check for terminal resize and update dimensions if needed
        self._update_dimensions()

        # Use pager temp buffer if pager is active, otherwise use main buffer
        if getattr(self, '_pager_active', False) and hasattr(self, '_pager_temp_buffer'):
            output_buffer = self._pager_temp_buffer
        elif self._agent_registry:
            output_buffer = self._agent_registry.get_selected_buffer()
            if not output_buffer:
                output_buffer = self._output_buffer
        else:
            output_buffer = self._output_buffer

        # NOTE: Do NOT flush here - render() uses _get_current_block_lines()
        # which reads streaming content without flushing, preserving chunk accumulation

        # Calculate available height for output (account for dynamic input height)
        input_height = self._get_input_height()
        # Subtract: session bar (1), status bar (1), tab bar (1 if present), input area
        tab_bar_height = 1 if self._agent_tab_bar else 0
        session_bar_height = 1  # Session bar is always visible
        available_height = self._height - session_bar_height - 1 - tab_bar_height - input_height

        # Output is now 100% width (tab bar is horizontal, not side panel)
        panel_width = self._width

        # Render output panel with correct width for word wrapping
        panel = output_buffer.render_panel(
            height=available_height,
            width=panel_width,
        )
        # Use full-width renderer - Panel's width parameter handles sizing
        return to_formatted_text(ANSI(self._renderer.render(panel)))

    def _sync_output_display(self):
        """Sync the output display buffer and fragments from ANSI content.

        Converts ANSI output to:
        1. Plain text in _output_pt_buffer (enables selection)
        2. Style fragments in _output_line_fragments (enables styling)
        """
        # Check for terminal resize
        self._update_dimensions()

        # Get ANSI content
        ansi_content = self._renderer.render(self._get_output_panel_content())

        # Convert to plain text and fragments
        plain_text, all_fragments = ansi_to_plain_and_fragments(ansi_content)

        # Split fragments by line for the processor
        self._output_line_fragments.clear()
        line_num = 0
        current_line_fragments = []
        char_pos = 0

        for style, text in all_fragments:
            # Split text by newlines
            parts = text.split('\n')
            for i, part in enumerate(parts):
                if part:
                    current_line_fragments.append((style, part))
                if i < len(parts) - 1:  # Newline encountered
                    self._output_line_fragments[line_num] = current_line_fragments
                    line_num += 1
                    current_line_fragments = []

        # Store last line if any content
        if current_line_fragments:
            self._output_line_fragments[line_num] = current_line_fragments

        # Update the buffer with plain text, preserving selection state and cursor position
        old_selection = self._output_pt_buffer.selection_state
        old_cursor = self._output_pt_buffer.cursor_position
        self._output_pt_buffer.set_document(
            Document(plain_text, len(plain_text)),
            bypass_readonly=True
        )
        # Restore selection and cursor position if selection was active (user might be selecting text)
        if old_selection:
            self._output_pt_buffer.selection_state = old_selection
            # Clamp cursor to new document length in case document got shorter
            self._output_pt_buffer.cursor_position = min(old_cursor, len(plain_text))

    def _get_output_panel_content(self):
        """Get the raw Rich panel for output (before ANSI rendering)."""
        # Use pager temp buffer if pager is active, otherwise use main buffer
        if getattr(self, '_pager_active', False) and hasattr(self, '_pager_temp_buffer'):
            output_buffer = self._pager_temp_buffer
        elif self._agent_registry:
            output_buffer = self._agent_registry.get_selected_buffer()
            if not output_buffer:
                output_buffer = self._output_buffer
        else:
            output_buffer = self._output_buffer

        # Calculate available height
        input_height = self._get_input_height()
        tab_bar_height = 1 if self._agent_tab_bar else 0
        session_bar_height = 1
        available_height = self._height - session_bar_height - 1 - tab_bar_height - input_height
        panel_width = self._width

        return output_buffer.render_panel(
            height=available_height,
            width=panel_width,
        )

    def _get_line_fragments(self, lineno: int) -> Optional[List[Tuple[str, str]]]:
        """Get style fragments for a specific line (for StyledOutputProcessor)."""
        return self._output_line_fragments.get(lineno)

    def _get_agent_tab_bar_content(self):
        """Get agent tab bar content as prompt_toolkit formatted text."""
        if not self._agent_tab_bar:
            return []

        # Check for popup auto-hide timeout
        if self._agent_tab_bar.check_popup_timeout():
            pass  # Popup was hidden, will render without it

        return self._agent_tab_bar.render()

    def _get_agent_popup_content(self):
        """Get agent details popup content as prompt_toolkit formatted text."""
        if not self._agent_tab_bar or not self._agent_tab_bar.is_popup_visible:
            return []

        # Get the selected agent
        if not self._agent_registry:
            return []

        agent = self._agent_registry.get_selected_agent()
        if not agent:
            return []

        return self._agent_tab_bar.render_popup(agent)

    def set_status_message(self, message: str, timeout: float = 2.0) -> None:
        """Set a temporary status bar message.

        Args:
            message: The message to display.
            timeout: How long to show the message (seconds).
        """
        self._status_message = message
        self._status_message_expires = time.time() + timeout
        if self._app:
            self._app.invalidate()

    def set_connection_status(
        self,
        message: Optional[str],
        style: str = "warning",
    ) -> None:
        """Set persistent connection status message.

        Unlike set_status_message, this persists until explicitly cleared.
        Used to show IPC reconnection status.

        Args:
            message: The message to display, or None to clear.
            style: Style to use ("warning" for reconnecting, "error" for failed).
        """
        self._connection_status = message
        self._connection_status_style = style
        if self._app:
            self._app.invalidate()

    def _yank_last_response(self) -> None:
        """Copy the last model response to clipboard (chrome-free)."""
        # Get the appropriate buffer (selected agent if registry present)
        if self._agent_registry:
            buffer = self._agent_registry.get_selected_buffer()
            if not buffer:
                buffer = self._output_buffer
        else:
            buffer = self._output_buffer

        # Extract chrome-free text using configured sources
        text = buffer.get_last_response_text(sources=self._clipboard_config.sources)

        if text:
            self._clipboard.copy(text)
            # Count lines for feedback
            line_count = text.count('\n') + 1
            self.set_status_message(f"Yanked {line_count} line{'s' if line_count != 1 else ''}")
        else:
            self.set_status_message("Nothing to yank")

    def _yank_selection(self, start_y: int, end_y: int) -> None:
        """Copy selected screen region to clipboard (chrome-free).

        Args:
            start_y: Start Y coordinate (screen row).
            end_y: End Y coordinate (screen row).
        """
        # Get the appropriate buffer
        if self._agent_registry:
            buffer = self._agent_registry.get_selected_buffer()
            if not buffer:
                buffer = self._output_buffer
        else:
            buffer = self._output_buffer

        # Calculate visible area height (account for status bar and input)
        input_height = self._get_input_height()
        visible_height = self._height - 1 - input_height  # minus status bar and input

        # Get visible line range in buffer coordinates
        visible_start, visible_end = buffer.get_visible_line_range(visible_height)

        # Map screen Y coordinates to buffer line indices
        # Screen Y is relative to output panel (after status bar)
        # Ensure start <= end
        if start_y > end_y:
            start_y, end_y = end_y, start_y

        # Convert screen coordinates to buffer coordinates
        buffer_start = visible_start + start_y
        buffer_end = visible_start + end_y

        # Extract chrome-free text
        text = buffer.get_text_in_line_range(
            buffer_start, buffer_end,
            sources=self._clipboard_config.sources
        )

        if text:
            self._clipboard.copy(text)
            line_count = text.count('\n') + 1
            self.set_status_message(f"Yanked {line_count} line{'s' if line_count != 1 else ''}")
        else:
            self.set_status_message("Nothing to yank (no matching content)")

    def _build_app(self) -> None:
        """Build the prompt_toolkit application."""
        # Key bindings - use configuration for customizable keys
        kb = KeyBindings()
        keys = self._keybinding_config

        # Condition for disabling keybindings when in search mode
        # Search mode is modal - most keys should be captured by search handlers
        not_in_search_mode = Condition(lambda: not getattr(self, '_search_mode', False))

        @kb.add(*keys.get_key_args("submit"), eager=True,
                filter=Condition(lambda: not self._budget_captures_keys()) & not_in_search_mode)
        def handle_enter(event):
            """Handle enter key - submit input, select permission option, or advance pager."""
            if getattr(self, '_pager_active', False):
                # In pager mode - advance page
                self._advance_pager_page()
                return
            # Check for permission mode - if input is empty, select focused option
            raw_text = self._input_buffer.text
            text = raw_text.strip()
            if (not text and
                getattr(self, '_waiting_for_channel_input', False) and
                self._permission_response_options):
                self._select_focused_permission_option()
                return
            # Normal mode - submit input
            # Expand paste placeholders BEFORE stripping to preserve line feeds in pasted content
            expanded_text = self._expand_paste_placeholders(raw_text)
            # Strip only leading/trailing whitespace from the final result
            expanded_text = expanded_text.strip()
            # Add to history before reset (like PromptSession does)
            if expanded_text and self._input_buffer.history:
                self._input_buffer.history.append_string(expanded_text)
            self._input_buffer.reset()
            if self._input_callback:
                self._input_callback(expanded_text)

        @kb.add(*keys.get_key_args("newline"), filter=not_in_search_mode)
        def handle_alt_enter(event):
            """Handle Alt+Enter / Escape+Enter - insert newline for multi-line input."""
            if not getattr(self, '_pager_active', False):
                event.current_buffer.insert_text('\n')

        @kb.add(*keys.get_key_args("clear_input"), filter=not_in_search_mode)
        def handle_escape_escape(event):
            """Handle Escape+Escape - clear input buffer contents."""
            if not getattr(self, '_pager_active', False):
                event.current_buffer.reset()

        @kb.add(*keys.get_key_args("tool_exit"), filter=not_in_search_mode)
        def handle_tool_exit(event):
            """Handle Escape - exit tool navigation mode if active."""
            buffer = self._get_active_buffer()
            if buffer.tool_nav_active:
                buffer.exit_tool_navigation()
                self._app.invalidate()
                # Don't let escape propagate further
                return
            # Not in tool nav mode - let other handlers process escape
            # (escape+enter for newline, escape+escape for clear)

        @kb.add(*keys.get_key_args("pager_quit"), eager=True, filter=not_in_search_mode)
        def handle_q(event):
            """Handle 'q' key - quit pager if active, otherwise type 'q'."""
            if getattr(self, '_pager_active', False):
                # In pager mode - quit pager directly
                self._stop_pager()
                return  # Don't insert 'q'
            else:
                # Normal mode - insert 'q' character
                event.current_buffer.insert_text("q")

        @kb.add(*keys.get_key_args("view_full"), filter=not_in_search_mode)
        def handle_v(event):
            """Handle 'v' key - view full prompt if truncated, otherwise type 'v'."""
            # Check if waiting for channel input with TRUNCATED pending prompt
            if getattr(self, '_waiting_for_channel_input', False):
                # Only zoom if the prompt is actually truncated
                # Use get_selected_buffer() for IPC compatibility (agent ID may not be "main")
                if self._agent_registry:
                    buffer = self._agent_registry.get_selected_buffer()
                    if buffer and buffer.has_truncated_pending_prompt():
                        # Trigger zoom via callback
                        if self._input_callback:
                            self._input_callback("v")
                        return
            # Normal mode - insert 'v' character
            event.current_buffer.insert_text("v")

        @kb.add(*keys.get_key_args("pager_next"), eager=True, filter=not_in_search_mode)
        def handle_space(event):
            """Handle space key - pager advance, tool expand toggle, permission select, or insert space."""
            if getattr(self, '_pager_active', False):
                # In pager mode - advance page
                self._advance_pager_page()
                return
            # Check for tool navigation mode
            buffer = self._agent_registry.get_selected_buffer() if self._agent_registry else self._output_buffer
            if buffer and buffer.tool_nav_active:
                # In tool navigation mode - toggle expand/collapse
                buffer.toggle_selected_tool_expanded()
                self._app.invalidate()
                return
            # Check for permission mode - select focused option
            if getattr(self, '_waiting_for_channel_input', False) and self._permission_response_options:
                self._select_focused_permission_option()
                return
            # Normal mode - insert space character
            event.current_buffer.insert_text(" ")

        @kb.add(*keys.get_key_args("permission_next"), eager=True,
                filter=Condition(lambda: not self._budget_captures_keys()) & not_in_search_mode)
        def handle_permission_next(event):
            """Handle TAB - cycle to next permission option, or complete in normal mode."""
            if getattr(self, '_waiting_for_channel_input', False) and self._permission_response_options:
                # Permission mode: cycle to next option (wrap around)
                self._permission_focus_index = (self._permission_focus_index + 1) % len(self._permission_response_options)
                # Update output buffer for inline highlighting (use correct buffer from agent registry)
                buffer = self._agent_registry.get_selected_buffer() if self._agent_registry else self._output_buffer
                buffer.set_permission_focus(
                    self._permission_response_options,
                    self._permission_focus_index
                )
                self._app.invalidate()
            else:
                # Normal mode: trigger tab completion
                buff = event.app.current_buffer
                if buff.complete_state:
                    buff.complete_next()
                else:
                    buff.start_completion()

        @kb.add(*keys.get_key_args("permission_prev"), eager=True,
                filter=Condition(lambda: not self._budget_captures_keys()) & not_in_search_mode)
        def handle_permission_prev(event):
            """Handle Shift+TAB - cycle to previous permission option, or complete prev in normal mode."""
            if getattr(self, '_waiting_for_channel_input', False) and self._permission_response_options:
                # Permission mode: cycle to previous option (wrap around)
                self._permission_focus_index = (self._permission_focus_index - 1) % len(self._permission_response_options)
                # Update output buffer for inline highlighting (use correct buffer from agent registry)
                buffer = self._agent_registry.get_selected_buffer() if self._agent_registry else self._output_buffer
                buffer.set_permission_focus(
                    self._permission_response_options,
                    self._permission_focus_index
                )
                self._app.invalidate()
            else:
                # Normal mode: trigger previous completion
                buff = event.app.current_buffer
                if buff.complete_state:
                    buff.complete_previous()

        @kb.add(*keys.get_key_args("cancel"))
        def handle_ctrl_c(event):
            """Handle Ctrl-C - stop if running, exit if not."""
            # If model is running, stop it instead of exiting
            if self._is_running_callback and self._is_running_callback():
                if self._stop_callback:
                    self._stop_callback()
                    # Flush pending streaming content (cancellation message comes from session)
                    self._output_buffer._flush_current_block()
                    self._app.invalidate()
                return
            # Otherwise exit the application
            event.app.exit(exception=KeyboardInterrupt())

        @kb.add(*keys.get_key_args("exit"))
        def handle_ctrl_d(event):
            """Handle Ctrl-D - EOF."""
            event.app.exit(exception=EOFError())

        @kb.add(*keys.get_key_args("scroll_up"))
        def handle_page_up(event):
            """Handle Page-Up - scroll output up."""
            # Use selected agent's buffer if registry present
            if self._agent_registry:
                buffer = self._agent_registry.get_selected_buffer()
                if buffer:
                    buffer.scroll_up(lines=self._get_scroll_page_size())
                else:
                    self._output_buffer.scroll_up(lines=self._get_scroll_page_size())
            else:
                self._output_buffer.scroll_up(lines=self._get_scroll_page_size())
            self._sync_output_display()
            self._app.invalidate()

        @kb.add(*keys.get_key_args("scroll_down"))
        def handle_page_down(event):
            """Handle Page-Down - scroll output down."""
            # Use selected agent's buffer if registry present
            if self._agent_registry:
                buffer = self._agent_registry.get_selected_buffer()
                if buffer:
                    buffer.scroll_down(lines=self._get_scroll_page_size())
                else:
                    self._output_buffer.scroll_down(lines=self._get_scroll_page_size())
            else:
                self._output_buffer.scroll_down(lines=self._get_scroll_page_size())
            self._sync_output_display()
            self._app.invalidate()

        @kb.add(*keys.get_key_args("scroll_top"))
        def handle_home(event):
            """Handle Home - scroll to top of output."""
            # Use selected agent's buffer if registry present
            if self._agent_registry:
                buffer = self._agent_registry.get_selected_buffer()
                if buffer:
                    buffer.scroll_to_top()
                else:
                    self._output_buffer.scroll_to_top()
            else:
                self._output_buffer.scroll_to_top()
            self._sync_output_display()
            self._app.invalidate()

        @kb.add(*keys.get_key_args("scroll_bottom"))
        def handle_end(event):
            """Handle End - scroll to bottom of output."""
            # Use selected agent's buffer if registry present
            if self._agent_registry:
                buffer = self._agent_registry.get_selected_buffer()
                if buffer:
                    buffer.scroll_to_bottom()
                else:
                    self._output_buffer.scroll_to_bottom()
            else:
                self._output_buffer.scroll_to_bottom()
            self._sync_output_display()
            self._app.invalidate()

        # Mouse scroll handlers - bind mouse wheel to page up/down
        @kb.add(*keys.get_key_args("mouse_scroll_up"), eager=True)
        def handle_mouse_scroll_up(event):
            """Handle mouse scroll up - scroll output up."""
            # Use selected agent's buffer if registry present
            if self._agent_registry:
                buffer = self._agent_registry.get_selected_buffer()
                if buffer:
                    buffer.scroll_up(lines=self._get_scroll_page_size())
                else:
                    self._output_buffer.scroll_up(lines=self._get_scroll_page_size())
            else:
                self._output_buffer.scroll_up(lines=self._get_scroll_page_size())
            self._sync_output_display()
            self._app.invalidate()

        @kb.add(*keys.get_key_args("mouse_scroll_down"), eager=True)
        def handle_mouse_scroll_down(event):
            """Handle mouse scroll down - scroll output down."""
            # Use selected agent's buffer if registry present
            if self._agent_registry:
                buffer = self._agent_registry.get_selected_buffer()
                if buffer:
                    buffer.scroll_down(lines=self._get_scroll_page_size())
                else:
                    self._output_buffer.scroll_down(lines=self._get_scroll_page_size())
            else:
                self._output_buffer.scroll_down(lines=self._get_scroll_page_size())
            self._sync_output_display()
            self._app.invalidate()

        @kb.add(*keys.get_key_args("nav_up"), eager=True, filter=not_in_search_mode)
        def handle_up(event):
            """Handle Up arrow - scroll popup, tool nav, or history/completion."""
            # Budget panel takes priority when visible AND input is empty
            if self._budget_captures_keys():
                self._budget_panel.scroll_up()
                self._app.invalidate()
                return
            # Plan popup takes priority when visible (it's an overlay)
            if self._plan_panel.is_popup_visible and self._current_plan_has_data():
                plan_data = self._get_current_plan_data()
                if self._plan_panel.scroll_popup_up(plan_data):
                    self._app.invalidate()
                return
            # Tool navigation mode
            buffer = self._get_active_buffer()
            if buffer.tool_nav_active:
                # If selected tool is expanded and has output, try scrolling
                selected_tool = buffer.get_selected_tool()
                if selected_tool and selected_tool.expanded:
                    if buffer.scroll_selected_tool_up():
                        self._app.invalidate()
                        return
                # Navigate to previous tool (either not expanded, no output, or at scroll boundary)
                buffer.select_prev_tool()
                self._app.invalidate()
                return
            # Normal mode - history/completion navigation
            event.current_buffer.auto_up()

        @kb.add(*keys.get_key_args("nav_down"), eager=True, filter=not_in_search_mode)
        def handle_down(event):
            """Handle Down arrow - scroll popup, tool nav, or history/completion."""
            # Budget panel takes priority when visible AND input is empty
            if self._budget_captures_keys():
                self._budget_panel.scroll_down()
                self._app.invalidate()
                return
            # Plan popup takes priority when visible (it's an overlay)
            if self._plan_panel.is_popup_visible and self._current_plan_has_data():
                plan_data = self._get_current_plan_data()
                if self._plan_panel.scroll_popup_down(plan_data):
                    self._app.invalidate()
                return
            # Tool navigation mode
            buffer = self._get_active_buffer()
            if buffer.tool_nav_active:
                # If selected tool is expanded and has output, try scrolling
                selected_tool = buffer.get_selected_tool()
                if selected_tool and selected_tool.expanded:
                    if buffer.scroll_selected_tool_down():
                        self._app.invalidate()
                        return
                # Navigate to next tool (either not expanded, no output, or at scroll boundary)
                buffer.select_next_tool()
                self._app.invalidate()
                return
            # Normal mode - history/completion navigation
            event.current_buffer.auto_down()

        @kb.add(*keys.get_key_args("toggle_plan"), filter=not_in_search_mode)
        def handle_ctrl_p(event):
            """Handle Ctrl+P - toggle plan popup visibility."""
            # Check if current agent has a plan (registry or fallback)
            if self._current_plan_has_data():
                self._plan_panel.toggle_popup()
                self._app.invalidate()

        @kb.add(*keys.get_key_args("toggle_budget"), filter=not_in_search_mode)
        def handle_ctrl_b(event):
            """Handle Ctrl+B - toggle budget panel visibility."""
            self._budget_panel.toggle()
            self._app.invalidate()

        # Budget panel navigation (only active when budget panel is visible and not in search mode)
        @kb.add("tab", filter=Condition(lambda: self._budget_captures_keys()) & not_in_search_mode)
        def handle_budget_tab(event):
            """Handle TAB in budget panel - cycle to next agent."""
            self._budget_panel.cycle_agent(forward=True)
            self._app.invalidate()

        @kb.add("s-tab", filter=Condition(lambda: self._budget_captures_keys()) & not_in_search_mode)
        def handle_budget_shift_tab(event):
            """Handle Shift+TAB in budget panel - cycle to previous agent."""
            self._budget_panel.cycle_agent(forward=False)
            self._app.invalidate()

        @kb.add("escape", "left", filter=Condition(lambda: self._budget_captures_keys()) & not_in_search_mode)
        def handle_budget_escape(event):
            """Handle ESC+left in budget panel - drill up or close."""
            if not self._budget_panel.drill_up():
                # Already at top level, close the panel
                self._budget_panel.hide()
            self._app.invalidate()

        @kb.add("enter", filter=Condition(lambda: self._budget_captures_keys()) & not_in_search_mode)
        def handle_budget_enter(event):
            """Handle Enter in budget panel - drill down into selected source."""
            self._budget_panel.drill_down()
            self._app.invalidate()

        @kb.add(*keys.get_key_args("cycle_agents"), filter=not_in_search_mode)
        def handle_f2(event):
            """Handle cycle_agents keybinding - cycle through agents."""
            if self._agent_registry:
                self._agent_registry.cycle_selection()
                # Sync output display to show new agent's buffer content
                self._sync_output_display()
                # Show the agent details popup briefly
                if self._agent_tab_bar:
                    self._agent_tab_bar.show_popup()
                self._app.invalidate()

        @kb.add(*keys.get_key_args("toggle_tools"), filter=not_in_search_mode)
        def handle_ctrl_t(event):
            """Handle Ctrl+T - toggle tool view between collapsed/expanded."""
            buffer = self._get_active_buffer()
            buffer.toggle_tools_expanded()
            self._app.invalidate()

        @kb.add(*keys.get_key_args("open_editor"), filter=not_in_search_mode)
        def handle_open_editor(event):
            """Handle Ctrl+G - open current input in external editor ($EDITOR)."""
            # Don't open editor in pager mode or when buffer is empty
            if getattr(self, '_pager_active', False):
                return
            # Open the buffer in external editor
            # This uses $EDITOR or $VISUAL, falling back to vi
            event.current_buffer.open_in_editor()

        # Search mode keybindings
        @kb.add(*keys.get_key_args("search"))
        def handle_search(event):
            """Handle Ctrl+F - toggle search mode."""
            if getattr(self, '_pager_active', False):
                return
            if self._search_mode:
                # Close search
                self._close_search()
            else:
                # Open search
                self._open_search()
            self._app.invalidate()

        @kb.add(*keys.get_key_args("search_next"),
                filter=Condition(lambda: getattr(self, '_search_mode', False)))
        def handle_search_next(event):
            """Handle Enter in search mode - go to next match."""
            buffer = self._get_active_buffer()
            buffer.search_next()
            # Sync display to update current match highlighting
            self._sync_output_display()
            self._app.invalidate()

        @kb.add(*keys.get_key_args("search_prev"),
                filter=Condition(lambda: getattr(self, '_search_mode', False)))
        def handle_search_prev(event):
            """Handle Ctrl+P in search mode - go to previous match."""
            buffer = self._get_active_buffer()
            buffer.search_prev()
            # Sync display to update current match highlighting
            self._sync_output_display()
            self._app.invalidate()

        @kb.add(*keys.get_key_args("search_close"),
                filter=Condition(lambda: getattr(self, '_search_mode', False)))
        def handle_search_close(event):
            """Handle Escape in search mode - close search."""
            self._close_search()
            self._app.invalidate()

        @kb.add(*keys.get_key_args("tool_nav_enter"), eager=True, filter=not_in_search_mode)
        def handle_tool_nav_enter(event):
            """Handle Ctrl+N - enter/exit tool navigation mode."""
            buffer = self._get_active_buffer()
            if buffer.tool_nav_active:
                # Already in nav mode - exit
                buffer.exit_tool_navigation()
            else:
                # Try to enter navigation mode (will check for tools)
                buffer.enter_tool_navigation()
            self._app.invalidate()

        @kb.add(*keys.get_key_args("tool_expand"), eager=True, filter=not_in_search_mode)
        def handle_tool_expand(event):
            """Handle right arrow - expand selected tool's output or move cursor."""
            buffer = self._get_active_buffer()
            if buffer.tool_nav_active:
                buffer.expand_selected_tool()
                self._app.invalidate()
            else:
                # Move cursor right by one character, wrapping to next line if needed
                buf = event.current_buffer
                if buf.cursor_position < len(buf.text):
                    buf.cursor_position += 1

        @kb.add(*keys.get_key_args("tool_collapse"), eager=True, filter=not_in_search_mode)
        def handle_tool_collapse(event):
            """Handle left arrow - collapse selected tool's output or move cursor."""
            buffer = self._get_active_buffer()
            if buffer.tool_nav_active:
                buffer.collapse_selected_tool()
                self._app.invalidate()
            else:
                # Move cursor left by one character, wrapping to previous line if needed
                buf = event.current_buffer
                if buf.cursor_position > 0:
                    buf.cursor_position -= 1

        @kb.add(*keys.get_key_args("yank"), filter=not_in_search_mode)
        def handle_ctrl_y(event):
            """Handle Ctrl+Y - yank (copy) last response to clipboard."""
            self._yank_last_response()

        # Bracketed paste handler - detects large pastes and uses placeholders
        @kb.add(Keys.BracketedPaste)
        def handle_bracketed_paste(event):
            """Handle bracketed paste - use placeholder for large pastes."""
            data = event.data
            if self._is_large_paste(data):
                # Register the paste and insert placeholder
                line_count = len(data.splitlines()) if data else 0
                paste_id = self._register_paste(data)
                placeholder = f"[paste #{paste_id}: +{line_count} lines]"
                event.current_buffer.insert_text(placeholder)
            else:
                # Normal paste for small content
                event.current_buffer.insert_text(data)

        # Agent tab bar at top (conditional - only if agent_registry present)
        agent_tab_bar = ConditionalContainer(
            Window(
                FormattedTextControl(self._get_agent_tab_bar_content),
                height=1,
                style="class:agent-tab-bar",
            ),
            filter=Condition(lambda: self._agent_tab_bar is not None),
        )

        # Session bar (shows current session and workspace)
        session_bar = Window(
            FormattedTextControl(self._get_session_bar_content),
            height=1,
            style="class:session-bar",
        )

        # Status bar (always visible, 1 line)
        status_bar = Window(
            FormattedTextControl(self._get_status_bar_content),
            height=1,
            style="class:status-bar",
        )

        # Dynamic height for output panel - accounts for all other components
        def get_output_height():
            """Calculate output panel height by subtracting all other components."""
            total = self._height
            fixed = 2  # session_bar (1) + status_bar (1)
            if self._agent_tab_bar is not None:
                fixed += 1  # agent tab bar
            pending = self._get_pending_prompts_height()
            input_h = self._get_input_height()
            return max(1, total - fixed - pending - input_h)

        # Mouse scroll callbacks for output panel
        # Use smaller scroll amount (3 lines) for smoother mouse scrolling
        mouse_scroll_lines = 3

        def on_mouse_scroll_up():
            """Handle mouse scroll up in output panel."""
            if self._agent_registry:
                buffer = self._agent_registry.get_selected_buffer()
                if buffer:
                    buffer.scroll_up(lines=mouse_scroll_lines)
                else:
                    self._output_buffer.scroll_up(lines=mouse_scroll_lines)
            else:
                self._output_buffer.scroll_up(lines=mouse_scroll_lines)
            # Re-sync display to reflect new scroll position, then invalidate
            self._sync_output_display()
            if self._app:
                self._app.invalidate()

        def on_mouse_scroll_down():
            """Handle mouse scroll down in output panel."""
            if self._agent_registry:
                buffer = self._agent_registry.get_selected_buffer()
                if buffer:
                    buffer.scroll_down(lines=mouse_scroll_lines)
                else:
                    self._output_buffer.scroll_down(lines=mouse_scroll_lines)
            else:
                self._output_buffer.scroll_down(lines=mouse_scroll_lines)
            # Re-sync display to reflect new scroll position, then invalidate
            self._sync_output_display()
            if self._app:
                self._app.invalidate()

        # Output panel (fills remaining space minus pending prompts)
        # Uses BufferControl for mouse selection support, with StyledOutputProcessor for ANSI styling
        styled_processor = StyledOutputProcessor(self._get_line_fragments, buffer=self._output_pt_buffer)
        def on_selection_complete(text: str) -> None:
            """Copy selected text to clipboard."""
            success = self._clipboard.copy(text)
            # Show status with provider info for debugging
            char_count = len(text)
            provider_name = getattr(self._clipboard, 'name', 'unknown')
            if success:
                self.set_status_message(f"Copied {char_count} chars via {provider_name}")
            else:
                # Try to get error details
                error = ""
                if hasattr(self._clipboard, '_native'):
                    error = getattr(self._clipboard._native, '_last_error', '') or ''
                elif hasattr(self._clipboard, '_last_error'):
                    error = getattr(self._clipboard, '_last_error', '') or ''
                if error:
                    self.set_status_message(f"Copy failed: {error[:50]}")
                else:
                    self.set_status_message(f"Copy failed ({provider_name})")

        self._output_control = ScrollableBufferControl(
            buffer=self._output_pt_buffer,
            input_processors=[styled_processor],
            focusable=True,
            on_scroll_up=on_mouse_scroll_up,
            on_scroll_down=on_mouse_scroll_down,
            input_buffer=self._input_buffer,  # For returning focus after selection
            on_selection_complete=on_selection_complete,
        )
        output_window = Window(
            self._output_control,
            height=get_output_height,
            wrap_lines=False,
            style="class:output-panel",
        )

        # Input prompt label - changes based on mode (pager, waiting for channel, search, normal)
        def get_prompt_text():
            # Custom prompt override (for exit confirmation, etc.)
            if getattr(self, '_custom_prompt', None):
                return [("class:prompt.permission", self._custom_prompt)]
            if getattr(self, '_pager_active', False):
                return [("class:prompt.pager", "── Enter: next, q: quit ──")]
            if getattr(self, '_search_mode', False):
                # Show search status with match count and keybinding hints
                buffer = self._get_active_buffer()
                query, current, total = buffer.get_search_status()
                if total > 0:
                    return [
                        ("class:prompt.search", f"Search ({current}/{total}) "),
                        ("class:prompt", "[Enter: next, Ctrl+P: prev, Esc: close]> "),
                    ]
                elif query:
                    return [
                        ("class:prompt.search", "Search (no matches) "),
                        ("class:prompt", "[Esc: close]> "),
                    ]
                else:
                    return [
                        ("class:prompt.search", "Search "),
                        ("class:prompt", "[Enter: next, Esc: close]> "),
                    ]
            if getattr(self, '_waiting_for_channel_input', False):
                # 'v' hint is already shown in the output panel truncation indicator
                return [("class:prompt.permission", "Answer> ")]
            # Show "User>" for main agent, "Parent>" for subagent
            if self._agent_registry:
                agent = self._agent_registry.get_selected_agent()
                if agent and agent.agent_type == "subagent":
                    return [("class:prompt", "Parent> ")]
            return [("class:prompt", "User> ")]

        prompt_label = Window(
            FormattedTextControl(get_prompt_text),
            height=self._get_input_height,
            dont_extend_width=True,
            style="class:output-panel",
        )

        # Input text area - hidden during pager mode (expandable height with word wrap)
        input_window = ConditionalContainer(
            Window(
                BufferControl(buffer=self._input_buffer),
                height=self._get_input_height,
                wrap_lines=True,
                style="class:output-panel",
            ),
            filter=Condition(lambda: not getattr(self, '_pager_active', False)),
        )

        # Input row (label + optional input area)
        input_row = VSplit([prompt_label, input_window])

        # Pending prompts bar (shown above input when prompts are queued)
        pending_prompts_bar = ConditionalContainer(
            Window(
                FormattedTextControl(self._get_pending_prompts_content),
                height=self._get_pending_prompts_height,
                style="class:pending-prompts-bar",
            ),
            filter=Condition(lambda: len(self._pending_prompts) > 0),
        )

        # Plan popup (floating overlay, shown with Ctrl+P)
        def get_popup_height():
            """Calculate popup height by rendering content and counting lines."""
            plan_data = self._get_current_plan_data()
            if not plan_data:
                return 6

            # Render the popup to get actual line count
            # Pass plan_data directly to avoid thread-unsafe state modification
            popup_width = max(40, min(80, int(self._width * 0.6)))
            rendered = self._plan_panel.render_popup(width=popup_width, plan_data=plan_data)

            # Render to string and count lines
            rendered_str = self._renderer.render(rendered)
            line_count = rendered_str.count('\n') + 1

            # Cap at available screen height
            return min(line_count, self._height - 4)

        plan_popup_window = ConditionalContainer(
            Window(
                FormattedTextControl(self._get_plan_popup_content),
                height=get_popup_height,
            ),
            filter=Condition(lambda: self._plan_panel.is_popup_visible and self._current_plan_has_data()),
        )

        # Budget popup (floating overlay, toggled with Ctrl+B)
        def get_budget_popup_height():
            """Calculate budget popup height by rendering content and counting lines."""
            if not self._budget_panel.has_data():
                return 4  # Minimal height for "no data" message

            # Render the popup to get actual line count
            popup_width = max(50, min(90, int(self._width * 0.7)))
            popup_height = max(10, min(20, int(self._height * 0.5)))
            rendered = self._budget_panel.render(popup_height, popup_width)

            # Render to string and count lines
            rendered_str = self._renderer.render(rendered)
            line_count = rendered_str.count('\n') + 1

            # Cap at available screen height
            return min(line_count, self._height - 4)

        budget_popup_window = ConditionalContainer(
            Window(
                FormattedTextControl(self._get_budget_popup_content),
                height=get_budget_popup_height,
            ),
            filter=Condition(lambda: self._budget_panel.is_visible),
        )

        # Agent details popup (floating overlay, shown on agent cycle)
        def get_agent_popup_height():
            """Calculate popup height based on content."""
            content = self._get_agent_popup_content()
            if not content:
                return 1
            # Count newlines in content
            line_count = sum(1 for _, text in content if '\n' in text)
            return max(8, line_count + 2)  # At least 8 lines

        agent_popup_window = ConditionalContainer(
            Window(
                FormattedTextControl(self._get_agent_popup_content),
                height=get_agent_popup_height,
            ),
            filter=Condition(lambda: self._agent_tab_bar is not None and self._agent_tab_bar.is_popup_visible),
        )

        # Root layout with session bar at top, then tab bar, status bar, output, bars, input
        from prompt_toolkit.layout.containers import FloatContainer, Float
        root = FloatContainer(
            content=HSplit([
                session_bar,             # Session info bar (top)
                agent_tab_bar,           # Agent tabs (conditional)
                status_bar,              # Status bar
                output_window,           # Output panel (fills remaining space)
                pending_prompts_bar,     # Queued prompts (dynamic, above input)
                input_row,               # Input area
            ]),
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=8),
                ),
                Float(
                    top=3,  # Same position as plan popup (they toggle separately)
                    right=2,  # Positioned on the right side
                    content=budget_popup_window,
                    z_index=50,
                ),
                Float(
                    top=2,  # Below session bar, aligned with selected tab
                    left=1,
                    content=agent_popup_window,
                    z_index=50,
                ),
                # Plan popup LAST so it renders on top of everything
                Float(
                    top=3,  # Below session bar, tab bar, and status bar
                    left=2,
                    content=plan_popup_window,
                    z_index=100,  # Highest priority
                ),
            ],
        )

        layout = Layout(root, focused_element=input_window)

        # Get style from theme and merge with input handler styles
        from prompt_toolkit.styles import merge_styles

        # Theme-based styles for all UI components
        default_style = self._theme.get_prompt_toolkit_style()

        input_style = self._input_handler._pt_style if self._input_handler else None
        if input_style:
            # Theme styles take precedence over input handler defaults
            style = merge_styles([input_style, default_style])
        else:
            style = default_style

        self._app = Application(
            layout=layout,
            key_bindings=kb,
            full_screen=True,
            mouse_support=True,
            style=style,
        )

    def refresh(self) -> None:
        """Refresh the display.

        Syncs the output buffer and invalidates the prompt_toolkit app to trigger re-render.
        The output sync updates the plain text buffer and style fragments for the
        BufferControl-based output panel.

        NOTE: We only call invalidate() - do NOT call renderer.render() directly
        as this may be called from background threads and would cause race
        conditions with the main event loop's rendering.
        """
        if self._app and self._app.is_running:
            # Sync output buffer and fragments for styled BufferControl
            self._sync_output_display()
            # Invalidate schedules a redraw in the main event loop
            self._app.invalidate()

    def _schedule_refresh(self) -> None:
        """Schedule a debounced refresh for streaming performance.

        Batches rapid refresh requests (e.g., during streaming) into a single
        refresh every _refresh_interval seconds. This reduces rendering overhead
        from 50+ refreshes/sec to ~20 refreshes/sec during streaming.
        """
        if self._refresh_pending:
            # Already scheduled, will happen soon
            return

        if not self._app or not self._app.is_running:
            return

        self._refresh_pending = True

        def _do_refresh() -> None:
            self._refresh_pending = False
            self.refresh()

        # Schedule refresh after debounce interval
        self._app.loop.call_later(self._refresh_interval, _do_refresh)

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

    async def run_input_loop_async(
        self,
        on_input: Callable[[str], None],
        initial_prompt: Optional[str] = None
    ) -> None:
        """Run the input loop asynchronously.

        This is an async version of run_input_loop that can be used with asyncio.
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

        await self._app.run_async()

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

    def set_session_info(
        self,
        session_id: str,
        description: str = "",
        workspace: str = ""
    ) -> None:
        """Set the session info for the session bar.

        Args:
            session_id: The session ID.
            description: Optional session description.
            workspace: Optional workspace path.
        """
        self._session_id = session_id
        self._session_description = description
        # Shorten home directory to ~
        if workspace:
            import os
            home = os.path.expanduser("~")
            if workspace.startswith(home):
                workspace = "~" + workspace[len(home):]
        self._session_workspace = workspace
        self.refresh()

    def set_permission_status(
        self,
        effective_default: str,
        suspension_scope: Optional[str] = None
    ) -> None:
        """Set the permission status for the session bar.

        Args:
            effective_default: The effective default policy ("allow", "deny", or "ask").
            suspension_scope: Optional suspension scope ("turn", "idle", "session", or None).
        """
        self._permission_default = effective_default
        self._permission_suspension = suspension_scope
        self.refresh()

    def set_stop_callbacks(
        self,
        stop_callback: Callable[[], bool],
        is_running_callback: Callable[[], bool]
    ) -> None:
        """Set callbacks for model stop functionality.

        These callbacks enable Ctrl-C to stop model generation when running.

        Args:
            stop_callback: Called to request stop. Returns True if stop was requested.
            is_running_callback: Called to check if model is running.
        """
        self._stop_callback = stop_callback
        self._is_running_callback = is_running_callback

    def set_prompt(self, prompt: Optional[str]) -> None:
        """Set a custom prompt override.

        Args:
            prompt: Custom prompt text, or None to restore default prompt.
        """
        self._custom_prompt = prompt
        self.refresh()

    def update_context_usage(self, usage: Dict[str, Any]) -> None:
        """Update context usage display in status bar.

        Args:
            usage: Dict with 'total_tokens', 'prompt_tokens', 'output_tokens', etc.
        """
        self._context_usage = usage
        self.refresh()

    def set_gc_threshold(
        self,
        threshold: Optional[float],
        strategy: Optional[str] = None,
        target_percent: Optional[float] = None,
        continuous_mode: bool = False
    ) -> None:
        """Set the GC threshold percentage and strategy for status bar display.

        Args:
            threshold: GC trigger threshold percentage (e.g., 80.0), or None to hide.
            strategy: GC strategy name (e.g., "truncate", "hybrid", "summarize").
            target_percent: Target usage after GC (e.g., 60.0).
            continuous_mode: True if GC runs after every turn.
        """
        self._gc_threshold = threshold
        self._gc_strategy = strategy
        self._gc_target_percent = target_percent
        self._gc_continuous_mode = continuous_mode
        self.refresh()

    def register_formatter(self, formatter: Any) -> None:
        """Register an additional formatter plugin with the pipeline.

        Args:
            formatter: A formatter implementing the FormatterPlugin protocol.
        """
        if self._formatter_pipeline:
            self._formatter_pipeline.register(formatter)
            # Also register with agent registry if present
            if self._agent_registry:
                self._agent_registry.set_formatter_pipeline_all(self._formatter_pipeline)

    def get_formatter_pipeline(self) -> Any:
        """Get the formatter pipeline for external access.

        Returns:
            The FormatterPipeline instance.
        """
        return self._formatter_pipeline

    # Plan panel methods

    def update_plan(self, plan_data: Dict[str, Any], agent_id: Optional[str] = None) -> None:
        """Update the plan for an agent.

        Args:
            plan_data: Plan status dict with title, status, steps, progress.
            agent_id: Which agent's plan to update. If None, updates the global
                     plan panel (backwards compatibility) or "main" in registry.
        """
        if self._agent_registry:
            # Route through registry for per-agent plan tracking
            target_id = agent_id or "main"
            self._agent_registry.update_plan(target_id, plan_data)
        else:
            # Fallback to global plan panel
            self._plan_panel.update_plan(plan_data)
        self.refresh()

    def clear_plan(self, agent_id: Optional[str] = None) -> None:
        """Clear the plan for an agent.

        Args:
            agent_id: Which agent's plan to clear. If None, clears the global
                     plan panel (backwards compatibility) or "main" in registry.
        """
        if self._agent_registry:
            target_id = agent_id or "main"
            self._agent_registry.clear_plan(target_id)
        else:
            self._plan_panel.clear()
        self.refresh()

    @property
    def has_plan(self) -> bool:
        """Check if there's an active plan for the current agent."""
        return self._current_plan_has_data()

    # Budget panel methods

    def register_agent_name(self, agent_id: str, name: str) -> None:
        """Register a display name for an agent (used by budget panel tabs).

        Args:
            agent_id: Agent identifier ("main", "subagent_1", etc.)
            name: Human-readable display name.
        """
        self._budget_panel.set_agent_name(agent_id, name)

    def update_instruction_budget(self, agent_id: str, budget_snapshot: Dict[str, Any]) -> None:
        """Update the instruction budget for an agent.

        Args:
            agent_id: Agent identifier ("main", "subagent_1", etc.)
            budget_snapshot: Budget snapshot from InstructionBudget.snapshot()
        """
        self._budget_panel.update_budget(agent_id, budget_snapshot)
        # Only refresh if the panel is visible to avoid unnecessary renders
        if self._budget_panel.is_visible:
            self.refresh()

    def clear_instruction_budget(self, agent_id: str) -> None:
        """Clear the instruction budget for an agent.

        Args:
            agent_id: Agent identifier to clear.
        """
        self._budget_panel.clear_budget(agent_id)
        if self._budget_panel.is_visible:
            self.refresh()

    @property
    def theme(self) -> ThemeConfig:
        """Get the current theme configuration."""
        return self._theme

    def set_theme(self, theme: ThemeConfig) -> None:
        """Set a new theme and update the application styles.

        Args:
            theme: New ThemeConfig to apply.
        """
        self._theme = theme

        # Propagate theme to all components that use Rich styles
        self._output_buffer.set_theme(theme)
        self._plan_panel.set_theme(theme)
        if self._agent_registry:
            self._agent_registry.set_theme_all(theme)

        # Update code block formatter syntax theme to match UI theme
        if self._code_block_formatter:
            self._code_block_formatter.set_syntax_theme(theme.name)

        # Update prompt_toolkit styles on the running application
        if self._app:
            from prompt_toolkit.styles import merge_styles

            default_style = self._theme.get_prompt_toolkit_style()
            input_style = self._input_handler._pt_style if self._input_handler else None

            if input_style:
                self._app.style = merge_styles([input_style, default_style])
            else:
                self._app.style = default_style

        self.refresh()

    # Output buffer methods

    def append_output(self, source: str, text: str, mode: str) -> None:
        """Append output to the scrolling panel."""
        # Use selected agent's buffer if registry present, otherwise use default
        if self._agent_registry:
            buffer = self._agent_registry.get_selected_buffer()
            if not buffer:
                buffer = self._output_buffer
        else:
            buffer = self._output_buffer

        buffer.append(source, text, mode)
        # Auto-scroll to bottom when new output arrives
        buffer.scroll_to_bottom()
        # Use debounced refresh during streaming for better performance
        self._schedule_refresh()

    def add_system_message(self, message: str, style: str = "system_info") -> None:
        """Add a system message to the output."""
        # Use selected agent's buffer if registry present, otherwise use default
        if self._agent_registry:
            buffer = self._agent_registry.get_selected_buffer()
            if not buffer:
                buffer = self._output_buffer
        else:
            buffer = self._output_buffer

        buffer.add_system_message(message, style)
        # Auto-scroll to bottom when new output arrives
        buffer.scroll_to_bottom()
        self.refresh()

    # =========================================================================
    # Pending Prompts Queue (displayed above input field)
    # =========================================================================

    def add_pending_prompt(self, prompt: str) -> None:
        """Add a prompt to the pending queue.

        Called when a prompt is queued for mid-turn injection (user types
        while model is running, or subagent injects a message).

        Args:
            prompt: The prompt text to queue.
        """
        import time
        self._pending_prompts.append((prompt, time.time()))
        self.refresh()

    def remove_pending_prompt(self, prompt: str) -> None:
        """Remove a prompt from the pending queue.

        Called when a queued prompt is processed/injected.

        Args:
            prompt: The prompt text that was processed.
        """
        # Remove first matching prompt
        for i, (p, _) in enumerate(self._pending_prompts):
            if p == prompt:
                self._pending_prompts.pop(i)
                break
        self.refresh()

    def clear_pending_prompts(self) -> None:
        """Clear all pending prompts."""
        self._pending_prompts.clear()
        self.refresh()

    def _get_pending_prompts_height(self) -> int:
        """Calculate height for pending prompts bar.

        Returns 0 if no pending prompts, otherwise 1 line per prompt
        up to a maximum of 5 lines.
        """
        count = len(self._pending_prompts)
        if count == 0:
            return 0
        return min(count, 5)  # Cap at 5 lines

    def _get_pending_prompts_content(self) -> List[Tuple[str, str]]:
        """Render pending prompts as formatted text for display above input.

        Returns:
            List of (style, text) tuples for prompt_toolkit.
        """
        if not self._pending_prompts:
            return []

        result = []
        max_display = 5
        count = len(self._pending_prompts)

        # Show most recent prompts (up to max_display)
        prompts_to_show = self._pending_prompts[-max_display:]

        for i, (prompt, timestamp) in enumerate(prompts_to_show):
            # Truncate long prompts
            preview = prompt.replace("\n", " ")
            if len(preview) > 60:
                preview = preview[:57] + "..."

            # Format: "⏳ [1] prompt text here..."
            queue_num = count - len(prompts_to_show) + i + 1
            line = f"⏳ [{queue_num}] {preview}"

            result.append(("class:pending-prompt", line))
            if i < len(prompts_to_show) - 1:
                result.append(("", "\n"))

        # If there are more prompts than we can show
        if count > max_display:
            hidden = count - max_display
            result.insert(0, ("class:pending-prompt.overflow", f"  ... {hidden} more queued ...\n"))

        return result

    def update_last_system_message(
        self,
        message: str,
        style: str = "system_info",
        prefix: Optional[str] = None
    ) -> bool:
        """Update the last system message in the output.

        Args:
            message: The new message text.
            style: Rich style for the message.
            prefix: If provided, only update a system message that starts with
                this prefix.

        Returns:
            True if updated, False if no system message found.
        """
        # Use selected agent's buffer if registry present, otherwise use default
        if self._agent_registry:
            buffer = self._agent_registry.get_selected_buffer()
            if not buffer:
                buffer = self._output_buffer
        else:
            buffer = self._output_buffer

        result = buffer.update_last_system_message(message, style, prefix)
        if result:
            self.refresh()
        return result

    def add_init_progress(self, message: str, style: str = "dim") -> None:
        """Add a line to the initialization progress display.

        Init progress lines are displayed during session initialization
        and cleared when initialization completes.
        """
        self._init_progress_lines.append((message, style))

    def clear_init_progress(self) -> None:
        """Clear all initialization progress lines."""
        self._init_progress_lines.clear()

    def show_init_progress(self) -> None:
        """Display the current initialization progress lines to the output buffer."""
        # Use selected agent's buffer if registry present, otherwise use default
        if self._agent_registry:
            buffer = self._agent_registry.get_selected_buffer()
            if not buffer:
                buffer = self._output_buffer
        else:
            buffer = self._output_buffer

        for message, style in self._init_progress_lines:
            buffer.add_system_message(message, style)

        buffer.scroll_to_bottom()

    def clear_output(self) -> None:
        """Clear the output buffer."""
        # Use selected agent's buffer if registry present
        if self._agent_registry:
            buffer = self._agent_registry.get_selected_buffer()
            if buffer:
                buffer.clear()
            else:
                self._output_buffer.clear()
        else:
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
            # Account for panel borders
            page_size = max(5, available - 4)

        # Use selected agent's buffer if registry present
        if self._agent_registry:
            buffer = self._agent_registry.get_selected_buffer()
            if not buffer:
                buffer = self._output_buffer
        else:
            buffer = self._output_buffer

        # Always use pager mode for zoom view (even if content fits on one page)
        # This ensures clean separation from original content and proper 'q' to exit
        self._start_pager(lines, page_size)

    def _start_pager(self, lines: list, page_size: int) -> None:
        """Start paged display mode (internal).

        Creates a temporary buffer for pager content, preserving the original
        buffer which is restored when pager exits.

        Args:
            lines: List of (text, style) tuples to display.
            page_size: Lines per page.
        """
        self._pager_lines = lines
        self._pager_page_size = page_size
        self._pager_current = 0
        self._pager_active = True

        # Create a temporary buffer for pager content (preserves original)
        self._pager_temp_buffer = OutputBuffer()
        # Copy width settings from the main buffer
        if self._agent_registry:
            original_buffer = self._agent_registry.get_selected_buffer()
            if original_buffer:
                self._pager_temp_buffer.set_width(original_buffer._console_width)
        else:
            self._pager_temp_buffer.set_width(self._output_buffer._console_width)
        # Apply formatter pipeline for output processing
        if self._formatter_pipeline:
            self._pager_temp_buffer.set_formatter_pipeline(self._formatter_pipeline)

        self._show_pager_page()

    def _show_pager_page(self) -> None:
        """Show the current pager page in the temporary pager buffer."""
        if not self._pager_active or not hasattr(self, '_pager_temp_buffer'):
            return

        lines = self._pager_lines
        page_size = self._pager_page_size
        current = self._pager_current

        total_lines = len(lines)
        total_pages = (total_lines + page_size - 1) // page_size
        page_num = (current // page_size) + 1

        # Use the temporary pager buffer (not the main buffer)
        buffer = self._pager_temp_buffer

        # Clear the temp buffer for fresh page
        buffer.clear()

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
                buffer.add_system_message(text, style)
            # Add a separator to show where new content starts
            buffer.add_system_message("─" * 40, style="separator")

        # Show current page content
        for text, style in lines[current:end_line]:
            buffer.add_system_message(text, style)

        # Show navigation hint
        if not is_last_page:
            buffer.add_system_message(
                f"── Page {page_num}/{total_pages} ── Press Enter/Space for more, 'q' to quit ──",
                style="pager_nav"
            )
        else:
            # Last page or single page - show how to exit
            if total_pages > 1:
                buffer.add_system_message(
                    f"── Page {page_num}/{total_pages} (end) ── Press 'q' to close ──",
                    style="pager_nav"
                )
            else:
                buffer.add_system_message(
                    "── Press 'q' to close ──",
                    style="pager_nav"
                )

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
            # Quit pager - restore original buffer
            self._stop_pager()
            return True

        if text.lower() == 'v':
            # Ignore 'v' when already zoomed in - we're already viewing
            return True

        # Empty string or any other input advances to next page
        self._pager_current += self._pager_page_size
        if self._pager_current >= len(self._pager_lines):
            # Reached end - restore original buffer
            self._stop_pager()
        else:
            self._show_pager_page()

        return True

    def _stop_pager(self) -> None:
        """Stop pager mode and restore original buffer."""
        self._pager_active = False
        # Clean up temporary buffer
        if hasattr(self, '_pager_temp_buffer'):
            del self._pager_temp_buffer
        # Refresh to show original buffer again
        self.refresh()

    def _advance_pager_page(self) -> None:
        """Advance to next pager page or exit if at end."""
        if not getattr(self, '_pager_active', False):
            return
        self._pager_current += self._pager_page_size
        if self._pager_current >= len(self._pager_lines):
            # Reached end - restore original buffer
            self._stop_pager()
        else:
            self._show_pager_page()

    @property
    def pager_active(self) -> bool:
        """Check if pager mode is active."""
        return getattr(self, '_pager_active', False)

    def start_spinner(self) -> None:
        """Start the spinner animation to show model is thinking.

        Thread-safe: can be called from background threads.
        """
        # Use selected agent's buffer if registry present
        if self._agent_registry:
            buffer = self._agent_registry.get_selected_buffer()
            if buffer:
                buffer.start_spinner()
            else:
                self._output_buffer.start_spinner()
        else:
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

        # Use selected agent's buffer if registry present
        if self._agent_registry:
            buffer = self._agent_registry.get_selected_buffer()
            if buffer:
                buffer.stop_spinner()
            else:
                self._output_buffer.stop_spinner()
        else:
            self._output_buffer.stop_spinner()

        self.refresh()

    def ensure_spinner_timer_running(self) -> None:
        """Ensure the spinner animation timer is running.

        Call this when an agent's status changes to 'active' to ensure
        the spinner animation loop is running. This doesn't touch any
        buffer's spinner state - it only ensures the timer is active.
        """
        if self._spinner_timer_active:
            return  # Already running

        self._spinner_timer_active = True
        # Schedule spinner advance in main event loop (thread-safe)
        if self._app and self._app.is_running:
            self._app.loop.call_soon_threadsafe(self._advance_spinner)
        else:
            self._advance_spinner()

    def _advance_spinner(self) -> None:
        """Advance spinner animation frame.

        Advances spinners on ALL agent buffers that have active spinners,
        and also the agent tab bar spinner for processing status display.
        This ensures that when you switch agents (F2), the spinner is
        already animating if that agent is thinking.
        """
        if not self._spinner_timer_active:
            return

        # Advance spinners on ALL agent buffers that have active spinners
        any_active = False
        if self._agent_registry:
            for agent_id in self._agent_registry.get_all_agent_ids():
                buffer = self._agent_registry.get_buffer(agent_id)
                if buffer and buffer.spinner_active:
                    buffer.advance_spinner()
                    any_active = True
        else:
            if self._output_buffer.spinner_active:
                self._output_buffer.advance_spinner()
                any_active = True

        # Advance tab bar spinner (for processing status in tabs)
        if self._agent_tab_bar and any_active:
            self._agent_tab_bar.advance_spinner()

        # Stop timer if no agents have active spinners
        if not any_active:
            self._spinner_timer_active = False
            return

        self.refresh()
        # Schedule next frame using prompt_toolkit's call_later
        if self._app and self._app.is_running:
            self._app.loop.call_later(0.1, self._advance_spinner)

    def _compute_valid_prefixes(self, response_options: list) -> set:
        """Compute all valid prefixes for the given response options.

        This generates all possible prefixes of valid responses so we can
        proactively filter input keystroke by keystroke.

        Args:
            response_options: List of PermissionResponseOption objects.

        Returns:
            Set of all valid prefixes (lowercase), including empty string.
        """
        prefixes = {''}  # Empty string is always valid (nothing typed yet)
        for option in response_options:
            # Handle both object attributes and dict keys
            if isinstance(option, dict):
                short = option.get('key', option.get('short', '')).lower()
                full = option.get('label', option.get('full', '')).lower()
            else:
                short = getattr(option, 'short', getattr(option, 'key', '')).lower()
                full = getattr(option, 'full', getattr(option, 'label', '')).lower()
            # Add all prefixes of short form (e.g., "y", "a", "n")
            for i in range(1, len(short) + 1):
                prefixes.add(short[:i])
            # Add all prefixes of full form (e.g., "yes", "ye", "y")
            for i in range(1, len(full) + 1):
                prefixes.add(full[:i])
        return prefixes

    def _is_valid_permission_input(self, text: str) -> bool:
        """Check if text is a valid prefix for permission input.

        Args:
            text: The current input text to validate.

        Returns:
            True if text could lead to a valid response, False otherwise.
        """
        return text.lower() in self._valid_input_prefixes

    def set_waiting_for_channel_input(
        self,
        waiting: bool,
        response_options: Optional[list] = None
    ) -> None:
        """Set whether we're waiting for channel (permission/clarification) input.

        When waiting for channel input (permission or clarification prompts),
        this method also switches the completion source to show only valid
        response options instead of the normal completions (commands, files, etc.).

        The valid response options are provided by the permission plugin,
        making it the single source of truth for what responses are valid.

        Args:
            waiting: True if waiting for channel input, False otherwise.
            response_options: List of PermissionResponseOption objects from the
                            permission plugin. Only used when waiting=True.
                            Each option should have: short, full, description attributes.
        """
        self._waiting_for_channel_input = waiting
        # Compute valid prefixes for keystroke filtering
        if waiting and response_options:
            self._valid_input_prefixes = self._compute_valid_prefixes(response_options)
            self._last_valid_permission_input = ""  # Reset for new permission prompt
            # Store options for keyboard navigation
            self._permission_response_options = response_options
            self._permission_focus_index = 0  # Default focus on first option (yes)
        else:
            self._valid_input_prefixes = set()
            self._last_valid_permission_input = ""
            self._permission_response_options = None
            self._permission_focus_index = 0
        # Update output buffer with focus state for inline highlighting (use correct buffer)
        buffer = self._agent_registry.get_selected_buffer() if self._agent_registry else self._output_buffer
        buffer.set_permission_focus(
            self._permission_response_options,
            self._permission_focus_index
        )
        self.refresh()

    def _select_focused_permission_option(self) -> None:
        """Select the currently focused permission option and submit it."""
        if not self._permission_response_options or not self._input_callback:
            return

        # Get the focused option
        option = self._permission_response_options[self._permission_focus_index]
        # Get the short form (the key to submit)
        if isinstance(option, dict):
            short = option.get('key', option.get('short', ''))
        else:
            short = getattr(option, 'short', getattr(option, 'key', ''))

        # Submit the selection via input callback
        self._input_callback(short)
