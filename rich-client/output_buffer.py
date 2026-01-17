"""Output buffer for the scrolling output panel.

Manages a ring buffer of output lines for display in the scrolling
region of the TUI.
"""

import os
import tempfile
import textwrap
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional, Tuple, Union


def _trace(msg: str) -> None:
    """Write trace message to log file for debugging."""
    trace_path = os.environ.get(
        'JAATO_TRACE_LOG',
        os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
    )
    if trace_path:
        try:
            with open(trace_path, "a") as f:
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                f.write(f"[{ts}] [OutputBuffer] {msg}\n")
                f.flush()
        except (IOError, OSError):
            pass

import re

from rich.console import Console, RenderableType
from rich.text import Text
from rich.panel import Panel
from rich.align import Align

# Type checking import for ThemeConfig
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from theme import ThemeConfig


# Pattern to strip ANSI escape codes for visible length calculation
_ANSI_ESCAPE_PATTERN = re.compile(r'\x1b\[[0-9;]*m')


def _visible_len(text: str) -> int:
    """Calculate visible length of text, ignoring ANSI escape codes."""
    return len(_ANSI_ESCAPE_PATTERN.sub('', text))


def _slice_ansi_string(text: str, start: int, width: int) -> tuple[str, bool, bool]:
    """Slice a string with ANSI codes to a visible width range.

    Args:
        text: String potentially containing ANSI escape codes.
        start: Starting visible character position (0-indexed).
        width: Maximum visible width to return.

    Returns:
        Tuple of (sliced_string, has_more_left, has_more_right):
        - sliced_string: The visible portion with ANSI codes preserved
        - has_more_left: True if there's content before start
        - has_more_right: True if there's content after start+width
    """
    # Pattern to find ANSI sequences
    ansi_pattern = re.compile(r'(\x1b\[[0-9;]*m)')

    result = []
    visible_pos = 0  # Current visible character position
    active_codes = []  # Track active ANSI codes for proper reset/restore

    # Split into segments (alternating text and ANSI codes)
    segments = ansi_pattern.split(text)

    for segment in segments:
        if not segment:
            continue

        if ansi_pattern.match(segment):
            # This is an ANSI code
            # Track it if we're in or before the viewport (for proper styling)
            if visible_pos < start + width:
                if segment == '\x1b[0m':
                    active_codes.clear()
                else:
                    active_codes.append(segment)
                # Only include in output if we've started outputting
                if visible_pos >= start or (visible_pos < start and visible_pos + len(active_codes) > 0):
                    result.append(segment)
        else:
            # This is regular text
            for char in segment:
                if visible_pos >= start and visible_pos < start + width:
                    # Character is in viewport
                    result.append(char)
                visible_pos += 1

                if visible_pos >= start + width:
                    # We've filled the viewport, but continue counting for has_more_right
                    pass

    # Calculate overflow indicators
    total_visible = _visible_len(text)
    has_more_left = start > 0
    has_more_right = total_visible > start + width

    # Ensure we close any open ANSI codes
    sliced = ''.join(result)
    if active_codes and sliced:
        sliced += '\x1b[0m'

    return sliced, has_more_left, has_more_right


@dataclass
class OutputLine:
    """A single line of output with metadata."""
    source: str
    text: str
    style: str
    display_lines: int = 1  # How many terminal lines this takes when rendered
    is_turn_start: bool = False  # True if this is the first line of a new turn
    # Render cache for performance (avoids rebuilding Text objects on each render)
    _rendered_cache: Optional[Text] = None
    _cache_width: int = 0  # Width used when cache was created (invalidate on change)


@dataclass
class ToolBlock:
    """A block of completed tools stored inline in the output buffer.

    This allows tool blocks to be rendered dynamically with expand/collapse
    state while maintaining their position in the output flow.
    """
    tools: List['ActiveToolCall']  # The tools in this block
    expanded: bool = True  # Whether the block is in expanded view
    selected_index: Optional[int] = None  # Which tool is selected (when navigating)


@dataclass
class ActiveToolsMarker:
    """Marker for where active tools should render in the output flow.

    This is a virtual item inserted into the render list at the placeholder
    position, so active tools render inline (at their chronological position)
    rather than always at the bottom.
    """
    pass  # No data needed - just a position marker


@dataclass
class ActiveToolCall:
    """Represents an actively executing or completed tool call."""
    name: str
    args_summary: str  # Truncated string representation of args
    call_id: Optional[str] = None  # Unique ID for correlating start/end of same tool call
    completed: bool = False  # True when tool execution finished
    success: bool = True  # Whether the tool succeeded (only valid when completed)
    duration_seconds: Optional[float] = None  # Execution time (only valid when completed)
    error_message: Optional[str] = None  # Error message if tool failed
    # Display override (e.g., show "writeFile" instead of "askPermission")
    display_name: Optional[str] = None  # If set, display this instead of name
    display_args_summary: Optional[str] = None  # If set, display this instead of args_summary
    # Permission tracking
    permission_state: Optional[str] = None  # None, "pending", "granted", "denied"
    permission_method: Optional[str] = None  # "yes", "always", "once", "never", "whitelist", "blacklist"
    permission_prompt_lines: Optional[List[str]] = None  # Expanded prompt (may contain ANSI codes)
    permission_format_hint: Optional[str] = None  # "diff" for pre-formatted content (skip box wrapping)
    permission_truncated: bool = False  # True if prompt is truncated
    permission_h_scroll: int = 0  # Horizontal scroll offset for diff viewport (stage 2)
    # Clarification tracking (per-question progressive display)
    clarification_state: Optional[str] = None  # None, "pending", "resolved"
    clarification_prompt_lines: Optional[List[str]] = None  # Current question lines
    clarification_truncated: bool = False  # True if prompt is truncated
    clarification_current_question: int = 0  # Current question index (1-based)
    clarification_total_questions: int = 0  # Total number of questions
    clarification_answered: Optional[List[Tuple[int, str]]] = None  # List of (question_index, answer_summary)
    clarification_summary: Optional[List[Tuple[str, str]]] = None  # Q&A pairs [(question, answer), ...] for overview
    # Live output tracking (tail -f style preview)
    output_lines: Optional[List[str]] = None  # Rolling buffer of recent output lines
    output_max_lines: int = 30  # Max lines to keep in buffer
    output_display_lines: int = 5  # Max lines to show at once when expanded
    output_scroll_offset: int = 0  # Scroll position (0 = show most recent lines)
    # Per-tool expand state for navigation
    expanded: bool = False  # Whether this tool's output is expanded


class OutputBuffer:
    """Manages output lines for the scrolling panel.

    Stores output in a ring buffer and renders to Rich Text
    for display in the output panel.
    """

    # Spinner animation frames
    SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    # Type alias for items that can be stored in the line buffer
    LineItem = Union[OutputLine, ToolBlock]

    def __init__(self, max_lines: int = 1000, agent_type: str = "main"):
        """Initialize the output buffer.

        Args:
            max_lines: Maximum number of lines to retain.
            agent_type: Type of agent ("main" or "subagent") for label display.
        """
        self._lines: deque[Union[OutputLine, ToolBlock]] = deque(maxlen=max_lines)
        self._current_block: Optional[Tuple[str, List[str], bool]] = None
        self._measure_console: Optional[Console] = None
        self._console_width: int = 80
        self._visible_height: int = 20  # Last known visible height for auto-scroll
        self._last_source: Optional[str] = None  # Track source for turn detection
        self._last_turn_source: Optional[str] = None  # Track user/model turns (ignores system/tool)
        self._scroll_offset: int = 0  # Lines scrolled up from bottom (0 = at bottom)
        self._spinner_active: bool = False
        self._spinner_index: int = 0
        self._active_tools: List[ActiveToolCall] = []  # Currently executing tools
        self._tools_expanded: bool = False  # Toggle between collapsed/expanded tool view
        self._rendering: bool = False  # Guard against flushes during render
        self._tool_placeholder_index: Optional[int] = None  # Position in _lines where tools render
        self._agent_type: str = agent_type  # "main" or "subagent" for user label
        # Tool navigation state
        self._tool_nav_active: bool = False  # True when navigating tools
        self._selected_block_index: Optional[int] = None  # Which ToolBlock is selected
        self._selected_tool_index: Optional[int] = None  # Which tool within selected block
        # Keybinding config for dynamic UI hints
        self._keybinding_config: Optional[Any] = None
        # Pending enrichment notifications (queued while tools are active)
        self._pending_enrichments: List[Tuple[str, str, str]] = []  # (source, text, mode)
        # Formatter pipeline for output processing (optional)
        self._formatter_pipeline: Optional[Any] = None
        # Theme configuration for styling (optional)
        self._theme: Optional["ThemeConfig"] = None

    def set_width(self, width: int) -> None:
        """Set the console width for measuring line wrapping.

        Args:
            width: Console width in characters.
        """
        if width != self._console_width:
            # Width changed - invalidate all line render caches
            self._invalidate_line_caches()
        self._console_width = width
        self._measure_console = Console(width=width, force_terminal=True)
        # Sync width with formatter pipeline if set
        if self._formatter_pipeline and hasattr(self._formatter_pipeline, 'set_console_width'):
            self._formatter_pipeline.set_console_width(width)

    def _invalidate_line_caches(self) -> None:
        """Invalidate render caches and recalculate display_lines for all lines (called on width change).

        This is critical for correct scroll position calculation - display_lines
        must reflect the actual rendered height at the current width, otherwise
        gaps or overflow will occur when scrolling.
        """
        for item in self._lines:
            if isinstance(item, OutputLine):
                item._rendered_cache = None
                item._cache_width = 0
                # Recalculate display_lines for the new width
                item.display_lines = self._measure_display_lines(
                    item.source, item.text, item.is_turn_start
                )

    def set_keybinding_config(self, config: Any) -> None:
        """Set the keybinding config for dynamic UI hints.

        Args:
            config: KeybindingConfig instance from keybindings module.
        """
        self._keybinding_config = config

    def set_formatter_pipeline(self, pipeline: Any) -> None:
        """Set the formatter pipeline for output processing.

        Args:
            pipeline: FormatterPipeline instance or None to disable.
        """
        self._formatter_pipeline = pipeline
        # Sync console width with pipeline if it has the method
        if pipeline and hasattr(pipeline, 'set_console_width'):
            pipeline.set_console_width(self._console_width)

    # Legacy alias for backwards compatibility
    def set_output_formatter(self, formatter: Any) -> None:
        """Deprecated: Use set_formatter_pipeline instead."""
        self.set_formatter_pipeline(formatter)

    def set_theme(self, theme: "ThemeConfig") -> None:
        """Set the theme configuration for styling.

        Args:
            theme: ThemeConfig instance for Rich style lookups.
        """
        self._theme = theme
        # Invalidate render caches so content re-renders with new theme colors
        self._invalidate_line_caches()

    # Known Rich style primitives that should be passed through without semantic lookup
    _RICH_STYLE_PRIMITIVES = frozenset({
        "bold", "dim", "italic", "underline", "blink", "reverse", "strike",
        "red", "green", "yellow", "blue", "magenta", "cyan", "white", "black",
        "bright_red", "bright_green", "bright_yellow", "bright_blue",
        "bright_magenta", "bright_cyan", "bright_white",
    })

    def _style(self, semantic_name: str, fallback: str = "") -> str:
        """Get a Rich style string from the theme.

        Args:
            semantic_name: Semantic style name (e.g., "tool_output", "user_header"),
                          or a raw Rich style string (e.g., "bold", "dim cyan").
            fallback: Fallback style if theme is not set or name not found.

        Returns:
            Rich style string.
        """
        # Handle empty semantic name - just return fallback
        if not semantic_name:
            return fallback

        # Check if this is a raw Rich style (primitive or compound like "bold cyan")
        # If any word is a Rich primitive, treat the whole thing as a raw style
        words = semantic_name.split()
        if any(word in self._RICH_STYLE_PRIMITIVES for word in words):
            return semantic_name

        if self._theme:
            style = self._theme.get_rich_style(semantic_name)
            if style:
                return style
        return fallback

    def _format_key_hint(self, action: str) -> str:
        """Format a keybinding for display in UI hints.

        Args:
            action: The action name (e.g., "tool_expand", "nav_up")

        Returns:
            Human-readable key representation (e.g., "→", "↑", "Ctrl+N")
        """
        if not self._keybinding_config:
            # Fallback defaults if no config
            defaults = {
                "nav_up": "↑", "nav_down": "↓",
                "tool_expand": "→", "tool_collapse": "←", "tool_exit": "Esc",
                "tool_output_up": "↑", "tool_output_down": "↓",
                "tool_nav_enter": "Ctrl+N", "toggle_tools": "Ctrl+T",
            }
            return defaults.get(action, action)

        try:
            key = getattr(self._keybinding_config, action, None)
            if key is None:
                return action

            # Convert key to human-readable format
            if isinstance(key, list):
                key = " ".join(key)

            # Map common keys to symbols/readable names
            key_map = {
                "up": "↑", "down": "↓", "left": "←", "right": "→",
                "escape": "Esc", "enter": "Enter", "space": "Space",
                "pageup": "PgUp", "pagedown": "PgDn",
                "home": "Home", "end": "End", "tab": "Tab",
            }

            # Handle control keys
            if key.startswith("c-"):
                letter = key[2:].upper()
                return f"Ctrl+{letter}"

            return key_map.get(key, key)
        except Exception:
            return action

    def _measure_display_lines(self, source: str, text: str, is_turn_start: bool = False) -> int:
        """Measure how many display lines a piece of text will take.

        Args:
            source: Source of the output.
            text: The text content.
            is_turn_start: Whether this is the first line of a turn (shows prefix).

        Returns:
            Number of display lines when rendered.
        """
        if not self._measure_console:
            self._measure_console = Console(width=self._console_width, force_terminal=True)

        # Build the text as it will be rendered
        # IMPORTANT: If text contains ANSI codes, use Text.from_ansi() to properly
        # measure visible width. Otherwise ANSI escape sequences are counted as
        # visible characters, causing massive overcounting for syntax-highlighted text.
        has_ansi = '\x1b[' in text
        if has_ansi:
            rendered = Text.from_ansi(text)
        else:
            rendered = Text()
            rendered.append(text)

        if source == "model":
            if is_turn_start:
                # Blank line (1) + header line (1) + wrapped content lines
                # Note: blank line is rendered at inter-item level for non-first visible items,
                # but we always count it in measurement. This slightly overcounts for first
                # visible item but ensures we don't try to show more items than fit.
                with self._measure_console.capture() as capture:
                    self._measure_console.print(rendered, end='')
                output = capture.get()
                content_lines = output.count('\n') + 1 if output else 1
                # 1 (blank) + 1 (header) + content lines
                return 2 + content_lines
        elif source == "thinking":
            if is_turn_start:
                # Blank line (1) + Model header (1) + thinking header (1) + content + footer (1)
                # Note: footer is only rendered for last thinking line, but we count it
                # for all thinking lines to ensure the last one fits on screen.
                with self._measure_console.capture() as capture:
                    self._measure_console.print(rendered, end='')
                output = capture.get()
                content_lines = output.count('\n') + 1 if output else 1
                # 1 (blank) + 1 (Model header) + 1 (thinking header) + content + 1 (footer)
                return 4 + content_lines
        elif source in ("user", "parent"):
            if is_turn_start:
                # Blank line (1) + header line (1) + wrapped content lines
                # Note: blank line is rendered at inter-item level for non-first visible items,
                # but we always count it in measurement. This slightly overcounts for first
                # visible item but ensures we don't try to show more items than fit.
                with self._measure_console.capture() as capture:
                    self._measure_console.print(rendered, end='')
                output = capture.get()
                content_lines = output.count('\n') + 1 if output else 1
                # 1 (blank) + 1 (header) + content lines
                return 2 + content_lines
        elif source not in ("system", "enrichment"):
            if is_turn_start:
                # Prepend source prefix for non-model sources
                prefix = Text(f"[{source}] ", style=self._style("tool_source_label", "dim magenta"))
                rendered = prefix + rendered

        # Measure by capturing output
        with self._measure_console.capture() as capture:
            self._measure_console.print(rendered, end='')

        output = capture.get()
        if not output:
            return 1
        return output.count('\n') + 1

    def _add_line(self, source: str, text: str, style: str, is_turn_start: bool = False) -> None:
        """Add a line to the buffer with measured display lines.

        Args:
            source: Source of the output.
            text: The text content.
            style: Style for the line.
            is_turn_start: Whether this is the first line of a new turn.
        """
        display_lines = self._measure_display_lines(source, text, is_turn_start)
        self._lines.append(OutputLine(source, text, style, display_lines, is_turn_start))

    def append(self, source: str, text: str, mode: str) -> None:
        """Append output to the buffer.

        Args:
            source: Source of the output ("model", plugin name, etc.)
            text: The output text.
            mode: "write" for new block, "append" to continue.
        """

        # Skip plan messages - they're shown in the sticky plan panel
        if source == "plan":
            return

        # Skip permission messages - they're shown inline under tool calls in the tree
        if source == "permission":
            return

        # Skip clarification messages - they're shown inline under tool calls in the tree
        if source == "clarification":
            return

        # Queue enrichment notifications while tools are active
        # They should appear AFTER the tool tree, not before
        if source == "enrichment" and self._active_tools:
            self._pending_enrichments.append((source, text, mode))
            return

        # If this is new user/parent input, exit navigation mode and finalize any remaining tools
        if source in ("user", "parent") and mode == "write":
            self._exit_navigation_mode()
            # Finalize any remaining active tools
            if self._active_tools:
                self._finalize_completed_tools()

        # If this is new model text and there are completed tools, finalize the tree first
        # This ensures the tool tree appears BEFORE the new response, not after
        if source == "model" and mode == "write" and self._active_tools:
            all_completed = all(tool.completed for tool in self._active_tools)
            any_pending = any(
                tool.permission_state == "pending" or tool.clarification_state == "pending"
                for tool in self._active_tools
            )
            if all_completed and not any_pending:
                self.finalize_tool_tree()

        if mode == "write":
            # Start a new block
            self._flush_current_block()
            # Only show header when switching between user/parent and model turns
            # (not for every model response within same agentic loop)
            is_new_turn = False
            if source in ("user", "parent", "model", "thinking"):
                # Treat "parent" like "user", "thinking" like "model" for turn tracking
                effective_source = "user" if source == "parent" else ("model" if source == "thinking" else source)
                is_new_turn = (self._last_turn_source != effective_source)
                self._last_turn_source = effective_source
            self._current_block = (source, [text], is_new_turn)
        elif mode == "append":
            # Append mode (streaming)
            if self._current_block and self._current_block[0] == source:
                # Append to existing block from same source
                self._current_block[1].append(text)
            else:
                # First streaming chunk or source changed - create new block
                self._flush_current_block()
                is_new_turn = False
                if source in ("user", "parent", "model", "thinking"):
                    # Treat "parent" like "user", "thinking" like "model" for turn tracking
                    effective_source = "user" if source == "parent" else ("model" if source == "thinking" else source)
                    is_new_turn = (self._last_turn_source != effective_source)
                    self._last_turn_source = effective_source
                self._current_block = (source, [text], is_new_turn)
        elif mode == "replace":
            # Replace mode: server has sent formatted text to replace accumulated output
            # Clear current block if from same source and replace with full text
            if self._current_block and self._current_block[0] == source:
                # Preserve is_new_turn from existing block
                _, _, is_new_turn = self._current_block
                self._current_block = (source, [text], is_new_turn)
            else:
                # First chunk or source changed - create new block
                self._flush_current_block()
                is_new_turn = False
                if source in ("user", "parent", "model", "thinking"):
                    # Treat "parent" like "user", "thinking" like "model" for turn tracking
                    effective_source = "user" if source == "parent" else ("model" if source == "thinking" else source)
                    is_new_turn = (self._last_turn_source != effective_source)
                    self._last_turn_source = effective_source
                self._current_block = (source, [text], is_new_turn)
        elif mode == "flush":
            # Flush mode: process pending output but don't add any content
            # Used to synchronize output before UI events like tool tree rendering
            self._flush_current_block()
        else:
            # Standalone line (unknown mode)
            self._flush_current_block()
            is_new_turn = self._last_source != source
            for i, line in enumerate(text.split('\n')):
                self._add_line(source, line, "line", is_turn_start=(i == 0 and is_new_turn))
            self._last_source = source

    def _flush_current_block(self) -> None:
        """Flush the current block to lines."""
        # Guard: don't flush during render cycles (prompt_toolkit calls render frequently)
        if self._rendering:
            return
        if self._current_block:
            source, parts, is_new_turn = self._current_block
            # Concatenate streaming chunks directly (no separator)
            # Then split by newlines for display
            full_text = ''.join(parts)

            # Apply formatter pipeline for model output
            if source == "model" and self._formatter_pipeline:
                if hasattr(self._formatter_pipeline, 'format'):
                    full_text = self._formatter_pipeline.format(full_text)

            lines = full_text.split('\n')
            for i, line in enumerate(lines):
                # Only first line of a new turn gets the prefix
                self._add_line(source, line, "line", is_turn_start=(i == 0 and is_new_turn))
            self._last_source = source
            self._current_block = None

    def _get_current_block_lines(self) -> List[OutputLine]:
        """Get lines from current block without flushing it.

        This allows render() to display streaming content without
        breaking the append chain.
        """
        if not self._current_block:
            return []

        source, parts, is_new_turn = self._current_block
        # Concatenate streaming chunks directly (no separator)
        full_text = ''.join(parts)

        # Apply formatter pipeline for model output
        # This handles complete code blocks during streaming
        if source == "model" and self._formatter_pipeline:
            if hasattr(self._formatter_pipeline, 'format'):
                full_text = self._formatter_pipeline.format(full_text)

        lines_text = full_text.split('\n')

        result = []
        for i, line_text in enumerate(lines_text):
            is_turn_start_line = (i == 0 and is_new_turn)
            display_lines = self._measure_display_lines(source, line_text, is_turn_start_line)
            result.append(OutputLine(
                source=source,
                text=line_text,
                style="system_info",
                display_lines=display_lines,
                is_turn_start=is_turn_start_line
            ))
        return result

    def add_system_message(self, message: str, style: str = "system_info") -> None:
        """Add a system message to the buffer.

        Args:
            message: The system message.
            style: Rich style for the message.
        """
        self._flush_current_block()
        # Handle None or empty messages gracefully
        if not message:
            return
        # Handle multi-line messages
        for line in message.split('\n'):
            self._add_line("system", line, style)

    def update_last_system_message(self, message: str, style: str = "system_info") -> bool:
        """Update the last system message in the buffer.

        Used for progressive updates like init progress that show "..." then "OK".

        Args:
            message: The new message text.
            style: Rich style for the message.

        Returns:
            True if a system message was found and updated, False otherwise.
        """
        # Search backwards for the last system message
        for i in range(len(self._lines) - 1, -1, -1):
            if self._lines[i].source == "system":
                # Update the line
                display_lines = self._measure_display_lines("system", message, False)
                self._lines[i] = OutputLine(
                    source="system",
                    text=message,
                    style=style,
                    display_lines=display_lines,
                    is_turn_start=False
                )
                return True
        return False

    def clear(self) -> None:
        """Clear all output."""
        self._lines.clear()
        self._current_block = None
        self._last_source = None
        self._last_turn_source = None
        self._scroll_offset = 0
        self._spinner_active = False
        self._active_tools.clear()
        self._tool_placeholder_index = None
        self._tool_nav_active = False
        self._selected_block_index = None
        self._selected_tool_index = None

    def start_spinner(self) -> None:
        """Start showing spinner in the output."""
        self._spinner_active = True
        self._spinner_index = 0

    def stop_spinner(self) -> None:
        """Stop showing spinner and finalize tool tree if complete."""
        self._spinner_active = False
        # Flush any pending streaming text before finalizing the turn
        self._flush_current_block()
        # Convert tool tree to scrollable lines if all tools are done
        self.finalize_tool_tree()

    def advance_spinner(self) -> None:
        """Advance spinner to next frame."""
        if self._spinner_active:
            self._spinner_index = (self._spinner_index + 1) % len(self.SPINNER_FRAMES)

    @property
    def spinner_active(self) -> bool:
        """Check if spinner is currently active."""
        return self._spinner_active

    def add_active_tool(self, tool_name: str, tool_args: dict,
                        call_id: Optional[str] = None) -> None:
        """Add a tool to the active tools list.

        Args:
            tool_name: Name of the tool being executed.
            tool_args: Arguments passed to the tool.
            call_id: Unique identifier for this tool call (for correlation).
        """
        # If all existing tools are completed, finalize them as a ToolBlock
        # This ensures proper ordering of text and tool output
        if self._active_tools and all(t.completed for t in self._active_tools):
            self._finalize_completed_tools()

        # Create a summary of args (truncated for display)
        args_str = str(tool_args)
        if len(args_str) > 60:
            args_str = args_str[:57] + "..."

        # Don't add duplicates - check by call_id if provided, otherwise by name
        for tool in self._active_tools:
            if call_id and tool.call_id == call_id:
                return
            # Fall back to name-based check only if no call_id provided
            if not call_id and tool.name == tool_name and not tool.call_id:
                return

        # When first tool is added, establish the placeholder position
        # Flush current block first so tools appear AFTER preceding text
        if not self._active_tools:
            self._flush_current_block()
            self._tool_placeholder_index = len(self._lines)

        self._active_tools.append(ActiveToolCall(
            name=tool_name, args_summary=args_str, call_id=call_id
        ))

    def mark_tool_completed(self, tool_name: str, success: bool = True,
                            duration_seconds: Optional[float] = None,
                            error_message: Optional[str] = None,
                            call_id: Optional[str] = None) -> None:
        """Mark a tool as completed (keeps it in the tree with completion status).

        Args:
            tool_name: Name of the tool that finished.
            success: Whether the tool execution succeeded.
            duration_seconds: How long the tool took to execute.
            error_message: Error message if the tool failed.
            call_id: Unique identifier for this tool call (for correlation).
        """
        for tool in self._active_tools:
            # Match by call_id if provided, otherwise by name (for backwards compatibility)
            if call_id:
                if tool.call_id == call_id and not tool.completed:
                    tool.completed = True
                    tool.success = success
                    tool.duration_seconds = duration_seconds
                    tool.error_message = error_message
                    return
            elif tool.name == tool_name and not tool.completed and not tool.call_id:
                tool.completed = True
                tool.success = success
                tool.duration_seconds = duration_seconds
                tool.error_message = error_message
                return

    def append_tool_output(self, call_id: str, chunk: str) -> None:
        """Append output chunk to a running tool's output buffer.

        Used for live "tail -f" style output preview in the tool tree.

        Args:
            call_id: Unique identifier for the tool call.
            chunk: Output text chunk (may contain newlines).
        """
        for tool in self._active_tools:
            if tool.call_id == call_id and not tool.completed:
                # Initialize output_lines if needed
                if tool.output_lines is None:
                    tool.output_lines = []

                # Split chunk by newlines and add to buffer
                lines = chunk.splitlines()
                for line in lines:
                    # Skip empty lines from split
                    if line or chunk == "\n":
                        tool.output_lines.append(line)

                # Trim to max size (keep most recent lines)
                if len(tool.output_lines) > tool.output_max_lines:
                    tool.output_lines = tool.output_lines[-tool.output_max_lines:]
                return

    def remove_active_tool(self, tool_name: str) -> None:
        """Remove a tool from the active tools list (legacy, now marks as completed).

        Args:
            tool_name: Name of the tool that finished.
        """
        # Instead of removing, mark as completed to keep the tree visible
        self.mark_tool_completed(tool_name)

    def clear_active_tools(self) -> None:
        """Clear all active tools and reset navigation state."""
        self._active_tools.clear()
        self._tool_nav_active = False
        self._selected_tool_index = None

    def toggle_tools_expanded(self) -> bool:
        """Toggle between collapsed and expanded tool view.

        Returns:
            True if now expanded, False if now collapsed.
        """
        self._tools_expanded = not self._tools_expanded
        return self._tools_expanded

    @property
    def tools_expanded(self) -> bool:
        """Check if tool view is currently expanded."""
        return self._tools_expanded

    def _get_approval_indicator(self, method: str) -> Optional[str]:
        """Convert a permission method to a short display indicator.

        Args:
            method: The permission method (whitelist, session_whitelist, etc.)

        Returns:
            Short indicator string like "[pre]", "[once]", "[session]", or None.
        """
        # Pre-approved via config whitelist
        if method in ("whitelist", "policy"):
            return "[pre]"
        # Approved for entire session
        if method in ("session_whitelist", "always"):
            return "[session]"
        # User chose to approve all future requests
        if method == "allow_all":
            return "[all]"
        # Single approval by user
        if method in ("user_approved", "yes", "once"):
            return "[once]"
        # Other methods - no indicator needed
        return None

    # Tool navigation methods
    def _get_tool_blocks(self) -> List[Tuple[int, ToolBlock]]:
        """Get all ToolBlocks in the line buffer with their indices.

        Returns:
            List of (index, ToolBlock) tuples.
        """
        blocks = []
        for i, item in enumerate(self._lines):
            if isinstance(item, ToolBlock):
                blocks.append((i, item))
        return blocks

    def _get_selected_block(self) -> Optional[ToolBlock]:
        """Get the currently selected ToolBlock."""
        if self._selected_block_index is None:
            return None
        blocks = self._get_tool_blocks()
        if 0 <= self._selected_block_index < len(blocks):
            return blocks[self._selected_block_index][1]
        return None

    def _exit_navigation_mode(self) -> None:
        """Exit navigation mode and clear all selection state."""
        self._tool_nav_active = False
        self._selected_block_index = None
        self._selected_tool_index = None
        # Clear selection from all blocks
        for _, block in self._get_tool_blocks():
            block.selected_index = None

    def enter_tool_navigation(self) -> bool:
        """Enter tool navigation mode.

        Returns:
            True if entered successfully, False if no tools available.
        """
        # First check active tools (currently executing)
        if self._active_tools:
            self._tool_nav_active = True
            self._selected_block_index = None  # No block, using active tools
            self._selected_tool_index = 0
            self._tools_expanded = True
            return True

        # Then check tool blocks in the buffer
        blocks = self._get_tool_blocks()
        if not blocks:
            return False

        # Select last tool in last block (most recent) for more intuitive navigation
        self._tool_nav_active = True
        last_block_idx = len(blocks) - 1
        self._selected_block_index = last_block_idx
        last_block = blocks[last_block_idx][1]
        last_tool_idx = len(last_block.tools) - 1
        self._selected_tool_index = last_tool_idx
        last_block.selected_index = last_tool_idx
        last_block.expanded = True
        self._tools_expanded = True  # Ensure tools are in expanded mode
        # Scroll to bottom to show the selected block
        self._scroll_offset = 0
        return True

    def exit_tool_navigation(self) -> None:
        """Exit tool navigation mode."""
        self._exit_navigation_mode()

    @property
    def tool_nav_active(self) -> bool:
        """Check if tool navigation is active."""
        return self._tool_nav_active

    @property
    def selected_tool_index(self) -> Optional[int]:
        """Get currently selected tool index."""
        return self._selected_tool_index

    def _get_all_tools_flat(self) -> List[Tuple[Optional[int], int, ActiveToolCall]]:
        """Get all tools flattened with (block_index, tool_index, tool).

        Returns:
            List of (block_index, tool_index, tool) where block_index is None for active tools.
        """
        result = []
        # Active tools first (block_index = None)
        for i, tool in enumerate(self._active_tools):
            result.append((None, i, tool))
        # Then tools from blocks
        for block_idx, (_, block) in enumerate(self._get_tool_blocks()):
            for tool_idx, tool in enumerate(block.tools):
                result.append((block_idx, tool_idx, tool))
        return result

    def select_next_tool(self) -> bool:
        """Select next tool in list (across all blocks).

        Returns:
            True if selection changed, False if already at end.
        """
        if not self._tool_nav_active:
            return False

        all_tools = self._get_all_tools_flat()
        if not all_tools:
            return False

        # Find current position
        current_flat_idx = None
        for i, (block_idx, tool_idx, _) in enumerate(all_tools):
            if block_idx == self._selected_block_index and tool_idx == self._selected_tool_index:
                current_flat_idx = i
                break

        if current_flat_idx is None:
            # No selection, select first
            block_idx, tool_idx, _ = all_tools[0]
            self._select_tool(block_idx, tool_idx)
            return True

        if current_flat_idx < len(all_tools) - 1:
            # Move to next
            block_idx, tool_idx, _ = all_tools[current_flat_idx + 1]
            self._select_tool(block_idx, tool_idx)
            return True

        return False  # Already at end

    def select_prev_tool(self) -> bool:
        """Select previous tool in list (across all blocks).

        Returns:
            True if selection changed, False if already at start.
        """
        if not self._tool_nav_active:
            return False

        all_tools = self._get_all_tools_flat()
        if not all_tools:
            return False

        # Find current position
        current_flat_idx = None
        for i, (block_idx, tool_idx, _) in enumerate(all_tools):
            if block_idx == self._selected_block_index and tool_idx == self._selected_tool_index:
                current_flat_idx = i
                break

        if current_flat_idx is None:
            # No selection, select first
            block_idx, tool_idx, _ = all_tools[0]
            self._select_tool(block_idx, tool_idx)
            return True

        if current_flat_idx > 0:
            # Move to previous
            block_idx, tool_idx, _ = all_tools[current_flat_idx - 1]
            self._select_tool(block_idx, tool_idx)
            return True

        return False  # Already at start

    def _select_tool(self, block_idx: Optional[int], tool_idx: int) -> None:
        """Select a specific tool."""
        # Clear selection from old block
        old_block = self._get_selected_block()
        if old_block:
            old_block.selected_index = None

        self._selected_block_index = block_idx
        self._selected_tool_index = tool_idx

        # Set selection on new block
        if block_idx is not None:
            blocks = self._get_tool_blocks()
            if 0 <= block_idx < len(blocks):
                new_block = blocks[block_idx][1]
                new_block.selected_index = tool_idx
                new_block.expanded = True

        # Auto-scroll to keep selected tool visible
        self._scroll_to_selected_tool()

    def _scroll_to_selected_tool(self) -> None:
        """Scroll the output panel to keep the selected tool visible."""
        if not self._tool_nav_active:
            return

        # For active tools (block_idx is None), they're always at the bottom
        if self._selected_block_index is None:
            self._scroll_offset = 0
            return

        # For ToolBlocks, calculate the position and scroll if needed
        blocks = self._get_tool_blocks()
        if not blocks or self._selected_block_index >= len(blocks):
            return

        # Calculate display lines up to the selected block
        display_line = 0
        for i, item in enumerate(self._lines):
            if isinstance(item, ToolBlock):
                # Find which block index this is
                block_list_idx = None
                for bi, (line_idx, _) in enumerate(blocks):
                    if line_idx == i:
                        block_list_idx = bi
                        break

                if block_list_idx == self._selected_block_index:
                    # Found the selected block, add lines for tools before selected
                    # Block header (separator + header line)
                    display_line += 2
                    # Add lines for each tool before the selected one
                    block = blocks[block_list_idx][1]
                    for ti in range(self._selected_tool_index or 0):
                        display_line += 1  # Tool line
                        tool = block.tools[ti]
                        if tool.expanded and tool.output_lines:
                            display_line += min(len(tool.output_lines), tool.output_display_lines)
                            if len(tool.output_lines) > tool.output_display_lines:
                                display_line += 2  # Scroll indicators
                    break

            display_line += self._get_item_display_lines(item)

        # Calculate total display lines
        total_lines = sum(self._get_item_display_lines(item) for item in self._lines)

        # Calculate the scroll offset needed to show this position
        # scroll_offset is "lines from bottom", so higher = older content shown
        # We want the selected tool roughly in the middle of the visible area
        target_from_bottom = total_lines - display_line
        margin = self._visible_height // 3  # Keep some margin

        # If tool would be above visible area, scroll up (increase offset)
        if target_from_bottom > self._scroll_offset + self._visible_height - margin:
            self._scroll_offset = max(0, target_from_bottom - self._visible_height + margin)
        # If tool would be below visible area, scroll down (decrease offset)
        elif target_from_bottom < self._scroll_offset + margin:
            self._scroll_offset = max(0, target_from_bottom - margin)

    def toggle_selected_tool_expanded(self) -> bool:
        """Toggle expand/collapse for selected tool's output.

        Returns:
            New expanded state of the tool, False if no selection.
        """
        tool = self.get_selected_tool()
        if tool is None:
            return False
        tool.expanded = not tool.expanded
        return tool.expanded

    def expand_selected_tool(self) -> bool:
        """Expand the selected tool's output.

        Only expands if the tool has output to show.

        Returns:
            True if tool was expanded, False if no selection or no output.
        """
        tool = self.get_selected_tool()
        if tool is None:
            return False
        # Only expand if there's output to show
        if not tool.output_lines or len(tool.output_lines) == 0:
            return False
        tool.expanded = True
        return True

    def collapse_selected_tool(self) -> bool:
        """Collapse the selected tool's output.

        Returns:
            True if tool was collapsed (or already collapsed), False if no selection.
        """
        tool = self.get_selected_tool()
        if tool is None:
            return False
        tool.expanded = False
        return True

    def get_selected_tool(self) -> Optional[ActiveToolCall]:
        """Get the currently selected tool."""
        if self._selected_tool_index is None:
            return None

        # Check active tools first
        if self._selected_block_index is None:
            if 0 <= self._selected_tool_index < len(self._active_tools):
                return self._active_tools[self._selected_tool_index]
            return None

        # Check tool blocks
        block = self._get_selected_block()
        if block and 0 <= self._selected_tool_index < len(block.tools):
            return block.tools[self._selected_tool_index]
        return None

    def scroll_selected_tool_up(self) -> bool:
        """Scroll up within the selected tool's output.

        Returns:
            True if scroll position changed, False if at top or no tool selected.
        """
        tool = self.get_selected_tool()
        if not tool or not tool.expanded or not tool.output_lines:
            return False
        # Scroll offset is from the end, so scrolling "up" means increasing offset
        max_offset = max(0, len(tool.output_lines) - tool.output_display_lines)
        if tool.output_scroll_offset < max_offset:
            tool.output_scroll_offset += 1
            return True
        return False

    def scroll_selected_tool_down(self) -> bool:
        """Scroll down within the selected tool's output.

        Returns:
            True if scroll position changed, False if at bottom or no tool selected.
        """
        tool = self.get_selected_tool()
        if not tool or not tool.expanded or not tool.output_lines:
            return False
        if tool.output_scroll_offset > 0:
            tool.output_scroll_offset -= 1
            return True
        return False

    def finalize_tool_tree(self) -> None:
        """Mark turn as complete and finalize active tools as a ToolBlock.

        Called when a turn is complete. Active tools are stored as a ToolBlock
        that can be navigated and expanded.
        """
        # Flush any pending streaming text first to ensure proper ordering
        self._flush_current_block()

        # Finalize any remaining active tools as a ToolBlock
        if self._active_tools:
            self._finalize_completed_tools()

    def _finalize_completed_tools(self) -> None:
        """Store completed tools as a ToolBlock in the line buffer.

        Called internally when a new tool is added while previous tools are
        all completed. Stores a ToolBlock that can be navigated and expanded.

        The ToolBlock is inserted at the placeholder position (established when
        the first tool was added), ensuring tools appear in chronological order
        after their preceding text.
        """
        if not self._active_tools:
            return

        # Don't finalize if any tool has pending permission or clarification
        any_pending = any(
            tool.permission_state == "pending" or tool.clarification_state == "pending"
            for tool in self._active_tools
        )
        if any_pending:
            return

        # Create a copy of the tools for the ToolBlock
        import copy
        tools_copy = copy.deepcopy(self._active_tools)

        # Create ToolBlock with the completed tools
        tool_block = ToolBlock(
            tools=tools_copy,
            expanded=self._tools_expanded,
            selected_index=None
        )

        # Insert at placeholder position (set when first tool was added)
        insert_pos = self._tool_placeholder_index

        # Create separator and trailing separator lines
        separator_line = OutputLine(
            source="system", text="", style="",
            display_lines=1, is_turn_start=False
        )
        trailing_line = OutputLine(
            source="system", text="", style="",
            display_lines=1, is_turn_start=False
        )

        # Insert: separator, tool_block, trailing separator
        self._lines.insert(insert_pos, separator_line)
        self._lines.insert(insert_pos + 1, tool_block)
        self._lines.insert(insert_pos + 2, trailing_line)

        # Flush any pending enrichment notifications
        # These were queued while tools were active so they appear AFTER the tool tree
        if self._pending_enrichments:
            enrich_pos = insert_pos + 3
            for enrich_source, enrich_text, enrich_mode in self._pending_enrichments:
                for line in enrich_text.split('\n'):
                    display_lines = self._measure_display_lines(enrich_source, line, False)
                    enrich_line = OutputLine(
                        source=enrich_source, text=line, style="line",
                        display_lines=display_lines, is_turn_start=False
                    )
                    self._lines.insert(enrich_pos, enrich_line)
                    enrich_pos += 1
            self._pending_enrichments.clear()

        # Clear placeholder and active tools
        self._tool_placeholder_index = None
        self._active_tools.clear()

        # Auto-scroll to bottom to show finalized content
        # This ensures enrichment notifications and tool results are visible
        self.scroll_to_bottom()

    @property
    def active_tools(self) -> List[ActiveToolCall]:
        """Get list of currently active tools."""
        return list(self._active_tools)

    def set_tool_permission_pending(
        self,
        tool_name: str,
        prompt_lines: List[str],
        format_hint: Optional[str] = None,
    ) -> None:
        """Mark a tool as awaiting permission with the prompt to display.

        Args:
            tool_name: Name of the tool awaiting permission (may be the tool being
                checked, not necessarily the currently executing tool).
            prompt_lines: Lines of the permission prompt to display (may contain ANSI codes).
            format_hint: Optional hint about content format ("diff" = pre-formatted, skip box).
        """
        _trace(f"set_tool_permission_pending: looking for tool={tool_name}")
        _trace(f"set_tool_permission_pending: active_tools={[(t.name, t.completed) for t in self._active_tools]}")

        # First try exact match by tool name
        for tool in self._active_tools:
            if tool.name == tool_name and not tool.completed:
                tool.permission_state = "pending"
                tool.permission_prompt_lines = prompt_lines
                tool.permission_format_hint = format_hint
                # Scroll to bottom to show the prompt
                self._scroll_offset = 0
                _trace(f"set_tool_permission_pending: FOUND exact match for {tool_name}")
                return

        # Fallback: attach to the last uncompleted tool (handles askPermission checking other tools)
        # When askPermission checks permission for "cli_based_tool", the active tool is "askPermission"
        for tool in reversed(self._active_tools):
            if not tool.completed:
                tool.permission_state = "pending"
                tool.permission_prompt_lines = prompt_lines
                tool.permission_format_hint = format_hint
                # If the requested tool is different from the active tool (e.g., askPermission
                # checking permission for writeNewFile), show the actual tool name being checked
                if tool.name != tool_name:
                    tool.display_name = tool_name
                    # Clear the args summary since it contains askPermission's args, not the target tool's
                    tool.display_args_summary = ""
                self._scroll_offset = 0
                _trace(f"set_tool_permission_pending: FALLBACK attached to {tool.name} (requested: {tool_name})")
                return

        _trace(f"set_tool_permission_pending: NO MATCH for {tool_name}")

    def set_tool_permission_resolved(self, tool_name: str, granted: bool,
                                      method: str) -> None:
        """Mark a tool's permission as resolved.

        Args:
            tool_name: Name of the tool (may be the tool being checked, not the executing tool).
            granted: Whether permission was granted.
            method: How permission was resolved (yes, always, once, never, whitelist, etc.)
        """
        _trace(f"set_tool_permission_resolved: looking for tool={tool_name}, granted={granted}")

        # First try exact match by tool name with pending permission
        for tool in self._active_tools:
            if tool.name == tool_name and tool.permission_state == "pending":
                tool.permission_state = "granted" if granted else "denied"
                tool.permission_method = method
                tool.permission_prompt_lines = None  # Clear expanded prompt
                _trace(f"set_tool_permission_resolved: FOUND exact match for {tool_name}")
                return

        # Fallback: find any tool with pending permission state
        # This handles askPermission checking other tools (permission attached to askPermission)
        for tool in self._active_tools:
            if tool.permission_state == "pending":
                tool.permission_state = "granted" if granted else "denied"
                tool.permission_method = method
                tool.permission_prompt_lines = None  # Clear expanded prompt
                _trace(f"set_tool_permission_resolved: FALLBACK resolved {tool.name} (requested: {tool_name})")
                return

        _trace(f"set_tool_permission_resolved: NO PENDING TOOL for {tool_name}")

    def set_tool_clarification_pending(self, tool_name: str, prompt_lines: List[str]) -> None:
        """Mark a tool as awaiting clarification (initial context only).

        Args:
            tool_name: Name of the tool awaiting clarification.
            prompt_lines: Initial context lines (not the questions).
        """
        for tool in self._active_tools:
            if tool.name == tool_name and not tool.completed:
                tool.clarification_state = "pending"
                tool.clarification_prompt_lines = prompt_lines
                tool.clarification_answered = []  # Initialize answered list
                # Scroll to bottom to show the prompt
                self._scroll_offset = 0
                return

    def set_tool_clarification_question(
        self,
        tool_name: str,
        question_index: int,
        total_questions: int,
        question_lines: List[str]
    ) -> None:
        """Set the current question being displayed for clarification.

        Args:
            tool_name: Name of the tool.
            question_index: Current question number (1-based).
            total_questions: Total number of questions.
            question_lines: Lines for this question's prompt.
        """
        for tool in self._active_tools:
            if tool.name == tool_name and not tool.completed:
                tool.clarification_state = "pending"
                tool.clarification_current_question = question_index
                tool.clarification_total_questions = total_questions
                tool.clarification_prompt_lines = question_lines
                if tool.clarification_answered is None:
                    tool.clarification_answered = []
                # Scroll to bottom to show the question
                self._scroll_offset = 0
                return

    def set_tool_question_answered(
        self,
        tool_name: str,
        question_index: int,
        answer_summary: str
    ) -> None:
        """Mark a clarification question as answered.

        Args:
            tool_name: Name of the tool.
            question_index: Question number that was answered (1-based).
            answer_summary: Brief summary of the answer.
        """
        for tool in self._active_tools:
            if tool.name == tool_name:
                if tool.clarification_answered is None:
                    tool.clarification_answered = []
                tool.clarification_answered.append((question_index, answer_summary))
                # Clear prompt lines since question is answered
                tool.clarification_prompt_lines = None
                return

    def set_tool_clarification_resolved(
        self,
        tool_name: str,
        qa_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> None:
        """Mark a tool's clarification as fully resolved.

        Args:
            tool_name: Name of the tool.
            qa_pairs: Optional list of (question, answer) tuples for overview display.
        """
        for tool in self._active_tools:
            if tool.name == tool_name:
                tool.clarification_state = "resolved"
                tool.clarification_prompt_lines = None
                tool.clarification_current_question = 0
                tool.clarification_total_questions = 0
                tool.clarification_summary = qa_pairs
                return

    def get_pending_prompt_for_pager(self) -> Optional[Tuple[str, List[str]]]:
        """Get the pending prompt that's awaiting user input for pager display.

        Returns:
            Tuple of (type, lines) where type is "permission" or "clarification",
            or None if no prompt is pending.
        """
        for tool in self._active_tools:
            if tool.permission_state == "pending" and tool.permission_prompt_lines:
                return ("permission", tool.permission_prompt_lines)
            if tool.clarification_state == "pending" and tool.clarification_prompt_lines:
                return ("clarification", tool.clarification_prompt_lines)
        return None

    def has_truncated_pending_prompt(self) -> bool:
        """Check if there's a truncated prompt awaiting user input.

        Returns:
            True if a truncated permission or clarification prompt is pending.
        """
        for tool in self._active_tools:
            if tool.permission_state == "pending" and tool.permission_truncated:
                return True
            if tool.clarification_state == "pending" and tool.clarification_truncated:
                return True
        return False

    def has_pending_prompt(self) -> bool:
        """Check if there's any pending prompt awaiting user input.

        Returns:
            True if any permission or clarification prompt is pending.
        """
        for tool in self._active_tools:
            if tool.permission_state == "pending" and tool.permission_prompt_lines:
                return True
            if tool.clarification_state == "pending" and tool.clarification_prompt_lines:
                return True
        return False

    def scroll_up(self, lines: int = 5) -> bool:
        """Scroll up (view older content).

        Args:
            lines: Number of display lines to scroll.

        Returns:
            True if scroll position changed.
        """
        # Calculate total display lines (handles both OutputLine and ToolBlock)
        total_display_lines = sum(self._get_item_display_lines(item) for item in self._lines)
        max_offset = max(0, total_display_lines - 1)

        old_offset = self._scroll_offset
        self._scroll_offset = min(self._scroll_offset + lines, max_offset)
        return self._scroll_offset != old_offset

    def scroll_down(self, lines: int = 5) -> bool:
        """Scroll down (view newer content).

        Args:
            lines: Number of display lines to scroll.

        Returns:
            True if scroll position changed.
        """
        old_offset = self._scroll_offset
        self._scroll_offset = max(0, self._scroll_offset - lines)
        return self._scroll_offset != old_offset

    def scroll_to_bottom(self) -> bool:
        """Scroll to the bottom (most recent content).

        Returns:
            True if scroll position changed.
        """
        old_offset = self._scroll_offset
        self._scroll_offset = 0
        return self._scroll_offset != old_offset

    def scroll_to_top(self) -> bool:
        """Scroll to the top (oldest content).

        Returns:
            True if scroll position changed.
        """
        total_display_lines = sum(self._get_item_display_lines(item) for item in self._lines)
        max_offset = max(0, total_display_lines - 1)
        old_offset = self._scroll_offset
        self._scroll_offset = max_offset
        return self._scroll_offset != old_offset

    @property
    def is_at_bottom(self) -> bool:
        """Check if scrolled to the bottom."""
        return self._scroll_offset == 0

    def _calculate_tool_tree_height(self) -> int:
        """Calculate the approximate height of the tool tree in display lines.

        This is used to reserve space when selecting which stored lines to display,
        ensuring the tool tree doesn't push content off the visible area.

        Note: This only calculates height for active tools. ToolBlocks in the
        line buffer are accounted for via _get_item_display_lines().

        Returns:
            Number of display lines the tool tree will occupy.
        """
        if not self._active_tools and not self._spinner_active:
            return 0

        height = 0

        if self._active_tools:
            if self._tools_expanded:
                # Expanded view: header + each tool on its own line
                height += 1  # Header line

                for tool in self._active_tools:
                    height += 1  # Tool line

                    # Output preview lines (persists after tool completes)
                    if tool.output_lines:
                        height += len(tool.output_lines)

                    # Permission denied message
                    if tool.permission_state == "denied" and tool.permission_method:
                        height += 1
                    # Error message (if failed and has error, but not permission denied)
                    elif tool.completed and not tool.success and tool.error_message:
                        height += 1
            else:
                # Collapsed view: just one summary line
                height += 1

            # Check for pending prompts (same logic for both views)
            for tool in self._active_tools:
                # Permission prompt (if pending)
                if tool.permission_state == "pending" and tool.permission_prompt_lines:
                    height += 1  # "Permission required" header

                    prompt_lines = tool.permission_prompt_lines

                    # Pre-formatted content (e.g., diff) - no box, just lines
                    if tool.permission_format_hint == "diff":
                        height += len(prompt_lines)
                    else:
                        # Box calculation
                        max_prompt_lines = 18
                        max_box_width = max(60, self._console_width - 22) if self._console_width > 40 else 60
                        box_width = min(max_box_width, max(len(line) for line in prompt_lines) + 4)
                        content_width = box_width - 4

                        # First count ALL wrapped lines to decide truncation (matches render logic)
                        total_wrapped_lines = 0
                        for prompt_line in prompt_lines:
                            if len(prompt_line) > content_width:
                                wrapped = textwrap.wrap(prompt_line, width=content_width, break_long_words=True)
                                total_wrapped_lines += len(wrapped) if wrapped else 1
                            else:
                                total_wrapped_lines += 1

                        if total_wrapped_lines > max_prompt_lines:
                            # Truncation triggered - render shows:
                            # - max_lines_before_truncation content lines
                            # - 1 truncation message
                            # - Last line (may wrap to multiple lines)
                            max_lines_before_truncation = max_prompt_lines - 3

                            # Calculate wrapped lines for last line
                            last_line = prompt_lines[-1]
                            if len(last_line) > content_width:
                                last_wrapped = textwrap.wrap(last_line, width=content_width, break_long_words=True)
                                last_line_count = len(last_wrapped) if last_wrapped else 1
                            else:
                                last_line_count = 1

                            rendered_lines = max_lines_before_truncation + 1 + last_line_count  # content + truncation + last
                        else:
                            # No truncation - show all wrapped lines
                            rendered_lines = total_wrapped_lines

                        height += 1  # top border
                        height += rendered_lines
                        height += 1  # bottom border

                # Clarification prompt (if pending)
                if tool.clarification_state == "pending":
                    height += 1  # header ("Clarification needed" or progress)

                    # Previously answered questions
                    if tool.clarification_answered:
                        height += len(tool.clarification_answered)

                    # Current question prompt box
                    if tool.clarification_prompt_lines:
                        prompt_lines = tool.clarification_prompt_lines
                        max_prompt_lines = 18
                        max_box_width = max(60, self._console_width - 22) if self._console_width > 40 else 60
                        box_width = min(max_box_width, max(len(line) for line in prompt_lines) + 4)
                        content_width = box_width - 4

                        # First count ALL wrapped lines to decide truncation (matches render logic)
                        total_wrapped_lines = 0
                        for prompt_line in prompt_lines:
                            if len(prompt_line) > content_width:
                                wrapped = textwrap.wrap(prompt_line, width=content_width, break_long_words=True)
                                total_wrapped_lines += len(wrapped) if wrapped else 1
                            else:
                                total_wrapped_lines += 1

                        if total_wrapped_lines > max_prompt_lines:
                            # Truncation triggered - render shows:
                            # - max_lines_before_truncation content lines
                            # - 1 truncation message
                            # - Last line (may wrap to multiple lines)
                            max_lines_before_truncation = max_prompt_lines - 3

                            # Calculate wrapped lines for last line
                            last_line = prompt_lines[-1]
                            if len(last_line) > content_width:
                                last_wrapped = textwrap.wrap(last_line, width=content_width, break_long_words=True)
                                last_line_count = len(last_wrapped) if last_wrapped else 1
                            else:
                                last_line_count = 1

                            rendered_lines = max_lines_before_truncation + 1 + last_line_count  # content + truncation + last
                        else:
                            # No truncation - show all wrapped lines
                            rendered_lines = total_wrapped_lines

                        height += 1  # top border
                        height += rendered_lines
                        height += 1  # bottom border

                # Clarification summary (if resolved with answers)
                if tool.completed and tool.clarification_summary:
                    height += 1  # header ("Answers (N)")
                    height += len(tool.clarification_summary)  # One line per Q&A pair

        elif self._spinner_active:
            # Spinner alone (no tools) - just 1 line
            height += 1

        return height

    def _get_item_display_lines(self, item: Union[OutputLine, ToolBlock, ActiveToolsMarker]) -> int:
        """Get display line count for a line item (OutputLine, ToolBlock, or ActiveToolsMarker)."""
        if isinstance(item, OutputLine):
            return item.display_lines
        elif isinstance(item, ToolBlock):
            return self._calculate_tool_block_height(item)
        elif isinstance(item, ActiveToolsMarker):
            # Active tools marker takes the same space as the active tool tree
            return self._calculate_tool_tree_height()
        return 1

    def _calculate_tool_block_height(self, block: ToolBlock) -> int:
        """Calculate display height of a ToolBlock."""
        height = 1  # Separator line
        if block.expanded:
            height += 1  # Header
            for tool in block.tools:
                height += 1  # Tool line
                # Output preview if tool has output and is expanded
                if tool.expanded and tool.output_lines:
                    display_count = min(len(tool.output_lines), tool.output_display_lines)
                    height += display_count
                    # Scroll indicators
                    if len(tool.output_lines) > tool.output_display_lines:
                        height += 2
                # Error message
                if not tool.success and tool.error_message:
                    height += 1
        else:
            height += 1  # Collapsed summary line
        return height

    def _render_active_tools_inline(self, output: Text, wrap_width: int) -> None:
        """Render active tools at their placeholder position (inline in the output flow).

        This is called when rendering an ActiveToolsMarker item, so that active tools
        appear at their chronological position rather than always at the bottom.
        """
        if not self._active_tools:
            return

        # Separator with navigation hints
        if self._tool_nav_active:
            pos = (self._selected_tool_index or 0) + 1
            total = len(self._active_tools)
            selected_tool = self._active_tools[self._selected_tool_index or 0]
            nav_up = self._format_key_hint("nav_up")
            nav_down = self._format_key_hint("nav_down")
            expand_key = self._format_key_hint("tool_expand")
            collapse_key = self._format_key_hint("tool_collapse")
            exit_key = self._format_key_hint("tool_exit")
            has_output = selected_tool.output_lines and len(selected_tool.output_lines) > 0
            if selected_tool.expanded and has_output:
                output.append(f"  ───  {nav_up}/{nav_down} scroll, {collapse_key} collapse, {exit_key} exit [{pos}/{total}]", style=self._style("hint", "dim"))
            elif has_output:
                output.append(f"  ───  {nav_up}/{nav_down} nav, {expand_key} expand, {exit_key} exit [{pos}/{total}]", style=self._style("hint", "dim"))
            else:
                output.append(f"  ───  {nav_up}/{nav_down} nav, {exit_key} exit [{pos}/{total}]", style=self._style("hint", "dim"))
        elif self._tools_expanded:
            toggle_tools = self._format_key_hint("toggle_tools")
            tool_nav = self._format_key_hint("tool_nav_enter")
            output.append(f"  ───  {toggle_tools} to collapse, {tool_nav} to navigate", style=self._style("hint", "dim"))
        else:
            toggle_tools = self._format_key_hint("toggle_tools")
            output.append(f"  ───  {toggle_tools} to expand", style=self._style("hint", "dim"))

        output.append("\n")

        # Check for pending tool
        pending_tool = None
        for tool in self._active_tools:
            if tool.permission_state == "pending" or tool.clarification_state == "pending":
                pending_tool = tool
                break

        tool_count = len(self._active_tools)
        any_uncompleted = any(not tool.completed for tool in self._active_tools)
        all_completed = all(tool.completed for tool in self._active_tools)
        show_spinner = self._spinner_active or any_uncompleted or (all_completed and not pending_tool)

        if self._tools_expanded:
            # Expanded view - show each tool on its own line
            if pending_tool:
                output.append("  ⏳ ", style=self._style("tool_pending", "bold yellow"))
            elif show_spinner:
                frame = self.SPINNER_FRAMES[self._spinner_index]
                output.append(f"  {frame} ", style=self._style("spinner", "cyan"))
            else:
                output.append("  ▾ ", style=self._style("tool_border", "dim"))
            output.append(f"{tool_count} tool{'s' if tool_count != 1 else ''}:", style=self._style("tool_border", "dim"))

            for i, tool in enumerate(self._active_tools):
                is_last = (i == len(self._active_tools) - 1)
                is_selected = (self._tool_nav_active and i == self._selected_tool_index)
                connector = "└─" if is_last else "├─"

                if tool.completed:
                    status_icon = "✓" if tool.success else "✗"
                    status_style = self._style("tool_success", "green") if tool.success else self._style("tool_error", "red")
                else:
                    status_icon = "○"
                    status_style = self._style("muted", "dim")

                expand_icon = "▾" if tool.expanded else "▸" if self._tool_nav_active else ""
                row_style = "reverse" if is_selected else self._style("muted", "dim")

                output.append("\n")
                if self._tool_nav_active:
                    output.append(f"  {expand_icon} {connector} ", style=row_style)
                else:
                    output.append(f"    {connector} ", style=row_style)

                tool_display_name = tool.display_name or tool.name
                output.append(tool_display_name, style=row_style)
                args_to_show = tool.display_args_summary if tool.display_args_summary is not None else tool.args_summary
                if args_to_show:
                    output.append(f"({args_to_show})", style=row_style)
                output.append(f" {status_icon}", style=status_style)

                if tool.permission_state == "granted" and tool.permission_method:
                    indicator = self._get_approval_indicator(tool.permission_method)
                    if indicator:
                        output.append(f" {indicator}", style=self._style("tool_indicator", "dim cyan"))

                if tool.completed and tool.duration_seconds is not None:
                    output.append(f" ({tool.duration_seconds:.1f}s)", style=self._style("tool_duration", "dim"))

                # Tool output preview
                show_output = tool.expanded if self._tool_nav_active else True
                if show_output and tool.output_lines:
                    self._render_tool_output_lines(output, tool, is_last)

                # Permission/error info
                if tool.permission_state == "denied" and tool.permission_method:
                    output.append("\n")
                    continuation = "   " if is_last else "│  "
                    output.append(f"    {continuation} ", style=self._style("tree_connector", "dim"))
                    output.append(f"  ⊘ Permission denied: User chose: {tool.permission_method}", style=self._style("permission_denied", "red dim"))
                elif tool.completed and not tool.success and tool.error_message:
                    output.append("\n")
                    continuation = "   " if is_last else "│  "
                    output.append(f"    {continuation} ", style=self._style("tree_connector", "dim"))
                    output.append(f"  ⚠ {tool.error_message}", style=self._style("tool_error", "red dim"))

                # Permission prompt
                if tool.permission_state == "pending" and tool.permission_prompt_lines:
                    self._render_permission_prompt(output, tool, is_last)

                # Clarification prompt
                if tool.clarification_state == "pending" and tool.clarification_prompt_lines:
                    self._render_clarification_prompt(output, tool, is_last)
        else:
            # Collapsed view
            if pending_tool:
                output.append("  ⏳ ", style=self._style("tool_pending", "bold yellow"))
            elif show_spinner:
                frame = self.SPINNER_FRAMES[self._spinner_index]
                output.append(f"  {frame} ", style=self._style("spinner", "cyan"))
            else:
                output.append("  ▸ ", style=self._style("tool_border", "dim"))

            completed = sum(1 for t in self._active_tools if t.completed)
            if completed == tool_count:
                output.append(f"{tool_count} tool{'s' if tool_count != 1 else ''} ✓", style=self._style("tool_border", "dim"))
            else:
                output.append(f"{completed}/{tool_count} tools", style=self._style("tool_border", "dim"))

    def _render_tool_output_lines(self, output: Text, tool: 'ActiveToolCall', is_last: bool) -> None:
        """Render output lines for a tool (shared helper)."""
        continuation = "   " if is_last else "│  "
        prefix = "    "
        total_lines = len(tool.output_lines)
        display_count = tool.output_display_lines

        end_idx = total_lines - tool.output_scroll_offset
        start_idx = max(0, end_idx - display_count)
        lines_above = start_idx
        lines_below = tool.output_scroll_offset

        if lines_above > 0:
            output.append("\n")
            output.append(f"{prefix}{continuation}   ", style=self._style("tree_connector", "dim"))
            scroll_up_key = self._format_key_hint("nav_up")
            output.append(f"▲ {lines_above} more line{'s' if lines_above != 1 else ''} ({scroll_up_key} to scroll)", style=self._style("scroll_indicator", "dim italic"))

        for output_line in tool.output_lines[start_idx:end_idx]:
            output.append("\n")
            output.append(f"{prefix}{continuation}   ", style=self._style("tree_connector", "dim"))
            max_line_width = max(40, self._console_width - 20) if self._console_width > 60 else 40
            if len(output_line) > max_line_width:
                display_line = output_line[:max_line_width - 3] + "..."
            else:
                display_line = output_line
            output.append(display_line, style=self._style("tool_output", "#87D7D7 italic"))

        if lines_below > 0:
            output.append("\n")
            output.append(f"{prefix}{continuation}   ", style=self._style("tree_connector", "dim"))
            scroll_down_key = self._format_key_hint("nav_down")
            output.append(f"▼ {lines_below} more line{'s' if lines_below != 1 else ''} ({scroll_down_key} to scroll)", style=self._style("scroll_indicator", "dim italic"))

    def _render_tool_block(self, block: ToolBlock, output: Text, wrap_width: int) -> None:
        """Render a ToolBlock inline in the output."""
        tool_count = len(block.tools)

        # Separator
        output.append("  ───", style=self._style("separator", "dim"))

        # Navigation hint if in nav mode and this block is selected
        if self._tool_nav_active and block.selected_index is not None:
            all_tools = self._get_all_tools_flat()
            # Find position in all tools
            pos = 0
            for i, (bidx, tidx, _) in enumerate(all_tools):
                if bidx == self._selected_block_index and tidx == self._selected_tool_index:
                    pos = i + 1
                    break
            total = len(all_tools)
            # Dynamic hint based on selected tool's expanded state and actual keybindings
            selected_tool = block.tools[block.selected_index]
            nav_up = self._format_key_hint("nav_up")
            nav_down = self._format_key_hint("nav_down")
            expand_key = self._format_key_hint("tool_expand")
            collapse_key = self._format_key_hint("tool_collapse")
            exit_key = self._format_key_hint("tool_exit")
            has_output = selected_tool.output_lines and len(selected_tool.output_lines) > 0
            if selected_tool.expanded and has_output:
                # When expanded: arrows scroll output, left collapses
                output.append(f"  {nav_up}/{nav_down} scroll, {collapse_key} collapse, {exit_key} exit [{pos}/{total}]", style=self._style("hint", "dim"))
            elif has_output:
                # When collapsed but has output: arrows navigate, right expands
                output.append(f"  {nav_up}/{nav_down} nav, {expand_key} expand, {exit_key} exit [{pos}/{total}]", style=self._style("hint", "dim"))
            else:
                # No output: just navigation hints
                output.append(f"  {nav_up}/{nav_down} nav, {exit_key} exit [{pos}/{total}]", style=self._style("hint", "dim"))

        output.append("\n")

        if block.expanded:
            # Expanded view - each tool on its own line
            output.append(f"  ▾ {tool_count} tool{'s' if tool_count != 1 else ''}:", style=self._style("tool_border", "dim"))

            for i, tool in enumerate(block.tools):
                is_last = (i == len(block.tools) - 1)
                is_selected = (block.selected_index == i)
                connector = "└─" if is_last else "├─"

                status_icon = "✓" if tool.success else "✗"
                status_style = self._style("tool_success", "green") if tool.success else self._style("tool_error", "red")

                # Expand indicator for tool output (only if tool has output)
                if tool.output_lines:
                    expand_icon = "▾" if tool.expanded else "▸"
                else:
                    expand_icon = " "

                # Selection highlight
                row_style = "reverse" if is_selected else self._style("muted", "dim")

                output.append("\n")
                output.append(f"  {expand_icon} {connector} ", style=row_style)
                # Use display_name if set (e.g., showing actual tool instead of askPermission)
                tool_display_name = tool.display_name or tool.name
                output.append(tool_display_name, style=row_style)
                # Use display_args_summary if set, otherwise fall back to args_summary
                args_to_show = tool.display_args_summary if tool.display_args_summary is not None else tool.args_summary
                if args_to_show:
                    output.append(f"({args_to_show})", style=row_style)
                output.append(f" {status_icon}", style=status_style)

                # Approval indicator
                if tool.permission_state == "granted" and tool.permission_method:
                    indicator = self._get_approval_indicator(tool.permission_method)
                    if indicator:
                        output.append(f" {indicator}", style=self._style("tool_indicator", "dim cyan"))

                # Duration
                if tool.duration_seconds is not None:
                    output.append(f" ({tool.duration_seconds:.1f}s)", style=self._style("tool_duration", "dim"))

                # Expanded output
                if tool.expanded and tool.output_lines:
                    continuation = "   " if is_last else "│  "
                    prefix = "    "
                    total_lines = len(tool.output_lines)
                    display_count = tool.output_display_lines

                    end_idx = total_lines - tool.output_scroll_offset
                    start_idx = max(0, end_idx - display_count)

                    lines_above = start_idx
                    lines_below = tool.output_scroll_offset

                    if lines_above > 0:
                        output.append("\n")
                        output.append(f"{prefix}{continuation}   ", style=self._style("tree_connector", "dim"))
                        scroll_up_key = self._format_key_hint("nav_up")
                        output.append(f"▲ {lines_above} more line{'s' if lines_above != 1 else ''} ({scroll_up_key} to scroll)", style=self._style("scroll_indicator", "dim italic"))

                    for output_line in tool.output_lines[start_idx:end_idx]:
                        output.append("\n")
                        output.append(f"{prefix}{continuation}   ", style=self._style("tree_connector", "dim"))
                        max_line_width = max(40, self._console_width - 20) if self._console_width > 60 else 40
                        if len(output_line) > max_line_width:
                            display_line = output_line[:max_line_width - 3] + "..."
                        else:
                            display_line = output_line
                        output.append(display_line, style=self._style("tool_output", "#87D7D7 italic"))

                    if lines_below > 0:
                        output.append("\n")
                        output.append(f"{prefix}{continuation}   ", style=self._style("tree_connector", "dim"))
                        scroll_down_key = self._format_key_hint("nav_down")
                        output.append(f"▼ {lines_below} more line{'s' if lines_below != 1 else ''} ({scroll_down_key} to scroll)", style=self._style("scroll_indicator", "dim italic"))

                # Error message
                if not tool.success and tool.error_message:
                    output.append("\n")
                    continuation = "   " if is_last else "│  "
                    output.append(f"    {continuation}   ⚠ {tool.error_message}", style=self._style("tool_error", "red dim"))
        else:
            # Collapsed view
            tool_summaries = []
            for tool in block.tools:
                status_icon = "✓" if tool.success else "✗"
                tool_display_name = tool.display_name or tool.name
                summary = f"{tool_display_name} {status_icon}"
                if tool.permission_state == "granted" and tool.permission_method:
                    indicator = self._get_approval_indicator(tool.permission_method)
                    if indicator:
                        summary += f" {indicator}"
                tool_summaries.append(summary)

            output.append(f"  ▸ {tool_count} tool{'s' if tool_count != 1 else ''}: ", style=self._style("tool_border", "dim"))
            output.append(" ".join(tool_summaries), style=self._style("tool_border", "dim"))

    def _get_cached_line_content(self, line: OutputLine, wrap_width: int) -> Optional[Text]:
        """Get cached rendered content for a line, or None if cache invalid.

        This avoids expensive Text.from_ansi() and text wrapping on each render
        for lines that haven't changed.

        Args:
            line: The OutputLine to get cached content for.
            wrap_width: Current wrap width (cache invalid if different).

        Returns:
            Cached Text object if valid, None otherwise.
        """
        if line._rendered_cache is not None and line._cache_width == wrap_width:
            return line._rendered_cache
        return None

    def _cache_line_content(self, line: OutputLine, content: Text, wrap_width: int) -> None:
        """Store rendered content in line cache.

        Args:
            line: The OutputLine to cache content for.
            content: The rendered Text object.
            wrap_width: Current wrap width.
        """
        line._rendered_cache = content
        line._cache_width = wrap_width

    def render(self, height: Optional[int] = None, width: Optional[int] = None) -> RenderableType:
        """Render the output buffer as Rich Text.

        Args:
            height: Optional height limit (in display lines).
            width: Optional width for calculating line wrapping.

        Returns:
            Rich renderable for the output panel.
        """
        # Set rendering guard to prevent flush during render cycle
        self._rendering = True
        try:
            return self._render_impl(height, width)
        finally:
            self._rendering = False

    def _render_impl(self, height: Optional[int] = None, width: Optional[int] = None) -> RenderableType:
        """Internal render implementation (called within rendering guard)."""
        # Update width BEFORE measuring current block lines (important for accurate measurement)
        if width and width != self._console_width:
            self.set_width(width)

        # Get current block lines without flushing (preserves streaming state)
        current_block_lines = self._get_current_block_lines()

        # If buffer is empty but spinner is active, show only spinner
        if not self._lines and not current_block_lines:
            if self._spinner_active:
                output = Text()
                frame = self.SPINNER_FRAMES[self._spinner_index]
                output.append(f"  {frame} ", style=self._style("spinner", "cyan"))
                output.append("thinking...", style=self._style("hint", "dim italic"))
                return output
            return Text("Waiting for output...", style=self._style("hint", "dim italic"))

        # Store visible height for auto-scroll calculations
        if height:
            self._visible_height = height

        # Work backwards from the end, using stored display line counts
        # First skip _scroll_offset lines, then collect 'height' lines
        # Include current block lines (streaming content) at the end
        all_items: List[Union[OutputLine, ToolBlock, ActiveToolsMarker]] = list(self._lines) + current_block_lines

        # Insert ActiveToolsMarker at placeholder position if active tools exist
        if self._active_tools and self._tool_placeholder_index is not None:
            # Insert marker at the placeholder position
            # Account for current_block_lines being appended at the end
            insert_pos = min(self._tool_placeholder_index, len(all_items))
            all_items.insert(insert_pos, ActiveToolsMarker())

        items_to_show: List[Union[OutputLine, ToolBlock, ActiveToolsMarker]] = []

        if height:
            # Tools render inline via ActiveToolsMarker - no separate space reservation needed
            # The marker's height is calculated in _get_item_display_lines()
            available_for_lines = height

            # Calculate total display lines (accounting for ToolBlocks)
            total_display_lines = sum(self._get_item_display_lines(item) for item in all_items)

            # Find the end position (bottom of visible window)
            # scroll_offset=0 means show the most recent content
            # scroll_offset>0 means we've scrolled up, showing older content
            end_display_line = total_display_lines - self._scroll_offset
            start_display_line = max(0, end_display_line - available_for_lines)

            # Collect items that fall within the visible range
            current_display_line = 0
            for item in all_items:
                item_height = self._get_item_display_lines(item)
                line_end = current_display_line + item_height
                # Include item if it overlaps with visible range
                if line_end > start_display_line and current_display_line < end_display_line:
                    items_to_show.append(item)
                current_display_line = line_end
                # Stop if we've passed the visible range
                if current_display_line >= end_display_line:
                    break
        else:
            items_to_show = all_items

        # Build output text with wrapping
        output = Text()

        # Calculate available width for content (accounting for prefixes)
        wrap_width = self._console_width if self._console_width > 20 else 80

        # Define wrap_text helper before the loops
        def wrap_text(text: str, prefix_width: int = 0) -> List[str]:
            """Wrap text to console width, accounting for prefix.

            Handles multi-line text by splitting on newlines first,
            then wrapping each line individually.
            """
            available = max(20, wrap_width - prefix_width)

            # Handle literal \n in text (escaped newlines) - convert to actual newlines
            text = text.replace('\\n', '\n')

            # Split on actual newlines first, then wrap each line
            result = []
            for paragraph in text.split('\n'):
                if not paragraph.strip():
                    # Preserve empty lines (paragraph breaks)
                    result.append('')
                elif len(paragraph) <= available:
                    result.append(paragraph)
                else:
                    # Use textwrap for clean word-based wrapping
                    wrapped = textwrap.wrap(paragraph, width=available, break_long_words=True, break_on_hyphens=False)
                    result.extend(wrapped)
            return result if result else ['']

        # Render items (OutputLines, ToolBlocks, and ActiveToolsMarker)
        for i, item in enumerate(items_to_show):
            if i > 0:
                output.append("\n")
                # Add extra blank line before turn_start items for visual separation
                if isinstance(item, OutputLine) and item.is_turn_start:
                    output.append("\n")

            # Handle ToolBlocks specially
            if isinstance(item, ToolBlock):
                self._render_tool_block(item, output, wrap_width)
                continue

            # Handle ActiveToolsMarker - render active tools inline at their position
            if isinstance(item, ActiveToolsMarker):
                self._render_active_tools_inline(output, wrap_width)
                continue

            # For OutputLine items, render based on source
            line = item
            if line.source == "system":
                # System messages resolve style through theme (semantic or fallback to raw)
                resolved_style = self._style(line.style, line.style)
                wrapped = wrap_text(line.text)
                for j, wrapped_line in enumerate(wrapped):
                    if j > 0:
                        output.append("\n")
                    output.append(wrapped_line, style=resolved_style)
            elif line.source in ("user", "parent"):
                # User/parent input - use header line for turn start
                if line.is_turn_start:
                    # Render header line: ── User ── or ── Parent ──
                    # Note: blank line for visual separation is added at inter-item level above
                    # "parent" source means message from parent agent to this subagent
                    user_label = "Parent" if line.source == "parent" else "User"
                    header_prefix = f"── {user_label} "
                    remaining = max(0, wrap_width - len(header_prefix))
                    output.append(header_prefix, style=self._style("user_header", "bold green"))
                    output.append("─" * remaining, style=self._style("user_header_separator", "dim green"))
                    output.append("\n")
                    # Then render the text content
                    wrapped = wrap_text(line.text, 0)
                    for j, wrapped_line in enumerate(wrapped):
                        if j > 0:
                            output.append("\n")
                        output.append(wrapped_line)
                else:
                    # Non-turn-start - just render text
                    wrapped = wrap_text(line.text, 0)
                    for j, wrapped_line in enumerate(wrapped):
                        if j > 0:
                            output.append("\n")
                        output.append(wrapped_line)
            elif line.source == "model":
                # Model output - use header line for turn start
                # Text may contain ANSI codes from output formatter (syntax highlighting)
                has_ansi = '\x1b[' in line.text
                if line.is_turn_start:
                    # Render header line: ── Model ─────────────────
                    # Note: blank line for visual separation is added at inter-item level above
                    header_prefix = "── Model "
                    remaining = max(0, wrap_width - len(header_prefix))
                    output.append(header_prefix, style=self._style("model_header", "bold cyan"))
                    output.append("─" * remaining, style=self._style("model_header_separator", "dim cyan"))
                    output.append("\n")
                    # Then render the text content (no prefix needed)
                    # Use cache for expensive ANSI parsing
                    cached = self._get_cached_line_content(line, wrap_width)
                    if cached is not None:
                        output.append_text(cached)
                    elif has_ansi:
                        # Text contains ANSI codes from syntax highlighting
                        content = Text.from_ansi(line.text)
                        self._cache_line_content(line, content, wrap_width)
                        output.append_text(content)
                    else:
                        wrapped = wrap_text(line.text, 0)
                        content = Text()
                        for j, wrapped_line in enumerate(wrapped):
                            if j > 0:
                                content.append("\n")
                            content.append(wrapped_line)
                        self._cache_line_content(line, content, wrap_width)
                        output.append_text(content)
                else:
                    # Non-turn-start - just render text, no prefix
                    # Use cache for expensive ANSI parsing
                    cached = self._get_cached_line_content(line, wrap_width)
                    if cached is not None:
                        output.append_text(cached)
                    elif has_ansi:
                        # Text contains ANSI codes from syntax highlighting
                        content = Text.from_ansi(line.text)
                        self._cache_line_content(line, content, wrap_width)
                        output.append_text(content)
                    else:
                        wrapped = wrap_text(line.text, 0)
                        content = Text()
                        for j, wrapped_line in enumerate(wrapped):
                            if j > 0:
                                content.append("\n")
                            content.append(wrapped_line)
                        self._cache_line_content(line, content, wrap_width)
                        output.append_text(content)
            elif line.source == "tool":
                # Tool output
                prefix_width = len(f"[{line.source}] ") if line.is_turn_start else 0
                wrapped = wrap_text(line.text, prefix_width)
                for j, wrapped_line in enumerate(wrapped):
                    if j > 0:
                        output.append("\n")
                    if j == 0 and line.is_turn_start:
                        output.append(f"[{line.source}] ", style=self._style("tool_source_label", "dim yellow"))
                    elif j > 0 and line.is_turn_start:
                        output.append(" " * (len(f"[{line.source}] ")))  # Indent continuation
                    output.append(wrapped_line, style=self._style("muted", "dim"))
            elif line.source == "permission":
                # Permission prompts - wrap but preserve ANSI codes
                text = line.text
                if "[askPermission]" in text:
                    text = text.replace("[askPermission]", "")
                    wrapped = wrap_text(text, 16)  # "[askPermission] " = 16 chars
                    for j, wrapped_line in enumerate(wrapped):
                        if j > 0:
                            output.append("\n                ")  # Indent continuation
                        if j == 0:
                            output.append("[askPermission] ", style=self._style("permission_prompt", "bold yellow"))
                        output.append_text(Text.from_ansi(wrapped_line))
                elif "Options:" in text or text.startswith(("===", "─", "=")) or "Enter choice" in text:
                    # Special lines - wrap normally
                    wrapped = wrap_text(text)
                    for j, wrapped_line in enumerate(wrapped):
                        if j > 0:
                            output.append("\n")
                        if "Options:" in text:
                            output.append_text(Text.from_ansi(wrapped_line, style=self._style("clarification_label", "cyan")))
                        elif text.startswith(("===", "─", "=")):
                            output.append(wrapped_line, style=self._style("separator", "dim"))
                        else:
                            output.append_text(Text.from_ansi(wrapped_line, style=self._style("clarification_label", "cyan")))
                else:
                    # Preserve ANSI codes with wrapping
                    wrapped = wrap_text(text)
                    for j, wrapped_line in enumerate(wrapped):
                        if j > 0:
                            output.append("\n")
                        output.append_text(Text.from_ansi(wrapped_line))
            elif line.source == "clarification":
                # Clarification prompts - wrap but preserve ANSI codes
                text = line.text
                wrapped = wrap_text(text)
                for j, wrapped_line in enumerate(wrapped):
                    if j > 0:
                        output.append("\n")
                    if "Clarification Needed" in wrapped_line:
                        output.append_text(Text.from_ansi(wrapped_line, style=self._style("clarification_label", "bold cyan")))
                    elif wrapped_line.startswith(("===", "─", "=")):
                        output.append(wrapped_line, style=self._style("separator", "dim"))
                    elif "Enter choice" in wrapped_line:
                        output.append_text(Text.from_ansi(wrapped_line, style=self._style("clarification_label", "cyan")))
                    elif wrapped_line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
                        output.append_text(Text.from_ansi(wrapped_line, style=self._style("clarification_label", "cyan")))
                    elif "Question" in wrapped_line and "/" in wrapped_line:
                        output.append_text(Text.from_ansi(wrapped_line, style=self._style("emphasis", "bold")))
                    elif "[*required]" in wrapped_line:
                        wrapped_line = wrapped_line.replace("[*required]", "")
                        output.append_text(Text.from_ansi(wrapped_line))
                        output.append("[*required]", style=self._style("clarification_required", "yellow"))
                    else:
                        output.append_text(Text.from_ansi(wrapped_line))
            elif line.source == "enrichment":
                # Enrichment notifications - render dimmed with proper wrapping
                # The formatter pre-aligns continuation lines, so we wrap each line
                wrapped = wrap_text(line.text)
                for j, wrapped_line in enumerate(wrapped):
                    if j > 0:
                        output.append("\n")
                    output.append(wrapped_line, style=self._style("muted", "dim"))
            elif line.source == "thinking":
                # Extended thinking output - render with header/footer and indentation
                # This is the model's internal reasoning before generating response
                # Box characters: ┌ (top-left), │ (vertical), └ (bottom-left), ┘ (bottom-right)
                border = "│ "
                border_width = len(border)
                if line.is_turn_start:
                    # First render Model header (thinking is part of model turn)
                    header_prefix = "── Model "
                    remaining = max(0, wrap_width - len(header_prefix))
                    output.append(header_prefix, style=self._style("model_header", "bold cyan"))
                    output.append("─" * remaining, style=self._style("model_header_separator", "dim cyan"))
                    output.append("\n")
                    # Then render thinking header: ┌─ Internal thinking ───────┐
                    thinking_header = "┌─ Internal thinking "
                    remaining = max(0, wrap_width - len(thinking_header) - 1)
                    output.append(thinking_header, style=self._style("thinking_header", "dim #D7AF5F"))
                    output.append("─" * remaining, style=self._style("thinking_header_separator", "dim #D7AF5F"))
                    output.append("┐", style=self._style("thinking_header", "dim #D7AF5F"))
                    output.append("\n")
                # Render thinking content with border (aligned with box)
                wrapped = wrap_text(line.text, border_width)
                for j, wrapped_line in enumerate(wrapped):
                    if j > 0:
                        output.append("\n")
                    output.append(border, style=self._style("thinking_border", "dim #D7AF5F"))
                    output.append(wrapped_line, style=self._style("thinking_content", "italic #D7AF87"))
                # Track that we're in thinking mode for footer rendering
                self._in_thinking = True
                # Check if this is the last thinking line in the visible items
                # to render footer (look ahead to next item)
                is_last_thinking = (i == len(items_to_show) - 1) or (
                    i + 1 < len(items_to_show) and items_to_show[i + 1].source != "thinking"
                )
                if is_last_thinking:
                    # Render footer: └─────────────────────────────────────────┘
                    output.append("\n")
                    remaining = max(0, wrap_width - 2)  # -2 for └ and ┘
                    output.append("└", style=self._style("thinking_footer", "dim #D7AF5F"))
                    output.append("─" * remaining, style=self._style("thinking_footer_separator", "dim #D7AF5F"))
                    output.append("┘", style=self._style("thinking_footer", "dim #D7AF5F"))
                    self._in_thinking = False
            else:
                # Other plugin output - wrap and preserve ANSI codes
                # Use cache for expensive ANSI parsing (only for non-turn-start simple cases)
                cached = self._get_cached_line_content(line, wrap_width) if not line.is_turn_start else None
                if cached is not None:
                    output.append_text(cached)
                else:
                    prefix_width = len(f"[{line.source}] ") if line.is_turn_start else 0
                    wrapped = wrap_text(line.text, prefix_width)
                    content = Text()
                    for j, wrapped_line in enumerate(wrapped):
                        if j > 0:
                            content.append("\n")
                            output.append("\n")
                        if j == 0 and line.is_turn_start:
                            output.append(f"[{line.source}] ", style=self._style("tool_source_label", "dim magenta"))
                        elif j > 0 and line.is_turn_start:
                            output.append(" " * (len(f"[{line.source}] ")))  # Indent continuation
                        parsed = Text.from_ansi(wrapped_line)
                        output.append_text(parsed)
                        if not line.is_turn_start:
                            content.append_text(parsed)
                    # Cache only non-turn-start lines (turn-start has prefix that varies)
                    if not line.is_turn_start:
                        self._cache_line_content(line, content, wrap_width)

        # Show spinner when model is thinking (no tools yet)
        if self._spinner_active and not self._active_tools:
            # Spinner active but no tools yet - show model header first
            if items_to_show:
                output.append("\n\n")  # Blank line before header
                # Show model header if this is a new turn
                if self._last_turn_source != "model":
                    header_prefix = "── Model "
                    remaining = max(0, wrap_width - len(header_prefix))
                    output.append(header_prefix, style=self._style("model_header", "bold cyan"))
                    output.append("─" * remaining, style=self._style("model_header_separator", "dim cyan"))
                    output.append("\n")
            frame = self.SPINNER_FRAMES[self._spinner_index]
            output.append(f"  {frame} ", style=self._style("spinner", "cyan"))
            output.append("thinking...", style=self._style("hint", "dim italic"))

        return output

    def render_panel(self, height: Optional[int] = None, width: Optional[int] = None) -> Panel:
        """Render as a Panel.

        Args:
            height: Optional height limit (for content lines, not panel height).
            width: Optional width for calculating line wrapping.

        Returns:
            Panel containing the output.
        """
        # Account for panel border (2 lines) and padding when calculating content height
        content_height = (height - 2) if height else None
        # Account for panel borders (2 chars each side) when calculating content width
        content_width = (width - 4) if width else None

        content = self.render(content_height, content_width)

        # Use Align to push content to bottom of panel
        aligned_content = Align(content, vertical="bottom")

        # Show scroll indicator in title when not at bottom
        if self._scroll_offset > 0:
            title = f"[bold]Output[/bold] [dim](↑{self._scroll_offset} lines)[/dim]"
        else:
            title = "[bold]Output[/bold]"

        return Panel(
            aligned_content,
            title=title,
            border_style=self._style("panel_border", "blue"),
            height=height,  # Constrain panel to exact height
            width=width,  # Constrain panel to exact width (preserves right border)
        )

    # -------------------------------------------------------------------------
    # Chrome-free text extraction (for clipboard operations)
    # -------------------------------------------------------------------------

    def _should_include_source(self, source: str, sources: Optional[set] = None) -> bool:
        """Check if a source should be included based on filter.

        Args:
            source: The source type to check.
            sources: Set of sources to include. If None, includes all.

        Returns:
            True if the source should be included.
        """
        if sources is None:
            return True
        if source in sources:
            return True
        # Handle prefix matching (e.g., "tool" matches "tool:grep")
        if ":" in source:
            prefix = source.split(":")[0]
            if prefix in sources:
                return True
        return False

    def get_last_response_text(self, sources: Optional[set] = None) -> Optional[str]:
        """Get text from the last model response (chrome-free).

        Extracts text from the most recent contiguous block of model output,
        excluding turn headers and other UI chrome.

        Args:
            sources: Set of sources to include (default: {"model"}).

        Returns:
            The extracted text, or None if no matching content.
        """
        if sources is None:
            sources = {"model"}

        # Get all items including current streaming block
        all_items = list(self._lines) + self._get_current_block_lines()

        if not all_items:
            return None

        # Find the last contiguous block of matching source lines
        result_lines: List[str] = []
        in_block = False

        # Walk backwards to find the last matching block
        for item in reversed(all_items):
            # Skip ToolBlocks - they don't have text content
            if isinstance(item, ToolBlock):
                if in_block:
                    # We've hit a ToolBlock while in a block, stop
                    break
                continue
            # item is an OutputLine
            if self._should_include_source(item.source, sources):
                result_lines.append(item.text)
                in_block = True
            elif in_block:
                # We've exited the block, stop
                break

        if not result_lines:
            return None

        # Reverse since we walked backwards
        result_lines.reverse()

        # Join and clean up
        text = "\n".join(result_lines)
        return text.strip() if text.strip() else None

    def get_text_in_line_range(
        self,
        start_line: int,
        end_line: int,
        sources: Optional[set] = None
    ) -> Optional[str]:
        """Get text from a range of display lines (chrome-free).

        Used for mouse selection - maps screen coordinates to content.

        Args:
            start_line: Start display line (0-indexed from top of buffer).
            end_line: End display line (inclusive).
            sources: Set of sources to include. If None, includes all.

        Returns:
            The extracted text, or None if no matching content.
        """
        all_items = list(self._lines) + self._get_current_block_lines()

        if not all_items:
            return None

        result_lines: List[str] = []
        current_display_line = 0

        for item in all_items:
            item_height = self._get_item_display_lines(item)
            line_end = current_display_line + item_height

            # Check if this item overlaps with the selection range
            if line_end > start_line and current_display_line <= end_line:
                # Only include OutputLine items (skip ToolBlocks)
                if isinstance(item, OutputLine):
                    if self._should_include_source(item.source, sources):
                        result_lines.append(item.text)

            current_display_line = line_end

            # Stop if we've passed the selection range
            if current_display_line > end_line:
                break

        if not result_lines:
            return None

        text = "\n".join(result_lines)
        return text.strip() if text.strip() else None

    def get_all_text(self, sources: Optional[set] = None) -> Optional[str]:
        """Get all text from the buffer (chrome-free).

        Args:
            sources: Set of sources to include. If None, includes all.

        Returns:
            The extracted text, or None if no matching content.
        """
        all_items = list(self._lines) + self._get_current_block_lines()

        if not all_items:
            return None

        result_lines: List[str] = []

        for item in all_items:
            # Only include OutputLine items (skip ToolBlocks)
            if isinstance(item, OutputLine):
                if self._should_include_source(item.source, sources):
                    result_lines.append(item.text)

        if not result_lines:
            return None

        text = "\n".join(result_lines)
        return text.strip() if text.strip() else None

    def get_visible_line_range(self, visible_height: int) -> Tuple[int, int]:
        """Get the display line range currently visible on screen.

        Used for mouse selection to map screen Y coordinates to buffer lines.

        Args:
            visible_height: Height of the visible area in display lines.

        Returns:
            Tuple of (start_line, end_line) display line indices.
        """
        all_items = list(self._lines) + self._get_current_block_lines()
        total_display_lines = sum(self._get_item_display_lines(item) for item in all_items)

        # Calculate visible range based on scroll offset
        end_line = total_display_lines - self._scroll_offset
        start_line = max(0, end_line - visible_height)

        return (start_line, end_line)
