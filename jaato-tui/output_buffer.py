"""Output buffer for the scrolling output panel.

Manages a ring buffer of output lines for display in the scrolling
region of the TUI.

Tracing:
    Set RICH_BUFFER_TRACE environment variable to control buffer tracing:
    - Not set: writes to {tempdir}/rich_render_trace.log
    - Empty string: disabled
    - Path: writes to specified file
"""

import os
import re
import tempfile
import textwrap
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union


def _trace(msg: str) -> None:
    """Write trace message to log file for debugging."""
    from jaato_sdk.trace import trace
    trace("OutputBuffer", msg)


def _get_buffer_trace_path() -> Optional[str]:
    """Get the buffer trace file path from environment variable."""
    from jaato_sdk.trace import resolve_trace_path
    return resolve_trace_path("RICH_BUFFER_TRACE",
                              default_filename="rich_render_trace.log")


def _buffer_trace(msg: str) -> None:
    """Write trace message to buffer trace log."""
    from jaato_sdk.trace import trace_write
    trace_write("OutputBuffer", msg, _get_buffer_trace_path())


from io import StringIO
from rich.console import Console, RenderableType
from rich.text import Text
from rich.panel import Panel
from rich.align import Align
from rich.table import Table
from rich import box

from shared.plugins.table_formatter.plugin import _display_width
from shared.plugins.formatter_pipeline import PRERENDERED_LINE_PREFIX
from ui_utils import format_tool_arg_value, format_tool_args_summary
from terminal_emulator import TerminalEmulator

# Type checking import for ThemeConfig
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from theme import ThemeConfig


# Pattern to strip ANSI escape codes for visible length calculation
_ANSI_ESCAPE_PATTERN = re.compile(r'\x1b\[[0-9;?]*[A-Za-z~]')


def _truncate_to_display_width(text: str, width: int) -> str:
    """Truncate text to fit within a target display width.

    Iterates character-by-character using _display_width() so that
    wide characters (CJK, emoji) are properly accounted for.

    Args:
        text: The string to truncate.
        width: Maximum display width.

    Returns:
        The truncated string (may be shorter than *width* characters).
    """
    current = 0
    for i, char in enumerate(text):
        cw = _display_width(char)
        if current + cw > width:
            return text[:i]
        current += cw
    return text


def _visible_len(text: str) -> int:
    """Calculate visible length of text, ignoring ANSI escape codes."""
    return len(_ANSI_ESCAPE_PATTERN.sub('', text))


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text."""
    return _ANSI_ESCAPE_PATTERN.sub('', text)


def _ljust_visible(text: str, width: int) -> str:
    """Left-justify text to *width* visible characters, preserving ANSI codes.

    Like ``str.ljust()`` but only counts characters that are actually visible
    on the terminal (i.e. ANSI escape sequences are ignored when measuring).

    Args:
        text: String that may contain ANSI escape codes.
        width: Target visible width to pad to.

    Returns:
        The original string with trailing spaces appended so its visible
        length equals *width*.  If the visible length already meets or
        exceeds *width*, the string is returned unchanged.
    """
    visible = _visible_len(text)
    if visible >= width:
        return text
    return text + ' ' * (width - visible)


def _wrap_visible(text: str, width: int) -> list[str]:
    """Wrap text to *width* visible characters, handling ANSI codes.

    ANSI escape codes are stripped for measurement/wrapping, then each
    resulting physical line is returned as plain text (no ANSI codes).
    This is appropriate for permission/clarification boxes where the
    content is rendered with a uniform box style anyway.

    Args:
        text: String that may contain ANSI escape codes.
        width: Maximum visible width per line.

    Returns:
        List of wrapped plain-text lines.
    """
    plain = _strip_ansi(text)
    if len(plain) <= width:
        return [plain]
    wrapped = textwrap.wrap(plain, width=width, break_long_words=True)
    return wrapped if wrapped else [plain]


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
    ansi_pattern = re.compile(r'(\x1b\[[0-9;?]*[A-Za-z~])')

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


def _get_trace_path() -> Optional[str]:
    """Get the trace file path from environment variable."""
    return _get_buffer_trace_path()


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
    """A finalized block of completed tools stored inline in the output buffer.

    Created by ``_finalize_completed_tools()`` when all active tools have
    completed and either a new tool arrives or the model's turn ends.
    The block holds deep-copied ``ActiveToolCall`` objects and is inserted
    into ``_lines`` at the chronological placeholder position.

    Rendered by ``_render_tool_block()`` with ``finalized=True`` — output is
    capped at 70 % of visible height using standard beginning + ellipsis + end
    truncation.
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
    """Represents a tool call through its full lifecycle.

    Lifecycle::

        add_active_tool()            → in _active_tools, completed=False  (running)
        mark_tool_completed()        → in _active_tools, completed=True   (completed)
        _finalize_completed_tools()  → deep-copied into ToolBlock in _lines,
                                       removed from _active_tools           (finalized)

    **Completed** means execution finished but the tool still lives in
    ``_active_tools`` and renders via ``_render_active_tools_inline()``.
    A tool can stay completed-but-not-finalized while sibling tools are
    still running or the model's turn hasn't ended.

    **Finalized** means the tool has been moved into a ``ToolBlock`` in
    ``_lines``. Rendering uses ``_render_tool_block()`` with
    ``finalized=True``.
    """
    name: str
    args_summary: str  # Truncated string representation of args (for tree display)
    args_full: Optional[str] = None  # Full untruncated args (for popup header)
    tool_args_dict: Optional[Dict[str, Any]] = None  # Raw args dict for multi-line param rendering
    call_id: Optional[str] = None  # Unique ID for correlating start/end of same tool call
    completed: bool = False  # True when tool execution finished
    backgrounded: bool = False  # True when auto-backgrounded (completed but still producing output)
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
    permission_content: Optional[str] = None  # Formatted content from unified flow (may contain ANSI codes)
    # Persisted file output (preserved after permission resolution for display in collapsed blocks)
    file_output_lines: Optional[List[str]] = None  # File content/diff lines that persist after tool completes
    file_output_display_lines: int = 5  # Max lines to show at once when expanded
    file_output_scroll_offset: int = 0  # Scroll position (0 = show most recent lines)
    # Clarification tracking (per-question progressive display)
    clarification_state: Optional[str] = None  # None, "pending", "resolved"
    clarification_prompt_lines: Optional[List[str]] = None  # Current question lines
    clarification_truncated: bool = False  # True if prompt is truncated
    clarification_current_question: int = 0  # Current question index (1-based)
    clarification_total_questions: int = 0  # Total number of questions
    clarification_answered: Optional[List[Tuple[int, str]]] = None  # List of (question_index, answer_summary)
    clarification_summary: Optional[List[Tuple[str, str]]] = None  # Q&A pairs [(question, answer), ...] for overview
    clarification_content: Optional[str] = None  # Formatted content from unified flow (may contain ANSI codes)
    # Live output tracking (preserves full output for smart truncation during rendering)
    output_lines: Optional[List[str]] = None  # Full output buffer (truncation happens at render time)
    output_max_lines: int = 1000  # Max lines to keep (high limit, rendering does smart truncation)
    output_display_lines: int = 5  # Max lines to show at once when expanded
    output_scroll_offset: int = 0  # Scroll position (0 = show most recent lines)
    # Continuation grouping (e.g., interactive shell sessions)
    continuation_id: Optional[str] = None  # Shared session ID for tools that belong together
    popup_header_override: Optional[str] = None  # If set, popup header shows this instead of args (for continuation groups)
    show_output: bool = True  # Whether to render output_lines in the main panel (popup unaffected)
    show_popup: bool = True  # Whether to track/update the tool output popup
    # Per-tool expand state for navigation
    expanded: bool = False  # Whether this tool's output is expanded
    # Progressive escalation: inline preview → popup
    # Output starts inline; after thresholds (line count + time), escalates to popup
    escalation_state: str = "inline"  # "inline" (showing in panel) | "escalated" (popup took over)
    escalation_line_threshold: int = 5  # Lines of output before escalation eligible
    escalation_time_threshold: float = 3.0  # Seconds after first output before escalation eligible
    first_output_time: Optional[float] = None  # monotonic timestamp of first output chunk
    inline_frozen_lines: Optional[List[str]] = None  # Snapshot of first N lines at escalation time
    popup_suppressed: bool = False  # User manually dismissed popup; don't re-escalate


class OutputBuffer:
    """Manages output lines for the scrolling panel.

    Stores output in a ring buffer and renders to Rich Text
    for display in the output panel.
    """

    # Spinner animation frames
    SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    # Type alias for items that can be stored in the line buffer
    LineItem = Union[OutputLine, ToolBlock]

    def __init__(self, max_lines: int = 1000, agent_type: str = "main", tools_expanded: bool = False):
        """Initialize the output buffer.

        Args:
            max_lines: Maximum number of lines to retain.
            agent_type: Type of agent ("main" or "subagent") for label display.
            tools_expanded: Initial tool block expansion state. False (collapsed) for
                interactive TUI, True (expanded) for headless mode.
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
        self._popup_tools: Dict[str, ActiveToolCall] = {}  # Backgrounded tools for popup (keyed by call_id)
        self._tools_expanded: bool = tools_expanded  # Toggle between collapsed/expanded tool view
        self._tools_expanded_before_prompt: Optional[bool] = None  # Saved state before permission/clarification forced expansion
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
        # Pending permission/clarification content (buffered until associated with tool)
        self._pending_permission_content: Optional[str] = None
        self._pending_clarification_content: Optional[str] = None
        # Permission keyboard navigation state (for focus highlighting in options line)
        self._permission_response_options: Optional[list] = None
        self._permission_focus_index: int = 0
        # Formatter pipeline for output processing (optional)
        self._formatter_pipeline: Optional[Any] = None
        # Theme configuration for styling (optional)
        self._theme: Optional["ThemeConfig"] = None
        # Search state
        self._search_query: str = ""
        self._search_matches: List[Tuple[int, int, int]] = []  # (line_index, start_pos, end_pos)
        self._search_current_idx: int = 0
        # Terminal emulators for tool output (pyte-backed, keyed by call_id)
        self._tool_emulators: Dict[str, TerminalEmulator] = {}
        # Step ID → step number mapping for human-readable display in tool args.
        # Updated when plan data changes; used by add_active_tool() to replace
        # raw UUIDs with step numbers (e.g., "Step #3") in the tool tree.
        self._step_id_to_number: Dict[str, int] = {}

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

    def set_permission_focus(self, options: Optional[list], focus_index: int) -> None:
        """Set permission keyboard navigation state for focus highlighting.

        When permission options and focus index are set, the options line in the
        permission prompt will be re-rendered with the focused option highlighted.

        Args:
            options: List of permission response options (or None to clear).
            focus_index: Index of the currently focused option.
        """
        # Check if focus actually changed to avoid unnecessary cache invalidation
        changed = (self._permission_response_options != options or
                   self._permission_focus_index != focus_index)
        self._permission_response_options = options
        self._permission_focus_index = focus_index
        # Invalidate render caches so permission prompt re-renders with new focus
        if changed:
            self._invalidate_line_caches()

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

    def _safe_insert_line(
        self, index: int, item: "Union[OutputLine, ToolBlock]"
    ) -> int:
        """Insert into _lines, removing from front if at max capacity.

        When the deque is at its maxlen, insert() raises IndexError.
        This method handles that by removing from the front first.

        Args:
            index: Position to insert at.
            item: The line or block to insert.

        Returns:
            The actual index where the item was inserted (may be less than
            requested if items were removed from front to make room).
        """
        if self._lines.maxlen is not None and len(self._lines) >= self._lines.maxlen:
            self._lines.popleft()
            index = max(0, index - 1)
        self._lines.insert(index, item)
        return index

    def _measure_display_lines(self, source: str, text: str, is_turn_start: bool = False) -> int:
        """Measure how many display lines a piece of text will take.

        Args:
            source: Source of the output.
            text: The text content.
            is_turn_start: Whether this is the first line of a turn (shows prefix).

        Returns:
            Number of display lines when rendered.
        """
        # Fast path: prerendered content (mermaid diagrams) — just count newlines
        if text.startswith(PRERENDERED_LINE_PREFIX):
            clean = text[len(PRERENDERED_LINE_PREFIX):]
            return clean.count('\n') + 1

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
            # Note: footer/header are only rendered for last/first thinking line in a
            # block, but we count them for all lines to ensure the block fits on screen.
            with self._measure_console.capture() as capture:
                self._measure_console.print(rendered, end='')
            output = capture.get()
            content_lines = output.count('\n') + 1 if output else 1
            if is_turn_start:
                # 1 (blank) + 1 (Model header) + 1 (thinking header) + content + 1 (footer)
                return 4 + content_lines
            else:
                # 1 (thinking header) + content + 1 (footer)
                return 2 + content_lines
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

    @staticmethod
    def _coalesce_prerendered_lines(lines: list) -> list:
        """Merge consecutive PRERENDERED_LINE_PREFIX lines into single entries.

        Rendered diagrams produce one line per pixel row. Keeping them as
        separate OutputLines causes inter-item spacing gaps and O(N)
        measurement overhead.  Coalescing into one entry with embedded
        newlines fixes both issues.
        """
        result = []
        buf = []
        for line in lines:
            if line.startswith(PRERENDERED_LINE_PREFIX):
                buf.append(line[len(PRERENDERED_LINE_PREFIX):])
            else:
                if buf:
                    result.append(PRERENDERED_LINE_PREFIX + '\n'.join(buf))
                    buf = []
                result.append(line)
        if buf:
            result.append(PRERENDERED_LINE_PREFIX + '\n'.join(buf))
        return result

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
        # Trace buffer append
        text_preview = text[:80] + "..." if len(text) > 80 else text
        text_preview = text_preview.replace("\n", "\\n")
        _buffer_trace(f"append: source={source} mode={mode} text={text_preview!r}")

        # Skip plan messages - they're shown in the sticky plan panel
        if source == "plan":
            return

        # Permission and clarification content is buffered until associated with a tool
        # This ensures correct association when multiple tools run in parallel
        if source == "permission" and self._active_tools:
            self._flush_current_block()
            if mode == "append" and self._pending_permission_content:
                self._pending_permission_content += "\n" + text
            else:
                self._pending_permission_content = text
            return

        if source == "clarification" and self._active_tools:
            self._flush_current_block()
            if mode == "append" and self._pending_clarification_content:
                self._pending_clarification_content += "\n" + text
            else:
                self._pending_clarification_content = text
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
        elif mode == "update":
            # Update mode: replace the last line from the same source
            # Used for progress indicators that update in place (e.g., OAuth countdown)
            self._flush_current_block()
            # Search backwards for the last line from this source
            updated = False
            for i in range(len(self._lines) - 1, -1, -1):
                if self._lines[i].source == source:
                    # Replace the line
                    display_lines = self._measure_display_lines(source, text, False)
                    self._lines[i] = OutputLine(
                        source=source,
                        text=text,
                        style=self._lines[i].style,
                        display_lines=display_lines,
                        is_turn_start=False,
                    )
                    updated = True
                    break
            if not updated:
                # No previous line from this source, just add a new one
                self._add_line(source, text, "line", is_turn_start=False)
            self._last_source = source
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
            lines = self._coalesce_prerendered_lines(lines)
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
        lines_text = self._coalesce_prerendered_lines(lines_text)

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
        msg_preview = message[:80] + "..." if len(message) > 80 else message
        msg_preview = msg_preview.replace("\n", "\\n")
        _buffer_trace(f"add_system_message: style={style} msg={msg_preview!r}")

        self._flush_current_block()
        # Handle None or empty messages gracefully
        if not message:
            return
        # Handle multi-line messages
        for line in message.split('\n'):
            self._add_line("system", line, style)

    def update_last_system_message(
        self,
        message: str,
        style: str = "system_info",
        prefix: Optional[str] = None
    ) -> bool:
        """Update the last system message in the buffer.

        Used for progressive updates like init progress that show "..." then "OK".

        Args:
            message: The new message text.
            style: Rich style for the message.
            prefix: If provided, only update a system message that starts with
                this prefix. This prevents updating the wrong message when other
                system messages are inserted between "running" and "done" states.

        Returns:
            True if a system message was found and updated, False otherwise.
        """
        # Search backwards for the last system message (optionally matching prefix)
        for i in range(len(self._lines) - 1, -1, -1):
            line = self._lines[i]
            if line.source == "system":
                # If prefix specified, check if this line starts with it
                if prefix is not None and not line.text.startswith(prefix):
                    continue
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
        _buffer_trace(f"clear: lines={len(self._lines)} active_tools={len(self._active_tools)}")

        self._lines.clear()
        self._current_block = None
        self._last_source = None
        self._last_turn_source = None
        self._scroll_offset = 0
        self._spinner_active = False
        self._active_tools.clear()
        self._tool_emulators.clear()
        self._tool_placeholder_index = None
        self._tool_nav_active = False
        self._selected_block_index = None
        self._selected_tool_index = None

    def start_spinner(self) -> None:
        """Start showing spinner in the output.

        This is called when a new turn starts. Flushes any pending content
        from the previous turn to ensure clean turn boundaries.
        """
        # Flush any pending content from the previous turn before starting
        # a new one. This prevents late-arriving chunks from the previous
        # turn from being concatenated with new turn content.
        self._flush_current_block()
        self._spinner_active = True
        self._spinner_index = 0

    def stop_spinner(self) -> None:
        """Stop showing spinner and finalize tool tree if complete."""
        self._spinner_active = False
        # Flush any pending streaming text before finalizing the turn
        self._flush_current_block()
        # Convert tool tree to scrollable lines if all tools are done
        self.finalize_tool_tree()

    def flush(self) -> None:
        """Flush the current streaming block to finalized lines.

        This should be called at turn boundaries to ensure all pending
        content from the current turn is rendered before the next turn starts.
        This prevents late-arriving chunks from being concatenated with
        chunks from a new turn.
        """
        self._flush_current_block()

    def advance_spinner(self) -> None:
        """Advance spinner to next frame."""
        if self._spinner_active:
            self._spinner_index = (self._spinner_index + 1) % len(self.SPINNER_FRAMES)

    @property
    def spinner_active(self) -> bool:
        """Check if spinner is currently active."""
        return self._spinner_active

    def update_step_id_map(self, step_id_map: Dict[str, int]) -> None:
        """Replace the step_id → step-number mapping used for display.

        When a PlanUpdatedEvent arrives, the caller builds a dict mapping
        each step's UUID to its 1-based sequence number and passes it here.
        Subsequent ``add_active_tool()`` calls will replace any ``step_id``
        argument value that appears in this map with ``Step #<N>`` so the
        user sees a human-readable reference instead of a raw UUID.

        Args:
            step_id_map: Mapping from step_id (UUID string) to 1-based
                sequence number.
        """
        self._step_id_to_number = step_id_map

    def add_active_tool(self, tool_name: str, tool_args: dict,
                        call_id: Optional[str] = None) -> None:
        """Add a tool to the active tools list.

        Args:
            tool_name: Name of the tool being executed.
            tool_args: Arguments passed to the tool.
            call_id: Unique identifier for this tool call (for correlation).
        """
        _buffer_trace(f"add_active_tool: name={tool_name} call_id={call_id}")

        # If all active tools are completed, finalize them as a ToolBlock
        # This ensures proper ordering of text and tool output.
        if self._active_tools and all(t.completed for t in self._active_tools):
            self._finalize_completed_tools()

        # Create a summary of args (truncated for tree display, full for popup)
        # Filter out intent args (message, summary, etc.) since they're shown as model text
        intent_arg_names = {"message", "summary", "intent", "rationale"}
        display_args = {k: v for k, v in (tool_args or {}).items() if k not in intent_arg_names}
        # Replace step_id UUIDs with human-readable step numbers when a plan is active
        if "step_id" in display_args and display_args["step_id"] in self._step_id_to_number:
            display_args = dict(display_args)  # shallow copy to avoid mutating caller's dict
            step_num = self._step_id_to_number[display_args["step_id"]]
            display_args["step_id"] = f"Step #{step_num}"
        args_full = str(display_args) if display_args else ""
        args_str = format_tool_args_summary(display_args) if display_args else ""

        # Don't add duplicates - check by call_id if provided, otherwise by name
        for tool in self._active_tools:
            if call_id and tool.call_id == call_id:
                _buffer_trace(f"add_active_tool: skipped duplicate call_id={call_id}")
                return
            # Fall back to name-based check only if no call_id provided
            if not call_id and tool.name == tool_name and not tool.call_id:
                _buffer_trace(f"add_active_tool: skipped duplicate name={tool_name}")
                return

        # When first tool is added, establish the placeholder position
        # Flush current block first so tools appear AFTER preceding text
        if not self._active_tools:
            self._flush_current_block()
            # If this is a new model turn (last source was user/parent), add model header
            # This ensures "── Model ──" appears before tool trees even when
            # the model makes tool calls without sending text first
            if self._last_turn_source in ("user", "parent", None):
                self._add_line("model", "", "line", is_turn_start=True)
                self._last_turn_source = "model"
            self._tool_placeholder_index = len(self._lines)

        # Detect continuation grouping: tools with a session_id arg participate
        continuation_id = (tool_args or {}).get("session_id")
        tool = ActiveToolCall(
            name=tool_name, args_summary=args_str,
            args_full=args_full or None,
            tool_args_dict=display_args or None,
            call_id=call_id, continuation_id=continuation_id,
        )
        # If joining an existing continuation group, carry over shared state
        if continuation_id and continuation_id in self._tool_emulators:
            emulator = self._tool_emulators[continuation_id]
            tool.output_lines = emulator.get_lines()
            tool.show_output = False  # Continuation follow-ups hide output in main panel
            # Inherit the original command as popup header (e.g., show "sh script.sh"
            # instead of "{'session_id': '...', 'input': '10\n'}")
            # Uses popup_header_override so the inline tree still shows this tool's own args.
            original = self._popup_tools.get(continuation_id)
            if original:
                tool.popup_header_override = original.popup_header_override or original.args_full or original.args_summary
        self._active_tools.append(tool)
        # Scroll to show the full tool tree (prioritizing top if it's taller than visible area)
        self.scroll_to_show_tool_tree()

    def mark_tool_completed(self, tool_name: str, success: bool = True,
                            duration_seconds: Optional[float] = None,
                            error_message: Optional[str] = None,
                            call_id: Optional[str] = None,
                            backgrounded: bool = False,
                            continuation_id: Optional[str] = None,
                            show_output: Optional[bool] = None,
                            show_popup: Optional[bool] = None) -> None:
        """Mark a tool as completed (keeps it in the tree with completion status).

        When backgrounded=True, the tool is marked completed for tree purposes
        but keeps accepting output and stays visible in the popup until the
        background task finishes (a second call with backgrounded=False).

        When continuation_id is provided, the tool belongs to a continuation
        group (e.g., interactive shell session) and the popup stays open
        across tools sharing the same continuation_id.

        Args:
            tool_name: Name of the tool that finished.
            success: Whether the tool execution succeeded.
            duration_seconds: How long the tool took to execute.
            error_message: Error message if the tool failed.
            call_id: Unique identifier for this tool call (for correlation).
            backgrounded: True if tool was auto-backgrounded (still producing output).
            continuation_id: Session ID for continuation grouping (popup stays open).
            show_output: Whether to render output_lines in the main panel.
                None means keep current value. The popup is unaffected.
            show_popup: Whether to track/update the tool output popup.
                None means keep current value. False prevents this tool from
                becoming the tracked popup tool.
        """
        _buffer_trace(
            f"mark_tool_completed: name={tool_name} success={success} "
            f"duration={duration_seconds} call_id={call_id} backgrounded={backgrounded}"
        )

        for tool in self._active_tools:
            # Match by call_id if provided, otherwise by name (for backwards compatibility)
            if call_id:
                if tool.call_id == call_id:
                    # Transition: backgrounded → truly completed (done callback fired)
                    if tool.completed and tool.backgrounded and not backgrounded:
                        tool.backgrounded = False
                        tool.success = success
                        if duration_seconds is not None:
                            tool.duration_seconds = duration_seconds
                        tool.error_message = error_message
                        self._popup_tools.pop(call_id, None)
                        self._tool_emulators.pop(call_id, None)
                        return
                    # Normal completion
                    if not tool.completed:
                        tool.completed = True
                        tool.backgrounded = backgrounded
                        tool.success = success
                        tool.duration_seconds = duration_seconds
                        tool.error_message = error_message
                        if show_output is not None:
                            tool.show_output = show_output
                        if show_popup is not None:
                            tool.show_popup = show_popup
                        if backgrounded and call_id and tool.show_popup:
                            self._popup_tools[call_id] = tool
                        # Continuation grouping (skip popup tracking if show_popup=False)
                        if tool.show_popup:
                            self._apply_continuation(tool, call_id, continuation_id)
                        return
            elif tool.name == tool_name and not tool.completed and not tool.call_id:
                tool.completed = True
                tool.backgrounded = backgrounded
                tool.success = success
                tool.duration_seconds = duration_seconds
                tool.error_message = error_message
                return

        # Tool not found in _active_tools — check _popup_tools (post-finalization)
        if call_id and call_id in self._popup_tools:
            tool = self._popup_tools[call_id]
            if tool.completed and tool.backgrounded and not backgrounded:
                tool.backgrounded = False
                tool.success = success
                if duration_seconds is not None:
                    tool.duration_seconds = duration_seconds
                tool.error_message = error_message
                self._popup_tools.pop(call_id)
                self._tool_emulators.pop(call_id, None)
                return

    def _apply_continuation(self, tool: ActiveToolCall, call_id: str,
                            continuation_id: Optional[str]) -> None:
        """Apply continuation grouping logic after a tool completes.

        If continuation_id is provided (session still alive), the tool joins or
        creates a continuation group. The emulator is re-keyed from call_id to
        continuation_id so subsequent tools in the group share the same emulator.

        If no continuation_id from events but the tool already had one from its
        args (e.g., shell_close closing the session), clean up the group.
        """
        if continuation_id:
            tool.continuation_id = continuation_id
            # Re-key emulator from call_id to continuation_id (first tool in group)
            if call_id in self._tool_emulators and continuation_id not in self._tool_emulators:
                self._tool_emulators[continuation_id] = self._tool_emulators.pop(call_id)
            self._popup_tools[continuation_id] = tool
        elif tool.continuation_id:
            # Tool had continuation_id from args but events didn't confirm it
            # — session is closing (e.g., shell_close)
            self._popup_tools.pop(tool.continuation_id, None)
            self._tool_emulators.pop(tool.continuation_id, None)
            tool.continuation_id = None

    def append_tool_output(self, call_id: str, chunk: str) -> None:
        """Append output chunk to a running tool's output buffer.

        Uses a pyte terminal emulator to correctly interpret ANSI control
        sequences (cursor movement, erase, carriage return, colors, etc.)
        and produce clean visual output lines.

        Used for live "tail -f" style output preview in the tool tree.
        Checks both _active_tools (pre-finalization) and _popup_tools
        (post-finalization, for backgrounded tools).

        Args:
            call_id: Unique identifier for the tool call.
            chunk: Output text chunk (may contain ANSI sequences, newlines).
        """
        tool = self._find_output_tool(call_id)
        if tool is None:
            return

        # Get or create terminal emulator for this tool
        # Use continuation_id as key when tool belongs to a continuation group
        emulator_key = tool.continuation_id or call_id
        emulator = self._tool_emulators.get(emulator_key)
        if emulator is None:
            emulator = TerminalEmulator()
            self._tool_emulators[emulator_key] = emulator

        # Feed chunk through pyte (re-add newline stripped by CLI plugin)
        emulator.feed(chunk + "\n")

        # Reconstruct output_lines from emulator state
        tool.output_lines = emulator.get_lines()

        # Progressive escalation: track first output time and check thresholds
        if tool.first_output_time is None:
            tool.first_output_time = time.monotonic()

        if (tool.escalation_state == "inline"
                and not tool.popup_suppressed
                and tool.show_popup
                and len(tool.output_lines) > tool.escalation_line_threshold
                and (time.monotonic() - tool.first_output_time) >= tool.escalation_time_threshold):
            tool.escalation_state = "escalated"
            tool.inline_frozen_lines = tool.output_lines[:tool.escalation_line_threshold]

    def _find_output_tool(self, call_id: str) -> Optional[ActiveToolCall]:
        """Find the tool that should receive output for a given call_id.

        Checks _active_tools first (running/backgrounded pre-finalization),
        then _popup_tools (backgrounded post-finalization).
        """
        for tool in self._active_tools:
            if tool.call_id == call_id and (not tool.completed or tool.backgrounded):
                return tool
        # Post-finalization: backgrounded tool lives in _popup_tools
        popup_tool = self._popup_tools.get(call_id)
        if popup_tool and popup_tool.backgrounded:
            return popup_tool
        return None

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
        self._popup_tools.clear()
        self._tool_emulators.clear()
        self._tool_nav_active = False
        self._selected_tool_index = None

    def toggle_tools_expanded(self) -> bool:
        """Toggle between collapsed and expanded tool view.

        Returns:
            True if now expanded, False if now collapsed.
        """
        old = self._tools_expanded
        self._tools_expanded = not self._tools_expanded
        # Clear saved state so user's manual toggle is respected and not overwritten
        # by the restore logic when permission/clarification resolves
        self._tools_expanded_before_prompt = None
        _trace(f"toggle_tools_expanded: {old} -> {self._tools_expanded}, cleared before_prompt")
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
                        if tool.expanded and tool.file_output_lines:
                            # File content header + content lines
                            display_line += 1  # "Content" header
                            max_display_lines = max(5, int(self._visible_height * 0.7))
                            display_line += min(len(tool.file_output_lines), max_display_lines)
                            if len(tool.file_output_lines) > max_display_lines:
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
        # Only expand if there's output to show (output_lines or file_output_lines)
        has_output = (tool.output_lines and len(tool.output_lines) > 0) or \
                     (tool.file_output_lines and len(tool.file_output_lines) > 0)
        if not has_output:
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
        """Scroll up within the selected tool's output or file content.

        Returns:
            True if scroll position changed, False if at top or no tool selected.
        """
        tool = self.get_selected_tool()
        if not tool or not tool.expanded:
            return False

        # Handle output_lines or file_output_lines (mutually exclusive)
        if tool.output_lines:
            max_offset = max(0, len(tool.output_lines) - tool.output_display_lines)
            if tool.output_scroll_offset < max_offset:
                tool.output_scroll_offset += 1
                return True
        elif tool.file_output_lines:
            # Use 70% of visible height for file content display
            max_display_lines = max(5, int(self._visible_height * 0.7))
            max_offset = max(0, len(tool.file_output_lines) - max_display_lines)
            if tool.file_output_scroll_offset < max_offset:
                tool.file_output_scroll_offset += 1
                return True
        return False

    def scroll_selected_tool_down(self) -> bool:
        """Scroll down within the selected tool's output or file content.

        Returns:
            True if scroll position changed, False if at bottom or no tool selected.
        """
        tool = self.get_selected_tool()
        if not tool or not tool.expanded:
            return False

        # Handle output_lines or file_output_lines (mutually exclusive)
        if tool.output_lines:
            if tool.output_scroll_offset > 0:
                tool.output_scroll_offset -= 1
                return True
        elif tool.file_output_lines:
            if tool.file_output_scroll_offset > 0:
                tool.file_output_scroll_offset -= 1
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

        Backgrounded tools are finalized normally into the ToolBlock. Their
        references survive in _popup_tools for the popup to continue tracking.
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

        # Finalize completed tools; keep non-completed tools active
        to_finalize = [t for t in self._active_tools if t.completed]
        to_keep = [t for t in self._active_tools if not t.completed]

        if not to_finalize:
            return

        # Create a copy of the tools for the ToolBlock
        import copy
        tools_copy = copy.deepcopy(to_finalize)

        # Determine expansion state for the ToolBlock:
        # - If _tools_expanded_before_prompt is set, it means expansion was system-forced
        #   (permission/clarification prompt), so use the saved user preference
        # - If None, user manually toggled (or never had prompt), so use current state
        if self._tools_expanded_before_prompt is not None:
            block_expanded = self._tools_expanded_before_prompt
        else:
            block_expanded = self._tools_expanded

        # Create ToolBlock with the completed tools
        _trace(f"_finalize_completed_tools: creating ToolBlock with expanded={block_expanded}, _before_prompt={self._tools_expanded_before_prompt}, current={self._tools_expanded}")
        tool_block = ToolBlock(
            tools=tools_copy,
            expanded=block_expanded,
            selected_index=None
        )

        # Insert at placeholder position (set when first tool was added)
        insert_pos = self._tool_placeholder_index

        # Insert just the tool_block - it renders its own separator (───)
        # Use _safe_insert_line to handle bounded deque (raises IndexError when full)
        insert_pos = self._safe_insert_line(insert_pos, tool_block)
        next_pos = insert_pos + 1

        # Flush any pending enrichment notifications
        # These were queued while tools were active so they appear AFTER the tool tree
        if self._pending_enrichments:
            for enrich_source, enrich_text, enrich_mode in self._pending_enrichments:
                for line in enrich_text.split('\n'):
                    display_lines = self._measure_display_lines(enrich_source, line, False)
                    # Use "dim" as style - it's a valid Rich style primitive
                    enrich_line = OutputLine(
                        source=enrich_source, text=line, style="dim",
                        display_lines=display_lines, is_turn_start=False
                    )
                    next_pos = self._safe_insert_line(next_pos, enrich_line) + 1
            self._pending_enrichments.clear()

        # Add trailing blank line after tool block (and enrichments)
        # Use empty style since it's just spacing
        trailing_line = OutputLine(
            source="model", text="", style="",
            display_lines=1, is_turn_start=False
        )
        self._safe_insert_line(next_pos, trailing_line)

        # Clear placeholder and active tools
        self._tool_placeholder_index = None
        self._active_tools = to_keep

        # Now that tools are finalized, restore the expanded state if we had saved one
        # This happens AFTER finalization so the ToolBlock captures the visible state
        self._maybe_restore_expanded_state()

        # Auto-scroll to bottom to show finalized content
        # This ensures enrichment notifications and tool results are visible
        self.scroll_to_bottom()

    @property
    def active_tools(self) -> List[ActiveToolCall]:
        """Get list of currently active tools."""
        return list(self._active_tools)

    @property
    def popup_tools(self) -> Dict[str, ActiveToolCall]:
        """Get backgrounded tools tracked by the popup (keyed by call_id)."""
        return self._popup_tools

    def _maybe_restore_expanded_state(self) -> None:
        """Restore expanded state if no more pending prompts require expansion.

        Called after permission/clarification is resolved to check if we can
        restore the user's previous collapsed state.

        IMPORTANT: We don't restore while tools are still active (incomplete).
        This ensures the ToolBlock captures the visible state when tools finalize,
        rather than restoring the state and then having finalization capture
        the restored state.
        """
        _trace(f"_maybe_restore_expanded_state: active_tools={len(self._active_tools)}, expanded={self._tools_expanded}, before_prompt={self._tools_expanded_before_prompt}")

        # Check if any tool still has a pending prompt
        for tool in self._active_tools:
            if tool.permission_state == "pending" or tool.clarification_state == "pending":
                _trace(f"_maybe_restore_expanded_state: skip - tool {tool.name} has pending prompt")
                return  # Still have pending prompts, keep expanded

        # Don't restore while tools are still active - let finalization capture the visible state
        # The restore will happen after finalization clears _active_tools
        if self._active_tools:
            _trace(f"_maybe_restore_expanded_state: skip restore - {len(self._active_tools)} active tools remaining")
            return

        # No active tools and no pending prompts, restore saved state if we have one
        if self._tools_expanded_before_prompt is not None:
            old = self._tools_expanded
            self._tools_expanded = self._tools_expanded_before_prompt
            _trace(f"_maybe_restore_expanded_state: RESTORED {old} -> {self._tools_expanded}")
            self._tools_expanded_before_prompt = None
        else:
            _trace(f"_maybe_restore_expanded_state: no saved state to restore")

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
                # Save current state and force expanded view so user can see the permission prompt
                if self._tools_expanded_before_prompt is None:
                    self._tools_expanded_before_prompt = self._tools_expanded
                self._tools_expanded = True
                # Scroll to show the tool tree with the permission prompt
                self.scroll_to_show_tool_tree()
                _trace(f"set_tool_permission_pending: FOUND exact match for {tool_name}, _tools_expanded now True")
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
                # Save current state and force expanded view so user can see the permission prompt
                if self._tools_expanded_before_prompt is None:
                    self._tools_expanded_before_prompt = self._tools_expanded
                self._tools_expanded = True
                # Scroll to show the tool tree with the permission prompt
                self.scroll_to_show_tool_tree()
                _trace(f"set_tool_permission_pending: FALLBACK attached to {tool.name} (requested: {tool_name})")
                return

        _trace(f"set_tool_permission_pending: NO MATCH for {tool_name}")

    def set_tool_awaiting_approval(self, tool_name: str, call_id: Optional[str] = None) -> None:
        """Mark tool as awaiting permission and associate buffered content.

        This is used with the unified output flow where permission content flows
        through AgentOutputEvent, gets buffered, and is then associated with
        the specific tool when this method is called.

        Args:
            tool_name: Name of the tool awaiting permission.
            call_id: Optional unique identifier for the tool call (for parallel tool matching).
        """
        _trace(f"set_tool_awaiting_approval: looking for tool={tool_name} call_id={call_id}")

        def _associate_content(tool: ActiveToolCall) -> None:
            """Associate buffered permission content with the tool."""
            tool.permission_state = "pending"
            tool.permission_content = self._pending_permission_content
            self._pending_permission_content = None  # Clear buffer
            # Save current state and force expanded view so user can see tool status
            if self._tools_expanded_before_prompt is None:
                self._tools_expanded_before_prompt = self._tools_expanded
            self._tools_expanded = True
            # Scroll to show the tool tree with the permission prompt
            self.scroll_to_show_tool_tree()

        # First try exact match by call_id (most reliable for parallel tools)
        if call_id:
            for tool in self._active_tools:
                if tool.call_id == call_id and not tool.completed:
                    _associate_content(tool)
                    _trace(f"set_tool_awaiting_approval: FOUND by call_id={call_id}")
                    return

        # Fallback: try exact match by tool name (for backwards compatibility)
        for tool in self._active_tools:
            if tool.name == tool_name and not tool.completed:
                _associate_content(tool)
                _trace(f"set_tool_awaiting_approval: FOUND exact match for {tool_name}")
                return

        # Last resort: attach to the last uncompleted tool
        for tool in reversed(self._active_tools):
            if not tool.completed:
                if tool.name != tool_name:
                    tool.display_name = tool_name
                    tool.display_args_summary = ""
                _associate_content(tool)
                _trace(f"set_tool_awaiting_approval: FALLBACK attached to {tool.name}")
                return

        _trace(f"set_tool_awaiting_approval: NO MATCH for {tool_name} call_id={call_id}")

    def set_tool_awaiting_clarification(self, tool_name: str, question_index: int, total_questions: int) -> None:
        """Mark tool as awaiting clarification and associate buffered content.

        This is used with the unified output flow where clarification content flows
        through AgentOutputEvent, gets buffered, and is then associated with
        the specific tool when this method is called.

        Args:
            tool_name: Name of the tool awaiting clarification.
            question_index: Current question number (1-based).
            total_questions: Total number of questions.
        """
        _trace(f"set_tool_awaiting_clarification: tool={tool_name}, q{question_index}/{total_questions}")

        for tool in self._active_tools:
            if tool.name == tool_name and not tool.completed:
                tool.clarification_state = "pending"
                tool.clarification_content = self._pending_clarification_content
                self._pending_clarification_content = None  # Clear buffer
                tool.clarification_current_question = question_index
                tool.clarification_total_questions = total_questions
                # Save current state and force expanded view
                if self._tools_expanded_before_prompt is None:
                    self._tools_expanded_before_prompt = self._tools_expanded
                self._tools_expanded = True
                # Scroll to show the tool tree with the clarification prompt
                self.scroll_to_show_tool_tree()
                _trace(f"set_tool_awaiting_clarification: FOUND for {tool_name}")
                return

        _trace(f"set_tool_awaiting_clarification: NO MATCH for {tool_name}")

    def set_tool_permission_resolved(self, tool_name: str, granted: bool,
                                      method: str) -> None:
        """Mark a tool's permission as resolved.

        Args:
            tool_name: Name of the tool (may be the tool being checked, not the executing tool).
            granted: Whether permission was granted.
            method: How permission was resolved (yes, always, once, never, whitelist, etc.)
        """
        _trace(f"set_tool_permission_resolved: looking for tool={tool_name}, granted={granted}")

        resolved = False

        # First try exact match by tool name with pending permission
        for tool in self._active_tools:
            if tool.name == tool_name and tool.permission_state == "pending":
                tool.permission_state = "granted" if granted else "denied"
                tool.permission_method = method
                # Clear permission content - it was just for the prompt display, not permanent storage
                tool.permission_content = None
                _trace(f"set_tool_permission_resolved: FOUND exact match for {tool_name}")
                resolved = True
                break

        # Fallback: find any tool with pending permission state
        if not resolved:
            for tool in self._active_tools:
                if tool.permission_state == "pending":
                    tool.permission_state = "granted" if granted else "denied"
                    tool.permission_method = method
                    # Clear permission content - it was just for the prompt display, not permanent storage
                    tool.permission_content = None
                    _trace(f"set_tool_permission_resolved: FALLBACK resolved {tool.name} (requested: {tool_name})")
                    resolved = True
                    break

        if not resolved:
            _trace(f"set_tool_permission_resolved: NO PENDING TOOL for {tool_name}")
            return

        # Restore expanded state if no more pending prompts
        self._maybe_restore_expanded_state()

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
                # Save current state and force expanded view so user can see the prompt
                if self._tools_expanded_before_prompt is None:
                    self._tools_expanded_before_prompt = self._tools_expanded
                self._tools_expanded = True
                # Scroll to show the tool tree with the clarification prompt
                self.scroll_to_show_tool_tree()
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
                # Save current state and force expanded view so user can see the prompt
                if self._tools_expanded_before_prompt is None:
                    self._tools_expanded_before_prompt = self._tools_expanded
                self._tools_expanded = True
                # Scroll to show the tool tree with the question
                self.scroll_to_show_tool_tree()
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
        resolved = False
        for tool in self._active_tools:
            if tool.name == tool_name:
                tool.clarification_state = "resolved"
                tool.clarification_content = None  # Clear clarification content
                tool.clarification_current_question = 0
                tool.clarification_total_questions = 0
                tool.clarification_summary = qa_pairs
                resolved = True
                break

        if resolved:
            # Restore expanded state if no more pending prompts
            self._maybe_restore_expanded_state()

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
            # Check pending state regardless of prompt_lines (unified flow case)
            if tool.permission_state == "pending":
                return True
            if tool.clarification_state == "pending":
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

    def scroll_to_show_tool_tree(self) -> bool:
        """Scroll to show the active tool tree.

        When there's a pending permission/clarification, prioritizes showing the
        bottom of the tree (where response options are). Otherwise shows from top.

        Returns:
            True if scroll position changed.
        """
        if not self._active_tools or self._tool_placeholder_index is None:
            return False

        old_offset = self._scroll_offset

        # Check if any tool has a pending prompt (permission or clarification)
        has_pending_prompt = any(
            tool.permission_state == "pending" or tool.clarification_state == "pending"
            for tool in self._active_tools
        )

        # Calculate total display lines before the tool tree
        # Note: _lines is a deque which doesn't support slicing, so we iterate with enumerate
        lines_before_tree = sum(
            self._get_item_display_lines(item)
            for i, item in enumerate(self._lines)
            if i < self._tool_placeholder_index
        )

        # Calculate tool tree height
        tree_height = self._calculate_tool_tree_height()

        # Calculate lines after the tool tree (remaining content in _lines + any streaming)
        lines_after_tree = sum(
            self._get_item_display_lines(item)
            for i, item in enumerate(self._lines)
            if i >= self._tool_placeholder_index
        )

        # Total content height
        total_height = lines_before_tree + tree_height + lines_after_tree

        # scroll_offset = how many lines from the bottom to skip
        # To show ending at line X, scroll_offset = total_height - X

        if tree_height <= self._visible_height:
            # Tree fits - show entire tree with some context before it
            tree_bottom = lines_before_tree + tree_height
            self._scroll_offset = max(0, total_height - tree_bottom - max(0, self._visible_height - tree_height))
        elif has_pending_prompt:
            # Tree is taller AND has pending prompt - prioritize showing BOTTOM
            # (where response options are), keeping minimal context at top
            context_lines = min(1, lines_before_tree)  # Keep just 1 line of context
            tree_bottom = lines_before_tree + tree_height
            # Scroll so bottom of tree is visible, with minimal context
            self._scroll_offset = max(0, total_height - tree_bottom - context_lines)
        else:
            # Tree is taller, no pending prompt - show from top of tree
            self._scroll_offset = max(0, total_height - lines_before_tree - self._visible_height)

        _trace(f"scroll_to_show_tool_tree: visible={self._visible_height}, tree={tree_height}, "
               f"before={lines_before_tree}, pending={has_pending_prompt}, offset={self._scroll_offset}")

        return self._scroll_offset != old_offset

    # ─────────────────────────────────────────────────────────────────────────
    # Search functionality
    # ─────────────────────────────────────────────────────────────────────────

    def search(self, query: str) -> int:
        """Search for text in all output lines.

        Args:
            query: The search query (case-insensitive).

        Returns:
            Number of matches found.
        """
        self._search_query = query
        self._search_matches = []
        self._search_current_idx = 0

        # Invalidate line caches so highlighting is applied on next render
        self._invalidate_line_caches()

        if not query:
            return 0

        query_lower = query.lower()

        # Search through all OutputLine items
        for line_idx, item in enumerate(self._lines):
            if isinstance(item, OutputLine):
                # Strip ANSI codes for search
                text = _strip_ansi(item.text).lower()
                start = 0
                while True:
                    pos = text.find(query_lower, start)
                    if pos == -1:
                        break
                    self._search_matches.append((line_idx, pos, pos + len(query)))
                    start = pos + 1

        # Jump to first match if found
        if self._search_matches:
            self._scroll_to_match(0)

        return len(self._search_matches)

    def search_next(self) -> bool:
        """Navigate to the next search match.

        Returns:
            True if navigation occurred, False if no matches.
        """
        if not self._search_matches:
            return False

        self._search_current_idx = (self._search_current_idx + 1) % len(self._search_matches)
        self._scroll_to_match(self._search_current_idx)
        # Invalidate cache so current match highlighting updates
        self._invalidate_line_caches()
        return True

    def search_prev(self) -> bool:
        """Navigate to the previous search match.

        Returns:
            True if navigation occurred, False if no matches.
        """
        if not self._search_matches:
            return False

        self._search_current_idx = (self._search_current_idx - 1) % len(self._search_matches)
        self._scroll_to_match(self._search_current_idx)
        # Invalidate cache so current match highlighting updates
        self._invalidate_line_caches()
        return True

    def clear_search(self) -> None:
        """Clear search state."""
        self._search_query = ""
        self._search_matches = []
        self._search_current_idx = 0
        # Invalidate line caches so highlights are removed on next render
        self._invalidate_line_caches()

    def get_search_status(self) -> Tuple[str, int, int]:
        """Get current search status for display.

        Returns:
            Tuple of (query, current_match_1_indexed, total_matches).
        """
        if not self._search_matches:
            return (self._search_query, 0, 0)
        return (self._search_query, self._search_current_idx + 1, len(self._search_matches))

    def _scroll_to_match(self, match_idx: int) -> None:
        """Scroll to show a specific search match.

        Args:
            match_idx: Index into _search_matches list.
        """
        if match_idx < 0 or match_idx >= len(self._search_matches):
            return

        line_idx, _, _ = self._search_matches[match_idx]

        # Calculate display lines before this item
        display_lines_before = 0
        for i, item in enumerate(self._lines):
            if i >= line_idx:
                break
            display_lines_before += self._get_item_display_lines(item)

        # Calculate total display lines
        total_display_lines = sum(self._get_item_display_lines(item) for item in self._lines)

        # Calculate scroll offset to center the match in the visible area
        # scroll_offset is measured from bottom (0 = at bottom)
        target_from_bottom = total_display_lines - display_lines_before - 1

        # Aim to show match in the middle of the visible area
        center_offset = max(0, self._visible_height // 2)
        self._scroll_offset = max(0, target_from_bottom - center_offset)

    def is_line_match(self, line_idx: int) -> bool:
        """Check if a line contains any search matches.

        Args:
            line_idx: Index in _lines.

        Returns:
            True if line has matches.
        """
        return any(idx == line_idx for idx, _, _ in self._search_matches)

    def get_line_matches(self, line_idx: int) -> List[Tuple[int, int, bool]]:
        """Get search match positions for a specific line.

        Args:
            line_idx: Index in _lines.

        Returns:
            List of (start_pos, end_pos, is_current) tuples.
        """
        matches = []
        for i, (idx, start, end) in enumerate(self._search_matches):
            if idx == line_idx:
                is_current = (i == self._search_current_idx)
                matches.append((start, end, is_current))
        return matches

    def _highlight_text_with_matches(
        self,
        text: str,
        matches: List[Tuple[int, int, bool]],
        base_style: Optional[str] = None
    ) -> Text:
        """Apply search highlighting to text.

        Args:
            text: The text to highlight.
            matches: List of (start_pos, end_pos, is_current) tuples.
            base_style: Optional base style for non-highlighted text.

        Returns:
            Text object with highlighting applied.
        """
        if not matches:
            result = Text()
            result.append(text, style=base_style)
            return result

        # Sort matches by start position
        sorted_matches = sorted(matches, key=lambda m: m[0])

        result = Text()
        pos = 0

        for start, end, is_current in sorted_matches:
            # Add text before this match
            if start > pos:
                result.append(text[pos:start], style=base_style)

            # Add highlighted match
            match_text = text[start:end]
            if is_current:
                style = self._style("search_match_current", "reverse bold yellow")
            else:
                style = self._style("search_match", "reverse yellow")
            result.append(match_text, style=style)
            pos = end

        # Add remaining text after last match
        if pos < len(text):
            result.append(text[pos:], style=base_style)

        return result

    def _wrap_and_highlight_text(
        self,
        text: str,
        line_idx: Optional[int],
        wrap_width: int,
        base_style: Optional[str] = None
    ) -> Text:
        """Wrap text and apply search highlighting.

        This method handles the complexity of wrapping text while preserving
        search highlight positions across wrapped line segments.

        Args:
            text: The text to wrap and highlight.
            line_idx: Index in self._lines for looking up search matches.
            wrap_width: Maximum width for wrapping.
            base_style: Optional base style for non-highlighted text.

        Returns:
            Text object with wrapping and highlighting applied.
        """
        # Handle literal \n in text (escaped newlines) - convert to actual newlines
        text = text.replace('\\n', '\n')

        # Get matches for this line
        matches: List[Tuple[int, int, bool]] = []
        if line_idx is not None and self._search_query:
            matches = self.get_line_matches(line_idx)

        result = Text()

        # Process each paragraph (split by newlines)
        paragraphs = text.split('\n')
        char_offset = 0  # Track position in original text

        for p_idx, paragraph in enumerate(paragraphs):
            if p_idx > 0:
                result.append("\n")
                char_offset += 1  # Account for the newline character

            if not paragraph.strip():
                # Empty paragraph
                char_offset += len(paragraph)
                continue

            # Wrap this paragraph
            available = max(20, wrap_width)
            if len(paragraph) <= available:
                wrapped_lines = [paragraph]
            else:
                wrapped_lines = textwrap.wrap(
                    paragraph, width=available,
                    break_long_words=True, break_on_hyphens=False
                )
                if not wrapped_lines:
                    wrapped_lines = [paragraph]

            # Track position within this paragraph for wrapping
            para_pos = 0

            for w_idx, wrapped_line in enumerate(wrapped_lines):
                if w_idx > 0:
                    result.append("\n")

                # Find where this wrapped line is in the original paragraph
                # textwrap may add/remove spaces, so we need to find the actual substring
                line_start = paragraph.find(wrapped_line.lstrip(), para_pos)
                if line_start == -1:
                    line_start = para_pos

                line_end = line_start + len(wrapped_line)
                para_pos = line_end

                # Calculate absolute positions in original text
                abs_start = char_offset + line_start
                abs_end = char_offset + line_end

                # Find matches that overlap with this wrapped line
                segment_matches: List[Tuple[int, int, bool]] = []
                for match_start, match_end, is_current in matches:
                    # Check if match overlaps with this segment
                    if match_end > abs_start and match_start < abs_end:
                        # Adjust positions relative to this segment
                        rel_start = max(0, match_start - abs_start)
                        rel_end = min(len(wrapped_line), match_end - abs_start)
                        segment_matches.append((rel_start, rel_end, is_current))

                # Apply highlighting to this segment
                if segment_matches:
                    highlighted = self._highlight_text_with_matches(
                        wrapped_line, segment_matches, base_style
                    )
                    result.append_text(highlighted)
                else:
                    result.append(wrapped_line, style=base_style)

            char_offset += len(paragraph)

        return result

    def _measure_content_lines(self, content: str) -> int:
        """Count display lines for content (lines are truncated, not wrapped).

        Args:
            content: The content string (may contain newlines).

        Returns:
            Number of display lines (1 logical line = 1 display line since we truncate).
        """
        if not content:
            return 0
        # Each logical line = 1 display line (long lines are truncated with indicator)
        return content.count('\n') + 1

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
                # Expanded view: separator + header + each tool on its own line
                height += 1  # Separator/hint line (e.g., "───  Ctrl+T to expand")
                height += 1  # Header line with tool count

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
                # Collapsed view: separator + summary line
                height += 1  # Separator/hint line
                height += 1  # Summary line with tool count

            # Check for pending prompts (same logic for both views)
            for tool in self._active_tools:
                # Permission prompt (if pending)
                if tool.permission_state == "pending":
                    height += 1  # "Permission required" header

                    # Unified flow: permission_content is rendered inline under the tool
                    if tool.permission_content:
                        # Lines are truncated (not wrapped), so 1 logical line = 1 display line
                        logical_lines = self._measure_content_lines(tool.permission_content)

                        # Calculate max content lines dynamically (same as _render_permission_prompt)
                        overhead = self._calculate_prompt_overhead(tool)
                        available_space = self._visible_height - overhead
                        max_content_lines = max(3, available_space)

                        if logical_lines > max_content_lines:
                            # Truncation will occur: lines_at_start + ellipsis (1) + lines_at_end
                            height += max_content_lines
                        else:
                            # No truncation - all lines shown
                            height += logical_lines
                        continue

                    # Legacy flow: no permission_content, use prompt_lines
                    if not tool.permission_prompt_lines:
                        continue

                    prompt_lines = tool.permission_prompt_lines

                    # Pre-formatted content (e.g., diff) - no box, just lines
                    # Count actual visual lines (diff lines may contain embedded newlines)
                    if tool.permission_format_hint == "diff":
                        for line in prompt_lines:
                            # Count newlines within each line + 1 for the line itself
                            height += line.count('\n') + 1
                    else:
                        # Box calculation — use _visible_len to ignore ANSI codes
                        max_prompt_lines = 18
                        max_box_width = max(60, self._console_width - 22) if self._console_width > 40 else 60
                        box_width = min(max_box_width, max(_visible_len(line) for line in prompt_lines) + 4)
                        content_width = box_width - 4

                        # First count ALL wrapped lines to decide truncation (matches render logic)
                        total_wrapped_lines = 0
                        for prompt_line in prompt_lines:
                            wrapped = _wrap_visible(prompt_line, content_width)
                            total_wrapped_lines += len(wrapped)

                        if total_wrapped_lines > max_prompt_lines:
                            # Truncation triggered - render shows:
                            # - max_lines_before_truncation content lines
                            # - 1 truncation message
                            # - Last line (may wrap to multiple lines)
                            max_lines_before_truncation = max_prompt_lines - 3

                            # Calculate wrapped lines for last line
                            last_line = prompt_lines[-1]
                            last_wrapped = _wrap_visible(last_line, content_width)
                            last_line_count = len(last_wrapped)

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

                    # Unified flow: clarification_content is rendered inline under the tool
                    if tool.clarification_content:
                        # Lines are truncated (not wrapped), so 1 logical line = 1 display line
                        logical_lines = self._measure_content_lines(tool.clarification_content)

                        # Calculate max content lines dynamically (same as _render_clarification_prompt)
                        overhead = self._calculate_prompt_overhead(tool)
                        available_space = self._visible_height - overhead
                        max_content_lines = max(3, available_space)

                        if logical_lines > max_content_lines:
                            # Truncation will occur: lines_at_start + ellipsis (1) + lines_at_end
                            height += max_content_lines
                        else:
                            # No truncation - all lines shown
                            height += logical_lines
                        continue

                    # Legacy flow: no clarification_content, use prompt_lines
                    if not tool.clarification_prompt_lines:
                        continue

                    # Previously answered questions
                    if tool.clarification_answered:
                        height += len(tool.clarification_answered)

                    # Current question prompt box
                    if tool.clarification_prompt_lines:
                        prompt_lines = tool.clarification_prompt_lines
                        max_prompt_lines = 18
                        max_box_width = max(60, self._console_width - 22) if self._console_width > 40 else 60
                        box_width = min(max_box_width, max(_visible_len(line) for line in prompt_lines) + 4)
                        content_width = box_width - 4

                        # First count ALL wrapped lines to decide truncation (matches render logic)
                        total_wrapped_lines = 0
                        for prompt_line in prompt_lines:
                            wrapped = _wrap_visible(prompt_line, content_width)
                            total_wrapped_lines += len(wrapped)

                        if total_wrapped_lines > max_prompt_lines:
                            # Truncation triggered - render shows:
                            # - max_lines_before_truncation content lines
                            # - 1 truncation message
                            # - Last line (may wrap to multiple lines)
                            max_lines_before_truncation = max_prompt_lines - 3

                            # Calculate wrapped lines for last line
                            last_line = prompt_lines[-1]
                            last_wrapped = _wrap_visible(last_line, content_width)
                            last_line_count = len(last_wrapped)

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
                # Output preview - always shown when block is expanded
                # Finalized ToolBlocks cap at 70% of visible height (same as file output)
                if tool.output_lines:
                    total_lines = len(tool.output_lines)
                    max_display_lines = max(5, int(self._visible_height * 0.7))
                    display_count = min(total_lines, max_display_lines)
                    height += display_count
                    # Truncation indicator if content exceeds display
                    if total_lines > max_display_lines:
                        height += 1
                # Error message
                if not tool.success and tool.error_message:
                    height += 1
                # Clarification summary (Q&A table) - always shown when block is expanded
                if tool.clarification_summary:
                    height += 1  # header ("Answers (N)")
                    height += len(tool.clarification_summary)  # One line per Q&A pair
                # File output content - always shown when block is expanded
                if tool.file_output_lines:
                    height += 1  # header ("Content")
                    total_lines = len(tool.file_output_lines)
                    # Use 70% of visible height for file content display
                    max_display_lines = max(5, int(self._visible_height * 0.7))
                    display_count = min(total_lines, max_display_lines)
                    height += display_count
                    # Scroll indicators (up/down)
                    if total_lines > max_display_lines:
                        height += 2
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
            toggle_key = self._format_key_hint("pager_next")  # Space key toggles expand
            exit_key = self._format_key_hint("tool_exit")
            has_output = (selected_tool.output_lines and len(selected_tool.output_lines) > 0) or \
                         (selected_tool.file_output_lines and len(selected_tool.file_output_lines) > 0)
            if selected_tool.expanded and has_output:
                output.append(f"  ───  {nav_up}/{nav_down} scroll, {toggle_key} collapse, {exit_key} exit [{pos}/{total}]", style=self._style("hint", "dim"))
            elif has_output:
                output.append(f"  ───  {nav_up}/{nav_down} nav, {toggle_key} expand, {exit_key} exit [{pos}/{total}]", style=self._style("hint", "dim"))
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

                # Show expand icon only if tool has content to expand
                has_output = (tool.output_lines and len(tool.output_lines) > 0) or \
                             (tool.file_output_lines and len(tool.file_output_lines) > 0)
                if self._tool_nav_active and has_output:
                    expand_icon = "▾" if tool.expanded else "▸"
                else:
                    expand_icon = " " if self._tool_nav_active else ""
                row_style = "reverse" if is_selected else self._style("muted", "dim")

                output.append("\n")
                if self._tool_nav_active:
                    output.append(f"  {expand_icon} {connector} ", style=row_style)
                else:
                    output.append(f"    {connector} ", style=row_style)

                tool_display_name = tool.display_name or tool.name
                output.append(tool_display_name, style=row_style)
                # Show single-line args summary only when display override is active
                # or multi-line dict is not available; otherwise params render below
                if tool.display_args_summary is not None:
                    if tool.display_args_summary:
                        output.append(f"({tool.display_args_summary})", style=row_style)
                elif not tool.tool_args_dict and tool.args_summary:
                    output.append(f"({tool.args_summary})", style=row_style)
                output.append(f" {status_icon}", style=status_style)

                if tool.permission_state == "granted" and tool.permission_method:
                    indicator = self._get_approval_indicator(tool.permission_method)
                    if indicator:
                        output.append(f" {indicator}", style=self._style("tool_indicator", "dim cyan"))

                if tool.completed and tool.duration_seconds is not None:
                    output.append(f" ({tool.duration_seconds:.1f}s)", style=self._style("tool_duration", "dim"))

                # Multi-line parameter display (when dict is available and no display override)
                if tool.display_args_summary is None and tool.tool_args_dict:
                    param_style = self._style("muted", "dim")
                    self._render_tool_params_multiline(output, tool, is_last, param_style, wrap_width=wrap_width)

                # Tool output preview
                show_output = tool.expanded if self._tool_nav_active else True
                if show_output and tool.show_output and tool.output_lines:
                    self._render_tool_output_lines(output, tool, is_last, wrap_width=wrap_width)

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

                # Permission prompt - render if pending (with or without prompt_lines for unified flow)
                if tool.permission_state == "pending":
                    self._render_permission_prompt(output, tool, is_last)

                # Clarification prompt - render if pending (with or without prompt_lines for unified flow)
                if tool.clarification_state == "pending":
                    self._render_clarification_prompt(output, tool, is_last)

                # Clarification summary table (Q&A pairs when expanded)
                show_summary = tool.expanded if self._tool_nav_active else True
                if show_summary and tool.completed and tool.clarification_summary:
                    self._render_clarification_summary(output, tool, is_last)

                # File output content (preserved from permission prompt when expanded)
                show_file_output = tool.expanded if self._tool_nav_active else True
                if show_file_output and tool.completed and tool.file_output_lines:
                    self._render_file_output(output, tool, is_last)
        else:
            # Collapsed view
            if pending_tool:
                output.append("  ⏳ ", style=self._style("tool_pending", "bold yellow"))
            elif show_spinner:
                frame = self.SPINNER_FRAMES[self._spinner_index]
                output.append(f"  {frame} ", style=self._style("spinner", "cyan"))
            else:
                output.append("  ▸ ", style=self._style("tool_border", "dim"))

            # Build tool summaries with names (like finalized tools do)
            tool_summaries = []
            for tool in self._active_tools:
                tool_display_name = tool.display_name or tool.name
                if tool.completed:
                    status_icon = "✓" if tool.success else "✗"
                    summary = f"{tool_display_name} {status_icon}"
                else:
                    summary = f"{tool_display_name}..."
                tool_summaries.append(summary)

            output.append(f"{tool_count} tool{'s' if tool_count != 1 else ''}: ", style=self._style("tool_border", "dim"))
            output.append(" ".join(tool_summaries), style=self._style("tool_border", "dim"))

    def _render_scrollable_content(
        self,
        output: Text,
        lines: List[str],
        scroll_offset: int,
        display_count: int,
        is_last: bool,
        preserve_ansi: bool = False,
        style: Optional[str] = None,
        smart_truncate: bool = False
    ) -> None:
        """Render scrollable content with scroll indicators or smart truncation.

        Args:
            output: Text object to append to.
            lines: List of content lines to render.
            scroll_offset: Current scroll position (0 = show most recent).
            display_count: Max lines to show at once.
            is_last: Whether this is the last tool in the list.
            preserve_ansi: If True, use _truncate_line_to_width for ANSI preservation.
            style: Style to apply to lines (only used when preserve_ansi=False).
            smart_truncate: If True, show beginning + "... N lines not shown ..." + end
                           instead of scroll indicators. Better for content where both
                           beginning and end are important (like notebook input/output).
        """
        continuation = "   " if is_last else "│  "
        prefix = "    "
        total_lines = len(lines)

        max_line_width = max(40, self._console_width - 20) if self._console_width > 60 else 40

        # Find the natural width of the content (max content width across all lines)
        # This preserves background styling up to the widest content
        if preserve_ansi and lines:
            natural_width = max((self._get_content_width(line) for line in lines), default=0)
            target_width = min(natural_width, max_line_width)
        else:
            target_width = max_line_width

        def render_line(line: str) -> None:
            """Helper to render a single line with proper styling."""
            output.append("\n")
            output.append(f"{prefix}{continuation}   ", style=self._style("tree_connector", "dim"))
            if preserve_ansi:
                output.append_text(self._truncate_line_to_width(line, target_width, max_line_width))
            else:
                # -1 italic margin: compensates for terminal italic font slant
                effective_width = max_line_width - 1
                if _display_width(line) > effective_width:
                    # Truncate by display width, not character count
                    truncated = _truncate_to_display_width(line, effective_width - 3)
                    display_line = truncated + "..."
                else:
                    display_line = line
                output.append(display_line, style=self._style(style or "tool_output", "#87D7D7 italic"))

        if smart_truncate:
            # Use shared helper for smart truncation (beginning + ellipsis + end)
            indent = f"{prefix}{continuation}   "
            self._render_truncated_lines(
                output=output,
                lines=lines,
                max_display_lines=display_count,
                indent=indent,
                target_width=target_width,
                max_width=max_line_width,
                preserve_ansi=preserve_ansi
            )
        else:
            # Standard scrollable view
            end_idx = total_lines - scroll_offset
            start_idx = max(0, end_idx - display_count)
            lines_above = start_idx
            lines_below = scroll_offset
            visible_lines = lines[start_idx:end_idx]

            if lines_above > 0:
                output.append("\n")
                output.append(f"{prefix}{continuation}   ", style=self._style("tree_connector", "dim"))
                scroll_up_key = self._format_key_hint("nav_up")
                output.append(f"▲ {lines_above} more line{'s' if lines_above != 1 else ''} ({scroll_up_key} to scroll)", style=self._style("scroll_indicator", "dim italic"))

            for line in visible_lines:
                render_line(line)

            if lines_below > 0:
                output.append("\n")
                output.append(f"{prefix}{continuation}   ", style=self._style("tree_connector", "dim"))
                scroll_down_key = self._format_key_hint("nav_down")
                output.append(f"▼ {lines_below} more line{'s' if lines_below != 1 else ''} ({scroll_down_key} to scroll)", style=self._style("scroll_indicator", "dim italic"))

    def _render_tool_params_multiline(
        self,
        output: Text,
        tool: 'ActiveToolCall',
        is_last: bool,
        style: str,
        wrap_width: Optional[int] = None,
    ) -> None:
        """Render tool parameters as one key: value pair per line.

        Each parameter is rendered on its own indented line below the tool
        name, using tree continuation characters to maintain the visual
        tree structure.  Values are truncated with ``...`` only when they
        exceed the available terminal width.

        Args:
            output: Rich Text object to append to.
            tool: The tool whose parameters to render.
            is_last: Whether this is the last tool in the block (affects
                tree continuation character: ``│`` vs space).
            style: Style string to apply to the parameter lines.
            wrap_width: Effective content width (already accounts for debug
                line-number gutters etc.).  Falls back to
                ``self._console_width`` when not supplied.
        """
        if not tool.tool_args_dict:
            return

        effective_width = wrap_width if wrap_width is not None else self._console_width
        cont_char = " " if is_last else "│"
        prefix = f"    {cont_char}  "
        prefix_width = len(prefix)
        # "key: " takes len(key) + 2 chars
        for key, value in tool.tool_args_dict.items():
            key_label = f"{key}: "
            available = max(10, effective_width - prefix_width - len(key_label))
            formatted = format_tool_arg_value(value, available)
            output.append("\n")
            output.append(prefix, style=self._style("tree_connector", "dim"))
            output.append(f"{key_label}{formatted}", style=style)

    def _render_tool_output_lines(self, output: Text, tool: 'ActiveToolCall', is_last: bool,
                                   finalized: bool = False,
                                   wrap_width: Optional[int] = None) -> None:
        """Render output lines for a tool.

        Uses the same approach as _render_permission_prompt: calculates available
        space dynamically and calls _render_truncated_lines directly.

        For notebook output (content with <nb-row> markers), renders as a 2-column
        table with labels (In[n]:, Out[n]:) on the left and content on the right.

        When the tool has been escalated to the popup, shows the frozen first
        few lines plus an ellipsis with line count (the popup has the full output).

        Args:
            finalized: If True, cap display lines like file output (70% of visible
                       height) instead of filling all available space.
            wrap_width: Effective content width (already accounts for debug
                line-number gutters etc.).  Falls back to
                ``self._console_width`` when not supplied.
        """
        effective_width = wrap_width if wrap_width is not None else self._console_width
        continuation = "   " if is_last else "│  "
        prefix = "    "

        # Escalated tools: show frozen preview lines + ellipsis (popup has the full output)
        # Skip for completed/finalized tools — they should show standard truncated output
        if (tool.escalation_state == "escalated" and not finalized
                and not tool.completed and tool.inline_frozen_lines):
            indent = f"{prefix}{continuation}     "
            indent_width = len(indent)
            max_width = max(20, effective_width - indent_width)
            natural_width = max((self._get_content_width(line) for line in tool.inline_frozen_lines), default=0)
            target_width = min(natural_width, max_width)
            for line in tool.inline_frozen_lines:
                output.append("\n")
                output.append(indent, style=self._style("tree_connector", "dim"))
                output.append_text(self._truncate_line_to_width(line, target_width, max_width))
            # Ellipsis with total line count
            total = len(tool.output_lines) if tool.output_lines else 0
            hidden = total - len(tool.inline_frozen_lines)
            if hidden > 0:
                output.append("\n")
                output.append(indent, style=self._style("tree_connector", "dim"))
                output.append(f"... {hidden} more lines ...", style=self._style("muted", "dim italic"))
            return

        content_lines = tool.output_lines

        # Check if content contains security warning markers
        content_text = "\n".join(content_lines)
        if "<security-warning " in content_text:
            # Extract and render security warnings separately, then render remaining content
            self._render_with_security_warnings(output, content_text, prefix, continuation, is_last)
            return

        # Check if content contains notebook row markers
        if "<nb-row " in content_text:
            self._render_notebook_rows(output, content_text, prefix, continuation, is_last)
            return

        # Same indent structure as permission prompt
        indent = f"{prefix}{continuation}     "
        indent_width = len(indent)

        # Calculate available width for content (same as permission prompt)
        max_width = max(20, effective_width - indent_width)

        # Find the natural width of the content (max content width across all lines)
        # This preserves background styling up to the widest content
        natural_width = max((self._get_content_width(line) for line in content_lines), default=0)
        target_width = min(natural_width, max_width)

        # Calculate max content lines dynamically based on actual overhead
        overhead = self._calculate_tool_output_overhead(tool)
        available_space = self._visible_height - overhead
        # Minimum 3 to show something useful even on tiny terminals
        max_content_lines = max(3, available_space)

        # Finalized ToolBlocks: cap like file output (70% of visible height)
        if finalized:
            max_content_lines = max(5, int(self._visible_height * 0.7))

        # Use shared helper for smart truncation (beginning + ellipsis + end)
        self._render_truncated_lines(
            output=output,
            lines=content_lines,
            max_display_lines=max_content_lines,
            indent=indent,
            target_width=target_width,
            max_width=max_width,
            preserve_ansi=True
        )

    def _render_notebook_rows(self, output: Text, content: str, prefix: str, continuation: str, is_last: bool) -> None:
        """Render notebook content with <nb-row> markers as a 2-column table.

        Format:
            In [1]:  │ code line 1
                     │ code line 2
            Out [1]: │ result

        Supports progressive rendering during streaming - incomplete markers
        are rendered with whatever content is available so far.
        """
        # Pattern to match complete <nb-row type="..." label="...">content</nb-row>
        # Note: Use (?:\n) instead of \s*\n? to preserve leading spaces in content
        complete_pattern = re.compile(
            r'<nb-row\s+type="([^"]+)"\s+label="([^"]*)">(?:\n)(.*?)(?:\n)?</nb-row>',
            re.DOTALL
        )
        # Pattern to match incomplete (streaming) marker - has opening but no closing
        incomplete_pattern = re.compile(
            r'<nb-row\s+type="([^"]+)"\s+label="([^"]*)">(.*)$',
            re.DOTALL
        )

        # Find label width for alignment (use max of common labels)
        label_width = 10  # "Out [99]:" is 9 chars, pad to 10 for alignment

        # Calculate base indent and available content width
        base_indent = f"{prefix}{continuation}"
        separator_width = 3  # " │ " = space + bar + space
        total_prefix_width = len(base_indent) + label_width + separator_width
        max_content_width = max(20, self._console_width - total_prefix_width)

        # Process each complete notebook row
        last_end = 0
        for match in complete_pattern.finditer(content):
            # Render any text before this row (shouldn't happen normally)
            if match.start() > last_end:
                pre_text = content[last_end:match.start()].strip()
                if pre_text:
                    output.append("\n")
                    output.append(base_indent, style=self._style("tree_connector", "dim"))
                    output.append(pre_text)

            cell_type = match.group(1)
            label = match.group(2)
            # Use strip('\n') to preserve leading space indentation from code formatter
            cell_content = match.group(3).strip('\n')

            # Render the row
            self._render_single_notebook_row(
                output, cell_type, label, cell_content,
                base_indent, label_width, max_content_width
            )

            last_end = match.end()

        # Check for incomplete marker after all complete ones (streaming case)
        remaining = content[last_end:]
        incomplete_match = incomplete_pattern.search(remaining)
        if incomplete_match:
            # Render any text before the incomplete marker
            if incomplete_match.start() > 0:
                pre_text = remaining[:incomplete_match.start()].strip()
                if pre_text:
                    output.append("\n")
                    output.append(base_indent, style=self._style("tree_connector", "dim"))
                    output.append(pre_text)

            # Render the incomplete marker with available content (progressive streaming)
            cell_type = incomplete_match.group(1)
            label = incomplete_match.group(2)
            # Use strip('\n') to preserve leading space indentation from code formatter
            cell_content = incomplete_match.group(3).strip('\n')

            # Only render if there's actual content (not just the opening tag)
            if cell_content:
                self._render_single_notebook_row(
                    output, cell_type, label, cell_content,
                    base_indent, label_width, max_content_width
                )
        elif last_end < len(content):
            # Render any trailing text that's not part of a marker
            trailing = remaining.strip()
            if trailing:
                output.append("\n")
                output.append(base_indent, style=self._style("tree_connector", "dim"))
                output.append(trailing)

    def _render_single_notebook_row(
        self,
        output: Text,
        cell_type: str,
        label: str,
        content: str,
        base_indent: str,
        label_width: int,
        max_content_width: int
    ) -> None:
        """Render a single notebook row with label and content columns.

        Args:
            output: Text object to append to
            cell_type: Type of cell (input, stdout, result, error, etc.)
            label: Label like "In [1]:" or "" for stdout
            content: Cell content (may include code fences with ANSI)
            base_indent: Base indentation string
            label_width: Width reserved for label column
            max_content_width: Maximum width for content column
        """
        # Split content into lines
        lines = content.split('\n')
        if not lines:
            return

        # Strip the 4-space indent added by code_block_formatter for notebook display
        # The code_block_formatter adds "    " (4 spaces) to each line for visual distinction,
        # but notebook cells have their own layout so this indent is unnecessary
        if lines and lines[0].startswith('    '):
            lines = [line[4:] if line.startswith('    ') else line for line in lines]

        # Determine label style based on cell type
        if cell_type == "input":
            label_style = self._style("notebook_input_label", "bold green")
        elif cell_type in ("result", "display", "stdout"):
            label_style = self._style("notebook_output_label", "bold cyan")
        elif cell_type in ("error", "stderr"):
            label_style = self._style("notebook_error_label", "bold red")
        else:
            label_style = self._style("muted", "dim")

        # Column layout: [base_indent][label_col + space][separator][content]
        # Total prefix width after base_indent: label_width + 1 + 2 = label_width + 3
        separator = "│ "  # 2 chars: bar + space

        # First line: label right-aligned in label_width + space + separator
        # Continuation: (label_width + 1) spaces + separator
        padded_label = label.rjust(label_width) if label else " " * label_width
        first_line_prefix = padded_label + " "  # label_width + 1 chars
        continuation_prefix = " " * (label_width + 1) + separator  # label_width + 1 + 2 chars

        # Render first line with label
        output.append("\n")
        output.append(base_indent, style=self._style("tree_connector", "dim"))
        output.append(first_line_prefix, style=label_style if label else self._style("tree_connector", "dim"))
        output.append(separator, style=self._style("tree_connector", "dim"))

        # First line of content
        first_line = lines[0] if lines else ""
        if '\x1b[' in first_line:
            output.append_text(self._truncate_line_to_width(first_line, max_content_width, max_content_width))
        else:
            if len(first_line) > max_content_width:
                output.append(first_line[:max_content_width - 1] + "…")
            else:
                output.append(first_line)

        # Continuation lines use pre-computed prefix for consistent alignment
        for line in lines[1:]:
            output.append("\n")
            output.append(base_indent, style=self._style("tree_connector", "dim"))
            output.append(continuation_prefix, style=self._style("tree_connector", "dim"))

            if '\x1b[' in line:
                output.append_text(self._truncate_line_to_width(line, max_content_width, max_content_width))
            else:
                if len(line) > max_content_width:
                    output.append(line[:max_content_width - 1] + "…")
                else:
                    output.append(line)

    def _extract_and_render_security_warnings(self, output: Text, content: str, prefix: str, continuation: str) -> str:
        """Extract and render security warnings, returning remaining content.

        Extracts <security-warning> blocks, renders them with colored styling,
        and returns the content with warning blocks removed so normal rendering
        can continue.

        Args:
            output: Text object to append rendered warnings to.
            content: Content that may contain security warning markers.
            prefix: Line prefix for indentation.
            continuation: Continuation character for tree structure.

        Returns:
            Content with security warning blocks removed.
        """
        import re

        indent = f"{prefix}{continuation}     "

        # Pattern to extract security warning blocks
        # Note: Use (?:\n) instead of \n? to preserve leading spaces in content
        warning_pattern = re.compile(
            r'<security-warning\s+level="([^"]+)">(?:\n)(.*?)(?:\n)?</security-warning>(?:\n)?',
            re.DOTALL
        )

        # Find all warnings
        warnings = warning_pattern.findall(content)
        # Remove warning blocks from content, strip only newlines to preserve leading spaces
        remaining_content = warning_pattern.sub('', content)
        remaining_content = remaining_content.strip('\n')
        # Normalize multiple consecutive newlines to single newlines
        import re as re_inner
        remaining_content = re_inner.sub(r'\n{3,}', '\n\n', remaining_content)

        # Render warnings with colored styling
        for level, warning_text in warnings:
            # Choose style and icon based on level
            if level == "error":
                icon = "⚠"
                header_style = self._style("security_error_header", "bold red")
                text_style = self._style("security_error_text", "red")
            elif level == "warning":
                icon = "⚠"
                header_style = self._style("security_warning_header", "bold yellow")
                text_style = self._style("security_warning_text", "yellow")
            else:  # info
                icon = "ℹ"
                header_style = self._style("security_info_header", "bold cyan")
                text_style = self._style("security_info_text", "cyan")

            # Render header
            output.append("\n")
            output.append(indent, style=self._style("tree_connector", "dim"))
            output.append(f"{icon} Security Analysis", style=header_style)

            # Render warning lines
            for line in warning_text.strip().split('\n'):
                output.append("\n")
                output.append(indent, style=self._style("tree_connector", "dim"))
                # Style the bracketed level markers
                if line.startswith('['):
                    bracket_end = line.find(']')
                    if bracket_end > 0:
                        marker = line[:bracket_end + 1]
                        rest = line[bracket_end + 1:]
                        output.append(marker, style=header_style)
                        output.append(rest, style=text_style)
                    else:
                        output.append(line, style=text_style)
                else:
                    output.append(line, style=text_style)

        # Add blank line after warnings to separate from code block
        if warnings:
            output.append("\n")

        return remaining_content

    def _render_with_security_warnings(self, output: Text, content: str, prefix: str, continuation: str, is_last: bool) -> None:
        """Render content that contains security warning markers.

        Extracts <security-warning> blocks and renders them with colored styling
        (yellow for warning, red for error), then renders remaining content normally.

        Format:
            ⚠ Security Analysis
            Detected: 1 medium, 1 low
            [MEDIUM] Import of 'os' module (line 1)
        """
        import re

        # Pattern to extract security warning blocks
        # Note: Use (?:\n) instead of \n? to preserve leading spaces in content
        warning_pattern = re.compile(
            r'<security-warning\s+level="([^"]+)">(?:\n)(.*?)(?:\n)?</security-warning>(?:\n)?',
            re.DOTALL
        )

        # Find all warnings
        warnings = warning_pattern.findall(content)
        # Remove warning blocks from content, strip only newlines to preserve leading spaces
        remaining_content = warning_pattern.sub('', content)
        remaining_content = remaining_content.strip('\n')
        # Normalize multiple consecutive newlines to single newlines
        import re as re_inner
        remaining_content = re_inner.sub(r'\n{3,}', '\n\n', remaining_content)

        indent = f"{prefix}{continuation}     "
        indent_width = len(indent)
        max_width = max(20, self._console_width - indent_width)

        # Render warnings first with colored styling
        # Use semantic styles from theme for consistent theming
        for level, warning_text in warnings:
            # Choose style and icon based on level
            if level == "error":
                icon = "⚠"
                header_style = self._style("security_error_header", "bold red")
                text_style = self._style("security_error_text", "red")
            elif level == "warning":
                icon = "⚠"
                header_style = self._style("security_warning_header", "bold yellow")
                text_style = self._style("security_warning_text", "yellow")
            else:  # info
                icon = "ℹ"
                header_style = self._style("security_info_header", "bold cyan")
                text_style = self._style("security_info_text", "cyan")

            # Render header
            output.append("\n")
            output.append(indent, style=self._style("tree_connector", "dim"))
            output.append(f"{icon} Security Analysis", style=header_style)

            # Render warning lines
            for line in warning_text.strip().split('\n'):
                output.append("\n")
                output.append(indent, style=self._style("tree_connector", "dim"))
                # Style the bracketed level markers
                if line.startswith('['):
                    bracket_end = line.find(']')
                    if bracket_end > 0:
                        marker = line[:bracket_end + 1]
                        rest = line[bracket_end + 1:]
                        output.append(marker, style=header_style)
                        output.append(rest, style=text_style)
                    else:
                        output.append(line, style=text_style)
                else:
                    output.append(line, style=text_style)

            output.append("\n")  # Blank line after warnings

        # Render remaining content (code block, permission prompt, options)
        if remaining_content:
            remaining_lines = remaining_content.split('\n')

            # Handle options line focus highlighting if permission options are set
            if self._permission_response_options:
                for i in range(len(remaining_lines) - 1, -1, -1):
                    if self._is_options_line(remaining_lines[i]):
                        # Replace options line with focused version
                        remaining_lines[i] = self._render_focused_options_line()
                        break

            # Check for notebook rows in remaining content
            if "<nb-row " in remaining_content:
                self._render_notebook_rows(output, remaining_content, prefix, continuation, is_last)
            else:
                for line in remaining_lines:
                    output.append("\n")
                    output.append(indent, style=self._style("tree_connector", "dim"))
                    if len(line) > max_width:
                        output.append(line[:max_width - 3] + "...", style=self._style("tool_output", "dim"))
                    else:
                        output.append(line, style=self._style("tool_output", "dim"))

    def _calculate_tool_output_overhead(self, tool: 'ActiveToolCall') -> int:
        """Calculate lines of overhead before tool output content.

        Similar to _calculate_prompt_overhead but for tool output display.
        For finalized ToolBlocks (no active tools/placeholder), returns minimal overhead.
        """
        overhead = 0

        # For finalized ToolBlocks, _tool_placeholder_index is None
        # Return minimal overhead (just the header line)
        if self._tool_placeholder_index is None:
            return 2  # Separator + header

        # Context lines kept at top by scroll logic
        lines_before_tree = sum(
            self._get_item_display_lines(item)
            for i, item in enumerate(self._lines)
            if i < self._tool_placeholder_index
        )
        context_lines = min(1, lines_before_tree)
        overhead += context_lines

        # Tool tree header (1 line when expanded)
        overhead += 1

        # Lines for tools up to and including the current one
        for t in self._active_tools:
            overhead += 1  # Tool name line
            if t is tool:
                break
            # Output lines for tools before the current one
            if t.expanded and t.output_lines:
                overhead += min(len(t.output_lines), t.output_display_lines)

        return overhead

    def _calculate_prompt_overhead(self, tool: 'ActiveToolCall') -> int:
        """Calculate actual lines of overhead before permission/clarification content.

        This includes:
        - Context lines kept at top by scroll logic
        - Tool tree header line
        - Tool name lines for tools up to and including current tool
        - Output lines for tools before current tool
        - Prompt header line ("Permission required" or "Clarification needed")
        """
        overhead = 0

        # Calculate context lines that scroll logic will keep at top
        # (mirrors the logic in scroll_to_show_tool_tree)
        lines_before_tree = sum(
            self._get_item_display_lines(item)
            for i, item in enumerate(self._lines)
            if i < self._tool_placeholder_index
        )
        context_lines = min(1, lines_before_tree)  # Scroll keeps 1 line max for pending prompts
        overhead += context_lines

        # Tool tree header (1 line when expanded - permissions force expanded view)
        overhead += 1

        # Lines for tools up to and including the current one
        for t in self._active_tools:
            overhead += 1  # Tool name line
            if t is tool:
                break
            # Output lines for tools before the current one
            if t.output_lines:
                overhead += len(t.output_lines)

        # "Permission required" header line
        overhead += 1

        return overhead

    def _render_truncated_lines(
        self,
        output: Text,
        lines: List[str],
        max_display_lines: int,
        indent: str,
        target_width: int,
        max_width: int,
        preserve_ansi: bool = True
    ) -> None:
        """Render lines with smart truncation (beginning + ellipsis + end).

        This is a shared helper used by tool output, permission prompts, and
        clarification prompts to display content that may exceed available space.

        Args:
            output: Text object to append to.
            lines: List of content lines to render.
            max_display_lines: Maximum lines to show (including ellipsis line).
            indent: Indentation string for each line.
            target_width: Target width for truncation (natural content width).
            max_width: Maximum allowed width.
            preserve_ansi: If True, use ANSI-aware truncation.
        """
        total_lines = len(lines)

        if total_lines > max_display_lines:
            # Truncate in the middle: show beginning, ellipsis, and end
            # Reserve more lines for the end (typically has results/output)
            lines_at_start = max_display_lines // 3
            lines_at_end = max_display_lines - lines_at_start - 1  # -1 for ellipsis line
            hidden_count = total_lines - lines_at_start - lines_at_end

            # Render first N lines
            for line in lines[:lines_at_start]:
                output.append("\n")
                output.append(indent, style=self._style("tree_connector", "dim"))
                if preserve_ansi:
                    output.append_text(self._truncate_line_to_width(line, target_width, max_width))
                else:
                    # -1 italic margin: tool_output theme style is italic
                    effective_width = max_width - 1
                    if _display_width(line) > effective_width:
                        truncated = _truncate_to_display_width(line, effective_width - 3)
                        output.append(truncated + "...", style=self._style("tool_output", "dim"))
                    else:
                        output.append(line, style=self._style("tool_output", "dim"))

            # Render ellipsis indicator
            output.append("\n")
            output.append(indent, style=self._style("tree_connector", "dim"))
            output.append(f"... {hidden_count} lines not shown ...", style=self._style("muted", "dim italic"))

            # Render last M lines
            for line in lines[-lines_at_end:]:
                output.append("\n")
                output.append(indent, style=self._style("tree_connector", "dim"))
                if preserve_ansi:
                    output.append_text(self._truncate_line_to_width(line, target_width, max_width))
                else:
                    # -1 italic margin: tool_output theme style is italic
                    effective_width = max_width - 1
                    if _display_width(line) > effective_width:
                        truncated = _truncate_to_display_width(line, effective_width - 3)
                        output.append(truncated + "...", style=self._style("tool_output", "dim"))
                    else:
                        output.append(line, style=self._style("tool_output", "dim"))
        else:
            # Content fits, render all lines
            for line in lines:
                output.append("\n")
                output.append(indent, style=self._style("tree_connector", "dim"))
                if preserve_ansi:
                    output.append_text(self._truncate_line_to_width(line, target_width, max_width))
                else:
                    # -1 italic margin: tool_output theme style is italic
                    effective_width = max_width - 1
                    if _display_width(line) > effective_width:
                        truncated = _truncate_to_display_width(line, effective_width - 3)
                        output.append(truncated + "...", style=self._style("tool_output", "dim"))
                    else:
                        output.append(line, style=self._style("tool_output", "dim"))

    def _get_content_width(self, line: str) -> int:
        """Get the display width of actual content in a line (excluding trailing whitespace).

        Uses _display_width for proper handling of wide characters and
        JAATO_AMBIGUOUS_WIDTH for CJK terminal compatibility.

        Args:
            line: The line (may contain ANSI codes).

        Returns:
            Display width of content (excluding trailing spaces).
        """
        if not line:
            return 0

        # Parse ANSI codes to get plain text
        if '\x1b[' in line:
            text = Text.from_ansi(line)
            plain = text.plain
        else:
            plain = line

        # Strip trailing spaces and measure content width
        content = plain.rstrip()
        if not content:
            return 0

        return _display_width(content)

    def _truncate_line_to_width(self, line: str, target_width: int, max_width: int, indicator: str = "▸") -> Text:
        """Truncate a line to target_width, adding indicator if it exceeds max_width.

        Uses _display_width for proper handling of wide characters and
        JAATO_AMBIGUOUS_WIDTH for CJK terminal compatibility.

        Args:
            line: The line to truncate (may contain ANSI codes).
            target_width: The natural width to truncate to (preserves background styling).
            max_width: Maximum allowed width - add indicator if content exceeds this.
            indicator: Character to show when line is truncated (default: ▸).

        Returns:
            Rich Text object, truncated to target_width with indicator if needed.
        """
        if not line:
            return Text("")

        # Parse ANSI codes to get styled text
        if '\x1b[' in line:
            text = Text.from_ansi(line)
        else:
            text = Text(line)

        # Get actual content width (excluding trailing spaces) using _display_width
        plain = text.plain
        content = plain.rstrip()
        content_width = _display_width(content) if content else 0

        # Determine if we need to show truncation indicator
        needs_indicator = content_width > max_width

        # Calculate final width: truncate to target_width but leave room for indicator if needed
        if needs_indicator:
            final_width = min(target_width, max_width) - 1
        else:
            final_width = min(target_width, max_width)

        if final_width <= 0:
            return Text(indicator, style="dim") if needs_indicator else Text("")

        # Truncate by character position to reach target display width
        # We need to find the character index that corresponds to final_width display columns
        current_width = 0
        char_count = 0
        for char in plain:
            char_width = _display_width(char)
            if current_width + char_width > final_width:
                break
            current_width += char_width
            char_count += 1

        # Copy the text up to char_count (preserves styling)
        if char_count > 0:
            result = text[:char_count]
        else:
            result = Text()

        if needs_indicator:
            result.append(indicator, style="dim")

        return result

    def _render_prerendered(self, text: str, width: int) -> Text:
        """Render pre-rendered content (mermaid pixel art), truncating rows that overflow.

        Each pixel row is truncated individually with _truncate_line_to_width
        so the art is cropped cleanly on the right instead of wrapping
        chaotically when the terminal is narrower than the rendered width.
        """
        clean = text[len(PRERENDERED_LINE_PREFIX):]
        rows = clean.split('\n')
        result = Text()
        for i, row in enumerate(rows):
            if i > 0:
                result.append("\n")
            result.append_text(self._truncate_line_to_width(row, width, width))
        return result

    def _render_table_to_text(self, table: Table, width: Optional[int] = None) -> Text:
        """Render a Rich Table to a Text object for appending to output.

        This utility enables using Rich's Table features (word-wrap, alignment,
        borders) while still composing output as Text objects.

        Args:
            table: Rich Table to render.
            width: Console width for rendering. Defaults to self._console_width.

        Returns:
            Text object containing the rendered table with ANSI styling preserved.
        """
        render_width = width or self._console_width or 80
        string_io = StringIO()
        temp_console = Console(
            file=string_io,
            force_terminal=True,
            width=render_width,
            no_color=False,
        )
        temp_console.print(table, end="")
        return Text.from_ansi(string_io.getvalue())

    def _render_permission_prompt(self, output: Text, tool: 'ActiveToolCall', is_last: bool) -> None:
        """Render permission prompt for a tool awaiting approval."""
        continuation = "   " if is_last else "│  "
        prefix = "    "
        prompt_lines = tool.permission_prompt_lines or []

        # Header - always show
        output.append("\n")
        output.append(f"{prefix}{continuation}", style=self._style("tree_connector", "dim"))
        output.append("  🔒 Permission required", style=self._style("permission_prompt", "bold yellow"))

        # Unified flow: permission_content contains formatted content to render inline
        if tool.permission_content:
            content_text = tool.permission_content

            # Extract and render security warnings with colored styling, then continue with remaining content
            if "<security-warning " in content_text:
                content_text = self._extract_and_render_security_warnings(output, content_text, prefix, continuation)

            indent = f"{prefix}{continuation}     "
            indent_width = len(indent)  # 12 chars
            content_lines = content_text.split('\n')

            # Search for options line from end (may not be exactly last due to trailing empty lines)
            # Options line typically looks like: "[y]es [n]o [a]lways ..."
            if self._permission_response_options:
                for i in range(len(content_lines) - 1, -1, -1):
                    if self._is_options_line(content_lines[i]):
                        # Replace options line with focused version
                        content_lines[i] = self._render_focused_options_line()
                        break

            # Calculate available width for content
            max_width = max(20, self._console_width - indent_width)

            # Find the natural width of the content (max content width across all lines)
            # This preserves background styling up to the widest content
            natural_width = max((self._get_content_width(line) for line in content_lines), default=0)
            target_width = min(natural_width, max_width)

            # Calculate max content lines dynamically based on actual overhead
            overhead = self._calculate_prompt_overhead(tool)
            available_space = self._visible_height - overhead
            # Minimum 3 to show something useful even on tiny terminals
            max_content_lines = max(3, available_space)

            # Log content structure for debugging
            last_lines_preview = [line[:60] for line in content_lines[-5:]]
            first_lines_preview = [line[:60] for line in content_lines[:3]]
            _trace(f"_render_permission_prompt: visible_height={self._visible_height}, "
                   f"overhead={overhead}, available_space={available_space}, "
                   f"content_lines={len(content_lines)}, max_content_lines={max_content_lines}, "
                   f"will_truncate={len(content_lines) > max_content_lines}")
            _trace(f"  first_3_lines={first_lines_preview}")
            _trace(f"  last_5_lines={last_lines_preview}")

            # Use shared helper for smart truncation (beginning + ellipsis + end)
            self._render_truncated_lines(
                output=output,
                lines=content_lines,
                max_display_lines=max_content_lines,
                indent=indent,
                target_width=target_width,
                max_width=max_width,
                preserve_ansi=True
            )
            return

        # Legacy flow: no permission_content means we use prompt_lines or show minimal indicator
        if not prompt_lines:
            return

        if tool.permission_format_hint == "diff":
            # Pre-formatted content (diff) - render lines directly without box
            # Handle embedded newlines by splitting and prefixing each visual line
            for line in prompt_lines:
                # Split on newlines to properly prefix each visual line
                visual_lines = line.split('\n')
                for visual_line in visual_lines:
                    output.append("\n")
                    output.append(f"{prefix}{continuation}  ", style=self._style("tree_connector", "dim"))
                    output.append(visual_line)
        else:
            # Standard box format
            # Use _visible_len to ignore ANSI escape codes when measuring
            # prompt line widths — prompt_lines may contain ANSI color codes
            # which are invisible but would inflate len() calculations.
            max_prompt_lines = 18
            max_box_width = max(60, self._console_width - 22) if self._console_width > 40 else 60
            box_width = min(max_box_width, max(_visible_len(line) for line in prompt_lines) + 4) if prompt_lines else 40
            content_width = box_width - 4

            # Wrap and potentially truncate lines (strip ANSI for clean box content)
            wrapped_lines: list[str] = []
            for line in prompt_lines:
                wrapped = _wrap_visible(line, content_width)
                wrapped_lines.extend(wrapped)

            # Check if truncation needed
            if len(wrapped_lines) > max_prompt_lines:
                max_before = max_prompt_lines - 3
                truncated_count = len(wrapped_lines) - max_before - 1
                display_lines = wrapped_lines[:max_before]
                display_lines.append(f"... ({truncated_count} more lines)")
                display_lines.append(wrapped_lines[-1])
            else:
                display_lines = wrapped_lines

            # Top border
            output.append("\n")
            output.append(f"{prefix}{continuation}  ", style=self._style("tree_connector", "dim"))
            output.append("┌" + "─" * (box_width - 2) + "┐", style=self._style("permission_text", "yellow"))

            # Content lines — use _ljust_visible so padding is based on
            # visible width, not byte length (handles residual ANSI codes)
            for line in display_lines:
                output.append("\n")
                output.append(f"{prefix}{continuation}  ", style=self._style("tree_connector", "dim"))
                padded = _ljust_visible(line, content_width)
                output.append("│ " + padded + " │", style=self._style("permission_text", "yellow"))

            # Bottom border
            output.append("\n")
            output.append(f"{prefix}{continuation}  ", style=self._style("tree_connector", "dim"))
            output.append("└" + "─" * (box_width - 2) + "┘", style=self._style("permission_text", "yellow"))

    def _is_options_line(self, line: str) -> bool:
        """Check if a line is the permission options line.

        The options line contains bracketed shortcuts like [y]es [n]o [a]lways.
        We detect it by looking for the pattern of multiple bracketed items.

        Args:
            line: The line to check.

        Returns:
            True if this appears to be an options line.
        """
        import re
        # Strip ANSI codes for pattern matching
        clean_line = re.sub(r'\x1b\[[0-9;?]*[A-Za-z~]', '', line)
        # Options line has multiple [x]word patterns
        # Pattern: [letter(s)]rest_of_word repeated multiple times
        pattern = r'\[[a-z]+\][a-z]*'
        matches = re.findall(pattern, clean_line.lower())
        # Need at least 3 options to be considered an options line
        return len(matches) >= 3

    def _render_focused_options_line(self) -> str:
        """Render the permission options line with the focused option highlighted.

        Uses ANSI escape codes for styling since the output is processed with preserve_ansi=True.

        Returns:
            The options line with the focused option highlighted using reverse video.
        """
        if not self._permission_response_options:
            return ""

        # ANSI escape codes for styling
        REVERSE = "\x1b[7m"  # Reverse video
        BOLD = "\x1b[1m"
        RESET = "\x1b[0m"
        DIM = "\x1b[2m"

        parts = []
        for i, option in enumerate(self._permission_response_options):
            # Extract option properties (handle both dict and object forms)
            if isinstance(option, dict):
                short = option.get('key', option.get('short', ''))
                full = option.get('label', option.get('full', ''))
            else:
                short = getattr(option, 'short', getattr(option, 'key', ''))
                full = getattr(option, 'full', getattr(option, 'label', ''))

            is_focused = (i == self._permission_focus_index)

            # Build the option text: [y]es or [once]
            if short != full and full.startswith(short):
                # Format: [y]es - short is prefix of full
                option_text = f"[{short}]{full[len(short):]}"
            else:
                # Format: [once] - short equals full or doesn't match
                option_text = f"[{full}]"

            # Apply styling based on focus state
            if is_focused:
                parts.append(f"{BOLD}{REVERSE} {option_text} {RESET}")
            else:
                parts.append(f"{DIM}{option_text}{RESET}")

        # Add hint at the end
        hint = f"{DIM}  ⇥ cycle  ↵ select{RESET}"

        return " ".join(parts) + hint

    def _render_clarification_prompt(self, output: Text, tool: 'ActiveToolCall', is_last: bool) -> None:
        """Render clarification prompt for a tool awaiting user input."""
        continuation = "   " if is_last else "│  "
        prefix = "    "
        prompt_lines = tool.clarification_prompt_lines or []

        # Header - always show with question progress if available
        output.append("\n")
        output.append(f"{prefix}{continuation}", style=self._style("tree_connector", "dim"))
        if tool.clarification_current_question and tool.clarification_total_questions:
            output.append(f"  ❓ Clarification needed ({tool.clarification_current_question}/{tool.clarification_total_questions})", style=self._style("clarification_required", "bold cyan"))
        else:
            output.append("  ❓ Clarification needed", style=self._style("clarification_required", "bold cyan"))

        # Unified flow: clarification_content contains formatted content to render inline
        if tool.clarification_content:
            indent = f"{prefix}{continuation}     "
            indent_width = len(indent)  # 12 chars
            content_lines = tool.clarification_content.split('\n')

            # Calculate available width for content
            max_width = max(20, self._console_width - indent_width)

            # Find the natural width of the content (max content width across all lines)
            # This preserves background styling up to the widest content
            natural_width = max((self._get_content_width(line) for line in content_lines), default=0)
            target_width = min(natural_width, max_width)

            # Calculate max content lines dynamically based on actual overhead
            overhead = self._calculate_prompt_overhead(tool)
            available_space = self._visible_height - overhead
            max_content_lines = max(3, available_space)

            # Use shared helper for smart truncation (beginning + ellipsis + end)
            self._render_truncated_lines(
                output=output,
                lines=content_lines,
                max_display_lines=max_content_lines,
                indent=indent,
                target_width=target_width,
                max_width=max_width,
                preserve_ansi=True
            )
            return

        # Legacy flow: no clarification_content means we use prompt_lines or show minimal indicator
        if not prompt_lines:
            return

        # Render prompt lines
        for line in prompt_lines:
            output.append("\n")
            output.append(f"{prefix}{continuation}  ", style=self._style("tree_connector", "dim"))
            output.append(line, style=self._style("clarification_label", "cyan"))

    def _render_clarification_summary(self, output: Text, tool: 'ActiveToolCall', is_last: bool) -> None:
        """Render clarification summary table (Q&A pairs) for a completed tool.

        Uses manual layout with dynamic column widths and leader dots filling
        between question and answer on the last line of each question.
        """
        continuation = "   " if is_last else "│  "
        prefix = "    "
        qa_pairs = tool.clarification_summary or []

        if not qa_pairs:
            return

        # Header
        output.append("\n")
        output.append(f"{prefix}{continuation}", style=self._style("tree_connector", "dim"))
        output.append(f"  📋 Answers ({len(qa_pairs)})", style=self._style("clarification_resolved", "bold green"))

        # Use a fixed content width for Q&A layout (not terminal width)
        # This ensures consistent, readable layout regardless of terminal size
        # The tool tree already has its own indentation we can't easily measure
        CONTENT_WIDTH = 70  # Reasonable width for Q&A content

        # Dynamic column sizing based on content
        # Answer column: sized to longest answer (min 10, max 40% of content width)
        max_answer_len = max(_display_width(a) for _, a in qa_pairs)
        answer_col_width = max(10, min(max_answer_len, int(CONTENT_WIDTH * 0.4)))

        # Question column: remaining space minus dots
        min_dots = 3
        question_col_width = CONTENT_WIDTH - answer_col_width - min_dots - 2

        question_style = self._style("clarification_question", "cyan")
        answer_style = self._style("clarification_answer", "green")
        dots_style = self._style("muted", "dim")

        for question, answer in qa_pairs:
            # Normalize question (collapse whitespace, remove newlines)
            question = ' '.join(question.split())

            # Wrap question to fit in question column
            wrapped = textwrap.wrap(question, width=question_col_width) or [question]

            # Render non-last lines (question only, no dots)
            for line in wrapped[:-1]:
                output.append("\n")
                output.append(f"{prefix}{continuation}  ", style=self._style("tree_connector", "dim"))
                output.append(line, style=question_style)

            # Last line: question + dots + answer (all on same line)
            # Dots fill from end of question to answer column start
            last_line = wrapped[-1]
            last_line_width = _display_width(last_line)
            dots_needed = question_col_width - last_line_width + min_dots

            output.append("\n")
            output.append(f"{prefix}{continuation}  ", style=self._style("tree_connector", "dim"))
            output.append(last_line, style=question_style)
            output.append(" " + "·" * max(min_dots, dots_needed) + " ", style=dots_style)
            output.append(answer, style=answer_style)

    def _render_file_output(self, output: Text, tool: 'ActiveToolCall', is_last: bool) -> None:
        """Render preserved file output content for a completed tool."""
        if not tool.file_output_lines:
            return

        continuation = "   " if is_last else "│  "
        prefix = "    "

        # Header - show content indicator
        output.append("\n")
        output.append(f"{prefix}{continuation}", style=self._style("tree_connector", "dim"))
        output.append("  📄 Content", style=self._style("file_output_header", "bold cyan"))

        # Calculate display count as 70% of visible height, with reasonable bounds
        max_display_lines = max(5, int(self._visible_height * 0.7))
        display_count = min(len(tool.file_output_lines), max_display_lines)

        # Use shared scrollable content renderer with ANSI preservation for diffs
        self._render_scrollable_content(
            output=output,
            lines=tool.file_output_lines,
            scroll_offset=tool.file_output_scroll_offset,
            display_count=display_count,
            is_last=is_last,
            preserve_ansi=True
        )

    def _render_tool_block(self, block: ToolBlock, output: Text, wrap_width: int) -> None:
        """Render a ToolBlock inline in the output."""
        tool_count = len(block.tools)
        _trace(f"_render_tool_block: block.expanded={block.expanded}, tool_count={tool_count}")

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
            toggle_key = self._format_key_hint("pager_next")  # Space key toggles expand
            exit_key = self._format_key_hint("tool_exit")
            has_output = (selected_tool.output_lines and len(selected_tool.output_lines) > 0) or \
                         (selected_tool.file_output_lines and len(selected_tool.file_output_lines) > 0)
            if selected_tool.expanded and has_output:
                # When expanded: arrows scroll output, space collapses
                output.append(f"  {nav_up}/{nav_down} scroll, {toggle_key} collapse, {exit_key} exit [{pos}/{total}]", style=self._style("hint", "dim"))
            elif has_output:
                # When collapsed but has output: arrows navigate, space expands
                output.append(f"  {nav_up}/{nav_down} nav, {toggle_key} expand, {exit_key} exit [{pos}/{total}]", style=self._style("hint", "dim"))
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

                # Determine if tool output should be shown:
                # - In nav mode: respect tool.expanded (user can toggle individual tools)
                # - Not in nav mode: always show output (consistent with active tools behavior)
                show_tool_output = tool.expanded if self._tool_nav_active else True

                # Expand indicator for tool output (only if tool has output or file content)
                has_output = (tool.output_lines and len(tool.output_lines) > 0) or \
                             (tool.file_output_lines and len(tool.file_output_lines) > 0)
                if has_output:
                    expand_icon = "▾" if show_tool_output else "▸"
                else:
                    expand_icon = " "

                # Selection highlight
                row_style = "reverse" if is_selected else self._style("muted", "dim")

                output.append("\n")
                output.append(f"  {expand_icon} {connector} ", style=row_style)
                # Use display_name if set (e.g., showing actual tool instead of askPermission)
                tool_display_name = tool.display_name or tool.name
                output.append(tool_display_name, style=row_style)
                # Show single-line args summary only when display override is active
                # or multi-line dict is not available; otherwise params render below
                if tool.display_args_summary is not None:
                    if tool.display_args_summary:
                        output.append(f"({tool.display_args_summary})", style=row_style)
                elif not tool.tool_args_dict and tool.args_summary:
                    output.append(f"({tool.args_summary})", style=row_style)
                output.append(f" {status_icon}", style=status_style)

                # Approval indicator
                if tool.permission_state == "granted" and tool.permission_method:
                    indicator = self._get_approval_indicator(tool.permission_method)
                    if indicator:
                        output.append(f" {indicator}", style=self._style("tool_indicator", "dim cyan"))

                # Duration
                if tool.duration_seconds is not None:
                    output.append(f" ({tool.duration_seconds:.1f}s)", style=self._style("tool_duration", "dim"))

                # Multi-line parameter display (when dict is available and no display override)
                if tool.display_args_summary is None and tool.tool_args_dict:
                    param_style = self._style("muted", "dim")
                    self._render_tool_params_multiline(output, tool, is_last, param_style, wrap_width=wrap_width)

                # Tool output - use shared rendering method with smart truncation
                if show_tool_output and tool.show_output and tool.output_lines:
                    self._render_tool_output_lines(output, tool, is_last, finalized=True, wrap_width=wrap_width)

                # Error message
                if not tool.success and tool.error_message:
                    output.append("\n")
                    continuation = "   " if is_last else "│  "
                    output.append(f"    {continuation}   ⚠ {tool.error_message}", style=self._style("tool_error", "red dim"))

                # Clarification summary table (Q&A pairs)
                if show_tool_output and tool.clarification_summary:
                    self._render_clarification_summary(output, tool, is_last)

                # File output content (preserved from permission prompt)
                if show_tool_output and tool.file_output_lines:
                    self._render_file_output(output, tool, is_last)
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

    def _wrap_ansi_text(self, text: str, width: int) -> Text:
        """Parse ANSI text and wrap it to the specified width.

        Args:
            text: Text containing ANSI escape codes.
            width: Maximum width for wrapping.

        Returns:
            Wrapped Text object with styles preserved.
        """
        # Parse ANSI codes into Rich Text
        content = Text.from_ansi(text)

        # Wrap using Rich's text wrapping (requires a console for measurement)
        if self._measure_console is None:
            self._measure_console = Console(width=width, force_terminal=True)

        wrapped_lines = content.wrap(self._measure_console, width)

        # Join wrapped lines back into a single Text with newlines
        result = Text()
        for i, wrapped_line in enumerate(wrapped_lines):
            if i > 0:
                result.append("\n")
            result.append_text(wrapped_line)

        return result

    def _add_debug_line_numbers(self, output: Text, start_line: int) -> Text:
        """Add debug line numbers to output text for debugging scroll/height issues.

        Each line gets prefixed with its display line number (1-based) to help
        identify where scroll calculation issues occur.

        Args:
            output: The rendered Text object.
            start_line: The starting display line number (0-based) for the visible range.

        Returns:
            New Text object with line numbers prepended to each line.
        """
        # Split preserves Rich styling on each line
        lines = output.split('\n')

        if not lines:
            return output

        # Calculate width for line numbers (based on max line number)
        max_line = start_line + len(lines)
        num_width = max(4, len(str(max_line)))  # Minimum 4 chars for padding

        # Build new output with line numbers
        result = Text()
        for i, line_text in enumerate(lines):
            line_num = start_line + i + 1  # 1-based display line number
            if i > 0:
                result.append("\n")
            # Prepend line number using semantic style (themeable)
            result.append(f"{line_num:>{num_width}}│ ", style=self._style("debug_line_number", "dim"))
            result.append_text(line_text)  # Preserve original styling

        return result

    def render(self, height: Optional[int] = None, width: Optional[int] = None) -> RenderableType:
        """Render the output buffer as Rich Text.

        Args:
            height: Optional height limit (in display lines).
            width: Optional width for calculating line wrapping.

        Returns:
            Rich renderable for the output panel.
        """
        _buffer_trace(
            f"render: lines={len(self._lines)} height={height} width={width} "
            f"scroll_offset={self._scroll_offset} spinner={self._spinner_active} "
            f"active_tools={len(self._active_tools)}"
        )

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
            if height != self._visible_height:
                _trace(f"render: visible_height changed from {self._visible_height} to {height}")
            self._visible_height = height

        # Work backwards from the end, using stored display line counts
        # First skip _scroll_offset lines, then collect 'height' lines
        # Include current block lines (streaming content) at the end
        all_items: List[Union[OutputLine, ToolBlock, ActiveToolsMarker]] = list(self._lines) + current_block_lines

        # Build line index mapping for search highlighting (map item ID -> index in self._lines)
        # This must be done before ActiveToolsMarker insertion to preserve original indices
        line_idx_map: Dict[int, int] = {id(item): i for i, item in enumerate(self._lines)}

        # Insert ActiveToolsMarker at placeholder position if active tools exist
        if self._active_tools and self._tool_placeholder_index is not None:
            # Insert marker at the placeholder position
            # Account for current_block_lines being appended at the end
            insert_pos = min(self._tool_placeholder_index, len(all_items))
            all_items.insert(insert_pos, ActiveToolsMarker())

        items_to_show: List[Union[OutputLine, ToolBlock, ActiveToolsMarker]] = []
        start_display_line = 0  # Track starting display line for debug numbering

        if height:
            # Tools render inline via ActiveToolsMarker - no separate space reservation needed
            # The marker's height is calculated in _get_item_display_lines()
            available_for_lines = height

            # Reserve space for the spinner when it will be rendered after items.
            # The spinner appends 1-3 extra lines that are NOT part of any item's
            # display_lines, so they must be subtracted from the budget.
            # Only reserve when at the bottom (scroll_offset==0) - when scrolled up,
            # the spinner at the bottom being clipped by the Panel is acceptable.
            if self._spinner_active and not self._active_tools and self._scroll_offset == 0:
                spinner_reserved = 1  # "thinking..." line
                if all_items:
                    spinner_reserved += 1  # blank line separator
                    if self._last_turn_source != "model":
                        spinner_reserved += 1  # model header line
                available_for_lines = max(1, available_for_lines - spinner_reserved)

            # Calculate total display lines (accounting for ToolBlocks)
            total_display_lines = sum(self._get_item_display_lines(item) for item in all_items)

            # Find the end position (bottom of visible window)
            # scroll_offset=0 means show the most recent content
            # scroll_offset>0 means we've scrolled up, showing older content
            end_display_line = total_display_lines - self._scroll_offset
            start_display_line = max(0, end_display_line - available_for_lines)

            # Log render viewport for debugging scroll issues
            has_pending = any(
                t.permission_state == "pending" or t.clarification_state == "pending"
                for t in self._active_tools
            ) if self._active_tools else False
            if has_pending:
                _trace(f"render: total={total_display_lines}, offset={self._scroll_offset}, "
                       f"start={start_display_line}, end={end_display_line}, available={available_for_lines}")

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

        # When debug line numbers are enabled, reduce wrap width to account for prefix
        # Prefix format: "{num:>4}│ " = 6 chars minimum (4-digit number + │ + space)
        if os.environ.get('JAATO_DEBUG_LINE_NUMBERS', '').lower() in ('1', 'true', 'yes'):
            wrap_width = max(20, wrap_width - 6)

        # Define wrap_text helper before the loops
        def _wrap_paragraph_by_display_width(paragraph: str, available: int) -> List[str]:
            """Wrap a single paragraph using display width instead of len().

            Uses _display_width() to correctly handle wide characters (CJK),
            East Asian Ambiguous characters (box-drawing), and zero-width chars.
            Falls back to textwrap for pure-ASCII text where len() == display width.
            """
            # Fast path: pure ASCII text where len() == display width
            if paragraph.isascii():
                return textwrap.wrap(paragraph, width=available, break_long_words=True, break_on_hyphens=False)

            # Display-width-aware wrapping for text with wide/ambiguous characters
            words = paragraph.split(' ')
            lines: List[str] = []
            current_parts: List[str] = []
            current_width = 0

            for word in words:
                word_width = _display_width(word)
                # Break words wider than available width
                if word_width > available:
                    # Flush current line
                    if current_parts:
                        lines.append(' '.join(current_parts))
                        current_parts = []
                        current_width = 0
                    # Break long word character by character
                    chunk = ''
                    chunk_width = 0
                    for char in word:
                        cw = _display_width(char)
                        if chunk_width + cw > available and chunk:
                            lines.append(chunk)
                            chunk = char
                            chunk_width = cw
                        else:
                            chunk += char
                            chunk_width += cw
                    if chunk:
                        current_parts = [chunk]
                        current_width = chunk_width
                    continue

                if current_parts:
                    # +1 for the space between words
                    if current_width + 1 + word_width <= available:
                        current_parts.append(word)
                        current_width += 1 + word_width
                    else:
                        lines.append(' '.join(current_parts))
                        current_parts = [word]
                        current_width = word_width
                else:
                    current_parts = [word]
                    current_width = word_width

            if current_parts:
                lines.append(' '.join(current_parts))
            return lines if lines else ['']

        def wrap_text(text: str, prefix_width: int = 0) -> List[str]:
            """Wrap text to console width, accounting for prefix.

            Handles multi-line text by splitting on newlines first,
            then wrapping each line individually.  Uses _display_width()
            for correct handling of wide/ambiguous-width characters.
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
                elif _display_width(paragraph) <= available:
                    result.append(paragraph)
                else:
                    wrapped = _wrap_paragraph_by_display_width(paragraph, available)
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
            # Get line index for search highlighting
            line_idx = line_idx_map.get(id(item))
            search_active = bool(self._search_query)

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
                    prefix_dw = _display_width(header_prefix)
                    dash_dw = _display_width("─") or 1
                    remaining = max(0, (wrap_width - prefix_dw) // dash_dw)
                    output.append(header_prefix, style=self._style("user_header", "bold green"))
                    output.append("─" * remaining, style=self._style("user_header_separator", "dim green"))
                    output.append("\n")
                    # Then render the text content with search highlighting
                    if search_active:
                        content = self._wrap_and_highlight_text(line.text, line_idx, wrap_width)
                        output.append_text(content)
                    else:
                        wrapped = wrap_text(line.text, 0)
                        for j, wrapped_line in enumerate(wrapped):
                            if j > 0:
                                output.append("\n")
                            output.append(wrapped_line)
                else:
                    # Non-turn-start - just render text with search highlighting
                    if search_active:
                        content = self._wrap_and_highlight_text(line.text, line_idx, wrap_width)
                        output.append_text(content)
                    else:
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
                    prefix_dw = _display_width(header_prefix)
                    dash_dw = _display_width("─") or 1
                    remaining = max(0, (wrap_width - prefix_dw) // dash_dw)
                    output.append(header_prefix, style=self._style("model_header", "bold cyan"))
                    output.append("─" * remaining, style=self._style("model_header_separator", "dim cyan"))
                    output.append("\n")
                    # Then render the text content (no prefix needed)
                    # Skip cache when search is active (highlights change dynamically)
                    if search_active and not has_ansi:
                        # Use highlighting-aware wrapping
                        content = self._wrap_and_highlight_text(line.text, line_idx, wrap_width)
                        output.append_text(content)
                    else:
                        # Use cache for expensive ANSI parsing
                        cached = self._get_cached_line_content(line, wrap_width)
                        if cached is not None:
                            output.append_text(cached)
                        elif line.text.startswith(PRERENDERED_LINE_PREFIX):
                            # Pre-rendered content (mermaid diagrams) — truncate per row, don't wrap
                            content = self._render_prerendered(line.text, wrap_width)
                            self._cache_line_content(line, content, wrap_width)
                            output.append_text(content)
                        elif has_ansi:
                            # Text contains ANSI codes from syntax highlighting - wrap to width
                            content = self._wrap_ansi_text(line.text, wrap_width)
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
                    # Skip cache when search is active (highlights change dynamically)
                    if search_active and not has_ansi:
                        # Use highlighting-aware wrapping
                        content = self._wrap_and_highlight_text(line.text, line_idx, wrap_width)
                        output.append_text(content)
                    else:
                        # Use cache for expensive ANSI parsing
                        cached = self._get_cached_line_content(line, wrap_width)
                        if cached is not None:
                            output.append_text(cached)
                        elif line.text.startswith(PRERENDERED_LINE_PREFIX):
                            # Pre-rendered content (mermaid diagrams) — truncate per row, don't wrap
                            content = self._render_prerendered(line.text, wrap_width)
                            self._cache_line_content(line, content, wrap_width)
                            output.append_text(content)
                        elif has_ansi:
                            # Text contains ANSI codes from syntax highlighting - wrap to width
                            content = self._wrap_ansi_text(line.text, wrap_width)
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
                # NOTE: Box-drawing chars have East Asian Ambiguous width, so we use
                # _display_width() throughout to handle terminals where they render as 2 columns.
                indent = "   "  # Left indent for the entire thinking block
                border = "│ "
                indent_dw = _display_width(indent)
                border_dw = _display_width(border)
                border_width = indent_dw + border_dw
                dash_dw = _display_width("─") or 1
                # Extra 1-char right margin compensates for italic font rendering in
                # terminals that slant glyphs rightward, clipping the last character.
                italic_margin = 1
                # Check if this is the first thinking line in a consecutive group
                is_first_thinking = (i == 0) or (items_to_show[i - 1].source != "thinking"
                                                  if isinstance(items_to_show[i - 1], OutputLine)
                                                  else True)
                if line.is_turn_start:
                    # First render Model header (thinking is part of model turn)
                    header_prefix = "── Model "
                    prefix_dw = _display_width(header_prefix)
                    remaining = max(0, (wrap_width - prefix_dw) // dash_dw)
                    output.append(header_prefix, style=self._style("model_header", "bold cyan"))
                    output.append("─" * remaining, style=self._style("model_header_separator", "dim cyan"))
                    output.append("\n")
                if is_first_thinking:
                    # Render thinking header (top border): ┌─ Internal thinking ───────┐
                    thinking_header = "┌─ Internal thinking "
                    box_display_width = wrap_width - indent_dw
                    header_dw = _display_width(thinking_header)
                    closing_dw = _display_width("┐")
                    remaining = max(0, (box_display_width - header_dw - closing_dw) // dash_dw)
                    output.append(indent)
                    output.append(thinking_header, style=self._style("thinking_header", "dim #D7AF5F"))
                    output.append("─" * remaining, style=self._style("thinking_header_separator", "dim #D7AF5F"))
                    output.append("┐", style=self._style("thinking_header", "dim #D7AF5F"))
                    output.append("\n")
                # Render thinking content with border (aligned with box)
                wrapped = wrap_text(line.text, border_width + italic_margin)
                for j, wrapped_line in enumerate(wrapped):
                    if j > 0:
                        output.append("\n")
                    output.append(indent)
                    output.append(border, style=self._style("thinking_border", "dim #D7AF5F"))
                    output.append(wrapped_line, style=self._style("thinking_content", "italic #D7AF87"))
                # Track that we're in thinking mode for footer rendering
                self._in_thinking = True
                # Check if this is the last thinking line in the visible items
                # to render footer (look ahead to next item)
                is_last_thinking = (i == len(items_to_show) - 1) or (
                    i + 1 < len(items_to_show) and getattr(items_to_show[i + 1], 'source', None) != "thinking"
                )
                if is_last_thinking:
                    # Render footer: └─────────────────────────────────────────┘
                    output.append("\n")
                    box_display_width = wrap_width - indent_dw
                    opening_dw = _display_width("└")
                    closing_dw = _display_width("┘")
                    remaining = max(0, (box_display_width - opening_dw - closing_dw) // dash_dw)
                    output.append(indent)
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
                    prefix_dw = _display_width(header_prefix)
                    dash_dw = _display_width("─") or 1
                    remaining = max(0, (wrap_width - prefix_dw) // dash_dw)
                    output.append(header_prefix, style=self._style("model_header", "bold cyan"))
                    output.append("─" * remaining, style=self._style("model_header_separator", "dim cyan"))
                    output.append("\n")
            frame = self.SPINNER_FRAMES[self._spinner_index]
            output.append(f"  {frame} ", style=self._style("spinner", "cyan"))
            output.append("thinking...", style=self._style("hint", "dim italic"))

        # Crop rendered content to exactly fit the viewport height.
        # This is the definitive safety net against ALL measurement inaccuracies
        # (word-wrap differences between textwrap and Rich, inter-item spacing edge
        # cases, spinner lines, tool tree height miscalculations, etc.).
        # Rich Panel clips from the BOTTOM when content exceeds height, which hides
        # the most recent output. By cropping from the TOP here, we guarantee the
        # most recent (bottom) content is always visible.
        #
        # Only crop when at the bottom (scroll_offset==0). When scrolled up, the
        # user wants to see the OLDER content at the top of the viewport; cropping
        # from the top would remove exactly what they scrolled up to see. In that
        # case, Panel's default bottom-clipping is correct (clips the newer edge
        # of the visible range, preserving the older content the user navigated to).
        if height and self._scroll_offset == 0 and isinstance(output, Text):
            plain = output.plain
            line_count = plain.count('\n') + 1 if plain else 0
            if line_count > height:
                trim_count = line_count - height
                pos = 0
                for _ in range(trim_count):
                    idx = plain.find('\n', pos)
                    if idx == -1:
                        break
                    pos = idx + 1
                if pos > 0:
                    output = output[pos:]

        # Add debug line numbers if enabled (checked at runtime for .env support)
        if os.environ.get('JAATO_DEBUG_LINE_NUMBERS', '').lower() in ('1', 'true', 'yes'):
            output = self._add_debug_line_numbers(output, start_display_line)

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
