"""Tests for the per-turn copy-to-clipboard button.

Covers:
  * ``format_turn_for_clipboard`` — markdown shape for User/Assistant
    role headers, ToolBlock rendering, system/error blockquotes,
    consecutive-same-role grouping.
  * Conversational-turn boundary detection — that a CopyButton lands
    in ``OutputBuffer._lines`` between turns and captures the right
    items in the right order.
  * Click-region scanning — that ``_compute_copy_button_regions`` finds
    every embedded marker in the rendered plain_text and recovers the
    correct button index.
"""

import os
import sys
from importlib.util import module_from_spec, spec_from_file_location

# Hoist the jaato-tui directory onto sys.path so submodule imports work
# without needing to install the package.
_TUI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TUI_DIR not in sys.path:
    sys.path.insert(0, _TUI_DIR)

# Load output_buffer via spec_from_file_location (matches the pattern
# used by test_output_buffer.py — avoids namespace collisions if the
# package gets installed in editable mode alongside).
_spec = spec_from_file_location(
    "output_buffer", os.path.join(_TUI_DIR, "output_buffer.py")
)
_ob = module_from_spec(_spec)
_spec.loader.exec_module(_ob)

OutputBuffer = _ob.OutputBuffer
OutputLine = _ob.OutputLine
ToolBlock = _ob.ToolBlock
CopyButton = _ob.CopyButton
ActiveToolCall = _ob.ActiveToolCall
format_turn_for_clipboard = _ob.format_turn_for_clipboard
COPY_BUTTON_RE = _ob.COPY_BUTTON_RE


# ---------------------------------------------------------------------------
# format_turn_for_clipboard
# ---------------------------------------------------------------------------


class TestFormatTurnForClipboard:

    def test_user_then_assistant_emits_role_headers(self):
        items = [
            OutputLine(source="user", text="hello", style="line", is_turn_start=True),
            OutputLine(source="model", text="hi back", style="line", is_turn_start=True),
        ]
        out = format_turn_for_clipboard(items)
        assert "### User" in out
        assert "hello" in out
        assert "### Assistant" in out
        assert "hi back" in out
        # User header must precede Assistant header.
        assert out.index("### User") < out.index("### Assistant")

    def test_consecutive_same_role_lines_grouped_under_one_header(self):
        """Multiple model lines in a row belong to one ### Assistant
        block — no extra header per line."""
        items = [
            OutputLine(source="model", text="line one", style="line", is_turn_start=True),
            OutputLine(source="model", text="line two", style="line", is_turn_start=False),
            OutputLine(source="model", text="line three", style="line", is_turn_start=False),
        ]
        out = format_turn_for_clipboard(items)
        assert out.count("### Assistant") == 1
        assert "line one\nline two\nline three" in out

    def test_role_change_flushes_and_starts_new_header(self):
        items = [
            OutputLine(source="model", text="thinking out loud", style="line", is_turn_start=True),
            OutputLine(source="user", text="wait, stop", style="line", is_turn_start=True),
            OutputLine(source="model", text="ok", style="line", is_turn_start=True),
        ]
        out = format_turn_for_clipboard(items)
        # Two ### Assistant headers (split by the user interjection).
        assert out.count("### Assistant") == 2
        assert out.count("### User") == 1
        assert out.index("### Assistant") < out.index("### User") < out.rindex("### Assistant")

    def test_parent_treated_as_user(self):
        items = [
            OutputLine(source="parent", text="from above", style="line", is_turn_start=True),
        ]
        out = format_turn_for_clipboard(items)
        assert "### User" in out
        assert "from above" in out

    def test_thinking_treated_as_assistant(self):
        items = [
            OutputLine(source="thinking", text="hmm...", style="line", is_turn_start=True),
        ]
        out = format_turn_for_clipboard(items)
        assert "### Assistant" in out

    def test_tool_block_renders_full_args_and_output(self):
        tool = ActiveToolCall(
            name="writeNewFile",
            args_summary="path=fb.py",
            args_full='{"path": "fb.py", "content": "..."}',
            completed=True,
            success=True,
            output_lines=["wrote 12 lines", "ok"],
        )
        items = [
            OutputLine(source="model", text="writing file", style="line", is_turn_start=True),
            ToolBlock(tools=[tool]),
        ]
        out = format_turn_for_clipboard(items)
        assert "**Tool: writeNewFile**" in out
        assert '"path": "fb.py"' in out  # full args
        assert "wrote 12 lines" in out  # output
        assert "_Output:_" in out

    def test_tool_error_message_included(self):
        tool = ActiveToolCall(
            name="cli",
            args_summary="rm bad",
            completed=True,
            success=False,
            error_message="permission denied",
        )
        items = [ToolBlock(tools=[tool])]
        out = format_turn_for_clipboard(items)
        assert "**Tool: cli**" in out
        assert "_Error:_ permission denied" in out

    def test_system_lines_render_as_blockquote(self):
        items = [
            OutputLine(source="model", text="ok", style="line", is_turn_start=True),
            OutputLine(source="system", text="permission required", style="line"),
            OutputLine(source="model", text="continuing", style="line", is_turn_start=True),
        ]
        out = format_turn_for_clipboard(items)
        assert "> _System:_ permission required" in out

    def test_strips_ansi_escapes(self):
        items = [
            OutputLine(
                source="model",
                text="\x1b[31mred text\x1b[0m and normal",
                style="line",
                is_turn_start=True,
            ),
        ]
        out = format_turn_for_clipboard(items)
        assert "\x1b[" not in out
        assert "red text and normal" in out

    def test_strips_leaked_button_labels(self):
        """If a previous render somehow leaked the [copy N] label into
        a captured OutputLine, format defensively strips it so the copy
        doesn't reveal the affordance text inside the prose copy."""
        leaked_text = "agent reply [copy 1] tail"
        items = [OutputLine(source="model", text=leaked_text, style="line", is_turn_start=True)]
        out = format_turn_for_clipboard(items)
        assert "[copy 1]" not in out
        assert "agent reply" in out

    def test_empty_items_returns_empty_string(self):
        assert format_turn_for_clipboard([]) == ""


# ---------------------------------------------------------------------------
# Conversational-turn boundary detection in OutputBuffer
# ---------------------------------------------------------------------------


class TestTurnBoundaryDetection:

    def _buffer(self):
        b = OutputBuffer(max_lines=200, agent_type="main")
        b.set_width(80)
        return b

    def test_initial_state_has_no_buttons(self):
        b = self._buffer()
        assert b._copy_buttons == []
        assert b._current_turn_items == []

    def test_first_turn_does_not_emit_button_until_next_user_input(self):
        """User → Model → ... no second user input yet → no button.
        Mirrors the streaming-not-flushed reality: the most recent
        in-progress turn doesn't get a button until the next turn
        starts (acceptable V1 behavior)."""
        b = self._buffer()
        b.append("user", "hello", mode="write")
        b.append("model", "hi", mode="write")
        # Nothing flushed the model line yet (no follow-up append).
        assert b._copy_buttons == []

    def test_second_user_input_closes_first_turn(self):
        """User → Model → User (+ flush) → CopyButton appears."""
        b = self._buffer()
        b.append("user", "hello", mode="write")
        b.append("model", "hi", mode="write")
        b.append("user", "second", mode="write")
        b.append("model", "again", mode="write")  # forces the second user to flush

        assert len(b._copy_buttons) == 1
        button = b._copy_buttons[0]
        # Button captures: user "hello" + model "hi"
        sources = [it.source for it in button.turn_items if isinstance(it, OutputLine)]
        assert sources == ["user", "model"]

    def test_copybutton_lands_in_lines_between_turns(self):
        """Buffer order must be: turn-1-items, CopyButton, turn-2-start.
        Otherwise the button would render in the wrong scrollback
        position."""
        b = self._buffer()
        b.append("user", "hello", mode="write")
        b.append("model", "hi", mode="write")
        b.append("user", "second", mode="write")
        b.append("model", "again", mode="write")

        items = list(b._lines)
        copy_idx = next(i for i, it in enumerate(items) if isinstance(it, CopyButton))
        # Items before the button belong to turn 1; items after start turn 2.
        before = items[:copy_idx]
        after = items[copy_idx + 1:]
        assert any(isinstance(it, OutputLine) and it.source == "user" and it.text == "hello" for it in before)
        assert any(isinstance(it, OutputLine) and it.source == "model" and it.text == "hi" for it in before)
        assert any(isinstance(it, OutputLine) and it.source == "user" and it.text == "second" for it in after)

    def test_get_copy_button_returns_registered_or_none(self):
        b = self._buffer()
        b.append("user", "a", mode="write")
        b.append("model", "b", mode="write")
        b.append("user", "c", mode="write")
        b.append("model", "d", mode="write")
        assert b.get_copy_button(0) is b._copy_buttons[0]
        assert b.get_copy_button(99) is None
        assert b.get_copy_button(-1) is None

    def test_clear_resets_buttons_and_accumulator(self):
        b = self._buffer()
        b.append("user", "hello", mode="write")
        b.append("model", "hi", mode="write")
        b.append("user", "second", mode="write")
        b.append("model", "x", mode="write")
        assert len(b._copy_buttons) == 1
        b.clear()
        assert b._copy_buttons == []
        assert b._current_turn_items == []


# ---------------------------------------------------------------------------
# Click-region scanning (the helper used by ScrollableBufferControl)
# ---------------------------------------------------------------------------


class TestComputeCopyButtonRegions:
    """Reimplements the helper inline (avoiding pulling in pt_display
    which has prompt_toolkit dependencies) — same logic, narrower
    surface for testing.

    The button label is a literal ``[copy N]`` (1-based ``N``).  The
    helper recovers the 0-based registry index for ``get_copy_button``.
    """

    @staticmethod
    def _compute(plain_text):
        regions = []
        for match in COPY_BUTTON_RE.finditer(plain_text):
            button_index = int(match.group(1)) - 1
            regions.append((match.start(), match.end(), button_index))
        return regions

    def _emit_button(self, index):
        return f"[copy {index + 1}]"

    def test_empty_text_no_regions(self):
        assert self._compute("") == []
        assert self._compute("just regular text\nwith newlines") == []

    def test_single_button_recovered_with_correct_index(self):
        text = "some output\n" + self._emit_button(0) + "\n"
        regions = self._compute(text)
        assert len(regions) == 1
        start, end, idx = regions[0]
        assert idx == 0
        assert text[start:end] == "[copy 1]"

    def test_multiple_buttons_distinct_indices(self):
        text = (
            "turn one output\n" + self._emit_button(0)
            + "\nturn two output\n" + self._emit_button(1)
            + "\nturn three output\n" + self._emit_button(2)
            + "\n"
        )
        regions = self._compute(text)
        assert [idx for _, _, idx in regions] == [0, 1, 2]

    def test_two_digit_index(self):
        text = self._emit_button(42)
        regions = self._compute(text)
        assert regions == [(0, len(text), 42)]


class TestEndToEndRender:
    """Lock in that the affordance survives the full render pipeline —
    OutputBuffer.render_panel → Rich → ANSI → plain_text.  This is the
    test that would have caught the regression where Rich's word-wrap
    pushed the (formerly) invisible marker tail onto a separate line
    from the visible label."""

    def _render_plain(self, buf, width=80, height=20):
        # Local import — pt_display brings in prompt_toolkit.
        from pt_display import RichRenderer, ansi_to_plain_and_fragments
        panel = buf.render_panel(height=height, width=width)
        ansi = RichRenderer(width).render(panel)
        plain, _ = ansi_to_plain_and_fragments(ansi)
        return plain

    def test_button_label_survives_render_unsplit(self):
        """The ``[copy N]`` label must appear intact in the rendered
        plain_text — i.e. the regex finds at least one match.  When
        the marker tail was used, Rich would word-wrap the line at the
        panel's trailing edge and split the marker onto a separate
        line, breaking the regex."""
        buf = OutputBuffer(max_lines=200, agent_type="main")
        buf.set_width(80)
        buf.append("user", "first", mode="write")
        buf.append("model", "reply one", mode="write")
        buf.append("user", "second", mode="write")
        buf.append("model", "reply two", mode="write")  # forces flush of second user
        plain = self._render_plain(buf, width=80)

        regions = []
        for m in COPY_BUTTON_RE.finditer(plain):
            regions.append((m.start(), m.end(), int(m.group(1)) - 1))
        assert len(regions) == 1, f"expected 1 button in plain_text, got {len(regions)}"
        assert regions[0][2] == 0  # first button is registry index 0


class TestStopSpinnerClosesTurn:
    """stop_spinner is the agent-idle hook.  Closing the turn from
    there is what makes the [copy N] button appear immediately when
    the agent finishes, instead of waiting for the user's next
    message to trigger the close from _add_line."""

    def test_stop_spinner_emits_button_for_completed_turn(self):
        buf = OutputBuffer(max_lines=200, agent_type="main")
        buf.set_width(80)
        buf.append("user", "hello", mode="write")
        buf.append("model", "hi", mode="write")
        # Simulate the agent finishing: bridge calls stop_spinner.
        buf.stop_spinner()
        # Button should now exist without needing a follow-up user msg.
        assert len(buf._copy_buttons) == 1

    def test_stop_spinner_on_empty_buffer_is_noop(self):
        """Calling stop_spinner before the agent has produced anything
        must not emit a button (nothing to copy)."""
        buf = OutputBuffer(max_lines=200, agent_type="main")
        buf.set_width(80)
        buf.stop_spinner()
        assert buf._copy_buttons == []

    def test_stop_spinner_idempotent(self):
        """Calling stop_spinner twice in a row (e.g. spurious idle
        events) must not emit a duplicate button."""
        buf = OutputBuffer(max_lines=200, agent_type="main")
        buf.set_width(80)
        buf.append("user", "q", mode="write")
        buf.append("model", "a", mode="write")
        buf.stop_spinner()
        buf.stop_spinner()
        assert len(buf._copy_buttons) == 1
