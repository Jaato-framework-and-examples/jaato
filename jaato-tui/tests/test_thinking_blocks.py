"""Reasoning blocks collapse by default and expand on a toggle (#755).

Reasoning (``source == "thinking"``) already rendered in its own bordered
``Internal thinking`` box; what it was not is *collapsed* — the longest
thing in a turn was the one block that could not be folded away, while the
tool tree beside it collapsed behind Ctrl+T.  These tests pin down the
collapsed rendering, the height the viewport is charged for it (which must
agree with what is drawn, or scrolling drifts), the streaming indicator,
the group split a ToolBlock introduces, and the keybinding registration.
"""

import os
import sys
from importlib.util import module_from_spec, spec_from_file_location

rich_client_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if rich_client_dir not in sys.path:
    sys.path.insert(0, rich_client_dir)

spec = spec_from_file_location("output_buffer", os.path.join(rich_client_dir, "output_buffer.py"))
output_buffer_module = module_from_spec(spec)
spec.loader.exec_module(output_buffer_module)

OutputBuffer = output_buffer_module.OutputBuffer
OutputLine = output_buffer_module.OutputLine
ToolBlock = output_buffer_module.ToolBlock

from keybindings import DEFAULT_KEYBINDINGS, KeybindingConfig, generate_example_config  # noqa: E402

REASONING = [
    "Let me look at the failing test first.",
    "The assertion compares a list to a tuple.",
    "So the fix is to normalise both sides before comparing.",
    "I will also add a regression test.",
]


def _plain(buffer: OutputBuffer, height=None) -> str:
    return buffer.render(height=height, width=80).plain


def _buffer_with_reasoning(**kwargs) -> OutputBuffer:
    buffer = OutputBuffer(**kwargs)
    buffer.set_width(80)
    buffer.append("user", "why does it fail?", "write")
    buffer.append("thinking", "\n".join(REASONING), "write")
    buffer.append("model", "The comparison mixes a list and a tuple.", "write")
    buffer._flush_current_block()
    return buffer


class TestDefaultState:
    def test_reasoning_collapsed_by_default(self):
        assert OutputBuffer().thinking_expanded is False

    def test_constructor_can_start_expanded(self):
        assert OutputBuffer(thinking_expanded=True).thinking_expanded is True

    def test_toggle_round_trips(self):
        buffer = OutputBuffer()
        assert buffer.toggle_thinking_expanded() is True
        assert buffer.thinking_expanded is True
        assert buffer.toggle_thinking_expanded() is False

    def test_tools_toggle_is_independent(self):
        """Ctrl+T and the reasoning toggle are siblings, not one switch."""
        buffer = OutputBuffer()
        buffer.toggle_tools_expanded()
        assert buffer.tools_expanded is True
        assert buffer.thinking_expanded is False

    def test_has_thinking(self):
        buffer = OutputBuffer()
        buffer.set_width(80)
        assert buffer.has_thinking() is False
        buffer.append("model", "just an answer", "write")
        buffer._flush_current_block()
        assert buffer.has_thinking() is False
        buffer.append("thinking", "hmm", "write")
        assert buffer.has_thinking() is True, "an in-flight reasoning block counts"
        buffer._flush_current_block()
        assert buffer.has_thinking() is True


class TestCollapsedRendering:
    def test_collapsed_renders_one_summary_line_and_no_reasoning_text(self):
        plain = _plain(_buffer_with_reasoning())
        assert "▸ Internal thinking (4 lines, 33 words)" in plain
        assert "to expand" in plain
        for sentence in REASONING:
            assert sentence not in plain, "collapsed reasoning must not leak its text"
        assert "┌─ Internal thinking" not in plain, "no box when collapsed"
        assert "The comparison mixes a list and a tuple." in plain, "the answer is untouched"

    def test_summary_precedes_the_answer_and_follows_model_header(self):
        plain = _plain(_buffer_with_reasoning())
        header = plain.index("── Model ")
        summary = plain.index("▸ Internal thinking")
        answer = plain.index("The comparison mixes")
        assert header < summary < answer

    def test_summary_carries_the_configured_key(self):
        buffer = _buffer_with_reasoning()
        buffer.set_keybinding_config(KeybindingConfig(toggle_thinking="c-x"))
        assert "Ctrl+X to expand" in _plain(buffer)

    def test_default_hint_without_keybinding_config(self):
        assert "Ctrl+R to expand" in _plain(_buffer_with_reasoning())

    def test_expanded_renders_the_box_with_every_line(self):
        buffer = _buffer_with_reasoning()
        buffer.toggle_thinking_expanded()
        plain = _plain(buffer)
        assert "┌─ Internal thinking" in plain
        assert "└" in plain
        for sentence in REASONING:
            assert sentence in plain
        assert "▸ Internal thinking" not in plain

    def test_toggle_back_collapses_again(self):
        buffer = _buffer_with_reasoning()
        buffer.toggle_thinking_expanded()
        buffer.toggle_thinking_expanded()
        assert "▸ Internal thinking (4 lines" in _plain(buffer)

    def test_reasoning_that_is_not_a_turn_start_has_no_model_header(self):
        """Reasoning after model prose within the same turn: summary only."""
        buffer = OutputBuffer()
        buffer.set_width(80)
        buffer.append("model", "First I will check.", "write")
        buffer.append("thinking", "now reasoning", "write")
        buffer._flush_current_block()
        plain = _plain(buffer)
        assert plain.count("── Model ") == 1
        assert "▸ Internal thinking (1 line, 2 words)" in plain

    def test_hint_dropped_when_too_narrow(self):
        buffer = _buffer_with_reasoning()
        buffer.set_width(40)
        plain = buffer.render(width=40).plain
        assert "▸ Internal thinking (4 lines, 33 words)" in plain
        assert "to expand" not in plain

    def test_hidden_lines_add_no_blank_lines(self):
        """Members render nothing — not even their inter-item newline."""
        plain = _plain(_buffer_with_reasoning())
        summary_line = next(l for l in plain.split("\n") if "▸ Internal thinking" in l)
        idx = plain.split("\n").index(summary_line)
        assert plain.split("\n")[idx + 1].strip() == "The comparison mixes a list and a tuple."


class TestHeightAccounting:
    """What the viewport is charged must equal what is drawn."""

    def test_collapsed_group_charged_summary_only(self):
        buffer = _buffer_with_reasoning()
        thinking = [l for l in buffer._lines if isinstance(l, OutputLine) and l.source == "thinking"]
        assert len(thinking) == 4
        heights = buffer._display_heights(buffer._lines)
        by_id = {id(item): h for item, h in zip(buffer._lines, heights)}
        # head starts the model turn: blank + Model header + summary
        assert by_id[id(thinking[0])] == 3
        assert all(by_id[id(t)] == 0 for t in thinking[1:])

    def test_non_turn_start_head_charged_one_line(self):
        buffer = OutputBuffer()
        buffer.set_width(80)
        buffer.append("model", "prose", "write")
        buffer.append("thinking", "a\nb", "write")
        buffer._flush_current_block()
        heights = buffer._display_heights(buffer._lines)
        assert heights[-2:] == [1, 0]

    def test_expanded_heights_are_the_per_line_measurement(self):
        buffer = _buffer_with_reasoning(thinking_expanded=True)
        heights = buffer._display_heights(buffer._lines)
        assert heights == [buffer._get_item_display_lines(i) for i in buffer._lines]

    def test_charged_height_matches_rendered_height(self):
        """Render unbounded and count lines; the collapsed charge must agree.

        (Expanded reasoning keeps the pre-existing measurement, which
        deliberately over-counts — header and footer are charged on every
        line "to ensure the block fits on screen" — so only the collapsed
        path is held to exact agreement.)
        """
        buffer = _buffer_with_reasoning()
        rendered_lines = _plain(buffer).count("\n") + 1
        # _display_heights counts the blank line before EVERY turn start,
        # while render draws it only between items (not before the first
        # item) — the same over-count the pre-existing measurement makes.
        charged = sum(buffer._display_heights(buffer._lines))
        first = buffer._lines[0]
        leading_blank = 1 if isinstance(first, OutputLine) and first.is_turn_start else 0
        assert charged - leading_blank == rendered_lines

    def test_scroll_range_shrinks_when_collapsed(self):
        collapsed = _buffer_with_reasoning()
        expanded = _buffer_with_reasoning(thinking_expanded=True)
        assert sum(collapsed._display_heights(collapsed._lines)) < sum(
            expanded._display_heights(expanded._lines)
        )

    def test_windowed_render_with_small_height_shows_summary(self):
        """The visible-window walk uses the collapsed heights."""
        buffer = _buffer_with_reasoning()
        plain = _plain(buffer, height=4)
        assert "▸ Internal thinking" in plain
        assert "The comparison mixes" in plain


class TestStreaming:
    def test_in_flight_reasoning_shows_streaming_indicator(self):
        buffer = OutputBuffer()
        buffer.set_width(80)
        buffer.append("thinking", "Let me think", "write")
        buffer.append("thinking", " about this.\nSecond line", "append")
        plain = _plain(buffer)
        assert "▸ Internal thinking (2 lines, streaming…)" in plain
        assert "Let me think" not in plain

    def test_deltas_join_into_one_block(self):
        """write-then-append deltas are one group, not one line per token."""
        buffer = OutputBuffer()
        buffer.set_width(80)
        for i, delta in enumerate(["), capture", " script (tcp", "dump wrappers", "), a", " checklist"]):
            buffer.append("thinking", delta, "write" if i == 0 else "append")
        buffer.append("model", "answer", "write")
        buffer._flush_current_block()
        thinking = [l for l in buffer._lines if isinstance(l, OutputLine) and l.source == "thinking"]
        assert [l.text for l in thinking] == ["), capture script (tcpdump wrappers), a checklist"]
        assert "▸ Internal thinking (1 line, 7 words)" in _plain(buffer)

    def test_word_count_once_the_block_is_flushed(self):
        buffer = OutputBuffer()
        buffer.set_width(80)
        buffer.append("thinking", "one two three", "write")
        buffer.append("model", "answer", "write")
        buffer._flush_current_block()
        assert "(1 line, 3 words)" in _plain(buffer)


class TestGrouping:
    def test_tool_block_splits_reasoning_into_two_groups(self):
        buffer = OutputBuffer()
        buffer.set_width(80)
        buffer.append("thinking", "before the tool", "write")
        buffer.add_active_tool("readFile", {"path": "x.py"}, call_id="c1")
        buffer.mark_tool_completed("readFile", success=True, call_id="c1")
        buffer.append("thinking", "after the tool", "write")
        buffer.append("model", "done", "write")
        buffer._flush_current_block()
        assert any(isinstance(i, ToolBlock) for i in buffer._lines)
        groups = buffer._thinking_groups(buffer._lines)
        assert len(groups) == 2
        plain = _plain(buffer)
        assert plain.count("▸ Internal thinking (1 line") == 2

    def test_replayed_thought_parts_are_one_visual_group(self):
        """A reconnect replays each thought part as its own write; adjacent
        reasoning lines still fold into one summary."""
        buffer = OutputBuffer()
        buffer.set_width(80)
        buffer.append("thinking", "first part", "write")
        buffer.append("thinking", "second part", "write")
        buffer.append("model", "answer", "write")
        buffer._flush_current_block()
        assert len(buffer._thinking_groups(buffer._lines)) == 1
        assert "▸ Internal thinking (2 lines, 4 words)" in _plain(buffer)

    def test_groups_keyed_by_head_hold_every_member(self):
        items = [
            OutputLine("model", "m", "line", 1),
            OutputLine("thinking", "a", "line", 1),
            OutputLine("thinking", "b", "line", 1),
            OutputLine("model", "m2", "line", 1),
            OutputLine("thinking", "c", "line", 1),
        ]
        groups = OutputBuffer._thinking_groups(items)
        assert [len(v) for v in groups.values()] == [2, 1]
        assert id(items[1]) in groups and id(items[2]) not in groups


class TestKeybinding:
    def test_default_binding(self):
        assert DEFAULT_KEYBINDINGS["toggle_thinking"] == "c-r"
        assert KeybindingConfig().toggle_thinking == "c-r"

    def test_does_not_collide_with_another_default(self):
        others = {k: v for k, v in DEFAULT_KEYBINDINGS.items() if k != "toggle_thinking"}
        assert "c-r" not in others.values()

    def test_round_trips_through_dict_and_example(self):
        assert KeybindingConfig().to_dict()["toggle_thinking"] == "c-r"
        assert KeybindingConfig.from_dict({"toggle_thinking": "c-x"}).toggle_thinking == "c-x"
        assert '"toggle_thinking": "c-r"' in generate_example_config()

    def test_key_hint_formatting(self):
        buffer = OutputBuffer()
        assert buffer._format_key_hint("toggle_thinking") == "Ctrl+R"
        buffer.set_keybinding_config(KeybindingConfig(toggle_thinking="c-x"))
        assert buffer._format_key_hint("toggle_thinking") == "Ctrl+X"
