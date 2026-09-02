"""A finished tool block closes when the model speaks.

The model narrates between tool calls.  Each narration should end the block
the previous calls were rendered into, so the next call opens its own — one
block per run of consecutive tool calls, prose in between.

What happened instead: every call after the first landed in the SAME block,
however much prose separated them.  The finalize was gated on
``mode == "write"``, and only the FIRST chunk of a model message is a write.
A model that keeps narrating inside one message emits every later chunk as
an ``append``, so after the second tool call the trace reads:

    ToolCallStart/End -> model write   (block closes)
    ToolCallStart/End -> model append  (block stays open)
    ToolCallStart                      (joins the open block)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from output_buffer import OutputBuffer  # noqa: E402


def _buffer_with_finished_tool():
    b = OutputBuffer()
    b.add_active_tool("tool_a", {}, call_id="c1")
    b.mark_tool_completed("tool_a", success=True, call_id="c1")
    return b


class TestProseClosesAFinishedBlock:

    def test_appended_prose_closes_the_block(self):
        """The regression: only 'write' used to close it."""
        b = _buffer_with_finished_tool()
        assert b._active_tools, "precondition: the block is open"
        b.append("model", "Now I will try something else.", "append")
        assert b._active_tools == [], "prose did not close the finished block"

    def test_written_prose_closes_the_block_too(self):
        b = _buffer_with_finished_tool()
        b.append("model", "Now I will try something else.", "write")
        assert b._active_tools == []

    def test_a_later_tool_call_opens_a_new_block(self):
        b = _buffer_with_finished_tool()
        b.append("model", "narration", "append")
        b.add_active_tool("tool_b", {}, call_id="c2")
        names = [t.name for t in b._active_tools]
        assert names == ["tool_b"], "the new call joined the previous block"


class TestBlocksThatMustStayOpen:
    """Closing early would render the tree before its own tools finish."""

    def test_the_block_stays_whole_until_every_tool_finishes(self):
        """The mixed case, which is the only one that can regress.

        With nothing completed there is no block to close and finalize
        returns early, so a lone running tool proves nothing.  With one of
        each, ``all_completed`` keeps the WHOLE block open — the finished
        tool is not sealed off on its own, because a tree split mid-run
        would render half a block before the rest of its calls arrive.
        """
        b = OutputBuffer()
        b.add_active_tool("done", {}, call_id="c1")
        b.add_active_tool("running", {}, call_id="c2")
        b.mark_tool_completed("done", success=True, call_id="c1")
        b.append("model", "narration while the second still runs", "append")
        assert [t.name for t in b._active_tools] == ["done", "running"]

    def test_a_pending_permission_keeps_the_block_open(self):
        b = _buffer_with_finished_tool()
        b._active_tools[0].permission_state = "pending"
        b.append("model", "narration", "append")
        assert len(b._active_tools) == 1

    def test_a_pending_clarification_keeps_the_block_open(self):
        b = _buffer_with_finished_tool()
        b._active_tools[0].clarification_state = "pending"
        b.append("model", "narration", "append")
        assert len(b._active_tools) == 1

    def test_non_model_output_does_not_close_the_block(self):
        """Only the model speaking ends a run of tool calls."""
        b = _buffer_with_finished_tool()
        b.append("user", "unrelated", "append")
        assert len(b._active_tools) == 1
