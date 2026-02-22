"""Tests for OutputBuffer tool tree positioning.

These tests verify that tool trees maintain their correct chronological position
relative to model output. The bug being tested: when tool trees are finalized
(collapsed), they should stay AFTER the model text that preceded them, not
move above the model header.
"""

import pytest
from unittest.mock import MagicMock

# Import the output buffer module - need to handle the path carefully
import sys
import os

# Add the jaato-tui directory to the path
rich_client_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if rich_client_dir not in sys.path:
    sys.path.insert(0, rich_client_dir)

# Now import directly from the module file
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("output_buffer", os.path.join(rich_client_dir, "output_buffer.py"))
output_buffer_module = module_from_spec(spec)
spec.loader.exec_module(output_buffer_module)

OutputBuffer = output_buffer_module.OutputBuffer
OutputLine = output_buffer_module.OutputLine
ToolBlock = output_buffer_module.ToolBlock
ActiveToolCall = output_buffer_module.ActiveToolCall


class TestToolTreePositioning:
    """Tests for tool tree position in output flow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.buffer = OutputBuffer()
        # Set a reasonable width for tests
        self.buffer.set_width(80)

    def test_tool_block_after_model_text_on_new_model_write(self):
        """Tool block should be positioned AFTER model text when new model text arrives.

        Scenario:
        1. Model streams "I will help you..."
        2. Tool executes and completes
        3. Model continues with more text

        Expected order in _lines:
        1. OutputLine(model, "I will help you...", is_turn_start=True)
        2. ToolBlock containing the completed tool
        3. (new model text is in _current_block, not yet in _lines)
        """
        # Step 1: Start model output
        self.buffer.append("model", "I will help you with the code diff preview.", "write")

        # Step 2: Add a tool and mark it completed
        self.buffer.add_active_tool("readFile", {"path": "test.py"}, call_id="call_1")
        self.buffer.mark_tool_completed("readFile", success=True, duration_seconds=0.1, call_id="call_1")

        # Step 3: New model text arrives - this should trigger finalization
        self.buffer.append("model", "Now let me analyze the results.", "write")

        # Verify the order in _lines
        assert len(self.buffer._lines) >= 2, f"Expected at least 2 items in _lines, got {len(self.buffer._lines)}"

        # Find the model text and tool block
        model_lines = [item for item in self.buffer._lines if isinstance(item, OutputLine) and item.source == "model"]
        tool_blocks = [item for item in self.buffer._lines if isinstance(item, ToolBlock)]

        assert len(model_lines) >= 1, "Expected at least 1 model line"
        assert len(tool_blocks) == 1, f"Expected 1 tool block, got {len(tool_blocks)}"

        # Find indices
        first_model_idx = next(i for i, item in enumerate(self.buffer._lines)
                               if isinstance(item, OutputLine) and item.source == "model")
        tool_block_idx = next(i for i, item in enumerate(self.buffer._lines)
                              if isinstance(item, ToolBlock))

        # Tool block should come AFTER the first model text
        assert tool_block_idx > first_model_idx, (
            f"Tool block (idx={tool_block_idx}) should come AFTER first model text (idx={first_model_idx}). "
            f"_lines order: {[(type(item).__name__, getattr(item, 'source', 'N/A')) for item in self.buffer._lines]}"
        )

    def test_tool_block_has_correct_model_header_position(self):
        """Model header should appear BEFORE tool block, not after.

        When model text "I will help you..." has is_turn_start=True, the header
        should be rendered first, then the text, then the tool block.
        """
        # Start fresh - user message first to establish turn context
        self.buffer.append("user", "Help me test the diff preview", "write")

        # Model responds
        self.buffer.append("model", "I will help you test the code diff preview.", "write")

        # Tool executes
        self.buffer.add_active_tool("glob_files", {"pattern": "**/*diff*"}, call_id="call_1")
        self.buffer.mark_tool_completed("glob_files", success=True, duration_seconds=0.2, call_id="call_1")

        # Model continues
        self.buffer.append("model", "Found the relevant files.", "write")

        # Check model line has is_turn_start=True
        model_lines = [item for item in self.buffer._lines
                       if isinstance(item, OutputLine) and item.source == "model"]

        assert len(model_lines) >= 1, "Should have at least one model line"

        # First model line should have is_turn_start=True (it's a new turn after user)
        first_model = model_lines[0]
        assert first_model.is_turn_start, "First model line should have is_turn_start=True"

    def test_multiple_tool_batches_maintain_order(self):
        """Multiple rounds of model text + tools should maintain chronological order.

        Scenario described by user:
        1. Model text "I will help you..."
        2. First tools execute
        3. Model continues
        4. Second tools execute
        5. Model continues

        Each tool batch should appear AFTER the model text that preceded it.
        """
        # Round 1
        self.buffer.append("model", "First, let me search for files.", "write")
        self.buffer.add_active_tool("glob_files", {"pattern": "*.py"}, call_id="call_1")
        self.buffer.mark_tool_completed("glob_files", success=True, duration_seconds=0.1, call_id="call_1")

        # Round 2
        self.buffer.append("model", "Found files. Now reading them.", "write")
        self.buffer.add_active_tool("readFile", {"path": "a.py"}, call_id="call_2")
        self.buffer.add_active_tool("readFile", {"path": "b.py"}, call_id="call_3")
        self.buffer.mark_tool_completed("readFile", success=True, duration_seconds=0.1, call_id="call_2")
        self.buffer.mark_tool_completed("readFile", success=True, duration_seconds=0.1, call_id="call_3")

        # Round 3
        self.buffer.append("model", "Analysis complete.", "write")

        # Verify we have 2 tool blocks
        tool_blocks = [item for item in self.buffer._lines if isinstance(item, ToolBlock)]
        assert len(tool_blocks) == 2, f"Expected 2 tool blocks, got {len(tool_blocks)}"

        # Collect indices for verification
        item_types = []
        for i, item in enumerate(self.buffer._lines):
            if isinstance(item, ToolBlock):
                item_types.append((i, "ToolBlock", len(item.tools)))
            elif isinstance(item, OutputLine):
                if item.source == "model" and item.text.strip():
                    item_types.append((i, "ModelText", item.text[:30]))
                elif item.source == "system" and not item.text.strip():
                    item_types.append((i, "Separator", ""))

        # Verify model text indices come before their respective tool blocks
        # Expected order: model1, toolblock1, model2, toolblock2, (model3 in current_block)
        model_indices = [i for i, t, _ in item_types if t == "ModelText"]
        block_indices = [i for i, t, _ in item_types if t == "ToolBlock"]

        assert len(model_indices) >= 2, f"Expected at least 2 model texts, found {model_indices}"
        assert len(block_indices) == 2, f"Expected 2 tool blocks, found {block_indices}"

        # First model text should come before first tool block
        assert model_indices[0] < block_indices[0], (
            f"First model text (idx={model_indices[0]}) should come before first tool block (idx={block_indices[0]})"
        )

        # First tool block should come before second model text
        assert block_indices[0] < model_indices[1], (
            f"First tool block (idx={block_indices[0]}) should come before second model text (idx={model_indices[1]})"
        )

        # Second model text should come before second tool block
        assert model_indices[1] < block_indices[1], (
            f"Second model text (idx={model_indices[1]}) should come before second tool block (idx={block_indices[1]})"
        )

    def test_active_tools_not_in_lines_until_finalized(self):
        """Active (running) tools should not be in _lines, only in _active_tools."""
        self.buffer.append("model", "Running tools now.", "write")
        self.buffer.add_active_tool("readFile", {"path": "test.py"}, call_id="call_1")

        # Tool is active but not completed
        tool_blocks = [item for item in self.buffer._lines if isinstance(item, ToolBlock)]
        assert len(tool_blocks) == 0, "No tool blocks should exist while tools are still active"
        assert len(self.buffer._active_tools) == 1, "Active tool should be in _active_tools"

    def test_finalize_tool_tree_called_on_model_write(self):
        """finalize_tool_tree should be called when new model text arrives."""
        self.buffer.append("model", "Initial text.", "write")
        self.buffer.add_active_tool("readFile", {"path": "test.py"}, call_id="call_1")
        self.buffer.mark_tool_completed("readFile", success=True, call_id="call_1")

        # Before new model text - active_tools should still have the completed tool
        assert len(self.buffer._active_tools) == 1, "Tool should still be in active_tools"

        # New model text triggers finalization
        self.buffer.append("model", "Continuation text.", "write")

        # After finalization - active_tools should be cleared
        assert len(self.buffer._active_tools) == 0, "Active tools should be cleared after finalization"

        # And there should be a tool block in _lines
        tool_blocks = [item for item in self.buffer._lines if isinstance(item, ToolBlock)]
        assert len(tool_blocks) == 1, "Tool block should exist in _lines after finalization"

    def test_pending_permission_blocks_finalization(self):
        """Tools with pending permission should not be finalized."""
        self.buffer.append("model", "Requesting permission.", "write")
        self.buffer.add_active_tool("writeFile", {"path": "test.py"}, call_id="call_1")

        # Set permission to pending
        for tool in self.buffer._active_tools:
            if tool.call_id == "call_1":
                tool.permission_state = "pending"

        # Try to trigger finalization with new model text
        self.buffer.append("model", "Waiting for permission.", "write")

        # Tool should NOT be finalized because permission is pending
        tool_blocks = [item for item in self.buffer._lines if isinstance(item, ToolBlock)]
        assert len(tool_blocks) == 0, "No tool block should be created while permission is pending"
        assert len(self.buffer._active_tools) == 1, "Tool should remain in active_tools"

    def test_render_order_matches_lines_order(self):
        """The rendered output should respect the order of items in _lines."""
        # Setup a scenario with model text and tool block
        self.buffer.append("model", "Hello world", "write")
        self.buffer.add_active_tool("testTool", {"arg": "value"}, call_id="call_1")
        self.buffer.mark_tool_completed("testTool", success=True, call_id="call_1")
        self.buffer.append("model", "Goodbye world", "write")

        # Get the rendered output
        rendered = self.buffer.render(height=50, width=80)
        rendered_str = str(rendered)

        # "Hello" should appear before the tool marker
        # "Goodbye" should appear after (or be in current block)
        hello_pos = rendered_str.find("Hello")
        goodbye_pos = rendered_str.find("Goodbye")

        # Both should be found (Goodbye might be in current_block which is rendered)
        assert hello_pos >= 0, "Should find 'Hello' in rendered output"
        # Note: Goodbye might be in current_block, not yet flushed

        # The tool should appear between them
        tool_pos = rendered_str.find("testTool")
        if goodbye_pos >= 0:
            assert hello_pos < tool_pos < goodbye_pos, (
                f"Expected order: Hello({hello_pos}) < tool({tool_pos}) < Goodbye({goodbye_pos})"
            )


class TestToolBlockRenderingPosition:
    """Tests specifically for how tool blocks are positioned in rendered output."""

    def setup_method(self):
        """Set up test fixtures."""
        self.buffer = OutputBuffer()
        self.buffer.set_width(80)

    def test_model_header_before_tool_block_in_render(self):
        """When rendered, the '── Model ──' header should appear BEFORE any tool blocks."""
        # User message to establish turn
        self.buffer.append("user", "Hello", "write")

        # Model responds (should get header)
        self.buffer.append("model", "I will help you.", "write")

        # Tool completes
        self.buffer.add_active_tool("readFile", {"path": "x"}, call_id="c1")
        self.buffer.mark_tool_completed("readFile", success=True, call_id="c1")

        # Trigger finalization
        self.buffer.append("model", "Done.", "write")

        # Render
        rendered = str(self.buffer.render(height=50, width=80))

        # Find positions
        model_header_pos = rendered.find("Model")
        tool_pos = rendered.find("readFile")

        assert model_header_pos >= 0, "Should find 'Model' header in output"
        assert tool_pos >= 0, "Should find 'readFile' tool in output"

        # Model header should come before tool
        assert model_header_pos < tool_pos, (
            f"Model header ({model_header_pos}) should appear before tool ({tool_pos})"
        )


class TestStreamingToolTreePositioning:
    """Tests for tool tree position with streaming model output (append mode)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.buffer = OutputBuffer()
        self.buffer.set_width(80)

    def test_tool_block_position_with_streaming_model(self):
        """Tool block position when model output is streamed via append mode.

        This simulates real-world streaming where:
        1. Model text starts with "write", continues with "append"
        2. Tools execute
        3. Model continues streaming

        The tool block should appear AFTER the initial model text.
        """
        # Start model output (write mode starts a new block)
        self.buffer.append("model", "I will ", "write")
        # Continue streaming (append mode)
        self.buffer.append("model", "help you ", "append")
        self.buffer.append("model", "test the diff preview.", "append")

        # Tool executes and completes
        self.buffer.add_active_tool("readFile", {"path": "a.py"}, call_id="c1")
        self.buffer.mark_tool_completed("readFile", success=True, call_id="c1")

        # More model text arrives (write mode triggers finalization)
        self.buffer.append("model", "Here are the results.", "write")

        # Verify order in _lines
        items_info = []
        for i, item in enumerate(self.buffer._lines):
            if isinstance(item, ToolBlock):
                items_info.append((i, "ToolBlock"))
            elif isinstance(item, OutputLine) and item.source == "model":
                items_info.append((i, f"Model: {item.text[:20]}..."))

        print(f"Items: {items_info}")

        # Find the model text that was streamed and the tool block
        model_indices = [i for i, t in items_info if t.startswith("Model")]
        block_indices = [i for i, t in items_info if t == "ToolBlock"]

        assert len(block_indices) == 1, f"Expected 1 tool block, got {len(block_indices)}"
        assert len(model_indices) >= 1, f"Expected at least 1 model text, got {len(model_indices)}"

        # First model text should come before tool block
        assert model_indices[0] < block_indices[0], (
            f"First model text ({model_indices[0]}) should come before tool block ({block_indices[0]})"
        )

    def test_multiple_streaming_rounds_with_tools(self):
        """Multiple rounds of streaming text + tools should maintain order.

        This is the scenario from the user's bug report:
        1. Model starts "I will help you..."
        2. First tools execute, tree rendered
        3. Model continues streaming
        4. Second tools execute
        5. etc.
        """
        # Round 1: User asks, model starts responding
        self.buffer.append("user", "Help me test diff preview", "write")

        # Model streams response
        self.buffer.append("model", "I'll ", "write")
        self.buffer.append("model", "help you test ", "append")
        self.buffer.append("model", "the code diff preview.", "append")

        # First batch of tools
        self.buffer.add_active_tool("glob_files", {"pattern": "**/*diff*"}, call_id="c1")
        self.buffer.add_active_tool("grep_content", {"pattern": "diff"}, call_id="c2")
        self.buffer.mark_tool_completed("glob_files", success=True, call_id="c1")
        self.buffer.mark_tool_completed("grep_content", success=True, call_id="c2")

        # Model continues (write mode - new text block)
        self.buffer.append("model", "I found ", "write")
        self.buffer.append("model", "several test files.", "append")

        # Second batch of tools
        self.buffer.add_active_tool("readFile", {"path": "test1.py"}, call_id="c3")
        self.buffer.add_active_tool("readFile", {"path": "test2.py"}, call_id="c4")
        self.buffer.mark_tool_completed("readFile", success=True, call_id="c3")
        self.buffer.mark_tool_completed("readFile", success=True, call_id="c4")

        # Model continues again
        self.buffer.append("model", "Now I will run the tests.", "write")

        # Collect item order
        order = []
        for i, item in enumerate(self.buffer._lines):
            if isinstance(item, ToolBlock):
                order.append(("ToolBlock", len(item.tools), i))
            elif isinstance(item, OutputLine):
                if item.source == "model" and item.text.strip():
                    order.append(("Model", item.text[:20], i))
                elif item.source == "user" and item.text.strip():
                    order.append(("User", item.text[:20], i))

        print(f"Order: {order}")

        # Expected order:
        # User, Model1, ToolBlock1 (2 tools), Model2, ToolBlock2 (2 tools)
        # (Model3 is in current_block, not yet in _lines)

        tool_blocks = [x for x in order if x[0] == "ToolBlock"]
        model_texts = [x for x in order if x[0] == "Model"]
        user_texts = [x for x in order if x[0] == "User"]

        assert len(tool_blocks) == 2, f"Expected 2 tool blocks, got {len(tool_blocks)}"
        assert len(model_texts) >= 2, f"Expected at least 2 model texts, got {len(model_texts)}"

        # Verify chronological order
        # User comes first
        assert user_texts[0][2] < model_texts[0][2], "User should come before first model"

        # First model before first tool block
        assert model_texts[0][2] < tool_blocks[0][2], (
            f"First model ({model_texts[0][2]}) should come before first tool block ({tool_blocks[0][2]})"
        )

        # First tool block before second model
        assert tool_blocks[0][2] < model_texts[1][2], (
            f"First tool block ({tool_blocks[0][2]}) should come before second model ({model_texts[1][2]})"
        )

        # Second model before second tool block
        assert model_texts[1][2] < tool_blocks[1][2], (
            f"Second model ({model_texts[1][2]}) should come before second tool block ({tool_blocks[1][2]})"
        )

    def test_render_output_maintains_streaming_order(self):
        """Rendered output should show items in correct chronological order."""
        # Setup scenario similar to user's report
        self.buffer.append("user", "Test the diff preview", "write")

        self.buffer.append("model", "I will help you test.", "write")
        self.buffer.add_active_tool("testTool", {"arg": 1}, call_id="c1")
        self.buffer.mark_tool_completed("testTool", success=True, call_id="c1")

        self.buffer.append("model", "Results are in.", "write")

        # Render
        rendered = str(self.buffer.render(height=50, width=80))

        # Find positions of key elements
        user_pos = rendered.find("Test the diff")
        model_header_pos = rendered.find("Model")
        help_pos = rendered.find("I will help")
        tool_pos = rendered.find("testTool")
        results_pos = rendered.find("Results are")

        # All should be found
        assert user_pos >= 0, "Should find user text"
        assert model_header_pos >= 0, "Should find Model header"
        assert help_pos >= 0, "Should find 'I will help'"
        assert tool_pos >= 0, "Should find tool"
        assert results_pos >= 0, "Should find 'Results are'"

        # Verify order: User < Model header < help text < tool < results
        assert user_pos < model_header_pos, f"User ({user_pos}) < Model header ({model_header_pos})"
        assert model_header_pos < help_pos, f"Model header ({model_header_pos}) < help ({help_pos})"
        assert help_pos < tool_pos, f"help ({help_pos}) < tool ({tool_pos})"
        assert tool_pos < results_pos, f"tool ({tool_pos}) < results ({results_pos})"


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.buffer = OutputBuffer()
        self.buffer.set_width(80)

    def test_finalize_tool_tree_explicit_call(self):
        """Explicit finalize_tool_tree() should preserve correct order.

        This simulates the scenario when generation is cancelled or completed
        and we need to explicitly finalize remaining tools.
        """
        self.buffer.append("model", "Starting analysis...", "write")

        # Add and complete tools
        self.buffer.add_active_tool("readFile", {"path": "a.py"}, call_id="c1")
        self.buffer.add_active_tool("readFile", {"path": "b.py"}, call_id="c2")
        self.buffer.mark_tool_completed("readFile", success=True, call_id="c1")
        self.buffer.mark_tool_completed("readFile", success=True, call_id="c2")

        # Explicitly finalize (e.g., on turn end or cancellation)
        self.buffer.finalize_tool_tree()

        # Verify order
        model_indices = []
        block_indices = []
        for i, item in enumerate(self.buffer._lines):
            if isinstance(item, OutputLine) and item.source == "model":
                model_indices.append(i)
            elif isinstance(item, ToolBlock):
                block_indices.append(i)

        assert len(block_indices) == 1, f"Expected 1 tool block, got {len(block_indices)}"
        assert len(model_indices) >= 1, f"Expected at least 1 model line, got {len(model_indices)}"

        # Model should come before tool block
        assert model_indices[0] < block_indices[0], (
            f"Model ({model_indices[0]}) should come before tool block ({block_indices[0]})"
        )

    def test_model_text_in_current_block_during_tool_execution(self):
        """Model text in _current_block should be flushed before ToolBlock is created.

        This tests the critical ordering requirement: when finalize_tool_tree() is called,
        it must flush _current_block first, then add the ToolBlock.
        """
        # Start model text (goes to _current_block)
        self.buffer.append("model", "Processing request...", "write")

        # Verify text is in _current_block
        assert self.buffer._current_block is not None, "Should have current block"
        assert self.buffer._current_block[0] == "model", "Current block should be model"

        # Add and complete tool
        self.buffer.add_active_tool("tool1", {}, call_id="c1")
        self.buffer.mark_tool_completed("tool1", success=True, call_id="c1")

        # Explicitly finalize
        self.buffer.finalize_tool_tree()

        # After finalization, current_block should be flushed
        assert self.buffer._current_block is None, "Current block should be flushed"

        # And the order should be: model text, then tool block
        item_types = [(type(item).__name__, getattr(item, 'source', 'N/A'))
                      for item in self.buffer._lines]
        print(f"Item types: {item_types}")

        # Find first model and first tool block
        first_model_idx = next(
            (i for i, item in enumerate(self.buffer._lines)
             if isinstance(item, OutputLine) and item.source == "model"),
            -1
        )
        first_block_idx = next(
            (i for i, item in enumerate(self.buffer._lines)
             if isinstance(item, ToolBlock)),
            -1
        )

        assert first_model_idx >= 0, "Should have model text"
        assert first_block_idx >= 0, "Should have tool block"
        assert first_model_idx < first_block_idx, (
            f"Model ({first_model_idx}) must come before tool block ({first_block_idx}). "
            f"Items: {item_types}"
        )

    def test_render_while_tools_active_vs_finalized(self):
        """Compare render output when tools are active vs finalized.

        Active tools now render inline at their placeholder position (not at bottom).
        Finalized tools (ToolBlock) also render inline. Both should show model text first.
        """
        # Setup
        self.buffer.append("model", "Working on it...", "write")
        self.buffer.add_active_tool("myTool", {"arg": 1}, call_id="c1")
        self.buffer.mark_tool_completed("myTool", success=True, call_id="c1")

        # Expand tools so tool names are visible
        self.buffer._tools_expanded = True

        # Render while tools are still "active" (not yet finalized)
        render_active = str(self.buffer.render(height=50, width=80))

        # Find positions
        model_pos_active = render_active.find("Working on it")
        tool_pos_active = render_active.find("myTool")

        assert model_pos_active >= 0, "Should find model text in active render"
        assert tool_pos_active >= 0, "Should find tool in active render (expanded view)"

        # Tools render inline at their position - model text should appear before tool section
        model_header_pos = render_active.find("Model")
        assert model_header_pos < tool_pos_active or model_pos_active < tool_pos_active, (
            "Model content should appear before tool section"
        )

        # Now finalize
        self.buffer.finalize_tool_tree()

        # Render after finalization
        render_finalized = str(self.buffer.render(height=50, width=80))

        model_pos_final = render_finalized.find("Working on it")
        tool_pos_final = render_finalized.find("myTool")

        assert model_pos_final >= 0, "Should find model text in finalized render"
        assert tool_pos_final >= 0, "Should find tool in finalized render"

        # After finalization, order should still be: model before tool
        assert model_pos_final < tool_pos_final, (
            f"Model ({model_pos_final}) should appear before tool ({tool_pos_final}) after finalization"
        )

    def test_turn_header_not_duplicated(self):
        """Model header should appear only once per turn, even with tool interruptions."""
        self.buffer.append("user", "Do something", "write")

        # First model chunk (should have turn header)
        self.buffer.append("model", "Starting", "write")

        # Tool
        self.buffer.add_active_tool("tool1", {}, call_id="c1")
        self.buffer.mark_tool_completed("tool1", success=True, call_id="c1")

        # More model text (should NOT have turn header - same turn)
        self.buffer.append("model", "Continuing", "write")

        # The continuation is still in _current_block, not _lines yet
        # Check that it has is_new_turn=False
        assert self.buffer._current_block is not None, "Should have current block"
        source, parts, is_new_turn = self.buffer._current_block
        assert source == "model", "Current block should be model"
        assert is_new_turn is False, "Continuation should NOT be a new turn"

        # Verify exactly one model line in _lines has is_turn_start=True
        # (tool finalization adds a trailing blank model line with is_turn_start=False)
        model_lines = [item for item in self.buffer._lines
                       if isinstance(item, OutputLine) and item.source == "model"]
        turn_starts = [l for l in model_lines if l.is_turn_start]
        assert len(turn_starts) == 1, f"Expected 1 turn start, got {len(turn_starts)}"
        assert turn_starts[0].text == "Starting", "Turn start should be the first model text"


class TestToolBlockExpansionPersistence:
    """Tests for tool block expansion state persistence after finalization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.buffer = OutputBuffer()
        self.buffer.set_width(80)

    def test_user_expanded_state_persists_after_finalization(self):
        """User's manual expansion should persist in the finalized ToolBlock.

        Scenario:
        1. Tool starts (default: collapsed)
        2. User toggles to expand
        3. Tool completes and finalizes
        4. ToolBlock should be expanded
        """
        # Start with default collapsed state
        assert self.buffer._tools_expanded is False

        # Add a tool
        self.buffer.add_active_tool("testTool", {"arg": "value"}, call_id="call_1")

        # User manually expands (like pressing Ctrl+T)
        self.buffer.toggle_tools_expanded()
        assert self.buffer._tools_expanded is True

        # Tool completes
        self.buffer.mark_tool_completed("testTool", success=True, call_id="call_1")

        # Finalize the tool tree
        self.buffer.finalize_tool_tree()

        # Find the tool block
        tool_blocks = [item for item in self.buffer._lines if isinstance(item, ToolBlock)]
        assert len(tool_blocks) == 1, "Expected 1 tool block"

        # The block should be expanded (user's choice persisted)
        assert tool_blocks[0].expanded is True, "ToolBlock should be expanded as user chose"

    def test_user_collapsed_state_persists_after_finalization(self):
        """User's manual collapse should persist in the finalized ToolBlock.

        Scenario:
        1. Tool starts
        2. State is expanded (e.g., from previous toggle or forced expansion)
        3. User toggles to collapse
        4. Tool completes and finalizes
        5. ToolBlock should be collapsed
        """
        # Start with expanded state
        self.buffer._tools_expanded = True

        # Add a tool
        self.buffer.add_active_tool("testTool", {"arg": "value"}, call_id="call_1")

        # User manually collapses (like pressing Ctrl+T)
        self.buffer.toggle_tools_expanded()
        assert self.buffer._tools_expanded is False

        # Tool completes
        self.buffer.mark_tool_completed("testTool", success=True, call_id="call_1")

        # Finalize the tool tree
        self.buffer.finalize_tool_tree()

        # Find the tool block
        tool_blocks = [item for item in self.buffer._lines if isinstance(item, ToolBlock)]
        assert len(tool_blocks) == 1, "Expected 1 tool block"

        # The block should be collapsed (user's choice persisted)
        assert tool_blocks[0].expanded is False, "ToolBlock should be collapsed as user chose"

    def test_user_toggle_overrides_permission_saved_state(self):
        """User's toggle during permission prompt should override the saved state.

        Scenario:
        1. Tool starts collapsed
        2. Permission pending forces expansion (saves collapsed state)
        3. User toggles to collapse (explicit choice)
        4. Permission resolves - should NOT restore to expanded
        5. Tool completes and finalizes
        6. ToolBlock should be collapsed (user's explicit choice)
        """
        # Start collapsed
        assert self.buffer._tools_expanded is False
        assert self.buffer._tools_expanded_before_prompt is None

        # Add a tool
        self.buffer.add_active_tool("dangerousTool", {"cmd": "rm -rf /"}, call_id="call_1")

        # Permission pending via unified flow - saves collapsed state and forces expansion
        self.buffer._pending_permission_content = "Allow dangerous operation?"
        self.buffer.set_tool_awaiting_approval("dangerousTool", call_id="call_1")

        # Verify state was saved and forced to expanded
        assert self.buffer._tools_expanded is True, "Should be forced to expanded"
        assert self.buffer._tools_expanded_before_prompt is False, "Should save original collapsed state"

        # User explicitly toggles to collapse (explicit choice)
        self.buffer.toggle_tools_expanded()
        assert self.buffer._tools_expanded is False, "User chose to collapse"
        # The saved state should be cleared so restore won't override user's choice
        assert self.buffer._tools_expanded_before_prompt is None, "Saved state should be cleared"

        # Permission resolves
        self.buffer.set_tool_permission_resolved("dangerousTool", "granted", "user_approved")

        # State should remain as user chose (collapsed), not restored to saved state
        assert self.buffer._tools_expanded is False, "Should remain collapsed per user's choice"

        # Tool completes
        self.buffer.mark_tool_completed("dangerousTool", success=True, call_id="call_1")

        # Finalize
        self.buffer.finalize_tool_tree()

        # Find the tool block
        tool_blocks = [item for item in self.buffer._lines if isinstance(item, ToolBlock)]
        assert len(tool_blocks) == 1, "Expected 1 tool block"

        # The block should be collapsed (user's explicit choice during execution)
        assert tool_blocks[0].expanded is False, "ToolBlock should be collapsed per user's choice"

    def test_system_forced_expansion_uses_original_preference(self):
        """System-forced expansion (permission) should use user's original preference in ToolBlock.

        When expansion is forced by the system (permission/clarification prompt) and user
        doesn't manually toggle, the finalized ToolBlock should use the user's ORIGINAL
        preference (before the system forced expansion), not the forced state.

        This ensures that temporary UI changes for prompts don't override user preferences.
        """
        # Start collapsed (user's preference)
        assert self.buffer._tools_expanded is False

        # Add a tool
        self.buffer.add_active_tool("dangerousTool", {"cmd": "rm"}, call_id="call_1")

        # Permission pending via unified flow - forces expansion, saves original preference
        self.buffer._pending_permission_content = "Allow?"
        self.buffer.set_tool_awaiting_approval("dangerousTool", call_id="call_1")
        assert self.buffer._tools_expanded is True, "Should be forced to expanded"
        assert self.buffer._tools_expanded_before_prompt is False, "Should save user's original preference"

        # Permission resolves (user approves, doesn't toggle)
        self.buffer.set_tool_permission_resolved("dangerousTool", "granted", "user_approved")

        # State should NOT be restored yet - tools are still active
        assert self.buffer._tools_expanded is True, "Should still be expanded (tools still active)"

        # Tool completes
        self.buffer.mark_tool_completed("dangerousTool", success=True, call_id="call_1")

        # Finalize
        self.buffer.finalize_tool_tree()

        # Find the tool block
        tool_blocks = [item for item in self.buffer._lines if isinstance(item, ToolBlock)]
        assert len(tool_blocks) == 1, "Expected 1 tool block"

        # The ToolBlock uses user's ORIGINAL preference (collapsed), not the forced state
        assert tool_blocks[0].expanded is False, (
            "ToolBlock should use user's original preference (collapsed), not system-forced state"
        )

        # After finalization, _tools_expanded is restored to original preference
        assert self.buffer._tools_expanded is False, (
            "_tools_expanded should be restored to collapsed for future tools"
        )


class TestToolsExpandedDefault:
    """Tests for configurable tools_expanded default (headless vs interactive)."""

    def test_default_tools_collapsed(self):
        """Interactive mode: tools default to collapsed."""
        buffer = OutputBuffer()
        assert buffer._tools_expanded is False

    def test_headless_tools_expanded(self):
        """Headless mode: tools can default to expanded via constructor parameter."""
        buffer = OutputBuffer(tools_expanded=True)
        assert buffer._tools_expanded is True

    def test_headless_expanded_persists_in_toolblock(self):
        """ToolBlocks created with expanded=True default should be expanded."""
        buffer = OutputBuffer(tools_expanded=True)
        buffer.set_width(80)

        buffer.add_active_tool("testTool", {"arg": "value"}, call_id="call_1")
        buffer.mark_tool_completed("testTool", success=True, call_id="call_1")
        buffer.finalize_tool_tree()

        # Find the ToolBlock in the lines
        tool_blocks = [item for item in buffer._lines if isinstance(item, ToolBlock)]
        assert len(tool_blocks) == 1, "Should have one ToolBlock"
        assert tool_blocks[0].expanded is True, "ToolBlock should be expanded in headless mode"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
