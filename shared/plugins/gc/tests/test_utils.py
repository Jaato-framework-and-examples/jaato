"""Tests for GC utility functions."""

import pytest

from shared.plugins.gc.utils import (
    Turn,
    ensure_tool_call_integrity,
    split_into_turns,
    flatten_turns,
    estimate_message_tokens,
    estimate_turn_tokens,
    estimate_history_tokens,
    create_summary_message,
    create_gc_notification_message,
    get_preserved_indices,
)
from jaato import Message, Part, Role, FunctionCall, ToolResult


def make_message(role: str, text: str) -> Message:
    """Helper to create Message objects."""
    r = Role.USER if role == "user" else Role.MODEL
    return Message(
        role=r,
        parts=[Part(text=text)]
    )


class TestTurn:
    def test_turn_creation(self):
        contents = [make_message("user", "Hello")]
        turn = Turn(index=0, contents=contents, estimated_tokens=10)

        assert turn.index == 0
        assert len(turn.contents) == 1
        assert turn.estimated_tokens == 10
        assert not turn.is_empty

    def test_empty_turn(self):
        turn = Turn(index=0, contents=[], estimated_tokens=0)
        assert turn.is_empty


class TestSplitIntoTurns:
    def test_empty_history(self):
        turns = split_into_turns([])
        assert turns == []

    def test_single_user_message(self):
        history = [make_message("user", "Hello")]
        turns = split_into_turns(history)

        assert len(turns) == 1
        assert turns[0].index == 0
        assert len(turns[0].contents) == 1

    def test_user_model_pair(self):
        history = [
            make_message("user", "Hello"),
            make_message("model", "Hi there!"),
        ]
        turns = split_into_turns(history)

        assert len(turns) == 1
        assert len(turns[0].contents) == 2

    def test_multiple_turns(self):
        history = [
            make_message("user", "Hello"),
            make_message("model", "Hi!"),
            make_message("user", "How are you?"),
            make_message("model", "I'm good!"),
        ]
        turns = split_into_turns(history)

        assert len(turns) == 2
        assert turns[0].index == 0
        assert turns[1].index == 1

    def test_function_response_grouped_with_turn(self):
        # Function responses have role='user' but should not start new turn
        fc_result = ToolResult(
            call_id="test_id",
            name="test_func",
            result={"result": "ok"}
        )
        fc_part = Part.from_function_response(fc_result)
        history = [
            make_message("user", "Call the function"),
            Message(role=Role.MODEL, parts=[
                Part(function_call=FunctionCall(
                    id="test_id",
                    name="test_func",
                    args={}
                ))
            ]),
            Message(role=Role.USER, parts=[fc_part]),
            make_message("model", "Function returned ok"),
        ]
        turns = split_into_turns(history)

        # All should be in same turn since function response doesn't start new turn
        assert len(turns) == 1


class TestFlattenTurns:
    def test_flatten_empty(self):
        assert flatten_turns([]) == []

    def test_flatten_single_turn(self):
        contents = [
            make_message("user", "Hello"),
            make_message("model", "Hi!"),
        ]
        turns = [Turn(index=0, contents=contents)]

        result = flatten_turns(turns)
        assert len(result) == 2

    def test_flatten_multiple_turns(self):
        turn1 = Turn(index=0, contents=[make_message("user", "A")])
        turn2 = Turn(index=1, contents=[make_message("user", "B")])

        result = flatten_turns([turn1, turn2])
        assert len(result) == 2

    def test_roundtrip(self):
        """split -> flatten should preserve content."""
        history = [
            make_message("user", "Hello"),
            make_message("model", "Hi!"),
            make_message("user", "How are you?"),
            make_message("model", "Good!"),
        ]

        turns = split_into_turns(history)
        result = flatten_turns(turns)

        assert len(result) == len(history)


class TestTokenEstimation:
    def test_estimate_message_tokens_text(self):
        content = make_message("user", "Hello world")  # 11 chars
        tokens = estimate_message_tokens(content)

        # ~4 chars per token, minimum 1
        assert tokens >= 1

    def test_estimate_message_tokens_empty(self):
        content = Message(role=Role.USER, parts=[])
        tokens = estimate_message_tokens(content)
        assert tokens >= 1

    def test_estimate_turn_tokens(self):
        contents = [
            make_message("user", "Hello"),
            make_message("model", "Hi there!"),
        ]
        tokens = estimate_turn_tokens(contents)
        assert tokens > 0

    def test_estimate_history_tokens(self):
        history = [
            make_message("user", "Hello world"),
            make_message("model", "Hi there, how can I help?"),
        ]
        tokens = estimate_history_tokens(history)
        assert tokens > 0


class TestCreateSummaryMessage:
    def test_creates_user_role_message(self):
        summary = create_summary_message("This is a summary")

        assert summary.role == Role.USER
        assert len(summary.parts) == 1

    def test_includes_markers(self):
        summary = create_summary_message("This is a summary")
        text = summary.parts[0].text

        assert "[Context Summary" in text
        assert "This is a summary" in text
        assert "[End Context Summary]" in text


class TestCreateGCNotificationMessage:
    def test_creates_notification(self):
        notification = create_gc_notification_message("GC happened")

        assert notification.role == Role.USER
        assert "[System:" in notification.parts[0].text
        assert "GC happened" in notification.parts[0].text


class TestGetPreservedIndices:
    def test_preserve_recent_only(self):
        preserved = get_preserved_indices(
            total_turns=10,
            preserve_recent=3,
            pinned_indices=None
        )

        assert preserved == {7, 8, 9}

    def test_preserve_all_when_few_turns(self):
        preserved = get_preserved_indices(
            total_turns=3,
            preserve_recent=5,
            pinned_indices=None
        )

        assert preserved == {0, 1, 2}

    def test_pinned_indices(self):
        preserved = get_preserved_indices(
            total_turns=10,
            preserve_recent=2,
            pinned_indices=[0, 3]
        )

        assert 0 in preserved  # pinned
        assert 3 in preserved  # pinned
        assert 8 in preserved  # recent
        assert 9 in preserved  # recent

    def test_pinned_out_of_range_ignored(self):
        preserved = get_preserved_indices(
            total_turns=5,
            preserve_recent=2,
            pinned_indices=[0, 100]  # 100 is out of range
        )

        assert 0 in preserved
        assert 100 not in preserved

    def test_zero_preserve_recent(self):
        preserved = get_preserved_indices(
            total_turns=5,
            preserve_recent=0,
            pinned_indices=[2]
        )

        assert preserved == {2}


# --- Helpers for tool call integrity tests ---

def make_model_with_tool_calls(call_ids, text=None):
    """Create a MODEL message with function calls."""
    parts = []
    if text:
        parts.append(Part(text=text))
    for cid in call_ids:
        parts.append(Part(function_call=FunctionCall(
            id=cid, name=f"tool_{cid}", args={}
        )))
    return Message(role=Role.MODEL, parts=parts)


def make_tool_results(call_ids):
    """Create a TOOL message with function responses."""
    parts = []
    for cid in call_ids:
        parts.append(Part(function_response=ToolResult(
            call_id=cid, name=f"tool_{cid}", result={"ok": True}
        )))
    return Message(role=Role.TOOL, parts=parts)


def make_user_tool_results(call_ids):
    """Create a USER message with function responses (Google GenAI style)."""
    parts = []
    for cid in call_ids:
        parts.append(Part(function_response=ToolResult(
            call_id=cid, name=f"tool_{cid}", result={"ok": True}
        )))
    return Message(role=Role.USER, parts=parts)


class TestEnsureToolCallIntegrity:
    """Tests for ensure_tool_call_integrity().

    This function repairs tool_use/tool_result pairing after GC removes
    messages individually, which can create orphaned entries.
    """

    def test_empty_history(self):
        assert ensure_tool_call_integrity([]) == []

    def test_no_tool_calls_unchanged(self):
        """History without tool calls passes through unchanged."""
        history = [
            make_message("user", "Hello"),
            make_message("model", "Hi there!"),
            make_message("user", "How are you?"),
            make_message("model", "Good!"),
        ]
        result = ensure_tool_call_integrity(history)
        assert len(result) == len(history)

    def test_valid_pairing_unchanged(self):
        """Correctly paired tool_use/tool_result passes through unchanged."""
        history = [
            make_message("user", "Search for X"),
            make_model_with_tool_calls(["call_1"]),
            make_tool_results(["call_1"]),
            make_message("model", "Here are the results"),
        ]
        result = ensure_tool_call_integrity(history)
        assert len(result) == 4

    def test_valid_multiple_calls_in_one_group(self):
        """Multiple tool calls in one MODEL message, all with results."""
        history = [
            make_message("user", "Do two things"),
            make_model_with_tool_calls(["call_A", "call_B"]),
            make_tool_results(["call_A", "call_B"]),
            make_message("model", "Done with both"),
        ]
        result = ensure_tool_call_integrity(history)
        assert len(result) == 4

    def test_orphaned_tool_result_removed(self):
        """Tool result whose call_id has no matching MODEL is removed.

        This is the primary bug scenario: GC removed the MODEL message
        containing the tool_call, leaving an orphaned tool_result.
        """
        history = [
            make_message("user", "Search for X"),
            # MODEL with tool_call was removed by GC
            make_tool_results(["call_orphaned"]),  # orphaned
            make_message("model", "Here are the results"),
        ]
        result = ensure_tool_call_integrity(history)
        # The orphaned tool_result should be removed (3 - 1 = 2)
        assert len(result) == 2
        assert result[0].role == Role.USER
        assert result[1].role == Role.MODEL
        assert result[1].parts[0].text == "Here are the results"

    def test_orphaned_tool_result_gc_scenario(self):
        """Simulates the actual GC bug: GC removes a large tool_result,
        leaving the MODEL's tool_call without a matching result, then
        another MODEL message's tool_result becomes orphaned.

        Before GC:
            user -> model(call_A) -> tool(A) -> model(call_B) -> tool(B) -> model(text)
        After GC removes tool(A) and model(call_A):
            user -> tool(B) -> model(text)  -- tool(B) is orphaned
        But actually GC removes by message_id, so more likely:
            user -> model(call_A) -> model(call_B) -> tool(B) -> model(text)
        Where model(call_A) has unpaired tool_call.
        """
        # Scenario: GC removed tool_result(A) but kept model(call_A)
        history = [
            make_message("user", "Do tasks"),
            make_model_with_tool_calls(["call_A"]),
            # tool_result(A) was removed by GC!
            make_model_with_tool_calls(["call_B"]),
            make_tool_results(["call_B"]),
            make_message("model", "All done"),
        ]
        result = ensure_tool_call_integrity(history)
        # model(call_A) should be removed because it has no tool_result
        assert len(result) == 4
        assert result[0].parts[0].text == "Do tasks"
        # The model with call_B should remain with its tool_result
        assert any(
            p.function_call and p.function_call.id == "call_B"
            for msg in result for p in msg.parts
        )

    def test_orphaned_tool_result_with_user_role(self):
        """USER-role tool results (Google GenAI style) are also validated."""
        history = [
            make_message("user", "Search for X"),
            # MODEL with tool_call was removed by GC
            make_user_tool_results(["call_orphaned"]),  # orphaned USER tool result
            make_message("model", "Results"),
        ]
        result = ensure_tool_call_integrity(history)
        assert len(result) == 2
        assert result[0].parts[0].text == "Search for X"
        assert result[1].parts[0].text == "Results"

    def test_unpaired_tool_use_at_end_removed(self):
        """MODEL with tool_calls at end of history with no results is removed."""
        history = [
            make_message("user", "Search for X"),
            make_model_with_tool_calls(["call_1"]),
            # No tool_result follows - end of history
        ]
        result = ensure_tool_call_integrity(history)
        assert len(result) == 1
        assert result[0].parts[0].text == "Search for X"

    def test_unpaired_tool_use_before_user_removed(self):
        """MODEL with tool_calls followed by USER (no results) is removed."""
        history = [
            make_message("user", "Do X"),
            make_model_with_tool_calls(["call_1"]),
            # No tool_result - next is a user message
            make_message("user", "Never mind, do Y"),
            make_message("model", "OK doing Y"),
        ]
        result = ensure_tool_call_integrity(history)
        assert len(result) == 3
        assert result[0].parts[0].text == "Do X"
        assert result[1].parts[0].text == "Never mind, do Y"
        assert result[2].parts[0].text == "OK doing Y"

    def test_mixed_valid_and_orphaned_across_turns(self):
        """Complex scenario: multiple turns, some corrupted by GC."""
        history = [
            make_message("user", "Turn 1"),
            make_model_with_tool_calls(["call_1"]),
            make_tool_results(["call_1"]),
            make_message("model", "Turn 1 complete"),
            make_message("user", "Turn 2"),
            # GC removed model(call_2), leaving orphaned tool_result
            make_tool_results(["call_2_orphaned"]),
            make_message("model", "Turn 2 complete"),
            make_message("user", "Turn 3"),
            make_model_with_tool_calls(["call_3"]),
            make_tool_results(["call_3"]),
            make_message("model", "Turn 3 complete"),
        ]
        result = ensure_tool_call_integrity(history)
        # Turn 1 intact (4 msgs), Turn 2 orphan removed (2 msgs), Turn 3 intact (4 msgs)
        assert len(result) == 10
        # Verify orphaned tool_result is gone
        for msg in result:
            for p in msg.parts:
                if p.function_response:
                    assert p.function_response.call_id != "call_2_orphaned"

    def test_trace_callback_called(self):
        """Trace function is called when messages are removed."""
        traces = []
        history = [
            make_message("user", "Search"),
            make_tool_results(["orphaned_call"]),
            make_message("model", "Done"),
        ]
        result = ensure_tool_call_integrity(history, trace_fn=traces.append)
        assert len(result) == 2
        assert any("orphaned" in t.lower() or "orphan" in t.lower() for t in traces)

    def test_model_with_text_and_tool_calls_removed_when_unpaired(self):
        """MODEL message that has BOTH text and tool_calls is removed
        when tool_calls are unpaired (even though it has text)."""
        history = [
            make_message("user", "Do X"),
            make_model_with_tool_calls(["call_1"], text="I'll search for you"),
            # No tool_result follows
            make_message("user", "OK what happened?"),
            make_message("model", "Sorry, let me try again"),
        ]
        result = ensure_tool_call_integrity(history)
        # The MODEL with unpaired tool_calls should be removed
        assert len(result) == 3

    def test_sequential_tool_call_chains(self):
        """Multiple sequential tool_use/tool_result pairs stay intact."""
        history = [
            make_message("user", "Multi-step task"),
            make_model_with_tool_calls(["call_1"]),
            make_tool_results(["call_1"]),
            make_model_with_tool_calls(["call_2"]),
            make_tool_results(["call_2"]),
            make_model_with_tool_calls(["call_3"]),
            make_tool_results(["call_3"]),
            make_message("model", "All steps complete"),
        ]
        result = ensure_tool_call_integrity(history)
        assert len(result) == 8  # All intact

    def test_gc_removes_middle_of_chain(self):
        """GC removes tool_result from middle of chain, cascading fix.

        Before GC: user -> model(A) -> tool(A) -> model(B) -> tool(B) -> model(text)
        After GC removes tool(A): user -> model(A) -> model(B) -> tool(B) -> model(text)
        model(A) is unpaired (no tool_result before next MODEL with tool_calls).
        """
        history = [
            make_message("user", "Multi-step"),
            make_model_with_tool_calls(["call_A"]),
            # tool_result(A) removed by GC!
            make_model_with_tool_calls(["call_B"]),
            make_tool_results(["call_B"]),
            make_message("model", "Done"),
        ]
        result = ensure_tool_call_integrity(history)
        # model(call_A) removed, rest stays
        assert len(result) == 4
        roles = [msg.role for msg in result]
        assert roles == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
