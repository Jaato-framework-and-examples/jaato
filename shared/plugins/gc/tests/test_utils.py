"""Tests for GC utility functions."""

import pytest

from shared.plugins.gc.utils import (
    Turn,
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
