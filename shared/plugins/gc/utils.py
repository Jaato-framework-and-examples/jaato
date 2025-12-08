"""Utility functions for Context Garbage Collection.

Provides helpers for turn splitting, token estimation, and history manipulation.
"""

from dataclasses import dataclass
from typing import List, Optional

from ..model_provider.types import Message, Part, Role


@dataclass
class Turn:
    """Represents a conversation turn (user message + model response(s)).

    A turn typically consists of:
    - One user Message (role=USER)
    - One or more model Message objects (role=MODEL)
    - Possibly function response Message objects (role=USER with function_response parts)
    """

    index: int
    """Turn index (0-based)."""

    contents: List[Message]
    """All Message objects in this turn."""

    estimated_tokens: int = 0
    """Estimated token count for this turn."""

    @property
    def is_empty(self) -> bool:
        """Check if this turn has no content."""
        return len(self.contents) == 0


def split_into_turns(history: List[Message]) -> List[Turn]:
    """Split conversation history into logical turns.

    A turn starts with a user message and includes all subsequent
    model responses until the next user message. Function responses
    (user role with function_response parts) are grouped with the
    preceding model response.

    Args:
        history: List of Message objects from conversation history.

    Returns:
        List of Turn objects, each containing related Message objects.
    """
    if not history:
        return []

    turns: List[Turn] = []
    current_turn_contents: List[Message] = []
    turn_index = 0

    for message in history:
        # Check if this is a new user message (not a function response)
        is_user_message = message.role == Role.USER
        is_function_response = False

        if is_user_message and message.parts:
            # Check if it's a function response (has function_response parts)
            is_function_response = any(
                part.function_response is not None
                for part in message.parts
            )

        # Start new turn on user message (not function response)
        if is_user_message and not is_function_response and current_turn_contents:
            # Save current turn
            turns.append(Turn(
                index=turn_index,
                contents=current_turn_contents,
                estimated_tokens=estimate_turn_tokens(current_turn_contents)
            ))
            turn_index += 1
            current_turn_contents = []

        current_turn_contents.append(message)

    # Don't forget the last turn
    if current_turn_contents:
        turns.append(Turn(
            index=turn_index,
            contents=current_turn_contents,
            estimated_tokens=estimate_turn_tokens(current_turn_contents)
        ))

    return turns


def flatten_turns(turns: List[Turn]) -> List[Message]:
    """Flatten a list of turns back into a content list.

    Args:
        turns: List of Turn objects.

    Returns:
        Flattened list of Message objects preserving order.
    """
    result: List[Message] = []
    for turn in turns:
        result.extend(turn.contents)
    return result


def estimate_message_tokens(message: Message) -> int:
    """Estimate token count for a single Message object.

    Uses a simple heuristic: ~4 characters per token.
    This is approximate but avoids API calls for counting.

    Args:
        message: A Message object to estimate.

    Returns:
        Estimated token count.
    """
    total_chars = 0

    if message.parts:
        for part in message.parts:
            # Text parts
            if part.text:
                total_chars += len(part.text)

            # Function call parts
            elif part.function_call:
                fc = part.function_call
                total_chars += len(fc.name) if fc.name else 0
                if fc.args:
                    # Args is typically a dict, estimate from string repr
                    total_chars += len(str(fc.args))

            # Function response parts
            elif part.function_response:
                fr = part.function_response
                total_chars += len(fr.name) if fr.name else 0
                if fr.response:
                    total_chars += len(str(fr.response))

    # Rough estimate: 4 chars per token (conservative)
    return max(1, total_chars // 4)


def estimate_turn_tokens(contents: List[Message]) -> int:
    """Estimate token count for a list of Message objects.

    Args:
        contents: List of Message objects.

    Returns:
        Total estimated token count.
    """
    return sum(estimate_message_tokens(c) for c in contents)


def estimate_history_tokens(history: List[Message]) -> int:
    """Estimate total token count for entire history.

    Args:
        history: Full conversation history.

    Returns:
        Total estimated token count.
    """
    return estimate_turn_tokens(history)


def create_summary_message(summary_text: str) -> Message:
    """Create a Message object containing a context summary.

    The summary is marked with special delimiters so the model
    understands it's compressed context, not a user message.

    Args:
        summary_text: The summary text to include.

    Returns:
        A Message object with role=USER containing the summary.
    """
    formatted_summary = (
        "[Context Summary - Previous conversation compressed]\n"
        f"{summary_text}\n"
        "[End Context Summary]"
    )

    return Message(
        role=Role.USER,
        parts=[Part(text=formatted_summary)]
    )


def create_gc_notification_message(message: str) -> Message:
    """Create a Message object notifying about GC.

    Args:
        message: The notification message.

    Returns:
        A Message object with the notification.
    """
    formatted_message = f"[System: {message}]"

    return Message(
        role=Role.USER,
        parts=[Part(text=formatted_message)]
    )


def get_preserved_indices(
    total_turns: int,
    preserve_recent: int,
    pinned_indices: Optional[List[int]] = None
) -> set:
    """Calculate which turn indices should be preserved.

    Args:
        total_turns: Total number of turns in history.
        preserve_recent: Number of recent turns to preserve.
        pinned_indices: Additional indices to preserve.

    Returns:
        Set of turn indices that should not be collected.
    """
    preserved = set()

    # Always preserve recent turns
    if preserve_recent > 0:
        start_recent = max(0, total_turns - preserve_recent)
        preserved.update(range(start_recent, total_turns))

    # Add pinned indices
    if pinned_indices:
        for idx in pinned_indices:
            if 0 <= idx < total_turns:
                preserved.add(idx)

    return preserved
