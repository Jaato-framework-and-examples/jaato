"""Utility functions for Context Garbage Collection.

Provides helpers for turn splitting, token estimation, and history manipulation.
"""

from dataclasses import dataclass
from typing import List, Optional

from google.genai import types


@dataclass
class Turn:
    """Represents a conversation turn (user message + model response(s)).

    A turn typically consists of:
    - One user Content (role='user')
    - One or more model Content objects (role='model')
    - Possibly function response Content objects (role='user' with function_response parts)
    """

    index: int
    """Turn index (0-based)."""

    contents: List[types.Content]
    """All Content objects in this turn."""

    estimated_tokens: int = 0
    """Estimated token count for this turn."""

    @property
    def is_empty(self) -> bool:
        """Check if this turn has no content."""
        return len(self.contents) == 0


def split_into_turns(history: List[types.Content]) -> List[Turn]:
    """Split conversation history into logical turns.

    A turn starts with a user message and includes all subsequent
    model responses until the next user message. Function responses
    (user role with function_response parts) are grouped with the
    preceding model response.

    Args:
        history: List of Content objects from conversation history.

    Returns:
        List of Turn objects, each containing related Content objects.
    """
    if not history:
        return []

    turns: List[Turn] = []
    current_turn_contents: List[types.Content] = []
    turn_index = 0

    for content in history:
        # Check if this is a new user message (not a function response)
        is_user_message = content.role == "user"
        is_function_response = False

        if is_user_message and content.parts:
            # Check if it's a function response (has function_response parts)
            is_function_response = any(
                hasattr(part, 'function_response') and part.function_response is not None
                for part in content.parts
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

        current_turn_contents.append(content)

    # Don't forget the last turn
    if current_turn_contents:
        turns.append(Turn(
            index=turn_index,
            contents=current_turn_contents,
            estimated_tokens=estimate_turn_tokens(current_turn_contents)
        ))

    return turns


def flatten_turns(turns: List[Turn]) -> List[types.Content]:
    """Flatten a list of turns back into a content list.

    Args:
        turns: List of Turn objects.

    Returns:
        Flattened list of Content objects preserving order.
    """
    result: List[types.Content] = []
    for turn in turns:
        result.extend(turn.contents)
    return result


def estimate_content_tokens(content: types.Content) -> int:
    """Estimate token count for a single Content object.

    Uses a simple heuristic: ~4 characters per token.
    This is approximate but avoids API calls for counting.

    Args:
        content: A Content object to estimate.

    Returns:
        Estimated token count.
    """
    total_chars = 0

    if content.parts:
        for part in content.parts:
            # Text parts
            if hasattr(part, 'text') and part.text:
                total_chars += len(part.text)

            # Function call parts
            elif hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                total_chars += len(fc.name) if fc.name else 0
                if fc.args:
                    # Args is typically a dict, estimate from string repr
                    total_chars += len(str(fc.args))

            # Function response parts
            elif hasattr(part, 'function_response') and part.function_response:
                fr = part.function_response
                total_chars += len(fr.name) if fr.name else 0
                if fr.response:
                    total_chars += len(str(fr.response))

    # Rough estimate: 4 chars per token (conservative)
    return max(1, total_chars // 4)


def estimate_turn_tokens(contents: List[types.Content]) -> int:
    """Estimate token count for a list of Content objects.

    Args:
        contents: List of Content objects.

    Returns:
        Total estimated token count.
    """
    return sum(estimate_content_tokens(c) for c in contents)


def estimate_history_tokens(history: List[types.Content]) -> int:
    """Estimate total token count for entire history.

    Args:
        history: Full conversation history.

    Returns:
        Total estimated token count.
    """
    return estimate_turn_tokens(history)


def create_summary_content(summary_text: str) -> types.Content:
    """Create a Content object containing a context summary.

    The summary is marked with special delimiters so the model
    understands it's compressed context, not a user message.

    Args:
        summary_text: The summary text to include.

    Returns:
        A Content object with role='user' containing the summary.
    """
    formatted_summary = (
        "[Context Summary - Previous conversation compressed]\n"
        f"{summary_text}\n"
        "[End Context Summary]"
    )

    return types.Content(
        role="user",
        parts=[types.Part(text=formatted_summary)]
    )


def create_gc_notification_content(message: str) -> types.Content:
    """Create a Content object notifying about GC.

    Args:
        message: The notification message.

    Returns:
        A Content object with the notification.
    """
    formatted_message = f"[System: {message}]"

    return types.Content(
        role="user",
        parts=[types.Part(text=formatted_message)]
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
