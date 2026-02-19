"""Utility functions for Context Garbage Collection.

Provides helpers for turn splitting, token estimation, and history manipulation.
"""

from dataclasses import dataclass
from typing import List, Optional

from jaato_sdk.plugins.model_provider.types import Message, Part, Role


@dataclass
class Turn:
    """Represents a conversation turn (user message + model response(s)).

    A turn typically consists of:
    - One user Message (role=USER)
    - One or more model Message objects (role=MODEL)
    - Possibly function response Message objects (role=USER with function_response parts,
      or role=TOOL for Anthropic provider)
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

        # Role.TOOL messages are always function responses (Anthropic provider uses this)
        if message.role == Role.TOOL:
            is_function_response = True
        elif is_user_message and message.parts:
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
                if fr.result:
                    total_chars += len(str(fr.result))

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


def ensure_tool_call_integrity(
    history: List[Message],
    trace_fn=None,
) -> List[Message]:
    """Validate and repair tool_use/tool_result pairing after GC.

    GC may remove individual messages from history, breaking the mandatory
    pairing between MODEL messages with function_call parts (tool_use) and
    TOOL/USER messages with function_response parts (tool_result). This
    function removes orphaned messages to restore a valid history that
    providers can accept.

    Handles two cases:
    1. Orphaned tool_results: A TOOL message references call_ids not present
       in any preceding MODEL message's function_calls.
    2. Unpaired tool_uses: A MODEL message has function_calls but no matching
       tool_result follows before the next USER or MODEL message.

    Args:
        history: Conversation history (potentially with broken pairs).
        trace_fn: Optional callable for trace logging, signature: (str) -> None.

    Returns:
        History with orphaned tool_use/tool_result messages removed.
    """
    if not history:
        return history

    def _trace(msg: str) -> None:
        if trace_fn:
            trace_fn(msg)

    # --- Pass 1: Remove orphaned tool_result messages ---
    # A tool_result is orphaned if its call_id doesn't appear in any preceding
    # MODEL message's function_calls.
    available_call_ids: set = set()
    pass1_result: List[Message] = []
    orphaned_tool_result_ids: set = set()

    for msg in history:
        if msg.role == Role.MODEL:
            # Collect function_call IDs from this MODEL message
            for p in msg.parts:
                if p.function_call and p.function_call.id:
                    available_call_ids.add(p.function_call.id)
            pass1_result.append(msg)

        elif msg.role == Role.TOOL or (
            msg.role == Role.USER
            and msg.parts
            and any(p.function_response is not None for p in msg.parts)
        ):
            # This is a tool result message - check if its call_ids are valid
            msg_call_ids = set()
            for p in msg.parts:
                if p.function_response and p.function_response.call_id:
                    msg_call_ids.add(p.function_response.call_id)

            valid_ids = msg_call_ids & available_call_ids
            if not valid_ids and msg_call_ids:
                # ALL call_ids in this message are orphaned - remove it
                _trace(
                    f"ensure_tool_call_integrity: removing orphaned tool_result "
                    f"message (call_ids={msg_call_ids})"
                )
                orphaned_tool_result_ids.update(msg_call_ids)
            else:
                pass1_result.append(msg)
                # Resolve matched call_ids (they now have results)
                available_call_ids -= valid_ids
        else:
            pass1_result.append(msg)

    # --- Pass 2: Remove MODEL messages with unresolved tool_calls ---
    # A MODEL message with function_calls is unpaired if no matching
    # tool_result follows before the next non-tool message or end of history.
    # We scan forward and track pending tool_call_ids.
    result: List[Message] = []
    pending_tool_use_ids: set = set()
    pending_model_idx: Optional[int] = None

    for msg in pass1_result:
        if msg.role == Role.MODEL:
            has_tool_calls = any(
                p.function_call is not None for p in msg.parts
            )
            if has_tool_calls:
                # If there's a previous MODEL with unresolved tool_calls,
                # that one is unpaired - remove it
                if pending_tool_use_ids and pending_model_idx is not None:
                    removed_msg = result[pending_model_idx]
                    _trace(
                        f"ensure_tool_call_integrity: removing unpaired tool_use "
                        f"MODEL message (pending_ids={pending_tool_use_ids})"
                    )
                    result = result[:pending_model_idx] + result[pending_model_idx + 1:]

                # Track this MODEL message and its tool_call IDs
                pending_model_idx = len(result)
                pending_tool_use_ids = set()
                for p in msg.parts:
                    if p.function_call and p.function_call.id:
                        pending_tool_use_ids.add(p.function_call.id)

            result.append(msg)

        elif msg.role == Role.TOOL or (
            msg.role == Role.USER
            and msg.parts
            and any(p.function_response is not None for p in msg.parts)
        ):
            # Resolve matching tool_call IDs
            for p in msg.parts:
                if p.function_response and p.function_response.call_id:
                    pending_tool_use_ids.discard(p.function_response.call_id)

            if not pending_tool_use_ids:
                pending_model_idx = None

            result.append(msg)

        elif msg.role == Role.USER:
            # User text message - any pending tool_calls are unpaired
            if pending_tool_use_ids and pending_model_idx is not None:
                _trace(
                    f"ensure_tool_call_integrity: removing unpaired tool_use "
                    f"MODEL message before USER message "
                    f"(pending_ids={pending_tool_use_ids})"
                )
                result = result[:pending_model_idx] + result[pending_model_idx + 1:]
                pending_tool_use_ids.clear()
                pending_model_idx = None

            result.append(msg)
        else:
            result.append(msg)

    # Final check: if history ends with unpaired tool_calls, remove the MODEL msg
    if pending_tool_use_ids and pending_model_idx is not None:
        _trace(
            f"ensure_tool_call_integrity: removing unpaired tool_use "
            f"MODEL message at end of history "
            f"(pending_ids={pending_tool_use_ids})"
        )
        result = result[:pending_model_idx] + result[pending_model_idx + 1:]

    removed_count = len(history) - len(result)
    if removed_count > 0:
        _trace(
            f"ensure_tool_call_integrity: removed {removed_count} message(s) "
            f"to restore tool_call pairing"
        )

    return result


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
