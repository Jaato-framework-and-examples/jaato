"""MessageQueue - Priority-aware message queue for agent communication.

A double-linked list implementation that supports:
- FIFO ordering within priority groups
- O(1) removal from any position (for mid-turn parent message processing)
- Separate iteration by source type (parent vs child)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Optional, Iterator


class SourceType(Enum):
    """Type of agent that sent the message."""
    PARENT = "parent"   # Controller messages - high priority, mid-turn processing
    CHILD = "child"     # Status updates - lower priority, process when idle
    USER = "user"       # User input - treated like parent (high priority)
    SYSTEM = "system"   # System messages - treated like parent (high priority)


@dataclass
class QueuedMessage:
    """A message in the queue with metadata about its source."""
    text: str
    source_id: str           # Who sent it (e.g., "main", "subagent_researcher")
    source_type: SourceType  # Parent, child, user, or system
    timestamp: datetime = field(default_factory=datetime.now)

    # Internal linked list pointers (managed by MessageQueue)
    _prev: Optional['QueuedMessage'] = field(default=None, repr=False)
    _next: Optional['QueuedMessage'] = field(default=None, repr=False)


class MessageQueue:
    """Double-linked list queue with priority-based message retrieval.

    Supports two processing modes:
    1. Mid-turn: Pop parent/user/system messages (high priority) from any position
    2. Idle: Pop child messages (status updates) in FIFO order

    The double-linked structure allows O(1) removal from the middle when
    processing parent messages mid-turn, without disrupting FIFO order
    for other messages.

    Thread-safe for concurrent access.

    Example:
        queue = MessageQueue()

        # Inject messages from different sources
        queue.put("guidance", source_id="main", source_type=SourceType.PARENT)
        queue.put("status update", source_id="subagent1", source_type=SourceType.CHILD)

        # Mid-turn: process parent messages immediately
        if queue.has_parent_messages():
            msg = queue.pop_first_parent_message()
            process(msg)

        # When idle: drain child messages
        while queue.has_child_messages():
            msg = queue.pop_first_child_message()
            process(msg)
    """

    def __init__(self):
        """Initialize an empty queue."""
        self._head: Optional[QueuedMessage] = None
        self._tail: Optional[QueuedMessage] = None
        self._size: int = 0
        self._lock = Lock()

    def put(self, text: str, source_id: str, source_type: SourceType) -> QueuedMessage:
        """Add a message to the end of the queue.

        Args:
            text: The message content.
            source_id: ID of the agent/entity that sent it.
            source_type: Type of source (parent, child, user, system).

        Returns:
            The created QueuedMessage (for reference if needed).
        """
        msg = QueuedMessage(
            text=text,
            source_id=source_id,
            source_type=source_type,
            timestamp=datetime.now()
        )

        with self._lock:
            if self._tail is None:
                # Empty list
                self._head = msg
                self._tail = msg
            else:
                # Append to tail
                msg._prev = self._tail
                self._tail._next = msg
                self._tail = msg

            self._size += 1

        return msg

    def _remove(self, msg: QueuedMessage) -> None:
        """Remove a message from anywhere in the list (internal, must hold lock)."""
        if msg._prev:
            msg._prev._next = msg._next
        else:
            # Was head
            self._head = msg._next

        if msg._next:
            msg._next._prev = msg._prev
        else:
            # Was tail
            self._tail = msg._prev

        msg._prev = None
        msg._next = None
        self._size -= 1

    def pop_first_parent_message(self) -> Optional[QueuedMessage]:
        """Find and remove the oldest parent/user/system message.

        Used for mid-turn processing where parent messages take priority.
        Searches the entire queue to find the oldest high-priority message.

        Returns:
            The oldest parent/user/system message, or None if none exist.
        """
        with self._lock:
            current = self._head
            while current:
                if current.source_type in (SourceType.PARENT, SourceType.USER, SourceType.SYSTEM):
                    self._remove(current)
                    return current
                current = current._next
        return None

    def pop_first_child_message(self) -> Optional[QueuedMessage]:
        """Find and remove the oldest child message.

        Used for idle processing where child status updates are handled.

        Returns:
            The oldest child message, or None if none exist.
        """
        with self._lock:
            current = self._head
            while current:
                if current.source_type == SourceType.CHILD:
                    self._remove(current)
                    return current
                current = current._next
        return None

    def pop_any(self) -> Optional[QueuedMessage]:
        """Pop the first message regardless of type (FIFO).

        Returns:
            The oldest message, or None if queue is empty.
        """
        with self._lock:
            if self._head is None:
                return None
            msg = self._head
            self._remove(msg)
            return msg

    def has_parent_messages(self) -> bool:
        """Check if there are any parent/user/system messages pending.

        Returns:
            True if at least one high-priority message exists.
        """
        with self._lock:
            current = self._head
            while current:
                if current.source_type in (SourceType.PARENT, SourceType.USER, SourceType.SYSTEM):
                    return True
                current = current._next
        return False

    def has_child_messages(self) -> bool:
        """Check if there are any child messages pending.

        Returns:
            True if at least one child message exists.
        """
        with self._lock:
            current = self._head
            while current:
                if current.source_type == SourceType.CHILD:
                    return True
                current = current._next
        return False

    def is_empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            True if no messages in queue.
        """
        with self._lock:
            return self._size == 0

    def __len__(self) -> int:
        """Return the number of messages in the queue."""
        with self._lock:
            return self._size

    def peek_all(self) -> list[QueuedMessage]:
        """Return a list of all messages without removing them.

        Useful for debugging and inspection.

        Returns:
            List of all messages in FIFO order.
        """
        result = []
        with self._lock:
            current = self._head
            while current:
                result.append(current)
                current = current._next
        return result

    def clear(self) -> int:
        """Remove all messages from the queue.

        Returns:
            Number of messages that were cleared.
        """
        with self._lock:
            count = self._size
            self._head = None
            self._tail = None
            self._size = 0
            return count
