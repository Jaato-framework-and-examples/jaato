"""SessionHistory - Canonical conversation history owned by the session.

The session is the sole owner of conversation history. Providers are stateless
and receive messages as parameters to ``complete()`` on each call.

Lifecycle:
    1. Created empty in ``JaatoSession.__init__``
    2. Populated via ``append()`` after each user message and model response
    3. Updated via ``replace()`` after GC operations or session restore
    4. Read via ``messages`` property by session code (budget tracking,
       turn boundaries, GC input, persistence)

State transitions:
    - ``append(msg)`` → adds a message, sets dirty flag
    - ``replace(msgs)`` → bulk replacement (after GC or restore), sets dirty flag
    - ``clear()`` → empties history and clears dirty flag

The ``_dirty`` flag tracks whether mutations have occurred since the last
``clear()`` or ``replace()`` call. Used by persistence logic to decide
whether to write.
"""

import logging
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from jaato_sdk.plugins.model_provider.types import Message

logger = logging.getLogger(__name__)


class SessionHistory:
    """Canonical conversation history owned by the session.

    Wraps ``List[Message]`` with mutation tracking. The session is the sole
    history owner; providers receive messages as parameters to ``complete()``.

    Attributes:
        _messages: The canonical message list.
        _dirty: Whether mutations have occurred since the last clear/replace.
    """

    def __init__(self) -> None:
        self._messages: List['Message'] = []
        self._dirty: bool = False

    def append(self, msg: 'Message') -> None:
        """Append a message to the history.

        Args:
            msg: The message to append.
        """
        self._messages.append(msg)
        self._dirty = True

    def replace(self, messages: List['Message']) -> None:
        """Replace the entire history (e.g. after GC or session restore).

        Args:
            messages: The new message list. A shallow copy is made to
                avoid aliasing the caller's list.
        """
        self._messages = list(messages)
        self._dirty = True

    def clear(self) -> None:
        """Clear all messages and reset the dirty flag."""
        self._messages = []
        self._dirty = False

    @property
    def messages(self) -> List['Message']:
        """Return a shallow copy of the message list.

        Returns a copy to prevent external mutation from bypassing the
        dirty-tracking mechanism.
        """
        return list(self._messages)

    @property
    def dirty(self) -> bool:
        """Whether mutations have occurred since last clear/replace."""
        return self._dirty

    def pop_last(self) -> Optional['Message']:
        """Remove and return the last message, or None if empty.

        Used for rollback when a provider call fails after the session
        has already appended a user/tool message.
        """
        if self._messages:
            msg = self._messages.pop()
            self._dirty = True
            return msg
        return None

    @property
    def last(self) -> Optional['Message']:
        """Return the last message without removing it, or None if empty."""
        return self._messages[-1] if self._messages else None

    @property
    def messages_ref(self) -> List['Message']:
        """Return a direct reference to the internal message list.

        Unlike ``messages`` (which returns a copy), this gives direct access
        for performance-critical reads where the caller guarantees it will
        not mutate the list. Used to avoid copying the full history on every
        ``complete()`` call.

        .. warning:: Do NOT mutate the returned list. Use ``append()``,
           ``replace()``, or ``pop_last()`` for mutations so dirty-tracking
           is maintained.
        """
        return self._messages

    def __len__(self) -> int:
        return len(self._messages)

    def __bool__(self) -> bool:
        return len(self._messages) > 0

    def __repr__(self) -> str:
        return f"SessionHistory(messages={len(self._messages)}, dirty={self._dirty})"
