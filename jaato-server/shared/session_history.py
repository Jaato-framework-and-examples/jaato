"""SessionHistory - Canonical conversation history owned by the session.

Phase 1 of the session-owned history migration. This wrapper makes history
ownership explicit: the session holds the canonical copy, synced
bidirectionally with the provider during the transition period.

After Phase 4 (legacy protocol removal), the provider sync is removed and
providers receive messages as parameters to a stateless ``complete()`` method.

Lifecycle:
    1. Created empty in ``JaatoSession.__init__``
    2. Populated via ``sync_from_provider()`` after each provider operation
       that mutates history (send_message, send_tool_results, etc.)
    3. Updated via ``replace()`` after GC operations
    4. Read via ``messages`` property by session code (budget tracking,
       turn boundaries, GC input, persistence)

State transitions:
    - ``append(msg)`` → adds a message, sets dirty flag
    - ``replace(msgs)`` → bulk replacement (after GC or restore), sets dirty flag
    - ``sync_from_provider(provider)`` → reads provider's history as the
      new canonical state, clears dirty flag
    - ``clear()`` → empties history and clears dirty flag

The ``_dirty`` flag tracks whether session-side mutations have occurred
that haven't been synced back to the provider. During Phase 1 this flag
is informational only — the provider is always the authoritative source
after its own operations. In later phases, the dirty flag will drive
provider sync decisions.
"""

import logging
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from jaato_sdk.plugins.model_provider.types import Message

logger = logging.getLogger(__name__)


class SessionHistory:
    """Canonical conversation history owned by the session.

    Wraps ``List[Message]`` with mutation tracking. During the Phase 1
    migration, changes are synced from the provider after each provider
    operation. After Phase 4, the provider sync is removed and the session
    becomes the sole history owner.

    Attributes:
        _messages: The canonical message list.
        _dirty: Whether session-side mutations have occurred since
            the last provider sync.
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
        """Whether session-side mutations have occurred since last sync."""
        return self._dirty

    def sync_from_provider(self, provider: object) -> None:
        """Sync history from the provider's authoritative copy.

        During Phase 1, the provider remains the authoritative source of
        history after its own operations (send_message, send_tool_results,
        create_session). This method reads the provider's history and
        replaces the session's copy.

        Args:
            provider: A ModelProviderPlugin instance with ``get_history()``.
        """
        get_history = getattr(provider, 'get_history', None)
        if get_history is None:
            logger.debug("SessionHistory.sync_from_provider: provider has no get_history()")
            return
        self._messages = list(get_history())
        self._dirty = False

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
        not mutate the list. Used by the stateless provider path to avoid
        copying the full history on every ``complete()`` call.

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
