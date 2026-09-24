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

**Optional transformers** (extension surface for redaction / pseudonymization
/ audit / content-filter consumers — see ``docs/design/daemon-extensions.md``
and ``project_backlog_pseudonymization_plugin_surface.md``):

* ``set_inbound_transformer(fn)`` — applied to every Message at ``append()``
  / ``replace()`` time before it lands in the container.  Use this for
  *write-side* transformation (e.g. pseudonymizing PII so the canonical
  history never holds raw values).
* ``set_raw_view_transformer(fn)`` — applied to every stored Message in
  the ``messages_raw`` property.  Use this for the *trusted-reader*
  view (e.g. un-pseudonymizing for the user-display path or for an
  audit logger that needs forensic access).

The two transformers are fully orthogonal — neither knows about the
other.  Consumers that need a coupled pair (inbound redact + raw-view
un-redact) wire them via closures over the consumer's own state
(e.g. a per-session lookup table).  The framework owns no plug-in
state beyond the transformer references themselves.
"""

import logging
from typing import Callable, List, Optional, TYPE_CHECKING

from jaato_sdk.plugins.model_provider.types import Message, Role

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Type alias for the per-Message transformer signature both extension
# points share.  Returning the same Message object is fine (identity is
# valid); returning a new Message lets the transformer rewrite parts.
MessageTransformer = Callable[[Message], Message]


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
        # Optional plug-in transformers.  Both default to None (identity);
        # set_*_transformer() registers them.  The container imposes no
        # constraints on what they do — the consumer is responsible for
        # consistency (e.g. a pseudonymization consumer should set both
        # inbound and raw-view at the same time, not one without the
        # other).
        self._inbound_transformer: Optional[MessageTransformer] = None
        self._raw_view_transformer: Optional[MessageTransformer] = None

    def set_inbound_transformer(
        self, fn: Optional[MessageTransformer]
    ) -> None:
        """Register a write-side transformer for ``append()`` / ``replace()``.

        The transformer runs on every Message immediately before it
        lands in the container.  Use ``None`` to clear (restores
        identity behaviour).

        Args:
            fn: Callable that takes a Message and returns the Message
                to actually store.  Returning the input unchanged is
                valid (identity).  Pass ``None`` to disable.
        """
        self._inbound_transformer = fn

    def set_raw_view_transformer(
        self, fn: Optional[MessageTransformer]
    ) -> None:
        """Register a read-side transformer for the ``messages_raw`` accessor.

        Trusted callers (e.g. the user-display swap-back path or an
        audit logger) reach into ``messages_raw`` and the transformer
        runs on each stored Message before it's returned.  Use ``None``
        to clear (``messages_raw`` then returns stored messages
        unchanged).

        Args:
            fn: Callable that takes a stored Message and returns the
                Message to actually surface to the trusted caller.
                Pass ``None`` to disable.
        """
        self._raw_view_transformer = fn

    def append(self, msg: 'Message') -> None:
        """Append a message to the history.

        Args:
            msg: The message to append.
        """
        if self._inbound_transformer is not None:
            msg = self._inbound_transformer(msg)
        self._messages.append(msg)
        self._dirty = True

    def replace(self, messages: List['Message']) -> None:
        """Replace the entire history (e.g. after GC or session restore).

        Args:
            messages: The new message list. A shallow copy is made to
                avoid aliasing the caller's list.
        """
        if self._inbound_transformer is not None:
            messages = [self._inbound_transformer(m) for m in messages]
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

    @property
    def messages_raw(self) -> List['Message']:
        """Return the trusted-caller view of the message list.

        When a raw-view transformer is registered (via
        :meth:`set_raw_view_transformer`), it runs on each stored
        Message before the result is returned to the caller.  The
        canonical use case is a pseudonymization consumer that
        un-redacts placeholders for the user-display path or an audit
        logger.

        When no transformer is registered, returns a copy of the stored
        messages (identical to :attr:`messages`).  Trusted callers
        should always use this accessor — it future-proofs against the
        framework gaining a redaction layer later without those callers
        needing to change.
        """
        if self._raw_view_transformer is None:
            return list(self._messages)
        return [self._raw_view_transformer(m) for m in self._messages]

    def rewrite_last_dropping_tool_use(self) -> Optional[str]:
        """Rewrite the last message to keep text parts and drop ``function_call`` parts.

        Used by the rewind-with-hint recovery path in
        :mod:`shared.jaato_session` when the provider returned a
        ``MAX_TOKENS``-truncated tool call: the model's narration
        text is worth preserving as an anchor for the corrective hint,
        but the half-serialized ``function_call`` part must go so the
        chat loop doesn't try to execute it.

        The last message is expected to be the just-appended assistant
        response.  When its role is NOT ``MODEL``, or no text parts
        survive the drop, we leave history untouched and return
        ``None`` — the caller treats that as "nothing to preserve" and
        falls through to the existing abnormal-termination handling.

        Returns:
            The concatenated text of the preserved parts when the
            rewrite succeeded, or ``None`` when no rewrite happened
            (empty history, wrong role, or no text to keep).
        """
        if not self._messages:
            return None

        last = self._messages[-1]
        if last.role != Role.MODEL:
            # Defensive: only assistant turns carry function_call parts
            # from the model.  If the history doesn't end with one,
            # something upstream is out of order — don't mutate.
            return None

        kept_parts = [p for p in last.parts if p.function_call is None]
        if not any(p.text for p in kept_parts):
            # No narration to preserve — signal caller to skip rewind.
            return None

        # Mutate in place via replacement.  We keep the same message_id
        # and model/provider so downstream consumers (GC, token ledger,
        # telemetry) continue to see a coherent assistant turn.
        rewritten = Message(
            role=last.role,
            parts=kept_parts,
            message_id=last.message_id,
            model=last.model,
            provider=last.provider,
        )
        self._messages[-1] = rewritten
        self._dirty = True

        return rewritten.text or ""

    def __len__(self) -> int:
        return len(self._messages)

    def __bool__(self) -> bool:
        return len(self._messages) > 0

    def __repr__(self) -> str:
        return f"SessionHistory(messages={len(self._messages)}, dirty={self._dirty})"
