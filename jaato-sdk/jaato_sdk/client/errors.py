"""Typed failures for session creation.

``create_session`` used to answer every failure with ``None``.  Five different
things produce that ``None`` -- the command never left the client, the daemon
refused it, the wait timed out, the connection dropped, or the client had no
inner client at all -- and a caller that receives it cannot tell which, cannot
tell whether a session now EXISTS on the daemon, and therefore cannot decide
whether re-sending is safe.

The five were measured returning ``None`` from the same call, four of them
indistinguishable even by wall-clock time.  The information was never missing:
these are five distinct branches inside the client.  It was DISCARDED at the
point of failure.

WHY EXCEPTIONS AND NOT A STATUS STRING.

``create_session``'s success value is the session id, so there is no room in
the return type for a status the way there is for a delivery verb (see
``shared.message_delivery``, which answers the same question with words
because it has no id to carry).  A companion out-parameter or a
``last_error`` attribute would be hidden state, wrong under concurrency, and
invisible to the two in-tree callers that DISCARD the return value entirely --
which is exactly how a create failure currently goes unnoticed.

WHY ``RuntimeError``.

Consumers already convert the ``None`` into a ``RuntimeError`` themselves, at
the call site, by hand: a survey of 18 repositories in this organisation found
that every handler wrapping ``create_session`` catches ``RuntimeError``,
because the line under the call reads ``if not sid: raise RuntimeError(...)``.
Inheriting from it means those handlers keep working through the change
instead of being bypassed by a fresh hierarchy.  The one place that would
otherwise have swallowed the new exception in a bare ``except Exception`` and
skipped unrelated work keeps its narrow ``except RuntimeError`` and behaves as
it does today.

THE AXIS THAT MATTERS IS ``may_exist``.

Not the mechanism -- the mechanism is what the message and ``cause`` name.
What a caller must branch on is whether a session may ALREADY EXIST on the
daemon, because ``session.new`` has no idempotency key: ``request_id`` is a
correlation token that is echoed back, never a dedupe key.  So a retry after
an unconfirmed create makes a SECOND session, holding a second runner
subprocess and a second pool slot.  A caller that must retry should look for
the session first (``list_sessions``) rather than create blindly.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "SessionCreateFailed",
    "SessionNotSent",
    "SessionRefused",
    "SessionNotConfirmed",
]


class SessionCreateFailed(RuntimeError):
    """Base: ``create_session`` did not produce a session.

    Subclasses say WHY, and — through :attr:`may_exist` — whether retrying is
    safe.  Catch this base to handle every creation failure; catch a subclass
    when the recovery differs.

    Attributes:
        cause: Machine-readable token naming the mechanism
            (``not_sent`` / ``refused`` / ``timeout`` / ``disconnect``).
            Stable, greppable, and safe to branch on; the human message is
            not.
        may_exist: Whether a session may have been created on the daemon
            despite this failure.  ``False`` means nothing was created and a
            retry cannot duplicate.  ``True`` means the outcome is genuinely
            unknown and a blind retry may create a SECOND session.
    """

    cause: str = "unknown"
    may_exist: bool = False

    def __init__(self, message: str) -> None:
        super().__init__(message)


class SessionNotSent(SessionCreateFailed):
    """The command never left this process.

    The socket write failed, or the client had no inner client to write
    through.  Nothing reached the daemon, so nothing was created.

    RETRY IS SAFE once the connection is restored, and pointless before then.

    This case used to burn the FULL timeout — 60 seconds by default — waiting
    for a reply to a command the client already knew it had not sent, then
    reported "timed out waiting for SessionInfoEvent", blaming the daemon for
    a local socket failure.  It now returns immediately.
    """

    cause = "not_sent"
    may_exist = False


class SessionRefused(SessionCreateFailed):
    """The daemon answered, and the answer was no.

    A profile or agent that does not exist, an invalid inline spec, a
    spawn-payload validation failure, an exhausted cascade budget, a provider
    that would not authenticate.  The daemon states the reason; it is carried
    here rather than guessed at.

    RETRY IS SAFE — nothing was created — but futile unless the request
    changes.

    Attributes:
        error_type: The daemon's own ``ErrorEvent.error_type``, or ``None``
            when it did not supply one.  Never inferred: a refusal whose type
            the daemon did not state must not be given a likely-looking one.
    """

    cause = "refused"
    may_exist = False

    def __init__(self, message: str, *, error_type: Optional[str] = None) -> None:
        super().__init__(message)
        self.error_type = error_type


class SessionNotConfirmed(SessionCreateFailed):
    """An answer was expected and never arrived.

    The wait timed out, or the connection dropped before the daemon answered.
    The command WAS sent, so the daemon may have created the session and only
    the confirmation was lost.

    RETRY MAY CREATE A SECOND SESSION.  ``session.new`` has no idempotency
    key, so a retry is a new create, not a resumption of this one — and each
    session holds a runner subprocess and a pool slot.  Prefer looking for the
    session (``list_sessions``) over creating another.

    This is the one failure where the correct action depends on something the
    caller cannot see, which is why it is a distinct type rather than a
    detail in a message.
    """

    #: Overwritten per-instance by ``__init__``; the class-level value is the
    #: same default so introspecting the CLASS does not read ``"unknown"``.
    cause = "timeout"
    may_exist = True

    def __init__(self, message: str, *, cause: str = "timeout") -> None:
        super().__init__(message)
        self.cause = cause
