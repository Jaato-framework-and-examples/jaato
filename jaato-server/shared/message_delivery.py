"""The ONE decision every message sender makes: queue it, or drive a turn.

Three senders reach a session that may be working or resting — a human via
``SendMessageRequest``, a parent via ``send_to_subagent``, a cascade peer via
``send_to_sibling`` — and all three face the same fork:

    target BUSY  -> queue on the sender's tier; the target picks it up at a
                    yield point (high-priority tiers) or when its turn ends
                    (idle-only tiers)
    target IDLE  -> DRIVE a turn; queueing here delivers nothing

**Why this is a module and not a comment.**  It was written three times.  Two
copies were right and the third — ``send_to_sibling`` — was not: it queued
into an idle peer and reported ``accepted``, which promises a turn started.
Nothing started.  The message sat in a tier that had no drainer and was
discarded when the session unloaded.

The reason the third copy got it wrong is worth keeping, because it is not
obvious from the call site: ``JaatoSession.inject_prompt`` DOES start a turn
on an idle session — but only when ``_on_continuation_needed`` is set, and
that callback is installed only for the DURATION of a ``session.send_message``
RPC (``server/runner/rpc.py`` installs it at entry and restores it in the
finally).  So "idle" and "able to start a turn on its own" are DIFFERENT
PROPERTIES, and a genuinely idle peer that nobody is driving has the first and
not the second.  Reading one as the other is what produced a receipt that
lied.

Cloning a path this similar means each copy has to rediscover that.  Sharing
the decision means the next sender inherits it.
"""

from __future__ import annotations

from typing import Callable

#: Delivered by queueing: the target was mid-turn.  It will be picked up
#: without another send — mid-turn for high-priority tiers, at the turn
#: boundary for idle-only ones.  NOT a claim that it has been read.
QUEUED = "queued"

#: Delivered by driving: the target was idle, so a turn was started on it.
#: NOT a claim that the target agreed, understood, or acted.
ACCEPTED = "accepted"

#: The target is loaded but TERMINAL -- its model thread ended with an error
#: or an exhausted budget and it will run no further turns.  Reported by the
#: DAEMON from the target's own terminal stamp, never inferred by the caller
#: from silence: a slow target and a dead one produce the same nothing, so a
#: caller that infers cannot be wrong-and-know-it.
TERMINATED = "terminated"

#: No session with that id is loaded.  Distinct from :data:`TERMINATED` on
#: purpose -- "gone" and "dead but present" are different situations for a
#: driver, and collapsing them is what makes an absence claim unfalsifiable.
NO_SESSION = "no_session"

#: The session is loaded and live, but the delivery mechanism failed -- no
#: runner channel, or the forward raised.  A transport fault, NOT a decision
#: by the target, which is why it is not spelled ``refused``.
UNREACHABLE = "unreachable"

#: The statuses that mean the message WILL be acted on.
#:
#: This set is the whole point of the vocabulary.  The failure states above
#: must never render as success, because a caller that assumes delivery and
#: is wrong gets a silent stall it cannot attribute -- the expensive
#: direction.  Anything not in here is a delivery that did not happen.
#:
#: Note what :data:`QUEUED` does and does not promise.  It promises a drainer
#: exists (the target is mid-turn, and a turn drains its queue at every yield
#: point and again when it ends).  It does NOT yet promise the drainer will
#: still be there a millisecond later: the runner's final drain runs strictly
#: BEFORE the daemon's busy flag clears, so a message routed into that tail is
#: queued against a turn that will never drain again.  Closing that requires
#: moving the decision to the runner, which owns the authoritative flag.
DELIVERED = frozenset({ACCEPTED, QUEUED})


def deliver(
    *,
    is_busy: Callable[[], bool],
    queue: Callable[[], object],
    drive: Callable[[], object],
) -> str:
    """Route one message and return :data:`QUEUED` or :data:`ACCEPTED`.

    The caller supplies the two mechanisms because they differ by tier —
    in-process (``session.inject_prompt`` / ``session.send_message``) for a
    parent reaching its own child, daemon-to-runner RPC
    (``inject_prompt_to_session`` / ``send_message_to_session``) for a peer
    reaching a sibling.  What does NOT differ, and therefore lives here, is
    WHICH of them to use.

    ``is_busy`` is called once and its answer decides the branch.  A target
    that becomes busy immediately afterwards is fine: the drive path goes
    through the same request gate a client send does, which queues rather
    than interleaving.

    Raises whatever ``queue``/``drive`` raise — a delivery that failed must
    not be reported as one that succeeded.
    """
    if is_busy():
        queue()
        return QUEUED
    drive()
    return ACCEPTED
