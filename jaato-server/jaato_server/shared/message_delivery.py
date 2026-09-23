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

#: The session is loaded and live, but NOTHING WAS PUT IN FLIGHT -- there is
#: no server attached, no runner channel, the runner is too old to answer the
#: offer verb, or the drive that was supposed to start a turn failed.  A
#: transport fault, NOT a decision by the target, which is why it is not
#: spelled ``refused``.
#:
#: RETRY IS SAFE HERE, and that is the whole reason this is a separate word
#: from :data:`NOT_CONFIRMED`.  No offer reached the target, so a retry cannot
#: duplicate anything.  It is also unlikely to help until whatever is missing
#: is restored -- the daemon log names which of the four it was.
UNREACHABLE = "unreachable"

#: An offer WAS made and the answer was lost -- the offer RPC raised or timed
#: out.  The message may be sitting in the target's queue right now, or may
#: never have arrived; from here those are indistinguishable.
#:
#: RETRY MAY DUPLICATE.  This is the one delivery failure where the caller's
#: correct move depends on something the caller cannot see, so it gets its own
#: word rather than being folded into :data:`UNREACHABLE`.
#:
#: Splitting these was not cosmetic.  Both used to be ``unreachable``, and the
#: single prose reason attached to that word described THIS case -- "the
#: message may still have been enqueued and only the acknowledgement lost".
#: Applied to the four structural cases it was not vague, it was FALSE: it
#: warned a sender about a duplicate that could not exist, and a careful
#: sender therefore declined to re-send a message that had definitely never
#: arrived.  The word that was meant to prevent a wrong action caused one.
NOT_CONFIRMED = "not_confirmed"

#: The target is mid-turn and the caller asked NOT to add to its queue --
#: backpressure, requested via ``require_idle``.  Nothing was enqueued.
#:
#: Answered by the TARGET, not inferred from a daemon-side replica: a peer
#: that has drained its backlog and gone idle must not be refused for a
#: backlog it no longer has.
BUSY = "busy"

#: The statuses that mean the message WILL be acted on.
#:
#: Membership is the ONLY thing a caller must branch on to be correct.  The
#: failure words differentiate what to do NEXT (retry safe / retry may
#: duplicate / target gone / target dead), never whether it arrived -- so a
#: consumer that only knows this set stays correct when a new failure word is
#: added, which is why :data:`NOT_CONFIRMED` could be introduced without
#: touching a single caller.
#:
#: This set is the whole point of the vocabulary.  The failure states above
#: must never render as success, because a caller that assumes delivery and
#: is wrong gets a silent stall it cannot attribute -- the expensive
#: direction.  Anything not in here is a delivery that did not happen.
#:
#: :data:`QUEUED` is a GUARANTEE, not a prediction (#620).  It is answered by
#: the target session's own :meth:`JaatoSession.offer_message` under
#: ``_delivery_lock``, which is held across BOTH the check-and-enqueue and the
#: turn's ``_is_running = False`` flip.  So an offer that observed a running
#: turn finishes its enqueue before that flip, and the end-of-turn drain runs
#: after it -- the message is necessarily collected.
#:
#: Before #620 this said the opposite, and said it here, in the module
#: consumers read to learn what the word means.  The daemon used to decide
#: from its own ``_model_running``, a replica that clears ~30s later, so
#: ``queued`` could mean "queued behind a drain that already ran".  That is
#: fixed; the caveat that described it is gone rather than softened.
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
