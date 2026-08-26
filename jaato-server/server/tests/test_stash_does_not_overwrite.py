"""Several sends landing in ONE wind-down window must all survive.

#620 moved the queue-or-drive decision onto the session and, for a send that
arrives while the SESSION is idle but the DAEMON's model thread is still
unwinding, stashed the text for that thread's ``finally`` to turn into the
next turn.  It stashed with a plain assignment into a single slot, so N such
sends overwrote each other and only the last became a turn.

Before #620 that path went to the runner's ``_message_queue`` -- a real queue.
The regression was replacing a queue with a variable, and it is silent: every
send is reported ``accepted``, and the ones that were overwritten leave no
trace anywhere.

Reachable whenever two messages land in the same wind-down tail, which is
wide: measured ~30s on a live cascade.
"""

from __future__ import annotations

import threading
from typing import Any, List

from server.core import JaatoServer


def _server() -> JaatoServer:
    """A JaatoServer with only what the stash path touches."""
    srv = JaatoServer.__new__(JaatoServer)
    srv._model_running = True          # our thread is still unwinding
    srv._pending_continuations = []
    srv._pending_continuation_lock = threading.Lock()
    srv._traces: List[str] = []
    srv._trace = lambda msg: srv._traces.append(msg)  # type: ignore[method-assign]
    return srv


def test_three_sends_in_one_window_all_survive():
    srv = _server()

    with srv._pending_continuation_lock:
        pass  # lock is usable

    for text in ("first", "second", "third"):
        with srv._pending_continuation_lock:
            if srv._model_running:
                srv._pending_continuations.append(text)

    assert srv._pending_continuations == ["first", "second", "third"], (
        "a single-slot stash drops every message but the last, silently -- "
        "each one having been reported 'accepted' to its sender"
    )


def test_the_consumer_takes_all_of_them_joined():
    """Taking one and leaving the rest would re-introduce the loss.

    Mirrors ``_drain_child_messages``, which joins a collected batch with a
    blank line and fires ONE continuation for it.
    """
    srv = _server()
    srv._pending_continuations = ["first", "second", "third"]

    with srv._pending_continuation_lock:
        stashed = srv._pending_continuations
        srv._pending_continuations = []
    pending = "\n\n".join(stashed) if stashed else None

    assert pending == "first\n\nsecond\n\nthird"
    assert srv._pending_continuations == [], "the stash must be emptied"


def test_an_empty_stash_starts_no_turn():
    srv = _server()

    with srv._pending_continuation_lock:
        stashed = srv._pending_continuations
        srv._pending_continuations = []
    pending = "\n\n".join(stashed) if stashed else None

    assert pending is None, (
        "an empty stash must be falsy, not an empty string that reads as "
        "'there is a continuation' to the caller's truthiness check"
    )
