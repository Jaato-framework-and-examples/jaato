"""``session.new``, timed phase by phase, up to the socket (#1452).

**Why.**  On 2026-09-30 two sessions were created in about 9 s and their
callers still gave up after 60 s: the confirmations could not be written,
because the daemon loop was stalled for 121 s.  The log stopped at
``Session created``, so it could not say when (or whether) the
confirmation left.  This module logs one line per phase, with elapsed
milliseconds since the request was read and since the previous phase:

=========================  ====================================================
phase                      where
=========================  ====================================================
received                   the transport read the ``session.new`` frame
handler_started            ``SessionManager.create_session`` began (the gap
                           from ``received`` is executor queueing)
runner_ready               a pool slot was claimed or a runner spawned
bootstrap_acked            the runner acknowledged ``session.bootstrap``
server_initialized         ``JaatoServer.initialize`` succeeded
workspace_monitor_started  the workspace monitor seeded its baseline (a walk
                           of the whole tree; #1553)
session_created            the session is registered and saved
answer_queued              the confirmation (or refusal) went to the client's
                           send queue
answer_written             the transport WROTE that frame to the socket -- the
                           phase the incident had no record of
=========================  ====================================================

Every line carries :data:`PHASE_TOKEN`, the request's correlation id and,
once known, the session id, so one ``grep`` reconstructs a create.

**Keying.**  Clocks are keyed by the ``session.new`` correlation id
(``request_id``), which is what the answer frame carries back to the
socket writer.  A request with no id (an old client) is still timed on the
creating thread, and its ``answer_written`` phase cannot be matched, which
the ``answer_queued`` line says.  Open clocks are bounded
(:data:`MAX_OPEN`, oldest dropped), so a caller that never gets an answer
cannot grow the table.

**The creating thread.**  ``create_session`` runs start to finish on one
executor worker, and the runner spawn and bootstrap run on it too (inside
``server.initialize``).  :func:`begin` binds the request to that thread, so
:func:`mark` needs no id and ``runner_spawn`` needs no new parameter.
Called on any other thread, :func:`mark` does nothing.

Never raises: timing must not be able to fail a create.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Greppable token on every phase line.
PHASE_TOKEN = "SESSION_NEW_PHASE"
#: Most open (unanswered) clocks kept.
MAX_OPEN = 256

_OPEN: "OrderedDict[str, _Clock]" = OrderedDict()
_OPEN_LOCK = threading.Lock()
_THREAD = threading.local()


@dataclass
class _Clock:
    """One ``session.new`` being timed.  ``last`` is the previous phase."""

    request_id: Optional[str]
    started: float
    last: float
    session_id: Optional[str] = None


def _log(clock: _Clock, phase: str, detail: str = "") -> None:
    now = time.monotonic()
    logger.info(
        "%s request=%s session=%s phase=%s t=+%.0fms step=%.0fms%s",
        PHASE_TOKEN, clock.request_id or "-", clock.session_id or "-", phase,
        (now - clock.started) * 1000.0, (now - clock.last) * 1000.0,
        f" {detail}" if detail else "",
    )
    clock.last = now


def _store(clock: _Clock) -> None:
    if clock.request_id is None:
        return
    with _OPEN_LOCK:
        _OPEN[clock.request_id] = clock
        _OPEN.move_to_end(clock.request_id)
        while len(_OPEN) > MAX_OPEN:
            _OPEN.popitem(last=False)


def note_received(request_id: Optional[str]) -> None:
    """The transport read a ``session.new`` frame carrying ``request_id``."""
    if not request_id:
        return
    now = time.monotonic()
    clock = _Clock(request_id=request_id, started=now, last=now)
    _store(clock)
    _log(clock, "received")


def note_request(event: Any) -> None:
    """Call for every request a transport reads; times ``session.new`` only.

    Duck-typed (a ``CommandRequest`` whose ``command`` is ``session.new``),
    so a transport makes one unconditional call and gains no branch.
    """
    command = getattr(event, "command", None)
    if not isinstance(command, str) or command.lower() != "session.new":
        return
    payload = getattr(event, "payload", None)
    if isinstance(payload, dict):
        note_received(payload.get("request_id"))


def begin(request_id: Optional[str]) -> None:
    """Bind this thread to a ``session.new`` and log ``handler_started``.

    Continues the clock :func:`note_received` opened when there is one, so
    the step shows how long the request waited for a worker thread.
    """
    try:
        clock = None
        if request_id:
            with _OPEN_LOCK:
                clock = _OPEN.get(request_id)
        if clock is None:
            now = time.monotonic()
            clock = _Clock(request_id=request_id, started=now, last=now)
            _store(clock)
        _THREAD.clock = clock
        _log(clock, "handler_started")
    except Exception:  # noqa: BLE001 -- timing must not fail a create
        logger.debug("session.new timing: begin failed", exc_info=True)


def end() -> None:
    """Unbind this thread (the clock stays open for ``answer_written``)."""
    _THREAD.clock = None


def mark(phase: str, *, session_id: Optional[str] = None,
         detail: str = "") -> None:
    """Log ``phase`` for the ``session.new`` running on this thread."""
    clock = getattr(_THREAD, "clock", None)
    if clock is None:
        return
    if session_id:
        clock.session_id = session_id
    _log(clock, phase, detail)


def note_written(event: Any, transport: str) -> None:
    """A transport wrote ``event`` to a client's socket.

    Closes the clock when ``event`` is the answer to a timed
    ``session.new``: any event whose ``request_id`` has an open clock.
    Called for every written frame, so it must stay a dict lookup.
    """
    request_id = getattr(event, "request_id", None)
    if not request_id:
        return
    with _OPEN_LOCK:
        clock = _OPEN.pop(request_id, None)
    if clock is None:
        return
    _log(clock, "answer_written",
         f"event={type(event).__name__} transport={transport}")
