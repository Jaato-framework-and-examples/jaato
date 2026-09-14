"""High-level convenience facade over the SDK session clients.

The facade is **client-agnostic**: the same :class:`Session` (``ask`` /
``complete`` / ``stream``) rides on every client that implements the small
session contract —

* :class:`IPCClient` and :class:`IPCRecoveryClient` — talk to a running daemon
  over IPC (the latter adds auto-reconnect);
* :class:`WSClient` and :class:`WSRecoveryClient` — talk to a (typically
  **remote**) daemon over WebSocket (``ws://`` / ``wss://``); the recovery
  variant adds auto-reconnect, mirroring the IPC pair. Needs the
  ``jaato-sdk[ws]`` extra;
* ``jaato.InProcessClient`` — runs the runtime **embedded** in your own process,
  no daemon / runner / socket.

The transport-agnostic ``jaato.session(mode="in_process"|"ipc"|"ws", ...)``
entry picks one, so the same facade code runs across transports with the mode
the only variable. The daemon transports (``ipc`` / ``ws``) also accept
``recovery=True`` to return the auto-reconnect client, plus ``ssl=`` / ``ca=``
on ``mode="ws"`` for self-signed / dev ``wss://`` certificates. The Python
``WSClient`` / ``WSRecoveryClient`` back the WS mode directly — Python WS access
is no longer raw frames.

The SDK's low-level surface is event-loop primitives (``subscribe`` /
``send_message`` / ``events``).  The common path — open a session, ask, get
the answer — costs ~10 lines of ``asyncio.Event`` + subscribe + ``done.wait()``
plumbing, and that recipe is subtle enough that the canonical scaffold template
once shipped an infinite hang (waiting on ``SESSION_TERMINATED`` only, which a
plain turn never emits — jaato PR #399).

This module owns that recipe so user code can't reproduce it (or its hangs):

    async with IPCClient.session(profile="researcher", agent="pirate") as s:
        print(await s.ask("Research tide pools."))

Owning it means ONE of it.  ``ask`` / ``stream`` / ``complete`` all settle
through :class:`_TurnWatch`, which is what the facade had not done: ``complete``
learned the daemon's settle rule in #767 and the two turn verbs kept a simpler
one, so a turn that its budget ceiling had ENDED came back through ``ask`` as
an ordinary reply -- and on a cascade-stamped session the next call never
returned at all (jaato #1007).  A second copy of a settle rule is how a hang
gets in here; a second copy is what there is now nowhere to put.

It is purely additive sugar over existing methods — every client method is
untouched, and both config styles are preserved because ``session(...)``
forwards ``create_session``'s parameters unchanged (``profile`` as a name =
declarative, as a dict = programmatic; ``agent`` composes with either).

See ``docs/design/sdk-convenience-layer.md`` and
``docs/design/in-process-facade.md``.
"""

from __future__ import annotations

import asyncio
from typing import (Any, AsyncIterator, Callable, Collection, Dict,
                    Optional)

from ..events import ClientType, EventType


#: Seconds ``Session.complete`` will wait for the ``AGENT_STATUS_CHANGED`` that
#: confirms or withdraws a turn's proposed terminus, before settling on the
#: turn anyway.
#:
#: The daemon emits that event from the same thread that emitted the turn
#: event, microseconds later, so this is never reached in a healthy session --
#: it exists for the one where the daemon stops talking mid-sequence (a lost
#: connection between a turn and its status event, or a daemon that dies
#: there).  Without it that session waits forever, and this module has shipped
#: an infinite hang before (PR #399).  Deliberately generous: tripping it early
#: costs the accuracy #767 bought back, so it should only ever fire when the
#: alternative is not returning at all.
SETTLE_GRACE = 10.0

#: Seconds a TURN-span call (``Session.ask`` / ``Session.stream``) will wait,
#: after a turn proposes a terminus, for the daemon to say whether that turn
#: also ended the SESSION -- before settling on the turn anyway.
#:
#: Separate from :data:`SETTLE_GRACE` because the two bound different
#: questions.  ``SETTLE_GRACE`` asks *is this agent going to take another
#: turn?* -- a question about the AGENT, which may legitimately pause for a
#: long time.  This asks *did the daemon emit a terminal in the same breath as
#: this turn event?* -- a question about ONE emit sequence on ONE daemon
#: thread, answered in milliseconds or not at all.  Measured on the jaato #1007
#: repro: the terminal follows its turn event by 2-3 ms and the confirming
#: status event by 10-18 ms, so one second is ~50x the signal it waits for.
#:
#: EXPIRY COSTS INFORMATION, NEVER CORRECTNESS.  When it fires, the call
#: settles on the turn and returns the turn's text -- exactly the pre-#1007
#: behaviour.  So a daemon that never emits the confirmation degrades to what
#: it always did, one second later, rather than hanging or lying.
TERMINUS_GRACE = 1.0

#: Terminal reasons a TURN verb returns on rather than raising
#: :class:`SessionEnded` for.  The session is over either way -- that is on
#: :attr:`Session.terminus` in every case -- and what this set decides is
#: whether it is over because something CUT THE TURN SHORT.
#:
#: ``natural`` is the agent finishing, which is not a failure of anything: a
#: scaffolded ``client`` pointed at a completion-gated profile drives it with
#: ``ask`` and gets exactly this, and must exit 0
#: (``shared/scaffold/tests/test_client_template_completion_wait.py``).
#: ``client_request`` and ``stopped`` are the caller's own doing, so telling
#: it by exception is telling it what it already knows.
#:
#: Everything else raises, INCLUDING A REASON THAT DOES NOT EXIST YET.  The
#: list is an allow-list rather than a deny-list because this whole issue is
#: a new reason arriving and going unconsidered: ``budget_exhausted`` was the
#: fifth, ``_apply_default_cascade_policy``'s docstring still said there were
#: four, and a turn that a ceiling had cut short came back as ``''``.  A
#: deny-list would let the sixth do it again.
CLEAN_TERMINAL_REASONS = frozenset({"natural", "client_request", "stopped"})


class AgentError(Exception):
    """A turn ended in error (``SESSION_TERMINATED(reason="error")`` /
    ``AgentErrorEvent``).  Carries the daemon's ``error_type`` /
    ``error_summary`` so callers can branch without parsing strings."""

    def __init__(self, error_type: Optional[str], error_summary: Optional[str]):
        self.error_type = error_type
        self.error_summary = error_summary
        super().__init__(f"{error_type or 'AgentError'}: {error_summary or ''}".rstrip(": "))


class Terminus:
    """How one ``ask`` / ``stream`` / ``complete`` call ended.

    Recorded on :attr:`Session.terminus` by every one of the three, so a
    driver can read the daemon's own account of the turn it just drove --
    ``reason`` plus the ``details`` the terminal carries (for a budget stop:
    the exhausted dimension and the usage that crossed it).

    ``session_ended`` is the question a turn verb exists to answer: ``False``
    means the turn ended and the session is still there, ``True`` means this
    turn was the session's last.  It is ``True`` for EVERY terminal, including
    the ones :data:`CLEAN_TERMINAL_REASONS` lets a turn verb return on -- so
    "the session is over" is always available here, and
    :class:`SessionEnded` is the narrower claim that something cut the turn
    short.  ``complete`` never raises for a terminal (a session settling is
    what it was asked to wait for), which makes this attribute its only route
    to the reason (jaato #1007: "the caller sees 'no payload', not 'budget
    exhausted'").

    ``None`` while a call is in flight, and after one that raised before
    settling -- a ``TurnTimeout`` writes no terminus rather than leaving the
    previous turn's standing, which would read exactly like an answer.
    """

    __slots__ = ("reason", "details", "error_type", "error_summary",
                 "session_ended")

    def __init__(self, reason: str, *, details=None, error_type=None,
                 error_summary=None, session_ended: bool = False):
        self.reason = reason
        self.details = details
        self.error_type = error_type
        self.error_summary = error_summary
        self.session_ended = session_ended

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (f"Terminus(reason={self.reason!r}, "
                f"session_ended={self.session_ended!r}, "
                f"details={self.details!r})")


class SessionEnded(Exception):
    """The session a TURN call was driving is over.

    Raised by :meth:`Session.ask` and :meth:`Session.stream` when the turn
    they drove was CUT SHORT by the session ending, and by all three verbs
    when the daemon answers that the session no longer exists.  Carries the
    daemon's ``reason`` and the terminal's ``details`` so a driver can branch
    without parsing prose.

    NOT every terminal: :data:`CLEAN_TERMINAL_REASONS` (``natural``,
    ``client_request``, ``stopped``) returns normally, because the turn ran to
    its end and the session finished or the caller stopped it.  A scaffolded
    ``client`` driving a completion-gated profile with ``ask`` sees ``natural``
    on its one successful turn, and raising there would fail every such run.
    The fact that the session is over still reaches the caller either way, on
    :attr:`Session.terminus`.

    WHY A TURN VERB RAISES AND ``complete`` DOES NOT.  ``ask`` / ``stream``
    promise one turn's output *with the session still there for the next
    one*; a terminal that cut the turn short breaks that promise, and every
    later call on the handle is dead.  Before jaato #1007 a budget-cut turn
    came back as ``''`` --
    indistinguishable from a model that said nothing -- and the caller found
    out on its NEXT send, which on a cascade-stamped session never returned at
    all.  ``complete`` promises to drive the session to its terminus, so a
    terminal is its success condition, not a breach; it records the same facts
    on :attr:`Session.terminus` instead.

    ``reason`` is the daemon's ``SessionTerminatedEvent.reason``
    (``"budget_exhausted"``, ``"natural"``, ``"stopped"``,
    ``"client_request"``, ``"cascade_cancelled"``, ...), except for the
    session-does-not-exist route, which uses ``"not_found"`` -- the daemon
    said so with an ``ErrorEvent``, not with a terminal.  ``reason="error"``
    never arrives here: that keeps raising :class:`AgentError`, as it always
    did.
    """

    def __init__(self, session_id: str, reason: str, *, details=None,
                 message: Optional[str] = None):
        self.session_id = session_id
        self.reason = reason
        self.details = details
        super().__init__(message or
                         f"session {session_id} ended ({reason})")


class TurnTimeout(TimeoutError):
    """A turn did not reach a terminus within the caller's ``timeout``.

    Raised by :meth:`Session.ask` / :meth:`Session.complete` /
    :meth:`Session.stream` when one is given a ``timeout=`` and the wait
    outlives it.  The session is NOT stopped -- it keeps running daemon-side
    and this exception says only that the caller stopped waiting; call
    ``s.client.stop()`` (or end the session) if the work should stop too.

    A WALL CLOCK IS THE CALLER'S, NOT THE CEILING'S.  It is tempting to
    delegate this to a cascade task pool's ``seconds`` limit, and that does
    not work: a pool is an aggregate over COMPLETED work, reconciled when a
    session ENDS, so a job that runs away never reconciles and never charges.
    The pool is coherent; it is simply not a timeout (jaato #826).

    Subclasses :class:`TimeoutError` (which ``asyncio.TimeoutError`` aliases
    from 3.11), so code already written around ``asyncio.wait_for`` keeps
    catching it.
    """

    def __init__(self, timeout: float, waiting_for: str = "a terminal event"):
        self.timeout = timeout
        self.waiting_for = waiting_for
        super().__init__(f"no {waiting_for} within {timeout}s")


class PermissionUnhandled(Exception):
    """A gated tool requested permission but no ``on_permission`` callback was
    supplied to :meth:`IPCClient.session`.  The facade auto-denied (to unstick
    the daemon) and raised this rather than hang or silently degrade.  Pass
    ``on_permission=`` to ``session(...)`` or drop to the low-level API."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(
            f"tool {tool_name!r} requested permission but no on_permission "
            f"callback was set — pass on_permission= to session(...) or use "
            f"the low-level subscribe(EventType.PERMISSION_REQUESTED) API"
        )


class _TurnWatch:
    """THE settle rule.  One implementation, used by all three verbs.

    ``complete`` owned this rule alone (jaato #767) and ``ask`` / ``stream``
    each carried their own, simpler one -- first-of ``{TURN_COMPLETED,
    SESSION_TERMINATED}``.  That is what jaato #1007 is: the daemon emits the
    turn event 2-3 ms BEFORE the ``SESSION_TERMINATED`` that says the turn
    also ended the session, from the same thread, so the simpler rule settles
    on the turn and never sees the terminal.  Two rules meant one of them had
    to be wrong; this is the one both use now.

    THE RULE.  A ``TURN_COMPLETED`` for the latched agent PROPOSES a
    terminus; the daemon's next word about that agent decides what the
    proposal was:

    * ``SESSION_TERMINATED`` -- an unconditional terminus, whatever was
      proposed.  It carries the reason and ``details``, and is the only event
      that sets ``session_ended``.
    * ``AGENT_STATUS_CHANGED`` for the latched agent -- the confirmation.
      What it means is the one thing that differs between the two spans, and
      it is the whole of the difference:

      ``span="turn"`` (``ask`` / ``stream``)
          ANY status settles.  The turn is over either way; whether the agent
          starts another one is not this call's question.
      ``span="session"`` (``complete``)
          ``"active"`` WITHDRAWS the proposal (a nudge, a drained send -- the
          session is still working); ``"idle"`` / ``"done"`` confirms it.

    NO HANG, ON EITHER SIDE OF THE LATCH -- #767's two escapes, kept:

    * *Never latched.*  No opening ``AGENT_STATUS_CHANGED(status="active")``
      arrived (an older daemon, or a connection whose cascade registration
      filters the type), so the first ``TURN_COMPLETED`` settles immediately,
      exactly as the pre-#767 rule did.
    * *Latched, then unconfirmed.*  The daemon stopped talking between a turn
      and its status event.  The proposal is settled after the span's grace
      (:data:`SETTLE_GRACE` / :data:`TERMINUS_GRACE`) rather than waited on
      forever -- this module has shipped an infinite hang before (PR #399).

    A SESSION THAT IS NOT THERE FAILS RATHER THAN WAITS.  The daemon answers
    a request for an unloaded session with ``ErrorEvent(error_type=
    "SessionError")`` at once, and nothing in the facade listened -- so on a
    cascade-stamped session, whose terminal ALSO unloads it, the next call
    waited forever (jaato #1007, measured at 12+ minutes of a real run).
    That event now settles the wait, and the verb raises
    :class:`SessionEnded`.
    """

    def __init__(self, session: 'Session', *, span: str, on_settle=None):
        self._session = session
        self._client = session.client
        self._span = span
        self._on_settle = on_settle
        self.box: Dict[str, Any] = {}
        self.done = asyncio.Event()
        self._wake = asyncio.Event()
        # The agent whose turns can settle this call, and the turn of its that
        # is currently PROPOSING a terminus.  ``None`` = nothing proposed, so
        # the wait is unbounded.
        self._state: Dict[str, Any] = {"agent": None, "proposal": None}
        self._grace = SETTLE_GRACE if span == "session" else TERMINUS_GRACE
        # A call is now in flight, so the last one's terminus stops
        # describing anything.  Cleared HERE rather than on return because
        # the failure mode is the call that does NOT return normally -- a
        # ``TurnTimeout`` never reaches ``_raise_if_needed``, and a stale
        # answer read as this turn's is worse than no answer.
        session.terminus = None

    # -- handlers (dispatched inline on the loop by the client registry) --

    def _finish(self) -> None:
        """End the wait, however it ended."""
        self.done.set()
        self._wake.set()
        if self._on_settle is not None:
            self._on_settle()

    def _settle(self, ev: Any) -> None:
        self._session._note_terminal(self.box, ev)
        self._finish()

    def on_terminal(self, ev: Any) -> None:
        """``SESSION_TERMINATED``: unconditional, and the only ``session_ended``.

        Recorded on its own keys rather than through ``_note_terminal``'s
        ``setdefault``: a terminal may arrive in the same batch as -- and so
        just after -- a status event that already settled the call, and the
        terminal's account of why the session ended outranks the absence of
        one on a status event.
        """
        self.box["ended"] = True
        self.box["end_reason"] = getattr(ev, "reason", None) or "natural"
        self.box["details"] = getattr(ev, "details", None)
        self._settle(ev)

    def on_turn(self, ev: Any) -> None:
        if self._state["agent"] is None:
            self._settle(ev)            # never latched -> pre-#767 fallback
            return
        if getattr(ev, "agent_id", None) == self._state["agent"]:
            self._state["proposal"] = ev
            self._wake.set()

    def on_status(self, ev: Any) -> None:
        status = getattr(ev, "status", None)
        if self._state["agent"] is None:
            if status == "active":
                self._state["agent"] = getattr(ev, "agent_id", None)
            return
        if getattr(ev, "agent_id", None) != self._state["agent"] \
                or self._state["proposal"] is None:
            return
        if status == "active" and self._span == "session":
            self._state["proposal"] = None      # another turn is starting
            self._wake.set()
        else:
            self._settle(ev)

    def on_error(self, ev: Any) -> None:
        """``ErrorEvent``: fail fast when it says this session is not there.

        Narrow on purpose.  ``error_type == "SessionError"`` is the daemon's
        four "this request could not reach a session" replies and nothing
        else, so an ordinary mid-turn error still belongs to the turn.
        ``recoverable`` is not the discriminator: the "Session not found"
        reply leaves it at its ``True`` default.

        An event naming a DIFFERENT session is ignored; one naming none is
        accepted, because a daemon predating the stamp on this reply sends it
        unattributed and degrading to "wait forever" is the defect.
        """
        if getattr(ev, "error_type", None) != "SessionError":
            return
        sid = getattr(ev, "session_id", None)
        if sid and sid != self._session.session_id:
            return
        # NOT through ``_settle``: an ``ErrorEvent`` is not a terminus
        # anybody drove to, and running it through ``_note_terminal`` would
        # stamp the box with a ``reason`` the daemon never gave.
        self.box["gone"] = getattr(ev, "error", "") or "session not found"
        self._finish()

    # -- wiring + wait ----------------------------------------------------

    def subscribe(self) -> Callable[[], None]:
        """Wire the four handlers; return one unsubscribe for all of them.

        ``TURN_COMPLETED`` and ``ERROR`` are NOT ``subscribe_once``: a
        subagent's turn and an unrelated error must not consume the slot the
        latched agent's turn needs.
        """
        unsubs = [
            self._client.subscribe_once(EventType.SESSION_TERMINATED,
                                        self.on_terminal),
            self._client.subscribe(EventType.TURN_COMPLETED, self.on_turn),
            self._client.subscribe(EventType.AGENT_STATUS_CHANGED,
                                   self.on_status),
            self._client.subscribe(EventType.ERROR, self.on_error),
        ]

        def _unsubscribe() -> None:
            for un in unsubs:
                un()

        return _unsubscribe

    async def wait(self) -> None:
        """Block until this call's terminus is known.

        Unbounded while nothing is proposed -- a turn's natural length is the
        model's to decide.  The caller's own ``timeout`` is layered on top by
        :meth:`Session._wait_bounded`, so the two clocks stay separate: this
        one settles, that one raises.
        """
        while not self.done.is_set():
            # Clear BEFORE reading the state the handlers write.  Sync
            # handlers run inline on this loop, so they cannot interleave
            # between the clear and the await -- whatever they had already
            # recorded is visible here, and anything later re-sets ``_wake``.
            self._wake.clear()
            proposal = self._state["proposal"]
            if proposal is None:
                await self._wake.wait()
                continue
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._grace)
            except asyncio.TimeoutError:
                # The confirmation never came.  Settle on what the turn
                # proposed rather than wait on a daemon that stopped talking.
                if self._state["proposal"] is proposal and not self.done.is_set():
                    self._settle(proposal)


class Session:
    """High-level handle over an open ``IPCClient`` session.

    Owns the send-and-wait recipe so a turn can never hang in user code —
    and, since jaato #1007, so a turn that ended the SESSION can never be
    handed back as though it had not.  All three verbs settle through the one
    rule in :class:`_TurnWatch`; each records :attr:`terminus`, and the turn
    verbs raise :class:`SessionEnded`.  Construct via
    :meth:`IPCClient.session`, not directly.
    """

    def __init__(self, client: Any, session_id: str, on_permission=None):
        self._client = client
        self.session_id = session_id
        self._on_permission = on_permission
        self._unhandled_perm: Optional[str] = None
        self._perm_callback_error: Optional[Exception] = None
        #: How the most recent ``ask`` / ``stream`` / ``complete`` on this
        #: session ended -- see :class:`Terminus`.  ``None`` before the first
        #: call.  It is the channel ``complete`` has for the daemon's reason
        #: and ``details``, since its return value is the payload; the turn
        #: verbs also raise :class:`SessionEnded`, which carries the same two.
        #: Well defined because a session is one conversation and the daemon
        #: serialises its turns -- it describes the call that last returned.
        self.terminus: Optional[Terminus] = None
        # Wire permissions once for the session's lifetime — fail loud, never
        # hang (the second hang trap after #399).
        client.subscribe(
            EventType.PERMISSION_REQUESTED,
            lambda ev: asyncio.ensure_future(self._on_perm(ev)),
        )

    @property
    def client(self):
        """The underlying low-level client (``IPCClient`` / ``IPCRecoveryClient``).

        The facade is purely additive — drop to the full event API on the SAME
        connection when you need it (custom ``subscribe``/``events`` routing,
        ``respond_to_permission(edited_arguments=...)``, ``cascade_events``,
        ``attach_session``, ...) while still using ``ask``/``complete``/``stream``
        for the common turns::

            async with IPCClient.session(profile=...) as s:
                s.client.subscribe(EventType.TOOL_CALL_END, observer)  # low-level
                print(await s.ask("..."))                              # facade

        ``ask``/``complete``/``stream`` clean up only their own subscriptions, so
        listeners you add via ``s.client`` are independent and persist across
        turns.
        """
        return self._client

    async def _on_perm(self, ev: Any) -> None:
        request_id = getattr(ev, "request_id", "")
        resp: Any = "n"
        if self._on_permission is not None:
            try:
                resp = self._on_permission(ev)
                if asyncio.iscoroutine(resp):
                    resp = await resp
            except Exception as exc:  # never let a user callback hang the turn
                # A raising on_permission must NOT deadlock the turn — the
                # in-process channel blocks a worker thread on the response with
                # no transport timeout, so a response is mandatory. Deny to
                # unstick, and surface the callback's error on the in-flight
                # ask()/complete() instead of hanging.
                self._perm_callback_error = exc
                resp = "n"
        else:
            # No policy to apply — deny to unstick, flag so the in-flight
            # ask()/complete() raises PermissionUnhandled.
            self._unhandled_perm = getattr(ev, "tool_name", "?")
        # ALWAYS respond — guaranteed even on a raising callback (the unconditional
        # await below is the no-deadlock invariant).
        await self._client.respond_to_permission(request_id, resp or "n")

    def _raise_if_needed(self, box: Dict[str, Any], *,
                         end_is_error: bool = False) -> None:
        """Record the terminus on the session, then raise what it implies.

        ``end_is_error`` is the one difference between the turn verbs and
        ``complete``: a session that ended is a broken promise to ``ask`` /
        ``stream`` and the success condition of ``complete``.  Both record it.

        Order matters.  ``reason == "error"`` keeps raising
        :class:`AgentError` for every verb, as it always has, so the richer
        type wins over the general "the session ended" one -- an error
        terminal is an error first and an ending second.
        """
        self.terminus = Terminus(
            box.get("end_reason") or box.get("reason") or "natural",
            details=box.get("details"),
            error_type=box.get("error_type"),
            error_summary=box.get("error_summary"),
            session_ended=bool(box.get("ended")) or box.get("gone") is not None,
        )
        if self._perm_callback_error is not None:
            exc = self._perm_callback_error
            self._perm_callback_error = None
            raise exc
        if self._unhandled_perm is not None:
            tool = self._unhandled_perm
            self._unhandled_perm = None
            raise PermissionUnhandled(tool)
        # BOTH keys, because a terminal can land in the same dispatch batch
        # as -- and just after -- the status event that already settled the
        # call.  ``_note_terminal`` writes ``reason`` with ``setdefault``, so
        # the status event's synthetic ``"natural"`` holds it; the terminal's
        # own account went to ``end_reason``.  Reading only the first would
        # turn an error terminal that arrived a microsecond late into a plain
        # ``SessionEnded``, losing the ``error_type`` the caller branches on.
        if "error" in (box.get("reason"), box.get("end_reason")):
            raise AgentError(box.get("error_type"), box.get("error_summary"))
        # The daemon told us the session is not there.  Every verb fails on
        # this -- it is not a terminus anybody drove to, it is a request that
        # reached nothing.
        if box.get("gone") is not None:
            raise SessionEnded(self.session_id, "not_found",
                               message=box["gone"])
        end_reason = box.get("end_reason") or "natural"
        if end_is_error and box.get("ended") \
                and end_reason not in CLEAN_TERMINAL_REASONS:
            raise SessionEnded(self.session_id, end_reason,
                               details=box.get("details"))

    @staticmethod
    def _note_terminal(box: Dict[str, Any], ev: Any) -> None:
        # first-of {TURN_COMPLETED, SESSION_TERMINATED}: TURN_COMPLETED carries
        # no reason (-> "natural"); SESSION_TERMINATED carries the rich
        # reason/error.  setdefault lets a real error reason win on a race.
        box.setdefault("reason", getattr(ev, "reason", None) or "natural")
        if getattr(ev, "error_type", None):
            box["error_type"] = ev.error_type
        if getattr(ev, "error_summary", None):
            box["error_summary"] = ev.error_summary

    @staticmethod
    async def _wait_bounded(waiter, timeout: Optional[float],
                            waiting_for: str) -> None:
        """``await waiter``, bounded by *timeout* seconds when one is given.

        The one place the facade's optional wall clock is enforced, so
        ``ask`` and ``stream`` cannot drift apart on what a ``timeout=``
        means.  ``None`` waits exactly as before -- an unbounded wait is
        still the default, because a turn's natural length is the model's to
        decide and guessing a ceiling for every caller would break long
        agentic work.
        """
        if timeout is None:
            await waiter
            return
        try:
            await asyncio.wait_for(waiter, timeout=timeout)
        except asyncio.TimeoutError:
            raise TurnTimeout(timeout, waiting_for) from None

    def _subscribe_media(self, on_media) -> Any:
        """Route this session's MODEL SPEECH to ``on_media``; return unsubscribe.

        Symmetric with the text these methods already hand back: ``ask``
        returns what the model wrote, ``on_media`` delivers what it said.
        Audio gets a sink rather than a return value because it streams --
        accumulating a whole utterance before returning would defeat
        playing it as it arrives, and would hold an unbounded buffer.

        MODEL speech only.  A tool's attachments ride the same event but
        belong to that tool call, and a caller after those is not asking
        about this session's answer; ``client.subscribe`` on
        ``TOOL_OUTPUT`` still reaches them.

        Returns a no-op unsubscribe when ``on_media`` is None, so callers
        wanting no media pay nothing and the three call sites stay flat.
        """
        if on_media is None:
            return lambda: None

        def _fan(ev: Any) -> None:
            if ev.is_model_speech():
                on_media(ev)

        return self._client.subscribe(EventType.TOOL_OUTPUT, _fan)

    async def ask(self, prompt: str, *,
                  sources: Optional[Collection[str]] = ("model",),
                  parallel_tools: Optional[bool] = None,
                  attachments: Optional[list] = None,
                  timeout: Optional[float] = None,
                  on_media: Optional[Callable[[Any], None]] = None) -> str:
        """Send ``prompt``, wait for the turn to finish, return collected text.

        Settles on :class:`_TurnWatch`'s rule at ``span="turn"`` — the turn
        proposes, the daemon's next word about that agent decides — so a plain
        turn never hangs and a turn that ALSO ended the session is not
        mistaken for one that did not.  ``sources`` selects which
        ``AGENT_OUTPUT`` chunks to keep by ``.source`` — default ``("model",)``
        (a clean answer); ``None`` collects everything.  Raises
        :class:`AgentError` on an error terminal, :class:`SessionEnded` when
        this turn was CUT SHORT by the session ending (see
        :data:`CLEAN_TERMINAL_REASONS`) or the daemon says the session is
        gone, :class:`PermissionUnhandled` if a gated tool went unanswered.
        ``on_media`` receives the model's own SPEECH as it streams (see
        :meth:`_subscribe_media`) — the text comes back, the audio is handed
        over.  :attr:`terminus` records how it ended either way.

        WHY IT RAISES RATHER THAN HANDING BACK THE TEXT.  Before jaato #1007
        this returned whatever the budget-cut turn had produced — ``''`` in
        the repro — and the session was already over: the next ``ask`` on a
        cascade-stamped session never returned, because that terminal also
        unloads it and the daemon's "Session not found" reply had no listener.
        A turn verb whose session is gone has nothing true to return, so it
        says so.  The partial text is not lost: chunks reach ``on_media`` and
        any ``s.client`` listener as they arrive.

        WHICH AGENT.  Only the agent this send started, latched from the
        opening ``AGENT_STATUS_CHANGED(status="active")`` — so a subagent's
        turn no longer settles the parent's call, which it did before.  A
        daemon that announces no agent falls back to settling on the first
        turn, exactly as this did pre-#1007.

        ``timeout`` (seconds, default ``None`` = wait as long as the turn
        takes) bounds the wait and raises :class:`TurnTimeout` on expiry.
        It exists because a fan-out driver needs a per-job wall clock and a
        cascade ceiling cannot be one — a pool reconciles when a session
        ENDS, so the runaway job is exactly the one it never charges for
        (jaato #826).  A timeout does NOT stop the session; it stops
        waiting.
        """
        chunks: list[str] = []
        watch = _TurnWatch(self, span="turn")

        def on_output(ev: Any) -> None:
            if sources is None or getattr(ev, "source", None) in sources:
                text = getattr(ev, "text", "")
                if text:
                    chunks.append(text)

        unsub_out = self._client.subscribe(EventType.AGENT_OUTPUT, on_output)
        unsub_media = self._subscribe_media(on_media)
        unsub_watch = watch.subscribe()
        try:
            await self._client.send_message(
                prompt, parallel_tools=parallel_tools, attachments=attachments)
            await self._wait_bounded(watch.wait(), timeout, "terminal event")
        finally:
            unsub_out()
            unsub_media()
            unsub_watch()
        self._raise_if_needed(watch.box, end_is_error=True)
        return "".join(chunks)

    async def complete(self, prompt: str, *,
                       parallel_tools: Optional[bool] = None,
                       attachments: Optional[list] = None,
                       timeout: Optional[float] = None,
                       on_media: Optional[Callable[[Any], None]] = None,
                       ) -> Optional[Dict[str, Any]]:
        """Send ``prompt`` and return the typed completion ``payload``.

        For completion-gated profiles: captures ``AGENT_COMPLETED.payload``
        (emitted before the terminal), and returns when the SESSION settles --
        not when its first turn ends.  Returns the payload (``None`` if the
        profile declared no schema or the model didn't complete).  Raises
        :class:`AgentError` on an error terminal.  ``on_media`` receives the
        model's own SPEECH as it streams — a completion-gated stage that
        answers OUT LOUD returns its payload here and its audio there.

        WHY A TURN BOUNDARY IS NOT THE SESSION'S TERMINUS.

        This waited on first-of ``{SESSION_TERMINATED, TURN_COMPLETED}``, and
        for a completion-gated profile that is the WRONG event.  Such a
        session ends at ``signal_completion``; when the model instead ends its
        loop in prose, the daemon RE-PROMPTS it (``COMPLETION_NUDGE``) and the
        session keeps working -- so ``TURN_COMPLETED`` fired while the agent
        had minutes of work left.  A caller that graded on the return value
        graded a half-finished tree, and did so silently and in both
        directions: it can read a FAIL from a defect the next turn fixes, or a
        PASS from a tree the next turn regresses (jaato #767, where an eval
        arm was graded 19s before the agent's first commit and recorded FAIL
        on a tree that compiles).

        THE RULE lives in :class:`_TurnWatch`, at ``span="session"``, and is
        shared with ``ask`` / ``stream`` (``span="turn"``) since jaato #1007 --
        one settle rule for the facade rather than one per verb.  A
        ``TURN_COMPLETED`` is the terminus only if the agent SETTLES on it: the
        turn event proposes and the agent's next ``AGENT_STATUS_CHANGED``
        confirms (``"idle"``/``"done"``) or withdraws it (``"active"`` -- a
        nudge, a stashed continuation, a drained user send).
        ``SESSION_TERMINATED`` remains an unconditional terminus -- it is what
        arrives when the nudge budget is spent.  The two escapes from an
        unarriving confirmation, and which agent may settle the call, are
        documented there.

        A CLAIM THAT USED TO STAND HERE AND DOES NOT.  This said "the daemon
        closes every turn of the main agent with exactly one
        ``AGENT_STATUS_CHANGED``".  It is false on a cascade-stamped session:
        the terminal ALSO unloads the session and detaches every client, so
        the ``AGENT_STATUS_CHANGED(status="done")`` emitted after it reaches
        nobody (jaato #1007).  ``complete`` is unaffected only because the
        terminal comes first and settles unconditionally -- so nothing here,
        or in ``ask``, may be built on the status event ALONE arriving.

        ``ask``/``stream`` keep turn semantics deliberately: their contract is
        one turn's output, and an interactive session must hand back each turn
        as it lands.  What they gained in #1007 is not the session span but
        the ability to SEE the terminal that the pre-#1007 rule settled 2-3 ms
        ahead of; a session that ended raises :class:`SessionEnded` there,
        while here it is the success condition and is recorded on
        :attr:`terminus` instead.

        ``timeout`` (seconds, default ``None`` = unbounded) is the CALLER's
        wall clock over the whole settle sequence, raising
        :class:`TurnTimeout` on expiry.  It is deliberately separate from
        ``SETTLE_GRACE``, which bounds only one proposal's confirmation and
        settles rather than raises: a nudged session legitimately runs many
        turns, so "the daemon went quiet after this turn" and "this job has
        taken too long" are different questions with different answers.  A
        timeout does NOT stop the session; it stops waiting (jaato #826).
        """
        watch = _TurnWatch(self, span="session")

        def on_completed(ev: Any) -> None:
            watch.box["payload"] = getattr(ev, "payload", None)

        unsub_media = self._subscribe_media(on_media)
        unsub_comp = self._client.subscribe_once(EventType.AGENT_COMPLETED,
                                                 on_completed)
        unsub_watch = watch.subscribe()
        try:
            await self._client.send_message(
                prompt, parallel_tools=parallel_tools, attachments=attachments)
            await self._wait_bounded(watch.wait(), timeout, "terminal event")
        finally:
            unsub_media()
            unsub_comp()
            unsub_watch()
        self._raise_if_needed(watch.box)
        return watch.box.get("payload")

    async def stream(self, prompt: str, *,
                     sources: Optional[Collection[str]] = ("model",),
                     parallel_tools: Optional[bool] = None,
                     attachments: Optional[list] = None,
                     timeout: Optional[float] = None,
                     on_media: Optional[Callable[[Any], None]] = None,
                     ) -> AsyncIterator[str]:
        """Send ``prompt`` and yield text chunks live as they arrive.

        Async-iterator counterpart to :meth:`ask` — yields each
        ``AGENT_OUTPUT`` chunk (filtered by ``sources``; ``None`` = all) the
        moment it streams in, then stops when :class:`_TurnWatch` settles at
        ``span="turn"``, the same rule :meth:`ask` uses.  ``TURN_COMPLETED``
        fires after all of the turn's output, so no chunk is dropped.  Raises
        :class:`AgentError` on an error terminal, :class:`SessionEnded` when
        the turn was CUT SHORT by the session ending (see
        :data:`CLEAN_TERMINAL_REASONS`) or the daemon says the session is
        gone, :class:`PermissionUnhandled` on an unanswered gated tool — all after
        the stream drains, so the chunks the turn did produce are yielded
        first and only then does the caller learn there will be no next turn.
        :attr:`terminus` records how it ended either way.  ``on_media``
        receives the model's own SPEECH as it streams, alongside the text
        this yields::

            async with IPCClient.session(profile=...) as s:
                async for chunk in s.stream("Tell me a story."):
                    print(chunk, end="", flush=True)

        ``timeout`` (seconds, default ``None`` = unbounded) bounds the WHOLE
        stream, not each chunk -- a model that emits one token a minute
        forever is the case a per-chunk bound would miss -- and raises
        :class:`TurnTimeout` from the iterator on expiry.  Chunks already
        yielded stay yielded; a timeout does NOT stop the session.
        """
        queue: "asyncio.Queue[Any]" = asyncio.Queue()
        sentinel = object()
        # The watch settles on its own schedule (it may hold a proposal open
        # for TERMINUS_GRACE waiting to hear whether the turn also ended the
        # session), so the sentinel is what it pushes when it does -- this
        # loop keeps draining chunks until then.
        watch = _TurnWatch(self, span="turn",
                           on_settle=lambda: queue.put_nowait(sentinel))

        def on_output(ev: Any) -> None:
            if sources is None or getattr(ev, "source", None) in sources:
                text = getattr(ev, "text", "")
                if text:
                    queue.put_nowait(text)

        unsub_out = self._client.subscribe(EventType.AGENT_OUTPUT, on_output)
        unsub_media = self._subscribe_media(on_media)
        unsub_watch = watch.subscribe()
        # The grace loop has to be RUNNING for a proposal to settle, and this
        # coroutine is busy on the queue -- so it runs beside it.
        waiter = asyncio.ensure_future(watch.wait())
        try:
            loop = asyncio.get_running_loop()
            deadline = None if timeout is None else loop.time() + timeout
            await self._client.send_message(
                prompt, parallel_tools=parallel_tools, attachments=attachments)
            while True:
                if deadline is None:
                    item = await queue.get()
                else:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise TurnTimeout(timeout, "terminal event")
                    try:
                        item = await asyncio.wait_for(queue.get(),
                                                      timeout=remaining)
                    except asyncio.TimeoutError:
                        raise TurnTimeout(timeout, "terminal event") from None
                if item is sentinel:
                    break
                yield item
        finally:
            waiter.cancel()
            unsub_out()
            unsub_media()
            unsub_watch()
        self._raise_if_needed(watch.box, end_is_error=True)


class _SessionContext:
    """Async context manager returned by :meth:`IPCClient.session`.

    Connects + creates the session on ``__aenter__`` (failing loud rather than
    yielding a dead session) and disconnects on ``__aexit__``.
    """

    def __init__(self, client: Any, create_kwargs: Dict[str, Any],
                 on_permission, connect_timeout: float, client_tools=None):
        self._client = client
        self._create_kwargs = create_kwargs
        self._on_permission = on_permission
        self._connect_timeout = connect_timeout
        self._client_tools = client_tools

    async def __aenter__(self) -> Session:
        if not await self._client.connect(timeout=self._connect_timeout):
            raise ConnectionError(
                "could not connect to / autostart the jaato daemon — "
                "run `python -m jaato_sdk.doctor`"
            )
        # Host/client tools MUST be registered AFTER connect but BEFORE
        # create_session — the runner-tier model only sees tools registered
        # before the session exists (mid-session registration isn't picked up
        # until a later turn).  This is the hook that lets host-tool clients use
        # the facade instead of dropping to the low-level connect/register/create
        # dance.  See register_client_tools().
        if self._client_tools:
            await self._client.register_client_tools(self._client_tools)
        # No ``if not sid`` guard: ``create_session`` raises now, and its
        # exception STATES the cause.  What stood here guessed one --
        # "check provider auth" -- which is one of five causes and wrong for
        # the other four: a failed socket write, a refusal for an unknown
        # profile, a timeout, a dropped connection.  A hardcoded likely-cause
        # sends the reader to the wrong place with full confidence, which is
        # worse than saying nothing.
        #
        # The disconnect stays, and moves into a ``finally``-shaped guard: it
        # used to run only on the ``not sid`` path, so an EXCEPTION out of
        # create_session (already possible today -- the recovery client raises
        # ReconnectingError / ConnectionClosedError from _check_can_send)
        # leaked the connection.
        try:
            sid = await self._client.create_session(**self._create_kwargs)
        except BaseException:
            await self._client.disconnect()
            raise
        return Session(self._client, sid, self._on_permission)

    async def __aexit__(self, *exc: Any) -> bool:
        await self._client.disconnect()
        return False


def open_session(client_cls, *, profile=None, agent=None, agent_params=None,
                 cascade_driver_id=None, on_permission=None, client_tools=None,
                 socket_path=None, env_file: str = ".env",
                 workspace_path=None, auto_start: bool = True,
                 client_type: ClientType = ClientType.API,
                 connect_timeout: float = 120.0,
                 config_root=None, apparmor=None,
                 on_status_change=None, presentation=None,
                 min_protocol_version=None) -> _SessionContext:
    """Build a client of ``client_cls`` and return a session context manager.

    Backs :meth:`IPCClient.session` and :meth:`IPCRecoveryClient.session`.
    ``profile`` / ``agent`` / ``agent_params`` / ``cascade_driver_id`` are
    forwarded to ``create_session`` unchanged, so both the declarative (named)
    and programmatic (inline-dict) styles work.  ``client_tools`` (a list of
    host-tool specs, same shape as :meth:`register_client_tools`) is registered
    after connect but before create_session, so host-tool clients can use the
    facade instead of the low-level connect/register/create dance.
    ``config_root`` (read-only-config root override) and ``apparmor`` (opt-in
    per-session AppArmor confinement) are accepted by BOTH client classes.
    They were ``IPCClient``-only until 2026-08-23, and this docstring used to
    record that as deliberate — it was not.  ``on_status_change`` is the only
    genuinely asymmetric arg: it is an ``IPCRecoveryClient``-only
    reconnection-status callback, meaningless on a plain ``IPCClient``.
    Each is forwarded to the constructor only when set — including
    ``min_protocol_version``, the wire protocol this caller requires, so
    adopting the facade never quietly drops a compatibility guarantee the
    plain constructor offers.
    """
    ctor_kwargs = dict(client_type=client_type, auto_start=auto_start,
                       env_file=env_file, workspace_path=workspace_path,
                       presentation=presentation)
    # Accepted by BOTH client classes since 2026-08-23; forwarded only when
    # set so the constructors keep their own defaults.  The previous comment
    # here claimed they were IPCClient-only "by design, same pattern as
    # on_status_change" — a false analogy that made an accident look
    # deliberate.  on_status_change IS recovery-only (a reconnection concept);
    # config_root/apparmor are ordinary session bootstrap config that goes
    # straight into ClientConfigRequest, and withholding them from recovery
    # clients left confinement SILENTLY incomplete for anything that had to
    # survive a daemon restart.
    if config_root is not None:
        ctor_kwargs["config_root"] = config_root
    if apparmor is not None:
        ctor_kwargs["apparmor"] = apparmor
    if on_status_change is not None:
        ctor_kwargs["on_status_change"] = on_status_change
    # The wire protocol this caller REQUIRES.  Without it the facade was a
    # silent downgrade: a client that adopted the sugar could no longer say
    # which protocol it depends on, so an older daemon simply never sent the
    # newer fields and the caller saw an empty result rather than a refusal
    # naming the version.  A convenience layer must not cost a guarantee the
    # low-level constructor offers.
    if min_protocol_version is not None:
        ctor_kwargs["min_protocol_version"] = min_protocol_version
    client = (client_cls(socket_path, **ctor_kwargs) if socket_path is not None
              else client_cls(**ctor_kwargs))
    create_kwargs = dict(profile=profile, agent=agent, agent_params=agent_params,
                         cascade_driver_id=cascade_driver_id)
    return _SessionContext(client, create_kwargs, on_permission, connect_timeout,
                           client_tools=client_tools)


async def ask(prompt: str, *, sources: Optional[Collection[str]] = ("model",),
              **session_kwargs: Any) -> str:
    """One-shot: open a session, ask, return the answer, tear down.

    Sugar over ``IPCClient.session`` — accepts the same kwargs (``profile``,
    ``agent``, connection knobs, ``on_permission``).  One daemon connect per
    call; for repeated calls use the ``IPCClient.session`` context manager.
    """
    from .ipc import IPCClient
    async with IPCClient.session(**session_kwargs) as s:
        return await s.ask(prompt, sources=sources)
