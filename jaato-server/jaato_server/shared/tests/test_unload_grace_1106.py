"""A session is not torn down because its audience blinked (#1106).

A WebSocket close -- a tab reload, a network blip, a phone backgrounding the
page -- reached ``SessionManager.detach_client`` and the daemon tore the
session down on the spot: saved it, closed its log handlers, stopped its
workspace monitor, tore down its isolated subagents, shut its server down and
returned its pool slot.  ``_maybe_unload_session`` had two gates -- "still has
clients" and "the model thread is running" -- and no time dimension at all, so
the browser came back holding a session id the daemon no longer had in memory
and every send was answered ``[SessionError] Session not found:``.

Nothing was ever LOST: ``attach_session`` revives from disk.  What was wrong
is the price -- a full teardown and a full respawn for a two-second network
event -- and the premise, which is the line this module guards:

    **A terminal ENDS a session; a disconnect only removes its audience.**

So three of the four callers of ``_maybe_unload_session`` now defer for
``runtime_limits.unload_grace_seconds`` (60 s by default), and the fourth --
``_apply_default_cascade_policy``, which is acting on a
``SessionTerminatedEvent`` -- passes ``immediate=True``.  That last one is not
a carve-out from the rule, it is the other side of it, and it is the
regression that matters most: its own docstring records headless handoffs
returning their slot in ~250 ms against one pinned discovery slot that stalled
a cascade for 6m43s.

WHAT THIS MODULE PINS, AND THE DECISION BEHIND EACH
====================================================

``test_a_terminal_still_unloads_immediately``
    The line above.  Fails the moment the terminal path starts waiting.

``test_attach_switching_away_gets_the_grace``
    Question 3, decided as GRACE.  An attach-switch removes a session's
    audience; it does not end the session, so it falls on the same side of
    the line as a disconnect, and the benefit is that clicking between two
    sessions in a UI stops costing a teardown and a respawn each way --
    which is this issue's own complaint arriving through a different verb.

    The cost is real and is stated rather than waved at: clicking through
    five sessions holds five instead of one, for a decaying 60 s window.
    What the grace does NOT do is cause loads that would not otherwise
    happen -- attaching already loads a session -- so the worst case is the
    transient working set of someone browsing their own sessions, which is
    exactly the set they may click back into.  And a rule with one exception
    carved by intuition is how the next call site gets it wrong.

``test_the_grace_does_not_read_the_clients_declared_kind``
    Question 4, decided as NO.  ``client_type`` is a PRESENTATION field
    ("values describe the kind of display surface, not specific apps"),
    it is client-declared and optional, and it is wrong in both directions --
    a localhost ``web`` UI blinks least of anything and a chat bot on a mobile
    network blinks most.  Deciding how long the daemon holds a session from
    what the client's screen looks like is the substitution #881 is about.

    The residual case it would buy is narrower than it first looks: a
    cascade-stamped or headless session is ALREADY exempt, because its
    terminal takes the ``immediate`` path above, so what remains is a
    non-cascade programmatic driver that simply disconnects.  For that case
    the honest instrument already exists and is visible: the deployment sets
    ``unload_grace_seconds: 0``.  One number an operator can read and change
    beats a second, silent lifetime policy keyed on a field nobody validates.

``test_a_reattach_inside_the_grace_costs_nothing``
    The property the whole design rests on.  ``_do_session_unload`` already
    re-checked under the lock and aborted; the grace makes that abort the
    common case rather than a race -- and, better, means the unload thread is
    never started at all.

``test_the_clock_is_derived_not_stamped_at_a_mutation_site`` (AST)
    ``server/session_lifetime.py``'s own docstring gives the reason in this
    repo's words: instrumenting the ten-odd sites that mutate
    ``attached_clients`` is "exactly the shape that lets one new call site
    disarm it" (#735).  Both writers of ``_clientless_since`` derive the fact
    from an empty ``attached_clients`` at the moment they observe it.

NO TEST HERE SLEEPS.  ``_maybe_unload_session`` and
``sweep_session_lifetimes`` both take ``now``, the way #996 and #713 made
their clocks injectable, so a case states the instant it means.  A sleeping
test here would be slow AND flaky.
"""

from __future__ import annotations

import ast
import pathlib
import threading
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from jaato_server.server.session_manager import Session, SessionManager
from jaato_server.server.session_lifetime import (
    describe_armed_bounds,
    resolve_unload_grace,
    unload_grace_remaining,
)
from jaato_server.shared.runtime_limits import (
    DEFAULT_UNLOAD_GRACE_SECONDS,
    RuntimeLimits,
)
from jaato_server.shared.tests.reversion import Reversion

_MANAGER = "jaato-server/jaato_server/server/session_manager.py"
_MANAGER_PY = pathlib.Path(__file__).resolve().parents[2] / (
    "server/session_manager.py")


REVERSIONS = [
    Reversion(
        target=_MANAGER,
        find="""        if not immediate and self._unload_grace_armed:
            limits = self._session_runtime_limits(session)
            remaining = unload_grace_remaining(clientless_since, now, limits)
            if remaining > 0:""",
        replace="""        if not immediate and self._unload_grace_armed:
            limits = self._session_runtime_limits(session)
            remaining = unload_grace_remaining(clientless_since, now, limits)
            if False:""",
        test="test_a_disconnect_does_not_unload_immediately",
        because="a disconnected session is torn down on the spot again",
    ),
    Reversion(
        target=_MANAGER,
        find="""            self._maybe_unload_session(session.session_id, immediate=True)""",
        replace="""            self._maybe_unload_session(session.session_id)""",
        test="test_a_terminal_still_unloads_immediately",
        because=(
            "a SessionTerminatedEvent would wait out the grace, pinning the "
            "pool slot a cascade's next stage is waiting for"
        ),
    ),
    Reversion(
        target=_MANAGER,
        find="""                self._maybe_unload_session(session_id, now=now)""",
        replace="""                pass  # self._maybe_unload_session(session_id, now=now)""",
        test="test_the_sweep_unloads_once_the_grace_has_elapsed",
        because=(
            "nothing carries out a deferred unload, so the grace becomes a "
            "leak rather than a delay"
        ),
    ),
    Reversion(
        target=_MANAGER,
        find="""                self._maybe_unload_session(current)""",
        replace="""                self._maybe_unload_session(current, immediate=True)""",
        test="test_attach_switching_away_gets_the_grace",
        because=(
            "switching away tears the old session down immediately, though "
            "the session was left rather than ended"
        ),
    ),
    Reversion(
        target=_MANAGER,
        find="""        now = time.monotonic() if now is None else now
        clientless_since = self._note_clientless(session_id, now)""",
        replace="""        now = time.monotonic() if now is None else now
        clientless_since = self._clientless_since.get(session_id)""",
        test="test_the_grace_starts_at_the_disconnect_not_at_the_next_sweep",
        because=(
            "the clock is only started by the sweep, so a session created "
            "and detached between two sweeps waits up to a whole extra "
            "interval"
        ),
    ),
]


# ------------------------------------------------------------------ harness

def _sm(*, armed: bool = True) -> SessionManager:
    """A ``SessionManager`` with only the attributes these paths touch.

    Same construction pattern as ``test_session_identity_and_bound_812.py``:
    the full ``__init__`` stands up transports, a workspace index on the real
    ``~/.jaato`` and a plugin registry, none of which the unload decision
    reads.

    Args:
        armed: Whether the unload grace is armed.  ``True`` here (the daemon's
            state, since ``start_lifetime_watchdog`` arms it) rather than the
            constructor's ``False``, because these cases are about the grace.
    """
    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {}
    sm._lock = threading.RLock()
    sm._orphan_since = {}
    sm._ever_attached = set()
    sm._clientless_since = {}
    sm._unload_grace_armed = armed
    sm._client_to_session = {}
    sm._cascade_clients = {}
    sm._cascade_clients_lock = threading.RLock()
    sm._lifetime_watchdog = None
    sm._lifetime_watchdog_stop = threading.Event()
    sm._lifetime_sweep_interval = 15.0
    sm._unloading = {}
    sm._client_config = {}
    # ``attach_session`` is a large method; these are the collaborators it
    # touches on the in-memory path the switch-away case drives.
    sm._session_workspace_index = MagicMock()
    sm._session_workspace_index.workspaces.return_value = []
    sm._workspace_monitors = {}
    sm._normalize_workspace = lambda p: p
    # The attach's own emission builds a pydantic ``SessionInfoEvent`` out of
    # the server's real fields; a MagicMock server cannot satisfy it and none
    # of it bears on the unload decision under test.
    sm._build_session_info_event = MagicMock()
    sm._session_workspace_index.identity.return_value = None
    sm._emit_to_session = MagicMock()
    sm._emit_to_client = MagicMock()
    sm.unloaded = []

    def _record(session_id: str) -> None:
        sm.unloaded.append(session_id)

    # ``_do_session_unload`` is the real teardown -- save, handler close,
    # workspace-monitor stop, ``server.shutdown`` -- and none of it bears on
    # what these cases test, which is whether the DECISION to run it was
    # taken.  So only the thread body is replaced: every gate above it, both
    # original ones and the grace, still runs for real.
    sm._do_session_unload = _record
    return sm


def _after_grace(sm: SessionManager, sid: str, offset: float) -> float:
    """The instant ``offset`` seconds past *sid*'s grace deadline.

    The daemon's own verbs (``detach_client``, ``attach_session``) take no
    ``now`` -- they stamp the clientless clock from the real monotonic clock,
    because they are edge handlers and the instant IS "now".  So a case that
    drives one of them anchors its timeline on the stamp the daemon just made
    rather than on a number of its own, and then names every later instant
    relative to it.

    Still no sleeping and still exact: the stamp is read, not waited for, and
    ``sweep_session_lifetimes`` takes the result as its ``now``.

    Args:
        sm: The manager holding the clock.
        sid: The session whose grace is being measured.
        offset: Seconds past the deadline; negative means still inside it.

    Returns:
        A monotonic instant to hand to ``sweep_session_lifetimes(now=)``.
    """
    since = sm._clientless_since[sid]
    limits = sm._session_runtime_limits(sm._sessions[sid])
    return since + resolve_unload_grace(limits) + offset


def _session(sid: str, *, clients=(), limits=None, running=False) -> Session:
    server = MagicMock()
    server._model_running = running
    server._main_agent_id = "main"
    server._profile = MagicMock()
    server._profile.runtime_limits = limits
    s = Session(session_id=sid, name=sid, server=server,
                created_at=datetime.now().isoformat(), loaded_at=0.0)
    s.attached_clients = set(clients)
    return s


@pytest.fixture(autouse=True)
def _no_real_unload_thread(monkeypatch):
    """Make ``threading.Thread(target=sm._do_session_unload, ...)`` inert.

    ``_maybe_unload_session`` builds the thread itself, so the stand-in above
    is only reached if the thread actually starts.  Starting it is fine --
    the target is the recorder -- but a daemon thread per case is noise, so
    the launch is run synchronously instead.  The DECISION is unchanged,
    which is the only thing under test.
    """
    real = threading.Thread

    class _Sync(real):  # type: ignore[misc,valid-type]
        def start(self):  # noqa: D102
            # ONLY the unload thread.  The lifetime watchdog is created by
            # the same module-level ``threading.Thread``, and running its
            # loop inline would block forever on its own stop event -- which
            # is exactly what happened the first time this fixture was
            # written without the name check.
            if self.name.startswith("unload-"):
                self.run()
            else:
                real.start(self)

    monkeypatch.setattr(threading, "Thread", _Sync)
    yield


# ---------------------------------------------------- the grace itself

def test_a_disconnect_does_not_unload_immediately():
    """The reported case: the only client's socket closes.

    Before #1106 this tore the session down inside ``detach_client``.
    """
    sm = _sm()
    sm._sessions["s"] = _session("s", clients=("client_28",))
    sm._client_to_session["client_28"] = "s"

    sm.detach_client("client_28")

    assert sm.unloaded == [], (
        "the session was unloaded the instant its client went away -- which "
        "is what makes a tab reload cost a full teardown and respawn (#1106)"
    )
    assert "s" in sm._sessions


def test_the_sweep_unloads_once_the_grace_has_elapsed():
    """A delay is only a delay if something comes back for it.

    The four callers are edge-triggered and none fires again just because
    time passed, so the lifetime sweep is the level trigger.
    """
    sm = _sm()
    sm._sessions["s"] = _session("s", clients=("c1",))
    sm._client_to_session["c1"] = "s"

    sm.detach_client("c1")
    assert sm.unloaded == []

    sm.sweep_session_lifetimes(now=_after_grace(sm, "s", -1))
    assert sm.unloaded == [], "unloaded before the grace elapsed"

    sm.sweep_session_lifetimes(now=_after_grace(sm, "s", +1))
    assert sm.unloaded == ["s"]


def test_a_reattach_inside_the_grace_costs_nothing():
    """The property the design rests on: the session is never touched.

    Not "the unload aborted" -- the unload is never STARTED, so there is no
    save, no handler close, no workspace-monitor stop, no ``server.shutdown``
    and no pool-slot return to undo.
    """
    sm = _sm()
    session = _session("s", clients=("c1",))
    sm._sessions["s"] = session
    sm._client_to_session["c1"] = "s"

    sm.detach_client("c1")
    deadline = _after_grace(sm, "s", +1)
    session.attached_clients.add("c2")           # the browser reconnects

    sm.sweep_session_lifetimes(now=deadline)
    assert sm.unloaded == []
    assert "s" in sm._sessions
    session.server.shutdown.assert_not_called()


def test_the_clock_measures_continuous_clientlessness():
    """A reconnect renews the session's claim on being wanted.

    Same shape as the orphan clock (#812): the grace is not "time since the
    first disconnect", it is "time since the LAST one", so a flapping client
    never accumulates its way to a teardown.
    """
    sm = _sm()
    session = _session("s", clients=("c1",))
    sm._sessions["s"] = session
    sm._client_to_session["c1"] = "s"

    sm.detach_client("c1")                        # clock starts
    first_deadline = _after_grace(sm, "s", +1)
    session.attached_clients.add("c2")
    sm.sweep_session_lifetimes(now=first_deadline - 30)  # attached: cleared
    assert "s" not in sm._clientless_since

    session.attached_clients.clear()
    sm._maybe_unload_session("s", now=first_deadline - 29)   # clock restarts

    sm.sweep_session_lifetimes(now=first_deadline)
    assert sm.unloaded == [], (
        "the grace was measured from the FIRST disconnect, so a client that "
        "reconnected and left again got less than a full window"
    )
    sm.sweep_session_lifetimes(now=_after_grace(sm, "s", +1))
    assert sm.unloaded == ["s"]


def test_the_grace_starts_at_the_disconnect_not_at_the_next_sweep():
    """The gate derives the clock itself rather than waiting for a sweep.

    A session created and detached between two sweeps has no entry yet.  If
    only the sweep wrote the clock, such a session would start its grace up
    to a whole sweep interval (15 s) late -- a 25% overshoot on a 60 s grace,
    paid by exactly the short-lived sessions that can least afford a pinned
    slot.
    """
    sm = _sm()
    sm._sessions["s"] = _session("s", clients=("c1",))
    sm._client_to_session["c1"] = "s"
    assert sm._clientless_since == {}, "the sweep has not run yet"

    sm._maybe_unload_session("s", now=100.0)      # still attached: no clock
    assert sm._clientless_since == {}

    sm._sessions["s"].attached_clients.clear()
    sm._maybe_unload_session("s", now=100.0)      # clientless: clock at 100.0

    assert sm._clientless_since.get("s") == 100.0, (
        "the gate did not start the clock, so it would begin only at the "
        "next sweep -- up to a whole interval late"
    )
    sm.sweep_session_lifetimes(now=100.0 + DEFAULT_UNLOAD_GRACE_SECONDS + 1)
    assert sm.unloaded == ["s"]


def test_zero_restores_the_pre_1106_behaviour():
    """``0`` is the operator's opt-out, per the 0-disables convention.

    Deliberately NOT "unbounded", which is what 0 means for the two wall-clock
    BOUNDS in the same block: here it is the tightest value there is.
    """
    sm = _sm()
    sm._sessions["s"] = _session(
        "s", clients=("c1",), limits=RuntimeLimits(unload_grace_seconds=0))
    sm._client_to_session["c1"] = "s"

    sm.detach_client("c1")
    assert sm.unloaded == ["s"]


def test_a_declared_grace_outranks_the_framework_default():
    sm = _sm()
    sm._sessions["s"] = _session(
        "s", clients=("c1",), limits=RuntimeLimits(unload_grace_seconds=5))
    sm._client_to_session["c1"] = "s"

    sm.detach_client("c1")
    assert sm.unloaded == []
    sm.sweep_session_lifetimes(now=_after_grace(sm, "s", -1))
    assert sm.unloaded == []
    sm.sweep_session_lifetimes(now=_after_grace(sm, "s", +1))
    assert sm.unloaded == ["s"]


def test_a_mid_turn_session_is_still_deferred_by_the_model_gate():
    """The grace composes with the gate that was already there.

    A long turn CONSUMES the grace -- the clock runs from the disconnect, not
    from the turn's end -- so a session whose model finishes after the window
    is unloaded at the next sweep with no further wait.
    """
    sm = _sm()
    session = _session("s", clients=("c1",), running=True)
    sm._sessions["s"] = session
    sm._client_to_session["c1"] = "s"

    sm.detach_client("c1")
    past = _after_grace(sm, "s", +1)
    sm.sweep_session_lifetimes(now=past)
    assert sm.unloaded == [], "unloaded while the model thread was running"

    session.server._model_running = False
    sm.sweep_session_lifetimes(now=past + 1)
    assert sm.unloaded == ["s"]


# ------------------------------------------- the line: terminal vs disconnect

def _terminated(sm: SessionManager, session: Session, reason: str = "natural"):
    from jaato_sdk.events import SessionTerminatedEvent
    sm._apply_default_cascade_policy(session, SessionTerminatedEvent(
        session_id=session.session_id, reason=reason))


def test_a_terminal_still_unloads_immediately():
    """*A terminal ENDS a session; a disconnect only removes its audience.*

    THE regression to protect.  ``_apply_default_cascade_policy``'s own
    docstring measures what a delay costs: every headless handoff returns its
    slot in ~250 ms, and one discovery slot that stayed pinned stalled a
    cascade for 6m43s.  A 60 s grace here would reintroduce that on every
    stage of every cascade.
    """
    sm = _sm()
    session = _session("stage", clients=(SessionManager._HEADLESS_CLIENT_ID,))
    sm._sessions["stage"] = session

    _terminated(sm, session)

    assert sm.unloaded == ["stage"], (
        "a SessionTerminatedEvent now waits out the unload grace -- the "
        "pool slot the cascade's next stage is waiting for stays pinned"
    )


def test_a_cascade_stamped_session_is_also_unloaded_immediately():
    """The other shape the terminal path covers: a driver-attached stage."""
    sm = _sm()
    session = _session("discovery", clients=("client_ipc_14",))
    session.cascade_driver_id = "cid-1"
    sm._sessions["discovery"] = session
    sm._client_to_session["client_ipc_14"] = "discovery"

    _terminated(sm, session, reason="budget_exhausted")

    assert sm.unloaded == ["discovery"]


# ------------------------------------------------- question 3: attach-switch

def test_attach_switching_away_gets_the_grace():
    """Decided as GRACE: the session was LEFT, not ended.

    The discriminator is the same sentence the terminal path is on the other
    side of.  Clicking between two sessions in a UI is the same event as a
    reload arriving through a different verb, and paying a teardown plus a
    respawn each way is the cost #1106 exists to remove.
    """
    sm = _sm()
    old = _session("old", clients=("c1",))
    new = _session("new")
    sm._sessions = {"old": old, "new": new}
    sm._client_to_session["c1"] = "old"

    sm.attach_session("c1", "new")

    assert sm.unloaded == [], (
        "switching away tore the old session down immediately; a user "
        "flipping back pays a full respawn"
    )
    assert "old" in sm._sessions
    assert sm._client_to_session["c1"] == "new"

    sm.sweep_session_lifetimes(now=_after_grace(sm, "old", +1))
    assert sm.unloaded == ["old"], "and it IS unloaded once the grace elapses"


# ------------------------------------------------ question 4: no client_type

def test_the_grace_does_not_read_the_clients_declared_kind():
    """Decided as NO: a presentation field must not decide a lifetime.

    The tempting rule was "exempt ``api``", on the grounds that a program's
    socket closing means the program exited.  It is client-DECLARED, optional,
    presentation-scoped, and wrong in both directions.  The case it would buy
    is already narrow -- a cascade or headless session takes the ``immediate``
    path above -- and the deployment that wants it has a visible knob:
    ``runtime_limits.unload_grace_seconds: 0``.
    """
    sm = _sm()
    sm._client_config = {
        "prog": {"presentation": {"client_type": "api"}},
        "tab": {"presentation": {"client_type": "web"}},
    }
    for sid, cid in (("api-driven", "prog"), ("browser", "tab")):
        sm._sessions[sid] = _session(sid, clients=(cid,))
        sm._client_to_session[cid] = sid

    sm.detach_client("prog")
    sm.detach_client("tab")

    assert sm.unloaded == [], (
        "the grace was scoped by client_type. The enum describes the kind of "
        "DISPLAY SURFACE, not how long a session should outlive its client "
        "-- and a deployment that wants no grace says so with a number."
    )


def test_the_grace_source_never_mentions_client_type():
    """Belt and braces for the case above, at the source level.

    A behavioural test can be satisfied by a rule that reads the field and
    happens to answer the same way for these two values.  The decision was
    that the field is not consulted AT ALL.
    """
    src = _MANAGER_PY.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for name in ("_maybe_unload_session", "_sweep_unload_grace",
                 "_note_clientless"):
        node = next(
            (n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == name),
            None,
        )
        assert node is not None, f"{name} is gone -- this guard asserts nothing"
        body = ast.unparse(node)
        assert "client_type" not in body and "presentation" not in body, (
            f"{name} consults the client's declared presentation kind. "
            "#1106 question 4 was decided the other way: that field is a "
            "display-surface descriptor, it is optional, and it is wrong in "
            "both directions (a localhost web UI blinks least of anything)."
        )


# --------------------------------------------------------- derived, not stamped

def _writers_of(attr: str) -> set:
    """Method names that decide membership of ``self.<attr>``.

    Same shape as ``test_orphan_bound_observes_attachment_812.py``, which
    guards the sibling clock.
    """
    tree = ast.parse(_MANAGER_PY.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            func = inner.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in {"setdefault", "pop", "clear", "update"}:
                continue
            owner = func.value
            if isinstance(owner, ast.Attribute) and owner.attr == attr:
                out.add(node.name)
    return out


def test_the_clock_is_derived_not_stamped_at_a_mutation_site():
    """``attached_clients`` is mutated in ten-odd places; none may stamp here.

    ``server/session_lifetime.py`` states the reason for its sibling clock:
    instrumenting every mutation is "exactly the shape that lets one new call
    site disarm it" (#735).  Both permitted writers DERIVE the fact from an
    empty ``attached_clients`` at the moment they observe it, so a future
    caller that forgets is still covered by the sweep.
    """
    writers = _writers_of("_clientless_since")
    assert writers, (
        "nothing writes _clientless_since -- did the clock change name? "
        "This guard asserts nothing if it cannot find the attribute."
    )
    allowed = {
        "_observe_session_lifetimes",   # the sweep, for every loaded session
        "_note_clientless",             # the gate, at the instant it observes
        "_maybe_unload_session",        # clears it on an observed re-attach
        "_do_session_unload",           # ditto, under the lock
    }
    assert writers <= allowed, (
        f"{sorted(writers - allowed)} write(s) _clientless_since. A "
        "timestamp stamped where attached_clients is MUTATED is the #735 "
        "shape: the next call site to forget it disarms the grace silently."
    )


def test_a_no_clock_session_defers_rather_than_unloading():
    """"I have no observation" must read as "wait", not as "unload now".

    The safe default in the direction that matters: a future call site that
    learns to unload and nothing else still defers, and the sweep picks it up.
    """
    assert unload_grace_remaining(None, 12_345.0, None) == (
        DEFAULT_UNLOAD_GRACE_SECONDS)


# ----------------------------------------------------------- question 5: arming

def test_the_arming_line_names_the_grace_and_its_effective_value():
    """#735: log what is armed, with the value that actually applies."""
    line = describe_armed_bounds(None)
    assert "unload_grace_seconds=60.0s (framework default)" in line
    assert "max_orphan_seconds=900.0s" in line

    declared = describe_armed_bounds(RuntimeLimits(unload_grace_seconds=5))
    assert "unload_grace_seconds=5.0s" in declared
    assert "framework default" not in declared.split(
        "unload_grace_seconds")[1]

    off = describe_armed_bounds(RuntimeLimits(unload_grace_seconds=0))
    assert "unload_grace_seconds=0.0s (no grace)" in off


def test_the_grace_is_armed_with_the_thread_that_carries_it_out():
    """A deferral nobody acts on is a leak, not a delay.

    A ``SessionManager`` in a test or an embedding process grows no watchdog
    on purpose; it must not silently acquire a mechanism whose other half is
    missing.  That is #735 in its worst form -- the symptom is sessions
    accumulating, not an error.
    """
    sm = _sm(armed=False)
    sm._sessions["s"] = _session("s", clients=("c1",))
    sm._client_to_session["c1"] = "s"

    sm.detach_client("c1")
    assert sm.unloaded == ["s"], "pre-#1106 behaviour without the sweep"

    assert sm.start_lifetime_watchdog(interval_seconds=3600) is True
    try:
        assert sm._unload_grace_armed is True
    finally:
        sm.stop_lifetime_watchdog()
    assert sm._unload_grace_armed is False


def test_the_listing_says_what_is_holding_a_session_in_memory():
    """An operator must be able to tell "kept for a client that may return"
    from "nothing has got round to unloading it"."""
    sm = _sm()
    session = _session("s", clients=("c1",))
    sm._sessions["s"] = session
    sm._client_to_session["c1"] = "s"
    sm.detach_client("c1")

    row = next(r for r in sm.list_orphan_sessions()
               if r["session_id"] == "s")
    assert row["unload_grace_seconds"] == DEFAULT_UNLOAD_GRACE_SECONDS
    assert row["unload_grace_remaining"] >= 0.0


# ------------------------------------------------------- resolution + inheritance

def test_resolution_is_three_valued_and_never_raises():
    assert resolve_unload_grace(None) == DEFAULT_UNLOAD_GRACE_SECONDS
    assert resolve_unload_grace(RuntimeLimits()) == DEFAULT_UNLOAD_GRACE_SECONDS
    assert resolve_unload_grace(RuntimeLimits(unload_grace_seconds=0)) == 0.0
    assert resolve_unload_grace(RuntimeLimits(unload_grace_seconds=12)) == 12.0
    # A snapshot written before the field existed, revived by a newer daemon.
    assert resolve_unload_grace(object()) == DEFAULT_UNLOAD_GRACE_SECONDS


def _merge(parent_limits, child_limits):
    from jaato_server.shared.plugins.subagent.config import _merge_runtime_limits
    parent = MagicMock()
    parent.name = "base"
    parent.runtime_limits = parent_limits
    child = MagicMock()
    child.name = "leaf"
    child.runtime_limits = child_limits
    return _merge_runtime_limits([parent], child)


def test_a_child_may_narrow_the_grace():
    merged, conflicts = _merge(
        RuntimeLimits(unload_grace_seconds=120),
        RuntimeLimits(unload_grace_seconds=30),
    )
    assert merged.unload_grace_seconds == 30
    assert conflicts == []


def test_a_child_may_not_widen_the_grace():
    merged, _ = _merge(
        RuntimeLimits(unload_grace_seconds=30),
        RuntimeLimits(unload_grace_seconds=600),
    )
    assert merged.unload_grace_seconds == 30


def test_zero_wins_the_min_here_where_it_loses_it_for_a_bound():
    """The two readings of ``0`` in one block, and why they differ.

    For a BOUND, 0 means "never stop this" -- the least restrictive thing a
    layer can say -- so a child cannot use it to escape an ancestor's ceiling.
    For the grace, 0 means "unload immediately", which is the MOST restrictive
    thing a layer can say: it releases the runner and the pool slot soonest.
    Routing both through one rule would invert the safety direction for one
    of them.
    """
    merged, _ = _merge(
        RuntimeLimits(unload_grace_seconds=60, max_orphan_seconds=60),
        RuntimeLimits(unload_grace_seconds=0, max_orphan_seconds=0),
    )
    assert merged.unload_grace_seconds == 0
    assert merged.max_orphan_seconds == 60


def test_two_parents_differing_only_in_the_grace_do_not_conflict():
    """The agreement test normalises every min-wins field out.

    Missing this is how adding a min-wins field turns two compatible base
    profiles into a reported conflict.
    """
    from jaato_server.shared.plugins.subagent.config import _merge_runtime_limits
    parents = []
    for name, grace in (("a", 30), ("b", 90)):
        p = MagicMock()
        p.name = name
        p.runtime_limits = RuntimeLimits(pids_max=64, unload_grace_seconds=grace)
        parents.append(p)
    child = MagicMock()
    child.name = "leaf"
    child.runtime_limits = None
    merged, conflicts = _merge_runtime_limits(parents, child)
    assert conflicts == []
    assert merged.unload_grace_seconds == 30
    assert merged.pids_max == 64


def test_a_negative_grace_is_refused_at_parse_time():
    with pytest.raises(ValueError, match="unload_grace_seconds"):
        RuntimeLimits(unload_grace_seconds=-1)
    with pytest.raises(ValueError, match="unload_grace_seconds"):
        RuntimeLimits(unload_grace_seconds=True)


def test_the_field_round_trips_through_from_dict():
    limits = RuntimeLimits.from_dict({"unload_grace_seconds": 45})
    assert limits.unload_grace_seconds == 45
    assert limits.extra == {}
