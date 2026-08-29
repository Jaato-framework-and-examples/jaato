"""Every event must say WHICH SESSION it is about.

A cascade observer receives events from every session under a cid on one
stream.  Before protocol 1.2, 12 of 112 event types declared ``session_id``
and the rest had none — and the split was not random:

  LIFECYCLE (created / woken / restored / terminated) carried it.
  ACTIVITY  (turn / tool / agent output)              did not.

So an observer could watch sessions appear, sleep and die, but never watch
them WORK.  ``agent_id`` cannot substitute: it is ``"main"`` for every
top-level session, so two siblings of one cascade were indistinguishable.

The framework's own client scaffold printed
``getattr(ev, "session_id", "")`` for every event — an idiom that cannot
tell "no such field" from "field is blank".  Three of the four event types
in its default subscription list rendered empty; the one that worked is the
one the ``cascade_events`` docstring example happens to use.  Both were
correct by accident, having picked from the minority.

Reported by the cascade-coordination probe (Finding 6), which gated its
claims on a declared ``turn_attribution`` capability rather than failing
them for a reason that was not their claim.
"""

import json
import threading

import pytest

import jaato_sdk.events as E
from jaato_sdk.events import (
    AgentOutputEvent,
    SlotSettledEvent,
    ToolCallStartEvent,
    TurnCompletedEvent,
    deserialize_event,
)
from server.session_manager import SessionManager, _stamp_session_id


def _event_classes():
    return [
        getattr(E, n) for n in dir(E)
        if isinstance(getattr(E, n), type)
        and issubclass(getattr(E, n), E.Event)
        and getattr(E, n) is not E.Event
    ]


from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

#: The defect, put back: the 64-call-site path stops attributing, and every
#: event a consumer sees through it goes back to arriving anonymous.
REVERSIONS = [
    Reversion(
        target="jaato-server/server/session_manager.py",
        find="        _stamp_session_id(event, self._client_to_session.get(client_id))\n",
        replace="",
        test="test_an_event_to_a_bound_client_is_attributed_to_its_session",
        because="the majority emit path not attributing its events",
    ),
]


def _make_sm() -> SessionManager:
    """SessionManager skeleton — same shape the cascade phase-1 tests use."""
    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {}
    sm._lock = threading.RLock()
    sm._client_to_session = {}
    sm._cascade_clients = {}
    sm._cascade_clients_lock = threading.Lock()
    sm._cascade_client_idle_timeout = 300.0
    sm._cascade_client_sweep_stop = None
    sm._cascade_client_sweep_thread = None
    sm._cid_last_session_ts = {}
    sm._HEADLESS_CLIENT_ID = "_headless"
    sm._pending_wakes = {}
    # No wake bindings here, so the GC sweep's wake-durability exemption
    # is a no-op.  Without this the lazily-started sweep THREAD raises
    # AttributeError in the background — which pytest does not fail on,
    # so the suite would stay green while printing a traceback.
    sm._wake_binding_registry = _NoWakeBindings()
    return sm


class _NoWakeBindings:
    def has_live_binding_for_cid(self, cid):
        return False


# ----------------------------------------------------------------------
# The protocol-wide claim
# ----------------------------------------------------------------------

def test_every_event_type_can_be_attributed():
    """No event type may be structurally unattributable.

    Asserted over the WHOLE class list rather than a sample, because the
    gap was a whole CATEGORY (activity events) and any sample drawn from
    lifecycle events would have passed while the category stayed broken.
    """
    missing = [
        c.__name__ for c in _event_classes()
        if "session_id" not in c.model_fields
    ]
    assert missing == [], (
        f"{len(missing)} event types cannot be attributed: {missing[:8]}"
    )


def test_the_activity_events_an_observer_needs():
    """The specific types the shipped client scaffold subscribes to.

    Three of these four used to render empty in generated code.
    """
    for cls in (ToolCallStartEvent, TurnCompletedEvent, AgentOutputEvent):
        assert "session_id" in cls.model_fields, cls.__name__


def test_attribution_survives_the_wire():
    """Declared, so it serializes AND is kept on deserialize.

    Both halves matter: the base ``Event`` sets ``extra='ignore'``, so a
    field the model does not declare is dropped at the CLIENT even if the
    server forces it onto the wire.  That is why stamping alone — without
    declaring — could not have worked.
    """
    ev = TurnCompletedEvent(agent_id="main", session_id="sess-A")
    assert "session_id" in ev.to_dict()
    assert deserialize_event(json.dumps(ev.to_dict())).session_id == "sess-A"


# ----------------------------------------------------------------------
# Central stamping
# ----------------------------------------------------------------------

def test_stamp_fills_a_blank_attribution():
    ev = TurnCompletedEvent(agent_id="main")
    _stamp_session_id(ev, "sess-A")
    assert ev.session_id == "sess-A"


def test_stamp_never_overwrites_a_stated_subject():
    """An event ABOUT another session keeps its own subject.

    ``SlotSettledEvent.session_id`` means "the session that just ended".
    Relabelling it with whoever emitted it would replace a true fact with
    a plausible one — worse than blank, because a wrong attribution is
    indistinguishable from a right one.
    """
    ev = SlotSettledEvent(session_id="the-session-that-ended")
    _stamp_session_id(ev, "the-emitter")
    assert ev.session_id == "the-session-that-ended"


def test_stamp_is_a_noop_without_a_session():
    ev = TurnCompletedEvent(agent_id="main")
    _stamp_session_id(ev, None)
    _stamp_session_id(ev, "")
    assert ev.session_id == ""


def test_stamp_tolerates_a_non_event_object():
    """A failed stamp must never break event delivery."""
    class Foreign:
        __slots__ = ()
    _stamp_session_id(Foreign(), "sess-A")   # must not raise


# ----------------------------------------------------------------------
# Both delivery paths
# ----------------------------------------------------------------------

def test_emit_to_session_attributes_both_delivery_paths():
    """One stamp at the chokepoint covers direct clients AND observers.

    These are two different fan-outs in ``_emit_to_session``; stamping in
    only one would attribute an event for an observer but not for the
    attached client watching the same session (or vice versa).
    """
    sm = _make_sm()
    direct, observed = [], []

    sm._event_callback = lambda cid, ev: direct.append(ev)
    sm.register_in_process_client(
        client_id="obs", callback=observed.append,
        cascade_driver_id="cid-1", role="observer",
    )

    session = type("S", (), {})()
    session.session_id = "sess-A"
    session.attached_clients = ["client-1"]
    session.cascade_driver_id = "cid-1"
    session.description = None
    session.is_dirty = False
    sm._sessions["sess-A"] = session
    sm._handle_turn_tracking_event = lambda *a, **k: None
    sm._accumulate_cascade_budget = lambda *a, **k: None
    sm._apply_default_cascade_policy = lambda *a, **k: None

    sm._emit_to_session("sess-A", TurnCompletedEvent(agent_id="main"))

    assert [e.session_id for e in direct] == ["sess-A"], "attached client"
    assert [e.session_id for e in observed] == ["sess-A"], "cascade observer"


def test_bootstrap_events_are_attributed_too():
    """The FIRST events of a session's life bypass ``_emit_to_session``.

    They are exactly the ones an observer uses to notice a session
    exists, so leaving them unattributed would leave the gap open at the
    only moment it is load-bearing.
    """
    sm = _make_sm()
    seen = []
    sm._event_callback = lambda cid, ev: seen.append(ev)

    sm._route_bootstrap_event(
        "client-1", None, TurnCompletedEvent(agent_id="main"), "sess-A",
    )
    assert [e.session_id for e in seen] == ["sess-A"]


def test_two_siblings_are_distinguishable_on_one_stream():
    """The claim the whole change exists to make.

    ``agent_id`` is "main" for both — which is precisely why the observer
    could not tell them apart before.
    """
    sm = _make_sm()
    received = []
    sm._event_callback = lambda cid, ev: None
    sm.register_in_process_client(
        client_id="obs", callback=received.append,
        cascade_driver_id="cid-1", role="observer",
    )
    sm._handle_turn_tracking_event = lambda *a, **k: None
    sm._accumulate_cascade_budget = lambda *a, **k: None
    sm._apply_default_cascade_policy = lambda *a, **k: None

    for sid in ("sibling-a", "sibling-b"):
        s = type("S", (), {})()
        s.session_id = sid
        s.attached_clients = []
        s.cascade_driver_id = "cid-1"
        s.description = None
        s.is_dirty = False
        sm._sessions[sid] = s
        sm._emit_to_session(sid, TurnCompletedEvent(agent_id="main"))

    assert {e.agent_id for e in received} == {"main"}, "agent_id cannot do it"
    assert [e.session_id for e in received] == ["sibling-a", "sibling-b"]


# ---------------------------------------------------------------------------
# The MAJORITY path
#
# ``_emit_to_session`` stamps and has 10 call sites.  ``_emit_to_client`` has
# 64 and did not, so most of what a consumer sees arrived unattributed --
# including every PermissionRequestedEvent, which the earlier audit recorded
# as "NOT verified: whether it arrives unstamped end-to-end".  It did.
#
# The audit concluded _emit_to_client "has nothing to stamp WITH" because it
# takes a client_id.  ``_client_to_session`` was one attribute away, and is
# read exactly that way elsewhere in the class.  A structural audit answered
# "does this emitter call the stamper", not "could it".
# ---------------------------------------------------------------------------

def _emitting_sm(bound: dict):
    """A manager whose ``_emit_to_client`` records what it delivered."""
    sm = _make_sm()
    sm._client_to_session = dict(bound)
    delivered = []
    sm._event_callback = lambda cid, ev: delivered.append((cid, ev))
    return sm, delivered


def test_an_event_to_a_bound_client_is_attributed_to_its_session():
    from jaato_sdk.events import PermissionRequestedEvent

    sm, delivered = _emitting_sm({"client-1": "sess-A"})
    sm._emit_to_client("client-1", PermissionRequestedEvent())

    assert delivered, "nothing was delivered"
    _cid, ev = delivered[0]
    assert ev.session_id == "sess-A", (
        "an event routed to a client bound to a session arrived without "
        "naming it. This is the 64-call-site path; an observer watching a "
        "cascade sees most of its traffic through here and cannot tell two "
        "siblings apart without it."
    )


def test_an_event_that_names_its_own_subject_is_not_relabelled():
    """SlotSettled means 'the session that ended', not 'who received this'.

    Overwriting would replace a true fact with a plausible one, which is
    worse than leaving it blank: a wrong attribution is indistinguishable
    from a right one.
    """
    from jaato_sdk.events import PermissionRequestedEvent

    sm, delivered = _emitting_sm({"client-1": "the-recipient"})
    ev = PermissionRequestedEvent()
    ev.session_id = "the-subject"
    sm._emit_to_client("client-1", ev)

    assert delivered[0][1].session_id == "the-subject", (
        "the recipient's session overwrote an explicitly-set subject"
    )


def test_an_unbound_client_leaves_the_event_unstamped():
    """The honest residue.

    ``_client_to_session`` is not populated pre-init, so events emitted
    while a session is being created have no session to name.  Those are
    genuinely unattributable rather than missed -- and this is the
    population worth counting before deciding whether the base field should
    become ``Optional[str] = None`` to say so out loud.
    """
    from jaato_sdk.events import PermissionRequestedEvent

    sm, delivered = _emitting_sm({})          # nothing bound yet
    sm._emit_to_client("client-unknown", PermissionRequestedEvent())

    assert delivered[0][1].session_id == "", (
        "an event for a client with no session was given one anyway"
    )
