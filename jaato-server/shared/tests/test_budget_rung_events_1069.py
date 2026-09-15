"""A fired degrade rung reaches the client as something it can branch on (#1069).

The issue reported that ``budget_control``'s degrade rungs are "observable
only in the trace log — no client-visible event fires when one applies".
That premise is not quite right, and getting it right is what decides the
fix: ``_apply_budget_rungs`` has always called ``_surface_budget_event``,
which emits ``AgentOutputEvent(source="system")`` carrying bracketed prose
like ``[budget[self-enforced] tokens 85%: degraded planner opus->flash]``.

So the gap is not *a signal*.  It is **a signal a client can branch on**,
and one that is not mixed into the stream the client renders as what the
agent said.  A bot that decorates tier switches and memory stores cannot
decorate this without string-matching ``[budget[``, and a client that
renders agent output verbatim shows the framework's cost machinery to users
as the agent talking.

Two changes, tested here:

A. ``BudgetRungFiredEvent`` — a typed event per APPLIED rung, carrying the
   threshold, the action, the ``origin`` mechanism, the pressure, the
   structured per-dimension usage and what the overlay actually rebound.
   The prose still fires; it has consumers.

B. ``action: notify`` — a rung that ONLY emits.  The issue proposed it as
   an alternative to "a rung with neither ``model_tiers`` nor an action
   treated as emit-only", and that alternative is not available:
   ``DegradeRung.from_dict`` refuses such a rung outright
   (*"degrade[N] does nothing"*), so the vocabulary had to grow.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from shared.budget_control import (
    ACTION_ABORT,
    ACTION_FINALIZE,
    ACTION_NOTIFY,
    BudgetControlConfigError,
    BudgetTracker,
    TERMINAL_ACTIONS,
    VALID_ACTIONS,
)
from shared.jaato_session import JaatoSession
from shared.tests.test_budget_runtime import _cfg, _session
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

_SESSION = "jaato-server/shared/jaato_session.py"

REVERSIONS = [
    Reversion(
        target=_SESSION,
        find="""            if rung.action in TERMINAL_ACTIONS:
                self._budget_terminal_action = rung.action""",
        replace="""            if rung.action:
                self._budget_terminal_action = rung.action""",
        test="test_a_notify_rung_does_not_latch_a_terminal_action",
        because=(
            "a rung declared to change nothing again gets a say in how the "
            "session winds down"
        ),
    ),
    Reversion(
        target=_SESSION,
        find="""            self._notify_budget_rung(rung, origin, detail, changes)""",
        replace="""            pass  # reverted: no typed notification""",
        test="test_the_callback_fires_once_per_applied_rung",
        because="a fired rung is once again visible only as prose and a trace line",
    ),
    Reversion(
        target=_SESSION,
        find="""        if origin == "self-enforced" and self._budget_tracker is not None:""",
        replace="""        if self._budget_tracker is not None:""",
        test="test_a_cascade_pushed_rung_omits_this_sessions_own_usage",
        because=(
            "a pool-triggered rung again reports the CHILD's fractions beside "
            "the pool's pressure -- the contradiction the prose line avoids"
        ),
    ),
    Reversion(
        target=_SESSION,
        find="""            self._notify_budget_rung(rung, origin, detail, changes)""",
        replace="""            self._notify_budget_rung(rung, origin, detail, dict(rung.model_tiers))""",
        test="test_tier_changes_report_what_rebound_not_what_was_declared",
        because="the event reports what the rung ASKED for rather than what it did",
    ),
]


_TIER_OVERLAY = {"planner": {"model": "flash"}}


def _notifying_session(**kw):
    """``_session`` with the typed callback installed and payloads collected."""
    s = _session(**kw)
    s._payloads = []
    s._on_budget_rung = s._payloads.append
    return s


def _prose_session(**kw):
    """``_session`` with the legacy prose channel captured."""
    s = _notifying_session(**kw)
    s._prose = []
    s._current_output_callback = (
        lambda source, text, mode: s._prose.append((source, text))
    )
    return s


# --------------------------------------------------------------------------
# B — the vocabulary
# --------------------------------------------------------------------------

def test_notify_is_a_valid_action_and_not_a_terminal_one():
    assert ACTION_NOTIFY in VALID_ACTIONS
    assert ACTION_NOTIFY not in TERMINAL_ACTIONS
    assert TERMINAL_ACTIONS == {ACTION_FINALIZE, ACTION_ABORT, "escalate"}


def test_a_notify_rung_parses():
    cfg = _cfg(limits={"tokens": 100},
               degrade=[{"at": 60, "action": "notify"}])
    assert [(r.at_percent, r.action) for r in cfg.degrade] == [(60.0, "notify")]


def test_a_rung_declaring_nothing_is_still_refused():
    """The alternative the issue floated is NOT already-legal config.

    ``from_dict`` rejects a rung carrying neither an overlay nor an action,
    which is why ``notify`` had to become a real action rather than the bare
    form being reinterpreted.
    """
    with pytest.raises(BudgetControlConfigError, match="does nothing"):
        _cfg(limits={"tokens": 100}, degrade=[{"at": 60}])


def test_has_abort_rung_is_unaffected_by_notify():
    """#947's ``budget_limits_without_abort`` finding reads this predicate.

    A ladder of pure checkpoints must not start looking like one that stops
    the run.
    """
    checkpoints = _cfg(limits={"tokens": 100},
                       degrade=[{"at": 60, "action": "notify"},
                                {"at": 80, "action": "notify"}])
    assert not checkpoints.has_abort_rung

    with_abort = _cfg(limits={"tokens": 100},
                      degrade=[{"at": 60, "action": "notify"},
                               {"at": 100, "action": "abort"}])
    assert with_abort.has_abort_rung


def test_a_notify_rung_does_not_latch_a_terminal_action():
    s = _notifying_session(budget=_cfg(
        limits={"usd": 10},
        degrade=[{"at": 50, "action": "notify"}]))
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=6))

    assert s._budget_terminal_action is None
    assert s._budget_exhausted_reason is None
    assert not hasattr(s, "_stopped")


def test_a_finalize_rung_still_latches():
    s = _notifying_session(budget=_cfg(
        limits={"usd": 10},
        degrade=[{"at": 50, "action": "finalize"}]))
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=6))

    assert s._budget_terminal_action == "finalize"


def test_an_abort_rung_still_stops_the_session():
    s = _notifying_session(budget=_cfg(
        limits={"usd": 10},
        degrade=[{"at": 100, "action": "abort"}]))
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=10))

    assert s._budget_terminal_action == "abort"
    assert s._stopped
    assert "budget_exhausted" in (s._budget_exhausted_reason or "")


# --------------------------------------------------------------------------
# A — the typed event
# --------------------------------------------------------------------------

def test_the_callback_fires_once_per_applied_rung():
    s = _notifying_session(budget=_cfg(
        limits={"usd": 10},
        degrade=[{"at": 50, "action": "notify"}]))
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=6))

    assert len(s._payloads) == 1
    p = s._payloads[0]
    assert p["at_percent"] == 50.0
    assert p["action"] == "notify"
    assert p["origin"] == "self-enforced"
    assert "usd" in p["pressure"]


def test_two_rungs_crossed_at_once_notify_twice():
    s = _notifying_session(budget=_cfg(
        limits={"usd": 10},
        degrade=[{"at": 50, "action": "notify"},
                 {"at": 80, "action": "notify"}]))
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=9))

    assert [p["at_percent"] for p in s._payloads] == [50.0, 80.0]


def test_a_skipped_rung_notifies_nobody():
    """The backwards-rebind guard changed nothing, so it says nothing.

    Reporting it would tell a user the model was downgraded when it was not.
    """
    s = _notifying_session(budget=_cfg(
        limits={"usd": 10},
        degrade=[{"at": 50, "action": "notify"}]))
    s._budget_applied_rung_pct = 90.0        # a higher rung already applied

    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=6))

    assert s._payloads == []


def test_no_callback_installed_is_not_an_error():
    s = _session(budget=_cfg(
        limits={"usd": 10},
        degrade=[{"at": 50, "action": "notify"}]))
    assert s._on_budget_rung is None
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=6))


def test_a_raising_callback_does_not_break_the_rung():
    """A client-notification failure must never fail the turn behind it."""
    def _boom(_payload):
        raise RuntimeError("client went away")

    s = _session(budget=_cfg(
        limits={"usd": 10},
        degrade=[{"at": 100, "action": "abort"}]))
    s._on_budget_rung = _boom

    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=10))

    # The rung still did its job.
    assert s._budget_terminal_action == "abort"
    assert s._stopped


def test_a_self_enforced_rung_carries_structured_usage():
    s = _notifying_session(budget=_cfg(
        limits={"usd": 10, "tokens": 1000},
        degrade=[{"at": 50, "action": "notify"}]))
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=6))

    p = s._payloads[0]
    assert p["usage"] == pytest.approx({"usd": 0.6, "tokens": 0.0})
    assert p["driving_dimension"] == "usd"


def test_a_cascade_pushed_rung_omits_this_sessions_own_usage():
    """The pool crossed, not this session -- so its fractions are absent.

    Publishing the child's own numbers beside the pool's pressure is the
    exact contradiction the prose line already avoids ("degrading at 50%
    (tokens 32%)").  Absent means "not measured here", not "zero".
    """
    s = _notifying_session(budget=_cfg(
        limits={"usd": 10},
        degrade=[{"at": 50, "action": "notify"}]))
    fired = s._budget_tracker.observe(usd=6)

    s._payloads.clear()
    JaatoSession._apply_budget_rungs(
        s, fired, origin="cascade-pushed", pressure="pool: usd 91%")

    p = s._payloads[-1]
    assert p["origin"] == "cascade-pushed"
    assert p["pressure"] == "pool: usd 91%"
    assert "usage" not in p
    assert "driving_dimension" not in p


def test_tier_changes_report_what_rebound_not_what_was_declared():
    """Both ends, in ``overlay_tier_table``'s own ``old -> new`` shape.

    That function's docstring already names this event as its consumer, and
    it carries the old model because the table is mutated in place — nothing
    downstream can recover it afterwards.
    """
    s = _notifying_session(budget=_cfg(
        limits={"usd": 10},
        degrade=[{"at": 50, "model_tiers": _TIER_OVERLAY}]))
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=6))

    p = s._payloads[0]
    assert p["tier_changes"] == {"planner": "opus -> flash"}
    assert p["action"] is None


def test_a_noop_overlay_still_notifies_with_no_changes():
    """The rung fired and rebound nothing, and the event says exactly that.

    ``overlay_tier_table``'s own no-op semantics are covered by
    ``test_overlay_to_identical_binding_is_a_noop`` in ``test_budget_runtime``;
    what is asserted here is that the rung is still REPORTED — it crossed its
    threshold — while ``tier_changes`` stays empty.  The binding must match on
    provider as well as model, or the overlay is not a no-op.
    """
    s = _notifying_session(budget=_cfg(
        limits={"usd": 10},
        degrade=[{"at": 50, "model_tiers": {
            "planner": {"model": "opus", "provider": "openrouter"}}}]))
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=6))

    assert len(s._payloads) == 1
    assert s._payloads[0]["tier_changes"] == {}


def test_an_action_only_rung_reports_no_tier_changes():
    s = _notifying_session(budget=_cfg(
        limits={"usd": 10},
        degrade=[{"at": 50, "action": "notify"}]))
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=6))

    assert s._payloads[0]["tier_changes"] == {}


def test_an_overlay_without_tier_config_still_notifies():
    """The rung fired; it simply rebound nothing."""
    s = _notifying_session(budget=_cfg(
        limits={"usd": 10},
        degrade=[{"at": 50, "model_tiers": _TIER_OVERLAY}]))
    s._tier_config = None

    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=6))

    assert len(s._payloads) == 1
    assert s._payloads[0]["tier_changes"] == {}


# --------------------------------------------------------------------------
# The prose channel the issue did not know about
# --------------------------------------------------------------------------

def test_the_prose_channel_still_fires_for_a_terminal_action():
    """#1069's premise was that nothing reached the client.  This is what did."""
    s = _prose_session(budget=_cfg(
        limits={"usd": 10},
        degrade=[{"at": 50, "action": "finalize"}]))
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=6))

    assert any(src == "system" and "finalize" in text
               for src, text in s._prose)


def test_the_prose_channel_still_fires_for_a_brownout():
    s = _prose_session(model="opus", budget=_cfg(
        limits={"usd": 10},
        degrade=[{"at": 50, "model_tiers": _TIER_OVERLAY}]))
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=6))

    assert any(src == "system" and "degraded" in text
               for src, text in s._prose)


def test_a_notify_rung_surfaces_a_checkpoint_on_the_prose_channel():
    """Opt-in by construction: you get this line only by writing ``notify``.

    A checkpoint visible only through an event type shipped in this same
    change would be invisible in exactly the deployments asking for it.
    """
    s = _prose_session(budget=_cfg(
        limits={"usd": 10},
        degrade=[{"at": 50, "action": "notify"}]))
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=6))

    assert any(src == "system" and "checkpoint" in text
               for src, text in s._prose)


# --------------------------------------------------------------------------
# The wire
# --------------------------------------------------------------------------

def test_the_event_round_trips_over_the_wire():
    from jaato_sdk.events import BudgetRungFiredEvent, deserialize_event

    sent = BudgetRungFiredEvent(
        at_percent=85.0, action="notify", origin="self-enforced",
        pressure="tokens 85%", usage={"tokens": 0.85},
        driving_dimension="tokens", tier_changes={"planner": "flash"},
    )
    got = deserialize_event(sent.model_dump_json())

    assert isinstance(got, BudgetRungFiredEvent)
    assert got.at_percent == 85.0
    assert got.usage == {"tokens": 0.85}
    assert got.tier_changes == {"planner": "flash"}
    assert got.driving_dimension == "tokens"


def test_the_event_is_registered_for_deserialisation():
    """An event class the registry does not know raises on arrival."""
    from jaato_sdk.events import EventType, _EVENT_CLASSES, BudgetRungFiredEvent

    assert _EVENT_CLASSES[EventType.BUDGET_RUNG_FIRED.value] is BudgetRungFiredEvent


def test_absent_usage_survives_the_wire_as_absent():
    """``None`` must not become ``{}`` -- "not measured" is not "no limits"."""
    from jaato_sdk.events import BudgetRungFiredEvent, deserialize_event

    sent = BudgetRungFiredEvent(
        at_percent=50.0, origin="cascade-pushed", pressure="pool: usd 91%")
    got = deserialize_event(sent.model_dump_json())

    assert got.usage is None
    assert got.driving_dimension is None


def test_protocol_version_was_bumped_for_the_new_event():
    """A new event type is not a harmlessly additive field.

    ``deserialize_event`` raises on an unrecognised ``type``; the SDK reader
    catches it, logs and continues, so an older client loses the event and
    logs a line per rung.  Bounded -- and a version bump, not a silent
    addition.
    """
    from jaato_sdk.events import PROTOCOL_VERSION

    major, minor = PROTOCOL_VERSION.split(".")[:2]
    assert (int(major), int(minor)) >= (1, 8)


# --------------------------------------------------------------------------
# The tracker helpers the payload is built from
# --------------------------------------------------------------------------

def test_pressure_by_dimension_omits_undeclared_dimensions():
    t = BudgetTracker(_cfg(limits={"usd": 10}))
    t.observe(usd=5, tokens=999)

    pressure = t.pressure_by_dimension()
    assert pressure == pytest.approx({"usd": 0.5})
    assert "tokens" not in pressure          # declared nowhere -> absent, not 0.0


def test_pressure_by_dimension_is_unclamped():
    t = BudgetTracker(_cfg(limits={"usd": 10}))
    t.observe(usd=25)

    assert t.pressure_by_dimension()["usd"] == pytest.approx(2.5)


def test_driving_dimension_names_the_highest_fraction():
    t = BudgetTracker(_cfg(limits={"usd": 10, "tokens": 1000}))
    t.observe(usd=1, tokens=900)

    assert t.driving_dimension() == "tokens"


def test_driving_dimension_is_none_when_nothing_is_measured():
    """The defensive branch, exercised without inventing a shape.

    ``BudgetControlConfig.from_dict({"limits": {}})`` is ``None`` — an empty
    budget is no budget — and a tracker is never built around that in
    production (``usage_fraction`` raises on it identically, and has since
    long before this change).  So the empty-pressure branch is reached
    through the method it actually depends on.
    """
    t = BudgetTracker(_cfg(limits={"usd": 10}))
    t.pressure_by_dimension = lambda: {}

    assert BudgetTracker.driving_dimension(t) is None
