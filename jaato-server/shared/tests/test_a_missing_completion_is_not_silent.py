"""A stage that was asked twice to signal completion, and never did, says so.

THE GAP.  When a session's profile puts ``signal_completion`` in its tool
surface and the model ends its loop without calling it, the framework nudges
it -- up to ``max_completion_nudges`` (default 2).  If it still never calls
it, the turn ends without a typed payload.  A cascade driver waiting for one
has to INVENT a reason for its absence, and the reason reached for is "the
profile must not declare a completion schema" -- which is the one thing that
was fine.  That misdirection is the defect; it cost a downstream consumer an
afternoon inspecting a correct schema.

WHAT ACTUALLY SIGNALS IT (#771).  This docstring used to say that no
``AgentCompletedEvent`` and no ``SessionTerminatedEvent`` fire, and that
``TurnCompletedEvent.completion_gap`` is therefore "the ONLY event that fires
on this path".  Both halves were wrong:

  * a terminal DOES fire -- ``server/core.py`` emits ``ErrorEvent`` and
    ``_emit_error_termination`` with ``error_type="NudgeExhausted"``; and
  * ``completion_gap`` does NOT reach a consumer on this path.  Its sole
    writer in the server runs after ``on_agent_turn_completed`` has built the
    turn event, read the field and cleared it, and the session then ends.

So the consumer signal is the ``NudgeExhausted`` terminal, which is typed,
terminal and unconditional; ``jaato_eval.sign_off`` already routes on it.
The tests below still pin the CARRIER (the field round-trips, and a clean
turn reports nothing) because the wire shape is released and must not drift
-- but carriage is not delivery, and
``test_a_nudge_exhausted_session_is_announced.py`` is the guard that the
signal a consumer is told to watch is really emitted.
"""

import ast
import pathlib


from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

#: The defect, put back.  ``not should_nudge`` alone is ALSO true when the
#: agent signalled, so this stamps a completion gap on clean completions.
REVERSIONS = [
    Reversion(
        target="jaato-server/server/core.py",
        find="""                if (
                    status == "done"
                    and signal_completion_in_surface
                    and not should_nudge
                    and nudges_fired >= MAX_COMPLETION_NUDGES
                ):""",
        replace="""                if (
                    not should_nudge
                ):""",
        test="TestGiveUpPredicate::test_the_gap_is_not_set_on_a_bare_should_nudge_false",
        because="a predicate that cannot tell 'budget spent' from 'agent signalled'",
    ),
]

CORE = (pathlib.Path(__file__).resolve().parents[2]
        / "server" / "core.py")
GAP = "not_signalled_after_nudges"


# ---------------------------------------------------------------------------
# Event contract
# ---------------------------------------------------------------------------

class TestEventContract:
    def test_a_normal_turn_reports_no_gap(self):
        from jaato_sdk.events import TurnCompletedEvent

        assert TurnCompletedEvent(agent_id="main").completion_gap is None

    def test_the_gap_is_carried_on_the_event(self):
        from jaato_sdk.events import TurnCompletedEvent

        evt = TurnCompletedEvent(agent_id="main", completion_gap=GAP)
        assert evt.completion_gap == GAP

    def test_the_gap_survives_the_wire_round_trip(self):
        """A contract carried only by a Python attribute dies at the wire."""
        from jaato_sdk.events import TurnCompletedEvent

        evt = TurnCompletedEvent(agent_id="main", completion_gap=GAP)
        restored = TurnCompletedEvent(**evt.to_dict())
        assert restored.completion_gap == GAP, (
            "completion_gap did not survive to_dict() -- a consumer reading "
            "the deserialized event would see None and be back to guessing"
        )


# ---------------------------------------------------------------------------
# The per-agent slot
# ---------------------------------------------------------------------------

class TestAgentStateSlot:
    def test_agent_state_starts_with_no_gap(self):
        from server.core import AgentState

        agent = AgentState(agent_id="a", name="n", agent_type="main")
        assert agent.completion_gap is None


# ---------------------------------------------------------------------------
# Structural guards on core.py
#
# The give-up point sits inside ``_start_model_thread``'s model loop, which
# cannot be driven from a unit test without standing up a provider, a runner
# and an RPC transport.  These assert the two properties that were actually
# easy to get wrong, and both were: the predicate that decides "gave up", and
# the read-AND-CLEAR that keeps the gap on one event.
#
# Count/identity claims only -- never "the first X after Y" (``ast.walk`` is
# breadth-first).  Anchors asserted first so a rename fails as STALE.
# ---------------------------------------------------------------------------

def _core_tree():
    return ast.parse(CORE.read_text(encoding="utf-8"))


def _assignments_of_attr(tree, attr):
    return [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Attribute) and t.attr == attr
    ]


class TestGiveUpPredicate:
    def test_core_still_has_the_pieces_this_guard_inspects(self):
        """Anchor. Without it the assertions below pass on an empty match."""
        assert CORE.exists(), f"{CORE} not found — guard is stale, not passing"
        src = CORE.read_text(encoding="utf-8")
        assert "MAX_COMPLETION_NUDGES" in src, (
            "the nudge ceiling is gone — this guard can no longer find what "
            "it inspects and must be re-aimed, NOT deleted"
        )
        assert _assignments_of_attr(_core_tree(), "completion_gap"), (
            "nothing assigns .completion_gap — re-aim the guard or explain "
            "how the gap reaches the event without it"
        )

    def test_the_gap_is_not_set_on_a_bare_should_nudge_false(self):
        """``should_nudge`` False ALSO means "the agent signalled".

        ``try_completion_nudge`` returns ``(False, current)`` both when the
        budget is spent AND when the agent already called
        ``signal_completion``.  Keying the gap on ``not should_nudge`` alone
        would therefore stamp it on every clean completion -- reporting a
        failure to signal on exactly the sessions that DID signal.

        The count is what separates them, so the predicate must consult it.
        """
        src = CORE.read_text(encoding="utf-8")
        marker = f'completion_gap = "{GAP}"'
        assert marker in src, (
            f"no assignment of {GAP!r} — guard is stale, re-aim it"
        )
        before = src[:src.index(marker)]
        # the guarding `if` is the last one opened before the assignment
        condition = before[before.rindex("if ("):]
        assert "nudges_fired" in condition, (
            "the give-up predicate does not consult nudges_fired, so it "
            "cannot distinguish 'the budget is spent' from 'the agent "
            "signalled' — both return should_nudge=False. As written it "
            "would report a completion gap on successful completions."
        )
        assert "signal_completion_in_surface" in condition, (
            "the give-up predicate does not check whether signal_completion "
            "was even in the tool surface. Interactive sessions filter it "
            "out and are never nudged; flagging them would report a gap on "
            "every ordinary chat turn."
        )


class TestGapRidesExactlyOneEvent:
    def test_the_gap_is_cleared_when_it_is_read(self):
        """Otherwise a session that goes on working re-reports the same gap.

        The give-up is a property of ONE turn. Left set, it would ride every
        subsequent TurnCompletedEvent of a session that was nudged, gave up,
        and then received more work -- turning a real one-turn signal into a
        permanent false one.
        """
        assigns = _assignments_of_attr(_core_tree(), "completion_gap")
        assert assigns, "nothing assigns .completion_gap — guard is stale"

        clears = [
            n for n in assigns
            if isinstance(n.value, ast.Constant) and n.value.value is None
        ]
        assert clears, (
            "no site ever sets .completion_gap back to None. The gap would "
            "persist for the life of the agent record and be re-reported on "
            "every later turn."
        )
