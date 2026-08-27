"""``finish_reason != "stop"`` is the obvious completeness rule and it is wrong.

A profile with a ``completion_payload_schema`` ends by calling
``signal_completion``, which terminates the session INSIDE a tool-use turn.
So a COMPLETE run's terminal turn reports ``"tool_use"`` and no later turn
ever says ``"stop"``.

``TurnCompletedEvent.finish_reason``'s own field comment recommended that
branch.  A consumer copied it into two graders and blocked every
schema-driven arm as truncated, with the finished artefact sitting on disk
beside the verdict.  The correct statement existed in the repo the whole
time — in ``scaffold/_client_templates.py``, which a reader only sees if
they happen to run the generator.

So the rule ships as a FUNCTION rather than as prose in a field a reader
will believe.  Written by the consumer who got it wrong first; upstreamed so
their copy can be deleted rather than kept in sync.
"""

from __future__ import annotations

import pytest

from jaato_sdk.helpers import truncation_reason


# ------------------------------------------------------- reached a terminus

def test_a_completion_payload_settles_it_regardless_of_finish_reason():
    """The whole schema-profile case, and the one the old advice broke."""
    assert truncation_reason(
        finish_reason="tool_use",
        payload={"status": "done"},
    ) is None


def test_a_plain_prose_turn_settles_on_stop():
    assert truncation_reason(finish_reason="stop") is None


def test_an_empty_payload_is_still_a_payload():
    """``{}`` is a declared terminus; ``None`` is the absence of one.

    Collapsing them would block an agent whose schema legitimately admits an
    empty object — absent-versus-empty, in the field that decides verdicts.
    """
    assert truncation_reason(finish_reason="tool_use", payload={}) is None


# ---------------------------------------------------- did not reach one

def test_tool_use_without_a_payload_names_the_mechanism():
    reason = truncation_reason(finish_reason="tool_use")

    assert reason is not None
    assert "tool_use" in reason
    assert "no completion payload" in reason


def test_an_unknown_finish_reason_is_named_not_guessed():
    reason = truncation_reason(finish_reason="max_tokens")

    assert reason == "finish_reason='max_tokens'", (
        "an unrecognised finish reason must be reported AS itself; inventing "
        "a category for it is how a grader reports something it did not see"
    )


# --------------------------------------------- termination outranks all

def test_a_budget_ceiling_outranks_a_payload():
    """The ordering that is load-bearing.

    A budget refusal short-circuits BEFORE any turn runs, so ``finish_reason``
    still holds whatever the previous turn left — and a payload cannot mean
    "complete" if the session then refused.
    """
    reason = truncation_reason(
        finish_reason="stop",
        payload={"status": "done"},
        termination_reason="budget_exhausted",
        termination_detail="tokens: 0 remaining",
    )

    assert reason is not None
    assert "budget ceiling" in reason
    assert "tokens: 0 remaining" in reason


def test_a_session_error_outranks_a_payload():
    reason = truncation_reason(
        finish_reason="stop",
        payload={"status": "done"},
        termination_reason="error",
        termination_detail="provider 500",
    )

    assert reason is not None
    assert "session error" in reason
    assert "provider 500" in reason


@pytest.mark.parametrize("termination_reason,fragment", [
    ("budget_exhausted", "budget ceiling"),
    ("error", "session error"),
])
def test_a_terminal_reason_without_detail_still_names_itself(
    termination_reason, fragment,
):
    """Detail is optional; the mechanism is not.

    A caller quoting this string into a verdict must never end up quoting
    nothing.
    """
    reason = truncation_reason(
        finish_reason="stop", termination_reason=termination_reason,
    )

    assert reason is not None and fragment in reason


def test_a_natural_termination_does_not_override_a_payload():
    """Only ``budget_exhausted`` and ``error`` outrank.

    ``reason="natural"`` is what a completion-gated session emits ON SUCCESS
    — treating every termination as disqualifying would block exactly the
    sessions that finished correctly.
    """
    assert truncation_reason(
        finish_reason="tool_use",
        payload={"status": "done"},
        termination_reason="natural",
    ) is None


# ------------------------------------------------------------- the field

def test_the_field_comment_no_longer_recommends_the_wrong_branch():
    """The prose that caused this must not come back.

    Checked in the source because the defect was never in the code — it was
    in a sentence a reader had no reason to doubt.
    """
    import pathlib

    src = pathlib.Path(
        "jaato-sdk/jaato_sdk/events.py").read_text(encoding="utf-8")

    assert "deterministically branch on ``finish_reason != \"stop\"``" not in src, (
        "the finish_reason comment again tells clients to branch on "
        "!= 'stop'; that is wrong for every completion-gated profile"
    )
    assert "truncation_reason" in src, (
        "the field no longer points at the rule that supersedes it"
    )
