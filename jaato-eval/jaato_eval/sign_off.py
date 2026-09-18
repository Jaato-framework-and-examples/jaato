"""Which error terminals still leave something to grade.

BLOCKED asserts exactly one thing, and both places that define it say the
same sentence: *there was nothing to grade*.
:attr:`~jaato_eval.arm.ArmResult.blocked_reason` — "set when the arm
itself never ran"; :mod:`jaato_eval.runner` — "BLOCKED means 'we learned
nothing about the configuration', NOT 'this configuration failed'".

Most error terminals fit that.  A daemon that died mid-turn, a runner
that crashed, a provider that stopped answering — each leaves a tree
whose state nobody can vouch for, and grading it would report on an
interruption rather than on the agent.

One does not.  When the framework's completion-nudge budget runs out
(``NudgeExhausted``), the agent ran, worked, and left a workspace on
disk; the only thing missing is its sign-off.  Recording that as
BLOCKED discards a verdict twice over (jaato #773):

* a passing tree is reported as unmeasured, so a success is lost;
* a failing tree is reported as unmeasured, so it leaves the pass-rate
  denominator entirely — ``report.py`` excludes blocked arms by design —
  and an arm that genuinely failed silently *improves* the model's
  score.  That is a measurement bias, not just a missing row.

So this module owns the one rule that separates the two, in one place,
because the runner (deciding whether to keep going) and the graders
(deciding which of them are still valid on an unsigned arm) must not
each carry their own copy of it.

WHAT AN UNSIGNED TERMINAL DOES *NOT* EXCUSE
===========================================

Grading such an arm is per-grader, never per-arm.  The missing sign-off
invalidates precisely the graders that read the sign-off:

* a **script** grader runs a command against the workspace — the tree is
  real, so it stays valid and returns a verdict;
* a **processor** grader validates the completion payload — there is no
  payload, so it BLOCKS (and says so in those words);
* a **judge** grader is handed the payload first and the workspace
  second, so it BLOCKS rather than score a rubric on half its input.

An arm whose manifest carries a payload-reading grader therefore still
rolls up BLOCKED unless something FAILs — :meth:`Report.state` holds
that an unexercised grader has not established the claim.  What changes
is that the reason now names the missing sign-off instead of blaming the
daemon, and that a workspace-only manifest gets its verdict back.
"""
from __future__ import annotations

from typing import Optional

#: How many times the framework re-prompts a session that ended without
#: calling ``signal_completion`` before giving up and producing the
#: terminal below.  ``server/core.py``'s ``MAX_COMPLETION_NUDGES``,
#: restated here because the engine must not import ``server.*`` and
#: because the number is what makes a reported ``2`` mean "at the
#: ceiling, one nudge from NudgeExhausted" rather than merely "twice".
#:
#: It lives beside :data:`UNSIGNED_TERMINALS` deliberately: this is the
#: budget whose exhaustion produces that terminal, and the runner (which
#: counts nudges) and the report (which renders them against the ceiling)
#: must not each carry their own copy of the number.
#:
#: If the framework raises its ceiling, this constant is stale in exactly
#: one direction — an arm reported at the ceiling that had a nudge left.
#: Since #919 that ceiling is a PER-PROFILE knob
#: (``max_completion_nudges:``, defaulting to the framework's
#: ``shared.completion_nudge.DEFAULT_MAX_COMPLETION_NUDGES`` = 2), so an arm
#: whose profile raised its own budget is reported against this default
#: rather than against the budget it actually had — the same one-directional
#: staleness, now reachable without a framework release.
MAX_COMPLETION_NUDGES = 2

#: The terminal a DRIVER arm (``harness.kind: driver``, jaato #1110) ends
#: in when its process exits with a code that is neither ``0`` nor
#: ``EX_TEMPFAIL``.  Not a daemon ``error_type`` — the engine mints it,
#: because a driver's exit code is the only account the engine has of how
#: the run ended, and the exit-code vocabulary in :mod:`jaato_eval.driver`
#: says such an exit means *ran and stopped short*: the tree on disk is
#: real, there is no sign-off, and the graders sort themselves by that
#: exactly as they do for a session that spent its nudge budget.
DRIVER_STOPPED_SHORT = "DriverStoppedShort"

#: Terminal types whose workspace is still worth grading.  Deliberately
#: named rather than pattern-matched: every other terminal keeps the
#: conservative reading, and widening this is a decision someone has to
#: make explicitly.  Two members, each the one unsigned terminal of its
#: harness kind — ``NudgeExhausted`` for a session arm, whose daemon says
#: it; :data:`DRIVER_STOPPED_SHORT` for a driver arm, whose exit code says
#: it.
UNSIGNED_TERMINALS = frozenset({"NudgeExhausted", DRIVER_STOPPED_SHORT})


def describe_unsigned(error_type: Optional[str]) -> str:
    """One clause naming what the missing sign-off WAS, for a verdict.

    The three graders each say why a payload-reading verdict is BLOCKED,
    or why a workspace verdict carries a caveat, and before the driver
    kind every such sentence read "the agent never called
    signal_completion".  That is false of a driver arm — the process
    exited non-zero; no agent was asked to sign anything — and a verdict
    that names the wrong mechanism sends its reader to check a completion
    schema that was never involved.  One function, so the graders cannot
    each carry their own paraphrase of the rule this module owns.
    """
    if error_type == DRIVER_STOPPED_SHORT:
        return f"the driver stopped short of its end ({error_type})"
    return f"the agent never called signal_completion ({error_type})"


def is_unsigned_terminal(error_type: Optional[str]) -> bool:
    """Did the agent finish working and merely fail to sign off?

    Args:
        error_type: The daemon's ``error_type`` — off
            ``SessionTerminatedEvent`` / ``AgentErrorEvent``, or off the
            :class:`jaato_sdk.client.convenience.AgentError` those raise
            through.  ``None`` for a failure the daemon never typed.

    Returns:
        ``True`` when the workspace this session left is still evidence
        about the configuration under test.

    The type is READ, never inferred.  A failure the daemon did not type
    is not guessed at — the same rule
    :func:`jaato_eval.runner._describe_session_failure` follows, and for
    the same reason: an untyped failure that is *treated* as a known one
    is how a broken daemon starts producing verdicts.
    """
    return bool(error_type) and str(error_type) in UNSIGNED_TERMINALS
