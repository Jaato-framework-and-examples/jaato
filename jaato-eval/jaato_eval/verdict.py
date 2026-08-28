"""Three-state verdicts.

PROVENANCE
==========

Lifted from ``jaato-cascade-coordination-example/certify/verdict.py``.
That repository is not installable, so this is a **vendored copy** — the
exact duplication its own README warns about::

    Any second copy of a fact rots unless something executes the
    comparison — and the copy that rots is the one that cannot fail.

``tests/test_verdict.py`` executes the comparison for the part that
matters (the state set and the exit-code mapping), so the copy can fail.
The durable fix is to extract this module into a package both repos
depend on; until then, treat ``certify/verdict.py`` as canonical and this
as a mirror.

WHY THREE STATES
================

An evaluation has many ways to produce no signal: the fixture failed to
materialise, the daemon was stale, the provider returned 429, a budget
ceiling tripped, the model hit ``max_tokens`` mid-payload.  Scoring any
of those as ``FAIL`` corrupts the comparison being run — you conclude the
cheap model is worse when its provider merely rate-limited you.

``PASS``     the claim was exercised and held.
``FAIL``     the claim was exercised and was violated.  A real defect.
``BLOCKED``  nothing was exercised.  NOT a pass, and NOT a failure of the
             thing under test.  Excluded from pass-rate denominators.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List

PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"

#: The complete state set.  Anything outside it is a programming error.
STATES = (PASS, FAIL, BLOCKED)

_GLYPH = {PASS: "✓", FAIL: "✘", BLOCKED: "○"}


@dataclass
class Verdict:
    """The outcome of one grader, with its evidence attached.

    Attributes:
        grader_id: Which grader produced this (``kind:identifier``).
        claim: What was being checked, in one line.
        state: One of :data:`STATES`.
        detail: One-line summary of the outcome.
        evidence: Lines supporting the verdict — command output, the
            errors a processor returned, a judge's reasoning.  Rendered
            indented beneath the verdict.
        blocked_reason: Why nothing was exercised.  **Required** when
            ``state`` is ``BLOCKED`` — a BLOCKED verdict that does not say
            what was absent is indistinguishable from a silent skip, which
            is the failure mode this state exists to prevent.
    """

    grader_id: str
    claim: str
    state: str
    detail: str = ""
    evidence: List[str] = field(default_factory=list)
    blocked_reason: str = ""

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ValueError(
                f"{self.state!r} is not a verdict state; expected one of {STATES}")
        if self.state == BLOCKED and not self.blocked_reason:
            raise ValueError(
                f"BLOCKED verdict {self.grader_id!r} carries no blocked_reason. "
                "A BLOCKED state that does not name what was absent is a "
                "silent skip wearing a verdict's clothes.")

    def note(self, line: str) -> None:
        self.evidence.append(line)

    def render(self) -> str:
        body = [f"{_GLYPH[self.state]} {self.grader_id} {self.state}: {self.claim}"]
        if self.detail:
            body.append(f"    {self.detail}")
        if self.state == BLOCKED:
            body.append(f"    blocked: {self.blocked_reason}")
        for line in self.evidence:
            body.append(f"      | {line}")
        return "\n".join(body)

    def to_dict(self) -> Dict[str, object]:
        return {
            "grader_id": self.grader_id,
            "claim": self.claim,
            "state": self.state,
            "detail": self.detail,
            "evidence": list(self.evidence),
            "blocked_reason": self.blocked_reason,
        }


@dataclass
class Report:
    """A collection of verdicts with an exit code."""

    verdicts: List[Verdict] = field(default_factory=list)

    def add(self, v: Verdict) -> Verdict:
        self.verdicts.append(v)
        return v

    def counts(self) -> Dict[str, int]:
        out = {PASS: 0, FAIL: 0, BLOCKED: 0}
        for v in self.verdicts:
            out[v.state] += 1
        return out

    def state(self) -> str:
        """Roll up to one state.

        Any FAIL makes the whole thing FAIL.  Otherwise any BLOCKED makes
        it BLOCKED — a run where one grader could not be exercised has not
        established the claim, even if every other grader passed.  Only an
        all-PASS report (with at least one verdict) is PASS.
        """
        c = self.counts()
        if c[FAIL]:
            return FAIL
        if c[BLOCKED] or not self.verdicts:
            return BLOCKED
        return PASS

    def render(self) -> str:
        lines = [v.render() for v in self.verdicts]
        c = self.counts()
        lines.append("")
        lines.append(
            f"{c[PASS]} passed, {c[FAIL]} failed, {c[BLOCKED]} blocked "
            f"(blocked = not exercised; NOT a pass)")
        return "\n".join(lines)

    def exit_code(self) -> int:
        """FAIL is 1.  BLOCKED is 2 — distinct, so CI cannot read a
        not-exercised run as success."""
        c = self.counts()
        if c[FAIL]:
            return 1
        if c[BLOCKED] or not self.verdicts:
            return 2
        return 0


def emit(report: Report, stream=sys.stdout) -> int:
    print(report.render(), file=stream)
    return report.exit_code()
