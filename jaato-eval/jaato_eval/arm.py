"""One arm of a sweep: the unit that gets run and scored.

An *arm* is one (task, profile set, repeat) triple.  It is the row in the
results file and the cell in the pivot.  Keeping it as an explicit value
rather than three loose loop variables is what lets the sweep driver
resume, shard, and report without re-deriving identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .manifest import TaskManifest
from .provenance import provenance
from .verdict import BLOCKED, Report, Verdict


@dataclass(frozen=True)
class ArmSpec:
    """What to run.

    Attributes:
        task: The manifest.
        profile_set: The profile set this arm exercises — the model /
            provider axis.  ``None`` uses whatever the manifest declares.
        repeat: 0-based repeat index.  Repeats of the same arm differ only
            by the workspace they get and the sampling the provider does,
            which is exactly what makes them a flakiness measurement.
    """

    task: TaskManifest
    profile_set: Optional[str]
    repeat: int

    @property
    def arm_id(self) -> str:
        """Stable identity: ``task@set#repeat``."""
        return f"{self.task.task_id}@{self.profile_set or 'default'}#{self.repeat}"


@dataclass
class ArmResult:
    """What one arm produced.

    Attributes:
        spec: The arm that was run.
        verdicts: One per grader, in manifest order.
        usage: Token and cost breakdown, summed across turns.  ``cost_usd``
            is ``None`` when neither the provider nor a pricing table knew
            — consumers must not read that as free.
        duration_seconds: Wall clock for the agent run, excluding grading.
        turns: Turns consumed.
        finish_reason: Provider finish reason for the terminal turn.
        payload_hash: Canonical-JSON sha256 of the completion payload.
            Repeats sharing a hash produced byte-identical output; the
            count of distinct hashes across repeats is the determinism
            measurement.
        error: Terminal error text, when the arm ended in an error the
            engine graded through anyway.  Today that is exactly the
            unsigned terminal (:mod:`jaato_eval.sign_off`): the agent
            worked, left a workspace, and never called
            ``signal_completion``.  Set ALONGSIDE a real ``state`` rather
            than instead of one — an ``error`` with ``blocked_reason``
            unset is the record that the arm produced evidence and ended
            badly, which is the distinction jaato #773 exists to keep.
        blocked_reason: Set when the arm itself never ran (fixture failed,
            daemon unreachable, budget tripped).  Distinct from a grader
            being blocked — this means there was nothing to grade.  An
            error terminal is therefore NOT automatically a blocked_reason:
            see ``error`` for the one that leaves a workspace behind.

    THE PROVENANCE BLOCK
    ====================

    Everything below this line answers "what happened to THIS arm, and can
    I go look at it upstream?" rather than "which configuration won".  The
    pivot ignores all of it; the per-arm report (:mod:`jaato_eval.report_html`)
    is made of it.

    Every one of these fields is ``Optional`` and ``None`` means UNKNOWN,
    never a default.  A zero ceiling and an unread ceiling are opposite
    facts, and so are "the model called no nudge" and "we could not count".

    Attributes:
        session_id: The daemon's id for this arm's session.  The runner has
            always known it and used to discard it, which left nothing in
            the results file identifying the session — and OpenRouter's
            console groups by exactly this id, so persisting it turns every
            row into a join onto the provider's own record of the arm
            (request count, upstream, per-request cost, generation ids).
        model: The model the daemon actually BOUND, from
            ``SessionInfoEvent.model_name``.  Not ``profile_set``, which is
            a naming convention rather than data — ``openrouter_gemini25flash``
            is what someone called the directory.
        provider: Likewise ``SessionInfoEvent.model_provider`` — the jaato
            provider plugin (``openrouter``, ``anthropic``, ...).
        upstream_provider: WHO THE GATEWAY ROUTED TO.  OpenRouter serves one
            model from several upstreams (observed: ``Google Vertex`` for
            Gemini 2.5 Flash), and an arm served by a different upstream is
            not the same measurement.  Only the provider knows this and the
            framework does not yet carry it off the wire, so this is
            ``None`` on every arm today; the runner reads it opportunistically
            so it populates itself the day the turn event reports it (the
            same plumbing jaato #766 needs for ``native_finish_reason``).
        native_finish_reason: The upstream's own finish word, behind
            OpenRouter's normalised one — Gemini's ``MALFORMED_FUNCTION_CALL``
            is the case that sent two arms to the provider's API for an
            explanation.  ``None`` until jaato #766 surfaces it; read
            opportunistically for the same reason as ``upstream_provider``.
        completion_nudges: How many times the framework asked this session
            to call ``signal_completion`` before giving up.  An arm sitting
            at the ceiling is one nudge from BLOCKED, and today that is
            visible only by grepping ``COMPLETION_NUDGE`` out of session
            logs — which is how three BLOCKED arms were finally explained.
        budget_ceiling: The arm's OWN ``budget_control.limits``, resolved
            from the profile it bound (:mod:`jaato_eval.profile`).  ``None``
            when the profile declares none: such a session draws on the task
            pool instead, which is the framework's rule and the reason both
            fields exist side by side.
        pool_limits: The task's cascade-pool ceilings, as declared in the
            manifest — the aggregate this arm's siblings shared.
        pool_on_arrival: What the pool had left when this arm STARTED
            (``{declared, limits, remaining, usage_fraction, pressure}``
            from ``cascade.budget.get``).  The column that makes an arm
            killed by an earlier arm's appetite readable as such: three arms
            on one $6.00 pool spent $3.81 + $0.17 + $2.03, and the third was
            terminated ``budget_exhausted`` and recorded BLOCKED — which
            reads as a model failure until you can see it arrived with 63%
            already gone.
    """

    spec: ArmSpec
    verdicts: List[Verdict] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    turns: int = 0
    finish_reason: str = "stop"
    payload_hash: Optional[str] = None
    error: Optional[str] = None
    blocked_reason: Optional[str] = None
    session_id: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    upstream_provider: Optional[str] = None
    native_finish_reason: Optional[str] = None
    completion_nudges: Optional[int] = None
    budget_ceiling: Optional[Dict[str, float]] = None
    pool_limits: Optional[Dict[str, float]] = None
    pool_on_arrival: Optional[Dict[str, Any]] = None

    @property
    def report(self) -> Report:
        return Report(verdicts=list(self.verdicts))

    @property
    def state(self) -> str:
        """Roll-up state for this arm.

        An arm that never ran is BLOCKED regardless of what its (absent)
        graders would have said.
        """
        if self.blocked_reason:
            return BLOCKED
        return self.report.state()

    def to_dict(self) -> Dict[str, Any]:
        """Flat record for the JSONL results file."""
        return {
            "arm_id": self.spec.arm_id,
            "task_id": self.spec.task.task_id,
            "profile_set": self.spec.profile_set,
            "repeat": self.spec.repeat,
            "state": self.state,
            "blocked_reason": self.blocked_reason,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "usage": self.usage,
            "duration_seconds": self.duration_seconds,
            "turns": self.turns,
            "finish_reason": self.finish_reason,
            "payload_hash": self.payload_hash,
            "error": self.error,
            # The per-arm provenance block.  Written unconditionally, nulls
            # included: a reader must be able to tell a field this engine
            # could not establish from a field a newer engine added, and an
            # omitted key looks like the latter.
            "session_id": self.session_id,
            "model": self.model,
            "provider": self.provider,
            "upstream_provider": self.upstream_provider,
            "native_finish_reason": self.native_finish_reason,
            "completion_nudges": self.completion_nudges,
            "budget_ceiling": self.budget_ceiling,
            "pool_limits": self.pool_limits,
            "pool_on_arrival": self.pool_on_arrival,
            # Which code this arm actually exercised.  The branch does not
            # determine it — see results.provenance.
            "provenance": provenance(),
        }
