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
        error: Terminal error text, when the arm ended in an error.
        blocked_reason: Set when the arm itself never ran (fixture failed,
            daemon unreachable, budget tripped).  Distinct from a grader
            being blocked — this means there was nothing to grade.
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
            # Which code this arm actually exercised.  The branch does not
            # determine it — see results.provenance.
            "provenance": provenance(),
        }
