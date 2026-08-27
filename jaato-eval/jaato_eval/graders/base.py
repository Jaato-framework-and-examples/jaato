"""The grader contract.

A grader takes everything one arm produced and returns exactly one
:class:`~jaato_eval.verdict.Verdict`.  It must never raise for a
condition it can describe: an adapter that cannot run returns BLOCKED
with a reason, because an exception escaping into the sweep driver would
take down an arm that might still have had other graders to run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from ..ledger import LedgerResult
from ..manifest import GraderSpec
from ..verdict import Verdict


@dataclass
class GraderContext:
    """Everything one arm produced, handed to every grader.

    Attributes:
        workspace_path: The mutated scratch workspace.  Ground truth for
            anything the agent claims to have written.
        config_root: The read-only task definition (``.jaato/``).
        agent_params: The persona parameters this arm ran with.
        payload: The typed ``signal_completion`` payload, or ``None`` when
            the profile declared no schema or the agent never completed.
        ledger: Reconstructed tool-call ledger.  Check ``ledger.faithful``
            before grading on it.
        history: Raw serialized history, for graders that want it.
        usage: The arm's ``UsageBreakdown`` as a dict (tokens, cost).
        finish_reason: Provider finish reason for the terminal turn.
            Do NOT read this directly to decide whether the arm ran to
            completion — use :attr:`truncation_reason`.  ``"tool_use"``
            means two opposite things (see that property).
        turns: How many turns the arm consumed.
        error: Terminal error text, when the arm ended in an error.
        prior_verdicts: ``grader_id`` -> state, for graders already run
            on this arm.  The runner fills this in as it goes, so a
            later grader can gate on an earlier one's outcome (see
            ``JudgeGrader``'s ``gate_on``).  Order in the manifest is
            therefore significant for gated graders.
    """

    workspace_path: Path
    config_root: Path
    agent_params: Dict[str, Any] = field(default_factory=dict)
    payload: Optional[Dict[str, Any]] = None
    ledger: LedgerResult = field(default_factory=LedgerResult)
    history: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    finish_reason: str = "stop"
    turns: int = 0
    error: Optional[str] = None
    prior_verdicts: Dict[str, str] = field(default_factory=dict)

    @property
    def truncation_reason(self) -> Optional[str]:
        """Why this arm never reached a terminus it declared, or ``None``.

        ``finish_reason`` alone cannot answer this, because ``"tool_use"``
        carries two OPPOSITE meanings.  The provider emits it for "stopped
        to execute tools" — mid-loop, more turns expected.  But a profile
        with a ``completion_payload_schema`` ends by calling
        ``signal_completion``, which terminates the session on the spot, so
        the terminal turn of a perfectly complete arm ALSO reports
        ``"tool_use"`` and no further turn arrives to say ``"stop"``.
        Reading the field directly therefore blocks every schema-driven
        arm as truncated — which is most of the tasks this engine exists
        to run.

        The question graders actually need answered is whether the arm
        reached a terminus it declared, and each profile shape declares a
        different one:

        - schema profile — the terminus is the completion payload, so a
          payload's presence settles it regardless of finish reason;
        - prose profile — the terminus is a turn that ended with no tool
          calls, i.e. ``"stop"``.

        Anything else (``max_tokens``, ``error``, ``cancelled``,
        ``safety``, or a bare ``"tool_use"`` with nothing signalled) cut
        the arm short, and the returned string names which — a BLOCKED
        verdict has to say what was absent.
        """
        if self.payload is not None:
            return None
        if self.finish_reason == "stop":
            return None
        if self.finish_reason == "tool_use":
            return ("ended mid-tool-loop (finish_reason='tool_use') having "
                    "signalled no completion payload")
        return f"finish_reason={self.finish_reason!r}"

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        """Ledger entries under the name the framework's processors expect.

        Completion processors written against the framework read
        ``context.tool_calls``; exposing the same attribute here lets an
        unmodified processor run as a grader.
        """
        return self.ledger.entries


class Grader(Protocol):
    """Adapter protocol.

    Implementations are constructed with their :class:`GraderSpec` and
    called once per arm.
    """

    spec: GraderSpec

    def __init__(self, spec: GraderSpec) -> None: ...

    def grade(self, context: GraderContext) -> Verdict: ...


def blocked(spec: GraderSpec, claim: str, reason: str) -> Verdict:
    """Build a BLOCKED verdict for ``spec``.

    Centralised so every adapter's blocked path carries a reason — the
    :class:`~jaato_eval.verdict.Verdict` constructor rejects one that
    does not, and this keeps adapters from each inventing the phrasing.
    """
    return Verdict(
        grader_id=f"{spec.kind}:{spec.identifier}",
        claim=claim,
        state="BLOCKED",
        blocked_reason=reason,
    )
