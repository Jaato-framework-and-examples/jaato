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

from jaato_sdk import truncation_reason as _sdk_truncation_reason

from ..ledger import LedgerResult
from ..manifest import GraderSpec
from ..sign_off import is_unsigned_terminal
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
        termination_reason: ``SessionTerminatedEvent.reason`` — the only
            place an abnormal stop names itself.  ``budget_exhausted``
            never reaches ``finish_reason`` at all, because the refusal
            short-circuits before any turn runs.
        termination_detail: The refusal prose / error summary that came
            with it, so a BLOCKED verdict can quote the mechanism.
        termination_error_type: The terminal's TYPE, when it had one
            (``"NudgeExhausted"``, ``"RunnerCallError"``, ...).  Read it
            through :attr:`missing_sign_off` rather than by comparing
            strings — the set of terminals that leave a gradeable
            workspace is one rule, and it lives in
            :mod:`jaato_eval.sign_off`.
        completion_gap: ``TurnCompletedEvent.completion_gap`` (jaato #654) —
            set when the framework asked the agent twice to signal
            completion and it never did.  Before this existed, that path
            emitted NO terminal event at all: no AgentCompletedEvent, and
            no SessionTerminatedEvent because quiescence is gated on
            signal_completion having been called.  There was nothing to
            misread, so a consumer had to invent a cause for an absence —
            which this engine did, blaming a rubric's schema that was
            correct.
        socket_path: The daemon the ARM ran on.  A grader that opens its
            own session (the judge) must use the same one — the socket is
            a property of the RUN, not of the task, so it cannot live in
            the manifest.  Without this the judge silently reached the
            client default while the arm ran elsewhere.
        turns: How many turns the arm consumed.
        error: Terminal error text, when the arm ended in an error the
            engine graded through anyway (see :attr:`missing_sign_off`).
            An error terminal that leaves nothing to grade never reaches
            a grader at all — the arm is BLOCKED before this object is
            built — so a populated ``error`` means "the agent produced
            evidence AND ended badly", not "grading failed".
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
    termination_reason: str = ""
    termination_detail: str = ""
    termination_error_type: str = ""
    completion_gap: Optional[str] = None
    turns: int = 0
    socket_path: Optional[str] = None
    error: Optional[str] = None
    prior_verdicts: Dict[str, str] = field(default_factory=dict)

    @property
    def missing_sign_off(self) -> bool:
        """The agent finished working but never called ``signal_completion``.

        ``True`` only for the terminals
        :func:`jaato_eval.sign_off.is_unsigned_terminal` names — today
        just ``NudgeExhausted``, the framework's completion-nudge budget
        running out.  Such an arm HAS a workspace, so it is graded rather
        than discarded (jaato #773), and this is how each adapter decides
        whether it is one of the graders that survives:

        * reading the workspace → still valid, run and return a verdict;
        * reading the completion payload → BLOCK, and say the sign-off
          is what is missing rather than blaming the schema or the daemon.

        Note this is NOT the negation of :attr:`truncation_reason`.  That
        one answers "did the session end where it meant to", and the
        honest answer here is still no — which is why it keeps naming the
        mechanism, and why an adapter that reads the workspace has to
        consult BOTH: truncated-and-unsigned is gradeable, truncated for
        any other reason is not.
        """
        return is_unsigned_terminal(self.termination_error_type)

    @property
    def truncation_reason(self) -> Optional[str]:
        """Why this arm never reached a terminus it declared, or ``None``.

        Delegates to :func:`jaato_sdk.truncation_reason`, which shipped in
        jaato #648 carrying this rule and its ordering.  The logic used to
        live here; it is gone rather than kept in sync, for the same reason
        the ledger pairing went in #640 — a second copy of a rule rots
        independently, and this one exists precisely because a consumer
        following ``finish_reason != "stop"`` reaches the wrong verdict.

        This property remains because it names the arm's four inputs and
        hands them over; the RULE is the SDK's.
        """
        return _sdk_truncation_reason(
            finish_reason=self.finish_reason,
            payload=self.payload,
            termination_reason=self.termination_reason,
            termination_detail=self.termination_detail,
        )

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
