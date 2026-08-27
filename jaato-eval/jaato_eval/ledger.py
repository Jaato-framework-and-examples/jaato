"""Tool-call ledger — obtained from the SDK, with a daemon-version guard.

WHAT THIS MODULE USED TO BE
===========================

A private reimplementation of the framework's call/response pairing,
written because the ledger was unreachable over the SDK: the history
serializer emitted ``call_id`` on the ``function_response`` Part and not
on the ``function_call`` Part, so the identifier the pairing depends on
survived on only one side of the wire.

jaato #639 and #640 closed that.  ``_serialize_part`` now emits the call
identifier (it is ``fc.id`` in-process, re-keyed to ``call_id`` on the
wire so both Parts expose it under one name), and
``jaato_sdk.completion_processors.build_ledger`` is the single pairing
rule — the server-side ``build_tool_call_ledger`` is now a thin alias of
it.  So the reimplementation is deleted rather than kept in sync, which
was the point of asking for it.

WHAT REMAINS
============

One thing the SDK cannot answer, because it is a property of the
*deployment* rather than of the data: **is this daemon new enough to have
emitted the identifier at all?**  A client on a current SDK talking to a
pre-#639 daemon receives call Parts with no ``call_id``, and
``build_ledger`` correctly pairs nothing.  Nothing is wrong with the
ledger — it is an accurate reading of a history that cannot be paired —
but a grader must not treat it as evidence.

:func:`build_ledger_result` therefore wraps the SDK's ledger with a
faithfulness verdict computed from the raw history.  It is a version-skew
guard now, not a capability gap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class LedgerResult:
    """The SDK's ledger plus whether this daemon could express the pairing.

    Attributes:
        entries: Whatever ``jaato_sdk.completion_processors.build_ledger``
            returned — :class:`ToolCallEntry` dicts.  Empty when no tool
            calls fired, which is a legitimate outcome, not a failure.
        faithful: ``False`` only when call Parts arrived without the
            identifier, i.e. the daemon predates jaato #639.  Graders that
            depend on the ledger must return BLOCKED rather than grade.
        reason: Why ``faithful`` is ``False``.  Empty when it is ``True``.
        unpaired_calls: Calls with no response — a terminal turn's pending
            calls.  Legitimate and present in the framework's own ledger
            too, as ``{"error": "no_response"}``.
    """

    entries: List[Dict[str, Any]] = field(default_factory=list)
    faithful: bool = True
    reason: str = ""
    unpaired_calls: int = 0


def history_carries_call_ids(history: List[Dict[str, Any]]) -> bool:
    """Does every serialized ``function_call`` Part carry an identifier?

    Witnesses the key rather than inferring it from a pairing that came
    back empty — an unpaired ledger and an unpairable one look identical
    from the entries alone, and telling them apart is this module's whole
    remaining job.

    A history with no call Parts returns ``True``: there was nothing to
    pair, which is not a failure to pair.
    """
    for message in history:
        for part in message.get("parts", []) or []:
            if part.get("type") == "function_call" and not part.get("call_id"):
                return False
    return True


def build_ledger_result(
        history: Optional[List[Dict[str, Any]]]) -> LedgerResult:
    """Build the ledger via the SDK and judge whether it can be graded on.

    Args:
        history: ``HistoryEvent.history`` — serialized Message dicts — or
            ``None`` when no ``HistoryEvent`` ever arrived.  The two are
            NOT the same and must not share a representation: an empty
            list means the agent made no tool calls, and grading on that
            is correct; ``None`` means the engine never learned what the
            agent did, and grading on it fabricates a verdict about the
            model out of the engine's own blind spot.  Measured: a pooled
            arm whose history request went unanswered produced "the agent
            reports writing answer.txt but the ledger holds no call to any
            of (writeNewFile, ...)" — about an agent that had written the
            file, in a call the engine simply never saw.

    Returns:
        A :class:`LedgerResult`.  ``faithful`` is the gate; consult it
        before letting a verdict depend on ``entries``.

    Raises:
        Nothing.  An SDK too old to expose ``build_ledger`` yields an
        unfaithful empty result rather than an exception, so a sweep
        against a mismatched install degrades to BLOCKED arms instead of
        dying — the same rule the rest of this package follows.
    """
    if history is None:
        return LedgerResult(
            faithful=False,
            reason="no HistoryEvent arrived, so there is no ledger to grade "
                   "on — an absent history is not an empty one, and the "
                   "difference is whether a 'never called it' verdict is "
                   "about the agent or about this engine")

    try:
        from jaato_sdk.completion_processors import build_ledger
    except ImportError as exc:
        return LedgerResult(
            faithful=False,
            reason=f"jaato_sdk.completion_processors.build_ledger unavailable "
                   f"({exc}); needs an SDK carrying jaato #640")

    entries = list(build_ledger(history))
    unpaired = sum(1 for e in entries
                   if isinstance(e.get("result"), dict)
                   and e["result"].get("error") == "no_response")

    if history_carries_call_ids(history):
        return LedgerResult(entries=entries, faithful=True, unpaired_calls=unpaired)

    return LedgerResult(
        entries=entries,
        faithful=False,
        reason="serialized function_call Parts carry no call_id, so calls "
               "cannot be paired to responses — this daemon predates jaato "
               "#639; upgrade it rather than grading on an unpairable ledger",
        unpaired_calls=unpaired,
    )
