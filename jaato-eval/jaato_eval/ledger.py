"""Tool-call ledger reconstruction from SDK session history.

WHY THIS IS NOT A SIMPLE MIRROR
===============================

The framework builds this ledger server-side in
``shared/completion_processors.build_tool_call_ledger``, pairing each
``function_call`` Part with its ``function_response`` Part **by
call_id**.  Completion processors receive the result as
``context.tool_calls``; the SDK publishes the entry shape as
``jaato_sdk.completion_processors.ToolCallEntry``.

That ledger is not reachable over the SDK.  ``request_history`` returns
serialized Message dicts, and the serializer
(``server/command_router.py::_serialize_part``) emits::

    function_call     -> {"type", "name", "args"}
    function_response -> {"type", "name", "call_id", "response", "is_error", ...}

**The call side carries no ``call_id``.**  The identifier that the
canonical builder pairs on exists on only one of the two Parts once it
crosses the wire, so an SDK-side consumer cannot reproduce the pairing.

Pairing by ``name`` in arrival order is the obvious substitute and it is
wrong in exactly the case that matters: an agent that calls a tool,
fails, and retries it produces two calls and two responses with the same
name, and a positional pairing silently attributes the retry's success to
the first call.  A grader built on that would report a fabricated file as
verified.

So this module builds what it *can* build, and reports faithfulness
honestly.  Consumers that need true pairing must treat an unfaithful
ledger as BLOCKED rather than grade on it.  See ``graders/processor.py``.

The durable fix is one line in ``_serialize_part`` (carry ``call_id`` on
the ``function_call`` branch, as the response branch already does), plus
exposing the canonical builder through the SDK so this module can be
deleted rather than kept in sync.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class LedgerResult:
    """A reconstructed ledger plus an honest account of its limits.

    Attributes:
        entries: Ledger entries in the shape of
            ``jaato_sdk.completion_processors.ToolCallEntry``, except that
            ``call_id`` is ``""`` on any entry whose pairing could not be
            established.
        faithful: ``True`` only when every call was paired to its response
            by ``call_id``.  ``False`` means the entries are a best-effort
            reconstruction and must not be graded on.
        reason: Why ``faithful`` is ``False``.  Empty when it is ``True``.
        unpaired_calls: Calls with no response at all (a terminal turn's
            pending calls).  These are legitimate and appear in the
            canonical ledger too, as ``{"error": "no_response"}``.
    """

    entries: List[Dict[str, Any]] = field(default_factory=list)
    faithful: bool = True
    reason: str = ""
    unpaired_calls: int = 0


def build_ledger(history: List[Dict[str, Any]]) -> LedgerResult:
    """Reconstruct a tool-call ledger from serialized session history.

    Args:
        history: ``HistoryEvent.history`` — a list of serialized Message
            dicts, each ``{"role": str, "parts": [...]}``.

    Returns:
        A :class:`LedgerResult`.  Inspect ``faithful`` before using
        ``entries`` for anything a verdict depends on.
    """
    calls: List[Dict[str, Any]] = []
    responses: List[Dict[str, Any]] = []

    for turn_index, message in enumerate(history):
        for part in message.get("parts", []) or []:
            ptype = part.get("type")
            if ptype == "function_call":
                calls.append({"part": part, "turn_index": turn_index})
            elif ptype == "function_response":
                responses.append({"part": part, "turn_index": turn_index})

    # Can the wire even express the pairing?  A call Part that carries no
    # call_id makes the question unanswerable — witness the key, do not
    # infer it from the value being absent.
    calls_with_id = [c for c in calls if c["part"].get("call_id")]
    # No calls at all is faithful: there was nothing to pair.  Writing this
    # as ``bool(calls) and ...`` made an empty ledger report unfaithful and
    # would have BLOCKED every grader on a prose-only run — the same
    # absent/empty collapse this module exists to refuse, in this module.
    faithful = len(calls_with_id) == len(calls)
    reason = ""
    if calls and not faithful:
        reason = (
            f"{len(calls) - len(calls_with_id)} of {len(calls)} function_call "
            "parts carry no call_id over the SDK wire "
            "(server/command_router.py::_serialize_part omits it on the call "
            "branch), so calls cannot be paired to responses by id")

    by_id: Dict[str, Dict[str, Any]] = {}
    for r in responses:
        cid = r["part"].get("call_id")
        if cid:
            by_id[cid] = r["part"]

    entries: List[Dict[str, Any]] = []
    unpaired = 0
    for c in calls:
        part = c["part"]
        cid = part.get("call_id", "")
        resp_part = by_id.get(cid) if cid else None
        if resp_part is None:
            unpaired += 1
            result: Any = {"error": "no_response"}
            success = False
        else:
            result = resp_part.get("response")
            # ``is_error`` is the framework's own boundary-computed flag.
            # Prefer it over inspecting the result dict, which is the
            # plugin's private return convention and may have been
            # re-encoded by a provider round-trip.
            success = not resp_part.get("is_error", False)
        entries.append({
            "name": part.get("name", ""),
            "args": part.get("args", {}) or {},
            "result": result,
            "success": success,
            "call_id": cid,
            "turn_index": c["turn_index"],
        })

    return LedgerResult(entries=entries, faithful=faithful, reason=reason,
                        unpaired_calls=unpaired)
