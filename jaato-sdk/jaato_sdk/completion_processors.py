"""Typed contract for cascade completion-processor context.

Cascade completion processors receive a ``RenderContext`` whose
``tool_calls`` field is a pre-paired ledger of every ``function_call``
+ ``function_response`` in the session.  The ledger entry shape is
stable but historically loosely typed (``List[Dict[str, Any]]``),
which makes it discoverable only by grepping the framework's
``shared/completion_processors.build_tool_call_ledger`` docstring.

This module exposes that shape as :class:`ToolCallEntry` so IDEs and
type checkers can surface the contract directly.  Existing processors
that read the ledger as plain dicts continue to work unchanged — the
TypedDict is an optional improvement, not a required import.

The framework's ``shared/completion_processors.build_tool_call_ledger``
is the canonical builder; it produces entries that satisfy this
TypedDict at runtime.  Server 0.6.158+ guarantees this contract.

**Canonical success indicator**: use ``entry["success"]``.  The inner
``entry["result"]["status"]`` is the plugin's own return convention
and may not survive provider-side serialization round-trips; the
top-level ``success`` flag is computed once at the framework boundary
and is the only stable success signal.

Server 0.6.158+ / SDK 0.14.0+.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypedDict


class ToolCallEntry(TypedDict):
    """One paired function_call + function_response from the
    completion-processor ledger.

    Each entry in :attr:`RenderContext.tool_calls` (which cascade
    completion processors receive via ``context.tool_calls``) conforms
    to this shape.

    Fields:
        name: Tool name as registered with the framework
            (e.g. ``"store_memory"``, ``"renderTemplateToFile"``).
        args: The argument dict the agent emitted on the
            ``function_call``.  Keys + types match the tool's schema.
        result: The tool's response dict.  Successful calls carry
            whatever the plugin returned (often ``{"status": "success",
            ...}``).  **Failed calls** carry
            ``{"error": "...", "message": "..."}``.  Calls without a
            matching ``function_response`` (e.g. terminal-turn pending
            calls) carry ``{"error": "no_response"}``.
        success: ``True`` iff ``"error"`` is not a key of ``result``.
            **THIS IS THE CANONICAL SUCCESS INDICATOR.**  Use it
            instead of reaching into ``result["status"]``:

            - ``result["status"]`` is the plugin's own return-shape
              convention; not every plugin uses it.
            - Provider-side serialization round-trips can re-encode
              ``result`` (e.g. as a JSON string) before the ledger
              builder sees it, breaking inner-field access.
            - ``success`` is computed once at the framework boundary
              and is stable across providers + serialization paths.

            Example (correct)::

                if not tc["success"]:
                    continue
                memory_id = tc["result"].get("memory_id")

            Example (fragile — DON'T DO THIS)::

                if tc["result"].get("status") != "success":  # may
                                                              # mis-fire
                    continue

        call_id: The provider's tool-call identifier.  Use for
            cross-referencing a call with its response when the
            ledger has been further transformed.
        turn_index: 0-based turn the call landed in.  Useful for
            ordering checks ("the agent called X before Y").
        enrichment_metadata: Structured per-plugin metadata from
            tool-result enrichment (LSP diagnostics, artifact tracking,
            ...), keyed by plugin name.  ``None`` when the call produced
            none, when it is in the ``no_response`` pending state — **or
            when the ledger was built from serialized history**, since
            enrichment is in-memory only and is not part of the wire
            shape.  Those three are not distinguishable from the value.

            This field was emitted by the builder and documented in its
            docstring from the start, and was MISSING here — so the
            published type described a narrower dict than the one
            consumers actually receive, and anyone typing against it saw
            no reason the key existed.
    """

    name: str
    args: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    call_id: str
    turn_index: int
    enrichment_metadata: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

def _read_part(part: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Normalise one Part into ``(kind, fields)``, or ``None`` if neither.

    TWO SHAPES REACH THIS, and only the READING differs between them:

    - **in-process** — ``Part`` objects, where the call is a
      ``FunctionCall`` (``.id`` / ``.name`` / ``.args``) and the response a
      ``ToolResult`` (``.call_id`` / ``.result`` / ``.enrichment_metadata``);
    - **over the wire** — the dicts ``CommandRouter._serialize_part``
      produces, where both sides expose the identifier as ``call_id`` and the
      response body is under ``response``.

    NOTE THE ASYMMETRY IN THE OBJECT SHAPE: the identifier is ``id`` on the
    call and ``call_id`` on the response.  Reading ``call_id`` from a
    ``FunctionCall`` yields nothing — silently, since it has no such field —
    which is how the wire came to carry no call identifier at all until it
    was fixed.  It is spelled out here because this is the only place that
    still has to know it.

    Everything downstream sees one shape, which is the point: the PAIRING
    RULE below exists once, and adding a third carrier means teaching this
    function to read it, not writing a second ledger.
    """
    # ---- wire shape: a plain dict tagged with ``type`` --------------------
    if isinstance(part, dict):
        kind = part.get("type")
        if kind == "function_call":
            return ("call", {
                "call_id": part.get("call_id") or "",
                "name": part.get("name") or "",
                "args": part.get("args"),
            })
        if kind == "function_response":
            return ("response", {
                "call_id": part.get("call_id") or "",
                "result": part.get("response"),
                # The wire carries no enrichment metadata: it is in-memory
                # only and is not part of the serialized part shape.  ``None``
                # here means "not transported", NOT "the tool produced none" —
                # a distinction a consumer cannot make, and one worth knowing
                # before building a check on it.
                "enrichment_metadata": None,
            })
        return None

    # ---- in-process shape: Part objects ----------------------------------
    fc = getattr(part, "function_call", None)
    if fc is not None:
        return ("call", {
            "call_id": getattr(fc, "id", None) or "",
            "name": getattr(fc, "name", "") or "",
            "args": getattr(fc, "args", None),
        })
    fr = getattr(part, "function_response", None)
    if fr is not None:
        return ("response", {
            "call_id": getattr(fr, "call_id", None) or "",
            "result": getattr(fr, "result", None),
            "enrichment_metadata": getattr(fr, "enrichment_metadata", None),
        })
    return None


def _parts_of(message: Any) -> List[Any]:
    """Parts of one message, from either carrier."""
    if isinstance(message, dict):
        return message.get("parts") or []
    return getattr(message, "parts", None) or []


def build_ledger(history: List[Any]) -> List[ToolCallEntry]:
    """Pair every ``function_call`` with its ``function_response``.

    Accepts EITHER carrier — the in-process ``Message`` objects from
    ``JaatoSession.get_history()``, or the serialized dicts a client receives
    on ``HistoryEvent.history`` — and returns the same
    :class:`ToolCallEntry` shape from both.

    WHY THIS LIVES IN THE SDK.  The SDK published the TYPE of a ledger entry
    and no way to obtain a ledger, so a consumer wanting one wrote its own
    pairing.  Every such copy has to re-derive that pairing is by identifier
    and not by name-in-order — and the copies rot independently, which is the
    failure this module's own contract exists to prevent.  The server-side
    ``shared.completion_processors.build_tool_call_ledger`` is now a thin
    alias of this function, so there is ONE implementation rather than one
    per consumer.

    PAIRING IS BY IDENTIFIER, NEVER BY NAME AND ORDER.  Name-in-order
    pairing fails on the case that matters most: a tool that errors and is
    retried produces two calls and two responses sharing a name, and
    positional pairing credits the retry's SUCCESS to the call that FAILED.
    A check built on that reports a fabricated artefact as verified — it
    inverts the verdict rather than weakening it.

    Args:
        history: Messages, either shape.  Mixed lists work; each message is
            read independently.

    Returns:
        Chronological ledger.  Empty when no tool calls fired.

    Notes:
        - A call with no matching response — a terminal turn whose calls are
          still pending — is emitted with ``success=False`` and
          ``result={"error": "no_response"}``, so a validator's "claimed but
          never successfully called" check fires deterministically rather
          than finding nothing to look at.
        - ``success`` is ``"error" not in result``, on BOTH carriers.  The
          wire shape also carries an explicit ``is_error`` flag; it is
          deliberately not consulted, because two success rules that usually
          agree are worse than one that always does.
        - ``enrichment_metadata`` is present only on the in-process carrier.
          Over the wire it is always ``None`` — not transported, rather than
          absent.
        - Responses are keyed by identifier, so if two responses somehow
          share one, the LAST wins.  That mirrors the original behaviour and
          is not a decision this function is in a position to make better.
    """
    responses: Dict[str, Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = {}
    for message in history:
        for part in _parts_of(message):
            read = _read_part(part)
            if read is None or read[0] != "response":
                continue
            fields = read[1]
            body = fields["result"]
            if not isinstance(body, dict):
                # A tool returned a scalar/None.  Wrap so every consumer sees
                # one shape and the ``"error" not in result`` rule still
                # applies rather than raising on a string.
                body = {"result": body}
            responses[fields["call_id"]] = (
                body, fields["enrichment_metadata"],
            )

    ledger: List[ToolCallEntry] = []
    for turn_index, message in enumerate(history):
        for part in _parts_of(message):
            read = _read_part(part)
            if read is None or read[0] != "call":
                continue
            fields = read[1]
            raw_args = fields["args"]
            args = raw_args if isinstance(raw_args, dict) else {"_raw": raw_args}
            paired = responses.get(fields["call_id"])
            if paired is None:
                result: Dict[str, Any] = {"error": "no_response"}
                enrichment: Optional[Dict[str, Any]] = None
            else:
                result, enrichment = paired
            ledger.append({
                "name": fields["name"],
                "args": args,
                "result": result,
                "success": "error" not in result,
                "call_id": fields["call_id"],
                "turn_index": turn_index,
                "enrichment_metadata": enrichment,
            })
    return ledger


__all__ = ["ToolCallEntry", "build_ledger"]
