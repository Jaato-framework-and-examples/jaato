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

from typing import Any, Dict, TypedDict


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
    """

    name: str
    args: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    call_id: str
    turn_index: int


__all__ = ["ToolCallEntry"]
