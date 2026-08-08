"""Tests for the typed completion-processor context contract.

The framework's ``shared/completion_processors.build_tool_call_ledger``
is the canonical builder; this test asserts the SDK's
:class:`ToolCallEntry` TypedDict matches the field set the builder
produces.  When the builder evolves, this test catches drift before
external cascade authors hit it.
"""

from __future__ import annotations

from typing import get_type_hints

import pytest

from jaato_sdk import ToolCallEntry
from jaato_sdk.completion_processors import ToolCallEntry as _Direct


def test_toolcallentry_exported_from_package_root():
    """SDK package root re-export — what cascade authors import.

    Both ``from jaato_sdk import ToolCallEntry`` and
    ``from jaato_sdk.completion_processors import ToolCallEntry`` must
    resolve to the same class.
    """
    assert ToolCallEntry is _Direct


def test_toolcallentry_field_set_matches_ledger_contract():
    """Field names + types match the ledger entry shape documented
    in :func:`shared.completion_processors.build_tool_call_ledger`
    (server-side).  Drift here breaks every kb-side cascade processor.
    """
    hints = get_type_hints(ToolCallEntry)
    assert set(hints.keys()) == {
        "name", "args", "result", "success", "call_id", "turn_index",
    }


def test_toolcallentry_accepts_runtime_dict_shape():
    """Confirm a dict matching the builder's output validates as a
    ``ToolCallEntry``.  TypedDicts are structural at runtime — this
    is a contract sanity check, not a strict-type assertion.
    """
    entry: ToolCallEntry = {
        "name": "store_memory",
        "args": {"content": "...", "tags": ["a", "b"]},
        "result": {"status": "success", "memory_id": "mem_..."},
        "success": True,
        "call_id": "call_123",
        "turn_index": 3,
    }
    # Access every documented field — fails at type-check time if
    # the TypedDict shape regressed.
    assert entry["name"] == "store_memory"
    assert isinstance(entry["args"], dict)
    assert isinstance(entry["result"], dict)
    assert isinstance(entry["success"], bool)
    assert isinstance(entry["call_id"], str)
    assert isinstance(entry["turn_index"], int)


def test_toolcallentry_failed_call_shape():
    """Failed-call shape per the builder's contract: result carries
    an ``"error"`` key and ``success`` is ``False``.  This is the
    shape cascade processors must handle in the rejection branch.
    """
    entry: ToolCallEntry = {
        "name": "store_memory",
        "args": {},
        "result": {"error": "PermissionError", "message": "..."},
        "success": False,
        "call_id": "call_456",
        "turn_index": 4,
    }
    assert not entry["success"]
    assert "error" in entry["result"]
