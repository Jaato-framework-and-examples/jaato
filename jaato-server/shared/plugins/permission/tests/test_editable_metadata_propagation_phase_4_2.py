"""Regression-pin tests for Phase 4 §4.2 — J.B editable_metadata
propagation.

**Root cause** (Phase 3 cycle-13 backlog J.B):

Pre-§4.2: ``PromptPayload`` didn't carry an ``editable_metadata``
field.  ``PromptOperatorHandler.handle()`` therefore emitted
``PermissionInputModeEvent`` with ``editable_metadata=None``,
breaking the TUI's "edit and approve" flow for editable tools
(e.g. ``todo_write`` which declares
``editable=EditableContent(parameters=["title", "steps"],
format="yaml")``).

**Phase 4 §4.2 fix** (per audit 271b7e0d, Option A narrowed):

The schema lookup is ALREADY done in the permission plugin's
``check_permission`` flow:

1. ``shared/plugins/permission/plugin.py:1411-1412`` calls
   ``_get_tool_schema(tool_name)`` and extracts ``schema.editable``.
2. ``shared/plugins/permission/plugin.py:1461`` stores it on
   ``PermissionRequest.editable``.

§4.2 hops:

A. ``shared/plugins/permission/runner_rpc_channel.py``
   :request_permission reads ``request.editable``
   (``Optional[EditableContent]``) and converts to canonical wire-
   shape dict ``{"parameters": list, "format": str}`` (mirrors the
   pre-§7c daemon-side hook at core.py:3062-3072).  Template
   omitted — editor-rendering concern, not part of the daemon/TUI
   wire contract.
B. ``shared/plugins/permission/types.py:PromptPayload`` gains
   ``editable_metadata: Optional[Dict[str, Any]] = None`` field
   with full to_dict/from_dict round-trip + backward-compat.
C. ``server/runner_rpc_handlers/prompt_operator.py:handle`` threads
   ``payload.editable_metadata`` to
   ``PermissionInputModeEvent.editable_metadata`` (was hardcoded
   ``None``).

Tests pin:

- PromptPayload.editable_metadata field default (None)
- to_dict includes editable_metadata key
- Round-trip preserves populated dict
- from_dict backward-compat (legacy wire dict → None)
- RunnerRPCChannel reads request.editable + threads with the
  canonical shape (parameters + format)
- Non-editable request → editable_metadata=None
- Defensive: malformed editable (missing fields) → None
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from jaato_sdk.plugins.model_provider.types import EditableContent
from shared.plugins.permission.types import PromptPayload, PromptResponse


# ----------------------------------------------------------------------
# B — PromptPayload field + serialization
# ----------------------------------------------------------------------


def test_prompt_payload_has_editable_metadata_field_default_none() -> None:
    """Pin: PromptPayload.editable_metadata field exists; defaults
    to None (non-editable tools)."""
    p = PromptPayload(
        request_id="r-1", session_id="s-1", tool_name="cli",
    )
    assert hasattr(p, "editable_metadata")
    assert p.editable_metadata is None


def test_prompt_payload_to_dict_includes_editable_metadata() -> None:
    """Pin: ``to_dict`` exposes ``editable_metadata`` on the wire."""
    p = PromptPayload(
        request_id="r-1", session_id="s-1", tool_name="todo",
        editable_metadata={"parameters": ["title"], "format": "yaml"},
    )
    d = p.to_dict()
    assert "editable_metadata" in d
    assert d["editable_metadata"] == {
        "parameters": ["title"], "format": "yaml",
    }


def test_prompt_payload_round_trip_preserves_editable_metadata() -> None:
    """Pin: to_dict → from_dict preserves the editable_metadata
    dict structure."""
    md = {
        "parameters": ["title", "steps"],
        "format": "yaml",
    }
    p1 = PromptPayload(
        request_id="r-1", session_id="s-1", tool_name="todo",
        editable_metadata=md,
    )
    p2 = PromptPayload.from_dict(p1.to_dict())
    assert p2.editable_metadata == md


def test_prompt_payload_from_dict_backward_compat_missing_editable() -> None:
    """Pin: from_dict on a legacy wire dict (pre-§4.2) WITHOUT an
    editable_metadata field deserializes to None."""
    legacy_dict = {
        "request_id": "r-1",
        "session_id": "s-1",
        "tool_name": "cli",
        "tool_args": {},
        "prompt_lines": None,
        "response_options": [],
        "format_hint": None,
        "warnings": None,
        "warning_level": None,
        "agent_id": "",
        "call_id": None,
        # No editable_metadata key — pre-§4.2 wire shape.
    }
    p = PromptPayload.from_dict(legacy_dict)
    assert p.editable_metadata is None


def test_prompt_payload_from_dict_rejects_non_dict_editable_metadata() -> None:
    """Pin (defensive): if a wire dict carries a non-dict
    editable_metadata (corrupted/forward-incompat), from_dict
    deserializes to None rather than propagating the bad value."""
    for bad in [42, "string", ["list"], True]:
        legacy_dict = {
            "request_id": "r-1",
            "session_id": "s-1",
            "tool_name": "cli",
            "editable_metadata": bad,
        }
        p = PromptPayload.from_dict(legacy_dict)
        assert p.editable_metadata is None, (
            f"Phase 4 §4.2 regression: non-dict editable_metadata="
            f"{bad!r} leaked through from_dict; defensive validation "
            f"missing."
        )


# ----------------------------------------------------------------------
# A — RunnerRPCChannel reads request.editable + threads to PromptPayload
# ----------------------------------------------------------------------


def test_runner_rpc_channel_threads_editable_metadata_from_request() -> None:
    """Pin: RunnerRPCChannel.request_permission reads
    ``request.editable`` (an EditableContent instance already
    resolved by the permission plugin) and converts to the
    canonical wire-shape dict on PromptPayload.editable_metadata."""
    from shared.plugins.permission.channels import PermissionRequest
    from shared.plugins.permission.runner_rpc_channel import RunnerRPCChannel

    captured_payloads: List[PromptPayload] = []

    def _fake_prompt_operator(payload: PromptPayload) -> PromptResponse:
        captured_payloads.append(payload)
        return PromptResponse(request_id=payload.request_id, response="y")

    channel = RunnerRPCChannel(prompt_operator=_fake_prompt_operator)

    editable = EditableContent(
        parameters=["title", "steps"],
        format="yaml",
        template="Optional header",
    )

    req = PermissionRequest.create(
        tool_name="todo_write",
        arguments={"title": "Plan A", "steps": []},
        context={"session_id": "s-1", "agent_id": "main"},
        editable=editable,
    )

    channel.request_permission(req)

    assert len(captured_payloads) == 1
    md = captured_payloads[0].editable_metadata
    assert md is not None, (
        "Phase 4 §4.2 regression: RunnerRPCChannel didn't propagate "
        "request.editable to PromptPayload.editable_metadata."
    )
    assert md["parameters"] == ["title", "steps"]
    assert md["format"] == "yaml"
    # Template is NOT in the wire shape — editor-rendering concern;
    # daemon/TUI don't need it for input-mode signaling.
    assert "template" not in md


def test_runner_rpc_channel_non_editable_request_yields_none() -> None:
    """Pin: when request.editable is None (non-editable tool), the
    PromptPayload's editable_metadata is None."""
    from shared.plugins.permission.channels import PermissionRequest
    from shared.plugins.permission.runner_rpc_channel import RunnerRPCChannel

    captured: List[PromptPayload] = []

    def _fake_prompt_operator(payload: PromptPayload) -> PromptResponse:
        captured.append(payload)
        return PromptResponse(request_id=payload.request_id, response="y")

    channel = RunnerRPCChannel(prompt_operator=_fake_prompt_operator)
    req = PermissionRequest.create(
        tool_name="cli",
        arguments={"command": "ls"},
        context={"session_id": "s-1", "agent_id": "main"},
        # No editable= → None by default.
    )
    channel.request_permission(req)

    assert captured[0].editable_metadata is None


def test_runner_rpc_channel_defensive_against_malformed_editable() -> None:
    """Pin (defensive): if request.editable is somehow malformed
    (missing parameters or format), the channel emits a
    minimally-valid editable_metadata or None — never crashes the
    ASK relay."""
    from shared.plugins.permission.channels import PermissionRequest
    from shared.plugins.permission.runner_rpc_channel import RunnerRPCChannel

    captured: List[PromptPayload] = []

    def _fake_prompt_operator(payload: PromptPayload) -> PromptResponse:
        captured.append(payload)
        return PromptResponse(request_id=payload.request_id, response="y")

    channel = RunnerRPCChannel(prompt_operator=_fake_prompt_operator)

    # Malformed editable: missing parameters attribute.
    class _BadEditable:
        format = "yaml"

    req = PermissionRequest.create(
        tool_name="todo_write",
        arguments={"title": "Plan"},
        context={"session_id": "s-1", "agent_id": "main"},
        editable=_BadEditable(),
    )
    # Should not raise.
    channel.request_permission(req)

    # The relay must have completed (captured the payload).  The
    # editable_metadata may be either {parameters: [], format:
    # "yaml"} (graceful default) or None (defensive bail) — both
    # acceptable.  What matters: no crash; ASK still propagates.
    assert len(captured) == 1
