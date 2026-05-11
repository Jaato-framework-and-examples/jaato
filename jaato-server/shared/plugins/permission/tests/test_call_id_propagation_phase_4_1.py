"""Regression-pin tests for Phase 4 §4.1 — J.A ``call_id`` propagation.

**Root cause** (Phase 3 cycle-13 backlog J.A):

``PromptPayload`` didn't carry a ``call_id`` field; the runner-side
permission plugin had the call_id (passed into ``check_permission``)
but dropped it at the channel boundary.  Daemon-side
``PromptOperatorHandler`` therefore emitted
``PermissionInputModeEvent`` with ``call_id=None``, breaking TUI
per-tool-block correlation when multiple tools were in flight.

**Phase 4 §4.1 fix** (per the J.A backlog doc and Phase 4 plan §3.1):

The call_id threads through 4 hops, each pinned here:

1. ``PermissionPlugin.check_permission`` stores ``call_id`` into
   ``request.context["call_id"]`` (mirrors session_id/agent_id
   pattern — keeps the Channel ABC stable).
2. ``RunnerRPCChannel.request_permission`` reads ``call_id`` from
   ``request.context`` and threads to ``PromptPayload.call_id``.
3. ``PromptPayload`` round-trips ``call_id`` through to_dict/
   from_dict (wire serialization).
4. ``PromptOperatorHandler.handle`` threads ``payload.call_id`` to
   ``PermissionInputModeEvent.call_id``.  (Pinned in
   ``test_permission_input_mode_emit_path_j.py``.)

Tests pin:

- PromptPayload.call_id field default (None)
- PromptPayload.to_dict includes call_id key
- PromptPayload round-trip (to_dict → from_dict) preserves
  populated call_id
- PromptPayload backward-compat: from_dict on a legacy wire dict
  WITHOUT call_id field deserializes to call_id=None
- PermissionPlugin.check_permission's request_context population
  (AST pin on plugin.py source)
- RunnerRPCChannel reads context['call_id'] and threads to
  PromptPayload (behavioral)
- End-to-end propagation pin (full chain in one test)
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from shared.plugins.permission.types import PromptPayload, PromptResponse


# ----------------------------------------------------------------------
# PromptPayload field + serialization
# ----------------------------------------------------------------------


def test_prompt_payload_has_call_id_field_default_none() -> None:
    """Pin: PromptPayload.call_id field exists; defaults to None."""
    p = PromptPayload(
        request_id="r-1", session_id="s-1", tool_name="cli",
    )
    assert hasattr(p, "call_id")
    assert p.call_id is None


def test_prompt_payload_call_id_accepts_string() -> None:
    """Pin: call_id accepts arbitrary string (tool-call correlator)."""
    p = PromptPayload(
        request_id="r-1", session_id="s-1", tool_name="cli",
        call_id="tool-call-uuid-abc-123",
    )
    assert p.call_id == "tool-call-uuid-abc-123"


def test_prompt_payload_to_dict_includes_call_id() -> None:
    """Pin: ``to_dict`` exposes call_id on the wire."""
    p = PromptPayload(
        request_id="r-1", session_id="s-1", tool_name="cli",
        call_id="c1",
    )
    d = p.to_dict()
    assert "call_id" in d
    assert d["call_id"] == "c1"


def test_prompt_payload_round_trip_preserves_call_id() -> None:
    """Pin: to_dict → from_dict preserves the call_id field."""
    p1 = PromptPayload(
        request_id="r-1", session_id="s-1", tool_name="cli",
        call_id="c1",
    )
    p2 = PromptPayload.from_dict(p1.to_dict())
    assert p2.call_id == "c1"


def test_prompt_payload_from_dict_backward_compat_missing_call_id() -> None:
    """Pin: ``from_dict`` on a legacy wire dict (pre-§4.1) WITHOUT
    a call_id field deserializes to call_id=None.  Forward/backward
    compat with older runners that don't populate the field."""
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
        # No call_id key — pre-§4.1 wire shape.
    }
    p = PromptPayload.from_dict(legacy_dict)
    assert p.call_id is None


def test_prompt_payload_to_dict_call_id_is_none_when_unset() -> None:
    """Pin: when call_id wasn't populated, ``to_dict`` emits None
    (not missing key, not empty string).  Lets the daemon-side
    handler distinguish "explicit None" from "absent" — both map
    to no call_id, but the dict shape is stable."""
    p = PromptPayload(
        request_id="r-1", session_id="s-1", tool_name="cli",
    )
    d = p.to_dict()
    assert d["call_id"] is None


# ----------------------------------------------------------------------
# PermissionPlugin populates context['call_id'] (AST pin)
# ----------------------------------------------------------------------


def test_check_permission_source_populates_call_id_in_context() -> None:
    """Pin via AST: ``check_permission`` body contains
    ``request_context["call_id"] = call_id``.

    Catches accidental deletion of the §4.1 context population —
    without it, RunnerRPCChannel can't read call_id from
    request.context and the propagation chain breaks.
    """
    from shared.plugins.permission.plugin import PermissionPlugin

    src = inspect.getsource(PermissionPlugin.check_permission)
    src = textwrap.dedent(src)
    tree = ast.parse(src)

    found = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "request_context"
        ):
            # Subscript slice is the key being written.
            slice_node = node.targets[0].slice
            if (
                isinstance(slice_node, ast.Constant)
                and slice_node.value == "call_id"
            ):
                found = True
                break

    assert found, (
        "Phase 4 §4.1 regression: PermissionPlugin.check_permission "
        "missing ``request_context['call_id'] = call_id`` assignment.  "
        "Without it, call_id never reaches the runner-RPC channel "
        "and J.A regression returns."
    )


# ----------------------------------------------------------------------
# RunnerRPCChannel reads call_id from context + threads to PromptPayload
# ----------------------------------------------------------------------


def test_runner_rpc_channel_reads_call_id_from_context() -> None:
    """Pin (behavioral): RunnerRPCChannel.request_permission reads
    ``call_id`` from ``request.context`` and threads it to
    ``PromptPayload.call_id``."""
    from shared.plugins.permission.channels import PermissionRequest
    from shared.plugins.permission.runner_rpc_channel import RunnerRPCChannel

    captured_payloads: List[PromptPayload] = []

    def _fake_prompt_operator(payload: PromptPayload) -> PromptResponse:
        captured_payloads.append(payload)
        return PromptResponse(request_id=payload.request_id, response="y")

    channel = RunnerRPCChannel(prompt_operator=_fake_prompt_operator)

    req = PermissionRequest.create(
        tool_name="cli",
        arguments={"command": "ls"},
        context={
            "session_id": "s-1",
            "agent_id": "main",
            "call_id": "tool-call-xyz-789",
        },
    )

    channel.request_permission(req)

    assert len(captured_payloads) == 1
    assert captured_payloads[0].call_id == "tool-call-xyz-789", (
        "Phase 4 §4.1 regression: RunnerRPCChannel didn't read "
        "call_id from request.context.  PromptPayload reached the "
        "daemon without the correlator; J.A propagation chain "
        "broken at the channel hop."
    )


def test_runner_rpc_channel_handles_missing_call_id_in_context() -> None:
    """Pin: when context doesn't carry call_id (legacy callers),
    PromptPayload.call_id is None (no crash, no empty string)."""
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
    )
    channel.request_permission(req)

    assert captured[0].call_id is None


def test_runner_rpc_channel_rejects_non_string_call_id() -> None:
    """Pin (defensive): if context['call_id'] is non-string (or
    empty), PromptPayload.call_id is None.  Prevents accidental
    int/dict leakage through the wire payload."""
    from shared.plugins.permission.channels import PermissionRequest
    from shared.plugins.permission.runner_rpc_channel import RunnerRPCChannel

    captured: List[PromptPayload] = []

    def _fake_prompt_operator(payload: PromptPayload) -> PromptResponse:
        captured.append(payload)
        return PromptResponse(request_id=payload.request_id, response="y")

    channel = RunnerRPCChannel(prompt_operator=_fake_prompt_operator)

    for bad in [42, None, "", {"nested": "x"}]:
        captured.clear()
        req = PermissionRequest.create(
            tool_name="cli",
            arguments={"command": "ls"},
            context={"call_id": bad},
        )
        channel.request_permission(req)
        assert captured[0].call_id is None, (
            f"Phase 4 §4.1 regression: non-string context['call_id']="
            f"{bad!r} leaked into PromptPayload.call_id; defensive "
            f"validation missing."
        )


# ----------------------------------------------------------------------
# End-to-end pin: plugin populates context → channel reads → payload carries
# ----------------------------------------------------------------------


def test_end_to_end_call_id_propagation_via_channel() -> None:
    """Pin: full §4.1 chain — when a caller hands ``call_id`` to
    PermissionPlugin.check_permission, the resulting PromptPayload
    (reached via the runner-RPC channel) carries the same value.

    This is the J.A acceptance gate test — if it fails, the TUI
    can't correlate parallel-tool permission prompts to their
    blocks.
    """
    from shared.plugins.permission.channels import PermissionRequest
    from shared.plugins.permission.runner_rpc_channel import RunnerRPCChannel

    # Simulate the plugin-side population (the actual plugin runs
    # extensive auth/eval logic before reaching this point; for the
    # propagation pin we exercise the boundary directly).
    request_context: Dict[str, Any] = {
        "session_id": "s-1",
        "agent_id": "main",
    }
    call_id = "ground-truth-tool-call-id"
    if call_id:  # mirrors plugin.py:1455-area condition
        request_context["call_id"] = call_id

    captured: List[PromptPayload] = []

    def _fake_prompt_operator(payload: PromptPayload) -> PromptResponse:
        captured.append(payload)
        return PromptResponse(request_id=payload.request_id, response="y")

    channel = RunnerRPCChannel(prompt_operator=_fake_prompt_operator)
    req = PermissionRequest.create(
        tool_name="write_file",
        arguments={"path": "/tmp/x", "content": "y"},
        context=request_context,
    )
    channel.request_permission(req)

    assert captured[0].call_id == call_id

    # And round-trips through the wire serializer.
    wire = captured[0].to_dict()
    restored = PromptPayload.from_dict(wire)
    assert restored.call_id == call_id
