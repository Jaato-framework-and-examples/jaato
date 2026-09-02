"""Tests for ``shared.plugins.permission.runner_rpc_channel`` (Phase 3 §3.7 deeper).

The :class:`RunnerRPCChannel` relays ASK decisions through the
daemon's ``client.prompt_operator`` RPC primitive (§3.2.1) when
the permission plugin runs runner-side.  These tests cover:

- Channel ABC contract (name, request_permission shape).
- PermissionRequest → PromptPayload translation.
- PromptResponse → ChannelResponse translation, including all
  the response-key → decision mappings.
- Edit-with-arguments path.
- Transport failure → DENY-with-reason posture.
- session_id + agent_id propagation through PermissionRequest.context.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from shared.plugins.permission.channels import (
    ChannelDecision,
    ChannelResponse,
    PermissionRequest,
    PermissionResponseOption,
    get_default_permission_options,
)
from shared.plugins.permission.runner_rpc_channel import RunnerRPCChannel
from shared.plugins.permission.types import PromptPayload, PromptResponse


# ----------------------------------------------------------------------
# Stub prompt_operator — captures payload, returns canned response
# ----------------------------------------------------------------------


class _StubOperator:
    """Records each prompt_operator invocation; returns
    ``next_response`` (or raises ``next_raise`` if set)."""

    def __init__(
        self,
        next_response: Optional[PromptResponse] = None,
        next_raise: Optional[BaseException] = None,
    ) -> None:
        self.next_response = next_response
        self.next_raise = next_raise
        self.calls: List[PromptPayload] = []

    def __call__(self, payload: PromptPayload) -> PromptResponse:
        self.calls.append(payload)
        if self.next_raise is not None:
            raise self.next_raise
        if self.next_response is None:
            raise AssertionError("test forgot to set next_response")
        return self.next_response


def _make_request(**overrides: Any) -> PermissionRequest:
    base = dict(
        tool_name="cli_based_tool",
        arguments={"command": "rm -rf /"},
        timeout=30,
    )
    base.update(overrides)
    return PermissionRequest.create(**base)


# ----------------------------------------------------------------------
# Channel ABC contract
# ----------------------------------------------------------------------


def test_channel_name() -> None:
    channel = RunnerRPCChannel(_StubOperator())
    assert channel.name == "runner_rpc"


def test_channel_implements_request_permission() -> None:
    """request_permission returns a ChannelResponse with the request_id."""
    op = _StubOperator(
        next_response=PromptResponse(request_id="r-1", response="y"),
    )
    channel = RunnerRPCChannel(op)
    request = _make_request()
    response = channel.request_permission(request)
    assert isinstance(response, ChannelResponse)
    assert response.request_id == request.request_id


# ----------------------------------------------------------------------
# Request → Payload translation
# ----------------------------------------------------------------------


def test_payload_carries_tool_name_and_args() -> None:
    op = _StubOperator(
        next_response=PromptResponse(request_id="r-1", response="y"),
    )
    channel = RunnerRPCChannel(op)
    request = _make_request(
        tool_name="writeNewFile",
        arguments={"path": "/tmp/x.txt", "content": "hi"},
    )
    channel.request_permission(request)
    assert len(op.calls) == 1
    payload = op.calls[0]
    assert payload.tool_name == "writeNewFile"
    assert payload.tool_args == {"path": "/tmp/x.txt", "content": "hi"}


def test_payload_carries_request_id() -> None:
    op = _StubOperator(
        next_response=PromptResponse(request_id="any", response="y"),
    )
    channel = RunnerRPCChannel(op)
    request = _make_request()
    channel.request_permission(request)
    assert op.calls[0].request_id == request.request_id


def test_payload_carries_session_id_from_context() -> None:
    op = _StubOperator(
        next_response=PromptResponse(request_id="r-1", response="y"),
    )
    channel = RunnerRPCChannel(op)
    request = _make_request(context={"session_id": "sess-42"})
    channel.request_permission(request)
    assert op.calls[0].session_id == "sess-42"


def test_payload_carries_agent_id_from_context() -> None:
    op = _StubOperator(
        next_response=PromptResponse(request_id="r-1", response="y"),
    )
    channel = RunnerRPCChannel(op)
    request = _make_request(context={"agent_id": "researcher"})
    channel.request_permission(request)
    assert op.calls[0].agent_id == "researcher"


def test_payload_session_and_agent_id_default_empty() -> None:
    """When PermissionRequest.context lacks session_id / agent_id,
    the payload carries empty strings (not raises)."""
    op = _StubOperator(
        next_response=PromptResponse(request_id="r-1", response="y"),
    )
    channel = RunnerRPCChannel(op)
    request = _make_request()  # default context = {}
    channel.request_permission(request)
    assert op.calls[0].session_id == ""
    assert op.calls[0].agent_id == ""


def test_payload_response_options_translated() -> None:
    """The default permission-options list survives translation
    to the wire format the handler advertises to the client."""
    op = _StubOperator(
        next_response=PromptResponse(request_id="r-1", response="y"),
    )
    channel = RunnerRPCChannel(op)
    request = _make_request()
    channel.request_permission(request)
    options = op.calls[0].response_options
    # Should carry the default option list (y, n, a, t, i, etc.).
    keys = {opt["key"] for opt in options}
    assert "y" in keys
    assert "n" in keys


def test_default_options_override_constructor_arg() -> None:
    """When the channel is constructed with default_options, those
    win over the request's own response_options."""
    custom = [
        PermissionResponseOption(
            "ok", "ok-yes", "Approve",
            ChannelDecision.ALLOW,
        ),
    ]
    op = _StubOperator(
        next_response=PromptResponse(request_id="r-1", response="ok"),
    )
    channel = RunnerRPCChannel(op, default_options=custom)
    channel.request_permission(_make_request())
    assert len(op.calls[0].response_options) == 1
    assert op.calls[0].response_options[0]["key"] == "ok"


# ----------------------------------------------------------------------
# Response → ChannelResponse decision mapping
# ----------------------------------------------------------------------


@pytest.mark.parametrize("response_key,expected_decision", [
    ("y", ChannelDecision.ALLOW),
    ("yes", ChannelDecision.ALLOW),
    ("allow", ChannelDecision.ALLOW),
    ("n", ChannelDecision.DENY),
    ("no", ChannelDecision.DENY),
    ("deny", ChannelDecision.DENY),
    ("a", ChannelDecision.ALLOW_SESSION),
    ("always", ChannelDecision.ALLOW_SESSION),
    ("never", ChannelDecision.DENY_SESSION),
    ("all", ChannelDecision.ALLOW_ALL),
    ("t", ChannelDecision.ALLOW_TURN),
    ("turn", ChannelDecision.ALLOW_TURN),
    ("i", ChannelDecision.ALLOW_UNTIL_IDLE),
    ("idle", ChannelDecision.ALLOW_UNTIL_IDLE),
    ("once", ChannelDecision.ALLOW_ONCE),
    ("e", ChannelDecision.EDIT),
    ("edit", ChannelDecision.EDIT),
    ("c", ChannelDecision.COMMENT),
    ("comment", ChannelDecision.COMMENT),
    ("yc", ChannelDecision.ALLOW_COMMENT),
    ("allow-comment", ChannelDecision.ALLOW_COMMENT),
])
def test_response_key_maps_to_decision(
    response_key: str, expected_decision: ChannelDecision,
) -> None:
    op = _StubOperator(
        next_response=PromptResponse(
            request_id="r-1", response=response_key,
        ),
    )
    channel = RunnerRPCChannel(op)
    response = channel.request_permission(_make_request())
    assert response.decision == expected_decision


def test_unknown_response_key_defaults_to_deny() -> None:
    """Unrecognized response strings default to DENY (same posture
    as the in-process channels)."""
    op = _StubOperator(
        next_response=PromptResponse(
            request_id="r-1", response="something_weird",
        ),
    )
    channel = RunnerRPCChannel(op)
    response = channel.request_permission(_make_request())
    assert response.decision == ChannelDecision.DENY


def test_response_key_case_insensitive() -> None:
    op = _StubOperator(
        next_response=PromptResponse(
            request_id="r-1", response="YES",
        ),
    )
    channel = RunnerRPCChannel(op)
    response = channel.request_permission(_make_request())
    assert response.decision == ChannelDecision.ALLOW


def test_response_key_strips_whitespace() -> None:
    op = _StubOperator(
        next_response=PromptResponse(
            request_id="r-1", response="  y  ",
        ),
    )
    channel = RunnerRPCChannel(op)
    response = channel.request_permission(_make_request())
    assert response.decision == ChannelDecision.ALLOW


# ----------------------------------------------------------------------
# Edit-with-arguments path
# ----------------------------------------------------------------------


def test_edit_response_propagates_edited_arguments() -> None:
    op = _StubOperator(
        next_response=PromptResponse(
            request_id="r-1",
            response="e",
            edited_arguments={"command": "ls -la"},
        ),
    )
    channel = RunnerRPCChannel(op)
    response = channel.request_permission(_make_request())
    assert response.decision == ChannelDecision.EDIT
    assert response.edited_arguments == {"command": "ls -la"}
    assert response.was_edited is True


def test_allow_with_edited_arguments_propagates() -> None:
    """ALLOW with edited_arguments is a legitimate combination
    (e.g., user edited then accepted)."""
    op = _StubOperator(
        next_response=PromptResponse(
            request_id="r-1",
            response="y",
            edited_arguments={"command": "ls -la"},
        ),
    )
    channel = RunnerRPCChannel(op)
    response = channel.request_permission(_make_request())
    assert response.decision == ChannelDecision.ALLOW
    assert response.edited_arguments == {"command": "ls -la"}
    assert response.was_edited is True


def test_deny_with_edited_arguments_does_not_set_was_edited() -> None:
    """DENY with edited_arguments shouldn't mark was_edited (the
    edit didn't take effect)."""
    op = _StubOperator(
        next_response=PromptResponse(
            request_id="r-1",
            response="n",
            edited_arguments={"command": "x"},
        ),
    )
    channel = RunnerRPCChannel(op)
    response = channel.request_permission(_make_request())
    assert response.decision == ChannelDecision.DENY
    assert response.was_edited is False


def test_no_edited_arguments_was_edited_false() -> None:
    op = _StubOperator(
        next_response=PromptResponse(request_id="r-1", response="y"),
    )
    channel = RunnerRPCChannel(op)
    response = channel.request_permission(_make_request())
    assert response.was_edited is False
    assert response.edited_arguments is None


# ----------------------------------------------------------------------
# Comment field propagation
# ----------------------------------------------------------------------


def test_response_comment_carried_into_reason() -> None:
    op = _StubOperator(
        next_response=PromptResponse(
            request_id="r-1",
            response="c",
            comment="this command is dangerous",
        ),
    )
    channel = RunnerRPCChannel(op)
    response = channel.request_permission(_make_request())
    assert response.reason == "this command is dangerous"


def test_no_comment_reason_empty() -> None:
    op = _StubOperator(
        next_response=PromptResponse(request_id="r-1", response="y"),
    )
    channel = RunnerRPCChannel(op)
    response = channel.request_permission(_make_request())
    assert response.reason == ""


# ----------------------------------------------------------------------
# Transport failure → DENY-with-reason
# ----------------------------------------------------------------------


def test_prompt_operator_raises_returns_deny() -> None:
    """An exception from the prompt_operator (RPC transport failure,
    handler crash, etc.) translates to a DENY response carrying the
    error message in the reason field — same posture as the
    in-process channels treat transport failures."""
    from server.runner_rpc_client import RunnerCallError

    op = _StubOperator(
        next_raise=RunnerCallError("daemon disconnected mid-prompt"),
    )
    channel = RunnerRPCChannel(op)
    request = _make_request()
    response = channel.request_permission(request)
    assert response.decision == ChannelDecision.DENY
    assert response.request_id == request.request_id
    assert "daemon disconnected" in response.reason


def test_prompt_operator_raises_generic_exception_returns_deny() -> None:
    op = _StubOperator(next_raise=ValueError("malformed payload"))
    channel = RunnerRPCChannel(op)
    response = channel.request_permission(_make_request())
    assert response.decision == ChannelDecision.DENY
    assert "malformed payload" in response.reason


class TestFeedbackOptionLabels:
    """The two feedback options read as a pair, and both labels are accepted.

    ``comment`` was relabelled ``deny-comment`` so it names its decision the
    way ``allow-comment`` does — the pair is deny-with-feedback vs
    allow-with-feedback, and "comment" alone said neither.  The response map
    keys off these strings, so the new label has to be accepted; the old one
    stays accepted for clients that have not been updated.
    """

    def test_the_option_is_labelled_deny_comment(self):
        from shared.plugins.permission.channels import (
            DEFAULT_PERMISSION_OPTIONS, ChannelDecision,
        )
        labels = {
            o.full for o in DEFAULT_PERMISSION_OPTIONS
            if o.decision is ChannelDecision.COMMENT
        }
        assert labels == {"deny-comment"}

    def test_it_pairs_with_allow_comment(self):
        from shared.plugins.permission.channels import (
            DEFAULT_PERMISSION_OPTIONS, ChannelDecision,
        )
        by_decision = {o.decision: o.full for o in DEFAULT_PERMISSION_OPTIONS}
        assert by_decision[ChannelDecision.COMMENT] == "deny-comment"
        assert by_decision[ChannelDecision.ALLOW_COMMENT] == "allow-comment"

    def test_the_new_label_resolves_to_the_comment_decision(self):
        from shared.plugins.permission.runner_rpc_channel import (
            _RESPONSE_KEY_TO_DECISION,
        )
        from shared.plugins.permission.channels import ChannelDecision
        assert _RESPONSE_KEY_TO_DECISION["deny-comment"] is ChannelDecision.COMMENT

    def test_the_old_label_is_still_accepted(self):
        """Clients predating the rename must keep working."""
        from shared.plugins.permission.runner_rpc_channel import (
            _RESPONSE_KEY_TO_DECISION,
        )
        from shared.plugins.permission.channels import ChannelDecision
        assert _RESPONSE_KEY_TO_DECISION["comment"] is ChannelDecision.COMMENT

    def test_the_shortcut_still_resolves(self):
        from shared.plugins.permission.runner_rpc_channel import (
            _RESPONSE_KEY_TO_DECISION,
        )
        from shared.plugins.permission.channels import ChannelDecision
        assert _RESPONSE_KEY_TO_DECISION["c"] is ChannelDecision.COMMENT
        assert _RESPONSE_KEY_TO_DECISION["yc"] is ChannelDecision.ALLOW_COMMENT
