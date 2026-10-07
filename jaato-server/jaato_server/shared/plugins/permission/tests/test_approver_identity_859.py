"""Who approved a tool call reaches the resolution (issue #859).

``PermissionResolvedEvent`` said HOW a decision was reached (``method``)
and nothing about WHO reached it, so an audit trail of approvals could
only be joined against telemetry spans by timestamp.  Two identities now
ride the resolution, kept apart because their provenance differs:

- ``user_id`` -- the identity the daemon authenticated for the client
  that answered.  Only the runner-RPC channel fills it, from a
  ``PromptResponse`` the daemon stamped from its transport.
- ``approver`` -- the name an external approval system attached to its
  webhook / file response.  Asserted, not verified; recorded as claimed.

Both are ``None`` for policy decisions, so "nobody was asked" stays
distinguishable from "somebody answered".
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import Mock

from ..channels import (
    ChannelDecision,
    ChannelResponse,
    _optional_identity,
)
from ..plugin import PermissionPlugin
from ..runner_rpc_channel import RunnerRPCChannel
from ..types import PromptResponse


# ---------------------------------------------------------------- channels


def test_channel_response_reads_approver_and_user_id_from_dict() -> None:
    """A webhook / file response names its approver with no extra contract."""
    response = ChannelResponse.from_dict({
        "request_id": "r",
        "decision": "allow",
        "reason": "Approved by admin",
        "approver": "alice@example.com",
        "user_id": "sso|alice",
    })
    assert response.approver == "alice@example.com"
    assert response.user_id == "sso|alice"


def test_channel_response_identity_absent_is_none() -> None:
    response = ChannelResponse.from_dict({"decision": "allow"})
    assert response.approver is None
    assert response.user_id is None


def test_channel_response_blank_identity_is_none() -> None:
    """An empty or whitespace name is no name -- never an empty string."""
    response = ChannelResponse.from_dict({"decision": "allow", "approver": "  "})
    assert response.approver is None
    assert _optional_identity(None) is None
    assert _optional_identity("") is None
    assert _optional_identity(42) == "42"


def test_channel_response_to_dict_emits_identity_only_when_set() -> None:
    bare = ChannelResponse(request_id="r", decision=ChannelDecision.ALLOW)
    assert "approver" not in bare.to_dict()
    assert "user_id" not in bare.to_dict()

    named = ChannelResponse(
        request_id="r", decision=ChannelDecision.ALLOW,
        user_id="u", approver="a",
    )
    d = named.to_dict()
    assert d["user_id"] == "u"
    assert d["approver"] == "a"
    assert ChannelResponse.from_dict(d) == named


def test_attribution_names_only_the_fields_that_are_set() -> None:
    assert ChannelResponse(
        request_id="r", decision=ChannelDecision.ALLOW,
    ).attribution() == {}
    assert ChannelResponse(
        request_id="r", decision=ChannelDecision.ALLOW, approver="a",
    ).attribution() == {"approver": "a"}
    assert ChannelResponse(
        request_id="r", decision=ChannelDecision.DENY, user_id="u", approver="a",
    ).attribution() == {"user_id": "u", "approver": "a"}


# -------------------------------------------------------- runner RPC wire


def test_prompt_response_user_id_round_trips() -> None:
    r = PromptResponse(request_id="r", response="y", user_id="sso|alice")
    back = PromptResponse.from_dict(r.to_dict())
    assert back == r
    assert back.user_id == "sso|alice"


def test_prompt_response_from_older_daemon_has_no_user() -> None:
    """A daemon that predates the field sends no key; the runner sees None."""
    back = PromptResponse.from_dict({"request_id": "r", "response": "y"})
    assert back.user_id is None
    assert PromptResponse.from_dict(
        {"request_id": "r", "response": "y", "user_id": ""},
    ).user_id is None


def test_runner_rpc_channel_carries_daemon_authenticated_user() -> None:
    """The daemon's stamp on the PromptResponse becomes the channel's
    ``user_id`` -- the verified half of the trail."""
    def _operator(payload: Any) -> PromptResponse:
        return PromptResponse(
            request_id=payload.request_id, response="y", user_id="sso|alice",
        )

    channel = RunnerRPCChannel(_operator)
    from ..channels import PermissionRequest
    response = channel.request_permission(
        PermissionRequest.create(tool_name="cli", arguments={}),
    )
    assert response.decision == ChannelDecision.ALLOW
    assert response.user_id == "sso|alice"
    assert response.approver is None


def test_runner_rpc_channel_without_user_leaves_none() -> None:
    channel = RunnerRPCChannel(
        lambda p: PromptResponse(request_id=p.request_id, response="n"),
    )
    from ..channels import PermissionRequest
    response = channel.request_permission(
        PermissionRequest.create(tool_name="cli", arguments={}),
    )
    assert response.user_id is None


# --------------------------------------------------------------- plugin


def _asking_plugin(response: ChannelResponse) -> PermissionPlugin:
    plugin = PermissionPlugin()
    plugin.initialize({"policy": {"defaultPolicy": "ask"}})
    channel = Mock()
    channel.name = "webhook"
    channel.request_permission.return_value = response
    plugin._channel = channel
    return plugin


def _capture_hook(plugin: PermissionPlugin) -> List[Dict[str, Any]]:
    seen: List[Dict[str, Any]] = []

    def on_resolved(tool_name, request_id, granted, method, **kw):
        seen.append({
            "tool_name": tool_name, "request_id": request_id,
            "granted": granted, "method": method, **kw,
        })

    plugin.set_permission_hooks(on_resolved=on_resolved)
    return seen


def test_channel_attribution_reaches_metadata_hook_and_audit_log() -> None:
    plugin = _asking_plugin(ChannelResponse(
        request_id="r", decision=ChannelDecision.ALLOW,
        reason="Approved by admin", user_id="sso|alice", approver="Alice",
    ))
    seen = _capture_hook(plugin)

    allowed, meta = plugin.check_permission("deploy", {"env": "prod"})

    assert allowed is True
    assert meta["method"] == "user_approved"
    assert meta["user_id"] == "sso|alice"
    assert meta["approver"] == "Alice"

    assert len(seen) == 1
    assert seen[0]["granted"] is True
    assert seen[0]["user_id"] == "sso|alice"
    assert seen[0]["approver"] == "Alice"

    # The plugin's own execution log -- its audit trail -- names them too.
    last = plugin._execution_log[-1]
    assert last["tool_name"] == "deploy"
    assert last["decision"] == "allow"
    assert last["user_id"] == "sso|alice"
    assert last["approver"] == "Alice"


def test_denial_is_attributed_too() -> None:
    plugin = _asking_plugin(ChannelResponse(
        request_id="r", decision=ChannelDecision.DENY,
        reason="no", approver="Bob",
    ))
    seen = _capture_hook(plugin)
    allowed, meta = plugin.check_permission("deploy", {})
    assert allowed is False
    assert meta["approver"] == "Bob"
    assert "user_id" not in meta
    assert seen[0]["approver"] == "Bob"
    assert seen[0]["user_id"] is None
    assert plugin._execution_log[-1]["approver"] == "Bob"


def test_unattributed_channel_answer_leaves_no_identity() -> None:
    """A console answer names nobody: no keys, no empty strings."""
    plugin = _asking_plugin(ChannelResponse(
        request_id="r", decision=ChannelDecision.ALLOW, reason="ok",
    ))
    seen = _capture_hook(plugin)
    _, meta = plugin.check_permission("deploy", {})
    assert "user_id" not in meta and "approver" not in meta
    assert seen[0]["user_id"] is None and seen[0]["approver"] is None
    assert "user_id" not in plugin._execution_log[-1]


def test_policy_decision_names_nobody() -> None:
    """A whitelist hit fires the hook without identity keywords, so a hook
    must accept them with defaults -- and 'nobody was asked' stays
    distinguishable from 'somebody answered'."""
    plugin = PermissionPlugin()
    plugin.initialize({"policy": {
        "defaultPolicy": "ask",
        "whitelist": {"tools": ["safe_tool"]},
    }})
    seen = _capture_hook(plugin)
    allowed, meta = plugin.check_permission("safe_tool", {})
    assert allowed is True
    assert "user_id" not in meta and "approver" not in meta
    assert seen and seen[0]["method"] == "whitelist"
    assert "user_id" not in seen[0] and "approver" not in seen[0]
