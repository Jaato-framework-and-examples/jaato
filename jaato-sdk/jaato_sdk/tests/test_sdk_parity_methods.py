"""Tests for the SDK-parity methods on :class:`IPCClient`.

The methods are thin wrappers that construct the correct typed
WS request and send it over the wire.  These tests intercept
``_send_event`` to assert the request shape — they don't touch
the daemon.

Together with the server-side handler tests in
``jaato-server/server/test_sdk_parity_handlers.py`` and the
wire-format baseline in ``test_events_wire_format.py``, the
three layers cover the full SDK feature parity contract — see
``project_backlog_sdk_feature_parity.md``.
"""

from typing import List

import pytest

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.events import ClientType
from jaato_sdk.events import (
    Event,
    InjectPromptRequest,
    PermissionAddBlacklistRequest,
    PermissionAddWhitelistRequest,
    PermissionClearRequest,
    PermissionPolicySnapshotRequest,
    PermissionRemoveRequest,
    PermissionSetDefaultRequest,
    ReplayMessagesRequest,
    ResolveForkPointRequest,
    SendMessageRequest,
)


@pytest.fixture
def client_capture():
    """Build an IPCClient and capture every event passed to _send_event."""
    client = IPCClient(client_type=ClientType.API)
    captured: List[Event] = []

    async def fake_send(event):
        captured.append(event)

    client._send_event = fake_send  # type: ignore[assignment]
    return client, captured


# ─────────────────────────────────────────────────────────────────────


class TestSendMessageParallelTools:

    @pytest.mark.asyncio
    async def test_default_parallel_tools_is_none(self, client_capture):
        client, captured = client_capture
        await client.send_message("hi")
        assert isinstance(captured[0], SendMessageRequest)
        assert captured[0].parallel_tools is None

    @pytest.mark.asyncio
    async def test_parallel_tools_true_propagates(self, client_capture):
        client, captured = client_capture
        await client.send_message("hi", parallel_tools=True)
        assert captured[0].parallel_tools is True

    @pytest.mark.asyncio
    async def test_parallel_tools_false_propagates(self, client_capture):
        client, captured = client_capture
        await client.send_message("hi", parallel_tools=False)
        assert captured[0].parallel_tools is False


class TestSessionPrimitives:

    @pytest.mark.asyncio
    async def test_inject_prompt_default_source_type_user(self, client_capture):
        client, captured = client_capture
        await client.inject_prompt("steer me")
        ev = captured[0]
        assert isinstance(ev, InjectPromptRequest)
        assert ev.text == "steer me"
        assert ev.source_type == "user"
        assert ev.source_id is None

    @pytest.mark.asyncio
    async def test_inject_prompt_followup_via_child(self, client_capture):
        client, captured = client_capture
        await client.inject_prompt("follow up", source_type="child", source_id="ui")
        ev = captured[0]
        assert ev.source_type == "child"
        assert ev.source_id == "ui"

    @pytest.mark.asyncio
    async def test_replay_messages_omitted_messages_means_continue(self, client_capture):
        client, captured = client_capture
        await client.replay_messages(request_id="r1")
        ev = captured[0]
        assert isinstance(ev, ReplayMessagesRequest)
        assert ev.request_id == "r1"
        assert ev.messages is None
        assert ev.timeout_seconds == 120.0

    @pytest.mark.asyncio
    async def test_replay_messages_explicit_args(self, client_capture):
        client, captured = client_capture
        msgs = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
        await client.replay_messages(
            request_id="r2", messages=msgs, timeout_seconds=30.0,
        )
        ev = captured[0]
        assert ev.messages == msgs
        assert ev.timeout_seconds == 30.0

    @pytest.mark.asyncio
    async def test_resolve_fork_point_after_message(self, client_capture):
        client, captured = client_capture
        await client.resolve_fork_point(request_id="r3", after_message=5)
        ev = captured[0]
        assert isinstance(ev, ResolveForkPointRequest)
        assert ev.after_message == 5
        assert ev.after_tool_call is None
        assert ev.after_timestamp is None

    @pytest.mark.asyncio
    async def test_resolve_fork_point_after_tool_call(self, client_capture):
        client, captured = client_capture
        await client.resolve_fork_point(request_id="r4", after_tool_call="call_42")
        ev = captured[0]
        assert ev.after_tool_call == "call_42"
        assert ev.after_message is None


class TestPermissionPolicyMethods:

    @pytest.mark.asyncio
    async def test_add_whitelist_tools_only(self, client_capture):
        client, captured = client_capture
        await client.add_whitelist_tools(tools=["read_file", "list_files"])
        ev = captured[0]
        assert isinstance(ev, PermissionAddWhitelistRequest)
        assert ev.tools == ["read_file", "list_files"]
        assert ev.patterns == []

    @pytest.mark.asyncio
    async def test_add_whitelist_tools_and_patterns(self, client_capture):
        client, captured = client_capture
        await client.add_whitelist_tools(
            tools=["t1"], patterns=["safe_*", "read_*"],
        )
        ev = captured[0]
        assert ev.tools == ["t1"]
        assert ev.patterns == ["safe_*", "read_*"]

    @pytest.mark.asyncio
    async def test_add_blacklist_tools(self, client_capture):
        client, captured = client_capture
        await client.add_blacklist_tools(tools=["dangerous"])
        ev = captured[0]
        assert isinstance(ev, PermissionAddBlacklistRequest)
        assert ev.tools == ["dangerous"]

    @pytest.mark.asyncio
    async def test_remove_permission_rules(self, client_capture):
        client, captured = client_capture
        await client.remove_permission_rules(
            target="whitelist", tools=["t1"], patterns=["p1"],
        )
        ev = captured[0]
        assert isinstance(ev, PermissionRemoveRequest)
        assert ev.target == "whitelist"
        assert ev.tools == ["t1"]
        assert ev.patterns == ["p1"]

    @pytest.mark.asyncio
    async def test_clear_permission_rules_default_target_all(self, client_capture):
        client, captured = client_capture
        await client.clear_permission_rules()
        ev = captured[0]
        assert isinstance(ev, PermissionClearRequest)
        assert ev.target == "all"

    @pytest.mark.asyncio
    async def test_clear_permission_rules_specific_target(self, client_capture):
        client, captured = client_capture
        await client.clear_permission_rules(target="blacklist")
        assert captured[0].target == "blacklist"

    @pytest.mark.asyncio
    async def test_set_default_policy(self, client_capture):
        client, captured = client_capture
        await client.set_default_policy("allow")
        ev = captured[0]
        assert isinstance(ev, PermissionSetDefaultRequest)
        assert ev.policy == "allow"

    @pytest.mark.asyncio
    async def test_request_policy_snapshot(self, client_capture):
        client, captured = client_capture
        await client.request_policy_snapshot(request_id="snap1")
        ev = captured[0]
        assert isinstance(ev, PermissionPolicySnapshotRequest)
        assert ev.request_id == "snap1"


class TestToolExecutionResponse:

    @pytest.mark.asyncio
    async def test_respond_to_tool_execution_success_path(self, client_capture):
        from jaato_sdk.events import ToolExecuteResultEvent
        client, captured = client_capture
        await client.respond_to_tool_execution(
            call_id="call_42", result='{"ok": true}',
        )
        ev = captured[0]
        assert isinstance(ev, ToolExecuteResultEvent)
        assert ev.call_id == "call_42"
        assert ev.result == '{"ok": true}'
        assert ev.error == ""

    @pytest.mark.asyncio
    async def test_respond_to_tool_execution_error_path(self, client_capture):
        from jaato_sdk.events import ToolExecuteResultEvent
        client, captured = client_capture
        await client.respond_to_tool_execution(
            call_id="call_99", error="tool crashed",
        )
        ev = captured[0]
        assert ev.call_id == "call_99"
        assert ev.result == ""
        assert ev.error == "tool crashed"


class TestSessionLifecycle:

    @pytest.mark.asyncio
    async def test_end_session_sends_session_end_command(self, client_capture):
        from jaato_sdk.events import CommandRequest
        client, captured = client_capture
        await client.end_session()
        ev = captured[0]
        assert isinstance(ev, CommandRequest)
        assert ev.command == "session.end"
        assert ev.args == []

    @pytest.mark.asyncio
    async def test_delete_session_carries_session_id(self, client_capture):
        from jaato_sdk.events import CommandRequest
        client, captured = client_capture
        await client.delete_session("sess_xyz")
        ev = captured[0]
        assert isinstance(ev, CommandRequest)
        assert ev.command == "session.delete"
        assert ev.args == ["sess_xyz"]
