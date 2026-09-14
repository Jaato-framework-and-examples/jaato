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

import asyncio
from typing import List

import pytest

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.events import ClientType
from jaato_sdk.events import (
    Event,
    InjectPromptRequest,
    InjectPromptResultEvent,
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
    async def test_inject_prompt_reports_status_when_daemon_can_answer(
        self, client_capture,
    ):
        """Protocol 1.3+: the call correlates and RETURNS the status.

        Before this the method returned ``None`` unconditionally -- the
        runner's receipt was discarded by the daemon and nothing reached the
        caller -- so a driver got identical silence whether its target was
        busy, idle, stranded, or dead, and read that silence as "sent".
        """
        client, captured = client_capture
        client._server_protocol_version = "1.3"

        async def fake_await(request_id):
            assert request_id, "must correlate the wait with the request"
            return "queued"

        client._await_inject_result = fake_await  # type: ignore[assignment]

        status = await client.inject_prompt("steer me")

        assert status == "queued"
        assert captured[0].request_id, "1.3 request must carry a request_id"

    @pytest.mark.asyncio
    async def test_inject_prompt_returns_none_on_older_daemon(
        self, client_capture,
    ):
        """An older daemon cannot answer, so the status is UNKNOWN.

        ``None`` means "I was not told", not "delivery failed" -- returned
        rather than a placeholder string so the difference stays checkable.
        The request still goes out with no ``request_id``, preserving the
        pre-1.3 fire-and-forget shape.
        """
        client, captured = client_capture
        client._server_protocol_version = "1.2"

        status = await client.inject_prompt("steer me")

        assert status is None
        assert captured[0].request_id is None

    @pytest.mark.asyncio
    async def test_await_inject_result_ignores_another_call_s_result(self):
        """Correlation is what makes the wait about THIS inject.

        Matching on shape alone would let a concurrent inject's result
        satisfy this wait -- the same defect ``_correlates`` was added to
        fix for ``session.new``.
        """
        client = IPCClient(client_type=ClientType.API)
        # ``_await_inject_result`` subscribes its OWN queue, and
        # ``_subscribe_events`` drains ``_buffered_events`` into it first --
        # so seeding the buffer is how a test feeds it deterministically,
        # with no sleep and no race against the subscribe.
        client._buffered_events = [
            InjectPromptResultEvent(request_id="req_other", status="terminated"),
            InjectPromptResultEvent(request_id="req_mine", status="accepted"),
        ]

        status = await asyncio.wait_for(
            client._await_inject_result("req_mine"), timeout=2.0,
        )

        assert status == "accepted", (
            "the other call's 'terminated' must not satisfy this wait"
        )

    def test_recovery_client_mirrors_the_inject_signature(self):
        """The recovery client must expose the SAME inject signature.

        Recovery-client parity has broken on two separate axes before (#585:
        the method surface AND constructor args), and a recovery client that
        silently kept the old ``-> None`` shape would hand a reconnecting
        driver the exact ambiguity this change removes -- only intermittently,
        which is worse than always.
        """
        import inspect

        from jaato_sdk.client.recovery import IPCRecoveryClient

        assert (
            inspect.signature(IPCRecoveryClient.inject_prompt)
            == inspect.signature(IPCClient.inject_prompt)
        )

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


def _answers(status):
    """Stand in for the daemon's InjectPromptResultEvent wait.

    Without it a 1.3+ inject blocks for its full timeout on a captured
    request nobody answers — ten seconds of nothing, per test.
    """
    async def _await(request_id):
        return status
    return _await


class TestResumeVerbsCarryAttachments:
    """#845 — ``session.wake`` and ``inject_prompt`` accept the ``attachments``
    ``send_message`` already accepted.

    A completion-gated voice session is designed to END, and the documented
    way back in is ``session.wake``.  With both resume verbs text-only, a
    session whose input is a spoken utterance could be started with audio and
    never driven again with it.  The limitation was invisible from the API
    surface — ``attachments`` is on the call a reader starts from, and was
    absent on the two they reach for next.
    """

    @pytest.mark.asyncio
    async def test_inject_normalizes_attachments_like_send_message(
            self, client_capture, tmp_path):
        client, captured = client_capture
        client._server_protocol_version = "1.5"
        client._await_inject_result = _answers("accepted")  # type: ignore[assignment]
        pic = tmp_path / "shot.png"
        pic.write_bytes(b"\x89PNG")

        await client.inject_prompt("look", attachments=[str(pic)])

        att = captured[0].attachments[0]
        assert att["mime_type"] == "image/png"
        assert att["display_name"] == "shot.png"
        # The same expansion send_message performs: a client-side PATH never
        # crosses the wire, because the daemon (esp. cross-host WS) cannot
        # read it.
        assert att["data"] == "iVBORw=="

    @pytest.mark.asyncio
    async def test_inject_without_attachments_sends_an_empty_list(
            self, client_capture):
        client, captured = client_capture
        client._server_protocol_version = "1.5"
        client._await_inject_result = _answers("accepted")  # type: ignore[assignment]
        await client.inject_prompt("steer")
        assert captured[0].attachments == []

    @pytest.mark.asyncio
    async def test_a_daemon_that_would_drop_the_bytes_is_refused(
            self, client_capture):
        """An additive optional field is normally safe to send blind — an
        older peer ignores it and the call degrades to what it always did.
        That reasoning holds for a ``request_id`` and NOT for an attachment:
        the degraded call is a turn driven WITHOUT the audio that was the
        whole message.  So it raises instead."""
        client, captured = client_capture
        client._server_protocol_version = "1.4"
        with pytest.raises(ValueError, match="DROP the attachments"):
            await client.inject_prompt("hi", attachments=[{"data": "QUJD"}])
        assert captured == []

    @pytest.mark.asyncio
    async def test_wake_session_puts_the_utterance_in_the_payload(
            self, client_capture):
        client, captured = client_capture
        client._server_protocol_version = "1.5"

        await client.wake_session(
            "sess_1", attachments=[{"mime_type": "audio/wav", "data": "QUJD"}],
            source="phone", event_id="e1")

        req = captured[0]
        assert req.command == "session.wake"
        assert req.payload["session_id"] == "sess_1"
        # Blank text is the NORMAL shape here: for a spoken message the
        # attachment IS the message (#838).
        assert req.payload["text"] == ""
        assert req.payload["source"] == "phone"
        assert req.payload["event_id"] == "e1"
        assert req.payload["attachments"][0]["mime_type"] == "audio/wav"

    @pytest.mark.asyncio
    async def test_wake_session_text_only_carries_no_attachments_key(
            self, client_capture):
        client, captured = client_capture
        client._server_protocol_version = "1.5"
        await client.wake_session("sess_1", "have another look")
        assert captured[0].payload["text"] == "have another look"
        assert "attachments" not in captured[0].payload

    @pytest.mark.asyncio
    async def test_wake_session_with_no_content_is_refused_client_side(
            self, client_capture):
        client, captured = client_capture
        client._server_protocol_version = "1.5"
        with pytest.raises(ValueError, match="text or attachments"):
            await client.wake_session("sess_1")
        assert captured == []

    @pytest.mark.asyncio
    async def test_wake_session_also_refuses_a_daemon_that_would_drop_bytes(
            self, client_capture):
        client, captured = client_capture
        client._server_protocol_version = "1.3"
        with pytest.raises(ValueError, match="DROP the attachments"):
            await client.wake_session("sess_1", attachments=[{"data": "QUJD"}])
        assert captured == []
