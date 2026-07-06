"""Unit tests for the in-process permission channel (PR2 mechanism).

Tests the cross-thread rendezvous in isolation — no embedded runtime / provider
/ network. The end-to-end demo (a real gated tool answered through the facade)
rides on the plugin-loading slice + the dual-mode example suite.
"""

import threading
import time

from jaato_embedded.permission import (
    InProcessChannel,
    PendingPermissions,
    decision_for,
)
from jaato_sdk.events import EventType
from shared.plugins.permission.channels import ChannelDecision, PermissionRequest


def test_decision_mapping_reuses_daemon_table():
    assert decision_for("y") == ChannelDecision.ALLOW
    assert decision_for("n") == ChannelDecision.DENY
    assert decision_for("a") == ChannelDecision.ALLOW_SESSION
    assert decision_for("t") == ChannelDecision.ALLOW_TURN
    # Fail closed: missing / unknown -> DENY.
    assert decision_for(None) == ChannelDecision.DENY
    assert decision_for("not-a-code") == ChannelDecision.DENY


class TestPendingPermissions:
    def test_resolve_roundtrip(self):
        pending = PendingPermissions()
        event = pending.register("req1")
        assert pending.resolve("req1", "y") is True
        assert event.is_set()
        assert pending.take("req1") == "y"

    def test_resolve_unknown_is_false(self):
        pending = PendingPermissions()
        assert pending.resolve("nope", "y") is False

    def test_take_after_pop_is_none(self):
        pending = PendingPermissions()
        pending.register("req1")
        pending.resolve("req1", "y")
        assert pending.take("req1") == "y"
        assert pending.take("req1") is None  # already popped


class TestInProcessChannelBridge:
    def test_request_permission_blocks_then_resolves(self):
        """``request_permission`` (worker thread) emits PERMISSION_REQUESTED and
        blocks; ``resolve`` (this thread) wakes it and the mapped decision comes
        back — the cross-thread bridge the facade rides on."""
        emitted = []
        pending = PendingPermissions()
        channel = InProcessChannel(emitted.append, pending)
        request = PermissionRequest.create(
            tool_name="cli", arguments={"command": "ls"}
        )

        result = {}

        def _worker():
            result["response"] = channel.request_permission(request)

        worker = threading.Thread(target=_worker)
        worker.start()

        # The worker emits the event and parks on the registry.
        for _ in range(200):
            if emitted:
                break
            time.sleep(0.01)
        assert len(emitted) == 1
        event = emitted[0]
        assert event.type == EventType.PERMISSION_REQUESTED
        assert event.tool_name == "cli"
        assert event.request_id == request.request_id
        assert event.tool_args == {"command": "ls"}
        assert worker.is_alive()  # still blocked, awaiting the response

        # Facade answers -> worker unblocks with the mapped decision.
        assert pending.resolve(request.request_id, "y") is True
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert result["response"].decision == ChannelDecision.ALLOW
        assert result["response"].request_id == request.request_id

    def test_deny_default_on_n(self):
        emitted = []
        pending = PendingPermissions()
        channel = InProcessChannel(emitted.append, pending)
        request = PermissionRequest.create(tool_name="cli", arguments={})

        result = {}
        worker = threading.Thread(
            target=lambda: result.__setitem__(
                "r", channel.request_permission(request)
            )
        )
        worker.start()
        for _ in range(200):
            if emitted:
                break
            time.sleep(0.01)
        pending.resolve(request.request_id, "n")
        worker.join(timeout=2)
        assert result["r"].decision == ChannelDecision.DENY
