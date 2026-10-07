"""``plugin_configs.permission.auto_allow_housekeeping`` (jaato/#1304 phase
3, server change #3): an explicit, WARNING-announced, default-off opt-in
that pre-whitelists ``HOUSEKEEPING_TOOLS`` so a housekeeping tool call
resolves via ``whitelist`` without ever reaching the approval channel --
matching this codebase's ``scrub_secret_env`` / ``allow_inline`` convention
for a posture that is weakened only by explicit, announced request.

Covers, each against a channel that raises if asked (so a pass can only
mean the call never reached it):

- the ROOT/runtime policy opt-in resolves a housekeeping tool with no ask;
- a NON-housekeeping tool is UNCHANGED by the opt-in and still asks;
- the opt-in is per-scope (#957's own contract): a subagent whose block
  opts in does not leak the whitelist onto the root policy or a sibling
  scope that did not opt in, and the reverse;
- ``get_permission_status()`` reports the flag and the resulting
  whitelist;
- a WARNING is logged on the opt-in, for both the root and the scoped
  routes;
- ``PermissionStatusEvent.auto_allow_housekeeping`` round-trips on the SDK
  wire.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pytest

from jaato_sdk.events import PermissionStatusEvent, deserialize_event, serialize_event

from ..channels import Channel, ChannelDecision, ChannelResponse, PermissionRequest
from ..plugin import PermissionPlugin


class _RaisingChannel(Channel):
    """A channel that must never be asked -- the load-bearing shape for
    every "resolves without a prompt" assertion in this file."""

    def __init__(self) -> None:
        self.requests: List[PermissionRequest] = []

    @property
    def name(self) -> str:
        return "raising"

    def request_permission(self, request: PermissionRequest) -> ChannelResponse:
        self.requests.append(request)
        raise AssertionError(
            f"channel asked for {request.tool_name!r} — a whitelisted "
            f"housekeeping tool must resolve without reaching the "
            f"approval channel"
        )

    def shutdown(self) -> None:
        pass


class _ScriptedChannel(Channel):
    """Answers every ask with a plain ALLOW_ONCE, recording each request."""

    def __init__(self) -> None:
        self.requests: List[PermissionRequest] = []

    @property
    def name(self) -> str:
        return "scripted"

    def request_permission(self, request: PermissionRequest) -> ChannelResponse:
        self.requests.append(request)
        return ChannelResponse(
            request_id=request.request_id,
            decision=ChannelDecision.ALLOW_ONCE,
            reason="ok",
        )

    def shutdown(self) -> None:
        pass


HOUSEKEEPING_TOOL = "createPlan"
NON_HOUSEKEEPING_TOOL = "writeNewFile"


class TestRootAutoAllowHousekeeping:
    def test_housekeeping_tool_resolves_without_asking(self) -> None:
        plugin = PermissionPlugin()
        plugin.initialize(
            {
                "policy": {"defaultPolicy": "ask"},
                "auto_allow_housekeeping": True,
            }
        )
        plugin._channel = _RaisingChannel()

        allowed, info = plugin.check_permission(HOUSEKEEPING_TOOL, {})

        assert allowed is True
        assert info["method"] == "whitelist"
        assert plugin._channel.requests == []

    def test_non_housekeeping_tool_still_asks(self) -> None:
        """The opt-in does not widen the whitelist beyond the closed
        ``HOUSEKEEPING_TOOLS`` set -- a file writer is unaffected."""
        plugin = PermissionPlugin()
        plugin.initialize(
            {
                "policy": {"defaultPolicy": "ask"},
                "auto_allow_housekeeping": True,
            }
        )
        channel = _ScriptedChannel()
        plugin._channel = channel

        allowed, info = plugin.check_permission(NON_HOUSEKEEPING_TOOL, {})

        assert allowed is True
        assert info["method"] == "user_approved"
        assert len(channel.requests) == 1
        assert channel.requests[0].tool_name == NON_HOUSEKEEPING_TOOL

    def test_default_off_still_asks_for_housekeeping_tools(self) -> None:
        """Control: with the knob absent (the default), a housekeeping
        tool is an ordinary ASK -- proving the whitelist above is the
        opt-in's doing, not something ``createPlan`` already had."""
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})
        channel = _ScriptedChannel()
        plugin._channel = channel

        allowed, info = plugin.check_permission(HOUSEKEEPING_TOOL, {})

        assert allowed is True
        assert info["method"] == "user_approved"
        assert len(channel.requests) == 1

    def test_logs_a_warning_on_opt_in(self, caplog: pytest.LogCaptureFixture) -> None:
        plugin = PermissionPlugin()
        with caplog.at_level(logging.WARNING, logger="jaato_server.shared.plugins.permission.plugin"):
            plugin.initialize(
                {
                    "policy": {"defaultPolicy": "ask"},
                    "auto_allow_housekeeping": True,
                }
            )
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("auto_allow_housekeeping" in r.getMessage() for r in warnings)

    def test_no_warning_when_left_off(self, caplog: pytest.LogCaptureFixture) -> None:
        plugin = PermissionPlugin()
        with caplog.at_level(logging.WARNING, logger="jaato_server.shared.plugins.permission.plugin"):
            plugin.initialize({"policy": {"defaultPolicy": "ask"}})
        warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and "auto_allow_housekeeping" in r.getMessage()
        ]
        assert warnings == []


class TestGetPermissionStatusReportsTheKnob:
    def test_status_reports_true_and_the_whitelist(self) -> None:
        plugin = PermissionPlugin()
        plugin.initialize(
            {
                "policy": {"defaultPolicy": "ask"},
                "auto_allow_housekeeping": True,
            }
        )
        status = plugin.get_permission_status()

        assert status["auto_allow_housekeeping"] is True
        assert HOUSEKEEPING_TOOL in status["whitelisted_tools"]
        assert status["whitelisted_tools"] == sorted(status["whitelisted_tools"])

    def test_status_reports_false_by_default(self) -> None:
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})
        status = plugin.get_permission_status()

        assert status["auto_allow_housekeeping"] is False
        assert HOUSEKEEPING_TOOL not in status["whitelisted_tools"]

    def test_shutdown_resets_the_flag(self) -> None:
        plugin = PermissionPlugin()
        plugin.initialize(
            {
                "policy": {"defaultPolicy": "ask"},
                "auto_allow_housekeeping": True,
            }
        )
        plugin.shutdown()
        assert plugin._auto_allow_housekeeping is False


class TestScopedAutoAllowHousekeepingDoesNotLeak:
    """The #957 contract, applied to this knob specifically: a subagent's
    own ``plugin_configs.permission`` block is what its own tool calls are
    judged by, and nothing about that block may change what a DIFFERENT
    session (the root, or a sibling subagent) is judged by."""

    def test_subagent_opt_in_does_not_leak_to_the_root_policy(self) -> None:
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})
        plugin.set_scoped_policy(
            "subagent-a",
            {
                "policy": {"defaultPolicy": "ask"},
                "auto_allow_housekeeping": True,
            },
        )

        # The subagent's own scope resolves the housekeeping tool with
        # no ask.
        plugin._channel = _RaisingChannel()
        allowed, info = plugin.check_permission(
            HOUSEKEEPING_TOOL, {}, context={"permission_scope": "subagent-a"}
        )
        assert allowed is True
        assert info["method"] == "whitelist"

        # The ROOT policy (no scope in context) was never touched by the
        # subagent's opt-in and still asks.
        channel = _ScriptedChannel()
        plugin._channel = channel
        allowed_root, info_root = plugin.check_permission(HOUSEKEEPING_TOOL, {})
        assert allowed_root is True
        assert info_root["method"] == "user_approved"
        assert len(channel.requests) == 1

    def test_root_opt_in_does_not_leak_to_a_subagent_that_declared_none(self) -> None:
        plugin = PermissionPlugin()
        plugin.initialize(
            {
                "policy": {"defaultPolicy": "ask"},
                "auto_allow_housekeeping": True,
            }
        )
        # A subagent whose OWN block declares a policy but no opt-in.
        plugin.set_scoped_policy(
            "subagent-b", {"policy": {"defaultPolicy": "ask"}}
        )

        channel = _ScriptedChannel()
        plugin._channel = channel
        allowed, info = plugin.check_permission(
            HOUSEKEEPING_TOOL, {}, context={"permission_scope": "subagent-b"}
        )
        assert allowed is True
        assert info["method"] == "user_approved"
        assert len(channel.requests) == 1

    def test_two_sibling_subagents_do_not_leak_to_each_other(self) -> None:
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})
        plugin.set_scoped_policy(
            "subagent-opted-in",
            {
                "policy": {"defaultPolicy": "ask"},
                "auto_allow_housekeeping": True,
            },
        )
        plugin.set_scoped_policy(
            "subagent-opted-out", {"policy": {"defaultPolicy": "ask"}}
        )

        plugin._channel = _RaisingChannel()
        allowed_a, info_a = plugin.check_permission(
            HOUSEKEEPING_TOOL, {},
            context={"permission_scope": "subagent-opted-in"},
        )
        assert allowed_a is True
        assert info_a["method"] == "whitelist"

        channel = _ScriptedChannel()
        plugin._channel = channel
        allowed_b, info_b = plugin.check_permission(
            HOUSEKEEPING_TOOL, {},
            context={"permission_scope": "subagent-opted-out"},
        )
        assert allowed_b is True
        assert info_b["method"] == "user_approved"
        assert len(channel.requests) == 1

    def test_scoped_opt_in_logs_a_warning_naming_the_scope(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})
        with caplog.at_level(logging.WARNING, logger="jaato_server.shared.plugins.permission.plugin"):
            plugin.set_scoped_policy(
                "subagent-a",
                {
                    "policy": {"defaultPolicy": "ask"},
                    "auto_allow_housekeeping": True,
                },
            )
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "auto_allow_housekeeping" in r.getMessage()
            and "subagent-a" in r.getMessage()
            for r in warnings
        )


class TestPermissionStatusEventRoundTrip:
    def test_auto_allow_housekeeping_round_trips(self) -> None:
        original = PermissionStatusEvent(
            effective_default="ask",
            suspension_scope=None,
            auto_allow_housekeeping=True,
        )
        restored = deserialize_event(serialize_event(original))
        assert isinstance(restored, PermissionStatusEvent)
        assert restored.auto_allow_housekeeping is True

    def test_defaults_to_none_when_unset(self) -> None:
        event = PermissionStatusEvent(effective_default="ask")
        assert event.auto_allow_housekeeping is None
        restored = deserialize_event(serialize_event(event))
        assert isinstance(restored, PermissionStatusEvent)
        assert restored.auto_allow_housekeeping is None
