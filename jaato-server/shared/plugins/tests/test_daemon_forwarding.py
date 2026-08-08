"""Tests for :mod:`shared.plugins.daemon_forwarding`.

Mirror of ``test_runner_forwarding`` for the reverse direction.
Covers:

- Daemon-side path: ``registry.runner_rpc_client`` absent → in-process
  executor runs.
- Runner-side path: ``registry.runner_rpc_client`` present →
  ``daemon_plugin_execute`` RPC is invoked with the right args.
- Per-call resolution: changing the registry attribute mid-life
  switches behaviour on the next call.
- Result + error translation: dict result returned unchanged; RPC
  errors translate to ``{"error": ...}``; CancelledException re-raises.
- ``__name__`` / ``__module__`` preserved on the wrapper.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from shared.plugins.daemon_forwarding import (
    DaemonForwardingMixin,
    _forward_via_daemon,
)


class _FakeRegistry:
    """Stand-in for the production plugin registry.

    Production runner-side wiring lives at
    ``server/runner/rpc.py:1097-1123`` and sets
    ``runner_rpc_client`` on the registry post-bootstrap.  Daemon-side
    never gains this attribute (its counterpart is ``runner_rpc`` for
    the opposite direction).  Tests model both sides by toggling
    whether the attribute exists.
    """

    def __init__(self, *, runner_rpc_client: Any = None) -> None:
        if runner_rpc_client is not None:
            self.runner_rpc_client = runner_rpc_client


class _FakePlugin(DaemonForwardingMixin):
    """Minimal plugin mirroring the cross-tier shape session_ops uses."""

    name = "fake_cross_tier"

    def __init__(self) -> None:
        self._plugin_registry: Any = None
        self.in_process_calls: list = []

    def _execute_demo(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Records the call + returns a dict; tests assert on the
        # recorded args to distinguish in-process from RPC paths.
        self.in_process_calls.append(args)
        return {"answer": "in-process", "echo": args}

    def get_executors(self) -> Dict[str, Any]:
        return self.wrap_executors_for_daemon_forwarding({
            "demo_tool": self._execute_demo,
        })


# ----------------------------------------------------------------------
# Daemon-side path (no runner_rpc_client attached → in-process body)
# ----------------------------------------------------------------------


def test_daemon_side_calls_in_process_body() -> None:
    """When the registry has NO ``runner_rpc_client`` (daemon-side),
    the wrapped executor calls the in-process body directly."""
    plugin = _FakePlugin()
    plugin._plugin_registry = _FakeRegistry()  # no rpc client

    executors = plugin.get_executors()
    result = executors["demo_tool"]({"q": "hello"})

    assert result == {"answer": "in-process", "echo": {"q": "hello"}}
    assert plugin.in_process_calls == [{"q": "hello"}]


def test_no_registry_attached_calls_in_process() -> None:
    """A plugin with no registry attribute at all (pre-set_plugin_registry
    state, or unit tests that bypass it) falls back to in-process.

    This matches RunnerForwardingMixin's behaviour when its
    ``_plugin_registry`` is None — both mixins are defensive against
    early-execution states.
    """
    plugin = _FakePlugin()
    # _plugin_registry stays None
    executors = plugin.get_executors()
    result = executors["demo_tool"]({"q": "early"})

    assert result == {"answer": "in-process", "echo": {"q": "early"}}


# ----------------------------------------------------------------------
# Runner-side path (runner_rpc_client present → forward via RPC)
# ----------------------------------------------------------------------


def test_runner_side_forwards_via_rpc() -> None:
    """When the registry has ``runner_rpc_client``, the wrapped
    executor forwards via ``daemon_plugin_execute``."""
    plugin = _FakePlugin()
    rpc_client = MagicMock()
    rpc_client.daemon_plugin_execute.return_value = {
        "answer": "from-daemon",
    }
    plugin._plugin_registry = _FakeRegistry(runner_rpc_client=rpc_client)

    executors = plugin.get_executors()
    result = executors["demo_tool"]({"q": "via-rpc"})

    assert result == {"answer": "from-daemon"}
    # In-process body NEVER ran.
    assert plugin.in_process_calls == []
    # RPC invoked with the right plugin / tool / args.
    rpc_client.daemon_plugin_execute.assert_called_once_with(
        plugin_name="fake_cross_tier",
        tool_name="demo_tool",
        args={"q": "via-rpc"},
    )


def test_runner_side_args_dict_is_defensively_copied() -> None:
    """The RPC args dict is a defensive copy — caller mutations
    don't leak into the on-the-wire payload."""
    plugin = _FakePlugin()
    captured_args: Dict[str, Any] = {}

    def _capture(*, plugin_name, tool_name, args):
        captured_args.update(args)
        # Mutate the caller's mutation surface AFTER capture to prove
        # the wire copy was made.
        return {"ok": True}

    rpc_client = MagicMock()
    rpc_client.daemon_plugin_execute.side_effect = _capture
    plugin._plugin_registry = _FakeRegistry(runner_rpc_client=rpc_client)

    user_args = {"key": "original"}
    executors = plugin.get_executors()
    executors["demo_tool"](user_args)
    user_args["key"] = "MUTATED-AFTER-CALL"

    # The dict we captured server-side wasn't the same object.
    assert captured_args == {"key": "original"}


# ----------------------------------------------------------------------
# Per-call resolution
# ----------------------------------------------------------------------


def test_per_call_resolution_picks_up_late_attached_rpc() -> None:
    """The registry attribute is set after plugin instantiation (the
    bootstrap order is plugin construct → set_plugin_registry →
    later: rpc client attaches).  The mixin must re-resolve on every
    call, not capture at decoration time."""
    plugin = _FakePlugin()
    registry = _FakeRegistry()  # no rpc client yet
    plugin._plugin_registry = registry

    executors = plugin.get_executors()
    # First call: no RPC → in-process.
    assert executors["demo_tool"]({"k": 1}) == {
        "answer": "in-process", "echo": {"k": 1},
    }
    assert plugin.in_process_calls == [{"k": 1}]

    # Attach the RPC client mid-life.
    rpc_client = MagicMock()
    rpc_client.daemon_plugin_execute.return_value = {"answer": "late-rpc"}
    registry.runner_rpc_client = rpc_client

    # Next call: RPC path.
    assert executors["demo_tool"]({"k": 2}) == {"answer": "late-rpc"}
    # In-process call list didn't grow.
    assert plugin.in_process_calls == [{"k": 1}]


# ----------------------------------------------------------------------
# Result translation
# ----------------------------------------------------------------------


def test_rpc_error_translates_to_error_dict() -> None:
    """When the wrapper method raises ``RunnerRPCError`` (typed RPC
    failure), the forwarder returns a clean error dict — same shape
    runner_forwarding produces."""
    from server.runner.rpc_client import RunnerRPCError

    plugin = _FakePlugin()
    rpc_client = MagicMock()
    rpc_client.daemon_plugin_execute.side_effect = RunnerRPCError(
        "daemon.plugin_execute failed: ValueError: bad args"
    )
    plugin._plugin_registry = _FakeRegistry(runner_rpc_client=rpc_client)

    executors = plugin.get_executors()
    result = executors["demo_tool"]({})

    assert isinstance(result, dict)
    assert "error" in result
    assert "demo_tool" in result["error"]
    assert "ValueError" in result["error"]


def test_transport_error_translates_to_error_dict() -> None:
    """Transport-level failures (channel closed, malformed frame, etc.)
    translate to ``{"error": ...}`` so the model sees a clean message
    rather than an opaque traceback."""
    plugin = _FakePlugin()
    rpc_client = MagicMock()
    rpc_client.daemon_plugin_execute.side_effect = OSError(
        "broken pipe"
    )
    plugin._plugin_registry = _FakeRegistry(runner_rpc_client=rpc_client)

    executors = plugin.get_executors()
    result = executors["demo_tool"]({})

    assert isinstance(result, dict)
    assert "error" in result
    assert "demo_tool" in result["error"]
    assert "OSError" in result["error"]


def test_cancelled_exception_re_raises() -> None:
    """When the RPC error indicates cancellation, the wrapper
    re-raises ``CancelledException`` so the in-process pipeline
    classifies the call as cancelled (same contract as
    runner_forwarding)."""
    from jaato_sdk.plugins.model_provider.types import CancelledException
    from server.runner.rpc_client import RunnerRPCError

    plugin = _FakePlugin()
    rpc_client = MagicMock()
    rpc_client.daemon_plugin_execute.side_effect = RunnerRPCError(
        "daemon.plugin_execute failed: CancelledException: user cancelled"
    )
    plugin._plugin_registry = _FakeRegistry(runner_rpc_client=rpc_client)

    executors = plugin.get_executors()
    with pytest.raises(CancelledException):
        executors["demo_tool"]({})


# ----------------------------------------------------------------------
# Function-attribute preservation
# ----------------------------------------------------------------------


def test_wrapper_preserves_name_and_module() -> None:
    """Wrapper preserves ``__name__`` + ``__module__`` for stack traces
    and debuggers — same shape runner_forwarding preserves."""
    plugin = _FakePlugin()
    executors = plugin.get_executors()
    wrapper = executors["demo_tool"]

    assert wrapper.__name__ == "_execute_demo"
    assert wrapper.__module__ == plugin._execute_demo.__module__


# ----------------------------------------------------------------------
# _forward_via_daemon module-level helper (unit-test without plugin)
# ----------------------------------------------------------------------


def test_forward_via_daemon_passes_args_through() -> None:
    """Module-level helper passes plugin/tool/args to the wrapper."""
    rpc_client = MagicMock()
    rpc_client.daemon_plugin_execute.return_value = "primitive-result"

    result = _forward_via_daemon(
        rpc_client, "session_ops", "interrogate_session",
        {"target": "abc"},
    )

    assert result == "primitive-result"
    rpc_client.daemon_plugin_execute.assert_called_once_with(
        plugin_name="session_ops",
        tool_name="interrogate_session",
        args={"target": "abc"},
    )
