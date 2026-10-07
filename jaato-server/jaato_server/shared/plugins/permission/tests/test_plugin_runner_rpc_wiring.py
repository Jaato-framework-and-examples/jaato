"""Tests for ``PermissionPlugin._get_channel()`` runner-RPC selection
(Phase 3 §3.7 deeper).

The runner-side permission plugin can't reach the connected client
through the in-process daemon-side ``Channel`` subclasses
(ConsoleChannel etc.) — the runner is in an AppArmor-confined
process with no client connection.  ASK decisions must relay through
the daemon's ``client.prompt_operator`` RPC primitive (§3.2.1).

The plugin handles this with a lazy resolution in ``_get_channel()``:
when ``self._registry.runner_rpc_client`` is set (the runner-side
:class:`server.runner.rpc_client.RunnerRPCClient` wrapper), an
instance of :class:`RunnerRPCChannel` is constructed and cached for
all subsequent ASK requests.

These tests pin the exact selection contract:

- Resolution order (thread-local > runner-RPC > default).
- ``runner_rpc_client`` attribute presence triggers selection.
- Absence falls back to the in-process channel.
- Cached on first hit (constructor not invoked twice).
- ``shutdown()`` resets the cache.
- Subagent thread-local channels still win over runner-RPC.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from ..channels import ConsoleChannel
from ..plugin import PermissionPlugin
from ..runner_rpc_channel import RunnerRPCChannel


class _StubRunnerRPCClient:
    """Minimal stand-in for ``server.runner.rpc_client.RunnerRPCClient``.

    The plugin only reads ``.prompt_operator`` off the attached
    instance; we don't need the full bidirectional RPC machinery in
    this unit test.
    """

    def __init__(self) -> None:
        self.prompt_operator = MagicMock(name="prompt_operator")


class _StubRegistry:
    """Stand-in for :class:`PluginRegistry` carrying a
    ``runner_rpc_client`` attribute.

    The plugin's :meth:`set_registry` accepts any object — the only
    thing the channel-resolution path reads is the
    ``runner_rpc_client`` attribute, mirroring the runtime contract
    documented on :class:`RunnerRPCChannel`."""

    def __init__(self, runner_rpc_client: Any = None) -> None:
        self.runner_rpc_client = runner_rpc_client


# ----------------------------------------------------------------------
# When no runner-RPC client is attached → fall back to default channel
# ----------------------------------------------------------------------


def test_no_runner_rpc_returns_default_channel() -> None:
    plugin = PermissionPlugin()
    plugin.initialize()  # default ConsoleChannel
    channel = plugin._get_channel()
    assert channel is plugin._channel
    assert isinstance(channel, ConsoleChannel)


def test_registry_without_runner_rpc_attribute_falls_back() -> None:
    """Daemon-side registry with no ``runner_rpc_client`` attribute
    at all → fall back to the in-process channel."""
    plugin = PermissionPlugin()
    plugin.initialize()
    plugin.set_registry(_StubRegistry(runner_rpc_client=None))
    channel = plugin._get_channel()
    assert channel is plugin._channel


def test_no_registry_falls_back() -> None:
    plugin = PermissionPlugin()
    plugin.initialize()
    # Don't call set_registry at all.
    channel = plugin._get_channel()
    assert channel is plugin._channel


# ----------------------------------------------------------------------
# When runner-RPC client is attached → return RunnerRPCChannel
# ----------------------------------------------------------------------


def test_runner_rpc_client_attached_returns_runner_channel() -> None:
    plugin = PermissionPlugin()
    plugin.initialize()
    plugin.set_registry(_StubRegistry(runner_rpc_client=_StubRunnerRPCClient()))
    channel = plugin._get_channel()
    assert isinstance(channel, RunnerRPCChannel)
    assert channel.name == "runner_rpc"


def test_runner_channel_carries_prompt_operator_callable() -> None:
    """The constructed RunnerRPCChannel uses
    ``runner_rpc_client.prompt_operator`` as its operator callable —
    proves the wiring is live, not a stub."""
    rpc_client = _StubRunnerRPCClient()
    plugin = PermissionPlugin()
    plugin.initialize()
    plugin.set_registry(_StubRegistry(runner_rpc_client=rpc_client))
    channel = plugin._get_channel()
    assert isinstance(channel, RunnerRPCChannel)
    # The channel stashes the callable as ``_prompt_operator``.
    assert channel._prompt_operator is rpc_client.prompt_operator


# ----------------------------------------------------------------------
# Caching: same instance returned on every call until shutdown.
# ----------------------------------------------------------------------


def test_runner_channel_cached_across_calls() -> None:
    plugin = PermissionPlugin()
    plugin.initialize()
    plugin.set_registry(_StubRegistry(runner_rpc_client=_StubRunnerRPCClient()))
    first = plugin._get_channel()
    second = plugin._get_channel()
    third = plugin._get_channel()
    assert first is second is third


def test_shutdown_clears_runner_channel_cache() -> None:
    """``shutdown()`` resets the cache so a subsequent
    ``initialize()`` (e.g., reconfigure) re-resolves cleanly."""
    plugin = PermissionPlugin()
    plugin.initialize()
    plugin.set_registry(_StubRegistry(runner_rpc_client=_StubRunnerRPCClient()))
    first = plugin._get_channel()
    assert isinstance(first, RunnerRPCChannel)

    plugin.shutdown()
    assert plugin._runner_rpc_channel is None

    # Re-init + re-attach a NEW rpc client — the channel must be a
    # different instance bound to the new operator.
    plugin.initialize()
    new_client = _StubRunnerRPCClient()
    plugin.set_registry(_StubRegistry(runner_rpc_client=new_client))
    second = plugin._get_channel()
    assert isinstance(second, RunnerRPCChannel)
    assert second is not first
    assert second._prompt_operator is new_client.prompt_operator


# ----------------------------------------------------------------------
# Resolution order: thread-local > runner-RPC > default.
# ----------------------------------------------------------------------


def test_thread_local_channel_wins_over_runner_rpc() -> None:
    """Subagents stash a per-thread channel via ``_thread_local.channel``.
    That channel must continue to win even when a runner-RPC client
    is attached — subagent isolation is independent of the runner-
    side ASK relay."""
    plugin = PermissionPlugin()
    plugin.initialize()
    plugin.set_registry(_StubRegistry(runner_rpc_client=_StubRunnerRPCClient()))

    custom = ConsoleChannel()
    plugin._thread_local.channel = custom
    try:
        channel = plugin._get_channel()
        assert channel is custom
    finally:
        # Cleanup: the thread-local persists across tests in the
        # same thread because PermissionPlugin._thread_local is a
        # class attribute.
        del plugin._thread_local.channel


# ----------------------------------------------------------------------
# Defensive: rpc_client without a prompt_operator attribute.
# ----------------------------------------------------------------------


class _IncompleteRPCClient:
    """An rpc_client that's missing ``prompt_operator`` — should not
    crash the plugin; falls back to the in-process channel."""

    pass


def test_rpc_client_without_prompt_operator_falls_back() -> None:
    plugin = PermissionPlugin()
    plugin.initialize()
    plugin.set_registry(_StubRegistry(runner_rpc_client=_IncompleteRPCClient()))
    channel = plugin._get_channel()
    # Falls back to the default channel rather than raising or
    # constructing a half-broken RunnerRPCChannel.
    assert channel is plugin._channel
