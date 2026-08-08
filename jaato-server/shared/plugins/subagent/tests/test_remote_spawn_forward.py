"""Tests for the runner→daemon remote-spawn bridge.

Post-seat-flip the gossip remote-spawn handler is registered on the
DAEMON-side subagent instance, but ``spawn_subagent`` executes
runner-side where ``_remote_spawn_handler`` is None ("Gap #1 trap").
``_execute_spawn_subagent``'s ``server=`` branch bridges runner→daemon
via ``daemon.plugin_execute``, stamping ``parent_session_id`` (the
invoking session's daemon id, from ``get_current_session()._daemon_session_id``)
into the forwarded args so the daemon-side handler can inject results
back via ``inject_prompt_to_session``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from shared.session_context import _current_session, set_current_session

from ..plugin import SubagentPlugin


@pytest.fixture(autouse=True)
def _isolate_session_ctx():
    token = _current_session.set(None)
    try:
        yield
    finally:
        _current_session.reset(token)


def _make_plugin(*, runner_rpc_client, remote_handler=None):
    plugin = SubagentPlugin()
    plugin._initialized = True
    plugin._self_profile_name = None
    plugin._remote_spawn_handler = remote_handler
    registry = MagicMock()
    registry.runner_rpc_client = runner_rpc_client
    runtime = MagicMock()
    runtime.registry = registry
    plugin._runtime = runtime
    return plugin


_OK = {
    "success": True, "subagent_id": "x", "status": "spawned",
    "remote_server": "peer1", "message": "ok",
}


def test_runner_side_forwards_with_parent_session_id_stamped():
    """Runner-side (handler None, rpc_client present): the server= branch
    forwards spawn_subagent via daemon.plugin_execute, stamping
    parent_session_id from the current session, and round-trips the
    daemon-side result dict unchanged."""
    rpc = MagicMock()
    rpc.daemon_plugin_execute.return_value = _OK
    plugin = _make_plugin(runner_rpc_client=rpc, remote_handler=None)
    set_current_session(SimpleNamespace(_daemon_session_id="sess-A"))

    result = plugin._execute_spawn_subagent({"task": "do it", "server": "peer1"})

    rpc.daemon_plugin_execute.assert_called_once()
    kwargs = rpc.daemon_plugin_execute.call_args.kwargs
    assert kwargs["plugin_name"] == "subagent"
    assert kwargs["tool_name"] == "spawn_subagent"
    assert kwargs["args"]["server"] == "peer1"
    assert kwargs["args"]["parent_session_id"] == "sess-A"
    # Result round-trips unchanged (no envelope translation).
    assert result == _OK


def test_runner_side_no_current_session_stamps_none():
    """No session in context → parent_session_id forwarded as None
    (the forward still happens; daemon-side decides)."""
    rpc = MagicMock()
    rpc.daemon_plugin_execute.return_value = _OK
    plugin = _make_plugin(runner_rpc_client=rpc, remote_handler=None)
    # autouse fixture leaves the ContextVar at None → get returns None.

    plugin._execute_spawn_subagent({"task": "do it", "server": "peer1"})

    args = rpc.daemon_plugin_execute.call_args.kwargs["args"]
    assert args["parent_session_id"] is None


def test_daemon_side_calls_handler_with_parent_session_id_kwarg():
    """Daemon-side re-entry (handler registered, no runner channel):
    the server= branch invokes the handler with parent_session_id read
    from the forwarded args."""
    captured = {}

    def handler(**kwargs):
        captured.update(kwargs)
        return _OK

    plugin = _make_plugin(runner_rpc_client=None, remote_handler=handler)
    result = plugin._execute_spawn_subagent({
        "task": "do it", "server": "peer1", "parent_session_id": "sess-A",
    })

    assert captured["server"] == "peer1"
    assert captured["task"] == "do it"
    assert captured["parent_session_id"] == "sess-A"
    assert result == _OK


def test_no_premium_and_no_channel_returns_install_error():
    """Neither a registered handler nor a runner→daemon channel →
    premium genuinely absent → actionable install error."""
    plugin = _make_plugin(runner_rpc_client=None, remote_handler=None)
    result = plugin._execute_spawn_subagent({"task": "do it", "server": "peer1"})
    assert result["success"] is False
    assert "jaato-premium" in result["error"]
