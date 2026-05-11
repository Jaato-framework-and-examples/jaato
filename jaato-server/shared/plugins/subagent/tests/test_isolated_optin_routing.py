"""Tests for ``SubagentPlugin._dispatch_isolated_spawn`` (Phase 4 §4.3.7).

Routing logic that fires when ``agent_params.isolated=true`` is
detected AFTER profile resolution.  Builds profile_payload per
Audit 5's wire shape and calls
``RunnerRPCClient.spawn_isolated_runner``.

Test surfaces:
1. No runner_rpc_client → returns None (caller falls back).
2. RPC raises → returns SubagentResult error dict.
3. RPC returns ok=True → returns spawn-success shape with
   subagent_id + isolation diagnostic fields.
4. RPC returns ok=False → returns SubagentResult error with
   stage + error message surfaced.
5. profile_payload contains expected fields (model, provider,
   plugins with preload annotations, plugin_configs, etc.).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ..plugin import SubagentPlugin
from ..config import SubagentProfile


def _make_plugin_with_registry(runner_rpc_client=None):
    """Build a plugin with a runtime + registry exposing
    runner_rpc_client (or None to test the fallback)."""
    plugin = SubagentPlugin()
    registry = MagicMock()
    registry.runner_rpc_client = runner_rpc_client
    runtime = MagicMock()
    runtime.registry = registry
    plugin._runtime = runtime
    # Parent session with session_id for confused-deputy echo.
    parent = MagicMock()
    parent._session_id = "sess-A"
    plugin._parent_session = parent
    return plugin


def _make_profile(**overrides):
    base = {
        "name": "researcher",
        "description": "Deep research profile",
        "model": "claude-sonnet-4-5",
        "provider": "anthropic",
        "plugins": ["cli", "web_search"],
        "plugin_configs": {"cli": {"timeout": 30}},
        "system_instructions": "You are a researcher.",
        "max_turns": 25,
        "env": {"FOO": "bar"},
    }
    base.update(overrides)
    return SubagentProfile(**base)


class TestNoRunnerRpcClient:
    """When the parent session has no runner_rpc_client wired,
    the supervisor's isolated=true request MUST surface as an
    error envelope — never silently downgrade to the default-share
    path.  Per peer-review fix: ``the supervisor asked for
    kernel-level isolation and got none`` was a security-violating
    fallback.  The helper now surfaces ``stage=rpc_unavailable``
    so the supervisor sees the failure."""

    def test_returns_error_envelope_with_stage_rpc_unavailable(self):
        plugin = _make_plugin_with_registry(runner_rpc_client=None)
        result = plugin._dispatch_isolated_spawn(
            agent_id="agent-1",
            profile=_make_profile(),
            task="do thing",
            workspace_path="/work",
            agent_params={"isolated": True},
            display_name="researcher",
        )
        # Returns a SubagentResult error dict, NOT None.
        assert result is not None
        assert result["success"] is False
        assert "rpc_unavailable" in result["error"].lower()
        # Recovery instructions surface BOTH options:
        # re-create with apparmor opt-in, OR use default-share path.
        error_lower = result["error"].lower()
        assert "apparmor" in error_lower
        assert "default-share" in error_lower
        # No silent downgrade — supervisor sees the failure
        # explicitly and chooses recovery.


class TestRpcRaises:
    def test_returns_subagent_error_dict(self):
        rpc = MagicMock()
        rpc.spawn_isolated_runner.side_effect = RuntimeError("transport closed")
        plugin = _make_plugin_with_registry(runner_rpc_client=rpc)

        result = plugin._dispatch_isolated_spawn(
            agent_id="agent-1",
            profile=_make_profile(),
            task="do thing",
            workspace_path="/work",
            agent_params={"isolated": True},
            display_name="researcher",
        )

        assert result is not None
        assert result["success"] is False
        assert "spawn RPC failed" in result["error"]
        assert "RuntimeError" in result["error"]
        assert "transport closed" in result["error"]


class TestRpcOk:
    def test_returns_spawn_success_shape(self):
        rpc = MagicMock()
        rpc.spawn_isolated_runner.return_value = {
            "ok": True,
            "session_id": "sess-A__sub_agent-1",
            "subagent_id": "agent-1",
            "runner_pid": 12345,
            "apparmor_profile": "jaato-ws-sess-A//agent-1",
            "cgroup_path": "/sys/fs/cgroup/jaato/jaato-ws-sess-A__sub_agent-1",
        }
        plugin = _make_plugin_with_registry(runner_rpc_client=rpc)

        result = plugin._dispatch_isolated_spawn(
            agent_id="agent-1",
            profile=_make_profile(),
            task="do thing",
            workspace_path="/work",
            agent_params={"isolated": True},
            display_name="researcher",
        )

        assert result["success"] is True
        assert result["subagent_id"] == "agent-1"
        assert result["status"] == "spawned"
        # Isolation-specific telemetry.
        telemetry = result["_telemetry"]
        assert telemetry["jaato.subagent.isolated"] is True
        assert (
            telemetry["jaato.subagent.apparmor_profile"]
            == "jaato-ws-sess-A//agent-1"
        )
        assert telemetry["jaato.subagent.cgroup_path"] != ""

    def test_rpc_called_with_expected_args(self):
        rpc = MagicMock()
        rpc.spawn_isolated_runner.return_value = {"ok": True}
        plugin = _make_plugin_with_registry(runner_rpc_client=rpc)

        plugin._dispatch_isolated_spawn(
            agent_id="agent-1",
            profile=_make_profile(),
            task="do thing",
            workspace_path="/work",
            agent_params={"isolated": True, "case_id": "42"},
            display_name="researcher",
        )

        rpc.spawn_isolated_runner.assert_called_once()
        kwargs = rpc.spawn_isolated_runner.call_args.kwargs
        assert kwargs["parent_session_id"] == "sess-A"
        assert kwargs["subagent_id"] == "agent-1"
        assert kwargs["task"] == "do thing"
        assert kwargs["workspace_path"] == "/work"
        assert kwargs["display_name"] == "researcher"
        # agent_params propagates (with isolated flag — daemon strips it).
        assert kwargs["agent_params"]["isolated"] is True
        assert kwargs["agent_params"]["case_id"] == "42"

    def test_profile_payload_carries_resolved_profile_fields(self):
        rpc = MagicMock()
        rpc.spawn_isolated_runner.return_value = {"ok": True}
        plugin = _make_plugin_with_registry(runner_rpc_client=rpc)

        plugin._dispatch_isolated_spawn(
            agent_id="agent-1",
            profile=_make_profile(),
            task="do thing",
            workspace_path="/work",
            agent_params={"isolated": True},
            display_name="researcher",
        )

        payload = rpc.spawn_isolated_runner.call_args.kwargs["profile_payload"]
        assert payload["name"] == "researcher"
        assert payload["model"] == "claude-sonnet-4-5"
        assert payload["provider"] == "anthropic"
        assert payload["plugins"] == ["cli", "web_search"]
        assert payload["plugin_configs"] == {"cli": {"timeout": 30}}
        assert payload["system_instructions"] == "You are a researcher."
        assert payload["max_turns"] == 25
        assert payload["env"] == {"FOO": "bar"}


class TestRpcOkFalse:
    def test_returns_subagent_error_with_stage(self):
        rpc = MagicMock()
        rpc.spawn_isolated_runner.return_value = {
            "ok": False,
            "stage": "sub_profile",
            "error": "AppArmor unavailable on this host",
        }
        plugin = _make_plugin_with_registry(runner_rpc_client=rpc)

        result = plugin._dispatch_isolated_spawn(
            agent_id="agent-1",
            profile=_make_profile(),
            task="do thing",
            workspace_path="/work",
            agent_params={"isolated": True},
            display_name="researcher",
        )

        assert result["success"] is False
        assert "stage=sub_profile" in result["error"]
        assert "AppArmor unavailable" in result["error"]


class TestPreloadAnnotations:
    def test_preload_round_trips_in_profile_payload(self):
        """``preloaded_plugins`` set serializes back to ``(preload)``
        annotations in the wire-shape plugins list."""
        rpc = MagicMock()
        rpc.spawn_isolated_runner.return_value = {"ok": True}
        plugin = _make_plugin_with_registry(runner_rpc_client=rpc)

        profile = _make_profile(
            plugins=["cli", "todo", "web_search"],
            preloaded_plugins={"todo", "web_search"},
        )

        plugin._dispatch_isolated_spawn(
            agent_id="agent-1",
            profile=profile,
            task="do thing",
            workspace_path="/work",
            agent_params={"isolated": True},
            display_name="researcher",
        )

        payload = rpc.spawn_isolated_runner.call_args.kwargs["profile_payload"]
        plugins_list = payload["plugins"]
        assert "cli" in plugins_list
        assert "todo(preload)" in plugins_list
        assert "web_search(preload)" in plugins_list
