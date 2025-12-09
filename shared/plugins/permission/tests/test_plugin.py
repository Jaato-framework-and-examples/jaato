"""Tests for the permission plugin integration."""

import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

import pytest

from ..plugin import PermissionPlugin, create_plugin
from ..actors import ActorDecision, ActorResponse, PermissionRequest
from ..policy import PermissionDecision


class TestPermissionPluginInitialization:
    """Tests for plugin initialization."""

    def test_create_plugin_factory(self):
        plugin = create_plugin()
        assert isinstance(plugin, PermissionPlugin)

    def test_plugin_name(self):
        plugin = PermissionPlugin()
        assert plugin.name == "permission"

    def test_initialize_without_config(self):
        plugin = PermissionPlugin()
        plugin.initialize()
        assert plugin._initialized is True

    def test_initialize_with_inline_policy(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {
                "defaultPolicy": "allow",
                "blacklist": {"tools": ["blocked_tool"]},
            }
        })
        assert plugin._initialized is True
        assert plugin._policy is not None

    def test_initialize_from_config_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "version": "1.0",
                "defaultPolicy": "deny",
                "blacklist": {"tools": ["dangerous"]},
            }
            json.dump(config, f)
            f.flush()

            try:
                plugin = PermissionPlugin()
                plugin.initialize({"config_path": f.name})
                assert plugin._initialized is True
            finally:
                os.unlink(f.name)

    def test_initialize_with_actor_type(self):
        plugin = PermissionPlugin()
        plugin.initialize({"actor_type": "console"})
        assert plugin._actor is not None
        assert plugin._actor.name == "console"

    def test_initialize_fallback_to_console_actor(self):
        plugin = PermissionPlugin()
        # Webhook without endpoint should fail and fall back to console
        plugin.initialize({"actor_type": "webhook"})
        assert plugin._actor is not None
        assert plugin._actor.name == "console"  # Fallback

    def test_shutdown(self):
        plugin = PermissionPlugin()
        plugin.initialize()
        plugin.shutdown()

        assert plugin._initialized is False
        assert plugin._policy is None
        assert plugin._actor is None


class TestPermissionPluginFunctionDeclarations:
    """Tests for function declarations."""

    def test_get_tool_schemas(self):
        plugin = PermissionPlugin()
        declarations = plugin.get_tool_schemas()

        assert len(declarations) == 1
        assert declarations[0].name == "askPermission"

    def test_askPermission_schema(self):
        plugin = PermissionPlugin()
        schemas = plugin.get_tool_schemas()
        schema = schemas[0].parameters

        assert schema["type"] == "object"
        assert "tool_name" in schema["properties"]
        assert "arguments" in schema["properties"]
        assert "tool_name" in schema["required"]


class TestPermissionPluginExecutors:
    """Tests for executor methods."""

    def test_get_executors(self):
        plugin = PermissionPlugin()
        executors = plugin.get_executors()

        assert "askPermission" in executors
        assert callable(executors["askPermission"])
        # User command executor
        assert "permissions" in executors
        assert callable(executors["permissions"])

    def test_execute_ask_permission_requires_tool_name(self):
        plugin = PermissionPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        result = executors["askPermission"]({})
        assert "error" in result

    def test_execute_ask_permission_allowed(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {
                "defaultPolicy": "allow",
            }
        })
        executors = plugin.get_executors()

        result = executors["askPermission"]({
            "tool_name": "some_tool",
            "arguments": {}
        })
        assert result["allowed"] is True
        assert result["tool_name"] == "some_tool"

    def test_execute_ask_permission_denied(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {
                "defaultPolicy": "deny",
                "blacklist": {"tools": ["blocked_tool"]}
            }
        })
        executors = plugin.get_executors()

        result = executors["askPermission"]({
            "tool_name": "blocked_tool",
            "arguments": {}
        })
        assert result["allowed"] is False


class TestPermissionPluginCheckPermission:
    """Tests for check_permission method."""

    def test_check_permission_not_initialized(self):
        plugin = PermissionPlugin()
        # Don't initialize
        allowed, reason = plugin.check_permission("any_tool", {})
        assert allowed is True  # Defaults to allow when not initialized
        assert "not initialized" in reason.lower()

    def test_check_permission_allow_by_default(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "allow"}
        })

        allowed, reason = plugin.check_permission("any_tool", {})
        assert allowed is True

    def test_check_permission_deny_by_default(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "deny"}
        })

        allowed, reason = plugin.check_permission("any_tool", {})
        assert allowed is False

    def test_check_permission_blacklisted_tool(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {
                "defaultPolicy": "allow",
                "blacklist": {"tools": ["dangerous_tool"]}
            }
        })

        allowed, reason = plugin.check_permission("dangerous_tool", {})
        assert allowed is False

    def test_check_permission_whitelisted_tool(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {
                "defaultPolicy": "deny",
                "whitelist": {"tools": ["safe_tool"]}
            }
        })

        allowed, reason = plugin.check_permission("safe_tool", {})
        assert allowed is True

    def test_check_permission_blacklist_pattern(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {
                "defaultPolicy": "allow",
                "blacklist": {"patterns": ["rm -rf *"]}
            }
        })

        allowed, reason = plugin.check_permission(
            "cli_based_tool",
            {"command": "rm -rf /tmp/test"}
        )
        assert allowed is False

    def test_check_permission_whitelist_pattern(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {
                "defaultPolicy": "deny",
                "whitelist": {"patterns": ["git *"]}
            }
        })

        allowed, reason = plugin.check_permission(
            "cli_based_tool",
            {"command": "git status"}
        )
        assert allowed is True

    def test_check_permission_logs_decision(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "allow"}
        })

        plugin.check_permission("test_tool", {"arg": "val"})

        log = plugin.get_execution_log()
        assert len(log) == 1
        assert log[0]["tool_name"] == "test_tool"
        assert log[0]["decision"] == "allow"


class TestPermissionPluginActorInteraction:
    """Tests for actor interaction when policy returns ASK_ACTOR."""

    def test_ask_actor_allow(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "ask"}
        })

        # Mock the actor
        mock_actor = Mock()
        mock_actor.request_permission.return_value = ActorResponse(
            request_id="test",
            decision=ActorDecision.ALLOW,
            reason="User approved"
        )
        plugin._actor = mock_actor

        allowed, reason = plugin.check_permission("test_tool", {})
        assert allowed is True
        assert "approved" in reason.lower()

    def test_ask_actor_deny(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "ask"}
        })

        mock_actor = Mock()
        mock_actor.request_permission.return_value = ActorResponse(
            request_id="test",
            decision=ActorDecision.DENY,
            reason="User denied"
        )
        plugin._actor = mock_actor

        allowed, reason = plugin.check_permission("test_tool", {})
        assert allowed is False

    def test_ask_actor_allow_session(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "ask"}
        })

        mock_actor = Mock()
        mock_actor.request_permission.return_value = ActorResponse(
            request_id="test",
            decision=ActorDecision.ALLOW_SESSION,
            reason="Approved for session",
            remember_pattern="test_tool"
        )
        plugin._actor = mock_actor

        allowed, reason = plugin.check_permission("test_tool", {})
        assert allowed is True

        # Should be allowed without asking again
        plugin._actor.request_permission.reset_mock()
        allowed2, reason2 = plugin.check_permission("test_tool", {})
        assert allowed2 is True
        # Actor should not be called again
        mock_actor.request_permission.assert_not_called()

    def test_ask_actor_deny_session(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "ask"}
        })

        mock_actor = Mock()
        mock_actor.request_permission.return_value = ActorResponse(
            request_id="test",
            decision=ActorDecision.DENY_SESSION,
            reason="Denied for session",
            remember_pattern="test_tool"
        )
        plugin._actor = mock_actor

        allowed, reason = plugin.check_permission("test_tool", {})
        assert allowed is False

        # Should be denied without asking again
        plugin._actor.request_permission.reset_mock()
        allowed2, reason2 = plugin.check_permission("test_tool", {})
        assert allowed2 is False
        mock_actor.request_permission.assert_not_called()

    def test_ask_actor_timeout(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "ask"}
        })

        mock_actor = Mock()
        mock_actor.request_permission.return_value = ActorResponse(
            request_id="test",
            decision=ActorDecision.TIMEOUT,
            reason="Timeout"
        )
        plugin._actor = mock_actor

        allowed, reason = plugin.check_permission("test_tool", {})
        assert allowed is False

    def test_no_actor_configured(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "ask"}
        })
        plugin._actor = None  # Remove actor

        allowed, reason = plugin.check_permission("test_tool", {})
        assert allowed is False
        assert "no actor" in reason.lower()


class TestPermissionPluginExecutionLog:
    """Tests for execution logging."""

    def test_get_execution_log(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "allow"}
        })

        plugin.check_permission("tool1", {"arg": "val1"})
        plugin.check_permission("tool2", {"arg": "val2"})

        log = plugin.get_execution_log()
        assert len(log) == 2

    def test_clear_execution_log(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "allow"}
        })

        plugin.check_permission("tool1", {})
        assert len(plugin.get_execution_log()) == 1

        plugin.clear_execution_log()
        assert len(plugin.get_execution_log()) == 0

    def test_log_is_copy(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "allow"}
        })

        plugin.check_permission("tool1", {})
        log = plugin.get_execution_log()
        log.clear()

        # Original should be unchanged
        assert len(plugin.get_execution_log()) == 1


class TestPermissionPluginWrapExecutor:
    """Tests for executor wrapping."""

    def test_wrap_executor(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "allow"}
        })

        original_executor = Mock(return_value={"result": "success"})
        wrapped = plugin.wrap_executor("test_tool", original_executor)

        result = wrapped({"arg": "val"})

        original_executor.assert_called_once_with({"arg": "val"})
        assert result == {"result": "success"}

    def test_wrap_executor_blocks_denied(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {
                "defaultPolicy": "deny",
                "blacklist": {"tools": ["blocked_tool"]}
            }
        })

        original_executor = Mock(return_value={"result": "success"})
        wrapped = plugin.wrap_executor("blocked_tool", original_executor)

        result = wrapped({"arg": "val"})

        original_executor.assert_not_called()
        assert "error" in result
        assert "Permission denied" in result["error"]

    def test_wrap_all_executors(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "allow"}
        })

        executors = {
            "tool1": Mock(return_value={"r": 1}),
            "tool2": Mock(return_value={"r": 2}),
            "askPermission": Mock(return_value={"r": 3}),
        }

        wrapped = plugin.wrap_all_executors(executors)

        # All should be wrapped except askPermission
        assert len(wrapped) == 3

        # askPermission should be the original
        assert wrapped["askPermission"] is executors["askPermission"]

        # Others should be wrapped
        wrapped["tool1"]({"a": 1})
        executors["tool1"].assert_called_once()

    def test_wrap_all_executors_blocks_blacklisted(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {
                "defaultPolicy": "allow",
                "blacklist": {"tools": ["blocked_tool"]}
            }
        })

        executors = {
            "safe_tool": Mock(return_value={"r": 1}),
            "blocked_tool": Mock(return_value={"r": 2}),
        }

        wrapped = plugin.wrap_all_executors(executors)

        # Safe tool should work
        result1 = wrapped["safe_tool"]({"a": 1})
        assert result1 == {"r": 1}

        # Blocked tool should be denied
        result2 = wrapped["blocked_tool"]({"a": 1})
        assert "error" in result2
        executors["blocked_tool"].assert_not_called()


class TestPermissionPluginContextPassing:
    """Tests for context passing to actors."""

    def test_check_permission_passes_context(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "ask"}
        })

        mock_actor = Mock()
        mock_actor.request_permission.return_value = ActorResponse(
            request_id="test",
            decision=ActorDecision.ALLOW,
            reason="OK"
        )
        plugin._actor = mock_actor

        context = {"session_id": "abc123", "turn": 5}
        plugin.check_permission("test_tool", {"arg": "val"}, context=context)

        # Check that actor received the context
        call_args = mock_actor.request_permission.call_args
        request = call_args[0][0]
        assert isinstance(request, PermissionRequest)
        assert request.context == context


class TestPermissionPluginConfigOptions:
    """Tests for various configuration options."""

    def test_actor_timeout_from_config(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "actor": {
                    "type": "console",
                    "timeout": 120
                }
            }
            json.dump(config, f)
            f.flush()

            try:
                plugin = PermissionPlugin()
                plugin.initialize({"config_path": f.name})
                assert plugin._config.actor_timeout == 120
            finally:
                os.unlink(f.name)

    def test_actor_config_override(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "actor_type": "console",
            "actor_config": {"timeout": 60}
        })
        # Actor should be initialized with the config


class TestPermissionPluginEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unknown_actor_decision(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "ask"}
        })

        # Create a response with an unexpected decision value
        mock_actor = Mock()
        response = ActorResponse(
            request_id="test",
            decision=ActorDecision.ALLOW,  # Will be modified
            reason="OK"
        )
        # Modify to simulate unknown decision
        response.decision = Mock()
        response.decision.name = "UNKNOWN"
        mock_actor.request_permission.return_value = response
        plugin._actor = mock_actor

        allowed, reason = plugin.check_permission("test_tool", {})
        # Should default to deny for unknown decisions
        assert allowed is False

    def test_empty_arguments(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "allow"}
        })

        allowed, reason = plugin.check_permission("tool", {})
        assert allowed is True

    def test_complex_arguments(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "allow"}
        })

        complex_args = {
            "nested": {"deep": {"value": 123}},
            "list": [1, 2, 3],
            "null": None,
        }
        allowed, reason = plugin.check_permission("tool", complex_args)
        assert allowed is True


class TestActorCommunication:
    """Comprehensive tests for actor communication."""

    def test_request_contains_tool_name(self):
        """Verify actor receives correct tool_name."""
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})

        mock_actor = Mock()
        mock_actor.request_permission.return_value = ActorResponse(
            request_id="test",
            decision=ActorDecision.ALLOW,
            reason="OK"
        )
        plugin._actor = mock_actor

        plugin.check_permission("my_specific_tool", {"arg": "val"})

        request = mock_actor.request_permission.call_args[0][0]
        assert request.tool_name == "my_specific_tool"

    def test_request_contains_arguments(self):
        """Verify actor receives correct arguments."""
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})

        mock_actor = Mock()
        mock_actor.request_permission.return_value = ActorResponse(
            request_id="test",
            decision=ActorDecision.ALLOW,
            reason="OK"
        )
        plugin._actor = mock_actor

        test_args = {"command": "git status", "cwd": "/home/user"}
        plugin.check_permission("cli_based_tool", test_args)

        request = mock_actor.request_permission.call_args[0][0]
        assert request.arguments == test_args

    def test_request_contains_timeout(self):
        """Verify actor receives configured timeout."""
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {"defaultPolicy": "ask"},
            "actor_config": {"timeout": 45}
        })

        mock_actor = Mock()
        mock_actor.request_permission.return_value = ActorResponse(
            request_id="test",
            decision=ActorDecision.ALLOW,
            reason="OK"
        )
        plugin._actor = mock_actor

        plugin.check_permission("test_tool", {})

        request = mock_actor.request_permission.call_args[0][0]
        # Timeout should come from config
        assert request.timeout_seconds == plugin._config.actor_timeout

    def test_allow_once_does_not_remember(self):
        """ALLOW_ONCE should execute but still ask next time."""
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})

        mock_actor = Mock()
        mock_actor.request_permission.return_value = ActorResponse(
            request_id="test",
            decision=ActorDecision.ALLOW_ONCE,
            reason="Allowed once"
        )
        plugin._actor = mock_actor

        # First call - should be allowed
        allowed1, _ = plugin.check_permission("test_tool", {})
        assert allowed1 is True
        assert mock_actor.request_permission.call_count == 1

        # Second call - should ask actor again (not remembered)
        allowed2, _ = plugin.check_permission("test_tool", {})
        assert allowed2 is True
        assert mock_actor.request_permission.call_count == 2

    def test_allow_session_remembers_exact_tool(self):
        """ALLOW_SESSION with tool name should remember for that tool."""
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})

        mock_actor = Mock()
        mock_actor.request_permission.return_value = ActorResponse(
            request_id="test",
            decision=ActorDecision.ALLOW_SESSION,
            reason="Approved for session",
            remember_pattern="specific_tool"
        )
        plugin._actor = mock_actor

        # First call - asks actor
        allowed1, _ = plugin.check_permission("specific_tool", {})
        assert allowed1 is True
        assert mock_actor.request_permission.call_count == 1

        # Second call - should NOT ask actor (remembered)
        allowed2, _ = plugin.check_permission("specific_tool", {})
        assert allowed2 is True
        assert mock_actor.request_permission.call_count == 1  # Still 1, not called again

    def test_allow_session_pattern_matching(self):
        """ALLOW_SESSION with pattern 'git *' should match git commands."""
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})

        mock_actor = Mock()
        mock_actor.request_permission.return_value = ActorResponse(
            request_id="test",
            decision=ActorDecision.ALLOW_SESSION,
            reason="Git commands approved",
            remember_pattern="git *"
        )
        plugin._actor = mock_actor

        # First call with "git status" - asks actor
        allowed1, _ = plugin.check_permission(
            "cli_based_tool",
            {"command": "git status"}
        )
        assert allowed1 is True
        assert mock_actor.request_permission.call_count == 1

        # Second call with "git push" - should NOT ask actor (pattern match)
        allowed2, _ = plugin.check_permission(
            "cli_based_tool",
            {"command": "git push origin main"}
        )
        assert allowed2 is True
        # Pattern should match, so actor not called again
        assert mock_actor.request_permission.call_count == 1

    def test_deny_session_remembers(self):
        """DENY_SESSION should block subsequent calls without asking."""
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})

        mock_actor = Mock()
        mock_actor.request_permission.return_value = ActorResponse(
            request_id="test",
            decision=ActorDecision.DENY_SESSION,
            reason="Blocked for session",
            remember_pattern="dangerous_tool"
        )
        plugin._actor = mock_actor

        # First call - asks actor, gets denied
        allowed1, _ = plugin.check_permission("dangerous_tool", {})
        assert allowed1 is False
        assert mock_actor.request_permission.call_count == 1

        # Second call - should NOT ask actor (remembered denial)
        allowed2, _ = plugin.check_permission("dangerous_tool", {})
        assert allowed2 is False
        assert mock_actor.request_permission.call_count == 1  # Not called again

    def test_deny_session_pattern_blocking(self):
        """DENY_SESSION with pattern should block matching commands."""
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})

        mock_actor = Mock()
        mock_actor.request_permission.return_value = ActorResponse(
            request_id="test",
            decision=ActorDecision.DENY_SESSION,
            reason="rm commands blocked",
            remember_pattern="rm *"
        )
        plugin._actor = mock_actor

        # First call with "rm -rf" - asks actor
        allowed1, _ = plugin.check_permission(
            "cli_based_tool",
            {"command": "rm -rf /tmp/test"}
        )
        assert allowed1 is False
        assert mock_actor.request_permission.call_count == 1

        # Second call with "rm file.txt" - should NOT ask (pattern blocks)
        allowed2, _ = plugin.check_permission(
            "cli_based_tool",
            {"command": "rm file.txt"}
        )
        assert allowed2 is False
        assert mock_actor.request_permission.call_count == 1

    def test_different_tool_still_asks(self):
        """Session rule for one tool should not affect different tools."""
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})

        # Set up actor to return ALLOW_SESSION for first tool
        call_count = [0]

        def mock_request_permission(request):
            call_count[0] += 1
            return ActorResponse(
                request_id="test",
                decision=ActorDecision.ALLOW_SESSION,
                reason="Approved",
                remember_pattern="tool_a"
            )

        mock_actor = Mock()
        mock_actor.request_permission = mock_request_permission
        plugin._actor = mock_actor

        # Call tool_a - gets remembered
        plugin.check_permission("tool_a", {})
        assert call_count[0] == 1

        # Call tool_a again - not asked (remembered)
        plugin.check_permission("tool_a", {})
        assert call_count[0] == 1

        # Call tool_b - should ask actor (different tool)
        plugin.check_permission("tool_b", {})
        assert call_count[0] == 2

    def test_request_has_unique_id(self):
        """Each request should have a unique ID."""
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})

        request_ids = []

        def capture_request(request):
            request_ids.append(request.request_id)
            return ActorResponse(
                request_id=request.request_id,
                decision=ActorDecision.ALLOW,
                reason="OK"
            )

        mock_actor = Mock()
        mock_actor.request_permission = capture_request
        plugin._actor = mock_actor

        # Make multiple requests
        plugin.check_permission("tool1", {})
        plugin.check_permission("tool2", {})
        plugin.check_permission("tool3", {})

        # All IDs should be unique
        assert len(request_ids) == 3
        assert len(set(request_ids)) == 3  # All unique

    def test_request_has_timestamp(self):
        """Request should have a timestamp."""
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})

        captured_request = [None]

        def capture_request(request):
            captured_request[0] = request
            return ActorResponse(
                request_id=request.request_id,
                decision=ActorDecision.ALLOW,
                reason="OK"
            )

        mock_actor = Mock()
        mock_actor.request_permission = capture_request
        plugin._actor = mock_actor

        plugin.check_permission("test_tool", {})

        assert captured_request[0] is not None
        assert captured_request[0].timestamp is not None
        assert len(captured_request[0].timestamp) > 0


class TestPermissionPluginUserCommands:
    """Tests for the permissions user command."""

    def test_get_user_commands_returns_permissions(self):
        plugin = PermissionPlugin()
        commands = plugin.get_user_commands()

        assert len(commands) == 1
        assert commands[0].name == "permissions"
        assert commands[0].share_with_model is False

    def test_permissions_show_without_init(self):
        plugin = PermissionPlugin()
        result = plugin.execute_permissions({"args": ["show"]})

        assert "not initialized" in result.lower()

    def test_permissions_show_basic(self):
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {
                "defaultPolicy": "ask",
                "whitelist": {"tools": ["git"], "patterns": ["npm *"]},
                "blacklist": {"patterns": ["rm -rf *"]}
            }
        })

        result = plugin.execute_permissions({"args": ["show"]})

        assert "Effective Permission Policy" in result
        assert "Default Policy: ask" in result
        assert "git" in result
        assert "npm *" in result
        assert "rm -rf *" in result

    def test_permissions_show_no_args_defaults_to_show(self):
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "allow"}})

        result = plugin.execute_permissions({"args": []})

        assert "Effective Permission Policy" in result

    def test_permissions_allow(self):
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "deny"}})

        result = plugin.execute_permissions({"args": ["allow", "docker", "*"]})

        assert "+ Added to session whitelist: docker *" in result
        assert "docker *" in plugin._policy.session_whitelist

    def test_permissions_allow_missing_pattern(self):
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "deny"}})

        result = plugin.execute_permissions({"args": ["allow"]})

        assert "Usage:" in result

    def test_permissions_deny(self):
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "allow"}})

        result = plugin.execute_permissions({"args": ["deny", "cli_based_tool"]})

        assert "- Added to session blacklist: cli_based_tool" in result
        assert "cli_based_tool" in plugin._policy.session_blacklist

    def test_permissions_deny_missing_pattern(self):
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "allow"}})

        result = plugin.execute_permissions({"args": ["deny"]})

        assert "Usage:" in result

    def test_permissions_default_allow(self):
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})

        result = plugin.execute_permissions({"args": ["default", "allow"]})

        assert "Session default policy: allow" in result
        assert "was: ask" in result
        assert plugin._policy.session_default_policy == "allow"

    def test_permissions_default_deny(self):
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "allow"}})

        result = plugin.execute_permissions({"args": ["default", "deny"]})

        assert "Session default policy: deny" in result
        assert plugin._policy.session_default_policy == "deny"

    def test_permissions_default_ask(self):
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "deny"}})

        result = plugin.execute_permissions({"args": ["default", "ask"]})

        assert "Session default policy: ask" in result
        assert plugin._policy.session_default_policy == "ask"

    def test_permissions_default_invalid(self):
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})

        result = plugin.execute_permissions({"args": ["default", "invalid"]})

        assert "Invalid policy" in result
        assert plugin._policy.session_default_policy is None

    def test_permissions_default_missing_policy(self):
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})

        result = plugin.execute_permissions({"args": ["default"]})

        assert "Usage:" in result

    def test_permissions_clear(self):
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})

        # Add some session rules
        plugin._policy.add_session_whitelist("docker *")
        plugin._policy.add_session_blacklist("dangerous_tool")
        plugin._policy.set_session_default_policy("allow")

        result = plugin.execute_permissions({"args": ["clear"]})

        assert "Session rules cleared" in result
        assert len(plugin._policy.session_whitelist) == 0
        assert len(plugin._policy.session_blacklist) == 0
        assert plugin._policy.session_default_policy is None

    def test_permissions_unknown_subcommand(self):
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})

        result = plugin.execute_permissions({"args": ["unknown"]})

        assert "Unknown subcommand" in result
        assert "show" in result
        assert "allow" in result
        assert "deny" in result
        assert "default" in result
        assert "clear" in result

    def test_permissions_show_with_session_rules(self):
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "ask"}})

        plugin._policy.add_session_whitelist("docker *")
        plugin._policy.add_session_blacklist("dangerous_tool")
        plugin._policy.set_session_default_policy("allow")

        result = plugin.execute_permissions({"args": ["show"]})

        assert "Default Policy: allow (session override, was: ask)" in result
        assert "+ allow: docker *" in result
        assert "- deny:  dangerous_tool" in result

    def test_permissions_check_basic(self):
        """Test permissions check shows correct decision for a tool."""
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "deny"}})

        result = plugin.execute_permissions({"args": ["check", "some_tool"]})

        assert "some_tool" in result
        assert "DENY" in result
        assert "default" in result.lower()

    def test_permissions_check_whitelist(self):
        """Test permissions check shows ALLOW for whitelisted tool."""
        plugin = PermissionPlugin()
        plugin.initialize({
            "policy": {
                "defaultPolicy": "deny",
                "whitelist": {"tools": ["allowed_tool"]}
            }
        })

        result = plugin.execute_permissions({"args": ["check", "allowed_tool"]})

        assert "allowed_tool" in result
        assert "ALLOW" in result
        assert "whitelist" in result.lower()

    def test_permissions_check_session_override(self):
        """Test permissions check reflects session whitelist overriding blacklist pattern."""
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "deny"}})

        # Add pattern deny and explicit allow
        plugin._policy.add_session_blacklist("create*")
        plugin._policy.add_session_whitelist("createPlan")

        # createPlan should be ALLOWED (explicit override)
        result = plugin.execute_permissions({"args": ["check", "createPlan"]})
        assert "ALLOW" in result
        assert "session_whitelist" in result

        # createIssue should be DENIED (no explicit override)
        result = plugin.execute_permissions({"args": ["check", "createIssue"]})
        assert "DENY" in result
        assert "session_blacklist" in result

    def test_permissions_check_missing_tool(self):
        """Test permissions check requires a tool name."""
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "deny"}})

        result = plugin.execute_permissions({"args": ["check"]})

        assert "Usage" in result
        assert "check" in result

    def test_permissions_check_without_init(self):
        """Test permissions check handles uninitialized plugin."""
        plugin = PermissionPlugin()

        result = plugin.execute_permissions({"args": ["check", "some_tool"]})

        assert "not initialized" in result.lower()

    def test_session_default_affects_permission_check(self):
        """Verify session default policy actually affects permission checks."""
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "deny"}})

        # Default deny - should be denied
        allowed, _ = plugin.check_permission("unknown_tool", {})
        assert allowed is False

        # Set session default to allow
        plugin.execute_permissions({"args": ["default", "allow"]})

        # Now should be allowed
        allowed, _ = plugin.check_permission("another_tool", {})
        assert allowed is True

    def test_session_allow_affects_permission_check(self):
        """Verify session whitelist actually affects permission checks."""
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "deny"}})

        # Should be denied by default
        allowed, _ = plugin.check_permission("my_tool", {})
        assert allowed is False

        # Add to session whitelist
        plugin.execute_permissions({"args": ["allow", "my_tool"]})

        # Now should be allowed
        allowed, _ = plugin.check_permission("my_tool", {})
        assert allowed is True

    def test_session_deny_affects_permission_check(self):
        """Verify session blacklist actually affects permission checks."""
        plugin = PermissionPlugin()
        plugin.initialize({"policy": {"defaultPolicy": "allow"}})

        # Should be allowed by default
        allowed, _ = plugin.check_permission("some_tool", {})
        assert allowed is True

        # Add to session blacklist
        plugin.execute_permissions({"args": ["deny", "some_tool"]})

        # Now should be denied
        allowed, _ = plugin.check_permission("some_tool", {})
        assert allowed is False
