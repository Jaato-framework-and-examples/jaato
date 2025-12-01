"""Tests for the actors module."""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from ..actors import (
    ActorDecision,
    PermissionRequest,
    ActorResponse,
    Actor,
    ConsoleActor,
    WebhookActor,
    FileActor,
    create_actor,
)


class TestActorDecision:
    """Tests for ActorDecision enum."""

    def test_decision_values(self):
        assert ActorDecision.ALLOW.value == "allow"
        assert ActorDecision.DENY.value == "deny"
        assert ActorDecision.ALLOW_ONCE.value == "allow_once"
        assert ActorDecision.ALLOW_SESSION.value == "allow_session"
        assert ActorDecision.DENY_SESSION.value == "deny_session"
        assert ActorDecision.TIMEOUT.value == "timeout"


class TestPermissionRequest:
    """Tests for PermissionRequest dataclass."""

    def test_create_request(self):
        request = PermissionRequest.create(
            tool_name="test_tool",
            arguments={"arg1": "value1"},
            timeout=60,
        )
        assert request.tool_name == "test_tool"
        assert request.arguments == {"arg1": "value1"}
        assert request.timeout_seconds == 60
        assert request.request_id  # Should be auto-generated
        assert request.timestamp  # Should be auto-generated
        assert request.context == {}

    def test_create_request_with_context(self):
        context = {"session_id": "123", "user": "test"}
        request = PermissionRequest.create(
            tool_name="test_tool",
            arguments={},
            context=context,
        )
        assert request.context == context

    def test_request_to_dict(self):
        request = PermissionRequest.create(
            tool_name="test_tool",
            arguments={"command": "ls"},
        )
        data = request.to_dict()
        assert data["tool_name"] == "test_tool"
        assert data["arguments"] == {"command": "ls"}
        assert "request_id" in data
        assert "timestamp" in data
        assert data["timeout_seconds"] == 30  # default
        assert data["default_on_timeout"] == "deny"

    def test_default_timeout(self):
        request = PermissionRequest.create(tool_name="test", arguments={})
        assert request.timeout_seconds == 30


class TestActorResponse:
    """Tests for ActorResponse dataclass."""

    def test_create_response(self):
        response = ActorResponse(
            request_id="abc123",
            decision=ActorDecision.ALLOW,
            reason="Test reason",
        )
        assert response.request_id == "abc123"
        assert response.decision == ActorDecision.ALLOW
        assert response.reason == "Test reason"
        assert response.remember is False
        assert response.remember_pattern is None

    def test_response_from_dict(self):
        data = {
            "request_id": "abc123",
            "decision": "allow",
            "reason": "Approved",
            "remember": True,
            "remember_pattern": "git *",
        }
        response = ActorResponse.from_dict(data)
        assert response.request_id == "abc123"
        assert response.decision == ActorDecision.ALLOW
        assert response.reason == "Approved"
        assert response.remember is True
        assert response.remember_pattern == "git *"

    def test_response_from_dict_unknown_decision(self):
        data = {"decision": "invalid_decision"}
        response = ActorResponse.from_dict(data)
        assert response.decision == ActorDecision.DENY  # Default to deny

    def test_response_from_dict_missing_fields(self):
        data = {}
        response = ActorResponse.from_dict(data)
        assert response.request_id == ""
        assert response.decision == ActorDecision.DENY
        assert response.reason == ""
        assert response.remember is False

    def test_response_to_dict(self):
        response = ActorResponse(
            request_id="abc123",
            decision=ActorDecision.ALLOW_SESSION,
            reason="Approved for session",
            remember=True,
            remember_pattern="npm *",
        )
        data = response.to_dict()
        assert data["request_id"] == "abc123"
        assert data["decision"] == "allow_session"
        assert data["reason"] == "Approved for session"
        assert data["remember"] is True
        assert data["remember_pattern"] == "npm *"


class TestConsoleActor:
    """Tests for ConsoleActor."""

    def test_name(self):
        actor = ConsoleActor()
        assert actor.name == "console"

    def test_yes_response(self):
        actor = ConsoleActor()
        outputs = []
        actor.initialize({
            "input_func": lambda: "y",
            "output_func": outputs.append,
        })

        request = PermissionRequest.create("test_tool", {"arg": "val"})
        response = actor.request_permission(request)

        assert response.decision == ActorDecision.ALLOW
        assert "approved" in response.reason.lower()

    def test_yes_full_word(self):
        actor = ConsoleActor()
        actor.initialize({
            "input_func": lambda: "yes",
            "output_func": lambda x: None,
        })

        request = PermissionRequest.create("test_tool", {})
        response = actor.request_permission(request)
        assert response.decision == ActorDecision.ALLOW

    def test_no_response(self):
        actor = ConsoleActor()
        actor.initialize({
            "input_func": lambda: "n",
            "output_func": lambda x: None,
        })

        request = PermissionRequest.create("test_tool", {})
        response = actor.request_permission(request)
        assert response.decision == ActorDecision.DENY

    def test_no_full_word(self):
        actor = ConsoleActor()
        actor.initialize({
            "input_func": lambda: "no",
            "output_func": lambda x: None,
        })

        request = PermissionRequest.create("test_tool", {})
        response = actor.request_permission(request)
        assert response.decision == ActorDecision.DENY

    def test_always_response(self):
        actor = ConsoleActor()
        actor.initialize({
            "input_func": lambda: "a",
            "output_func": lambda x: None,
        })

        request = PermissionRequest.create("test_tool", {})
        response = actor.request_permission(request)
        assert response.decision == ActorDecision.ALLOW_SESSION
        assert response.remember is True

    def test_always_full_word(self):
        actor = ConsoleActor()
        actor.initialize({
            "input_func": lambda: "always",
            "output_func": lambda x: None,
        })

        request = PermissionRequest.create("test_tool", {})
        response = actor.request_permission(request)
        assert response.decision == ActorDecision.ALLOW_SESSION

    def test_never_response(self):
        actor = ConsoleActor()
        actor.initialize({
            "input_func": lambda: "never",
            "output_func": lambda x: None,
        })

        request = PermissionRequest.create("test_tool", {})
        response = actor.request_permission(request)
        assert response.decision == ActorDecision.DENY_SESSION
        assert response.remember is True

    def test_once_response(self):
        actor = ConsoleActor()
        actor.initialize({
            "input_func": lambda: "once",
            "output_func": lambda x: None,
        })

        request = PermissionRequest.create("test_tool", {})
        response = actor.request_permission(request)
        assert response.decision == ActorDecision.ALLOW_ONCE

    def test_unknown_response(self):
        actor = ConsoleActor()
        actor.initialize({
            "input_func": lambda: "maybe",
            "output_func": lambda x: None,
        })

        request = PermissionRequest.create("test_tool", {})
        response = actor.request_permission(request)
        assert response.decision == ActorDecision.DENY
        assert "unknown" in response.reason.lower()

    def test_eof_error(self):
        def raise_eof():
            raise EOFError()

        actor = ConsoleActor()
        actor.initialize({
            "input_func": raise_eof,
            "output_func": lambda x: None,
        })

        request = PermissionRequest.create("test_tool", {})
        response = actor.request_permission(request)
        assert response.decision == ActorDecision.DENY
        assert "cancelled" in response.reason.lower()

    def test_keyboard_interrupt(self):
        def raise_interrupt():
            raise KeyboardInterrupt()

        actor = ConsoleActor()
        actor.initialize({
            "input_func": raise_interrupt,
            "output_func": lambda x: None,
        })

        request = PermissionRequest.create("test_tool", {})
        response = actor.request_permission(request)
        assert response.decision == ActorDecision.DENY

    def test_remember_pattern_cli_tool(self):
        actor = ConsoleActor()
        actor.initialize({
            "input_func": lambda: "always",
            "output_func": lambda x: None,
        })

        request = PermissionRequest.create(
            "cli_based_tool",
            {"command": "git status"}
        )
        response = actor.request_permission(request)
        assert response.remember_pattern == "git *"

    def test_remember_pattern_other_tool(self):
        actor = ConsoleActor()
        actor.initialize({
            "input_func": lambda: "always",
            "output_func": lambda x: None,
        })

        request = PermissionRequest.create("custom_tool", {"arg": "val"})
        response = actor.request_permission(request)
        assert response.remember_pattern == "custom_tool"

    def test_output_format(self):
        outputs = []
        actor = ConsoleActor()
        actor.initialize({
            "input_func": lambda: "y",
            "output_func": outputs.append,
        })

        request = PermissionRequest.create(
            "test_tool",
            {"command": "ls -la"},
            context={"session": "test123"}
        )
        actor.request_permission(request)

        output_text = "\n".join(outputs)
        assert "test_tool" in output_text
        assert "ls -la" in output_text
        assert "[y]es" in output_text
        assert "[n]o" in output_text


class TestWebhookActor:
    """Tests for WebhookActor."""

    def test_name(self):
        actor = WebhookActor()
        assert actor.name == "webhook"

    def test_initialize_requires_config(self):
        actor = WebhookActor()
        with pytest.raises(ValueError, match="requires configuration"):
            actor.initialize(None)

    def test_initialize_requires_endpoint(self):
        actor = WebhookActor()
        with pytest.raises(ValueError, match="requires configuration with 'endpoint'"):
            actor.initialize({})

    def test_no_endpoint_configured(self):
        actor = WebhookActor()
        # Don't initialize, endpoint will be None
        request = PermissionRequest.create("test", {})
        response = actor.request_permission(request)
        assert response.decision == ActorDecision.DENY
        assert "not configured" in response.reason

    @patch("shared.plugins.permission.actors.requests")
    def test_successful_allow(self, mock_requests):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "request_id": "test",
            "decision": "allow",
            "reason": "Approved by webhook",
        }
        mock_requests.post.return_value = mock_response
        mock_requests.Timeout = Exception
        mock_requests.RequestException = Exception

        actor = WebhookActor()
        actor.initialize({"endpoint": "http://example.com/webhook"})

        request = PermissionRequest.create("test_tool", {"arg": "val"})
        response = actor.request_permission(request)

        assert response.decision == ActorDecision.ALLOW
        assert "Approved" in response.reason

    @patch("shared.plugins.permission.actors.requests")
    def test_successful_deny(self, mock_requests):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "request_id": "test",
            "decision": "deny",
            "reason": "Denied by policy",
        }
        mock_requests.post.return_value = mock_response
        mock_requests.Timeout = Exception
        mock_requests.RequestException = Exception

        actor = WebhookActor()
        actor.initialize({"endpoint": "http://example.com/webhook"})

        request = PermissionRequest.create("test_tool", {})
        response = actor.request_permission(request)

        assert response.decision == ActorDecision.DENY

    @patch("shared.plugins.permission.actors.requests")
    def test_non_200_status(self, mock_requests):
        mock_response = Mock()
        mock_response.status_code = 500
        mock_requests.post.return_value = mock_response
        mock_requests.Timeout = Exception
        mock_requests.RequestException = Exception

        actor = WebhookActor()
        actor.initialize({"endpoint": "http://example.com/webhook"})

        request = PermissionRequest.create("test_tool", {})
        response = actor.request_permission(request)

        assert response.decision == ActorDecision.DENY
        assert "500" in response.reason

    @patch("shared.plugins.permission.actors.requests")
    def test_timeout_default_deny(self, mock_requests):
        mock_requests.Timeout = Exception

        def raise_timeout(*args, **kwargs):
            raise mock_requests.Timeout()

        mock_requests.post = raise_timeout
        mock_requests.RequestException = Exception

        actor = WebhookActor()
        actor.initialize({"endpoint": "http://example.com/webhook", "timeout": 5})

        request = PermissionRequest.create("test_tool", {})
        response = actor.request_permission(request)

        assert response.decision == ActorDecision.DENY
        assert "timeout" in response.reason.lower()

    @patch("shared.plugins.permission.actors.requests")
    def test_timeout_allow_on_timeout(self, mock_requests):
        mock_requests.Timeout = Exception

        def raise_timeout(*args, **kwargs):
            raise mock_requests.Timeout()

        mock_requests.post = raise_timeout
        mock_requests.RequestException = Exception

        actor = WebhookActor()
        actor.initialize({"endpoint": "http://example.com/webhook"})

        request = PermissionRequest(
            request_id="test",
            timestamp="2024-01-01T00:00:00Z",
            tool_name="test_tool",
            arguments={},
            default_on_timeout="allow"
        )
        response = actor.request_permission(request)

        assert response.decision == ActorDecision.ALLOW

    @patch("shared.plugins.permission.actors.requests")
    def test_request_exception(self, mock_requests):
        # Create distinct exception classes for proper exception handling
        class MockTimeout(Exception):
            pass

        class MockRequestException(Exception):
            pass

        mock_requests.Timeout = MockTimeout
        mock_requests.RequestException = MockRequestException

        def raise_error(*args, **kwargs):
            raise MockRequestException("Connection refused")

        mock_requests.post = raise_error

        actor = WebhookActor()
        actor.initialize({"endpoint": "http://example.com/webhook"})

        request = PermissionRequest.create("test_tool", {})
        response = actor.request_permission(request)

        assert response.decision == ActorDecision.DENY
        assert "failed" in response.reason.lower() or "connection" in response.reason.lower()

    @patch("shared.plugins.permission.actors.requests")
    def test_auth_token_header(self, mock_requests):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"decision": "allow"}
        mock_requests.post.return_value = mock_response
        mock_requests.Timeout = Exception
        mock_requests.RequestException = Exception

        actor = WebhookActor()
        actor.initialize({
            "endpoint": "http://example.com/webhook",
            "auth_token": "secret123",
        })

        request = PermissionRequest.create("test_tool", {})
        actor.request_permission(request)

        call_kwargs = mock_requests.post.call_args.kwargs
        assert "Authorization" in call_kwargs["headers"]
        assert "Bearer secret123" in call_kwargs["headers"]["Authorization"]

    @patch("shared.plugins.permission.actors.requests")
    def test_custom_headers(self, mock_requests):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"decision": "allow"}
        mock_requests.post.return_value = mock_response
        mock_requests.Timeout = Exception
        mock_requests.RequestException = Exception

        actor = WebhookActor()
        actor.initialize({
            "endpoint": "http://example.com/webhook",
            "headers": {"X-Custom": "value"},
        })

        request = PermissionRequest.create("test_tool", {})
        actor.request_permission(request)

        call_kwargs = mock_requests.post.call_args.kwargs
        assert "X-Custom" in call_kwargs["headers"]


class TestFileActor:
    """Tests for FileActor."""

    def test_name(self):
        actor = FileActor()
        assert actor.name == "file"

    def test_initialize_requires_config(self):
        actor = FileActor()
        with pytest.raises(ValueError, match="requires configuration"):
            actor.initialize(None)

    def test_initialize_requires_base_path(self):
        actor = FileActor()
        with pytest.raises(ValueError, match="requires 'base_path'|requires configuration with 'base_path'"):
            actor.initialize({})

    def test_initialize_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "permissions")
            actor = FileActor()
            actor.initialize({"base_path": base_path})

            assert os.path.isdir(os.path.join(base_path, "requests"))
            assert os.path.isdir(os.path.join(base_path, "responses"))

    def test_no_base_path_configured(self):
        actor = FileActor()
        # Don't initialize, base_path will be None
        request = PermissionRequest.create("test", {})
        response = actor.request_permission(request)
        assert response.decision == ActorDecision.DENY
        assert "not configured" in response.reason

    def test_request_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            actor = FileActor()
            actor.initialize({
                "base_path": tmpdir,
                "poll_interval": 0.1,
            })

            request = PermissionRequest(
                request_id="test123",
                timestamp="2024-01-01T00:00:00Z",
                tool_name="test_tool",
                arguments={"arg": "val"},
                timeout_seconds=1,
            )

            # Start request in a thread, write response file quickly
            import threading

            def write_response():
                time.sleep(0.2)
                response_file = os.path.join(tmpdir, "responses", "test123.json")
                with open(response_file, "w") as f:
                    json.dump({"decision": "allow", "reason": "Approved"}, f)

            thread = threading.Thread(target=write_response)
            thread.start()

            response = actor.request_permission(request)
            thread.join()

            assert response.decision == ActorDecision.ALLOW

    def test_timeout_default_deny(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            actor = FileActor()
            actor.initialize({
                "base_path": tmpdir,
                "poll_interval": 0.1,
            })

            request = PermissionRequest(
                request_id="test123",
                timestamp="2024-01-01T00:00:00Z",
                tool_name="test_tool",
                arguments={},
                timeout_seconds=0.3,  # Very short timeout
            )

            response = actor.request_permission(request)

            assert response.decision == ActorDecision.DENY
            assert "timeout" in response.reason.lower()

    def test_timeout_allow_on_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            actor = FileActor()
            actor.initialize({
                "base_path": tmpdir,
                "poll_interval": 0.1,
            })

            request = PermissionRequest(
                request_id="test123",
                timestamp="2024-01-01T00:00:00Z",
                tool_name="test_tool",
                arguments={},
                timeout_seconds=0.3,
                default_on_timeout="allow",
            )

            response = actor.request_permission(request)

            assert response.decision == ActorDecision.ALLOW

    def test_request_file_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            actor = FileActor()
            actor.initialize({
                "base_path": tmpdir,
                "poll_interval": 0.1,
            })

            request = PermissionRequest(
                request_id="test456",
                timestamp="2024-01-01T00:00:00Z",
                tool_name="test_tool",
                arguments={"command": "ls"},
                timeout_seconds=0.2,
            )

            import threading

            request_content = None

            def read_request():
                nonlocal request_content
                time.sleep(0.05)
                request_file = os.path.join(tmpdir, "requests", "test456.json")
                if os.path.exists(request_file):
                    with open(request_file) as f:
                        request_content = json.load(f)

            thread = threading.Thread(target=read_request)
            thread.start()
            actor.request_permission(request)
            thread.join()

            if request_content:
                assert request_content["tool_name"] == "test_tool"
                assert request_content["arguments"] == {"command": "ls"}

    def test_cleanup_files_on_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            actor = FileActor()
            actor.initialize({
                "base_path": tmpdir,
                "poll_interval": 0.1,
            })

            request = PermissionRequest(
                request_id="test789",
                timestamp="2024-01-01T00:00:00Z",
                tool_name="test_tool",
                arguments={},
                timeout_seconds=1,
            )

            import threading

            def write_response():
                time.sleep(0.1)
                response_file = os.path.join(tmpdir, "responses", "test789.json")
                with open(response_file, "w") as f:
                    json.dump({"decision": "allow"}, f)

            thread = threading.Thread(target=write_response)
            thread.start()

            actor.request_permission(request)
            thread.join()

            # Both files should be cleaned up
            request_file = os.path.join(tmpdir, "requests", "test789.json")
            response_file = os.path.join(tmpdir, "responses", "test789.json")
            assert not os.path.exists(request_file)
            assert not os.path.exists(response_file)

    def test_cleanup_request_file_on_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            actor = FileActor()
            actor.initialize({
                "base_path": tmpdir,
                "poll_interval": 0.1,
            })

            request = PermissionRequest(
                request_id="test_cleanup",
                timestamp="2024-01-01T00:00:00Z",
                tool_name="test_tool",
                arguments={},
                timeout_seconds=0.2,
            )

            actor.request_permission(request)

            # Request file should be cleaned up
            request_file = os.path.join(tmpdir, "requests", "test_cleanup.json")
            assert not os.path.exists(request_file)

    def test_shutdown_cleans_pending_requests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            actor = FileActor()
            actor.initialize({"base_path": tmpdir})

            # Create some pending request files
            requests_dir = os.path.join(tmpdir, "requests")
            for i in range(3):
                with open(os.path.join(requests_dir, f"pending_{i}.json"), "w") as f:
                    json.dump({}, f)

            assert len(os.listdir(requests_dir)) == 3

            actor.shutdown()

            assert len(os.listdir(requests_dir)) == 0


class TestCreateActor:
    """Tests for the create_actor factory function."""

    def test_create_console_actor(self):
        actor = create_actor("console")
        assert isinstance(actor, ConsoleActor)
        assert actor.name == "console"

    def test_create_file_actor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            actor = create_actor("file", {"base_path": tmpdir})
            assert isinstance(actor, FileActor)
            assert actor.name == "file"

    def test_create_webhook_actor(self):
        actor = create_actor("webhook", {"endpoint": "http://example.com"})
        assert isinstance(actor, WebhookActor)
        assert actor.name == "webhook"

    def test_create_unknown_actor(self):
        with pytest.raises(ValueError, match="Unknown actor type"):
            create_actor("unknown_type")

    def test_create_actor_with_invalid_config(self):
        with pytest.raises(ValueError):
            create_actor("webhook", {})  # Missing required endpoint
