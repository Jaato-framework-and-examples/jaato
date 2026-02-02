"""Tests for reliability plugin core functionality."""

import pytest
from datetime import datetime, timedelta

from ..types import (
    EscalationRule,
    FailureKey,
    FailureRecord,
    FailureSeverity,
    ReliabilityConfig,
    ToolReliabilityState,
    TrustState,
    classify_failure,
)
from ..plugin import ReliabilityPlugin


class TestFailureKey:
    """Tests for FailureKey generation and parsing."""

    def test_from_invocation_file_read(self):
        """Test failure key for file read operations."""
        key = FailureKey.from_invocation("readFile", {"path": "/etc/passwd"})
        assert key.tool_name == "readFile"
        assert "path_prefix=/etc" in key.parameter_signature

    def test_from_invocation_http_request(self):
        """Test failure key for HTTP requests."""
        key = FailureKey.from_invocation("http_request", {"url": "https://api.example.com/v1/users"})
        assert key.tool_name == "http_request"
        assert "domain=api.example.com" in key.parameter_signature
        assert "path_prefix=v1" in key.parameter_signature

    def test_from_invocation_bash_command(self):
        """Test failure key for bash commands."""
        key = FailureKey.from_invocation("bash", {"command": "git status"})
        assert key.tool_name == "bash"
        assert "command=git" in key.parameter_signature

    def test_from_invocation_no_special_params(self):
        """Test failure key for unknown tools."""
        key = FailureKey.from_invocation("unknown_tool", {"foo": "bar"})
        assert key.tool_name == "unknown_tool"
        assert key.parameter_signature == ""  # No special extraction

    def test_to_string_and_from_string(self):
        """Test roundtrip serialization."""
        original = FailureKey(tool_name="readFile", parameter_signature="path_prefix=/etc")
        key_str = original.to_string()
        restored = FailureKey.from_string(key_str)

        assert restored.tool_name == original.tool_name
        assert restored.parameter_signature == original.parameter_signature

    def test_to_string_no_signature(self):
        """Test to_string with empty signature."""
        key = FailureKey(tool_name="simple_tool", parameter_signature="")
        assert key.to_string() == "simple_tool"


class TestClassifyFailure:
    """Tests for failure classification."""

    def test_http_404(self):
        """Test classification of HTTP 404."""
        severity, weight = classify_failure("http", {"http_status": 404})
        assert severity == FailureSeverity.NOT_FOUND
        assert weight == 1.0

    def test_http_429(self):
        """Test classification of HTTP 429 (rate limit)."""
        severity, weight = classify_failure("http", {"http_status": 429})
        assert severity == FailureSeverity.TRANSIENT
        assert weight == 0.5  # Lower weight for transient

    def test_http_500(self):
        """Test classification of HTTP 500."""
        severity, weight = classify_failure("http", {"http_status": 500})
        assert severity == FailureSeverity.SERVER_ERROR
        assert weight == 1.0

    def test_permission_denied_message(self):
        """Test classification from error message."""
        severity, weight = classify_failure("bash", {"error": "Permission denied"})
        assert severity == FailureSeverity.PERMISSION

    def test_invalid_input_message(self):
        """Test classification of model errors (lower weight)."""
        severity, weight = classify_failure("tool", {"error": "Invalid parameter: foo is required"})
        assert severity == FailureSeverity.INVALID_INPUT
        assert weight == 0.5  # Lower weight for model errors

    def test_timeout_message(self):
        """Test classification of timeout."""
        severity, weight = classify_failure("http", {"error": "Request timed out"})
        assert severity == FailureSeverity.TIMEOUT

    def test_default_classification(self):
        """Test default classification for unknown errors."""
        severity, weight = classify_failure("tool", {"error": "Something went wrong"})
        assert severity == FailureSeverity.SERVER_ERROR
        assert weight == 1.0


class TestToolReliabilityState:
    """Tests for ToolReliabilityState serialization."""

    def test_to_dict_and_from_dict(self):
        """Test roundtrip serialization."""
        original = ToolReliabilityState(
            failure_key="readFile|path_prefix=/etc",
            tool_name="readFile",
            state=TrustState.ESCALATED,
            failures_in_window=3.5,
            consecutive_failures=2,
            total_failures=5,
            total_successes=10,
            last_failure=datetime.now(),
            escalated_at=datetime.now(),
            escalation_reason="3 failures in 3600s",
        )

        data = original.to_dict()
        restored = ToolReliabilityState.from_dict(data)

        assert restored.failure_key == original.failure_key
        assert restored.tool_name == original.tool_name
        assert restored.state == original.state
        assert restored.failures_in_window == original.failures_in_window
        assert restored.escalation_reason == original.escalation_reason


class TestReliabilityPlugin:
    """Tests for ReliabilityPlugin core functionality."""

    @pytest.fixture
    def plugin(self):
        """Create a plugin instance for testing."""
        config = ReliabilityConfig(
            default_rule=EscalationRule(
                count_threshold=3,
                window_seconds=3600,
                escalation_duration_seconds=1800,
                success_count_to_recover=2,
            )
        )
        return ReliabilityPlugin(config)

    def test_initial_state(self, plugin):
        """Test plugin starts with no tracked tools."""
        assert len(plugin.get_all_states()) == 0
        assert len(plugin.get_escalated_tools()) == 0

    def test_success_creates_state(self, plugin):
        """Test that successful execution creates tracking state."""
        plugin.on_tool_result(
            tool_name="readFile",
            args={"path": "/tmp/test.txt"},
            success=True,
            result={"content": "hello"},
            call_id="123",
        )

        states = plugin.get_all_states()
        assert len(states) == 1

        key = "readFile|path_prefix=/tmp"
        assert key in states
        assert states[key].state == TrustState.TRUSTED
        assert states[key].total_successes == 1

    def test_failure_increments_counters(self, plugin):
        """Test that failures increment counters."""
        plugin.on_tool_result(
            tool_name="readFile",
            args={"path": "/etc/secret"},
            success=False,
            result={"error": "Permission denied"},
            call_id="123",
        )

        key = "readFile|path_prefix=/etc"
        state = plugin.get_state(key)
        assert state is not None
        assert state.total_failures == 1
        assert state.consecutive_failures == 1

    def test_escalation_after_threshold(self, plugin):
        """Test that tool is escalated after reaching failure threshold."""
        # Generate 3 failures (threshold)
        for i in range(3):
            plugin.on_tool_result(
                tool_name="http_request",
                args={"url": "https://api.bad.com/endpoint"},
                success=False,
                result={"error": "Service unavailable", "http_status": 503},
                call_id=f"call_{i}",
            )

        key = "http_request|domain=api.bad.com|path_prefix=endpoint"
        state = plugin.get_state(key)
        assert state is not None
        assert state.state == TrustState.ESCALATED
        assert state.escalation_reason is not None

    def test_success_resets_consecutive(self, plugin):
        """Test that success resets consecutive failure count."""
        # One failure
        plugin.on_tool_result(
            tool_name="bash",
            args={"command": "ls /tmp"},
            success=False,
            result={"error": "ls: error"},
            call_id="1",
        )

        key = "bash|command=ls"
        state = plugin.get_state(key)
        assert state.consecutive_failures == 1

        # One success
        plugin.on_tool_result(
            tool_name="bash",
            args={"command": "ls /home"},
            success=True,
            result={"output": "files"},
            call_id="2",
        )

        assert state.consecutive_failures == 0

    def test_is_escalated_check(self, plugin):
        """Test is_escalated method."""
        # Initially not escalated
        is_esc, reason = plugin.is_escalated("readFile", {"path": "/etc/test"})
        assert is_esc is False
        assert reason is None

        # Escalate through failures
        for i in range(3):
            plugin.on_tool_result(
                tool_name="readFile",
                args={"path": "/etc/test"},
                success=False,
                result={"error": "Permission denied"},
                call_id=f"call_{i}",
            )

        is_esc, reason = plugin.is_escalated("readFile", {"path": "/etc/test"})
        assert is_esc is True
        assert reason is not None

    def test_manual_reset(self, plugin):
        """Test manual reset of escalated tool."""
        # Escalate
        for i in range(3):
            plugin.on_tool_result(
                tool_name="bash",
                args={"command": "rm -rf /"},
                success=False,
                result={"error": "Permission denied"},
                call_id=f"call_{i}",
            )

        key = "bash|command=rm"
        state = plugin.get_state(key)
        assert state.state == TrustState.ESCALATED

        # Manual reset
        success = plugin.reset_tool(key)
        assert success is True
        assert state.state == TrustState.TRUSTED
        assert state.consecutive_failures == 0

    def test_different_params_tracked_separately(self, plugin):
        """Test that different parameters create separate tracking."""
        # Failures to /etc
        for i in range(3):
            plugin.on_tool_result(
                tool_name="readFile",
                args={"path": "/etc/passwd"},
                success=False,
                result={"error": "Permission denied"},
                call_id=f"etc_{i}",
            )

        # Success to /tmp
        plugin.on_tool_result(
            tool_name="readFile",
            args={"path": "/tmp/test.txt"},
            success=True,
            result={"content": "ok"},
            call_id="tmp_1",
        )

        # Check states
        etc_state = plugin.get_state("readFile|path_prefix=/etc")
        tmp_state = plugin.get_state("readFile|path_prefix=/tmp")

        assert etc_state.state == TrustState.ESCALATED
        assert tmp_state.state == TrustState.TRUSTED

    def test_security_severity_blocks_immediately(self, plugin):
        """Test that security-classified failures block immediately."""
        plugin.on_tool_result(
            tool_name="bash",
            args={"command": "eval dangerous"},
            success=False,
            result={"error": "Security violation: malicious command detected"},
            call_id="1",
        )

        key = "bash|command=eval"
        state = plugin.get_state(key)
        assert state.state == TrustState.BLOCKED

    def test_recovery_after_successes(self, plugin):
        """Test recovery flow after escalation."""
        # Escalate
        for i in range(3):
            plugin.on_tool_result(
                tool_name="http_request",
                args={"url": "https://api.test.com/data"},
                success=False,
                result={"error": "Server error", "http_status": 500},
                call_id=f"fail_{i}",
            )

        key = "http_request|domain=api.test.com|path_prefix=data"
        state = plugin.get_state(key)
        assert state.state == TrustState.ESCALATED

        # Simulate time passing (expire escalation)
        state.escalation_expires = datetime.now() - timedelta(seconds=1)

        # Success should start recovery
        plugin.on_tool_result(
            tool_name="http_request",
            args={"url": "https://api.test.com/data"},
            success=True,
            result={"data": "ok"},
            call_id="success_1",
        )

        assert state.state == TrustState.RECOVERING
        assert state.successes_since_recovery == 1

        # Another success should complete recovery (threshold=2)
        plugin.on_tool_result(
            tool_name="http_request",
            args={"url": "https://api.test.com/data"},
            success=True,
            result={"data": "ok"},
            call_id="success_2",
        )

        assert state.state == TrustState.TRUSTED


class TestReliabilityPluginCommands:
    """Tests for user command handling."""

    @pytest.fixture
    def plugin(self):
        """Create a plugin with some state for testing."""
        plugin = ReliabilityPlugin()

        # Add some failures
        for i in range(3):
            plugin.on_tool_result(
                tool_name="http_request",
                args={"url": "https://api.bad.com/test"},
                success=False,
                result={"error": "Service unavailable"},
                call_id=f"call_{i}",
            )

        return plugin

    def test_status_command(self, plugin):
        """Test status command output."""
        output = plugin.handle_command("status", "")
        assert "http_request" in output
        assert "ESCALATED" in output.upper() or "escalated" in output

    def test_reset_command(self, plugin):
        """Test reset command."""
        key = "http_request|domain=api.bad.com|path_prefix=test"
        output = plugin.handle_command("reset", key)
        assert "TRUSTED" in output

        state = plugin.get_state(key)
        assert state.state == TrustState.TRUSTED

    def test_history_command(self, plugin):
        """Test history command."""
        output = plugin.handle_command("history", "")
        assert "http_request" in output

    def test_config_command(self, plugin):
        """Test config command."""
        output = plugin.handle_command("config", "")
        assert "Count threshold" in output
        assert "Window" in output

    def test_recovery_command(self, plugin):
        """Test recovery mode command."""
        output = plugin.handle_command("recovery", "auto")
        assert "AUTO" in output

        output = plugin.handle_command("recovery", "ask")
        assert "ASK" in output
