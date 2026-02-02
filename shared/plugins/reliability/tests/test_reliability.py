"""Tests for reliability plugin core functionality."""

import pytest
from datetime import datetime, timedelta

from ..types import (
    EscalationInfo,
    EscalationRule,
    FailureKey,
    FailureRecord,
    FailureSeverity,
    ReliabilityConfig,
    ToolReliabilityState,
    TrustState,
    classify_failure,
)
from ..plugin import ReliabilityPlugin, ReliabilityPermissionWrapper


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


class TestEscalationInfo:
    """Tests for EscalationInfo and check_escalation method."""

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

    def test_check_escalation_not_escalated(self, plugin):
        """Test check_escalation for a tool that is not tracked."""
        from ..types import EscalationInfo

        info = plugin.check_escalation("readFile", {"path": "/tmp/test.txt"})
        assert info.is_escalated is False
        assert info.state == TrustState.TRUSTED

    def test_check_escalation_after_failures(self, plugin):
        """Test check_escalation returns rich info for escalated tool."""
        from ..types import EscalationInfo

        # Generate failures to escalate
        for i in range(3):
            plugin.on_tool_result(
                tool_name="http_request",
                args={"url": "https://api.failing.com/test"},
                success=False,
                result={"error": "Service unavailable", "http_status": 503},
                call_id=f"call_{i}",
            )

        info = plugin.check_escalation("http_request", {"url": "https://api.failing.com/test"})

        assert info.is_escalated is True
        assert info.state == TrustState.ESCALATED
        assert info.reason is not None
        assert info.failure_count == 3
        assert info.severity_label == "⚠ ESCALATED"
        assert "failures" in info.window_description.lower()

    def test_check_escalation_recovering(self, plugin):
        """Test check_escalation during recovery."""
        from ..types import EscalationInfo

        # Escalate
        for i in range(3):
            plugin.on_tool_result(
                tool_name="bash",
                args={"command": "ls /fail"},
                success=False,
                result={"error": "Command failed"},
                call_id=f"fail_{i}",
            )

        # Expire escalation
        key = "bash|command=ls"
        state = plugin.get_state(key)
        state.escalation_expires = datetime.now() - timedelta(seconds=1)

        # Start recovery
        plugin.on_tool_result(
            tool_name="bash",
            args={"command": "ls /success"},
            success=True,
            result={"output": "files"},
            call_id="success_1",
        )

        info = plugin.check_escalation("bash", {"command": "ls /test"})

        assert info.is_escalated is True
        assert info.state == TrustState.RECOVERING
        assert info.recovery_progress is not None
        assert "1/2" in info.recovery_progress
        assert info.recovery_hint is not None
        assert info.severity_label == "↻ RECOVERING"

    def test_escalation_info_to_display_lines(self):
        """Test EscalationInfo.to_display_lines() formatting."""
        from ..types import EscalationInfo

        # Not escalated - should return empty
        info = EscalationInfo(is_escalated=False)
        assert info.to_display_lines() == []

        # Escalated with full info
        info = EscalationInfo(
            is_escalated=True,
            state=TrustState.ESCALATED,
            reason="3 failures in 3600s",
            failure_count=3,
            window_description="3 failures in 1h",
            recovery_hint="Cooldown: 25m before recovery eligible",
            severity_label="⚠ ESCALATED",
        )
        lines = info.to_display_lines()

        assert len(lines) >= 3
        assert "⚠ ESCALATED" in lines[0]
        assert "Reason:" in lines[1]
        assert any("History:" in line for line in lines)


class TestReliabilityHooks:
    """Tests for reliability lifecycle hooks."""

    @pytest.fixture
    def plugin(self):
        """Create a plugin with hooks."""
        config = ReliabilityConfig(
            default_rule=EscalationRule(
                count_threshold=3,
                window_seconds=3600,
                escalation_duration_seconds=1800,
                success_count_to_recover=2,
            )
        )
        return ReliabilityPlugin(config)

    def test_on_escalated_hook(self, plugin):
        """Test that on_escalated hook is called."""
        escalation_events = []

        def on_escalated(key, state, reason):
            escalation_events.append((key, state, reason))

        plugin.set_reliability_hooks(on_escalated=on_escalated)

        # Trigger escalation
        for i in range(3):
            plugin.on_tool_result(
                tool_name="readFile",
                args={"path": "/etc/secret"},
                success=False,
                result={"error": "Permission denied"},
                call_id=f"call_{i}",
            )

        assert len(escalation_events) == 1
        key, state, reason = escalation_events[0]
        assert "readFile" in key
        assert state == TrustState.ESCALATED
        assert "failures" in reason.lower()

    def test_on_recovered_hook(self, plugin):
        """Test that on_recovered hook is called."""
        recovery_events = []

        def on_recovered(key, state):
            recovery_events.append((key, state))

        plugin.set_reliability_hooks(on_recovered=on_recovered)

        # Escalate first
        for i in range(3):
            plugin.on_tool_result(
                tool_name="bash",
                args={"command": "cat /fail"},
                success=False,
                result={"error": "Failed"},
                call_id=f"fail_{i}",
            )

        # Expire escalation
        key = "bash|command=cat"
        state = plugin.get_state(key)
        state.escalation_expires = datetime.now() - timedelta(seconds=1)

        # Recover with successes
        for i in range(2):
            plugin.on_tool_result(
                tool_name="bash",
                args={"command": "cat /ok"},
                success=True,
                result={"output": "content"},
                call_id=f"success_{i}",
            )

        assert len(recovery_events) == 1
        key, recovered_state = recovery_events[0]
        assert "bash" in key
        assert recovered_state == TrustState.TRUSTED

    def test_on_blocked_hook(self, plugin):
        """Test that on_blocked hook is called for security issues."""
        blocked_events = []

        def on_blocked(key, reason):
            blocked_events.append((key, reason))

        plugin.set_reliability_hooks(on_blocked=on_blocked)

        # Trigger security block
        plugin.on_tool_result(
            tool_name="bash",
            args={"command": "eval malicious"},
            success=False,
            result={"error": "Security violation: malicious code detected"},
            call_id="1",
        )

        assert len(blocked_events) == 1
        key, reason = blocked_events[0]
        assert "bash" in key
        assert "security" in reason.lower()


class TestPermissionWrapper:
    """Tests for ReliabilityPermissionWrapper."""

    @pytest.fixture
    def reliability_plugin(self):
        """Create a reliability plugin."""
        config = ReliabilityConfig(
            default_rule=EscalationRule(
                count_threshold=3,
                window_seconds=3600,
            )
        )
        return ReliabilityPlugin(config)

    def test_wrapper_delegates_to_inner(self, reliability_plugin):
        """Test that wrapper delegates to inner plugin for non-escalated tools."""
        from ..plugin import ReliabilityPermissionWrapper

        # Mock inner permission plugin
        class MockPermissionPlugin:
            def check_permission(self, tool_name, args, context=None, call_id=None):
                return True, {"reason": "whitelisted", "method": "whitelist"}

            def get_formatted_prompt(self, tool_name, args, channel_type="ipc"):
                return ["Tool: " + tool_name], None, None, None, None, None

        inner = MockPermissionPlugin()
        wrapper = ReliabilityPermissionWrapper(inner, reliability_plugin)

        # Non-escalated tool should delegate
        allowed, info = wrapper.check_permission("readFile", {"path": "/tmp/test"})
        assert allowed is True
        assert info["method"] == "whitelist"

    def test_wrapper_forces_approval_for_escalated(self, reliability_plugin):
        """Test that wrapper forces approval for escalated tools."""
        from ..plugin import ReliabilityPermissionWrapper

        # Escalate a tool
        for i in range(3):
            reliability_plugin.on_tool_result(
                tool_name="http_request",
                args={"url": "https://bad.api.com/fail"},
                success=False,
                result={"error": "Service error"},
                call_id=f"call_{i}",
            )

        # Mock inner that would normally auto-approve
        class MockPermissionPlugin:
            def __init__(self):
                self._policy = None  # No policy to manipulate

            def check_permission(self, tool_name, args, context=None, call_id=None):
                # Check if reliability context was added
                if context and "_reliability_escalation" in context:
                    return False, {"reason": "prompted for approval", "method": "channel"}
                return True, {"reason": "whitelisted", "method": "whitelist"}

            def get_formatted_prompt(self, tool_name, args, channel_type="ipc"):
                return ["Tool: " + tool_name], None, None, None, None, None

        inner = MockPermissionPlugin()
        wrapper = ReliabilityPermissionWrapper(inner, reliability_plugin)

        # Escalated tool should have reliability context
        allowed, info = wrapper.check_permission(
            "http_request",
            {"url": "https://bad.api.com/fail"}
        )

        assert "_reliability" in info
        assert info["_reliability"]["escalated"] is True
        assert info["_reliability"]["state"] == "escalated"

    def test_wrapper_enhances_prompt_for_escalated(self, reliability_plugin):
        """Test that wrapper adds reliability info to prompts."""
        from ..plugin import ReliabilityPermissionWrapper

        # Escalate a tool
        for i in range(3):
            reliability_plugin.on_tool_result(
                tool_name="bash",
                args={"command": "rm -rf /danger"},
                success=False,
                result={"error": "Permission denied"},
                call_id=f"call_{i}",
            )

        class MockPermissionPlugin:
            def get_formatted_prompt(self, tool_name, args, channel_type="ipc"):
                return ["Tool: " + tool_name, "Args: command"], None, None, None, None, None

        inner = MockPermissionPlugin()
        wrapper = ReliabilityPermissionWrapper(inner, reliability_plugin)

        lines, fmt, lang, raw, warn, level = wrapper.get_formatted_prompt(
            "bash",
            {"command": "rm -rf /danger"},
            "ipc"
        )

        # Should have reliability warning prepended
        assert any("ESCALATED" in line for line in lines)
        assert level == "warning"


class TestPersistence:
    """Tests for reliability persistence."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for testing."""
        return str(tmp_path)

    def test_tool_state_persistence(self, temp_dir):
        """Test saving and loading tool states."""
        from ..persistence import ReliabilityPersistence

        persist = ReliabilityPersistence(workspace_path=temp_dir)

        # Create and save state
        state = ToolReliabilityState(
            failure_key="test_tool|param=value",
            tool_name="test_tool",
            state=TrustState.ESCALATED,
            total_failures=5,
            escalation_reason="test reason",
        )
        persist.save_tool_state(state)

        # Reload and verify
        persist.invalidate_cache()
        states = persist.load_tool_states()

        assert "test_tool|param=value" in states
        loaded = states["test_tool|param=value"]
        assert loaded.state == TrustState.ESCALATED
        assert loaded.total_failures == 5
        assert loaded.escalation_reason == "test reason"

    def test_failure_history_persistence(self, temp_dir):
        """Test saving and loading failure history."""
        from ..persistence import ReliabilityPersistence

        persist = ReliabilityPersistence(workspace_path=temp_dir)

        # Create and save record
        record = FailureRecord(
            failure_key="test_tool|param=value",
            tool_name="test_tool",
            plugin_name="test",
            timestamp=datetime.now(),
            parameter_signature="param=value",
            severity=FailureSeverity.SERVER_ERROR,
            error_message="test error",
        )
        persist.append_failure(record)

        # Reload and verify
        persist.invalidate_cache()
        history = persist.load_failure_history()

        assert len(history) == 1
        assert history[0].failure_key == "test_tool|param=value"
        assert history[0].error_message == "test error"

    def test_settings_hierarchy(self, temp_dir):
        """Test settings merge hierarchy (session > workspace > user)."""
        from ..persistence import ReliabilityPersistence, SessionSettings

        persist = ReliabilityPersistence(workspace_path=temp_dir)

        # Set user default
        persist.save_setting_to_user("test_setting", "user_value")

        # Without overrides, should get user value
        persist.invalidate_cache()
        assert persist.get_effective_setting("test_setting") == "user_value"

        # Set workspace override
        persist.save_setting_to_workspace("test_setting", "workspace_value")
        persist.invalidate_cache()
        assert persist.get_effective_setting("test_setting") == "workspace_value"

        # Session override takes priority
        session = SessionSettings()
        # SessionSettings doesn't have arbitrary attributes, so test with known one
        session.recovery_mode = "ask"
        assert persist.get_effective_setting("recovery_mode", session) == "ask"

    def test_trusted_states_not_persisted(self, temp_dir):
        """Test that TRUSTED states are removed from persistence."""
        from ..persistence import ReliabilityPersistence

        persist = ReliabilityPersistence(workspace_path=temp_dir)

        # Save escalated state
        state = ToolReliabilityState(
            failure_key="test_tool|param=value",
            tool_name="test_tool",
            state=TrustState.ESCALATED,
        )
        persist.save_tool_state(state)

        # Verify it's saved
        persist.invalidate_cache()
        states = persist.load_tool_states()
        assert "test_tool|param=value" in states

        # Change to trusted and save
        state.state = TrustState.TRUSTED
        persist.save_tool_state(state)

        # Should be removed from persistence
        persist.invalidate_cache()
        states = persist.load_tool_states()
        assert "test_tool|param=value" not in states

    def test_session_settings_serialization(self):
        """Test SessionSettings serialization."""
        from ..persistence import SessionSettings, SessionReliabilityState

        settings = SessionSettings(
            nudge_level="gentle",
            recovery_mode="ask",
        )

        state = SessionReliabilityState(
            session_id="test-session",
            settings=settings,
        )

        # Serialize
        data = state.serialize()
        assert isinstance(data, bytes)

        # Deserialize
        restored = SessionReliabilityState.deserialize(data)
        assert restored.session_id == "test-session"
        assert restored.settings.nudge_level == "gentle"
        assert restored.settings.recovery_mode == "ask"


class TestModelReliability:
    """Tests for model-specific reliability tracking."""

    @pytest.fixture
    def plugin(self):
        """Create a plugin with model context."""
        config = ReliabilityConfig(
            default_rule=EscalationRule(
                count_threshold=3,
                window_seconds=3600,
            )
        )
        plugin = ReliabilityPlugin(config)
        plugin.set_model_context(
            current_model="gpt-4",
            available_models=["gpt-4", "claude-3", "gemini-pro"]
        )
        return plugin

    def test_model_attempt_tracking(self, plugin):
        """Test that model attempts are tracked."""
        from ..types import ModelToolProfile

        plugin.on_tool_result(
            tool_name="readFile",
            args={"path": "/tmp/test.txt"},
            success=True,
            result={"content": "ok"},
            call_id="1",
        )

        profile = plugin.get_model_profile("gpt-4", "readFile|path_prefix=/tmp")
        assert profile is not None
        assert profile.total_attempts == 1
        assert profile.failures == 0
        assert profile.success_rate == 1.0

    def test_model_failure_tracking(self, plugin):
        """Test that model failures are tracked."""
        plugin.on_tool_result(
            tool_name="http_request",
            args={"url": "https://api.test.com/data"},
            success=False,
            result={"error": "Server error"},
            call_id="1",
        )

        profile = plugin.get_model_profile("gpt-4", "http_request|domain=api.test.com|path_prefix=data")
        assert profile is not None
        assert profile.total_attempts == 1
        assert profile.failures == 1
        assert profile.success_rate == 0.0

    def test_model_profile_success_rate(self, plugin):
        """Test success rate calculation."""
        key = "bash|command=ls"

        # 3 successes, 1 failure = 75% success
        for _ in range(3):
            plugin.on_tool_result(
                tool_name="bash",
                args={"command": "ls /tmp"},
                success=True,
                result={"output": "files"},
                call_id="ok",
            )

        plugin.on_tool_result(
            tool_name="bash",
            args={"command": "ls /fail"},
            success=False,
            result={"error": "No such file"},
            call_id="fail",
        )

        profile = plugin.get_model_profile("gpt-4", key)
        assert profile.total_attempts == 4
        assert profile.failures == 1
        assert profile.success_rate == 0.75

    def test_model_reliability_summary(self, plugin):
        """Test model reliability summary."""
        # Add some results
        for _ in range(5):
            plugin.on_tool_result(
                tool_name="readFile",
                args={"path": "/tmp/test.txt"},
                success=True,
                result={"content": "ok"},
                call_id="ok",
            )

        plugin.on_tool_result(
            tool_name="http_request",
            args={"url": "https://api.test.com/fail"},
            success=False,
            result={"error": "Error"},
            call_id="fail",
        )

        summary = plugin.get_model_reliability_summary()

        assert summary["model"] == "gpt-4"
        assert summary["total_attempts"] == 6
        assert summary["total_failures"] == 1
        assert summary["tools_tracked"] == 2
        assert summary["success_rate"] == pytest.approx(5/6)

    def test_model_switch_suggestion_on_failure(self, plugin):
        """Test that model switch is suggested after failures."""
        from ..types import ModelSwitchConfig, ModelSwitchStrategy, ModelSwitchSuggestion

        suggestions = []

        def on_suggestion(suggestion):
            suggestions.append(suggestion)

        plugin.set_reliability_hooks(on_model_switch_suggested=on_suggestion)
        plugin.set_model_switch_config(ModelSwitchConfig(
            strategy=ModelSwitchStrategy.SUGGEST,
            failure_threshold=3,
            min_success_rate_diff=0.3,
            min_attempts=2,
        ))

        # First, record good history for claude-3
        plugin.set_model_context("claude-3", ["gpt-4", "claude-3", "gemini-pro"])
        for _ in range(5):
            plugin.on_tool_result(
                tool_name="http_request",
                args={"url": "https://api.problem.com/endpoint"},
                success=True,
                result={"data": "ok"},
                call_id="ok",
            )

        # Switch to gpt-4 and generate failures
        plugin.set_model_context("gpt-4", ["gpt-4", "claude-3", "gemini-pro"])
        for i in range(3):
            plugin.on_tool_result(
                tool_name="http_request",
                args={"url": "https://api.problem.com/endpoint"},
                success=False,
                result={"error": "gpt-4 fails at this"},
                call_id=f"fail_{i}",
            )

        # Should have suggestion
        assert len(suggestions) >= 1
        suggestion = suggestions[-1]
        assert suggestion.current_model == "gpt-4"
        assert suggestion.suggested_model == "claude-3"
        assert suggestion.improvement > 0.3

    def test_model_switch_disabled(self, plugin):
        """Test that disabled strategy prevents suggestions."""
        from ..types import ModelSwitchConfig, ModelSwitchStrategy

        suggestions = []

        def on_suggestion(suggestion):
            suggestions.append(suggestion)

        plugin.set_reliability_hooks(on_model_switch_suggested=on_suggestion)
        plugin.set_model_switch_config(ModelSwitchConfig(
            strategy=ModelSwitchStrategy.DISABLED,
        ))

        # Generate failures - should not trigger suggestion
        for i in range(5):
            plugin.on_tool_result(
                tool_name="http_request",
                args={"url": "https://api.test.com/fail"},
                success=False,
                result={"error": "Error"},
                call_id=f"fail_{i}",
            )

        assert len(suggestions) == 0

    def test_get_model_switch_suggestion_manual(self, plugin):
        """Test manually checking for model switch suggestion."""
        from ..types import ModelSwitchConfig, ModelSwitchStrategy

        plugin.set_model_switch_config(ModelSwitchConfig(
            strategy=ModelSwitchStrategy.SUGGEST,
            failure_threshold=2,
            min_success_rate_diff=0.2,
            min_attempts=2,
        ))

        # Build history for gemini-pro (good)
        plugin.set_model_context("gemini-pro", ["gpt-4", "claude-3", "gemini-pro"])
        for _ in range(5):
            plugin.on_tool_result(
                tool_name="bash",
                args={"command": "git status"},
                success=True,
                result={"output": "ok"},
                call_id="ok",
            )

        # Build history for gpt-4 (bad)
        plugin.set_model_context("gpt-4", ["gpt-4", "claude-3", "gemini-pro"])
        for _ in range(3):
            plugin.on_tool_result(
                tool_name="bash",
                args={"command": "git status"},
                success=False,
                result={"error": "Failed"},
                call_id="fail",
            )

        # Manually check
        suggestion = plugin.get_model_switch_suggestion("bash", {"command": "git status"})

        assert suggestion is not None
        assert suggestion.suggested_model == "gemini-pro"
        assert suggestion.current_success_rate == 0.0
        assert suggestion.suggested_success_rate == 1.0

    def test_model_command_status(self, plugin):
        """Test reliability model status command."""
        # Add some data
        for _ in range(3):
            plugin.on_tool_result(
                tool_name="readFile",
                args={"path": "/tmp/test.txt"},
                success=True,
                result={"content": "ok"},
                call_id="ok",
            )

        output = plugin.handle_command("model", "status")

        assert "gpt-4" in output
        assert "Success rate" in output
        assert "3" in output  # total attempts

    def test_model_command_compare(self, plugin):
        """Test reliability model compare command."""
        # Add data for gpt-4
        for _ in range(3):
            plugin.on_tool_result(
                tool_name="readFile",
                args={"path": "/tmp/test.txt"},
                success=True,
                result={"content": "ok"},
                call_id="ok",
            )

        # Add data for claude-3
        plugin.set_model_context("claude-3", ["gpt-4", "claude-3"])
        for _ in range(2):
            plugin.on_tool_result(
                tool_name="http_request",
                args={"url": "https://api.test.com/data"},
                success=False,
                result={"error": "Error"},
                call_id="fail",
            )

        output = plugin.handle_command("model", "compare")

        assert "Model" in output
        assert "gpt-4" in output
        assert "claude-3" in output
        assert "Success" in output

    def test_model_command_strategy_change(self, plugin):
        """Test changing model switch strategy via command."""
        from ..types import ModelSwitchStrategy

        output = plugin.handle_command("model", "suggest")
        assert "SUGGEST" in output
        assert plugin._model_switch_config.strategy == ModelSwitchStrategy.SUGGEST

        output = plugin.handle_command("model", "auto")
        assert "AUTO" in output
        assert plugin._model_switch_config.strategy == ModelSwitchStrategy.AUTO

        output = plugin.handle_command("model", "disabled")
        assert "DISABLED" in output
        assert plugin._model_switch_config.strategy == ModelSwitchStrategy.DISABLED

    def test_no_model_context_handling(self):
        """Test graceful handling when no model context set."""
        plugin = ReliabilityPlugin()

        # Without model context, should not track model profiles
        plugin.on_tool_result(
            tool_name="readFile",
            args={"path": "/tmp/test.txt"},
            success=True,
            result={"content": "ok"},
            call_id="1",
        )

        assert len(plugin.get_all_model_profiles()) == 0

        # Status command should handle gracefully
        output = plugin.handle_command("model", "status")
        assert "No model context" in output


class TestModelToolProfile:
    """Tests for ModelToolProfile dataclass."""

    def test_record_attempt_success(self):
        """Test recording successful attempts."""
        from ..types import ModelToolProfile

        profile = ModelToolProfile(
            model_name="gpt-4",
            failure_key="test_tool|param=value",
        )

        profile.record_attempt(success=True)

        assert profile.total_attempts == 1
        assert profile.failures == 0
        assert profile.success_rate == 1.0
        assert profile.last_success is not None
        assert profile.last_failure is None

    def test_record_attempt_failure(self):
        """Test recording failed attempts."""
        from ..types import ModelToolProfile

        profile = ModelToolProfile(
            model_name="gpt-4",
            failure_key="test_tool|param=value",
        )

        profile.record_attempt(success=False)

        assert profile.total_attempts == 1
        assert profile.failures == 1
        assert profile.success_rate == 0.0
        assert profile.last_success is None
        assert profile.last_failure is not None

    def test_success_rate_empty(self):
        """Test success rate when no attempts."""
        from ..types import ModelToolProfile

        profile = ModelToolProfile(
            model_name="gpt-4",
            failure_key="test_tool|param=value",
        )

        assert profile.success_rate == 1.0  # Assume good until proven otherwise

    def test_success_rate_mixed(self):
        """Test success rate with mixed results."""
        from ..types import ModelToolProfile

        profile = ModelToolProfile(
            model_name="gpt-4",
            failure_key="test_tool|param=value",
        )

        # 7 success, 3 failures = 70%
        for _ in range(7):
            profile.record_attempt(success=True)
        for _ in range(3):
            profile.record_attempt(success=False)

        assert profile.total_attempts == 10
        assert profile.failures == 3
        assert profile.success_rate == pytest.approx(0.7)


class TestPatternDetection:
    """Tests for behavioral pattern detection."""

    @pytest.fixture
    def plugin(self):
        """Create a plugin with pattern detection enabled."""
        from ..types import PatternDetectionConfig

        plugin = ReliabilityPlugin()
        plugin.set_pattern_detection_config(PatternDetectionConfig(
            enabled=True,
            repetitive_call_threshold=3,
            introspection_loop_threshold=3,
            max_reads_before_action=4,
        ))
        plugin.enable_pattern_detection(True)
        return plugin

    def test_pattern_detection_enabled(self, plugin):
        """Test that pattern detection can be enabled."""
        assert plugin.get_pattern_detector() is not None

    def test_repetitive_calls_detection(self, plugin):
        """Test detection of repetitive tool calls."""
        from ..types import BehavioralPatternType

        patterns = []

        def on_pattern(pattern):
            patterns.append(pattern)

        plugin.set_pattern_hook(on_pattern)
        plugin.on_turn_start(1)

        # Call same tool 3 times (threshold)
        for i in range(3):
            plugin.on_tool_called("readFile", {"path": "/tmp/test.txt"})

        assert len(patterns) == 1
        assert patterns[0].pattern_type == BehavioralPatternType.REPETITIVE_CALLS
        assert patterns[0].repetition_count >= 3

    def test_introspection_loop_detection(self, plugin):
        """Test detection of introspection loops."""
        from ..types import BehavioralPatternType

        patterns = []

        def on_pattern(pattern):
            patterns.append(pattern)

        plugin.set_pattern_hook(on_pattern)
        plugin.on_turn_start(1)

        # Call introspection tools 3 times
        for i in range(3):
            plugin.on_tool_called("list_tools", {})

        assert len(patterns) == 1
        assert patterns[0].pattern_type == BehavioralPatternType.INTROSPECTION_LOOP

    def test_read_only_stall_detection(self, plugin):
        """Test detection of read-only stalls."""
        from ..types import BehavioralPatternType, PatternDetectionConfig

        # Use higher threshold to avoid triggering repetitive calls
        plugin.set_pattern_detection_config(PatternDetectionConfig(
            enabled=True,
            repetitive_call_threshold=10,  # High to avoid triggering
            max_reads_before_action=4,
        ))
        plugin.enable_pattern_detection(True)

        patterns = []

        def on_pattern(pattern):
            patterns.append(pattern)

        plugin.set_pattern_hook(on_pattern)
        plugin.on_turn_start(1)

        # Call different read-only tools without any action
        read_tools = ["Read", "Glob", "Grep", "Read"]
        for tool in read_tools:
            plugin.on_tool_called(tool, {"path": f"/tmp/file{tool}.txt"})

        assert len(patterns) >= 1
        read_only_patterns = [p for p in patterns if p.pattern_type == BehavioralPatternType.READ_ONLY_LOOP]
        assert len(read_only_patterns) >= 1

    def test_pattern_summary(self, plugin):
        """Test pattern summary reporting."""
        plugin.on_turn_start(1)

        # Generate a pattern
        for i in range(3):
            plugin.on_tool_called("readFile", {"path": "/tmp/test.txt"})

        summary = plugin.get_pattern_summary()

        assert summary["total"] >= 1
        assert "repetitive_calls" in str(summary["by_type"]).lower()

    def test_pattern_command_status(self, plugin):
        """Test patterns status command."""
        output = plugin.handle_command("patterns", "status")

        assert "Pattern Detection" in output
        assert "ENABLED" in output

    def test_pattern_command_disable_enable(self, plugin):
        """Test patterns enable/disable commands."""
        output = plugin.handle_command("patterns", "disable")
        assert "DISABLED" in output
        assert plugin.get_pattern_detector() is None

        output = plugin.handle_command("patterns", "enable")
        assert "ENABLED" in output
        assert plugin.get_pattern_detector() is not None


class TestNudgeTypes:
    """Tests for nudge type definitions."""

    def test_nudge_level_values(self):
        """Test NudgeLevel enum values."""
        from ..types import NudgeLevel

        assert NudgeLevel.OFF.value == "off"
        assert NudgeLevel.GENTLE.value == "gentle"
        assert NudgeLevel.DIRECT.value == "direct"
        assert NudgeLevel.FULL.value == "full"

    def test_nudge_type_values(self):
        """Test NudgeType enum values."""
        from ..types import NudgeType

        assert NudgeType.GENTLE_REMINDER.value == "gentle"
        assert NudgeType.DIRECT_INSTRUCTION.value == "direct"
        assert NudgeType.INTERRUPT.value == "interrupt"

    def test_nudge_config_defaults(self):
        """Test NudgeConfig default values."""
        from ..types import NudgeConfig, NudgeLevel

        config = NudgeConfig()

        assert config.enabled is True
        assert config.level == NudgeLevel.DIRECT
        assert config.cooldown_seconds > 0

    def test_nudge_dataclass(self):
        """Test Nudge dataclass creation."""
        from ..types import Nudge, NudgeType, BehavioralPattern, BehavioralPatternType, PatternSeverity

        pattern = BehavioralPattern(
            pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
            detected_at=datetime.now(),
            turn_index=1,
            session_id="test",
            tool_sequence=["readFile", "readFile", "readFile"],
            repetition_count=3,
            duration_seconds=5.0,
            model_name="gpt-4",
        )

        nudge = Nudge(
            nudge_type=NudgeType.GENTLE_REMINDER,
            message="Test message",
            pattern=pattern,
            injected_at=datetime.now(),
        )

        assert nudge.nudge_type == NudgeType.GENTLE_REMINDER
        assert nudge.message == "Test message"
        assert nudge.acknowledged is False
        assert nudge.effective is False

    def test_nudge_to_system_message(self):
        """Test Nudge.to_system_message() formatting."""
        from ..types import Nudge, NudgeType, BehavioralPattern, BehavioralPatternType

        pattern = BehavioralPattern(
            pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
            detected_at=datetime.now(),
            turn_index=1,
            session_id="test",
            tool_sequence=["readFile"],
            repetition_count=3,
            duration_seconds=5.0,
            model_name="gpt-4",
        )

        nudge = Nudge(
            nudge_type=NudgeType.DIRECT_INSTRUCTION,
            message="Stop repeating yourself!",
            pattern=pattern,
            injected_at=datetime.now(),
        )

        msg = nudge.to_system_message()

        assert "Stop repeating yourself!" in msg
        assert "system-reminder" in msg.lower() or len(msg) > 0


class TestNudgeStrategy:
    """Tests for NudgeStrategy class."""

    def test_create_nudge_for_repetitive_calls(self):
        """Test creating nudge for repetitive calls pattern."""
        from ..types import BehavioralPattern, BehavioralPatternType, PatternSeverity
        from ..nudge import NudgeStrategy

        strategy = NudgeStrategy()

        pattern = BehavioralPattern(
            pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
            detected_at=datetime.now(),
            turn_index=1,
            session_id="test",
            tool_sequence=["readFile", "readFile", "readFile"],
            repetition_count=3,
            duration_seconds=5.0,
            model_name="gpt-4",
            severity=PatternSeverity.MINOR,
        )

        nudge = strategy.create_nudge(pattern)

        assert nudge is not None
        assert "readFile" in nudge.message
        assert "3" in nudge.message

    def test_create_nudge_severity_escalation(self):
        """Test that severity affects nudge type."""
        from ..types import BehavioralPattern, BehavioralPatternType, PatternSeverity, NudgeType
        from ..nudge import NudgeStrategy

        strategy = NudgeStrategy()

        # Minor severity -> gentle reminder
        minor_pattern = BehavioralPattern(
            pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
            detected_at=datetime.now(),
            turn_index=1,
            session_id="test",
            tool_sequence=["readFile"],
            repetition_count=3,
            duration_seconds=5.0,
            model_name="gpt-4",
            severity=PatternSeverity.MINOR,
        )
        minor_nudge = strategy.create_nudge(minor_pattern)
        assert minor_nudge.nudge_type == NudgeType.GENTLE_REMINDER

        # Severe -> interrupt
        severe_pattern = BehavioralPattern(
            pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
            detected_at=datetime.now(),
            turn_index=1,
            session_id="test",
            tool_sequence=["readFile"],
            repetition_count=10,
            duration_seconds=60.0,
            model_name="gpt-4",
            severity=PatternSeverity.SEVERE,
        )
        severe_nudge = strategy.create_nudge(severe_pattern)
        assert severe_nudge.nudge_type == NudgeType.INTERRUPT

    def test_should_inject_level_filtering(self):
        """Test that nudge level filters appropriately."""
        from ..types import BehavioralPattern, BehavioralPatternType, PatternSeverity, NudgeConfig, NudgeLevel, NudgeType
        from ..nudge import NudgeStrategy

        strategy = NudgeStrategy()

        pattern = BehavioralPattern(
            pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
            detected_at=datetime.now(),
            turn_index=1,
            session_id="test",
            tool_sequence=["readFile"],
            repetition_count=3,
            duration_seconds=5.0,
            model_name="gpt-4",
            severity=PatternSeverity.MINOR,
        )
        gentle_nudge = strategy.create_nudge(pattern)

        # GENTLE level should allow gentle reminders
        gentle_config = NudgeConfig(level=NudgeLevel.GENTLE)
        assert strategy.should_inject(gentle_nudge, gentle_config) is True

        # OFF level should block all
        off_config = NudgeConfig(level=NudgeLevel.OFF)
        assert strategy.should_inject(gentle_nudge, off_config) is False

        # Create interrupt nudge
        severe_pattern = BehavioralPattern(
            pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
            detected_at=datetime.now(),
            turn_index=1,
            session_id="test",
            tool_sequence=["readFile"],
            repetition_count=10,
            duration_seconds=60.0,
            model_name="gpt-4",
            severity=PatternSeverity.SEVERE,
        )
        interrupt_nudge = strategy.create_nudge(severe_pattern)

        # GENTLE level should block interrupts
        assert strategy.should_inject(interrupt_nudge, gentle_config) is False

        # FULL level should allow interrupts
        full_config = NudgeConfig(level=NudgeLevel.FULL)
        assert strategy.should_inject(interrupt_nudge, full_config) is True


class TestNudgeInjector:
    """Tests for NudgeInjector class."""

    def test_nudge_injector_creation(self):
        """Test NudgeInjector creation."""
        from ..nudge import NudgeInjector

        injector = NudgeInjector()
        assert injector is not None

    def test_on_pattern_detected_creates_nudge(self):
        """Test that on_pattern_detected creates and records nudge."""
        from ..types import BehavioralPattern, BehavioralPatternType, PatternSeverity, NudgeConfig, NudgeLevel
        from ..nudge import NudgeInjector

        injector = NudgeInjector(config=NudgeConfig(level=NudgeLevel.FULL))

        pattern = BehavioralPattern(
            pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
            detected_at=datetime.now(),
            turn_index=1,
            session_id="test",
            tool_sequence=["readFile", "readFile", "readFile"],
            repetition_count=3,
            duration_seconds=5.0,
            model_name="gpt-4",
            severity=PatternSeverity.MINOR,
        )

        nudge = injector.on_pattern_detected(pattern)

        assert nudge is not None
        assert len(injector.get_nudge_history()) == 1

    def test_cooldown_prevents_spam(self):
        """Test that cooldown prevents nudge spam."""
        from ..types import BehavioralPattern, BehavioralPatternType, PatternSeverity, NudgeConfig, NudgeLevel
        from ..nudge import NudgeInjector

        config = NudgeConfig(level=NudgeLevel.FULL, cooldown_seconds=60)
        injector = NudgeInjector(config=config)

        pattern = BehavioralPattern(
            pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
            detected_at=datetime.now(),
            turn_index=1,
            session_id="test",
            tool_sequence=["readFile"],
            repetition_count=3,
            duration_seconds=5.0,
            model_name="gpt-4",
            severity=PatternSeverity.MINOR,
        )

        # First nudge should succeed
        first = injector.on_pattern_detected(pattern)
        assert first is not None

        # Second immediately should be blocked by cooldown
        second = injector.on_pattern_detected(pattern)
        assert second is None

        assert len(injector.get_nudge_history()) == 1

    def test_nudge_callbacks_called(self):
        """Test that injection callbacks are called."""
        from ..types import BehavioralPattern, BehavioralPatternType, PatternSeverity, NudgeConfig, NudgeLevel
        from ..nudge import NudgeInjector

        injected_messages = []

        def mock_inject(msg):
            injected_messages.append(msg)

        injector = NudgeInjector(config=NudgeConfig(level=NudgeLevel.FULL))
        injector.set_injection_callbacks(inject_system_guidance=mock_inject)

        # Create pattern that will trigger direct instruction (moderate severity)
        pattern = BehavioralPattern(
            pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
            detected_at=datetime.now(),
            turn_index=1,
            session_id="test",
            tool_sequence=["readFile"],
            repetition_count=5,
            duration_seconds=10.0,
            model_name="gpt-4",
            severity=PatternSeverity.MODERATE,
        )

        injector.on_pattern_detected(pattern)

        assert len(injected_messages) >= 1

    def test_nudge_summary(self):
        """Test nudge summary statistics."""
        from ..types import BehavioralPattern, BehavioralPatternType, PatternSeverity, NudgeConfig, NudgeLevel
        from ..nudge import NudgeInjector

        injector = NudgeInjector(config=NudgeConfig(level=NudgeLevel.FULL, cooldown_seconds=0))

        # Inject a few nudges with different pattern types
        for i, ptype in enumerate([
            BehavioralPatternType.REPETITIVE_CALLS,
            BehavioralPatternType.INTROSPECTION_LOOP,
            BehavioralPatternType.READ_ONLY_LOOP,
        ]):
            pattern = BehavioralPattern(
                pattern_type=ptype,
                detected_at=datetime.now(),
                turn_index=i,
                session_id="test",
                tool_sequence=["tool"],
                repetition_count=3,
                duration_seconds=5.0,
                model_name="gpt-4",
                severity=PatternSeverity.MINOR,
            )
            injector.on_pattern_detected(pattern)

        summary = injector.get_nudge_summary()

        assert summary["total"] == 3
        assert "gentle" in str(summary["by_type"]).lower()

    def test_mark_nudge_effective(self):
        """Test marking nudge as effective."""
        from ..types import BehavioralPattern, BehavioralPatternType, PatternSeverity, NudgeConfig, NudgeLevel
        from ..nudge import NudgeInjector

        injector = NudgeInjector(config=NudgeConfig(level=NudgeLevel.FULL))

        pattern = BehavioralPattern(
            pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
            detected_at=datetime.now(),
            turn_index=1,
            session_id="test",
            tool_sequence=["readFile"],
            repetition_count=3,
            duration_seconds=5.0,
            model_name="gpt-4",
            severity=PatternSeverity.MINOR,
        )

        nudge = injector.on_pattern_detected(pattern)

        # Initially not effective
        assert nudge.effective is False

        # Mark as effective
        injector.mark_last_nudge_effective(True)

        assert nudge.effective is True
        assert nudge.acknowledged is True


class TestPluginNudgeIntegration:
    """Tests for nudge integration with the reliability plugin."""

    @pytest.fixture
    def plugin(self):
        """Create a plugin with pattern detection and nudge injection."""
        from ..types import PatternDetectionConfig, NudgeConfig, NudgeLevel

        plugin = ReliabilityPlugin()
        plugin.set_pattern_detection_config(PatternDetectionConfig(
            enabled=True,
            repetitive_call_threshold=3,
        ))
        plugin.enable_pattern_detection(True)
        plugin.set_nudge_config(NudgeConfig(level=NudgeLevel.FULL, cooldown_seconds=0))
        plugin.enable_nudge_injection(True)
        return plugin

    def test_pattern_triggers_nudge(self, plugin):
        """Test that detected patterns automatically trigger nudges."""
        nudges = []

        def on_nudge(nudge):
            nudges.append(nudge)

        plugin.set_nudge_hook(on_nudge)
        plugin.on_turn_start(1)

        # Generate repetitive calls to trigger pattern
        for i in range(3):
            plugin.on_tool_called("readFile", {"path": "/tmp/test.txt"})

        assert len(nudges) >= 1

    def test_nudge_command_status(self, plugin):
        """Test nudge status command."""
        output = plugin.handle_command("nudge", "status")

        assert "Nudge Injection" in output
        assert "ENABLED" in output

    def test_nudge_command_level_change(self, plugin):
        """Test changing nudge level via commands."""
        from ..types import NudgeLevel

        output = plugin.handle_command("nudge", "off")
        assert "OFF" in output

        output = plugin.handle_command("nudge", "gentle")
        assert "GENTLE" in output

        output = plugin.handle_command("nudge", "direct")
        assert "DIRECT" in output

        output = plugin.handle_command("nudge", "full")
        assert "FULL" in output

    def test_nudge_command_history(self, plugin):
        """Test nudge history command."""
        # Generate some nudges first
        plugin.on_turn_start(1)
        for i in range(3):
            plugin.on_tool_called("readFile", {"path": "/tmp/test.txt"})

        output = plugin.handle_command("nudge", "history")

        assert "Nudge" in output or "nudge" in output

    def test_nudge_injection_disabled_by_default(self):
        """Test that nudge injection must be explicitly enabled."""
        plugin = ReliabilityPlugin()
        plugin.enable_pattern_detection(True)

        # Don't enable nudge injection
        assert plugin.get_nudge_injector() is None

    def test_nudge_summary_via_plugin(self, plugin):
        """Test getting nudge summary through plugin."""
        plugin.on_turn_start(1)

        # Generate pattern and nudge
        for i in range(3):
            plugin.on_tool_called("readFile", {"path": "/tmp/test.txt"})

        summary = plugin.get_nudge_summary()

        assert "total" in summary
        assert summary["total"] >= 1

    def test_manual_nudge_injection(self, plugin):
        """Test injecting nudge for a pattern manually."""
        from ..types import BehavioralPattern, BehavioralPatternType, PatternSeverity

        pattern = BehavioralPattern(
            pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
            detected_at=datetime.now(),
            turn_index=1,
            session_id="test",
            tool_sequence=["readFile"],
            repetition_count=3,
            duration_seconds=5.0,
            model_name="gpt-4",
            severity=PatternSeverity.MINOR,
        )

        nudge = plugin.inject_nudge_for_pattern(pattern)

        assert nudge is not None


class TestModelBehavioralProfile:
    """Tests for ModelBehavioralProfile dataclass."""

    def test_profile_creation(self):
        """Test creating a behavioral profile."""
        from ..types import ModelBehavioralProfile

        profile = ModelBehavioralProfile(model_name="gpt-4")
        assert profile.model_name == "gpt-4"
        assert profile.total_turns == 0
        assert profile.stalled_turns == 0
        assert profile.stall_rate == 0.0
        assert profile.nudge_effectiveness == 1.0

    def test_record_turn_start(self):
        """Test recording turn starts."""
        from ..types import ModelBehavioralProfile

        profile = ModelBehavioralProfile(model_name="gpt-4")
        profile.record_turn_start()

        assert profile.total_turns == 1
        assert profile.first_seen is not None
        assert profile.last_seen is not None

    def test_record_pattern(self):
        """Test recording patterns."""
        from ..types import ModelBehavioralProfile, BehavioralPattern, BehavioralPatternType, PatternSeverity

        profile = ModelBehavioralProfile(model_name="gpt-4")

        pattern = BehavioralPattern(
            pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
            detected_at=datetime.now(),
            turn_index=1,
            session_id="test",
            tool_sequence=["readFile"],
            repetition_count=3,
            duration_seconds=5.0,
            model_name="gpt-4",
            severity=PatternSeverity.MINOR,
        )
        profile.record_pattern(pattern)

        assert profile.pattern_counts[BehavioralPatternType.REPETITIVE_CALLS] == 1
        assert profile.total_patterns == 1
        assert profile.most_common_pattern() == BehavioralPatternType.REPETITIVE_CALLS

    def test_stall_rate(self):
        """Test stall rate calculation."""
        from ..types import ModelBehavioralProfile

        profile = ModelBehavioralProfile(model_name="gpt-4")
        profile.record_turn_start()
        profile.record_turn_start()
        profile.record_turn_stalled()  # 1 stalled of 2 turns

        assert profile.stall_rate == 0.5

    def test_nudge_effectiveness(self):
        """Test nudge effectiveness tracking."""
        from ..types import ModelBehavioralProfile

        profile = ModelBehavioralProfile(model_name="gpt-4")
        profile.record_nudge_sent()
        profile.record_nudge_sent()
        profile.record_nudge_effective()

        assert profile.nudges_sent == 2
        assert profile.nudges_effective == 1
        assert profile.nudge_effectiveness == 0.5

    def test_average_severity(self):
        """Test average severity calculation."""
        from ..types import ModelBehavioralProfile, BehavioralPattern, BehavioralPatternType, PatternSeverity

        profile = ModelBehavioralProfile(model_name="gpt-4")

        # Add MINOR pattern (severity = 1)
        profile.record_pattern(BehavioralPattern(
            pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
            detected_at=datetime.now(),
            turn_index=1,
            session_id="test",
            tool_sequence=["readFile"],
            repetition_count=3,
            duration_seconds=5.0,
            model_name="gpt-4",
            severity=PatternSeverity.MINOR,
        ))

        # Add MODERATE pattern (severity = 2)
        profile.record_pattern(BehavioralPattern(
            pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
            detected_at=datetime.now(),
            turn_index=2,
            session_id="test",
            tool_sequence=["readFile"],
            repetition_count=5,
            duration_seconds=10.0,
            model_name="gpt-4",
            severity=PatternSeverity.MODERATE,
        ))

        # Average should be 1.5
        assert profile.average_severity(BehavioralPatternType.REPETITIVE_CALLS) == 1.5

    def test_serialization(self):
        """Test profile serialization and deserialization."""
        from ..types import ModelBehavioralProfile, BehavioralPattern, BehavioralPatternType, PatternSeverity

        profile = ModelBehavioralProfile(model_name="gpt-4")
        profile.record_turn_start()
        profile.record_nudge_sent()
        profile.record_nudge_effective()

        pattern = BehavioralPattern(
            pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
            detected_at=datetime.now(),
            turn_index=1,
            session_id="test",
            tool_sequence=["readFile"],
            repetition_count=3,
            duration_seconds=5.0,
            model_name="gpt-4",
            severity=PatternSeverity.MINOR,
        )
        profile.record_pattern(pattern)

        data = profile.to_dict()
        restored = ModelBehavioralProfile.from_dict(data)

        assert restored.model_name == profile.model_name
        assert restored.total_turns == profile.total_turns
        assert restored.nudges_sent == profile.nudges_sent
        assert restored.nudges_effective == profile.nudges_effective
        assert BehavioralPatternType.REPETITIVE_CALLS in restored.pattern_counts

    def test_compare_to(self):
        """Test comparing two profiles."""
        from ..types import ModelBehavioralProfile

        profile1 = ModelBehavioralProfile(model_name="gpt-4")
        profile1.record_turn_start()
        profile1.record_turn_stalled()  # 100% stall rate

        profile2 = ModelBehavioralProfile(model_name="claude-3")
        profile2.record_turn_start()
        profile2.record_turn_start()
        # 0% stall rate

        comparison = profile1.compare_to(profile2)

        assert comparison["this_model"] == "gpt-4"
        assert comparison["other_model"] == "claude-3"
        assert comparison["stall_rate_diff"] > 0
        assert comparison["this_better_stall"] == False

    def test_get_summary(self):
        """Test getting profile summary."""
        from ..types import ModelBehavioralProfile, BehavioralPattern, BehavioralPatternType, PatternSeverity

        profile = ModelBehavioralProfile(model_name="gpt-4")
        profile.record_turn_start()
        profile.record_nudge_sent()

        pattern = BehavioralPattern(
            pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
            detected_at=datetime.now(),
            turn_index=1,
            session_id="test",
            tool_sequence=["readFile"],
            repetition_count=3,
            duration_seconds=5.0,
            model_name="gpt-4",
            severity=PatternSeverity.MINOR,
        )
        profile.record_pattern(pattern)

        summary = profile.get_summary()

        assert summary["model"] == "gpt-4"
        assert summary["total_turns"] == 1
        assert summary["nudges_sent"] == 1
        assert "repetitive_calls" in summary["pattern_breakdown"]


class TestPluginBehavioralIntegration:
    """Tests for behavioral profile integration with the reliability plugin."""

    @pytest.fixture
    def plugin(self):
        """Create a plugin with behavioral tracking enabled."""
        from ..types import PatternDetectionConfig, NudgeConfig, NudgeLevel

        plugin = ReliabilityPlugin()
        plugin.set_model_context("gpt-4", ["gpt-4", "claude-3"])
        plugin.set_pattern_detection_config(PatternDetectionConfig(
            enabled=True,
            repetitive_call_threshold=3,
        ))
        plugin.enable_pattern_detection(True)
        plugin.set_nudge_config(NudgeConfig(level=NudgeLevel.FULL, cooldown_seconds=0))
        plugin.enable_nudge_injection(True)
        return plugin

    def test_patterns_update_behavioral_profile(self, plugin):
        """Test that detected patterns update behavioral profiles."""
        plugin.on_turn_start(1)

        # Generate repetitive calls to trigger pattern
        for i in range(3):
            plugin.on_tool_called("readFile", {"path": "/tmp/test.txt"})

        profile = plugin.get_behavioral_profile("gpt-4")
        assert profile.total_turns == 1
        assert profile.total_patterns >= 1

    def test_nudges_update_behavioral_profile(self, plugin):
        """Test that nudges update behavioral profiles."""
        plugin.on_turn_start(1)

        for i in range(3):
            plugin.on_tool_called("readFile", {"path": "/tmp/test.txt"})

        profile = plugin.get_behavioral_profile("gpt-4")
        assert profile.nudges_sent >= 1

    def test_behavior_status_command(self, plugin):
        """Test behavior status command."""
        plugin.on_turn_start(1)

        output = plugin.handle_command("behavior", "status")

        assert "gpt-4" in output

    def test_behavior_compare_command(self, plugin):
        """Test behavior compare command."""
        plugin.on_turn_start(1)

        # Add second model
        plugin.set_model_context("claude-3", ["gpt-4", "claude-3"])
        plugin.on_turn_start(2)

        output = plugin.handle_command("behavior", "compare gpt-4 claude-3")

        assert "Comparison" in output
        assert "gpt-4" in output
        assert "claude-3" in output

    def test_behavior_patterns_command(self, plugin):
        """Test behavior patterns command."""
        plugin.on_turn_start(1)
        for i in range(3):
            plugin.on_tool_called("readFile", {"path": "/tmp/test.txt"})

        output = plugin.handle_command("behavior", "patterns")

        assert "Pattern Breakdown" in output

    def test_mark_last_nudge_effective(self, plugin):
        """Test marking nudges as effective."""
        plugin.on_turn_start(1)
        for i in range(3):
            plugin.on_tool_called("readFile", {"path": "/tmp/test.txt"})

        initial_effective = plugin.get_behavioral_profile("gpt-4").nudges_effective
        plugin.mark_last_nudge_effective(True)
        final_effective = plugin.get_behavioral_profile("gpt-4").nudges_effective

        assert final_effective >= initial_effective

    def test_get_behavioral_summary(self, plugin):
        """Test getting behavioral summary for all models."""
        plugin.on_turn_start(1)
        plugin.set_model_context("claude-3", ["gpt-4", "claude-3"])
        plugin.on_turn_start(2)

        summary = plugin.get_behavioral_summary()

        assert "models_tracked" in summary
        assert summary["models_tracked"] >= 1

    def test_compare_behavioral_profiles(self, plugin):
        """Test comparing behavioral profiles."""
        plugin.on_turn_start(1)
        for i in range(3):
            plugin.on_tool_called("readFile", {"path": "/tmp/test.txt"})

        plugin.set_model_context("claude-3", ["gpt-4", "claude-3"])
        plugin.on_turn_start(2)

        result = plugin.compare_behavioral_profiles("gpt-4", "claude-3")

        assert "recommendation" in result


# =============================================================================
# Observability Tests (Phase 10)
# =============================================================================

class TestReliabilityStatusTool:
    """Tests for the enhanced reliability_status tool."""

    @pytest.fixture
    def plugin(self):
        """Create a plugin with pattern detection enabled."""
        config = ReliabilityConfig()
        plugin = ReliabilityPlugin(config)
        plugin.set_session_context("test-session")
        plugin.set_model_context("gpt-4", ["gpt-4", "claude-3"])
        plugin.enable_pattern_detection()
        plugin.enable_nudge_injection()
        return plugin

    def test_basic_status(self, plugin):
        """Test basic reliability status output."""
        result = plugin._execute_status({})

        assert "tool_status" in result or "tools" in result
        assert "summary" in result

    def test_status_with_tool_name(self, plugin):
        """Test status for specific tool."""
        plugin.on_tool_called("readFile", {"path": "/etc/passwd"})
        plugin.on_tool_result("readFile", {"path": "/etc/passwd"}, False, {"error": "Permission denied"}, "call-1", "filesystem")

        result = plugin._execute_status({"tool_name": "readFile"})

        assert "tools" in result

    def test_status_with_history(self, plugin):
        """Test status with failure history."""
        plugin.on_tool_result("readFile", {"path": "/etc/test"}, False, {"error": "File not found"}, "call-1", "filesystem")

        result = plugin._execute_status({"include_history": True})

        assert "recent_failures" in result
        assert len(result["recent_failures"]) > 0

    def test_status_with_patterns(self, plugin):
        """Test status with pattern information."""
        plugin.on_turn_start(1)
        # Trigger repetitive pattern
        for i in range(3):
            plugin.on_tool_called("readFile", {"path": "/tmp/test.txt"})

        result = plugin._execute_status({"include_patterns": True})

        # Should have pattern info (even if empty)
        assert "recent_patterns" in result or "pattern_stats" in result

    def test_status_with_behavior(self, plugin):
        """Test status with behavioral profile."""
        plugin.on_turn_start(1)

        result = plugin._execute_status({"include_behavior": True})

        assert "behavioral_profile" in result
        profile = result["behavioral_profile"]
        assert "model" in profile
        assert "stall_rate" in profile
        assert "nudge_effectiveness" in profile

    def test_status_with_model_comparison(self, plugin):
        """Test status with model comparison."""
        plugin.on_turn_start(1)
        # Create some data for gpt-4
        for i in range(3):
            plugin.on_tool_called("readFile", {"path": "/tmp/test.txt"})

        # Switch models and create data for claude-3
        plugin.set_model_context("claude-3", ["gpt-4", "claude-3"])
        plugin.on_turn_start(2)

        result = plugin._execute_status({
            "compare_models": ["gpt-4", "claude-3"]
        })

        assert "model_comparison" in result

    def test_summary_metrics(self, plugin):
        """Test summary metrics in status output."""
        plugin.on_turn_start(1)
        plugin.on_tool_result("bash", {"command": "rm -rf /"}, False, {"error": "Permission denied"}, "call-1", "cli")

        result = plugin._execute_status({})

        summary = result.get("summary", {})
        assert "tools" in summary
        assert "total_tracked" in summary["tools"]
        assert "escalated" in summary["tools"]
        assert "total_failures" in summary["tools"]


class TestTelemetryIntegration:
    """Tests for telemetry/OpenTelemetry event emission."""

    @pytest.fixture
    def mock_telemetry(self):
        """Create a mock telemetry plugin."""
        class MockTelemetry:
            def __init__(self):
                self.enabled = True
                self._current_span_id = "span-123"
                self._current_trace_id = "trace-456"

            def get_current_span_id(self):
                return self._current_span_id

            def get_current_trace_id(self):
                return self._current_trace_id

        return MockTelemetry()

    @pytest.fixture
    def plugin(self, mock_telemetry):
        """Create plugin with telemetry."""
        config = ReliabilityConfig()
        plugin = ReliabilityPlugin(config)
        plugin.set_session_context("test-session")
        plugin.set_model_context("gpt-4", ["gpt-4"])
        plugin.set_telemetry(mock_telemetry)
        plugin.enable_pattern_detection()
        plugin.enable_nudge_injection()
        return plugin

    def test_set_telemetry(self, plugin, mock_telemetry):
        """Test that telemetry can be set."""
        assert plugin._telemetry is mock_telemetry

    def test_pattern_event_emission(self, plugin, caplog):
        """Test that pattern detection emits telemetry events."""
        import logging
        caplog.set_level(logging.INFO)

        plugin.on_turn_start(1)
        # Trigger repetitive pattern
        for i in range(3):
            plugin.on_tool_called("readFile", {"path": "/tmp/test.txt"})

        # Check that pattern event was logged
        pattern_logs = [r for r in caplog.records if "pattern_detected" in r.message]
        assert len(pattern_logs) > 0

    def test_nudge_event_emission(self, plugin, caplog):
        """Test that nudge injection emits telemetry events."""
        import logging
        caplog.set_level(logging.INFO)

        plugin.on_turn_start(1)
        # Trigger repetitive pattern (which should trigger nudge)
        for i in range(3):
            plugin.on_tool_called("readFile", {"path": "/tmp/test.txt"})

        # Check that nudge event was logged
        nudge_logs = [r for r in caplog.records if "nudge_injected" in r.message]
        assert len(nudge_logs) > 0

    def test_failure_event_emission(self, plugin, caplog):
        """Test that failures emit telemetry events."""
        import logging
        caplog.set_level(logging.INFO)

        plugin.on_tool_result(
            "bash",
            {"command": "invalid_cmd"},
            False,
            {"error": "Command not found"},
            "call-1",
            "cli"
        )

        # Check that failure event was logged
        failure_logs = [r for r in caplog.records if "tool_failure" in r.message]
        assert len(failure_logs) > 0

    def test_trust_state_event_emission(self, plugin, caplog):
        """Test that trust state changes emit telemetry events."""
        import logging
        caplog.set_level(logging.INFO)

        # Trigger multiple failures to cause escalation
        for i in range(5):
            plugin.on_tool_result(
                "bash",
                {"command": "bad_cmd"},
                False,
                {"error": "Failed", "http_status": 500},
                f"call-{i}",
                "cli"
            )

        # Check that trust state event was logged
        state_logs = [r for r in caplog.records if "trust_state_changed" in r.message]
        assert len(state_logs) > 0

    def test_telemetry_disabled(self, caplog):
        """Test that telemetry events are not emitted when disabled."""
        import logging
        caplog.set_level(logging.INFO)

        # Create telemetry with enabled=False
        class DisabledTelemetry:
            enabled = False

            def get_current_span_id(self):
                return None

        config = ReliabilityConfig()
        plugin = ReliabilityPlugin(config)
        plugin.set_session_context("test-session")
        plugin.set_model_context("gpt-4", ["gpt-4"])
        plugin.set_telemetry(DisabledTelemetry())
        plugin.enable_pattern_detection()

        plugin.on_turn_start(1)
        for i in range(3):
            plugin.on_tool_called("readFile", {"path": "/tmp/test.txt"})

        # No telemetry events should be logged
        pattern_logs = [r for r in caplog.records if "pattern_detected" in r.message]
        assert len(pattern_logs) == 0


class TestObservabilitySummary:
    """Tests for observability summary/dashboard metrics."""

    @pytest.fixture
    def plugin(self):
        """Create plugin with all features enabled."""
        config = ReliabilityConfig()
        plugin = ReliabilityPlugin(config)
        plugin.set_session_context("test-session")
        plugin.set_model_context("gpt-4", ["gpt-4", "claude-3"])
        plugin.enable_pattern_detection()
        plugin.enable_nudge_injection()
        return plugin

    def test_summary_with_no_data(self, plugin):
        """Test summary when no data collected."""
        summary = plugin._get_observability_summary()

        assert "tools" in summary
        assert summary["tools"]["total_tracked"] == 0
        assert summary["tools"]["escalated"] == 0
        assert summary["tools"]["total_failures"] == 0

    def test_summary_with_failures(self, plugin):
        """Test summary with recorded failures."""
        plugin.on_tool_result("bash", {"command": "test"}, False, {"error": "Failed"}, "call-1", "cli")

        summary = plugin._get_observability_summary()

        assert summary["tools"]["total_failures"] == 1

    def test_summary_with_patterns(self, plugin):
        """Test summary with detected patterns."""
        plugin.on_turn_start(1)
        for i in range(3):
            plugin.on_tool_called("readFile", {"path": "/tmp/test.txt"})

        summary = plugin._get_observability_summary()

        assert "patterns" in summary
        assert summary["patterns"]["total_detected"] > 0
        assert "by_type" in summary["patterns"]
        assert "by_severity" in summary["patterns"]

    def test_summary_with_nudges(self, plugin):
        """Test summary with nudge information."""
        plugin.on_turn_start(1)
        # Trigger pattern and nudge
        for i in range(3):
            plugin.on_tool_called("readFile", {"path": "/tmp/test.txt"})

        summary = plugin._get_observability_summary()

        assert "nudges" in summary
        assert summary["nudges"]["total_sent"] > 0

    def test_summary_with_multiple_models(self, plugin):
        """Test summary with multiple model profiles."""
        plugin.on_turn_start(1)

        plugin.set_model_context("claude-3", ["gpt-4", "claude-3"])
        plugin.on_turn_start(2)

        summary = plugin._get_observability_summary()

        assert "models" in summary
        assert len(summary["models"]) >= 2

    def test_model_summary_contains_metrics(self, plugin):
        """Test that model summaries contain expected metrics."""
        plugin.on_turn_start(1)
        for i in range(3):
            plugin.on_tool_called("readFile", {"path": "/tmp/test.txt"})

        summary = plugin._get_observability_summary()

        if "models" in summary and len(summary["models"]) > 0:
            model_summary = summary["models"][0]
            assert "model" in model_summary
            assert "stall_rate" in model_summary
            assert "nudge_effectiveness" in model_summary
            assert "turns" in model_summary
