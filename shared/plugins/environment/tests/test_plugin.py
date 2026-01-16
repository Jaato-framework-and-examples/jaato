"""Tests for the environment awareness plugin."""

import json
import os
import platform
import sys
import pytest

from ..plugin import EnvironmentPlugin


class TestEnvironmentPluginInitialization:
    """Tests for plugin initialization."""

    def test_plugin_name(self):
        plugin = EnvironmentPlugin()
        assert plugin.name == "environment"

    def test_initialize_without_config(self):
        plugin = EnvironmentPlugin()
        plugin.initialize()
        # No config required, should complete without error

    def test_initialize_with_config(self):
        plugin = EnvironmentPlugin()
        plugin.initialize({"some_key": "some_value"})
        # Unknown config keys are ignored

    def test_shutdown(self):
        plugin = EnvironmentPlugin()
        plugin.initialize()
        plugin.shutdown()
        # Should complete without error


class TestEnvironmentPluginToolSchemas:
    """Tests for tool schemas."""

    def test_get_tool_schemas(self):
        plugin = EnvironmentPlugin()
        schemas = plugin.get_tool_schemas()

        assert len(schemas) == 1
        assert schemas[0].name == "get_environment"

    def test_get_environment_schema(self):
        plugin = EnvironmentPlugin()
        schemas = plugin.get_tool_schemas()
        env_tool = schemas[0]
        schema = env_tool.parameters

        assert schema["type"] == "object"
        assert "aspect" in schema["properties"]
        assert schema["required"] == []

    def test_get_environment_aspect_enum(self):
        plugin = EnvironmentPlugin()
        schemas = plugin.get_tool_schemas()
        env_tool = schemas[0]
        schema = env_tool.parameters

        aspect_enum = schema["properties"]["aspect"]["enum"]
        assert "os" in aspect_enum
        assert "shell" in aspect_enum
        assert "arch" in aspect_enum
        assert "cwd" in aspect_enum
        assert "terminal" in aspect_enum
        assert "context" in aspect_enum
        assert "session" in aspect_enum
        assert "all" in aspect_enum


class TestEnvironmentPluginExecutors:
    """Tests for executor mapping."""

    def test_get_executors(self):
        plugin = EnvironmentPlugin()
        executors = plugin.get_executors()

        assert "get_environment" in executors
        assert callable(executors["get_environment"])


class TestEnvironmentPluginExecution:
    """Tests for get_environment execution."""

    def test_get_environment_all(self):
        """Test getting all environment info (default)."""
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({}))

        assert "os" in result
        assert "shell" in result
        assert "arch" in result
        assert "cwd" in result
        assert "terminal" in result

    def test_get_environment_explicit_all(self):
        """Test getting all environment info explicitly."""
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "all"}))

        assert "os" in result
        assert "shell" in result
        assert "arch" in result
        assert "cwd" in result
        assert "terminal" in result

    def test_get_environment_os_aspect(self):
        """Test getting OS info only."""
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "os"}))

        # Single aspect returns flattened result
        assert "type" in result
        assert "name" in result
        assert "friendly_name" in result

    def test_get_environment_shell_aspect(self):
        """Test getting shell info only."""
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "shell"}))

        assert "default" in result
        assert "current" in result
        assert "path_separator" in result
        assert "dir_separator" in result

    def test_get_environment_arch_aspect(self):
        """Test getting architecture info only."""
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "arch"}))

        assert "machine" in result
        assert "normalized" in result

    def test_get_environment_cwd_aspect(self):
        """Test getting current working directory only."""
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "cwd"}))

        # cwd returns a string directly
        assert isinstance(result, str)
        assert os.path.isabs(result)

    def test_get_environment_terminal_aspect(self):
        """Test getting terminal info only."""
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "terminal"}))

        # Terminal info should have these keys
        assert "term" in result
        assert "term_program" in result
        assert "colorterm" in result
        assert "multiplexer" in result
        assert "color_depth" in result
        assert "emulator" in result

    def test_get_environment_invalid_aspect(self):
        """Test handling of invalid aspect."""
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "invalid"}))

        assert "error" in result
        assert "Invalid aspect" in result["error"]

    def test_get_environment_context_aspect_without_session(self):
        """Test getting context info without session returns error."""
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "context"}))

        # Without session, should return error
        assert "error" in result
        assert "Session not available" in result["error"]

    def test_get_environment_all_includes_context(self):
        """Test that 'all' includes context key."""
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "all"}))

        # Context should be in result (even if it's an error without session)
        assert "context" in result


class TestEnvironmentPluginOSInfo:
    """Tests for OS information accuracy."""

    def test_os_type_lowercase(self):
        """Test that OS type is lowercase."""
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "os"}))

        assert result["type"] == platform.system().lower()

    def test_os_friendly_name(self):
        """Test that friendly name is provided."""
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "os"}))

        system = platform.system()
        if system == "Darwin":
            assert result["friendly_name"] == "macOS"
        elif system == "Linux":
            assert result["friendly_name"] == "Linux"
        elif system == "Windows":
            assert result["friendly_name"] == "Windows"


class TestEnvironmentPluginShellInfo:
    """Tests for shell information accuracy."""

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_shell_info_unix(self):
        """Test shell info on Unix."""
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "shell"}))

        assert result["path_separator"] == ":"
        assert result["dir_separator"] == "/"
        # Shell should be detected from SHELL env var
        if "SHELL" in os.environ:
            assert result["path"] == os.environ["SHELL"]

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_shell_info_windows(self):
        """Test shell info on Windows."""
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "shell"}))

        assert result["path_separator"] == ";"
        assert result["dir_separator"] == "\\"
        assert "powershell_available" in result


class TestEnvironmentPluginTerminalInfo:
    """Tests for terminal information accuracy."""

    def test_terminal_term_from_env(self, monkeypatch):
        """Test that TERM is read from environment."""
        monkeypatch.setenv("TERM", "xterm-256color")
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "terminal"}))

        assert result["term"] == "xterm-256color"

    def test_terminal_color_depth_256(self, monkeypatch):
        """Test 256 color detection."""
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.delenv("COLORTERM", raising=False)
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "terminal"}))

        assert result["color_depth"] == "256"

    def test_terminal_color_depth_truecolor(self, monkeypatch):
        """Test truecolor detection via COLORTERM."""
        monkeypatch.setenv("TERM", "xterm")
        monkeypatch.setenv("COLORTERM", "truecolor")
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "terminal"}))

        assert result["color_depth"] == "24bit"

    def test_terminal_tmux_detection(self, monkeypatch):
        """Test tmux multiplexer detection."""
        monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,12345,0")
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "terminal"}))

        assert result["multiplexer"] == "tmux"

    def test_terminal_screen_detection(self, monkeypatch):
        """Test screen multiplexer detection."""
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.setenv("STY", "12345.pts-0.hostname")
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "terminal"}))

        assert result["multiplexer"] == "screen"

    def test_terminal_no_multiplexer(self, monkeypatch):
        """Test no multiplexer detected."""
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.delenv("STY", raising=False)
        monkeypatch.setenv("TERM", "xterm")
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "terminal"}))

        assert result["multiplexer"] is None

    def test_terminal_emulator_from_term_program(self, monkeypatch):
        """Test emulator detection from TERM_PROGRAM."""
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "terminal"}))

        assert result["emulator"] == "iTerm.app"


class TestEnvironmentPluginArchInfo:
    """Tests for architecture information accuracy."""

    def test_arch_machine_matches_platform(self):
        """Test that machine matches platform.machine()."""
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "arch"}))

        assert result["machine"] == platform.machine()

    def test_arch_normalized(self):
        """Test that normalized architecture is provided."""
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "arch"}))

        # Normalized should be one of the common names
        assert result["normalized"] in ["x86_64", "arm64", "x86", platform.machine().lower()]


class TestEnvironmentPluginContextInfo:
    """Tests for context/token usage information."""

    def test_set_session(self):
        """Test that set_session stores the session reference."""
        plugin = EnvironmentPlugin()

        class MockSession:
            pass

        mock_session = MockSession()
        plugin.set_session(mock_session)

        assert plugin._session is mock_session

    def test_shutdown_clears_session(self):
        """Test that shutdown clears the session reference."""
        plugin = EnvironmentPlugin()

        class MockSession:
            pass

        plugin.set_session(MockSession())
        assert plugin._session is not None

        plugin.shutdown()
        assert plugin._session is None

    def test_context_aspect_with_session(self):
        """Test getting context info with a mocked session."""
        plugin = EnvironmentPlugin()

        class MockSession:
            def get_context_usage(self):
                return {
                    "model": "test-model",
                    "context_limit": 100000,
                    "total_tokens": 5000,
                    "prompt_tokens": 4000,
                    "output_tokens": 1000,
                    "tokens_remaining": 95000,
                    "percent_used": 5.0,
                    "turns": 3,
                }

        plugin.set_session(MockSession())
        result = json.loads(plugin._get_environment({"aspect": "context"}))

        assert result["model"] == "test-model"
        assert result["context_limit"] == 100000
        assert result["total_tokens"] == 5000
        assert result["prompt_tokens"] == 4000
        assert result["output_tokens"] == 1000
        assert result["tokens_remaining"] == 95000
        assert result["percent_used"] == 5.0
        assert result["turns"] == 3
        assert result["gc"] is None  # No GC config on mock

    def test_context_aspect_with_gc_config(self):
        """Test that GC config is included when available."""
        plugin = EnvironmentPlugin()

        class MockGCConfig:
            threshold_percent = 80.0
            auto_trigger = True
            preserve_recent_turns = 5
            max_turns = None

        class MockSession:
            _gc_config = MockGCConfig()

            def get_context_usage(self):
                return {
                    "model": "test-model",
                    "context_limit": 100000,
                    "total_tokens": 5000,
                    "prompt_tokens": 4000,
                    "output_tokens": 1000,
                    "tokens_remaining": 95000,
                    "percent_used": 5.0,
                    "turns": 3,
                }

        plugin.set_session(MockSession())
        result = json.loads(plugin._get_environment({"aspect": "context"}))

        assert result["gc"] is not None
        assert result["gc"]["threshold_percent"] == 80.0
        assert result["gc"]["auto_trigger"] is True
        assert result["gc"]["preserve_recent_turns"] == 5
        assert "max_turns" not in result["gc"]  # None is not included

    def test_context_aspect_with_max_turns(self):
        """Test that max_turns is included when set."""
        plugin = EnvironmentPlugin()

        class MockGCConfig:
            threshold_percent = 80.0
            auto_trigger = True
            preserve_recent_turns = 5
            max_turns = 100

        class MockSession:
            _gc_config = MockGCConfig()

            def get_context_usage(self):
                return {
                    "model": "test-model",
                    "context_limit": 100000,
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                    "tokens_remaining": 100000,
                    "percent_used": 0,
                    "turns": 0,
                }

        plugin.set_session(MockSession())
        result = json.loads(plugin._get_environment({"aspect": "context"}))

        assert result["gc"]["max_turns"] == 100


class TestEnvironmentPluginSessionInfo:
    """Tests for session identifier information."""

    def test_session_aspect_without_session(self):
        """Test getting session info without session returns error."""
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "session"}))

        # Without session, should return error
        assert "error" in result
        assert "Session not available" in result["error"]

    def test_session_aspect_with_session(self):
        """Test getting session info with a mocked session."""
        plugin = EnvironmentPlugin()

        class MockSession:
            _agent_id = "main"
            _agent_type = "main"
            _agent_name = None
            _session_plugin = None

        plugin.set_session(MockSession())
        result = json.loads(plugin._get_environment({"aspect": "session"}))

        assert result["agent_id"] == "main"
        assert result["agent_type"] == "main"
        assert "agent_name" not in result  # None is not included
        assert "session_id" not in result  # No session plugin

    def test_session_aspect_with_session_plugin(self):
        """Test getting session info with session plugin providing session ID."""
        plugin = EnvironmentPlugin()

        class MockSessionPlugin:
            def get_current_session_id(self):
                return "20251207_143022"

        class MockSession:
            _agent_id = "main"
            _agent_type = "main"
            _agent_name = None
            _session_plugin = MockSessionPlugin()

        plugin.set_session(MockSession())
        result = json.loads(plugin._get_environment({"aspect": "session"}))

        assert result["session_id"] == "20251207_143022"
        assert result["agent_id"] == "main"
        assert result["agent_type"] == "main"

    def test_session_aspect_with_subagent(self):
        """Test getting session info for a subagent."""
        plugin = EnvironmentPlugin()

        class MockSession:
            _agent_id = "subagent_1"
            _agent_type = "subagent"
            _agent_name = "researcher"
            _session_plugin = None

        plugin.set_session(MockSession())
        result = json.loads(plugin._get_environment({"aspect": "session"}))

        assert result["agent_id"] == "subagent_1"
        assert result["agent_type"] == "subagent"
        assert result["agent_name"] == "researcher"

    def test_session_aspect_with_env_variable(self, monkeypatch):
        """Test that JAATO_SESSION_ID env var is included when set."""
        monkeypatch.setenv("JAATO_SESSION_ID", "env-session-123")
        plugin = EnvironmentPlugin()

        class MockSession:
            _agent_id = "main"
            _agent_type = "main"
            _agent_name = None
            _session_plugin = None

        plugin.set_session(MockSession())
        result = json.loads(plugin._get_environment({"aspect": "session"}))

        assert result["agent_id"] == "main"
        assert result["env_session_id"] == "env-session-123"

    def test_all_aspect_includes_session(self):
        """Test that 'all' includes session key."""
        plugin = EnvironmentPlugin()
        result = json.loads(plugin._get_environment({"aspect": "all"}))

        # Session should be in result (even if it's an error without session)
        assert "session" in result


class TestEnvironmentPluginProtocol:
    """Tests for required protocol methods."""

    def test_get_system_instructions(self):
        plugin = EnvironmentPlugin()
        instructions = plugin.get_system_instructions()

        # Environment plugin doesn't need special instructions
        assert instructions is None

    def test_get_auto_approved_tools(self):
        plugin = EnvironmentPlugin()
        auto_approved = plugin.get_auto_approved_tools()

        # Environment tools are read-only and safe
        assert "get_environment" in auto_approved

    def test_get_user_commands(self):
        plugin = EnvironmentPlugin()
        commands = plugin.get_user_commands()

        # No user commands provided
        assert commands == []
