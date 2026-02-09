"""Tests for ShellSession — the pexpect/wexpect wrapper with idle detection."""

import sys
import time

import pytest

from shared.plugins.interactive_shell.session import ShellSession

# Helpers for cross-platform test commands
IS_WINDOWS = sys.platform == "win32"

# Use platform-appropriate shell for tests
if IS_WINDOWS:
    SHELL_CMD = "cmd.exe /Q"     # /Q disables echo
    ECHO_CMD = "echo hello_world"
    EXIT_CMD = "exit"
else:
    SHELL_CMD = "bash --norc --noprofile"
    ECHO_CMD = "echo hello_world"
    EXIT_CMD = "exit"


class TestShellSessionLifecycle:
    """Test session creation, interaction, and cleanup."""

    def test_spawn_and_read_initial_output(self):
        """Spawning a shell produces initial output (prompt)."""
        session = ShellSession(
            command=SHELL_CMD,
            session_id="test_bash",
            idle_timeout=0.3,
        )
        try:
            output = session.read_initial_output()
            assert session.is_alive
            # Shell with minimal config may or may not produce a prompt,
            # but the session should be alive
        finally:
            session.close()

    def test_spawn_echo_command(self):
        """Spawning a simple echo captures its output."""
        if IS_WINDOWS:
            command = "cmd.exe /C echo hello_world"
        else:
            command = "echo hello_world"
        session = ShellSession(
            command=command,
            session_id="test_echo",
            idle_timeout=0.3,
        )
        try:
            output = session.read_initial_output()
            assert "hello_world" in output
        finally:
            session.close()

    def test_send_input_and_get_response(self):
        """Send a command to the shell and get output back."""
        session = ShellSession(
            command=SHELL_CMD,
            session_id="test_input",
            idle_timeout=0.3,
        )
        try:
            session.read_initial_output()

            output = session.send_input("echo test_response_42\n")
            assert "test_response_42" in output
        finally:
            session.close()

    def test_multiple_interactions(self):
        """Multiple send/receive cycles work correctly."""
        session = ShellSession(
            command=SHELL_CMD,
            session_id="test_multi",
            idle_timeout=0.3,
        )
        try:
            session.read_initial_output()

            out1 = session.send_input("echo first\n")
            assert "first" in out1

            out2 = session.send_input("echo second\n")
            assert "second" in out2

            out3 = session.send_input("echo third\n")
            assert "third" in out3
        finally:
            session.close()

    def test_is_alive_after_exit(self):
        """Session reports not alive after process exits."""
        session = ShellSession(
            command=SHELL_CMD,
            session_id="test_exit",
            idle_timeout=0.3,
        )
        try:
            session.read_initial_output()
            assert session.is_alive

            session.send_input("exit\n")
            # Give process a moment to exit
            time.sleep(0.5)
            assert not session.is_alive
        finally:
            session.close()

    @pytest.mark.skipif(IS_WINDOWS, reason="exit-code syntax differs on Windows")
    def test_close_returns_exit_status(self):
        """Closing a session returns exit status."""
        session = ShellSession(
            command="bash --norc --noprofile -c 'exit 42'",
            session_id="test_status",
            idle_timeout=0.3,
        )
        session.read_initial_output()
        result = session.close()
        assert result['exit_status'] == 42

    @pytest.mark.skipif(not IS_WINDOWS, reason="Windows-only test")
    def test_close_returns_exit_status_windows(self):
        """Closing a session returns exit status on Windows."""
        session = ShellSession(
            command="cmd.exe /C exit 42",
            session_id="test_status_win",
            idle_timeout=0.3,
        )
        session.read_initial_output()
        result = session.close()
        assert result['exit_status'] == 42

    def test_close_idempotent(self):
        """Closing an already-closed session doesn't error."""
        if IS_WINDOWS:
            command = "cmd.exe /C echo done"
        else:
            command = "echo done"
        session = ShellSession(
            command=command,
            session_id="test_idem",
            idle_timeout=0.3,
        )
        session.read_initial_output()
        result1 = session.close()
        # Second close should not raise
        result2 = session.close()
        assert isinstance(result2, dict)


class TestShellSessionControl:
    """Test control character sending."""

    def test_ctrl_c_interrupts(self):
        """Ctrl+C interrupts a running command."""
        session = ShellSession(
            command=SHELL_CMD,
            session_id="test_ctrlc",
            idle_timeout=0.3,
        )
        try:
            session.read_initial_output()

            # Start a long-running command, then interrupt it
            if IS_WINDOWS:
                session.send_input("ping -n 60 127.0.0.1\n")
            else:
                session.send_input("sleep 60\n")
            time.sleep(0.2)

            output = session.send_control("c-c")
            # After Ctrl+C, the shell should give us a prompt back
            assert session.is_alive
        finally:
            session.close()

    @pytest.mark.skipif(IS_WINDOWS, reason="cat not available on Windows")
    def test_ctrl_d_sends_eof(self):
        """Ctrl+D sends EOF, which exits many programs."""
        session = ShellSession(
            command="cat",
            session_id="test_ctrld",
            idle_timeout=0.3,
        )
        try:
            session.read_initial_output()
            session.send_control("c-d")
            time.sleep(0.5)
            assert not session.is_alive
        finally:
            session.close()

    def test_invalid_control_key(self):
        """Unknown control key raises ValueError."""
        session = ShellSession(
            command=SHELL_CMD,
            session_id="test_badkey",
            idle_timeout=0.3,
        )
        try:
            session.read_initial_output()
            with pytest.raises(ValueError, match="Unknown control key"):
                session.send_control("c-x")
        finally:
            session.close()


class TestShellSessionTimers:
    """Test age, idle, and expiry tracking."""

    def test_age_increases(self):
        """Session age increases over time."""
        session = ShellSession(
            command=SHELL_CMD,
            session_id="test_age",
            idle_timeout=0.3,
        )
        try:
            session.read_initial_output()
            age1 = session.age_seconds
            time.sleep(0.2)
            age2 = session.age_seconds
            assert age2 > age1
        finally:
            session.close()

    def test_is_expired(self):
        """Session with very short lifetime expires quickly."""
        session = ShellSession(
            command=SHELL_CMD,
            session_id="test_expired",
            idle_timeout=0.3,
            max_lifetime=0.1,  # 100ms lifetime
        )
        try:
            session.read_initial_output()
            time.sleep(0.2)
            assert session.is_expired
        finally:
            session.close()

    def test_interaction_resets_idle(self):
        """Sending input resets the idle timer."""
        session = ShellSession(
            command=SHELL_CMD,
            session_id="test_idle",
            idle_timeout=0.3,
        )
        try:
            session.read_initial_output()
            time.sleep(0.3)
            idle_before = session.idle_seconds

            session.send_input("echo hi\n")
            idle_after = session.idle_seconds

            assert idle_after < idle_before
        finally:
            session.close()


class TestShellSessionCwd:
    """Test working directory support."""

    def test_cwd_is_respected(self, tmp_path):
        """Session starts in the specified working directory."""
        session = ShellSession(
            command=SHELL_CMD,
            session_id="test_cwd",
            idle_timeout=0.3,
            cwd=str(tmp_path),
        )
        try:
            session.read_initial_output()
            if IS_WINDOWS:
                output = session.send_input("cd\n")
            else:
                output = session.send_input("pwd\n")
            assert str(tmp_path) in output
        finally:
            session.close()


class TestShellSessionReadOutput:
    """Test non-blocking output reading."""

    def test_read_with_no_pending_output(self):
        """Reading when nothing is pending returns empty-ish output."""
        session = ShellSession(
            command=SHELL_CMD,
            session_id="test_read_empty",
            idle_timeout=0.3,
        )
        try:
            session.read_initial_output()
            # Nothing new to read
            output = session.read_output(timeout=0.2)
            # Could be empty or just a prompt — shouldn't error
            assert isinstance(output, str)
        finally:
            session.close()
