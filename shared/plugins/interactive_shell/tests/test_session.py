"""Tests for ShellSession — the pexpect wrapper with idle detection."""

import time

import pytest

from shared.plugins.interactive_shell.session import ShellSession


class TestShellSessionLifecycle:
    """Test session creation, interaction, and cleanup."""

    def test_spawn_and_read_initial_output(self):
        """Spawning a shell produces initial output (prompt)."""
        session = ShellSession(
            command="bash --norc --noprofile",
            session_id="test_bash",
            idle_timeout=0.3,
        )
        try:
            output = session.read_initial_output()
            assert session.is_alive
            # bash with --norc may or may not produce a prompt, but
            # the session should be alive
        finally:
            session.close()

    def test_spawn_echo_command(self):
        """Spawning a simple echo captures its output."""
        session = ShellSession(
            command="echo hello_world",
            session_id="test_echo",
            idle_timeout=0.3,
        )
        try:
            output = session.read_initial_output()
            assert "hello_world" in output
        finally:
            session.close()

    def test_send_input_and_get_response(self):
        """Send a command to bash and get output back."""
        session = ShellSession(
            command="bash --norc --noprofile",
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
            command="bash --norc --noprofile",
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
            command="bash --norc --noprofile",
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

    def test_close_idempotent(self):
        """Closing an already-closed session doesn't error."""
        session = ShellSession(
            command="echo done",
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
            command="bash --norc --noprofile",
            session_id="test_ctrlc",
            idle_timeout=0.3,
        )
        try:
            session.read_initial_output()

            # Start a sleep, then interrupt it
            session.send_input("sleep 60\n")
            time.sleep(0.2)

            output = session.send_control("c-c")
            # After Ctrl+C, bash should give us a prompt back
            assert session.is_alive
        finally:
            session.close()

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
            command="bash --norc --noprofile",
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
            command="bash --norc --noprofile",
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
            command="bash --norc --noprofile",
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
            command="bash --norc --noprofile",
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
            command="bash --norc --noprofile",
            session_id="test_cwd",
            idle_timeout=0.3,
            cwd=str(tmp_path),
        )
        try:
            session.read_initial_output()
            output = session.send_input("pwd\n")
            assert str(tmp_path) in output
        finally:
            session.close()


class TestShellSessionReadOutput:
    """Test non-blocking output reading."""

    def test_read_with_no_pending_output(self):
        """Reading when nothing is pending returns empty-ish output."""
        session = ShellSession(
            command="bash --norc --noprofile",
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
