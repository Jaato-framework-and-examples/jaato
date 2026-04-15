"""Tests for the ``--instructions`` / ``--no-instructions`` flags on session.new.

Covers the full plumbing the IPC path uses to fit a session into a small
context window:

  command_router._handle_session_new (parse flags)
       ↓ system_instruction_override
  SessionManager.create_session
       ↓
  JaatoServer(__init__, _build_profile_session_kwargs)
       ↓ kwargs["system_instruction_override"]
  JaatoRuntime.create_session → JaatoSession.configure(...)

Each test isolates one hop with mocks so a failure points at the layer
that broke the contract.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ──────────────────────────────────────────────────────────────────
# command_router parsing
# ──────────────────────────────────────────────────────────────────


@pytest.fixture
def router():
    """Build a router with mocks for everything except the args-parsing logic."""
    from server.command_router import CommandRouter

    sm = MagicMock()
    sm.create_session.return_value = "20260415_120000"
    sm.get_client_session.return_value = MagicMock(server=MagicMock(get_all_session_env=lambda: {}))

    sink = MagicMock()
    sink.get_client_user.return_value = "tester"

    r = CommandRouter(
        session_manager=sm,
        event_sink=sink,
        daemon_plugins={},
    )
    return r, sm


class TestSessionNewParsing:

    def test_no_instruction_flag_means_no_override(self, router):
        r, sm = router
        r._handle_session_new("client1", ["my-session"], workspace_path="/tmp/ws")
        # No --instructions / --no-instructions → override is None
        assert sm.create_session.call_args.kwargs["system_instruction_override"] is None

    def test_no_instructions_flag_resolves_to_empty_string(self, router):
        r, sm = router
        r._handle_session_new("client1", ["--no-instructions"], workspace_path="/tmp/ws")
        # Empty string is the trigger that "send no system message at all"
        assert sm.create_session.call_args.kwargs["system_instruction_override"] == ""

    def test_instructions_with_literal_text(self, router):
        r, sm = router
        r._handle_session_new(
            "client1",
            ["--instructions", "You are terse."],
            workspace_path="/tmp/ws",
        )
        assert sm.create_session.call_args.kwargs["system_instruction_override"] == "You are terse."

    def test_instructions_at_path_reads_file(self, router, tmp_path):
        r, sm = router
        prompt_file = tmp_path / "min.md"
        prompt_file.write_text("Brief: ship the code.\n")

        # Absolute path: works regardless of workspace
        r._handle_session_new(
            "client1",
            ["--instructions", f"@{prompt_file}"],
            workspace_path="/tmp/some-ws",
        )
        # Trailing newline is stripped so the override isn't accidentally
        # double-newlined when the runtime re-emits.
        assert sm.create_session.call_args.kwargs["system_instruction_override"] == "Brief: ship the code."

    def test_instructions_at_relative_path_resolves_against_workspace(self, router, tmp_path):
        r, sm = router
        prompt_file = tmp_path / "prompts" / "x.md"
        prompt_file.parent.mkdir(parents=True)
        prompt_file.write_text("relative-path test")

        r._handle_session_new(
            "client1",
            ["--instructions", "@prompts/x.md"],
            workspace_path=str(tmp_path),
        )
        assert sm.create_session.call_args.kwargs["system_instruction_override"] == "relative-path test"

    def test_instructions_missing_value_emits_error_and_aborts(self, router):
        r, sm = router
        # Trailing --instructions with no value → error, no session created
        r._handle_session_new("client1", ["--instructions"], workspace_path="/tmp/ws")
        sm.create_session.assert_not_called()
        # Error event was emitted to the client
        emit_calls = [
            c for c in r._event_sink.method_calls
            if c[0] == "_emit_to_client" or "ErrorEvent" in str(c)
        ]
        assert any("ErrorEvent" in str(c) or "UsageError" in str(c) or "--instructions" in str(c)
                   for c in r._event_sink.method_calls)

    def test_instructions_at_nonexistent_path_emits_error_and_aborts(self, router, tmp_path):
        r, sm = router
        missing = tmp_path / "does-not-exist.md"
        r._handle_session_new(
            "client1",
            ["--instructions", f"@{missing}"],
            workspace_path=str(tmp_path),
        )
        sm.create_session.assert_not_called()


# ──────────────────────────────────────────────────────────────────
# JaatoServer kwargs assembly
# ──────────────────────────────────────────────────────────────────


class TestBuildProfileSessionKwargsOverride:
    """``_build_profile_session_kwargs`` must surface the override even when
    no profile is in play, and it must override profile.system_instructions
    when both are present."""

    def _make_server(self, *, profile=None, override=None):
        from server.core import JaatoServer

        # Construct minimally — we only exercise _build_profile_session_kwargs
        # which doesn't need plugin discovery, network, or the runtime.
        return JaatoServer(
            env_file=".env",
            profile=profile,
            system_instruction_override=override,
        )

    def test_no_profile_no_override_returns_none(self):
        server = self._make_server()
        assert server._build_profile_session_kwargs() is None

    def test_no_profile_with_override_returns_kwargs_with_override(self):
        server = self._make_server(override="")
        kwargs = server._build_profile_session_kwargs()
        assert kwargs == {"system_instruction_override": ""}

    def test_override_wins_against_profile_system_instructions(self):
        # Synthetic profile carrying the deprecated system_instructions field
        profile = MagicMock()
        profile.plugins = []
        profile.preloaded_plugins = []
        profile.plugin_configs = {}
        profile.system_instructions = "FROM PROFILE"
        profile.provider = None

        server = self._make_server(profile=profile, override="FROM CLI")
        kwargs = server._build_profile_session_kwargs()
        # Both keys are present (system_instructions reaches the runtime as
        # "additional"; the override replaces the assembled output entirely).
        # Order matters: override must be set so the session.configure path
        # picks it up.
        assert kwargs["system_instructions"] == "FROM PROFILE"
        assert kwargs["system_instruction_override"] == "FROM CLI"

    def test_override_none_omits_the_key(self):
        """``None`` means "no override" — must not appear in kwargs at all
        so JaatoSession.configure's default behaviour kicks in."""
        profile = MagicMock()
        profile.plugins = []
        profile.preloaded_plugins = []
        profile.plugin_configs = {}
        profile.system_instructions = None
        profile.provider = "anthropic"

        server = self._make_server(profile=profile, override=None)
        kwargs = server._build_profile_session_kwargs()
        assert "system_instruction_override" not in kwargs
        assert kwargs.get("provider_name") == "anthropic"
