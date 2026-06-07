"""Tests for the §3.12 ephemeral path migration to
``_construct_and_initialize_server``.

The ephemeral path used to construct ``JaatoServer()`` and call
``server.initialize(model=..., provider_name=..., tools=...,
system_instructions=...)`` — but ``JaatoServer.initialize`` takes
no kwargs, so that signature was effectively dead code.  Phase 3
§3.12 ephemeral migration routes through the unified
``_construct_and_initialize_server`` sub-helper by composing the
ephemeral inputs into an inline ``SubagentProfile`` (via
``shared.plugins.subagent.config.build_inline_profile``).

These tests pin the migration:

- The ephemeral impl invokes the sub-helper with a
  ``BootstrapEnvelope`` carrying an inline profile composed from
  the merged profile/inline data.
- ``client_id=None`` (no client-driven apparmor opt-in on
  ephemeral fan-out).
- Workspace context (chdir + ``JAATO_WORKSPACE_ROOT``) is
  preserved when ``workspace_path`` is set.
- Init failure surfaces as a hint string rather than crashing the
  caller.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

from shared.session_envelope import BootstrapEnvelope


class _FakeServer:
    def __init__(self) -> None:
        self.shutdown_called = False
        self.send_message_calls: List[Tuple[str, Any]] = []

    def send_message(self, prompt: str, on_output: Any = None) -> str:
        self.send_message_calls.append((prompt, on_output))
        if on_output:
            on_output("model", "ephemeral response", "text")
        return "ephemeral response"

    def shutdown(self) -> None:
        self.shutdown_called = True

    @property
    def registry(self) -> Any:
        return None


def _make_session_manager():
    """Construct a SessionManager with the sub-helper patched to
    capture the BootstrapEnvelope and return a fake server."""
    from server.session_manager import SessionManager

    sm = SessionManager()
    captured: List[BootstrapEnvelope] = []
    fake_server = _FakeServer()

    def _stub_subhelper(self: Any, envelope: BootstrapEnvelope):
        captured.append(envelope)
        return fake_server, None

    sm._construct_and_initialize_server = _stub_subhelper.__get__(sm, type(sm))  # type: ignore[method-assign]
    return sm, captured, fake_server


# ----------------------------------------------------------------------
# Envelope construction
# ----------------------------------------------------------------------


def test_ephemeral_routes_through_sub_helper() -> None:
    sm, captured, fake_server = _make_session_manager()

    profile_json = json.dumps({
        "model": "claude-sonnet-4-5",
        "provider": "anthropic",
        "plugins": ["cli", "todo"],
        "system_instructions": "You are an ephemeral agent.",
    })
    inline_json = ""

    result = sm._run_ephemeral_session_impl(
        profile_json=profile_json,
        inline_config_json=inline_json,
        prompt="hello",
        agent_name="ephemeral-1",
        on_output=None,
        workspace_path=None,
    )

    assert len(captured) == 1
    envelope = captured[0]
    assert envelope.client_id is None
    assert envelope.profile is not None
    # The inline profile composed from the merged data carries the
    # model + provider + plugins.
    assert envelope.profile.model == "claude-sonnet-4-5"
    assert envelope.profile.provider == "anthropic"
    assert "cli" in envelope.profile.plugins
    assert "todo" in envelope.profile.plugins
    assert envelope.agent_name == "ephemeral-1"
    # Result is whatever the fake server returned.
    assert result == "ephemeral response"
    assert fake_server.shutdown_called is True


def test_ephemeral_inline_data_overrides_profile_data_partially() -> None:
    """Merge order: ``profile_data`` setdefaults first, then
    ``inline_data`` only fills missing keys.  Tests this precedence
    by setting model only in inline_data."""
    sm, captured, _ = _make_session_manager()

    profile_json = json.dumps({"plugins": ["cli"]})
    inline_json = json.dumps({"model": "claude-haiku-4-5"})

    sm._run_ephemeral_session_impl(
        profile_json=profile_json,
        inline_config_json=inline_json,
        prompt="x",
        agent_name="merge-test",
        on_output=None,
    )

    envelope = captured[0]
    assert envelope.profile.model == "claude-haiku-4-5"
    assert "cli" in envelope.profile.plugins


def test_ephemeral_session_id_is_unique() -> None:
    sm, captured, _ = _make_session_manager()

    sm._run_ephemeral_session_impl(
        profile_json=json.dumps({"plugins": []}), inline_config_json="",
        prompt="x", agent_name="a", on_output=None,
    )
    sm._run_ephemeral_session_impl(
        profile_json=json.dumps({"plugins": []}), inline_config_json="",
        prompt="x", agent_name="b", on_output=None,
    )

    sids = [env.session_id for env in captured]
    assert len(set(sids)) == 2  # each run gets a fresh ephemeral-* id
    assert all(sid.startswith("ephemeral-") for sid in sids)


# ----------------------------------------------------------------------
# Workspace context preservation
# ----------------------------------------------------------------------


def test_ephemeral_chdir_restored_after_run(tmp_path) -> None:
    """The chdir to workspace_path must be reverted after the
    ephemeral run, even if it succeeds."""
    sm, _, _ = _make_session_manager()

    original_cwd = os.getcwd()
    sub = tmp_path / "ws"
    sub.mkdir()

    sm._run_ephemeral_session_impl(
        profile_json=json.dumps({"plugins": []}), inline_config_json="",
        prompt="x", agent_name="ws-test", on_output=None,
        workspace_path=str(sub),
    )

    assert os.getcwd() == original_cwd


def test_ephemeral_workspace_root_env_restored(tmp_path) -> None:
    """``JAATO_WORKSPACE_ROOT`` is restored to its prior value (or
    removed if unset before)."""
    sm, _, _ = _make_session_manager()

    sub = tmp_path / "ws"
    sub.mkdir()

    # Snapshot env, ensure JAATO_WORKSPACE_ROOT is unset.
    original = os.environ.pop("JAATO_WORKSPACE_ROOT", None)
    try:
        sm._run_ephemeral_session_impl(
            profile_json=json.dumps({"plugins": []}), inline_config_json="",
            prompt="x", agent_name="ws-test", on_output=None,
            workspace_path=str(sub),
        )
        # Must be unset again (matched the prior unset state).
        assert "JAATO_WORKSPACE_ROOT" not in os.environ
    finally:
        if original is not None:
            os.environ["JAATO_WORKSPACE_ROOT"] = original


# ----------------------------------------------------------------------
# Init failure path
# ----------------------------------------------------------------------


def test_ephemeral_init_failure_returns_hint_string() -> None:
    """When the sub-helper returns ``(None, None)`` the ephemeral
    impl must surface a hint rather than crashing."""
    from server.session_manager import SessionManager

    sm = SessionManager()

    def _failing_subhelper(self: Any, envelope: BootstrapEnvelope):
        return None, None

    sm._construct_and_initialize_server = _failing_subhelper.__get__(sm, type(sm))  # type: ignore[method-assign]

    result = sm._run_ephemeral_session_impl(
        profile_json=json.dumps({"plugins": []}), inline_config_json="",
        prompt="x", agent_name="fail", on_output=None,
    )
    assert "failed" in result.lower()
