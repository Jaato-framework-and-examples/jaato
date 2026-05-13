"""Pin the profile-less envelope-construction fallback (PR 100).

PR 100 closes the regression where ``jaato --new-session`` without a
profile failed with ``envelope.model_name is empty`` post-§7c (commit
6406fe35, 2026-05-09).  ``build_session_envelope`` now reads
``MODEL_NAME`` / ``JAATO_PROVIDER`` from the daemon's resolved
session env (workspace ``.env`` + profile.env + env_overrides) when
the profile is absent or doesn't supply them.

Also removes the pre-existing hardcoded ``provider_name="anthropic"``
fallback — per the user's standing "no hardcoded fallbacks" rule
(global CLAUDE.md).

Contract pinned by these tests:

1. Profile present + sets model/provider → envelope uses profile values.
2. Profile absent + workspace .env sets MODEL_NAME / JAATO_PROVIDER
   → envelope uses those values.
3. Profile absent + .env silent → envelope's model_name/provider_name
   stay empty.  Runner-side validation catches this audibly via
   ``BootstrapError(stage="validate")``.
4. Profile partial (sets model but not provider) + .env supplements
   → profile wins for model, .env wins for provider.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from server.runner_spawn import build_session_envelope


class _FakeProfile:
    """Stand-in for SubagentProfile."""

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.env = env or {}
        self.plugins: list = []
        self.plugin_configs: dict = {}
        self.preloaded_plugins: set = set()
        self.system_instructions: Optional[str] = None
        self.gc = None
        self.completion_payload_schema = None


class _FakeServer:
    """Stand-in for JaatoServer."""

    def __init__(
        self,
        profile: Optional[_FakeProfile] = None,
        session_env: Optional[Dict[str, str]] = None,
    ) -> None:
        self._profile = profile
        self._session_env = session_env or {}
        self._agent_params: Dict[str, str] = {}
        self._main_agent_id = "main"
        self.config_root: Optional[str] = None


# ----------------------------------------------------------------------
# Profile wins
# ----------------------------------------------------------------------


def test_profile_model_and_provider_win() -> None:
    """When the profile sets model + provider, envelope uses those
    regardless of session_env."""
    profile = _FakeProfile(model="claude-sonnet-4-7", provider="anthropic")
    server = _FakeServer(
        profile=profile,
        session_env={
            "MODEL_NAME": "gpt-4-from-env",
            "JAATO_PROVIDER": "openai",
        },
    )
    env = build_session_envelope(
        server=server,
        session_id="sess-profile-wins",
        workspace_path="/tmp/ws",
        profile_name="my-profile",
    )
    assert env.model_name == "claude-sonnet-4-7"
    assert env.provider_name == "anthropic"


# ----------------------------------------------------------------------
# Profile absent, session_env supplies
# ----------------------------------------------------------------------


def test_profile_absent_session_env_supplies_model_and_provider() -> None:
    """Profile-less session + workspace ``.env`` carries MODEL_NAME
    + JAATO_PROVIDER → envelope picks them up.

    This is the load-bearing fix for the profile-less
    ``jaato --new-session`` regression."""
    server = _FakeServer(
        profile=None,
        session_env={
            "MODEL_NAME": "claude-sonnet-4-7",
            "JAATO_PROVIDER": "anthropic",
        },
    )
    env = build_session_envelope(
        server=server,
        session_id="sess-env-supplies",
        workspace_path="/tmp/ws",
        profile_name="",
    )
    assert env.model_name == "claude-sonnet-4-7"
    assert env.provider_name == "anthropic"


def test_profile_absent_session_env_supplies_provider_only() -> None:
    """Common case: workspace .env has JAATO_PROVIDER but no MODEL_NAME
    (model picked from provider's default).  Envelope honors that."""
    server = _FakeServer(
        profile=None,
        session_env={"JAATO_PROVIDER": "openrouter"},
    )
    env = build_session_envelope(
        server=server,
        session_id="sess-provider-only",
        workspace_path="/tmp/ws",
        profile_name="",
    )
    assert env.model_name == ""  # absent; runner-side validation catches
    assert env.provider_name == "openrouter"


# ----------------------------------------------------------------------
# Both silent → empty (no hardcoded fallback)
# ----------------------------------------------------------------------


def test_profile_and_env_both_silent_leaves_fields_empty() -> None:
    """When neither profile nor session_env declares model/provider,
    the envelope's fields stay empty.  No hardcoded default.  Runner-
    side ``_validate_envelope`` will raise ``BootstrapError`` —
    loud failure beats silent guess.

    Pre-PR-100 the provider had a silent ``"anthropic"`` default;
    this test pins the removal."""
    server = _FakeServer(profile=None, session_env={})
    env = build_session_envelope(
        server=server,
        session_id="sess-both-silent",
        workspace_path="/tmp/ws",
        profile_name="",
    )
    assert env.model_name == ""
    assert env.provider_name == ""


def test_profile_silent_env_silent_model_provider_stays_empty() -> None:
    """Even with workspace_path set + everything wired, an
    intentionally empty session_env doesn't fall back to anthropic
    via a hardcoded default.  Mirrors today's user-facing situation
    where the .env is missing keys."""
    server = _FakeServer(profile=None, session_env={"SOME_OTHER_VAR": "x"})
    env = build_session_envelope(
        server=server,
        session_id="sess-no-model",
        workspace_path="/tmp/ws",
        profile_name="",
    )
    assert env.model_name == ""
    assert env.provider_name == ""


# ----------------------------------------------------------------------
# Mixed: profile partial + session_env supplements
# ----------------------------------------------------------------------


def test_profile_sets_model_only_env_supplies_provider() -> None:
    """Profile declares model but not provider; workspace .env has
    JAATO_PROVIDER → envelope combines them.  Each field's fallback
    is independent."""
    profile = _FakeProfile(model="claude-sonnet-4-7", provider=None)
    server = _FakeServer(
        profile=profile,
        session_env={"JAATO_PROVIDER": "anthropic"},
    )
    env = build_session_envelope(
        server=server,
        session_id="sess-mixed-1",
        workspace_path="/tmp/ws",
        profile_name="my-profile",
    )
    assert env.model_name == "claude-sonnet-4-7"
    assert env.provider_name == "anthropic"


def test_profile_sets_provider_only_env_supplies_model() -> None:
    """Inverse: profile sets provider, .env supplies MODEL_NAME."""
    profile = _FakeProfile(model=None, provider="anthropic")
    server = _FakeServer(
        profile=profile,
        session_env={"MODEL_NAME": "claude-haiku-4-5"},
    )
    env = build_session_envelope(
        server=server,
        session_id="sess-mixed-2",
        workspace_path="/tmp/ws",
        profile_name="my-profile",
    )
    assert env.model_name == "claude-haiku-4-5"
    assert env.provider_name == "anthropic"
