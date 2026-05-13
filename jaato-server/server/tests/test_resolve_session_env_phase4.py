"""Daemon-side ``_resolve_session_env`` contract (Phase 4 §B + Shape 3 PR 1).

Phase 4 §B pinned: workspace .env values reach the runner via
daemon-side resolution + ``_with_session_env`` ``os.environ`` overlay
inherited by the cold-spawn fork.

Shape 3 PR 1 (2026-05-13) moved workspace .env reading to the
runner-side ``bootstrap_session`` — the per-workspace process where
the file is colocated.  Daemon-side ``_resolve_session_env`` no
longer reads ``<workspace>/.env``; it only resolves profile.env +
env_overrides for the SessionInitEnvelope's wire field.  Workspace
.env reading is pinned in
``server/runner/tests/test_runner_session_workspace_env.py``.

This file's pins now cover only the daemon-side surface that
remains:

1. ``_resolve_session_env`` populates ``self._session_env`` from
   profile.env + env_overrides (NOT workspace .env).
2. ``_resolve_session_env`` is idempotent (second call is a no-op).
3. ``_with_session_env`` applies the resolved (profile.env +
   env_overrides) values to ``os.environ`` for the duration of the
   context — still load-bearing for daemon-side code paths reading
   ``os.environ`` during runtime construction + plugin discovery.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pytest

from server.core import JaatoServer


@pytest.fixture
def env_file(tmp_path):
    """Workspace .env file — present but no longer read daemon-side."""
    f = tmp_path / ".env"
    f.write_text(
        "PHASE4_FOO=bar-resolved\n"
        "PHASE4_BAZ=qux-${PHASE4_FOO}\n"
    )
    return f


class _ProfileWithEnv:
    """Stand-in for a SubagentProfile carrying profile-level env."""

    def __init__(self, env: Dict[str, str]) -> None:
        self.env = env
        # Other attributes that JaatoServer may probe during init.
        self.plugins: List[Any] = []
        self.plugin_configs: Dict[str, Any] = {}
        self.system_instructions: Optional[str] = None
        self.gc = None
        self.completion_payload_schema = None
        self.runtime_limits = None
        self.suppress_base_instructions = False


def test_resolve_session_env_does_not_read_workspace_dotenv(
    env_file, tmp_path,
):
    """Shape 3 PR 1: daemon-side ``_resolve_session_env`` MUST NOT
    populate ``self._session_env`` from ``<workspace>/.env``.  That
    file is read runner-side now (see
    ``test_runner_session_workspace_env.py``).  Daemon reading it
    here was the pre-Shape-3 fork-inheritance mechanism; pool slots
    silently broke that channel, so the resolution moved to the
    per-workspace process where it belongs."""
    server = JaatoServer(env_file=str(env_file), workspace_path=str(tmp_path))
    server._resolve_session_env()
    assert server._session_env_resolved is True
    # The workspace .env keys are NOT in the daemon-side resolved env.
    assert "PHASE4_FOO" not in server._session_env
    assert "PHASE4_BAZ" not in server._session_env


def test_resolve_session_env_populates_from_profile_env(tmp_path):
    """Profile.env values still resolve daemon-side because the
    daemon constructed the SubagentProfile (Shape 3 PR 2 will
    relocate this).  Until then, ``self._session_env`` carries
    profile.env entries.  These also feed the envelope's
    ``env_overrides`` field that the runner applies on top of its
    own workspace .env read."""
    server = JaatoServer(env_file=None, workspace_path=str(tmp_path))
    server._profile = _ProfileWithEnv(env={"PROFILE_KEY": "profile-value"})
    server._resolve_session_env()
    assert server._session_env_resolved is True
    assert server._session_env["PROFILE_KEY"] == "profile-value"


def test_resolve_session_env_populates_from_env_overrides(tmp_path):
    """Post-auth wizard overrides (constructor ``env_overrides=...``)
    have highest precedence — applied last."""
    server = JaatoServer(
        env_file=None,
        workspace_path=str(tmp_path),
        env_overrides={"OVERRIDE_KEY": "override-value"},
    )
    server._resolve_session_env()
    assert server._session_env["OVERRIDE_KEY"] == "override-value"


def test_resolve_session_env_overrides_win_over_profile(tmp_path):
    """env_overrides > profile.env on key conflicts."""
    server = JaatoServer(
        env_file=None,
        workspace_path=str(tmp_path),
        env_overrides={"K": "from-overrides"},
    )
    server._profile = _ProfileWithEnv(env={"K": "from-profile"})
    server._resolve_session_env()
    assert server._session_env["K"] == "from-overrides"


def test_resolve_session_env_is_idempotent(tmp_path):
    """Second call is a no-op."""
    server = JaatoServer(env_file=None, workspace_path=str(tmp_path))
    server._profile = _ProfileWithEnv(env={"K": "v1"})
    server._resolve_session_env()
    # Mutate after first resolve — second resolve should not overwrite.
    server._session_env["K"] = "mutated"
    server._resolve_session_env()
    assert server._session_env["K"] == "mutated"


def test_resolve_session_env_no_env_file(tmp_path):
    """env_file=None path: ``_session_env`` stays {} (no profile, no
    overrides) but flag flips so subsequent calls are no-ops."""
    server = JaatoServer(env_file=None, workspace_path=str(tmp_path))
    server._resolve_session_env()
    assert server._session_env_resolved is True
    assert server._session_env == {}


def test_with_session_env_propagates_profile_env_to_environ(
    tmp_path, monkeypatch,
):
    """``_with_session_env`` applies profile.env (the only daemon-side
    layer that remains workspace-tied) to ``os.environ`` for the
    duration.  Daemon-side runtime construction + plugin discovery
    reads ``os.environ`` inside this context."""
    monkeypatch.delenv("PROFILE_KEY", raising=False)
    server = JaatoServer(env_file=None, workspace_path=str(tmp_path))
    server._profile = _ProfileWithEnv(env={"PROFILE_KEY": "in-context"})
    server._resolve_session_env()

    with server._with_session_env():
        assert os.environ["PROFILE_KEY"] == "in-context"
    assert "PROFILE_KEY" not in os.environ


def test_with_session_env_restores_previous_value(tmp_path, monkeypatch):
    """If ``os.environ`` had a different value before the with-block,
    exit restores the previous value rather than deleting."""
    monkeypatch.setenv("PROFILE_KEY", "pre-existing-value")
    server = JaatoServer(env_file=None, workspace_path=str(tmp_path))
    server._profile = _ProfileWithEnv(env={"PROFILE_KEY": "in-context"})
    server._resolve_session_env()

    with server._with_session_env():
        assert os.environ["PROFILE_KEY"] == "in-context"
    assert os.environ["PROFILE_KEY"] == "pre-existing-value"
