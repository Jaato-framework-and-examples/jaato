"""Tests for the runner-side session env wiring (PR #92 Y fix).

PR #91 attempted to move workspace ``.env`` reading runner-side
(Shape 3 PR 1 principle: "per-workspace state belongs to the per-
workspace process").  Under AppArmor confinement that path was
broken — the runner can't exec ``pass``, so secret-URI resolution
fails and ``pass://`` survives as a literal in ``os.environ``.

PR #92 (this file) pins the Y shape: the daemon (unconfined)
resolves all session env — including ``${VAR}`` cross-references AND
secret URIs — and ships the fully-resolved dict via
``envelope.session_env``.  The runner applies the dict to
``os.environ`` verbatim during bootstrap, never running its own
resolver.

Pins:
1. ``bootstrap_session`` applies ``envelope.session_env`` to
   ``os.environ`` BEFORE constructing the runtime / configuring
   plugins.
2. The resolved env is attached to the constructed
   :class:`JaatoSession` as ``_session_env`` so plugin code can call
   ``session.get_session_env(key)``.
3. Runner does NOT read ``<workspace>/.env`` itself — daemon owns
   that file read.
4. Envelope.project / envelope.location are used verbatim; runner
   does not fall back to ``os.environ`` lookups (the daemon
   populates the envelope fields when relevant).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pytest

from server.runner.session import bootstrap_session
from shared.session_envelope import SessionInitEnvelope


def _good_envelope(**overrides: Any) -> SessionInitEnvelope:
    base = dict(
        session_id="sess-y",
        workspace_path=None,
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[{"name": "signal_completion", "preload": True}],
    )
    base.update(overrides)
    return SessionInitEnvelope(**base)


class _RecordingSession:
    """Stand-in for JaatoSession that records what was attached."""

    def __init__(self, **kwargs: Any) -> None:
        self.create_session_kwargs = dict(kwargs)
        self._session_env: Dict[str, str] = {}


class _RecordingRuntime:
    """Stand-in for JaatoRuntime; captures connect args + skips Path D."""

    def __init__(self) -> None:
        self.create_session_kwargs: Optional[Dict[str, Any]] = None
        self.connect_args: Optional[tuple] = None
        self._registry = object()  # tells Path D guard to skip
        self.is_connected = False

    def connect(self, project: str, location: str) -> None:
        self.connect_args = (project, location)
        self.is_connected = True

    def create_session(self, **kwargs: Any) -> _RecordingSession:
        self.create_session_kwargs = dict(kwargs)
        return _RecordingSession(**kwargs)


@pytest.fixture
def isolated_env(monkeypatch):
    """Ensure each test starts with a clean ``os.environ`` for the
    keys we touch."""
    keys = [
        "JAATO_INPUTS_DIR", "TEST_Y_KEY_A", "TEST_Y_KEY_B",
        "PROJECT_ID", "LOCATION", "ZHIPUAI_API_KEY",
    ]
    for k in keys:
        monkeypatch.delenv(k, raising=False)
    yield monkeypatch


# ----------------------------------------------------------------------
# 1. envelope.session_env → os.environ
# ----------------------------------------------------------------------


def test_bootstrap_applies_envelope_session_env_to_os_environ(
    tmp_path, isolated_env,
) -> None:
    """Core load-bearing path: daemon-resolved env values reach the
    runner's ``os.environ`` via ``envelope.session_env``.  Closes the
    pool-routing env-propagation bug — pool slots get the env via
    this wire path independent of fork inheritance."""
    runtime = _RecordingRuntime()
    env = _good_envelope(
        workspace_path=str(tmp_path),
        session_env={
            "JAATO_INPUTS_DIR": "/data/cascade",
            "TEST_Y_KEY_A": "value-a",
        },
    )
    host = bootstrap_session(env, runtime_factory=lambda e: runtime)

    assert host.is_ready
    assert os.environ.get("JAATO_INPUTS_DIR") == "/data/cascade"
    assert os.environ.get("TEST_Y_KEY_A") == "value-a"


def test_bootstrap_applies_pre_resolved_secret_value(
    tmp_path, isolated_env,
) -> None:
    """Regression pin for PR #91 → #92: the daemon resolved
    ``pass://jaato/zhipuai/api-key`` to a 49-byte value daemon-side
    and shipped the resolved literal in ``envelope.session_env``.
    The runner applies it verbatim — no resolver discovery, no
    ``pass`` exec (which AppArmor would block)."""
    resolved_key = "a85e05a017a746b98a3df41c864826ea.WHhuugYQUbRanAfM"
    runtime = _RecordingRuntime()
    env = _good_envelope(
        workspace_path=str(tmp_path),
        session_env={"ZHIPUAI_API_KEY": resolved_key},
    )
    bootstrap_session(env, runtime_factory=lambda e: runtime)

    assert os.environ.get("ZHIPUAI_API_KEY") == resolved_key
    # Sanity: not the literal URI shape.
    assert not os.environ["ZHIPUAI_API_KEY"].startswith("pass://")


def test_bootstrap_empty_session_env_is_noop(isolated_env) -> None:
    """Envelope without ``session_env`` (daemon didn't resolve any
    env) doesn't error."""
    runtime = _RecordingRuntime()
    env = _good_envelope(session_env={})
    host = bootstrap_session(env, runtime_factory=lambda e: runtime)

    assert host.is_ready
    assert os.environ.get("JAATO_INPUTS_DIR") is None


# ----------------------------------------------------------------------
# 2. JaatoSession._session_env attribute
# ----------------------------------------------------------------------


def test_bootstrap_attaches_session_env_to_session(
    tmp_path, isolated_env,
) -> None:
    """The applied env is attached to the JaatoSession as
    ``_session_env`` so plugin code can call
    ``session.get_session_env(key)``."""
    runtime = _RecordingRuntime()
    env = _good_envelope(
        workspace_path=str(tmp_path),
        session_env={"TEST_Y_KEY_B": "attached"},
    )
    host = bootstrap_session(env, runtime_factory=lambda e: runtime)

    assert host.session is not None
    assert host.session._session_env.get("TEST_Y_KEY_B") == "attached"


# ----------------------------------------------------------------------
# 3. Runner does NOT read workspace .env itself
# ----------------------------------------------------------------------


def test_bootstrap_does_not_read_workspace_dotenv_runner_side(
    tmp_path, isolated_env,
) -> None:
    """The runner must NOT consume ``<workspace>/.env`` directly.
    Daemon owns that file read.  This pins the architectural shape:
    if a future change makes the runner read the file, it would
    re-introduce the AppArmor exec failure on pass://."""
    (tmp_path / ".env").write_text("JAATO_INPUTS_DIR=should-not-be-read\n")

    runtime = _RecordingRuntime()
    # Empty session_env — daemon "didn't" resolve anything.  If the
    # runner reads workspace .env itself, JAATO_INPUTS_DIR would
    # appear in os.environ.
    env = _good_envelope(
        workspace_path=str(tmp_path),
        session_env={},
    )
    bootstrap_session(env, runtime_factory=lambda e: runtime)

    # The .env file is present but should NOT have been read.
    assert os.environ.get("JAATO_INPUTS_DIR") is None


# ----------------------------------------------------------------------
# 4. runtime.connect uses envelope project/location verbatim
# ----------------------------------------------------------------------


def test_bootstrap_connect_uses_envelope_project(
    tmp_path, isolated_env,
) -> None:
    """The runner uses ``envelope.project`` / ``envelope.location``
    verbatim — no ``os.environ`` fallback runner-side, because the
    daemon now resolves them daemon-side and ships via the
    envelope."""
    runtime = _RecordingRuntime()
    env = _good_envelope(
        workspace_path=str(tmp_path),
        project="my-gcp-project",
        location="us-central1",
    )
    bootstrap_session(env, runtime_factory=lambda e: runtime)

    assert runtime.connect_args == ("my-gcp-project", "us-central1")


def test_bootstrap_connect_passes_empty_when_envelope_empty(
    tmp_path, isolated_env,
) -> None:
    """When daemon didn't carry project/location, runner passes
    empty strings — provider plugins that need them (Vertex AI)
    fail at provider initialize, not silently fall back to a global
    env knob."""
    runtime = _RecordingRuntime()
    env = _good_envelope(
        workspace_path=str(tmp_path),
        project="",
        location="",
    )
    bootstrap_session(env, runtime_factory=lambda e: runtime)

    assert runtime.connect_args == ("", "")
