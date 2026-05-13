"""Tests for Shape 3 PR 1 — runner-side workspace ``.env`` reading.

Pins:
1. ``bootstrap_session`` reads ``<workspace_path>/.env`` and applies
   the resolved key-value pairs to ``os.environ`` BEFORE constructing
   the runtime / configuring plugins.  Closes the pool-routing env-
   propagation bug surfaced by v63 cascade smoke.
2. The resolved env is attached to the constructed
   :class:`JaatoSession` as ``_session_env`` so plugin code can call
   ``session.get_session_env(key)``.
3. ``envelope.env_overrides`` (profile.env + post-auth wizard
   overrides) layer on top of workspace ``.env``.
4. ``PROJECT_ID`` / ``LOCATION`` fall back to ``os.environ`` (i.e.
   the workspace .env values applied in step 1b) when the envelope
   doesn't carry them — closes the daemon-side dependency on
   provider-specific env knobs that Shape 3 PR 1 removes.

End-to-end relevance: pre-PR-1 the daemon read workspace .env and
fork-inherited the values into cold-spawned runners via
``_with_session_env`` ``os.environ`` overlay.  Pool slots were
forked from the template at daemon startup so they never saw the
overlay → prefetch scripts reading ``os.environ["JAATO_INPUTS_DIR"]``
crashed.  Post-PR-1 the runner reads workspace .env itself and
populates ``os.environ`` from the per-workspace process, independent
of which spawn path delivered the runner.
"""

from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, Optional

import pytest

from server.runner.session import bootstrap_session
from shared.session_envelope import SessionInitEnvelope


def _good_envelope(**overrides: Any) -> SessionInitEnvelope:
    base = dict(
        session_id="sess-shape3",
        workspace_path=None,  # tests override with the tmp workspace
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
    keys we touch.  Pre-existing values would mask the assertion that
    bootstrap put them there."""
    keys = [
        "JAATO_INPUTS_DIR", "TEST_S3PR1_KEY_A", "TEST_S3PR1_KEY_B",
        "TEST_S3PR1_CROSSREF", "PROJECT_ID", "LOCATION",
    ]
    for k in keys:
        monkeypatch.delenv(k, raising=False)
    yield monkeypatch


# ----------------------------------------------------------------------
# 1. Workspace .env reading
# ----------------------------------------------------------------------


def test_bootstrap_reads_workspace_dotenv_into_os_environ(
    tmp_path, isolated_env,
) -> None:
    """Core load-bearing fix: workspace ``.env`` values reach
    ``os.environ`` via runner-side bootstrap_session."""
    (tmp_path / ".env").write_text(textwrap.dedent("""\
        JAATO_INPUTS_DIR=/data/cascade
        TEST_S3PR1_KEY_A=value-a
    """))

    runtime = _RecordingRuntime()
    env = _good_envelope(workspace_path=str(tmp_path))
    host = bootstrap_session(env, runtime_factory=lambda e: runtime)

    assert host.is_ready
    assert os.environ.get("JAATO_INPUTS_DIR") == "/data/cascade"
    assert os.environ.get("TEST_S3PR1_KEY_A") == "value-a"


def test_bootstrap_no_workspace_path_is_noop(isolated_env) -> None:
    """Headless sessions (no workspace) skip the env read cleanly."""
    runtime = _RecordingRuntime()
    env = _good_envelope(workspace_path=None)
    host = bootstrap_session(env, runtime_factory=lambda e: runtime)

    assert host.is_ready
    # Nothing was set because there's nothing to read.
    assert os.environ.get("JAATO_INPUTS_DIR") is None


def test_bootstrap_missing_dotenv_file_is_noop(
    tmp_path, isolated_env,
) -> None:
    """Workspace without a ``.env`` file doesn't error."""
    runtime = _RecordingRuntime()
    env = _good_envelope(workspace_path=str(tmp_path))
    host = bootstrap_session(env, runtime_factory=lambda e: runtime)

    assert host.is_ready
    assert os.environ.get("JAATO_INPUTS_DIR") is None


def test_bootstrap_expands_dotenv_crossrefs(
    tmp_path, isolated_env,
) -> None:
    """``${VAR}`` cross-references within ``.env`` resolve against
    sibling entries — mirrors daemon-side ``_resolve_session_env``
    behavior so the runner produces the same resolved values."""
    (tmp_path / ".env").write_text(textwrap.dedent("""\
        TEST_S3PR1_KEY_A=base-value
        TEST_S3PR1_CROSSREF=${TEST_S3PR1_KEY_A}/suffix
    """))

    runtime = _RecordingRuntime()
    env = _good_envelope(workspace_path=str(tmp_path))
    bootstrap_session(env, runtime_factory=lambda e: runtime)

    assert os.environ.get("TEST_S3PR1_CROSSREF") == "base-value/suffix"


# ----------------------------------------------------------------------
# 2. Session attribute
# ----------------------------------------------------------------------


def test_bootstrap_attaches_session_env_to_session(
    tmp_path, isolated_env,
) -> None:
    """The resolved env is attached to the JaatoSession as
    ``_session_env`` so plugin code can call
    ``session.get_session_env(key)``."""
    (tmp_path / ".env").write_text("TEST_S3PR1_KEY_B=attached")

    runtime = _RecordingRuntime()
    env = _good_envelope(workspace_path=str(tmp_path))
    host = bootstrap_session(env, runtime_factory=lambda e: runtime)

    assert host.session is not None
    assert host.session._session_env.get("TEST_S3PR1_KEY_B") == "attached"


# ----------------------------------------------------------------------
# 3. envelope.env_overrides layer
# ----------------------------------------------------------------------


def test_bootstrap_applies_envelope_env_overrides(
    tmp_path, isolated_env,
) -> None:
    """envelope.env_overrides (profile.env + post-auth wizard overrides)
    layer on top of workspace .env and reach os.environ."""
    (tmp_path / ".env").write_text("TEST_S3PR1_KEY_A=from-workspace")

    runtime = _RecordingRuntime()
    env = _good_envelope(
        workspace_path=str(tmp_path),
        env_overrides={
            "TEST_S3PR1_KEY_A": "from-overrides",  # wins over .env
            "TEST_S3PR1_KEY_B": "from-overrides-only",
        },
    )
    bootstrap_session(env, runtime_factory=lambda e: runtime)

    assert os.environ.get("TEST_S3PR1_KEY_A") == "from-overrides"
    assert os.environ.get("TEST_S3PR1_KEY_B") == "from-overrides-only"


# ----------------------------------------------------------------------
# 4. PROJECT_ID / LOCATION fallback
# ----------------------------------------------------------------------


def test_bootstrap_connect_uses_envelope_project_when_set(
    tmp_path, isolated_env,
) -> None:
    """When the envelope carries project/location, the runner uses
    those (back-compat with daemon-side resolution still in transit)."""
    runtime = _RecordingRuntime()
    env = _good_envelope(
        workspace_path=str(tmp_path),
        project="my-gcp-project",
        location="us-central1",
    )
    bootstrap_session(env, runtime_factory=lambda e: runtime)

    assert runtime.connect_args == ("my-gcp-project", "us-central1")


def test_bootstrap_connect_falls_back_to_os_environ_for_project(
    tmp_path, isolated_env,
) -> None:
    """When envelope.project is empty, runner falls back to
    ``os.environ.get("PROJECT_ID")`` which step 1b populated from
    workspace .env.  Closes the daemon-side ``PROJECT_ID`` /
    ``LOCATION`` read that Shape 3 PR 1 removes."""
    (tmp_path / ".env").write_text(textwrap.dedent("""\
        PROJECT_ID=ws-derived-gcp
        LOCATION=eu-west4
    """))

    runtime = _RecordingRuntime()
    env = _good_envelope(workspace_path=str(tmp_path), project="", location="")
    bootstrap_session(env, runtime_factory=lambda e: runtime)

    assert runtime.connect_args == ("ws-derived-gcp", "eu-west4")
