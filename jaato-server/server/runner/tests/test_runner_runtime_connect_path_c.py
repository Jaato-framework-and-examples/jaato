"""Regression-pin tests for Path C — runner-side runtime.connect()
in bootstrap_session.

**Root cause** (cycle-4 integration-test diagnosis):

Path B (commit c68151a8) closed the missing-dispatch class — IPC
sessions now synchronously call ``dispatch_bootstrap_envelope``
after spawn.  But the runner-side ``_handle_session_bootstrap``
itself failed with "Runtime not connected. Call connect() first."
because ``runtime.create_session(...)`` guards on
``self._connected`` (jaato_runtime.py:964) and the runner-side
bootstrap path never called ``runtime.connect(...)``.

The fix has two parts:

1. **Envelope schema extension**: add ``project: str`` +
   ``location: str`` fields (defaults to empty strings preserving
   backward compat).  Daemon-side ``build_session_envelope`` reads
   ``PROJECT_ID`` + ``LOCATION`` env vars (matching the daemon's
   own ``core.py:1550-1551`` reads).

2. **Runner-side connect**: ``bootstrap_session`` calls
   ``runtime.connect(envelope.project, envelope.location)`` after
   constructing the runtime, before ``_build_session`` (which
   internally calls ``runtime.create_session``).

Non-Vertex providers (anthropic, openrouter, ollama, zhipuai,
etc.) tolerate empty strings — provider plugin ``initialize()``
ignores ``project``/``location``.  Vertex AI / Google GenAI use
the real values.

**Bug chain at cycle 4** (3 layers, all from §7c structural
regressions surfaced by integration testing):

| Layer | Commit | Fix |
|---|---|---|
| Snapshot caller wraps no_session | 20b16326 | Path A narrow |
| IPC dispatches session.bootstrap | c68151a8 | Path B architectural |
| Runner connects runtime pre-create_session | (this) | Path C envelope+connect |

Tests pin:

- ``SessionInitEnvelope`` declares ``project`` + ``location`` fields
- Round-trip preserves project/location through ``to_dict`` /
  ``from_dict``
- Default values are empty strings (backward-compat)
- ``build_session_envelope`` reads ``PROJECT_ID`` + ``LOCATION`` env
- ``bootstrap_session`` calls ``runtime.connect(...)`` after
  ``_default_runtime_factory`` returns
- ``bootstrap_session`` is idempotent against already-connected
  runtimes (test stubs that pre-connect for isolated testing)
- ``BootstrapError("connect", ...)`` surfaces when ``connect()``
  itself raises
"""

from __future__ import annotations

import ast
import inspect
from typing import Any, Optional

import pytest

import server.runner.session as runner_session_module
from server.runner.session import (
    BootstrapError,
    RunnerSessionHost,
    bootstrap_session,
)
from shared.session_envelope import SessionInitEnvelope


# ----------------------------------------------------------------------
# Envelope schema pins
# ----------------------------------------------------------------------


def test_envelope_declares_project_field() -> None:
    """Pin: ``SessionInitEnvelope`` carries ``project`` field."""
    env = SessionInitEnvelope(
        session_id="s",
        workspace_path=None,
        profile_name=None,
        provider_name="anthropic",
        model_name="claude",
    )
    assert hasattr(env, "project")
    assert env.project == ""  # default empty


def test_envelope_declares_location_field() -> None:
    env = SessionInitEnvelope(
        session_id="s",
        workspace_path=None,
        profile_name=None,
        provider_name="anthropic",
        model_name="claude",
    )
    assert hasattr(env, "location")
    assert env.location == ""


def test_envelope_round_trip_preserves_project_location() -> None:
    """Wire-shape round-trip: ``to_dict`` → ``from_dict`` preserves
    the new fields."""
    env = SessionInitEnvelope(
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="p",
        provider_name="google_genai",
        model_name="gemini",
        project="my-gcp-project",
        location="us-central1",
    )
    d = env.to_dict()
    assert d["project"] == "my-gcp-project"
    assert d["location"] == "us-central1"
    back = SessionInitEnvelope.from_dict(d)
    assert back.project == "my-gcp-project"
    assert back.location == "us-central1"


def test_envelope_from_dict_handles_missing_project_location() -> None:
    """Backward-compat: ``from_dict`` on a dict missing the new
    fields defaults to empty strings (pre-Path-C envelopes)."""
    legacy_dict = {
        "schema_version": 1,
        "session_id": "s",
        "workspace_path": None,
        "profile_name": None,
        "provider_name": "anthropic",
        "model_name": "claude",
        # No project / location keys.
    }
    env = SessionInitEnvelope.from_dict(legacy_dict)
    assert env.project == ""
    assert env.location == ""


# ----------------------------------------------------------------------
# Daemon-side envelope-build reads env vars
# ----------------------------------------------------------------------


def test_build_session_envelope_reads_project_location_env(
    monkeypatch,
) -> None:
    """``build_session_envelope`` mirrors the daemon-side
    ``core.py:1550-1551`` reads — picks up ``PROJECT_ID`` +
    ``LOCATION`` env vars and threads them into the envelope so
    the runner can call ``runtime.connect()``."""
    from server.runner_spawn import build_session_envelope

    monkeypatch.setenv("PROJECT_ID", "vertex-proj-1")
    monkeypatch.setenv("LOCATION", "us-west1")

    class _MinimalServer:
        _profile = None
        config_root = None

    env = build_session_envelope(
        server=_MinimalServer(),
        session_id="sess",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
    )
    assert env.project == "vertex-proj-1"
    assert env.location == "us-west1"


def test_build_session_envelope_empty_env_yields_empty_strings(
    monkeypatch,
) -> None:
    monkeypatch.delenv("PROJECT_ID", raising=False)
    monkeypatch.delenv("LOCATION", raising=False)
    from server.runner_spawn import build_session_envelope

    class _MinimalServer:
        _profile = None
        config_root = None

    env = build_session_envelope(
        server=_MinimalServer(),
        session_id="sess",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
    )
    assert env.project == ""
    assert env.location == ""


# ----------------------------------------------------------------------
# Runner-side runtime.connect call
# ----------------------------------------------------------------------


class _StubRuntime:
    """Stand-in JaatoRuntime that tracks connect() calls + supports
    create_session() returning a stub session.

    ``_registry`` set truthy so the Path D ``_configure_runtime_plugins``
    guard skips for these Path-C-focused tests (the stub doesn't
    implement ``configure_plugins`` and Path C tests are pinning the
    connect call, not plugin configuration).
    """

    def __init__(self) -> None:
        self.connect_calls: list = []
        self.is_connected: bool = False
        # Sentinel: tells Path D's "already configured?" guard to skip.
        self._registry = object()

    def connect(self, project: str, location: str) -> None:
        self.connect_calls.append((project, location))
        self.is_connected = True

    def create_session(self, **kwargs: Any) -> Any:
        # Return a stub object — runner_session.bootstrap_session
        # treats the return as opaque.
        class _StubSession:
            pass

        return _StubSession()


def _envelope(
    *, project: str = "", location: str = "",
) -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-pc",
        workspace_path="/tmp/ws",
        profile_name="p",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
        project=project,
        location=location,
    )


def test_bootstrap_session_calls_runtime_connect() -> None:
    """Pin: ``bootstrap_session`` calls ``runtime.connect()`` after
    construction — closes the cycle-4 bug class.  Without this,
    ``runtime.create_session(...)`` would raise
    ``RuntimeError("Runtime not connected.")``."""
    stub = _StubRuntime()
    host = bootstrap_session(
        _envelope(project="my-proj", location="us-east1"),
        runtime_factory=lambda env: stub,
    )
    assert stub.connect_calls == [("my-proj", "us-east1")]
    assert isinstance(host, RunnerSessionHost)


def test_bootstrap_session_passes_envelope_project_location() -> None:
    """Pin: the values passed to ``runtime.connect`` are the
    envelope's ``project`` / ``location`` (not hardcoded)."""
    stub = _StubRuntime()
    bootstrap_session(
        _envelope(project="vertex-proj", location="europe-west1"),
        runtime_factory=lambda env: stub,
    )
    assert stub.connect_calls == [("vertex-proj", "europe-west1")]


def test_bootstrap_session_empty_strings_for_non_vertex() -> None:
    """Pin: non-Vertex providers (default empty strings) still
    successfully bootstrap — connect() just sets internal state."""
    stub = _StubRuntime()
    host = bootstrap_session(
        _envelope(),
        runtime_factory=lambda env: stub,
    )
    assert stub.connect_calls == [("", "")]
    assert host is not None


def test_bootstrap_session_idempotent_on_pre_connected_runtime() -> None:
    """If the factory returns an already-connected runtime (test
    stubs), the bootstrap skips the connect call.  Pin: no
    double-connect."""

    class _PreConnectedStub(_StubRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.is_connected = True

    stub = _PreConnectedStub()
    bootstrap_session(
        _envelope(project="x", location="y"),
        runtime_factory=lambda env: stub,
    )
    assert stub.connect_calls == [], (
        "Path C regression: bootstrap_session connected an already-"
        "connected runtime; the ``is_connected`` guard didn't fire."
    )


def test_bootstrap_session_connect_raise_surfaces_as_bootstrap_error() -> None:
    """When ``runtime.connect`` raises, the wrapper surfaces a
    typed ``BootstrapError("connect", ...)`` (matches the existing
    "runtime" / "configure" stage codes)."""

    class _BoomConnect(_StubRuntime):
        def connect(self, project: str, location: str) -> None:
            raise RuntimeError("provider config invalid")

    with pytest.raises(BootstrapError) as exc_info:
        bootstrap_session(
            _envelope(),
            runtime_factory=lambda env: _BoomConnect(),
        )
    assert exc_info.value.stage == "connect"
    assert "provider config invalid" in exc_info.value.message


# ----------------------------------------------------------------------
# AST-level pin: bootstrap_session contains a connect call
# ----------------------------------------------------------------------


def test_bootstrap_session_source_calls_runtime_connect() -> None:
    """Pin via AST: ``bootstrap_session`` body contains
    ``runtime.connect(...)``.  Catches accidental deletion of the
    Path C fix."""
    src = inspect.getsource(bootstrap_session)
    import textwrap
    src = textwrap.dedent(src)
    tree = ast.parse(src)
    calls = [
        n for n in ast.walk(tree)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "connect"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "runtime"
        )
    ]
    assert calls, (
        "Path C regression: ``runtime.connect(...)`` call missing "
        "from ``bootstrap_session`` body.  Without it, "
        "``runtime.create_session`` raises ``RuntimeError(\"Runtime "
        "not connected. Call connect() first.\")``."
    )
