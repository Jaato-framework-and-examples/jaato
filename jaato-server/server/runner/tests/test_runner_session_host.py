"""Tests for the runner-side session host (Phase 3 §3.3b).

Two test surfaces:
1. Envelope validation (bad fields → ``BootstrapError("validate", ...)``).
2. Bootstrap orchestration with a stubbed runtime factory (verifies
   the envelope's plugin list / provider / model flow into the
   runtime correctly without standing up a real provider).

§3.3c will land the daemon-side dispatch (the RPC method that the
daemon sends on session-spawn).  §3.3b's tests are Python-API level
to keep the commit reviewable in isolation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from server.runner.session import (
    BootstrapError,
    RunnerSessionHost,
    bootstrap_session,
)
from shared.session_envelope import SessionInitEnvelope


# ----------------------------------------------------------------------
# Helpers — minimal envelope + recording stub runtime.
# ----------------------------------------------------------------------


def _good_envelope(**overrides: Any) -> SessionInitEnvelope:
    base = dict(
        session_id="sess-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[
            {"name": "signal_completion", "preload": True},
            {"name": "cli", "preload": False},
        ],
        # Phase 4 §C: per-plugin configs now live in the top-level
        # envelope.plugin_configs map (schema v2), not in plugins[i].config.
        plugin_configs={"cli": {"max_workers": 4}},
        system_instructions="You are a helpful agent.",
        agent_id="main",
    )
    base.update(overrides)
    return SessionInitEnvelope(**base)


class _StubSession:
    """Stand-in for JaatoSession; records the args it was constructed
    with.  Returned by the stub runtime's ``create_session``."""

    def __init__(self, **kwargs: Any) -> None:
        self.create_session_kwargs = dict(kwargs)
        self._daemon_session_id: Optional[str] = None

    def set_daemon_session_id(self, session_id: str) -> None:
        # Bootstrap stamps envelope.session_id onto the session.
        self._daemon_session_id = session_id


class _StubRuntime:
    """Stand-in for JaatoRuntime; records the create_session call so
    tests can assert the envelope flowed through correctly.

    ``_registry`` set truthy so the Path D ``_configure_runtime_plugins``
    guard skips for these envelope-flow tests (these don't pin
    plugin configuration; the Path D-specific tests live in
    ``test_runner_configure_plugins_path_d.py``).
    """

    def __init__(self, *, fail: Optional[BaseException] = None) -> None:
        self._fail = fail
        self.create_session_kwargs: Optional[Dict[str, Any]] = None
        # Sentinel: tells Path D's "already configured?" guard to skip.
        self._registry = object()

    def create_session(self, **kwargs: Any) -> _StubSession:
        if self._fail is not None:
            raise self._fail
        self.create_session_kwargs = dict(kwargs)
        return _StubSession(**kwargs)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_bootstrap_rejects_empty_session_id() -> None:
    env = _good_envelope(session_id="")
    with pytest.raises(BootstrapError) as excinfo:
        bootstrap_session(env, runtime_factory=lambda e: _StubRuntime())
    assert excinfo.value.stage == "validate"
    assert "session_id" in excinfo.value.message


def test_bootstrap_rejects_empty_model_name() -> None:
    env = _good_envelope(model_name="")
    with pytest.raises(BootstrapError) as excinfo:
        bootstrap_session(env, runtime_factory=lambda e: _StubRuntime())
    assert excinfo.value.stage == "validate"
    assert "model_name" in excinfo.value.message


def test_bootstrap_rejects_empty_provider_name() -> None:
    env = _good_envelope(provider_name="")
    with pytest.raises(BootstrapError) as excinfo:
        bootstrap_session(env, runtime_factory=lambda e: _StubRuntime())
    assert excinfo.value.stage == "validate"
    assert "provider_name" in excinfo.value.message


def test_bootstrap_accepts_no_workspace_path() -> None:
    """Headless / no-workspace sessions are legitimate per parent
    §3."""
    env = _good_envelope(workspace_path=None)
    runtime = _StubRuntime()
    host = bootstrap_session(env, runtime_factory=lambda e: runtime)
    assert host.workspace_path is None
    assert host.is_ready is True


def test_bootstrap_accepts_no_profile_name() -> None:
    """Inline-spec sessions don't carry a profile name."""
    env = _good_envelope(profile_name=None)
    runtime = _StubRuntime()
    host = bootstrap_session(env, runtime_factory=lambda e: runtime)
    assert host.is_ready is True


# ----------------------------------------------------------------------
# Plugin spec extraction → runtime.create_session args
# ----------------------------------------------------------------------


def test_bootstrap_passes_plugin_list_to_runtime() -> None:
    env = _good_envelope()
    runtime = _StubRuntime()
    bootstrap_session(env, runtime_factory=lambda e: runtime)
    assert runtime.create_session_kwargs is not None
    assert runtime.create_session_kwargs["tools"] == [
        "signal_completion", "cli",
    ]


def test_bootstrap_passes_plugin_configs_to_runtime() -> None:
    env = _good_envelope()
    runtime = _StubRuntime()
    bootstrap_session(env, runtime_factory=lambda e: runtime)
    assert runtime.create_session_kwargs["plugin_configs"] == {
        "cli": {"max_workers": 4},
    }


def test_bootstrap_passes_preloaded_plugins_to_runtime() -> None:
    env = _good_envelope()
    runtime = _StubRuntime()
    bootstrap_session(env, runtime_factory=lambda e: runtime)
    assert runtime.create_session_kwargs["preloaded_plugins"] == {
        "signal_completion",
    }


def test_bootstrap_passes_provider_and_model() -> None:
    env = _good_envelope(provider_name="openrouter", model_name="anthropic/claude-3.5")
    runtime = _StubRuntime()
    bootstrap_session(env, runtime_factory=lambda e: runtime)
    assert runtime.create_session_kwargs["model"] == "anthropic/claude-3.5"
    assert runtime.create_session_kwargs["provider_name"] == "openrouter"


def test_bootstrap_passes_system_instructions() -> None:
    env = _good_envelope(system_instructions="Be concise.")
    runtime = _StubRuntime()
    bootstrap_session(env, runtime_factory=lambda e: runtime)
    assert (
        runtime.create_session_kwargs["system_instructions"] == "Be concise."
    )


def test_bootstrap_passes_completion_payload_schema() -> None:
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
    env = _good_envelope(completion_payload_schema=schema)
    runtime = _StubRuntime()
    bootstrap_session(env, runtime_factory=lambda e: runtime)
    assert (
        runtime.create_session_kwargs["completion_payload_schema"] == schema
    )


def test_bootstrap_passes_agent_params() -> None:
    env = _good_envelope(agent_params={"case_id": "c-42"})
    runtime = _StubRuntime()
    bootstrap_session(env, runtime_factory=lambda e: runtime)
    assert runtime.create_session_kwargs["agent_params"] == {"case_id": "c-42"}


def test_bootstrap_with_empty_plugins_passes_empty_list() -> None:
    """Empty plugin list flows as ``tools=[]`` so the runtime
    produces the minimal framework set (introspection + lifecycle +
    framework infra) rather than expanding to all exposed plugins.

    Pre-2026-06-07 this assertion was ``tools is None`` because the
    helper had ``tools=tool_names or None`` — falsy coercion swapped
    empty-list for None, which the runtime then interpreted as
    "load ALL exposed plugins" per its docstring.  The vLLM smoke
    surfaced the resulting ~22-tool wire surface (instead of ~8);
    PR ``fix/runner-session-tools-falsy-none`` (2026-06-07) drops
    the falsy coercion so ``plugins: []`` in a profile actually
    produces an empty tool surface.

    Phase 4 §C: plugin_configs is a separate top-level envelope
    field (no longer derived from plugins[i].config), so the test
    clears it explicitly to assert the "no configs" path.
    """
    env = _good_envelope(plugins=[], plugin_configs={})
    runtime = _StubRuntime()
    bootstrap_session(env, runtime_factory=lambda e: runtime)
    assert runtime.create_session_kwargs["tools"] == [], (
        "Empty profile.plugins must produce tools=[] (NOT None).  "
        "None would cascade to the runtime's 'load all exposed "
        "plugins' fallback — the bug the vLLM smoke 2026-06-07 "
        "surfaced."
    )
    # plugin_configs and preloaded_plugins still get the empty→None
    # falsy coercion since their downstream semantics treat
    # empty-dict and empty-set as equivalent to None (no overrides,
    # no preloaded plugins).  Distinct from tools where None has
    # meaningfully different semantics.
    assert runtime.create_session_kwargs["plugin_configs"] is None
    assert runtime.create_session_kwargs["preloaded_plugins"] is None


def test_bootstrap_rejects_plugin_entry_missing_name() -> None:
    env = _good_envelope(plugins=[{"preload": True}])
    with pytest.raises(BootstrapError) as excinfo:
        bootstrap_session(env, runtime_factory=lambda e: _StubRuntime())
    assert excinfo.value.stage == "configure"
    assert "missing 'name'" in excinfo.value.message


# ----------------------------------------------------------------------
# Failure-stage attribution
# ----------------------------------------------------------------------


def test_bootstrap_runtime_construction_failure_attributed_to_runtime_stage() -> None:
    env = _good_envelope()

    def _bad_factory(e: SessionInitEnvelope) -> _StubRuntime:
        raise RuntimeError("provider credentials missing")

    with pytest.raises(BootstrapError) as excinfo:
        bootstrap_session(env, runtime_factory=_bad_factory)
    assert excinfo.value.stage == "runtime"
    assert "provider credentials missing" in excinfo.value.message


def test_bootstrap_session_construction_failure_attributed_to_configure_stage() -> None:
    env = _good_envelope()
    runtime = _StubRuntime(fail=ValueError("plugin discovery: no entry points"))

    with pytest.raises(BootstrapError) as excinfo:
        bootstrap_session(env, runtime_factory=lambda e: runtime)
    assert excinfo.value.stage == "configure"
    assert "no entry points" in excinfo.value.message


# ----------------------------------------------------------------------
# Test-only "skip runtime" path
# ----------------------------------------------------------------------


def test_bootstrap_skip_runtime_returns_host_without_session() -> None:
    """``runtime_factory=None`` skips runtime construction; the host
    still validates the envelope but ``is_ready`` is False."""
    env = _good_envelope()
    host = bootstrap_session(env, runtime_factory=None)
    assert host.envelope is env
    assert host.runtime is None
    assert host.session is None
    assert host.is_ready is False


def test_bootstrap_skip_runtime_still_validates() -> None:
    """Validation runs even on the skip-runtime test path."""
    env = _good_envelope(session_id="")
    with pytest.raises(BootstrapError) as excinfo:
        bootstrap_session(env, runtime_factory=None)
    assert excinfo.value.stage == "validate"


# ----------------------------------------------------------------------
# Host attribute access
# ----------------------------------------------------------------------


def test_host_exposes_session_id_and_workspace_path() -> None:
    env = _good_envelope()
    runtime = _StubRuntime()
    host = bootstrap_session(env, runtime_factory=lambda e: runtime)
    assert host.session_id == "sess-1"
    assert host.workspace_path == "/tmp/ws"


def test_host_is_ready_true_after_bootstrap() -> None:
    env = _good_envelope()
    runtime = _StubRuntime()
    host = bootstrap_session(env, runtime_factory=lambda e: runtime)
    assert host.is_ready is True
    assert host.session is not None
    assert host.runtime is runtime
