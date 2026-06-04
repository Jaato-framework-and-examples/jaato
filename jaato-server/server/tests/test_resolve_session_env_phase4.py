"""Daemon-side ``_resolve_session_env`` contract (Phase 4 §B + PR #92 Y fix).

Phase 4 §B pinned: workspace .env values reach the runner via
daemon-side resolution + ``_with_session_env`` ``os.environ`` overlay
inherited by the cold-spawn fork.

PR #91 attempted to relocate workspace .env reading to the runner
(Shape 3 PR 1 principle), which broke under AppArmor confinement
(``pass`` exec blocked → secret URIs survive as literals).

PR #92 (Y fix) restored daemon-side workspace .env reading AND added
``envelope.session_env`` so the daemon ships the FULLY-RESOLVED env
to the runner over the wire.  The runner no longer needs the
``_with_session_env`` fork-inherit channel — it gets the same
values via the envelope.  Daemon-side reading remains because:
1. The daemon is unconfined and CAN exec ``pass`` / ``vault`` / etc.
2. The runner subprocess (under AppArmor) cannot, so daemon-side
   resolution is the only viable place.

Pins:

1. ``_resolve_session_env`` reads workspace ``.env`` + profile.env +
   env_overrides daemon-side, in that precedence (low → high).
2. ``_resolve_session_env`` is idempotent (second call is a no-op).
3. ``_with_session_env`` applies the resolved values to ``os.environ``
   for the duration — still load-bearing for daemon-side code paths
   that read ``os.environ`` during runtime construction (e.g.
   ``build_session_envelope`` reading ``PROJECT_ID`` / ``LOCATION``).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pytest

from server.core import JaatoServer


@pytest.fixture
def env_file(tmp_path):
    """Workspace .env file with one literal + one cross-reference."""
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
        self.plugins: List[Any] = []
        self.plugin_configs: Dict[str, Any] = {}
        self.system_instructions: Optional[str] = None
        self.gc = None
        self.completion_payload_schema = None
        self.runtime_limits = None
        self.suppress_base_instructions = False


# ----------------------------------------------------------------------
# Resolution
# ----------------------------------------------------------------------


def test_resolve_session_env_populates_from_workspace_dotenv(env_file, tmp_path):
    """Daemon reads workspace .env into ``self._session_env``,
    expanding ``${VAR}`` cross-references against sibling entries."""
    server = JaatoServer(env_file=str(env_file), workspace_path=str(tmp_path))
    server._resolve_session_env()
    assert server._session_env_resolved is True
    assert server._session_env["PHASE4_FOO"] == "bar-resolved"
    assert server._session_env["PHASE4_BAZ"] == "qux-bar-resolved"


def test_resolve_session_env_populates_from_profile_env(tmp_path):
    """profile.env values resolve daemon-side."""
    server = JaatoServer(env_file=None, workspace_path=str(tmp_path))
    server._profile = _ProfileWithEnv(env={"PROFILE_KEY": "profile-value"})
    server._resolve_session_env()
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


def test_resolve_session_env_layering_precedence(env_file, tmp_path):
    """Precedence: workspace .env < profile.env < env_overrides.
    Same key in all three lands with env_overrides value."""
    server = JaatoServer(
        env_file=str(env_file),
        workspace_path=str(tmp_path),
        env_overrides={"PHASE4_FOO": "from-overrides"},
    )
    server._profile = _ProfileWithEnv(env={"PHASE4_FOO": "from-profile"})
    server._resolve_session_env()
    assert server._session_env["PHASE4_FOO"] == "from-overrides"


def test_resolve_session_env_is_idempotent(env_file, tmp_path):
    """Second call is a no-op — important so secrets don't get
    re-resolved (re-running pass/vault subprocesses) when the runner
    re-invokes."""
    server = JaatoServer(env_file=str(env_file), workspace_path=str(tmp_path))
    server._resolve_session_env()
    server._session_env["PHASE4_FOO"] = "mutated"
    server._resolve_session_env()  # no-op
    assert server._session_env["PHASE4_FOO"] == "mutated"


def test_resolve_session_env_no_env_file(tmp_path):
    """env_file=None path: ``_session_env`` carries only framework-
    managed keys (no profile.env, no env_overrides).  Pre-Family-IV
    this was {} exactly; post-Family-IV (PR-217) the framework
    populates ``JAATO_JDTLS_STATE_DIR`` whenever workspace_path is set.
    The contract here is "no operator/profile-supplied values when
    env_file=None", not "literally empty dict".
    """
    server = JaatoServer(env_file=None, workspace_path=str(tmp_path))
    server._resolve_session_env()
    assert server._session_env_resolved is True
    # Strip framework-managed keys; what remains must be empty.
    operator_keys = {
        k: v for k, v in server._session_env.items()
        if k != "JAATO_JDTLS_STATE_DIR"
    }
    assert operator_keys == {}


# ----------------------------------------------------------------------
# _with_session_env os.environ overlay (daemon-side consumers still
# rely on this for the brief runtime-construct window)
# ----------------------------------------------------------------------


def test_with_session_env_propagates_to_environ_after_resolve(
    env_file, tmp_path, monkeypatch,
):
    """``_with_session_env`` applies the resolved env to ``os.environ``
    for the duration of the context.  Daemon-side runtime construction
    + ``build_session_envelope`` read these from ``os.environ`` (e.g.
    PROJECT_ID for Vertex AI)."""
    monkeypatch.delenv("PHASE4_FOO", raising=False)
    monkeypatch.delenv("PHASE4_BAZ", raising=False)

    server = JaatoServer(env_file=str(env_file), workspace_path=str(tmp_path))
    server._resolve_session_env()

    with server._with_session_env():
        assert os.environ["PHASE4_FOO"] == "bar-resolved"
        assert os.environ["PHASE4_BAZ"] == "qux-bar-resolved"

    assert "PHASE4_FOO" not in os.environ
    assert "PHASE4_BAZ" not in os.environ


def test_with_session_env_restores_previous_value(env_file, tmp_path, monkeypatch):
    """If ``os.environ`` had a different value before the with-block,
    exit restores the previous value rather than deleting."""
    monkeypatch.setenv("PHASE4_FOO", "pre-existing-value")

    server = JaatoServer(env_file=str(env_file), workspace_path=str(tmp_path))
    server._resolve_session_env()

    with server._with_session_env():
        assert os.environ["PHASE4_FOO"] == "bar-resolved"
    assert os.environ["PHASE4_FOO"] == "pre-existing-value"


def test_initialize_step1_treats_pre_resolved_env_as_authoritative(
    env_file, tmp_path,
):
    """Pre-spawn resolution → runner-side re-call is a no-op via the
    idempotency flag.  Mirrors the daemon-side pre-spawn + runner-side
    initialize() double-call path."""
    server = JaatoServer(env_file=str(env_file), workspace_path=str(tmp_path))
    server._resolve_session_env()
    snapshot = dict(server._session_env)

    server._resolve_session_env()  # second call — no-op
    assert server._session_env == snapshot


# ----------------------------------------------------------------------
# Family IV (PR-217): jdtls state sibling dir provisioning
# ----------------------------------------------------------------------


class TestFamilyIVJdtlsStateProvisioning:
    """Pins the Family IV contract: _resolve_session_env provisions
    the sibling jdtls state directory + exports JAATO_JDTLS_STATE_DIR
    so runner-side LSP / .lsp.json finds it.

    Family IV closes the 2026-06-04 smoke bug where jdtls's -data arg
    pointed at <workspace>/.jaato/jdtls-data — Eclipse Platform Core
    forbids workspace_location ⊆ project_location, so jdtls failed
    project import and silently dropped publishDiagnostics.
    Sibling-of-workspace location at
    <workspace.parent>/.<workspace.basename>-jdtls-state/ satisfies
    Eclipse + stays under the same per-session AppArmor profile (no
    data-leak — indexed source copies stay inside tenant boundary).
    """

    def test_provisions_sibling_jdtls_state_dir(self, tmp_path):
        """After _resolve_session_env, the sibling dir exists on disk."""
        workspace = tmp_path / "cascade_smoke"
        workspace.mkdir()
        sibling_expected = tmp_path / ".cascade_smoke-jdtls-state"
        assert not sibling_expected.exists()

        server = JaatoServer(
            env_file=None, workspace_path=str(workspace),
        )
        server._resolve_session_env()

        assert sibling_expected.exists()
        assert sibling_expected.is_dir()

    def test_exports_jaato_jdtls_state_dir_env_var(self, tmp_path):
        """After _resolve_session_env, JAATO_JDTLS_STATE_DIR is set in
        self._session_env to the absolute sibling path.  This flows
        to the runner via envelope.session_env."""
        workspace = tmp_path / "cascade_smoke"
        workspace.mkdir()

        server = JaatoServer(
            env_file=None, workspace_path=str(workspace),
        )
        server._resolve_session_env()

        assert "JAATO_JDTLS_STATE_DIR" in server._session_env
        assert server._session_env["JAATO_JDTLS_STATE_DIR"] == str(
            tmp_path / ".cascade_smoke-jdtls-state",
        )

    def test_idempotent_provisioning(self, tmp_path):
        """Sibling exists already → mkdir exist_ok=True succeeds without
        error.  Second _resolve_session_env call is a no-op via the
        idempotency flag (so the env var stays set, no re-provisioning
        race)."""
        workspace = tmp_path / "cascade_smoke"
        workspace.mkdir()
        sibling = tmp_path / ".cascade_smoke-jdtls-state"
        sibling.mkdir()  # pre-existing

        server = JaatoServer(
            env_file=None, workspace_path=str(workspace),
        )
        server._resolve_session_env()
        assert sibling.exists()

        # Second call — idempotent via _session_env_resolved flag
        snapshot = dict(server._session_env)
        server._resolve_session_env()
        assert server._session_env == snapshot

    def test_no_provisioning_when_workspace_path_unset(self, tmp_path):
        """No workspace_path → no sibling computation/provisioning.
        Symmetric with the existing 'no env_file' path: degrade
        cleanly, don't crash.  JAATO_JDTLS_STATE_DIR stays absent."""
        server = JaatoServer(env_file=None, workspace_path=None)
        server._resolve_session_env()

        assert "JAATO_JDTLS_STATE_DIR" not in server._session_env

    def test_sibling_dir_is_writable_post_provisioning(self, tmp_path):
        """Sanity probe: the provisioned sibling dir is actually
        writable (the inline write-probe in _resolve_session_env
        confirmed this; here we re-verify externally for the test
        harness).  Without write access, jdtls bootstrap would silently
        drop publishDiagnostics — the failure mode Family IV closes."""
        workspace = tmp_path / "cascade_smoke"
        workspace.mkdir()

        server = JaatoServer(
            env_file=None, workspace_path=str(workspace),
        )
        server._resolve_session_env()

        sibling_path = server._session_env.get("JAATO_JDTLS_STATE_DIR")
        assert sibling_path
        # Write a marker file post-provisioning to confirm writability.
        marker = os.path.join(sibling_path, ".test-marker")
        with open(marker, "w") as f:
            f.write("ok")
        with open(marker, "r") as f:
            assert f.read() == "ok"
        os.unlink(marker)
