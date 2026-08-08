"""Tests for the notebook local-backend in-process execution gate.

The local backend executes model-authored cells via exec()/eval() in the
host interpreter, so it fails closed: execution is permitted only under
kernel-enforced AppArmor confinement or with an explicit operator opt-in.
These tests live in their own module (no autouse opt-in fixture) so the
refuse-by-default path can be exercised directly.
"""

from __future__ import annotations

import pytest

from ..backends.local import (
    INPROCESS_OPT_IN_ENV,
    LocalJupyterBackend,
    _apparmor_enforced_profile,
)
from ..types import ExecutionStatus


@pytest.fixture(autouse=True)
def _clear_opt_in(monkeypatch):
    """Ensure no ambient opt-in leaks in from the environment."""
    monkeypatch.delenv(INPROCESS_OPT_IN_ENV, raising=False)


def _notebook(backend):
    return backend.create_notebook("gate-test").notebook_id


class TestRefusedByDefault:
    def test_execute_refused_without_opt_in_or_confinement(self, monkeypatch):
        # Force the "unconfined" view regardless of the host's real state.
        monkeypatch.setattr(
            "shared.plugins.notebook.backends.local._apparmor_enforced_profile",
            lambda: None,
        )
        backend = LocalJupyterBackend()
        backend.initialize()
        nb = _notebook(backend)

        result = backend.execute(nb, "x = 1")
        assert result.status == ExecutionStatus.FAILED
        assert result.error_name == "InProcessExecutionRefused"
        assert INPROCESS_OPT_IN_ENV in result.error_message
        # The cell did not run: no variable was bound.
        assert backend.get_variables(nb) == {}


class TestEnvOptIn:
    def test_env_opt_in_allows_execution(self, monkeypatch):
        monkeypatch.setattr(
            "shared.plugins.notebook.backends.local._apparmor_enforced_profile",
            lambda: None,
        )
        monkeypatch.setenv(INPROCESS_OPT_IN_ENV, "1")
        backend = LocalJupyterBackend()
        backend.initialize()
        nb = _notebook(backend)

        result = backend.execute(nb, "2 + 3")
        assert result.status == ExecutionStatus.COMPLETED
        assert any(o.content == "5" for o in result.outputs)


class TestConfigOptIn:
    def test_config_opt_in_allows_execution(self, monkeypatch):
        monkeypatch.setattr(
            "shared.plugins.notebook.backends.local._apparmor_enforced_profile",
            lambda: None,
        )
        backend = LocalJupyterBackend()
        backend.initialize({"allow_inprocess_exec": True})
        nb = _notebook(backend)

        result = backend.execute(nb, "x = 7\nx * 2")
        assert result.status == ExecutionStatus.COMPLETED
        assert any(o.content == "14" for o in result.outputs)


class TestApparmorConfinementAllows:
    def test_enforced_profile_allows_without_opt_in(self, monkeypatch):
        monkeypatch.setattr(
            "shared.plugins.notebook.backends.local._apparmor_enforced_profile",
            lambda: "jaato-ws-abc//child",
        )
        backend = LocalJupyterBackend()
        backend.initialize()  # no opt-in
        nb = _notebook(backend)

        result = backend.execute(nb, "1 + 1")
        assert result.status == ExecutionStatus.COMPLETED


class TestApparmorProfileParsing:
    def test_unconfined_is_none(self, monkeypatch, tmp_path):
        self._patch_attr(monkeypatch, tmp_path, "unconfined\n")
        assert _apparmor_enforced_profile() is None

    def test_enforce_profile_returned(self, monkeypatch, tmp_path):
        self._patch_attr(monkeypatch, tmp_path, "jaato-ws-x//child (enforce)\n")
        assert _apparmor_enforced_profile() == "jaato-ws-x//child"

    def test_nul_terminated_enforce_profile(self, monkeypatch, tmp_path):
        # The kernel NUL-terminates the value; a whitespace-only strip would
        # leave the NUL and false-negative the enforced profile, breaking
        # the confined path. Regression for that parse bug.
        self._patch_attr(monkeypatch, tmp_path, "jaato-ws-x//child (enforce)\n\x00")
        assert _apparmor_enforced_profile() == "jaato-ws-x//child"

    def test_nul_then_newline_order(self, monkeypatch, tmp_path):
        self._patch_attr(monkeypatch, tmp_path, "jaato-ws-x//child (enforce)\x00")
        assert _apparmor_enforced_profile() == "jaato-ws-x//child"

    def test_complain_mode_is_none(self, monkeypatch, tmp_path):
        # complain mode logs but does not block — not a boundary.
        self._patch_attr(monkeypatch, tmp_path, "jaato-ws-x (complain)\n")
        assert _apparmor_enforced_profile() is None

    def test_missing_file_is_none(self, monkeypatch):
        def _raise(*a, **k):
            raise OSError("no /proc on this platform")
        monkeypatch.setattr("builtins.open", _raise)
        assert _apparmor_enforced_profile() is None

    @staticmethod
    def _patch_attr(monkeypatch, tmp_path, content):
        f = tmp_path / "attr_current"
        f.write_text(content)
        real_open = open

        def fake_open(path, *a, **k):
            if path == "/proc/self/attr/current":
                return real_open(f, *a, **k)
            return real_open(path, *a, **k)

        monkeypatch.setattr("builtins.open", fake_open)
