"""Tests for shared.plugins.workspace_venv — workspace-scoped tool venvs."""

import os
import sys

import pytest

from shared.plugins.workspace_venv import (
    resolve_venv_path,
    ensure_workspace_venv,
    apply_venv_to_env,
    venv_python,
    venv_site_packages,
)


# ---- resolve_venv_path ------------------------------------------------------

@pytest.mark.parametrize("raw", ["", "   ", None])
def test_resolve_empty_is_off(raw):
    assert resolve_venv_path(raw, "/ws") is None


def test_resolve_absolute_realpaths(tmp_path):
    p = str(tmp_path / "v")
    assert resolve_venv_path(p, None) == os.path.realpath(p)


def test_resolve_relative_joins_workspace(tmp_path):
    ws = str(tmp_path)
    got = resolve_venv_path(".jaato/tool-venv", ws)
    assert got == os.path.realpath(os.path.join(ws, ".jaato/tool-venv"))


def test_resolve_relative_without_workspace_raises():
    # No silent cwd fallback — the target would be non-deterministic.
    with pytest.raises(ValueError):
        resolve_venv_path(".jaato/tool-venv", None)


# ---- apply_venv_to_env ------------------------------------------------------

def test_apply_venv_activates_env():
    venv = "/tmp/example-venv"
    env = {"PATH": "/usr/bin", "PYTHONPATH": "/src", "PYTHONHOME": "/base"}
    apply_venv_to_env(env, venv)

    bindir = os.path.join(venv, "Scripts" if os.name == "nt" else "bin")
    assert env["PATH"].split(os.pathsep)[0] == bindir      # venv bin wins
    assert "/usr/bin" in env["PATH"]                        # base preserved
    assert env["VIRTUAL_ENV"] == venv
    assert "PYTHONHOME" not in env                          # cleared
    # site-packages only prepended when it exists on disk (this venv is fake)
    assert env["PYTHONPATH"].endswith("/src")


def test_apply_venv_empty_env_ok():
    env = {}
    apply_venv_to_env(env, "/tmp/v")
    assert env["PATH"]           # populated, no leading separator
    assert not env["PATH"].startswith(os.pathsep)


# ---- ensure_workspace_venv (idempotency + real creation) --------------------

def test_ensure_fastpath_when_pyvenv_cfg_exists(tmp_path):
    # Pre-existing pyvenv.cfg short-circuits: no subprocess, returns the path.
    venv = tmp_path / "v"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /x\n")
    assert ensure_workspace_venv(str(venv)) == str(venv)


def test_ensure_creates_real_venv_and_resolves(tmp_path):
    venv = str(tmp_path / "tool-venv")
    ensure_workspace_venv(venv)
    assert os.path.exists(os.path.join(venv, "pyvenv.cfg"))
    assert os.path.exists(venv_python(venv))
    site = venv_site_packages(venv)
    assert site and os.path.isdir(site)
    # Second call is a no-op fast-path (idempotent).
    assert ensure_workspace_venv(venv) == venv
    # An activated env prepends the resolved site-packages.
    env = {"PATH": "/usr/bin"}
    apply_venv_to_env(env, venv)
    assert env["PYTHONPATH"].split(os.pathsep)[0] == site
