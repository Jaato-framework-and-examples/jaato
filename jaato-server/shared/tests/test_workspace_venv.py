"""Tests for shared.plugins.workspace_venv — workspace-scoped tool venvs."""

import os
import subprocess
import sys

import pytest

from shared.plugins.workspace_venv import (
    resolve_venv_path,
    ensure_workspace_venv,
    apply_venv_to_env,
    venv_python,
    venv_site_packages,
    runner_site_dirs,
    _BRIDGE_PTH,
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

def test_ensure_fastpath_skips_create_but_refreshes_bridge(tmp_path):
    # Pre-existing pyvenv.cfg short-circuits CREATION (no subprocess), but the
    # runner-import bridge is (re)written even for a venv made elsewhere.
    venv = tmp_path / "v"
    site = venv / "lib" / "python3.99" / "site-packages"
    site.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /x\n")
    assert ensure_workspace_venv(str(venv)) == str(venv)
    bridge = site / _BRIDGE_PTH
    assert bridge.exists()
    body = bridge.read_text()
    assert body.startswith("import site;")
    assert "addsitedir" in body


# ---- runner-import bridge ---------------------------------------------------

def test_runner_site_dirs_nonempty_and_existing():
    dirs = runner_site_dirs()
    assert dirs, "runner must have at least one site-packages dir"
    assert all(os.path.isdir(d) for d in dirs)


def test_ensure_creates_real_venv_and_resolves(tmp_path):
    venv = str(tmp_path / "tool-venv")
    ensure_workspace_venv(venv)
    assert os.path.exists(os.path.join(venv, "pyvenv.cfg"))
    assert os.path.exists(venv_python(venv))
    site = venv_site_packages(venv)
    assert site and os.path.isdir(site)
    # Bridge .pth written into the venv, targeting the runner's site dirs.
    bridge = os.path.join(site, _BRIDGE_PTH)
    assert os.path.exists(bridge)
    for d in runner_site_dirs():
        assert repr(d) in open(bridge).read()
    # Second call is idempotent (create short-circuits; bridge refreshed).
    assert ensure_workspace_venv(venv) == venv
    # An activated env prepends the resolved site-packages.
    env = {"PATH": "/usr/bin"}
    apply_venv_to_env(env, venv)
    assert env["PYTHONPATH"].split(os.pathsep)[0] == site


def test_bridged_venv_python_can_import_shared(tmp_path):
    # The authoritative e2e (mirrors the peer's notebook failure): a tool-venv
    # created from the runner venv, once bridged, lets its OWN interpreter
    # import jaato's `shared` — for the editable install this repo uses.
    venv = str(tmp_path / "tool-venv")
    ensure_workspace_venv(venv)
    out = subprocess.run(
        [venv_python(venv), "-c", "import shared; print(shared.__file__)"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, f"import shared failed: {out.stderr}"
    assert "shared" in out.stdout
