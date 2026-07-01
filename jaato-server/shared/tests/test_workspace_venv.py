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
    jaato_source_dirs,
    _BRIDGE_PTH,
    PIP_APPARMOR_RULES,
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
    # Plain directory lines only — NO ``import``/``addsitedir`` (that would run
    # the base dir's editable .pth finders and leak unrelated src dirs).
    assert "import" not in body
    assert "addsitedir" not in body
    assert all(os.path.isdir(ln) for ln in body.split() if ln)


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
    # Bridge .pth written into the venv: plain path lines for the base
    # site-packages + the explicit jaato source roots (no addsitedir, so no
    # editable-finder execution / unrelated-src leak).
    bridge = os.path.join(site, _BRIDGE_PTH)
    assert os.path.exists(bridge)
    body = open(bridge).read()
    assert "addsitedir" not in body
    for d in runner_site_dirs() + jaato_source_dirs():
        assert d in body
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


# ---- pip AppArmor contribution (least-privilege, per-tool) ------------------

def test_pip_apparmor_rules_cover_distro_reads():
    body = "\n".join(PIP_APPARMOR_RULES)
    assert "/etc/debian_version" in body   # the file the peer's pip UA crashed on
    assert "/etc/os-release" in body
    assert all(r.rstrip().endswith("r,") for r in PIP_APPARMOR_RULES)  # read-only


def test_pip_tools_contribute_rules_others_dont():
    from shared.plugins.cli.plugin import create_plugin as mk_cli
    from shared.plugins.notebook.plugin import create_plugin as mk_nb
    from shared.plugins.interactive_shell.plugin import create_plugin as mk_sh

    kw = dict(workspace_path="/ws", session_id="s", config_root=None, plugin_config={})
    for mk in (mk_cli, mk_nb, mk_sh):
        assert mk().get_apparmor_rules(**kw) == list(PIP_APPARMOR_RULES)


def test_resolver_dedups_identical_tool_contributions():
    # cli + notebook + interactive_shell all contribute the SAME reads; the
    # resolver must collapse them (one copy in the rendered profile).
    from server.apparmor import resolve_plugin_apparmor_rules
    from shared.plugins.cli.plugin import create_plugin as mk_cli
    from shared.plugins.notebook.plugin import create_plugin as mk_nb
    from shared.plugins.interactive_shell.plugin import create_plugin as mk_sh

    plugins = {"cli": mk_cli(), "notebook": mk_nb(),
               "interactive_shell": mk_sh(), "todo": object()}

    class Reg:
        def get_plugin(self, n): return plugins.get(n)

    class Server:
        registry = Reg()

    class Profile:
        plugins = ["cli(preload)", "notebook", "interactive_shell", "todo"]
        plugin_configs = {}

    rules = resolve_plugin_apparmor_rules(Server(), Profile(), "s", "/ws", None)
    assert rules.count("/etc/debian_version  r,") == 1   # deduped, not 3
    assert set(PIP_APPARMOR_RULES).issubset(set(rules))


def test_cli_foreground_path_activates_workspace_venv(tmp_path, monkeypatch):
    # Regression for the foreground gap: _execute -> run_command must carry the
    # venv activation (previously only the streaming path did).
    from shared.plugins.cli import plugin as cli_mod
    venv = str(tmp_path / "tool-venv")
    ensure_workspace_venv(venv)
    p = cli_mod.create_plugin()
    p.initialize({"workspace_root": str(tmp_path), "workspace_venv": venv})

    captured = {}

    class _R:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run_command(command, **kw):
        captured["extra_env"] = kw.get("extra_env")
        return _R()

    monkeypatch.setattr(cli_mod, "run_command", fake_run_command)
    p._execute({"command": "python3 --version"})

    ee = captured["extra_env"]
    assert ee and ee.get("VIRTUAL_ENV") == venv
    assert ee["PATH"].split(os.pathsep)[0] == os.path.join(venv, "bin")


def test_jaato_source_dirs_are_shared_and_sdk_roots():
    dirs = jaato_source_dirs()
    assert len(dirs) == 2 and all(os.path.isdir(d) for d in dirs)
    import shared, jaato_sdk
    assert os.path.dirname(shared.__path__[0]) in dirs
    assert os.path.dirname(jaato_sdk.__path__[0]) in dirs


def test_bridge_does_not_leak_unrelated_editable_src(tmp_path):
    # The scoped bridge must NOT surface an unrelated editable install's src
    # dir on the tool-venv sys.path (the over-share the peer hit: the client's
    # own package). Structural guarantee: no addsitedir -> no .pth finder runs,
    # and only base-site-packages + jaato source roots are listed.
    venv = str(tmp_path / "tool-venv")
    ensure_workspace_venv(venv)
    body = open(os.path.join(venv_site_packages(venv), _BRIDGE_PTH)).read()
    lines = [ln for ln in body.splitlines() if ln.strip()]
    allowed = set(runner_site_dirs() + jaato_source_dirs())
    assert set(lines) == allowed          # nothing beyond the scoped set
    assert "addsitedir" not in body       # no finder execution


# ---- pip materialization (bare `!pip` / `pip` -> tool-venv) ------------------

def test_ensure_materializes_pip_when_venv_created_without_it(tmp_path):
    # Simulate a client-created venv WITHOUT pip (the bot's case): ensure must
    # materialize <venv>/bin/pip so bare `pip`/`!pip` resolves to the tool-venv.
    venv = str(tmp_path / "v")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", venv], check=True)
    pip = os.path.join(venv, "bin", "pip")
    assert not os.path.exists(pip)          # precondition: no pip
    ensure_workspace_venv(venv)
    assert os.path.exists(pip)              # ensurepip materialized it


def test_real_venv_has_pip_and_it_targets_the_venv(tmp_path):
    venv = str(tmp_path / "tool-venv")
    ensure_workspace_venv(venv)
    pip = os.path.join(venv, "bin", "pip")
    assert os.path.exists(pip)
    # the pip console script's shebang points at the venv python -> installs here
    assert venv in open(pip).readline()
