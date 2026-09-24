"""#1225 -- a per-workspace HOME for model-driven subprocesses.

The runner's subprocesses (``cli`` commands, ``interactive_shell`` PTY
children, the notebook kernel) otherwise inherit the daemon's ``HOME``, so
every tool that writes to ``~`` shares state across tenants.  A
``workspace_home`` knob (shaped like ``workspace_venv``) points ``HOME`` + the
XDG base dirs at ``<ws>/.home/`` for those three surfaces, while the runner's
own ``HOME`` is left untouched.

These are behavioural guards (no ``REVERSIONS`` list): they exercise the env
each surface builds without spawning a real subprocess -- the ``cli`` path end
to end, and ``interactive_shell`` / the notebook kernel by capturing the env
handed to the spawn primitive.  The env's own daemon-side machinery
(``effective_workspace_home`` / ``inject_workspace_home`` /
``ensure_workspace_home``) and the two "kept out of sight" obligations (git,
the Files panel) are exercised directly.
"""

import os
import subprocess

import pytest

from jaato_server.shared.plugins.workspace_home import (
    DEFAULT_WORKSPACE_HOME,
    apply_home_to_env,
    effective_workspace_home,
    ensure_workspace_home,
    ensure_workspace_home_dir,
    inject_workspace_home,
    resolve_home_path,
)

_XDG_KEYS = (
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
)


def _assert_home_env(env, home):
    """Every HOME/XDG var in *env* points under *home*."""
    assert env["HOME"] == home
    for key in _XDG_KEYS:
        assert env[key].startswith(home + os.sep), (key, env[key])


# --------------------------------------------------------------------------
# The primitive
# --------------------------------------------------------------------------

def test_apply_home_to_env_redirects_home_and_xdg():
    env = {"HOME": "/runner/home", "XDG_CONFIG_HOME": "/runner/home/.config"}
    apply_home_to_env(env, "/ws/.home")
    _assert_home_env(env, "/ws/.home")
    # XDG dirs are subdirectories, matching the spec's own defaults, so a
    # tool writing config and cache does not collide them onto one dir.
    assert env["XDG_CONFIG_HOME"] == os.path.join("/ws/.home", ".config")


def test_resolve_home_path_mirrors_workspace_venv_rules():
    assert resolve_home_path("", "/ws") is None
    assert resolve_home_path(None, "/ws") is None
    assert resolve_home_path("/abs/home", "/ws") == os.path.realpath("/abs/home")
    assert resolve_home_path(".home", "/ws") == os.path.realpath("/ws/.home")
    with pytest.raises(ValueError):
        resolve_home_path(".home", None)  # relative + no workspace = refused


# --------------------------------------------------------------------------
# Surface 1: cli (end to end through _build_subprocess_env)
# --------------------------------------------------------------------------

def test_cli_command_sees_workspace_home_runner_home_unchanged(tmp_path, monkeypatch):
    from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin

    monkeypatch.setenv("HOME", "/runner/home")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/runner/home/.config")
    ws = str(tmp_path)
    home = os.path.join(os.path.realpath(ws), ".home")

    plugin = CLIToolPlugin()
    plugin.initialize({"workspace_root": ws, "workspace_home": ".home"})
    env, _venv = plugin._build_subprocess_env()

    _assert_home_env(env, home)
    # The runner's own HOME is untouched -- ~/.jaato lookups + the ${HOME}
    # profile-expansion variable keep their meaning (§8).
    assert os.environ["HOME"] == "/runner/home"


def test_cli_without_the_knob_does_not_redirect_home(tmp_path, monkeypatch):
    from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin

    monkeypatch.setenv("HOME", "/runner/home")
    plugin = CLIToolPlugin()
    plugin.initialize({"workspace_root": str(tmp_path)})  # no workspace_home
    env, _ = plugin._build_subprocess_env()
    assert env["HOME"] == "/runner/home"


# --------------------------------------------------------------------------
# Surface 2: interactive_shell (capture the env handed to the PTY spawn)
# --------------------------------------------------------------------------

def test_interactive_shell_pty_child_sees_workspace_home(tmp_path, monkeypatch):
    from jaato_server.shared.plugins.interactive_shell import session as sess_mod

    monkeypatch.setenv("HOME", "/runner/home")
    ws = str(tmp_path)
    home = os.path.join(os.path.realpath(ws), ".home")
    captured = {}

    class _FakeProc:
        def isalive(self):
            return True

    def _fake_spawn(command, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return _FakeProc()

    monkeypatch.setattr(sess_mod, "_spawn", _fake_spawn)

    sess_mod.ShellSession(
        command="cat",
        session_id="s1",
        cwd=ws,
        workspace_root=ws,
        workspace_home=home,
    )

    _assert_home_env(captured["env"], home)
    assert os.environ["HOME"] == "/runner/home"


# --------------------------------------------------------------------------
# Surface 3: the notebook kernel (capture the env handed to Popen)
# --------------------------------------------------------------------------

def test_notebook_kernel_subprocess_sees_workspace_home(tmp_path, monkeypatch):
    from jaato_server.shared.plugins.notebook.backends import (
        subprocess_kernel as sk_mod,
    )

    monkeypatch.setenv("HOME", "/runner/home")
    ws = str(tmp_path)
    home = os.path.join(os.path.realpath(ws), ".home")
    captured = {}

    class _StopSpawn(Exception):
        pass

    def _fake_popen(*args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        raise _StopSpawn

    monkeypatch.setattr(sk_mod.subprocess, "Popen", _fake_popen)

    backend = sk_mod.SubprocessKernelBackend()
    backend.initialize({"workspace_root": ws, "workspace_home": ".home"})
    # create_notebook spawns the kernel eagerly, which reaches the patched
    # Popen and stops there.
    with pytest.raises(_StopSpawn):
        backend.create_notebook("nb")

    _assert_home_env(captured["env"], home)
    assert os.environ["HOME"] == "/runner/home"


def test_notebook_kernel_without_home_does_not_force_an_env(tmp_path, monkeypatch):
    # Without a workspace_home (and no venv), kernel_env stays None so Popen
    # inherits the daemon environment exactly as before -- the redirect is
    # opt-in, never forced.
    from jaato_server.shared.plugins.notebook.backends import (
        subprocess_kernel as sk_mod,
    )

    ws = str(tmp_path)
    captured = {}

    class _StopSpawn(Exception):
        pass

    def _fake_popen(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        raise _StopSpawn

    monkeypatch.setattr(sk_mod.subprocess, "Popen", _fake_popen)
    # No jaato tools to expose either (#1273), so nothing forces an env.
    monkeypatch.setattr(sk_mod, "jaato_tools_dir", lambda: None)
    backend = sk_mod.SubprocessKernelBackend()
    backend.initialize({"workspace_root": ws})
    with pytest.raises(_StopSpawn):
        backend.create_notebook("nb")
    assert captured["env"] is None


# --------------------------------------------------------------------------
# Default-on for daemon-managed workspaces, opt-in elsewhere (§8)
# --------------------------------------------------------------------------

def test_managed_workspace_defaults_home_on():
    # A workspace under the daemon's provisioning root defaults .home on.
    assert effective_workspace_home(
        None, "/root/sessions/s1", "/root",
    ) == DEFAULT_WORKSPACE_HOME


def test_unmanaged_workspace_is_opt_in_only():
    # A session pointed at a user's own checkout (managed root None, or
    # outside it) never grows a .home unless it asked for one.
    assert effective_workspace_home(None, "/home/u/proj", "/root") is None
    assert effective_workspace_home(None, "/home/u/proj", None) is None


def test_explicit_config_wins_and_can_opt_out():
    assert effective_workspace_home(
        {"workspace_home": ".h"}, "/root/s", "/root",
    ) == ".h"
    # A blank explicit value opts out, exactly as workspace_venv does, and
    # the managed default is NOT layered back on top.
    assert effective_workspace_home(
        {"workspace_home": ""}, "/root/s", "/root",
    ) is None


def test_inject_mirrors_the_knob_across_the_three_surfaces():
    plugin_configs = {}
    raw = inject_workspace_home(plugin_configs, "/root/s1", "/root")
    assert raw == DEFAULT_WORKSPACE_HOME
    for surface in ("cli", "interactive_shell", "notebook"):
        assert plugin_configs[surface]["workspace_home"] == DEFAULT_WORKSPACE_HOME


def test_inject_never_overwrites_an_explicit_per_surface_value():
    plugin_configs = {
        "cli": {"workspace_home": ".home"},
        "notebook": {"workspace_home": "/custom/home"},
    }
    inject_workspace_home(plugin_configs, "/root/s1", "/root")
    # cli's explicit value is mirrored to interactive_shell; notebook keeps
    # its own.
    assert plugin_configs["interactive_shell"]["workspace_home"] == ".home"
    assert plugin_configs["notebook"]["workspace_home"] == "/custom/home"


# --------------------------------------------------------------------------
# The daemon creates the directory + gitignore before spawn (#1171 pattern)
# --------------------------------------------------------------------------

def test_ensure_workspace_home_creates_dir_and_gitignore(tmp_path):
    home = str(tmp_path / ".home")
    ensure_workspace_home(home)
    assert os.path.isdir(home)
    with open(os.path.join(home, ".gitignore"), encoding="utf-8") as fh:
        assert fh.read() == "*\n"


def test_ensure_workspace_home_dir_applies_managed_default(tmp_path):
    # A profile-less (None) daemon-managed session -- the bare-.env shape a
    # web-created workspace has -- gets .home created under it.
    ws = str(tmp_path)
    created = ensure_workspace_home_dir(None, ws, managed_workspace_root=ws)
    assert created == os.path.join(os.path.realpath(ws), ".home")
    assert os.path.isdir(created)


def test_ensure_workspace_home_dir_skips_unmanaged(tmp_path):
    ws = str(tmp_path)
    assert ensure_workspace_home_dir(None, ws, managed_workspace_root=None) is None
    assert not os.path.isdir(os.path.join(ws, ".home"))


# --------------------------------------------------------------------------
# .home appears in neither `git status` nor the Files panel
# --------------------------------------------------------------------------

def _git(args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
    )


def test_workspace_home_is_kept_out_of_git(tmp_path):
    ws = tmp_path
    if _git(["init"], str(ws)).returncode != 0:
        pytest.skip("git not available")
    _git(["config", "user.email", "t@t"], str(ws))
    _git(["config", "user.name", "t"], str(ws))
    home = ws / ".home"
    home.mkdir()
    ensure_workspace_home(str(home))
    (home / ".gitconfig").write_text("[user]\n\tname = agent\n")

    # git check-ignore reports the tool's dotfile ignored, via the nested
    # `*` gitignore -- the user's own .gitignore is never edited.
    r = _git(["check-ignore", ".home/.gitconfig"], str(ws))
    assert r.returncode == 0 and ".home/.gitconfig" in r.stdout

    # And `git status` lists nothing under .home (a directory of only-ignored
    # files is not shown).
    status = _git(["status", "--porcelain"], str(ws)).stdout
    assert ".home" not in status


def test_workspace_monitor_hides_the_home_from_the_files_panel(tmp_path):
    from jaato_server.shared.utils.gitignore import GitignoreParser
    from jaato_server.server.workspace_monitor import _WORKSPACE_HOME_IGNORE

    ws = tmp_path
    (ws / ".home").mkdir()
    (ws / ".home" / ".gitconfig").write_text("x")
    (ws / "real.py").write_text("y")

    parser = GitignoreParser(
        ws, include_defaults=True, extra_patterns=list(_WORKSPACE_HOME_IGNORE),
    )
    assert parser.is_ignored(ws / ".home" / ".gitconfig")
    assert parser.is_ignored(ws / ".home")
    # A genuine project file is still surfaced.
    assert not parser.is_ignored(ws / "real.py")
