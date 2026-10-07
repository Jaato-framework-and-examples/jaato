"""#1273 / #1274 -- what a model-driven subprocess finds when it reaches for a tool.

Reported from a web session: the ``jaato-sdk`` skill told the model to run
``jaato-doctor``, and ``which jaato-doctor`` found nothing.  The daemon ran
as ``<venv>/bin/python -m jaato_server`` without its venv activated, so its
console scripts were on no ``PATH`` a subprocess inherits -- and #1202 lets
the model name a program by path only when that directory IS on the PATH.

Three answers, each attached to the tool class it serves:

* **jaato's own tools** (#1273): an allow-listed symlink directory APPENDED
  to the PATH -- ``jaato-doctor`` and ``jaato-scaffold``, never the daemon
  venv's ``python`` / ``pip`` (which would let a model's ``pip install``
  mutate the daemon's own environment) and never ``jaato-server``.
* **tools the model installs** (#1273): ``uv tool`` / ``pipx`` /
  ``pip --user`` put programs in ``<home>/.local/bin``; with HOME redirected
  per workspace (#1225) that directory is now appended too, with the
  AppArmor ``ix`` grants that let them run under confinement.
* **python / pip / uv pip** (#1274): a daemon-managed workspace gets a tool
  venv by default (``.jaato/tool-venv``), the rule #1225 applies to the
  home.  Before, a profile-less web workspace -- every one the web client
  creates -- had none, so ``pip install`` ran the HOST's pip, as root on a
  root daemon.  Folded into the envelope, the AppArmor rules and the Files
  panel's ignore list, so the venv is used, allowed to run, and not shown.

APPENDED is the security property throughout: an entry added here supplies
a name the PATH lacks and can never shadow one it already has.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jaato_server.shared.plugins import jaato_tools_path as tools_mod
from jaato_server.shared.plugins.jaato_tools_path import append_path_entry
from jaato_server.shared.plugins.workspace_home import (
    apply_home_to_env,
    home_exec_apparmor_rules,
)
from jaato_server.shared.plugins.workspace_venv import (
    DEFAULT_WORKSPACE_VENV,
    VENV_SURFACES,
    effective_workspace_venv,
    inject_workspace_venv,
)
from jaato_server.shared.tests.reversion import Reversion

_TOOLS = "jaato-server/jaato_server/shared/plugins/jaato_tools_path.py"
_HOME = "jaato-server/jaato_server/shared/plugins/workspace_home.py"
_VENV = "jaato-server/jaato_server/shared/plugins/workspace_venv.py"
_APPARMOR = "jaato-server/jaato_server/server/apparmor.py"
_CLI = "jaato-server/jaato_server/shared/plugins/cli/plugin.py"

REVERSIONS = [
    Reversion(
        target=_TOOLS,
        find='JAATO_TOOL_NAMES = ("jaato-doctor", "jaato-scaffold")',
        replace='JAATO_TOOL_NAMES = ("jaato-doctor", "jaato-scaffold", "python", "pip")',
        test="test_only_the_allow_listed_tools_are_exposed",
        because="exposing the daemon venv's python/pip lets a model's pip install mutate the daemon's own environment",
    ),
    Reversion(
        target=_TOOLS,
        find='''    env["PATH"] = f"{current}{os.pathsep}{directory}" if current else directory''',
        replace='''    env["PATH"] = f"{directory}{os.pathsep}{current}" if current else directory''',
        test="test_an_appended_entry_never_shadows_an_existing_binary",
        because="a prepended directory would shadow a host binary of the same name",
    ),
    Reversion(
        target=_CLI,
        find='''        apply_jaato_tools_to_env(env)
''',
        replace='''''',
        test="test_a_cli_command_finds_jaato_doctor_by_name",
        because="without it the skill's jaato-doctor is on no PATH a command inherits",
    ),
    Reversion(
        target=_HOME,
        find='''    append_path_entry(env, local_bin_dir(home_path))
''',
        replace='''''',
        test="test_programs_installed_under_the_workspace_home_resolve_by_name",
        because="uv tool / pipx / pip --user install into <home>/.local/bin, which was on no PATH",
    ),
    Reversion(
        target=_VENV,
        find='''    if is_daemon_managed(workspace_path, managed_workspace_root):
        return DEFAULT_WORKSPACE_VENV
    return None''',
        replace='''    return None''',
        test="test_a_managed_workspace_gets_a_tool_venv_by_default",
        because="a profile-less managed workspace has no plugin_configs channel, so pip runs the host's interpreter",
    ),
    Reversion(
        target=_APPARMOR,
        find='''    venv = inject_workspace_venv(configs, workspace_path, managed_workspace_root)
''',
        replace='''    venv = None
''',
        test="test_the_apparmor_rules_follow_the_managed_venv",
        because="a venv the envelope carries but the profile does not grant ix on cannot run under confinement",
    ),
]


# --------------------------------------------------------------------------
# jaato's own tools
# --------------------------------------------------------------------------

@pytest.fixture
def fake_scripts(tmp_path, monkeypatch):
    """A console-script dir holding jaato's tools AND the ones that must stay hidden."""
    scripts = tmp_path / "venv-bin"
    scripts.mkdir()
    for name in ("jaato-doctor", "jaato-scaffold", "jaato-server", "jaato", "python", "pip"):
        exe = scripts / name
        exe.write_text("#!/bin/sh\necho " + name + "\n")
        exe.chmod(0o755)
    tmpdir = tmp_path / "session-tmp"
    tmpdir.mkdir()
    monkeypatch.setattr(tools_mod, "_scripts_dir", lambda: str(scripts))
    monkeypatch.setattr(tools_mod.tempfile, "gettempdir", lambda: str(tmpdir))
    monkeypatch.setattr(tools_mod, "_built", {})
    return scripts


def test_only_the_allow_listed_tools_are_exposed(fake_scripts):
    directory = tools_mod.jaato_tools_dir()
    assert directory is not None
    # Spelled out rather than read from the constant under test: a guard
    # that compares the directory with the allow-list it is guarding passes
    # whatever the allow-list says.
    exposed = set(os.listdir(directory))
    assert exposed == {"jaato-doctor", "jaato-scaffold"}
    assert not exposed & {"python", "pip", "jaato-server", "jaato"}
    # Each is a symlink to the interpreter's own script, so AppArmor judges
    # the exec on the target the runner profile already allows.
    for name in sorted(exposed):
        link = os.path.join(directory, name)
        assert os.path.islink(link)
        assert os.readlink(link) == str(fake_scripts / name)


def test_the_daemon_venv_bin_itself_never_reaches_the_path(fake_scripts, tmp_path, monkeypatch):
    from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    plugin = CLIToolPlugin()
    plugin.initialize({"workspace_root": str(tmp_path)})
    env, _ = plugin._build_subprocess_env()
    assert str(fake_scripts) not in env["PATH"].split(os.pathsep)


def test_a_cli_command_finds_jaato_doctor_by_name(fake_scripts, tmp_path, monkeypatch):
    import shutil
    from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    plugin = CLIToolPlugin()
    plugin.initialize({"workspace_root": str(tmp_path)})
    env, _ = plugin._build_subprocess_env()
    found = shutil.which("jaato-doctor", path=env["PATH"])
    assert found is not None
    assert os.path.realpath(found) == str(fake_scripts / "jaato-doctor")
    # Appended: the host PATH still comes first.
    assert env["PATH"].startswith("/usr/bin:/bin")


def test_no_tools_installed_exposes_nothing(tmp_path, monkeypatch):
    """A daemon run from a checkout via PYTHONPATH has no console scripts."""
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setattr(tools_mod, "_scripts_dir", lambda: str(empty))
    monkeypatch.setattr(tools_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(tools_mod, "_built", {})
    assert tools_mod.jaato_tools_dir() is None
    env = {"PATH": "/usr/bin"}
    tools_mod.apply_jaato_tools_to_env(env)
    assert env["PATH"] == "/usr/bin"


def test_a_reaped_tmpdir_is_rebuilt(fake_scripts):
    import shutil
    first = tools_mod.jaato_tools_dir()
    shutil.rmtree(first)
    assert tools_mod.jaato_tools_dir() == first
    assert os.path.islink(os.path.join(first, "jaato-doctor"))


def test_an_appended_entry_never_shadows_an_existing_binary():
    env = {"PATH": "/usr/bin:/bin"}
    append_path_entry(env, "/extra")
    assert env["PATH"] == "/usr/bin:/bin:/extra"
    append_path_entry(env, "/extra")
    assert env["PATH"] == "/usr/bin:/bin:/extra", "appended twice"
    empty: dict = {}
    append_path_entry(empty, "/extra")
    assert empty["PATH"] == "/extra"


# --------------------------------------------------------------------------
# Tools the model installs under the workspace HOME
# --------------------------------------------------------------------------

def test_programs_installed_under_the_workspace_home_resolve_by_name():
    env = {"PATH": "/usr/bin:/bin"}
    apply_home_to_env(env, "/ws/.home")
    assert env["PATH"] == "/usr/bin:/bin:" + os.path.join("/ws/.home", ".local", "bin")


def test_the_home_bin_dirs_are_granted_exec_under_apparmor():
    rules = home_exec_apparmor_rules(".home", "/ws")
    assert "/ws/.home/.local/bin/* ix," in rules
    # Where the uv tool / pipx symlinks in .local/bin resolve to.
    assert "/ws/.home/.local/share/**/bin/* ix," in rules
    assert home_exec_apparmor_rules(None, "/ws") == []
    assert home_exec_apparmor_rules(".home", None) == []


def test_the_cli_contributes_the_home_exec_grant():
    from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin

    rules = CLIToolPlugin.get_apparmor_rules(
        workspace_path="/ws", session_id="s", config_root=None,
        plugin_config={"workspace_home": ".home"},
    )
    assert "/ws/.home/.local/bin/* ix," in rules


# --------------------------------------------------------------------------
# The managed-default tool venv
# --------------------------------------------------------------------------

def test_a_managed_workspace_gets_a_tool_venv_by_default(tmp_path):
    root = tmp_path / "workspaces"
    ws = root / "mine"
    ws.mkdir(parents=True)
    assert effective_workspace_venv(None, str(ws), str(root)) == DEFAULT_WORKSPACE_VENV


def test_an_unmanaged_workspace_gets_none(tmp_path):
    assert effective_workspace_venv(None, str(tmp_path), None) is None
    other = tmp_path / "elsewhere"
    other.mkdir()
    assert effective_workspace_venv(None, str(other), str(tmp_path / "workspaces")) is None


def test_a_profile_decides_first_and_blank_opts_out(tmp_path):
    root = tmp_path / "workspaces"
    ws = root / "mine"
    ws.mkdir(parents=True)
    assert effective_workspace_venv({"workspace_venv": ""}, str(ws), str(root)) is None
    assert effective_workspace_venv({"workspace_venv": "v"}, str(ws), str(root)) is None


def test_inject_fills_every_surface_and_keeps_explicit_values(tmp_path):
    root = tmp_path / "workspaces"
    ws = root / "mine"
    ws.mkdir(parents=True)
    configs = {"notebook": {"workspace_venv": "nb-venv"}}
    assert inject_workspace_venv(configs, str(ws), str(root)) == DEFAULT_WORKSPACE_VENV
    assert configs["cli"]["workspace_venv"] == DEFAULT_WORKSPACE_VENV
    assert configs["interactive_shell"]["workspace_venv"] == DEFAULT_WORKSPACE_VENV
    assert configs["notebook"]["workspace_venv"] == "nb-venv"
    assert set(VENV_SURFACES) <= set(configs)


def test_an_explicit_cli_venv_is_not_mirrored(tmp_path):
    """The pre-#1274 meaning of a cli-only value is kept."""
    root = tmp_path / "workspaces"
    ws = root / "mine"
    ws.mkdir(parents=True)
    configs = {"cli": {"workspace_venv": "custom"}}
    assert inject_workspace_venv(configs, str(ws), str(root)) is None
    assert "interactive_shell" not in configs


def test_the_envelope_builder_folds_the_managed_venv():
    """AST: ``build_session_envelope`` calls the fold beside the home's.

    A call-site check, because the defect this guards is a fold that exists
    and is never invoked -- the #1133 shape.
    """
    src = (Path(__file__).resolve().parents[2] / "server" / "runner_spawn.py").read_text()
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "build_session_envelope"
    )
    called = {
        c.func.id for c in ast.walk(fn)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    }
    assert "inject_workspace_venv" in called
    assert "inject_workspace_home" in called


def test_the_apparmor_rules_follow_the_managed_venv(tmp_path):
    from jaato_server.server.apparmor import resolve_plugin_apparmor_rules
    from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin

    root = tmp_path / "workspaces"
    ws = root / "mine"
    ws.mkdir(parents=True)
    registry = MagicMock()
    registry.all_plugins.return_value = {"cli": CLIToolPlugin}
    server = MagicMock()
    server.registry = registry

    # A profile-less session on a managed workspace: the web client's shape.
    rules = resolve_plugin_apparmor_rules(
        server=server, profile=None, session_id="s",
        workspace_path=str(ws), config_root=None,
        managed_workspace_root=str(root),
    ) or []
    venv_bin = os.path.join(os.path.realpath(str(ws)), DEFAULT_WORKSPACE_VENV, "bin")
    assert f"{venv_bin}/* ix," in rules
    assert any(r.endswith("/.home/.local/bin/* ix,") for r in rules)

    # Unmanaged and profile-less: unchanged, nothing to grant.
    assert resolve_plugin_apparmor_rules(
        server=server, profile=None, session_id="s",
        workspace_path=str(ws), config_root=None,
    ) is None


def test_the_resolver_does_not_write_into_the_profile(tmp_path):
    from jaato_server.server.apparmor import resolve_plugin_apparmor_rules

    root = tmp_path / "workspaces"
    ws = root / "mine"
    ws.mkdir(parents=True)
    profile = MagicMock()
    profile.plugin_configs = {"cli": {"max_output_chars": 10}}
    profile.gc = None
    registry = MagicMock()
    registry.all_plugins.return_value = {}
    server = MagicMock()
    server.registry = registry
    resolve_plugin_apparmor_rules(
        server=server, profile=profile, session_id="s",
        workspace_path=str(ws), config_root=None,
        managed_workspace_root=str(root),
    )
    assert profile.plugin_configs == {"cli": {"max_output_chars": 10}}


def test_the_files_panel_hides_the_tool_venv(tmp_path):
    from jaato_server.shared.utils.gitignore import GitignoreParser
    from jaato_server.server.workspace_monitor import _WORKSPACE_HOME_IGNORE

    venv = tmp_path / DEFAULT_WORKSPACE_VENV / "lib"
    venv.mkdir(parents=True)
    (venv / "six.py").write_text("x")
    (tmp_path / "real.py").write_text("y")
    parser = GitignoreParser(
        tmp_path, include_defaults=True, extra_patterns=list(_WORKSPACE_HOME_IGNORE),
    )
    assert parser.is_ignored(venv / "six.py")
    assert not parser.is_ignored(tmp_path / "real.py")
