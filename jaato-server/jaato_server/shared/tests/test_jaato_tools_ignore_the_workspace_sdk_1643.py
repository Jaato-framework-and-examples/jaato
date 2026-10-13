"""#1643 -- a jaato tool inside a session describes the installed framework.

#1273 put ``jaato-doctor`` and ``jaato-scaffold`` on a session's ``PATH`` as
symlinks to the daemon venv's console scripts.  A symlink cannot reset the
environment, and ``apply_venv_to_env`` puts the workspace venv's
site-packages first on ``PYTHONPATH``.  An agent writing a cascade driver
installs ``jaato-sdk`` there, and from then on the daemon's own
``jaato-scaffold`` imported the WORKSPACE's ``jaato_sdk``: exit 0, three
plugins missing from ``explain``, the cause only on stderr.

The fix has two halves, and each case below pins one:

* the runner stamps ``JAATO_TOOLS_TRUSTED_PYTHONPATH`` -- its OWN
  ``PYTHONPATH``, never the one it is building -- wherever it exposes the
  tools (cli, interactive_shell, the notebook kernel);
* the console scripts start in ``jaato_tool_launch`` (jaato-sdk, top level),
  which drops every other ``PYTHONPATH`` entry from ``sys.path`` before
  ``jaato_sdk`` is imported.

The shadow used here is the real ``jaato_sdk`` with one name removed from
``plugins/base.py`` -- the issue's failure (``cannot import name
'RESOLVED_MENTIONS_METADATA_KEY'``) without depending on an old release
being downloadable.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import jaato_sdk
import jaato_tool_launch
from jaato_server.shared.plugins import jaato_tools_path as tools_mod
from jaato_server.shared.tests.reversion import Reversion

_LAUNCH = "jaato-sdk/jaato_tool_launch.py"
_SDK_TOML = "jaato-sdk/pyproject.toml"
_TOOLS = "jaato-server/jaato_server/shared/plugins/jaato_tools_path.py"
_KERNEL = "jaato-server/jaato_server/shared/plugins/notebook/backends/subprocess_kernel.py"

MARKER = "JAATO_TOOLS_TRUSTED_PYTHONPATH"

REVERSIONS = [
    Reversion(
        target=_LAUNCH,
        find='''    return given - trusted - _own_dirs()''',
        replace='''    return set()''',
        test="test_a_shadowing_sdk_does_not_change_what_scaffold_describes",
        because="with nothing dropped, the workspace's jaato_sdk answers for the daemon's tool and plugins vanish with exit 0",
    ),
    Reversion(
        target=_SDK_TOML,
        find='''jaato-scaffold = "jaato_tool_launch:scaffold"''',
        replace='''jaato-scaffold = "jaato_sdk.scaffold.cli:main"''',
        test="test_both_console_scripts_start_in_the_launcher",
        because="a console script importing jaato_sdk first has loaded the shadow's __init__ before anything can isolate",
    ),
    Reversion(
        target=_TOOLS,
        find='''        append_path_entry(env, directory)
        stamp_trusted_pythonpath(env)''',
        replace='''        append_path_entry(env, directory)''',
        test="test_a_cli_command_carries_the_runners_own_pythonpath_as_trusted",
        because="with no marker the launcher cannot tell a session's launch from a developer's, and isolates nothing",
    ),
    Reversion(
        target=_TOOLS,
        find='''    env[TRUSTED_PYTHONPATH_ENV] = os.environ.get("PYTHONPATH", "")''',
        replace='''    env[TRUSTED_PYTHONPATH_ENV] = env.get("PYTHONPATH", "")''',
        test="test_a_cli_command_carries_the_runners_own_pythonpath_as_trusted",
        because="read from the env being built, the trusted value already holds the workspace venv's site-packages",
    ),
    Reversion(
        target=_KERNEL,
        find='''            stamp_trusted_pythonpath(kernel_env)
''',
        replace='''''',
        test="test_the_notebook_kernel_stamps_where_it_exposes_the_tools",
        because="a cell's !jaato-scaffold would import the workspace venv's jaato_sdk",
    ),
]


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "jaato-sdk" / "pyproject.toml").is_file():
            return parent
    raise AssertionError("repository root not found above " + str(here))


@pytest.fixture
def shadow_sdk(tmp_path) -> Path:
    """A ``jaato_sdk`` one release behind: a name its plugins import is gone."""
    root = tmp_path / "workspace-site-packages"
    root.mkdir()
    src = Path(jaato_sdk.__file__).parent
    dest = root / "jaato_sdk"
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__"))
    base = dest / "plugins" / "base.py"
    text = base.read_text()
    assert "RESOLVED_MENTIONS_METADATA_KEY =" in text, "the shadow must lack a name the plugins import"
    base.write_text(text.replace("RESOLVED_MENTIONS_METADATA_KEY =", "_GONE_IN_THIS_RELEASE ="))
    return root


def _session_env(shadow: Path) -> dict:
    """What a session's command sees: the shadow first, the runner's own after it.

    The runner's own ``PYTHONPATH`` is this process's -- empty in a normal
    run, the sandbox's import roots under the reversion meta-guard -- and is
    what the marker says to keep.
    """
    env = os.environ.copy()
    own = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(shadow) + (os.pathsep + own if own else "")
    env[MARKER] = own
    return env


def _console_script(tmp_path: Path, func: str) -> Path:
    """The script pip writes for ``jaato-<x> = "jaato_tool_launch:<func>"``."""
    script = tmp_path / f"jaato-{func}"
    script.write_text(
        "import sys\n"
        f"from jaato_tool_launch import {func}\n"
        f"sys.exit({func}())\n"
    )
    return script


def test_a_shadowing_sdk_does_not_change_what_scaffold_describes(shadow_sdk, tmp_path):
    script = _console_script(tmp_path, "scaffold")
    proc = subprocess.run(
        [sys.executable, str(script), "explain"],
        env=_session_env(shadow_sdk), cwd=str(tmp_path),
        capture_output=True, text=True, timeout=300,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined[-2000:]
    assert str(shadow_sdk) not in combined, (
        "jaato-scaffold imported the workspace's jaato_sdk:\n" + proc.stderr[-2000:])
    assert "Error loading entry point" not in proc.stderr, proc.stderr[-2000:]
    assert "plugins" in proc.stdout


def test_the_launcher_resolves_jaato_sdk_from_its_own_environment(shadow_sdk, tmp_path):
    probe = (
        "import jaato_tool_launch as L\n"
        "removed = L.isolate()\n"
        "import os, jaato_sdk\n"
        "print(jaato_sdk.__file__)\n"
        "print(repr(os.environ.get('PYTHONPATH')))\n"
        "print(removed)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], env=_session_env(shadow_sdk),
        cwd=str(tmp_path), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    sdk_file, pythonpath, removed = proc.stdout.splitlines()
    assert not sdk_file.startswith(str(shadow_sdk)), sdk_file
    # The tool's own children inherit the narrowed PYTHONPATH, not the shadow.
    assert str(shadow_sdk) not in pythonpath
    assert str(shadow_sdk) in removed


def test_without_the_marker_pythonpath_is_the_callers_choice(shadow_sdk):
    """Outside a session nothing changes: a developer's PYTHONPATH is honoured."""
    path = [str(shadow_sdk), "/somewhere/else"]
    environ = {"PYTHONPATH": str(shadow_sdk)}
    assert jaato_tool_launch.isolate(environ, path) == []
    assert path == [str(shadow_sdk), "/somewhere/else"]
    assert environ == {"PYTHONPATH": str(shadow_sdk)}


def test_trusted_entries_stay_and_only_untrusted_ones_go(tmp_path):
    own, workspace = str(tmp_path / "own"), str(tmp_path / "ws-site")
    environ = {"PYTHONPATH": os.pathsep.join([workspace, own]), MARKER: own + os.sep}
    path = ["/script-dir", workspace, own, "/lib/python3"]
    assert jaato_tool_launch.isolate(environ, path) == [workspace]
    assert path == ["/script-dir", own, "/lib/python3"]
    assert environ["PYTHONPATH"] == own
    # An empty marker keeps nothing, and PYTHONPATH goes with it.
    environ = {"PYTHONPATH": workspace, MARKER: ""}
    path = [workspace, "/lib/python3"]
    jaato_tool_launch.isolate(environ, path)
    assert path == ["/lib/python3"]
    assert "PYTHONPATH" not in environ


def test_both_console_scripts_start_in_the_launcher():
    scripts = tomllib.loads((_repo_root() / _SDK_TOML).read_text())["project"]["scripts"]
    assert scripts["jaato-scaffold"] == "jaato_tool_launch:scaffold"
    assert scripts["jaato-doctor"] == "jaato_tool_launch:doctor"


def test_a_cli_command_carries_the_runners_own_pythonpath_as_trusted(tmp_path, monkeypatch):
    from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin

    scripts = tmp_path / "venv-bin"
    scripts.mkdir()
    for name in tools_mod.JAATO_TOOL_NAMES:
        exe = scripts / name
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
    session_tmp = tmp_path / "session-tmp"
    session_tmp.mkdir()
    monkeypatch.setattr(tools_mod, "_scripts_dir", lambda: str(scripts))
    monkeypatch.setattr(tools_mod.tempfile, "gettempdir", lambda: str(session_tmp))
    monkeypatch.setattr(tools_mod, "_built", {})

    ws = tmp_path / "ws"
    venv = ws / ".venv"
    site = venv / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}" / "site-packages"
    site.mkdir(parents=True)
    (venv / "bin").mkdir()
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    monkeypatch.setenv("PYTHONPATH", "/runner/own")

    plugin = CLIToolPlugin()
    plugin.initialize({"workspace_root": str(ws), "workspace_venv": ".venv"})
    env, venv_path = plugin._build_subprocess_env()
    assert venv_path, "the workspace venv must be active for this case to mean anything"
    assert str(site) in env["PYTHONPATH"].split(os.pathsep)
    assert env[MARKER] == "/runner/own"


def test_the_notebook_kernel_stamps_where_it_exposes_the_tools():
    """Every module that puts the tools dir on a PATH also stamps the marker."""
    tree = ast.parse((_repo_root() / _KERNEL).read_text())
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "jaato_tools_dir" in called
    assert "stamp_trusted_pythonpath" in called
