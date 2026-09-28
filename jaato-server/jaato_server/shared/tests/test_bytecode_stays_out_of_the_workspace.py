"""Running a project's Python leaves no bytecode in the workspace.

A model-driven ``python`` / ``pytest`` writes ``__pycache__/*.pyc`` beside
every module it imports.  A project that does not ignore them gets them in
``git status`` and in the Files panel.  With a workspace HOME in force,
``apply_home_to_env`` sets ``PYTHONPYCACHEPREFIX`` to
``<home>/.cache/pycache``, so the bytecode lands under the home, which is
already kept out of git (its own ``*`` gitignore) and out of the panel.

The guard runs a real interpreter through the real ``cli`` plugin, on both
execution paths, and looks at the filesystem afterwards.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

import pytest

from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin
from jaato_server.shared.tests.reversion import Reversion

_HOME = "jaato-server/jaato_server/shared/plugins/workspace_home.py"

REVERSIONS = [
    Reversion(
        target=_HOME,
        find='    env["PYTHONPYCACHEPREFIX"] = pycache_dir(home_path)\n',
        replace="",
        test="test_importing_a_module_writes_no_pycache_in_the_workspace",
        because="without the prefix the interpreter writes __pycache__ beside the source",
    ),
]

pytestmark = pytest.mark.skipif(shutil.which("python3") is None, reason="needs python3 on PATH")

_COMMAND = "python3 -c 'import app, pkg.mod'"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "app.py").write_text("X = 1\n")
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "mod.py").write_text("Y = 2\n")
    return tmp_path


def _cli(ws: Path, home: bool = True) -> CLIToolPlugin:
    plugin = CLIToolPlugin()
    config = {"workspace_root": str(ws)}
    if home:
        config["workspace_home"] = ".home"
    plugin.initialize(config)
    return plugin


def _stray_bytecode(ws: Path) -> List[Path]:
    return [p for p in ws.rglob("*") if p.name == "__pycache__" or p.suffix == ".pyc"
            if ".home" not in p.relative_to(ws).parts]


def test_importing_a_module_writes_no_pycache_in_the_workspace(workspace):
    plugin = _cli(workspace)
    try:
        result = plugin._execute({"command": f"cd repo && {_COMMAND}"})
    finally:
        plugin.shutdown()
    assert result.get("returncode") == 0, result
    assert _stray_bytecode(workspace) == []
    cached = list((workspace / ".home" / ".cache" / "pycache").rglob("*.pyc"))
    assert any(p.name.startswith("mod.") for p in cached), "bytecode is still cached, under the home"


def test_the_streaming_path_redirects_too(workspace):
    plugin = _cli(workspace)
    try:
        result = plugin._execute_streaming(
            {"command": f"cd repo && {_COMMAND}"},
            on_stdout=lambda _b: None, on_stderr=lambda _b: None, on_returncode=lambda _c: None,
        )
    finally:
        plugin.shutdown()
    assert result.get("returncode") == 0, result
    assert _stray_bytecode(workspace) == []


def test_without_a_workspace_home_python_behaves_as_before(workspace, monkeypatch):
    """The control: no home, no redirect, so the probe is not vacuous."""
    monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    plugin = _cli(workspace, home=False)
    try:
        result = plugin._execute({"command": f"cd repo && {_COMMAND}"})
    finally:
        plugin.shutdown()
    assert result.get("returncode") == 0, result
    assert _stray_bytecode(workspace), "an unredirected interpreter writes __pycache__"
