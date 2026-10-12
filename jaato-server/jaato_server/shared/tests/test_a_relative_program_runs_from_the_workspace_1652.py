"""A relative program path resolves against the command's cwd (#1652).

In argv mode (a command needing no shell) ``cli`` resolved the executable
with ``shutil.which(exe, path=PATH)``.  For a word with a directory
component ``which`` ignores ``path`` and tests the word relative to the
CALLING process's cwd, which is the runner's, not the workspace the command
runs in.  So ``pipeline/venv/bin/python -c ...`` was refused as "not found in
PATH" right after containment (which judges against the subprocess cwd) had
allowed it, while the same command behind ``&&`` ran through the shell.

These tests run a workspace program by relative path, from a test process
whose cwd is NOT the workspace, through both cli paths and through
``run_command`` directly.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import List

import pytest

from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin
from jaato_server.shared.subprocess_runner import resolve_executable, run_command
from jaato_server.shared.tests.reversion import Reversion

_RUNNER = "jaato-server/jaato_server/shared/subprocess_runner.py"

REVERSIONS = [
    Reversion(
        target=_RUNNER,
        find="    if has_dir and cwd and not os.path.isabs(exe):\n",
        replace="    if False:\n",
        test="test_both_cli_paths_run_a_relative_workspace_program",
        because="a path-shaped word is resolved against the runner's cwd again",
    ),
    Reversion(
        target=_RUNNER,
        find="        argv = _resolve_argv(command, env, cwd)\n",
        replace="        argv = _resolve_argv(command, env)\n",
        test="test_run_command_resolves_against_its_cwd",
        because="run_command stops passing its cwd to the resolution",
    ),
]


def _write_program(root: Path) -> Path:
    bin_dir = root / "pipeline" / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    prog = bin_dir / "probe"
    prog.write_text("#!/bin/sh\necho ran-$1\n")
    prog.chmod(prog.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return prog


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_program(ws)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return ws


@pytest.fixture
def cli(workspace: Path):
    plugin = CLIToolPlugin()
    plugin.initialize({
        "workspace_root": str(workspace),
        "extra_paths": [str(workspace / "pipeline" / "venv" / "bin")],
    })
    yield plugin
    plugin.shutdown()


def test_both_cli_paths_run_a_relative_workspace_program(cli):
    command = "pipeline/venv/bin/probe ok"

    result = cli._execute({"command": command})
    assert result.get("returncode") == 0, result
    assert result["stdout"].strip() == "ran-ok"

    out: List[bytes] = []
    result = cli._execute_streaming(
        {"command": command},
        on_stdout=out.append,
        on_stderr=lambda _b: None,
        on_returncode=lambda _c: None,
    )
    assert "error" not in result, result
    assert b"".join(out).decode().strip() == "ran-ok"


def test_run_command_resolves_against_its_cwd(workspace):
    result = run_command("pipeline/venv/bin/probe ok", cwd=str(workspace))
    assert result.returncode == 0, result
    assert result.stdout.strip() == "ran-ok"


def test_bare_names_and_absolute_paths_are_unchanged(workspace):
    sh = resolve_executable("sh", os.environ.get("PATH"), str(workspace))
    assert sh and os.path.isabs(sh)
    prog = str(workspace / "pipeline" / "venv" / "bin" / "probe")
    assert resolve_executable(prog, None, "/nonexistent") == prog
    # A bare name is never looked up in cwd.
    assert resolve_executable("probe", "", str(workspace / "pipeline" / "venv" / "bin")) is None


def test_a_relative_path_missing_from_cwd_is_not_found(workspace, tmp_path):
    assert resolve_executable(
        "pipeline/venv/bin/probe", os.environ.get("PATH"), str(tmp_path),
    ) is None
