"""The Files panel's ignore decision agrees with git inside a checkout.

The workspace monitor filtered paths through ``GitignoreParser``, which reads
only ``<workspace>/.gitignore``.  A clone under ``<ws>/<name>`` has its own
``.gitignore`` and ``.git/info/exclude`` the parser never saw, so a clone's
``node_modules/`` and ``build/`` showed in the panel.  ``GitIgnoreOracle``
now answers for a path inside a checkout by asking that checkout's own git.

The parser still wins where it force-hides (``.home/``, ``.tmp/``, ``.git/``)
and for the workspace-root ``.gitignore``; git is only consulted for a path
the parser lets through.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from jaato_server.server.workspace_gitignore_query import GitIgnoreOracle
from jaato_server.server.workspace_monitor import WorkspaceMonitor
from jaato_server.shared.tests.reversion import Reversion

_MON = "jaato-server/jaato_server/server/workspace_monitor.py"

REVERSIONS = [
    Reversion(
        target=_MON,
        find=(
            "        if self._git_ignore is not None:\n"
            "            verdict = self._git_ignore.is_ignored(abs_path)\n"
            "            if verdict is not None:\n"
            "                return verdict\n"
        ),
        replace="",
        test="test_a_clone_gitignore_hides_its_own_build_output",
        because="without asking git, a clone's own .gitignore is invisible to the panel",
    ),
]

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _init(repo: Path, *, gitignore: str = "") -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    if gitignore:
        (repo / ".gitignore").write_text(gitignore)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    # A clone under the workspace with its own ignore rules; the workspace
    # root is NOT itself a checkout.
    clone = tmp_path / "project"
    _init(clone, gitignore="node_modules/\n*.log\n")
    (clone / "node_modules" / "pkg").mkdir(parents=True)
    (clone / "node_modules" / "pkg" / "index.js").write_text("//\n")
    (clone / "src").mkdir()
    (clone / "src" / "main.py").write_text("x = 1\n")
    (clone / "run.log").write_text("noise\n")
    return tmp_path


def _oracle(ws: Path) -> GitIgnoreOracle:
    return GitIgnoreOracle(str(ws))


def test_oracle_matches_git_inside_a_clone(workspace):
    oracle = _oracle(workspace)
    try:
        clone = workspace / "project"
        assert oracle.is_ignored(str(clone / "node_modules" / "pkg" / "index.js")) is True
        assert oracle.is_ignored(str(clone / "run.log")) is True
        assert oracle.is_ignored(str(clone / "src" / "main.py")) is False
        # A path in no checkout is git's to decline.
        assert oracle.is_ignored(str(workspace / "loose.txt")) is None
    finally:
        oracle.close()


def test_a_negated_pattern_is_not_ignored(tmp_path):
    clone = tmp_path / "p"
    _init(clone, gitignore="*.pyc\n!keep.pyc\n")
    oracle = _oracle(tmp_path)
    try:
        assert oracle.is_ignored(str(clone / "a.pyc")) is True
        assert oracle.is_ignored(str(clone / "keep.pyc")) is False
    finally:
        oracle.close()


def test_info_exclude_is_honoured(tmp_path):
    clone = tmp_path / "p"
    _init(clone)
    (clone / ".git" / "info" / "exclude").write_text("secret/\n")
    (clone / "secret").mkdir()
    oracle = _oracle(tmp_path)
    try:
        assert oracle.is_ignored(str(clone / "secret" / "x")) is True
    finally:
        oracle.close()


def test_a_clone_appearing_later_is_picked_up(tmp_path):
    oracle = _oracle(tmp_path)
    try:
        clone = tmp_path / "late"
        _init(clone, gitignore="dist/\n")
        (clone / "dist").mkdir()
        # The first query for a path it cannot place triggers a rescan.
        assert oracle.is_ignored(str(clone / "dist" / "o")) is True
    finally:
        oracle.close()


def test_a_repo_config_hook_does_not_run(tmp_path):
    """A repo-local git config must not make a read-only query run a program.

    ``core.fsmonitor`` names a program git would run; the oracle neutralises
    it on the command line, so the marker file is never created.
    """
    clone = tmp_path / "p"
    _init(clone, gitignore="x/\n")
    marker = tmp_path / "ran"
    hook = tmp_path / "hook.sh"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
    hook.chmod(0o755)
    subprocess.run(["git", "-C", str(clone), "config", "core.fsmonitor", str(hook)], check=True)
    oracle = _oracle(tmp_path)
    try:
        (clone / "x").mkdir()
        oracle.is_ignored(str(clone / "x" / "f"))
    finally:
        oracle.close()
    assert not marker.exists(), "the repo's configured program must not have run"


def test_a_clone_gitignore_hides_its_own_build_output(tmp_path):
    """End to end through the monitor: the parser lets it through, git catches it."""
    clone = tmp_path / "proj"
    _init(clone, gitignore="build/\n")
    (clone / "build").mkdir()
    (clone / "build" / "artifact.o").write_text("x\n")
    monitor = WorkspaceMonitor(str(tmp_path), on_changed=lambda c: None)
    try:
        assert monitor._is_ignored(str(clone / "build" / "artifact.o")) is True
        assert monitor._is_ignored(str(clone / "build"), is_dir=True) is True
    finally:
        monitor.stop()


def test_the_home_extras_still_win_in_a_root_checkout(tmp_path):
    """A root checkout whose .gitignore omits .home must still hide it."""
    _init(tmp_path, gitignore="# nothing about .home\n")
    home = tmp_path / ".home" / ".cache"
    home.mkdir(parents=True)
    (home / "x").write_text("x\n")
    monitor = WorkspaceMonitor(str(tmp_path), on_changed=lambda c: None)
    try:
        # git in the root checkout would say .home is NOT ignored; the
        # parser's force-hide wins first.
        assert monitor._is_ignored(str(home / "x")) is True
    finally:
        monitor.stop()
