"""Four concurrent ``session.new`` do not spend a minute walking (#1553).

A cascade driver created four sessions at once against warm pool slots.
Every runner was ready in 3-5 s; three of the four confirmations missed the
caller's 60 s budget anyway.  Each session's last runner RPC was
``get_auth_info`` (the end of ``JaatoServer.initialize``) and its next was
``get_history`` (the first save), ~45 s later, all four resuming together.

Between the two, ``_create_session_impl`` starts the workspace monitor,
whose ``start()`` walks the whole workspace to seed its baseline.  Since the
Files panel started asking git (``GitIgnoreOracle``), that walk paid, per
FILE: a ``pathlib`` ``relative_to`` per checkout, a full rescan of the
workspace root for every path outside a checkout, the parser's own
``relative_to`` + ``stat`` + every ancestor again, and one git round trip
read back one byte per Python call.  It is pure Python on one GIL, and
every blocking git read waited out the switch interval to get the GIL back
from the other walkers, so four walks took ~6x one walk each and finished
together.  Measured on a 12k-file workspace: 3.0 s alone, 18.2 s each with
four at once; after this change 0.2 s and 1.5 s.

The guards here are about the SHAPE of the work, not a clock (a timing
assertion is a bet on the CI machine):

* the walk gives the same answer as asking ``_is_ignored`` per path;
* git is asked once per directory listing, never once per file;
* a path outside every checkout does not rescan the workspace root;
* a directory larger than git's pipe buffer does not deadlock the batch;
* a clone that appears after the oracle was built is still found.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Set

import pytest

from jaato_server.server import workspace_gitignore_query as wq
from jaato_server.server.workspace_monitor import WorkspaceMonitor
from jaato_server.shared.tests.reversion import Reversion

_MON = "jaato-server/jaato_server/server/workspace_monitor.py"
_ORACLE = "jaato-server/jaato_server/server/workspace_gitignore_query.py"

REVERSIONS = [
    Reversion(
        target=_MON,
        find=(
            "        verdicts = self._git_ignore.is_ignored_many(\n"
            "            [os.path.join(dirpath, n) for n in names])\n"
        ),
        replace=(
            "        verdicts = [self._git_ignore.is_ignored("
            "os.path.join(dirpath, n)) for n in names]\n"
        ),
        test="test_git_is_asked_per_directory_not_per_file",
        because="one git round trip per file is what four walkers convoyed on",
    ),
    Reversion(
        target=_ORACLE,
        find=(
            "                checkout = self._checkout_for(abs_path)\n"
            "                if checkout is None and self._new_checkout_for(abs_path):\n"
        ),
        replace=(
            "                checkout = self._checkout_for(abs_path)\n"
            "                if checkout is None and (self._scan_checkouts() or True):\n"
        ),
        test="test_a_path_outside_every_checkout_does_not_rescan_the_root",
        because="rescanning the root for every non-checkout file was a listdir per file",
    ),
    Reversion(
        target=_ORACLE,
        find="_BATCH = 128\n",
        replace="_BATCH = 10 ** 9\n",
        test="test_a_directory_larger_than_the_pipe_does_not_deadlock",
        because="writing a whole listing before reading fills git's stdout and both block",
    ),
    Reversion(
        target=_ORACLE,
        find=(
            "        if child in self._checkouts or not (child / \".git\").exists():\n"
            "            return False\n"
        ),
        replace="        return False\n",
        test="test_a_clone_appearing_later_is_found_by_the_batch",
        because="the cheap miss must still notice a clone that appeared mid-session",
    ),
]

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git_init(repo: Path, gitignore: str = "") -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    if gitignore:
        (repo / ".gitignore").write_text(gitignore)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace that exercises every rule the walk must keep."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".gitignore").write_text("*.tmp\nscratch/\n")
    # A clone with its own ignore rules, an info/exclude and a negation.
    clone = ws / "project"
    _git_init(clone, "node_modules/\n*.log\n!keep.log\nbuild/\n")
    (clone / ".git" / "info").mkdir(exist_ok=True)
    (clone / ".git" / "info" / "exclude").write_text("secret.txt\n")
    for rel in ("src/a.py", "src/deep/b.py", "node_modules/pkg/index.js",
                "debug.log", "keep.log", "build/out.bin", "secret.txt",
                "notes.tmp", "README.md"):
        _touch(clone / rel)
    # Files outside every checkout, ignored and not.
    for rel in ("docs/guide.md", "docs/draft.tmp", "scratch/x.txt",
                ".home/.cache/c", ".jaato/profiles/p.yaml", "top.txt"):
        _touch(ws / rel)
    os.symlink(ws / "missing", ws / "dangling")
    return ws


def _reference_walk(monitor: WorkspaceMonitor) -> Set[str]:
    """The walk as it was: ``_is_ignored`` per path, pruning on the way."""
    files: Set[str] = set()
    for dirpath, dirnames, filenames in os.walk(monitor.workspace_path):
        dirnames[:] = [d for d in dirnames
                       if not monitor._is_ignored(os.path.join(dirpath, d), is_dir=True)]
        for name in filenames:
            full = os.path.join(dirpath, name)
            if not monitor._is_ignored(full, is_dir=False):
                files.add(os.path.relpath(full, monitor.workspace_path))
    return files


def test_the_walk_agrees_with_asking_per_path(workspace):
    monitor = WorkspaceMonitor(str(workspace), on_changed=lambda c: None)
    walked = monitor._walk_unignored()
    assert walked == _reference_walk(monitor)
    # And the rules really did something, so the comparison is not vacuous.
    assert "project/src/deep/b.py" in walked
    assert "project/keep.log" in walked
    for hidden in ("project/node_modules/pkg/index.js", "project/debug.log",
                   "project/build/out.bin", "project/secret.txt",
                   "docs/draft.tmp", "scratch/x.txt", ".home/.cache/c"):
        assert hidden not in walked, hidden


def test_git_is_asked_per_directory_not_per_file(workspace, monkeypatch):
    for i in range(40):
        _touch(workspace / "project" / "src" / f"m{i}.py")
    per_file = []
    batches = []
    real_one = wq._CheckIgnore.is_ignored
    real_many = wq._CheckIgnore.is_ignored_many

    def one(self, rel):
        per_file.append(rel)
        return real_one(self, rel)

    def many(self, rels):
        batches.append(len(rels))
        return real_many(self, rels)

    monkeypatch.setattr(wq._CheckIgnore, "is_ignored", one)
    monkeypatch.setattr(wq._CheckIgnore, "is_ignored_many", many)
    monitor = WorkspaceMonitor(str(workspace), on_changed=lambda c: None)
    monitor._seed_baseline()
    assert per_file == []
    dirs_in_clone = sum(1 for _ in os.walk(workspace / "project"))
    assert 0 < len(batches) <= dirs_in_clone
    assert max(batches) > 40


def test_a_path_outside_every_checkout_does_not_rescan_the_root(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    _git_init(ws / "project")
    for i in range(50):
        _touch(ws / "notes" / f"n{i}.md")
    scans = []
    real = wq.GitIgnoreOracle._scan_checkouts

    def counting(self):
        scans.append(1)
        return real(self)

    monkeypatch.setattr(wq.GitIgnoreOracle, "_scan_checkouts", counting)
    monitor = WorkspaceMonitor(str(ws), on_changed=lambda c: None)
    assert len(scans) == 1  # the oracle's own construction
    monitor._seed_baseline()
    assert len(scans) == 1
    assert len(monitor.baseline) == 50


def test_a_directory_larger_than_the_pipe_does_not_deadlock(tmp_path):
    clone = tmp_path / "ws" / "project"
    _git_init(clone)
    big = clone / "many"
    big.mkdir()
    for i in range(2000):  # ~2000 * 2 * 200 bytes, far past a 64 KiB pipe
        (big / (f"{i:05d}" + "x" * 195)).write_text("")
    oracle = wq.GitIgnoreOracle(str(tmp_path / "ws"))
    paths = [str(p) for p in big.iterdir()]
    result = {}
    worker = threading.Thread(
        target=lambda: result.setdefault("v", oracle.is_ignored_many(paths)),
        daemon=True)
    worker.start()
    worker.join(timeout=30)
    try:
        assert not worker.is_alive(), "the batched check-ignore deadlocked"
        assert result["v"] == [False] * len(paths)
    finally:
        for check in list(oracle._checks.values()):
            proc = check._proc
            if proc is not None:
                proc.kill()


def test_a_clone_appearing_later_is_found_by_the_batch(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    oracle = wq.GitIgnoreOracle(str(ws))
    clone = ws / "later"
    _git_init(clone, "dist/\n")
    _touch(clone / "dist" / "x.js")
    _touch(clone / "src" / "y.js")
    try:
        assert oracle.is_ignored_many(
            [str(clone / "dist"), str(clone / "src" / "y.js")]) == [True, False]
    finally:
        oracle.close()
