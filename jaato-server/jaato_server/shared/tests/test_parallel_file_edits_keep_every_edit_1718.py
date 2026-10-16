"""Parallel edits of one file keep every edit (#1718).

Tool calls in one model response run in parallel.  ``updateFile`` read the
file, computed the edit and wrote the result with nothing held, so three
targeted edits of one file in one response all read the original and the
last write discarded the other two, while each call reported success.

Each file_edit call that reads a file in order to rewrite it now holds that
file's lock (``path_locks.locked_paths``) from the read through the write,
and ``updateFile`` refuses to write when the file changed underneath it from
outside the process.

The races are forced, not hoped for: a gate on ``restore`` (which every
writer calls after reading and before writing) holds the first caller until
a second one has read too, or until a short timeout.  Without the lock the
second caller arrives inside that window and both write from the same read;
with it the second cannot read until the first has written, so the gate
times out and the calls run one after the other.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from jaato_server.shared.plugins import path_locks
from jaato_server.shared.plugins.file_edit import find_replace as find_replace_mod
from jaato_server.shared.plugins.file_edit import multi_file as multi_file_mod
from jaato_server.shared.plugins.file_edit import plugin as plugin_mod
from jaato_server.shared.plugins.file_edit.plugin import FileEditPlugin
from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/file_edit/plugin.py"
_MULTI = "jaato-server/jaato_server/shared/plugins/file_edit/multi_file.py"
_FIND = "jaato-server/jaato_server/shared/plugins/file_edit/find_replace.py"
_LOCKS = "jaato-server/jaato_server/shared/plugins/path_locks.py"
_LSP = "jaato-server/jaato_server/shared/plugins/lsp/plugin.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find=(
            "        with locked_paths(file_path):\n"
            "            return self._update_file_locked(args, path, file_path)\n"
        ),
        replace="        return self._update_file_locked(args, path, file_path)\n",
        test="test_parallel_update_file_calls_keep_every_edit",
        because="updateFile reads and writes with no lock: the last write wins",
    ),
    Reversion(
        target=_MULTI,
        find=(
            "        with locked_paths(*named):\n"
            "            return self._execute_locked(parsed_ops, operations)\n"
        ),
        replace="        return self._execute_locked(parsed_ops, operations)\n",
        test="test_update_file_and_multi_file_edit_keep_both_edits",
        because="multiFileEdit takes no lock and overwrites a parallel updateFile",
    ),
    Reversion(
        target=_FIND,
        find=(
            "        with locked_paths(*files):\n"
            "            return self._execute_locked(\n"
        ),
        replace=(
            "        if True:\n"
            "            return self._execute_locked(\n"
        ),
        test="test_update_file_and_find_and_replace_keep_both_edits",
        because="findAndReplace takes no lock and overwrites a parallel updateFile",
    ),
    Reversion(
        target=_PLUGIN,
        find="                assert_unchanged(file_path, read_fingerprint)\n",
        replace="                pass\n",
        test="test_a_write_from_outside_the_process_is_refused_not_lost",
        because="an edit computed from a stale read overwrites another process's write",
    ),
    Reversion(
        target=_LOCKS,
        find="    return os.path.normcase(os.path.realpath(os.fspath(path)))\n",
        replace="    return os.fspath(path)\n",
        test="test_two_spellings_of_one_file_share_one_lock",
        because="two spellings of one file get two locks",
    ),
    Reversion(
        target=_LSP,
        find=(
            "            with locked_paths(file_path):\n"
            "                change_info = _apply_edits_to_file(file_path, edits, dry_run)\n"
        ),
        replace=(
            "            if True:\n"
            "                change_info = _apply_edits_to_file(file_path, edits, dry_run)\n"
        ),
        test="test_an_lsp_rename_waits_for_a_file_edit_in_progress",
        because="an LSP workspace edit rewrites the file under a file_edit call",
    ),
]

#: How long the first writer waits for a second one to have read too.
_WINDOW = 1.0


class _Gate:
    """Hold the first caller of a wrapped function until a second arrives."""

    def __init__(self, fn):
        self._fn = fn
        self._lock = threading.Lock()
        self._arrived = 0
        self._second = threading.Event()

    def __call__(self, *args, **kwargs):
        with self._lock:
            self._arrived += 1
            first = self._arrived == 1
        if first:
            self._second.wait(_WINDOW)
        else:
            self._second.set()
        return self._fn(*args, **kwargs)


@pytest.fixture
def gate(monkeypatch):
    """One gate on ``restore`` in every module that rewrites a file."""
    g = _Gate(plugin_mod.restore)
    for mod in (plugin_mod, multi_file_mod, find_replace_mod):
        monkeypatch.setattr(mod, "restore", g)
    return g


def _plugin(tmp_path: Path) -> FileEditPlugin:
    plugin = FileEditPlugin()
    plugin.initialize({"backup_dir": str(tmp_path / "backups")})
    return plugin


def _run_parallel(*calls):
    results = [None] * len(calls)

    def run(i, fn, args):
        results[i] = fn(args)

    threads = [
        threading.Thread(target=run, args=(i, fn, args))
        for i, (fn, args) in enumerate(calls)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    return results


def _ok(result) -> bool:
    return not isinstance(result, tuple) and "error" not in result


def _target(tmp_path: Path) -> Path:
    target = tmp_path / "ws" / "run.py"
    target.parent.mkdir(parents=True)
    target.write_text("alpha = 1\nbeta = 2\ngamma = 3\n", encoding="utf-8")
    return target


def test_parallel_update_file_calls_keep_every_edit(tmp_path, gate):
    target = _target(tmp_path)
    update = _plugin(tmp_path).get_executors()["updateFile"]

    results = _run_parallel(
        (update, {"path": str(target), "old": "alpha = 1", "new": "alpha = 10"}),
        (update, {"path": str(target), "old": "gamma = 3", "new": "gamma = 30"}),
    )

    assert all(_ok(r) for r in results), results
    text = target.read_text(encoding="utf-8")
    assert "alpha = 10" in text and "gamma = 30" in text, text
    assert path_locks.held_key_count() == 0


def test_update_file_and_multi_file_edit_keep_both_edits(tmp_path, gate):
    target = _target(tmp_path)
    executors = _plugin(tmp_path).get_executors()

    results = _run_parallel(
        (executors["updateFile"],
         {"path": str(target), "old": "alpha = 1", "new": "alpha = 10"}),
        (executors["multiFileEdit"],
         {"operations": [{"action": "edit", "path": str(target),
                          "old": "beta = 2", "new": "beta = 20"}]}),
    )

    assert all(_ok(r) for r in results), results
    text = target.read_text(encoding="utf-8")
    assert "alpha = 10" in text and "beta = 20" in text, text


def test_update_file_and_find_and_replace_keep_both_edits(tmp_path, gate):
    target = _target(tmp_path)
    plugin = _plugin(tmp_path)
    plugin._workspace_root = str(target.parent)
    executors = plugin.get_executors()

    results = _run_parallel(
        (executors["updateFile"],
         {"path": str(target), "old": "alpha = 1", "new": "alpha = 10"}),
        (executors["findAndReplace"],
         {"pattern": "gamma = 3", "replacement": "gamma = 30", "paths": "*.py"}),
    )

    assert all(_ok(r) for r in results), results
    text = target.read_text(encoding="utf-8")
    assert "alpha = 10" in text and "gamma = 30" in text, text


def test_a_write_from_outside_the_process_is_refused_not_lost(tmp_path, monkeypatch):
    target = _target(tmp_path)
    plugin = _plugin(tmp_path)
    real_restore = plugin_mod.restore

    def another_process_writes(*args, **kwargs):
        # The edit has been computed from the read; now something outside
        # this process's locks rewrites the file.
        target.write_text("alpha = 1\nbeta = 2\ngamma = 3\ndelta = 4\n",
                          encoding="utf-8")
        return real_restore(*args, **kwargs)

    monkeypatch.setattr(plugin_mod, "restore", another_process_writes)

    result = plugin.get_executors()["updateFile"](
        {"path": str(target), "old": "alpha = 1", "new": "alpha = 10"})

    assert isinstance(result, tuple) and result[0] is False, result
    assert "modified by another process" in result[1]["error"]
    assert "delta = 4" in target.read_text(encoding="utf-8")


def test_two_spellings_of_one_file_share_one_lock(tmp_path, gate):
    target = _target(tmp_path)
    (target.parent / "sub").mkdir()
    link = target.parent / "sub" / ".." / "run.py"
    update = _plugin(tmp_path).get_executors()["updateFile"]

    results = _run_parallel(
        (update, {"path": str(target), "old": "alpha = 1", "new": "alpha = 10"}),
        (update, {"path": str(link), "old": "gamma = 3", "new": "gamma = 30"}),
    )

    assert all(_ok(r) for r in results), results
    text = target.read_text(encoding="utf-8")
    assert "alpha = 10" in text and "gamma = 30" in text, text


def test_overlapping_batches_in_opposite_orders_do_not_deadlock(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    done = []

    def take(first, second):
        for _ in range(200):
            with path_locks.locked_paths(first, second):
                pass
        done.append(1)

    threads = [threading.Thread(target=take, args=(a, b)),
               threading.Thread(target=take, args=(b, a))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(done) == 2
    assert path_locks.held_key_count() == 0


def test_an_lsp_rename_waits_for_a_file_edit_in_progress(tmp_path):
    from jaato_server.shared.plugins.lsp.lsp_client import (
        Position, Range, TextEdit, WorkspaceEdit,
    )
    from jaato_server.shared.plugins.lsp.plugin import apply_workspace_edit

    target = _target(tmp_path)
    edit = WorkspaceEdit(changes={
        f"file://{target}": [TextEdit(
            range=Range(start=Position(0, 0), end=Position(0, 5)),
            new_text="ALPHA",
        )],
    })
    finished = threading.Event()

    def rename():
        apply_workspace_edit(edit)
        finished.set()

    # A file_edit call holds the file: the LSP edit must not run under it.
    with path_locks.locked_paths(target):
        worker = threading.Thread(target=rename)
        worker.start()
        assert not finished.wait(0.5), "the LSP edit ran while the file was locked"
        target.write_text("alpha = 1\nbeta = 20\ngamma = 3\n", encoding="utf-8")
    worker.join(10)

    assert finished.is_set()
    # It read what the file_edit call wrote, so both changes are there.
    assert target.read_text(encoding="utf-8") == "ALPHA = 1\nbeta = 20\ngamma = 3\n"
