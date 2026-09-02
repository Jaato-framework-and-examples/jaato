"""``file_edit`` must act on the object its sandbox check approved.

The tools resolve a path, ask the sandbox about it, and then read or write —
addressed by path, so the kernel resolves the symlinks a second time. Anything
that changed in between redirects the operation to a different object than the
one that was approved (jaato issue #669, shape A).

Racing a real attacker would make a flaky test, so instead the swap is driven
from inside the sandbox check itself: ``_is_path_allowed`` is wrapped so that
answering the question also repoints the symlink. That is deterministic and
reproduces the exact ordering — validate target A, act on target B.

The plain (unraced) symlink escapes are covered too, since they are what an
attacker gets without needing to win any race at all.
"""

import os
import sys

import pytest

from shared.plugins.file_edit.plugin import FileEditPlugin

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX symlink/FIFO semantics; Windows needs elevation for links",
)


@pytest.fixture
def workspace(tmp_path):
    """A sandboxed workspace plus an off-limits directory beside it.

    Returns:
        Tuple of (workspace path, outside path).
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOPSECRET\n")

    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "ordinary.txt").write_text("ordinary\n")
    return ws, outside


@pytest.fixture(autouse=True)
def no_tmp_allowance(monkeypatch):
    """Take ``/tmp`` off the default allowlist for these tests.

    ``check_path_with_jaato_containment`` allows anything under a system temp
    directory by default, and ``tmp_path`` puts the whole fixture there — so
    without this every "escaped outside the workspace" assertion would pass
    for the wrong reason.  Emptying ``SYSTEM_TEMP_PATHS`` isolates the symlink
    behaviour from the (deliberate, unrelated) ``/tmp`` allowance.
    """
    monkeypatch.setattr("shared.plugins.sandbox_utils.SYSTEM_TEMP_PATHS", [])


@pytest.fixture
def plugin(workspace, tmp_path):
    """A ``FileEditPlugin`` sandboxed to the workspace.

    ``backup_dir`` is supplied explicitly and points outside the workspace:
    the plugin refuses to initialize without a resolvable backup root, and
    keeping backups out of the tree stops them showing up as extra files in
    the assertions below.
    """
    ws, _ = workspace
    p = FileEditPlugin()
    p.initialize(
        {"workspace_root": str(ws), "backup_dir": str(tmp_path / "backups")}
    )
    p.set_workspace_path(str(ws))
    yield p
    p.shutdown()


def after_check(plugin, mutate, *, after_n=1):
    """Run ``mutate`` immediately after the ``after_n``-th sandbox check.

    This is the check-then-act window made deterministic.  Racing a real
    attacker would make a flaky test; instead the wrapper answers the sandbox
    question honestly about the state that exists *now* and then, on the way
    out, applies the substitution the attacker would be trying to land.  What
    follows is exactly the ordering the bug needs: validate object A, act on
    object B.

    Args:
        plugin: The plugin whose ``_is_path_allowed`` to wrap.
        mutate: Zero-argument callable applied once the count is reached.
        after_n: Fire on this check and every one after it.  Tools that check
            more than one path (``moveFile`` checks source and destination)
            need the substitution to land after the *last* of them, or the
            tool's own later check catches it and the race is never reached.
    """
    original = plugin._is_path_allowed
    calls = {"n": 0}

    def wrapper(path, mode="read"):
        verdict = original(path, mode=mode)
        calls["n"] += 1
        if calls["n"] >= after_n:
            mutate()
        return verdict

    plugin._is_path_allowed = wrapper


def swap_on_check(plugin, link, new_target):
    """Repoint ``link`` at ``new_target`` after each sandbox check.

    Args:
        plugin: The plugin whose ``_is_path_allowed`` to wrap.
        link: The symlink to repoint.
        new_target: What it should point at after the check.
    """

    def mutate():
        if link.is_symlink():
            link.unlink()
            link.symlink_to(new_target)

    after_check(plugin, mutate)


class TestReadFileContainment:
    """readFile must not read outside the approved location."""

    def test_plain_symlink_escape_is_refused(self, plugin, workspace):
        ws, outside = workspace
        (ws / "escape.txt").symlink_to(outside / "secret.txt")

        result = plugin._execute_read_file({"path": "escape.txt"})
        assert "error" in result
        assert "TOPSECRET" not in str(result)

    def test_swapped_symlink_is_caught(self, plugin, workspace):
        ws, outside = workspace
        decoy = ws / "decoy.txt"
        decoy.write_text("harmless\n")
        link = ws / "handle.txt"
        link.symlink_to(decoy)
        swap_on_check(plugin, link, outside / "secret.txt")

        result = plugin._execute_read_file({"path": "handle.txt"})
        assert "TOPSECRET" not in str(result), result

    def test_ordinary_read_still_works(self, plugin, workspace):
        """readFile returns the text directly for a plain text file."""
        result = plugin._execute_read_file({"path": "ordinary.txt"})
        assert isinstance(result, str)
        assert "ordinary" in result

    def test_fifo_is_refused_rather_than_blocking(self, plugin, workspace):
        """Opening a named pipe would block until a writer appears."""
        ws, _ = workspace
        os.mkfifo(str(ws / "pipe.txt"))

        result = plugin._execute_read_file({"path": "pipe.txt"})
        assert "error" in result


class TestWriteContainment:
    """Writes must land on the approved object, never through a planted link."""

    def test_update_through_a_symlink_out_is_refused(self, plugin, workspace):
        ws, outside = workspace
        (ws / "escape.txt").symlink_to(outside / "secret.txt")

        result = plugin._execute_update_file(
            {"path": "escape.txt", "new_content": "pwned\n"}
        )
        assert "error" in result
        assert (outside / "secret.txt").read_text() == "TOPSECRET\n"

    def test_update_with_a_swapped_symlink_is_caught(self, plugin, workspace):
        ws, outside = workspace
        decoy = ws / "decoy.txt"
        decoy.write_text("harmless\n")
        link = ws / "handle.txt"
        link.symlink_to(decoy)
        swap_on_check(plugin, link, outside / "secret.txt")

        plugin._execute_update_file({"path": "handle.txt", "new_content": "pwned\n"})
        assert (outside / "secret.txt").read_text() == "TOPSECRET\n"

    def test_new_file_through_a_symlinked_parent_is_refused(self, plugin, workspace):
        """The *parent* being the symlink is its own escape route."""
        ws, outside = workspace
        (ws / "vendor").symlink_to(outside, target_is_directory=True)

        result = plugin._execute_write_new_file(
            {"path": "vendor/planted.txt", "content": "pwned\n"}
        )
        assert "error" in result
        assert not (outside / "planted.txt").exists()

    def test_new_file_refuses_a_symlink_planted_after_the_check(self, plugin, workspace):
        """The window ``writeNewFile`` used to leave open.

        The target does not exist when the sandbox is consulted, so the check
        passes on its own merits; the link appears only afterwards.  A plain
        ``write_text`` then follows it and writes outside the workspace.  The
        ``O_EXCL``/``O_NOFOLLOW`` create relative to the resolved parent has
        nothing to follow.
        """
        ws, outside = workspace
        target = ws / "brand-new.txt"
        escape = outside / "planted.txt"

        def plant():
            if not target.is_symlink():
                target.symlink_to(escape)

        after_check(plugin, plant)

        plugin._execute_write_new_file({"path": "brand-new.txt", "content": "pwned\n"})
        assert not escape.exists(), "write followed a symlink planted after the check"

    def test_ordinary_write_still_works(self, plugin, workspace):
        ws, _ = workspace
        result = plugin._execute_write_new_file(
            {"path": "fresh.txt", "content": "hello\n"}
        )
        assert result.get("success") is True
        assert (ws / "fresh.txt").read_text() == "hello\n"

    def test_ordinary_update_still_works(self, plugin, workspace):
        ws, _ = workspace
        result = plugin._execute_update_file(
            {"path": "ordinary.txt", "new_content": "changed\n"}
        )
        assert result.get("success") is True
        assert (ws / "ordinary.txt").read_text() == "changed\n"


class TestDeleteAndMoveContainment:
    """Deletes and moves act on entries inside the validated parent."""

    def test_delete_through_a_symlinked_parent_is_refused(self, plugin, workspace):
        ws, outside = workspace
        (ws / "vendor").symlink_to(outside, target_is_directory=True)

        result = plugin._execute_remove_file({"path": "vendor/secret.txt"})
        assert "error" in result
        assert (outside / "secret.txt").exists()

    def test_delete_refuses_a_parent_swapped_after_the_check(self, plugin, workspace):
        """The delete must land in the directory that was validated.

        ``unlink`` never follows a symlink at the final component, so the only
        way a delete escapes is a swap of a *parent*.  Here ``vendor`` is a
        real in-workspace directory when the check runs and a symlink to the
        off-limits directory by the time the delete would happen.  Unlinking
        relative to the resolved parent's descriptor cannot be redirected.
        """
        ws, outside = workspace
        vendor = ws / "vendor"
        vendor.mkdir()
        (vendor / "secret.txt").write_text("in-workspace\n")

        def swap_parent():
            if vendor.is_dir() and not vendor.is_symlink():
                for child in vendor.iterdir():
                    child.unlink()
                vendor.rmdir()
                vendor.symlink_to(outside, target_is_directory=True)

        after_check(plugin, swap_parent)

        plugin._execute_remove_file({"path": "vendor/secret.txt"})
        assert (outside / "secret.txt").exists()

    def test_delete_removes_the_link_not_its_target(self, plugin, workspace):
        """Deleting a symlink must not delete what it points at."""
        ws, _ = workspace
        target = ws / "target.txt"
        target.write_text("keep me\n")
        link = ws / "link.txt"
        link.symlink_to(target)

        result = plugin._execute_remove_file({"path": "link.txt"})
        assert result.get("success") is True
        assert not link.is_symlink()
        assert target.read_text() == "keep me\n"

    def test_move_to_a_symlinked_parent_is_refused(self, plugin, workspace):
        ws, outside = workspace
        (ws / "vendor").symlink_to(outside, target_is_directory=True)
        (ws / "payload.txt").write_text("payload\n")

        result = plugin._execute_move_file(
            {"source_path": "payload.txt", "destination_path": "vendor/payload.txt"}
        )
        assert "error" in result
        assert not (outside / "payload.txt").exists()

    def test_move_refuses_a_destination_parent_swapped_after_the_check(
        self, plugin, workspace
    ):
        """The destination must land in the directory that was validated."""
        ws, outside = workspace
        (ws / "payload.txt").write_text("payload\n")
        vendor = ws / "vendor"
        vendor.mkdir()

        def swap_parent():
            if vendor.is_dir() and not vendor.is_symlink():
                vendor.rmdir()
                vendor.symlink_to(outside, target_is_directory=True)

        # moveFile checks source then destination; swap only after both.
        after_check(plugin, swap_parent, after_n=2)

        plugin._execute_move_file(
            {"source_path": "payload.txt", "destination_path": "vendor/payload.txt"}
        )
        assert not (outside / "payload.txt").exists()

    def test_ordinary_move_still_works(self, plugin, workspace):
        ws, _ = workspace
        (ws / "payload.txt").write_text("payload\n")

        result = plugin._execute_move_file(
            {"source_path": "payload.txt", "destination_path": "moved.txt"}
        )
        assert result.get("success") is True
        assert (ws / "moved.txt").read_text() == "payload\n"
        assert not (ws / "payload.txt").exists()

    def test_ordinary_delete_still_works(self, plugin, workspace):
        ws, _ = workspace
        (ws / "doomed.txt").write_text("x")
        result = plugin._execute_remove_file({"path": "doomed.txt"})
        assert result.get("success") is True
        assert not (ws / "doomed.txt").exists()
