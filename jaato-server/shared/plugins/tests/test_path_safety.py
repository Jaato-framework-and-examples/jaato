"""Tests for the symlink-safe filesystem primitives.

Each class pins one of the three guards described in
``shared/plugins/path_safety.py``:

* the check-then-open (TOCTOU) window, exercised by validating one target and
  then swapping the symlink before the open — the sequence that made file
  tools read and write outside their approved location (jaato issue #669,
  shape A);
* the special-file guard, which is what keeps a FIFO on an approved path from
  blocking a worker forever;
* the pre-planted-directory refusal for predictable paths on shared ``/tmp``.
"""

import errno
import os
import stat
import sys

import pytest

from shared.plugins.path_safety import (
    PathSwappedError,
    SpecialFileError,
    UnsafePathError,
    describe_special,
    ensure_private_dir,
    move_verified,
    open_verified,
    read_bytes_verified,
    read_text_verified,
    unlink_verified,
    write_text_verified,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX symlink/FIFO semantics; Windows needs elevation for links",
)


@pytest.fixture
def sandbox(tmp_path):
    """An ``approved`` directory, a ``forbidden`` one, and a validator.

    Returns:
        Tuple of (approved dir, forbidden dir, validator restricting to
        ``approved``).
    """
    approved = tmp_path / "approved"
    approved.mkdir()
    forbidden = tmp_path / "forbidden"
    forbidden.mkdir()
    (forbidden / "secret.txt").write_text("TOPSECRET\n")

    root = str(approved)

    def validate(resolved: str) -> bool:
        return resolved == root or resolved.startswith(root + os.sep)

    return approved, forbidden, validate


class TestDescribeSpecial:
    """describe_special names exactly the kinds we refuse to open."""

    def test_regular_file_is_not_special(self, tmp_path):
        target = tmp_path / "f.txt"
        target.write_text("x")
        assert describe_special(target) is None

    def test_directory_is_not_special(self, tmp_path):
        assert describe_special(tmp_path) is None

    def test_missing_path_is_not_special(self, tmp_path):
        assert describe_special(tmp_path / "nope") is None

    def test_fifo_is_named(self, tmp_path):
        fifo = tmp_path / "p"
        os.mkfifo(str(fifo))
        assert describe_special(fifo) == "FIFO (named pipe)"

    def test_character_device_is_named(self):
        assert describe_special("/dev/null") == "character device"

    def test_symlink_to_fifo_is_named_when_followed(self, tmp_path):
        fifo = tmp_path / "p"
        os.mkfifo(str(fifo))
        link = tmp_path / "link"
        link.symlink_to(fifo)
        assert describe_special(link) == "FIFO (named pipe)"
        assert describe_special(link, follow_symlinks=False) is None


class TestReadPathValidation:
    """The validator sees the canonical path, so symlinks cannot dodge it."""

    def test_reads_a_file_inside_the_sandbox(self, sandbox):
        approved, _, validate = sandbox
        (approved / "ok.txt").write_text("hello\n")
        assert read_text_verified(approved / "ok.txt", validate=validate) == "hello\n"

    def test_refuses_a_symlink_pointing_out(self, sandbox):
        approved, forbidden, validate = sandbox
        (approved / "escape.txt").symlink_to(forbidden / "secret.txt")

        with pytest.raises(UnsafePathError) as exc:
            read_text_verified(approved / "escape.txt", validate=validate)
        assert exc.value.errno == errno.EACCES

    def test_refuses_a_symlinked_parent_directory(self, sandbox):
        approved, forbidden, validate = sandbox
        (approved / "vendor").symlink_to(forbidden, target_is_directory=True)

        with pytest.raises(UnsafePathError):
            read_text_verified(approved / "vendor" / "secret.txt", validate=validate)

    def test_allows_a_symlink_that_stays_inside(self, sandbox):
        approved, _, validate = sandbox
        (approved / "real.txt").write_text("inside\n")
        (approved / "alias.txt").symlink_to(approved / "real.txt")
        assert read_text_verified(approved / "alias.txt", validate=validate) == "inside\n"

    def test_read_bytes_honours_the_validator(self, sandbox):
        approved, forbidden, validate = sandbox
        (approved / "escape.bin").symlink_to(forbidden / "secret.txt")
        with pytest.raises(UnsafePathError):
            read_bytes_verified(approved / "escape.bin", validate=validate)

    def test_crlf_round_trips_like_path_read_text(self, sandbox):
        """Universal-newline translation must match ``Path.read_text``."""
        approved, _, validate = sandbox
        target = approved / "crlf.txt"
        target.write_bytes(b"a\r\nb\r\n")
        assert read_text_verified(target, validate=validate) == target.read_text()


class TestCheckThenOpenRace:
    """The descriptor must be the object the validator approved."""

    def test_swapped_symlink_is_caught(self, sandbox, monkeypatch):
        """Swap the link after validation, before the open.

        ``open_verified`` validates the canonical path and then opens; hooking
        the validator to repoint the symlink reproduces exactly the window an
        attacker races for, with none of the flakiness of a real race.
        """
        approved, forbidden, validate = sandbox
        decoy = approved / "decoy.txt"
        decoy.write_text("harmless\n")
        link = approved / "handle.txt"
        link.symlink_to(decoy)

        def validate_then_swap(resolved: str) -> bool:
            verdict = validate(resolved)
            link.unlink()
            link.symlink_to(forbidden / "secret.txt")
            return verdict

        with pytest.raises(PathSwappedError):
            read_text_verified(link, validate=validate_then_swap)

    def test_target_removed_after_validation_is_caught(self, sandbox):
        approved, _, validate = sandbox
        decoy = approved / "decoy.txt"
        decoy.write_text("harmless\n")
        link = approved / "handle.txt"
        link.symlink_to(decoy)

        def validate_then_replace(resolved: str) -> bool:
            verdict = validate(resolved)
            decoy.unlink()
            link.unlink()
            other = approved / "other.txt"
            other.write_text("different object\n")
            link.symlink_to(other)
            return verdict

        with pytest.raises(PathSwappedError):
            read_text_verified(link, validate=validate_then_replace)


class TestSpecialFileGuard:
    """FIFOs and devices are refused rather than opened."""

    def test_fifo_is_refused_without_blocking(self, sandbox):
        approved, _, validate = sandbox
        fifo = approved / "pipe"
        os.mkfifo(str(fifo))

        with pytest.raises(SpecialFileError):
            read_text_verified(fifo, validate=validate)

    def test_character_device_is_refused(self):
        with pytest.raises(SpecialFileError):
            read_bytes_verified("/dev/zero", max_bytes=1)

    def test_symlink_to_fifo_is_refused(self, sandbox):
        approved, _, validate = sandbox
        fifo = approved / "pipe"
        os.mkfifo(str(fifo))
        (approved / "link").symlink_to(fifo)

        with pytest.raises(SpecialFileError):
            read_text_verified(approved / "link", validate=validate)

    def test_allow_special_opts_back_in(self, sandbox):
        approved, _, validate = sandbox
        fd = open_verified(approved, os.O_RDONLY, validate=validate, allow_special=True)
        os.close(fd)


class TestWriteVerified:
    """Writes land on the validated object, never through a planted link."""

    def test_writes_inside_the_sandbox(self, sandbox):
        approved, _, validate = sandbox
        write_text_verified(approved / "out.txt", "data\n", validate=validate)
        assert (approved / "out.txt").read_text() == "data\n"

    def test_truncating_write_refuses_a_symlink_out(self, sandbox):
        approved, forbidden, validate = sandbox
        (approved / "out.txt").symlink_to(forbidden / "secret.txt")

        with pytest.raises(UnsafePathError):
            write_text_verified(approved / "out.txt", "pwned\n", validate=validate)
        assert (forbidden / "secret.txt").read_text() == "TOPSECRET\n"

    def test_exclusive_write_refuses_a_planted_symlink(self, sandbox):
        """The new-file path must not follow a link planted at the target.

        The link points *inside* the sandbox, so the validator is happy: what
        stops the write is ``O_EXCL``/``O_NOFOLLOW`` on the leaf, which is the
        guard being pinned here.
        """
        approved, _, validate = sandbox
        victim = approved / "victim.txt"
        victim.write_text("original\n")
        (approved / "new.txt").symlink_to(victim)

        with pytest.raises(OSError):
            write_text_verified(
                approved / "new.txt", "pwned\n", validate=validate, exclusive=True
            )
        assert victim.read_text() == "original\n"

    def test_exclusive_write_refuses_a_symlinked_parent_escape(self, sandbox):
        approved, forbidden, validate = sandbox
        (approved / "vendor").symlink_to(forbidden, target_is_directory=True)

        with pytest.raises(UnsafePathError):
            write_text_verified(
                approved / "vendor" / "new.txt",
                "pwned\n",
                validate=validate,
                exclusive=True,
            )
        assert not (forbidden / "new.txt").exists()

    def test_writing_to_a_fifo_is_refused(self, sandbox):
        approved, _, validate = sandbox
        fifo = approved / "pipe"
        os.mkfifo(str(fifo))

        with pytest.raises(OSError):
            write_text_verified(approved / "pipe", "x", validate=validate)


class TestUnlinkVerified:
    """Deletes act on the entry inside the validated parent."""

    def test_deletes_inside_the_sandbox(self, sandbox):
        approved, _, validate = sandbox
        target = approved / "gone.txt"
        target.write_text("x")
        unlink_verified(target, validate=validate)
        assert not target.exists()

    def test_deletes_the_link_not_its_target(self, sandbox):
        approved, forbidden, validate = sandbox
        link = approved / "link.txt"
        link.symlink_to(forbidden / "secret.txt")

        unlink_verified(link, validate=validate)
        assert not link.is_symlink()
        assert (forbidden / "secret.txt").exists()

    def test_refuses_through_a_symlinked_parent(self, sandbox):
        approved, forbidden, validate = sandbox
        (approved / "vendor").symlink_to(forbidden, target_is_directory=True)

        with pytest.raises(UnsafePathError):
            unlink_verified(approved / "vendor" / "secret.txt", validate=validate)
        assert (forbidden / "secret.txt").exists()


class TestMoveVerified:
    """Moves pin both ends to their validated parents."""

    def test_moves_inside_the_sandbox(self, sandbox):
        approved, _, validate = sandbox
        src = approved / "a.txt"
        src.write_text("payload\n")
        move_verified(src, approved / "b.txt", validate=validate)
        assert (approved / "b.txt").read_text() == "payload\n"
        assert not src.exists()

    def test_refuses_a_destination_outside(self, sandbox):
        approved, forbidden, validate = sandbox
        src = approved / "a.txt"
        src.write_text("payload\n")

        with pytest.raises(UnsafePathError):
            move_verified(src, forbidden / "a.txt", validate=validate)
        assert src.exists()

    def test_refuses_a_destination_through_a_symlinked_parent(self, sandbox):
        approved, forbidden, validate = sandbox
        src = approved / "a.txt"
        src.write_text("payload\n")
        (approved / "vendor").symlink_to(forbidden, target_is_directory=True)

        with pytest.raises(UnsafePathError):
            move_verified(src, approved / "vendor" / "a.txt", validate=validate)
        assert not (forbidden / "a.txt").exists()


class TestEnsurePrivateDir:
    """Pre-planted paths on shared /tmp are refused, not adopted."""

    def test_creates_a_new_directory_owner_only(self, tmp_path):
        target = tmp_path / "fresh"
        assert ensure_private_dir(target) == str(target)
        mode = stat.S_IMODE(os.lstat(target).st_mode)
        assert mode & (stat.S_IRWXG | stat.S_IRWXO) == 0

    def test_adopts_our_own_existing_directory(self, tmp_path):
        target = tmp_path / "existing"
        target.mkdir(mode=0o700)
        assert ensure_private_dir(target) == str(target)

    def test_refuses_a_pre_planted_symlink(self, tmp_path):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        planted = tmp_path / "planted"
        planted.symlink_to(elsewhere, target_is_directory=True)

        with pytest.raises(UnsafePathError) as exc:
            ensure_private_dir(planted)
        assert "symlink" in str(exc.value)

    def test_refuses_a_regular_file(self, tmp_path):
        occupied = tmp_path / "occupied"
        occupied.write_text("x")

        with pytest.raises(UnsafePathError) as exc:
            ensure_private_dir(occupied)
        assert exc.value.errno == errno.ENOTDIR

    def test_refuses_a_world_writable_directory(self, tmp_path):
        loose = tmp_path / "loose"
        loose.mkdir(mode=0o777)
        os.chmod(loose, 0o777)

        with pytest.raises(UnsafePathError) as exc:
            ensure_private_dir(loose)
        assert "sticky" in str(exc.value)

    def test_tolerates_the_sticky_bit(self, tmp_path):
        """`/tmp` itself is 1777 and must remain usable."""
        sticky = tmp_path / "sticky"
        sticky.mkdir()
        os.chmod(sticky, 0o1777)
        assert ensure_private_dir(sticky) == str(sticky)
