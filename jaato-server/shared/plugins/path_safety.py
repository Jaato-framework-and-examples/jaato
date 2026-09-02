"""Symlink-safe filesystem primitives for sandboxed plugins.

``sandbox_utils.check_path_with_jaato_containment`` answers *"is this path
allowed?"* by canonicalising with :func:`os.path.realpath` and comparing
strings.  That answer is only as good as the instant it was computed: if the
caller then re-opens the file **by path**, the kernel resolves the symlinks a
second time, and anything that changed in between silently redirects the
operation.  This module supplies the primitives that close that window, plus
the two related guards the same class of bug keeps producing.

Three problems, three primitives
================================

**1. Check-then-open (TOCTOU).**  ``check(path)`` followed by
``open(path)`` validates one object and acts on another if a symlink on the
path is swapped in between.  :func:`open_verified` resolves once, validates
the resolved path, opens it, and then proves via :func:`os.fstat` that the
descriptor it holds is the very object that was validated — a mismatch raises
:class:`PathSwappedError` instead of returning a usable handle.  For creation
(``O_CREAT``) there is no pre-existing object to pin, so it opens the
*resolved, symlink-free* parent directory and creates the leaf relative to
that descriptor with ``O_NOFOLLOW`` — a symlink planted at the leaf, or a
swap of the parent, both fail rather than escape.

**2. Special files.**  A FIFO reached through an approved path blocks the
whole worker on ``open()`` until a writer appears; character devices such as
``/dev/zero`` are unbounded reads.  :func:`open_verified` opens with
``O_NONBLOCK``, then rejects anything that is not a regular file (or a
directory, when asked for one) via :func:`describe_special`.  Callers that
only need the predicate — e.g. to skip an entry during a directory walk —
can use :func:`describe_special` directly.

**3. Pre-planted paths on shared ``/tmp``.**  A predictable path under a
world-writable directory can be created *first* by another user, as a symlink
or as a directory they own, and whatever we write there lands wherever they
chose.  :func:`ensure_private_dir` refuses such a path instead of using it.

Portability
===========

``O_NOFOLLOW`` and ``dir_fd`` are POSIX.  Where the platform lacks them
(notably Windows) the corresponding hardening is skipped and the function
degrades to the ordinary resolve-validate-open sequence; the ``fstat``
identity check still runs, because it needs nothing platform-specific.  Every
capability is probed at import time rather than by ``sys.platform`` string
comparison.

All failures raise :class:`UnsafePathError`, an :class:`OSError` subclass, so
existing ``except OSError`` handlers in plugin call sites degrade to a normal
tool error rather than crashing the turn.
"""

import errno
import logging
import os
import stat
from typing import Callable, Optional, Union

logger = logging.getLogger(__name__)

__all__ = [
    "UnsafePathError",
    "PathSwappedError",
    "SpecialFileError",
    "describe_special",
    "open_verified",
    "read_bytes_verified",
    "read_text_verified",
    "write_text_verified",
    "unlink_verified",
    "move_verified",
    "ensure_private_dir",
]

PathLike = Union[str, "os.PathLike[str]"]

# --- Platform capability probes (evaluated once, at import) ---

_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_O_NOCTTY = getattr(os, "O_NOCTTY", 0)
_O_BINARY = getattr(os, "O_BINARY", 0)  # Windows: don't translate newlines
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_SUPPORTS_DIR_FD = os.open in getattr(os, "supports_dir_fd", set())

#: Human-readable names for the ``stat`` type predicates we refuse to open.
#: Regular files and directories are absent by design — those are the two
#: kinds every caller here expects.
_SPECIAL_KINDS = (
    (stat.S_ISFIFO, "FIFO (named pipe)"),
    (stat.S_ISSOCK, "socket"),
    (stat.S_ISCHR, "character device"),
    (stat.S_ISBLK, "block device"),
)


class UnsafePathError(OSError):
    """A filesystem operation was refused because the path is not safe.

    Subclasses :class:`OSError` deliberately: plugin call sites already wrap
    their reads and writes in ``except OSError`` and turn the result into a
    tool-level error message, so a refusal surfaces to the model as an
    ordinary failure rather than an unhandled exception.
    """


class PathSwappedError(UnsafePathError):
    """The object opened is not the object that passed the sandbox check.

    Raised when the descriptor's ``(st_dev, st_ino)`` does not match the
    canonical path that was validated — i.e. a symlink on the path was
    swapped between the check and the open.
    """


class SpecialFileError(UnsafePathError):
    """The path resolves to a FIFO, socket, or device rather than a file."""


def describe_special(path: PathLike, *, follow_symlinks: bool = True) -> Optional[str]:
    """Name the special-file kind at ``path``, or ``None`` if it is ordinary.

    Used to skip entries during directory walks and to explain refusals.  A
    path that does not exist, or that cannot be stat-ed at all, is reported as
    ``None`` — absence is not this function's concern, and the caller's own
    open will produce the accurate error.

    Args:
        path: Path to inspect.
        follow_symlinks: When True (default) classify the symlink *target*;
            when False classify the link itself (which is never one of the
            kinds reported here, so this mostly means "don't traverse").

    Returns:
        A human-readable kind (``"FIFO (named pipe)"``, ``"socket"``,
        ``"character device"``, ``"block device"``) or ``None`` for regular
        files, directories, symlinks-not-followed, and non-existent paths.
    """
    try:
        st = os.stat(path, follow_symlinks=follow_symlinks)
    except (OSError, ValueError):
        return None
    return _describe_mode(st.st_mode)


def _describe_mode(st_mode: int) -> Optional[str]:
    """Map a ``st_mode`` to a special-file kind name, or ``None``."""
    for predicate, label in _SPECIAL_KINDS:
        if predicate(st_mode):
            return label
    return None


def _validated_resolution(
    path: PathLike,
    validate: Optional[Callable[[str], bool]],
    creating: bool,
) -> "tuple[str, str, str, str]":
    """Resolve ``path`` **once** and run ``validate`` against the canonical form.

    The canonical path is returned rather than recomputed by the caller, and
    that is the whole point: resolving a second time after the open would walk
    whatever the symlinks say *then*, which is exactly the state an attacker
    controls.  Everything downstream must be pinned to the value this function
    produced.

    Returns:
        ``(abs_path, canonical, resolved_parent, leaf)``.  ``canonical`` and
        ``resolved_parent`` are symlink-free, so acting on them later cannot
        be redirected.

    Raises:
        UnsafePathError: if ``validate`` rejects the canonical path.
    """
    abs_path = os.path.abspath(os.fspath(path))
    parent, leaf = os.path.split(abs_path)
    resolved_parent = os.path.realpath(parent)

    # Validate the canonical target.  For creation the leaf does not exist
    # yet, so the canonical form is "resolved parent + leaf name"; that is
    # exactly what the O_NOFOLLOW creation below will produce.
    if creating:
        canonical = os.path.join(resolved_parent, leaf)
    else:
        canonical = os.path.realpath(abs_path)

    if validate is not None and not validate(canonical):
        raise UnsafePathError(
            errno.EACCES,
            f"Path denied by sandbox after symlink resolution: {canonical}",
            abs_path,
        )

    return abs_path, canonical, resolved_parent, leaf


def _assert_same_object(fd: int, canonical: str, abs_path: str) -> os.stat_result:
    """Prove the open descriptor is the object at ``canonical``.

    ``canonical`` is the output of :func:`os.path.realpath`, so it contains no
    symlinks: stat-ing it names the approved object no matter what has been
    swapped in the meantime.  Comparing that against ``fstat(fd)`` therefore
    detects exactly the check-then-open race.

    Raises:
        PathSwappedError: on a device/inode mismatch, or if the approved
            object has since disappeared.
    """
    fd_stat = os.fstat(fd)
    try:
        expected = os.stat(canonical)
    except OSError as exc:
        raise PathSwappedError(
            errno.EACCES,
            f"Path vanished between the sandbox check and the open: {abs_path} ({exc})",
            abs_path,
        ) from exc

    if (fd_stat.st_dev, fd_stat.st_ino) != (expected.st_dev, expected.st_ino):
        raise PathSwappedError(
            errno.EACCES,
            "Path was swapped between the sandbox check and the open "
            f"(symlink race): {abs_path}",
            abs_path,
        )
    return fd_stat


def open_verified(
    path: PathLike,
    flags: int,
    mode: int = 0o666,
    *,
    validate: Optional[Callable[[str], bool]] = None,
    allow_special: bool = False,
) -> int:
    """Open ``path`` so that the descriptor provably refers to a checked object.

    This is the resolve-once-then-use-the-handle replacement for the
    ``check(path)`` / ``open(path)`` pattern.  Two distinct strategies, chosen
    by whether ``O_CREAT`` is set:

    * **Existing file** (no ``O_CREAT``): resolve, validate the canonical
      path, open (following symlinks — a symlink *within* the sandbox is
      legitimate), then ``fstat`` the descriptor and require it to be the same
      ``(st_dev, st_ino)`` as the validated canonical path.
    * **Creation** (``O_CREAT``): resolve and validate the parent, open the
      *resolved* parent directory, and create the leaf relative to that
      descriptor with ``O_NOFOLLOW``.  A symlink planted at the leaf and a
      swap of the parent directory both fail closed.  Without ``dir_fd``
      support the leaf is created by path with ``O_NOFOLLOW`` only.

    ``O_NONBLOCK`` is always requested so that a FIFO on the path cannot hang
    the worker; unless ``allow_special`` is set the descriptor is then
    rejected if it is not a regular file.  ``O_NONBLOCK`` is cleared from the
    returned descriptor for regular files, where it has no meaning, so the
    caller can wrap it in a normal file object.

    Args:
        path: Path to open.
        flags: ``os.open`` flags (e.g. ``os.O_RDONLY``).
        mode: Creation mode, used only with ``O_CREAT``.
        validate: Callback receiving the **canonical** (symlink-free) path and
            returning True if the sandbox permits it.  ``None`` skips the
            sandbox check but keeps the swap and special-file guards.
        allow_special: Permit non-regular files (used when the caller has
            already decided a directory or device is expected).

    Returns:
        An open file descriptor the caller owns and must close.

    Raises:
        UnsafePathError: the sandbox rejected the canonical path.
        PathSwappedError: the descriptor is not the validated object.
        SpecialFileError: the target is a FIFO, socket, or device.
        OSError: any ordinary open failure (missing file, permissions, ...).
    """
    creating = bool(flags & os.O_CREAT)
    abs_path, canonical, resolved_parent, leaf = _validated_resolution(
        path, validate, creating
    )

    open_flags = flags | _O_NONBLOCK | _O_NOCTTY | _O_BINARY
    dir_fd: Optional[int] = None
    fd: Optional[int] = None

    try:
        if creating:
            # The leaf must not be an existing symlink, and the directory we
            # create it in must be the one we validated.
            open_flags |= _O_NOFOLLOW
            if _SUPPORTS_DIR_FD:
                dir_fd = os.open(resolved_parent, os.O_RDONLY | _O_DIRECTORY)
                fd = os.open(leaf, open_flags, mode, dir_fd=dir_fd)
            else:
                fd = os.open(os.path.join(resolved_parent, leaf), open_flags, mode)
        else:
            fd = os.open(abs_path, open_flags, mode)
            # Pin to the canonical path captured *before* the open.  Calling
            # realpath() again here would re-walk the symlinks as they stand
            # now and happily confirm the attacker's substitution.
            _assert_same_object(fd, canonical, abs_path)

        if not allow_special:
            kind = _describe_mode(os.fstat(fd).st_mode)
            if kind is not None:
                raise SpecialFileError(
                    errno.EINVAL,
                    f"Refusing to operate on a {kind}: {abs_path}",
                    abs_path,
                )

        _clear_nonblock(fd)
        result, fd = fd, None
        return result
    except OSError as exc:
        # ELOOP from O_NOFOLLOW is a planted symlink, not a generic failure —
        # say so, since the distinction is the whole point of the guard.
        if creating and getattr(exc, "errno", None) == errno.ELOOP:
            raise UnsafePathError(
                errno.ELOOP,
                f"Refusing to create through a symlink at the final path component: {abs_path}",
                abs_path,
            ) from exc
        raise
    finally:
        if fd is not None:
            os.close(fd)
        if dir_fd is not None:
            os.close(dir_fd)


def _clear_nonblock(fd: int) -> None:
    """Drop ``O_NONBLOCK`` from ``fd``, which is meaningless for regular files.

    Best-effort: platforms without :mod:`fcntl` (Windows) never set the flag
    in the first place, so there is nothing to undo.
    """
    if not _O_NONBLOCK:
        return
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        return
    try:
        current = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, current & ~_O_NONBLOCK)
    except OSError:  # pragma: no cover - defensive
        logger.debug("Could not clear O_NONBLOCK on fd %d", fd, exc_info=True)


def read_bytes_verified(
    path: PathLike,
    *,
    validate: Optional[Callable[[str], bool]] = None,
    max_bytes: Optional[int] = None,
) -> bytes:
    """Read a file's bytes through :func:`open_verified`.

    Args:
        path: File to read.
        validate: Sandbox callback, as for :func:`open_verified`.
        max_bytes: Stop after this many bytes; ``None`` reads to EOF.

    Returns:
        The file's contents.

    Raises:
        UnsafePathError: (or a subclass) if the path is unsafe.
        OSError: on ordinary read failures.
    """
    fd = open_verified(path, os.O_RDONLY, validate=validate)
    with os.fdopen(fd, "rb") as handle:
        return handle.read() if max_bytes is None else handle.read(max_bytes)


def read_text_verified(
    path: PathLike,
    *,
    validate: Optional[Callable[[str], bool]] = None,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> str:
    """Read a file's text through :func:`open_verified`.

    Args:
        path: File to read.
        validate: Sandbox callback, as for :func:`open_verified`.
        encoding: Text encoding.
        errors: Decoding error policy (e.g. ``"replace"``).

    Returns:
        The decoded contents.

    Raises:
        UnsafePathError: (or a subclass) if the path is unsafe.
        OSError: on ordinary read failures.
    """
    fd = open_verified(path, os.O_RDONLY, validate=validate)
    with os.fdopen(fd, "r", encoding=encoding, errors=errors) as handle:
        return handle.read()


def write_text_verified(
    path: PathLike,
    data: str,
    *,
    validate: Optional[Callable[[str], bool]] = None,
    encoding: str = "utf-8",
    exclusive: bool = False,
    mode: int = 0o666,
) -> int:
    """Write ``data`` to ``path`` through :func:`open_verified`.

    Args:
        path: File to write.
        data: Text to write.
        validate: Sandbox callback, as for :func:`open_verified`.
        encoding: Text encoding.
        exclusive: Require creation (``O_EXCL``) — the write fails if
            anything already exists at the path.  Use for "new file" tools so
            a symlink planted at the target cannot be followed.
        mode: Creation mode for new files.

    Returns:
        Number of characters written.

    Raises:
        UnsafePathError: (or a subclass) if the path is unsafe.
        FileExistsError: with ``exclusive=True`` when the path exists.
        OSError: on ordinary write failures.
    """
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_EXCL if exclusive else os.O_TRUNC
    fd = open_verified(path, flags, mode, validate=validate)
    with os.fdopen(fd, "w", encoding=encoding) as handle:
        return handle.write(data)


def _resolved_parent_and_leaf(
    path: PathLike,
    validate: Optional[Callable[[str], bool]],
) -> "tuple[str, str, str]":
    """Resolve and validate a path whose *leaf* must not be dereferenced.

    Unlike :func:`_validated_resolution`, the leaf is deliberately left
    unresolved: ``unlink`` and ``rename`` act on the directory entry itself,
    never on a symlink's target.  What still has to be pinned is the parent,
    so the operation lands in the directory that was validated.

    Returns:
        ``(abs_path, resolved_parent, leaf)``.

    Raises:
        UnsafePathError: if ``validate`` rejects the path.
    """
    abs_path = os.path.abspath(os.fspath(path))
    parent, leaf = os.path.split(abs_path)
    resolved_parent = os.path.realpath(parent)
    canonical = os.path.join(resolved_parent, leaf)

    if validate is not None and not validate(canonical):
        raise UnsafePathError(
            errno.EACCES,
            f"Path denied by sandbox after symlink resolution: {canonical}",
            abs_path,
        )
    return abs_path, resolved_parent, leaf


def unlink_verified(
    path: PathLike,
    *,
    validate: Optional[Callable[[str], bool]] = None,
) -> None:
    """Delete ``path`` relative to its validated, symlink-free parent.

    ``unlink`` never follows a symlink at the final component, so the only
    way a delete can land outside the approved location is a swap of a
    *parent* directory between the check and the call.  Opening the resolved
    parent and unlinking relative to that descriptor removes the window.
    Where ``dir_fd`` is unsupported the delete is issued against the resolved
    parent by path, which still avoids re-walking the caller's symlinks.

    Args:
        path: File to delete.
        validate: Sandbox callback receiving the canonical path.

    Raises:
        UnsafePathError: the sandbox rejected the path.
        OSError: on ordinary delete failures.
    """
    _, resolved_parent, leaf = _resolved_parent_and_leaf(path, validate)

    if _SUPPORTS_DIR_FD and os.unlink in getattr(os, "supports_dir_fd", set()):
        dir_fd = os.open(resolved_parent, os.O_RDONLY | _O_DIRECTORY)
        try:
            os.unlink(leaf, dir_fd=dir_fd)
        finally:
            os.close(dir_fd)
    else:  # pragma: no cover - Windows
        os.unlink(os.path.join(resolved_parent, leaf))


def move_verified(
    source: PathLike,
    destination: PathLike,
    *,
    validate: Optional[Callable[[str], bool]] = None,
) -> None:
    """Move ``source`` to ``destination``, both pinned to validated parents.

    Like :func:`unlink_verified`, this acts on directory entries rather than
    on symlink targets — moving a symlink moves the link, which is the
    behaviour callers expect — while pinning both parents so neither end can
    be redirected after its check.  Falls back to :func:`shutil.move` when the
    two ends are on different filesystems (``EXDEV``), where ``rename`` cannot
    work; the fallback uses the resolved paths, so it too acts on what was
    validated rather than re-walking the caller's input.

    Args:
        source: File to move.
        destination: Target path.  Overwritten if it exists — callers that
            want "no clobber" must check first.
        validate: Sandbox callback applied to both canonical paths.

    Raises:
        UnsafePathError: the sandbox rejected either end.
        OSError: on ordinary move failures.
    """
    _, src_parent, src_leaf = _resolved_parent_and_leaf(source, validate)
    _, dst_parent, dst_leaf = _resolved_parent_and_leaf(destination, validate)

    src_full = os.path.join(src_parent, src_leaf)
    dst_full = os.path.join(dst_parent, dst_leaf)

    use_dir_fd = _SUPPORTS_DIR_FD and os.rename in getattr(os, "supports_dir_fd", set())
    src_fd = dst_fd = None
    try:
        if use_dir_fd:
            src_fd = os.open(src_parent, os.O_RDONLY | _O_DIRECTORY)
            dst_fd = os.open(dst_parent, os.O_RDONLY | _O_DIRECTORY)
            os.rename(src_leaf, dst_leaf, src_dir_fd=src_fd, dst_dir_fd=dst_fd)
        else:  # pragma: no cover - Windows
            os.rename(src_full, dst_full)
    except OSError as exc:
        if getattr(exc, "errno", None) != errno.EXDEV:
            raise
        import shutil

        shutil.move(src_full, dst_full)
    finally:
        if src_fd is not None:
            os.close(src_fd)
        if dst_fd is not None:
            os.close(dst_fd)


def ensure_private_dir(path: PathLike, *, mode: int = 0o700) -> str:
    """Create or adopt ``path`` as a directory only we control.

    Predictable paths under a shared, world-writable ``/tmp`` are the classic
    pre-planting target: an attacker creates the path first — as a symlink to
    somewhere of their choosing, or as a directory they own and can read —
    and every later write lands where they decided.  ``mkdir(exist_ok=True)``
    happily adopts both.  This refuses both instead.

    A path is accepted only if, after creation or adoption, it is a real
    directory (not a symlink), owned by the current user where ownership is
    observable, and not writable by group or other unless it carries the
    sticky bit.

    Args:
        path: Directory to create or adopt.
        mode: Creation mode for a directory we create (default owner-only).

    Returns:
        The absolute path, unchanged, on success.

    Raises:
        UnsafePathError: the path exists but is a symlink, is not a
            directory, is owned by another user, or is group/other-writable.
        OSError: on ordinary creation failures.
    """
    abs_path = os.path.abspath(os.fspath(path))

    try:
        os.makedirs(abs_path, mode=mode, exist_ok=False)
    except FileExistsError:
        pass  # Adopt it, subject to the checks below.

    # lstat, never stat: a symlink must be seen as a symlink.
    st = os.lstat(abs_path)

    if stat.S_ISLNK(st.st_mode):
        raise UnsafePathError(
            errno.EACCES,
            f"Refusing to use a pre-planted symlink as a private directory: {abs_path}",
            abs_path,
        )

    if not stat.S_ISDIR(st.st_mode):
        raise UnsafePathError(
            errno.ENOTDIR,
            f"Refusing to use a non-directory as a private directory: {abs_path}",
            abs_path,
        )

    # st_uid is 0 for every entry on Windows, where it carries no meaning.
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise UnsafePathError(
            errno.EACCES,
            f"Refusing to use a directory owned by another user (uid {st.st_uid}): {abs_path}",
            abs_path,
        )

    # Group/other-writable is only tolerable with the sticky bit, which is how
    # /tmp itself is safe: others may create entries but not remove ours.
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH) and not st.st_mode & stat.S_ISVTX:
        raise UnsafePathError(
            errno.EACCES,
            "Refusing to use a group/world-writable directory without the sticky bit "
            f"(mode {stat.S_IMODE(st.st_mode):04o}): {abs_path}",
            abs_path,
        )

    return abs_path
