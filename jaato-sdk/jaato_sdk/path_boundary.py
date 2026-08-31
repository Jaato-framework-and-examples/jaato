"""The cross-process path boundary: relative paths do not cross it.

A *relative* path is not a portable value across a process boundary.  Its
meaning depends on the receiver's cwd — ambient state the sender does not
share and cannot see.  When a client sends
``workspace_path=".jaato-eval-workspaces/arm0"`` to a daemon started from a
different directory, both processes behave "correctly" and the session's
workspace silently splits in half: the client writes its fixture into one
directory, the agent works in another.  Nothing errors; the run completes,
burns tokens, and produces nothing gradeable.

Issue #742 is the worked example.  An eval arm made 25 provider calls and
committed nothing, because the tree it was told to fix was not in the
directory it was given; the grader then reported it blocked for a missing
``.base_commit`` that existed the whole time in the other half.  The
failure is also timing-dependent — the identical command had worked before
the daemon was restarted from elsewhere — so it cannot be fixed by
convention.  The value must be REJECTED, not resolved: resolving is
exactly what supplies the wrong missing half.

Both ends of the boundary use this module.  Senders (the SDK clients)
refuse to put a relative path on the wire; receivers (the daemon
handshake, the session bootstrap envelope) refuse to accept one.  It lives
in the SDK because the contract belongs to the protocol, not to either
side of it.

See ``docs/path-boundary-pattern.md`` for the sibling (MSYS2) boundary.
"""

import os
from typing import Any, Optional


class RelativePathAcrossBoundaryError(ValueError):
    """A relative path arrived across a process boundary.

    Raised by :func:`require_absolute_path`.  Subclasses ``ValueError``
    so existing ``except ValueError`` envelope-validation handlers keep
    working; caught by name where the boundary wants to report the
    field and the offending value to the sender.

    Attributes:
        field: Name of the offending field as the sender knows it
            (e.g. ``"workspace_path"``, ``"working_dir"``).
        value: The relative value exactly as received.
    """

    def __init__(self, message: str, *, field: str, value: str) -> None:
        super().__init__(message)
        self.field = field
        self.value = value


def describe_relative_path(
    field: str,
    value: Any,
    *,
    origin: str = "the daemon boundary",
) -> Optional[str]:
    """Return an error message iff *value* is a relative path.

    The message names the field, quotes the value verbatim, and states
    the cwd the receiver *would* have resolved it against — the two
    halves whose disagreement is otherwise only visible by comparing
    two filesystems.

    Args:
        field: Field name as the sender knows it.
        value: The received value — a ``str``, an ``os.PathLike``, or
            ``None``.  Empty / ``None`` is not a violation (an absent path
            is a different contract from a wrong one).
        origin: Human-readable name of the boundary, used in the
            message (e.g. ``"the session bootstrap envelope"``).

    Returns:
        The error message, or ``None`` when *value* is absent or
        already absolute.
    """
    if not value:
        return None
    # ``os.PathLike`` (a ``pathlib.Path``) is as common at these call sites
    # as a plain str, and carries exactly the same defect.
    value = os.fspath(value)
    # MSYS2-style ``/c/Users/foo`` is absolute under both posixpath and
    # ntpath rules, so no conversion is needed before the test.
    if os.path.isabs(value):
        return None
    return (
        f"{field}={value!r} is a relative path and cannot cross "
        f"{origin}: its meaning depends on the cwd of whichever process "
        f"reads it, and the two sides do not share one. Resolve it to an "
        f"absolute path in the process whose cwd it is relative to "
        f"(this process's cwd is {os.getcwd()!r})."
    )


def require_absolute_path(
    value: Any,
    *,
    field: str,
    origin: str = "the daemon boundary",
) -> Optional[str]:
    """Return *value* unchanged, or raise if it is relative.

    Deliberately does NOT absolutise: resolving a relative path against
    the receiver's cwd is the defect this guard exists to prevent.

    Args:
        value: The received path, or ``None``.
        field: Field name as the sender knows it.
        origin: Human-readable name of the boundary.

    Returns:
        *value*, unchanged (``None`` and ``""`` pass through).

    Raises:
        RelativePathAcrossBoundaryError: when *value* is relative.
    """
    message = describe_relative_path(field, value, origin=origin)
    if message is not None:
        raise RelativePathAcrossBoundaryError(
            message, field=field, value=str(value))
    return value
