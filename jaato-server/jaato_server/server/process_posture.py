"""What the daemon PROCESS is, and what that costs the files it writes.

Two attributes of the daemon process decide what every file an agent
writes into a workspace looks like on disk — the **uid** it runs as and
its **umask** — and until #1168 the daemon could neither announce the
first nor express the second.

WHY THE UID MATTERS HERE.  The runner executes as the daemon's uid:
``RunnerSpawner.spawn`` is ``os.fork()`` + ``os.execvpe`` with no
``setuid`` / ``setgid`` / ``initgroups`` anywhere on the path, and
``session_manager`` says so in its own words — *"the daemon acts on these
paths with ITS credential, not the caller's — and so does the runner,
which is a separate process under the same uid"*.  So on a root daemon
**everything** the agent writes into a user's workspace is root-owned:
``writeNewFile`` and ``file_edit`` backups, the intermediate directories
``mkdir(parents=True)`` creates beneath them, and whatever a ``cli``
subprocess produces — it runs with ``cwd=<workspace_root>``, so one
``git clone`` drops a whole root-owned tree in there.  The workspace's
owner then needs ``sudo`` to overwrite or delete their own files.

AND NOTHING SAID SO.  ``grep -rn 'geteuid'`` across ``server/`` returned
only the egress proxy's sudo decision and a cgroups writability message:
the daemon never noticed.  That is out of step with this codebase, where
every weakened posture announces itself at WARNING —
``scrub_secret_env: none``, ``--ws-unsafe-no-auth``, complain-mode
AppArmor (#1014), ``interactive_shell`` without ``require_confinement``.
The deployment guides are already written around a service user
(``docs/apparmor-setup.md`` and ``docs/runtime-limits-setup.md`` both
``chown jaato:jaato``), so a root daemon is off the documented path; it
simply was not *said*.

WHY A UMASK KNOB AND NOT A CHOWN.  ``grep -rn 'os.umask'`` returned
nothing tree-wide, so a shared-group deployment could not be expressed at
all.  ``umask 002`` plus a setgid workspace makes those root-written
files **group-writable**, which fixes the reported pain — overwrite and
delete without ``sudo`` — without touching ownership.  Chowning after
each write cannot be made complete (it misses the intermediate
directories, every file a subprocess writes, and anything an out-of-tree
plugin writes); a umask is applied once to the process and inherited by
everything downstream, because ``fork`` inherits it and ``exec``
preserves it.

WHAT THIS MODULE DOES **NOT** DO.  It does not drop privileges.  That is
step 3 of #1168 and is deliberately out of scope: it needs a uid field on
``SlotKey`` (a pool slot that has dropped is permanently that uid, which
is #1033's own generating rule), a policy for *which* uid, and an answer
for the WS transport, which has no OS principal to read (``get_client_peer``
returns ``None`` on WS by design, #1074).  A warning and a umask change
neither the process model nor what any file is owned by; they make the
posture visible and its consequence survivable.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

#: The env var carrying the daemon's umask.  ``host``-scoped in
#: ``shared/env_scope.py``: ``os.umask`` is a property of the PROCESS, and
#: the daemon serves every session from one process, so a per-session
#: value could not be applied without racing every other session's turn.
UMASK_ENV_VAR = "JAATO_UMASK"

#: Largest meaningful umask — the permission bits it can mask.
_MAX_UMASK = 0o777

# Once per PROCESS, not once per session.  A daemon serves many sessions
# and one line per session is noise an operator filters out, which is the
# same outcome as silence — the reasoning ``announce_complain_mode_once``
# records for the same class of announcement.
_root_announced = threading.Event()


def running_as_root() -> bool:
    """Whether this process's effective uid is 0.

    ``False`` where the platform has no ``geteuid`` (Windows), which is
    the honest answer: the ownership problem this reports is a POSIX one,
    and inventing a verdict for a platform that cannot be asked would put
    a warning in front of operators it does not describe.
    """
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:
        return False
    return geteuid() == 0


def announce_root_daemon_once() -> None:
    """WARN, once per daemon process, that the agent writes as root.

    No-op unless :func:`running_as_root`, so an ordinary deployment gains
    nothing — and no-op on every call after the first, so the line does
    not repeat per session.

    The message names the consequence (root-owned files in a workspace
    the daemon does not own) and both remedies, in order of preference:
    run as a service user, which is what the deployment guides already
    assume; or, where that is impossible, set a umask so the files stay
    group-writable.
    """
    if not running_as_root():
        return
    if _root_announced.is_set():
        return
    _root_announced.set()
    logger.warning(
        "This daemon is running as ROOT (euid 0) and never drops "
        "privileges — the per-session runner is forked and exec'd under "
        "the same uid.  Every file the agent writes into a workspace is "
        "therefore root-owned: writeNewFile and file_edit backups, the "
        "parent directories they create, and anything a cli subprocess "
        "produces (it runs with cwd=<workspace_root>, so one git clone "
        "or npm install leaves a whole root-owned tree there).  The "
        "workspace's owner then needs sudo to overwrite or delete their "
        "own files.  The deployment guides (docs/apparmor-setup.md, "
        "docs/runtime-limits-setup.md) assume a service user — run the "
        "daemon as one.  Where that is not possible, --umask 002 (or "
        "%s=002) plus a setgid workspace keeps those files "
        "group-writable; it does not change who owns them.",
        UMASK_ENV_VAR,
    )


def parse_umask(raw: Optional[str]) -> Optional[int]:
    """Parse an octal umask string, or explain why it was ignored.

    Args:
        raw: The operator's value — ``--umask``'s argument or
            :data:`UMASK_ENV_VAR`.  ``None`` and blank both mean "not
            configured".

    Returns:
        The mask as an int, or ``None`` when nothing was configured **or**
        the value was malformed.

    Read with ``int(text, 8)``, so the shell spelling (``002``, ``22``)
    works and Python's own ``0o22`` is accepted alongside it — the two
    denote the same mask, and refusing a value whose meaning is
    unambiguous would be pedantry with a cost.

    A malformed value leaves the inherited umask alone and logs at ERROR.
    Substituting an invented mask would silently change the mode of every
    file the daemon and its runners write, for a reason no operator could
    find — the posture ``references.max_transitive_references`` takes
    about a bad ceiling, and the direction that matters: the cost of
    ignoring a typo is a visible ERROR, while the cost of guessing is a
    permission change nobody asked for.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        value = int(text, 8)
    except ValueError:
        logger.error(
            "umask %r is not an octal number — ignoring it and keeping "
            "the umask this process inherited.  Write it the way the "
            "shell builtin does, e.g. 002 or 022.", raw,
        )
        return None
    if not 0 <= value <= _MAX_UMASK:
        logger.error(
            "umask %r is outside 000..777 — ignoring it and keeping the "
            "umask this process inherited.", raw,
        )
        return None
    return value


def resolve_umask(explicit: Optional[str] = None) -> Optional[int]:
    """Resolve the configured umask: the flag, else the env var.

    Args:
        explicit: ``--umask``'s argument, or ``None`` when the flag was
            not passed.

    Returns:
        What :func:`parse_umask` made of whichever source answered, or
        ``None`` when neither did.

    A malformed flag does **not** fall through to the env var.  The
    operator named a value on the command line; quietly serving a
    different one from the environment would answer a question they did
    not ask.
    """
    if explicit is not None:
        return parse_umask(explicit)
    return parse_umask(os.environ.get(UMASK_ENV_VAR))  # env: octal umask for the daemon process, inherited by every runner it forks (e.g. 002 for a shared-group workspace)


def apply_umask(value: Optional[int]) -> Optional[int]:
    """Set the process umask, and say so.

    Args:
        value: The mask from :func:`resolve_umask`; ``None`` is a no-op,
            which is what "nobody configured one" has to mean — the
            inherited umask is the operator's, and replacing it with a
            framework default would be a mode change nobody asked for.

    Returns:
        The previous umask, or ``None`` when nothing was applied.

    Logged at INFO rather than read back on demand: ``os`` exposes no
    umask getter, so reporting an *unconfigured* umask would mean a
    set-and-restore pair — a race against every other thread in a daemon
    that is already running its loop watchdog.  What this reports is the
    value it just wrote, which it knows without asking.
    """
    if value is None:
        return None
    previous = os.umask(value)
    logger.info(
        "umask set to %03o for this daemon process (was %03o); every "
        "runner forked from here inherits it, so it governs what the "
        "agent's files are chmod'd to.", value, previous,
    )
    return previous


def apply_process_posture(umask: Optional[str] = None) -> None:
    """Apply the umask, then announce a root daemon.  Once, at startup.

    The single entry point, so a caller wires one line and cannot wire
    half of it.  Call it from the daemon's ``start()`` **before** anything
    forks or writes: the pre-warm template is forked from this process and
    every pool slot forks from the template, so a umask set afterwards
    would not reach the slots that serve the default path.

    Args:
        umask: ``--umask``'s argument for entry points that have the flag.
            Omit it and :data:`UMASK_ENV_VAR` is the only source, which is
            what the standalone WS server uses.

    Order is deliberate: the umask is applied first so it governs every
    file this process opens from here on, and the root warning is emitted
    second so it appears beside the value that mitigates it.
    """
    apply_umask(resolve_umask(umask))
    announce_root_daemon_once()
