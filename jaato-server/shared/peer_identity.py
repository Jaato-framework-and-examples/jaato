"""Who is on the other end of a Unix socket, and what they may reach.

The IPC transport had no notion of a peer at all: ``get_client_user``
returned a hardcoded ``None`` with the comment *"IPC connections are
local and unauthenticated -- user identity is a WS/SSO concept"*.  That
is true of a daemon serving one human, which is what ``--socket-mode``'s
default ``0o660`` encodes.  It is false of the deployment the same flag
documents -- *"pass 0o666 to opt into world-accessible (e.g. cross-user
containers on a trusted host)"* -- where several OS accounts share one
socket and the daemon cannot tell them apart.

TWO PRINCIPALS, NOT ONE.  The account the daemon RUNS as (owner of
``~/.jaato``, the pooled provider credentials, ``ws.token``) and the
accounts that CONNECT are orthogonal, exactly as they are for any shared
Unix service.  The framework already supports the second axis at the
configuration layer: :func:`shared.config_resolver.resolve_config_search_path`
puts a CLIENT-SUPPLIED ``config_root`` (or ``<workspace>/.jaato``) at the
primary tier and the daemon's ``~/.jaato`` only as a fallback, so each
connecting user can bring their own profiles, agents, instructions -- and
their own ``<provider>_auth.json``.  Session records and logs likewise land
under the user's own workspace (:func:`~shared.config_resolver.workspace_state_path`
never honours ``config_root``).

WHAT WAS MISSING IS THE BINDING.  Nothing tied *which workspace or
config_root you may name* to *who you are*.  Those arrive as plain strings
(``CommandRouter._handle_set_workspace`` reads ``args[0]``;
``ClientConfigRequest`` carries ``working_dir`` / ``config_root`` /
``env_file``) and the only validation on the way down is that they be
absolute -- an anti-ambiguity guard (#742), not an access check.  The
daemon then acts on them with ITS OWN credential, and so does the runner:
``RunnerSpawner._exec_runner`` is ``fork()`` + ``os.execvpe`` with no
``setuid`` anywhere on the path, so a session runs in a different PROCESS
but under the same UID.  A peer who cannot read another user's tree can
therefore have the agent read it for them -- the classic confused deputy,
with the service account as the amplifier.

Every boundary below the transport is *workspace*-shaped and so cannot
answer this: the AppArmor profile is keyed on ``workspace_root`` plus the
rendered profile body (#1033), and ``check_path_with_jaato_containment``
tests paths against the session's own ``workspace_root``.  Point a session
at somebody else's tree and they all do their job perfectly -- they lock
the runner INTO that tree.  The only place the question is answerable is
before the path is accepted, where the peer credential exists and the
session does not.

WHAT THIS MODULE DOES.  Reads the kernel-vouched peer credential off the
socket and answers *could this uid reach this path by itself*.  The daemon
then refuses to act on a path its peer could not have reached, which
collapses the deputy back onto the OS's own model without inventing an ACL
system.

WHAT IT DELIBERATELY DOES NOT DO.  It is an ENTITLEMENT check, not a
sandbox: the session still runs as the daemon's uid, so within a tree the
peer can read, the daemon's own rights still apply.  Closing that residue
means dropping privileges between ``fork()`` and ``exec()``, which needs a
privileged daemon and a uid-keyed slot pool (a uid is a property the next
session cannot change, so :class:`server.runner_pool.SlotKey` would have to
carry it, by that class's own stated rule).  That is a separate decision
about the process model; this module is what makes the current model
honest.

ONE DEFINITION, NO JAATO IMPORTS.  Pure stdlib, the shape
``shared/apparmor_label.py`` and ``shared/runtime_limits.py`` already have,
so the transport can import it before any plugin discovery has run and so
the answer cannot be derived twice in two places.

POSITIVE EVIDENCE ONLY.  Every reader here returns ``None`` for *"could not
tell"* and never guesses -- the posture #1014 and #1023 take about
confinement labels.  What the CALLER does with ``None`` is a policy
decision stated at the call site, and for an entitlement it is to refuse:
granting access on ignorance is the failure this exists to prevent.
"""

from __future__ import annotations

import logging
import os
import socket
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Optional

logger = logging.getLogger(__name__)

#: ``struct ucred { pid_t pid; uid_t uid; gid_t gid; }`` -- three 32-bit
#: ints on every Linux ABI CPython supports.
_UCRED_FORMAT = "3i"
_UCRED_SIZE = struct.calcsize(_UCRED_FORMAT)

#: Permission bits, named so a call reads as a question rather than as
#: octal arithmetic.
READ = 0o4
WRITE = 0o2
SEARCH = 0o1


@dataclass(frozen=True)
class PeerCredentials:
    """The OS account on the far end of a Unix domain socket.

    Obtained from the kernel at ``connect(2)`` time, so it is asserted by
    nobody and cannot be forged by the client -- unlike the WS path's
    bearer token, which is a string the client chose to send.

    ``username`` is ``None`` when the uid resolves to no passwd entry
    (a container mapping, a deleted account).  That is not an error: the
    credential is still valid and :attr:`identity` falls back to the
    numeric form, because an unresolvable name must not become an
    unattributable session.
    """

    uid: int
    gid: int
    pid: Optional[int] = None
    username: Optional[str] = None

    @property
    def identity(self) -> str:
        """The string form used for attribution (``created_by``, the ledger
        ``user_id``, ``PermissionResolvedEvent.user_id``).

        The username when one resolves, else ``uid:<n>``.  Never empty, so
        a caller can always tell "this session had a peer" from "this
        session had none".
        """
        return self.username or f"uid:{self.uid}"

    def __str__(self) -> str:  # pragma: no cover - diagnostic only
        pid = f" pid={self.pid}" if self.pid is not None else ""
        return f"{self.identity} (uid={self.uid} gid={self.gid}{pid})"


def _resolve_username(uid: int) -> Optional[str]:
    """The passwd name for *uid*, or ``None`` when it resolves to none."""
    try:
        import pwd  # noqa: PLC0415 - absent on Windows, imported where used

        return pwd.getpwuid(uid).pw_name
    except (ImportError, KeyError, OSError):
        return None


def peer_credentials(sock: Optional[socket.socket]) -> Optional[PeerCredentials]:
    """Read the kernel's record of who opened *sock*.

    Linux only, via ``SO_PEERCRED``.  Returns ``None`` everywhere else --
    a Windows named pipe, a BSD/macOS socket (whose ``LOCAL_PEERCRED``
    carries a different, un-stdlib'd ``struct xucred``), a TCP socket, or
    a socket the option is refused on.

    ``None`` means EXACTLY "this transport cannot tell me", and callers
    must read it that way: on a platform with no peer credential the
    daemon is in the same position it was in before this module existed,
    and the pre-existing posture -- the socket's file mode is the access
    control -- is the one that applies.

    Args:
        sock: The accepted connection, or ``None`` when the transport
            exposes no socket object.

    Returns:
        The peer's credential, or ``None`` when it cannot be read.
    """
    if sock is None:
        return None
    option = getattr(socket, "SO_PEERCRED", None)
    if option is None:
        return None
    try:
        raw = sock.getsockopt(socket.SOL_SOCKET, option, _UCRED_SIZE)
        pid, uid, gid = struct.unpack(_UCRED_FORMAT, raw)
    except (OSError, struct.error) as exc:
        logger.debug("peer_credentials: SO_PEERCRED unavailable: %s", exc)
        return None
    return PeerCredentials(
        uid=uid, gid=gid, pid=pid or None, username=_resolve_username(uid),
    )


def daemon_uid() -> Optional[int]:
    """The uid this process runs as, or ``None`` on a platform without one."""
    getuid = getattr(os, "getuid", None)
    return getuid() if getuid is not None else None


def peer_is_the_daemon(peer: Optional[PeerCredentials]) -> bool:
    """True when *peer* is the account the daemon itself runs as.

    A question about ONE CONNECTION, not about the daemon.  Nothing here
    classifies a deployment as single- or multi-user, and nothing could:
    the same daemon answers True for a client started from its own login
    and False for a colleague's, one connection at a time.  Under a
    dedicated service account with humans connecting as themselves it is
    False for every human connection, so the check runs on all of them.

    Callers skip the entitlement check on a True, and the reason is NOT
    that such a deployment has one user.  It is that the refusal would
    deny nothing: a process under the daemon's own uid can already
    ``ptrace`` it, read its memory and read ``~/.jaato`` directly, so
    every path the daemon can open is reachable by that account by
    simpler means.  What the skip buys is that the common single-login
    case pays no ``stat`` and no group lookup.
    """
    if peer is None:
        return False
    own = daemon_uid()
    return own is not None and peer.uid == own


def peer_group_ids(peer: PeerCredentials) -> FrozenSet[int]:
    """Every gid *peer* is a member of -- primary plus supplementary.

    The supplementary set needs a username, so a uid with no passwd entry
    contributes its primary gid alone.  That direction is the safe one: a
    missing group can only cause a refusal, never an unearned grant.
    """
    groups = {peer.gid}
    getgrouplist = getattr(os, "getgrouplist", None)
    if peer.username and getgrouplist is not None:
        try:
            groups.update(getgrouplist(peer.username, peer.gid))
        except (KeyError, OSError) as exc:
            logger.debug(
                "peer_group_ids: supplementary groups unavailable for %s: %s",
                peer.username, exc,
            )
    return frozenset(groups)


def _mode_permits(
    st: os.stat_result, peer: PeerCredentials, groups: FrozenSet[int], want: int,
) -> bool:
    """Whether the classic ``rwx`` bits grant *want* to *peer* on *st*.

    Evaluates owner / group / other in the kernel's own order -- the first
    matching class decides, so a file owned by the peer with mode ``0o077``
    denies them, which is what the kernel does too.

    ACLs, capabilities and MAC labels are NOT consulted, so this can be
    stricter than the kernel but never looser.  Stricter is the direction
    that fails safe: the worst outcome is refusing a path the peer could
    in fact reach, which is a visible error naming the path, not a silent
    grant.
    """
    mode = st.st_mode
    if peer.uid == st.st_uid:
        return bool((mode >> 6) & want == want)
    if st.st_gid in groups:
        return bool((mode >> 3) & want == want)
    return bool(mode & want == want)


def _nearest_existing(path: Path) -> Optional[Path]:
    """*path* if it exists, else its closest existing ancestor.

    A workspace the daemon has not provisioned yet does not exist, and
    ``provisioned`` in the session record says the daemon creates them --
    so refusing every not-yet-created path would refuse the normal case.
    The question for such a path is whether the peer could CREATE it,
    which is a question about the directory that would hold it.
    """
    for candidate in (path, *path.parents):
        try:
            if candidate.exists():
                return candidate
        except OSError:
            return None
    return None


def path_reachable_by(
    path: str, peer: PeerCredentials,
) -> Optional[bool]:
    """Could *peer*, acting as themselves, reach *path*?

    The rule, and why each half is what it is:

    ========================  ==================================================
    *path* exists             ``r-x`` on it -- the minimum to open the tree at
                              all.  Deliberately NOT ``w``: a read-only tier is
                              a legitimate and desirable shape in exactly the
                              multi-user deployment this serves (an org-wide
                              ``config_root`` of shared profiles under
                              ``/opt``, readable by everyone and writable by
                              none of them), and requiring write would refuse
                              it.
    *path* does not exist     ``-wx`` on the nearest existing ancestor -- the
                              minimum to create it.  The daemon provisions
                              workspaces, so this is the ordinary case for a
                              first session in a new directory.
    every ancestor            ``--x`` (search).  Without it the peer cannot
                              traverse to the leaf however permissive the leaf
                              itself is, which is precisely the protection a
                              ``0o700`` home directory provides.
    ``peer.uid == 0``         True, for any absolute path.  root bypasses DAC
                              in the kernel, so a check that refused root
                              would be reporting something untrue.  A
                              RELATIVE path is still unknowable and still
                              answers ``None``, whoever asks -- the question
                              is what the string means, not who may read it.
    ========================  ==================================================

    Symlinks are resolved before any of it, so a link planted inside a
    world-writable directory is judged by what it points AT -- the same
    rule ``_start_unix_socket_server`` applies to the socket path itself.

    Args:
        path: An absolute path.  A relative one is refused by the #742
            guards before reaching here; it is treated as unknowable
            rather than resolved against the daemon's cwd.
        peer: The credential to evaluate against.

    Returns:
        ``True`` (reachable), ``False`` (demonstrably not), or ``None``
        when it could not be determined -- the path could not be resolved
        or stat'd, or it is relative.  ``None`` is not a grant; see the
        module docstring.
    """
    if not path or not os.path.isabs(path):
        return None
    if peer.uid == 0:
        return True
    try:
        resolved = Path(path).resolve()
    except OSError:
        return None

    target = _nearest_existing(resolved)
    if target is None:
        return None

    groups = peer_group_ids(peer)
    try:
        for ancestor in reversed(target.parents):
            if not _mode_permits(os.stat(ancestor), peer, groups, SEARCH):
                return False
        leaf_wants = READ | SEARCH if target == resolved else WRITE | SEARCH
        return _mode_permits(os.stat(target), peer, groups, leaf_wants)
    except OSError as exc:
        logger.debug("path_reachable_by: cannot stat under %s: %s", resolved, exc)
        return None


def describe_unreachable_path(
    field: str, path: Optional[str], peer: PeerCredentials,
) -> Optional[str]:
    """A refusal message for *path*, or ``None`` when the peer may use it.

    Shaped after :func:`shared.path_utils.describe_relative_path` so the
    two client-path guards report the same way: one line per offending
    field, naming the field, the value and the reason, collected by the
    caller into a single error rather than one round trip per field.

    An absent or empty *path* is not a violation -- the field is simply
    not set, and a daemon-side default applies.
    """
    if not path:
        return None
    verdict = path_reachable_by(path, peer)
    if verdict is True:
        return None
    if verdict is False:
        return (
            f"{field}={path!r} is not reachable by {peer.identity} — the "
            f"daemon will not open on your behalf a path you could not "
            f"open yourself"
        )
    return (
        f"{field}={path!r} could not be checked against {peer.identity} — "
        f"refusing rather than granting on an unknown"
    )


#: Operator opt-out.  Truthy disables every entitlement check, so the daemon
#: acts on whatever path a peer names — the behaviour of every release before
#: this module.  Announced at WARNING the first time it takes effect, the
#: posture ``--ws-unsafe-no-auth`` and ``scrub_secret_env: none`` already
#: take: a weakened boundary is never silent.
TRUST_PEER_PATHS_ENV = "JAATO_IPC_TRUST_PEER_PATHS"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Guards the one-per-process opt-out warning.  A per-call WARNING on the
#: handshake path would be one line per connect, which is how an important
#: line becomes noise nobody reads.
_announced_opt_out = False


def path_checks_disabled() -> bool:
    """Whether the operator has switched entitlement checks off.

    Reads the environment live rather than freezing it, unlike
    ``server.revive_policy.capture()``: this is consulted only on the
    handshake path, never inside a turn, so the window
    ``JaatoServer._with_session_env`` opens over the daemon's ``os.environ``
    cannot reach it — a session's own ``.env`` is applied around
    ``send_message``, long after a client's paths were accepted.
    """
    global _announced_opt_out
    disabled = os.environ.get(TRUST_PEER_PATHS_ENV, "").strip().lower() in _TRUTHY  # env: switch OFF the IPC peer check -- act on any workspace / config_root a client names without verifying the connecting account could reach it
    if disabled and not _announced_opt_out:
        _announced_opt_out = True
        logger.warning(
            "%s is set: the daemon will act on any workspace or config_root "
            "a client names, without checking that the connecting account "
            "could reach it. On a shared socket this restores the "
            "confused-deputy exposure the check exists to close.",
            TRUST_PEER_PATHS_ENV,
        )
    return disabled


def unreachable_client_paths(
    fields: "list[tuple[str, Optional[str]]]",
    peer: Optional[PeerCredentials],
) -> "list[str]":
    """The refusal messages for client-supplied paths *peer* may not use.

    The single policy both client-path chokepoints call, so the workspace
    command and the config handshake cannot drift apart on when a check
    applies.

    Returns ``[]`` — check not applicable — in three cases, each for a
    different reason:

    ``peer is None``
        The transport reports no peer (WS, Windows pipes, non-Linux
        sockets).  There is nothing to evaluate against, and that
        transport's own access control is what applies.

    :func:`peer_is_the_daemon`
        THIS CONNECTION is from the account the daemon runs as, which can
        already reach anything the daemon can by simpler means (``ptrace``,
        its memory, ``~/.jaato``), so a refusal would deny nothing.  Note
        what this is not: a statement about the deployment.  The same
        daemon checks a colleague's connection and skips its owner's, and
        under a dedicated service account it checks every human connection
        there is.

    :func:`path_checks_disabled`
        The operator asked for the pre-check behaviour, loudly.

    Everything else is a shared socket, where the check arms itself: there
    is no knob to remember to switch on, because a control nobody enables
    is a control nobody has.

    Args:
        fields: ``(field_name, value)`` pairs, reported in the order given.
            A ``None`` or empty value is not a violation — the field is
            simply unset and a daemon-side default applies.
        peer: The connecting account, or ``None``.

    Returns:
        One message per offending field, empty when the peer may use all
        of them or the check does not apply.
    """
    if peer is None or peer_is_the_daemon(peer) or path_checks_disabled():
        return []
    return [
        message
        for message in (
            describe_unreachable_path(field, value, peer)
            for field, value in fields
        )
        if message
    ]
