"""Per-session runner subprocess spawn helper (Phase 3 §3.12).

Extracted from ``server/__main__.py:_spawn_session_runner`` (Phase 2
task 2.3) so both the IPC apparmor pre-init hook and the WS-server
apparmor pre-init hook can spawn runners through the same code
path.

Lifecycle:

1. Caller (an apparmor pre-init hook) finishes
   ``apparmor.provision_profile`` for the session.
2. Caller invokes :func:`spawn_session_runner`.
3. Function spawns the runner via :class:`RunnerSpawner`, opens a
   :class:`RunnerRPCClient` against the parent end of the
   socketpair, starts the read-loop on the daemon's asyncio loop,
   and attaches the RPC handle onto the JaatoServer via
   ``server.set_runner_rpc(rpc, spawned)``.
4. After this returns, plugins discovered during
   ``server.initialize()`` see ``registry.runner_rpc`` set at
   configure time.
5. Caller invokes :func:`dispatch_bootstrap_envelope` to send the
   ``session.bootstrap`` RPC so the runner-side
   :class:`shared.jaato_session.JaatoSession` host is populated.

Failures raise; the caller catches and downgrades the session to
``sandbox_mode = "soft"`` per the §4.6 fallback contract.

Phase 3 §7c step 2: the bootstrap-envelope dispatch + the
envelope builder live in this module so both IPC + WS callers
share the implementation.  Pre-§7c-step-2 the helpers lived only
in ``server/__main__.py`` (the IPC entry point) and the WS path
had no bootstrap dispatch at all — every WS session left the
runner-side ``JaatoSession`` host unpopulated, blocking the
seat-flip on WS sessions.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

# Leaf module (no package-internal deps of its own): the strict-posture egress
# failure is the one wire-up error the spawn path must NOT swallow, so the type
# has to be nameable in an ``except`` clause without a lazy import.
from jaato_server.server.egress_proxy.errors import (
    EgressEnforcementError as _EgressEnforcementError,
)

from jaato_server.server.confinement_id import (
    confinement_id_from_profile_name, session_tmpdir,
)

# Named at import time because ``_profile_runtime_limits`` type-CHECKS the
# profile's declared block (#735) rather than duck-typing it, and a lazy
# import inside a per-session helper would pay the lookup on every spawn.
from jaato_server.shared.runtime_limits import RuntimeLimits
from jaato_server.server import session_new_timing
from jaato_server.shared.plugins.workspace_venv import inject_workspace_venv
from jaato_server.shared.plugins.workspace_home import (
    ensure_workspace_home_dir, inject_workspace_home,
)
from jaato_server.shared.utils.errors import exc_message
from jaato_server.shared.peer_identity import PeerCredentials
from jaato_server.shared.privilege_drop import RunnerUser
from jaato_server.server.runner_user import (
    current_policy as current_runner_uid_policy,
    prepare_runner_owned_paths,
    resolve_runner_user,
    runner_owned_paths,
    runner_uid_of,
)
from jaato_server.shared.private_tmp import (
    ensure_private_tmp_dir, resolve_private_tmp,
)


if TYPE_CHECKING:  # pragma: no cover — types only
    from jaato_server.shared.session_envelope import SessionInitEnvelope


logger = logging.getLogger(__name__)


def _pool_enabled() -> bool:
    """Read the ``JAATO_RUNNER_POOL_ENABLED`` env var.

    Pool PR 5e: default flipped to **enabled**.  Sessions consume
    pre-warm pool slots unless the operator explicitly disables the
    pool by setting ``JAATO_RUNNER_POOL_ENABLED=false`` (or
    ``0`` / ``no`` / ``off``).

    Rationale: PRs 4-5d shipped the pool's structural correctness,
    operational robustness (subreaper + watchdog), startup determinism
    (READY handshake), and observability (telemetry).  Default-on
    delivers the cascade speedup to every operator without per-deployment
    configuration.

    Disable cases (set env var to a falsy value):
      - Suspected pool regression — bisect against cold-spawn.
      - Operator host with extremely tight memory budget where the
        idle pool's ~150-300 MiB footprint is undesirable.
      - Custom plugin set that doesn't work with fork-from-template
        (would need to be a runner-tier plugin with module-global
        non-fork-safe state — none exist today per the 2026-05-13
        audit).

    Pre-PR-5e the default was opt-in (off); explicit ``true`` was
    required.  Operators who set ``true`` explicitly are unaffected
    by the flip.  Operators who never set the var get the pool
    automatically post-PR-5e.
    """
    raw = os.environ.get("JAATO_RUNNER_POOL_ENABLED", "").strip().lower()  # env: use the pre-warm runner pool (faster session bootstrap); false to always cold-spawn
    # Empty (unset) → enabled.  Explicit-falsy → disabled.  Anything
    # else (truthy or unrecognised) → enabled.
    return raw not in ("0", "false", "no", "off")


def _ensure_session_tmpdir(
    session_id: str, profile_name: Optional[str], confinement: Any = None,
) -> None:
    """Create the directory the runner's ``TMPDIR`` will point at.

    The daemon runs unconfined, so it is the only party that can make
    the boundary directory.  See the call site for why this is here
    rather than in either spawn branch.
    """
    path = session_tmpdir(session_id, _confinement_id_of(profile_name, confinement))
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "spawn_session_runner: failed to create session tmpdir %s "
            "(%s: %s) — the runner will fail its tempfile probe",
            path, type(exc).__name__, exc,
        )


def _ensure_runner_log_dir(log_path: Optional[str]) -> None:
    """Create ``<ws>/.jaato/logs`` before either spawn branch.

    Best-effort and audible, as :func:`_ensure_session_tmpdir` is.  Created
    before the runner user is resolved, so the hand-over (#1168) finds it
    owned by the daemon and gives it to that user.  ``None`` does nothing.
    """
    if not log_path:
        return
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    except OSError as exc:
        logger.warning(
            "spawn_session_runner: failed to create the runner log "
            "directory for %s (%s: %s) — the runner will have no log",
            log_path, type(exc).__name__, exc,
        )


def resolve_session_private_tmp(
    server: Any,
    workspace_path: Optional[str],
    managed_workspace_root: Optional[str],
) -> Optional[str]:
    """Decide this CONFINED session's private ``/tmp`` and stash it (#1381).

    Called on the confined provisioning path only (the WS pre-init hook and
    the IPC ``_provision_apparmor_for_session``), before the profile is
    rendered: the answer changes the rendered body and so the confinement
    id.  Stashed on the server as ``_private_tmp_dir`` so the spawn, the
    cold-spawn pre-exec and the envelope all read the value the profile was
    rendered with, rather than deciding it a second time.

    Returns:
        ``<ws>/.tmp`` when the session gets a private ``/tmp``, else ``None``.
    """
    value = resolve_private_tmp(
        getattr(server, "_profile", None), workspace_path, managed_workspace_root,
    )
    server._private_tmp_dir = value
    return value


def private_tmp_kwargs(private_tmp_dir: Optional[str]) -> Dict[str, str]:
    """``{"private_tmp_dir": v}`` when set, else ``{}`` (#1381).

    Splatted into ``provision_profile`` / ``confinement_id_for_boundary``
    so a session WITHOUT a private ``/tmp`` calls them exactly as before --
    a test double or an out-of-tree manager that predates the keyword keeps
    working for every session that does not use the feature.
    """
    return {"private_tmp_dir": private_tmp_dir} if private_tmp_dir else {}


def _confinement_id_of(profile_name: Optional[str], confinement: Any) -> Optional[str]:
    """The boundary's id: the handle's, else read out of an AppArmor name."""
    if confinement is not None:
        return confinement.confinement_id
    return confinement_id_from_profile_name(profile_name or "")


def _envelope_descriptor_of(profile_name: Optional[str], confinement: Any) -> Any:
    """``SessionInitEnvelope.confinement``: SELinux from the handle, AppArmor
    from the profile name (which an AppArmor handle's label equals)."""
    from jaato_server.server.confinement.base import is_selinux, selinux_descriptor

    if is_selinux(confinement):
        return selinux_descriptor(confinement)
    return _apparmor_descriptor(profile_name or "")


def _boundary_label(profile_name: Optional[str], confinement: Any) -> str:
    """What the runner is confined to, for the log: the AppArmor profile
    or the SELinux context, ``(none)`` when unconfined."""
    if confinement is not None:
        return confinement.label
    return profile_name or "(none)"


def runner_log_path(workspace_path: Optional[str], session_id: str) -> Optional[str]:
    """The session's runner log, ``<ws>/.jaato/logs/runner-<id>.log``.

    ``None`` without a workspace.  One definition for the two readers: the
    cold spawn's child opens it before exec, and the envelope carries it so
    every bootstrap, a pool slot's included, points fds 1 and 2 at it.
    """
    if not workspace_path:
        return None
    return os.path.join(workspace_path, ".jaato", "logs", f"runner-{session_id}.log")


def _slot_boundary_note(profile_name: Optional[str], confinement: Any) -> str:
    """How a pool slot ends up in its boundary, for the spawn log line."""
    from jaato_server.server.confinement.base import is_selinux

    if is_selinux(confinement):
        return f"{confinement.label}, entered at fork"
    if profile_name:
        return f"{profile_name}, entered at bootstrap"
    return "(unconfined)"


def _pool_may_serve(pool_manager: Any, cgroup_attach: Any) -> bool:
    """May a pre-warm slot serve this session?

    Not with a cgroup to attach (a slot is already in the daemon's).  An
    SELinux session may (phase 4): it is served by a slot forked into its
    boundary, see :func:`_acquire_pool_slot`.
    """
    return (pool_manager is not None and _pool_enabled()
            and cgroup_attach is None)


def _acquire_pool_slot(
    pool_manager: Any, server: Any, *, cascade_driver_id: Optional[str],
    workspace_path: Optional[str], profile_name: Optional[str],
    runner_user: Any, confinement: Any, log_path: Optional[str] = None,
) -> Any:
    """A pool slot for this session, or ``None`` (cold spawn).

    An idle slot that fits the key if there is one; otherwise one forked
    for this session from the template (:meth:`PoolManager.fork_slot_on_demand`),
    so a miss costs a fork, not a cold-spawned runner that holds its whole
    working set privately.  ``None`` only when the template cannot fork.

    The key: a slot's warm plugin state was built from ITS config root;
    its threads are stuck in the AppArmor profile it was last confined to
    (#1023); a dropped slot is that uid for life (#1168); and an SELinux
    slot is in the domain it was forked into (phase 4).  See
    ``runner_pool.SlotKey``.  Passed raw; ``SlotKey.build`` folds ``""``
    to ``None``.

    An SELinux session no idle slot fits gets a slot forked into its
    boundary (selinux-backend.md §7.2): the child points its output at
    *log_path* (the session's runner log), then enters the private
    ``/tmp``, the uid and the domain while it has one thread.
    """
    from jaato_server.server.confinement.base import is_selinux
    from jaato_server.server.runner_pool import SlotKey

    selinux = is_selinux(confinement)
    fields = dict(
        cascade_driver_id=cascade_driver_id,
        config_root=getattr(server, "config_root", None),
        workspace_root=workspace_path,
        profile_name=profile_name,
        runner_uid=runner_uid_of(runner_user),
        selinux_boundary=confinement.confinement_id if selinux else None,
    )
    slot = pool_manager.acquire_slot(**fields)
    if slot is not None:
        return slot
    if not selinux:
        return pool_manager.fork_slot_on_demand(SlotKey.build(**fields))
    entry = {
        "context": confinement.label,
        "log_path": log_path,
        "private_tmp": session_private_tmp(server, profile_name, confinement),
        "runner_user": runner_user.to_dict() if runner_user is not None else None,
    }
    return pool_manager.fork_slot_into(SlotKey.build(**fields), entry)


def session_private_tmp(
    server: Any, profile_name: Optional[str], confinement: Any = None,
) -> Optional[str]:
    """The private ``/tmp`` this runner must enter, or ``None`` (#1381).

    Only a CONFINED runner gets one: the namespace exists to make the
    profile's ``/tmp`` grant safe, and an unconfined runner has none.  So an
    empty ``profile_name`` (unconfined, or provisioning downgraded to soft)
    answers ``None`` whatever was stashed, which also keeps an unconfined
    pool slot free of a namespace another workspace would inherit.
    """
    if not profile_name and confinement is None:
        return None
    return stashed_private_tmp(server)


def stashed_private_tmp(server: Any) -> Optional[str]:
    """The value :func:`resolve_session_private_tmp` stashed, or ``None``.

    ``isinstance`` rather than truthiness: a test double's attribute is a
    mock, and only a real path may reach a render, the envelope or a mount.
    """
    value = getattr(server, "_private_tmp_dir", None)
    return value if isinstance(value, str) and value else None


def spawn_session_runner(
    *,
    server: Any,  # JaatoServer (forward-typed; importing the real
                  # type creates a cycle through server/core.py).
    session_id: str,
    workspace_path: str,
    profile_name: str,
    daemon_loop: asyncio.AbstractEventLoop,
    disable_confine: bool = False,
    cgroup_attach: Optional[Callable[[], None]] = None,
    pool_manager: Any = None,
    cascade_driver_id: Optional[str] = None,
    managed_workspace_root: Optional[str] = None,
    peer: Any = None,
    confinement: Any = None,
) -> None:
    """Spawn the per-session runner subprocess and wire its RPC handle
    onto the JaatoServer.

    Pool PR 4: when *pool_manager* is supplied AND
    ``JAATO_RUNNER_POOL_ENABLED`` is set AND the pool has an idle
    slot, the session reuses that slot's pre-warm runner subprocess
    instead of paying the per-session fork+exec+plugin-imports cost
    (~10-15s on v62 step 6).  The slot's daemon-side socket is
    wrapped in a :class:`SpawnedRunner` and the rest of the bootstrap
    flow (RunnerRPCClient + session.bootstrap RPC) is identical to
    the cold-spawn path — by design, so PR 4 doesn't fork a parallel
    code path that has to be maintained alongside the existing one.

    Args:
        server: The session's :class:`JaatoServer` instance.
        session_id: Session identifier (passed via env to the runner).
        confinement: The provisioned ``ConfinementHandle`` when it is an
            SELinux one, else ``None`` (an AppArmor boundary is carried by
            *profile_name* as before).  An SELinux session is always cold
            spawned, and its runner enters the domain by the exec.
        workspace_path: Session workspace; used both as the runner's
            cwd and as the prefix for the per-session log file path
            (plan §5.1).
        profile_name: AppArmor profile name (already loaded in the
            kernel).  Required unless *disable_confine* is set —
            then it can be empty (the runner runs unconfined).
        daemon_loop: The daemon's main asyncio loop — needed to run
            ``RunnerRPCClient.start()`` since it's async.
        disable_confine: Phase 3 §7a — skip kernel-level
            confinement.  Used by the always-spawn path when the
            client did not opt into apparmor.  The runner spawns
            with ``JAATO_RUNNER_DISABLE_CONFINE=1``; tool execution
            runs in the runner subprocess but without an AppArmor
            profile applied.  The runner-RPC dispatch surface is
            still available; the trade-off is process isolation
            without kernel-enforced FS confinement.
        cgroup_attach: Phase 3 §7d — optional zero-arg callable
            forwarded to ``RunnerSpawner.spawn`` to migrate the
            forked child into the per-session cgroup before exec.
            Caller (the WS pre-init hook) obtains via
            ``CgroupsManager.make_attach_callback(session_id)``
            after provisioning the cgroup.  ``None`` means no
            cgroup attach — the IPC session path (no cgroup
            provisioned) passes ``None``.
        managed_workspace_root: The WS server's provisioning root
            (#1225).  Non-None means this session's workspace is one the
            daemon manages, so the workspace HOME (``<ws>/.home/``)
            defaults on; ``None`` (IPC / user-CWD sessions) leaves it
            opt-in via ``plugin_configs.cli.workspace_home``.  Used only
            to create the home directory + its gitignore before spawn,
            as the session tmpdir is (#1171).
        peer: The requesting connection's
            :class:`~shared.peer_identity.PeerCredentials` (IPC), or
            ``None`` (WS, or a caller with no connection).  Read only by
            the ``peer`` runner uid policy (#1168 step 3,
            :mod:`server.runner_user`): the runner then drops to that uid
            (cold spawn: in the forked child after cgroup + private
            ``/tmp``; pool slot: bootstrap step 1b3), the pool key carries
            the uid, and the directories the daemon makes for the runner
            are handed to it first.
        cascade_driver_id: Phase 2 cascade-sharing tenant ID.  When
            non-None, the pool acquire walks for a slot already
            affined to this cascade (warm plugin state, warm LSP
            connections from prior sessions).  ``None`` (default)
            means standalone session — pool acquires any PURE IDLE
            slot, no cascade affinity stamped.
        pool_manager: Pool PR 4 — the daemon's
            :class:`server.runner_pool.PoolManager`.  When non-None
            AND the pool routing flag is enabled AND the pool has an
            idle slot, the session is served by a pre-warm slot
            instead of a cold-spawned runner.  ``None`` (or empty
            pool, or flag disabled) falls back to cold-spawn.

            Pool routing gates (post-PR 5a):
              - ``pool_manager`` is wired by the daemon.
              - ``JAATO_RUNNER_POOL_ENABLED`` is set.
              - No ``cgroup_attach`` supplied (PR 5b territory —
                slot is template-child, mid-life cgroup migration
                requires subreaper coordination).

            Per-slot AppArmor self-confinement landed in PR 5a:
            slots accept ``envelope.profile_name`` and call
            ``aa_change_profile`` in ``bootstrap_session`` step 1c
            BEFORE plugin initialize / prefetch.  The pre-PR-5a
            ``disable_confine`` gate is removed — sessions with
            AppArmor opt-in are now eligible for pool routing.

            #1033: a slot that has ALREADY been confined does not
            transition again — ``aa_change_profile`` is per-task and its
            other threads could not follow (#1023).  What makes reuse
            safe is that the profile name is part of the reuse key, so
            such a slot is only handed to a session wanting the profile
            it already wears; the transition above is the first-session
            path.

    Raises:
        RunnerUserRefused: the runner uid policy names a user who cannot
            read the daemon's interpreter or jaato packages.  Both callers
            REFUSE the session on it — never the in-process fallback, which
            would run the model's tools in the root daemon.
        RuntimeError: when *daemon_loop* is None or the runner-RPC
            start times out.  Caller catches and downgrades to
            ``sandbox_mode = "soft"`` (or omits the field entirely
            for the always-spawn-no-apparmor path).
        Exception: any spawn / RPC failure.  Caller catches and
            downgrades.
    """
    from jaato_server.server.runner_spawner import SpawnedRunner
    from jaato_server.server.runner_rpc_client import RunnerRPCClient

    if daemon_loop is None:
        raise RuntimeError(
            "spawn_session_runner: daemon loop unavailable; cannot "
            "start RunnerRPCClient"
        )

    log_path = runner_log_path(workspace_path, session_id)

    # ----- The session's tmpdir, before either branch (#1171) -----
    # ``RunnerSpawner.spawn`` makes this directory before it forks,
    # which covers the cold-spawn branch and nothing else: a pool-served
    # session never calls ``spawn``, so on the DEFAULT path nobody
    # created it.  The runner cannot create it for itself — the profile
    # grants ``/tmp/jaato-<confinement_id>/**`` but not ``/tmp/``, so
    # the confined child may make its own subdirectory and not the
    # boundary directory above it.
    #
    # Done here rather than in the pool branch because one boundary
    # covers both, and a per-branch mkdir is the shape that leaves one
    # path armed and the other silently not (#735).  ``spawn``'s own
    # call is then an idempotent second one, kept because it also
    # serves callers that reach the spawner directly (the isolated
    # sub-runner).
    #
    # Best-effort, and audible: a runner whose tmpdir is missing fails
    # at plugin-import time with ``No usable temporary directory``,
    # which is #1171 itself — so this must not fail silently, and must
    # not take down a session that would otherwise run.
    _ensure_session_tmpdir(session_id, profile_name, confinement)
    # ...and the runner log's directory, for the same reason: a cold
    # spawn's child made it before exec, a pool slot never did, so the
    # first pool-served session of a fresh workspace had nowhere to log
    # (SELinux phase 4 kernel run).
    _ensure_runner_log_dir(log_path)

    # ----- The session's workspace HOME, before either branch (#1225) -----
    # The daemon creates ``<ws>/.home/`` (+ its ``*`` gitignore) here, for
    # the same reason and at the same seam as the tmpdir above: a confined
    # runner cannot be relied on to make it, and one call covers both the
    # cold-spawn and pool-served branches.  The plugins (cli /
    # interactive_shell / notebook) only redirect HOME at the env they build;
    # they do not create the directory.  Best-effort and audible inside the
    # helper, which no-ops for a falsy workspace / an off session -- so it is
    # called unconditionally (a guard here would only add a decision point to
    # a ratcheted function).
    workspace_home = ensure_workspace_home_dir(
        getattr(server, "_profile", None),
        workspace_path,
        managed_workspace_root,
    )
    # ----- The workspace's private /tmp, before either branch (#1381) -----
    # Same seam and reason: ``<ws>/.tmp`` must exist before the cold-spawn
    # child or the pool slot binds it over ``/tmp``.  No-op when the session
    # has no private /tmp.  A failure is logged; the runner then refuses to
    # start, naming the missing directory.
    ensure_private_tmp_dir(session_private_tmp(server, profile_name, confinement))

    # ----- Which uid the runner runs as, before either branch (#1168) -----
    # Resolved once, stashed on the server (the cold spawn and the envelope
    # both read it), and the directories above handed to that user before
    # anything forks.  ``None`` = the daemon's uid, the default policy.
    runner_user = _prepare_runner_user(
        server, peer=peer, session_id=session_id,
        workspace_path=workspace_path, profile_name=profile_name,
        log_path=log_path, workspace_home=workspace_home,
        confinement=confinement,
    )

    # ----- Pool routing (pool PR 4 + 5a) -----
    # Pool-served path is gated to sessions that:
    #   (a) Have a pool_manager wired by the daemon.
    #   (b) Have JAATO_RUNNER_POOL_ENABLED set.
    #   (c) Don't need a cgroup_attach.  Pool slots are forked from
    #       the template (already in daemon cgroup); migrating
    #       mid-life is PR 5b + subreaper-fix territory.
    # (PR 5a removed the disable_confine gate — apparmor sessions are
    # now eligible for pool routing because the slot self-confines
    # to envelope.profile_name in bootstrap_session step 1c.)
    spawned: Optional[SpawnedRunner] = None
    pool_served = False
    if _pool_may_serve(pool_manager, cgroup_attach):
        slot = _acquire_pool_slot(
            pool_manager, server, cascade_driver_id=cascade_driver_id,
            workspace_path=workspace_path, profile_name=profile_name,
            runner_user=runner_user, confinement=confinement,
            log_path=log_path,
        )
        if slot is not None:
            spawned = SpawnedRunner(
                pid=slot.pid,
                parent_socket=slot.sock,
                # Slot will transition to ``profile_name`` itself in
                # bootstrap_session step 1c.  Record the target on
                # the SpawnedRunner for audit / diagnostic surfaces.
                profile_name=profile_name,
                session_id=session_id,
            )
            spawned.pool_slot = slot  # Phase 2: keep ref for return path.
            pool_served = True
            # Phase 2 cascade-sharing teardown path: server.shutdown()
            # needs the pool_manager to return the slot after a
            # successful session_end RPC.  Stash it alongside the
            # SpawnedRunner so JaatoServer.shutdown() can find it.
            # Only set when the slot actually came from the pool; cold-
            # spawn paths leave it None.
            server._pool_manager_ref = pool_manager
            logger.info(
                "spawn_session_runner: session %s served by pool slot "
                "pid=%d cascade=%s (warm imports inherited; boundary %s)",
                session_id, slot.pid, slot.cascade_id or "(standalone)",
                _slot_boundary_note(profile_name, confinement),
            )
        else:
            logger.info(
                "spawn_session_runner: session %s — no slot fitted and the "
                "template could not fork one; falling back to cold-spawn",
                session_id,
            )

    # ----- Cold-spawn fallback (pre-PR-4 behavior) -----
    if spawned is None:
        spawned = _cold_spawn_runner(
            server,
            profile_name=profile_name,
            session_id=session_id,
            workspace_path=workspace_path,
            log_path=log_path,
            disable_confine=disable_confine,
            cgroup_attach=cgroup_attach,
            confinement=confinement,
        )

    # Phase 3 cascade-sharing hotfix (server 0.6.150+): reuse the
    # slot's existing RunnerRPCClient when it's a returned pool slot.
    # Creating a second RunnerRPCClient on the same socket fails —
    # the asyncio transport adopted by ``rpc.start`` binds the
    # socket exclusively.  Subsequent ``start`` calls on the same
    # socket fail; ``server._runner_rpc`` ends up None; sessions
    # crash with ``NoneType.session_send_message_threadsafe``.
    # See PR #173.
    #
    # #1058: ...and reuse it only if it is LIVE.  ``PoolManager`` now
    # refuses to hand out a slot whose client has died, so reaching this
    # branch with a corpse takes a race — ``acquire_slot`` ran off the
    # daemon loop, and the read loop that sets the flag runs on it — but
    # a race is exactly what the reported symptom looks like from here,
    # and verifying at the point of USE is what makes the reuse a
    # checked assumption rather than an inherited one.
    pool_slot = getattr(spawned, "pool_slot", None) if pool_served else None
    existing_rpc, slot_discarded = _reusable_slot_rpc(
        pool_slot, session_id, pool_manager, daemon_loop)
    if slot_discarded:
        # The slot was discarded with its dead client.  Its socket was
        # adopted by that client's asyncio transport, so a second
        # ``RunnerRPCClient`` on the same socket is the PR #173 failure
        # this whole reuse path exists to avoid — the session gets a
        # freshly spawned runner instead of an unknown-state transport.
        spawned = _cold_spawn_runner(
            server,
            profile_name=profile_name,
            session_id=session_id,
            workspace_path=workspace_path,
            log_path=log_path,
            disable_confine=disable_confine,
            cgroup_attach=cgroup_attach,
            confinement=confinement,
        )
        pool_slot = None
        pool_served = False
        server._pool_manager_ref = None
    if existing_rpc is not None:
        # Returned slot — reuse the rpc client.  Per-session state
        # was cleared by the slot-return path's
        # reset_for_slot_reuse call (see core.py shutdown cascade
        # branch).  Transport (reader/writer/read_task) stays
        # bound to the slot's socket.
        rpc = existing_rpc
        logger.info(
            "spawn_session_runner: session %s reused slot's rpc client "
            "(no new transport adopt needed)", session_id,
        )
    else:
        rpc = RunnerRPCClient(
            spawned.parent_socket,
            runner_pid=spawned.pid,
            loop=daemon_loop,
        )
        fut = asyncio.run_coroutine_threadsafe(rpc.start(), daemon_loop)
        fut.result(timeout=10.0)
        # First session on this slot — stash the rpc on the slot
        # so subsequent same-cascade sessions can reuse it.
        if pool_slot is not None:
            pool_slot.rpc = rpc
            logger.debug(
                "spawn_session_runner: session %s — new rpc client "
                "created + stashed on pool slot pid=%d",
                session_id, pool_slot.pid,
            )

    server.set_runner_rpc(rpc, spawned)
    logger.info(
        "runner spawned for session %s: pid=%d profile=%s log=%s "
        "confined=%s pool_served=%s runner_uid_policy=%s runs_as=%s",
        session_id, spawned.pid,
        _boundary_label(profile_name, confinement),
        log_path or "(inherited)",
        not disable_confine,
        pool_served,
        current_runner_uid_policy(),
        describe_runs_as(runner_user),
    )
    session_new_timing.mark(
        "runner_ready", session_id=session_id,
        detail=f"pid={spawned.pid} pool_served={pool_served}")


def _cold_spawn_runner(
    server: Any,
    *,
    profile_name: str,
    session_id: str,
    workspace_path: Optional[str],
    log_path: Optional[str],
    disable_confine: bool,
    cgroup_attach: Optional[Any],
    confinement: Any = None,
) -> Any:
    """Spawn a fresh session-mode runner subprocess.

    The pre-pool behaviour, and the fallback for every path the pool
    cannot serve: pool disabled, pool empty, a ``cgroup_attach`` the
    pool cannot honour, or — since #1058 — a pool slot whose RPC client
    turned out to be dead at the point of use.

    A free function with two call sites rather than a block inside
    :func:`spawn_session_runner`: the second call site is what lets the
    dead-slot path produce a working session instead of a refusal, and
    a copied forty-line block is a copy that rots.

    Args:
        server: The session's ``JaatoServer`` — read only for its
            resolved profile's ``runtime_limits``.
        profile_name: AppArmor profile to confine to (``""`` =
            unconfined).
        session_id: Session this runner will serve.
        workspace_path: The runner's cwd.
        log_path: Where the runner writes its own log, or ``None`` to
            inherit the daemon's stderr.
        disable_confine: Skip the AppArmor transition entirely.
        cgroup_attach: Optional ``preexec`` callback migrating the
            forked runner into a per-session cgroup.

    Returns:
        The ``SpawnedRunner`` handle.  It carries NO ``pool_slot``, which
        is what makes the teardown path (``JaatoServer.shutdown``) take
        the cold close rather than trying to return a slot to the pool.
    """
    from jaato_server.server.runner_spawner import RunnerSpawner

    spawner = RunnerSpawner()

    # Phase 5 §5.1b, re-scoped by #735: these two kwargs become the
    # ``JAATO_RUNNER_MAX_OUTPUT_CHARS`` /
    # ``JAATO_RUNNER_TOOL_TIMEOUT_SECONDS`` env vars, and those
    # configure ``server/runner/tool_executor.ToolExecutor`` — the
    # PHASE-2, cli-only ``execute_fn`` surface.  They are NOT this
    # session's enforcement path and never were: ``RunnerRPC``
    # routes to ``host.session._executor`` whenever a session host
    # exists, which is on every path that dispatches
    # ``session.bootstrap`` (all of them).  The measured consequence
    # was a ``tool_timeout_seconds: 2`` profile running a
    # ``sleep 60`` for 60.02 s on cold-spawn as well as on the pool.
    #
    # They are kept, not deleted, because the Phase-2 executor is a
    # LIVE fallback for a runner with no session host (cli-only
    # runners, harnesses, tests) and deleting them would silently
    # drop that surface back to its compile-time defaults — the same
    # class of regression, in the same direction.  Both values come
    # from the one source of truth, ``server._profile.runtime_limits``,
    # so the two surfaces cannot disagree about a number; they simply
    # bound different executors.  The session's own caps ride
    # ``SessionInitEnvelope.runtime_limits`` (v7), built by
    # ``build_session_envelope`` and applied in
    # ``JaatoSession.configure``.
    runtime_limits = _profile_runtime_limits(getattr(server, "_profile", None))
    max_output_chars = (
        runtime_limits.max_output_bytes
        if runtime_limits is not None else None
    )
    tool_timeout_seconds = (
        runtime_limits.tool_timeout_seconds
        if runtime_limits is not None else None
    )

    return spawner.spawn(
        profile_name=profile_name,
        session_id=session_id,
        workspace_path=workspace_path,
        log_path=log_path,
        max_output_chars=max_output_chars,
        tool_timeout_seconds=tool_timeout_seconds,
        disable_confine=disable_confine,
        cgroup_attach=cgroup_attach,
        # #1381: entered in the forked child before exec, i.e. before
        # ``runner/__main__`` confines itself.
        private_tmp_dir=session_private_tmp(server, profile_name, confinement),
        # #1168: dropped in the forked child after the private /tmp and
        # before exec -- so before ``runner/__main__`` confines itself.
        runner_user=stashed_runner_user(server),
        # SELinux: the child sets its exec context last, so the exec
        # enters the domain (selinux-backend.md §7.1).
        confinement=confinement,
    )


def _prepare_runner_user(
    server: Any,
    *,
    peer: Any,
    session_id: str,
    workspace_path: Optional[str],
    profile_name: Optional[str],
    log_path: Optional[str],
    workspace_home: Optional[str],
    confinement: Any = None,
) -> Optional[RunnerUser]:
    """Resolve this session's runner user, stash it, hand it its paths (#1168).

    The resolution is :func:`server.runner_user.resolve_runner_user`
    (``None`` under the default policy and every case the policy cannot
    honour).  Stashed as ``server._runner_user`` so the cold spawn and
    :func:`build_session_envelope` read the value decided here rather than
    deciding again.  Then the directories the daemon creates FOR the
    runner — session tmpdir, private ``/tmp``, workspace HOME,
    ``.jaato/logs`` + the runner log, ``.jaato/sessions/<id>`` — are handed
    to that user before anything forks, because a dropped runner could not
    write into them otherwise.

    Raises:
        RunnerUserRefused: see :func:`server.runner_user.refuse_if_unreachable`.
    """
    user = resolve_runner_user(
        peer=peer if isinstance(peer, PeerCredentials) else None,
        workspace_path=workspace_path,
    )
    server._runner_user = user
    dirs, files = runner_owned_paths(
        session_id=session_id,
        workspace_path=workspace_path,
        session_tmp=session_tmpdir(
            session_id, _confinement_id_of(profile_name, confinement)),
        private_tmp=session_private_tmp(server, profile_name, confinement),
        workspace_home=workspace_home,
        log_path=log_path,
    )
    prepare_runner_owned_paths(user, dirs, files)
    return user


def stashed_runner_user(server: Any) -> Optional[RunnerUser]:
    """The :class:`RunnerUser` :func:`_prepare_runner_user` stashed, or ``None``.

    ``isinstance`` for the reason :func:`stashed_private_tmp` gives: a test
    double's attribute is a mock, and only a real user may reach a drop.
    """
    value = getattr(server, "_runner_user", None)
    return value if isinstance(value, RunnerUser) else None


def describe_runs_as(user: Optional[RunnerUser]) -> str:
    """``name(uid=..,gid=..)`` or ``daemon uid N`` for the per-session log."""
    if user is not None:
        return user.describe()
    getuid = getattr(os, "getuid", None)
    return f"daemon uid {getuid() if getuid else '?'}"


def _revive_slot_rpc(rpc: Any, daemon_loop: Any) -> bool:
    """Reopen a pooled client whose read task was cancelled (#1058).

    Driven from the spawn path rather than from the pool because
    restarting a reader needs the event loop, and ``PoolManager`` has
    none — it runs on the replenish thread and on whichever thread is
    acquiring.  ``spawn_session_runner`` already holds ``daemon_loop``
    and already schedules :meth:`RunnerRPCClient.start` on it the same
    way.

    Best-effort and bounded: a revive that raises, times out, or is
    refused by the client's own evidence check answers ``False``, and
    the caller discards the slot and cold-spawns.  A slot is worth one
    short attempt, never a stalled session.

    Args:
        rpc: The slot's ``RunnerRPCClient``.
        daemon_loop: The loop the client's transport is bound to.

    Returns:
        ``True`` when the channel is usable again.
    """
    revive = getattr(rpc, "revive_read_loop", None)
    if not callable(revive) or daemon_loop is None:
        return False
    try:
        fut = asyncio.run_coroutine_threadsafe(revive(), daemon_loop)
        return bool(fut.result(timeout=5.0))
    except Exception as exc:  # noqa: BLE001 — a failed revive is a discard
        logger.warning(
            "spawn_session_runner: reviving the slot's rpc read loop "
            "raised %s — discarding the slot instead", exc,
        )
        return False


def _reusable_slot_rpc(
    pool_slot: Any,
    session_id: str,
    pool_manager: Any,
    daemon_loop: Any = None,
) -> Tuple[Optional[Any], bool]:
    """The slot's RPC client, but only when it is still alive (#1058).

    Layer 2 of the dead-slot fix.  ``PoolManager.acquire_slot`` is layer
    1 and is the one that makes the reported failure unreachable; this is
    the check at the point of USE, where the assumption is actually
    consumed.  It is not decorative: ``spawn_session_runner`` runs OFF
    the daemon loop (it blocks on ``run_coroutine_threadsafe``), so the
    read loop that sets ``is_closed`` can run between the acquire and
    here.

    A closed client gets ONE repair, and only on positive evidence.  If
    its ``close_reason`` says its read task was CANCELLED, then the
    runner was never signalled and the transport was never closed — what
    died is the reader, and a task cannot be un-cancelled, so a fresh
    read task is started on the surviving transport
    (``RunnerRPCClient.revive_read_loop``).  Clearing the flag alone
    would hand back a channel nobody is reading, which fails later and
    more confusingly than failing here.

    Every other cause DISCARDS the slot: the peer is gone (``eof``), the
    runner was reaped (``explicit-close``), or the stream is
    desynchronised mid-frame (``oversized-frame`` / ``malformed-frame``).
    Re-ADOPTING the socket is not among the options in any of those
    cases: it is already owned by the dead client's asyncio transport,
    and a second ``connect_accepted_socket`` on it is the exact PR #173
    failure the shared-client design exists to avoid.  The caller
    cold-spawns instead.

    Args:
        pool_slot: The acquired slot, or ``None`` for a cold-spawned
            session (in which case there is nothing to reuse).
        session_id: The arriving session — for the log line.
        pool_manager: The daemon's :class:`~server.runner_pool.PoolManager`,
            used to queue a discarded slot for teardown so its runner
            process is reaped rather than leaked.  ``None`` is tolerated:
            an unwired pool cannot reap, and refusing the reuse still
            beats handing back a closed client.
        daemon_loop: The loop the slot's transport is bound to, needed to
            start a new read task.  ``None`` disables the revive, which
            degrades to the discard path rather than to a wrong answer.

    Returns:
        ``(rpc, discarded)``.  ``rpc`` is the usable client or ``None``;
        ``discarded`` says the SLOT is gone and the caller must
        cold-spawn.  Two values because ``None`` alone conflates the
        routine case — the first session on a slot, which has no client
        yet and must create one on the slot's socket — with the failure
        case, where the socket belongs to a dead client and must not be
        touched.  Cold-spawning for the first case would bypass the pool
        entirely, silently, on every fresh slot.
    """
    rpc = getattr(pool_slot, "rpc", None) if pool_slot is not None else None
    if rpc is None:
        return None, False

    from jaato_server.server.runner_pool import slot_rpc_death

    if not getattr(rpc, "is_closed", False):
        return rpc, False

    # Closed, but possibly only because its READER was cancelled — the
    # transport and the runner survive that, and a task cannot be
    # un-cancelled, so clearing a flag would hand back a channel nobody
    # is reading.  A fresh read task on the surviving transport is the
    # repair; ``can_revive`` is the evidence that it is the right one.
    if getattr(rpc, "can_revive", False) and _revive_slot_rpc(
            rpc, daemon_loop):
        logger.info(
            "spawn_session_runner: session %s revived the read loop of "
            "pool slot pid=%s before reusing its rpc client",
            session_id, getattr(pool_slot, "pid", "?"),
        )
        return rpc, False

    reason = slot_rpc_death(pool_slot) or "closed"

    logger.error(
        "spawn_session_runner: session %s was handed pool slot pid=%s "
        "whose rpc client is closed (reason=%s) — discarding the slot "
        "and cold-spawning instead (#1058)",
        session_id, getattr(pool_slot, "pid", "?"), reason,
    )
    discard = getattr(pool_manager, "discard_acquired_slot", None)
    if callable(discard):
        discard(pool_slot, reason=f"dead-rpc:{reason}")
    return None, True


def _profile_runtime_limits(profile: Any) -> Optional[RuntimeLimits]:
    """A profile's ``runtime_limits``, but only if it really is one.

    ``build_session_envelope`` accepts whatever object the daemon stashed
    on ``server._profile``, which in practice is a ``SubagentProfile``
    and in tests is sometimes a ``MagicMock``.  ``_runtime_limits_to_dict``
    calls ``dataclasses.asdict``, which raises ``TypeError`` on anything
    else -- and an exception here aborts the whole envelope build, so a
    stand-in profile would take the session down rather than the one
    field it cannot describe.

    Type-checking rather than catching is deliberate: the attribute is
    DECLARED ``Optional[RuntimeLimits]``, so anything else is "nobody
    declared limits", which is exactly what the framework defaults mean.

    A free function because ``build_session_envelope`` sits on its
    cyclomatic-complexity baseline and may not grow.

    Args:
        profile: The resolved profile, a stand-in, or ``None``.

    Returns:
        The declared limits, or ``None``.
    """
    limits = getattr(profile, "runtime_limits", None)
    return limits if isinstance(limits, RuntimeLimits) else None


def _apply_cache_field(
    profile: Any,
    provider_name: Optional[str],
    plugin_configs_dict: Dict[str, Any],
) -> None:
    # ``provider_name`` falls back to the profile's own inside the helper,
    # not at the call site: ``build_session_envelope`` sits at the
    # complexity ceiling and one ``or`` there costs a decision point.
    """Fold a profile's common ``cache:`` field into ``plugin_configs``.

    Resolved HERE, into the provider's own knobs, rather than threaded
    down as a new envelope field: caching is delivered three different
    ways with three spellings and two layers, and ``plugin_configs`` is
    already the channel that carries all three to the runner.  A parallel
    channel would mean a schema bump and a fourth copy of the same
    routing.

    Laid down BENEATH the profile's explicit knobs -- ``setdefault`` for
    scalars, and for a sub-dict the cache values first with the explicit
    ones over the top -- so ``plugin_configs.<provider>`` still wins.  The
    escape hatch has to be able to escape; see the precedence note in §7.

    Mutates ``plugin_configs_dict`` in place and returns nothing, matching
    the quirks injection directly below its call site.
    """
    provider_name = provider_name or getattr(profile, "provider", None)
    profile_cache = getattr(profile, "cache", None)
    if profile_cache is None or not provider_name:
        return

    from jaato_server.shared.jaato_runtime import cache_field_to_provider_extra
    import importlib
    try:
        mod = importlib.import_module(
            f"jaato_server.shared.plugins.model_provider.{provider_name}")
        supports = bool(getattr(
            getattr(mod, "PROVIDER_CAPABILITIES", None),
            "prompt_caching", False))
    except Exception:  # noqa: BLE001
        # An unknown or unimportable provider cannot be shown to support
        # caching, and a capability lookup is not worth failing a spawn.
        supports = False

    cache_extra = cache_field_to_provider_extra(
        profile_cache, provider_name, supports_caching=supports)
    if not cache_extra:
        return

    existing = dict(plugin_configs_dict.get(provider_name) or {})
    for key, value in cache_extra.items():
        if isinstance(value, dict) and isinstance(existing.get(key), dict):
            merged = dict(value)           # cache field first,
            merged.update(existing[key])   # explicit knob over the top
            existing[key] = merged
        else:
            existing.setdefault(key, value)
    plugin_configs_dict[provider_name] = existing


def _apply_egress_env(
    session_id: str,
    agent_params_dict: dict,
    resolved_session_env: dict,
) -> None:
    """Merge the session's egress-proxy env into ``resolved_session_env``.

    Per-session egress allowlist (Phase 5 §5.11, opt-in).  If the profile's
    ``agent_params.egress_allowlist`` is set, start a per-session CONNECT proxy
    and point the runner's proxy env at it so all outbound HTTPS is confined to
    the allowlisted hosts.  No config (the common case) leaves
    ``resolved_session_env`` untouched.

    Fail-safe by default: a wire-up error is swallowed so it can never break
    session spawn.  The one exception is ``EgressEnforcementError``, raised
    only under ``JAATO_EGRESS_NFT_ENFORCE=strict`` — the operator asked for a
    kernel-enforced gate, so a session that would run without one is denied
    rather than started unconfined.
    """
    try:
        from jaato_server.server.egress_proxy import wireup as _egress_wireup
        _egress_env, _egress_errs = _egress_wireup.egress_env_for_session(
            session_id, agent_params_dict.get("egress_allowlist"))
        if _egress_env:
            resolved_session_env.update(_egress_env)
    except _EgressEnforcementError:
        logger.error("egress: denying session %s — hard enforcement was "
                     "required but could not be installed", session_id)
        raise
    except Exception:  # pragma: no cover - defensive: never block spawn
        logger.warning(
            "egress proxy wire-up failed for session %s (continuing without "
            "egress restriction)", session_id, exc_info=True)


def build_session_envelope(
    *,
    server: Any,  # JaatoServer (forward-typed; importing the real type
                  # creates a cycle through server/core.py).
    session_id: str,
    workspace_path: Optional[str],
    profile_name: str,
    managed_workspace_root: Optional[str] = None,
    confinement_required: bool = False,
    confinement: Any = None,
) -> "SessionInitEnvelope":
    """Build a :class:`SessionInitEnvelope` from a pre-init JaatoServer.

    Phase 3 §7c step 2: relocated from ``server/__main__.py`` so
    both IPC and WS callers share the implementation.  Reads the
    resolved profile from the server (set in
    ``SessionManager._create_session_impl`` before the pre-init
    hooks fire) and constructs the envelope the runner-side host
    needs.

    Fallback rules (no hardcoded defaults):
    - ``model_name``: profile.model → ``model_tiers[initial].model``
      → ``session_env["MODEL_NAME"]``.
      Empty if neither declares; runner-side ``_validate_envelope``
      raises ``BootstrapError(stage="validate")`` audibly.
    - ``provider_name``: profile.provider → ``model_tiers[initial].provider``
      → ``session_env["JAATO_PROVIDER"]``.
      Same empty-stays-empty rule.  The tier step is what lets a profile whose
      tiers fully declare model AND provider drop both top-level keys, which
      is what the profile loader's own both-declared warning tells authors is
      fine (jaato #822).

    Args:
        server: The :class:`JaatoServer` instance — has ``_profile``
            set to a :class:`SubagentProfile` if a profile was
            resolved.  ``None`` for inline-spec / no-profile sessions.
        session_id: Stable session identifier.
        workspace_path: Session's workspace; ``None`` for headless.
        profile_name: AppArmor profile name (informational; the
            envelope's ``profile_name`` field carries it for
            audit attribution).
        confinement_required: #1253 — whether this session was
            configured for AppArmor confinement, stamped onto
            ``SessionInitEnvelope.confinement_required`` so the runner
            gate can tell an empty profile that SHOULD have been
            confined from a legitimately-unconfined session.

    Returns:
        A :class:`SessionInitEnvelope` ready for
        :meth:`server.runner_rpc_client.RunnerRPCClient.bootstrap_session_threadsafe`.
    """
    from jaato_server.shared.session_envelope import SessionInitEnvelope

    profile = getattr(server, "_profile", None)
    provider_name = ""
    model_name = ""
    # ``None`` until a profile says otherwise: a profile-less session must
    # reach ``create_session(plugins=None)`` ("all exposed plugins", so the
    # eager wire keeps the core tools and the model can discover the rest).
    # Seeding this with ``[]`` made that case indistinguishable from a
    # profile declaring ``plugins: []`` (the minimal set), which since
    # a3df70a4 is passed to the runtime verbatim — a profile-less TUI
    # session then got introspection only, and the empty-wire gate in
    # ``jaato_session._should_drop_introspection`` dropped even that.
    plugin_specs: Optional[list] = None
    plugin_configs_dict: dict = {}
    preloaded: set = set()
    system_instructions: Optional[str] = None
    gc_dict: Optional[dict] = None
    env_overrides: dict = {}
    model_tiers_dict: Optional[Dict[str, Any]] = None
    budget_control_dict: Optional[Dict[str, Any]] = None
    max_parallel_tools: Optional[int] = None
    runtime_limits_dict: Optional[Dict[str, Any]] = None

    if profile is not None:
        # Same source the bootstrap gate uses (core._profile_binds_a_model):
        # flat ``model``, else ``model_tiers[initial].model``.  Reading
        # ``profile.model`` alone made the gate and the envelope disagree --
        # a tiers-only profile passed the gate, then the runner rejected the
        # envelope with "envelope.model_name is empty" and the caller saw a
        # dropped connection rather than a config error.
        #
        # ``provider`` follows the SAME rule, and did not until #822: it read
        # the flat key alone, so a profile whose tiers declared model AND
        # provider per tier -- and which therefore dropped both top-level
        # keys, exactly as the profile loader's own warning advises -- was
        # refused with "envelope.provider_name is empty".  Client-side that is
        # a 60-second create_session timeout, which reads as a hung daemon
        # rather than a config error.  The initial tier is what the session
        # uses on turn 1, so deriving from it is what makes that advice true.
        from jaato_server.shared.model_tiers import (bound_model_for_profile,
                                        bound_provider_for_profile)
        provider_name = bound_provider_for_profile(profile) or ""
        model_name = bound_model_for_profile(profile) or ""
        # A profile is present, so its answer is explicit -- even the
        # empty one.  Materialise here rather than at the first append:
        # ``plugins: []`` yields an empty ``names`` that never enters the
        # loop below, and leaving the spec list at ``None`` would report
        # that profile as "no profile" and expand it to every plugin.
        plugin_specs = []
        names = list(getattr(profile, "plugins", []) or [])
        preloaded = set(getattr(profile, "preloaded_plugins", set()) or set())
        # v3 (2026-05-14): forward profile.model_tiers to the runner so
        # it can resolve ``ModelTierConfig`` and register the
        # ``enter_tier`` lifecycle tool.  Pre-v3 this never reached the
        # runner — sessions silently ran in single-model mode regardless
        # of profile config.  Empty dict means "no tiers declared";
        # serialise as ``None`` to keep the envelope minimal.
        raw_tiers = getattr(profile, "model_tiers", None) or None
        if raw_tiers:
            model_tiers_dict = {str(k): v for k, v in raw_tiers.items()}
        # Envelope v5: same contract as model_tiers above — without this
        # the runner leaves ``JaatoSession._budget_tracker = None`` and the
        # session silently runs UNBUDGETED regardless of profile config.
        # The profile holds a parsed ``BudgetControlConfig``; the wire
        # carries its re-serialised dict (runner re-parses + revalidates).
        _budget = getattr(profile, "budget_control", None)
        # Envelope v6 (#862) / v7 (#735): the resolved ``runtime_limits``.
        # The width rides its own field for daemon/runner skew; the whole
        # block rides ``runtime_limits`` so the runner-side session can
        # ARM the subprocess plugins with ``tool_timeout_seconds`` /
        # ``max_output_bytes``.  Built HERE, outside the cold-spawn
        # branch in ``spawn_session_runner``, because that branch is the
        # one a pool slot -- forked before this session existed -- never
        # reaches, and pool-served is the default path.  The env pair the
        # cold-spawn branch still writes configures the Phase-2 cli-only
        # executor, which no bootstrapped session dispatches through; see
        # ``SessionInitEnvelope`` v7's note.
        from jaato_server.shared.plugins.subagent.config import _runtime_limits_to_dict
        _limits = _profile_runtime_limits(profile)
        max_parallel_tools = getattr(_limits, "max_parallel_tools", None)
        runtime_limits_dict = _runtime_limits_to_dict(_limits)
        # Cascade clamp (design note §3.1/§8b).  When this session belongs to
        # a cascade with a declared cap, its EFFECTIVE ceiling is
        # min(profile, cascade_remaining) per dimension — a child may only
        # ever be tightened, never widened, and never exceed what the cascade
        # still has.  Both inputs are logged, not just the outcome, so an
        # operator can SEE min() clamp rather than infer it from one number.
        _pool = getattr(server, "_cascade_budget_pool", None)
        if _pool is not None:
            _effective, _limits = _pool.child_config(_budget)
            logger.info(
                "cascade %s budget clamp for session %s -> %s",
                _pool.cascade_driver_id, session_id, _limits.describe(),
            )
            _budget = _effective
        if _budget is not None:
            budget_control_dict = _budget.to_dict()
        # Record the EFFECTIVE budget (post cascade-clamp) on the server so
        # ``_save_session`` can persist what this session ACTUALLY ran under.
        # The profile is the only vehicle that carries a budget to the runner,
        # so a budget declared anywhere else -- notably ``cascade_budget_set``
        # on the driver, where limits are a per-RUN operator choice rather
        # than a property of the agent -- has nothing to restore from once the
        # session unloads, and the revived session runs UNBUDGETED by
        # construction.  Persisting the resolved config here is what lets the
        # restore path re-attach it (session_manager: ``state.budget_control``
        # -> the rebuilt profile), the same way ``profile_spec`` lets an
        # inline session restore its own recipe.
        server._effective_budget_control = budget_control_dict
        # Phase 4 §C: carry the full profile.plugin_configs at the
        # envelope's top level (schema v2) so auto-loaded plugins not
        # named in profile.plugins (e.g. permission) receive their
        # profile overrides.  Closes backlog §3.3c.X.  Per-entry
        # ``config`` is no longer set — entries are {name, preload} only.
        #
        # Server 0.6.123+: values flow through ``expand_plugin_configs``
        # so ``${VAR}`` references AND secret URIs (``pass://`` /
        # ``vault://``) resolve daemon-side before the runner sees
        # them.  Pre-0.6.123 the envelope copied profile.plugin_configs
        # LITERALLY — any ``pass://`` in
        # ``plugin_configs.<provider>.api_key`` reached the runner
        # unresolved, and the AppArmor-confined runner can't exec
        # ``pass`` to resolve it itself (per
        # ``feedback_secret_resolution_stays_daemon_side`` memory).
        # Symmetric to the ``envelope.session_env`` resolution channel
        # (PR #91 → #92).  Same trust posture: resolved plaintext on
        # the daemon↔runner socketpair, never logged or forwarded.
        from jaato_server.shared.plugins.subagent.config import (
            expand_plugin_configs, inject_scrub_secret_env,
        )
        raw_plugin_configs = {
            k: dict(v)
            for k, v in (getattr(profile, "plugin_configs", {}) or {}).items()
        }
        plugin_configs_dict = expand_plugin_configs(
            raw_plugin_configs,
            workspace_root_override=getattr(server, "_workspace_path", None),
        )
        # Quirks injection (server 0.6.194+).  Top-level
        # ``profile.quirks`` is threaded into the provider's
        # plugin_configs namespace under the ``"quirks"`` key so the
        # runner-side provider plugin reads it from
        # ``ProviderConfig.extra["quirks"]`` at session init.  Lives
        # here at the envelope-build site (NOT at
        # ``core.py:_build_profile_session_kwargs`` — that's dead code
        # per PR #240's diagnosis comment in runner/session.py:~1015)
        # so the value actually reaches the runner.  Mirror in
        # ``shared/plugins/subagent/plugin.py`` covers the
        # daemon-spawned subagent path.  See
        # ``SubagentProfile.quirks`` docstring +
        # ``feedback_llama31_vllm_auto_mode_stringifies_args``.
        # Common ``cache:`` field (§7) -- resolved into the provider's own
        # knobs and laid BENEATH the profile's explicit ones.  Body lives
        # in a helper because this function is already 42 on the
        # complexity ratchet: growing it is what the guard exists to stop.
        _apply_cache_field(profile, provider_name, plugin_configs_dict)

        profile_quirks = getattr(profile, "quirks", None) or {}
        effective_provider_for_quirks = (
            provider_name or getattr(profile, "provider", None)
        )
        if profile_quirks and effective_provider_for_quirks:
            provider_cfg = dict(
                plugin_configs_dict.get(effective_provider_for_quirks) or {}
            )
            provider_cfg["quirks"] = dict(profile_quirks)
            plugin_configs_dict[effective_provider_for_quirks] = provider_cfg
        # Profile-level ``scrub_secret_env`` (#863) -> the cli /
        # interactive_shell / mcp sections, beneath their explicit knobs.
        # Same channel as cache + quirks above, for the same reason: the
        # plugin_configs dict is what already reaches the runner.
        inject_scrub_secret_env(profile, plugin_configs_dict)
        profile_tool_scopes = getattr(profile, "tool_scopes", {}) or {}
        for name in names:
            spec = {"name": name, "preload": name in preloaded}
            # Per-plugin tool allow-list (profile ``tools:[...]`` modifier)
            # rides alongside name/preload on the envelope entry so the
            # runner-side ``_build_session`` can scope this session's
            # wire surface.  Absent → all of the plugin's tools exposed.
            scope = profile_tool_scopes.get(name)
            if scope:
                spec["tools"] = list(scope)
            plugin_specs.append(spec)
        system_instructions = getattr(profile, "system_instructions", None)
        gc_obj = getattr(profile, "gc", None)
        if gc_obj is not None:
            # #1133: this read ``getattr(gc_obj, "config", None)`` — an
            # attribute ``GCProfileConfig`` does not have — so it always
            # resolved to ``{}`` and the envelope carried nothing but
            # ``{"type": ...}``.  Every declared knob (threshold,
            # target, pressure, preserve_recent_turns, the three #850
            # media keys) was dropped here, silently, before the
            # consumer ever saw it.  ``to_dict`` is the complete half.
            # No partial path, deliberately.  The previous code had one —
            # it kept ``type`` and whatever ``.config`` held — and a
            # partial carry is indistinguishable from a complete one at
            # the consumer, which is how a block reduced to its strategy
            # name travelled for as long as it did.  An object that
            # cannot serialize itself carries NOTHING, and the consumer
            # then falls back to ``gc.json`` rather than installing a
            # strategy at defaults the profile never asked for.
            to_dict = getattr(gc_obj, "to_dict", None)
            if callable(to_dict):
                gc_dict = to_dict()
        env_overrides = dict(getattr(profile, "env", {}) or {})

    # Profile-less fallback: read ``MODEL_NAME`` / ``JAATO_PROVIDER``
    # from the daemon's resolved session env (workspace ``.env`` +
    # profile.env + env_overrides, populated by
    # ``JaatoServer._resolve_session_env``).  Closes the profile-less
    # ``jaato --new-session`` regression introduced by §7c step 1
    # (commit 6406fe35, 2026-05-09) which made the runner-side
    # bootstrap always run + always validate ``envelope.model_name``.
    # Pre-§7c profile-less worked because the runner never validated
    # the envelope's model_name field (it was read from the daemon's
    # session env post-fork).
    #
    # No hardcoded default — if neither the profile NOR the session
    # env declares ``MODEL_NAME`` / ``JAATO_PROVIDER``, the field
    # stays empty and the runner-side ``_validate_envelope`` raises
    # ``BootstrapError(stage="validate")`` with the missing field
    # surfaced.  Loud failure beats a silent guess; the user's
    # standing rule against hardcoded fallbacks (global CLAUDE.md)
    # is honored here.  Pre-PR-100 (this PR) the provider had a
    # silent ``"anthropic"`` default which papered over the
    # configuration gap and could mask a Vertex / OpenRouter
    # / etc. session creation slipping through with wrong provider.
    session_env = getattr(server, "_session_env", {}) or {}
    if not model_name:
        model_name = session_env.get("MODEL_NAME", "") or ""
    if not provider_name:
        provider_name = session_env.get("JAATO_PROVIDER", "") or ""

    # Phase 3 post-Step-7 Path C: read PROJECT_ID + LOCATION daemon-
    # side so the envelope carries the provider-connect args the
    # runner-side ``bootstrap_session`` needs to call
    # ``runtime.connect(project, location)`` before
    # ``runtime.create_session`` (which guards on ``_connected``).
    # Non-Vertex providers tolerate empty strings; Vertex AI sessions
    # pick up real values from env.  Mirrors the daemon-side reads
    # at ``core.py:1550-1551``.
    try:
        import os as _os
        project_val = _os.environ.get("PROJECT_ID", "") or ""
        location_val = _os.environ.get("LOCATION", "") or ""
    except Exception:
        project_val = ""
        location_val = ""

    # Phase 4 §D: read agent_params from the per-session JaatoServer.
    # SessionManager._construct_and_initialize_server stashes the
    # originating create_session.agent_params there so this builder
    # can forward them on the wire envelope.  Pre-§D this was
    # hard-coded to ``{}`` and any runner-side prefetch script
    # reading ``context.agent_params`` saw an empty dict.
    agent_params_dict = dict(getattr(server, "_agent_params", {}) or {})

    # Read the daemon-side resolved agent identity + profile's
    # completion_payload_schema.  Both were previously hardcoded /
    # omitted:
    #
    # - ``agent_id`` was hardcoded ``"main"`` regardless of
    #   ``--agent <name>`` resolution.  Result: the runner-side
    #   ``JaatoSession._agent_id`` stayed at its ``__init__`` default
    #   of ``"main"`` for EVERY session, because the only attribute
    #   that updates ``_agent_id`` post-construction is
    #   ``set_ui_hooks`` — and the runner-side bootstrap installs the
    #   UI hooks shim via direct attribute write (rpc.py:3178-3185),
    #   bypassing ``set_ui_hooks``.  Downstream consequence:
    #   ``AgentCompletedEvent.agent_id`` always carried ``"main"``;
    #   reactor where-clauses keying on the logical agent identity
    #   (e.g. ``agent_id == "discovery"``) silently missed.
    #
    # - ``completion_payload_schema`` was missing from the
    #   constructor entirely.  Even though
    #   :class:`SessionInitEnvelope` declared the field,
    #   ``build_session_envelope`` never populated it — the runner-
    #   side ``JaatoSession._completion_payload_schema`` stayed
    #   ``None`` for profile-declared payload schemas, and
    #   ``LifecycleTools._execute_signal_completion`` fell back to
    #   the legacy ``summary`` string path instead of validating
    #   the typed payload.
    #
    # Both regressions surfaced 2026-05-12 by the kb-enablement-2.0
    # cascade smoke test.  See:
    # - ``docs/design/per_session_confined_runner_phase5_plan.md``
    # - The §7c step-2 relocation commit that copied the body
    #   verbatim from server/__main__.py without catching the gaps.
    profile_completion_schema = None
    # Profile-declared completion processors (server 0.6.125+).
    # Replaces the prior split between completion_artifacts +
    # completion_validators.  Both were broken in the runner path
    # pre-0.6.122 (envelope didn't ship them); collapsed into one
    # unified surface as of 0.6.125.  Serialise to wire-dict shape
    # the runner side reconstructs without importing
    # CompletionProcessor on the wire boundary.
    profile_completion_processors: List[Dict[str, Any]] = []
    if profile is not None:
        profile_completion_schema = getattr(
            profile, "completion_payload_schema", None,
        )
        # Serialised from the dataclass, never field-by-field: this list
        # named five of CompletionProcessor's eight fields, so `name`,
        # `max_refusals` and `on_exhausted` were dropped crossing into the
        # runner and a declared refusal ceiling had no effect on the session
        # that ran (jaato #770).
        from jaato_server.shared.plugins.subagent.config import (
            completion_processors_to_wire,
        )
        profile_completion_processors = completion_processors_to_wire(
            getattr(profile, "completion_processors", []) or []
        )

    # PR #91 Y fix: ship the FULLY-RESOLVED per-session env to the
    # runner.  ``server._session_env`` is populated by the daemon's
    # :meth:`JaatoServer._resolve_session_env` from workspace ``.env``
    # + profile.env + env_overrides, with ``${VAR}`` cross-references
    # expanded AND secret URIs (``pass://`` / ``vault://`` / etc.)
    # resolved via the daemon's SecretResolver entry points.  The
    # runner applies this dict to ``os.environ`` verbatim during
    # bootstrap — no resolver discovery, no ``pass`` exec (which
    # AppArmor correctly blocks).
    #
    # Trust posture: the runner-rpc socketpair (daemon ↔ runner) is
    # FD-pass only — not in the filesystem, not on the network.
    # Resolved secrets transit that channel in plaintext, same as
    # pre-PR-91 fork-inherit ``os.environ`` semantics.  The audit
    # behind PR #92 verified envelope.session_env is never logged,
    # persisted, or forwarded to clients.
    resolved_session_env = dict(getattr(server, "_session_env", {}) or {})

    _apply_egress_env(session_id, agent_params_dict, resolved_session_env)

    # Workspace HOME (#1225): fold the effective ``workspace_home`` into the
    # cli / interactive_shell / notebook sections so each plugin reads it
    # from its own namespace, beneath any explicit per-surface value.  Placed
    # here (not inside the ``profile is not None`` block above) so a
    # profile-less WS-provisioned session -- the bare ``.env`` shape every
    # web-created workspace has -- still gets the managed default.  The
    # daemon created the directory in ``spawn_session_runner``.
    inject_workspace_home(
        plugin_configs_dict, workspace_path, managed_workspace_root,
    )
    # Workspace venv (#1274): the same managed-default rule, so a profile-less
    # WS-provisioned session's pip / python / uv pip land in the workspace,
    # never in the host (or daemon) interpreter.
    inject_workspace_venv(
        plugin_configs_dict, workspace_path, managed_workspace_root,
    )

    return SessionInitEnvelope(
        session_id=session_id,
        workspace_path=workspace_path,
        profile_name=profile_name,
        provider_name=provider_name,
        model_name=model_name,
        plugins=plugin_specs,
        plugin_configs=plugin_configs_dict,
        system_instructions=system_instructions,
        agent_id=getattr(server, "_main_agent_id", "main"),
        gc=gc_dict,
        agent_params=agent_params_dict,
        config_root=getattr(server, "config_root", None),
        env_overrides=env_overrides,
        session_env=resolved_session_env,
        project=project_val,
        location=location_val,
        completion_payload_schema=profile_completion_schema,
        completion_processors=profile_completion_processors,
        model_tiers=model_tiers_dict,
        budget_control=budget_control_dict,
        max_parallel_tools=max_parallel_tools,
        runtime_limits=runtime_limits_dict,
        # Phase 2 cascade-sharing (envelope v4): forward the cascade
        # tenant ID stashed on the server by
        # ``SessionManager._construct_and_initialize_server``.  Runner
        # stashes onto JaatoSession so subagent create_session calls
        # auto-inherit via runtime.create_session().
        cascade_driver_id=getattr(server, "_cascade_driver_id", None),
        # 2026-06-06: ferry the two daemon-resolved system-instruction
        # knobs to the runner.  See SessionInitEnvelope field docstrings
        # for the bug history — both knobs were set correctly on
        # ``JaatoServer`` daemon-side but never reached the runner's
        # ``JaatoSession.configure`` over the wire, making them silent
        # no-ops.  ``getattr`` with the default keeps backward compat
        # with daemons that predate these attributes (the attributes
        # are set in JaatoServer.__init__ from BootstrapEnvelope so
        # they should always be present, but the defaults are also the
        # documented "no-op" values for both knobs).
        # Default frozenset() (suppress nothing); the envelope's __post_init__
        # also normalizes a legacy bool defensively.
        suppress_base_instructions=getattr(
            server, "_suppress_base_instructions", frozenset(),
        ),
        system_instruction_override=getattr(
            server, "_system_instruction_override", None,
        ),
        # #859: the authenticated creator, stashed on the per-session
        # server by ``_construct_and_initialize_server`` from
        # ``BootstrapEnvelope.created_by``, so the runner-side session
        # can stamp telemetry spans and ledger records with a user.
        created_by=getattr(server, "_client_user_id", None),
        # 2026-06-21: client-provided ("host") tools registered via the WS/IPC
        # protocol BEFORE session.new (e.g. a telegram client's send_to_telegram),
        # ferried so the RUNNER-tier model SEES them in list_tools.  Pre-fix they
        # registered only on the daemon registry and the runner model was blind
        # (#344-sibling daemon-vs-runner split).  Execution forwards back to the
        # daemon's proxy executor via daemon.plugin_execute (sentinel name).
        client_tools=list(
            getattr(server, "client_tool_schemas", {}).values()
        ),
        # #1253: carry the confinement invariant to the runner gate.  An
        # empty ``profile_name`` with this True is the silent-bypass case
        # ``_maybe_self_confine`` must refuse rather than run unconfined.
        confinement_required=confinement_required,
        # The names in ``session_env`` the daemon resolved from ``app://``;
        # the runner exempts exactly these from the cli / interactive_shell
        # scrub.  ``getattr`` because a test server may carry no accessor.
        granted_env_names=_granted_env_names_of(server),
        # #1348: the //child grants, so a refused command can name its cause.
        confinement_grants=_confinement_grants_of(profile_name),
        # #1381: the <ws>/.tmp the profile's /tmp grant was rendered for.
        private_tmp_dir=session_private_tmp(server, profile_name, confinement),
        # v8: which LSM provisioned the boundary.  Only AppArmor provisions
        # today, so the descriptor is derived from ``profile_name`` -- the
        # one value every spawn path already carries -- and cannot disagree
        # with it (the runner refuses a descriptor that does).
        confinement=_envelope_descriptor_of(profile_name, confinement),
        # #1508: the session's seccomp filter, compiled HERE (the daemon is
        # unconfined); the runner only installs the bytes.
        seccomp_program=seccomp_program_of(
            _envelope_descriptor_of(profile_name, confinement),
            _profile_runtime_limits(profile),
        ),
        # #1168: the user the runner drops to at step 1b3 (a pool slot) or
        # already dropped to before exec (a cold spawn), or ``None``.
        runner_user=_runner_user_wire(server),
        # #1465: the user tier the runner reads and is not granted.
        user_tier_files=user_tier_snapshot(stashed_runner_user(server)),
        # SELinux phase 4: where this session's runner logs, so a pool
        # slot stops writing the daemon's log (which it may not).
        runner_log_path=runner_log_path(workspace_path, session_id),
        # #1606: a plugin-tool call bootstraps a plugin host, not a session.
        plugin_host=bool(getattr(server, "_plugin_host", False)),
    )


def seccomp_program_of(
    descriptor: Optional[Dict[str, Any]], limits: Optional[RuntimeLimits],
) -> Optional[Dict[str, Any]]:
    """``SessionInitEnvelope.seccomp_program`` for one session (#1508).

    Compiled per session, because the program depends on the profile's
    ``runtime_limits.seccomp`` and ``seccomp_allow``.  Compiled HERE, in the
    unconfined daemon, because the confined runner may neither load
    libseccomp by search (``find_library`` execs ``ldconfig``) nor write
    the memfd libseccomp exports through (``tmpfs_t`` under SELinux).
    Both spawn paths, cold and pool slot, read the envelope at bootstrap,
    so both receive it.

    ``None`` when the session has no kernel boundary (the runner records
    ``unconfined``) or is a ``//`` sub-profile (an isolated sub-runner,
    recorded ``absent``), and for ``seccomp: off``.
    """
    from jaato_server.shared import seccomp_filter

    label = (descriptor or {}).get("label") or ""
    if not label or "//" in label:
        return None
    return seccomp_filter.compile_for_envelope(
        getattr(limits, "seccomp", None), getattr(limits, "seccomp_allow", None))


def _apparmor_descriptor(profile_name: str) -> Optional[Dict[str, str]]:
    from jaato_server.server.confinement.apparmor import envelope_descriptor

    return envelope_descriptor(profile_name)


def user_tier_snapshot(user: Optional[RunnerUser]) -> Optional[Dict[str, str]]:
    """The ``~/.jaato`` snapshot for a runner that runs as *user* (#1465).

    The runner reads the user tier of the home it runs with: the daemon's
    own, or, once #1168 dropped it to *user*, that user's passwd home.  A
    target with no home has no user tier, which is ``{}`` (looked, found
    nothing), never the daemon's.
    """
    from jaato_server.shared import user_tier

    if user is not None:
        if not user.home:
            return {}
        return user_tier.collect(os.path.join(user.home, ".jaato"))
    return user_tier.collect(os.path.expanduser("~/.jaato"))


def _runner_user_wire(server: Any) -> Optional[Dict[str, Any]]:
    """``server._runner_user`` in its envelope form, or ``None``."""
    user = stashed_runner_user(server)
    return user.to_dict() if user is not None else None


def _confinement_grants_of(profile_name: str) -> Optional[Dict[str, Any]]:
    """``apparmor.envelope_grants(profile_name)``, or ``None`` if it fails.

    A helper for the same reason as :func:`_granted_env_names_of`.  A
    missing record means no hint, never a refused bootstrap.
    """
    if not profile_name:
        return None
    try:
        from jaato_server.server.apparmor import envelope_grants
        return envelope_grants(profile_name)
    except Exception:  # noqa: BLE001 -- no grants means no hint, nothing worse
        logger.warning(
            "could not read the AppArmor grant record for %s", profile_name,
            exc_info=True,
        )
        return None


def _granted_env_names_of(server: Any) -> List[str]:
    """``server.granted_env_names()``, or ``[]`` for a server without it.

    A helper rather than an inline conditional because
    ``build_session_envelope`` sits on the complexity baseline.
    """
    accessor = getattr(server, "granted_env_names", None)
    if not callable(accessor):
        return []
    try:
        return list(accessor())
    except Exception:  # noqa: BLE001 -- granting nothing is the safe answer
        logger.warning(
            "could not read the app:// granted names; granting none",
            exc_info=True,
        )
        return []


def dispatch_bootstrap_envelope(
    *,
    server: Any,  # JaatoServer (forward-typed; see above).
    session_id: str,
    workspace_path: Optional[str],
    profile_name: str,
    timeout: float = 30.0,
    managed_workspace_root: Optional[str] = None,
    confinement_required: bool = False,
    confinement: Any = None,
) -> None:
    """Send the ``session.bootstrap`` RPC so the runner-side
    :class:`shared.jaato_session.JaatoSession` host is populated.

    Phase 3 §7c step 2: shared between the IPC + WS spawn paths.
    Pre-§7c-step-2 the dispatch lived only in
    ``server/__main__.py`` (IPC); WS sessions left the runner-side
    host unpopulated.

    Bootstrap failure does NOT propagate — the daemon-side
    :class:`JaatoSession` is still authoritative during the §7c
    rollout window (steps 3-7 progressively migrate authority away).
    Failures log at WARNING so operators notice the runner host
    isn't actually populated.

    Args:
        server: The session's :class:`JaatoServer` instance.  Must
            have ``runner_rpc`` set (i.e. :func:`spawn_session_runner`
            already ran).
        session_id: Session identifier.
        workspace_path: Session workspace; threaded into the
            envelope.
        profile_name: AppArmor profile name; threaded into the
            envelope for audit attribution.
        timeout: Wall-clock cap on the bootstrap RPC, seconds.
            Default 30s — generous to absorb runner-side plugin
            discovery + provider connect latency.
        managed_workspace_root: The WS provisioning root (#1225),
            forwarded to :func:`build_session_envelope` so the
            workspace-HOME default is folded into the plugin configs
            for daemon-managed workspaces.  ``None`` for IPC / user-CWD.
        confinement_required: #1253 — the session was configured for
            AppArmor confinement.  Carried onto the envelope so the
            runner-side ``_maybe_self_confine`` REFUSES to bootstrap
            when it is ``True`` and ``profile_name`` is empty, rather
            than serving work unconfined.  ``False`` (the default, and
            every non-confined session) leaves that gate inert.
    """
    rpc = server.runner_rpc
    if rpc is None:
        # Defensive: a caller invoking this without spawn_session_runner
        # having succeeded means the spawn helper raised.  Log + return;
        # no point dispatching to a None handle.
        logger.debug(
            "dispatch_bootstrap_envelope: server.runner_rpc is None for "
            "session %s — skipping bootstrap (spawn likely failed)",
            session_id,
        )
        # Server 0.6.169+ (bootstrap-time visibility): surface this
        # as a terminal event so cascade observers + reactor rules on
        # ``session.terminated where reason='error'`` can react.
        # Without this, spawn-helper failures dead-end the cascade
        # silently (driver hits IPC timeout instead of getting the
        # actionable error_type/error_summary).  See
        # ``_emit_bootstrap_terminated`` helper for the exception-safe
        # emit + the rationale memory
        # ``project_backlog_bootstrap_time_visibility_gap``.
        _note_bootstrap_outcome(
            server,
            "spawn_session_runner did not populate server.runner_rpc — "
            "no session.bootstrap was dispatched, so the runner hosts no "
            "session",
        )
        _emit_bootstrap_terminated(
            server=server,
            session_id=session_id,
            exc=RuntimeError(
                "spawn_session_runner did not populate "
                "server.runner_rpc — session never reached bootstrap "
                "phase.  Check earlier ERROR logs in this turn for "
                "the spawn-helper failure (apparmor compose, slot "
                "acquisition, runner-spawn fork, etc.)."
            ),
        )
        return

    try:
        envelope = build_session_envelope(
            server=server,
            session_id=session_id,
            workspace_path=workspace_path,
            profile_name=profile_name,
            managed_workspace_root=managed_workspace_root,
            confinement_required=confinement_required,
            confinement=confinement,
        )
        result = rpc.bootstrap_session_threadsafe(envelope, timeout=timeout)
        _note_bootstrap_outcome(server, None)
        _note_seccomp_posture(server, result, session_id)
        _note_capability_posture(server, result, session_id)
        session_new_timing.mark("bootstrap_acked", session_id=session_id)
        logger.info(
            "runner session.bootstrap acknowledged for %s: %s",
            session_id, result,
        )
    except Exception as exc:  # noqa: BLE001 — boundary surface
        # %s on `exc` collapses arg-less exceptions (e.g. ``TimeoutError()``)
        # to "", leaving the colon trailing into the suffix and the operator
        # blind to root cause.  Surface ``error_type=`` + ``error=`` (same
        # shape as MODEL_THREAD_TERMINAL_ERROR in core.py) so the class name
        # is always visible, plus ``exc_info=True`` for the traceback when
        # the logger config preserves it.
        # #1033: ERROR, not WARNING, and no longer claiming a fallback.
        # "daemon-side JaatoSession remains authoritative" described a §7c
        # rollout window that has closed — ``initialize()`` itself now reads
        # the runner's session (``session_get_context_usage_threadsafe``),
        # and every ``session.*`` verb on a hostless runner answers
        # ``no_host``.  This line is the ONE place the real cause is
        # recorded, so it is logged at the severity of what it decides.
        logger.error(
            "runner session.bootstrap FAILED for %s: error_type=%s error=%s — "
            "this runner hosts no session; the session will be refused",
            session_id, type(exc).__name__, exc, exc_info=True,
        )
        _note_bootstrap_outcome(
            server, f"{type(exc).__name__}: {exc_message(exc)}",
        )
        # Server 0.6.169+ (bootstrap-time visibility): emit
        # SessionTerminatedEvent so cascade observers + reactor rules
        # see the failure.  Covers ANY bootstrap-time failure class
        # (SecretResolutionError, apparmor compose, runner RPC timeout,
        # plugin discovery errors) — generic ``except Exception``
        # mechanically catches everything that escapes the bootstrap
        # RPC.  Empirical motivation (peer 7:1, 2026-05-31): gpg-agent
        # passphrase expiry → SecretResolutionError → bootstrap WARNING
        # logged but cascade.py hung 3min on IPC timeout because no
        # terminal event surfaced.
        _emit_bootstrap_terminated(
            server=server, session_id=session_id, exc=exc,
        )
    finally:
        # Mark the runner ready REGARDLESS of bootstrap outcome: on success it
        # can service mid-session client-tool pushes + sends; on the
        # daemon-authoritative failure path the daemon-side JaatoSession still
        # handles the turn — either way, don't strand the push / send-gate on a
        # 30s readiness timeout.  This bootstrap-settled point is what the gates
        # now wait for, instead of racing the reused warm pool slot's
        # live-but-not-ready rpc handle (the re-attach client-tool-push stall).
        server.mark_runner_ready()
        # Runner is bootstrap-settled — re-emit the tool-id registry OFF the
        # event loop so the runner-tier tool names (prompt.* etc.) reach the
        # client.  Every ON-loop emit caller (emit_current_state / initialize /
        # _register_client_tools) now skips runner-tier to avoid the
        # daemon-side prompt_library filesystem walk on the loop (the re-attach
        # self-block); those names come ONLY from the runner
        # (session_get_tool_schemas), which can only run off-loop — here.  On a
        # bootstrap failure the runner RPC yields [] and this maps daemon-tier
        # only (harmless).
        try:
            server._emit_tool_id_registry_from_schemas()
        except Exception:  # noqa: BLE001 — re-emit must not strand bootstrap
            logger.debug(
                "post-bootstrap tool-id re-emit failed for %s",
                session_id, exc_info=True,
            )


def _note_seccomp_posture(
    server: Any, result: Any, session_id: str = "",
) -> None:
    """Record the runner's reported seccomp posture on *server* (#1503).

    And say it in the DAEMON log at the level it deserves (#1510): the
    runner logs its own WARNING, but in the runner's log, which is not the
    one an operator reads first.

    ======================================  =========  ====================
    posture                                 level      says
    ======================================  =========  ====================
    ``filter``, ``unconfined``              INFO       the posture
    ``off``                                 WARNING    no syscall filter
    ``absent``, best effort                 WARNING    no filter, and why
    ``absent`` with ``spawns_refused``      ERROR      every model-driven
                                                       subprocess spawn in
                                                       this session will be
                                                       refused, and why
    any, with ``ignored_families``          WARNING    the unknown
                                                       ``seccomp_allow``
                                                       names, which allowed
                                                       nothing back
    ======================================  =========  ====================

    Each names the session.  Best-effort, like
    :func:`_note_bootstrap_outcome`: a recorder must not turn an
    acknowledged bootstrap into a failed one.
    """
    note = getattr(server, "note_seccomp_posture", None)
    if not callable(note):
        return
    try:
        posture = result.get("seccomp") if isinstance(result, dict) else None
        note(posture)
        if isinstance(posture, dict):
            _log_seccomp_posture(session_id, posture)
    except Exception:  # noqa: BLE001 — a recorder must not fail the path
        logger.debug("note_seccomp_posture raised", exc_info=True)


def _log_seccomp_posture(session_id: str, posture: Dict[str, Any]) -> None:
    """The daemon-log line for one session's seccomp posture (#1510)."""
    from jaato_server.shared import seccomp_filter as sf

    kind = posture.get("posture")
    reason = posture.get("reason") or "no reason given"
    ignored = posture.get("ignored_families")
    if ignored:
        logger.warning(
            "seccomp: session %s: runtime_limits.seccomp_allow names unknown "
            "famil%s %s (known: %s); ignored, the filter is unchanged and "
            "nothing was allowed back for them",
            session_id, "y" if len(ignored) == 1 else "ies", list(ignored),
            ", ".join(sf.FAMILIES))
    if kind == sf.POSTURE_ABSENT and posture.get("spawns_refused"):
        logger.error(
            "seccomp: session %s: the filter is REQUIRED and unavailable "
            "(%s) -- every model-driven subprocess spawn in this session "
            "(cli, interactive_shell, notebook) will be refused",
            session_id, reason)
    elif kind == sf.POSTURE_ABSENT:
        logger.warning(
            "seccomp: session %s: posture absent (%s) -- model-driven "
            "subprocesses run behind the LSM boundary with no syscall filter",
            session_id, reason)
    elif kind == sf.POSTURE_OFF:
        logger.warning(
            "seccomp: session %s: posture off (%s) -- model-driven "
            "subprocesses reach the whole syscall table behind the LSM "
            "boundary alone", session_id, reason)
    else:
        logger.info("seccomp: session %s: runner posture %s",
                    session_id, posture)


def _note_capability_posture(
    server: Any, result: Any, session_id: str = "",
) -> None:
    """Record the runner's capability-drop posture on *server* (#1543).

    The sibling of :func:`_note_seccomp_posture`, and says it in the DAEMON
    log for the same reason: ``dropped`` / ``unconfined`` at INFO,
    ``partial`` / ``inherit`` / ``absent`` at WARNING with the reason, so
    "LSM yes, capabilities kept" is never silent.  Best-effort.
    """
    note = getattr(server, "note_capability_posture", None)
    if not callable(note):
        return
    try:
        posture = result.get("capabilities") if isinstance(result, dict) else None
        note(posture)
        if not isinstance(posture, dict):
            return
        kind = posture.get("posture")
        if kind in ("dropped", "unconfined"):
            logger.info("capabilities: session %s: runner posture %s",
                        session_id, posture)
        else:
            logger.warning(
                "capabilities: session %s: posture %s (%s) -- model-driven "
                "subprocesses are not fully stripped of capabilities",
                session_id, kind, posture.get("reason") or "no reason given")
    except Exception:  # noqa: BLE001 — a recorder must not fail the path
        logger.debug("note_capability_posture raised", exc_info=True)


def _note_bootstrap_outcome(server: Any, error: Optional[str]) -> None:
    """Record the bootstrap outcome on *server* (#1033).

    Best-effort by construction: this function is called from the
    dispatch's own failure paths, and a diagnostic that raises there would
    replace a reportable failure with an unreportable one.  Test doubles
    for ``JaatoServer`` that predate the method simply do not record —
    which reads downstream as "nothing known to be wrong", the same answer
    a session with no runner gives.

    Args:
        server: the session's ``JaatoServer``.
        error: one-line summary of the failure, or ``None`` on success.
    """
    note = getattr(server, "note_runner_bootstrap_outcome", None)
    if not callable(note):
        return
    try:
        note(error)
    except Exception:  # noqa: BLE001 — a recorder must not fail the path
        logger.debug("note_runner_bootstrap_outcome raised", exc_info=True)


def _emit_bootstrap_terminated(
    *, server: Any, session_id: str, exc: BaseException,
) -> None:
    """Emit ``SessionTerminatedEvent(reason="error")`` for a
    bootstrap-time failure.

    Server 0.6.169+ helper used by :func:`dispatch_bootstrap_envelope`
    at both failure paths (spawn-didn't-populate-rpc + bootstrap-rpc-
    raised).  By the time ``dispatch_bootstrap_envelope`` is invoked,
    ``server.set_event_callback`` has already been wired by
    ``session_manager.create_session`` (line ~4217), so emitted events
    flow through ``_emit_to_session`` → ``_dispatch_to_cascade_clients``
    and reach cascade observers + reactor rules.

    Routes through the single error-termination chokepoint
    ``JaatoServer._emit_error_termination_from_exc`` — which emits
    ``AgentErrorEvent`` (recovery first refusal; bootstrap has no auto-retry to
    wait on, the framework is out of moves immediately) THEN
    ``SessionTerminatedEvent(reason="error")`` (carrying ``error_summary`` /
    ``error_type``) — so the "AgentErrorEvent precedes every reason=error"
    invariant is structural here too.  ``agent_id`` falls back to ``"main"`` when
    ``server._main_agent_id`` isn't set (early bootstrap fail).

    The call is wrapped in a defensive try/except: a failure of the visibility
    path must not mask the underlying bootstrap failure or disrupt the
    session-creation caller's error handling.  Logs the failure to keep the
    audit trail intact.
    """
    try:
        agent_id = getattr(server, "_main_agent_id", None) or "main"
        server._emit_error_termination_from_exc(
            exc, session_id=session_id, agent_id=agent_id,
        )
    except Exception as emit_exc:  # noqa: BLE001 — defensive
        logger.warning(
            "_emit_bootstrap_terminated: SessionTerminatedEvent emit "
            "failed for session %s (root cause was %s: %s); cascade "
            "observers will not see the bootstrap failure for this "
            "session — investigate emit chain.  Emit error: %s: %s",
            session_id, type(exc).__name__, exc,
            type(emit_exc).__name__, emit_exc,
            exc_info=True,
        )
