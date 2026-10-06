"""Pool slot manager for pre-warm runner pool (pool PR 3).

Owns the collection of pre-forked pool slots between the template
subprocess (PR 2) and the session-routing layer (PR 4).

Each pool slot is:
  - A child process forked from the template (inherits warm imports)
  - Identified by its PID
  - Reachable via a per-slot socket on the daemon side

PR 3 ships only the pool's idle-management:
  - ``spawn_initial_slots(n)`` — ask the template to fork N slots
    at daemon startup
  - ``shutdown_all()`` — politely SHUTDOWN every idle slot at daemon
    exit
  - Idle-slot tracking dict

PR 4 will add:
  - ``acquire_slot()`` — pop an idle slot off the pool, return it
    to the session-routing code to send a bootstrap envelope
  - Replenishment thread that asks the template for a new slot
    whenever the pool drops below the target size

This module is **daemon-tier** infrastructure, like
``runner_template.py``.  Not a plugin in the discovery sense.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


# Cascade-sharing idle timeout (Phase 2).  An IDLE_FOR_CASCADE slot
# whose last_session_end_ts is older than this gets torn down by the
# replenish loop.  Default matches design doc §3 decision 5 (300s).
DEFAULT_CASCADE_IDLE_TIMEOUT_SECONDS = 300.0


@dataclass
class PoolSlot:
    """Pre-warm pool slot — daemon's handle on a template-forked child.

    Replaces the Phase 1 ``Tuple[int, socket.socket]`` shape.  Carries
    cascade-affinity bookkeeping for Phase 2 slot reuse.

    Attributes:
        pid: Slot process id.  Template-child; reaped via waitpid by
            the daemon's subreaper bit (or by shutdown_all on exit).
        sock: Daemon-side end of the slot's RPC socket.  Closing this
            socket signals the slot to exit (peer-EOF).
        config_root: The ``.jaato`` config root the slot's WARM PLUGIN STATE
            was built from.  Reuse requires a match.

            ``cascade_driver_id`` alone was the whole reuse key, so two
            sessions differing in config root -- different profiles, agents,
            prompt library, permission config -- shared a slot as long as the
            cascade id matched.  The path variables self-heal (the runner
            re-runs ``set_config_root`` every bootstrap); what does not is
            everything DERIVED from that root during the first session's
            bootstrap, which the pool exists to keep warm.  A slot's identity
            has to include what the slot actually carries.

            ``None`` for a slot that has never served.
        workspace_root: The workspace the slot's last session ran in
            (#1033).  Part of the reuse key for two independent reasons.

            The load-bearing one is CONFINEMENT.  A slot self-confines to
            the session's AppArmor profile, and the profile's whole point
            is that it grants one workspace — so a slot that has served a
            session in workspace A is wearing a boundary that does not
            fit workspace B.  Since ``aa_change_profile`` is per-task and
            existing threads cannot be re-confined (#1023), the slot
            cannot be re-fitted; it can only be refused.

            The second is warmth: #890 already declines to adopt
            ``TRAIT_SLOT_SCOPED`` plugin instances on a workspace
            mismatch, so a workspace-crossing reuse was already half
            wasted.  It simply was not UNSAFE until per-thread
            verification made the straddle visible.

            ``None`` for a slot that has never served.
        profile_name: The AppArmor profile this slot's threads wear, as
            of its last session (``""``/``None`` = unconfined).  The one
            property of a slot that the next session cannot change, so it
            is in the key and the key is what the profile is NAMED after
            (``server.confinement_id``) — which makes "reused slot" and
            "same profile" the same statement, and leaves nothing for a
            transition to do.

            This is also what gates a slot that carries no cascade.  A
            standalone session returns its slot to the pool as PURE IDLE,
            where any later session may take it — profile and all.  The
            cascade-affinity key never covered that path.
        cascade_id: Optional cascade-driver ID this slot is currently
            affined to.  ``None`` for a fresh slot or a slot that has
            never served a session (PURE IDLE).  Set when a session
            with ``cascade_driver_id != None`` is acquired against
            this slot; cleared only on slot teardown (slots never
            switch cascades mid-life).
        last_session_end_ts: Wall-clock monotonic timestamp of the
            most recent ``session_end`` RPC for this slot.  ``None``
            until the slot has served (+ returned from) at least one
            session.  Used by the idle-teardown sweep to detect
            cascades that have gone idle longer than
            ``cascade_idle_timeout_seconds``.
    """

    pid: int
    sock: socket.socket
    cascade_id: Optional[str] = None
    config_root: Optional[str] = None
    workspace_root: Optional[str] = None
    profile_name: Optional[str] = None
    #: The uid this slot's process runs as once it has served (#1168):
    #: ``None`` = the daemon's own uid (it never dropped).  A slot that
    #: dropped cannot get root back — ``drop_to`` proves ``setuid(0)``
    #: fails — so this is a property the next session cannot change and is
    #: in the key for that reason.  Stamped by :meth:`SlotKey.stamp`.
    runner_uid: Optional[int] = None
    #: The SELinux boundary (its ``confinement_id``) this slot was forked
    #: into (phase 4, selinux-backend.md §7.2), ``None`` for a slot that
    #: was not.  A slot enters an SELinux domain only at fork, while it
    #: has one thread, and never leaves it: the runner domain holds no
    #: ``dyntransition``.  So a slot carrying a boundary fits only that
    #: boundary, and a slot carrying none never fits an SELinux session.
    #: Kept out of ``profile_name``, whose last wearer's death unloads an
    #: AppArmor profile of that name (:meth:`PoolManager._reap_slot_profile`).
    selinux_boundary: Optional[str] = None
    last_session_end_ts: Optional[float] = None
    # Phase 3 cascade-sharing (server 0.6.146+): identifier of the
    # most recent session this slot served.  Set when the slot
    # returns to the pool after a successful ``session_end`` RPC
    # (JaatoServer.shutdown sets it via the
    # ``return_slot_after_session`` call site).  The NEXT session
    # to acquire this slot uses this value to
    # ``apparmor_parser --remove`` the prior session's profile
    # AFTER its own ``aa_change_profile`` transition succeeds.
    # ``None`` for slots that have never served a session.
    last_session_id: Optional[str] = None
    # Phase 3 cascade-sharing hotfix (server 0.6.150+): the
    # ``RunnerRPCClient`` adopted onto this slot's socket.  Created
    # ONCE per slot lifetime by the first session's spawn helper;
    # reset via ``reset_for_slot_reuse`` between cascade sessions;
    # closed at slot teardown (cascade-idle sweep OR daemon shutdown).
    #
    # Why the rpc lives on the slot, not on JaatoServer: the asyncio
    # transport that ``RunnerRPCClient.start`` adopts via
    # ``loop.connect_accepted_socket`` binds the socket exclusively.
    # Creating a second ``RunnerRPCClient`` on the same socket fails;
    # session 2 spawn falls back to in-process and ``server._runner_rpc``
    # is None — observed v152-retry-4 as
    # ``NoneType.session_send_message_threadsafe``.  See PR #173.
    #
    # Typed ``Any`` to avoid the cycle through ``runner_rpc_client``.
    rpc: Optional[Any] = None
    #: Why this slot was queued for teardown, set when it is put on
    #: ``PoolManager._pending_teardown`` and read by the drain so the
    #: teardown log and the AppArmor-profile reap name the real cause
    #: (#1058).  The drain used to label EVERY queued slot
    #: ``"over-capacity"``, which was true of its only producer and
    #: became a lie the moment a second one existed.  ``None`` for a
    #: slot that was never queued.
    teardown_reason: Optional[str] = None
    #: Has this slot ever been handed to a session? (#1100)
    #:
    #: Set by :meth:`SlotKey.stamp`, i.e. the moment the slot is claimed
    #: on the unaffined acquire path -- before the bootstrap envelope is
    #: sent, because what matters is that the process starts running a
    #: session's work, not that the work succeeded.  Never cleared: a
    #: slot cannot un-run a session, and the threads that session left
    #: behind cannot be re-confined (``aa_change_profile`` is per-task).
    #:
    #: Distinct from ``profile_name``, and that distinction IS the fix.
    #: ``SlotKey.build`` folds ``""`` to ``None`` so unconfined has one
    #: spelling -- right for a KEY, and it made a slot that SERVED an
    #: unconfined session indistinguishable from one that has served
    #: nothing, while it carries that session's ``unconfined`` threads.
    #: "Never confined" and "has no threads to worry about" are not the
    #: same statement.
    has_served: bool = False


@dataclass(frozen=True)
class SlotKey:
    """What a slot IS, and therefore who may be handed it (#1033).

    The rule that generates this tuple: **the key must contain every
    property of the slot that the next session cannot change.**  Applied
    field by field:

    ======================================  ==========================
    property the slot carries               in the key
    ======================================  ==========================
    warm plugin state from the config tree  ``config_root``
    warm plugin instances, tenant isolation ``cascade_driver_id``
    the AppArmor profile its threads wear   ``workspace_root`` +
                                            ``profile_name``
    the mount namespace (#1381 private      ``profile_name`` — the
    ``/tmp``: ``<ws>/.tmp`` over ``/tmp``)  body names the bound
                                            directory, so it is part of
                                            the confinement id
    the cgroup (``runtime_limits`` trio)    not needed — a session
                                            needing one is routed away
                                            from the pool entirely
                                            (``spawn_session_runner``
                                            consults the pool only when
                                            ``cgroup_attach is None``)
    session ``env:`` / secrets              not needed — overlaid per
                                            turn, never baked in
    the uid the process runs as (#1168)     ``runner_uid`` — a dropped
                                            slot can never become root
                                            or anyone else again
    the SELinux domain and level (phase 4)  ``selinux_boundary`` — the
                                            domain is entered at fork
                                            and never left
    ======================================  ==========================

    ``runner_uid`` (#1168 step 3): ``None`` is the daemon's own uid.  A
    virgin slot is still root, so it fits any uid — it drops at
    ``session.bootstrap`` step 1b3; a slot that has served fits only a
    session wanting its uid, in BOTH acquire paths (path (1) compares the
    whole key; path (2) goes through :meth:`accepts_unaffined`).  Cost,
    stated: on a multi-user daemon a slot one user's session dropped is
    never handed to another's, so that arrival cold-spawns — counted in
    ``pool_uid_mismatch_skips_total``.

    The third row is the one that was missing, and it cost a P0: the key
    said "reusable" while the profile name (``jaato-ws-{session_id}``)
    said "new boundary", and once #1023 made confinement per-task both
    could not be true.  Every reused slot straddled two profiles and its
    bootstrap was correctly refused.

    ``workspace_root`` and ``profile_name`` are both present rather than
    one standing in for the other: ``profile_name`` is ``""`` on a host
    with no AppArmor, where the workspace is still the thing #890 cares
    about; and ``workspace_root`` alone does not distinguish two
    boundaries over one workspace.

    The mount-namespace row (#1381) needs no field of its own, and that is
    checked rather than assumed (``test_private_tmp_1381``): only a
    CONFINED runner enters a private ``/tmp``, a confined slot is handed
    only to a session wanting the ``profile_name`` it wears, and that name
    is derived from a body that names the bound directory.  A confined slot
    can no longer mount (every body denies ``mount`` and ``sys_admin``), so
    a key that let it reach a different boundary would be a refused
    bootstrap, never a shared ``/tmp``.  An unconfined slot never enters a
    namespace, so ``profile_name=None`` keeps meaning "no namespace".

    ``PoolSlot.has_served`` (#1100) is deliberately NOT a fifth field.  A
    key says what an ARRIVING SESSION wants; ``has_served`` says what a
    SLOT has done, and a session cannot ask for a slot that has served —
    only :meth:`accepts_unaffined` reads it, which is the one comparison
    that is about the slot's history rather than the session's request.
    Putting it in the tuple would also break path (1), where the whole
    key is compared for equality and a served slot must still match a
    session that wants exactly its boundary.
    """

    cascade_driver_id: Optional[str] = None
    config_root: Optional[str] = None
    workspace_root: Optional[str] = None
    profile_name: Optional[str] = None
    runner_uid: Optional[int] = None
    selinux_boundary: Optional[str] = None

    @classmethod
    def build(
        cls,
        cascade_driver_id: Optional[str] = None,
        config_root: Optional[str] = None,
        workspace_root: Optional[str] = None,
        profile_name: Optional[str] = None,
        runner_uid: Optional[int] = None,
        selinux_boundary: Optional[str] = None,
    ) -> "SlotKey":
        """Normalize the caller's values into a comparable key.

        Paths are canonicalised (``realpath``) so two spellings of one
        directory do not read as two tenants, and ``""`` is folded to
        ``None`` so "unconfined" has one spelling rather than two.
        """
        return cls(
            cascade_driver_id=cascade_driver_id or None,
            config_root=_canonical_path(config_root),
            workspace_root=_canonical_path(workspace_root),
            profile_name=profile_name or None,
            runner_uid=runner_uid,
            selinux_boundary=selinux_boundary or None,
        )

    @classmethod
    def of_slot(cls, slot: "PoolSlot") -> "SlotKey":
        """The key a slot currently carries."""
        return cls(
            cascade_driver_id=slot.cascade_id or None,
            config_root=_canonical_path(slot.config_root),
            workspace_root=_canonical_path(slot.workspace_root),
            profile_name=slot.profile_name or None,
            runner_uid=getattr(slot, "runner_uid", None),
            selinux_boundary=getattr(slot, "selinux_boundary", None) or None,
        )

    def stamp(self, slot: "PoolSlot") -> None:
        """Record this key on *slot* — the four fields move together.

        Stamped as a unit because they ARE the slot's identity: a slot
        affined to a cascade without recording the boundary it was
        confined to is a slot whose reuse check cannot be right.

        ``has_served`` is raised here too (#1100), and never lowered.
        This is the one place a slot stops being virgin, so it is the one
        place that has to say so; DERIVING it from the four fields is
        exactly what did not work, since all four are legitimately
        ``None`` for an unconfined standalone session.
        """
        slot.cascade_id = self.cascade_driver_id
        slot.config_root = self.config_root
        slot.workspace_root = self.workspace_root
        slot.profile_name = self.profile_name
        slot.runner_uid = self.runner_uid
        slot.selinux_boundary = self.selinux_boundary
        slot.has_served = True

    def accepts_unaffined(self, slot: "PoolSlot") -> bool:
        """May this key take *slot*, which carries no cascade affinity?

        Warm state is not the question here — a PURE IDLE slot's warm
        state is either absent (never served) or already forfeit.  The
        question is the KERNEL BOUNDARY its threads are stuck inside.

        So: a slot that has served **nothing** fits anyone, and a slot
        that has served **anything** fits only a session wanting the
        boundary it already has.

        The first clause used to read ``not slot.profile_name`` — "was
        never confined" — and that is a different statement (#1100).  An
        unconfined session stamps ``profile_name=None`` (``SlotKey.build``
        folds ``""`` to ``None`` so unconfined has one spelling, which is
        right for a key), so its slot became indistinguishable from a
        virgin one **while carrying that session's ``unconfined``
        threads**.  Handed next to a confined session, the main thread
        transitions, the two RPC lanes are recycled, and #1023's
        per-thread verification correctly refuses the bootstrap for the
        leftovers — which cannot be confined, only retired, because
        ``aa_change_profile`` is per-task.

        Confirmed on a live daemon: slot pid 95942 served an unconfined
        session at 12:06, returned to the pool at 12:11, was handed to a
        confined session at 12:17, and was refused with 6 of 7 threads
        reporting ``unconfined``.

        This is #1033's own generating rule — *the key must contain every
        property of the slot that the next session cannot change* —
        applied to the property #1033 missed.  ``has_served`` is that
        property; it cannot be derived from the four key fields, because
        every one of them is legitimately ``None`` for an unconfined
        standalone session.

        **Cost**, stated: on a daemon that mixes unconfined and confined
        sessions, an unconfined session's slot is no longer offered to a
        confined one, so that arrival cold-spawns (~7s) instead.  Already
        visible in ``pool_profile_mismatch_skips_total`` beside
        ``pool_acquire_miss_total`` — the counter the skip branch
        increments does not care WHY the boundary did not fit — and
        remedied by raising ``JAATO_RUNNER_POOL_MAX_SIZE``.  The same cost
        #1033 accepted for the multi-boundary case.

        On a host with no AppArmor every profile name is empty, so a
        served slot's ``profile_name`` is ``None`` and every session's
        key wants ``None``: the second clause is a tautology and nothing
        changes.  The "no AppArmor at all: completely unchanged"
        requirement therefore still holds — the new clause only refuses a
        slot whose profile differs, which on such a host never happens.

        ``or None`` on the slot's side is not decoration.  ``self`` is a
        normalised key (``SlotKey.build`` folds ``""``), the slot's field
        is whatever is stored on it, and the OLD gate's first clause
        (``not slot.profile_name``) absorbed that asymmetry by accident.
        Removing the short-circuit removes the accident, so the
        normalisation is stated instead — the two spellings of
        "unconfined" must not read as two boundaries.
        """
        if not slot.has_served:
            # A virgin slot is threaded and in the template's domain, so
            # it can enter an AppArmor profile at bootstrap and never an
            # SELinux domain (phase 0: setcon refused in a threaded
            # process).  An SELinux session takes a slot forked into its
            # boundary (:meth:`PoolManager.fork_slot_into`) or this one's.
            return self.selinux_boundary is None
        return (
            (slot.profile_name or None) == self.profile_name
            and (getattr(slot, "selinux_boundary", None) or None) == self.selinux_boundary
            and self.uid_fits(slot)
        )

    def uid_fits(self, slot: "PoolSlot") -> bool:
        """Does *slot* run as the uid this key wants? (#1168)

        A virgin slot is still the daemon's uid and has not dropped, so it
        fits anyone — the drop happens at its first bootstrap.  A served
        slot fits only its own uid: ``None`` (never dropped) is the daemon's
        uid and a dropped slot cannot become another.
        """
        if not slot.has_served:
            return True
        return getattr(slot, "runner_uid", None) == self.runner_uid


#: A slot's POSTURE: the part of its identity that decides which sessions
#: a slot that has served may still be handed (#1507).  The three fields
#: :meth:`SlotKey.accepts_unaffined` compares on a served slot, and nothing
#: else -- workspace and config root do not gate a PURE IDLE slot.
Posture = Tuple[Optional[str], Optional[int], Optional[str]]


def posture_of_key(key: "SlotKey") -> Posture:
    """The posture a session with *key* asks for (#1507)."""
    return (key.profile_name or None, key.runner_uid,
            key.selinux_boundary or None)


def posture_of_slot(slot: "PoolSlot") -> Posture:
    """The posture a SERVED slot is stuck in (#1507).

    Only meaningful when ``slot.has_served``: a virgin slot has no posture,
    it fits any session an unaffined slot can fit.
    """
    return ((slot.profile_name or None),
            getattr(slot, "runner_uid", None),
            (getattr(slot, "selinux_boundary", None) or None))


def posture_label(posture: Posture) -> str:
    """``<profile|unconfined>/uid=<uid|daemon>[/selinux=<id>]`` for telemetry."""
    profile, uid, selinux = posture
    label = f"{profile or 'unconfined'}/uid={'daemon' if uid is None else uid}"
    if selinux:
        label += f"/selinux={selinux}"
    return label


def _canonical_path(path: Optional[str]) -> Optional[str]:
    """``realpath`` for key comparison, tolerant of a missing directory.

    ``realpath`` answers with its input when nothing is there, which is
    what is wanted: the key has to be comparable before anything is
    created on disk.
    """
    if not path:
        return None
    try:
        return os.path.realpath(str(path))
    except OSError:  # pragma: no cover — realpath is near-total
        return str(path)


# Phase 1 alias.  All in-tree callers were updated in Phase 2 to use
# ``PoolSlot``; the alias remains so external imports of
# ``server.runner_pool.SlotHandle`` keep type-checking.  No tuple
# semantics — ``PoolSlot`` is a dataclass with ``.pid`` / ``.sock``.
SlotHandle = PoolSlot


def _non_negative_int(name: str, value: Any) -> int:
    """``value`` as a non-negative ``int``, or ``ValueError`` naming it.

    ``bool`` is refused although it is an ``int``: ``resize(True)`` is a
    caller bug, not a request for one slot.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def slot_rpc_death(slot: PoolSlot) -> Optional[str]:
    """Positive evidence that *slot*'s RPC client is dead, else ``None``.

    The pool's ONE liveness predicate (#1058).  A slot whose
    ``RunnerRPCClient`` has closed cannot serve anybody: every later
    ``call`` raises up front, so the next session to take it fails
    ``session.bootstrap`` with ``RunnerCallError: RunnerRPCClient is
    closed`` — discovered by the consumer, one layer too late.

    A POLL, deliberately.  The read loop's ``finally`` already knows the
    channel died and already sets the flag; what was missing was anybody
    asking.  Asking at the point of use cannot silently stop working and
    is checkable by a test, where a callback registered at slot creation
    is inert the moment one construction path forgets it — the #735
    shape, in a pool whose slots are built on two different paths
    (``spawn_initial_slots`` and the replenish loop).

    **Only positive evidence counts**, the posture #1014 and #1023 take
    about confinement labels:

    ==================================  =========================
    Slot state                          Verdict
    ==================================  =========================
    no ``rpc`` (never served a session)  alive — there is no
                                         channel to have died, and
                                         the first session on the
                                         slot creates one
    ``rpc`` reporting ``is_closed``      **dead**, with its reason
    ...unless it reports ``can_revive``  alive — its reader was
                                         cancelled but its transport
                                         and its runner survived, and
                                         the spawn path reopens it
                                         (``revive_read_loop``).  The
                                         pool cannot do that itself:
                                         it holds no event loop, and
                                         a new read task needs one
    ``rpc`` with no ``is_closed``        alive — a duck-typed or
                                         out-of-tree client that
                                         cannot answer proves
                                         nothing, and reading
                                         absence as death would
                                         empty the pool
    ==================================  =========================

    Args:
        slot: The idle slot to judge.

    Returns:
        The client's ``close_reason`` (``"eof"``, ``"read-loop-crash"``,
        ``"explicit-close"``, ...), or the bare string ``"closed"`` when
        a client reports death without naming a cause, or ``None`` when
        there is no evidence the slot is dead.
    """
    rpc = getattr(slot, "rpc", None)
    if rpc is None:
        return None
    if not getattr(rpc, "is_closed", False):
        return None
    if getattr(rpc, "can_revive", False):
        return None
    return getattr(rpc, "close_reason", None) or "closed"


class PoolManager:
    """Manages the pre-warm pool of forked-from-template slots.

    Lifecycle:

      daemon startup ─→ template spawned (PR 2)
                         │
                         ▼
      daemon startup ─→ PoolManager(template_mgr).spawn_initial_slots(n=2)
                         │  forks N slots via template's FORK_SLOT
                         ▼
                  (N slots sit idle, waiting for bootstrap envelopes)
                         │
                         ▼
      session arrives ─→ acquire_slot() (PR 4)
                         │
                         ▼
                  (slot serves the session; replenish in background)
                         │
                         ▼
      daemon shutdown ─→ shutdown_all() — SHUTDOWN every idle slot

    Thread-safe — concurrent ``acquire_slot``/``shutdown_all`` calls
    serialize via the manager's lock.

    Attributes:
        target_size: Floor on VIRGIN idle slots — no cascade affinity and
            never served, so any arriving session may take them (default
            2).  Served unaffined slots fit one posture and sit on top of
            the floor (#1507).
        max_size: Ceiling on TOTAL idle slots, reservations included
            (default ``2 * target_size``).  Floor and ceiling count
            different things on purpose: a cascade-affined idle slot
            is capacity for exactly one tenant, so counting it as pool
            capacity starves everybody else (#898).
        _idle_slots: List of currently-idle slot handles.
    """

    def __init__(
        self,
        template_manager,
        target_size: int = 2,
        replenish_interval: float = 0.5,
        cascade_idle_timeout_seconds: float = DEFAULT_CASCADE_IDLE_TIMEOUT_SECONDS,
        max_size: Optional[int] = None,
    ) -> None:
        """Initialize the pool manager.

        Args:
            template_manager: The daemon's :class:`TemplateManager`.
                Source of fork-slot requests; must already have
                ``spawn()`` been called.
            target_size: Number of **unreserved** idle slots to keep
                available — slots any arriving session may take,
                i.e. those with no cascade affinity stamped.  Default
                2 — reasonable for typical workstation; cascade
                harnesses spawning many concurrent sessions can raise
                this via ``JAATO_RUNNER_POOL_SIZE`` env var (consumed
                by daemon ``__main__.py`` and threaded here).  Values
                <= 0 disable the pool (sessions fall back to cold-spawn
                session-mode).

                VIRGIN since #1507: a slot that has served fits only its
                own posture (#1100), so it is a reservation keyed on the
                boundary and does not count either.

                UNRESERVED, not total (#898).  A cascade-affined idle
                slot is a RESERVATION: cross-cascade reuse is forbidden
                by design, so such a slot is capacity for exactly one
                tenant and is unavailable to everybody else.  Counting
                it as pool capacity is what let a pool read "full" while
                being empty from the point of view of every tenant but
                one — two idle slots affined to cascade B, and cascade
                A's ``acquire_slot`` returning ``None`` with nothing in
                the system that would ever create a slot A could use.
            max_size: Hard ceiling on TOTAL idle slots (unreserved +
                reservations).  ``None`` (default) means
                ``2 * target_size``.  This is what bounds the growth
                that the unreserved-accounting above implies: without
                it, one reservation per live cascade accumulates
                without limit, and a slot is 129-187 MB by this file's
                own measurement.  Clamped up to ``target_size`` — a
                ceiling below the floor is not a policy.  Operators
                raise it via ``JAATO_RUNNER_POOL_MAX_SIZE``.
            replenish_interval: Seconds the replenishment thread sleeps
                between idle-count checks.  Default 0.5s — fast enough
                that a 6-step cascade refilling slots between steps
                doesn't perceive replenishment latency, slow enough
                that the thread doesn't burn CPU when the pool is at
                target size.  Pool PR 4.
            cascade_idle_timeout_seconds: Phase 2 — how long an
                IDLE_FOR_CASCADE slot may sit idle before the
                replenish loop tears it down.  Default 300s (per
                design doc §3 decision 5).  Override for tests that
                exercise the teardown path without sleeping for 5
                minutes.
        """
        self._template_manager = template_manager
        self.target_size = max(0, int(target_size))
        #: Hard ceiling on ``len(self._idle_slots)``.  ``target_size``
        #: is the floor on the UNRESERVED subset; this is the ceiling on
        #: the whole list, and the two are deliberately different
        #: numbers because reservations sit on top of the floor rather
        #: than consuming it (#898).
        self.max_size = (
            max(self.target_size, int(max_size))
            if max_size is not None else self.target_size * 2
        )
        #: Whether ``max_size`` was chosen by somebody rather than derived
        #: as ``2 * target_size``.  Decides what :meth:`resize` does with
        #: the ceiling when only the floor is given: a derived ceiling
        #: follows the floor, a chosen one is kept (clamped up).
        self._max_size_explicit = max_size is not None
        #: Slots dropped by the capacity check, awaiting teardown by the
        #: replenish thread.  They are already out of ``_idle_slots``, so
        #: the pool count is bounded the moment a slot is queued here.
        self._pending_teardown: List[PoolSlot] = []
        self._idle_slots: List[PoolSlot] = []
        self._lock = threading.Lock()
        # Pool PR 4 replenishment thread state.
        self._replenish_interval = float(replenish_interval)
        self._replenish_stop = threading.Event()
        #: Set to cut a replenish-loop pause short without stopping the
        #: loop -- :meth:`resize` uses it so a larger target starts
        #: forking now rather than after the next interval.
        self._replenish_wake = threading.Event()
        self._replenish_thread: Optional[threading.Thread] = None
        # Phase 2 cascade-sharing idle timeout.
        self._cascade_idle_timeout = float(cascade_idle_timeout_seconds)
        #: When each posture was last asked for by an unaffined acquire
        #: (#1507), monotonic.  Read only to decide which SERVED slot is
        #: the least useful to keep at the ceiling; bounded by the number
        #: of distinct postures a deployment has, like boundary-derived
        #: profile names (#1033).
        self._posture_last_demand: Dict[Posture, float] = {}

        # Optional ``(profile_name) -> None`` hook the daemon wires so a
        # slot's AppArmor profile is unloaded when the LAST slot wearing
        # it dies (#1033).  An attribute rather than a constructor
        # argument: the pool is built at daemon startup, long before any
        # ``AppArmorManager`` exists, and every test-built PoolManager
        # should keep working without knowing this exists.
        self.profile_reaper: Optional[Callable[[str], None]] = None
        # Pool PR 5d telemetry counters.  Monotonically-incrementing
        # process-lifetime totals.  Snapshot via :meth:`get_telemetry`
        # for diagnostic surfaces (logs, admin commands, OTel
        # exporters wired by external code).
        self._counters: Dict[str, int] = {
            # Number of times ``acquire_slot`` returned a real handle.
            "pool_slot_acquired_total": 0,
            # Phase 4: slots forked INTO an SELinux boundary for a session
            # no idle slot fitted, and the requests the template refused.
            "pool_selinux_fork_total": 0,
            "pool_selinux_fork_failures_total": 0,
            # A virgin slot forked for a session no idle slot fitted,
            # instead of a cold-spawned runner, and the requests the
            # template could not fork (the session then cold-spawns).
            "pool_demand_fork_total": 0,
            "pool_demand_fork_failures_total": 0,
            # Number of times ``acquire_slot`` returned None (pool
            # empty when called).  Sessions in this state fall back
            # to cold-spawn — useful for sizing decisions.
            "pool_acquire_miss_total": 0,
            # Number of times the replenishment thread forked a new
            # slot to top up the pool.
            "pool_replenish_success_total": 0,
            # Number of times ``request_fork_slot`` returned None
            # during replenishment (template stuck, IPC error, etc.).
            "pool_replenish_failures_total": 0,
            # Number of times the watchdog initiated a template
            # respawn.  Each unit of this counter corresponds to one
            # detected template death.
            "template_respawn_attempts_total": 0,
            # Number of those respawn attempts that raised an
            # exception (template_manager.spawn() failed).  Watchdog
            # retries on next iteration, so attempts >> failures is
            # normal during a flaky-template incident.
            "template_respawn_failures_total": 0,
            # Phase 2 cascade-sharing.  Number of times acquire_slot
            # found a slot already affined to the requested cascade
            # and reused it (warm pool slot AND warm plugin state
            # AND warm LSP server connections).
            "cascade_slot_reuse_hits_total": 0,
            # Phase 2.  Number of times acquire_slot received a
            # cascade_driver_id but no IDLE_FOR_CASCADE(C) slot
            # existed — fell through to a PURE-IDLE slot or to
            # cold-spawn.  Pairs with hits_total to measure
            # cascade-reuse efficacy on the workload.
            "cascade_slot_reuse_misses_total": 0,
            # Phase 2.  Number of cascade-affined idle slots torn
            # down by the replenish-loop sweep because they exceeded
            # ``cascade_idle_timeout_seconds`` without serving a new
            # session.  Should match the number of cascades that
            # actually finished (long-pause cascades within timeout
            # don't count).
            "cascade_slots_idle_torndown_total": 0,
            # Slots dropped by the capacity check (queued for
            # teardown), and the subset of those actually reclaimed by
            # the replenish thread's drain.  Declared here rather than
            # sprung into existence by ``_incr`` so ``get_telemetry``
            # reports 0 instead of omitting the key on a daemon that
            # has never hit the ceiling.
            "pool_slots_over_cap_total": 0,
            "pool_slots_over_cap_torndown_total": 0,
            # #898.  Times a RESERVATION (cascade-affined idle slot)
            # was evicted to admit a returning slot whose cascade is
            # demonstrably live.  Growth in this counter alongside
            # ``cascade_slot_reuse_misses_total`` means ``max_size`` is
            # too small for the number of concurrent tenants.
            "pool_stale_reservation_evicted_total": 0,
            # #898.  Times the replenish loop wanted to fork an
            # unreserved slot (unreserved < target_size) and could not
            # because the pool was at ``max_size``.  THE sizing signal
            # for a multi-tenant daemon: a nonzero value means some
            # tenant is being served by cold-spawn while reservations
            # hold the ceiling.  Raise ``JAATO_RUNNER_POOL_MAX_SIZE``.
            "pool_replenish_ceiling_blocked_total": 0,
            # #1058.  Idle slots dropped because their RunnerRPCClient
            # had died while the slot sat in the pool.  Declared here
            # rather than sprung into existence by ``_incr`` so
            # ``get_telemetry`` reports 0 on a healthy daemon instead of
            # omitting the key — an absent counter reads as "never
            # measured", which is the one thing this must not say.
            #
            # NONZERO IS NOT THE POOL RECOVERING QUIETLY.  Each unit is a
            # runner that died without anybody asking it to, so a growing
            # value is evidence of an external kill (OOM, a signal) or of
            # a frame-level fault on the channel.  The WARNING beside it
            # names which.
            "pool_dead_slot_evicted_total": 0,
            # #1058.  Returns refused because the slot was already in
            # the idle pool.  Nonzero means some session is being torn
            # down twice; the ERROR beside it names the slot.
            "pool_duplicate_return_refused_total": 0,
            # #1565.  Slots torn down at session end instead of returned,
            # because a plugin said the process holds memory nothing frees
            # (an embedding model loaded there).  Each unit is a cold fork
            # the next session of that posture pays for.
            "pool_slot_retired_total": 0,
            # #1168 step 3.  Served idle slots passed over because they
            # run as another uid than the arriving session drops to.  Also
            # counted in pool_profile_mismatch_skips_total (the skip
            # branch does not care why the slot did not fit); this one
            # says the uid was the reason.
            "pool_uid_mismatch_skips_total": 0,
            # #1507: an unaffined acquire that passed over SERVED slots of
            # another posture and found nothing.  Before the virgin floor
            # this was every second-posture arrival on a mixed daemon;
            # now it should stay near zero once the pool has warmed.
            # Growing means the floor is too small for the arrival rate.
            "pool_posture_miss_total": 0,
            # #1507: a served slot (no cascade) dropped so the floor could
            # be refilled with a virgin one at the ceiling, or displaced by
            # a returner of a posture asked for more recently.
            "pool_served_slot_evicted_total": 0,
        }
        self._counters_lock = threading.Lock()

    def spawn_initial_slots(self) -> int:
        """Fork ``target_size`` slots from the template.

        Called once at daemon startup, AFTER the template has
        finished its plugin discovery (which is asynchronous in
        the template — daemon should wait briefly or hope the
        template is fast enough).  Failures are individually
        non-fatal; the pool ends up smaller than target.

        PR 3 ships this as a synchronous loop at daemon startup —
        callers ride out the blocking time.  PR 5 will move
        replenishment into a background thread + add the
        slot-replenishment trigger on acquire.

        Returns:
            Number of slots successfully forked (≤ ``target_size``).
        """
        if self.target_size <= 0:
            logger.info(
                "PoolManager: target_size=%d; pool disabled",
                self.target_size,
            )
            return 0

        forked = 0
        with self._lock:
            for i in range(self.target_size):
                raw = self._template_manager.request_fork_slot()
                if raw is None:
                    logger.warning(
                        "PoolManager: fork-slot %d/%d failed; pool "
                        "will be smaller than target.  Sessions fall "
                        "back to cold-spawn session-mode when pool is "
                        "empty.", i + 1, self.target_size,
                    )
                    break
                pid, sock = raw
                self._idle_slots.append(PoolSlot(pid=pid, sock=sock))
                forked += 1

        logger.info(
            "PoolManager: spawned %d pool slot(s); target=%d",
            forked, self.target_size,
        )
        return forked

    def _evict_dead_slots_locked(self) -> int:
        """Drop every idle slot whose RPC client has died (#1058).

        **Caller must hold ``self._lock``.**  Evicted slots are moved to
        ``_pending_teardown``, which the replenish thread drains through
        :meth:`_teardown_slot` — so the runner process is reaped and its
        AppArmor profile unloaded.  Skipping a corpse and leaving it in
        the list would have been cheaper and wrong twice over: the
        process leaks, and — the sharper half — the corpse keeps counting
        toward ``unreserved_idle_count`` and ``max_size``, so
        replenishment reads the pool as full and never forks the
        replacement.  That is #898's failure in a new costume: capacity
        that no arriving session can use, counted as capacity for
        everybody.

        The ONE removal point, called from :meth:`acquire_slot` (so the
        check cannot be forgotten on one of the two affinity paths) and
        from :meth:`_sweep_dead_slots` (so a corpse is reaped and its
        capacity restored even when no session arrives to trip over it).

        Logs at WARNING and counts ``pool_dead_slot_evicted_total``: a
        dead runner is never routine, and a pool that recovered silently
        would hide whatever killed it.  Both happen while the lock is
        held, matching :meth:`acquire_slot`, which already calls
        :meth:`_incr` inside it.

        Returns:
            How many slots were evicted.
        """
        survivors: List[PoolSlot] = []
        dead: List[Tuple[PoolSlot, str]] = []
        for slot in self._idle_slots:
            reason = slot_rpc_death(slot)
            if reason is None:
                survivors.append(slot)
            else:
                dead.append((slot, reason))
        if not dead:
            return 0
        self._idle_slots = survivors
        for slot, reason in dead:
            slot.teardown_reason = f"dead-rpc:{reason}"
            self._pending_teardown.append(slot)
            self._incr("pool_dead_slot_evicted_total")
            logger.warning(
                "PoolManager: idle slot pid=%d (cascade=%s) dropped — its "
                "RPC client is closed (reason=%s); the runner died while "
                "the slot sat in the pool.  Queued for teardown; the pool "
                "will refill (#1058).",
                slot.pid, slot.cascade_id or "(pure)", reason,
            )
        return len(dead)

    def discard_acquired_slot(self, slot: PoolSlot, *, reason: str) -> None:
        """Give up on a slot that was already handed out (#1058).

        An acquired slot is NOT in ``_idle_slots`` — :meth:`acquire_slot`
        popped it — so nothing in the pool would ever reap it if its
        consumer simply walked away.  This queues it for the same
        teardown a dropped idle slot gets: transport closed, runner
        process reaped, AppArmor profile unloaded when the last slot
        wearing it dies.

        The alternative, letting the caller abandon the handle, leaks a
        runner process per occurrence — worse than the failure being
        recovered from.

        Args:
            slot: The acquired slot to give up on.
            reason: Short token for the log line, e.g.
                ``"dead-rpc:eof"``.
        """
        slot.teardown_reason = reason
        with self._lock:
            self._pending_teardown.append(slot)
        self._incr("pool_dead_slot_evicted_total")
        logger.warning(
            "PoolManager.discard_acquired_slot: slot pid=%d discarded "
            "(%s); queued for teardown",
            slot.pid, reason,
        )

    def _sweep_dead_slots(self) -> None:
        """Reap idle slots whose RPC client has died.

        Runs every replenish iteration beside :meth:`_sweep_cascade_idle`
        and is deliberately NOT exempt for ``cascade_id is None``.

        That exemption is right for the cascade sweep — a PURE IDLE slot
        has no cascade affinity to time out, and tearing it down on a
        timer would throw away exactly the warmth the pool exists for.
        It is wrong as a health policy, and it is why the reported
        incident went unnoticed: a standalone session returns its slot as
        PURE IDLE (#1033), so the one reaper the pool had was exempt from
        looking at it, and nothing else looked at all.  Warmth and
        liveness are different questions; only the first has a timeout.
        """
        with self._lock:
            self._evict_dead_slots_locked()

    def idle_count(self) -> int:
        """Return the current count of idle slots — reservations included.

        This is the number ``max_size`` bounds.  It is NOT the number a
        waiter can draw on: see :meth:`unreserved_idle_count`.
        """
        with self._lock:
            return len(self._idle_slots)

    def unreserved_idle_count(self) -> int:
        """Return the count of idle slots usable by ANY arriving session.

        A slot carrying a ``cascade_id`` is a reservation — cross-cascade
        reuse is forbidden (warm plugin state belongs to the original
        cascade), so it is capacity for one tenant and for nobody else.
        Only PURE-IDLE slots are the pool's free capacity (#898).  Since
        #1507 the replenish loop compares the narrower
        :meth:`virgin_idle_count` against ``target_size``: a pure-idle slot
        that has served is free capacity for its own posture only.

        Reading total idle there instead meant a pool holding two
        B-affined slots read "full" while ``acquire_slot(cascade=A)``
        returned ``None`` — no affine match, no pure-idle, and no
        replenishment that would ever produce one.  A starved until B
        released, which for the reporting incident was 31 s past the
        client's 60 s budget.
        """
        with self._lock:
            return sum(1 for s in self._idle_slots if s.cascade_id is None)

    def virgin_idle_count(self) -> int:
        """Return the count of idle slots that have never served (#1507).

        THIS is the floor ``target_size`` keeps stocked, and the reason is
        the same as #898's one layer in.  Since #1100 a slot that has
        served fits only a session of its own posture (AppArmor profile,
        uid, SELinux boundary), so an unreserved but SERVED slot is
        capacity for one posture and for nobody else -- a reservation
        keyed on the boundary instead of the cascade.  Counting it toward
        the floor let a daemon mixing unconfined IPC sessions and confined
        WS sessions read "full" on slots the second posture could not
        take, so that posture cold-spawned every time (21/21 in the
        reporting bench).  A virgin slot fits every session an unaffined
        slot can fit, so it is the only capacity that is capacity for all.

        Served-but-unaffined slots sit on top of the floor, like
        reservations, bounded by ``max_size``.
        """
        with self._lock:
            return self._virgin_idle_count_locked()

    def _virgin_idle_count_locked(self) -> int:
        """:meth:`virgin_idle_count`, for a caller holding ``self._lock``."""
        return sum(1 for s in self._idle_slots
                   if s.cascade_id is None and not s.has_served)

    def posture_idle_counts(self) -> Dict[str, int]:
        """Idle slots by what they can still serve (#1507).

        ``virgin`` (never served, fits any posture), ``reserved``
        (cascade-affined), and one ``served:<posture_label>`` entry per
        posture that has served-but-unaffined slots idle.
        """
        counts: Dict[str, int] = {"virgin": 0, "reserved": 0}
        with self._lock:
            for slot in self._idle_slots:
                if slot.cascade_id is not None:
                    counts["reserved"] += 1
                elif not slot.has_served:
                    counts["virgin"] += 1
                else:
                    name = "served:" + posture_label(posture_of_slot(slot))
                    counts[name] = counts.get(name, 0) + 1
        return counts

    def acquire_slot(
        self, cascade_driver_id: Optional[str] = None,
        config_root: Optional[str] = None,
        workspace_root: Optional[str] = None,
        profile_name: Optional[str] = None,
        runner_uid: Optional[int] = None,
        selinux_boundary: Optional[str] = None,
    ) -> Optional[PoolSlot]:
        """Pop an idle slot off the pool — cascade-affinity aware (Phase 2).

        Every idle slot whose RPC client has died is dropped FIRST
        (:meth:`_evict_dead_slots_locked`), so both affinity paths below
        walk a list that contains only slots a session can actually be
        served by.  Liveness is a property of the pool, not a
        precondition the caller has to re-check — see #1058 for what the
        caller's-problem version cost.

        Affinity routing (per design doc §4.2 claim flow):

        1. If ``cascade_driver_id`` is provided, walk the idle list for
           the first slot whose whole :class:`SlotKey` matches.  Match
           → return it (reuse hit).
        2. Else (no cascade requested OR no match), take a PURE IDLE
           slot (``cascade_id is None``) **whose kernel boundary fits**
           — see :meth:`SlotKey.accepts_unaffined`.  Stamp the key on it;
           the slot is now that key's for the rest of its life.
        3. If no idle slot qualifies, return ``None``.  Caller falls
           back to cold-spawn (which spawns a fresh session-mode
           runner, NOT a pool slot).

        Caller becomes responsible for the slot — must either send
        a bootstrap envelope (the daemon's spawn_session_runner path)
        or close the daemon-side socket (which signals the slot to
        exit).

        **The boundary gate is on BOTH paths, and path (2) is the one
        that is easy to miss** (#1033).  A standalone session — no
        cascade — returns its slot to the pool as PURE IDLE, confined to
        the profile it served under; path (2) then hands that slot to
        the next arrival whatever profile IT wants.  Gating only the
        cascade path would leave every standalone reuse producing the
        cross-profile thread population #1023 refuses.

        Telemetry (in addition to the Phase 1 counters):
          - ``cascade_slot_reuse_hits_total`` incremented on path (1).
          - ``cascade_slot_reuse_misses_total`` incremented when
            ``cascade_driver_id`` is supplied but path (1) didn't fire
            (whether or not we fell through to a PURE IDLE on path
            (2)).  Pairs with hits_total to score cascade-reuse
            efficacy on the workload.
          - ``pool_profile_mismatch_skips_total`` counts idle slots that
            path (2) passed over because their kernel boundary did not
            fit.  Nonzero on a daemon whose sessions span several
            workspaces or several profile shapes; growing alongside
            ``pool_acquire_miss_total`` means the pool is being split
            across more boundaries than ``target_size`` keeps stocked.
            Since #1100 it also counts a slot that served an UNCONFINED
            session being passed over by a confined one — the counter
            does not care why the boundary did not fit, so the cost of
            that gate needs no new instrumentation.

        Args:
            cascade_driver_id: Optional cascade tenant ID.  ``None``
                (default) means "no cascade affinity" — the slot is
                taken from, and returned to, the unaffined pool.
            config_root: The ``.jaato`` root this session's warm plugin
                state would be built from.
            workspace_root: The workspace this session runs in.  Part of
                the key since #1033 — see :class:`SlotKey`.
            profile_name: The AppArmor profile this session's runner will
                confine to, ``""``/``None`` for unconfined.
            runner_uid: The uid this session's runner will drop to (#1168),
                ``None`` for the daemon's own.  A served slot of another
                uid is passed over and counted in
                ``pool_uid_mismatch_skips_total``.
            selinux_boundary: The SELinux boundary's ``confinement_id``
                (phase 4), ``None`` when the session is not SELinux-
                confined.  Only a slot forked into that boundary fits; on
                a miss the caller forks one (:meth:`fork_slot_into`).

        Returns:
            A :class:`PoolSlot` carrying the requested key and a LIVE
            RPC client, or ``None`` when nothing in the pool qualifies.
        """
        key = SlotKey.build(
            cascade_driver_id=cascade_driver_id,
            config_root=config_root,
            workspace_root=workspace_root,
            profile_name=profile_name,
            runner_uid=runner_uid,
            selinux_boundary=selinux_boundary,
        )
        mismatch_skips = 0
        uid_skips = 0
        with self._lock:
            # #1058: drop corpses BEFORE either affinity path walks the
            # list.  Here rather than inside each path because there are
            # two of them and a third would be written one day; a filter
            # applied at the single point they both read from is one that
            # cannot be forgotten on one branch.
            self._evict_dead_slots_locked()

            # Path (1): whole-key match.  Every field is load-bearing and
            # the reasons differ per field — see :class:`SlotKey`.
            if key.cascade_driver_id is not None:
                for i, slot in enumerate(self._idle_slots):
                    if SlotKey.of_slot(slot) == key:
                        slot = self._idle_slots.pop(i)
                        self._incr("pool_slot_acquired_total")
                        self._incr("cascade_slot_reuse_hits_total")
                        logger.info(
                            "PoolManager.acquire_slot: cascade reuse "
                            "HIT — slot pid=%d cascade=%s profile=%s",
                            slot.pid, key.cascade_driver_id,
                            key.profile_name or "(unconfined)",
                        )
                        return slot
                # Cascade requested but no match.
                self._incr("cascade_slot_reuse_misses_total")

            # Path (2) / (3): a PURE IDLE slot whose boundary fits, else
            # None.  PURE IDLE is preferred over affined-to-another-
            # cascade because cross-cascade reuse is forbidden by design
            # (warm plugin state belongs to the original cascade).
            #
            # A SERVED slot that fits is preferred over a virgin one
            # (#1507): it can serve only this posture, while a virgin
            # can serve any, so spending the served one keeps the floor
            # of virgins intact and avoids a fork.
            self._posture_last_demand[posture_of_key(key)] = time.monotonic()
            pure_idle_idx, mismatch_skips, uid_skips, served_skips = (
                self._pick_unaffined_locked(key))
            if uid_skips:
                self._incr("pool_uid_mismatch_skips_total", uid_skips)
            if pure_idle_idx is None:
                self._incr("pool_acquire_miss_total")
                if served_skips:
                    # The #1507 case: warm runners were there, all of
                    # another posture.  Wake the replenish loop so the
                    # virgin floor is refilled now rather than after its
                    # pause.
                    self._incr("pool_posture_miss_total")
                    self._replenish_wake.set()
                if mismatch_skips:
                    self._incr(
                        "pool_profile_mismatch_skips_total", mismatch_skips)
                    logger.info(
                        "PoolManager.acquire_slot: no usable idle slot — "
                        "%d idle slot(s) passed over because they are "
                        "confined to another profile (wanted %s)",
                        mismatch_skips, key.profile_name or "(unconfined)",
                    )
                return None
            slot = self._idle_slots.pop(pure_idle_idx)
        if mismatch_skips:
            self._incr("pool_profile_mismatch_skips_total", mismatch_skips)

        # Stamp the key.  A slot picked up on path (2) may have been
        # unconfined and is about to be confined, or may already wear
        # exactly this profile; either way the four fields now describe
        # it, and they are written together for the reason
        # :meth:`SlotKey.stamp` gives.
        key.stamp(slot)
        if key.cascade_driver_id is not None:
            logger.info(
                "PoolManager.acquire_slot: cascade reuse MISS — fresh "
                "slot pid=%d stamped cascade=%s profile=%s",
                slot.pid, key.cascade_driver_id,
                key.profile_name or "(unconfined)",
            )
        self._incr("pool_slot_acquired_total")
        return slot

    def _pick_unaffined_locked(
        self, key: SlotKey,
    ) -> Tuple[Optional[int], int, int, int]:
        """Path (2) of :meth:`acquire_slot`: which PURE IDLE slot *key* gets.

        Caller holds ``self._lock``.  Returns ``(index or None,
        mismatch_skips, uid_skips, served_skips)``.  A served slot that
        fits is preferred over a virgin (#1507); ``served_skips`` counts
        the served slots of ANOTHER posture passed over, which is what
        makes a miss a posture miss.
        """
        virgin_idx: Optional[int] = None
        mismatch_skips = uid_skips = served_skips = 0
        for i, candidate in enumerate(self._idle_slots):
            if candidate.cascade_id is not None:
                continue
            if not key.accepts_unaffined(candidate):
                uid_skips += not key.uid_fits(candidate)
                served_skips += candidate.has_served
                mismatch_skips += 1
                continue
            if not candidate.has_served:
                if virgin_idx is None:
                    virgin_idx = i
                continue
            return i, mismatch_skips, uid_skips, served_skips
        return virgin_idx, mismatch_skips, uid_skips, served_skips

    def fork_slot_into(
        self, key: SlotKey, entry: Dict[str, Any],
    ) -> Optional[PoolSlot]:
        """Fork a slot that enters an SELinux boundary at fork (phase 4).

        For a session :meth:`acquire_slot` had no slot for: a virgin slot
        is threaded and cannot ``setcon`` (selinux-backend.md §7.2), so
        the template forks one for this session, and the child enters
        *entry* (``context``, ``private_tmp``, ``runner_user``) before it
        starts a thread.  The returned slot is checked out to the caller,
        stamped with *key*, and returns to the pool like any other, where
        it fits only *key*'s boundary and uid.

        ``None`` when the template could not fork (dead, timed out); the
        caller cold-spawns, as on any miss.  A child that cannot ENTER the
        boundary exits instead (``SELINUX_ENTRY_EXIT_CODE``), which the
        caller sees as a bootstrap that failed, not as a miss: a session
        whose boundary cannot be entered must not run anywhere else.
        """
        raw = self._template_manager.request_fork_slot(entry=entry)
        if raw is None:
            self._incr("pool_selinux_fork_failures_total")
            return None
        pid, sock = raw
        slot = PoolSlot(pid=pid, sock=sock)
        key.stamp(slot)
        self._incr("pool_selinux_fork_total")
        logger.info(
            "PoolManager: forked slot pid=%d into SELinux boundary %s "
            "(context %s)", pid, key.selinux_boundary, entry.get("context"),
        )
        return slot

    def fork_slot_on_demand(self, key: SlotKey) -> Optional[PoolSlot]:
        """Fork a virgin slot for a session no idle slot fitted.

        The miss used to be served by a cold-spawned runner, which
        imports everything itself and so holds its whole working set as
        private memory (~135 MB measured, against ~25 MB for a slot
        sharing the template's pages), and takes seconds to start.  A
        cascade fanning out past the pool's floor paid that per stage.

        A slot forked here is exactly what the replenish loop would have
        forked: virgin, with the template's warm imports.  Everything
        that binds it to the session (private ``/tmp``, the uid drop,
        the AppArmor transition) happens at ``session.bootstrap``, as for
        an idle virgin slot.  It is checked out to the caller, stamped
        with *key*, and returns to the pool like any other.

        Not bounded by ``max_size``: that bounds IDLE slots, and the
        session this serves would have had a process either way.

        ``None`` when the template could not fork; the caller cold-spawns.
        """
        raw = self._template_manager.request_fork_slot()
        if raw is None:
            self._incr("pool_demand_fork_failures_total")
            return None
        pid, sock = raw
        slot = PoolSlot(pid=pid, sock=sock)
        key.stamp(slot)
        self._incr("pool_demand_fork_total")
        logger.info(
            "PoolManager: forked slot pid=%d on demand (no idle slot fitted; "
            "cascade=%s)", pid, key.cascade_driver_id or "(standalone)",
        )
        return slot

    def return_slot_after_session(self, slot: PoolSlot) -> bool:
        """Return a slot to the idle pool after its session ended (Phase 2).

        Pre-condition: caller has ALREADY invoked the slot's
        ``session_end`` RPC successfully (which fired
        ``reset_for_next_session`` on each plugin).  This method is
        purely the pool-side bookkeeping — it does NOT call the
        slot's RPC.  On ``session_end`` failure, caller must NOT
        invoke this method; the slot should be torn down via
        ``slot.sock.close()`` + waitpid (the slot is in undefined
        state and cannot be reused).

        Stamps ``last_session_end_ts`` to the current monotonic
        wall-clock for the idle-teardown sweep.

        **ONE SLOT, ONE ENTRY.**  A slot object already in
        ``_idle_slots`` is refused (by identity) and reported at ERROR.
        The list is a set of distinct runners in disguise, and a
        duplicate is not a cosmetic double-count: the two entries share
        one ``rpc``, so the first to be acquired and torn down leaves the
        SECOND sitting in the pool pointing at a closed client and a
        reaped process — #1058's exact state, arrived at without
        anything having gone wrong on the channel.

        The producer of a duplicate is a double teardown of one session:
        :meth:`JaatoServer.shutdown` captures ``_runner_rpc`` /
        ``_spawned_runner`` / ``_pool_manager_ref`` and nulls them
        WITHOUT a lock, and it is called from several places (session
        unload, ``session.stop``, the #812 orphan sweep, daemon
        shutdown), so two concurrent callers can both see the live
        triple and both return the same slot.  Guarding here rather than
        there is the #674 argument: the pool owns this list, one
        boundary check covers every present and future caller, and the
        alternative is auditing every teardown path forever.  It does
        not FIX the double shutdown — it makes the double shutdown
        visible (an ERROR naming the slot) instead of silently
        corrupting the pool.

        Args:
            slot: The slot to return.  ``slot.cascade_id`` controls
                whether the slot becomes IDLE_FOR_CASCADE (cascade_id
                set) or PURE IDLE (cascade_id None — only legal if
                the slot was never stamped with a cascade, i.e. it
                served a standalone session).

        Returns:
            ``True`` when the slot is now IN the pool, ``False`` when it
            was dropped at capacity or refused as a duplicate.  Callers
            log their own teardown lines and used to assert the first
            outcome unconditionally; see the log-line note below.
        """
        slot.last_session_end_ts = time.monotonic()
        evicted: Optional[PoolSlot] = None
        stale_reservation_evicted = False
        served_evicted = False
        with self._lock:
            if any(s is slot for s in self._idle_slots):
                logger.error(
                    "PoolManager.return_slot_after_session: slot pid=%d is "
                    "ALREADY in the idle pool — refusing to add it twice.  "
                    "Its session was torn down more than once; the two "
                    "entries would share one rpc client, so tearing either "
                    "down would leave the other pointing at a closed "
                    "channel (#1058).",
                    slot.pid,
                )
                self._incr("pool_duplicate_return_refused_total")
                return False
            if len(self._idle_slots) >= self.max_size:
                # AT CAPACITY.  ``target_size`` said how many to keep at
                # LEAST and nothing said how many at most, so this list
                # grew by one runner per cleanly-ended pool session and
                # never shrank while the daemon lived.  Measured live at
                # 32 slots / 4237 MB, surfacing as ``pass show`` failing
                # because gpg could not fork -- a resource leak wearing a
                # credentials error.
                #
                # The ceiling is ``max_size``, not ``target_size``:
                # ``target_size`` is the floor on the UNRESERVED subset,
                # and reservations sit on top of it (#898).
                #
                # DEFAULT: the RETURNING slot goes.  It has just served a
                # session, so it carries that session's accumulated heap
                # (measured 129-187 MB on long-lived trees), while a
                # resident may still be a CoW-cheap template fork.  Keeping
                # what is already here also avoids churning the pool on
                # every return once it is full.
                victim_ix = self._pick_capacity_victim(slot)
                if victim_ix is not None:
                    evicted = self._idle_slots.pop(victim_ix)
                    stale_reservation_evicted = evicted.cascade_id is not None
                    served_evicted = (not stale_reservation_evicted
                                      and evicted.has_served)
                    self._idle_slots.append(slot)
                else:
                    evicted = slot
            else:
                self._idle_slots.append(slot)
        if stale_reservation_evicted:
            self._incr("pool_stale_reservation_evicted_total")
        if served_evicted:
            self._incr("pool_served_slot_evicted_total")

        if evicted is not None:
            # Handed to the replenish thread rather than torn down here.
            # ``_teardown_slot`` blocks on the daemon loop, and THIS path
            # runs wherever ``JaatoServer.shutdown`` runs.  The count is
            # already bounded above; only the process teardown is deferred,
            # by at most one replenish interval (~0.5s).
            with self._lock:
                self._pending_teardown.append(evicted)
            self._incr("pool_slots_over_cap_total")
            logger.info(
                "PoolManager: pool at capacity (%d/%d); slot pid=%d "
                "(cascade=%s) queued for teardown",
                len(self._idle_slots), self.max_size,
                evicted.pid, evicted.cascade_id or "(pure)",
            )
        # The outcome, said accurately.  This line used to read
        # "returned to pool" whatever happened — including when ``slot``
        # was the one dropped at capacity, which is the branch a pool
        # reading ``idle_count=N/N`` is in.  A log that asserts the slot
        # is pooled when it is on its way to teardown is not a small
        # thing: it is what an operator reads when a later session fails
        # on that slot, and it sent the #1058 diagnosis down the wrong
        # path for two rounds.
        pooled = evicted is not slot
        logger.info(
            "PoolManager.return_slot_after_session: slot pid=%d %s "
            "(cascade=%s; idle_count=%d/%d, unreserved=%d/%d)",
            slot.pid,
            "returned to pool" if pooled
            else "NOT pooled — dropped at capacity, queued for teardown",
            slot.cascade_id or "(pure)",
            len(self._idle_slots), self.max_size,
            self.unreserved_idle_count(), self.target_size,
        )
        return pooled

    def _pick_capacity_victim(
        self, returner: Optional[PoolSlot] = None,
    ) -> Optional[int]:
        """Choose which resident a returning slot (or a shrink) drops.

        Caller holds ``self._lock``.  Returns an index into
        ``self._idle_slots``, or ``None`` when the returner should be
        the one dropped.  ``returner=None`` (a shrink, no slot arriving)
        follows the affine order below.

        An UNAFFINED returner (#1507) displaces only a SERVED unaffined
        resident whose posture was last asked for longer ago than the
        returner's own -- warm capacity for a posture nobody is using
        traded for warm capacity for one somebody is.  It never displaces
        a reservation or a virgin slot; with no such resident it goes.

        LIVENESS OUTRANKS WARMTH (#898).  The rule used to be "an affine
        slot displaces a PURE-IDLE resident, otherwise the returner
        goes", which fires only against *unaffiliated* residents.  When
        both residents were affined to a second cascade, the returner --
        belonging to a cascade demonstrably mid-run, whose next stage
        was already queued -- was destroyed to preserve two slots of a
        cascade that might never come back.

        So the order is:

        1. **The stalest RESERVATION** (largest time since its
           ``last_session_end_ts``; a reservation that never served
           sorts first, since it carries no warm state at all).  Losing
           a reservation costs its cascade a cold plugin bootstrap on
           its next stage; that cascade still RUNS, because it falls
           through to a pure-idle slot.  A slot idle long enough to be
           the stalest is also the one the 300 s cascade-idle sweep was
           going to reap anyway.
        2. **A pure-idle resident**, when there is no reservation to
           drop.  This is the original rule and its original reason: a
           pure-idle slot carries no warm state, so trading it for a
           slot that does keeps the warm path the pool exists to serve.
           Among pure-idle residents a SERVED one goes before a virgin
           (#1507), least recently demanded posture first: a served slot
           fits one posture, a virgin fits all of them.
        3. **Neither** (``None``) -- the pool holds nothing at all, so
           the returner is the only candidate.

        Note what (1) does NOT do: it never leaves a tenant with no way
        to run.  Reservations are the thing being spent, and a cascade
        that loses one takes an unreserved slot instead -- which
        ``target_size`` keeps stocked.  That is the whole trade: warm
        state for one tenant is negotiable, capacity for every tenant
        is not.
        """
        if returner is not None and returner.cascade_id is None:
            served = self._served_unaffined_by_usefulness()
            if not served:
                return None
            ix, victim = served[0]
            if (self._posture_demand(victim)
                    < self._posture_demand(returner)):
                return ix
            return None
        reservations = [
            (i, s) for i, s in enumerate(self._idle_slots)
            if s.cascade_id is not None
        ]
        if reservations:
            return min(
                reservations,
                key=lambda pair: (
                    pair[1].last_session_end_ts
                    if pair[1].last_session_end_ts is not None
                    else float("-inf")
                ),
            )[0]
        served = self._served_unaffined_by_usefulness()
        if served:
            return served[0][0]
        return next(
            (i for i, s in enumerate(self._idle_slots)
             if s.cascade_id is None),
            None,
        )

    def _posture_demand(self, slot: PoolSlot) -> float:
        """When *slot*'s posture was last asked for; ``-inf`` if never."""
        return self._posture_last_demand.get(
            posture_of_slot(slot), float("-inf"))

    def _served_unaffined_by_usefulness(self) -> List[Tuple[int, PoolSlot]]:
        """Served, unaffined idle slots, least useful first (#1507).

        Caller holds ``self._lock``.  Ordered by when the slot's posture
        was last asked for (oldest first), then by how long the slot has
        sat idle.  The head of the list is what the ceiling spends first
        among slots that fit only one posture.
        """
        served = [(i, s) for i, s in enumerate(self._idle_slots)
                  if s.cascade_id is None and s.has_served]
        return sorted(served, key=lambda pair: (
            self._posture_demand(pair[1]),
            pair[1].last_session_end_ts
            if pair[1].last_session_end_ts is not None else float("-inf"),
        ))

    def note_slot_retired(self, slot: "PoolSlot") -> None:
        """Count a slot the daemon retired at session end (#1565).

        The slot is not in ``_idle_slots`` (it was acquired) and is closed
        by the caller's cold path, so this is bookkeeping only; the
        replenish loop restores the floor as for any other lost slot.
        """
        self._incr("pool_slot_retired_total")

    def _incr(self, key: str, delta: int = 1) -> None:
        """Atomic-ish bump of a telemetry counter.  Internal helper."""
        with self._counters_lock:
            self._counters[key] = self._counters.get(key, 0) + delta

    def get_telemetry(self) -> Dict[str, int]:
        """Return a snapshot of the pool's telemetry counters.

        Snapshot semantics: caller gets a stable dict copy at the
        moment of the call.  Counters keep incrementing concurrently
        on the replenishment thread + acquire path; subsequent
        snapshots reflect those increments.

        Keys (pool PR 5d):
          - ``pool_slot_acquired_total``: sessions served by a pool
            slot.  High value relative to ``pool_acquire_miss_total``
            means pool is well-sized for the workload.
          - ``pool_acquire_miss_total``: ``acquire_slot`` returned
            None.  The session then gets a slot forked for it
            (:meth:`fork_slot_on_demand`, or :meth:`fork_slot_into` for
            an SELinux session), and a cold spawn only if the template
            cannot fork.  Growing fast means the floor is too low for
            the workload: each miss pays a fork and a plugin bootstrap.
          - ``pool_demand_fork_total`` / ``..._failures_total``: virgin
            slots forked for a session at a miss, and the requests the
            template could not fork (those sessions cold-spawned).
          - ``pool_selinux_fork_total`` / ``..._failures_total`` (phase
            4): slots forked into an SELinux boundary, and the requests
            the template could not fork.
          - ``pool_stale_reservation_evicted_total``: a cascade-affined
            idle slot was dropped at the ceiling to admit a returning
            slot of a live cascade.  Its cascade pays a cold plugin
            bootstrap next stage; it does not stall.
          - ``pool_replenish_ceiling_blocked_total``: replenishment
            wanted an unreserved slot and ``max_size`` forbade it.
            Nonzero on a multi-tenant daemon means raise
            ``JAATO_RUNNER_POOL_MAX_SIZE``.
          - ``pool_replenish_success_total``: replenishment thread
            successfully forked a new slot.
          - ``pool_replenish_failures_total``: replenishment thread's
            ``request_fork_slot`` returned None.  Investigate
            template health if growing.
          - ``template_respawn_attempts_total``: watchdog initiated a
            template respawn (template died).
          - ``template_respawn_failures_total``: respawn ``spawn()``
            raised.  Attempts >> failures means flaky template;
            attempts == failures means template can't be respawned
            (operator action needed).
          - ``pool_dead_slot_evicted_total`` (#1058): idle slots
            dropped because their RPC client had died in the pool.
            NOT a success metric — each unit is a runner that died
            unasked, so a growing value points at an external kill
            (OOM, a signal) or a frame-level channel fault.  The
            WARNING beside each eviction names which.
          - ``pool_posture_miss_total`` (#1507): an unaffined acquire
            found only served slots of another posture.  Should stay
            near zero once the pool is warm; growing means raise the
            floor.
          - ``pool_served_slot_evicted_total`` (#1507): a served slot
            spent at the ceiling, for the virgin floor or for a returner
            of a posture asked for more recently.

        Gauges (#1507), read at the call rather than accumulated:
          - ``pool_idle_virgin``: idle slots that never served — the
            quantity ``target_size`` keeps stocked.
          - ``pool_idle_reserved``: cascade-affined idle slots.
          - ``pool_idle_served:<profile|unconfined>/uid=<uid|daemon>``:
            served, unaffined idle slots per posture (an SELinux one
            adds ``/selinux=<id>``).
        """
        # Taken before ``_counters_lock``: the acquire path takes
        # ``_lock`` then ``_counters_lock``, so the reverse order here
        # would be a lock-order inversion.
        gauges = {f"pool_idle_{name}": n
                  for name, n in self.posture_idle_counts().items()}
        with self._counters_lock:
            out = dict(self._counters)
        out.update(gauges)
        return out

    def shutdown_all(self) -> None:
        """Tear down the replenishment thread + every idle slot.

        Called once at daemon shutdown.  Idempotent: a second call
        after a successful first call finds an empty pool + stopped
        thread and returns.

        Order:
          1. Stop the replenishment thread (so it doesn't race-fork
             a new slot while we're closing the pool).
          2. Close each idle slot's socket — the slot's serve loop
             reads EOF and exits cleanly (pool PR 4 replaced PR 3's
             ``SHUTDOWN\\n`` command-loop with RPC-serve; an RPC
             slot exits on peer-EOF, not on a custom shutdown line).
          3. Reap each slot's PID via ``waitpid``.  Slots are template-
             children not daemon-children, so this can raise
             ``ChildProcessError`` (the init process already reaped
             them on SIGCHLD).  Treating that as success is fine for
             PR 4 — the subreaper fix is PR 5 work.

        Best-effort: any per-slot error logs but doesn't stop the
        rest of the teardown.
        """
        # 1. Stop the replenishment thread first.
        self.stop_replenishment()

        # 2. Drain idle slots.
        with self._lock:
            slots = list(self._idle_slots)
            self._idle_slots.clear()

        for slot in slots:
            try:
                slot.sock.close()
            except OSError:
                pass
            # 3. Reap.  Slots are template-children; ChildProcessError
            # is expected and benign.
            try:
                os.waitpid(slot.pid, 0)
            except ChildProcessError:
                pass
            # Then the profile, for the same reason the sweep does it
            # (#1033) and in the same order: the daemon owns the last
            # wearer, and a boundary-derived profile is not tied to any
            # one session that could have unloaded it earlier.
            self._reap_slot_profile(slot, reason="daemon-shutdown")

        if slots:
            logger.info(
                "PoolManager.shutdown_all: tore down %d idle slot(s)",
                len(slots),
            )

    # --------------------- runtime sizing ----------------------------

    def resize(
        self, target_size: int, max_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Change the pool's floor and ceiling on a running daemon.

        ``target_size`` and ``max_size`` were read once, at daemon
        startup, from ``JAATO_RUNNER_POOL_SIZE`` /
        ``JAATO_RUNNER_POOL_MAX_SIZE``, so adding warm runners meant a
        restart.  The replenish loop already reads both fields on every
        pass, so a resize is: set them, queue whatever no longer fits,
        and wake the loop.

        Growing: the loop forks the missing unreserved slots one per
        pass, with no pause between forks while it is short.

        Shrinking: only IDLE slots are touched -- an acquired slot is
        not in ``_idle_slots`` and finishes its session undisturbed.
        Virgin slots above the new floor go first (the floor counts
        virgins since #1507; at 0 every unaffined slot goes); then, if the
        pool is still over the new ceiling, the stalest reservations
        (the order :meth:`_pick_capacity_victim` uses).  Dropped slots
        are queued on ``_pending_teardown`` rather than torn down here,
        because ``_teardown_slot`` blocks on the daemon loop and the
        caller of this method usually IS the daemon loop.  The replenish
        thread drains the queue, and is started for that purpose if it
        was not running (a daemon booted with a disabled pool).

        The ceiling: an explicit ``max_size`` replaces it (clamped up to
        ``target_size``, as at startup).  When none is given, a ceiling
        that was derived (``2 * target_size``) follows the new floor and
        a ceiling somebody chose is kept, clamped up to the new floor.

        Args:
            target_size: New floor on unreserved idle slots.  0 disables
                the pool (sessions cold-spawn).
            max_size: New ceiling on total idle slots, or ``None``.

        Returns:
            ``{"previous": {...}, "current": {...}, "queued_for_teardown":
            n}``.  Both sizing dicts carry ``target_size`` and
            ``max_size``; ``current`` also carries ``max_size_explicit``.

        Raises:
            ValueError: A negative size, or a non-integer.
        """
        target = _non_negative_int("target_size", target_size)
        ceiling = (None if max_size is None
                   else _non_negative_int("max_size", max_size))
        with self._lock:
            previous = {"target_size": self.target_size,
                        "max_size": self.max_size}
            self.target_size = target
            if ceiling is not None:
                self.max_size = max(target, ceiling)
                self._max_size_explicit = True
            elif self._max_size_explicit:
                self.max_size = max(target, self.max_size)
            else:
                self.max_size = target * 2
            dropped = self._select_over_size_locked()
            for slot in dropped:
                slot.teardown_reason = "pool-resize"
                self._pending_teardown.append(slot)
            current = {"target_size": self.target_size,
                       "max_size": self.max_size,
                       "max_size_explicit": self._max_size_explicit}
        logger.info(
            "PoolManager.resize: target_size %d -> %d, max_size %d -> %d; "
            "%d idle slot(s) queued for teardown",
            previous["target_size"], current["target_size"],
            previous["max_size"], current["max_size"], len(dropped),
        )
        self._start_replenish_thread()
        self._replenish_wake.set()
        return {"previous": previous, "current": current,
                "queued_for_teardown": len(dropped)}

    def _select_over_size_locked(self) -> List[PoolSlot]:
        """Remove and return the idle slots the current sizes do not keep.

        Caller holds ``self._lock``.  First the virgin slots above
        ``target_size`` (newest first, so the longest-warm ones stay) --
        the floor counts virgins since #1507 -- then, while the pool is
        still above ``max_size``, the victims :meth:`_pick_capacity_victim`
        names: the stalest reservation, then the served slot of the least
        recently demanded posture, then a virgin.
        """
        dropped: List[PoolSlot] = []
        # At target 0 the pool is disabled, so a served slot is not kept
        # on top of a floor that no longer exists: every unaffined slot
        # goes, as before #1507.
        unreserved = [s for s in self._idle_slots
                      if s.cascade_id is None
                      and (not s.has_served or self.target_size == 0)]
        for slot in reversed(unreserved[self.target_size:]):
            self._idle_slots.remove(slot)
            dropped.append(slot)
        while len(self._idle_slots) > self.max_size:
            victim = self._pick_capacity_victim()
            if victim is None:
                break
            dropped.append(self._idle_slots.pop(victim))
        return dropped

    def snapshot(self) -> Dict[str, Any]:
        """The pool's sizing and state, for ``pool.status`` and logs.

        Keys: ``target_size``, ``max_size``, ``max_size_explicit``,
        ``idle`` (all idle slots), ``unreserved``, ``virgin`` (never
        served: what the floor counts, #1507), ``served`` (unaffined but
        served: fits one posture), ``reserved``,
        ``pending_teardown``, ``replenishing`` (the replenish thread is
        alive), ``template_alive`` and ``telemetry`` (the
        :meth:`get_telemetry` counters).
        """
        with self._lock:
            idle = len(self._idle_slots)
            unreserved = sum(1 for s in self._idle_slots if s.cascade_id is None)
            virgin = self._virgin_idle_count_locked()
            pending = len(self._pending_teardown)
            sizes = (self.target_size, self.max_size, self._max_size_explicit)
        thread = self._replenish_thread
        try:
            template_alive = bool(self._template_manager.is_alive())
        except Exception:  # noqa: BLE001 — a status read must not raise
            template_alive = False
        return {
            "target_size": sizes[0],
            "max_size": sizes[1],
            "max_size_explicit": sizes[2],
            "idle": idle,
            "unreserved": unreserved,
            "virgin": virgin,
            "served": unreserved - virgin,
            "reserved": idle - unreserved,
            "pending_teardown": pending,
            "replenishing": bool(thread is not None and thread.is_alive()),
            "template_alive": template_alive,
            "telemetry": self.get_telemetry(),
        }

    # --------------------- replenishment thread ----------------------

    def start_replenishment(self) -> None:
        """Start the background thread that keeps the pool topped up.

        Pool PR 4: the thread watches
        :meth:`virgin_idle_count` (#1507; :meth:`unreserved_idle_count`
        before it) against ``target_size`` and,
        whenever the count of slots ANY tenant could take drops below
        target, asks the template for a fresh fork-slot — subject to
        the ``max_size`` ceiling on total idle slots.  This is what makes
        the pool useful for cascades — a 6-step cascade with
        target_size=2 cold-spawns 4 of 6 steps otherwise; with
        replenishment a slot's gone-and-refilled cycle is small
        compared to the model-call time of each step, so cascade
        steps 3+ find a warm slot waiting.

        Idempotent: a second :meth:`start_replenishment` call after
        a successful first one is a no-op (logs a warning).

        Skipped when ``target_size <= 0`` (pool is disabled).
        """
        if self.target_size <= 0:
            logger.debug(
                "PoolManager.start_replenishment: target_size=%d; "
                "thread not started", self.target_size,
            )
            return
        if self._replenish_thread is not None and self._replenish_thread.is_alive():
            logger.warning(
                "PoolManager.start_replenishment: already running; ignoring",
            )
            return
        self._start_replenish_thread()

    def _start_replenish_thread(self) -> None:
        """Start the replenish thread, with no ``target_size`` gate.

        :meth:`start_replenishment` refuses a disabled pool (it is the
        startup path, and a daemon started with ``target_size=0`` asked
        for no pool).  :meth:`resize` needs the thread whatever the new
        target is: it is the only thread allowed to tear slots down
        (``_teardown_slot`` must run off the daemon loop), so a shrink to
        0 has to reach it too.  At target 0 the loop's capacity check is
        always satisfied, so a running thread forks nothing.
        """
        if self._replenish_thread is not None and self._replenish_thread.is_alive():
            return
        self._replenish_stop.clear()
        self._replenish_wake.clear()
        self._replenish_thread = threading.Thread(
            target=self._replenish_loop,
            name="jaato-pool-replenish",
            daemon=True,
        )
        self._replenish_thread.start()
        logger.info(
            "PoolManager: replenishment thread started "
            "(target_size=%d interval=%.2fs)",
            self.target_size, self._replenish_interval,
        )

    def stop_replenishment(self, timeout: float = 5.0) -> None:
        """Signal the replenishment thread to stop and wait for it.

        Idempotent: safe to call when no thread is running.
        """
        if self._replenish_thread is None:
            return
        self._replenish_stop.set()
        self._replenish_wake.set()
        self._replenish_thread.join(timeout=timeout)
        if self._replenish_thread.is_alive():
            logger.warning(
                "PoolManager.stop_replenishment: thread didn't join "
                "within %.1fs; leaking as daemon thread", timeout,
            )
        self._replenish_thread = None

    def _pause(self) -> None:
        """Sleep one replenish interval, cut short by a wake or a stop.

        Every pause in the loop goes through here, so :meth:`resize` and
        :meth:`stop_replenishment` (which set ``_replenish_wake``) end it
        at once.  The loop re-checks ``_replenish_stop`` at its top.
        """
        self._replenish_wake.wait(self._replenish_interval)
        self._replenish_wake.clear()

    def _replenish_loop(self) -> None:
        """Background loop body — wakes every ``_replenish_interval``,
        watchdogs the template + tops up the pool by ONE slot per
        iteration.

        One-slot-per-iteration matters: when the cascade rapidly
        drains the pool (multiple steps in a tight loop) the loop
        keeps refilling without blocking the daemon's main asyncio
        loop or holding the pool lock across a multi-slot batch.
        On a hot cascade the loop body fires effectively-continuously
        until the pool is back at target_size.

        Pool PR 5b watchdog (this iteration): detect template death
        + auto-respawn.  When ``template_manager.is_alive()`` flips
        False (template OOM-killed, segfaulted, or otherwise gone)
        the loop:

          1. Drains any idle slot handles — slots may have re-parented
             to the daemon via subreaper (PR 5b daemon startup bit);
             ``waitpid`` them so they don't zombie.
          2. Respawns the template via ``template_manager.spawn()``.
          3. Sleeps briefly to let the new template's plugin discovery
             complete before the next iteration tries fork-slot.
             (PR 5c will replace this sleep with a proper ready
             handshake.)

        Stops cleanly when ``_replenish_stop`` is set (daemon
        shutdown).
        """
        while not self._replenish_stop.is_set():
            try:
                # Template watchdog (PR 5b): detect death + respawn.
                # PR 5c: ``template_manager.spawn`` (called inside
                # _handle_template_death) blocks for the new
                # template's READY signal, so no explicit sleep
                # needed before the next iteration tries fork-slot.
                if not self._template_manager.is_alive():
                    self._handle_template_death()
                    continue

                # #1058: reap slots whose RPC client died in the pool.
                # BEFORE the cascade sweep, because that sweep is what
                # drains ``_pending_teardown`` — running it first means a
                # corpse is evicted and reaped in the SAME iteration, so
                # the capacity check below never reads it as capacity.
                self._sweep_dead_slots()

                # Phase 2: cascade-idle teardown sweep.  Reaps slots
                # that have sat IDLE_FOR_CASCADE longer than the
                # configured timeout — these are cascades that
                # finished (no further sessions arriving) or stalled
                # past the operator's tolerance.
                self._sweep_cascade_idle()

                # Cheap check: enough UNRESERVED slots, sleep.
                #
                # Unreserved, not total (#898).  ``idle_count()``
                # counts slots a waiting tenant is FORBIDDEN to use:
                # with two idle slots affined to cascade B and
                # target_size=2 the pool read "full" and never forked
                # another, while ``acquire_slot(cascade=A)`` returned
                # None.  Nothing to run on, and nothing in the system
                # that would ever create one, until B released.
                #
                # And VIRGIN, not unreserved (#1507): a slot that has
                # served fits only its own posture (#1100), so it is a
                # reservation keyed on the boundary.  Counting it here let
                # a daemon mixing unconfined and confined sessions read
                # "full" on slots the second posture could not take.
                if self.virgin_idle_count() >= self.target_size:
                    self._pause()
                    continue
                # ... but reservations still occupy memory, so the
                # total is what the ceiling bounds.  At the ceiling a
                # SERVED unaffined slot is spent for the virgin (liveness
                # outranks warmth: the virgin serves every posture);
                # reservations are not.  Hitting the ceiling with nothing
                # to spend is the signal that ``max_size`` is too small
                # for the number of concurrent tenants.
                if (self.idle_count() >= self.max_size
                        and not self._spend_served_slot_for_floor()):
                    self._incr("pool_replenish_ceiling_blocked_total")
                    self._pause()
                    continue
                raw = self._template_manager.request_fork_slot()
                if raw is None:
                    # request_fork_slot already logged the cause; back
                    # off briefly so we don't tight-loop on a flaky
                    # template.
                    self._incr("pool_replenish_failures_total")
                    self._pause()
                    continue
                pid, sock = raw
                new_slot = PoolSlot(pid=pid, sock=sock)
                with self._lock:
                    self._idle_slots.append(new_slot)
                self._incr("pool_replenish_success_total")
                logger.info(
                    "PoolManager replenish: forked slot pid=%d "
                    "(virgin=%d/%d, idle_count=%d/%d)",
                    new_slot.pid, self.virgin_idle_count(),
                    self.target_size, len(self._idle_slots), self.max_size,
                )
            except Exception:  # noqa: BLE001 — boundary surface
                logger.exception(
                    "PoolManager replenish: unhandled error; sleeping "
                    "and continuing",
                )
                self._pause()

    def _spend_served_slot_for_floor(self) -> bool:
        """Drop the least useful served slot to make room for a virgin.

        Called by the replenish loop when the virgin floor is short and
        the pool is at ``max_size`` (#1507).  Picks the served, unaffined
        slot whose posture was asked for least recently (then the one idle
        longest) and queues it for teardown; the caller forks the virgin.
        Reservations are never spent here -- that is #898's ceiling, and
        ``pool_replenish_ceiling_blocked_total`` still reports it.

        Returns ``True`` when a slot was queued.
        """
        with self._lock:
            served = self._served_unaffined_by_usefulness()
            if not served:
                return False
            victim = self._idle_slots.pop(served[0][0])
            victim.teardown_reason = "spent-for-virgin-floor"
            self._pending_teardown.append(victim)
        self._incr("pool_served_slot_evicted_total")
        logger.info(
            "PoolManager replenish: at the ceiling with the virgin floor "
            "short; slot pid=%d (served %s) queued for teardown to make "
            "room", victim.pid, posture_label(posture_of_slot(victim)),
        )
        return True

    def _teardown_slot(self, slot: PoolSlot, *, reason: str) -> None:
        """Close one idle slot's transport and reap its process.

        ONE COPY, called by every path that drops a slot.  It was the
        cascade-idle sweep's inline tail; the capacity check needs the
        identical sequence, and a second copy of a forty-line teardown
        is a second copy that rots.

        MUST RUN OFF THE DAEMON LOOP.  It closes the rpc via
        ``run_coroutine_threadsafe`` and BLOCKS on the future, so calling
        it from the loop thread would have the loop wait for itself.
        Both callers are the replenish thread, which is why the capacity
        check hands slots here instead of tearing them down on the
        session-return path -- that path runs wherever
        ``JaatoServer.shutdown`` runs, and a cap is not worth inheriting
        a thread-context assumption to enforce.
        """
        # Phase 3 hotfix (server 0.6.150+): close the slot's rpc
        # client before tearing down the socket.  rpc.close()
        # closes the writer + sigterm-ladder the runner pid;
        # doing it via the rpc rather than bare ``sock.close()``
        # cancels the read task + drains in-flight futures
        # cleanly.  Falls back to bare sock.close() if no rpc
        # was ever stashed (rare — slot has run zero sessions).
        rpc = slot.rpc
        if rpc is not None:
            # Best-effort close via the daemon loop; we run
            # close synchronously here because we're already
            # in the replenish thread, not the daemon loop.
            try:
                import asyncio as _asyncio
                loop = getattr(rpc, "_loop", None)
                if loop is not None and loop.is_running():
                    fut = _asyncio.run_coroutine_threadsafe(
                        rpc.close(timeout=5.0), loop,
                    )
                    fut.result(timeout=8.0)
                else:
                    # No loop running — fall back to bare
                    # socket close.  Runner gets EOF + dies.
                    slot.sock.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PoolManager cascade-idle sweep: rpc.close "
                    "raised for pid=%d: %s — falling back to "
                    "bare sock.close()", slot.pid, exc,
                )
                try:
                    slot.sock.close()
                except OSError:
                    pass
        else:
            try:
                slot.sock.close()
            except OSError:
                pass
        try:
            os.waitpid(slot.pid, 0)
        except ChildProcessError:
            pass
        # AFTER the process is gone, not before: unloading a profile
        # while a task is still confined to it is the thing this whole
        # change exists to avoid, and the slot is a task until waitpid
        # returns.
        self._reap_slot_profile(slot, reason=reason)

    def profile_in_use(self, profile_name: str) -> bool:
        """Is any idle slot still confined to *profile_name*? (#1033)

        For callers OUTSIDE the pool that are about to unload a profile —
        the WS workspace reaper is the one in this tree.  A
        boundary-derived profile outlives its session, so "no session is
        using it" is no longer sufficient grounds to unload it: a pooled
        slot may be sitting idle inside it, waiting for the next session
        of its cascade.

        Only IDLE slots, deliberately.  A checked-out slot is not in this
        list at all, and it does not need to be: such a slot always has a
        live session, and that is the case
        ``AppArmorManager.teardown_profile`` refuses on its own.  The two
        guards together cover every slot the pool knows about.
        """
        if not profile_name:
            return False
        with self._lock:
            return any(s.profile_name == profile_name
                       for s in self._idle_slots)

    def _reap_slot_profile(self, slot: PoolSlot, *, reason: str) -> None:
        """Unload the AppArmor profile this slot was the last to wear.

        **Profile teardown is per-SLOT now, not per-session** (#1033
        consequence 1).  It used to run right after the next session's
        ``aa_change_profile`` succeeded, keyed off
        ``PoolSlot.last_session_id`` — which was sound only while a
        profile belonged to exactly one session.  A boundary-derived
        profile OUTLIVES its session: the slot goes on wearing it, so
        unloading at session end would strip the kernel boundary off a
        live runner.  The thing that may reap it is the death of the
        last wearer.

        Two guards, and they cover different wearers:

        - here, another IDLE slot wearing the same profile.  Boundaries
          are shared by construction now, so a cascade that fans out has
          several slots inside one profile;
        - inside the reaper (``AppArmorManager.teardown_profile``), any
          live SESSION still claiming the id.  A checked-out slot is not
          in ``_idle_slots`` at all, and that is the guard that covers
          it.

        Best-effort in both directions: no reaper wired (an embedded or
        test PoolManager) and no profile on the slot are both ordinary,
        and a reaper that raises must not abort the teardown it is part
        of — a leaked kernel profile is a bounded, visible cost, while a
        half-torn-down slot is a leaked process.
        """
        reaper = self.profile_reaper
        profile_name = slot.profile_name
        if reaper is None or not profile_name:
            return
        with self._lock:
            still_worn = any(
                other is not slot and other.profile_name == profile_name
                for other in self._idle_slots
            )
        if still_worn:
            logger.debug(
                "PoolManager: slot pid=%d torn down (%s) but profile %s "
                "is still worn by another idle slot; leaving it loaded",
                slot.pid, reason, profile_name,
            )
            return
        try:
            reaper(profile_name)
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "PoolManager: profile reaper raised for %s while tearing "
                "down slot pid=%d (%s): %s — leaving the kernel profile "
                "loaded",
                profile_name, slot.pid, reason, exc,
            )

    def _drain_pending_teardown(self) -> None:
        """Tear down slots the capacity check dropped.

        Runs in the replenish thread, which is the context
        ``_teardown_slot`` requires.  Called from the sweep so both
        reclaim paths share one caller and one thread.
        """
        with self._lock:
            pending, self._pending_teardown = self._pending_teardown, []
        for slot in pending:
            reason = slot.teardown_reason or "over-capacity"
            self._teardown_slot(slot, reason=reason)
            self._incr("pool_slots_over_cap_torndown_total")
            logger.info(
                "PoolManager: torn down queued slot pid=%d "
                "(cascade=%s, reason=%s)",
                slot.pid, slot.cascade_id or "(pure)", reason,
            )

    def _sweep_cascade_idle(self) -> None:
        """Tear down IDLE_FOR_CASCADE slots whose idle window expired.

        Phase 2.  An IDLE_FOR_CASCADE(C) slot is one that returned
        from a session (``last_session_end_ts`` set) AND has a
        ``cascade_id`` set.  If the elapsed monotonic-time since
        the last session end exceeds ``cascade_idle_timeout_seconds``,
        the cascade is considered finished (or paused longer than
        the operator's tolerance).  We close the daemon-side socket
        (slot sees EOF, exits), waitpid it, and let the
        replenishment loop top the pool back up.

        Acquired (in-use) slots are NEVER in ``_idle_slots`` so this
        sweep cannot race with an active session.  PURE IDLE slots
        (``cascade_id is None``) are also exempt — they have no
        cascade affinity to time out.

        That exemption is about WARMTH and says nothing about health.
        A pure-idle slot whose runner has died is reaped by
        :meth:`_sweep_dead_slots`, which runs immediately before this
        one and exempts nobody (#1058).

        Called every replenish iteration (~0.5s).  Cheap when the
        pool has no IDLE_FOR_CASCADE entries.
        """
        self._drain_pending_teardown()

        now = time.monotonic()
        timeout = self._cascade_idle_timeout
        to_reap: List[PoolSlot] = []
        with self._lock:
            survivors: List[PoolSlot] = []
            for slot in self._idle_slots:
                if (
                    slot.cascade_id is not None
                    and slot.last_session_end_ts is not None
                    and (now - slot.last_session_end_ts) > timeout
                ):
                    to_reap.append(slot)
                else:
                    survivors.append(slot)
            if to_reap:
                self._idle_slots = survivors

        for slot in to_reap:
            self._teardown_slot(slot, reason="cascade-idle")
            self._incr("cascade_slots_idle_torndown_total")
            logger.info(
                "PoolManager cascade-idle sweep: torn down slot pid=%d "
                "cascade=%s (idle %.1fs > timeout %.0fs)",
                slot.pid, slot.cascade_id,
                now - (slot.last_session_end_ts or now), timeout,
            )

    def _handle_template_death(self) -> None:
        """Recover from template-subprocess death (pool PR 5b watchdog).

        Pool slots are template-children (forked, no exec).  When the
        template dies, slots get orphaned.  With the daemon's
        ``PR_SET_CHILD_SUBREAPER`` bit set (PR 5b startup), orphaned
        descendants re-parent to the daemon — so ``waitpid`` works.

        Recovery sequence:
          1. Drain idle slot handles — close their sockets (slot sees
             EOF, exits), waitpid them.  Acquired slots (handed out to
             active sessions) keep running independently; the
             session's bootstrap already completed and the slot serves
             RPC until session end.  Only IDLE slot handles in our
             possession need cleanup here.
          2. Respawn the template.  ``spawn`` blocks for the new
             template's READY signal (PR 5c) before returning, so
             the next replenish iteration can immediately fork-slot
             without a sleep.  On spawn failure, the outer loop's
             exception handler retries on the next iteration.

        Best-effort throughout — daemon shutdown signals via
        ``_replenish_stop`` are honored.
        """
        with self._lock:
            stale = list(self._idle_slots)
            self._idle_slots.clear()

        for slot in stale:
            try:
                slot.sock.close()
            except OSError:
                pass
            # Subreaper re-parented slot to daemon → waitpid works.
            # If subreaper wasn't set OR if init already reaped, we
            # get ChildProcessError — tolerate it.
            try:
                os.waitpid(slot.pid, 0)
            except ChildProcessError:
                pass

        if stale:
            logger.warning(
                "PoolManager watchdog: template died; reaped %d "
                "orphaned idle slot(s).  Acquired (in-use) slots "
                "keep running independently.", len(stale),
            )
        else:
            logger.warning(
                "PoolManager watchdog: template died; idle pool was "
                "empty so no orphaned slots to reap.",
            )

        # Respawn the template.  ``spawn`` is idempotent if the
        # previous spawn cleaned up properly (is_alive() returning
        # False clears self.pid via waitpid).
        self._incr("template_respawn_attempts_total")
        try:
            self._template_manager.spawn()
        except Exception as exc:  # noqa: BLE001 — boundary surface
            self._incr("template_respawn_failures_total")
            logger.error(
                "PoolManager watchdog: template respawn failed: %s; "
                "will retry on next iteration", exc,
            )
            return

        logger.info(
            "PoolManager watchdog: template respawned successfully — "
            "pool will refill on subsequent iterations",
        )
