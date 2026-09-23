"""Which process is running this session — recorded where an operator looks.

A daemon-side session is executed by a *runner subprocess*, either cold-spawned
for it or claimed from the pre-warm pool.  The daemon has always known which
one: :func:`server.runner_spawn.spawn_session_runner` builds a ``SpawnedRunner``
carrying the pid and (for a pool-served session) the ``PoolSlot`` it came from,
and hands it to ``JaatoServer.set_runner_rpc``.  It then went nowhere an
outside observer could read.

Issue #812 measured the cost.  A session whose client had died kept executing
tools and spending money, and the reporter — who could SEE the session in
``~/.jaato/session_workspace_index.json`` and in its own record on disk — could
not act on it:

* the workspace index mapped the id to a workspace and recorded no runner, pid
  or slot;
* the session JSON had no runner-, slot- or pid-shaped key;
* the per-session logs named only an IPC connection number (``client_ipc_14``);
* no runner process had the workspace as ``cwd`` or held an fd in it.

So the only options were killing a circumstantially-identified process on a
daemon shared with another live session, or waiting for the budget to burn.
Being able to see a session and not being able to act on it is the defect this
module closes.

**Reusable, not eval-specific.**  #806 needs the same fact for a different
question — which runner holds which language server — so the record is a
general "who is executing this session", written once at spawn and read by
anything that needs to correlate a session id with a process.

Lifecycle
---------

======================================  ==========================================
When                                    What happens
======================================  ==========================================
``_spawn_session_runner_unconditional`` :func:`identity_from_server` reads the
                                        ``SpawnedRunner`` the spawn helper left on
                                        the ``JaatoServer`` and stamps
                                        ``Session.runner_identity``.
every ``_save_session``                 serialised into
                                        ``SessionState.runner_identity`` (record
                                        version **2.10**) and into the workspace
                                        index's ``identity`` section.
``_load_session``                       the persisted record is restored as the
                                        LAST KNOWN identity with ``stale=True``;
                                        a re-spawn immediately overwrites it with
                                        a live one.
======================================  ==========================================

A stale record is deliberately kept rather than cleared: for a session that is
no longer loaded, "the last process that ran this" is exactly what a
post-mortem wants, and ``stale`` is what stops a reader mistaking it for a live
pid.  Nothing in the framework acts on a stale identity — it is evidence, not a
handle.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

#: Bumped alongside the session-record version when a field is added, so a
#: reader can tell a partially-populated old record from a complete new one.
RUNNER_IDENTITY_SCHEMA = 1


@dataclass(frozen=True)
class RunnerIdentity:
    """The process (and pool slot) executing one session.

    Immutable: a session that changes runner — a revive, or a pool slot handed
    on to the next cascade stage — gets a NEW instance rather than a mutated
    one, so a reader that captured the record cannot have it change underneath.

    Attributes:
        runner_pid: The runner subprocess pid.  For a pool-served session this
            is the slot's pid (the slot IS the runner), so the two fields agree
            and ``pool_served`` is what distinguishes the case.  ``None`` when
            the session runs in-process with no runner at all.
        pool_served: True when the runner came from the pre-warm pool rather
            than being cold-spawned for this session.  Load-bearing for an
            operator: killing a pool-served runner destroys a slot other
            sessions of the same cascade were going to reuse, so the two are
            not interchangeable targets.
        pool_slot_pid: The ``PoolSlot`` pid when pool-served, else ``None``.
            Recorded separately from ``runner_pid`` because they are distinct
            concepts that happen to be equal today; #806 correlates slots, not
            runners.
        cascade_driver_id: The cascade this session belongs to, when it has
            one.  A slot is affined to a cascade, so this is what says whose
            warm state a kill would destroy.
        apparmor_profile: The AppArmor profile the runner confined itself to,
            or ``""`` when unconfined.  The kernel-visible name, so an operator
            can find the process by profile as well as by pid.
        recorded_at: Unix timestamp of the stamp.  Distinguishes a record
            written this minute from one left by a previous daemon.
        stale: True when this was restored from disk rather than stamped at a
            live spawn — the pid named may belong to nothing, or to something
            else entirely.  NEVER act on a stale record.
    """

    runner_pid: Optional[int] = None
    pool_served: bool = False
    pool_slot_pid: Optional[int] = None
    cascade_driver_id: Optional[str] = None
    apparmor_profile: str = ""
    recorded_at: float = 0.0
    stale: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for the session record and the workspace index.

        Returns:
            A JSON-compatible dict carrying every field plus ``schema``.
        """
        data = asdict(self)
        data["schema"] = RUNNER_IDENTITY_SCHEMA
        return data

    @classmethod
    def from_dict(
        cls, data: Optional[Dict[str, Any]], *, stale: bool = False,
    ) -> Optional["RunnerIdentity"]:
        """Rebuild from a persisted dict, tolerating anything unexpected.

        Every field is read with a default, and an unknown key (a record
        written by a newer server) is ignored rather than raising — this sits
        on the session LOAD path, and a diagnostic field must never be able to
        stop a session from reviving.

        Args:
            data: The persisted dict, or ``None`` / a non-dict for a record
                that predates the field.
            stale: Force the restored record to declare itself stale.  The
                load path passes ``True``: the pid it names is from a previous
                process lifetime.

        Returns:
            The identity, or ``None`` when there was nothing to rebuild.
        """
        if not isinstance(data, dict):
            return None
        pid = data.get("runner_pid")
        slot_pid = data.get("pool_slot_pid")
        return cls(
            runner_pid=pid if isinstance(pid, int) else None,
            pool_served=bool(data.get("pool_served", False)),
            pool_slot_pid=slot_pid if isinstance(slot_pid, int) else None,
            cascade_driver_id=data.get("cascade_driver_id") or None,
            apparmor_profile=str(data.get("apparmor_profile") or ""),
            recorded_at=float(data.get("recorded_at") or 0.0),
            stale=bool(stale or data.get("stale", False)),
        )

    def describe(self) -> str:
        """One line naming the process, for a log or an operator listing.

        Returns:
            e.g. ``"runner_pid=41237 pool_slot_pid=41237 cascade=cid-7"``, or
            ``"runner_pid=41237 (STALE - from a previous daemon)"``.
        """
        parts = [f"runner_pid={self.runner_pid}"]
        if self.pool_served:
            parts.append(f"pool_slot_pid={self.pool_slot_pid}")
        else:
            parts.append("pool_served=False")
        if self.cascade_driver_id:
            parts.append(f"cascade={self.cascade_driver_id}")
        if self.apparmor_profile:
            parts.append(f"apparmor={self.apparmor_profile}")
        if self.stale:
            parts.append("(STALE - from a previous daemon)")
        return " ".join(parts)


def identity_from_server(
    server: Any,
    *,
    cascade_driver_id: Optional[str] = None,
) -> Optional[RunnerIdentity]:
    """Read the live runner identity off a ``JaatoServer`` after its spawn.

    ``spawn_session_runner`` ends with ``server.set_runner_rpc(rpc, spawned)``,
    so by the time the caller returns the ``SpawnedRunner`` is reachable as
    ``server._spawned_runner`` — and, when the session was pool-served, its
    ``pool_slot`` attribute carries the ``PoolSlot``.  This function is the one
    reader of that shape, so the attribute names live in exactly one place.

    Defensive by design: it runs on the session-creation path, where a stand-in
    ``JaatoServer`` (tests, harnesses) may carry none of these attributes, and
    a diagnostic must never break a spawn.  Anything missing yields ``None``
    rather than an exception.

    Args:
        server: The session's ``JaatoServer``, after ``spawn_session_runner``.
        cascade_driver_id: The cascade the session was created under, when
            known.  Falls back to the slot's own ``cascade_id``.

    Returns:
        A live (non-stale) :class:`RunnerIdentity`, or ``None`` when no runner
        pid could be read — which means the session has no runner subprocess
        to identify, not that identification failed.
    """
    spawned = getattr(server, "_spawned_runner", None)
    if spawned is None:
        return None
    pid = getattr(spawned, "pid", None)
    if not isinstance(pid, int):
        return None
    slot = getattr(spawned, "pool_slot", None)
    slot_pid = getattr(slot, "pid", None) if slot is not None else None
    return RunnerIdentity(
        runner_pid=pid,
        pool_served=slot is not None,
        pool_slot_pid=slot_pid if isinstance(slot_pid, int) else None,
        cascade_driver_id=(
            cascade_driver_id
            or (getattr(slot, "cascade_id", None) if slot is not None else None)
        ),
        apparmor_profile=str(getattr(spawned, "profile_name", "") or ""),
        recorded_at=time.time(),
        stale=False,
    )
