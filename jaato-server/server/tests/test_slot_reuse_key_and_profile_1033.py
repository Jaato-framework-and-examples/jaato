"""#1033 — the slot key said "reusable" and the profile name said "new".

A pre-warm pool slot is handed to session after session.  Its threads are
confined by ``aa_change_profile``, which is **per task**: the kernel
refuses to let one thread re-confine another (``current != task ->
-EACCES``), and #1026 can retire only the two RPC executor lanes.  So the
AppArmor profile a slot wears is, in practice, immutable for the life of
the slot.

The profile was named ``jaato-ws-{session_id}`` — the one property
guaranteed to DIFFER on every reuse — while the reuse key was
``(cascade_id, config_root)``, which says nothing about it.  Both
statements cannot be true at once, and on the maintainer's enforcing host
they were not: five consecutive session creations died with
``RunnerCallError``, each naming four of five threads still labelled with
the PRIOR session's profile (``OtelBatchSpanRe``, the reader/``Thread-N``
set).  Deterministic per slot — a fresh slot bootstrapped, a reused one
refused — which is exactly why it read as intermittent.

What these tests pin:

* the key contains the boundary (workspace + profile), so a reuse cannot
  cross one;
* the name is derived from that boundary, so a reuse never needs a
  transition at all;
* **the gate is on BOTH acquire paths.**  A standalone session (no
  cascade) returns its slot as PURE IDLE, profile and all, and the
  cascade-affinity key never covered that path — the brief for this fix
  named only path (1), and path (2) reproduces the same straddle for
  every unrelated pair of workspaces on one daemon;
* a profile now outlives its session, so nothing unloads it while a slot
  or a sibling session is still wearing it.

NO CLOCK.  Every case here is structural: a pool pre-seeded with the
slots it is meant to choose between, and an ``AppArmorManager`` whose
availability and ``apparmor_parser`` are stubbed so the naming can be
exercised on a host with no AppArmor LSM at all — which is what this
container is.  Nothing in this file proves the kernel behaves as #1023
describes; it proves the framework stops ASKING the kernel to do the
thing #1023 says it cannot.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from server.runner_pool import PoolManager, PoolSlot


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


#: Each entry puts one piece of the defect back.
REVERSIONS = [
    Reversion(
        target="jaato-server/server/runner_pool.py",
        find="                    if SlotKey.of_slot(slot) == key:",
        replace=("                    if (slot.cascade_id == "
                 "key.cascade_driver_id and _canonical_path("
                 "slot.config_root) == key.config_root):"),
        test=("TestReuseKey::"
              "test_a_cascade_does_not_reuse_a_slot_from_another_workspace"),
        because=("the reuse key ignoring the workspace again, so a slot "
                 "confined to workspace A is handed to a session in "
                 "workspace B and its threads straddle two profiles"),
    ),
    Reversion(
        target="jaato-server/server/runner_pool.py",
        find=("        return not slot.profile_name or "
              "slot.profile_name == self.profile_name"),
        replace="        return True",
        test=("TestReuseKey::"
              "test_a_pure_idle_slot_is_not_handed_across_profiles"),
        because=("the unaffined acquire path handing out a slot that is "
                 "already confined to somebody else's profile — the half "
                 "of the bug that has nothing to do with cascades"),
    ),
    Reversion(
        target="jaato-server/server/session_manager.py",
        find="            confinement_id=confinement_id,\n",
        replace="",
        test=("TestProfileNaming::"
              "test_one_boundary_gets_one_profile_name"),
        because=("the profile being named after the session again, so "
                 "every slot reuse needs an aa_change_profile that the "
                 "kernel cannot perform for threads that already exist"),
    ),
    Reversion(
        target="jaato-server/server/session_manager.py",
        find="        if prior_profile_name == current_profile_name:",
        replace="        if False:",
        test=("TestProfileLifetime::"
              "test_the_profile_a_slot_still_wears_is_not_unloaded"),
        because=("session-end teardown unloading the profile the slot is "
                 "still confined to, which is worse than the stale-profile "
                 "accumulation it was written to prevent"),
    ),
    Reversion(
        target="jaato-server/server/apparmor.py",
        find="        if confinement_id in self._confinement_ids.values():",
        replace="        if False:",
        test=("TestProfileLifetime::"
              "test_a_boundary_another_session_holds_survives_teardown"),
        because=("one session's exit stripping the kernel boundary off "
                 "every live sibling that shares it"),
    ),
    Reversion(
        target="jaato-server/server/runner_pool.py",
        find="        if still_worn:",
        replace="        if False:",
        test=("TestProfileLifetime::"
              "test_slot_teardown_spares_a_profile_another_slot_wears"),
        because=("slot teardown unloading a profile a sibling slot of the "
                 "same fan-out is still confined to"),
    ),
]


# ======================================================================
# Fixtures — a pool with no threads, a manager with no kernel
# ======================================================================


def _slot(
    pid: int,
    *,
    cascade_id: Optional[str] = None,
    config_root: Optional[str] = None,
    workspace_root: Optional[str] = None,
    profile_name: Optional[str] = None,
) -> PoolSlot:
    """An idle slot with the identity a previous session left on it."""
    return PoolSlot(
        pid=pid,
        sock=MagicMock(name=f"slot-{pid}-sock"),
        cascade_id=cascade_id,
        config_root=config_root,
        workspace_root=workspace_root,
        profile_name=profile_name,
    )


def _pool(*slots: PoolSlot) -> PoolManager:
    """A PoolManager holding *slots*, with no thread of any kind.

    The constructor starts nothing — ``spawn_initial_slots`` and
    ``start_replenishment`` are explicit calls these tests never make —
    so this is a pure data structure under test.
    """
    pool = PoolManager(template_manager=MagicMock(), target_size=8)
    pool._idle_slots = list(slots)
    return pool


def _manager(tmp_path, *, workspace: Optional[str] = None):
    """A real ``AppArmorManager`` that will never touch a kernel.

    ``_available`` is the cached answer ``is_available()`` returns, and
    ``subprocess.run`` is stubbed by the caller, so everything about
    NAMING and lifetime is exercisable on a host with no AppArmor — the
    only kind of host this change could be developed on.
    """
    from server.apparmor import AppArmorManager

    profile_dir = tmp_path / "apparmor.d"
    profile_dir.mkdir(parents=True, exist_ok=True)
    mgr = AppArmorManager(
        workspace_root=workspace or str(tmp_path / "ws"),
        profile_dir=str(profile_dir),
    )
    mgr._available = True
    return mgr


@pytest.fixture
def no_parser(monkeypatch):
    """Make every ``apparmor_parser`` invocation succeed, and record it."""
    calls: List[Tuple[str, ...]] = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def _run(cmd, *a, **kw):
        calls.append(tuple(cmd))
        return _Result()

    monkeypatch.setattr("server.apparmor.subprocess.run", _run)
    return calls


def _session_manager(mgr):
    """A ``SessionManager`` wired to *mgr* and to nothing else."""
    from server.session_manager import SessionManager

    sm = SessionManager()
    sm._apparmor_manager = mgr
    sm._emit_to_client = lambda cid, ev: None  # type: ignore[assignment]
    return sm


def _provision(sm, session_id: str, workspace: str, **kw) -> str:
    """Run the daemon's own provisioning path; return the profile name."""
    profile_name, _mode = sm._provision_apparmor_for_session(
        session_id=session_id,
        workspace_path=workspace,
        client_id="c-1",
        config_root=kw.pop("config_root", None),
        env_file=kw.pop("env_file", None),
        **kw,
    )
    return profile_name


# ======================================================================
# The key
# ======================================================================


class TestReuseKey:
    """What ``acquire_slot`` may and may not hand back."""

    def test_a_full_key_match_reuses_the_slot(self) -> None:
        """The warm path still works — this is not a fix by disabling."""
        warm = _slot(
            11, cascade_id="casc-1", config_root="/ws/.jaato",
            workspace_root="/ws", profile_name="jaato-ws-ws-aaaaaaaaaaaa",
        )
        pool = _pool(warm)

        got = pool.acquire_slot(
            cascade_driver_id="casc-1",
            config_root="/ws/.jaato",
            workspace_root="/ws",
            profile_name="jaato-ws-ws-aaaaaaaaaaaa",
        )

        assert got is warm
        assert pool.get_telemetry()["cascade_slot_reuse_hits_total"] == 1

    def test_a_cascade_does_not_reuse_a_slot_from_another_workspace(
        self,
    ) -> None:
        """Same cascade, same config root, DIFFERENT workspace.

        A cascade author may legitimately drive two workspaces from one
        driver.  The slot serving the first is confined to a profile that
        grants that workspace and no other, and it cannot be re-confined
        — so it must not be handed over.  #890 already declined to adopt
        its slot-scoped plugin instances across the same mismatch; what
        was missing is that the mismatch is also a KERNEL boundary.
        """
        warm = _slot(
            11, cascade_id="casc-1", config_root="/shared/.jaato",
            workspace_root="/ws-a", profile_name="jaato-ws-ws-a-aaaaaaaaaaaa",
        )
        pool = _pool(warm)

        got = pool.acquire_slot(
            cascade_driver_id="casc-1",
            config_root="/shared/.jaato",
            workspace_root="/ws-b",
            profile_name="jaato-ws-ws-b-bbbbbbbbbbbb",
        )

        assert got is None, (
            "a slot confined to /ws-a was handed to a session in /ws-b; "
            "its threads would straddle two profiles and #1023's "
            "verification would refuse the bootstrap"
        )
        assert pool.get_telemetry()["cascade_slot_reuse_hits_total"] == 0

    def test_the_shared_config_root_hole_is_closed(self) -> None:
        """The residual hole ``config_root`` alone could not cover.

        ``config_root`` defaults to ``<workspace>/.jaato`` and so usually
        moves with the workspace — but an operator may root config
        elsewhere deliberately ("to keep it out of the agent's filesystem
        tools", per CLAUDE.md).  Under THAT configuration the old
        two-field key matched across two different workspaces.  Same case
        as above, stated as the hole it closes.
        """
        warm = _slot(
            11, cascade_id="casc-1", config_root="/srv/operator/.jaato",
            workspace_root="/repos/alpha",
            profile_name="jaato-ws-alpha-aaaaaaaaaaaa",
        )
        pool = _pool(warm)

        assert pool.acquire_slot(
            cascade_driver_id="casc-1",
            config_root="/srv/operator/.jaato",
            workspace_root="/repos/beta",
            profile_name="jaato-ws-beta-bbbbbbbbbbbb",
        ) is None

    def test_a_pure_idle_slot_is_not_handed_across_profiles(self) -> None:
        """Acquire path (2) — the one the cascade key never covered.

        A standalone session carries no ``cascade_driver_id``, so its
        slot returns to the pool as PURE IDLE and the next arrival of any
        kind may take it.  It is still confined to the profile it served
        under.  Gating only the cascade path would leave every
        standalone reuse producing the divergence #1023 refuses.
        """
        used = _slot(
            11, workspace_root="/ws-a",
            profile_name="jaato-ws-ws-a-aaaaaaaaaaaa",
        )
        pool = _pool(used)

        assert pool.acquire_slot(
            workspace_root="/ws-b",
            profile_name="jaato-ws-ws-b-bbbbbbbbbbbb",
        ) is None
        assert pool.get_telemetry()["pool_profile_mismatch_skips_total"] == 1

    def test_a_never_confined_slot_fits_anybody(self) -> None:
        """A fresh template fork wears no profile, so it is not fussy.

        This is what keeps the pool a pool: the ``target_size`` floor is
        stocked with slots that have never served, and those must remain
        takeable by any arriving session.
        """
        fresh = _slot(11)
        pool = _pool(fresh)

        got = pool.acquire_slot(
            workspace_root="/ws-b",
            profile_name="jaato-ws-ws-b-bbbbbbbbbbbb",
        )

        assert got is fresh
        # ...and it is stamped, so the NEXT arrival is measured against
        # the boundary this one is about to impose on it.
        assert got.profile_name == "jaato-ws-ws-b-bbbbbbbbbbbb"
        assert got.workspace_root == "/ws-b"

    def test_no_apparmor_changes_nothing_about_the_profile_gate(
        self,
    ) -> None:
        """Every profile name is empty, so the gate is a tautology.

        The workspace field still splits the key — that is #890's
        argument and it is deliberate — but nothing here refuses a reuse
        on confinement grounds when there is no confinement.
        """
        used = _slot(11, workspace_root="/ws", profile_name=None)
        pool = _pool(used)

        got = pool.acquire_slot(workspace_root="/ws", profile_name=None)

        assert got is used

    def test_the_key_names_every_property_the_next_session_cannot_change(
        self,
    ) -> None:
        """The rule, stated as a test rather than only as a comment.

        A field added to ``PoolSlot`` that a session cannot change has to
        be in the key; this pins the four that are, so dropping one is a
        deliberate edit here rather than a silent omission there.
        """
        from server.runner_pool import SlotKey

        assert set(SlotKey.__dataclass_fields__) == {
            "cascade_driver_id",
            "config_root",
            "workspace_root",
            "profile_name",
        }

    def test_paths_are_compared_canonically(self, tmp_path) -> None:
        """Two spellings of one workspace are one workspace.

        Without this a trailing slash or a symlinked checkout would read
        as a different tenant and quietly cost every reuse in the
        deployment.
        """
        real = tmp_path / "ws"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)

        warm = _slot(11, cascade_id="c", workspace_root=str(real))
        pool = _pool(warm)

        assert pool.acquire_slot(
            cascade_driver_id="c", workspace_root=str(link),
        ) is warm


# ======================================================================
# The name
# ======================================================================


class TestProfileNaming:
    """Where ``jaato-ws-<...>`` comes from now."""

    def test_one_boundary_gets_one_profile_name(
        self, tmp_path, no_parser,
    ) -> None:
        """Two sessions, one workspace, one config root -> ONE profile.

        This is the whole fix in one assertion, and it is asserted
        through the daemon's own provisioning path rather than through a
        naming helper, because the defect was that the path did not use
        one.  With the profile named after the session, these two names
        differ and every reuse of the slot between them needs a
        transition the kernel cannot perform on its existing threads.
        """
        ws = str(tmp_path / "ws")
        mgr = _manager(tmp_path, workspace=ws)
        sm = _session_manager(mgr)

        first = _provision(sm, "20260913_171825", ws)
        second = _provision(sm, "20260913_193206", ws)

        assert first and first.startswith("jaato-ws-")
        assert first == second, (
            "two sessions on one boundary got two profile names; a pool "
            "slot serving both would have to change profile between them"
        )
        assert "20260913_171825" not in first

    def test_a_different_workspace_gets_a_different_profile(
        self, tmp_path, no_parser,
    ) -> None:
        """Sharing a name across boundaries would be the opposite bug."""
        mgr = _manager(tmp_path)
        sm = _session_manager(mgr)

        a = _provision(sm, "s-a", str(tmp_path / "ws-a"))
        b = _provision(sm, "s-b", str(tmp_path / "ws-b"))

        assert a != b

    def test_a_different_config_root_gets_a_different_profile(
        self, tmp_path, no_parser,
    ) -> None:
        """The config root is granted BY the profile, so it is part of it."""
        ws = str(tmp_path / "ws")
        mgr = _manager(tmp_path, workspace=ws)
        sm = _session_manager(mgr)

        a = _provision(sm, "s-a", ws, config_root=str(tmp_path / "cfg-a"))
        b = _provision(sm, "s-b", ws, config_root=str(tmp_path / "cfg-b"))

        assert a != b

    def test_different_rules_get_different_profiles(
        self, tmp_path, no_parser,
    ) -> None:
        """Two stages of one cascade whose plugins grant different paths.

        They share workspace and config root and they are NOT the same
        boundary.  Giving them one name would mean provisioning the
        second reloads the name the first is confined to — a silent
        widening (or narrowing) of a live session's boundary, which is a
        worse failure than the one being fixed.  Two names means two
        slots, which is the correct cost.
        """
        ws = str(tmp_path / "ws")
        mgr = _manager(tmp_path, workspace=ws)
        sm = _session_manager(mgr)

        narrow = _provision(sm, "s-a", ws, plugin_rules=None)
        broad = _provision(sm, "s-b", ws,
                           plugin_rules=["/srv/models/** r,"])

        assert narrow != broad

    def test_the_name_stays_readable(self, tmp_path, no_parser) -> None:
        """An operator greps this string; it is not allowed to be a uuid.

        Session identity is not lost with it — #812's ``runner_identity``
        records which runner ran which session — but the profile still
        has to be recognisable in ``dmesg`` and in
        ``/proc/<pid>/attr/current``.
        """
        ws = str(tmp_path / "my-repo")
        mgr = _manager(tmp_path, workspace=ws)
        sm = _session_manager(mgr)

        name = _provision(sm, "s-a", ws)

        assert name.startswith("jaato-ws-my-repo-")

    def test_the_id_is_stable_across_manager_instances(
        self, tmp_path, no_parser,
    ) -> None:
        """A daemon restart must land on the same name for the same
        boundary, or the first session after a restart reintroduces the
        straddle against slots that outlived nothing but still wear it.
        """
        ws = str(tmp_path / "ws")
        first = _manager(tmp_path, workspace=ws).confinement_id_for_boundary(ws)
        second = _manager(tmp_path, workspace=ws).confinement_id_for_boundary(ws)

        assert first == second


# ======================================================================
# The lifetime
# ======================================================================


class TestProfileLifetime:
    """A profile now outlives the session that created it."""

    def _profile_path(self, mgr, name: str):
        return mgr._profile_dir / name

    def test_a_boundary_another_session_holds_survives_teardown(
        self, tmp_path, no_parser,
    ) -> None:
        """Two live sessions, one boundary; one of them ends.

        Sharing a profile name is what makes a slot reusable, and it is
        also what makes the old per-session unload dangerous: session A
        ending must not take the kernel boundary away from session B.
        """
        ws = str(tmp_path / "ws")
        mgr = _manager(tmp_path, workspace=ws)
        sm = _session_manager(mgr)

        name = _provision(sm, "s-a", ws)
        assert _provision(sm, "s-b", ws) == name
        assert self._profile_path(mgr, name).exists()

        mgr.teardown_profile("s-a")

        assert self._profile_path(mgr, name).exists(), (
            "session A's exit unloaded the profile session B is confined to"
        )

        mgr.teardown_profile("s-b")
        assert not self._profile_path(mgr, name).exists(), (
            "the last holder released it and it was still left loaded"
        )

    def test_the_profile_a_slot_still_wears_is_not_unloaded(
        self, tmp_path, no_parser,
    ) -> None:
        """The post-transition teardown, against a slot that did not
        transition.

        ``_teardown_prior_apparmor_profile_after_transition`` fires right
        after a reused slot's bootstrap and used to unload
        ``slot.last_session_id``'s profile unconditionally.  With the
        name derived from the boundary, "the prior session's profile" and
        "the profile this runner is confined to right now" are the same
        profile.
        """
        ws = str(tmp_path / "ws")
        mgr = _manager(tmp_path, workspace=ws)
        sm = _session_manager(mgr)

        name = _provision(sm, "s-prev", ws)
        current = _provision(sm, "s-next", ws)
        assert current == name

        slot = _slot(11, workspace_root=ws, profile_name=name)
        slot.last_session_id = "s-prev"
        server = MagicMock()
        server._spawned_runner.pool_slot = slot

        # Asserted on the DECISION, not only on the file.  The manager
        # has its own shared-boundary guard, so a file check alone would
        # pass whichever of the two layers is doing the work — and this
        # test is about the one that decides not to ask.
        asked: List[str] = []
        mgr.teardown_profile = asked.append  # type: ignore[assignment]

        sm._teardown_prior_apparmor_profile_after_transition(
            server=server,
            current_session_id="s-next",
            current_profile_name=current,
        )

        assert asked == [], (
            "teardown was requested for a session whose profile is the "
            "one the slot is still confined to"
        )

    def test_slot_teardown_reaps_the_profile_it_last_wore(self) -> None:
        """Per-SLOT, because that is who the last wearer is.

        The reaper is what replaces the per-session unload: a
        boundary-derived profile is released when the last slot wearing
        it dies, not when a session ends.
        """
        pool = _pool()
        reaped: List[str] = []
        pool.profile_reaper = reaped.append

        dying = _slot(11, cascade_id="c", profile_name="jaato-ws-ws-aaaa")
        pool._teardown_slot(dying, reason="cascade-idle")

        assert reaped == ["jaato-ws-ws-aaaa"]

    def test_slot_teardown_spares_a_profile_another_slot_wears(self) -> None:
        """A fan-out puts several slots inside one boundary."""
        sibling = _slot(12, cascade_id="c", profile_name="jaato-ws-ws-aaaa")
        pool = _pool(sibling)
        reaped: List[str] = []
        pool.profile_reaper = reaped.append

        dying = _slot(11, cascade_id="c", profile_name="jaato-ws-ws-aaaa")
        pool._teardown_slot(dying, reason="cascade-idle")

        assert reaped == [], (
            "a profile a sibling slot is still confined to was unloaded"
        )

    def test_the_profile_outlives_the_process_that_wears_it(
        self, monkeypatch,
    ) -> None:
        """Order, not just outcome: reap the process, THEN the profile.

        Unloading a profile while a task is still confined to it is the
        thing this whole change exists to avoid, and the slot is a task
        until ``waitpid`` returns.
        """
        pool = _pool()
        order: List[str] = []
        pool.profile_reaper = lambda name: order.append("profile")
        monkeypatch.setattr(
            "server.runner_pool.os.waitpid",
            lambda *a, **kw: order.append("waitpid"),
        )

        pool._teardown_slot(
            _slot(11, profile_name="jaato-ws-ws-aaaa"), reason="cascade-idle",
        )

        assert order == ["waitpid", "profile"]

    def test_the_pool_answers_whether_a_profile_is_still_worn(self) -> None:
        """The question an outside unloader has to ask now.

        The WS workspace reaper unloads by session id on an hourly
        sweep.  "No session is using it" stopped being sufficient
        grounds the moment a profile could outlive its session, so the
        reaper consults this before unloading.
        """
        pool = _pool(_slot(11, profile_name="jaato-ws-ws-aaaa"))

        assert pool.profile_in_use("jaato-ws-ws-aaaa")
        assert not pool.profile_in_use("jaato-ws-other-bbbb")
        # An unconfined slot claims nothing, so neither does the empty
        # name every session on a host without AppArmor carries.
        assert not pool.profile_in_use("")

    def test_an_unconfined_slot_reaps_nothing(self) -> None:
        """Hosts without AppArmor tear slots down exactly as before."""
        pool = _pool()
        reaped: List[str] = []
        pool.profile_reaper = reaped.append

        pool._teardown_slot(_slot(11), reason="over-capacity")

        assert reaped == []

    def test_a_raising_reaper_does_not_abort_the_teardown(self) -> None:
        """A leaked kernel profile is bounded and visible; a leaked
        process is neither.
        """
        pool = _pool()

        def _boom(_name: str) -> None:
            raise RuntimeError("apparmor_parser is not here")

        pool.profile_reaper = _boom
        slot = _slot(11, profile_name="jaato-ws-ws-aaaa")

        pool._teardown_slot(slot, reason="cascade-idle")

        slot.sock.close.assert_called()
