"""#1506 — IPC sessions never released their AppArmor boundary profile.

``SessionManager._do_session_unload`` had no AppArmor release; the only
daemon callers of ``teardown_profile*`` were the cascade slot transition,
the dead-slot reaper and the WS workspace reaper.  ``_idle_since`` is
written only by ``_teardown_profile_impl``, so #1501's grace never started
for an IPC session, and profiles stayed loaded and on disk after the last
session and after daemon stop (148 files on one host).

What these tests pin:

* every session-end path (unload — which the #812 orphan stop,
  ``session.stop`` and the #1106 grace expiry reach — ``delete_session``
  and daemon stop) releases through ``teardown_profile``, AFTER
  ``server.shutdown()``, on the manager that provisioned the session;
* the release decrements, never unloads outright: another live session or
  an idle pool slot wearing the boundary keeps it loaded (#1033), and the
  slot's death releases it;
* with grace 0, two sequential sessions on one boundary reload;
* daemon stop leaves nothing loaded that it held;
* the startup reconcile removes an orphan (a dead owner's, or a legacy
  file of this uid nobody wears) and keeps a profile another live daemon
  owns, one a task wears, one of another uid, and one this daemon holds.

No kernel for most: ``apparmor_parser`` and securityfs are stubbed, as in
``test_apparmor_profile_reuse_1501.py``.  ``TestOnAnEnforcingKernel`` runs
only as root on a host whose AppArmor is enforcing.
"""

from __future__ import annotations

import ast
import logging
import os
import pathlib
import re
import shutil
import sys
import time
from typing import List, Tuple

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from jaato_server.shared.tests.reversion import Reversion


_APPARMOR = "jaato-server/jaato_server/server/apparmor.py"
_RECLAIM = "jaato-server/jaato_server/server/apparmor_reclaim.py"
_SM = "jaato-server/jaato_server/server/session_manager.py"
_MAIN = "jaato-server/jaato_server/server/__main__.py"

REVERSIONS = [
    Reversion(
        target=_SM,
        find=("        # After the shutdown: its slot is back in the pool by now (#1506).\n"
              "        self._release_apparmor_boundary(session_id)\n"),
        replace="",
        test="TestSessionEnd::test_an_unload_releases_the_boundary",
        because="the #1506 defect itself: an IPC unload releasing nothing",
    ),
    Reversion(
        target=_SM,
        find=("            session.server.shutdown()\n"
              "        self._release_apparmor_boundary(session_id)\n"),
        replace="            session.server.shutdown()\n",
        test="TestSessionEnd::test_delete_releases_the_boundary",
        because="a deleted session's boundary held for the daemon's lifetime",
    ),
    Reversion(
        target=_SM,
        find="        self._stop_apparmor_grace()\n",
        replace="",
        test="TestSessionEnd::test_daemon_stop_leaves_nothing_loaded",
        because=("a boundary released on the way out entering a grace no "
                 "watchdog will ever expire"),
    ),
    Reversion(
        target=_SM,
        find="                if holds is None or not holds(session_id):\n",
        replace="                if holds is None:\n",
        test="TestSessionEnd::test_only_the_manager_that_provisioned_is_asked",
        because="releasing on a manager that never provisioned the session",
    ),
    Reversion(
        target=_APPARMOR,
        find=("        if self._slot_holds(confinement_id):\n"
              "            logger.info(\n"
              "                \"AppArmor: leaving profile %s loaded — session %s released \"\n"),
        replace=("        if False:\n"
                 "            logger.info(\n"
                 "                \"AppArmor: leaving profile %s loaded — session %s released \"\n"),
        test="TestSlots::test_a_session_end_never_unloads_a_profile_a_slot_wears",
        because=("a session end with grace 0 stripping the profile off a pool "
                 "slot that returned to the pool still wearing it (#1033)"),
    ),
    Reversion(
        target=_SM,
        find=("        if any(cid in getattr(m, \"_confinement_ids\", {}).values()\n"
              "               for m in managers):\n"
              "            return\n"),
        replace="",
        test="TestSlots::test_a_dead_slot_keeps_a_boundary_a_ws_session_holds",
        because=("the slot reaper unloading through the IPC manager a "
                 "boundary a WS-manager session still holds"),
    ),
    Reversion(
        target=_RECLAIM,
        find=("        if owner_alive(ledger_owner, proc_root):\n"
              "            return KEEP_OWNER_ALIVE\n"),
        replace="",
        test="TestStartupReconcile::test_another_live_daemons_profile_is_kept",
        because="a starting daemon removing a profile another live daemon uses",
    ),
    Reversion(
        target=_RECLAIM,
        find=("    if kernel_name_for_file(profile_path.name) in worn:\n"
              "        return KEEP_WORN\n"),
        replace="",
        test="TestStartupReconcile::test_a_worn_profile_is_kept",
        because="unloading a profile a running task is confined to",
    ),
    Reversion(
        target=_MAIN,
        find="        await self._reconcile_apparmor_profiles()\n",
        replace="",
        test="TestStartupReconcile::test_the_daemon_reconciles_before_the_pool_forks",
        because="a reconcile nothing calls, so orphans survive every restart",
    ),
]


# ----------------------------------------------------------------------
# Fixtures (the stub kernel of test_apparmor_profile_reuse_1501.py)
# ----------------------------------------------------------------------

_FLAGS = re.compile(r"^profile (\S+) flags=\(([^)]*)\)", re.M)


class _Kernel:
    """A fake ``apparmor_parser`` plus the securityfs listing it keeps."""

    def __init__(self, listing: pathlib.Path) -> None:
        self.listing = listing
        self.calls: List[Tuple[str, ...]] = []
        self.loaded: dict = {}
        self._write()

    def _write(self) -> None:
        self.listing.write_text("".join(
            f"{name} ({mode})\n" for name, mode in sorted(self.loaded.items())))

    def run(self, cmd, *a, **kw):
        self.calls.append(tuple(cmd))
        path = pathlib.Path(cmd[-1])
        if "-r" in cmd:
            m = _FLAGS.search(path.read_text())
            if m:
                self.loaded[m.group(1)] = (
                    "complain" if "complain" in m.group(2) else "enforce")
        elif "-R" in cmd:
            self.loaded.pop(path.name, None)
        self._write()

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""
        return _R()

    def loads(self) -> List[str]:
        return [pathlib.Path(c[-1]).name for c in self.calls if "-r" in c]

    def unloads(self) -> List[str]:
        return [pathlib.Path(c[-1]).name for c in self.calls if "-R" in c]


@pytest.fixture
def kernel(tmp_path, monkeypatch):
    monkeypatch.delenv("JAATO_APPARMOR_COMPLAIN", raising=False)
    monkeypatch.delenv("JAATO_APPARMOR_PROFILE_GRACE_SECONDS", raising=False)
    k = _Kernel(tmp_path / "securityfs-profiles")
    monkeypatch.setattr("jaato_server.server.apparmor.subprocess.run", k.run)
    return k


def _manager(tmp_path, kernel, name: str = "apparmor.d"):
    from jaato_server.server.apparmor import AppArmorManager

    profile_dir = tmp_path / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    mgr = AppArmorManager(workspace_root=str(tmp_path / "ws"),
                          profile_dir=str(profile_dir))
    mgr._available = True
    mgr._securityfs_profiles = kernel.listing
    mgr._securityfs_policy_dir = tmp_path / "no-policy-tree"
    mgr._owner_ledger_path = tmp_path / "ledger.json"
    return mgr


def _ws(tmp_path) -> str:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return str(ws)


def _provision(mgr, session_id: str, ws: str) -> str:
    from jaato_server.server.confinement import Boundary
    from jaato_server.server.confinement.apparmor import AppArmorBackend

    handle = AppArmorBackend(mgr).provision(session_id, Boundary(workspace_path=ws))
    assert handle is not None, "provisioning failed"
    return handle.label


class _Server:
    """What ``_do_session_unload`` reads off a JaatoServer, and an order log."""

    def __init__(self, log: List[str]) -> None:
        self._model_running = False
        self._log = log

    def shutdown(self) -> None:
        self._log.append("shutdown")

    def _find_plugin_for_command(self, _name):
        return None


def _sm(*managers):
    from jaato_server.server.session_manager import SessionManager

    sm = SessionManager()
    sm._apparmor_manager = managers[0] if managers else None
    if len(managers) > 1:
        class _WS:
            pass
        ws = _WS()
        ws._apparmor = managers[1]
        sm._ws_server_ref = ws
    return sm


def _load(sm, session_id: str, log: List[str]) -> None:
    from jaato_server.server.session_manager import Session

    sm._sessions[session_id] = Session(
        session_id=session_id, name=session_id, server=_Server(log),
        created_at="2026-10-05T00:00:00+00:00")


def _spy(mgr, log: List[str]) -> None:
    real = mgr.teardown_profile

    def teardown(sid):
        log.append(f"release:{sid}")
        return real(sid)
    mgr.teardown_profile = teardown


# ----------------------------------------------------------------------
# Every session end releases
# ----------------------------------------------------------------------


class TestSessionEnd:

    def test_an_unload_releases_the_boundary(self, tmp_path, kernel, monkeypatch) -> None:
        monkeypatch.setenv("JAATO_APPARMOR_PROFILE_GRACE_SECONDS", "0")
        mgr = _manager(tmp_path, kernel)
        name = _provision(mgr, "s-a", _ws(tmp_path))
        sm = _sm(mgr)
        log: List[str] = []
        _spy(mgr, log)
        _load(sm, "s-a", log)

        sm._do_session_unload("s-a")

        assert log == ["shutdown", "release:s-a"], (
            "the release must come after server.shutdown(), when a pooled "
            "slot is already back in the pool")
        assert kernel.unloads() == [name]
        assert not (mgr._profile_dir / name).exists()

    def test_with_a_grace_the_unload_starts_it(self, tmp_path, kernel) -> None:
        mgr = _manager(tmp_path, kernel)
        name = _provision(mgr, "s-a", _ws(tmp_path))
        sm = _sm(mgr)
        _load(sm, "s-a", [])

        sm._do_session_unload("s-a")

        assert list(mgr.idle_profiles()) == [mgr.confinement_id_of(name[len("jaato-ws-"):])]
        assert kernel.unloads() == []
        assert mgr.sweep_idle_profiles(now=time.monotonic() + 61)
        assert name not in kernel.loaded

    def test_grace_zero_two_sequential_sessions_reload(
        self, tmp_path, kernel, monkeypatch, caplog,
    ) -> None:
        monkeypatch.setenv("JAATO_APPARMOR_PROFILE_GRACE_SECONDS", "0")
        mgr = _manager(tmp_path, kernel)
        ws = _ws(tmp_path)
        sm = _sm(mgr)
        name = _provision(mgr, "s-a", ws)
        _load(sm, "s-a", [])
        sm._do_session_unload("s-a")

        with caplog.at_level(logging.INFO, logger="jaato_server.server.apparmor"):
            assert _provision(mgr, "s-b", ws) == name

        assert kernel.loads() == [name, name]
        timing = [r.getMessage() for r in caplog.records
                  if "provision timings" in r.getMessage()]
        assert timing and "reload=ran" in timing[-1]

    def test_another_live_session_keeps_the_boundary(self, tmp_path, kernel, monkeypatch) -> None:
        monkeypatch.setenv("JAATO_APPARMOR_PROFILE_GRACE_SECONDS", "0")
        mgr = _manager(tmp_path, kernel)
        ws = _ws(tmp_path)
        name = _provision(mgr, "s-a", ws)
        _provision(mgr, "s-b", ws)
        sm = _sm(mgr)
        _load(sm, "s-a", [])

        sm._do_session_unload("s-a")

        assert name in kernel.loaded and kernel.unloads() == []

    def test_delete_releases_the_boundary(self, tmp_path, kernel, monkeypatch) -> None:
        monkeypatch.setenv("JAATO_APPARMOR_PROFILE_GRACE_SECONDS", "0")
        mgr = _manager(tmp_path, kernel)
        name = _provision(mgr, "s-a", _ws(tmp_path))
        sm = _sm(mgr)
        log: List[str] = []
        _spy(mgr, log)
        _load(sm, "s-a", log)

        sm.delete_session("s-a")

        assert log[:2] == ["shutdown", "release:s-a"]
        assert name not in kernel.loaded

    def test_daemon_stop_leaves_nothing_loaded(self, tmp_path, kernel) -> None:
        # Default grace (60 s).  The session's slot is back in the pool, so
        # the session release keeps the profile; the pool's teardown, which
        # runs AFTER the session manager's, then reaps the slot.  Without
        # the grace switched off that release would enter a grace no
        # watchdog will ever expire.
        mgr = _manager(tmp_path, kernel)
        name = _provision(mgr, "s-a", _ws(tmp_path))
        sm = _sm(mgr)
        saved: List[str] = []
        sm._save_session = lambda s: saved.append(s.session_id)
        worn = {name}

        class _Pool:
            def profile_in_use(self, n):
                return n in worn

        sm._pool_manager_ref = _Pool()
        mgr.slot_in_use = sm._apparmor_slot_in_use
        _load(sm, "s-a", [])

        sm.shutdown()
        assert saved == ["s-a"]
        assert name in kernel.loaded, "a slot still wears it"

        worn.clear()                      # PoolManager.shutdown_all
        sm._reap_apparmor_profile_for_dead_slot(name)

        assert name not in kernel.loaded
        assert not (mgr._profile_dir / name).exists()
        assert mgr.idle_profiles() == {}

    def test_only_the_manager_that_provisioned_is_asked(self, tmp_path, kernel) -> None:
        ipc = _manager(tmp_path, kernel, "ipc.d")
        ws_mgr = _manager(tmp_path, kernel, "ws.d")
        _provision(ws_mgr, "s-a", _ws(tmp_path))
        sm = _sm(ipc, ws_mgr)
        ipc_log: List[str] = []
        ws_log: List[str] = []
        _spy(ipc, ipc_log)
        _spy(ws_mgr, ws_log)

        sm._release_apparmor_boundary("s-a")

        assert ipc_log == [] and ws_log == ["release:s-a"]


# ----------------------------------------------------------------------
# Pool slots own their own release (#1033)
# ----------------------------------------------------------------------


class TestSlots:

    def test_a_session_end_never_unloads_a_profile_a_slot_wears(
        self, tmp_path, kernel, monkeypatch,
    ) -> None:
        monkeypatch.setenv("JAATO_APPARMOR_PROFILE_GRACE_SECONDS", "0")
        mgr = _manager(tmp_path, kernel)
        name = _provision(mgr, "s-a", _ws(tmp_path))
        sm = _sm(mgr)
        worn = {name}

        class _Pool:
            def profile_in_use(self, n):
                return n in worn

        sm._pool_manager_ref = _Pool()
        mgr.slot_in_use = sm._apparmor_slot_in_use
        _load(sm, "s-a", [])

        sm._do_session_unload("s-a")
        assert name in kernel.loaded, "stripped off a slot back in the pool"

        # The last slot wearing it dies: the pool's reaper releases it.
        worn.clear()
        sm._reap_apparmor_profile_for_dead_slot(name)
        assert name not in kernel.loaded

    def test_a_dead_slot_keeps_a_boundary_a_ws_session_holds(
        self, tmp_path, kernel, monkeypatch,
    ) -> None:
        monkeypatch.setenv("JAATO_APPARMOR_PROFILE_GRACE_SECONDS", "0")
        ipc = _manager(tmp_path, kernel)
        ws_mgr = _manager(tmp_path, kernel)  # same profile dir, as on a daemon
        name = _provision(ws_mgr, "s-ws", _ws(tmp_path))
        # The IPC manager provisioned this boundary once too, so it is the
        # one the reaper releases on; it has no session of its own on it.
        ipc._boundary_ids.add(name[len("jaato-ws-"):])
        sm = _sm(ipc, ws_mgr)

        sm._reap_apparmor_profile_for_dead_slot(name)

        assert name in kernel.loaded


# ----------------------------------------------------------------------
# Startup reconcile
# ----------------------------------------------------------------------


def _stat(pid: int, start: str) -> str:
    # Fields 3..22 after ``)``; field 22 is the start time.
    return f"{pid} (py thon) S " + " ".join(["0"] * 18) + f" {start} 0 0\n"


def _proc(tmp_path, *, pids=(), worn=()):
    root = tmp_path / "proc"
    root.mkdir(exist_ok=True)
    for pid, start in pids:
        (root / str(pid)).mkdir(exist_ok=True)
        (root / str(pid) / "stat").write_text(_stat(pid, start))
    for i, label in enumerate(worn):
        task = root / str(9000 + i) / "task" / str(9000 + i) / "attr"
        task.mkdir(parents=True)
        (task / "current").write_text(label + "\n")
    return root


def _write_profile(mgr, name: str) -> pathlib.Path:
    path = mgr._profile_dir / name
    path.write_text(f"profile {name} flags=(attach_disconnected) {{\n}}\n")
    return path


class TestStartupReconcile:

    def _setup(self, tmp_path, kernel, *, pids=(), worn=()):
        mgr = _manager(tmp_path, kernel)
        mgr._proc_root = _proc(tmp_path, pids=[(os.getpid(), "100"), *pids],
                               worn=worn)
        return mgr

    def test_a_dead_owners_profile_is_reclaimed(self, tmp_path, kernel) -> None:
        mgr = self._setup(tmp_path, kernel)
        path = _write_profile(mgr, "jaato-ws-orphan")
        kernel.loaded["jaato-ws-orphan"] = "enforce"
        mgr._ledger().record("jaato-ws-orphan", "424242:5")
        (mgr._profile_dir / "jaato-ws-orphan.refs.d").mkdir()

        assert mgr.reclaim_orphaned_profiles() == ["jaato-ws-orphan"]
        assert not path.exists()
        assert "jaato-ws-orphan" not in kernel.loaded
        assert not (mgr._profile_dir / "jaato-ws-orphan.refs.d").exists()
        assert "jaato-ws-orphan" not in mgr._ledger().entries()

    def test_a_legacy_unledgered_file_of_this_uid_is_reclaimed(self, tmp_path, kernel) -> None:
        # The VPS's 148: written before the ledger existed.
        mgr = self._setup(tmp_path, kernel)
        _write_profile(mgr, "jaato-ws-legacy-1")
        _write_profile(mgr, "jaato-ws-legacy-1__sub_iso")

        reclaimed = mgr.reclaim_orphaned_profiles()

        assert sorted(reclaimed) == ["jaato-ws-legacy-1", "jaato-ws-legacy-1__sub_iso"]
        assert list(mgr._profile_dir.iterdir()) == []

    def test_another_live_daemons_profile_is_kept(self, tmp_path, kernel) -> None:
        mgr = self._setup(tmp_path, kernel, pids=[(777, "55")])
        path = _write_profile(mgr, "jaato-ws-theirs")
        mgr._ledger().record("jaato-ws-theirs", "777:55")

        assert mgr.reclaim_orphaned_profiles() == []
        assert path.exists()

    def test_a_recycled_pid_does_not_keep_a_profile(self, tmp_path, kernel) -> None:
        mgr = self._setup(tmp_path, kernel, pids=[(777, "99")])
        _write_profile(mgr, "jaato-ws-recycled")
        mgr._ledger().record("jaato-ws-recycled", "777:55")

        assert mgr.reclaim_orphaned_profiles() == ["jaato-ws-recycled"]

    def test_a_worn_profile_is_kept(self, tmp_path, kernel) -> None:
        mgr = self._setup(tmp_path, kernel,
                          worn=["jaato-ws-busy//child (enforce)"])
        path = _write_profile(mgr, "jaato-ws-busy")

        assert mgr.reclaim_orphaned_profiles() == []
        assert path.exists()

    def test_a_profile_this_daemon_holds_is_kept(self, tmp_path, kernel) -> None:
        mgr = self._setup(tmp_path, kernel)
        name = _provision(mgr, "s-a", _ws(tmp_path))   # this daemon's, ledgered
        mgr.slot_in_use = lambda n: n == "jaato-ws-slot"
        _write_profile(mgr, "jaato-ws-slot")

        assert mgr.reclaim_orphaned_profiles() == []
        assert (mgr._profile_dir / name).exists()

    def test_a_file_of_another_uid_is_kept(self, tmp_path) -> None:
        from jaato_server.server.apparmor_reclaim import (
            KEEP_FOREIGN_UID, RECLAIM, reclaim_verdict,
        )

        path = tmp_path / "jaato-ws-foreign"
        path.write_text("x")
        mine = path.stat().st_uid
        kw = dict(ledger_owner=None, this_owner="1:1", worn=set(),
                  proc_root=tmp_path)
        assert reclaim_verdict(path, euid=mine + 1, **kw) == KEEP_FOREIGN_UID
        assert reclaim_verdict(path, euid=mine, **kw) == RECLAIM

    def test_the_reconcile_says_what_it_reclaimed(self, tmp_path, kernel, caplog) -> None:
        mgr = self._setup(tmp_path, kernel)
        _write_profile(mgr, "jaato-ws-old")

        with caplog.at_level(logging.INFO, logger="jaato_server.server.apparmor"):
            mgr.reclaim_orphaned_profiles()

        line = [r.getMessage() for r in caplog.records
                if "startup reconcile" in r.getMessage()]
        assert line and "reclaimed 1" in line[0] and "jaato-ws-old" in line[0]

    def test_the_daemon_reconciles_before_the_pool_forks(self) -> None:
        src = (pathlib.Path(__file__).resolve().parents[1] / "__main__.py").read_text()
        start = next(n for n in ast.walk(ast.parse(src))
                     if isinstance(n, ast.AsyncFunctionDef) and n.name == "start")
        lines = {}
        for node in ast.walk(start):
            if isinstance(node, ast.Attribute) and node.attr in (
                    "_reconcile_apparmor_profiles", "spawn_initial_slots"):
                lines.setdefault(node.attr, node.lineno)
        assert "_reconcile_apparmor_profiles" in lines, "nothing calls the reconcile"
        assert lines["_reconcile_apparmor_profiles"] < lines["spawn_initial_slots"]


# ----------------------------------------------------------------------
# The issue's kernel-gated checks
# ----------------------------------------------------------------------


def _enforcing_apparmor() -> bool:
    try:
        enabled = pathlib.Path("/sys/module/apparmor/parameters/enabled").read_text()
    except OSError:
        return False
    return (enabled.strip() == "Y" and os.geteuid() == 0
            and shutil.which("apparmor_parser") is not None)


@pytest.mark.skipif(not _enforcing_apparmor(),
                    reason="needs root and an enforcing AppArmor kernel")
class TestOnAnEnforcingKernel:

    def _real(self, tmp_path):
        from jaato_server.server.apparmor import AppArmorManager

        profile_dir = tmp_path / "apparmor.d"
        profile_dir.mkdir()
        mgr = AppArmorManager(workspace_root=str(tmp_path / "ws"),
                              profile_dir=str(profile_dir))
        mgr._owner_ledger_path = tmp_path / "ledger.json"
        assert mgr.is_available()
        return mgr

    def test_grace_zero_second_session_reloads_and_nothing_survives_stop(
        self, tmp_path, monkeypatch, caplog,
    ) -> None:
        monkeypatch.setenv("JAATO_APPARMOR_PROFILE_GRACE_SECONDS", "0")
        mgr = self._real(tmp_path)
        ws = _ws(tmp_path)
        sm = _sm(mgr)
        sm._save_session = lambda s: None
        name = _provision(mgr, "s-a", ws)
        _load(sm, "s-a", [])
        sm._do_session_unload("s-a")
        assert name not in mgr._securityfs_profiles.read_text()

        with caplog.at_level(logging.INFO, logger="jaato_server.server.apparmor"):
            _provision(mgr, "s-b", ws)
        assert any("reload=ran" in r.getMessage() for r in caplog.records)

        _load(sm, "s-b", [])
        sm.shutdown()
        assert name not in mgr._securityfs_profiles.read_text()
        assert not (mgr._profile_dir / name).exists()

    def test_with_a_grace_the_profile_goes_after_it(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("JAATO_APPARMOR_PROFILE_GRACE_SECONDS", "1")
        mgr = self._real(tmp_path)
        sm = _sm(mgr)
        name = _provision(mgr, "s-a", _ws(tmp_path))
        _load(sm, "s-a", [])
        sm._do_session_unload("s-a")
        assert name in mgr._securityfs_profiles.read_text()

        mgr.sweep_idle_profiles(now=time.monotonic() + 2)
        assert name not in mgr._securityfs_profiles.read_text()
