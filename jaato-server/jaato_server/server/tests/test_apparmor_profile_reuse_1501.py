"""#1501 — a confined session reloaded a profile the kernel already had.

``provision_profile`` rendered, rewrote and ``apparmor_parser -r``-ed the
boundary profile for every confined session, and the last release unloaded
it, so sequential sessions on one workspace paid an unload plus a load of
byte-identical rules (+1.3-1.7 s per session on the reporting host).

What these tests pin:

* a reload is skipped only when the boundary has no reference fragments,
  the file on disk is byte-identical to the render, and the kernel lists
  the exact name in the mode the render asks for (#1014);
* anything that cannot be read LOADS (#1253 / #1299);
* the last release keeps a boundary loaded for a grace, a claim within it
  reuses it, the sweep unloads it after; grace ``0`` unloads at once;
* a boundary carrying session-added reference grants unloads at once, so
  they never reach the next session;
* a profile a live pool slot wears is never unloaded by the sweep.

No kernel: ``apparmor_parser`` and the securityfs listing are stubbed.  The
stub parser keeps the fake listing in step (``-r`` adds the name with the
mode its ``flags=`` line asks for, ``-R`` removes it), so "loaded" means
what it would on a host.
"""

from __future__ import annotations

import ast
import logging
import pathlib
import re
import sys
import time
from typing import List, Tuple

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from jaato_server.shared.tests.reversion import Reversion


_APPARMOR = "jaato-server/jaato_server/server/apparmor.py"

REVERSIONS = [
    Reversion(
        target=_APPARMOR,
        find="        if on_disk != profile_content.encode():\n",
        replace="        if False:\n",
        test="TestSkipReload::test_a_file_that_differs_from_the_render_reloads",
        because=("trusting the name's digest alone, so a profile file that "
                 "no longer matches the render is never reloaded"),
    ),
    Reversion(
        target=_APPARMOR,
        find="        if not known:\n            return False, \"securityfs unreadable\"\n",
        replace="        if not known:\n            mode = \"enforce\"\n",
        test="TestSkipReload::test_an_unreadable_securityfs_loads",
        because=("assuming the profile is loaded when the kernel could not "
                 "be asked — the fail-open #1253 forbids"),
    ),
    Reversion(
        target=_APPARMOR,
        find="        if mode != expected:\n",
        replace="        if False:\n",
        test="TestSkipReload::test_a_kernel_mode_other_than_the_render_reloads",
        because=("reusing a complain-mode profile for an enforce request "
                 "(or the reverse), so the record claims a mode the kernel "
                 "is not applying (#1014)"),
    ),
    Reversion(
        target=_APPARMOR,
        find="        if self._refs_present(render_id):\n            return False, \"reference fragments present\"\n",
        replace="",
        test="TestReferenceGrants::test_a_boundary_with_reference_grants_reloads",
        because=("skipping the reload while an include the render cannot "
                 "see carries rules, so the bytes compared say nothing"),
    ),
    Reversion(
        target=_APPARMOR,
        find=("        if (self._grace_seconds > 0 and confinement_id in self._boundary_ids\n"
              "                and profile_path.exists()\n"
              "                and not self._refs_present(confinement_id)):\n"),
        replace=("        if (self._grace_seconds > 0 and confinement_id in self._boundary_ids\n"
                 "                and profile_path.exists()):\n"),
        test="TestReferenceGrants::test_reference_grants_never_reach_the_next_session",
        because=("keeping a boundary loaded with a session's reference "
                 "grants, so the next session starts with grants its own "
                 "render does not give"),
    ),
    Reversion(
        target=_APPARMOR,
        find="        if (self._grace_seconds > 0 and confinement_id in self._boundary_ids\n",
        replace="        if (confinement_id in self._boundary_ids\n",
        test="TestGrace::test_grace_zero_unloads_at_once",
        because="grace 0 no longer meaning 'unload at once, as before'",
    ),
    Reversion(
        target=_APPARMOR,
        find="            return bool(check(self.profile_name_for_confinement_id(confinement_id)))\n",
        replace="            return False\n",
        test="TestGrace::test_a_profile_a_live_slot_wears_is_never_unloaded",
        because=("the grace sweep unloading the profile an idle pool slot "
                 "is still confined to"),
    ),
    Reversion(
        target=_APPARMOR,
        find="        with self._idle_lock:\n            self._idle_since.pop(render_id, None)\n        self._sweep_idle_impl()\n",
        replace="        self._sweep_idle_impl()\n",
        test="TestGrace::test_a_claim_within_the_grace_reuses_the_profile",
        because=("a session reusing an idle boundary without claiming it, "
                 "so it stays listed as idle while a session runs in it"),
    ),
    Reversion(
        target="jaato-server/jaato_server/server/session_manager.py",
        find="                self._sweep_idle_apparmor_profiles()\n",
        replace="                pass\n",
        test="TestWiring::test_the_watchdog_sweeps_idle_profiles",
        because=("an idle boundary nobody's thread ever expires, so the "
                 "grace silently becomes 'forever'"),
    ),
]


# ----------------------------------------------------------------------
# Fixtures
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
                mode = "complain" if "complain" in m.group(2) else "enforce"
                self.loaded[m.group(1)] = mode
        elif "-R" in cmd:
            self.loaded.pop(path.name, None)
        self._write()

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""
        return _R()

    def loads(self) -> List[str]:
        return [c[-1] for c in self.calls if "-r" in c]

    def unloads(self) -> List[str]:
        return [c[-1] for c in self.calls if "-R" in c]


@pytest.fixture
def kernel(tmp_path, monkeypatch):
    monkeypatch.delenv("JAATO_APPARMOR_COMPLAIN", raising=False)
    monkeypatch.delenv("JAATO_APPARMOR_PROFILE_GRACE_SECONDS", raising=False)
    k = _Kernel(tmp_path / "securityfs-profiles")
    monkeypatch.setattr("jaato_server.server.apparmor.subprocess.run", k.run)
    return k


def _manager(tmp_path, kernel):
    from jaato_server.server.apparmor import AppArmorManager

    profile_dir = tmp_path / "apparmor.d"
    profile_dir.mkdir(parents=True, exist_ok=True)
    mgr = AppArmorManager(workspace_root=str(tmp_path / "ws"),
                          profile_dir=str(profile_dir))
    mgr._available = True
    mgr._securityfs_profiles = kernel.listing
    mgr._securityfs_policy_dir = tmp_path / "no-policy-tree"
    return mgr


def _ws(tmp_path) -> str:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return str(ws)


def _provision(mgr, session_id: str, ws: str, **kw) -> str:
    """Provision through the confinement seam, as the daemon does."""
    from jaato_server.server.confinement import Boundary
    from jaato_server.server.confinement.apparmor import AppArmorBackend

    handle = AppArmorBackend(mgr).provision(
        session_id, Boundary(workspace_path=ws, **kw))
    assert handle is not None, "provisioning failed"
    return handle.label


# ----------------------------------------------------------------------
# Skip the reload
# ----------------------------------------------------------------------


class TestSkipReload:

    def test_a_second_session_on_a_loaded_boundary_runs_no_parser(
        self, tmp_path, kernel, caplog,
    ) -> None:
        mgr = _manager(tmp_path, kernel)
        ws = _ws(tmp_path)
        name = _provision(mgr, "s-a", ws)
        assert len(kernel.loads()) == 1

        with caplog.at_level(logging.INFO, logger="jaato_server.server.apparmor"):
            assert _provision(mgr, "s-b", ws) == name

        assert len(kernel.loads()) == 1, "an identical, loaded profile was reloaded"
        timing = [r.getMessage() for r in caplog.records
                  if "provision timings" in r.getMessage()]
        assert timing and "reload=skipped" in timing[-1]
        assert "render=" in timing[-1] and "parser=" not in timing[-1]
        assert mgr.loaded_mode("s-b") == "enforce"
        # #1326: the grant record still describes the loaded profile.
        from jaato_server.server.apparmor import recorded_grants
        assert recorded_grants(name)["profile_name"] == name

    def test_a_changed_render_loads_under_a_new_name(self, tmp_path, kernel) -> None:
        mgr = _manager(tmp_path, kernel)
        ws = _ws(tmp_path)
        first = _provision(mgr, "s-a", ws)
        second = _provision(mgr, "s-b", ws, plugin_rules=("/srv/models/** r,",))

        assert first != second
        assert [pathlib.Path(p).name for p in kernel.loads()] == [first, second]

    def test_a_file_that_differs_from_the_render_reloads(self, tmp_path, kernel) -> None:
        mgr = _manager(tmp_path, kernel)
        ws = _ws(tmp_path)
        name = _provision(mgr, "s-a", ws)
        path = mgr._profile_dir / name
        path.write_text(path.read_text() + "\n# edited\n")

        _provision(mgr, "s-b", ws)

        assert len(kernel.loads()) == 2

    def test_an_unreadable_securityfs_loads(self, tmp_path, kernel) -> None:
        mgr = _manager(tmp_path, kernel)
        ws = _ws(tmp_path)
        _provision(mgr, "s-a", ws)
        mgr._securityfs_profiles = tmp_path / "gone"

        _provision(mgr, "s-b", ws)

        assert len(kernel.loads()) == 2, "the reload was skipped on an unknown"

    def test_the_policy_tree_is_read_when_the_listing_is_not(
        self, tmp_path, kernel,
    ) -> None:
        mgr = _manager(tmp_path, kernel)
        ws = _ws(tmp_path)
        name = _provision(mgr, "s-a", ws)
        tree = tmp_path / "policy-profiles"
        entry = tree / f"{name}.3"
        entry.mkdir(parents=True)
        (entry / "name").write_text(name + "\n")
        (entry / "mode").write_text("enforce\n")
        mgr._securityfs_profiles = tmp_path / "gone"
        mgr._securityfs_policy_dir = tree

        _provision(mgr, "s-b", ws)

        assert len(kernel.loads()) == 1

    def test_a_profile_the_kernel_does_not_list_loads(self, tmp_path, kernel) -> None:
        mgr = _manager(tmp_path, kernel)
        ws = _ws(tmp_path)
        name = _provision(mgr, "s-a", ws)
        kernel.loaded.pop(name)
        kernel._write()

        _provision(mgr, "s-b", ws)

        assert len(kernel.loads()) == 2

    def test_a_kernel_mode_other_than_the_render_reloads(self, tmp_path, kernel) -> None:
        mgr = _manager(tmp_path, kernel)
        ws = _ws(tmp_path)
        name = _provision(mgr, "s-a", ws)
        kernel.loaded[name] = "complain"
        kernel._write()

        _provision(mgr, "s-b", ws)

        assert len(kernel.loads()) == 2
        assert kernel.loaded[name] == "enforce"

    def test_complain_and_enforce_never_share_a_profile(
        self, tmp_path, kernel, monkeypatch,
    ) -> None:
        mgr = _manager(tmp_path, kernel)
        ws = _ws(tmp_path)
        enforce = _provision(mgr, "s-a", ws)
        monkeypatch.setenv("JAATO_APPARMOR_COMPLAIN", "1")
        complain = _provision(mgr, "s-b", ws)
        monkeypatch.delenv("JAATO_APPARMOR_COMPLAIN")
        again = _provision(mgr, "s-c", ws)

        assert complain != enforce and again == enforce
        assert kernel.loaded[complain] == "complain"
        assert kernel.loaded[enforce] == "enforce"
        assert mgr.profile_is_complain_mode("s-b") is True
        assert mgr.profile_is_complain_mode("s-c") is False
        assert len(kernel.loads()) == 2  # s-c reused the enforce profile


# ----------------------------------------------------------------------
# Reference grants
# ----------------------------------------------------------------------


class TestReferenceGrants:

    def test_a_boundary_with_reference_grants_reloads(self, tmp_path, kernel) -> None:
        mgr = _manager(tmp_path, kernel)
        ws = _ws(tmp_path)
        _provision(mgr, "s-a", ws)
        ref = tmp_path / "corpus"
        ref.mkdir()
        assert mgr.add_reference_fragment("s-a", "ref-1", str(ref))
        loads_before = len(kernel.loads())

        _provision(mgr, "s-b", ws)

        assert len(kernel.loads()) == loads_before + 1

    def test_reference_grants_never_reach_the_next_session(
        self, tmp_path, kernel,
    ) -> None:
        mgr = _manager(tmp_path, kernel)
        ws = _ws(tmp_path)
        name = _provision(mgr, "s-a", ws)
        ref = tmp_path / "corpus"
        ref.mkdir()
        assert mgr.add_reference_fragment("s-a", "ref-1", str(ref))

        mgr.teardown_profile("s-a")

        assert name in [pathlib.Path(p).name for p in kernel.unloads()], (
            "a boundary carrying a session's reference grants was kept loaded")
        assert not mgr._refs_dir(mgr.confinement_id_of("s-a")).exists()
        _provision(mgr, "s-b", ws)
        assert not any(mgr._refs_dir(mgr.confinement_id_of("s-b")).glob("*"))

    def test_a_failed_reload_after_a_removal_forces_the_next_reload(
        self, tmp_path, kernel, monkeypatch,
    ) -> None:
        mgr = _manager(tmp_path, kernel)
        ws = _ws(tmp_path)
        _provision(mgr, "s-a", ws)
        ref = tmp_path / "corpus"
        ref.mkdir()
        assert mgr.add_reference_fragment("s-a", "ref-1", str(ref))
        monkeypatch.setattr(mgr, "_reload_profile", lambda p: (False, "boom"))
        assert mgr.remove_reference_fragment("s-a", "ref-1")
        loads_before = len(kernel.loads())

        _provision(mgr, "s-b", ws)

        assert len(kernel.loads()) == loads_before + 1


# ----------------------------------------------------------------------
# The grace
# ----------------------------------------------------------------------


class TestGrace:

    def test_the_last_release_keeps_the_profile_until_the_grace_expires(
        self, tmp_path, kernel,
    ) -> None:
        mgr = _manager(tmp_path, kernel)
        ws = _ws(tmp_path)
        name = _provision(mgr, "s-a", ws)

        mgr.teardown_profile("s-a")

        assert kernel.unloads() == []
        assert name in kernel.loaded
        assert mgr.sweep_idle_profiles(now=time.monotonic()) == []
        unloaded = mgr.sweep_idle_profiles(now=time.monotonic() + 61)
        assert [mgr.profile_name_for_confinement_id(c) for c in unloaded] == [name]
        assert name not in kernel.loaded
        assert not (mgr._profile_dir / name).exists()

    def test_a_claim_within_the_grace_reuses_the_profile(self, tmp_path, kernel) -> None:
        mgr = _manager(tmp_path, kernel)
        ws = _ws(tmp_path)
        name = _provision(mgr, "s-a", ws)
        mgr.teardown_profile("s-a")

        assert _provision(mgr, "s-b", ws) == name

        assert len(kernel.loads()) == 1
        assert kernel.unloads() == []
        assert mgr.idle_profiles() == {}

    def test_grace_zero_unloads_at_once(self, tmp_path, kernel, monkeypatch) -> None:
        monkeypatch.setenv("JAATO_APPARMOR_PROFILE_GRACE_SECONDS", "0")
        mgr = _manager(tmp_path, kernel)
        ws = _ws(tmp_path)
        name = _provision(mgr, "s-a", ws)

        mgr.teardown_profile("s-a")

        assert [pathlib.Path(p).name for p in kernel.unloads()] == [name]

    def test_a_profile_a_live_slot_wears_is_never_unloaded(
        self, tmp_path, kernel,
    ) -> None:
        mgr = _manager(tmp_path, kernel)
        ws = _ws(tmp_path)
        name = _provision(mgr, "s-a", ws)
        mgr.slot_in_use = lambda n: n == name
        mgr.teardown_profile("s-a")

        mgr.sweep_idle_profiles(now=time.monotonic() + 61)
        mgr.unload_idle_profiles()

        assert name in kernel.loaded
        assert kernel.unloads() == []

    def test_unparseable_grace_reads_as_the_default(self, monkeypatch) -> None:
        from jaato_server.server import apparmor

        for raw, expected in (("", 60.0), ("abc", 60.0), ("-5", 60.0),
                              ("0", 0.0), ("12.5", 12.5)):
            monkeypatch.setenv("JAATO_APPARMOR_PROFILE_GRACE_SECONDS", raw)
            assert apparmor.profile_grace_seconds() == expected


# ----------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------


class TestWiring:

    def _sm(self, mgr):
        from jaato_server.server.session_manager import SessionManager

        sm = SessionManager()
        sm._apparmor_manager = mgr
        return sm

    def test_the_watchdog_sweeps_idle_profiles(self, tmp_path, kernel, monkeypatch) -> None:
        mgr = _manager(tmp_path, kernel)
        name = _provision(mgr, "s-a", _ws(tmp_path))
        mgr.teardown_profile("s-a")
        for cid in list(mgr._idle_since):
            mgr._idle_since[cid] -= 1000
        sm = self._sm(mgr)
        # One pass of the loop body: the wait returns False once, then True.
        answers = iter([False, True])
        monkeypatch.setattr(sm._lifetime_watchdog_stop, "wait",
                            lambda _t: next(answers))
        for attr in ("sweep_session_lifetimes", "_maybe_sweep_retention",
                     "_sweep_app_secret_expiry", "_sweep_inbox"):
            monkeypatch.setattr(sm, attr, lambda *a, **k: None)

        sm._lifetime_watchdog_loop()

        assert name not in kernel.loaded

    def test_daemon_shutdown_unloads_idle_profiles(self, tmp_path, kernel) -> None:
        mgr = _manager(tmp_path, kernel)
        name = _provision(mgr, "s-a", _ws(tmp_path))
        mgr.teardown_profile("s-a")
        sm = self._sm(mgr)

        sm._unload_idle_apparmor_profiles()

        assert name not in kernel.loaded

    def test_the_session_manager_wires_the_slot_predicate(self, tmp_path, kernel) -> None:
        mgr = _manager(tmp_path, kernel)
        sm = self._sm(mgr)

        class _Pool:
            def profile_in_use(self, n):
                return n == "jaato-ws-x"

        sm._pool_manager_ref = _Pool()
        assert sm._apparmor_managers() == [mgr]
        assert mgr.slot_in_use("jaato-ws-x") is True
        assert mgr.slot_in_use("jaato-ws-y") is False

    def test_the_grace_env_var_is_catalogued(self) -> None:
        from jaato_server.shared.env_scope import CATALOG, HOST

        assert CATALOG["JAATO_APPARMOR_PROFILE_GRACE_SECONDS"].scope == HOST
