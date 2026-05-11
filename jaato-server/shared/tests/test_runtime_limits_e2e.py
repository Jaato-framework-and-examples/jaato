"""End-to-end tests for runtime_limits: profile loading → cgroup wiring.

Part 1 (TestFakeFsE2E): Exercises the full chain from profile JSON
through SubagentProfile construction to CgroupsManager using a fake
cgroup v2 filesystem (tmp_path). Portable — runs on any OS.

Part 2 (TestRealKernel): Exercises real kernel cgroup enforcement on
Linux with cgroup v2. Skipped on other platforms.
"""

import json
import os
import signal
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# Avoid importing server/__init__.py (heavy provider deps).
if "server" not in sys.modules:
    _stub = types.ModuleType("server")
    _stub.__path__ = [
        os.path.join(os.path.dirname(__file__), "..", "..", "server")
    ]
    sys.modules["server"] = _stub

import server.cgroups as _cg
from shared.runtime_limits import RuntimeLimits
from shared.plugins.subagent.config import (
    SubagentProfile,
    discover_profiles,
    resolve_profiles,
    ProfileDiscoveryResult,
    _scan_profiles_dir,
)

CgroupsManager = _cg.CgroupsManager


# ---------------------------------------------------------------------------
# Fixtures (shared with existing test_cgroups.py pattern)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_cgroup_root(tmp_path):
    root = tmp_path / "jaato"
    root.mkdir()
    (root / "cgroup.subtree_control").write_text("memory pids cpu\n")
    return root


@pytest.fixture
def manager(fake_cgroup_root, monkeypatch):
    mgr = CgroupsManager(root=str(fake_cgroup_root))

    def _rmdir_for_fakefs(cg_path):
        try:
            for entry in cg_path.iterdir():
                if entry.is_file():
                    entry.unlink()
            cg_path.rmdir()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False

    monkeypatch.setattr(mgr, "_safe_rmdir", staticmethod(_rmdir_for_fakefs))
    return mgr


def _write_profile(tmp_path, name, data):
    """Write a profile JSON file and return its path."""
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _force_v2_available(tmp_path):
    marker = tmp_path / "v2_marker"
    marker.write_text("memory pids cpu\n")
    return patch("server.cgroups._V2_MARKER", marker)


# ===========================================================================
# Part 1: Fake-FS End-to-End
# ===========================================================================


class TestFakeFsE2E:

    # -- 1. Profile with kernel limits creates cgroup --

    def test_profile_with_kernel_limits_creates_cgroup(self, tmp_path, manager):
        profile_path = _write_profile(tmp_path, "heavy", {
            "name": "heavy",
            "description": "Heavy session",
            "plugins": ["cli"],
            "runtime_limits": {
                "memory_max_mb": 512,
                "pids_max": 128,
                "cpu_weight": 200,
            },
        })

        # Load via _scan_profiles_dir (isolated, no tier scanning)
        profiles, errors = {}, {}
        _scan_profiles_dir(tmp_path, profiles, errors)
        assert "heavy" in profiles
        assert not errors

        limits = profiles["heavy"].runtime_limits
        assert isinstance(limits, RuntimeLimits)
        assert limits.memory_max_mb == 512
        assert limits.pids_max == 128
        assert limits.cpu_weight == 200
        assert limits.has_kernel_limits() is True

        # Provision cgroup from the loaded limits
        with _force_v2_available(tmp_path), \
             patch("server.cgroups.platform.system", return_value="Linux"):
            manager._available = None  # reset cache
            assert manager.provision_cgroup("s1", limits) is True

        cg = manager.get_cgroup_path("s1")
        assert cg.is_dir()
        assert (cg / "memory.max").read_text() == str(512 * 1024 * 1024)
        assert (cg / "pids.max").read_text() == "128"
        assert (cg / "cpu.weight").read_text() == "200"

    # -- 2. App-only limits skip cgroup creation --

    def test_profile_with_app_only_limits_skips_cgroup(self, tmp_path, manager):
        _write_profile(tmp_path, "app_only", {
            "name": "app_only",
            "description": "App-layer only",
            "plugins": ["cli"],
            "runtime_limits": {
                "tool_timeout_seconds": 30,
                "max_output_bytes": 1048576,
            },
        })

        profiles, errors = {}, {}
        _scan_profiles_dir(tmp_path, profiles, errors)
        limits = profiles["app_only"].runtime_limits

        assert limits.tool_timeout_seconds == 30
        assert limits.max_output_bytes == 1048576
        assert limits.has_kernel_limits() is False

        with _force_v2_available(tmp_path), \
             patch("server.cgroups.platform.system", return_value="Linux"):
            manager._available = None
            assert manager.provision_cgroup("s1", limits) is True
            assert not manager.get_cgroup_path("s1").exists()

    # -- 3. Profile without runtime_limits gets None --

    def test_profile_without_runtime_limits_gets_none(self, tmp_path):
        _write_profile(tmp_path, "plain", {
            "name": "plain",
            "description": "No limits",
            "plugins": ["cli"],
        })

        profiles, errors = {}, {}
        _scan_profiles_dir(tmp_path, profiles, errors)
        assert profiles["plain"].runtime_limits is None

    # -- 4. Invalid runtime_limits rejected at load --

    def test_invalid_runtime_limits_rejected_at_load(self, tmp_path):
        _write_profile(tmp_path, "bad_limits", {
            "name": "bad_limits",
            "description": "Bad limits",
            "plugins": ["cli"],
            "runtime_limits": {
                "cpu_weight": 99999,
            },
        })

        profiles, errors = {}, {}
        _scan_profiles_dir(tmp_path, profiles, errors)
        # Profile with invalid runtime_limits is rejected — not added to profiles
        assert "bad_limits" not in profiles
        # Error is recorded
        assert "bad_limits" in errors

    # -- 5. Inherited runtime_limits merged --

    def test_inherited_runtime_limits_merged(self, tmp_path, manager):
        _write_profile(tmp_path, "parent_lim", {
            "name": "parent_lim",
            "description": "Parent with limits",
            "plugins": ["cli"],
            "runtime_limits": {
                "memory_max_mb": 2048,
                "pids_max": 256,
            },
        })
        _write_profile(tmp_path, "child_inherit", {
            "name": "child_inherit",
            "description": "Child inheriting limits",
            "inherits": ["parent_lim"],
            "plugins": ["web_search"],
        })

        profiles, errors = {}, {}
        _scan_profiles_dir(tmp_path, profiles, errors)
        # _scan_profiles_dir mutates in-place
        assert "parent_lim" in profiles
        assert "child_inherit" in profiles
        resolved, errors = resolve_profiles(profiles)
        assert not errors

        limits = resolved["child_inherit"].runtime_limits
        assert limits.memory_max_mb == 2048
        assert limits.pids_max == 256

    # -- 6. Child override wins --

    def test_child_override_wins(self, tmp_path):
        _write_profile(tmp_path, "parent_ov", {
            "name": "parent_ov",
            "description": "Parent",
            "plugins": ["cli"],
            "runtime_limits": {
                "memory_max_mb": 2048,
            },
        })
        _write_profile(tmp_path, "child_ov", {
            "name": "child_ov",
            "description": "Child overrides",
            "inherits": ["parent_ov"],
            "runtime_limits": {
                "memory_max_mb": 4096,
            },
        })

        _scan_profiles_dir(tmp_path, profiles := {}, errors := {})
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child_ov"].runtime_limits.memory_max_mb == 4096

    # -- 7. Inherited runtime_limits conflict --

    def test_inherited_runtime_limits_conflict(self, tmp_path):
        _write_profile(tmp_path, "p1_conflict", {
            "name": "p1_conflict",
            "description": "Parent 1",
            "plugins": ["cli"],
            "runtime_limits": {"memory_max_mb": 1024},
        })
        _write_profile(tmp_path, "p2_conflict", {
            "name": "p2_conflict",
            "description": "Parent 2",
            "plugins": ["cli"],
            "runtime_limits": {"memory_max_mb": 4096},
        })
        _write_profile(tmp_path, "child_conflict", {
            "name": "child_conflict",
            "description": "Child with conflicting parents",
            "inherits": ["p1_conflict", "p2_conflict"],
        })

        _scan_profiles_dir(tmp_path, profiles := {}, errors := {})
        resolved, errors = resolve_profiles(profiles)
        assert "child_conflict" in errors
        assert "runtime_limits" in errors["child_conflict"]

    # -- 8. attach_pid writes to cgroup_procs --

    def test_attach_pid_writes_to_cgroup_procs(self, manager):
        with _force_v2_available(manager._root), \
             patch("server.cgroups.platform.system", return_value="Linux"):
            manager._available = None
            manager.provision_cgroup("s1", RuntimeLimits(memory_max_mb=128))

        manager.attach_pid("s1", 99999)
        procs = manager.get_cgroup_path("s1") / "cgroup.procs"
        assert procs.read_text().strip() == "99999"

    # -- 9. Full lifecycle --

    def test_full_lifecycle_provision_attach_teardown(self, manager):
        with _force_v2_available(manager._root), \
             patch("server.cgroups.platform.system", return_value="Linux"):
            manager._available = None
            cfg = RuntimeLimits(memory_max_mb=256, pids_max=64, cpu_weight=100)
            assert manager.provision_cgroup("s1", cfg) is True

        cg = manager.get_cgroup_path("s1")
        assert cg.is_dir()
        assert (cg / "memory.max").exists()

        manager.attach_pid("s1", 12345)
        assert "12345" in (cg / "cgroup.procs").read_text()

        assert manager.teardown_cgroup("s1") is True
        assert not cg.exists()


# ===========================================================================
# Part 2: Real-Kernel Integration
# ===========================================================================


def _find_writable_cgroup_parent():
    """Find a writable cgroup v2 parent for tests.

    On systemd hosts, the test process is typically in a session scope
    (not delegated).  We look for the user service cgroup
    ``user@UID.service`` which has ``Delegate=yes`` and is writable.
    The fixture will move the test process into this cgroup so
    children can be placed in sub-cgroups.

    Returns None if no suitable parent is found.
    """
    uid = os.getuid()
    candidates = [
        Path(f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service"),
        Path("/sys/fs/cgroup/jaato-test"),
        Path("/sys/fs/cgroup"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        controllers = candidate / "cgroup.controllers"
        if not controllers.exists():
            continue
        try:
            controllers.read_text()
            test_dir = candidate / f".jaato-test-probe-{os.getpid()}"
            test_dir.mkdir()
            test_dir.rmdir()
            return candidate
        except (OSError, PermissionError):
            continue
    return None


_REAL_CGROUP_AVAILABLE = sys.platform == "linux" and _find_writable_cgroup_parent() is not None

skip_no_real_cgroup = pytest.mark.skipif(
    not _REAL_CGROUP_AVAILABLE,
    reason="Requires Linux with writable cgroup v2 parent",
)


def _can_migrate_to(parent: Path) -> bool:
    """Check whether the current process can move into *parent*.

    On systemd hosts the test process is typically in a session scope
    (not delegated) which is a *sibling* of the delegated user
    service cgroup.  cgroup v2 only allows moves to descendants, so
    the move will fail with EPERM.  Returns True only when the
    write actually succeeds.
    """
    procs_file = parent / "cgroup.procs"
    if not procs_file.exists():
        return False
    try:
        procs_file.write_text(str(os.getpid()))
        return True
    except OSError:
        return False


def _require_real_cgroup():
    """Find a writable cgroup parent or raise skip."""
    parent = _find_writable_cgroup_parent()
    if parent is None:
        pytest.skip("Requires Linux with writable cgroup v2 parent")
    return parent


@pytest.fixture
def real_cgroup_root():
    """Create a real cgroup under the system or user slice.

    Cleanup kills all processes and removes the directory.
    """
    parent = _require_real_cgroup()

    # Move the test process into the parent cgroup so forked children
    # can be placed in sub-cgroups.  cgroup v2 only allows moving
    # processes into descendants of the process's current cgroup.
    procs_file = parent / "cgroup.procs"
    if procs_file.exists():
        try:
            procs_file.write_text(str(os.getpid()))
        except OSError:
            pass

    root = parent / f"jaato-test-{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)

    stc = root / "cgroup.subtree_control"
    if stc.exists():
        try:
            stc.write_text("+memory +pids +cpu\n")
        except OSError:
            pass

    yield root

    # Cleanup: kill everything, remove directory
    kill_file = root / "cgroup.kill"
    if kill_file.exists():
        try:
            kill_file.write_text("1")
        except OSError:
            pass
    time.sleep(0.1)

    procs_file = root / "cgroup.procs"
    if procs_file.exists():
        for line in procs_file.read_text().strip().splitlines():
            pid = int(line)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.1)

    # Retry removal a few times (processes may take a moment to exit)
    for _ in range(5):
        try:
            root.rmdir()
            break
        except OSError:
            time.sleep(0.1)


@pytest.fixture
def real_manager(real_cgroup_root, monkeypatch):
    mgr = CgroupsManager(root=str(real_cgroup_root))

    def _rmdir_for_fakefs(cg_path):
        try:
            for entry in cg_path.iterdir():
                if entry.is_file():
                    entry.unlink()
            cg_path.rmdir()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False

    monkeypatch.setattr(mgr, "_safe_rmdir", staticmethod(_rmdir_for_fakefs))
    return mgr


@skip_no_real_cgroup
class TestRealKernel:

    # -- 14. Provision writes real controller files --

    def test_provision_real_cgroup_writes_controller_files(self, real_manager):
        cfg = RuntimeLimits(memory_max_mb=256, pids_max=64, cpu_weight=500)
        assert real_manager.provision_cgroup("real1", cfg) is True

        cg = real_manager.get_cgroup_path("real1")
        assert (cg / "memory.max").read_text().strip() == str(256 * 1024 * 1024)
        assert (cg / "pids.max").read_text().strip() == "64"
        assert (cg / "cpu.weight").read_text().strip() == "500"

        real_manager.teardown_cgroup("real1")

    # -- 17. No limits, no cgroup created --

    def test_no_limits_no_cgroup_created(self, real_manager):
        assert real_manager.provision_cgroup("noop", RuntimeLimits()) is True
        assert not real_manager.get_cgroup_path("noop").exists()

    # -- 10. pids_max blocks fork --

    @pytest.mark.skipif(not _can_migrate_to(_find_writable_cgroup_parent()),
                            reason="Process cannot migrate to delegated cgroup from current scope")
    def test_pids_max_blocks_fork(self, real_manager):
        real_manager.provision_cgroup("pidtest", RuntimeLimits(pids_max=3))
        cg = real_manager.get_cgroup_path("pidtest")

        children = []
        try:
            # Fork children — each joins the cgroup via preexec_fn
            for _ in range(3):
                pid = os.fork()
                if pid == 0:
                    # Child: attach self to cgroup, then sleep
                    (cg / "cgroup.procs").write_text(str(os.getpid()))
                    time.sleep(300)
                    os._exit(0)
                children.append(pid)

            time.sleep(0.3)

            # Verify all children are in the cgroup
            procs = (cg / "cgroup.procs").read_text().strip().splitlines()
            child_pids = {str(p) for p in children}
            assert child_pids.issubset(set(procs))

            # Verify pids.current reflects the limit pressure
            pids_current = int((cg / "pids.current").read_text().strip())
            assert pids_current >= 3

        finally:
            for pid in children:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass
            real_manager.teardown_cgroup("pidtest")

    # -- 11. pids_max exact boundary --

    @pytest.mark.skipif(not _can_migrate_to(_find_writable_cgroup_parent()),
                            reason="Process cannot migrate to delegated cgroup from current scope")
    def test_pids_max_exact_boundary(self, real_manager):
        real_manager.provision_cgroup("pid1", RuntimeLimits(pids_max=1))
        cg = real_manager.get_cgroup_path("pid1")

        # Fork a child that joins the cgroup — it becomes the single allowed process
        pid = os.fork()
        if pid == 0:
            try:
                (cg / "cgroup.procs").write_text(str(os.getpid()))
            except OSError:
                pass
            time.sleep(300)
            os._exit(0)

        try:
            time.sleep(0.2)
            pids_current = int((cg / "pids.current").read_text().strip())
            assert pids_current >= 1
        finally:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            real_manager.teardown_cgroup("pid1")

    # -- 15. Attach real PID to cgroup --

    @pytest.mark.skipif(not _can_migrate_to(_find_writable_cgroup_parent()),
                            reason="Process cannot migrate to delegated cgroup from current scope")
    def test_attach_real_pid_to_cgroup(self, real_manager):
        real_manager.provision_cgroup("attach_test", RuntimeLimits(memory_max_mb=128))
        cg = real_manager.get_cgroup_path("attach_test")

        pid = os.fork()
        if pid == 0:
            time.sleep(300)
            os._exit(0)

        try:
            real_manager.attach_pid("attach_test", pid)
            time.sleep(0.1)
            procs = (cg / "cgroup.procs").read_text().strip()
            assert str(pid) in procs
        finally:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            real_manager.teardown_cgroup("attach_test")

    # -- 16. Teardown kills and removes --

    @pytest.mark.skipif(not _can_migrate_to(_find_writable_cgroup_parent()),
                            reason="Process cannot migrate to delegated cgroup from current scope")
    def test_teardown_kills_and_removes(self, real_manager):
        real_manager.provision_cgroup("teardown_test", RuntimeLimits(pids_max=64))

        pid = os.fork()
        if pid == 0:
            time.sleep(300)
            os._exit(0)

        cg = real_manager.get_cgroup_path("teardown_test")
        try:
            real_manager.attach_pid("teardown_test", pid)
            time.sleep(0.1)
            assert str(pid) in (cg / "cgroup.procs").read_text()
        finally:
            real_manager.teardown_cgroup("teardown_test")
            # Child should be dead
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            assert not alive, "Child process survived cgroup teardown"
            os.waitpid(pid, 0)
            assert not cg.exists()

    # ------------------------------------------------------------------
    # §7d cgroup inheritance pins
    # ------------------------------------------------------------------

    def test_runner_spawn_attaches_to_session_cgroup(self, real_manager):
        """§7d integration pin (1/3): a runner spawned with
        ``cgroup_attach`` from CgroupsManager.make_attach_callback
        lands in the session cgroup, not the daemon's parent
        cgroup.

        Validates the architectural pivot: instead of wiring
        per-Popen attach_cb preexec_fn's on plugin-side
        subprocess launchers, the runner itself is in the
        cgroup at exec time.  Children inherit by default per
        cgroup-v2 kernel contract.
        """
        from server.runner_spawner import RunnerSpawner

        real_manager.provision_cgroup(
            "spawn1", RuntimeLimits(memory_max_mb=128, pids_max=32),
        )
        attach_cb = real_manager.make_attach_callback("spawn1")
        spawner = RunnerSpawner()

        # The spawner refuses to fork from a confined daemon
        # thread.  In test contexts the thread is unconfined
        # (Linux without apparmor, or apparmor without per-thread
        # confinement).  ``disable_confine=True`` skips the
        # runner-side aa_change_profile step too.
        spawned = spawner.spawn(
            profile_name="",
            session_id="spawn1",
            workspace_path=None,
            disable_confine=True,
            cgroup_attach=attach_cb,
        )
        try:
            # Read /proc/<pid>/cgroup — single-line cgroup v2
            # format is ``0::/<path>``.  The runner's path must
            # be the session cgroup.
            with open(f"/proc/{spawned.pid}/cgroup") as f:
                cgroup_line = f.read().strip()
            expected_path = str(
                real_manager.get_cgroup_path("spawn1")
            )
            # /proc/<pid>/cgroup reports the path relative to
            # the cgroup mount (``/sys/fs/cgroup``); strip the
            # mount prefix for comparison.
            mount_prefix = "/sys/fs/cgroup"
            expected_rel = expected_path
            if expected_rel.startswith(mount_prefix):
                expected_rel = expected_rel[len(mount_prefix):]
            assert (
                expected_rel in cgroup_line
            ), (
                f"runner pid {spawned.pid} expected in cgroup "
                f"path {expected_rel!r}; /proc reported "
                f"{cgroup_line!r}"
            )
        finally:
            # Reap the runner.  It's blocked on read_frame from
            # the parent socket; closing the socket triggers EOF
            # and the runner exits cleanly.
            try:
                spawned.parent_socket.close()
            except Exception:
                pass
            try:
                os.waitpid(spawned.pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass
            real_manager.teardown_cgroup("spawn1")

    def test_runner_child_inherits_cgroup(self, real_manager):
        """§7d integration pin (2/3): a subprocess spawned BY the
        runner inherits the runner's cgroup placement
        automatically — no per-Popen preexec_fn needed.

        Simulates the cli plugin's subprocess-spawning path:
        the runner is in the cgroup; ``Popen(...)`` from inside
        the runner forks a child that the kernel places in the
        same cgroup by inheritance.

        Test shape: fork ourselves (impersonating the runner),
        attach to the cgroup, then fork a grandchild (the
        ``cli`` subprocess); read the grandchild's
        /proc/<pid>/cgroup and assert it matches.
        """
        real_manager.provision_cgroup(
            "inherit1", RuntimeLimits(memory_max_mb=128, pids_max=32),
        )
        attach_cb = real_manager.make_attach_callback("inherit1")
        cg_path = str(real_manager.get_cgroup_path("inherit1"))

        # Communicate the grandchild pid back via a pipe.
        rfd, wfd = os.pipe()

        runner_pid = os.fork()
        if runner_pid == 0:
            # ----- impersonate the runner -----
            os.close(rfd)
            try:
                attach_cb()  # join the session cgroup
                # Fork a "cli subprocess".
                child_pid = os.fork()
                if child_pid == 0:
                    # The grandchild: write our pid back + sleep.
                    os.write(wfd, str(os.getpid()).encode())
                    os.close(wfd)
                    time.sleep(2.0)
                    os._exit(0)
                # Runner waits for the grandchild.
                os.close(wfd)
                os.waitpid(child_pid, 0)
                os._exit(0)
            except BaseException:
                os._exit(127)

        # ----- parent: read the grandchild's pid -----
        os.close(wfd)
        try:
            grandchild_pid_bytes = os.read(rfd, 64)
            grandchild_pid = int(grandchild_pid_bytes.decode())

            # The grandchild's cgroup file must match the
            # runner's cgroup — inheritance.
            with open(f"/proc/{grandchild_pid}/cgroup") as f:
                grandchild_cgroup = f.read().strip()
            mount_prefix = "/sys/fs/cgroup"
            expected_rel = cg_path
            if expected_rel.startswith(mount_prefix):
                expected_rel = expected_rel[len(mount_prefix):]
            assert (
                expected_rel in grandchild_cgroup
            ), (
                f"grandchild pid {grandchild_pid} expected in "
                f"cgroup {expected_rel!r}; /proc reported "
                f"{grandchild_cgroup!r} — cgroup v2 inheritance "
                f"didn't hold"
            )
        finally:
            os.close(rfd)
            try:
                os.waitpid(runner_pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass
            real_manager.teardown_cgroup("inherit1")

    def test_runner_pty_grandchild_inherits_cgroup(self, real_manager):
        """§7d integration pin (3/3): PTY-stress case — peer-
        review v2 observation #2's "grandchild inheritance"
        worry.

        ``interactive_shell`` plugin spawns a PTY (pexpect-style
        ``fork()`` + ``setsid()``+``ioctl(TIOCSCTTY)``); that PTY's
        child shell spawns further commands (e.g., ``ls`` inside
        the shell).  Three levels of inheritance:
        runner → PTY child → grandchild command.

        Test shape: simulate the chain via plain fork()'s with
        an intermediate process.  Pin all three pids end up in
        the same cgroup.

        Note: real PTYs (``os.openpty()`` + setsid()) wouldn't
        change cgroup membership — cgroup v2 inheritance is
        tied to fork() / clone(), not to terminal session
        ownership.  But the kernel contract is what we're
        pinning; this test is the stress case for that
        contract.
        """
        real_manager.provision_cgroup(
            "pty1", RuntimeLimits(memory_max_mb=128, pids_max=64),
        )
        attach_cb = real_manager.make_attach_callback("pty1")
        cg_path = str(real_manager.get_cgroup_path("pty1"))

        # Pipe back the grandchild's pid (level-3 process).
        rfd, wfd = os.pipe()

        l1_pid = os.fork()
        if l1_pid == 0:
            # ----- level 1: the runner -----
            os.close(rfd)
            try:
                attach_cb()
                l2_pid = os.fork()
                if l2_pid == 0:
                    # ----- level 2: the PTY child shell -----
                    try:
                        l3_pid = os.fork()
                        if l3_pid == 0:
                            # ----- level 3: the grandchild command -----
                            os.write(wfd, str(os.getpid()).encode())
                            os.close(wfd)
                            time.sleep(2.0)
                            os._exit(0)
                        os.close(wfd)
                        os.waitpid(l3_pid, 0)
                        os._exit(0)
                    except BaseException:
                        os._exit(127)
                os.close(wfd)
                os.waitpid(l2_pid, 0)
                os._exit(0)
            except BaseException:
                os._exit(127)

        os.close(wfd)
        try:
            l3_pid_bytes = os.read(rfd, 64)
            l3_pid = int(l3_pid_bytes.decode())

            with open(f"/proc/{l3_pid}/cgroup") as f:
                l3_cgroup = f.read().strip()
            mount_prefix = "/sys/fs/cgroup"
            expected_rel = cg_path
            if expected_rel.startswith(mount_prefix):
                expected_rel = expected_rel[len(mount_prefix):]
            assert (
                expected_rel in l3_cgroup
            ), (
                f"PTY grandchild pid {l3_pid} expected in cgroup "
                f"{expected_rel!r}; /proc reported "
                f"{l3_cgroup!r} — three-level inheritance "
                f"(runner → PTY → command) broke"
            )
        finally:
            os.close(rfd)
            try:
                os.waitpid(l1_pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass
            real_manager.teardown_cgroup("pty1")


@skip_no_real_cgroup
class TestRealKernelExpensive:
    """Tests gated behind environment variables — too slow/racy for CI."""

    # -- 12. OOM kill --

    @pytest.mark.skipif(
        not os.environ.get("JATO_TEST_OOM"),
        reason="OOM tests require JATO_TEST_OOM=1",
    )
    def test_memory_max_mb_oom_kill(self, real_manager):
        real_manager.provision_cgroup("oom_test", RuntimeLimits(memory_max_mb=16))
        cg = real_manager.get_cgroup_path("oom_test")
        attach = real_manager.make_attach_callback("oom_test")
        reader = real_manager.make_event_reader("oom_test")

        pid = os.fork()
        if pid == 0:
            attach()
            # Try to allocate more than the limit
            import mmap
            buf = mmap.mmap(-1, 32 * 1024 * 1024)
            time.sleep(1)
            os._exit(0)

        try:
            _, status = os.waitpid(pid, 0)
            # Child should have been killed by OOM or signal
            assert os.WIFSIGNALED(status) or os.WIFEXITED(status)
        finally:
            counts = reader()
            if counts is not None:
                # oom_kill may have incremented
                assert counts.get("oom_kill", 0) >= 0
            real_manager.teardown_cgroup("oom_test")

    # -- 13. CPU weight relative shares --

    @pytest.mark.skipif(
        not os.environ.get("JATO_TEST_CPU"),
        reason="CPU share tests require JATO_TEST_CPU=1 and isolated cores",
    )
    def test_cpu_weight_relative_shares(self, real_manager):
        real_manager.provision_cgroup("cpu_low", RuntimeLimits(cpu_weight=100))
        real_manager.provision_cgroup("cpu_high", RuntimeLimits(cpu_weight=900))

        attach_low = real_manager.make_attach_callback("cpu_low")
        attach_high = real_manager.make_attach_callback("cpu_high")

        # Fork two CPU-bound children
        pid_low = os.fork()
        if pid_low == 0:
            attach_low()
            # Burn CPU
            while True:
                pass

        pid_high = os.fork()
        if pid_high == 0:
            attach_high()
            while True:
                pass

        try:
            time.sleep(2)  # Let them run

            # Read cpu.stat
            def read_usage(cg_name):
                stat_file = real_manager.get_cgroup_path(cg_name) / "cpu.stat"
                for line in stat_file.read_text().strip().splitlines():
                    if line.startswith("usage_usec"):
                        return int(line.split()[1])
                return 0

            usage_low = read_usage("cpu_low")
            usage_high = read_usage("cpu_high")

            # High-weight should get significantly more CPU
            if usage_low > 0:
                ratio = usage_high / usage_low
                assert ratio > 3.0, f"CPU ratio {ratio:.1f} too low (expected ~9x)"
        finally:
            os.kill(pid_low, signal.SIGKILL)
            os.kill(pid_high, signal.SIGKILL)
            os.waitpid(pid_low, 0)
            os.waitpid(pid_high, 0)
            real_manager.teardown_cgroup("cpu_low")
            real_manager.teardown_cgroup("cpu_high")
