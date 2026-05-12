"""Tests for RuntimeLimits and CgroupsManager.

These tests run on any OS — they fake the cgroup v2 filesystem inside
``tmp_path`` and stub out the v2 marker check.  Real-kernel integration
exercise lives in adversarial/integration suites, not here.
"""

import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# Mirror the test_apparmor.py shim: avoid importing server/__init__.py,
# which pulls heavy provider deps the unit tests don't need.
if "server" not in sys.modules:
    _stub = types.ModuleType("server")
    _stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "server")]
    sys.modules["server"] = _stub

import server.cgroups as _cg
from shared.runtime_limits import RuntimeLimits
CgroupsManager = _cg.CgroupsManager


# ---------------------------------------------------------------------------
# RuntimeLimits
# ---------------------------------------------------------------------------

class TestRuntimeLimitsDefaults:
    def test_all_none_means_no_limits(self):
        cfg = RuntimeLimits()
        assert cfg.memory_max_mb is None
        assert cfg.pids_max is None
        assert cfg.cpu_weight is None
        assert cfg.tool_timeout_seconds is None
        assert cfg.max_output_bytes is None
        assert cfg.has_kernel_limits() is False

    def test_has_kernel_limits_only_when_kernel_field_set(self):
        # App-layer fields don't count as kernel limits.
        assert RuntimeLimits(tool_timeout_seconds=30).has_kernel_limits() is False
        assert RuntimeLimits(max_output_bytes=1024).has_kernel_limits() is False
        # Kernel fields do.
        assert RuntimeLimits(memory_max_mb=512).has_kernel_limits() is True
        assert RuntimeLimits(pids_max=256).has_kernel_limits() is True
        assert RuntimeLimits(cpu_weight=200).has_kernel_limits() is True


class TestRuntimeLimitsValidation:
    @pytest.mark.parametrize("value", [0, -1, "512", 1.5])
    def test_memory_max_mb_invalid(self, value):
        with pytest.raises(ValueError, match="memory_max_mb"):
            RuntimeLimits(memory_max_mb=value)

    def test_memory_max_mb_sanity_ceiling(self):
        # 2 TiB — likely a unit typo.
        with pytest.raises(ValueError, match="sanity ceiling"):
            RuntimeLimits(memory_max_mb=2 * 1024 * 1024)

    @pytest.mark.parametrize("value", [0, -1, "256"])
    def test_pids_max_invalid(self, value):
        with pytest.raises(ValueError, match="pids_max"):
            RuntimeLimits(pids_max=value)

    @pytest.mark.parametrize("value", [0, 10_001, "100"])
    def test_cpu_weight_out_of_range(self, value):
        with pytest.raises(ValueError, match="cpu_weight"):
            RuntimeLimits(cpu_weight=value)

    @pytest.mark.parametrize("value", [1, 100, 10_000])
    def test_cpu_weight_valid_bounds(self, value):
        RuntimeLimits(cpu_weight=value)

    @pytest.mark.parametrize("value", [0, -1.0, "30"])
    def test_tool_timeout_invalid(self, value):
        with pytest.raises(ValueError, match="tool_timeout_seconds"):
            RuntimeLimits(tool_timeout_seconds=value)

    def test_max_output_bytes_invalid(self):
        with pytest.raises(ValueError, match="max_output_bytes"):
            RuntimeLimits(max_output_bytes=0)


class TestRuntimeLimitsFromDict:
    def test_none_returns_default(self):
        assert RuntimeLimits.from_dict(None) == RuntimeLimits()

    def test_empty_dict_returns_default(self):
        assert RuntimeLimits.from_dict({}) == RuntimeLimits()

    def test_partial_dict(self):
        cfg = RuntimeLimits.from_dict({"memory_max_mb": 1024, "pids_max": 512})
        assert cfg.memory_max_mb == 1024
        assert cfg.pids_max == 512
        assert cfg.cpu_weight is None

    def test_unknown_keys_land_in_extra_not_kwargs(self):
        # Unknown keys must NOT crash from_dict (forward-compat).
        cfg = RuntimeLimits.from_dict({
            "memory_max_mb": 256,
            "future_knob": "value",
        })
        assert cfg.memory_max_mb == 256
        assert cfg.extra == {"future_knob": "value"}

    def test_validation_runs_on_from_dict(self):
        with pytest.raises(ValueError):
            RuntimeLimits.from_dict({"cpu_weight": 99999})


# ---------------------------------------------------------------------------
# CgroupsManager — availability
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_cgroup_root(tmp_path):
    """Build a fake cgroup v2 hierarchy under tmp_path.

    Creates a parent dir with cgroup.subtree_control listing all
    required controllers.  The v2 marker (/sys/fs/cgroup/cgroup.controllers)
    is stubbed by patching server.cgroups._V2_MARKER per-test.
    """
    root = tmp_path / "jaato"
    root.mkdir()
    (root / "cgroup.subtree_control").write_text("memory pids cpu io\n")
    return root


@pytest.fixture
def manager(fake_cgroup_root, monkeypatch):
    """Manager with rmdir patched to mimic kernel cgroup-fs semantics.

    A real cgroup v2 directory rmdir's even with controller files
    inside (kernel pseudofiles don't block rmdir(2)).  Our test fake
    is a real tmpfs dir, so we rewrite ``_safe_rmdir`` to clear files
    before rmdir — the production rmdir path stays exactly as-shipped.
    """
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


def _force_v2_available(tmp_path):
    """Return a patched _V2_MARKER pointing at an existing file."""
    marker = tmp_path / "v2_marker"
    marker.write_text("memory pids cpu io\n")
    return patch("server.cgroups._V2_MARKER", marker)


class TestAvailability:
    def test_not_available_on_non_linux(self, manager, tmp_path):
        with patch("server.cgroups.platform.system", return_value="Darwin"):
            manager._available = None
            assert manager.is_available() is False
            assert "not Linux" in manager._missing_reason

    def test_not_available_without_v2_marker(self, manager, tmp_path):
        missing = tmp_path / "does_not_exist"
        with patch("server.cgroups.platform.system", return_value="Linux"), \
             patch("server.cgroups._V2_MARKER", missing):
            manager._available = None
            assert manager.is_available() is False
            assert "cgroup v2 not mounted" in manager._missing_reason

    def test_not_available_when_root_missing(self, tmp_path):
        with _force_v2_available(tmp_path), \
             patch("server.cgroups.platform.system", return_value="Linux"):
            mgr = CgroupsManager(root=str(tmp_path / "nonexistent"))
            assert mgr.is_available() is False
            assert "does not exist" in mgr._missing_reason

    def test_not_available_when_controllers_missing(self, tmp_path, fake_cgroup_root):
        # Drop "cpu" from subtree_control.
        (fake_cgroup_root / "cgroup.subtree_control").write_text("memory pids io\n")
        mgr = CgroupsManager(root=str(fake_cgroup_root))
        with _force_v2_available(tmp_path), \
             patch("server.cgroups.platform.system", return_value="Linux"):
            assert mgr.is_available() is False
            assert "['cpu']" in mgr._missing_reason

    def test_available_when_all_prereqs_met(self, tmp_path, manager):
        with _force_v2_available(tmp_path), \
             patch("server.cgroups.platform.system", return_value="Linux"):
            assert manager.is_available() is True

    def test_caches_result(self, manager):
        manager._available = True
        assert manager.is_available() is True
        manager._available = False
        assert manager.is_available() is False  # cached, no re-check


# ---------------------------------------------------------------------------
# CgroupsManager — provision / teardown / attach
# ---------------------------------------------------------------------------

class TestCgroupNaming:
    def test_get_cgroup_name(self, manager):
        assert manager.get_cgroup_name("20260101_120000") == "jaato-ws-20260101_120000"

    def test_get_cgroup_path(self, manager, fake_cgroup_root):
        assert manager.get_cgroup_path("abc") == fake_cgroup_root / "jaato-ws-abc"


class TestProvisionCgroup:
    def test_unavailable_returns_true_no_op(self, manager):
        manager._available = False
        cfg = RuntimeLimits(memory_max_mb=512)
        assert manager.provision_cgroup("s1", cfg) is True

    def test_no_kernel_limits_skips_creation(self, manager):
        manager._available = True
        cfg = RuntimeLimits(tool_timeout_seconds=30)  # app-only
        assert manager.provision_cgroup("s1", cfg) is True
        assert not manager.get_cgroup_path("s1").exists()

    def test_writes_memory_limit_in_bytes(self, manager):
        manager._available = True
        cfg = RuntimeLimits(memory_max_mb=512)
        assert manager.provision_cgroup("s1", cfg) is True
        cg = manager.get_cgroup_path("s1")
        assert cg.is_dir()
        assert (cg / "memory.max").read_text() == str(512 * 1024 * 1024)

    def test_writes_pids_and_cpu_limits(self, manager):
        manager._available = True
        cfg = RuntimeLimits(pids_max=256, cpu_weight=200)
        assert manager.provision_cgroup("s1", cfg) is True
        cg = manager.get_cgroup_path("s1")
        assert (cg / "pids.max").read_text() == "256"
        assert (cg / "cpu.weight").read_text() == "200"

    def test_skips_unset_limits(self, manager):
        manager._available = True
        cfg = RuntimeLimits(memory_max_mb=128)  # only memory
        assert manager.provision_cgroup("s1", cfg) is True
        cg = manager.get_cgroup_path("s1")
        assert (cg / "memory.max").exists()
        assert not (cg / "pids.max").exists()
        assert not (cg / "cpu.weight").exists()

    def test_idempotent_reprovision_overwrites(self, manager):
        manager._available = True
        manager.provision_cgroup("s1", RuntimeLimits(memory_max_mb=128))
        manager.provision_cgroup("s1", RuntimeLimits(memory_max_mb=256))
        cg = manager.get_cgroup_path("s1")
        assert (cg / "memory.max").read_text() == str(256 * 1024 * 1024)

    def test_rolls_back_on_write_failure(self, manager):
        manager._available = True
        cfg = RuntimeLimits(memory_max_mb=128, pids_max=64)

        # Make the second write fail by patching _write_one.
        original = manager._write_one
        calls = {"n": 0}

        def flaky_write(path, value):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("simulated EIO")
            original(path, value)

        with patch.object(manager, "_write_one", side_effect=flaky_write):
            assert manager.provision_cgroup("s1", cfg) is False
        # Cgroup directory should be rolled back (rmdir succeeded
        # because nothing was attached to cgroup.procs).
        assert not manager.get_cgroup_path("s1").exists()


class TestTeardownCgroup:
    def test_unavailable_returns_true_no_op(self, manager):
        manager._available = False
        assert manager.teardown_cgroup("s1") is True

    def test_missing_cgroup_returns_true(self, manager):
        manager._available = True
        assert manager.teardown_cgroup("never-provisioned") is True

    def test_removes_cgroup_directory(self, manager):
        manager._available = True
        manager.provision_cgroup("s1", RuntimeLimits(memory_max_mb=128))
        assert manager.get_cgroup_path("s1").exists()
        assert manager.teardown_cgroup("s1") is True
        assert not manager.get_cgroup_path("s1").exists()

    def test_writes_cgroup_kill_when_present(self, manager):
        """Modern kernels (5.14+) — cgroup.kill exists, we write '1' to it."""
        manager._available = True
        manager.provision_cgroup("s1", RuntimeLimits(pids_max=64))
        cg = manager.get_cgroup_path("s1")
        # Simulate kernel ≥ 5.14: cgroup.kill exists.
        (cg / "cgroup.kill").write_text("0")
        assert manager.teardown_cgroup("s1") is True
        # rmdir succeeded (no real procs attached in fake fs).
        assert not cg.exists()


class TestAttachPid:
    def test_unavailable_is_silent_noop(self, manager):
        manager._available = False
        # No cgroup, no exception.
        manager.attach_pid("s1", 12345)

    def test_missing_cgroup_is_silent_noop(self, manager):
        manager._available = True
        # Session never provisioned a cgroup (no kernel limits).
        manager.attach_pid("s1", 12345)  # must not raise

    def test_writes_pid_to_cgroup_procs(self, manager):
        manager._available = True
        manager.provision_cgroup("s1", RuntimeLimits(memory_max_mb=128))
        manager.attach_pid("s1", 99999)
        procs = manager.get_cgroup_path("s1") / "cgroup.procs"
        assert procs.read_text() == "99999"


class TestReadEventCounts:
    """Tests for ``CgroupsManager.read_event_counts`` — the telemetry
    primitive that surfaces ``cgroup.events`` (populated, oom_kill,
    etc.) so callers can attribute kernel-killed exits.
    """

    def test_unavailable_returns_none(self, manager):
        manager._available = False
        assert manager.read_event_counts("s1") is None

    def test_missing_cgroup_returns_none(self, manager):
        # Session never provisioned a cgroup — read returns None,
        # not a crash.  This is the "session has only app-layer
        # limits" path.
        manager._available = True
        assert manager.read_event_counts("never-provisioned") is None

    def test_missing_events_file_returns_none(self, manager):
        # Cgroup exists but cgroup.events file is absent (older
        # kernel?  partially-constructed fake?) — read returns None.
        manager._available = True
        manager.provision_cgroup("s1", RuntimeLimits(memory_max_mb=64))
        # Provisioning doesn't auto-create cgroup.events; the kernel
        # would in real cgroupfs.  No file → None.
        assert manager.read_event_counts("s1") is None

    def test_parses_kernel_format(self, manager):
        # cgroup.events is `key value\n` per line.  Verify the parser
        # picks up all keys and converts to int.
        manager._available = True
        manager.provision_cgroup("s1", RuntimeLimits(memory_max_mb=64))
        cg = manager.get_cgroup_path("s1")
        (cg / "cgroup.events").write_text(
            "populated 1\n"
            "frozen 0\n"
            "oom 2\n"
            "oom_kill 3\n"
        )
        counts = manager.read_event_counts("s1")
        assert counts == {
            "populated": 1,
            "frozen": 0,
            "oom": 2,
            "oom_kill": 3,
        }

    def test_skips_malformed_lines(self, manager):
        # A future kernel could add non-integer values or
        # multi-token formats.  We must not crash — skip the line.
        manager._available = True
        manager.provision_cgroup("s1", RuntimeLimits(memory_max_mb=64))
        cg = manager.get_cgroup_path("s1")
        (cg / "cgroup.events").write_text(
            "populated 1\n"
            "future_key not_an_int\n"
            "weird three tokens here\n"
            "oom_kill 0\n"
        )
        counts = manager.read_event_counts("s1")
        assert counts == {"populated": 1, "oom_kill": 0}

    def test_empty_file_returns_empty_dict(self, manager):
        manager._available = True
        manager.provision_cgroup("s1", RuntimeLimits(memory_max_mb=64))
        cg = manager.get_cgroup_path("s1")
        (cg / "cgroup.events").write_text("")
        # Distinct from None — file exists but has no content.
        assert manager.read_event_counts("s1") == {}

    def test_attribution_pattern_works(self, manager):
        # The intended use: snapshot before spawn, snapshot after
        # wait, compare deltas.  The kernel writes are monotonic.
        manager._available = True
        manager.provision_cgroup("s1", RuntimeLimits(memory_max_mb=64))
        cg = manager.get_cgroup_path("s1")
        (cg / "cgroup.events").write_text("oom_kill 0\n")

        before = manager.read_event_counts("s1")
        # Simulate the kernel bumping the counter mid-call.
        (cg / "cgroup.events").write_text("oom_kill 2\n")
        after = manager.read_event_counts("s1")

        assert after["oom_kill"] - before["oom_kill"] == 2


class TestMakeEventReader:
    """Tests for ``CgroupsManager.make_event_reader`` — the
    zero-arg-callable form of :meth:`read_event_counts`, used to plumb
    snapshot access across the server → shared boundary without
    exposing the manager itself.
    """

    def test_unavailable_returns_noop(self, manager):
        manager._available = False
        reader = manager.make_event_reader("s1")
        # No-op reader returns None on every call.
        assert reader() is None

    def test_missing_cgroup_returns_noop(self, manager):
        manager._available = True
        reader = manager.make_event_reader("never-provisioned")
        assert reader() is None

    def test_reader_returns_current_snapshot(self, manager):
        manager._available = True
        manager.provision_cgroup("s1", RuntimeLimits(memory_max_mb=64))
        cg = manager.get_cgroup_path("s1")
        (cg / "cgroup.events").write_text("oom_kill 1\npopulated 1\n")

        reader = manager.make_event_reader("s1")
        assert reader() == {"oom_kill": 1, "populated": 1}

    def test_reader_reflects_kernel_updates(self, manager):
        # The reader closes over manager + session_id and re-reads on
        # every call — the kernel's monotonic counters appear via
        # subsequent invocations.
        manager._available = True
        manager.provision_cgroup("s1", RuntimeLimits(memory_max_mb=64))
        cg = manager.get_cgroup_path("s1")
        events = cg / "cgroup.events"

        events.write_text("oom_kill 0\n")
        reader = manager.make_event_reader("s1")
        assert reader() == {"oom_kill": 0}

        events.write_text("oom_kill 5\n")
        assert reader() == {"oom_kill": 5}


class TestMakeAttachCallback:
    def test_unavailable_returns_noop(self, manager):
        manager._available = False
        cb = manager.make_attach_callback("s1")
        cb()  # must not raise; must not write anywhere

    def test_missing_cgroup_returns_noop(self, manager):
        manager._available = True
        cb = manager.make_attach_callback("never-provisioned")
        cb()  # silent no-op

    def test_callback_writes_current_pid(self, manager):
        manager._available = True
        manager.provision_cgroup("s1", RuntimeLimits(pids_max=16))
        cb = manager.make_attach_callback("s1")
        cb()
        procs = manager.get_cgroup_path("s1") / "cgroup.procs"
        assert procs.read_text() == str(os.getpid())


# ---------------------------------------------------------------------------
# Phase 5 §5.3 — orphan sub-cgroup scanner
# ---------------------------------------------------------------------------


class TestListOrphanSubCgroups:
    """Phase 5 §5.3: name-pattern scan that finds sub-cgroups whose
    directories exist on disk but aren't in the caller's known-set.

    Used by ``SessionManager._cascade_teardown_isolated_subagents`` to
    catch sub-cgroups left behind by rollback failures or mid-spawn
    crashes — the §4.3.9 item 4 hardening gap."""

    def test_empty_when_no_matching_dirs(self, manager):
        manager._available = True
        result = manager.list_orphan_sub_cgroups(
            "parent-A", known_isolated_session_ids=set(),
        )
        assert result == []

    def test_returns_dirs_matching_pattern_not_in_known_set(
        self, manager, fake_cgroup_root,
    ):
        manager._available = True
        # Build three sub-cgroup dirs for parent-A; mark one as known.
        for subagent in ("X", "Y", "Z"):
            (fake_cgroup_root / f"jaato-ws-parent-A__sub_{subagent}").mkdir()

        result = manager.list_orphan_sub_cgroups(
            "parent-A",
            known_isolated_session_ids={"parent-A__sub_X"},
        )
        # X is known; Y and Z are orphans.
        assert sorted(result) == ["parent-A__sub_Y", "parent-A__sub_Z"]

    def test_ignores_other_parents(self, manager, fake_cgroup_root):
        """A scan for parent A must not return sub-cgroups belonging
        to parent B — confused-deputy protection."""
        manager._available = True
        (fake_cgroup_root / "jaato-ws-parent-A__sub_X").mkdir()
        (fake_cgroup_root / "jaato-ws-parent-B__sub_Y").mkdir()

        result_a = manager.list_orphan_sub_cgroups(
            "parent-A", known_isolated_session_ids=set(),
        )
        result_b = manager.list_orphan_sub_cgroups(
            "parent-B", known_isolated_session_ids=set(),
        )
        assert result_a == ["parent-A__sub_X"]
        assert result_b == ["parent-B__sub_Y"]

    def test_ignores_parent_cgroup_dir_itself(
        self, manager, fake_cgroup_root,
    ):
        """The parent's own cgroup (``jaato-ws-parent-A``) is NOT a
        sub-cgroup and must not be returned even when nothing is
        known."""
        manager._available = True
        (fake_cgroup_root / "jaato-ws-parent-A").mkdir()
        (fake_cgroup_root / "jaato-ws-parent-A__sub_X").mkdir()

        result = manager.list_orphan_sub_cgroups(
            "parent-A", known_isolated_session_ids=set(),
        )
        # Only the __sub_ entry — the parent dir doesn't match the
        # leaf-prefix pattern.
        assert result == ["parent-A__sub_X"]

    def test_silent_when_cgroups_unavailable(self, manager):
        manager._available = False
        result = manager.list_orphan_sub_cgroups(
            "parent-A", known_isolated_session_ids=set(),
        )
        assert result == []

    def test_tolerates_non_directory_entries(
        self, manager, fake_cgroup_root,
    ):
        """A regular file matching the pattern (defensive: shouldn't
        happen on a real cgroup-fs, but protects against odd hosts)
        is skipped — only directories count."""
        manager._available = True
        (fake_cgroup_root / "jaato-ws-parent-A__sub_X").mkdir()
        (fake_cgroup_root / "jaato-ws-parent-A__sub_FILE").write_text("x")

        result = manager.list_orphan_sub_cgroups(
            "parent-A", known_isolated_session_ids=set(),
        )
        assert result == ["parent-A__sub_X"]

    def test_returns_empty_when_root_missing(self, tmp_path):
        """Root directory absent (host without cgroup root provisioned)
        → empty list, no exception."""
        mgr = CgroupsManager(root=str(tmp_path / "nonexistent"))
        mgr._available = True  # Force past is_available() short-circuit.
        result = mgr.list_orphan_sub_cgroups(
            "parent-A", known_isolated_session_ids=set(),
        )
        assert result == []
