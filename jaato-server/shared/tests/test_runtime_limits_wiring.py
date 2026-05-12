"""Tests for ToolExecutor.set_runtime_limits forwarding.

The executor stores attach_callback + limits and forwards them to any
exposed plugin that implements ``set_runtime_limits`` — same pattern
as ``set_tool_output_callback``.  Subprocess-launching plugins (cli,
interactive_shell) are the intended recipients.
"""

import os
import sys
import types
from unittest.mock import MagicMock

import pytest

# Match the existing test-isolation pattern (avoid heavy server/__init__).
if "server" not in sys.modules:
    _stub = types.ModuleType("server")
    _stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "server")]
    sys.modules["server"] = _stub

from shared.ai_tool_runner import ToolExecutor
from shared.runtime_limits import RuntimeLimits


class _DummyPluginWithSetter:
    """Fake plugin that implements set_runtime_limits.

    Records its calls so tests can assert the executor forwarded
    correctly.  Other ``ToolPlugin`` methods are absent — the registry
    forwarder only checks ``hasattr``.
    """

    def __init__(self):
        self.calls = []

    def set_runtime_limits(self, attach, limits):
        self.calls.append((attach, limits))


class _DummyPluginWithoutSetter:
    """Fake plugin that does NOT implement set_runtime_limits.

    The forwarder must skip it silently — plugins that don't launch
    subprocesses don't need to know about runtime limits.
    """
    pass


def _make_executor_with_registry(plugins: dict) -> ToolExecutor:
    """Construct a ToolExecutor with a fake registry exposing *plugins*.

    The registry only needs ``list_exposed`` and ``get_plugin``; the
    forwarder doesn't touch anything else.
    """
    executor = ToolExecutor()
    registry = MagicMock()
    registry.list_exposed.return_value = list(plugins.keys())
    registry.get_plugin.side_effect = lambda name: plugins.get(name)
    executor.set_registry(registry)
    return executor


class TestSetRuntimeLimitsStorage:
    def test_default_state_is_empty(self):
        executor = ToolExecutor()
        assert executor.get_cgroup_attach() is None
        assert executor.get_runtime_limits() is None

    def test_setter_stores_both_pieces(self):
        executor = ToolExecutor()
        attach = lambda: None
        limits = RuntimeLimits(memory_max_mb=256)
        executor.set_runtime_limits(attach, limits)
        assert executor.get_cgroup_attach() is attach
        assert executor.get_runtime_limits() is limits

    def test_setter_accepts_none(self):
        executor = ToolExecutor()
        executor.set_runtime_limits(lambda: None, RuntimeLimits(pids_max=64))
        executor.set_runtime_limits(None, None)
        assert executor.get_cgroup_attach() is None
        assert executor.get_runtime_limits() is None


class TestForwardingToPlugins:
    def test_forwards_to_plugin_with_setter(self):
        plugin = _DummyPluginWithSetter()
        executor = _make_executor_with_registry({"cli": plugin})

        attach = lambda: None
        limits = RuntimeLimits(memory_max_mb=128, tool_timeout_seconds=30)
        executor.set_runtime_limits(attach, limits)

        assert plugin.calls == [(attach, limits)]

    def test_skips_plugin_without_setter(self):
        # No exception, no crash — the forwarder must hasattr-gate.
        plugin = _DummyPluginWithoutSetter()
        executor = _make_executor_with_registry({"todo": plugin})
        executor.set_runtime_limits(lambda: None, RuntimeLimits(pids_max=8))

    def test_forwards_to_only_plugins_with_setter(self):
        # Mixed registry: one supports it, one doesn't.  The forwarder
        # must call the supporting plugin and skip the other.
        good = _DummyPluginWithSetter()
        bad = _DummyPluginWithoutSetter()
        executor = _make_executor_with_registry({"cli": good, "todo": bad})

        attach = lambda: None
        limits = RuntimeLimits(cpu_weight=200)
        executor.set_runtime_limits(attach, limits)

        assert good.calls == [(attach, limits)]

    def test_no_registry_means_no_forwarding(self):
        # When no registry is set, the executor still stores the values
        # — accessor reads remain valid for any caller that pulls them
        # via get_*().
        plugin = _DummyPluginWithSetter()
        executor = ToolExecutor()  # no registry
        executor.set_runtime_limits(lambda: None, RuntimeLimits(pids_max=32))
        # The plugin was never registered, so it received nothing.
        assert plugin.calls == []

    def test_repeat_call_overwrites(self):
        # Re-calling the setter must replace the prior values both on
        # the executor and on the forwarded plugins.
        plugin = _DummyPluginWithSetter()
        executor = _make_executor_with_registry({"cli": plugin})

        executor.set_runtime_limits(None, RuntimeLimits(memory_max_mb=64))
        executor.set_runtime_limits(lambda: None, RuntimeLimits(memory_max_mb=512))

        assert len(plugin.calls) == 2
        # Second call's limits override the first.
        assert plugin.calls[-1][1].memory_max_mb == 512


class TestEventReaderTelemetryInjection:
    """Tests for ``ToolExecutor.execute`` snapshotting cgroup.events
    before/after each tool call and injecting deltas into the result's
    ``_telemetry`` dict.  The session's tool span auto-forwards every
    key in ``_telemetry`` as an OTel attribute, so we don't need to
    test the OTel side here — just that the deltas land in the right
    shape.
    """

    @staticmethod
    def _make_executor_with_reader(reader):
        executor = ToolExecutor()
        # Register a no-op tool that returns an empty dict so we can
        # observe the injected _telemetry keys.
        executor.register("noop", lambda args: {"ok": True})
        executor.set_runtime_limits(None, None, reader)
        return executor

    def test_no_reader_means_no_injection(self):
        # When no event_reader is set, execute() must not invent a
        # _telemetry dict on the result.  This is the IPC / no-cgroups
        # path — runs that don't have a cgroup shouldn't get spurious
        # zero-delta keys.
        executor = ToolExecutor()
        executor.register("noop", lambda args: {"ok": True})
        success, result = executor.execute("noop", {})
        assert success is True
        assert "_telemetry" not in result

    def test_zero_delta_means_no_injection(self):
        # The reader returns the same dict before and after — no
        # kernel events occurred during the tool call, so no delta
        # keys are added.  Keeping spans clean in the common case.
        snapshot = {"oom_kill": 0, "populated": 1}
        executor = self._make_executor_with_reader(lambda: snapshot.copy())
        success, result = executor.execute("noop", {})
        assert success is True
        assert "_telemetry" not in result

    def test_oom_kill_delta_is_injected(self):
        # The reader returns 0 before, 2 after — we should see
        # ``jaato.cgroup.oom_kill_delta = 2`` in the result.
        calls = {"n": 0}

        def reader():
            calls["n"] += 1
            if calls["n"] == 1:
                return {"oom_kill": 0, "populated": 1}
            return {"oom_kill": 2, "populated": 1}

        executor = self._make_executor_with_reader(reader)
        success, result = executor.execute("noop", {})
        assert success is True
        assert result["_telemetry"]["jaato.cgroup.oom_kill_delta"] == 2

    def test_oom_event_delta_is_injected(self):
        # ``oom`` (the cgroup-level OOM-event count) increments when
        # the OOM killer activates inside the cgroup, separate from
        # ``oom_kill`` (per-process count).  Both get surfaced.
        calls = {"n": 0}

        def reader():
            calls["n"] += 1
            if calls["n"] == 1:
                return {"oom": 0, "oom_kill": 0}
            return {"oom": 1, "oom_kill": 3}

        executor = self._make_executor_with_reader(reader)
        success, result = executor.execute("noop", {})
        telem = result["_telemetry"]
        assert telem["jaato.cgroup.oom_delta"] == 1
        assert telem["jaato.cgroup.oom_kill_delta"] == 3

    def test_populated_is_not_treated_as_a_delta(self):
        # ``populated`` is a level (0/1), not a counter — its delta
        # would be noisy and meaningless.  Verify we skip it even
        # when it changes between snapshots.
        calls = {"n": 0}

        def reader():
            calls["n"] += 1
            if calls["n"] == 1:
                return {"populated": 0}
            return {"populated": 1}

        executor = self._make_executor_with_reader(reader)
        success, result = executor.execute("noop", {})
        # No deltas surfaced — populated transitions are skipped.
        assert "_telemetry" not in result

    def test_preserves_existing_telemetry_dict(self):
        # If the tool already produced a ``_telemetry`` dict, our
        # additions must merge in, not overwrite the plugin's keys.
        executor = ToolExecutor()
        executor.register(
            "with_telem",
            lambda args: {"_telemetry": {"plugin.attr": "value"}, "ok": True},
        )
        calls = {"n": 0}

        def reader():
            calls["n"] += 1
            return {"oom_kill": 0 if calls["n"] == 1 else 1}

        executor.set_runtime_limits(None, None, reader)
        success, result = executor.execute("with_telem", {})
        telem = result["_telemetry"]
        # Plugin's attr survives.
        assert telem["plugin.attr"] == "value"
        # Our delta lands alongside it.
        assert telem["jaato.cgroup.oom_kill_delta"] == 1

    def test_reader_returning_none_after_does_not_crash(self):
        # If cgroup is torn down mid-call, the after-snapshot returns
        # None.  Wrapper must skip injection rather than crash on the
        # subtraction.
        calls = {"n": 0}

        def reader():
            calls["n"] += 1
            if calls["n"] == 1:
                return {"oom_kill": 0}
            return None  # cgroup vanished

        executor = self._make_executor_with_reader(reader)
        success, result = executor.execute("noop", {})
        assert success is True
        assert "_telemetry" not in result


# ---------------------------------------------------------------------------
# Phase 5 §5.10c — AppArmor child-profile transition callback forwarding
# ---------------------------------------------------------------------------


class _DummyPluginWithApparmorSetter:
    """Fake plugin that implements set_apparmor_child_transition_callback.

    Mirrors `_DummyPluginWithSetter` for the §5.10c forwarding chain."""

    def __init__(self):
        self.calls = []

    def set_apparmor_child_transition_callback(self, callback):
        self.calls.append(callback)


class TestSetApparmorChildTransitionCallback:
    """Phase 5 §5.10c: the executor stores the callback and forwards
    it to any plugin that implements
    ``set_apparmor_child_transition_callback``.  Same shape as
    :meth:`set_runtime_limits`'s forwarding — plugins that don't
    implement the method are silently skipped."""

    def test_default_state_is_none(self):
        executor = ToolExecutor()
        assert executor.get_apparmor_child_transition_callback() is None

    def test_setter_stores_callback(self):
        executor = ToolExecutor()
        cb = lambda: None
        executor.set_apparmor_child_transition_callback(cb)
        assert executor.get_apparmor_child_transition_callback() is cb

    def test_setter_accepts_none(self):
        """Passing None clears a previously installed callback."""
        executor = ToolExecutor()
        executor.set_apparmor_child_transition_callback(lambda: None)
        executor.set_apparmor_child_transition_callback(None)
        assert executor.get_apparmor_child_transition_callback() is None

    def test_forwards_to_plugin_with_setter(self):
        plugin = _DummyPluginWithApparmorSetter()
        executor = _make_executor_with_registry({"cli": plugin})

        cb = lambda: None
        executor.set_apparmor_child_transition_callback(cb)

        assert plugin.calls == [cb]

    def test_skips_plugin_without_setter(self):
        """A plugin lacking the method (todo, file_edit, ...) is
        silently skipped — only subprocess-spawning plugins care
        about the //child transition."""
        plugin = _DummyPluginWithoutSetter()
        executor = _make_executor_with_registry({"todo": plugin})
        executor.set_apparmor_child_transition_callback(lambda: None)
        # No crash, no AttributeError surfaced.

    def test_forwards_only_to_plugins_with_setter(self):
        good = _DummyPluginWithApparmorSetter()
        bad = _DummyPluginWithoutSetter()
        executor = _make_executor_with_registry(
            {"cli": good, "todo": bad},
        )

        cb = lambda: None
        executor.set_apparmor_child_transition_callback(cb)

        assert good.calls == [cb]

    def test_repeat_call_overwrites(self):
        plugin = _DummyPluginWithApparmorSetter()
        executor = _make_executor_with_registry({"cli": plugin})

        cb1 = lambda: None
        cb2 = lambda: None
        executor.set_apparmor_child_transition_callback(cb1)
        executor.set_apparmor_child_transition_callback(cb2)

        assert plugin.calls == [cb1, cb2]


class TestCliPluginPreexecComposition:
    """Phase 5 §5.10c: the cli plugin composes apparmor + cgroup
    callbacks into a single preexec_fn.  Apparmor-first ordering per
    §6.1 of the audit doc — the new profile must apply during the
    cgroup write."""

    def _make_cli_plugin(self):
        from shared.plugins.cli.plugin import CLIToolPlugin
        return CLIToolPlugin()

    def test_both_none_returns_none(self):
        """No callbacks installed → preexec_fn is None (today's
        pre-§5.10c Popen behavior)."""
        plugin = self._make_cli_plugin()
        assert plugin._build_subprocess_preexec_fn() is None

    def test_only_cgroup_returns_cgroup_directly(self):
        """Cgroup-only path — no apparmor wrapping overhead."""
        plugin = self._make_cli_plugin()
        cgroup_cb = lambda: None
        plugin.set_runtime_limits(cgroup_cb, None)
        assert plugin._build_subprocess_preexec_fn() is cgroup_cb

    def test_only_apparmor_returns_apparmor_directly(self):
        """Apparmor-only path — e.g., a confined runner without
        kernel runtime_limits configured."""
        plugin = self._make_cli_plugin()
        apparmor_cb = lambda: None
        plugin.set_apparmor_child_transition_callback(apparmor_cb)
        assert plugin._build_subprocess_preexec_fn() is apparmor_cb

    def test_both_compose_apparmor_first(self):
        """Composite preexec_fn runs apparmor FIRST, then cgroup.
        Order matters per §6.1 of the audit doc."""
        plugin = self._make_cli_plugin()
        call_order = []

        def apparmor_cb():
            call_order.append("apparmor")

        def cgroup_cb():
            call_order.append("cgroup")

        plugin.set_runtime_limits(cgroup_cb, None)
        plugin.set_apparmor_child_transition_callback(apparmor_cb)

        composite = plugin._build_subprocess_preexec_fn()
        assert composite is not None
        composite()
        assert call_order == ["apparmor", "cgroup"]

    def test_apparmor_failure_propagates(self):
        """Fail-closed: an apparmor transition failure must propagate
        as an exception so Popen treats it as a spawn failure.  A
        silent failure would leave the child in the parent profile
        with escape rules intact — exactly the gap §5.10 closes."""
        plugin = self._make_cli_plugin()

        def apparmor_cb():
            raise OSError("EACCES")

        def cgroup_cb():
            # Must not be reached.
            raise AssertionError("cgroup ran after apparmor failed")

        plugin.set_runtime_limits(cgroup_cb, None)
        plugin.set_apparmor_child_transition_callback(apparmor_cb)

        composite = plugin._build_subprocess_preexec_fn()
        try:
            composite()
        except OSError as exc:
            assert "EACCES" in str(exc)
        else:
            raise AssertionError(
                "apparmor failure must propagate (fail-closed)",
            )

    def test_set_apparmor_callback_accepts_none(self):
        """Passing None clears a previously installed callback —
        cleanup path for sessions transitioning between confined and
        unconfined modes (rare but supported)."""
        plugin = self._make_cli_plugin()
        plugin.set_apparmor_child_transition_callback(lambda: None)
        plugin.set_apparmor_child_transition_callback(None)
        assert plugin._apparmor_child_transition is None
