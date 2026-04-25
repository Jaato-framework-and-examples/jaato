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
