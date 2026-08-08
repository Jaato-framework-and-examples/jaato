"""get_exposed_tool_schemas(exclude_runner_tier=True) skips runner-tier plugins.

Re-attach loop-stall fix: the daemon-side tool-id-registry emit must NOT invoke
runner-tier plugins' get_tool_schemas (e.g. prompt_library's filesystem walk) on
the event-loop thread — runner-tier names come from the runner via
session_get_tool_schemas.  The opt-in flag keeps every OTHER caller unchanged.
"""
from types import SimpleNamespace

from shared.plugins.registry import PluginRegistry


def _make_registry():
    reg = PluginRegistry()
    reg._core_tools = {}

    class RunnerP:
        name = "rp"

        def get_tool_schemas(self):
            return [SimpleNamespace(name="runner_tool", category=None)]

    class DaemonP:
        name = "dp"

        def get_tool_schemas(self):
            return [SimpleNamespace(name="daemon_tool", category=None)]

    # _lookup_module_tier resolves PLUGIN_TIER from the plugin's module name.
    RunnerP.__module__ = "fake.runner.plugin"
    DaemonP.__module__ = "fake.daemon.plugin"
    reg._plugins = {"rp": RunnerP(), "dp": DaemonP()}
    reg._exposed = {"rp", "dp"}
    # Instance attr shadows the staticmethod (plain function — not bound).
    reg._lookup_module_tier = lambda m: "runner" if "runner" in m else "daemon"
    return reg


def test_default_includes_all_tiers():
    reg = _make_registry()
    names = {s.name for s in reg.get_exposed_tool_schemas()}
    assert names == {"runner_tool", "daemon_tool"}


def test_exclude_runner_tier_skips_runner_plugins():
    reg = _make_registry()
    names = {s.name for s in reg.get_exposed_tool_schemas(exclude_runner_tier=True)}
    assert names == {"daemon_tool"}
    assert "runner_tool" not in names
