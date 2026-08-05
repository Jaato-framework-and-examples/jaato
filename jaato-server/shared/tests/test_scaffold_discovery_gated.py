"""Discovery-gated tool surfacing in `validate`.

A tool is in the model's INITIAL schema iff it is [core] OR its plugin is
(preload)-ed.  [disc] tools of non-preloaded plugins are DEFERRED — reachable
only after the model calls list_tools/get_tool_schemas (introspection is always
core, so they're never lost).  `validate` surfaces them as an INFO diagnostic
(not a defect), so an author who assumed a tool was immediately available isn't
surprised.  See `explain plugin <name>` for the matching legend.
"""

from types import SimpleNamespace

from shared.scaffold import validate as V
from shared.scaffold import introspect as I


def _plugins():
    return {
        "cli": I.PluginInfo(name="cli", tools=[
            I.ToolInfo(name="cli_based_tool", discoverability="discoverable")]),
        "todo": I.PluginInfo(name="todo", tools=[
            I.ToolInfo(name="createPlan", discoverability="core")]),
    }


def _profile(plugins, preloaded=frozenset(), tool_scopes=None):
    return SimpleNamespace(
        name="p", provider=None, model=None,
        plugins=list(plugins), preloaded_plugins=set(preloaded),
        tool_scopes=tool_scopes or {}, plugin_configs={}, quirks=None, gc=None)


def _codes(diags):
    return {d.code: d for d in diags}


def test_discoverable_tool_not_preloaded_is_flagged():
    c = _codes(V.validate_profile(_profile(["cli"]), providers={},
                                  plugins=_plugins(), gc_names=[]))
    assert "discovery_gated_tools" in c
    assert c["discovery_gated_tools"].severity == "info"      # advisory, not a defect
    assert "cli_based_tool" in c["discovery_gated_tools"].message


def test_preloaded_plugin_tools_not_gated():
    c = _codes(V.validate_profile(_profile(["cli"], preloaded={"cli"}),
                                  providers={}, plugins=_plugins(), gc_names=[]))
    assert "discovery_gated_tools" not in c


def test_core_only_plugin_not_gated():
    c = _codes(V.validate_profile(_profile(["todo"]), providers={},
                                  plugins=_plugins(), gc_names=[]))
    assert "discovery_gated_tools" not in c


def test_disc_tool_scoped_out_is_not_gated():
    # cli's only [disc] tool is scoped OUT → not exposed at all → not "gated".
    c = _codes(V.validate_profile(
        _profile(["cli"], tool_scopes={"cli": ["nonexistent"]}),
        providers={}, plugins=_plugins(), gc_names=[]))
    assert "discovery_gated_tools" not in c
