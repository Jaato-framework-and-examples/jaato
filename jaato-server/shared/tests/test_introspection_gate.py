"""Introspection's discovery tools are gated on the wire, not just on discovery.

`list_tools`/`get_tool_schemas` are [core], so they start in the initial schema.
`_has_deferred_to_discover` is the pure predicate for "is something pending
discovery"; `_should_drop_introspection` is the actual drop decision layered on
top — it keeps introspection whenever real tools exist (so the model can
re-inspect them after GC offloads their schemas), and drops it ONLY for a truly
empty wire (e.g. `plugins=[]`).  This locks in both truth tables.
"""

from types import SimpleNamespace

from shared.jaato_session import (
    _has_deferred_to_discover,
    _should_drop_introspection,
)


def _sc(name, disc="discoverable"):
    return SimpleNamespace(name=name, discoverability=disc)


def _plugin_of(mapping):
    return lambda n: mapping.get(n)


def test_discoverable_nonpreloaded_tool_is_something_to_discover():
    assert _has_deferred_to_discover(
        [_sc("cli_based_tool")], ["cli"], set(), {},
        _plugin_of({"cli_based_tool": "cli"})) is True


def test_all_core_is_nothing_to_discover():
    assert _has_deferred_to_discover(
        [_sc("createPlan", "core"), _sc("list_tools", "core")],
        ["todo"], set(), {}, _plugin_of({"createPlan": "todo"})) is False


def test_preloaded_plugin_tool_does_not_count():
    assert _has_deferred_to_discover(
        [_sc("cli_based_tool")], ["cli"], {"cli"}, {},
        _plugin_of({"cli_based_tool": "cli"})) is False


def test_tool_outside_profile_does_not_count():
    assert _has_deferred_to_discover(
        [_sc("cli_based_tool")], ["todo"], set(), {},
        _plugin_of({"cli_based_tool": "cli"})) is False


def test_scoped_out_tool_does_not_count():
    assert _has_deferred_to_discover(
        [_sc("cli_based_tool")], ["cli"], set(), {"cli": ["other"]},
        _plugin_of({"cli_based_tool": "cli"})) is False


def test_introspection_tools_never_count_as_discoverable():
    # Even if (oddly) marked discoverable, they aren't "something to discover".
    assert _has_deferred_to_discover(
        [_sc("list_tools"), _sc("get_tool_schemas")],
        ["introspection"], set(), {},
        _plugin_of({"list_tools": "introspection",
                    "get_tool_schemas": "introspection"})) is False


def test_mixed_keeps_when_any_discoverable_present():
    schemas = [_sc("createPlan", "core"), _sc("cli_based_tool", "discoverable")]
    assert _has_deferred_to_discover(
        schemas, ["todo", "cli"], {"todo"}, {},
        _plugin_of({"createPlan": "todo", "cli_based_tool": "cli"})) is True


# --- _should_drop_introspection: the drop decision (wire-aware) ---

_INTRO = ["list_tools", "get_tool_schemas"]


def test_drop_when_empty_wire_nothing_deferred():
    # plugins=[] shape: only introspection on the wire, nothing deferred -> drop.
    assert _should_drop_introspection(False, list(_INTRO)) is True


def test_keep_when_real_eager_tools_even_if_nothing_deferred():
    # Preloaded/eager lead (e.g. spawn_subagent on the wire) with nothing
    # deferred -> KEEP introspection so the model can re-inspect after GC offload.
    assert _should_drop_introspection(
        False, _INTRO + ["spawn_subagent", "signal_completion"]) is False


def test_keep_when_something_deferred_regardless_of_wire():
    # Something pending discovery -> always keep, even with no eager real tools.
    assert _should_drop_introspection(True, list(_INTRO)) is False
    assert _should_drop_introspection(True, _INTRO + ["cli"]) is False


def test_drop_when_wire_has_only_introspection_tools():
    # No real tools at all -> empty wire -> drop.
    assert _should_drop_introspection(False, ["list_tools"]) is True
    assert _should_drop_introspection(False, []) is True
