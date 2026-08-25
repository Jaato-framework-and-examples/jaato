"""``explain plugin`` must publish a tool's PARAMETERS, not only its prose.

Without them a tool SIGNATURE is not reachable from the CLI: a consumer can
compare names and descriptions but not arguments.  That is not hypothetical —
the cascade-coordination example tried to validate its published spec against
the framework, found ``explain plugin --json`` surfaced name, discoverability
and description only, and had to anchor its guard elsewhere.  Four drifts had
already survived in that public spec:

    a parameter that was never implemented   (wake=False)
    a renamed one                            (name -> sibling_name)
    two stale return shapes                  ({delivered, status}, bare list)

None of them were checkable against the framework, so none of them failed.

A second copy of a fact rots unless something executes the comparison — and
the comparison could not be executed because the CLI did not publish the fact.
"""

import json

import pytest

from shared.scaffold import explain, introspect


def _tool(plugin_name, tool_name):
    data, _text = explain.plugin(plugin_name)
    return next((t for t in data["tools"] if t["name"] == tool_name), None)


def test_a_tools_parameters_are_published():
    t = _tool("subagent", "send_to_sibling")
    assert t is not None
    assert t["parameters"]["properties"].keys() == {"sibling_name", "message"}
    assert t["parameters"]["required"] == ["sibling_name", "message"]


def test_the_published_signature_matches_the_live_schema():
    """The claim a consumer relies on: this IS the shipped signature.

    Compared against ``get_tool_schemas()`` rather than a copy, because a
    guard that compares two documents proves only that they agree.
    """
    from shared.plugins.subagent.plugin import SubagentPlugin
    live = {s.name: s for s in SubagentPlugin().get_tool_schemas()}
    data, _ = explain.plugin("subagent")
    for entry in data["tools"]:
        schema = live.get(entry["name"])
        if schema is None:      # dynamic/registry-only tools
            continue
        assert entry["parameters"] == schema.parameters, entry["name"]


def test_no_arguments_and_unknown_arguments_are_different_facts():
    """``{}`` means "takes none"; ``None`` means "the schema omitted it".

    Collapsing them would be the absent-vs-empty defect in the very surface
    built to let consumers detect drift.
    """
    assert explain._signature({"type": "object", "properties": {}}) == "()"
    assert explain._signature(None) == "(?)"


def test_the_rendering_distinguishes_required_from_optional():
    sig = explain._signature({
        "type": "object",
        "properties": {"query": {}, "max_results": {}},
        "required": ["query"],
    })
    assert sig == "(query, max_results=...)"


def test_the_human_rendering_shows_the_signature():
    _data, text = explain.plugin("subagent")
    assert "send_to_sibling(sibling_name, message)" in text


def test_introspection_copies_rather_than_shares_the_schema():
    """A consumer mutating what it was handed must not reshape the plugin."""
    PL = introspect.plugins()
    tool = next(t for t in PL["subagent"].tools if t.name == "send_to_sibling")
    tool.parameters["properties"]["injected"] = {"type": "string"}

    from shared.plugins.subagent.plugin import SubagentPlugin
    live = next(s for s in SubagentPlugin().get_tool_schemas()
                if s.name == "send_to_sibling")
    assert "injected" not in live.parameters["properties"]


def test_the_json_surface_round_trips():
    """It is consumed as JSON by a CLI caller, so it must serialize."""
    data, _ = explain.plugin("subagent")
    back = json.loads(json.dumps(data, default=str))
    entry = next(t for t in back["tools"] if t["name"] == "send_to_sibling")
    assert entry["parameters"]["required"] == ["sibling_name", "message"]
