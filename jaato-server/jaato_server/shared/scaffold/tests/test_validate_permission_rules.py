"""The permission whitelist is checked against the same inventory ``tool_scopes`` is.

One key over from a check that has existed since the validator shipped, and
the key where getting it wrong is expensive: under ``defaultPolicy: deny`` a
name that matches nothing is a permanent denial of a tool the author believes
they approved, and the runtime symptom is the one #951 exists to make legible
— the call reaches the gate and vanishes.  A whitelist entry is also the
cheapest thing in a profile to misspell, because nothing else in the file
repeats the name.
"""

from __future__ import annotations

import pytest

from jaato_server.shared.plugins.subagent.config import SubagentProfile
from jaato_server.shared.scaffold import introspect, validate


@pytest.fixture(scope="module")
def registry():
    return introspect.providers(), introspect.plugins()


def _run(registry, plugins, policy):
    providers, plugin_info = registry
    profile = SubagentProfile(
        name="t", description="d", plugins=list(plugins),
        plugin_configs={"permission": {"policy": policy}})
    return validate.validate_profile(
        profile, providers=providers, plugins=plugin_info, gc_names=[])


def _codes(diags):
    return [(d.code, d.message) for d in diags
            if d.code in ("unknown_tool", "permission_rule_without_plugin")]


def test_a_misspelled_whitelist_entry_is_reported(registry):
    out = _codes(_run(registry, ["cli"], {
        "defaultPolicy": "deny",
        "whitelist": {"tools": ["cli_based_tool", "cli_based_tol"]}}))
    assert len(out) == 1
    assert out[0][0] == "unknown_tool"
    assert "cli_based_tol" in out[0][1]


def test_a_correct_whitelist_is_silent(registry):
    assert _codes(_run(registry, ["cli", "file_edit"], {
        "defaultPolicy": "deny",
        "whitelist": {"tools": ["cli_based_tool", "writeNewFile"]}})) == []


def test_a_whitelist_for_a_plugin_not_enabled_is_reported(registry):
    # The 2026-09 shape: `plugin_configs.permission.whitelist` gained
    # writeNewFile and `plugins:` never gained file_edit.
    out = _codes(_run(registry, ["cli"], {
        "whitelist": {"tools": ["writeNewFile"]}}))
    assert [c for c, _ in out] == ["permission_rule_without_plugin"]
    assert "file_edit" in out[0][1]


def test_a_blacklist_for_a_plugin_not_enabled_is_silent(registry):
    # Denying a tool the profile does not enable is defence in depth, and a
    # profile that adds the plugin later keeps the protection it wrote.
    assert _codes(_run(registry, ["cli"], {
        "blacklist": {"tools": ["writeNewFile"]}})) == []


def test_a_blacklist_typo_is_still_reported(registry):
    out = _codes(_run(registry, ["cli"], {
        "blacklist": {"tools": ["writeNewFil"]}}))
    assert [c for c, _ in out] == ["unknown_tool"]


def test_framework_session_tools_are_accepted(registry):
    # signal_completion belongs to no entry in plugins: — it is wired by
    # JaatoSession.configure() — so whitelisting it must not read as a typo.
    assert _codes(_run(registry, ["cli"], {
        "whitelist": {"tools": ["signal_completion", "askPermission"]}})) == []


def test_mcp_shaped_names_are_exempt(registry):
    # An MCP tool inventory comes from the servers a LIVE session connects to.
    assert _codes(_run(registry, ["mcp"], {
        "whitelist": {"tools": ["mcp__Atlassian__jira_search",
                                "mcp.atlassian.jira_search"]}})) == []


def test_an_abstract_base_is_silent_about_enablement(registry):
    # A profile declaring no plugins declares no surface, so it cannot be
    # said to be missing one — the carve-out `missing_model` already uses.
    out = _codes(_run(registry, [], {"whitelist": {"tools": ["writeNewFile"]}}))
    assert [c for c, _ in out] == []


def test_patterns_are_not_read_as_tool_names(registry):
    # A glob is not a tool name and nothing here could tell a deliberate
    # wildcard from a typo.
    assert _codes(_run(registry, ["cli"], {
        "whitelist": {"patterns": ["cli_*", "nonsense_*"]}})) == []


def test_a_malformed_policy_is_not_a_crash(registry):
    for policy in ("nonsense", [], {"whitelist": "nope"},
                   {"whitelist": {"tools": "nope"}}):
        _run(registry, ["cli"], policy)      # must not raise


def test_a_dynamic_plugin_silences_the_whole_check(registry, monkeypatch):
    # A live-session inventory could supply any of these names, so nothing
    # here is knowable — the rule `tool_scopes` has always applied to one
    # plugin at a time, widened to the profile.
    providers, plugins = registry
    patched = dict(plugins)
    patched["cli"] = type(plugins["cli"])(**{
        **{f: getattr(plugins["cli"], f)
           for f in plugins["cli"].__dataclass_fields__},
        "dynamic": True})
    profile = SubagentProfile(
        name="t", description="d", plugins=["cli"],
        plugin_configs={"permission": {"policy": {
            "whitelist": {"tools": ["utterly_made_up"]}}}})
    diags = validate.validate_profile(profile, providers=providers,
                                      plugins=patched, gc_names=[])
    assert _codes(diags) == []
