"""``validate`` says when a plugin is configured but not enabled (#950).

Configuring a plugin and putting its tools on the model's wire are two
decisions, and only the second is spelled ``plugins:``.  Since #950 the config
block IS applied either way, so this is not a "the block does nothing"
finding — it is the narrower one that survives that fix: the author wrote
``plugin_configs.memory`` and the agent has no memory tools, which is almost
always half of a pair.

The exemptions are the point of the check being usable at all.  A plugin that
exposes NO tools is configured-only by construction — ``permission``'s
``get_tool_schemas()`` returns ``[]`` on purpose ("askPermission is not
exposed to the model"), and ``sandbox_manager`` likewise — so naming it in
``plugins:`` would be ceremony and flagging it would be noise on nine in-tree
profiles that are correct as written.
"""
from types import SimpleNamespace

import pytest

from shared.scaffold import introspect
from shared.scaffold.validate import _check_plugin_configs_expose_tools


@pytest.fixture(scope="module")
def plugins():
    return introspect.plugins()


def _check(plugins, enabled, configs):
    prof = SimpleNamespace(plugins=list(enabled), plugin_configs=dict(configs))
    out = []

    def add(severity, code, message, where=None):
        out.append(SimpleNamespace(severity=severity, code=code,
                                   message=message, where=where))

    _check_plugin_configs_expose_tools(prof, plugins, add)
    return out


def _codes(diags):
    return [d.code for d in diags]


# ------------------------------------------------------------------ it fires

def test_a_tool_bearing_plugin_configured_but_not_enabled_is_flagged(plugins):
    found = _check(plugins, ["cli"], {"memory": {"storage": "sqlite"}})

    assert _codes(found) == ["plugin_config_without_plugin"]
    assert found[0].severity == "warn"
    assert found[0].where == "plugin_configs.memory"


def test_the_message_names_the_tools_the_agent_cannot_call(plugins):
    found = _check(plugins, [], {"memory": {}})

    assert "store_memory" in found[0].message
    assert "plugins:" in found[0].message


def test_a_plugin_whose_tools_are_not_statically_knowable_is_left_alone(plugins):
    """``mcp``'s tools come from servers a live session connects to, so
    offline introspection reports none.  The check reads that the same way it
    reads a genuinely toolless plugin and stays quiet — a false negative, and
    the right one: the alternative is asserting a missing tool surface the
    validator cannot see."""
    assert _check(plugins, ["cli"], {"mcp": {"workspace_path": "/ws"}}) == []


# --------------------------------------------------------------- it stays quiet

def test_an_enabled_plugin_is_not_flagged(plugins):
    assert _check(plugins, ["memory"], {"memory": {}}) == []


def test_a_toolless_plugin_is_never_flagged(plugins):
    """``permission`` is the whole reason the check needs this exemption."""
    policy = {"policy": {"defaultPolicy": "ask",
                         "whitelist": {"tools": ["writeNewFile"]}}}
    assert _check(plugins, ["cli"], {"permission": policy}) == []
    assert _check(plugins, ["cli"], {"sandbox_manager": {"x": 1}}) == []


def test_introspection_is_never_flagged(plugins):
    """Its tools are core: they reach the wire whatever ``plugins:`` says."""
    assert _check(plugins, ["cli"], {"introspection": {}}) == []


def test_a_provider_section_is_not_a_plugin(plugins):
    configs = {"openrouter": {"api_key": "sk-or-x"},
               "anthropic": {"api_params": {"temperature": 0.0}}}
    assert _check(plugins, ["cli"], configs) == []


def test_an_uninstalled_name_is_left_to_the_knob_checks(plugins):
    """Nothing reliable to say about a plugin this tree does not have."""
    assert _check(plugins, ["cli"], {"totally_not_a_plugin": {}}) == []


# ------------------------------------------------------- the in-tree profiles

def test_the_frameworks_own_smoke_profiles_stay_clean(plugins):
    """Every in-tree profile carrying an undeclared config carries a toolless
    one (``permission`` / ``sandbox_manager``).  A check that flagged those
    would be the "actively misleading" shape #947 warns against."""
    configs = {"permission": {"policy": {"defaultPolicy": "allow"}},
               "sandbox_manager": {"session_id": "s1"},
               "anthropic": {"api_key": "${ANTHROPIC_API_KEY}"}}
    assert _check(plugins, ["cli"], configs) == []
