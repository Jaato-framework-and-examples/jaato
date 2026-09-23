"""``explain plugin <name>`` descends a knob whose value is a declared object.

The knob this is about is ``permission.policy``.  Its whole vocabulary —
``defaultPolicy`` and its enum, ``whitelist.tools``, the ``sanitization``
tree — has always been machine-readable in ``get_config_schema()``, and the
page rendered one line reading ``policy  object  Permission policy rules``.
The only honest route to the real shape was
``shared/plugins/permission/policy.py``, which is the one thing this page
exists to make unnecessary; a 2026-09 workspace bring-up took exactly that
route.

Nesting is rendered at every declared depth, unlike a tool PARAMETER, which
stops at one (``_nested_params``).  The reader's need is the opposite: a
nested parameter is one call's argument, while ``policy`` IS the plugin's
whole configuration surface.
"""

from __future__ import annotations

import pytest

from jaato_server.shared.scaffold import explain, introspect


@pytest.fixture(scope="module")
def page():
    return explain.plugin("permission")


def _by_name(settings, name):
    for s in settings:
        if s["name"] == name:
            return s
    raise AssertionError(f"{name} not in {[s['name'] for s in settings]}")


def test_the_policy_tree_reaches_the_text(page):
    _data, text = page
    for token in ("defaultPolicy", "whitelist", "blacklist", "sanitization",
                  "path_scope", "allowed_roots"):
        assert token in text, token


def test_a_nested_enum_is_rendered_because_it_is_checked(page):
    _data, text = page
    assert "one of: 'allow', 'deny', 'ask'" in text


def test_the_json_view_carries_the_same_tree(page):
    data, _text = page
    policy = _by_name(data["config"], "policy")
    default = _by_name(policy["children"], "defaultPolicy")
    assert default["enum"] == ["allow", "deny", "ask"]
    scope = _by_name(_by_name(policy["children"], "sanitization")["children"],
                     "path_scope")
    assert {c["name"] for c in scope["children"]} >= {"allowed_roots",
                                                      "block_absolute"}


def test_an_open_key_set_is_marked_not_silently_bottomed_out(page):
    data, text = page
    assert _by_name(data["config"], "evaluators")["free_form"] is True
    assert "open key set" in text


def test_a_scalar_knob_carries_no_children(page):
    data, _text = page
    assert _by_name(data["config"], "emit_decision_events")["children"] is None


def test_descent_is_depth_capped():
    # A plugin declaring a pathological or recursive schema must not hang an
    # `explain`; the cap is the deepest real declaration plus headroom.
    spec = {"type": "object", "properties": {}}
    node = spec
    for _ in range(40):
        child = {"type": "object", "properties": {}}
        node["properties"]["deeper"] = child
        node = child
    settings = introspect._settings_from_properties(spec["properties"])
    depth = 0
    cur = settings
    while cur:
        depth += 1
        cur = cur[0].children
    assert depth <= introspect._MAX_KNOB_DEPTH + 1


# --------------------------------------------- the gate that is not a key

def test_the_client_gate_is_probed_not_spelled():
    gate = introspect.client_gate()
    # A headless driver completes; the interactive surfaces do not.
    assert gate["keeps"] == ["api"]
    assert set(gate["hides"]) >= {"terminal", "web"}


def test_both_pages_state_the_client_gate():
    for _data, text in (explain.completion(), explain.lifecycle()):
        assert "NOT A PROFILE KEY" in text
        assert "api" in text


def test_the_lifecycle_json_carries_it():
    data, _text = explain.lifecycle()
    assert data["client_gate"]["keeps"] == ["api"]
