"""Scaffold coverage for NON-provider plugin config knobs.

Closes the gap where tool-plugins that return a raw JSON-schema dict from
``get_config_schema`` (permission / cli / notebook / interactive_shell) were
invisible to introspect/explain and unchecked by validate — so e.g.
``plugin_configs.permission.evaluators`` was neither surfaced nor typo-checked.
"""
from types import SimpleNamespace

from shared.scaffold import introspect, explain
from shared.scaffold.validate import validate_profile


def _validate(plugin_configs, plugins=None):
    prof = SimpleNamespace(provider=None, model="m", plugin_configs=plugin_configs)
    return validate_profile(
        prof, providers=introspect.providers(),
        plugins=plugins if plugins is not None else introspect.plugins(),
        gc_names=list(introspect.gc_strategies().keys()))


def _unknown_knobs(diags):
    return [d.where for d in diags if d.code == "unknown_knob"]


# ---- introspect: raw-dict get_config_schema is normalized -------------------

def test_introspect_surfaces_permission_raw_dict_knobs():
    perm = introspect.plugins().get("permission")
    assert perm is not None
    assert "evaluators" in perm.config_keys
    assert "policy" in perm.config_keys


def test_introspect_surfaces_cli_workspace_venv():
    cli = introspect.plugins().get("cli")
    assert cli is not None and "workspace_venv" in cli.config_keys


# ---- explain plugin <name> now lists the knobs ------------------------------

def test_explain_plugin_permission_lists_evaluators():
    data, text = explain.plugin("permission")
    names = {c["name"] for c in data.get("config", [])}
    assert {"evaluators", "policy"} <= names
    assert "evaluators" in text


# ---- validate: top-level knob names checked against the schema --------------

def test_permission_evaluators_and_policy_accepted():
    diags = _validate({"permission": {"evaluators": {"default": "e.py"},
                                       "policy": {"defaultPolicy": "ask"}}})
    assert _unknown_knobs(diags) == []


def test_permission_typo_knob_flagged():
    diags = _validate({"permission": {"evaluatorss": {}}})
    assert any("evaluatorss" in w for w in _unknown_knobs(diags))
    # warn, not error — plugin schemas may be incomplete.
    assert all(d.severity == "warn" for d in diags
               if d.code == "unknown_knob" and "evaluatorss" in (d.where or ""))


def test_cli_workspace_venv_accepted_typo_flagged():
    assert _unknown_knobs(_validate({"cli": {"workspace_venv": "v"}})) == []
    assert any("workspace_venvv" in w for w in
               _unknown_knobs(_validate({"cli": {"workspace_venvv": "v"}})))


def test_nested_and_freeform_substructures_not_descended():
    # permission.evaluators is a free-form tool->path map; permission.policy is
    # a deep tree.  Only top-level knob names are checked — inner keys pass.
    diags = _validate({"permission": {
        "evaluators": {"cli_based_tool": "x.py", "made_up_tool": "y.py"},
        "policy": {"defaultPolicy": "deny", "whatever_inner": {"a": 1}},
    }})
    assert _unknown_knobs(diags) == []


def test_unknown_plugin_name_is_skipped_not_crashed():
    # A plugin_configs block for a plugin that doesn't exist: no crash, and no
    # knob diagnostics (nothing to validate against).
    diags = _validate({"totally_made_up_plugin": {"foo": "bar"}})
    assert _unknown_knobs(diags) == []
