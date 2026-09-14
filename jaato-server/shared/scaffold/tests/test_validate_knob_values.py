"""Scaffold coverage for a knob's VALUE, not only its name (#925).

``validate`` already read every plugin's ``get_config_schema()`` and already
kept each knob's declared ``type`` — then checked the knob NAMES and threw the
rest away.  So a knob violating the plugin's own declared ``enum`` validated
clean and exited 0, and the failure surfaced only at runtime as a silent
fallback: ``todo.storage_type: sqlite`` raises inside ``create_storage``, is
caught, printed to daemon stdout, and replaced by in-memory storage.  An
operator who asked for persistence got none, and nothing failed.

The two severities are deliberately different, and each test below says which
it asserts and why:

* an ``enum`` violation is an **error** — a plugin spelling out
  ``["memory","file","hybrid"]`` has left no incompleteness to be generous
  about;
* a ``type`` mismatch is a **warn** — YAML scalar typing is easy to trip over
  and a declared type can be an incomplete summary.

The check is generic (driven by the declaration a plugin already publishes),
which is what makes it reachable from an OUT-OF-TREE plugin: the alternative,
``_PLUGIN_VALUE_CHECKS``, is a hardcoded jaato-server dict keyed by plugin
name that a third-party distribution cannot register into.
"""
from types import SimpleNamespace

import pytest

from shared.scaffold import explain, introspect
from shared.scaffold.validate import validate_profile


@pytest.fixture(scope="module")
def env():
    return (introspect.providers(), introspect.plugins(),
            list(introspect.gc_strategies().keys()))


def _validate(env, plugin_configs):
    providers, plugins, gc_names = env
    prof = SimpleNamespace(provider=None, model="m",
                           plugin_configs=plugin_configs)
    return validate_profile(prof, providers=providers, plugins=plugins,
                            gc_names=gc_names)


def _find(diags, code, needle):
    return [d for d in diags if d.code == code and needle in (d.where or "")]


def _value_diags(diags):
    return [d for d in diags
            if d.code in ("invalid_knob_value", "knob_type_mismatch")]


# ---- introspect keeps the declaration ---------------------------------------

def test_declared_enum_survives_introspection():
    """``enum`` was the one field discarded at introspect time."""
    todo = introspect.plugins()["todo"]
    setting = next(s for s in todo.config_settings if s.name == "storage_type")
    assert setting.enum == ["memory", "file", "hybrid"]


def test_a_union_type_is_not_rendered_as_a_python_repr():
    """``"type": ["string","array"]`` — ``str()`` on it yields a repr."""
    cli = introspect.plugins()["cli"]
    setting = next(s for s in cli.config_settings
                   if s.name == "scrub_secret_env")
    assert setting.type == "string|array"


def test_a_knob_declaring_no_enum_reports_none_not_empty():
    """``None`` is "declares no enum"; ``[]`` would mean "permits nothing"."""
    web_fetch = introspect.plugins()["web_fetch"]
    setting = next(s for s in web_fetch.config_settings if s.name == "timeout")
    assert setting.enum is None


def test_explain_lists_the_permitted_values():
    """The listing is where an author reads the set validate enforces."""
    data, text = explain.plugin("todo")
    entry = next(c for c in data["config"] if c["name"] == "storage_type")
    assert entry["enum"] == ["memory", "file", "hybrid"]
    assert "'hybrid'" in text


# ---- the issue's own repro ---------------------------------------------------

def test_the_five_invalid_knobs_from_the_issue_are_all_reported(env):
    """Every knob here violates the plugin's OWN schema; all five were clean."""
    diags = _validate(env, {
        "todo": {"storage_type": "sqlite"},          # enum
        "references": {"lookup_strategy": "vibes"},  # enum
        "cli": {"max_output_chars": "lots"},         # integer
        "web_fetch": {"timeout": "thirty",           # integer
                      "follow_redirects": "maybe"},  # boolean
    })
    assert len(_find(diags, "invalid_knob_value", "todo.storage_type")) == 1
    assert len(_find(diags, "invalid_knob_value",
                     "references.lookup_strategy")) == 1
    assert len(_find(diags, "knob_type_mismatch", "cli.max_output_chars")) == 1
    assert len(_find(diags, "knob_type_mismatch", "web_fetch.timeout")) == 1
    assert len(_find(diags, "knob_type_mismatch",
                     "web_fetch.follow_redirects")) == 1


def test_an_enum_violation_is_an_error_and_names_the_permitted_values(env):
    diags = _find(_validate(env, {"todo": {"storage_type": "sqlite"}}),
                  "invalid_knob_value", "storage_type")
    assert [d.severity for d in diags] == ["error"]
    assert "'memory'" in diags[0].message and "'hybrid'" in diags[0].message


def test_a_type_mismatch_is_a_warn_not_an_error(env):
    diags = _find(_validate(env, {"web_fetch": {"timeout": "thirty"}}),
                  "knob_type_mismatch", "timeout")
    assert [d.severity for d in diags] == ["warn"]


def test_a_valid_value_is_clean(env):
    assert _value_diags(_validate(env, {
        "todo": {"storage_type": "file"},
        "references": {"lookup_strategy": "tags_only", "preselected": ["a"]},
        "web_fetch": {"timeout": 30, "follow_redirects": True,
                      "secret_host_bindings": {"T": ["h.example"]}},
    })) == []


# ---- what the check must NOT judge ------------------------------------------

@pytest.mark.parametrize("value", ["${TODO_STORAGE}", "pass://jaato/storage",
                                   "vault://secret/x#k"])
def test_a_deferred_value_is_not_judged(env, value):
    """``${VAR}`` and secret URIs are resolved later, against an environment
    the validator does not have.  Their literal form is a ``str`` whatever the
    knob declares, so judging it would flag every working profile that defers
    a knob to its deployment."""
    assert _value_diags(_validate(env, {"todo": {"storage_type": value}})) == []


def test_a_deferred_value_is_not_judged_for_type_either(env):
    assert _value_diags(
        _validate(env, {"web_fetch": {"timeout": "${HTTP_TIMEOUT}"}})) == []


def test_null_is_unset_not_wrongly_typed(env):
    assert _value_diags(_validate(env, {"todo": {"storage_type": None}})) == []


def test_a_knob_with_no_declared_type_asserts_nothing(env):
    """The predicate table is a source of findings, never of guesses."""
    from shared.scaffold.validate import _check_knob_value
    found = []
    _check_knob_value("p", "k", 1, introspect.ConfigSetting(name="k", type=""),
                      lambda sev, code, msg, where=None:
                      found.append((sev, code)))
    _check_knob_value("p", "k", 1,
                      introspect.ConfigSetting(name="k", type="Widget"),
                      lambda sev, code, msg, where=None:
                      found.append((sev, code)))
    assert found == []


def test_free_form_substructures_are_still_not_descended(env):
    """Only a top-level knob's own shape is judged — the inner keys of a
    declared ``object`` knob remain the plugin's business."""
    assert _value_diags(_validate(env, {"permission": {
        "evaluators": {"cli_based_tool": "x.py", "made_up_tool": "y.py"},
        "policy": {"defaultPolicy": "deny", "whatever_inner": {"a": 1}},
    }})) == []


# ---- the type vocabulary ----------------------------------------------------

def test_a_union_type_accepts_either_member(env):
    """``cli.scrub_secret_env`` declares ``["string","array"]`` and both the
    ``none`` spelling and the ``[default, '!GH_TOKEN']`` list are valid."""
    assert _find(_validate(env, {"cli": {"scrub_secret_env": "none"}}),
                 "knob_type_mismatch", "scrub_secret_env") == []
    assert _find(_validate(env, {"cli": {
        "scrub_secret_env": ["default", "!GH_TOKEN"]}}),
        "knob_type_mismatch", "scrub_secret_env") == []


def test_an_integer_knob_does_not_accept_a_bool(env):
    """Python makes ``True`` an ``int``; a knob declared ``integer`` must not
    quietly accept ``timeout: true`` on that technicality."""
    assert len(_find(_validate(env, {"web_fetch": {"timeout": True}}),
                     "knob_type_mismatch", "timeout")) == 1


def test_a_number_knob_accepts_an_int(env):
    """JSON Schema makes every integer a valid ``number``."""
    assert _value_diags(
        _validate(env, {"multimodal": {"max_image_size_mb": 10}})) == []


def test_the_object_form_vocabulary_is_understood(env):
    """``lsp`` declares its schema as ``PluginSetting`` objects, whose type
    names are Python-ish (``dict`` / ``str`` / ``float``) rather than JSON
    Schema's."""
    assert _value_diags(_validate(env, {"lsp": {
        "languageServers": {}, "config_path": "x.json",
        "connect_timeout_seconds": 3.0}})) == []
    assert len(_find(_validate(env, {"lsp": {"config_path": 7}}),
                     "knob_type_mismatch", "config_path")) == 1


def test_the_object_form_spells_its_closed_set_choices(env):
    """``PluginSetting.choices`` is the object form's ``enum``.  No in-tree
    plugin declares one today, so the branch is exercised against a synthetic
    plugin — which is also the out-of-tree shape this check exists to reach."""
    providers, plugins, gc_names = env
    fake = introspect.PluginInfo(name="acme")
    fake.config_settings = [introspect.ConfigSetting(
        name="region", type="str", enum=["wt-wt", "us-en"])]
    fake.config_keys = ["region"]
    patched = dict(plugins, acme=fake)

    prof = SimpleNamespace(provider=None, model="m",
                           plugin_configs={"acme": {"region": "mars"}})
    diags = _find(validate_profile(prof, providers=providers, plugins=patched,
                                   gc_names=gc_names),
                  "invalid_knob_value", "acme.region")
    assert [d.severity for d in diags] == ["error"]
    assert "'us-en'" in diags[0].message


def test_choices_reaches_config_setting_from_the_object_form():
    from jaato_sdk.plugins.base import PluginSetting
    from shared.scaffold.introspect import _schema_enum, _schema_type
    setting = PluginSetting(name="region", type="str", default="wt-wt",
                            description="", choices=["wt-wt", "us-en"])
    assert _schema_enum(setting.choices) == ["wt-wt", "us-en"]
    assert _schema_enum(None) is None and _schema_enum([]) is None
    assert _schema_type(["string", "array"]) == "string|array"
    assert _schema_type(None) == ""


# ---- the name check is unchanged --------------------------------------------

def test_an_unknown_name_still_warns_and_does_not_become_a_value_check(env):
    diags = _validate(env, {"todo": {"storage_typo": "file"}})
    assert [d.severity for d in _find(diags, "unknown_knob",
                                      "storage_typo")] == ["warn"]
    assert _value_diags(diags) == []


def test_a_plugin_that_declares_no_schema_opts_out(env):
    """No schema means no judgement — a typo and a value are indistinguishable
    from an accepted free-form key."""
    diags = _validate(env, {"totally_made_up_plugin": {"foo": "bar"}})
    assert _value_diags(diags) == []
    assert _find(diags, "unknown_knob", "totally_made_up_plugin") == []
