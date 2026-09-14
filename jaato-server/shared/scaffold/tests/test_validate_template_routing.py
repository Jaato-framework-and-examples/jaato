"""Scaffold coverage for ``plugin_configs.template.file_conventions`` (#900).

Two independent guarantees, and both are what the knob buys over the
``template_routing.yaml`` file it supersedes:

1. The knob is DECLARED, so a typo in its name is reported as
   ``unknown_knob`` — the file tier has no such check at all.
2. Its VALUE shape is checked, because a malformed routing table is not
   a runtime error: the plugin drops the bad rules and carries on with
   no routing, and generated files then land outside the declared source
   root, where a validator gate does not look.
"""
from types import SimpleNamespace

from shared.scaffold import introspect
from shared.scaffold.validate import validate_profile


def _validate(template_cfg):
    prof = SimpleNamespace(provider=None, model="m",
                           plugin_configs={"template": template_cfg})
    return validate_profile(
        prof, providers=introspect.providers(), plugins=introspect.plugins(),
        gc_names=list(introspect.gc_strategies().keys()))


def _codes(diags, code):
    return [d for d in diags if d.code == code]


VALID = {
    "file_conventions": {
        "output_path_routing": [
            {"glob": "pom.xml", "prefix": ""},
            {"glob": "**/*Test.java", "prefix": "src/test/java"},
            {"glob": "**/*.java", "prefix": "src/main/java"},
        ],
    },
}


# ---- the knob is declared ---------------------------------------------------

def test_introspect_surfaces_the_knob():
    tpl = introspect.plugins().get("template")
    assert tpl is not None
    assert "file_conventions" in tpl.config_keys


def test_valid_routing_table_is_clean():
    diags = _validate(VALID)
    assert _codes(diags, "unknown_knob") == []
    assert _codes(diags, "invalid_template_routing") == []
    assert _codes(diags, "template_routing_empty") == []


def test_typo_in_the_knob_name_is_flagged():
    diags = _validate({"file_convention": VALID["file_conventions"]})
    assert any("file_convention" in (d.where or "")
               for d in _codes(diags, "unknown_knob"))


# ---- the value shape is checked --------------------------------------------

def test_non_mapping_file_conventions_is_an_error():
    diags = _validate({"file_conventions": "src/main/java"})
    assert _codes(diags, "invalid_template_routing")


def test_file_conventions_without_rules_warns_that_routing_stops():
    """A knowledge base's stack block carries other keys under this name
    (source_dirs, source_extension, build_file).  Carried into a profile
    verbatim it suppresses the file — the key's presence IS the
    declaration — while declaring no rules of its own."""
    diags = _validate({"file_conventions": {
        "source_dirs": {"main": "src/main/java"},
        "source_extension": ".java",
    }})
    assert _codes(diags, "template_routing_empty")


def test_non_list_routing_is_an_error():
    diags = _validate({"file_conventions": {
        "output_path_routing": {"glob": "**/*.java"},
    }})
    assert _codes(diags, "invalid_template_routing")


def test_malformed_entries_are_flagged_individually():
    diags = _validate({"file_conventions": {"output_path_routing": [
        "**/*.java",                                # not a mapping
        {"prefix": "src/main/java"},                # no glob
        {"glob": "**/*.java", "prefix": ["nope"]},  # non-str prefix
        {"glob": "**/*.yml", "prefix": "src/main/resources"},   # fine
    ]}})
    wheres = [d.where for d in _codes(diags, "invalid_template_routing")]
    assert any("[0]" in w for w in wheres)
    assert any("[1].glob" in w for w in wheres)
    assert any("[2].prefix" in w for w in wheres)
    assert not any("[3]" in w for w in wheres)


def test_a_profile_declaring_no_template_config_is_untouched():
    prof = SimpleNamespace(provider=None, model="m", plugin_configs={})
    diags = validate_profile(
        prof, providers=introspect.providers(), plugins=introspect.plugins(),
        gc_names=list(introspect.gc_strategies().keys()))
    assert _codes(diags, "invalid_template_routing") == []
    assert _codes(diags, "template_routing_empty") == []
