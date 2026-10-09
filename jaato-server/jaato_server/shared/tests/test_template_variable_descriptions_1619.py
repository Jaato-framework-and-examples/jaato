"""A catalog index describes its variables, and callers are told (#1619).

``listTemplateVariables`` answered each variable's name and shape and
nothing about what to put in it.  The provisioner often knows (kbwiki
writes page templates from a plan that gives every section guidance) and
had nowhere to say it.  A catalog ``index.json`` entry may now carry
``variable_descriptions: {name: text}`` beside ``variables``:

1. ``listTemplateVariables`` adds ``description`` to each variable that
   has one, for Jinja2 and Mustache alike; an undescribed variable keeps
   exactly its old shape.
2. A description for a name the template does not use is ignored; a
   malformed field costs the entry nothing at load.
3. ``validateTemplateIndex`` accepts the field and refuses a malformed
   one; a name missing from ``variables`` is a warning.
4. ``explain plugin template`` states the rules from the plugin's table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaato_sdk.templates import template_id
from jaato_server.shared import tool_id_map
from jaato_server.shared.plugins.template.plugin import (
    VARIABLE_DESCRIPTION_RULES,
    TemplatePlugin,
)
from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/template/plugin.py"
_EXPLAIN = "jaato-server/jaato_server/shared/scaffold/explain.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find='                    variable_descriptions=_coerce_variable_descriptions(\n'
             '                        entry_data.get(VARIABLE_DESCRIPTIONS_KEY),\n'
             '                    ),\n',
        replace='',
        because="the loader drops the index's descriptions, so no caller sees them",
        test="TestListTemplateVariablesReturnsDescriptions::test_mustache",
    ),
    Reversion(
        target=_PLUGIN,
        find='            self._attach_variable_descriptions(structured, template_entry)\n',
        replace='',
        because="a Mustache template's variables come back without descriptions",
        test="TestListTemplateVariablesReturnsDescriptions::test_mustache",
    ),
    Reversion(
        target=_PLUGIN,
        find='                self._attach_variable_descriptions(final_list, template_entry)\n',
        replace='',
        because="a Jinja2 template's variables come back without descriptions",
        test="TestListTemplateVariablesReturnsDescriptions::test_jinja2",
    ),
    Reversion(
        target=_PLUGIN,
        find='            self._validate_variable_descriptions(entry, prefix, errors, warnings)\n',
        replace='',
        because="validateTemplateIndex passes a malformed variable_descriptions",
        test="TestValidateTemplateIndex::test_non_object_is_refused",
    ),
    Reversion(
        target=_EXPLAIN,
        find='    lines.extend(f"    - {rule}" for rule in TP.VARIABLE_DESCRIPTION_RULES)\n',
        replace='',
        because="explain plugin template would not document the field",
        test="TestExplainDocumentsTheField::test_rules_rendered",
    ),
]

_DESCRIPTIONS = {
    "title": "The page's H1: the component's name, no punctuation",
    "purpose": "One paragraph: what the component is for",
    "nowhere": "names no variable of the template",
}


def _workspace(tmp_path: Path, name: str, body: str, descriptions) -> Path:
    ws = tmp_path / "ws"
    cat = ws / ".jaato" / "templates"
    cat.mkdir(parents=True)
    (cat / name).write_text(body)
    entry = {
        "name": name,
        "source": str(cat / name),
        "syntax": "jinja2" if name.endswith(".j2") else "mustache",
        "variables": ["title", "purpose", "usage"],
        "origin": "standalone",
    }
    if descriptions is not None:
        entry["variable_descriptions"] = descriptions
    (cat / "index.json").write_text(json.dumps({"schema": "x", "entries": [entry]}))
    return ws


def _list(tmp_path: Path, name: str, body: str, descriptions=_DESCRIPTIONS):
    ws = _workspace(tmp_path, name, body, descriptions)
    tool_id_map._reverse.clear()
    p = TemplatePlugin()
    p.initialize({"base_path": str(ws)})
    out = p._execute_list_template_variables({"template_id": template_id(name)})
    assert "error" not in out, out
    return {v["name"]: v for v in out["variables"]}


# A section makes the body unambiguously Mustache: plain ``{{x}}`` is
# read as Jinja2 by syntax detection.
_MUSTACHE = "# {{title}}\n\n{{purpose}}\n\n{{usage}}\n{{#notes}}- {{.}}\n{{/notes}}\n"
_JINJA2 = "# {% if title %}{{ title }}{% endif %}\n\n{{ purpose }}\n\n{{ usage }}\n"


class TestListTemplateVariablesReturnsDescriptions:
    def test_mustache(self, tmp_path):
        got = _list(tmp_path, "page.md.tpl", _MUSTACHE)
        assert got["title"]["description"] == _DESCRIPTIONS["title"]
        assert got["purpose"]["description"] == _DESCRIPTIONS["purpose"]
        assert got["title"]["kind"] == "scalar"
        assert got["notes"]["kind"] == "section"

    def test_jinja2(self, tmp_path):
        got = _list(tmp_path, "page.md.j2", _JINJA2)
        assert got["purpose"]["description"] == _DESCRIPTIONS["purpose"]

    def test_undescribed_variable_keeps_its_shape(self, tmp_path):
        plain = _list(tmp_path / "a", "page.md.tpl", _MUSTACHE, descriptions=None)
        got = _list(tmp_path / "b", "page.md.tpl", _MUSTACHE)
        assert got["usage"] == plain["usage"]
        assert "description" not in got["usage"]

    def test_description_for_unused_name_is_ignored(self, tmp_path):
        got = _list(tmp_path, "page.md.tpl", _MUSTACHE)
        assert "nowhere" not in got

    @pytest.mark.parametrize("bad", [
        "not an object", ["title"], {"title": 3}, {"title": "   "},
    ])
    def test_malformed_field_costs_the_template_nothing(self, tmp_path, bad):
        got = _list(tmp_path, "page.md.tpl", _MUSTACHE, descriptions=bad)
        assert set(got) == {"title", "purpose", "usage", "notes"}
        assert all("description" not in v for v in got.values())


def _index(descriptions, variables=("title", "purpose")):
    entry = {
        "name": "page.md.tpl",
        "source_path": "templates/page.md.tpl",
        "syntax": "mustache",
        "variables": list(variables),
        "origin": "standalone",
        "variable_descriptions": descriptions,
    }
    return {"generated_at": "2026-10-09T00:00:00", "template_count": 1,
            "templates": {"page.md.tpl": entry}}


class TestValidateTemplateIndex:
    @pytest.fixture
    def plugin(self):
        p = TemplatePlugin()
        p.initialize()
        return p

    def test_valid_field_is_accepted(self, plugin):
        ok, errors, warnings = plugin._validate_template_index(
            _index({"title": "the H1", "purpose": "why"}))
        assert ok and errors == [] and warnings == []

    def test_non_object_is_refused(self, plugin):
        ok, errors, _ = plugin._validate_template_index(_index(["title"]))
        assert not ok
        assert any("variable_descriptions" in e and "object" in e for e in errors)

    def test_non_string_and_blank_are_refused(self, plugin):
        ok, errors, _ = plugin._validate_template_index(
            _index({"title": 3, "purpose": " "}))
        assert not ok
        assert any("['title'] must be a string" in e for e in errors)
        assert any("['purpose'] must not be blank" in e for e in errors)

    def test_unlisted_name_is_a_warning(self, plugin):
        ok, errors, warnings = plugin._validate_template_index(
            _index({"usage": "how to use it"}))
        assert ok and errors == []
        assert any("not listed in 'variables'" in w for w in warnings)


class TestExplainDocumentsTheField:
    def test_rules_rendered(self):
        from jaato_server.shared.scaffold import explain
        data, text = explain.plugin("template")
        assert data["variable_descriptions"]["key"] == "variable_descriptions"
        for rule in VARIABLE_DESCRIPTION_RULES:
            assert rule in text
