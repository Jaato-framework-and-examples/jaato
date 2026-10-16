"""listTemplateVariables reports the Jinja2 kinds the shape check uses (#1705).

For a Jinja2 template ``listTemplateVariables`` answered ``kind:
"scalar"`` for every variable, while ``_parse_jinja2_structure`` (the
render-time shape check) already classified ``{% for r in rules %}`` as
``iterable``.  A session read ``rules: scalar``, passed it as a string in
every encoding it could think of, and spent 17 renders on ``'str object'
has no attribute 'statement'``.

1. Each body variable carries the kind ``_parse_jinja2_structure``
   computes, so the report and the check cannot disagree.
2. A path-only variable is still ``scalar``.
3. The tool description documents the Jinja2 kinds it can now report.
"""

from __future__ import annotations

import json
from pathlib import Path

from jaato_sdk.templates import template_id
from jaato_server.shared import tool_id_map
from jaato_server.shared.plugins.template.plugin import TemplatePlugin
from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/template/plugin.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find='            for v in self._parse_jinja2_structure(template_content)\n'
             '            if v["name"] in variables\n',
        replace='            for v in []\n',
        because="every Jinja2 variable is reported without its structural kind",
        test="TestJinja2KindsAreReported::test_kinds_match_the_shape_check",
    ),
    Reversion(
        target=_PLUGIN,
        find="                    \"  - ``kind == 'iterable'`` (Jinja2): looped over as \"\n",
        replace="                    \"  - \"\n",
        because="the tool description would not say what 'iterable' asks for",
        test="TestTheToolDescriptionNamesTheKinds::test_jinja2_kinds_documented",
    ),
]

_BODY = (
    "// Output: docs/{{ slug }}.md\n"
    "# {{ title }}\n"
    "{% for r in rules %}\n"
    "### R{{ loop.index }} · {{ r.title }}\n"
    "{{ r.statement }}\n"
    "{% endfor %}\n"
    "{% if variant %}{{ variant }}{% endif %}\n"
    "{{ decisions | join(', ') }}\n"
    "{{ footer | default('') }}\n"
)


def _plugin(tmp_path: Path, name: str, body: str) -> TemplatePlugin:
    ws = tmp_path / "ws"
    cat = ws / ".jaato" / "templates"
    cat.mkdir(parents=True)
    (cat / name).write_text(body)
    entry = {
        "name": name,
        "source": str(cat / name),
        "syntax": "jinja2",
        "variables": [],
        "origin": "standalone",
    }
    (cat / "index.json").write_text(json.dumps({"schema": "x", "entries": [entry]}))
    tool_id_map._reverse.clear()
    p = TemplatePlugin()
    p.initialize({"base_path": str(ws)})
    return p


def _list(p: TemplatePlugin, name: str):
    out = p._execute_list_template_variables({"template_id": template_id(name)})
    assert "error" not in out, out
    assert out["syntax"] == "jinja2"
    return {v["name"]: v["kind"] for v in out["variables"]}


class TestJinja2KindsAreReported:
    def test_kinds_match_the_shape_check(self, tmp_path):
        p = _plugin(tmp_path, "page.md.j2", _BODY)
        got = _list(p, "page.md.j2")
        assert got["rules"] == "iterable"
        assert got["variant"] == "conditional_test"
        assert got["decisions"] == "operand"
        assert got["footer"] == "scalar_with_default"
        assert got["title"] == "scalar"
        expected = {
            v["name"]: v["kind"] for v in p._parse_jinja2_structure(_BODY)
        }
        for name, kind in expected.items():
            assert got[name] == kind, name

    def test_loop_locals_are_not_reported(self, tmp_path):
        p = _plugin(tmp_path, "page.md.j2", _BODY)
        got = _list(p, "page.md.j2")
        assert "r" not in got
        assert "loop" not in got

    def test_path_only_variable_is_scalar(self, tmp_path):
        p = _plugin(tmp_path, "page.md.j2", _BODY)
        assert _list(p, "page.md.j2")["slug"] == "scalar"


class TestTheToolDescriptionNamesTheKinds:
    def test_jinja2_kinds_documented(self):
        schema = next(
            s for s in TemplatePlugin().get_tool_schemas()
            if s.name == "listTemplateVariables"
        )
        for kind in ("iterable", "conditional_test", "scalar_with_default", "operand"):
            assert f"kind == '{kind}'" in schema.description, kind
