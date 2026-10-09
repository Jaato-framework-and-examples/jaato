"""A template's id can be derived outside a session, and is accepted (#1611).

kbwiki lists a workspace's catalog templates itself and lets a caller
render one through jaato's template tools, so the two must agree on the
``template_id``.  Before, the derivation lived only in jaato-server
internals, and even a correctly derived id was refused by a process that
had not issued it: id -> name went through an in-process reverse map that
only ``listAvailableTemplates`` (or a hint) populates.  A session-less
plugin-tool call (#1606) spawns a fresh runner per call, so an id handed
in from outside never resolved there.

1. ``jaato_sdk.templates.template_id`` equals what the plugin reports.
2. A fresh process accepts an id it never issued, for both
   ``listTemplateVariables`` and ``renderTemplateToFile``.
3. The extracts ``index.json`` carries each entry's id.
4. ``explain plugin template`` renders the derivation from the SDK.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jaato_sdk.templates import TEMPLATE_ID_DERIVATION, template_id
from jaato_server.shared import tool_id_map
from jaato_server.shared.plugins.template.plugin import (
    TemplatePlugin,
    _template_id,
)
from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/template/plugin.py"
_SDK = "jaato-sdk/jaato_sdk/templates.py"
_EXPLAIN = "jaato-server/jaato_server/shared/scaffold/explain.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find='        for candidate in list(self._template_index):\n'
             '            if _template_id(candidate) == template_id:\n'
             '                return candidate\n',
        replace='',
        because=(
            "without the index search an id this process never issued is "
            "'template not found', which is the reported defect on a fresh "
            "runner"
        ),
        test="TestAFreshProcessAcceptsAnIdItNeverIssued::test_list_template_variables",
    ),
    Reversion(
        target=_SDK,
        find='    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]\n',
        replace='    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]\n',
        because=(
            "an SDK derivation that disagrees with the plugin hands out ids "
            "renderTemplateToFile refuses"
        ),
        test="TestTheSdkDerivationIsThePlugins::test_sdk_equals_plugin",
    ),
    Reversion(
        target=_PLUGIN,
        find='name: {**asdict(entry), "id": _template_id(name)}',
        replace='name: asdict(entry)',
        because="a reader of the persisted index would have to derive the id",
        test="TestThePersistedIndexCarriesTheId::test_extracts_index_has_id",
    ),
    Reversion(
        target=_EXPLAIN,
        find='    if name == "template":\n        return _template_id_lines()\n',
        replace='',
        because="explain plugin template would not state the derivation",
        test="TestExplainStatesTheDerivation::test_explain_plugin_template",
    ),
]

_NAMES = ["Entity.java.tpl", "orders/Repository.java.tpl", "café.md.tpl", ""]


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    cat = ws / ".jaato" / "templates"
    cat.mkdir(parents=True)
    (cat / "Entity.java.tpl").write_text("class {{Entity}} {}\n")
    (cat / "index.json").write_text(json.dumps({
        "schema": "x",
        "entries": [{
            "name": "Entity.java.tpl",
            "source": str(cat / "Entity.java.tpl"),
            "syntax": "mustache",
            "variables": ["Entity"],
            "origin": "standalone",
        }],
    }))
    return ws


def _fresh_plugin(ws: Path) -> TemplatePlugin:
    # A fresh process has issued no ids: empty the process-wide reverse map.
    tool_id_map._reverse.clear()
    p = TemplatePlugin()
    p.initialize({"base_path": str(ws)})
    return p


class TestTheSdkDerivationIsThePlugins:
    @pytest.mark.parametrize("name", _NAMES)
    def test_sdk_equals_plugin(self, name):
        assert template_id(name) == _template_id(name)

    def test_the_stated_formula(self):
        name = "Entity.java.tpl"
        expected = "tpl_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        assert template_id(name) == expected

    def test_list_available_reports_the_sdk_id(self, tmp_path):
        p = _fresh_plugin(_make_workspace(tmp_path))
        ids = {t["id"] for t in p._execute_list_available({})["templates"]}
        assert template_id("Entity.java.tpl") in ids


class TestAFreshProcessAcceptsAnIdItNeverIssued:
    def test_list_template_variables(self, tmp_path):
        p = _fresh_plugin(_make_workspace(tmp_path))
        out = p._execute_list_template_variables(
            {"template_id": template_id("Entity.java.tpl")})
        assert "error" not in out, out
        assert [v["name"] for v in out["variables"]] == ["Entity"]

    def test_render_template_to_file(self, tmp_path):
        ws = _make_workspace(tmp_path)
        p = _fresh_plugin(ws)
        out = p._execute_render_template_to_file({
            "template_id": template_id("Entity.java.tpl"),
            "variables": {"Entity": "Order"},
            "output_path": str(ws / "Order.java"),
        })
        assert "error" not in out, out
        assert "class Order" in (ws / "Order.java").read_text()

    def test_unknown_id_is_still_not_found(self, tmp_path):
        p = _fresh_plugin(_make_workspace(tmp_path))
        out = p._execute_list_template_variables({"template_id": "tpl_00000000"})
        assert "not found" in out["error"]


class TestThePersistedIndexCarriesTheId:
    def test_extracts_index_has_id(self, tmp_path):
        p = _fresh_plugin(_make_workspace(tmp_path))
        p._persist_index()
        data = json.loads((p._extracts_dir / "index.json").read_text())
        entry = data["templates"]["Entity.java.tpl"]
        assert entry["id"] == template_id("Entity.java.tpl")


class TestExplainStatesTheDerivation:
    def test_explain_plugin_template(self):
        from jaato_server.shared.scaffold import explain
        data, text = explain.plugin("template")
        assert data["template_id"]["prefix"] == "tpl"
        for rule in TEMPLATE_ID_DERIVATION:
            assert rule in text
