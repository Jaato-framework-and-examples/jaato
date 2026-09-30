"""A profile can make catalog templates the only way to write a reference page.

Two parameters bypassed a profile that gates tools: ``renderTemplateToFile``
took an inline ``template`` (an arbitrary file write under another name) and
``proposeReference`` took the page inline as ``content``.  Tool gating
(``plugins:`` and ``tools:[...]`` scopes) cannot reach a parameter, so:

1. ``plugin_configs.template.allow_inline_template: false`` refuses an inline
   template, and ``narrow_tool_schema`` removes the parameter from the
   schema the model sees (through ``tool_visibility``, which both the wire
   and the deferred-tool catalog use).
2. ``plugin_configs.references.allow_inline_content: false`` does the same
   for ``proposeReference``'s ``content``.
3. Both are read per SESSION (``session_plugin_setting``): the plugin
   instance is shared with in-process subagents, and a gate must hold for
   the session that declared it.
4. A page ``renderTemplateToFile`` wrote from a catalog template is stamped
   on the claim as ``origin.rendered_from``, and flagged
   ``edited_after_render`` when the file no longer matches the render.
5. ``jaato-scaffold validate`` warns when a profile sets either knob and can
   still write a file another way (``template_only_gate_leaks``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

from jaato_sdk.plugins.model_provider.types import ToolSchema
from jaato_server.shared.plugins.references import claims
from jaato_server.shared.plugins.references.plugin import create_plugin as create_references
from jaato_server.shared.plugins.template.plugin import TemplatePlugin, _template_id
from jaato_server.shared.scaffold import validate as validate_mod
from jaato_server.shared.scaffold.introspect import PluginInfo, ToolInfo
from jaato_server.shared.session_context import (
    isolated_current_session,
    set_current_session,
)
from jaato_server.shared.tests.reversion import Reversion
from jaato_server.shared.tool_visibility import filter_visible_tool_schemas

_TPL = "jaato-server/jaato_server/shared/plugins/template/plugin.py"
_REFS = "jaato-server/jaato_server/shared/plugins/references/plugin.py"
_CLAIMS = "jaato-server/jaato_server/shared/plugins/references/claims.py"
_CTX = "jaato-server/jaato_server/shared/session_context.py"
_VIS = "jaato-server/jaato_server/shared/tool_visibility.py"
_VAL = "jaato-server/jaato_server/shared/scaffold/validate.py"

REVERSIONS = [
    Reversion(
        target=_TPL,
        find="        if template and not self._inline_template_allowed():\n",
        replace="        if False:\n",
        because="an inline template would still write any file the model asked for",
        test="TestInlineTemplates::test_refused_when_off",
    ),
    Reversion(
        target=_TPL,
        find='        if schema.name != "renderTemplateToFile" or self._inline_template_allowed():\n',
        replace="        if True:\n",
        because="the model would be offered a parameter the executor refuses",
        test="TestInlineTemplates::test_the_schema_does_not_offer_it",
    ),
    Reversion(
        target=_VIS,
        find="    return [_narrowed(schema, narrowers, on_error) for schema in visible]\n",
        replace="    return visible\n",
        because="no plugin's narrow_tool_schema would reach the wire or the catalog",
        test="TestInlineTemplates::test_the_schema_does_not_offer_it",
    ),
    Reversion(
        target=_CTX,
        find="    if isinstance(block, dict) and key in block:\n        return block[key]\n",
        replace="",
        because=(
            "a subagent's gate would be read off the shared instance, which "
            "holds whichever session re-initialized it last"
        ),
        test="TestPerSession::test_the_sessions_own_block_wins",
    ),
    Reversion(
        target=_REFS,
        find='        if args.get("content") and not self._inline_content_allowed():\n',
        replace="        if False:\n",
        because="a page could still be proposed inline, bypassing every template",
        test="TestInlineContent::test_refused_when_off",
    ),
    Reversion(
        target=_CLAIMS,
        find='    if file_digest(target) != record["digest"]:\n',
        replace="    if False:\n",
        because="a page edited after its render would read as the template's output",
        test="TestRenderedFrom::test_an_edit_is_flagged",
    ),
    Reversion(
        target=_VAL,
        find="    leaks: List[str] = [p for p in enabled if p in _COMMAND_PLUGINS]\n",
        replace="    leaks: List[str] = []\n",
        because="a profile with cli on the wire would read as template-only",
        test="TestValidate::test_a_command_plugin_is_a_leak",
    ),
]


class _Session:
    """The one method ``session_plugin_setting`` reads."""

    def __init__(self, configs: Dict[str, Dict[str, Any]]):
        self._configs = configs

    def declared_plugin_config(self, name: str) -> Optional[Dict[str, Any]]:
        return self._configs.get(name)


def _template_plugin(ws: Path, **config: Any) -> TemplatePlugin:
    plugin = TemplatePlugin()
    plugin.initialize({"base_path": str(ws), **config})
    catalog = ws / "catalog"
    catalog.mkdir(parents=True)
    (catalog / "runbook.md.tpl").write_text("# {{title}}\n\n## Steps\n{{steps}}\n", encoding="utf-8")
    for entry in plugin._discover_standalone_templates(catalog):
        plugin._template_index[entry.name] = entry
    return plugin


def _render_schema(plugin: TemplatePlugin) -> ToolSchema:
    return next(s for s in plugin.get_tool_schemas() if s.name == "renderTemplateToFile")


class _Registry:
    def __init__(self, **plugins: Any):
        self._plugins = plugins

    def list_exposed(self):
        return list(self._plugins)

    def get_plugin(self, name):
        return self._plugins.get(name)


class TestInlineTemplates:
    def test_refused_when_off(self, tmp_path):
        plugin = _template_plugin(tmp_path, allow_inline_template=False)
        result = plugin._execute_render_template_to_file(
            {"template": "anything", "variables": {}, "output_path": str(tmp_path / "x.md")})
        assert "turned off" in result["error"]
        assert not (tmp_path / "x.md").exists()
        # A catalog template still renders.
        ok = plugin._execute_render_template_to_file(
            {"template_id": _template_id("runbook.md.tpl"),
             "variables": {"title": "t", "steps": "s"},
             "output_path": str(tmp_path / "page.md")})
        assert ok.get("success") is True, ok

    def test_the_schema_does_not_offer_it(self, tmp_path):
        plugin = _template_plugin(tmp_path, allow_inline_template=False)
        wire = filter_visible_tool_schemas(_Registry(template=plugin), [_render_schema(plugin)])
        assert "template" not in wire[0].parameters["properties"]
        assert "template_id" in wire[0].parameters["properties"]

    def test_on_by_default(self, tmp_path):
        plugin = _template_plugin(tmp_path)
        wire = filter_visible_tool_schemas(_Registry(template=plugin), [_render_schema(plugin)])
        assert "template" in wire[0].parameters["properties"]


class TestPerSession:
    def test_the_sessions_own_block_wins(self, tmp_path):
        plugin = _template_plugin(tmp_path)  # the instance allows inline
        with isolated_current_session():
            set_current_session(_Session({"template": {"allow_inline_template": False}}))
            result = plugin._execute_render_template_to_file(
                {"template": "x", "variables": {}, "output_path": str(tmp_path / "x.md")})
        assert "turned off" in result["error"]

    def test_a_session_without_a_block_reads_the_instance(self, tmp_path):
        plugin = _template_plugin(tmp_path, allow_inline_template=False)
        with isolated_current_session():
            set_current_session(_Session({}))
            assert plugin._inline_template_allowed() is False


def _references(ws: Path, **config: Any):
    plugin = create_references()
    plugin.initialize({"workspace_path": str(ws), "lookup_strategy": "tags_only", **config})
    plugin.set_workspace_path(str(ws))
    return plugin


class TestInlineContent:
    def test_refused_when_off(self, tmp_path):
        refs = _references(tmp_path, allow_inline_content=False)
        with isolated_current_session():
            ok, payload = refs._execute_propose({"id": "p", "name": "p", "content": "# p"})
        assert ok is False and "turned off" in payload["error"]
        assert not claims.claims_dir(str(tmp_path)).exists()

    def test_the_schema_requires_a_path(self, tmp_path):
        refs = _references(tmp_path, allow_inline_content=False)
        schema = next(s for s in refs.get_tool_schemas() if s.name == "proposeReference")
        narrowed = filter_visible_tool_schemas(_Registry(references=refs), [schema])[0]
        assert "content" not in narrowed.parameters["properties"]
        assert "path" in narrowed.parameters["required"]


class TestRenderedFrom:
    def _propose_rendered(self, ws: Path, edit: bool) -> Dict[str, Any]:
        tpl = _template_plugin(ws)
        out = ws / "docs" / "runbook.md"
        tpl._execute_render_template_to_file(
            {"template_id": _template_id("runbook.md.tpl"),
             "variables": {"title": "Rollback", "steps": "1. revert"},
             "output_path": str(out)})
        if edit:
            out.write_text(out.read_text() + "\nextra\n", encoding="utf-8")
        refs = _references(ws)
        refs._plugin_registry = SimpleNamespace(get_plugin=lambda n: tpl if n == "template" else None)
        with isolated_current_session():
            result = refs._execute_propose({"id": "rollback", "name": "Rollback",
                                            "path": "docs/runbook.md"})
        claim, = claims.load_claims(str(ws))[0]
        assert claim["claim_id"] == result["claim_id"]
        return claim["origin"]["rendered_from"]

    def test_the_template_is_stamped(self, tmp_path):
        stamp = self._propose_rendered(tmp_path, edit=False)
        assert stamp["template"] == "runbook.md.tpl"
        assert stamp["template_id"] == _template_id("runbook.md.tpl")
        assert "edited_after_render" not in stamp

    def test_an_edit_is_flagged(self, tmp_path):
        stamp = self._propose_rendered(tmp_path, edit=True)
        assert stamp["edited_after_render"] is True

    def test_the_daemon_rechecks_the_file(self, tmp_path):
        stamp = self._propose_rendered(tmp_path, edit=False)
        entry = {"type": "local", "path": "docs/runbook.md"}
        (tmp_path / "docs" / "runbook.md").write_text("changed\n", encoding="utf-8")
        assert claims.rendered_stamp_now(str(tmp_path), entry, stamp)["edited_after_render"] is True


def _plugins_inventory() -> Dict[str, PluginInfo]:
    def info(name, tools):
        return PluginInfo(name=name, tools=[ToolInfo(name=t, traits=tr) for t, tr in tools])
    return {
        "template": info("template", [("renderTemplateToFile", ["file_writer"])]),
        "references": info("references", [("proposeReference", [])]),
        "file_edit": info("file_edit", [("readFile", []), ("writeNewFile", ["file_writer"])]),
        "cli": info("cli", [("cli_based_tool", [])]),
    }


def _findings(plugins, scopes=None, configs=None):
    profile = SimpleNamespace(
        plugins=plugins, tool_scopes=scopes or {},
        plugin_configs=configs if configs is not None else {
            "template": {"allow_inline_template": False},
            "references": {"allow_inline_content": False},
        })
    out = []
    validate_mod._check_template_only_gate(
        profile, _plugins_inventory(), lambda sev, code, msg, where=None: out.append((code, msg)))
    return out


class TestValidate:
    def test_a_command_plugin_is_a_leak(self):
        found = _findings(["template", "references", "cli"])
        assert [c for c, _ in found] == ["template_only_gate_leaks"]
        assert "cli" in found[0][1]

    def test_a_write_tool_is_a_leak_unless_scoped_out(self):
        assert _findings(["template", "references", "file_edit"])
        assert not _findings(["template", "references", "file_edit"],
                             scopes={"file_edit": ["readFile"]})

    def test_half_a_gate_is_a_leak(self):
        found = _findings(["template", "references"],
                          configs={"template": {"allow_inline_template": False}})
        assert "proposeReference" in found[0][1]

    def test_no_knob_no_finding(self):
        assert not _findings(["template", "references", "cli"], configs={})
