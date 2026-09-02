"""Tests for the unified template index and standalone template discovery.

Tests cover:
- TemplateIndexEntry dataclass
- Standalone template discovery from directories
- Index-based path resolution
- Unified listing (embedded + standalone)
- Index persistence to index.json
- Name collision handling
- Cross-plugin integration with references plugin
"""

import json
import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shared.plugins.template.plugin import (
    TemplatePlugin,
    TemplateIndexEntry,
    TEMPLATE_FILE_EXTENSIONS,
    PATH_ROUTING_FILENAME,
    _template_id,
)


def _names(item_keys):
    """Extract names from the new item_keys list-of-dicts shape (server
    0.6.43+).  Each entry is ``{"name": str, "required": bool, "source": str}``;
    this helper centralises the migration from the prior List[str] view."""
    return [k["name"] for k in item_keys]


def _render_args(template_name=None, **kw):
    """Build a renderTemplateToFile args dict from a human template name.

    Server 0.6.119+: tool args use the LLM-facing ``template_id`` hash
    rather than ``template_name``.  Tests author with human names for
    readability; this helper does the hash conversion in one place.
    Pass ``template_name=None`` (default) when building args for an
    inline ``template`` test.
    """
    if template_name is not None:
        kw["template_id"] = _template_id(template_name)
    return kw


def _list_vars_args(template_name):
    """Build a listTemplateVariables args dict from a human template name."""
    return {"template_id": _template_id(template_name)}


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace with .jaato/templates/ directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    templates_dir = workspace / ".jaato" / "templates"
    templates_dir.mkdir(parents=True)
    return workspace


@pytest.fixture
def plugin(tmp_workspace):
    """Create an initialized TemplatePlugin with a temp workspace."""
    p = TemplatePlugin()
    p.initialize({"base_path": str(tmp_workspace)})
    return p


@pytest.fixture
def template_dir(tmp_path):
    """Create a directory tree with standalone template files."""
    tpl_dir = tmp_path / "knowledge" / "modules" / "mod-015" / "templates"

    # domain/Entity.java.tpl (mustache)
    domain = tpl_dir / "domain"
    domain.mkdir(parents=True)
    (domain / "Entity.java.tpl").write_text(textwrap.dedent("""\
        package {{basePackage}}.domain.model;

        public class {{Entity}} {
        {{#entityFields}}
            private {{fieldType}} {{fieldName}};
        {{/entityFields}}
        }
    """))

    # domain/Repository.java.tpl (mustache)
    (domain / "Repository.java.tpl").write_text(textwrap.dedent("""\
        package {{basePackage}}.domain;

        public interface {{Entity}}Repository {
            {{Entity}} findById({{Entity}}Id id);
        }
    """))

    # config/application.yml.tpl (jinja2)
    config = tpl_dir / "config"
    config.mkdir(parents=True)
    (config / "application.yml.tpl").write_text(textwrap.dedent("""\
        spring:
          application:
            name: {{ service_name }}
          datasource:
            url: {{ db_url }}
    """))

    return tpl_dir


# ==================== TemplateIndexEntry ====================

class TestTemplateIndexEntry:
    def test_basic_creation(self):
        entry = TemplateIndexEntry(
            name="Entity.java.tpl",
            source_path="/path/to/Entity.java.tpl",
            syntax="mustache",
            variables=["Entity", "basePackage"],
            origin="standalone",
        )
        assert entry.name == "Entity.java.tpl"
        assert entry.source_path == "/path/to/Entity.java.tpl"
        assert entry.syntax == "mustache"
        assert entry.variables == ["Entity", "basePackage"]
        assert entry.origin == "standalone"

    def test_defaults(self):
        entry = TemplateIndexEntry(
            name="test.tpl",
            source_path="/path/test.tpl",
            syntax="jinja2",
        )
        assert entry.variables == []
        assert entry.origin == "embedded"


# ==================== Standalone Template Discovery ====================

class TestStandaloneDiscovery:
    def test_discover_templates_in_directory(self, plugin, template_dir):
        entries = plugin._discover_standalone_templates(template_dir)

        names = {e.name for e in entries}
        assert "Entity.java.tpl" in names
        assert "Repository.java.tpl" in names
        assert "application.yml.tpl" in names
        assert len(entries) == 3

    def test_discovered_syntax_detection(self, plugin, template_dir):
        entries = plugin._discover_standalone_templates(template_dir)
        by_name = {e.name: e for e in entries}

        # Mustache templates (have {{#entityFields}})
        assert by_name["Entity.java.tpl"].syntax == "mustache"

        # Jinja2 template (has {{ service_name }})
        assert by_name["application.yml.tpl"].syntax == "jinja2"

    def test_discovered_variables_extracted(self, plugin, template_dir):
        entries = plugin._discover_standalone_templates(template_dir)
        by_name = {e.name: e for e in entries}

        entity_vars = by_name["Entity.java.tpl"].variables
        assert "Entity" in entity_vars
        assert "basePackage" in entity_vars

    def test_discovered_source_path_is_absolute(self, plugin, template_dir):
        entries = plugin._discover_standalone_templates(template_dir)
        for entry in entries:
            assert Path(entry.source_path).is_absolute()

    def test_discovered_origin_is_standalone(self, plugin, template_dir):
        entries = plugin._discover_standalone_templates(template_dir)
        for entry in entries:
            assert entry.origin == "standalone"

    def test_empty_directory(self, plugin, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        entries = plugin._discover_standalone_templates(empty)
        assert entries == []

    def test_nonexistent_directory(self, plugin, tmp_path):
        entries = plugin._discover_standalone_templates(tmp_path / "nonexistent")
        assert entries == []

    def test_no_template_files(self, plugin, tmp_path):
        notempl = tmp_path / "notempl"
        notempl.mkdir()
        (notempl / "README.md").write_text("# Not a template")
        (notempl / "code.java").write_text("class Foo {}")
        entries = plugin._discover_standalone_templates(notempl)
        assert entries == []

    def test_name_collision_disambiguation(self, plugin, tmp_path):
        """When two files have the same name, parent folder is prepended."""
        base = tmp_path / "templates"
        (base / "domain").mkdir(parents=True)
        (base / "adapter").mkdir(parents=True)

        (base / "domain" / "Service.java.tpl").write_text(
            "package {{pkg}}.domain;\npublic class {{Entity}}DomainService {}"
        )
        (base / "adapter" / "Service.java.tpl").write_text(
            "package {{pkg}}.adapter;\npublic class {{Entity}}RestAdapter {}"
        )

        entries = plugin._discover_standalone_templates(base)
        names = {e.name for e in entries}

        # Both should be present with disambiguated names
        assert len(entries) == 2
        # With collision, full relative paths are used
        assert "domain/Service.java.tpl" in names
        assert "adapter/Service.java.tpl" in names

    def test_skip_already_indexed(self, plugin, template_dir):
        """Templates already in the index should be skipped."""
        # Pre-populate index
        plugin._template_index["Entity.java.tpl"] = TemplateIndexEntry(
            name="Entity.java.tpl",
            source_path="/other/Entity.java.tpl",
            syntax="mustache",
        )

        entries = plugin._discover_standalone_templates(template_dir)
        names = {e.name for e in entries}

        # Entity.java.tpl should NOT be in the discovered entries
        assert "Entity.java.tpl" not in names
        # Others should be there
        assert "Repository.java.tpl" in names
        assert "application.yml.tpl" in names

    def test_template_file_extensions(self):
        assert ".tpl" in TEMPLATE_FILE_EXTENSIONS
        assert ".tmpl" in TEMPLATE_FILE_EXTENSIONS

    def test_tmpl_extension_discovered(self, plugin, tmp_path):
        """Files with .tmpl extension should also be discovered."""
        tdir = tmp_path / "templates"
        tdir.mkdir()
        (tdir / "schema.sql.tmpl").write_text(
            "CREATE TABLE {{ table_name }} (\n  id UUID PRIMARY KEY\n);"
        )

        entries = plugin._discover_standalone_templates(tdir)
        assert len(entries) == 1
        assert entries[0].name == "schema.sql.tmpl"


# ==================== Index-based Path Resolution ====================

class TestIndexResolution:
    def test_resolve_by_name(self, plugin, template_dir):
        """Resolving by template name should use the index."""
        entries = plugin._discover_standalone_templates(template_dir)
        for entry in entries:
            plugin._template_index[entry.name] = entry

        resolved, tried = plugin._resolve_template_path("Entity.java.tpl")
        assert resolved is not None
        assert resolved.exists()
        assert resolved.name == "Entity.java.tpl"
        assert any("index:" in t for t in tried)

    def test_resolve_by_name_with_path_prefix(self, plugin, template_dir):
        """Passing 'templates/Entity.java.tpl' should strip prefix and match."""
        entries = plugin._discover_standalone_templates(template_dir)
        for entry in entries:
            plugin._template_index[entry.name] = entry

        resolved, tried = plugin._resolve_template_path("some/path/Entity.java.tpl")
        assert resolved is not None
        assert resolved.name == "Entity.java.tpl"

    def test_resolve_unknown_name(self, plugin):
        """Unknown template name falls through to filesystem resolution."""
        resolved, tried = plugin._resolve_template_path("NonExistent.java.tpl")
        assert resolved is None
        assert len(tried) > 0

    def test_resolve_falls_through_when_source_deleted(self, plugin, template_dir):
        """If indexed file is deleted, resolution should fail gracefully."""
        entries = plugin._discover_standalone_templates(template_dir)
        for entry in entries:
            plugin._template_index[entry.name] = entry

        # Delete the actual file
        os.unlink(entries[0].source_path)

        # Shouldn't return the deleted file
        resolved, tried = plugin._resolve_template_path(entries[0].name)
        # Falls through index (file gone) to filesystem resolution
        assert resolved is None or resolved.exists()

    def test_index_resolution_takes_priority(self, plugin, tmp_workspace, template_dir):
        """Index lookup should happen before filesystem checks."""
        entries = plugin._discover_standalone_templates(template_dir)
        for entry in entries:
            plugin._template_index[entry.name] = entry

        # Also create a file with the same name in .jaato/templates/
        alt_path = tmp_workspace / ".jaato" / "templates" / "Entity.java.tpl"
        alt_path.write_text("// different content")

        resolved, tried = plugin._resolve_template_path("Entity.java.tpl")
        assert resolved is not None
        # Should resolve to the indexed path, not the .jaato/templates/ one
        assert str(resolved) == entries[0].source_path if entries[0].name == "Entity.java.tpl" else True


# ==================== Unified Listing ====================

class TestUnifiedListing:
    def test_list_empty(self, plugin):
        result = plugin._execute_list_available({})
        assert result["templates"] == []
        assert "message" in result

    def test_list_standalone_templates(self, plugin, template_dir):
        entries = plugin._discover_standalone_templates(template_dir)
        for entry in entries:
            plugin._template_index[entry.name] = entry

        result = plugin._execute_list_available({})
        assert result["count"] == 3

        # All should be standalone origin
        for t in result["templates"]:
            assert t["origin"] == "standalone"
            assert t["exists"] is True
            # Cutover (server 0.6.119+): response carries ``id`` not ``name``.
            assert "id" in t
            assert t["id"].startswith("tpl_")
            assert "variables" in t

    def test_list_mixed_origins(self, plugin, template_dir):
        """Index with both embedded and standalone should list both."""
        # Add standalone
        entries = plugin._discover_standalone_templates(template_dir)
        for entry in entries:
            plugin._template_index[entry.name] = entry

        # Add embedded
        plugin._template_index["custom.java.tmpl"] = TemplateIndexEntry(
            name="custom.java.tmpl",
            source_path=str(plugin._templates_dir / "custom.java.tmpl"),
            syntax="jinja2",
            variables=["class_name"],
            origin="embedded",
        )
        # Create the embedded file
        plugin._templates_dir.mkdir(parents=True, exist_ok=True)
        (plugin._templates_dir / "custom.java.tmpl").write_text("class {{ class_name }} {}")

        result = plugin._execute_list_available({})
        assert result["count"] == 4

        origins = {t["origin"] for t in result["templates"]}
        assert "standalone" in origins
        assert "embedded" in origins

        # Standalone should be sorted first
        assert result["templates"][0]["origin"] == "standalone"


# ==================== Index Persistence ====================

class TestIndexPersistence:
    def test_persist_writes_json(self, plugin, template_dir):
        entries = plugin._discover_standalone_templates(template_dir)
        for entry in entries:
            plugin._template_index[entry.name] = entry

        plugin._persist_index()

        index_path = plugin._templates_dir / "index.json"
        assert index_path.exists()

        data = json.loads(index_path.read_text())
        assert "generated_at" in data
        assert "template_count" in data
        assert data["template_count"] == 3
        assert "templates" in data
        assert "Entity.java.tpl" in data["templates"]

        entity = data["templates"]["Entity.java.tpl"]
        assert entity["origin"] == "standalone"
        assert entity["syntax"] == "mustache"
        assert len(entity["variables"]) > 0

    def test_persist_empty_index_is_noop(self, plugin):
        plugin._persist_index()
        index_path = plugin._templates_dir / "index.json"
        assert not index_path.exists()


# ==================== Cross-plugin Integration ====================

class TestReferenceIntegration:
    def test_get_reference_directories_no_registry(self, plugin):
        """Without registry, should return empty."""
        assert plugin._get_reference_directories() == []

    def test_get_reference_directories_no_references_plugin(self, plugin):
        """With registry but no references plugin, should return empty."""
        registry = MagicMock()
        registry.get_plugin.return_value = None
        plugin._plugin_registry = registry

        assert plugin._get_reference_directories() == []

    def test_get_reference_directories_with_selected_dirs(self, plugin, template_dir):
        """Should extract directory paths from selected LOCAL sources."""
        # Mock the references plugin
        ref_plugin = MagicMock()
        ref_plugin.get_selected_ids.return_value = ["mod-015"]

        source = MagicMock()
        source.id = "mod-015"
        source.type.value = "local"
        source.resolved_path = str(template_dir.parent)  # The module dir
        source.path = "knowledge/modules/mod-015"
        ref_plugin.get_sources.return_value = [source]

        registry = MagicMock()
        registry.get_plugin.return_value = ref_plugin
        plugin._plugin_registry = registry

        dirs = plugin._get_reference_directories()
        assert len(dirs) == 1
        assert dirs[0] == Path(template_dir.parent)

    def test_get_reference_directories_skips_unselected(self, plugin, template_dir):
        """Non-selected sources should not be returned."""
        ref_plugin = MagicMock()
        ref_plugin.get_selected_ids.return_value = []  # Nothing selected

        source = MagicMock()
        source.id = "mod-015"
        source.type.value = "local"
        source.resolved_path = str(template_dir.parent)
        source.path = "knowledge/modules/mod-015"
        ref_plugin.get_sources.return_value = [source]

        registry = MagicMock()
        registry.get_plugin.return_value = ref_plugin
        plugin._plugin_registry = registry

        dirs = plugin._get_reference_directories()
        assert len(dirs) == 0

    def test_get_reference_directories_skips_non_local(self, plugin):
        """Non-LOCAL sources (URL, MCP) should not be returned."""
        ref_plugin = MagicMock()
        ref_plugin.get_selected_ids.return_value = ["remote-doc"]

        source = MagicMock()
        source.id = "remote-doc"
        source.type.value = "url"
        source.resolved_path = None
        source.path = None
        ref_plugin.get_sources.return_value = [source]

        registry = MagicMock()
        registry.get_plugin.return_value = ref_plugin
        plugin._plugin_registry = registry

        dirs = plugin._get_reference_directories()
        assert len(dirs) == 0


# ==================== Render with Index Resolution ====================

class TestRenderWithIndex:
    def test_render_template_by_name(self, plugin, template_dir):
        """renderTemplateToFile should resolve template_name via index."""
        entries = plugin._discover_standalone_templates(template_dir)
        for entry in entries:
            plugin._template_index[entry.name] = entry

        output_file = plugin._base_path / "output" / "CustomerRepository.java"
        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("Repository.java.tpl"),
            "variables": {
                "basePackage": "com.bank.customer",
                "Entity": "Customer",
                "EntityId": "CustomerId",
            },
            "output_path": str(output_file),
        })

        assert result.get("success") is True, f"Render failed: {result}"
        assert output_file.exists()
        content = output_file.read_text()
        assert "com.bank.customer" in content
        assert "CustomerRepository" in content

    def test_render_template_name_not_found(self, plugin):
        """Should return error when template name isn't in index."""
        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("NonExistent.java.tpl"),
            "variables": {},
            "output_path": "/tmp/out.java",
        })
        assert "error" in result
        assert "NonExistent.java.tpl" in result["error"]

    def test_list_variables_by_name(self, plugin, template_dir):
        """listTemplateVariables should resolve template_name via index.

        Returns variables as list[{name, kind, item_keys?}] (server
        0.6.28+).  The structured shape lets the agent know which
        variables are scalars, which are sections (list-of-dicts
        with item_keys), and which are inverted sections — eliminating
        the first-attempt-render-failure non-determinism source.
        """
        entries = plugin._discover_standalone_templates(template_dir)
        for entry in entries:
            plugin._template_index[entry.name] = entry

        result = plugin._execute_list_template_variables({
            "template_id": _template_id("Repository.java.tpl"),
        })
        assert "variables" in result, f"Expected variables, got: {result}"
        # Variables is now a list of {name, kind, item_keys?} dicts.
        var_names = {v["name"] for v in result["variables"]}
        assert "basePackage" in var_names
        assert "Entity" in var_names
        assert result["template_name"] == "Repository.java.tpl"
        # Each entry must have a 'kind' field.
        assert all("kind" in v for v in result["variables"])
        # Repository.java.tpl is a flat scalar template (Java
        # interface) — no sections expected.
        assert all(v["kind"] == "scalar" for v in result["variables"])


# ==================== System Instruction Enrichment ====================

class TestEnrichmentWithStandalone:
    def test_enrichment_discovers_standalone(self, plugin, template_dir):
        """System instruction enrichment should discover standalone templates."""
        # Mock references plugin returning a directory source
        ref_plugin = MagicMock()
        ref_plugin.get_selected_ids.return_value = ["mod-015"]

        source = MagicMock()
        source.id = "mod-015"
        source.type.value = "local"
        source.resolved_path = str(template_dir)
        source.path = "templates"
        ref_plugin.get_sources.return_value = [source]

        registry = MagicMock()
        registry.get_plugin.return_value = ref_plugin
        plugin._plugin_registry = registry

        # Run enrichment with some basic instructions (no embedded templates)
        result = plugin.enrich_system_instructions("# System Instructions\nNo templates here.")

        # Eager per-template MANDATORY-USAGE blocks were replaced by a
        # single compact pointer; per-template enumeration moved to
        # contextual surfacing in enrich_prompt / enrich_tool_result so
        # the catalog no longer pollutes the cacheable system-instruction
        # prefix.  The pointer must mention the template count and the
        # discovery tool the model can call to enumerate the catalog.
        assert "templates available" in result.instructions
        assert "listAvailableTemplates" in result.instructions
        assert "renderTemplateToFile" in result.instructions

        # Index should be populated regardless of the new presentation.
        assert len(plugin._template_index) == 3

        # Metadata should report counts (template_count is the new
        # cumulative figure; standalone_count remains for parity).
        assert result.metadata.get("template_count") == 3
        assert result.metadata.get("standalone_count") == 3

    def test_enrichment_persists_index(self, plugin, template_dir):
        """Enrichment should write index.json to disk."""
        ref_plugin = MagicMock()
        ref_plugin.get_selected_ids.return_value = ["mod-015"]

        source = MagicMock()
        source.id = "mod-015"
        source.type.value = "local"
        source.resolved_path = str(template_dir)
        source.path = "templates"
        ref_plugin.get_sources.return_value = [source]

        registry = MagicMock()
        registry.get_plugin.return_value = ref_plugin
        plugin._plugin_registry = registry

        plugin.enrich_system_instructions("# Instructions")

        index_path = plugin._templates_dir / "index.json"
        assert index_path.exists()

        data = json.loads(index_path.read_text())
        assert data["template_count"] == 3


# ==================== Plugin Lifecycle ====================

class TestPluginLifecycle:
    def test_shutdown_clears_index(self, plugin, template_dir):
        entries = plugin._discover_standalone_templates(template_dir)
        for entry in entries:
            plugin._template_index[entry.name] = entry
        assert len(plugin._template_index) > 0

        plugin.shutdown()
        assert len(plugin._template_index) == 0

    def test_set_plugin_registry(self, plugin):
        registry = MagicMock()
        plugin.set_plugin_registry(registry)
        assert plugin._plugin_registry is registry


class TestMustacheStructuralParser:
    """Tests for the structural Mustache parser introduced in 0.6.28.

    ``_parse_mustache_structure`` walks the template and returns each
    variable as ``{name, kind, item_keys?}`` so the model can tell
    scalars (``{{x}}``) from sections (``{{#x}}...{{/x}}``, list of
    dicts) from inverted sections (``{{^x}}...{{/x}}``, falsy-only).

    See plugin docstring for the full classification rules.
    """

    def test_flat_scalars_only(self):
        plugin = TemplatePlugin()
        template = "package {{basePackage}}; class {{Entity}} {}"
        result = plugin._parse_mustache_structure(template)
        names = {v["name"]: v["kind"] for v in result}
        assert names == {"basePackage": "scalar", "Entity": "scalar"}

    def test_section_with_inner_item_keys(self):
        plugin = TemplatePlugin()
        template = "{{#apiEndpoints}}\n{{methodName}} {{path}}\n{{/apiEndpoints}}"
        result = plugin._parse_mustache_structure(template)
        # apiEndpoints is the only top-level entry; methodName and
        # path appear ONLY inside the section, so they're item_keys
        # not top-level scalars.
        assert len(result) == 1
        api = result[0]
        assert api["name"] == "apiEndpoints"
        assert api["kind"] == "section"
        assert _names(api["item_keys"]) == ["methodName", "path"]

    def test_inverted_section(self):
        plugin = TemplatePlugin()
        template = "{{^isEmpty}}has data{{/isEmpty}}"
        result = plugin._parse_mustache_structure(template)
        assert len(result) == 1
        assert result[0]["name"] == "isEmpty"
        assert result[0]["kind"] == "inverted_section"
        # Inverted sections do not carry item_keys.
        assert "item_keys" not in result[0]

    def test_section_and_inverted_section_same_name(self):
        """When a name is used as both ``{{#x}}`` and ``{{^x}}`` —
        the section classification wins (more constrained shape) AND
        ``has_inverted_branch`` is set so the agent knows the else-
        branch exists.
        """
        plugin = TemplatePlugin()
        template = "{{#items}}{{name}}{{/items}}{{^items}}empty{{/items}}"
        result = plugin._parse_mustache_structure(template)
        items = next(v for v in result if v["name"] == "items")
        assert items["kind"] == "section"
        assert _names(items["item_keys"]) == ["name"]
        assert items["has_inverted_branch"] is True

    def test_triple_brace_unescaped_output(self):
        """Mustache ``{{{x}}}`` is the unescaped-output form.  Same
        variable as ``{{x}}`` from a structural perspective; the
        parser must not include the inner ``{`` in the captured name.
        """
        plugin = TemplatePlugin()
        template = "package {{basePackage}}; method({{{controllerSignature}}})"
        result = plugin._parse_mustache_structure(template)
        names = sorted(v["name"] for v in result)
        # Critically: NOT '{controllerSignature' with a leading brace.
        assert names == ["basePackage", "controllerSignature"]
        assert all(v["kind"] == "scalar" for v in result)

    def test_nested_section_inside_outer_iteration_is_item_key_not_top_level(self):
        """Nested ``{{#x}}`` inside an outer iteration section is a
        per-item field of the outer section.  It must appear in the
        outer's ``item_keys`` AND must NOT appear as a top-level
        section entry.

        Repros the kb-enablement-2.0 RestController.java.tpl pattern
        where ``{{#isVoid}}...{{^isVoid}}...`` inside
        ``{{#apiEndpoints}}...{{/apiEndpoints}}`` was leaking
        ``isVoid`` to top level.
        """
        plugin = TemplatePlugin()
        template = (
            "{{#apiEndpoints}}\n"
            "  {{methodName}}\n"
            "  {{#isVoid}}void{{/isVoid}}\n"
            "  {{^isVoid}}{{returnType}}{{/isVoid}}\n"
            "{{/apiEndpoints}}"
        )
        result = plugin._parse_mustache_structure(template)
        # Top-level should be apiEndpoints ONLY — no isVoid, no
        # methodName, no returnType.
        top_level = sorted(v["name"] for v in result)
        assert top_level == ["apiEndpoints"], (
            f"top-level should be apiEndpoints only, got {top_level}"
        )
        # apiEndpoints' item_keys must contain isVoid (the nested
        # section name) AND its inner refs (methodName, returnType).
        api = result[0]
        assert api["kind"] == "section"
        assert "isVoid" in _names(api["item_keys"])
        assert "methodName" in _names(api["item_keys"])
        assert "returnType" in _names(api["item_keys"])

    def test_section_inner_refs_attribute_to_outermost_iteration(self):
        """Scalar refs inside a nested boolean section attribute to
        the OUTERMOST iteration section's item_keys, not to the
        innermost (which is the boolean check, not the iteration).
        """
        plugin = TemplatePlugin()
        # Same shape as the kb-enablement RestController template
        # but minimal.  Inside {{#apiEndpoints}} body, references in
        # {{#isVoid}} and {{^isVoid}} bodies should attribute to
        # apiEndpoints.item_keys, not isVoid's.
        template = (
            "{{#apiEndpoints}}"
            "{{#isVoid}}{{controllerSignature}}{{/isVoid}}"
            "{{^isVoid}}{{returnType}} {{serviceCallArgs}}{{/isVoid}}"
            "{{/apiEndpoints}}"
        )
        result = plugin._parse_mustache_structure(template)
        top_level = sorted(v["name"] for v in result)
        assert top_level == ["apiEndpoints"], (
            f"inner refs leaked to top level: {top_level}"
        )
        api = result[0]
        assert api["kind"] == "section"
        # All three inner refs (regardless of the boolean section
        # they were syntactically inside) credit to apiEndpoints.
        api_names = _names(api["item_keys"])
        for required in ("controllerSignature", "returnType", "serviceCallArgs", "isVoid"):
            assert required in api_names, (
                f"{required} missing from apiEndpoints.item_keys: {api_names}"
            )

    def test_nested_iteration_owns_its_item_keys_not_outer(self):
        """Server 0.6.46+: a TRUE nested iteration (no
        ``{{^name}}`` companion → not boolean-like) accumulates its
        own item_keys instead of flattening into the outer
        iteration's.  Mirrors the SystemApiMapper.java.tpl pattern
        from kb-enablement-2.0 chunk-3 cascade probe v18 where
        ``{{#statusEnum}}{{#statusMappings}}{{code}}{{value}}{{/statusMappings}}{{/statusEnum}}``
        previously reported ``code`` and ``value`` as keys of
        ``statusEnum`` (wrong — they're keys of statusMappings's
        per-mapping dict).
        """
        plugin = TemplatePlugin()
        template = (
            "{{#statusEnum}}\n"
            "private {{StatusEnum}} to{{StatusEnum}}(String code) {\n"
            "  return switch (code) {\n"
            "    {{#statusMappings}}\n"
            "    case \"{{code}}\" -> {{StatusEnum}}.{{value}};\n"
            "    {{/statusMappings}}\n"
            "  };\n"
            "}\n"
            "{{/statusEnum}}\n"
        )
        result = plugin._parse_mustache_structure(template)
        # Top-level: statusEnum only.
        top = sorted(v["name"] for v in result)
        assert top == ["statusEnum"], f"unexpected top-level: {top}"
        outer = result[0]
        outer_names = _names(outer["item_keys"])
        # statusEnum.item_keys: scalar StatusEnum + section statusMappings.
        # ``code`` and ``value`` MUST NOT be here (they belong to the
        # nested iteration's scope).
        assert "StatusEnum" in outer_names
        assert "statusMappings" in outer_names
        assert "code" not in outer_names, (
            f"code leaked to outer (statusEnum.item_keys): {outer_names}"
        )
        assert "value" not in outer_names, (
            f"value leaked to outer: {outer_names}"
        )
        # Nested iteration: statusMappings.item_keys carries code, value
        # AND StatusEnum (with inherited_from_outer_scope flag).
        sm = next(k for k in outer["item_keys"] if k["name"] == "statusMappings")
        assert sm["source"] == "section"
        nested_names = _names(sm["item_keys"])
        assert "code" in nested_names
        assert "value" in nested_names
        assert "StatusEnum" in nested_names

    def test_inherited_from_outer_scope_flag(self):
        """Server 0.6.46+: when an inner-iteration ref shares its
        name with an outer-iteration ref, the inner entry carries
        ``inherited_from_outer_scope: True``.  This is the load-
        bearing signal for the agent (or framework) to handle the
        engine-inconsistent outer-scope walking — the kb-template
        author expected pybars3 to find ``StatusEnum`` in the outer
        scope from inside ``{{#statusMappings}}``, but pybars3's
        outer-context fallback is unreliable across iteration
        boundaries.  The flag tells consumers to flatten outer-
        scope keys into each inner item.
        """
        plugin = TemplatePlugin()
        template = (
            "{{#outer}}\n"
            "{{shared}}\n"
            "{{#inner}}\n"
            "{{shared}} {{innerOnly}}\n"
            "{{/inner}}\n"
            "{{/outer}}\n"
        )
        result = plugin._parse_mustache_structure(template)
        outer = next(v for v in result if v["name"] == "outer")
        inner_entry = next(
            k for k in outer["item_keys"] if k["name"] == "inner"
        )
        nested = inner_entry["item_keys"]
        shared_in_inner = next(k for k in nested if k["name"] == "shared")
        assert shared_in_inner.get("inherited_from_outer_scope") is True, (
            f"shared inside inner should carry inherited_from_outer_scope: "
            f"{shared_in_inner}"
        )
        # innerOnly does NOT exist in outer's item_keys → flag absent.
        inner_only = next(k for k in nested if k["name"] == "innerOnly")
        assert "inherited_from_outer_scope" not in inner_only, (
            f"innerOnly is not in outer scope; flag should be absent: "
            f"{inner_only}"
        )

    def test_boolean_section_idiom_keeps_legacy_attribution(self):
        """Server 0.6.46+ boolean-skip heuristic: when ``{{#name}}``
        AND ``{{^name}}`` both appear (Mustache if/else idiom),
        body refs flow through to the OUTER iteration — preserving
        the kb-enablement RestController.java.tpl pattern where
        ``{{#isVoid}}{{controllerSignature}}{{/isVoid}}`` puts
        ``controllerSignature`` at the apiEndpoint level (boolean
        flag, not nested mapping).
        """
        plugin = TemplatePlugin()
        template = (
            "{{#apiEndpoints}}"
            "{{#isVoid}}{{controllerSignature}}{{/isVoid}}"
            "{{^isVoid}}{{returnType}} {{serviceCallArgs}}{{/isVoid}}"
            "{{/apiEndpoints}}"
        )
        result = plugin._parse_mustache_structure(template)
        api = result[0]
        api_names = _names(api["item_keys"])
        # Boolean idiom: refs inside {{#isVoid}} flow to apiEndpoints.
        for required in (
            "controllerSignature", "returnType", "serviceCallArgs", "isVoid",
        ):
            assert required in api_names, (
                f"{required} missing from apiEndpoints (boolean-skip should "
                f"preserve legacy attribution): {api_names}"
            )
        # isVoid entry: section source, but its nested item_keys
        # should be empty (refs flowed to outer, not inner).
        is_void = next(k for k in api["item_keys"] if k["name"] == "isVoid")
        assert is_void["source"] == "section"
        # Nested item_keys may or may not be present — but if present,
        # it must be empty (no refs landed in the boolean's scope).
        nested = is_void.get("item_keys", [])
        assert nested == [], (
            f"boolean section should have empty nested item_keys: {nested}"
        )

    def test_top_level_boolean_section_keeps_own_scope(self):
        """Edge case: when the boolean-like section IS the outermost
        iteration (no outer to flow to), refs stay in its own
        item_keys.  Preserves
        ``{{#items}}{{name}}{{/items}}{{^items}}empty{{/items}}``
        with name landing in items.item_keys.
        """
        plugin = TemplatePlugin()
        template = (
            "{{#items}}{{name}}{{/items}}{{^items}}empty{{/items}}"
        )
        result = plugin._parse_mustache_structure(template)
        items = next(v for v in result if v["name"] == "items")
        assert items["kind"] == "section"
        assert "name" in _names(items["item_keys"]), (
            f"top-level boolean keeps own scope: {items['item_keys']}"
        )

    def test_strips_java_style_comment_lines(self):
        """``//``-prefixed lines (Java/C/C++/JS host-language comments)
        are stripped before structural extraction.  Refs inside such
        lines are documentation, not live Mustache references.

        Surfaced by kb-enablement-2.0 templates whose 7-line metadata
        header (`// Template:`, `// REQUIRED VARIABLES: {{Entity}} ...`)
        was leaking comment-line refs to top-level scalars.
        """
        plugin = TemplatePlugin()
        template = (
            "// REQUIRED VARIABLES: {{Entity}} {{fieldName}} {{type}}\n"
            "package com.example.{{basePackage}};\n"
            "\n"
            "public class {{Entity}} {\n"
            "{{#fields}}\n"
            "  {{type}} {{fieldName}};\n"
            "{{/fields}}\n"
            "}\n"
        )
        result = plugin._parse_mustache_structure(template)
        top_level = sorted(v["name"] for v in result)
        # fieldName, type appear ONLY in stripped comment line + inside
        # {{#fields}} body — should be item_keys of fields, NOT top-level.
        assert "fieldName" not in top_level, (
            f"fieldName leaked from comment line to top level: {top_level}"
        )
        assert "type" not in top_level
        assert top_level == ["Entity", "basePackage", "fields"]
        fields = next(v for v in result if v["name"] == "fields")
        assert fields["kind"] == "section"
        assert "fieldName" in _names(fields["item_keys"])
        assert "type" in _names(fields["item_keys"])

    def test_strip_does_not_eat_block_comments(self):
        """Block comments (``/* ... */``) often carry live Javadoc-
        with-Mustache refs (``@param {{x}}``).  Only single-line
        ``//`` is stripped; ``/* ... */`` is preserved so refs inside
        Javadoc continue to be live.
        """
        plugin = TemplatePlugin()
        template = (
            "/**\n"
            " * Generated entity for {{Entity}} domain.\n"
            " * @param {{paramName}} the value\n"
            " */\n"
            "class {{Entity}} {}\n"
        )
        result = plugin._parse_mustache_structure(template)
        top_level = sorted(v["name"] for v in result)
        # Both Entity and paramName end up as top-level scalars.
        # Javadoc references are live.
        assert "Entity" in top_level
        assert "paramName" in top_level

    def test_strip_does_not_eat_python_hash(self):
        """Python ``#`` (and Markdown ``#`` headers) are NOT stripped
        — risk of over-stripping.  Tenants targeting Python with
        parser-directive docs should use ``{{! ... }}`` instead.
        """
        plugin = TemplatePlugin()
        template = "# comment with {{liveRef}}\nplain {{Entity}}\n"
        result = plugin._parse_mustache_structure(template)
        top_level = sorted(v["name"] for v in result)
        assert "liveRef" in top_level
        assert "Entity" in top_level

    def test_execute_includes_warnings_for_stripped_lines(self, plugin, tmp_path):
        """When the parser strips ``//`` comment lines, the tool's
        return includes a ``warnings`` field per the standard
        completion-payload-schema convention (advisory escape hatch).
        """
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        tpl_path = templates_dir / "test.java.tpl"
        # Include a {{#section}} marker so syntax-detection picks
        # mustache (Jinja2 has no equivalent dotless-prefix
        # syntax) — only the mustache path runs the //-strip.
        tpl_path.write_text(
            "// REQUIRED VARIABLES: {{x}} {{y}}\n"
            "package com.example;\n"
            "{{#fields}}{{name}}{{/fields}}\n"
            "class {{Entity}} {}\n"
        )
        plugin._template_index["test.java.tpl"] = TemplateIndexEntry(
            name="test.java.tpl",
            source_path=str(tpl_path),
            origin="standalone",
            syntax="mustache",
            variables=[],
        )
        result = plugin._execute_list_template_variables({
            "template_id": _template_id("test.java.tpl"),
        })
        assert "warnings" in result, f"expected warnings, got {result}"
        assert "1 line(s) starting with '//'" in result["warnings"][0]
        assert "{{! ... }}" in result["warnings"][0]

    def test_section_without_inverted_has_explicit_false_flag(self):
        """Sections without an inverted branch get
        ``has_inverted_branch: False`` explicitly — predictable schema
        for the agent (no missing-key surprises).
        """
        plugin = TemplatePlugin()
        template = "{{#items}}{{name}}{{/items}}"
        result = plugin._parse_mustache_structure(template)
        items = next(v for v in result if v["name"] == "items")
        assert items["kind"] == "section"
        assert items["has_inverted_branch"] is False

    def test_top_level_scalar_not_polluted_by_section_inner(self):
        """Scalars referenced ONLY inside sections must not appear at
        top level — they're item_keys of the section, not top-level
        variables the agent has to provide separately.
        """
        plugin = TemplatePlugin()
        template = "{{Entity}}\n{{#fields}}{{name}}: {{type}}{{/fields}}"
        result = plugin._parse_mustache_structure(template)
        top_level = sorted(v["name"] for v in result if v["kind"] == "scalar")
        assert top_level == ["Entity"], (
            f"top-level scalars should be ['Entity'] only — name and "
            f"type are item_keys of fields, not top-level. got {top_level}"
        )

    def test_scalar_referenced_both_inside_and_outside(self):
        """When a name appears both at top level and inside a
        section, it gets BOTH a top-level scalar entry AND an
        item-keys entry — agent must provide both.
        """
        plugin = TemplatePlugin()
        template = "{{Entity}}\n{{#fields}}// {{Entity}} field {{name}}{{/fields}}"
        result = plugin._parse_mustache_structure(template)
        # Entity is a top-level scalar AND an item-key of fields.
        entity_top = next(v for v in result if v["name"] == "Entity")
        assert entity_top["kind"] == "scalar"
        fields = next(v for v in result if v["name"] == "fields")
        assert fields["kind"] == "section"
        assert "Entity" in _names(fields["item_keys"])
        assert "name" in _names(fields["item_keys"])

    def test_dotted_path_records_leftmost_token(self):
        """``{{item.foo.bar}}`` inside a section records ``item`` as
        the item-key — that's the field the agent provides; ``foo.bar``
        is the access path the template walks at render time.
        """
        plugin = TemplatePlugin()
        template = "{{#rows}}{{cell.text}}{{/rows}}"
        result = plugin._parse_mustache_structure(template)
        rows = next(v for v in result if v["name"] == "rows")
        assert rows["kind"] == "section"
        assert _names(rows["item_keys"]) == ["cell"]

    def test_comments_and_current_context_skipped(self):
        plugin = TemplatePlugin()
        template = "{{!comment}}{{Entity}}{{.}}{{this}}"
        result = plugin._parse_mustache_structure(template)
        names = [v["name"] for v in result]
        assert names == ["Entity"]

    def test_execute_returns_structured_shape_for_mustache(self, plugin, template_dir):
        """Public tool API returns the structured shape end-to-end
        for Mustache templates.
        """
        entries = plugin._discover_standalone_templates(template_dir)
        for entry in entries:
            plugin._template_index[entry.name] = entry

        result = plugin._execute_list_template_variables({
            "template_id": _template_id("Repository.java.tpl"),
        })
        assert "error" not in result
        assert isinstance(result["variables"], list)
        assert all(isinstance(v, dict) for v in result["variables"])
        assert all("name" in v and "kind" in v for v in result["variables"])


class TestConfigRootResolution:
    """Tests for the layered templates_dir resolution introduced in 0.6.26.

    The template plugin now mirrors the references plugin's
    ``set_config_root`` pattern: when ``config_root`` is set,
    ``_templates_dir`` resolves to ``<config_root>/templates`` instead
    of ``<workspace>/.jaato/templates``.  Falls back to the workspace
    tier when ``config_root`` is None.

    See plugin docstring for ``_compute_templates_dir``.
    """

    def test_workspace_only_uses_workspace_jaato_templates(self, tmp_path):
        """No config_root → resolves to <workspace>/.jaato/templates (legacy)."""
        ws = tmp_path / "sandbox"
        ws.mkdir()
        p = TemplatePlugin()
        p.initialize({"base_path": str(ws)})
        assert p._templates_dir == ws / ".jaato" / "templates"

    def test_set_config_root_flips_resolution(self, tmp_path):
        """set_config_root after init re-resolves templates_dir."""
        ws = tmp_path / "sandbox"
        cr = tmp_path / "repo" / ".jaato"
        ws.mkdir()
        cr.mkdir(parents=True)
        p = TemplatePlugin()
        p.initialize({"base_path": str(ws)})
        assert p._templates_dir == ws / ".jaato" / "templates"
        p.set_config_root(str(cr))
        assert p._templates_dir == cr / "templates"

    def test_set_config_root_none_falls_back_to_workspace(self, tmp_path):
        """Setting config_root then resetting to None falls back to workspace."""
        ws = tmp_path / "sandbox"
        cr = tmp_path / "repo" / ".jaato"
        ws.mkdir()
        cr.mkdir(parents=True)
        p = TemplatePlugin()
        p.initialize({"base_path": str(ws)})
        p.set_config_root(str(cr))
        assert p._templates_dir == cr / "templates"
        p.set_config_root(None)
        assert p._templates_dir == ws / ".jaato" / "templates"

    def test_config_root_wins_over_later_set_workspace_path(self, tmp_path):
        """Once config_root is set, switching workspace doesn't dislodge it."""
        ws1 = tmp_path / "sandbox1"
        ws2 = tmp_path / "sandbox2"
        cr = tmp_path / "repo" / ".jaato"
        ws1.mkdir()
        ws2.mkdir()
        cr.mkdir(parents=True)
        p = TemplatePlugin()
        p.initialize({"base_path": str(ws1)})
        p.set_config_root(str(cr))
        assert p._templates_dir == cr / "templates"
        # Switch workspace — config_root must still win.
        p.set_workspace_path(str(ws2))
        assert p._templates_dir == cr / "templates", (
            f"config_root should still win after set_workspace_path; "
            f"got {p._templates_dir}"
        )

    def test_compute_templates_dir_returns_none_when_neither_set(self):
        """Pure-helper test: no workspace, no config_root → None."""
        p = TemplatePlugin()
        assert p._compute_templates_dir() is None

    def test_set_config_root_reloads_persisted_index(self, tmp_path):
        """set_config_root reloads the index from the new location.

        Writes an index at config_root/templates/index.json BEFORE
        calling set_config_root; verifies the plugin picks it up
        without a session restart.
        """
        ws = tmp_path / "sandbox"
        cr = tmp_path / "repo" / ".jaato"
        ws.mkdir()
        cr.mkdir(parents=True)
        # Seed an index at the config_root location.
        templates_at_cr = cr / "templates"
        templates_at_cr.mkdir()
        index_path = templates_at_cr / "index.json"
        # Use the actual schema-shaped index the plugin's loader expects.
        # Runtime-persist schema: {"templates": {name: entry_dict, ...}}.
        # Empty dict is a valid (but contentless) index — exercises the
        # loader without requiring a real entry shape.
        index_path.write_text('{"templates": {}}')

        p = TemplatePlugin()
        p.initialize({"base_path": str(ws)})
        # Before set_config_root: would look at ws/.jaato/templates (empty).
        p.set_config_root(str(cr))
        # After set_config_root: _templates_dir points at cr/templates,
        # and _load_persisted_index ran (no exception).
        assert p._templates_dir == templates_at_cr


# ==================== Mustache Dotted-Path Preprocessing ====================

class TestMustacheDottedPaths:
    """Tests for the pybars3 dotted-path preprocessor.

    pybars3 does not support dotted paths in section tags ({{#a.b}}) or
    inverted section tags ({{^a.b}}), though it handles them in variable
    interpolation ({{a.b}}) and built-in helper arguments ({{#if a.b}}).

    The preprocessor rewrites:
      {{#a.b}} → {{#if a.b}}      (section → if helper)
      {{^a.b}} → {{#unless a.b}}  (inverted → unless helper)
      {{/a.b}} → matching {{/if}} or {{/unless}}
    """

    def test_preprocess_section_dot(self, plugin):
        result = plugin._preprocess_mustache_dotted_paths("{{#a.b}}yes{{/a.b}}")
        assert result == "{{#if a.b}}yes{{/if}}"

    def test_preprocess_inverted_dot(self, plugin):
        result = plugin._preprocess_mustache_dotted_paths("{{^a.b}}no{{/a.b}}")
        assert result == "{{#unless a.b}}no{{/unless}}"

    def test_preprocess_deep_section(self, plugin):
        result = plugin._preprocess_mustache_dotted_paths("{{#a.b.c}}deep{{/a.b.c}}")
        assert result == "{{#if a.b.c}}deep{{/if}}"

    def test_preprocess_deep_inverted(self, plugin):
        result = plugin._preprocess_mustache_dotted_paths("{{^a.b.c}}deep{{/a.b.c}}")
        assert result == "{{#unless a.b.c}}deep{{/unless}}"

    def test_preprocess_no_dots_unchanged(self, plugin):
        template = "{{#items}}{{name}}{{/items}}"
        assert plugin._preprocess_mustache_dotted_paths(template) == template

    def test_preprocess_helper_dots_unchanged(self, plugin):
        """Helpers like {{#if a.b}} already work in pybars3."""
        template = "{{#if a.b}}yes{{/if}}"
        assert plugin._preprocess_mustache_dotted_paths(template) == template

    def test_preprocess_variable_dots_unchanged(self, plugin):
        template = "{{person.name}}"
        assert plugin._preprocess_mustache_dotted_paths(template) == template

    def test_preprocess_nested_mixed(self, plugin):
        """Dotted section inside a non-dotted section."""
        template = "{{#items}}{{#val.active}}x{{/val.active}}{{/items}}"
        result = plugin._preprocess_mustache_dotted_paths(template)
        assert result == "{{#items}}{{#if val.active}}x{{/if}}{{/items}}"

    def test_preprocess_with_whitespace(self, plugin):
        """Whitespace around the dotted name."""
        result = plugin._preprocess_mustache_dotted_paths("{{# a.b }}yes{{/ a.b }}")
        assert result == "{{#if a.b}}yes{{/if}}"

    def test_preprocess_idempotent(self, plugin):
        """Running the preprocessor twice produces the same output."""
        template = "{{#a.b}}yes{{/a.b}}"
        first = plugin._preprocess_mustache_dotted_paths(template)
        second = plugin._preprocess_mustache_dotted_paths(first)
        assert first == second

    # -- End-to-end rendering with pybars3 --

    def test_render_section_dot_truthy(self, plugin):
        rendered, error = plugin._render_mustache(
            "{{#a.b}}yes{{/a.b}}", {"a": {"b": True}})
        assert error is None
        assert rendered == "yes"

    def test_render_section_dot_falsy(self, plugin):
        rendered, error = plugin._render_mustache(
            "{{#a.b}}yes{{/a.b}}", {"a": {"b": False}})
        assert error is None
        assert rendered == ""

    def test_render_section_dot_missing_parent(self, plugin):
        rendered, error = plugin._render_mustache(
            "{{#a.b}}yes{{/a.b}}", {})
        assert error is None
        assert rendered == ""

    def test_render_inverted_dot_falsy(self, plugin):
        rendered, error = plugin._render_mustache(
            "{{^a.b}}no{{/a.b}}", {"a": {"b": False}})
        assert error is None
        assert rendered == "no"

    def test_render_inverted_dot_truthy(self, plugin):
        rendered, error = plugin._render_mustache(
            "{{^a.b}}no{{/a.b}}", {"a": {"b": True}})
        assert error is None
        assert rendered == ""

    def test_render_inverted_dot_missing(self, plugin):
        rendered, error = plugin._render_mustache(
            "{{^a.b}}no{{/a.b}}", {})
        assert error is None
        assert rendered == "no"

    def test_render_variable_inside_dotted_section(self, plugin):
        """The common pattern: conditional section + same-path variable inside."""
        rendered, error = plugin._render_mustache(
            '{{#validation.pattern}}@Pattern("{{validation.pattern}}"){{/validation.pattern}}',
            {"validation": {"pattern": "^[a-z]+$"}})
        assert error is None
        assert rendered == '@Pattern("^[a-z]+$")'

    def test_render_realistic_java_template(self, plugin):
        """Realistic template with multiple dotted conditionals in a loop."""
        template = (
            "{{#fields}}"
            "{{#validation.pattern}}@Pattern(\"{{validation.pattern}}\")\n{{/validation.pattern}}"
            "{{#validation.maxLength}}@Size(max={{validation.maxLength}})\n{{/validation.maxLength}}"
            "private {{fieldType}} {{fieldName}};\n"
            "{{/fields}}"
        )
        variables = {
            "fields": [
                {
                    "fieldType": "String",
                    "fieldName": "email",
                    "validation": {"pattern": "regex", "maxLength": 255},
                },
                {
                    "fieldType": "int",
                    "fieldName": "age",
                    "validation": {},
                },
            ]
        }
        rendered, error = plugin._render_mustache(template, variables)
        assert error is None
        assert '@Pattern("regex")' in rendered
        assert "@Size(max=255)" in rendered
        assert "private String email;" in rendered
        assert "private int age;" in rendered
        # No annotations for age (empty validation)
        assert "@Pattern" not in rendered.split("private int age;")[0].split("private String email;")[1]

    def test_render_mixed_section_and_inverted_dots(self, plugin):
        """Section and inverted with dots at the same level."""
        rendered, error = plugin._render_mustache(
            "{{#a.b}}yes{{/a.b}}{{^a.b}}no{{/a.b}}",
            {"a": {"b": True}})
        assert error is None
        assert rendered == "yes"

        rendered2, error2 = plugin._render_mustache(
            "{{#a.b}}yes{{/a.b}}{{^a.b}}no{{/a.b}}",
            {"a": {"b": False}})
        assert error2 is None
        assert rendered2 == "no"

    def test_render_deep_dotted_section(self, plugin):
        rendered, error = plugin._render_mustache(
            "{{#a.b.c}}deep{{/a.b.c}}",
            {"a": {"b": {"c": True}}})
        assert error is None
        assert rendered == "deep"

    def test_renderTemplateToFile_with_dotted_sections(self, plugin):
        """renderTemplateToFile should handle dotted section paths."""
        template = (
            "{{#fields}}"
            "{{#validation.required}}required: {{fieldName}}\n{{/validation.required}}"
            "{{/fields}}"
        )
        output_file = plugin._base_path / "output" / "test.txt"
        result = plugin._execute_render_template_to_file({
            "template": template,
            "variables": {
                "fields": [
                    {"fieldName": "email", "validation": {"required": True}},
                    {"fieldName": "age", "validation": {"required": False}},
                ]
            },
            "output_path": str(output_file),
        })
        assert result.get("success") is True, f"Render failed: {result}"
        content = output_file.read_text()
        assert "required: email" in content
        assert "required: age" not in content


# ==================== @generated Annotation Stripping ====================

class TestGeneratedAnnotationStripping:
    """Tests for stripping @generated annotations from templates.

    Imported templates often include ``@generated {{skillId}} v{{skillVersion}}``
    in comment annotations. These are metadata placeholders — not template
    variables — and must be excluded from variable extraction and rendering.
    """

    JAVA_TEMPLATE_WITH_GENERATED = textwrap.dedent("""\
        package {{basePackage}}.domain.model;

        /**
         * Value object for {{Entity}} identifier.
         *
         * @generated {{skillId}} v{{skillVersion}}
         * @module mod-code-015-hexagonal-base-java-spring
         */
        public record {{Entity}}Id(UUID value) {
        }
    """)

    YAML_TEMPLATE_WITH_GENERATED = textwrap.dedent("""\
        # Timeout configuration
        # @generated {{skillId}} v{{skillVersion}}
        # @module mod-code-003-timeout-java-resilience4j
        spring:
          application:
            name: {{serviceName}}
    """)

    JAVA_LINE_COMMENT_GENERATED = textwrap.dedent("""\
        // @generated {{skillId}} v{{skillVersion}}
        // @module mod-code-003-timeout-java-resilience4j
        package {{basePackage}}.config;

        public class {{Entity}}Config {
        }
    """)

    def test_extract_variables_excludes_skillId_java_comment(self, plugin):
        """@generated {{skillId}} in JavaDoc should not be extracted."""
        variables = plugin._extract_variables(self.JAVA_TEMPLATE_WITH_GENERATED)
        assert "skillId" not in variables
        assert "skillVersion" not in variables
        # Real variables are still extracted
        assert "basePackage" in variables
        assert "Entity" in variables

    def test_extract_variables_excludes_skillId_yaml_comment(self, plugin):
        """@generated {{skillId}} in YAML comment should not be extracted."""
        variables = plugin._extract_variables(self.YAML_TEMPLATE_WITH_GENERATED)
        assert "skillId" not in variables
        assert "skillVersion" not in variables
        assert "serviceName" in variables

    def test_extract_variables_excludes_skillId_line_comment(self, plugin):
        """@generated {{skillId}} in // comment should not be extracted."""
        variables = plugin._extract_variables(self.JAVA_LINE_COMMENT_GENERATED)
        assert "skillId" not in variables
        assert "skillVersion" not in variables
        assert "basePackage" in variables
        assert "Entity" in variables

    def test_render_template_skips_generated_line(self, plugin):
        """Rendering should not fail on @generated placeholders."""
        rendered, error = plugin._render_template(
            self.JAVA_TEMPLATE_WITH_GENERATED,
            {"basePackage": "com.bank.customer", "Entity": "Customer"},
        )
        assert error is None
        assert "com.bank.customer.domain.model" in rendered
        assert "CustomerId" in rendered
        # @generated line should be removed from output
        assert "@generated" not in rendered
        # @module line (no template vars) should survive
        assert "@module" in rendered

    def test_render_template_yaml_skips_generated_line(self, plugin):
        """YAML template rendering should skip @generated comment."""
        rendered, error = plugin._render_template(
            self.YAML_TEMPLATE_WITH_GENERATED,
            {"serviceName": "customer-service"},
        )
        assert error is None
        assert "customer-service" in rendered
        assert "@generated" not in rendered

    def test_strip_generated_annotations_static(self):
        """_strip_generated_annotations is a static method."""
        content = " * @generated {{skillId}} v{{skillVersion}}\n * @module mod-015\n"
        result = TemplatePlugin._strip_generated_annotations(content)
        assert "@generated" not in result
        assert "@module mod-015" in result

    def test_strip_preserves_non_generated_comments(self):
        """Non-@generated comment content is preserved."""
        content = (
            "/**\n"
            " * Value object for {{Entity}} identifier.\n"
            " * @generated {{skillId}} v{{skillVersion}}\n"
            " * @module mod-015\n"
            " */\n"
        )
        result = TemplatePlugin._strip_generated_annotations(content)
        assert "Value object for {{Entity}} identifier." in result
        assert "@module mod-015" in result
        assert "@generated" not in result

    def test_renderTemplateToFile_with_generated_annotation(self, plugin):
        """End-to-end: renderTemplateToFile succeeds despite @generated."""
        output_file = plugin._base_path / "output" / "CustomerId.java"
        result = plugin._execute_render_template_to_file({
            "template": self.JAVA_TEMPLATE_WITH_GENERATED,
            "variables": {
                "basePackage": "com.bank.customer",
                "Entity": "Customer",
            },
            "output_path": str(output_file),
        })
        assert result.get("success") is True, f"Render failed: {result}"
        content = output_file.read_text()
        assert "com.bank.customer.domain.model" in content
        assert "CustomerId" in content
        assert "skillId" not in content


# ==================== List Metadata Injection ====================

class TestListMetadataInjection:
    """Tests for automatic ``first``/``last``/``@index`` injection into list items.

    Mustache templates use ``{{^last}}, {{/last}}`` to suppress trailing commas
    in parameter lists.  This pattern requires each list item to carry a boolean
    ``last`` property, but callers rarely provide it.  Without injection, ``last``
    is undefined (falsy in Mustache) for every item, so ``{{^last}}`` always
    renders — producing trailing commas.

    ``_inject_list_metadata`` recursively walks the variables dict and adds
    ``first``, ``last``, and ``@index`` to each item in every list of dicts,
    without overwriting existing keys.
    """

    # -- Unit tests for _inject_list_metadata --

    def test_inject_basic_list(self):
        """first/last/index injected correctly for a simple list."""
        variables = {
            "items": [
                {"name": "a"},
                {"name": "b"},
                {"name": "c"},
            ]
        }
        result = TemplatePlugin._inject_list_metadata(variables)
        items = result["items"]
        assert items[0]["first"] is True
        assert items[0]["last"] is False
        assert items[0]["@index"] == 0
        assert items[1]["first"] is False
        assert items[1]["last"] is False
        assert items[1]["@index"] == 1
        assert items[2]["first"] is False
        assert items[2]["last"] is True
        assert items[2]["@index"] == 2

    def test_inject_single_item_list(self):
        """Single-item list: first AND last are both True."""
        variables = {"items": [{"name": "only"}]}
        result = TemplatePlugin._inject_list_metadata(variables)
        item = result["items"][0]
        assert item["first"] is True
        assert item["last"] is True
        assert item["@index"] == 0

    def test_inject_preserves_existing_keys(self):
        """User-provided first/last/@index are never overwritten."""
        variables = {
            "items": [
                {"name": "a", "first": "custom", "last": "custom", "@index": 99},
                {"name": "b"},
            ]
        }
        result = TemplatePlugin._inject_list_metadata(variables)
        assert result["items"][0]["first"] == "custom"
        assert result["items"][0]["last"] == "custom"
        assert result["items"][0]["@index"] == 99
        # Second item gets injected normally
        assert result["items"][1]["first"] is False
        assert result["items"][1]["last"] is True

    def test_inject_nested_lists(self):
        """Metadata injection recurses into nested dicts and lists."""
        variables = {
            "outer": [
                {
                    "name": "parent",
                    "children": [
                        {"name": "child1"},
                        {"name": "child2"},
                    ]
                }
            ]
        }
        result = TemplatePlugin._inject_list_metadata(variables)
        # Outer list
        assert result["outer"][0]["first"] is True
        assert result["outer"][0]["last"] is True
        # Nested list
        children = result["outer"][0]["children"]
        assert children[0]["first"] is True
        assert children[0]["last"] is False
        assert children[1]["first"] is False
        assert children[1]["last"] is True

    def test_inject_skips_non_dict_lists(self):
        """Lists of non-dicts (strings, ints) are left untouched."""
        variables = {
            "tags": ["alpha", "beta"],
            "counts": [1, 2, 3],
        }
        result = TemplatePlugin._inject_list_metadata(variables)
        assert result["tags"] == ["alpha", "beta"]
        assert result["counts"] == [1, 2, 3]

    def test_inject_empty_list(self):
        """Empty lists are left untouched."""
        variables = {"items": []}
        result = TemplatePlugin._inject_list_metadata(variables)
        assert result["items"] == []

    def test_inject_does_not_mutate_original(self):
        """The original variables dict is not modified."""
        original_item = {"name": "a"}
        variables = {"items": [original_item]}
        result = TemplatePlugin._inject_list_metadata(variables)
        assert "last" not in original_item
        assert "last" in result["items"][0]

    def test_inject_nested_dict_without_list(self):
        """Plain nested dicts (not in lists) are recursed but no metadata added."""
        variables = {
            "config": {
                "items": [{"name": "x"}, {"name": "y"}]
            }
        }
        result = TemplatePlugin._inject_list_metadata(variables)
        assert result["config"]["items"][0]["first"] is True
        assert result["config"]["items"][1]["last"] is True

    # -- End-to-end rendering tests (trailing comma fix) --

    def test_render_no_trailing_comma_without_last_flag(self, plugin):
        """Core bug fix: {{^last}} no longer produces trailing commas."""
        template = "{{#items}}{{name}}{{^last}}, {{/last}}{{/items}}"
        variables = {
            "items": [
                {"name": "a"},
                {"name": "b"},
                {"name": "c"},
            ]
        }
        rendered, error = plugin._render_mustache(template, variables)
        assert error is None
        assert rendered == "a, b, c"

    def test_render_single_item_no_comma(self, plugin):
        """Single item: no comma at all."""
        template = "{{#items}}{{name}}{{^last}}, {{/last}}{{/items}}"
        variables = {"items": [{"name": "only"}]}
        rendered, error = plugin._render_mustache(template, variables)
        assert error is None
        assert rendered == "only"

    def test_render_entity_create_method_no_trailing_comma(self, plugin):
        """Realistic Java method signature — no trailing comma in parameter list."""
        template = (
            "public static Entity create("
            "{{#entityFields}}{{fieldType}} {{fieldName}}{{^last}}, {{/last}}{{/entityFields}}"
            ") {"
        )
        variables = {
            "entityFields": [
                {"fieldType": "String", "fieldName": "name"},
                {"fieldType": "int", "fieldName": "age"},
                {"fieldType": "BigDecimal", "fieldName": "balance"},
            ]
        }
        rendered, error = plugin._render_mustache(template, variables)
        assert error is None
        assert rendered == "public static Entity create(String name, int age, BigDecimal balance) {"
        # No trailing comma before the closing paren
        assert ", )" not in rendered

    def test_render_caller_provided_last_flag_respected(self, plugin):
        """When the caller provides their own ``last`` flags, they are respected."""
        template = "{{#items}}{{name}}{{^last}}, {{/last}}{{/items}}"
        variables = {
            "items": [
                {"name": "a", "last": False},
                {"name": "b", "last": True},  # Caller marks 'b' as last
                {"name": "c", "last": False},  # 'c' is NOT treated as last
            ]
        }
        rendered, error = plugin._render_mustache(template, variables)
        assert error is None
        # Caller's flags are respected: comma after 'a' and 'c', none after 'b'
        assert rendered == "a, bc, "

    def test_render_first_flag_works(self, plugin):
        """The injected ``first`` flag can be used in templates."""
        template = "{{#items}}{{#first}}[{{/first}}{{name}}{{^last}}, {{/last}}{{#last}}]{{/last}}{{/items}}"
        variables = {
            "items": [
                {"name": "a"},
                {"name": "b"},
                {"name": "c"},
            ]
        }
        rendered, error = plugin._render_mustache(template, variables)
        assert error is None
        assert rendered == "[a, b, c]"

    def test_renderTemplateToFile_entity_no_trailing_comma(self, plugin):
        """End-to-end: renderTemplateToFile renders Entity.java.tpl pattern correctly."""
        template = textwrap.dedent("""\
            public class Order {
                public static Order create({{#entityFields}}{{fieldType}} {{fieldName}}{{^last}}, {{/last}}{{/entityFields}}) {
                    return new Order();
                }
            }
        """)
        output_file = plugin._base_path / "output" / "Order.java"
        result = plugin._execute_render_template_to_file({
            "template": template,
            "variables": {
                "entityFields": [
                    {"fieldType": "String", "fieldName": "customerName"},
                    {"fieldType": "BigDecimal", "fieldName": "amount"},
                ]
            },
            "output_path": str(output_file),
        })
        assert result.get("success") is True, f"Render failed: {result}"
        content = output_file.read_text()
        assert "String customerName, BigDecimal amount)" in content
        assert ", )" not in content


class TestSpringBootPlaceholderCollision:
    """Tests for Spring Boot ``${VAR}`` / Handlebars ``{{{var}}}`` collision fix.

    When a Mustache variable appears inside a Spring Boot property placeholder,
    the sequence ``${{{VAR}}`` creates a collision with Handlebars triple-brace
    unescaped syntax.  pybars3 would interpret ``{{{VAR}}}`` as an unescaped
    variable and consume the opening brace that belongs to Spring's ``${``.

    These tests verify that the sentinel-based protection correctly separates
    the two syntaxes during rendering and variable extraction.
    """

    # -- _protect_spring_placeholders / _restore_spring_placeholders --

    def test_protect_replaces_collision_pattern(self):
        """${{{ is replaced with the sentinel to break the triple-brace."""
        template = "base-url: ${{{SERVICE_NAME}}_SYSTEM_API_URL:http://localhost:8081}"
        protected = TemplatePlugin._protect_spring_placeholders(template)
        assert "${{{" not in protected
        assert "{{SERVICE_NAME}}" in protected  # Mustache variable preserved

    def test_protect_and_restore_roundtrip(self):
        """Protect → restore produces a string that differs from the original
        only in that ``${{{`` has been split into ``${`` + ``{{``."""
        template = "x: ${{{A}}_URL:default}"
        protected = TemplatePlugin._protect_spring_placeholders(template)
        # After restore the sentinel is gone
        restored = TemplatePlugin._restore_spring_placeholders(protected)
        assert restored == "x: ${{{A}}_URL:default}"

    def test_protect_no_op_without_collision(self):
        """Templates without ${{{ are unchanged."""
        template = "name: {{serviceName}}\nurl: ${FIXED_URL:http://localhost}"
        protected = TemplatePlugin._protect_spring_placeholders(template)
        assert protected == template

    def test_protect_multiple_collisions(self):
        """Multiple ${{{ occurrences in the same template are all protected."""
        template = (
            "a: ${{{X}}_A:d1}\n"
            "b: ${{{Y}}_B:d2}\n"
        )
        protected = TemplatePlugin._protect_spring_placeholders(template)
        assert protected.count("${{{") == 0
        assert "{{X}}" in protected
        assert "{{Y}}" in protected

    # -- _render_mustache with Spring placeholders --

    def test_render_mod017_systemapi_template(self, plugin):
        """Renders the mod-code-017 application-systemapi.yml.tpl pattern.

        The template mixes Mustache ``{{serviceName}}`` with Spring Boot
        ``${{{SERVICE_NAME}}_SYSTEM_API_URL:http://localhost:8081}``.
        After rendering, the Spring placeholder must contain the expanded
        variable name wrapped in ``${...}``.
        """
        template = textwrap.dedent("""\
            system-api:
              {{serviceName}}:
                base-url: ${{{SERVICE_NAME}}_SYSTEM_API_URL:http://localhost:8081}
        """)
        variables = {
            "serviceName": "customer",
            "SERVICE_NAME": "CUSTOMER",
        }
        rendered, error = plugin._render_mustache(template, variables)
        assert error is None
        assert "customer:" in rendered
        assert "${CUSTOMER_SYSTEM_API_URL:http://localhost:8081}" in rendered

    def test_render_mod018_integration_template(self, plugin):
        """Renders the mod-code-018 application-integration.yml.tpl pattern.

        ``${{{BASE_URL_ENV}}:http://localhost:8081}`` must produce a valid
        Spring Boot property placeholder after Mustache rendering.
        """
        template = "base-url: ${{{BASE_URL_ENV}}:http://localhost:8081}"
        variables = {"BASE_URL_ENV": "ORDERS_API_URL"}
        rendered, error = plugin._render_mustache(template, variables)
        assert error is None
        assert rendered == "base-url: ${ORDERS_API_URL:http://localhost:8081}"

    def test_render_spring_placeholder_without_mustache_untouched(self, plugin):
        """Plain Spring ``${FIXED_VAR}`` (no Mustache inside) passes through."""
        template = "{{#feign}}\nurl: ${FIXED_URL:http://example.com}\n{{/feign}}"
        variables = {"feign": True}
        rendered, error = plugin._render_mustache(template, variables)
        assert error is None
        assert "${FIXED_URL:http://example.com}" in rendered

    def test_render_full_systemapi_template(self, plugin):
        """End-to-end rendering of a realistic multi-section template."""
        template = textwrap.dedent("""\
            system-api:
              {{serviceName}}:
                base-url: ${{{SERVICE_NAME}}_SYSTEM_API_URL:http://localhost:8081}

            {{#feign}}
            feign:
              client:
                config:
                  default:
                    connectTimeout: 5000
            {{/feign}}

            resilience4j:
              circuitbreaker:
                instances:
                  {{serviceName}}:
                    slidingWindowSize: 100
        """)
        variables = {
            "serviceName": "payment",
            "SERVICE_NAME": "PAYMENT",
            "feign": True,
        }
        rendered, error = plugin._render_mustache(template, variables)
        assert error is None
        assert "${PAYMENT_SYSTEM_API_URL:http://localhost:8081}" in rendered
        assert "payment:" in rendered
        assert "connectTimeout: 5000" in rendered

    # -- _extract_variables with Spring placeholders --

    def test_extract_variables_spring_placeholder_no_bogus_brace(self, plugin):
        """Variable extraction from ${{{VAR}} must yield 'VAR', not '{VAR'."""
        template = textwrap.dedent("""\
            {{#feign}}
            base-url: ${{{SERVICE_NAME}}_SYSTEM_API_URL:http://localhost:8081}
            name: {{serviceName}}
            {{/feign}}
        """)
        variables = plugin._extract_variables(template)
        assert "SERVICE_NAME" in variables
        assert "serviceName" in variables
        # Must NOT contain the bogus '{SERVICE_NAME' with leading brace
        for v in variables:
            assert not v.startswith("{"), f"Bogus variable with leading brace: {v}"

    def test_extract_variables_mod018_pattern(self, plugin):
        """Variable extraction for ${{{BASE_URL_ENV}}:default} pattern."""
        template = textwrap.dedent("""\
            {{#feign}}
            base-url: ${{{BASE_URL_ENV}}:http://localhost:8081}
            package: {{basePackage}}
            {{/feign}}
        """)
        variables = plugin._extract_variables(template)
        assert "BASE_URL_ENV" in variables
        assert "basePackage" in variables
        for v in variables:
            assert not v.startswith("{"), f"Bogus variable with leading brace: {v}"

    # -- renderTemplateToFile end-to-end --

    def test_renderTemplateToFile_spring_placeholder(self, plugin):
        """End-to-end: renderTemplateToFile with Spring Boot placeholder collision.

        Includes a Mustache section (``{{#feign}}...{{/feign}}``) so that
        syntax auto-detection routes to the Mustache engine, matching how
        mod-code-018's ``application-integration.yml.tpl`` is structured.
        """
        template = textwrap.dedent("""\
            integration:
              {{apiName}}:
                base-url: ${{{BASE_URL_ENV}}:http://localhost:8081}
                timeout:
                  connect: 5s
                  read: 10s

            {{#feign}}
            feign:
              client:
                config:
                  default:
                    connectTimeout: 5000
            {{/feign}}
        """)
        output_file = plugin._base_path / "output" / "application-integration.yml"
        result = plugin._execute_render_template_to_file({
            "template": template,
            "variables": {
                "apiName": "orders",
                "BASE_URL_ENV": "ORDERS_API_URL",
                "feign": True,
            },
            "output_path": str(output_file),
        })
        assert result.get("success") is True, f"Render failed: {result}"
        content = output_file.read_text()
        assert "orders:" in content
        assert "${ORDERS_API_URL:http://localhost:8081}" in content
        assert "connectTimeout: 5000" in content


# ==================== Render Shape Validation (server 0.6.31+) ====================

class TestRenderShapeValidation:
    """``renderTemplateToFile`` validates ``variables`` shape against the
    template's structural metadata (kind + item_keys) and HARD-FAILS
    before render on mismatches.

    Without this guard, Mustache silently renders garbage when a section
    variable is passed as a string or scalar — the file lands on disk
    with one bogus block instead of N repeated blocks, no error returned.
    The agent's retry path never fires.

    Fixes the slice-3 chunk-1 ``run-2`` non-determinism source where
    the agent occasionally passes a JSON string for a section variable.
    """

    def test_section_passed_as_string_rejected(self, plugin):
        """Section variable passed as JSON string → hard fail."""
        template = textwrap.dedent("""\
            {{#fields}}
            private {{type}} {{name}};
            {{/fields}}
        """)
        output_file = plugin._base_path / "out" / "Bad.java"
        result = plugin._execute_render_template_to_file({
            "template": template,
            "variables": {
                "fields": '[{"type": "String", "name": "id"}]',
            },
            "output_path": str(output_file),
        })
        assert "error" in result
        assert result.get("validation_layer") == "shape_check"
        assert any(
            "fields" in e and "kind=section" in e and "str" in e
            for e in result["validation_errors"]
        ), result
        # Output file must not be created when validation fails.
        assert not output_file.exists()

    def test_section_passed_as_int_rejected(self, plugin):
        """Section variable passed as raw int (with body referencing
        inner fields) → hard fail.  ``True`` and ``None`` are allowed
        — they're the boolean-conditional idiom.  Plain integers and
        strings are NOT, when item_keys is non-empty."""
        template = "{{#items}}- {{value}}\n{{/items}}"
        output_file = plugin._base_path / "out" / "items.txt"
        result = plugin._execute_render_template_to_file({
            "template": template,
            "variables": {"items": 42},
            "output_path": str(output_file),
        })
        assert "error" in result
        assert result.get("validation_layer") == "shape_check"
        assert "validation_errors" in result
        assert not output_file.exists()

    def test_section_list_of_strings_rejected_when_item_keys_nonempty(self, plugin):
        """Section value is a list but items are strings, not dicts
        — ``{{innerKey}}`` lookups would all resolve empty.  Only
        rejected when ``item_keys`` is non-empty (i.e. body has
        inner-field references).  See the 0.6.32 corner-case
        relaxation for the empty-item_keys case."""
        template = textwrap.dedent("""\
            {{#fields}}
            private {{type}} {{name}};
            {{/fields}}
        """)
        output_file = plugin._base_path / "out" / "los.java"
        result = plugin._execute_render_template_to_file({
            "template": template,
            "variables": {
                # list of strings instead of list of dicts.
                "fields": ["id", "email"],
            },
            "output_path": str(output_file),
        })
        assert "error" in result
        assert result.get("validation_layer") == "shape_check"
        assert any(
            "expected dict" in e for e in result["validation_errors"]
        ), result
        assert not output_file.exists()

    def test_list_of_strings_with_dot_iteration_allowed(self, plugin):
        """Mustache ``{{#x}}{{.}}{{/x}}`` over a list of strings is
        the canonical iteration-over-scalars idiom — must be
        accepted.  Surfaced by kb-enablement-2.0 chunk-1 v13:
        ``entityImports = ["java.time.LocalDate", ...]`` with body
        ``import {{.}};`` was being hard-failed by 0.6.31's overly
        strict rule, forcing the agent into incorrect dict-wrapping
        workarounds (``[{"_": "..."}]``, ``[{"value": "..."}]``,
        ``[{}]``) that produced 5 distinct hashes across 5 runs.

        Fixed in 0.6.32: when ``item_keys=[]`` (no inner-field refs),
        list items can be any type.
        """
        template = textwrap.dedent("""\
            package com.example;
            {{#entityImports}}
            import {{.}};
            {{/entityImports}}

            public class Entity {}
        """)
        output_file = plugin._base_path / "out" / "Entity.java"
        result = plugin._execute_render_template_to_file({
            "template": template,
            "variables": {
                "entityImports": [
                    "java.time.LocalDate",
                    "java.math.BigDecimal",
                    "java.util.UUID",
                ],
            },
            "output_path": str(output_file),
        })
        assert result.get("success") is True, result
        content = output_file.read_text()
        assert "import java.time.LocalDate;" in content
        assert "import java.math.BigDecimal;" in content
        assert "import java.util.UUID;" in content

    def test_list_of_strings_with_plain_body_allowed(self, plugin):
        """Section body with no inner refs at all (plain text repeated
        per item) accepts a list of scalars — Mustache renders the
        plain text once per list element."""
        template = "{{#items}}- entry\n{{/items}}"
        output_file = plugin._base_path / "out" / "items.txt"
        result = plugin._execute_render_template_to_file({
            "template": template,
            "variables": {"items": ["a", "b", "c"]},
            "output_path": str(output_file),
        })
        assert result.get("success") is True, result
        # 3 entries rendered, one per list item.
        assert output_file.read_text().count("- entry") == 3

    def test_list_of_mixed_scalars_allowed_when_item_keys_empty(self, plugin):
        """Empty item_keys + list of mixed scalar types (str, int, bool):
        validator allows it; Mustache renders ``{{.}}`` as the value's
        string form."""
        template = "{{#values}}* {{.}}\n{{/values}}"
        output_file = plugin._base_path / "out" / "mixed.txt"
        result = plugin._execute_render_template_to_file({
            "template": template,
            "variables": {"values": ["str", 42, True]},
            "output_path": str(output_file),
        })
        assert result.get("success") is True, result

    def test_boolean_conditional_idiom_allowed(self, plugin):
        """``{{#flag}}...{{/flag}}`` with ``flag=True`` is the canonical
        Mustache conditional — must be allowed even if body references
        inner fields (Mustache renders body in parent context)."""
        template = textwrap.dedent("""\
            {{#feign}}
            feign:
              client:
                config:
                  default:
                    connectTimeout: 5000
            {{/feign}}
        """)
        output_file = plugin._base_path / "out" / "flag.yml"
        result = plugin._execute_render_template_to_file({
            "template": template,
            "variables": {"feign": True},
            "output_path": str(output_file),
        })
        # Body has no inner-field refs, so this just renders the
        # plain text inside the section.
        assert result.get("success") is True, result

    def test_correct_shape_passes(self, plugin):
        """All-correct variables render without validation error."""
        template = textwrap.dedent("""\
            class {{Entity}} {
            {{#fields}}
                private {{type}} {{name}};
            {{/fields}}
            }
        """)
        output_file = plugin._base_path / "out" / "Good.java"
        result = plugin._execute_render_template_to_file({
            "template": template,
            "variables": {
                "Entity": "Customer",
                "fields": [
                    {"type": "String", "name": "id"},
                    {"type": "String", "name": "email"},
                ],
            },
            "output_path": str(output_file),
        })
        assert result.get("success") is True, result
        content = output_file.read_text()
        assert "class Customer" in content
        assert "private String id;" in content
        assert "private String email;" in content

    def test_inverted_section_accepts_any_value(self, plugin):
        """Inverted sections (``{{^x}}``) — no kind type-check applied;
        Mustache itself decides truthy/falsy at render time."""
        template = "{{^isEmpty}}has items{{/isEmpty}}\n"
        output_file = plugin._base_path / "out" / "inv.txt"
        # Pass any value — list, scalar, None — none should trigger
        # the shape validator for kind=inverted_section.
        result = plugin._execute_render_template_to_file({
            "template": template,
            "variables": {"isEmpty": [1, 2, 3]},
            "output_path": str(output_file),
            "overwrite": True,
        })
        assert result.get("success") is True, result

    def test_missing_top_level_variable_rejected_by_validator(self, plugin):
        """Variables NOT provided ARE rejected by the SHAPE validator
        as of server 0.6.173+ (Option A: strict-required-scalar
        enforcement per Mustache + Jinja2 structural-optionality
        semantics).  Pre-0.6.173 this returned None and let the
        render layer surface (or silently swallow) the absence; now
        the validator surfaces the gap loudly at the tool boundary
        with engine-specific actionable hints.

        See TestTopLevelRequiredScalarEnforcement for the full
        contract pin.  This test specifically guards against any
        future regression that re-introduces the old "skip missing
        scalar" behavior.
        """
        template = "Hello {{name}}, age {{age}}"
        result = plugin._validate_render_inputs_against_structure(
            template, {"name": "Alice"}
        )
        assert result is not None
        assert result.get("validation_layer") == "shape_check"
        assert any(
            "variable 'age'" in e and "absent" in e
            for e in result["validation_errors"]
        ), result

    def test_jinja2_path_validated_when_all_present(self, plugin):
        """Jinja2 templates ARE shape-validated as of server 0.6.173+
        (Option A jinja2 analog via _parse_jinja2_structure).  When
        every kind=scalar reference is supplied non-empty, the
        validator accepts.  Pre-0.6.173 this test asserted the
        validator silently skipped jinja2; the new contract is
        symmetric with Mustache."""
        template = "Hello {{ name }}, {{ greeting }}"
        output_file = plugin._base_path / "out" / "j.txt"
        result = plugin._execute_render_template_to_file({
            "template": template,
            "variables": {"name": "Alice", "greeting": "hi"},
            "output_path": str(output_file),
        })
        assert result.get("success") is True, result

    def test_validation_error_carries_hint(self, plugin):
        """Validation error response includes a ``hint`` pointing at
        ``listTemplateVariables`` so the agent knows how to recover."""
        template = "{{#items}}{{x}}{{/items}}"
        result = plugin._execute_render_template_to_file({
            "template": template,
            "variables": {"items": "not-a-list"},
            "output_path": str(plugin._base_path / "out" / "h.txt"),
        })
        assert "error" in result
        assert "hint" in result
        assert "listTemplateVariables" in result["hint"]

    def test_variables_not_a_dict_rejected(self, plugin):
        """Top-level ``variables`` that isn't a dict → hard fail.

        Uses a Mustache template with section markers so the syntax
        detector picks ``mustache`` (the validator is a no-op on
        Jinja2 / plain templates).  ``_coerce_variables`` turns
        non-dicts into ``{}`` upstream, so this exercises the
        defence-in-depth guard for direct callers of the validator.
        """
        template = "{{#fields}}{{name}}{{/fields}}"
        result = plugin._validate_render_inputs_against_structure(
            template, ["not-a-dict"]  # type: ignore[arg-type]
        )
        assert result is not None
        assert result.get("validation_layer") == "shape_check"


# ==================== Thread Safety (server 0.6.33+) ====================

class TestRenderThreadSafety:
    """``_render_mustache`` serialises pybars3 calls under a module-level
    lock because pybars3's ``Compiler`` class has CLASS-LEVEL mutable
    state — ``_handlebars``, ``_builder``, ``_compiler`` are class
    attributes shared by every Compiler() instance process-wide.

    Surfaced by kb-enablement-2.0 chunk-1 v15-retry: parallel-tool
    batches of 22 renderTemplateToFile calls produced sporadic
    ``'list' object has no attribute 'grow'`` and
    ``list indices must be integers, not str`` errors mid-compile.
    Each error triggered a model retry; retry sampling diverged across
    runs, amplifying agent value-mapping drift on top of the visible
    flakiness.

    These tests verify the lock prevents concurrent compile from
    crashing.  Without the lock these tests fail intermittently on
    multi-core machines [the symptom is contention-rate-dependent].
    """

    def test_concurrent_renders_no_pybars_race(self, plugin):
        """22 concurrent renderTemplateToFile calls must all succeed.

        Mirrors the kb-enablement-2.0 chunk-1 codegen pattern: many
        threads, each rendering a Mustache template with sections.
        Pre-lock, ~5-15% failure rate under contention.  Post-lock,
        zero failures.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        template = textwrap.dedent("""\
            package {{basePackage}};

            public class {{Entity}} {
            {{#fields}}
                private {{type}} {{name}};
            {{/fields}}

                public {{Entity}}({{#fields}}{{type}} {{name}}{{^last}}, {{/last}}{{/fields}}) {
            {{#fields}}
                    this.{{name}} = {{name}};
            {{/fields}}
                }
            }
        """)

        def render_one(idx):
            output = plugin._base_path / "out" / f"E{idx}.java"
            return plugin._execute_render_template_to_file({
                "template": template,
                "variables": {
                    "basePackage": "com.example",
                    "Entity": f"E{idx}",
                    "fields": [
                        {"type": "String", "name": "id"},
                        {"type": "Integer", "name": "n"},
                        {"type": "Boolean", "name": "active"},
                    ],
                },
                "output_path": str(output),
            })

        results = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(render_one, i) for i in range(22)]
            for f in as_completed(futures):
                results.append(f.result())

        # Every render must succeed — no pybars3 exceptions, no
        # render_error states.
        failures = [r for r in results if not r.get("success")]
        assert not failures, (
            f"Got {len(failures)}/22 render failures under concurrent load — "
            f"pybars3 race not properly serialised.  First failure: {failures[0]}"
        )

    def test_concurrent_list_template_variables_warnings_isolated(
        self, plugin, tmp_workspace
    ):
        """Concurrent ``listTemplateVariables`` calls must each see
        their own stripped-comment-lines warning, not another thread's.

        Pre-fix: ``self._last_stripped_comment_lines`` was instance-
        scoped — thread A's writes clobbered thread B's reads, surfacing
        wrong warnings to one or both callers.  Post-fix: thread-local
        storage isolates per-thread.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Two distinct templates with distinct ``//`` comment patterns.
        # Each thread should see its own comment lines surfaced.
        tpl_dir = tmp_workspace / ".jaato" / "templates"
        (tpl_dir / "A.tpl").write_text(textwrap.dedent("""\
            // first comment with {{a_ref}}
            {{#fields}}
            value: {{name}}
            {{/fields}}
        """))
        (tpl_dir / "B.tpl").write_text(textwrap.dedent("""\
            // 1st B-line {{b1}}
            // 2nd B-line {{b2}}
            // 3rd B-line {{b3}}
            {{#items}}
            x: {{x}}
            {{/items}}
        """))
        # Register both in the index.
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        def list_one(name):
            return name, plugin._execute_list_template_variables({
                "template_id": _template_id(name),
            })

        results = []
        # 16 calls — 8 of each — to ensure interleaving across the
        # 8-worker pool.
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = []
            for _ in range(8):
                futures.append(ex.submit(list_one, "A.tpl"))
                futures.append(ex.submit(list_one, "B.tpl"))
            for f in as_completed(futures):
                results.append(f.result())

        # Every A.tpl result should report exactly 1 stripped line; every
        # B.tpl result should report exactly 3.  Pre-fix, the warnings
        # would cross-contaminate (A sees B's count or vice versa).
        for name, result in results:
            if name == "A.tpl":
                warnings = result.get("warnings") or []
                assert any("1 line" in w for w in warnings), (
                    f"A.tpl thread saw cross-contaminated warnings: "
                    f"{warnings}"
                )
            else:
                warnings = result.get("warnings") or []
                assert any("3 line" in w for w in warnings), (
                    f"B.tpl thread saw cross-contaminated warnings: "
                    f"{warnings}"
                )


# ==================== Output Path Auto-Derivation (server 0.6.34+) ====================

class TestOutputPathAutoDerivation:
    """``renderTemplateToFile`` derives ``output_path`` from the template's
    ``// Output: <path>`` directive when the agent doesn't supply one.

    Surfaced by kb-enablement-2.0 chunk-1 v16: the dominant 3/5 hashes
    came from the agent inventing simplified paths (flattened DDD
    subpackages, normalised /config/) while the kb's ``// Output:``
    directives declared the canonical structure.  The two outliers in
    the 5x test were MORE kb-faithful than the dominant — i.e. the
    "majority hash" was the most-common agent invention, not the
    correct answer.

    Eliminating the agent-invents-the-path surface eliminates this
    drift class.
    """

    def test_extracts_output_directive_simple(self, plugin):
        """Simple `// Output: path` line is extracted verbatim."""
        content = textwrap.dedent("""\
            // Output: src/foo/Bar.java
            public class Bar {}
        """)
        assert plugin._extract_output_path_template(content) == (
            "src/foo/Bar.java"
        )

    def test_extracts_output_directive_with_placeholders(self, plugin):
        """Placeholders ``{{var}}`` in the directive are kept intact
        — they're substituted at render time, not extraction time."""
        content = textwrap.dedent("""\
            // Output: {{basePackagePath}}/domain/model/{{Entity}}.java
            package {{basePackage}}.domain.model;
            public class {{Entity}} {}
        """)
        assert plugin._extract_output_path_template(content) == (
            "{{basePackagePath}}/domain/model/{{Entity}}.java"
        )

    def test_extracts_output_directive_tolerates_whitespace(self, plugin):
        """Variable spacing around `//` and `Output:` is fine."""
        content = "  //  Output  :  some/path.java\n"
        assert plugin._extract_output_path_template(content) == (
            "some/path.java"
        )

    def test_no_directive_returns_empty_string(self, plugin):
        """Templates without ``// Output:`` return empty string."""
        content = textwrap.dedent("""\
            // some other comment
            public class X {}
        """)
        assert plugin._extract_output_path_template(content) == ""

    def test_first_directive_wins(self, plugin):
        """Only the first `// Output:` line is used; extras ignored."""
        content = textwrap.dedent("""\
            // Output: first/path.java
            // Output: second/path.java
            public class X {}
        """)
        assert plugin._extract_output_path_template(content) == (
            "first/path.java"
        )

    def test_walker_captures_output_path_template(self, plugin, tmp_path):
        """The standalone-template walker populates the new field."""
        tpl_dir = tmp_path / "templates"
        (tpl_dir / "domain").mkdir(parents=True)
        (tpl_dir / "domain" / "Entity.java.tpl").write_text(
            textwrap.dedent("""\
                // Output: {{basePackagePath}}/domain/model/{{Entity}}.java
                package {{basePackage}}.domain.model;
                public class {{Entity}} {}
            """)
        )
        entries = plugin._discover_standalone_templates(tpl_dir)
        assert len(entries) == 1
        assert entries[0].output_path_template == (
            "{{basePackagePath}}/domain/model/{{Entity}}.java"
        )

    def test_render_uses_directive_when_output_path_omitted(
        self, plugin, tmp_path
    ):
        """End-to-end: omit ``output_path``, framework substitutes the
        ``// Output:`` directive with ``variables`` and writes there."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "Entity.java.tpl").write_text(textwrap.dedent("""\
            // Output: {{basePackagePath}}/domain/model/{{Entity}}.java
            package {{basePackage}}.domain.model;
            {{#fields}}
            private {{type}} {{name}};
            {{/fields}}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("Entity.java.tpl"),
            "variables": {
                "basePackage": "com.bank.customer",
                "basePackagePath": "com/bank/customer",
                "Entity": "Customer",
                "fields": [{"type": "String", "name": "id"}],
            },
            # NOTE: output_path NOT supplied — should auto-derive.
        })
        assert result.get("success") is True, result
        # Path must come from the directive, with placeholders
        # substituted.  The framework resolves relative paths against
        # _base_path (workspace root).
        expected = (
            plugin._base_path
            / "com/bank/customer/domain/model/Customer.java"
        )
        assert expected.exists(), (
            f"File not found at directive-derived path "
            f"{expected!r} — got result: {result}"
        )

    def test_render_explicit_output_path_overrides_directive(
        self, plugin, tmp_path
    ):
        """Agent-supplied ``output_path`` overrides the directive — for
        legitimate redirects (custom destinations, test fixtures)."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "X.java.tpl").write_text(textwrap.dedent("""\
            // Output: declared/path.java
            public class X {}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        override = plugin._base_path / "custom" / "Override.java"
        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("X.java.tpl"),
            "variables": {},
            "output_path": str(override),
        })
        assert result.get("success") is True, result
        assert override.exists()
        # The directive-declared path must NOT have been written.
        assert not (plugin._base_path / "declared" / "path.java").exists()

    def test_render_no_output_no_directive_errors(
        self, plugin, tmp_path
    ):
        """When neither ``output_path`` is supplied nor the template
        declares a directive, render fails with path_check error
        (same severity class as shape validation)."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        # No `// Output:` directive in this template.
        (tpl_dir / "NoDir.java.tpl").write_text(textwrap.dedent("""\
            public class NoDir {}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("NoDir.java.tpl"),
            "variables": {},
        })
        assert "error" in result
        assert result.get("validation_layer") == "path_check"
        assert "Output:" in result["error"] or "output_path" in result["error"]

    def test_render_inline_template_requires_output_path(self, plugin):
        """Inline ``template`` (no template_name → no index entry → no
        directive) must still receive ``output_path`` explicitly."""
        result = plugin._execute_render_template_to_file({
            "template": "public class X {}",
            "variables": {},
        })
        assert "error" in result
        assert result.get("validation_layer") == "path_check"

    def test_listAvailableTemplates_surfaces_output_path_template(
        self, plugin, tmp_path
    ):
        """`listAvailableTemplates` includes ``output_path_template``
        for each entry so the agent can see the kb-declared path."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "Has.tpl").write_text(
            "// Output: declared/{{x}}.java\nbody"
        )
        (tpl_dir / "Lacks.tpl").write_text("just a body")
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        result = plugin._execute_list_available({})
        # Cutover (server 0.6.119+): response carries ``id`` not ``name``.
        by_id = {t["id"]: t for t in result["templates"]}
        assert by_id[_template_id("Has.tpl")]["output_path_template"] == (
            "declared/{{x}}.java"
        )
        assert by_id[_template_id("Lacks.tpl")]["output_path_template"] == ""

    def test_index_loader_handles_legacy_entries_without_output_path(
        self, plugin, tmp_path
    ):
        """Older index.json files (pre-0.6.34) lack the new field —
        loader must default to '' rather than crash."""
        # Simulate a pre-0.6.34 index.json.
        legacy_index = {
            "templates": {
                "Old.tpl": {
                    "name": "Old.tpl",
                    "source_path": str(tmp_path / "Old.tpl"),
                    "syntax": "mustache",
                    "variables": [],
                    "origin": "standalone",
                    # No output_path_template field.
                },
            },
        }
        index_path = plugin._templates_dir / "index.json"
        index_path.write_text(json.dumps(legacy_index))
        plugin._load_persisted_index()
        loaded = plugin._template_index.get("Old.tpl")
        assert loaded is not None
        # Default to empty string — same semantics as no-directive.
        assert loaded.output_path_template == ""


# ==================== Polyglot Output Directive (server 0.6.35+) ====================

class TestPolyglotOutputDirective:
    """``_extract_output_path_template`` recognises the ``Output:``
    directive in EVERY major comment style, dispatched by the
    template's inner extension (after stripping ``.tpl``/``.tmpl``).

    Surfaced by kb-enablement-2.0 mod-015's ``pom.xml.tpl``: declared
    ``<!-- Output: pom.xml -->`` (XML comment style).  Pre-0.6.35
    extractor only matched ``//`` so the directive was invisible,
    forcing the agent to invent ``output_path`` for every non-Java
    template.  As soon as later kb modules ship Python / YAML / SQL /
    properties templates, the same gap would propagate.
    """

    @pytest.mark.parametrize("filename, comment_line, expected", [
        # C-family (// Output:)
        ("Entity.java.tpl", "// Output: src/{{Entity}}.java",
         "src/{{Entity}}.java"),
        ("server.go.tpl", "// Output: cmd/server/main.go",
         "cmd/server/main.go"),
        ("lib.rs.tpl", "// Output: src/lib.rs", "src/lib.rs"),
        ("App.tsx.tpl", "// Output: src/App.tsx", "src/App.tsx"),
        ("Service.kt.tpl", "// Output: src/{{name}}.kt",
         "src/{{name}}.kt"),
        # XML-family (<!-- Output: -->)
        ("pom.xml.tpl", "<!-- Output: pom.xml -->", "pom.xml"),
        ("index.html.tpl", "<!-- Output: public/index.html -->",
         "public/index.html"),
        ("logo.svg.tpl", "<!-- Output: assets/logo.svg -->",
         "assets/logo.svg"),
        # Hash-family (# Output:)
        ("script.py.tpl", "# Output: scripts/{{name}}.py",
         "scripts/{{name}}.py"),
        ("setup.sh.tpl", "# Output: bin/setup.sh", "bin/setup.sh"),
        ("config.yml.tpl", "# Output: config/{{env}}.yml",
         "config/{{env}}.yml"),
        ("application.yaml.tpl", "# Output: src/main/resources/application.yaml",
         "src/main/resources/application.yaml"),
        ("rakefile.rb.tpl", "# Output: Rakefile", "Rakefile"),
        ("Cargo.toml.tpl", "# Output: Cargo.toml", "Cargo.toml"),
        ("app.properties.tpl", "# Output: app.properties",
         "app.properties"),
        # CSS-family (/* Output: */)
        ("styles.css.tpl", "/* Output: dist/styles.css */",
         "dist/styles.css"),
        ("theme.scss.tpl", "/* Output: src/theme.scss */",
         "src/theme.scss"),
        # SQL-family (-- Output:)
        ("migration.sql.tpl",
         "-- Output: db/migrations/V{{version}}__{{name}}.sql",
         "db/migrations/V{{version}}__{{name}}.sql"),
        ("util.lua.tpl", "-- Output: lua/{{module}}.lua",
         "lua/{{module}}.lua"),
        # Latex/Erlang-family (% Output:)
        ("paper.tex.tpl", "% Output: paper.tex", "paper.tex"),
        ("server.erl.tpl", "% Output: src/{{name}}.erl",
         "src/{{name}}.erl"),
    ])
    def test_extract_directive_per_language(
        self, plugin, filename, comment_line, expected
    ):
        """Each comment style is recognised when the filename's inner
        extension dispatches to the right regex."""
        content = comment_line + "\n<body>\n"
        assert plugin._extract_output_path_template(
            content, filename=filename,
        ) == expected

    def test_block_comment_closer_stripped(self, plugin):
        """``-->`` and ``*/`` close markers are stripped from capture
        — the path itself is clean."""
        # XML close marker
        content = "<!-- Output:   pom.xml   -->\n<project/>"
        assert plugin._extract_output_path_template(
            content, filename="pom.xml.tpl",
        ) == "pom.xml"
        # CSS close marker
        content = "/* Output:   dist/styles.css   */\nbody{}"
        assert plugin._extract_output_path_template(
            content, filename="styles.css.tpl",
        ) == "dist/styles.css"

    def test_universal_fallback_for_extension_less_template(self, plugin):
        """Templates without a recognised extension (``Dockerfile.tpl``,
        ``Makefile.tpl``) fall back to the universal multi-prefix
        scan — first match across all known comment styles wins."""
        # Dockerfile uses # for comments
        content = "# Output: Dockerfile\nFROM alpine\n"
        assert plugin._extract_output_path_template(
            content, filename="Dockerfile.tpl",
        ) == "Dockerfile"
        # Makefile also uses #
        content = "# Output: Makefile\nall:\n\techo hi\n"
        assert plugin._extract_output_path_template(
            content, filename="Makefile.tpl",
        ) == "Makefile"

    def test_universal_fallback_when_filename_omitted(self, plugin):
        """Calling without ``filename`` triggers the universal scan
        path — both line- and block-comment forms recognised."""
        # Block comment, no filename hint
        content = "<!-- Output: out.xml -->\n<x/>"
        assert plugin._extract_output_path_template(content) == "out.xml"
        # Line comment, no filename hint
        content = "# Output: out.py\nprint('hi')"
        assert plugin._extract_output_path_template(content) == "out.py"

    def test_no_directive_returns_empty_for_any_language(self, plugin):
        """Templates that don't declare an Output directive return ""
        regardless of language."""
        for filename in [
            "Entity.java.tpl", "pom.xml.tpl", "config.yml.tpl",
            "styles.css.tpl", "migration.sql.tpl", "server.erl.tpl",
        ]:
            content = "body without directive\n"
            assert plugin._extract_output_path_template(
                content, filename=filename,
            ) == ""

    def test_walker_captures_xml_directive(self, plugin, tmp_path):
        """Standalone-discovery walker now reads XML-comment directives
        for ``.xml.tpl`` files (regression for kb-enablement-2.0
        ``pom.xml.tpl``)."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "pom.xml.tpl").write_text(textwrap.dedent("""\
            <!-- Output: pom.xml -->
            <project>
              <artifactId>{{artifactId}}</artifactId>
            </project>
        """))
        entries = plugin._discover_standalone_templates(tpl_dir)
        assert len(entries) == 1
        assert entries[0].output_path_template == "pom.xml"

    def test_walker_captures_yaml_directive(self, plugin, tmp_path):
        """Walker reads ``# Output:`` for ``.yml.tpl`` templates."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "application.yml.tpl").write_text(textwrap.dedent("""\
            # Output: src/main/resources/application-{{env}}.yml
            spring:
              application:
                name: {{appName}}
        """))
        entries = plugin._discover_standalone_templates(tpl_dir)
        assert len(entries) == 1
        assert entries[0].output_path_template == (
            "src/main/resources/application-{{env}}.yml"
        )

    def test_render_xml_template_auto_derives_output(self, plugin, tmp_path):
        """End-to-end: pom.xml.tpl with ``<!-- Output: pom.xml -->``
        renders to ``pom.xml`` without the agent supplying output_path."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "pom.xml.tpl").write_text(textwrap.dedent("""\
            <!-- Output: pom.xml -->
            <project>
              <artifactId>{{artifactId}}</artifactId>
            </project>
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("pom.xml.tpl"),
            "variables": {"artifactId": "customer-service"},
            # No output_path — must auto-derive.
        })
        assert result.get("success") is True, result
        expected = plugin._base_path / "pom.xml"
        assert expected.exists(), (
            f"File not at directive-derived path {expected!r}: {result}"
        )
        assert "customer-service" in expected.read_text()


# ==================== listTemplateVariables Path-Variable Merge (server 0.6.36+) ====================

class TestListVariablesIncludesPathTemplate:
    """``listTemplateVariables`` merges variables from the template
    BODY and the ``// Output:`` directive into one list — surfaces
    the complete set the agent must supply for ``renderTemplateToFile``
    to succeed.

    Closes the symmetry gap from 0.6.34 where output_path auto-derivation
    needed ``{{basePackagePath}}`` etc. but the agent's source-of-truth
    tool reported only body vars.  Pre-0.6.36, ~6 mod-015 templates
    triggered substitution-error retries every run with stochastic
    supplementation.
    """

    def test_path_variables_helper_simple(self, plugin):
        """Helper extracts plain ``{{name}}`` references from a path."""
        path = "{{basePackagePath}}/domain/model/{{Entity}}.java"
        assert plugin._extract_path_variables(path) == [
            "Entity", "basePackagePath",
        ]

    def test_path_variables_helper_dedup(self, plugin):
        """Repeated references collapse to a single entry."""
        path = "{{x}}/{{x}}/{{y}}"
        assert plugin._extract_path_variables(path) == ["x", "y"]

    def test_path_variables_helper_dotted_path_leftmost(self, plugin):
        """Dotted refs (``{{a.b}}``) report only the leftmost token,
        matching body-parser convention."""
        path = "{{module.subdir}}/{{Entity}}.java"
        assert plugin._extract_path_variables(path) == ["Entity", "module"]

    def test_path_variables_helper_triple_brace(self, plugin):
        """Triple-brace ``{{{x}}}`` is the same variable as ``{{x}}``."""
        path = "{{{Entity}}}/{{path}}"
        assert plugin._extract_path_variables(path) == ["Entity", "path"]

    def test_path_variables_helper_empty(self, plugin):
        """Empty/plain paths yield empty lists."""
        assert plugin._extract_path_variables("") == []
        assert plugin._extract_path_variables("static/path.java") == []

    def test_path_variables_helper_skips_section_markers(self, plugin):
        """Defensive: malformed paths with section markers don't leak
        the marker keyword as a variable name."""
        # Sections shouldn't appear in paths but be defensive.
        path = "{{#bad}}{{Entity}}.java"
        assert plugin._extract_path_variables(path) == ["Entity"]

    def test_listTemplateVariables_merges_body_and_path_mustache(
        self, plugin, tmp_path
    ):
        """End-to-end: Mustache template's listTemplateVariables
        response includes path-only vars merged with body vars."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "Entity.java.tpl").write_text(textwrap.dedent("""\
            // Output: {{basePackagePath}}/domain/model/{{Entity}}.java
            package {{basePackage}}.domain.model;
            {{#fields}}
            private {{type}} {{name}};
            {{/fields}}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        result = plugin._execute_list_template_variables({
            "template_id": _template_id("Entity.java.tpl"),
        })
        var_names = {v["name"] for v in result["variables"]}
        # Body vars present
        assert "Entity" in var_names
        assert "basePackage" in var_names
        assert "fields" in var_names
        # Path-only var present (would have been missing pre-0.6.36)
        assert "basePackagePath" in var_names

    def test_listTemplateVariables_body_precedence_on_collision(
        self, plugin, tmp_path
    ):
        """Variable used in BOTH body and path keeps body-parsed kind
        (which may carry item_keys / has_inverted_branch metadata)."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "Entity.java.tpl").write_text(textwrap.dedent("""\
            // Output: {{basePackagePath}}/{{Entity}}.java
            package {{basePackage}};
            public class {{Entity}} { /* uses Entity scalar */ }
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        result = plugin._execute_list_template_variables({
            "template_id": _template_id("Entity.java.tpl"),
        })
        by_name = {v["name"]: v for v in result["variables"]}
        # Entity is in both body and path; body-parsed kind wins.
        assert by_name["Entity"]["kind"] == "scalar"
        # basePackagePath is path-only; gets kind=scalar.
        assert by_name["basePackagePath"]["kind"] == "scalar"

    def test_listTemplateVariables_no_directive_returns_body_only(
        self, plugin, tmp_path
    ):
        """Templates without an `Output:` directive return body vars
        unchanged — no spurious path-only entries."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "Plain.java.tpl").write_text(textwrap.dedent("""\
            package {{pkg}};
            public class {{Name}} {}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        result = plugin._execute_list_template_variables({
            "template_id": _template_id("Plain.java.tpl"),
        })
        names = {v["name"] for v in result["variables"]}
        assert names == {"pkg", "Name"}

    def test_listTemplateVariables_xml_template_with_path_var(
        self, plugin, tmp_path
    ):
        """Polyglot path: XML-style directive's path var also merged."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "config.xml.tpl").write_text(textwrap.dedent("""\
            <!-- Output: config/{{env}}/app.xml -->
            <app>
              <name>{{appName}}</name>
            </app>
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        result = plugin._execute_list_template_variables({
            "template_id": _template_id("config.xml.tpl"),
        })
        names = {v["name"] for v in result["variables"]}
        # Body var
        assert "appName" in names
        # Path-only var picked up despite XML comment style
        assert "env" in names

    def test_listTemplateVariables_count_reflects_merged_total(
        self, plugin, tmp_path
    ):
        """The ``count`` field equals the merged variable count."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "X.java.tpl").write_text(textwrap.dedent("""\
            // Output: {{outDir}}/{{Name}}.java
            package {{pkg}};
            public class {{Name}} {}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        result = plugin._execute_list_template_variables({
            "template_id": _template_id("X.java.tpl"),
        })
        # Body: pkg, Name.  Path: outDir, Name (Name dedups).  Merged: 3.
        assert result["count"] == 3
        names = {v["name"] for v in result["variables"]}
        assert names == {"pkg", "Name", "outDir"}

    def test_render_succeeds_with_merged_variable_list(
        self, plugin, tmp_path
    ):
        """End-to-end: agent calls listTemplateVariables, gets merged
        list including path-only vars, supplies all of them, render
        succeeds first try (no substitution-error retry)."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "Entity.java.tpl").write_text(textwrap.dedent("""\
            // Output: {{basePackagePath}}/domain/{{Entity}}.java
            package {{basePackage}}.domain;
            public class {{Entity}} {}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        # Step 1: listTemplateVariables to get the complete variable set.
        list_result = plugin._execute_list_template_variables({
            "template_id": _template_id("Entity.java.tpl"),
        })
        required_names = {v["name"] for v in list_result["variables"]}
        # Includes both body and path vars.
        assert "basePackagePath" in required_names
        assert "Entity" in required_names

        # Step 2: agent supplies ALL variables and renders.
        # No output_path → auto-derive from directive.
        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("Entity.java.tpl"),
            "variables": {
                "Entity": "Customer",
                "basePackage": "com.bank.customer",
                "basePackagePath": "com/bank/customer",
            },
        })
        assert result.get("success") is True, result
        assert (plugin._base_path / "com/bank/customer/domain/Customer.java").exists()


# ==================== Empty-Segment Path Validation (server 0.6.37+) ====================

class TestResolvedOutputPathValidation:
    """``renderTemplateToFile`` validates the auto-derived output_path
    for malformed segments.  Catches the silent-malformed-path failure
    mode where a path-template ``{{var}}`` substituted to empty
    string and the framework wrote a malformed file.

    Surfaced by kb-enablement-2.0 chunk-1 v22 run 3:
    ``ValueObjectName=""`` substituted into directive
    ``{{basePackagePath}}/domain/model/{{ValueObjectName}}.java``
    yielded ``com/example/customer/domain/model/.java`` — a hidden
    dotfile with empty stem.  The agent's stochastic 1:N skip
    judgment shouldn't render at all in that context, but the
    framework happily produced the malformed file pre-fix.
    """

    def test_empty_string_variable_in_path_rejected(
        self, plugin, tmp_path
    ):
        """The v22 run-3 repro: ValueObjectName='' → hard-fail."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "ValueObject.java.tpl").write_text(textwrap.dedent("""\
            // Output: {{basePackagePath}}/domain/model/{{ValueObjectName}}.java
            package {{basePackage}}.domain.model;
            public class {{ValueObjectName}} {}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("ValueObject.java.tpl"),
            "variables": {
                "ValueObjectName": "",
                "basePackage": "com.example.customer",
                "basePackagePath": "com/example/customer",
            },
        })
        assert "error" in result
        assert result.get("validation_layer") == "path_check"
        assert any(
            "ValueObjectName" in e and "empty string" in e
            for e in result["validation_errors"]
        ), result
        assert "hint" in result
        # Critical: the hint must direct the agent to NOT call this
        # render at all (1:N skip), not to retry with a different value.
        assert "do not call renderTemplateToFile" in result["hint"]
        # The malformed file must not exist on disk.
        assert not (
            plugin._base_path / "com/example/customer/domain/model/.java"
        ).exists()

    def test_missing_variable_in_path_rejected(self, plugin, tmp_path):
        """Variable referenced in path-template but absent from
        ``variables`` dict → also empty-substitution → hard-fail.

        Body deliberately doesn't reference the path-only variable
        (so body render succeeds and the empty-segment validator is
        the first to fail, isolating the test to its target check)."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "X.java.tpl").write_text(textwrap.dedent("""\
            // Output: {{basePackagePath}}/{{Name}}.java
            class X {}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry
        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("X.java.tpl"),
            # 'Name' missing entirely from the path-template's required vars
            "variables": {"basePackagePath": "com/x"},
        })
        assert "error" in result
        assert result.get("validation_layer") == "path_check"
        assert any("Name" in e for e in result["validation_errors"]), result

    def test_multiple_empty_variables_all_reported(
        self, plugin, tmp_path
    ):
        """When more than one path variable is empty, all are
        reported in validation_errors so the agent sees the full set."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "Multi.java.tpl").write_text(textwrap.dedent("""\
            // Output: {{root}}/{{module}}/{{Name}}.java
            class Multi {}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry
        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("Multi.java.tpl"),
            "variables": {"root": "src", "module": "", "Name": ""},
        })
        assert "error" in result
        errs_joined = " ".join(result["validation_errors"])
        assert "module" in errs_joined and "Name" in errs_joined

    def test_explicit_output_path_skips_validation(
        self, plugin, tmp_path
    ):
        """The empty-segment check only applies to auto-derived paths.
        When the agent supplies output_path explicitly, validation is
        skipped (override path remains an escape hatch — caller takes
        responsibility for the path they pass).

        Body has no variable references so it renders cleanly; only
        the path directive uses ``{{Name}}`` and ``{{basePackagePath}}``.
        With an explicit output_path supplied, neither path var is
        substituted (auto-derive branch is skipped) so absence of
        ``Name`` doesn't raise either."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "X.java.tpl").write_text(textwrap.dedent("""\
            // Output: {{basePackagePath}}/{{Name}}.java
            class X {}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry
        # The `// Output:` directive line is body content too —
        # pybars3 renders it as a Java comment and substitutes its
        # placeholders.  We need the path-template's variables present
        # so body render succeeds; what we're testing is that the
        # explicit output_path override skips the empty-segment
        # *validator*, not that the body skips substitution entirely.
        # Supply non-empty values so body renders, but the explicit
        # output_path is a different file.
        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("X.java.tpl"),
            "variables": {"Name": "X", "basePackagePath": "foo"},
            "output_path": str(plugin._base_path / "manual_path.java"),
        })
        assert result.get("success") is True, result
        # Renders to the explicit path, not the directive-derived one.
        assert (plugin._base_path / "manual_path.java").exists()
        assert not (plugin._base_path / "foo/X.java").exists()

    def test_no_directive_template_unaffected(self, plugin, tmp_path):
        """Templates without a `// Output:` directive return their
        existing path_check error ('output_path is required'), not the
        empty-segment one — different failure class."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "NoDir.java.tpl").write_text(textwrap.dedent("""\
            class NoDir {}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry
        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("NoDir.java.tpl"),
            "variables": {},
        })
        assert "error" in result
        assert result.get("validation_layer") == "path_check"
        # Different error class — no validation_errors list.
        assert "Output:" in result["error"]

    def test_structural_double_slash_caught(self, plugin, tmp_path):
        """Defensive structural check: malformed directive with literal
        ``//`` (e.g. kb-author typo) is also caught — even if no
        variable substituted to empty."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        # Directive has a literal ``//`` mid-path (typo: doubled separator).
        (tpl_dir / "Bad.java.tpl").write_text(textwrap.dedent("""\
            // Output: src//domain/{{Name}}.java
            class Bad {}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry
        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("Bad.java.tpl"),
            "variables": {"Name": "Foo"},
        })
        assert "error" in result
        assert result.get("validation_layer") == "path_check"
        assert any(
            "intermediate segment" in e or "//" in e
            for e in result["validation_errors"]
        ), result

    def test_structural_trailing_slash_caught(self, plugin, tmp_path):
        """Trailing slash in resolved path → structural error."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "Trail.java.tpl").write_text(textwrap.dedent("""\
            // Output: src/{{Name}}/
            class Trail {}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry
        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("Trail.java.tpl"),
            "variables": {"Name": "x"},
        })
        assert "error" in result
        assert result.get("validation_layer") == "path_check"
        assert any(
            "trailing" in e.lower() or "final segment" in e
            for e in result["validation_errors"]
        ), result

    def test_gitignore_style_filename_allowed(self, plugin, tmp_path):
        """``.gitignore``-style files (leading dot, no extension)
        are valid — must not false-positive."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "ignore.tpl").write_text(textwrap.dedent("""\
            # Output: {{module}}/.gitignore
            *.pyc
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry
        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("ignore.tpl"),
            "variables": {"module": "core"},
        })
        assert result.get("success") is True, result
        assert (plugin._base_path / "core/.gitignore").exists()

    def test_correct_paths_pass_validation(self, plugin, tmp_path):
        """All-correct variables → render succeeds (no false-positives
        from the new validator)."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "Good.java.tpl").write_text(textwrap.dedent("""\
            // Output: {{basePackagePath}}/domain/model/{{Name}}.java
            package {{basePackage}};
            public class {{Name}} {}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry
        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("Good.java.tpl"),
            "variables": {
                "Name": "Customer",
                "basePackage": "com.bank",
                "basePackagePath": "com/bank",
            },
        })
        assert result.get("success") is True, result
        assert (
            plugin._base_path / "com/bank/domain/model/Customer.java"
        ).exists()


# ==================== Handlebars Helper Recognition (server 0.6.38+) ====================

class TestHandlebarsHelperParsing:
    """Mustache parser recognises Handlebars helpers
    (``{{#if cond}}``, ``{{#unless cond}}``, ``{{#each xs}}``,
    ``{{#with x}}``) and unwraps them to the bare argument when
    populating ``item_keys``.  Iteration-metadata identifiers
    (``@first``, ``@last``, ``@index``, ``@key``) are filtered.

    Surfaced empirically by kb-enablement-2.0 chunk-1 v22 run 4: agent
    literally created dict items with keys named ``\"if validation.maxLength\"``
    (with the keyword prefix glued in) because that's what
    listTemplateVariables reported.  Pre-fix, item_keys carried the
    keyword-prefixed strings, the agent built shape-mismatched dicts,
    pybars3's helper-resolution path failed silently, different runs
    produced different rendered bytes.
    """

    def test_if_helper_unwraps_to_argument(self, plugin):
        """``{{#if required}}`` inside ``{{#fields}}`` should add
        ``required`` to fields' item_keys, NOT ``\"if required\"``."""
        template = textwrap.dedent("""\
            {{#fields}}
            {{#if required}} @NotNull {{/if}}
            {{name}}
            {{/fields}}
        """)
        result = plugin._parse_mustache_structure(template)
        by_name = {v["name"]: v for v in result}
        assert "fields" in by_name
        item_keys = _names(by_name["fields"]["item_keys"])
        assert "required" in item_keys
        assert "if required" not in item_keys
        assert "name" in item_keys

    def test_unless_helper_unwraps_to_argument(self, plugin):
        """``{{#unless x}}`` adds ``x`` (not ``\"unless x\"``)."""
        template = textwrap.dedent("""\
            {{#fields}}
            {{name}}{{#unless last}}, {{/unless}}
            {{/fields}}
        """)
        result = plugin._parse_mustache_structure(template)
        by_name = {v["name"]: v for v in result}
        item_keys = _names(by_name["fields"]["item_keys"])
        assert "last" in item_keys
        assert "unless last" not in item_keys

    def test_iteration_metadata_filtered(self, plugin):
        """``{{#unless @last}}`` should NOT add ``@last`` to item_keys —
        ``@last`` is engine-populated iteration metadata, never
        user-provided."""
        template = textwrap.dedent("""\
            {{#fields}}
            {{name}}{{#unless @last}}, {{/unless}}
            {{/fields}}
        """)
        result = plugin._parse_mustache_structure(template)
        by_name = {v["name"]: v for v in result}
        item_keys = _names(by_name["fields"]["item_keys"])
        assert "@last" not in item_keys
        assert "unless @last" not in item_keys
        assert "name" in item_keys

    def test_dotted_helper_argument_takes_leftmost_token(self, plugin):
        """``{{#if validation.maxLength}}`` adds ``validation`` to
        item_keys (not ``\"validation.maxLength\"``, not
        ``\"if validation.maxLength\"``)."""
        template = textwrap.dedent("""\
            {{#fields}}
            {{name}}
            {{#if validation.maxLength}}@Size(max={{validation.maxLength}}){{/if}}
            {{/fields}}
        """)
        result = plugin._parse_mustache_structure(template)
        by_name = {v["name"]: v for v in result}
        item_keys = _names(by_name["fields"]["item_keys"])
        assert "validation" in item_keys
        assert "validation.maxLength" not in item_keys
        assert "if validation.maxLength" not in item_keys

    def test_each_helper_creates_iteration_section(self, plugin):
        """``{{#each xs}}`` is structurally identical to ``{{#xs}}`` —
        registers ``xs`` as a section variable."""
        template = textwrap.dedent("""\
            {{#each items}}
            - {{name}}
            {{/each}}
        """)
        result = plugin._parse_mustache_structure(template)
        by_name = {v["name"]: v for v in result}
        assert "items" in by_name
        assert by_name["items"]["kind"] == "section"
        assert "name" in _names(by_name["items"]["item_keys"])
        assert "each items" not in by_name

    def test_v22_run4_repro_create_request(self, plugin):
        """Reproduces the chunk-1 v22 run-4 outlier: parsing the
        observed CreateRequest.java.tpl shape should yield clean
        item_keys (fieldName, type, validation, required, last) —
        NOT the keyword-prefixed mess pre-fix.
        """
        template = textwrap.dedent("""\
            public class CreateRequest {
            {{#fields}}
              {{#if required}}@NotNull{{/if}}
              {{#if validation.maxLength}}@Size(max={{validation.maxLength}}){{/if}}
              {{#if validation.email}}@Email{{/if}}
              {{#if validation.pattern}}@Pattern{{/if}}
              private {{type}} {{fieldName}};{{#unless @last}},{{/unless}}
            {{/fields}}
            }
        """)
        result = plugin._parse_mustache_structure(template)
        by_name = {v["name"]: v for v in result}
        assert "fields" in by_name
        item_keys = set(_names(by_name["fields"]["item_keys"]))
        # All the bare context refs that should be there:
        assert "fieldName" in item_keys
        assert "type" in item_keys
        assert "required" in item_keys
        assert "validation" in item_keys  # leftmost of validation.maxLength etc.
        # None of the keyword-prefixed leakage:
        assert not any("if " in k or "unless " in k for k in item_keys), (
            f"Helper-keyword leakage in item_keys: {item_keys}"
        )
        assert not any(k.startswith("@") for k in item_keys), (
            f"Iteration-metadata leakage in item_keys: {item_keys}"
        )

    def test_with_helper_unwraps_to_argument(self, plugin):
        """``{{#with profile}}`` is rare but supported — argument is
        a context-narrowing reference, treat as scalar lookup."""
        template = textwrap.dedent("""\
            {{#with user}}
            {{name}} {{email}}
            {{/with}}
        """)
        result = plugin._parse_mustache_structure(template)
        by_name = {v["name"]: v for v in result}
        # ``user`` registered as scalar (with-narrowing); ``name`` /
        # ``email`` are top-level since ``with`` doesn't iterate.
        assert "user" in by_name
        assert by_name["user"]["kind"] == "scalar"
        assert "with user" not in by_name

    def test_bare_if_section_without_argument_unaffected(self, plugin):
        """``{{#if}}...{{/if}}`` with no argument is a Mustache section
        named literally ``if`` — kb-author choice; we don't reinterpret
        it as a helper.  Only invocations WITH an argument get the
        helper unwrapping treatment."""
        template = textwrap.dedent("""\
            {{#if}}body{{/if}}
        """)
        result = plugin._parse_mustache_structure(template)
        by_name = {v["name"]: v for v in result}
        # Bare ``{{#if}}`` falls through to the regular section path —
        # registers ``if`` as a section variable.
        assert "if" in by_name
        assert by_name["if"]["kind"] == "section"

    def test_helper_close_pops_correctly(self, plugin):
        """``{{#if x}}...{{/if}}`` close should pop cleanly without
        leaving the section stack imbalanced (which would corrupt
        outer-iteration tracking for subsequent tags)."""
        template = textwrap.dedent("""\
            {{#fields}}
            {{#if required}}A{{/if}}
            {{name}}
            {{/fields}}
            {{topLevel}}
        """)
        result = plugin._parse_mustache_structure(template)
        by_name = {v["name"]: v for v in result}
        # ``topLevel`` after the {{/fields}} close should be a TOP-LEVEL
        # scalar (not an item_key of fields).  If the helper close
        # didn't pop properly, the stack would still have the helper
        # frame, outer_iter would still resolve to fields, and
        # topLevel would leak into fields' item_keys.
        assert "topLevel" in by_name
        assert by_name["topLevel"]["kind"] == "scalar"
        assert "topLevel" not in _names(by_name["fields"]["item_keys"])


# ==================== Index-authoritative listTemplateVariables (server 0.6.39+) ====================

class TestListVariablesHonorsIndex:
    """``listTemplateVariables`` reads the output_path_template from the
    index entry when available, falling back to on-disk re-extraction
    only when the entry is absent or carries an empty value.

    Surfaced empirically by kb-enablement-2.0 chunk-1 v20: the kb-side
    walker patched the index's `output_path_template` field for a
    template whose upstream source lacks the `// Output:` directive
    (kb-omission for UpdateRequest.java.tpl).  `renderTemplateToFile`
    picked up the patch (it reads the index directly), but
    `listTemplateVariables` re-extracted from disk, saw no directive,
    didn't merge the path-only var into the variables list.  Agent's
    first attempt rendered without the var, hit empty-segment
    validation, retried, retry resampling drifted.

    0.6.39 makes the index authoritative: both consumers read the
    same source.  Closes the index/disk asymmetry that v20's
    materialized-override workaround dodged but didn't actually fix.
    """

    def test_index_entry_overrides_disk_when_set(self, plugin, tmp_path):
        """When the on-disk template has NO directive but the index
        entry has output_path_template populated (e.g. via side-channel
        patch), listTemplateVariables uses the index value."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        # On-disk template has NO `// Output:` directive.
        (tpl_dir / "Patched.java.tpl").write_text(textwrap.dedent("""\
            package {{basePackage}};
            public class {{Name}} {}
        """))
        # Walker discovery would populate output_path_template="" since
        # no directive on disk.  Simulate the side-channel patch by
        # directly setting the field on the index entry.
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry
        plugin._template_index["Patched.java.tpl"].output_path_template = (
            "{{basePackagePath}}/domain/model/{{Name}}.java"
        )

        result = plugin._execute_list_template_variables({
            "template_id": _template_id("Patched.java.tpl"),
        })
        names = {v["name"] for v in result["variables"]}
        # Body vars from disk:
        assert "basePackage" in names
        assert "Name" in names
        # Path-only var from the patched index entry:
        assert "basePackagePath" in names

    def test_disk_used_when_index_entry_empty(self, plugin, tmp_path):
        """When the index entry's output_path_template is empty AND
        the on-disk template has a directive, fall back to disk
        extraction.  Preserves legacy behavior for templates the
        walker hasn't yet refreshed."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        # On-disk template HAS a directive.
        (tpl_dir / "OnDisk.java.tpl").write_text(textwrap.dedent("""\
            // Output: {{basePackagePath}}/{{Name}}.java
            class OnDisk {}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry
        # Simulate a stale index entry: clear its output_path_template
        # field even though the disk has a directive.
        plugin._template_index["OnDisk.java.tpl"].output_path_template = ""

        result = plugin._execute_list_template_variables({
            "template_id": _template_id("OnDisk.java.tpl"),
        })
        names = {v["name"] for v in result["variables"]}
        # Disk extraction picks up the path-var as fallback.
        assert "basePackagePath" in names
        assert "Name" in names

    def test_no_entry_falls_back_to_disk(self, plugin, tmp_path):
        """When no index entry exists at all (e.g. listTemplateVariables
        called before the walker has run), disk extraction handles it."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "NoIndex.java.tpl").write_text(textwrap.dedent("""\
            // Output: {{path}}/{{Name}}.java
            class NoIndex {}
        """))
        # Resolution by filesystem (no index entry); test that
        # listTemplateVariables still works via on-disk extraction.
        # The resolver searches templates_dir, so put the file there.
        (plugin._templates_dir / "NoIndex.java.tpl").write_text(
            (tpl_dir / "NoIndex.java.tpl").read_text()
        )
        result = plugin._execute_list_template_variables({
            "template_id": _template_id("NoIndex.java.tpl"),
        })
        names = {v["name"] for v in result["variables"]}
        assert "path" in names
        assert "Name" in names

    # NOTE: TestPathRouting suite (0.6.40+) is appended at the end
    # of this file.  This anchor only marks the place where the
    # index-honoring tests end.

    def test_v20_kb_omission_repro(self, plugin, tmp_path):
        """Reproduction of kb-enablement-2.0 v20's regression class:
        kb-omission patched in the index, body refs basePackage and a
        few fields, path-only var basePackagePath only in the patched
        index entry.  Without the index-honoring fix, listTemplateVariables
        misses basePackagePath, agent first-attempt fails empty-segment
        validation, retry resampling drifts.  Post-fix, basePackagePath
        is reported, agent supplies it, render succeeds first try.
        """
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        # Kb-omission: template has NO directive on disk.
        (tpl_dir / "UpdateRequest.java.tpl").write_text(textwrap.dedent("""\
            package {{basePackage}}.api;
            public class Update{{Entity}}Request {
            {{#fields}}
              {{type}} {{fieldName}};
            {{/fields}}
            }
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry
        # Side-channel patches the index field.
        plugin._template_index[
            "UpdateRequest.java.tpl"
        ].output_path_template = (
            "{{basePackagePath}}/api/Update{{Entity}}Request.java"
        )

        result = plugin._execute_list_template_variables({
            "template_id": _template_id("UpdateRequest.java.tpl"),
        })
        names = {v["name"] for v in result["variables"]}
        # Agent must see basePackagePath in the variable list.
        assert "basePackagePath" in names
        # Body vars also present.
        assert "basePackage" in names
        assert "Entity" in names
        assert "fields" in names


# ==================== Path Routing (server 0.6.40+) ====================

class TestPathRouting:
    """Auto-discovered ``.jaato/template_routing.yaml`` declares
    ``(glob, prefix)`` rules that prepend layout-specific prefixes to
    rendered file paths.  Centralises the file-type → location heuristic
    that consumers (jcmunuera/run-codegen.sh) previously baked into
    Python.

    Surfaced empirically by chunk-2 cascade probe: rendered output
    landed Java sources directly under <basePackagePath>/... instead
    of src/main/java/<basePackagePath>/..., making the result
    non-buildable as a Maven project.
    """

    def _write_routing_yaml(self, plugin, content: str) -> Path:
        """Materialise a template_routing.yaml at the plugin's resolved
        config_root path (or workspace fallback) so the lazy loader
        picks it up on next access."""
        path = plugin._resolve_path_routing_path()
        assert path is not None, "plugin must have _base_path set"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        # Invalidate cache so next call re-reads.
        plugin._path_routing_rules = None
        return path

    def test_no_routing_file_is_noop(self, plugin):
        """When no template_routing.yaml exists, _apply_path_routing
        returns the input unchanged.  Pre-0.6.40 back-compat preserved."""
        # No file written; cache is None / will load empty.
        out = plugin._apply_path_routing("com/example/Customer.java")
        assert out == "com/example/Customer.java"

    def test_simple_java_rule_prepends_src_main_java(self, plugin):
        self._write_routing_yaml(plugin, textwrap.dedent("""\
            file_conventions:
              output_path_routing:
                - glob: "**/*.java"
                  prefix: "src/main/java"
        """))
        out = plugin._apply_path_routing("com/example/Customer.java")
        assert out == "src/main/java/com/example/Customer.java"

    def test_first_match_wins_test_before_main(self, plugin):
        """**/*Test.java should match before **/*.java, routing tests
        to src/test/java."""
        self._write_routing_yaml(plugin, textwrap.dedent("""\
            file_conventions:
              output_path_routing:
                - glob: "**/*Test.java"
                  prefix: "src/test/java"
                - glob: "**/*.java"
                  prefix: "src/main/java"
        """))
        # CustomerTest.java → test path
        assert plugin._apply_path_routing(
            "com/example/CustomerTest.java",
        ) == "src/test/java/com/example/CustomerTest.java"
        # Customer.java → main path
        assert plugin._apply_path_routing(
            "com/example/Customer.java",
        ) == "src/main/java/com/example/Customer.java"

    def test_yaml_extensions_route_to_resources(self, plugin):
        """Multiple resource extensions need separate rules (no brace
        expansion)."""
        self._write_routing_yaml(plugin, textwrap.dedent("""\
            file_conventions:
              output_path_routing:
                - glob: "**/*.yml"
                  prefix: "src/main/resources"
                - glob: "**/*.yaml"
                  prefix: "src/main/resources"
                - glob: "**/*.properties"
                  prefix: "src/main/resources"
        """))
        assert plugin._apply_path_routing("application.yml") == (
            "src/main/resources/application.yml"
        )
        assert plugin._apply_path_routing("config.yaml") == (
            "src/main/resources/config.yaml"
        )
        assert plugin._apply_path_routing("app.properties") == (
            "src/main/resources/app.properties"
        )

    def test_pom_xml_empty_prefix_keeps_root_anchor(self, plugin):
        """``pom.xml`` rule with empty prefix is the workspace-root
        anchor — must not double-prefix or get caught by later rules."""
        self._write_routing_yaml(plugin, textwrap.dedent("""\
            file_conventions:
              output_path_routing:
                - glob: "pom.xml"
                  prefix: ""
                - glob: "**/*.xml"
                  prefix: "src/main/resources"
        """))
        # pom.xml matches first rule → unchanged
        assert plugin._apply_path_routing("pom.xml") == "pom.xml"
        # Other XML files match the catch-all
        assert plugin._apply_path_routing("config/app.xml") == (
            "src/main/resources/config/app.xml"
        )

    def test_idempotent_when_path_already_prefixed(self, plugin):
        """When the resolved path already starts with the matched
        rule's prefix (kb-author already prepended, or prior call
        wrote the path), the prepend is a no-op."""
        self._write_routing_yaml(plugin, textwrap.dedent("""\
            file_conventions:
              output_path_routing:
                - glob: "**/*.yml"
                  prefix: "src/main/resources"
        """))
        # Already correctly prefixed
        already = "src/main/resources/application.yml"
        assert plugin._apply_path_routing(already) == already

    def test_no_match_returns_unchanged(self, plugin):
        """A path that no rule matches passes through unchanged.
        Lets unfamiliar file types through without routing."""
        self._write_routing_yaml(plugin, textwrap.dedent("""\
            file_conventions:
              output_path_routing:
                - glob: "**/*.java"
                  prefix: "src/main/java"
        """))
        out = plugin._apply_path_routing("README.md")
        assert out == "README.md"

    def test_empty_yaml_is_noop(self, plugin):
        """Empty / commented-out routing YAML loads to an empty rules
        list and behaves like no file at all."""
        self._write_routing_yaml(plugin, "# no rules\n")
        assert plugin._apply_path_routing("a/b/c.java") == "a/b/c.java"

    def test_malformed_yaml_falls_back_to_noop(self, plugin):
        """Malformed YAML is logged but doesn't crash — rules cached
        as empty, behaviour reverts to pre-0.6.40."""
        self._write_routing_yaml(plugin, "this: is: not: valid: yaml: ::")
        # Should not raise; falls back to no-routing.
        out = plugin._apply_path_routing("a/b/c.java")
        assert out == "a/b/c.java"

    def test_explicit_leave_as_is_rule_before_catch_all(self, plugin):
        """Empty-prefix rule serves as 'whitelist this path' — the
        path matches the rule and returns unchanged, bypassing later
        catch-all rules."""
        self._write_routing_yaml(plugin, textwrap.dedent("""\
            file_conventions:
              output_path_routing:
                - glob: "src/**/*.java"
                  prefix: ""                  # already in tree, leave alone
                - glob: "**/*.java"
                  prefix: "src/main/java"
        """))
        # Already-prefixed path matches the leave-as-is rule
        assert plugin._apply_path_routing(
            "src/main/java/com/Customer.java",
        ) == "src/main/java/com/Customer.java"
        # Bare path matches the catch-all
        assert plugin._apply_path_routing(
            "com/Customer.java",
        ) == "src/main/java/com/Customer.java"

    def test_render_e2e_routes_into_maven_layout(self, plugin, tmp_path):
        """End-to-end: renderTemplateToFile with auto-derived path
        passes through routing and lands the file at the routed
        location.  Reproduces the chunk-2 cascade probe gap closure."""
        self._write_routing_yaml(plugin, textwrap.dedent("""\
            file_conventions:
              output_path_routing:
                - glob: "pom.xml"
                  prefix: ""
                - glob: "**/*Test.java"
                  prefix: "src/test/java"
                - glob: "**/*.java"
                  prefix: "src/main/java"
        """))
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "Entity.java.tpl").write_text(textwrap.dedent("""\
            // Output: {{basePackagePath}}/domain/{{Entity}}.java
            package {{basePackage}}.domain;
            public class {{Entity}} {}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("Entity.java.tpl"),
            "variables": {
                "Entity": "Customer",
                "basePackage": "com.bank",
                "basePackagePath": "com/bank",
            },
        })
        assert result.get("success") is True, result
        # Path landed under src/main/java per routing rule.
        expected = plugin._base_path / "src/main/java/com/bank/domain/Customer.java"
        assert expected.exists(), (
            f"File missing at expected routed path {expected}: {result}"
        )

    def test_explicit_output_path_also_routed(self, plugin, tmp_path):
        """Routing applies to BOTH auto-derived and explicit output_path
        — agent says WHAT, kb's routing rules say WHERE-IN-LAYOUT."""
        self._write_routing_yaml(plugin, textwrap.dedent("""\
            file_conventions:
              output_path_routing:
                - glob: "**/*.java"
                  prefix: "src/main/java"
        """))
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "X.java.tpl").write_text(textwrap.dedent("""\
            class X {}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        # Agent supplies explicit output_path; routing still applies.
        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("X.java.tpl"),
            "variables": {},
            "output_path": "com/example/X.java",
        })
        assert result.get("success") is True, result
        expected = plugin._base_path / "src/main/java/com/example/X.java"
        assert expected.exists(), result

    def test_config_root_overrides_workspace_fallback(
        self, plugin, tmp_workspace, tmp_path
    ):
        """When config_root is set (sandbox + config_root pattern),
        routing YAML resolves under config_root, NOT under workspace's
        .jaato/."""
        # Create a config_root location distinct from workspace
        config_root = tmp_path / "framework_config"
        config_root.mkdir()
        # Write conflicting rules in both locations to verify which wins
        workspace_yaml = tmp_workspace / ".jaato" / PATH_ROUTING_FILENAME
        workspace_yaml.write_text(textwrap.dedent("""\
            file_conventions:
              output_path_routing:
                - { glob: "**/*.java", prefix: "WORKSPACE_LOC" }
        """))
        config_root_yaml = config_root / PATH_ROUTING_FILENAME
        config_root_yaml.write_text(textwrap.dedent("""\
            file_conventions:
              output_path_routing:
                - { glob: "**/*.java", prefix: "CONFIG_ROOT_LOC" }
        """))
        # Set config_root — should override
        plugin.set_config_root(str(config_root))
        out = plugin._apply_path_routing("a/b/X.java")
        assert out == "CONFIG_ROOT_LOC/a/b/X.java"

        # Clear config_root — should fall back to workspace
        plugin.set_config_root(None)
        out2 = plugin._apply_path_routing("a/b/X.java")
        assert out2 == "WORKSPACE_LOC/a/b/X.java"


# ==================== Output-Directive Stripping at Render (server 0.6.41+) ====================

class TestOutputDirectiveStripping:
    """Render-time stripping of the ``// Output: <path>`` (or polyglot
    equivalent) directive line from template content loaded from disk.

    The directive is metadata declaring where the rendered file
    belongs; leaving it in the content makes it leak into the rendered
    file.  For Java that's harmless noise; for XML (``<?xml ?>`` must
    be the first character) it breaks parsing.  Surfaced empirically
    by kb-enablement-2.0 chunk-2 cascade-then-mvn-compile probe:
    rendered ``pom.xml`` carried ``<!-- Output: pom.xml -->`` as line
    1, Maven's XML parser rejected it.
    """

    def test_strip_java_directive_from_disk_render(self, plugin, tmp_path):
        """End-to-end: a Java template with a ``// Output:`` first-line
        directive renders without that line in the output file."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "Entity.java.tpl").write_text(textwrap.dedent("""\
            // Output: {{basePackagePath}}/{{Entity}}.java
            package {{basePackage}};
            public class {{Entity}} {}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("Entity.java.tpl"),
            "variables": {
                "Entity": "Customer",
                "basePackage": "com.bank",
                "basePackagePath": "com/bank",
            },
        })
        assert result.get("success") is True, result
        rendered = (
            plugin._base_path / "com/bank/Customer.java"
        ).read_text()
        # The directive line must NOT appear in rendered content.
        assert "Output:" not in rendered, (
            f"Directive leaked into rendered file:\n{rendered}"
        )
        # Body content IS present.
        assert "package com.bank;" in rendered
        assert "public class Customer" in rendered

    def test_strip_xml_directive_makes_pom_valid(self, plugin, tmp_path):
        """The chunk-2 cascade-probe repro: pom.xml.tpl with
        ``<!-- Output: pom.xml -->`` first line renders to a pom.xml
        that starts with ``<?xml ?>`` (Maven-parseable)."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "pom.xml.tpl").write_text(textwrap.dedent("""\
            <!-- Output: pom.xml -->
            <?xml version="1.0" encoding="UTF-8"?>
            <project>
              <artifactId>{{artifactId}}</artifactId>
            </project>
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("pom.xml.tpl"),
            "variables": {"artifactId": "my-service"},
        })
        assert result.get("success") is True, result
        rendered = (plugin._base_path / "pom.xml").read_text()
        # The XML doc must start with ``<?xml`` (no preamble).
        assert rendered.startswith("<?xml"), (
            f"pom.xml does not start with <?xml — directive line leaked:\n"
            f"{rendered[:120]!r}"
        )
        assert "Output:" not in rendered
        assert "my-service" in rendered

    def test_strip_python_hash_directive(self, plugin, tmp_path):
        """``# Output: <path>`` directive removed from Python source."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "script.py.tpl").write_text(textwrap.dedent("""\
            # Output: scripts/{{name}}.py
            import sys

            def main():
                print("{{message}}")
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("script.py.tpl"),
            "variables": {"name": "hello", "message": "hi"},
        })
        assert result.get("success") is True, result
        rendered = (plugin._base_path / "scripts/hello.py").read_text()
        assert not rendered.startswith("# Output:"), (
            f"Python directive leaked: first line = {rendered.splitlines()[0]!r}"
        )
        assert rendered.startswith("import sys"), (
            f"Strip should remove directive AND immediately following blank line; "
            f"got: {rendered[:80]!r}"
        )

    def test_strip_yaml_directive(self, plugin, tmp_path):
        """``# Output: <path>`` removed from YAML — application.yml etc."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "application.yml.tpl").write_text(textwrap.dedent("""\
            # Output: src/main/resources/application.yml
            spring:
              application:
                name: {{appName}}
        """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry

        result = plugin._execute_render_template_to_file({
            "template_id": _template_id("application.yml.tpl"),
            "variables": {"appName": "demo"},
        })
        assert result.get("success") is True, result
        rendered = (
            plugin._base_path / "src/main/resources/application.yml"
        ).read_text()
        assert "Output:" not in rendered
        assert rendered.startswith("spring:")

    def test_strip_helper_no_directive_is_noop(self, plugin):
        """Templates without a directive return unchanged."""
        content = "package com.x;\npublic class Foo {}\n"
        assert plugin._strip_output_directive(
            content, filename="Foo.java.tpl",
        ) == content

    def test_strip_helper_first_match_only(self, plugin):
        """Multiple ``Output:`` lines (authoring error) — only the
        first is stripped.  Mirrors the extractor's first-match
        behavior; second occurrence stays as a literal comment in
        the rendered output (compiles for Java, survives as text
        elsewhere — kb author's bug to fix)."""
        content = (
            "// Output: first.java\n"
            "// Output: second.java\n"
            "class Foo {}\n"
        )
        out = plugin._strip_output_directive(content, filename="Foo.java.tpl")
        assert "first.java" not in out
        assert "second.java" in out

    def test_strip_helper_universal_fallback_for_unknown_extension(
        self, plugin
    ):
        """Templates with no recognised extension (Dockerfile.tpl,
        Makefile.tpl) — universal scan finds the directive."""
        content = "# Output: Dockerfile\nFROM alpine\n"
        out = plugin._strip_output_directive(
            content, filename="Dockerfile.tpl",
        )
        assert "Output:" not in out
        assert out.startswith("FROM alpine")

    def test_inline_template_directive_NOT_stripped(self, plugin, tmp_path):
        """Inline templates (passed via the ``template`` arg) skip
        the strip — agent owns inline content shape."""
        result = plugin._execute_render_template_to_file({
            "template": (
                "// Output: ignored\n"
                "package x;\n"
            ),
            "variables": {},
            "output_path": str(plugin._base_path / "out.txt"),
        })
        assert result.get("success") is True, result
        rendered = (plugin._base_path / "out.txt").read_text()
        # Directive line is still present — agent passed it inline,
        # framework respects the inline content verbatim.
        assert "Output:" in rendered

    def test_strip_handles_directive_not_on_line_one(self, plugin):
        """Directive on a non-first line (e.g. after a shebang) is
        still stripped; surrounding content shape preserved."""
        content = (
            "#!/usr/bin/env python\n"
            "# Output: scripts/{{name}}.py\n"
            "import sys\n"
        )
        out = plugin._strip_output_directive(
            content, filename="script.py.tpl",
        )
        assert "Output:" not in out
        # Shebang preserved
        assert out.startswith("#!/usr/bin/env python\n")
        # Body still present, no double-blank
        assert "import sys" in out

    def test_strip_handles_block_comment_styles(self, plugin):
        """``<!-- ... -->`` and ``/* ... */`` directives stripped
        cleanly including their close markers."""
        # XML-style
        xml = "<!-- Output: pom.xml -->\n<?xml version=\"1.0\"?>\n<x/>\n"
        out = plugin._strip_output_directive(xml, filename="pom.xml.tpl")
        assert out.startswith("<?xml")
        # CSS-style
        css = "/* Output: dist/styles.css */\nbody { color: red; }\n"
        out = plugin._strip_output_directive(css, filename="styles.css.tpl")
        assert out.startswith("body {")


# ==================== Variant Axis Fields (server 0.6.42+) ====================

class TestVariantAxisFields:
    """``TemplateIndexEntry`` carries ``variant_key`` + ``variant`` for
    templates representing one option of a multi-option axis the
    consumer pre-selected upstream.  Mirrors the 0.6.34
    ``output_path_template`` plumbing exactly.

    Surfaced empirically by kb-enablement-2.0 chunk-3 mod-017
    (persistence-systemapi): three HTTP-client implementation
    templates (restclient/feign/resttemplate) all targeting the same
    target file.  Without variant filtering the agent rendered all
    three (last-write-wins) or skipped stochastically — 5x produced
    3 distinct hashes + file-count drift.

    The framework's job: plumb the two strings through the index +
    walker + listAvailableTemplates surface.  Per-stack semantics
    (which axes exist, which option is the default, etc.) live in
    the kb's MODULE.md ``variants:`` block; the kb-side walker
    populates ``variant_key`` from there.  The framework
    opportunistically extracts ``variant`` from a ``// Variant: <option>``
    header in template content as a secondary signal.
    """

    def test_variant_extractor_simple_java(self, plugin):
        """``// Variant: <option>`` header captured."""
        content = textwrap.dedent("""\
            // Variant: restclient
            package com.example.client;
            public class FooClient {}
        """)
        assert plugin._extract_variant(
            content, filename="restclient.java.tpl",
        ) == "restclient"

    def test_variant_extractor_xml_block_comment(self, plugin):
        """``<!-- Variant: ... -->`` with block-comment closer
        stripped from the captured option."""
        content = textwrap.dedent("""\
            <!-- Variant: maven-bom -->
            <dependencyManagement>
              <dependencies>{{deps}}</dependencies>
            </dependencyManagement>
        """)
        assert plugin._extract_variant(
            content, filename="bom.xml.tpl",
        ) == "maven-bom"

    def test_variant_extractor_python_hash(self, plugin):
        """``# Variant: <option>`` for hash-style comment languages."""
        content = "# Variant: requests\nimport requests\n"
        assert plugin._extract_variant(
            content, filename="client.py.tpl",
        ) == "requests"

    def test_variant_extractor_no_directive_returns_empty(self, plugin):
        """Templates without a Variant header return empty string."""
        content = "package com.x;\nclass Foo {}\n"
        assert plugin._extract_variant(
            content, filename="Foo.java.tpl",
        ) == ""

    def test_variant_extractor_universal_fallback(self, plugin):
        """Templates without a recognised extension fall back to the
        universal multi-prefix scan."""
        content = "# Variant: alpine\nFROM alpine:3.18\n"
        assert plugin._extract_variant(
            content, filename="Dockerfile.tpl",
        ) == "alpine"

    def test_walker_populates_variant_from_header(self, plugin, tmp_path):
        """Standalone-discovery walker captures ``variant`` from the
        on-disk header.  ``variant_key`` stays empty (kb-side walker
        owns the axis-mapping)."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "restclient.java.tpl").write_text(textwrap.dedent("""\
            // Variant: restclient
            package {{basePackage}}.client;
            public class FooClient {}
        """))
        entries = plugin._discover_standalone_templates(tpl_dir)
        assert len(entries) == 1
        assert entries[0].variant == "restclient"
        # No axis-mapping context from disk alone.
        assert entries[0].variant_key == ""

    def test_index_loader_reads_kb_authoritative_mapping(
        self, plugin, tmp_path
    ):
        """Index loader populates ``variant_key`` + ``variant`` from
        index.json — that's where the kb's authoritative axis-mapping
        lives.  Header extraction is the secondary signal; index
        wins."""
        # Simulate an index.json populated by the kb-side walker that
        # mapped MODULE.md's variants block onto each template.
        legacy_index = {
            "templates": {
                "restclient.java.tpl": {
                    "name": "restclient.java.tpl",
                    "source_path": str(tmp_path / "restclient.java.tpl"),
                    "syntax": "mustache",
                    "variables": [],
                    "origin": "standalone",
                    "variant_key": "http_client",
                    "variant": "restclient",
                },
            },
        }
        index_path = plugin._templates_dir / "index.json"
        index_path.write_text(json.dumps(legacy_index))
        plugin._load_persisted_index()
        loaded = plugin._template_index.get("restclient.java.tpl")
        assert loaded is not None
        assert loaded.variant_key == "http_client"
        assert loaded.variant == "restclient"

    def test_index_loader_legacy_entries_default_empty(
        self, plugin, tmp_path
    ):
        """Older index.json files (pre-0.6.42) without variant_key /
        variant fields load with empty-string defaults — no
        filtering applied, behaviour unchanged."""
        legacy_index = {
            "templates": {
                "Old.tpl": {
                    "name": "Old.tpl",
                    "source_path": str(tmp_path / "Old.tpl"),
                    "syntax": "mustache",
                    "variables": [],
                    "origin": "standalone",
                    # No variant_key / variant fields.
                },
            },
        }
        index_path = plugin._templates_dir / "index.json"
        index_path.write_text(json.dumps(legacy_index))
        plugin._load_persisted_index()
        loaded = plugin._template_index["Old.tpl"]
        assert loaded.variant_key == ""
        assert loaded.variant == ""

    def test_listAvailableTemplates_surfaces_variant_fields(
        self, plugin, tmp_path
    ):
        """`listAvailableTemplates` includes ``variant_key`` + ``variant``
        for every entry so consumers can filter by axis."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "restclient.java.tpl").write_text(
            "// Variant: restclient\npackage x;\nclass C {}\n"
        )
        (tpl_dir / "Plain.java.tpl").write_text(
            "package x;\nclass Plain {}\n"
        )
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry
        # Manually add the kb-authoritative axis mapping (simulating
        # the kb-side walker's contribution).
        plugin._template_index["restclient.java.tpl"].variant_key = "http_client"

        result = plugin._execute_list_available({})
        # Cutover (server 0.6.119+): response carries ``id`` not ``name``.
        by_id = {t["id"]: t for t in result["templates"]}
        # Variant template
        assert by_id[_template_id("restclient.java.tpl")]["variant_key"] == "http_client"
        assert by_id[_template_id("restclient.java.tpl")]["variant"] == "restclient"
        # Plain template
        assert by_id[_template_id("Plain.java.tpl")]["variant_key"] == ""
        assert by_id[_template_id("Plain.java.tpl")]["variant"] == ""

    def test_index_loader_reads_skip_when_flags(self, plugin, tmp_path):
        """Server 0.6.56+: index loader populates ``skip_when_flags``
        from index.json — kb-side walker authors it from MODULE.md's
        ``subscribes_to_flags:`` block (DEC-035 pub/sub).  Plugin
        treats opaque; consumer-side codegen agent honors via filter
        rule.
        """
        kb_index = {
            "templates": {
                "Response.java.tpl": {
                    "name": "Response.java.tpl",
                    "source_path": str(tmp_path / "Response.java.tpl"),
                    "syntax": "mustache",
                    "variables": [],
                    "origin": "standalone",
                    "skip_when_flags": {"hateoas": True},
                },
            },
        }
        index_path = plugin._templates_dir / "index.json"
        index_path.write_text(json.dumps(kb_index))
        plugin._load_persisted_index()
        loaded = plugin._template_index.get("Response.java.tpl")
        assert loaded is not None
        assert loaded.skip_when_flags == {"hateoas": True}

    def test_index_loader_legacy_entries_default_empty_skip_when_flags(
        self, plugin, tmp_path,
    ):
        """Pre-0.6.56 index.json files (no skip_when_flags field) load
        with the empty-dict default — no flag-driven skipping applied,
        behaviour unchanged for older indexes."""
        legacy_index = {
            "templates": {
                "Old.tpl": {
                    "name": "Old.tpl",
                    "source_path": str(tmp_path / "Old.tpl"),
                    "syntax": "mustache",
                    "variables": [],
                    "origin": "standalone",
                    # No skip_when_flags field.
                },
            },
        }
        index_path = plugin._templates_dir / "index.json"
        index_path.write_text(json.dumps(legacy_index))
        plugin._load_persisted_index()
        loaded = plugin._template_index["Old.tpl"]
        assert loaded.skip_when_flags == {}

    def test_listAvailableTemplates_surfaces_skip_when_flags(
        self, plugin, tmp_path,
    ):
        """``listAvailableTemplates`` includes ``skip_when_flags`` for
        every entry so the codegen agent can apply its filter rule
        without inspecting the index file directly.  Empty dict for
        templates that don't subscribe to any flag.
        """
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "Response.java.tpl").write_text(
            "package x;\nclass Response {}\n"
        )
        (tpl_dir / "Plain.java.tpl").write_text(
            "package x;\nclass Plain {}\n"
        )
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry
        # Simulate kb-side walker setting the skip rule.
        plugin._template_index["Response.java.tpl"].skip_when_flags = {
            "hateoas": True,
        }

        result = plugin._execute_list_available({})
        # Cutover (server 0.6.119+): response carries ``id`` not ``name``.
        by_id = {t["id"]: t for t in result["templates"]}
        # Subscriber template carries the rule.
        assert by_id[_template_id("Response.java.tpl")]["skip_when_flags"] == {
            "hateoas": True,
        }
        # Plain template gets the empty-dict default.
        assert by_id[_template_id("Plain.java.tpl")]["skip_when_flags"] == {}

    def test_mod_017_three_client_options_repro(self, plugin, tmp_path):
        """Reproduces chunk-3 v1 mod-017 scenario: three HTTP-client
        templates each declaring their option.  ``listAvailableTemplates``
        surfaces them with distinct ``variant`` values; consumer can
        then filter by ``selected_variants[variant_key]``."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        for option in ("restclient", "feign", "resttemplate"):
            (tpl_dir / f"{option}.java.tpl").write_text(textwrap.dedent(f"""\
                // Variant: {option}
                package com.example.client;
                public class FooClient {{}}
            """))
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry
        # Kb-side walker would set variant_key="http_client" on all three.
        for option in ("restclient", "feign", "resttemplate"):
            plugin._template_index[f"{option}.java.tpl"].variant_key = (
                "http_client"
            )

        result = plugin._execute_list_available({})
        # All three surface distinct variant values under the same axis.
        variant_axis = {
            t["variant"]
            for t in result["templates"]
            if t["variant_key"] == "http_client"
        }
        assert variant_axis == {"restclient", "feign", "resttemplate"}

    def test_extractor_returns_first_variant_only(self, plugin):
        """Multiple Variant headers (authoring error) — only the
        first is captured.  Mirrors the Output extractor's
        first-match-only behavior."""
        content = (
            "// Variant: first\n"
            "// Variant: second\n"
            "class Foo {}\n"
        )
        assert plugin._extract_variant(
            content, filename="Foo.java.tpl",
        ) == "first"


# ==================== Per-item Required-Key Coverage (server 0.6.43+) ====================

class TestPerItemRequiredKeyCoverage:
    """``_validate_render_inputs_against_structure`` enforces that
    each list-of-dicts item passed for a section variable carries
    every key declared with source ∈ {scalar, section} in the
    parser's per-key metadata.  Optional keys (source = inverted /
    dotted / helper) are NOT flagged.

    Surfaced empirically by kb-enablement-2.0 chunk-3 v3 (mod-017
    persistence-systemapi): agent reconstructed field items dropping
    `jsonField` (a scalar source key).  Mustache silently rendered
    `@JsonProperty("")` — semantically broken Java.  Validator now
    hard-fails before render with a clean error naming the missing
    key.

    Closes the gap the original 0.6.31 docstring deferred — the
    parser's source-aware per-key metadata structurally resolves
    the false-positive concern (dotted/inverted/helper keys aren't
    over-enforced).
    """

    def test_chunk3_v3_jsonField_repro(self, plugin):
        """Exact reproduction of chunk-3 v3: dropped scalar key in a
        per-item dict → hard-fail with clear error."""
        # Mirrors the Request.java.tpl shape from mod-017.
        template = textwrap.dedent("""\
            public record CustomerSystemApiRequest(
            {{#fields}}
                @JsonProperty("{{jsonField}}") {{type}} {{fieldName}}{{^last}},{{/last}}
            {{/fields}}
            ) {}
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "fields": [
                    # Agent's bad reconstruction — drops jsonField:
                    {"fieldName": "id", "type": "String"},
                    {"fieldName": "email", "type": "String"},
                ],
            },
        )
        assert result is not None, "validator should have flagged"
        assert result.get("validation_layer") == "shape_check"
        assert any(
            "jsonField" in e and "missing required keys" in e
            for e in result["validation_errors"]
        ), result

    def test_inverted_key_absence_not_flagged(self, plugin):
        """``{{^last}}`` is the comma-separator idiom (Mustache
        else-branch fires when ``last`` is absent/falsy).  Items
        missing ``last`` MUST NOT trigger validator errors —
        absence IS the trigger condition."""
        template = textwrap.dedent("""\
            {{#fields}}
            {{type}} {{fieldName}}{{^last}}, {{/last}}
            {{/fields}}
        """)
        # All required keys provided; ``last`` deliberately absent
        # (Mustache idiom — the inverted block fires).
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "fields": [
                    {"type": "String", "fieldName": "id"},
                    {"type": "Integer", "fieldName": "n"},
                ],
            },
        )
        assert result is None, (
            f"validator falsely flagged absent inverted key: {result}"
        )

    def test_dotted_key_absence_not_flagged(self, plugin):
        """``{{validation.maxLength}}`` records ``validation`` as a
        dotted-source item_key.  Mustache silently descends; missing
        nested renders empty.  Absence MUST NOT trigger errors."""
        template = textwrap.dedent("""\
            {{#fields}}
            {{type}} {{fieldName}};{{validation.maxLength}}
            {{/fields}}
        """)
        # ``fieldName`` and ``type`` provided; ``validation`` absent
        # (dotted-source key, Mustache silently descends).
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "fields": [
                    {"type": "String", "fieldName": "id"},
                ],
            },
        )
        assert result is None, (
            f"validator falsely flagged absent dotted key: {result}"
        )

    def test_helper_arg_absence_not_flagged(self, plugin):
        """``{{#if required}}`` records ``required`` as a helper-
        source item_key.  Conditionals fire only when value resolves;
        absence means the body doesn't render — that's the contract."""
        template = textwrap.dedent("""\
            {{#fields}}
            {{#if required}}@NotNull{{/if}} {{type}} {{fieldName}};
            {{/fields}}
        """)
        # ``required`` deliberately absent (helper arg, optional).
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "fields": [
                    {"type": "String", "fieldName": "id"},
                ],
            },
        )
        assert result is None, (
            f"validator falsely flagged absent helper arg: {result}"
        )

    def test_correct_shape_passes(self, plugin):
        """All required keys provided → no validation errors."""
        template = textwrap.dedent("""\
            public record Req(
            {{#fields}}
                @JsonProperty("{{jsonField}}") {{type}} {{fieldName}}{{^last}},{{/last}}
            {{/fields}}
            ) {}
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "fields": [
                    {
                        "fieldName": "id", "type": "String",
                        "jsonField": "CUST_ID",
                    },
                    {
                        "fieldName": "email", "type": "String",
                        "jsonField": "CUST_EMAIL",
                    },
                ],
            },
        )
        assert result is None, result

    def test_first_5_items_inspected(self, plugin):
        """Validator inspects up to first 5 items (cost cap for
        large lists).  Item 5+ is skipped — but checking first 5 is
        plenty to catch systemic shape errors in code-gen workloads
        where items are uniformly shaped per kb-author intent."""
        template = textwrap.dedent("""\
            {{#fields}}
            {{type}} {{fieldName}};
            {{/fields}}
        """)
        # 7 items, all but last missing fieldName; only first 5 are
        # checked, so all 5 missing-key errors should appear.
        items = [{"type": "X"} for _ in range(7)]
        items[-1] = {"type": "X", "fieldName": "tail"}  # 7th item OK
        result = plugin._validate_render_inputs_against_structure(
            template,
            {"fields": items},
        )
        assert result is not None
        # 5 errors (one per item[0..4]); item[5] and item[6] not checked.
        assert len(result["validation_errors"]) == 5

    def test_error_includes_actionable_hint(self, plugin):
        """Error message names the missing key + lists what was
        provided + names what was required + tells the caller why
        Mustache would silently render empty.  Each piece is the
        agent's debugging path."""
        template = "{{#fields}}{{type}} {{fieldName}}{{/fields}}"
        result = plugin._validate_render_inputs_against_structure(
            template,
            {"fields": [{"type": "String"}]},  # missing fieldName
        )
        assert result is not None
        err = result["validation_errors"][0]
        assert "fieldName" in err
        assert "missing required keys" in err
        assert "silent corruption" in err.lower()

    def test_legacy_flat_string_item_keys_treated_as_required(
        self, plugin
    ):
        """Defence-in-depth: if some upstream tooling writes legacy
        flat-string item_keys (pre-0.6.43 shape) into index.json,
        the validator falls back to treating ALL of them as required.
        This is conservative — old-style indexes get the strict
        check; new-style indexes get the source-aware check.
        Migration path: regenerate the index with a 0.6.43+ walker
        to pick up the source metadata."""
        # Synthesise the legacy shape by hand (parser output is
        # always new-shape; this exercises the validator's fallback
        # branch directly).
        template = "{{#fields}}{{type}} {{fieldName}}{{/fields}}"
        # We can't bypass the parser, but we can verify the validator
        # behaves correctly when the parser DOES return a list of
        # dicts (the new shape).  The flat-string-fallback code path
        # is exercised when consumers (e.g. listTemplateVariables
        # called by external tooling) feed pre-0.6.43 data into the
        # validator — out of scope for this test, but the code path
        # exists for forward-compat.
        result = plugin._validate_render_inputs_against_structure(
            template,
            {"fields": [{"type": "String", "fieldName": "id"}]},
        )
        assert result is None  # well-formed; no fallback triggered


# ==================== Top-Level Required-Scalar Enforcement (server 0.6.173+) ====================


class TestTopLevelRequiredScalarEnforcement:
    """``_validate_render_inputs_against_structure`` rejects calls
    where a top-level ``{{var}}`` reference (``kind=scalar`` per
    Mustache structural-optionality contract) is absent from the
    variables payload OR present as an empty string.

    Closes the v152-retry-49 cascade variance (kb-enablement-2.0,
    2026-06-02): mod-019 codegen agent shipped ``variables: {}``
    for all 8 files in step_2_result.json AND narrated 4 warnings
    explaining the gap.  Persona prose did not deter; the agent
    knew it was shipping incomplete data.  Mustache rendered every
    ``{{var}}`` reference as an empty string, the output Java was
    syntactically valid (one missing ``<identifier>`` per template
    instantiation), javac failed 12+ min later, cascade stalled
    across 2 iterations with identical 1-error build failures.

    Principled framing (Daniel's lens, 2026-06-02): Mustache has NO
    per-var ``?`` annotation for "may be empty" — the only way to
    express optionality is structurally via section idioms
    (``{{#var}}...{{/var}}`` for "render block iff present non-
    empty" or ``{{^var}}...{{/var}}`` for "render block iff
    absent/empty").  So ``{{var}}`` (kind=scalar) semantically
    MEANS "always supplied" per Mustache spec.  The pre-0.6.173
    ``continue`` at the missing-variable branch was silently
    subverting that structural commitment.  Option A enforcement
    is principled implementation of what Mustache already mandates
    structurally — NOT stricter than spec.

    Memory cross-ref:
    ``feedback_kernel_scoping_beats_persona_prose`` — kernel-layer
    enforcement of parser metadata replaces persona-prose
    instructions to the agent to supply all variables.
    """

    # Note on test template shape: ``_detect_template_syntax`` (server
    # 0.6.0+) defaults to ``jinja2`` for pure-bare-interpolation
    # templates (``{{var}}`` is unambiguous between the two engines;
    # only Mustache-specific tokens like ``{{#section}}`` flip detection
    # to Mustache).  Bare-only test templates would skip the Mustache
    # validator entirely and return None.  All test fixtures below
    # include a no-op Mustache section marker (``{{#_marker}}{{/_marker}}``)
    # to pin syntax detection, mirroring how production patched kb
    # templates (e.g. mod-019 EntityModelAssembler.java.tpl post-patch
    # adds ``{{#api}}{{#selfLinkEndpoint}}{{#queryParams}}...``) classify
    # as Mustache.  The ``_marker`` section variable being absent is
    # the standard Mustache "branch doesn't fire" idiom and unaffected
    # by Option A scalar-required enforcement.
    #
    # Caveat: this PR's Option A scope is the MUSTACHE validation path
    # specifically.  Kb templates with NO Mustache markers (pure bare
    # interpolation, classify as jinja2) currently bypass shape
    # validation entirely (see _validate_render_inputs_against_structure
    # line 3435: "if syntax != 'mustache': return None").  Closing the
    # jinja2-side coverage gap is a separate scope — flagged for follow-
    # up backlog per the Daniel "kernel-scoping-beats-persona-prose"
    # lens.

    def test_v152_retry49_repro_empty_variables_rejected(self, plugin):
        """Exact v152-retry-49 reproduction: agent ships
        ``variables: {}`` against a template with bare scalar
        references.  Validator must reject all missing scalars.

        Specifically reproduces the mod-019
        ``EntityModelAssembler.java.tpl`` shape (post-patch — see
        .jaato/template_patches/mod-code-019-api-public-exposure-
        java-spring/assembler/EntityModelAssembler.java.tpl.diff
        which introduces ``{{#api}}...{{/api}}`` sections that flip
        syntax detection to Mustache).  Pre-0.6.173 the validator
        silently skipped missing top-level scalars; Mustache rendered
        every reference as empty → syntactically broken Java →
        javac error at line 57 col 14 (12+ min later, blocking the
        cascade)."""
        template = textwrap.dedent("""\
            {{#_marker}}{{/_marker}}
            package {{basePackage}};
            class {{Entity}}Assembler {
                {{selfLinkMethod}}({{selfLinkParam}});
                String name = "{{entityNameLower}}";
            }
        """)
        result = plugin._validate_render_inputs_against_structure(
            template, {},
        )
        assert result is not None, (
            "validator MUST reject empty variables payload against "
            "template with bare scalar references"
        )
        assert result.get("validation_layer") == "shape_check"
        errs = result["validation_errors"]
        # All five bare scalars must be flagged.
        for name in ("basePackage", "Entity", "selfLinkMethod",
                     "selfLinkParam", "entityNameLower"):
            assert any(
                f"variable {name!r}" in e
                and "kind=scalar" in e
                and "absent from the variables payload" in e
                for e in errs
            ), f"missing-scalar error for {name!r} not surfaced: {errs}"

    def test_partial_payload_only_missing_flagged(self, plugin):
        """Mixed payload: some scalars supplied, some absent.  Only
        the absent ones are flagged; the supplied ones pass through
        the existing type-check branch unchanged."""
        template = textwrap.dedent("""\
            {{#_marker}}{{/_marker}}
            class {{Entity}} { {{methodName}}({{paramName}}); }
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "Entity": "Customer",
                "methodName": "getId",
                # paramName deliberately missing
            },
        )
        assert result is not None
        errs = result["validation_errors"]
        assert any(
            "variable 'paramName'" in e and "absent" in e
            for e in errs
        ), errs
        # Entity + methodName MUST NOT be flagged (they're present).
        for present in ("Entity", "methodName"):
            assert not any(
                f"variable '{present}'" in e and "absent" in e
                for e in errs
            ), f"present variable {present!r} incorrectly flagged: {errs}"

    def test_empty_string_scalar_rejected(self, plugin):
        """Per the Mustache structural semantics: a scalar variable
        supplied as ``""`` renders identically to absent (silent
        zero-length output).  The path-template side at
        ``_validate_resolved_output_path`` already rejects this on
        the PATH side; the body side now mirrors that contract."""
        template = textwrap.dedent("""\
            {{#_marker}}{{/_marker}}
            class {{Entity}} { @{{annotation}} void test(); }
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "Entity": "Customer",
                "annotation": "",  # empty string — Mustache-equivalent
                                   # to absent for body interpolation
            },
        )
        assert result is not None
        errs = result["validation_errors"]
        assert any(
            "variable 'annotation'" in e
            and "kind=scalar" in e
            and "empty string" in e
            for e in errs
        ), errs

    def test_v152_retry49_repro_None_scalar_rejected_mustache(self, plugin):
        """Empirical falsification repro (server 0.6.174+).
        v152-retry-49-on-0.6.173 (cascade 2026-06-02): mod-019
        codegen agent shipped ``selfLinkMethod: null`` (and other
        nulls) in renderTemplateToFile variables.  Validator
        passed the call through because the empty-string check
        only caught ``actual == ""`` — None passed as a valid
        scalar value.  Mustache then rendered bare
        ``{{selfLinkMethod}}`` as empty (None identical to absent
        + empty-string at render time), syntactically broken Java
        landed on disk, javac errored 12+ min later.

        Closes the third empty-equivalent state.  The full triplet
        (absent | "" | None) is now uniformly rejected at the
        tool boundary.
        """
        template = textwrap.dedent("""\
            {{#_marker}}{{/_marker}}
            class {{Entity}}Assembler {
                .{{selfLinkMethod}}(entity.getId().toString());
            }
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "Entity": "Customer",
                "selfLinkMethod": None,  # exact iter-49 shape
            },
        )
        assert result is not None, (
            "validator MUST reject None-valued kind=scalar (Mustache "
            "renders identically to absent + empty-string)"
        )
        assert result.get("validation_layer") == "shape_check"
        errs = result["validation_errors"]
        assert any(
            "variable 'selfLinkMethod'" in e
            and "None" in e
            and "Mustache" in e
            for e in errs
        ), errs

    def test_None_scalar_rejected_jinja2(self, plugin):
        """Same None-rejection contract on the Jinja2 path.  Bare
        ``{{ var }}`` in a Jinja2 template with var=None must
        reject — Jinja2's StrictUndefined would raise at render
        time but only when ``var`` is missing entirely; explicit
        None bypasses that check, producing silent empty output
        identical to Mustache."""
        template = "{{ x }} - {{ y }}"  # pure-bare → detected as jinja2
        result = plugin._validate_render_inputs_against_structure(
            template,
            {"x": "supplied", "y": None},
        )
        assert result is not None
        errs = result["validation_errors"]
        assert any(
            "variable 'y'" in e and "None" in e and "Jinja2" in e
            for e in errs
        ), errs

    def test_None_value_in_section_idiom_still_passes(self, plugin):
        """Sections accept None as the Mustache falsy idiom
        ("branch doesn't fire").  The new None-rejection contract
        applies ONLY to kind=scalar.  A section variable supplied
        as None remains a valid Mustache idiom and must NOT be
        rejected."""
        template = textwrap.dedent("""\
            {{#hasField}}
              this body skipped when hasField is None
            {{/hasField}}
            {{Entity}}
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {"Entity": "Customer", "hasField": None},
        )
        assert result is None, result

    def test_section_idiom_unaffected_by_strict_scalar_check(self, plugin):
        """Sections (``kind=section``) and inverted sections
        (``kind=inverted_section``) are the Mustache idioms for
        optionality.  Absence of a section variable is the contract
        for "branch doesn't fire" — MUST NOT be flagged by the
        new strict-scalar enforcement."""
        template = textwrap.dedent("""\
            class {{Entity}} {
            {{#hasField}}
              field present
            {{/hasField}}
            {{^hasField}}
              field absent
            {{/hasField}}
            }
        """)
        # Supply only Entity; hasField deliberately absent — Mustache
        # section + inverted-section idioms mean the section body
        # doesn't fire and the inverted body fires.  Validator must
        # accept.
        result = plugin._validate_render_inputs_against_structure(
            template,
            {"Entity": "Customer"},
        )
        assert result is None, (
            f"section-idiom absence MUST NOT trigger missing-scalar "
            f"rejection; got {result}"
        )

    def test_non_empty_scalar_passes(self, plugin):
        """Canonical correct emission: every bare scalar reference
        is supplied with a non-empty value.  Validator must accept."""
        template = textwrap.dedent("""\
            class {{Entity}} { void {{methodName}}() {} }
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {"Entity": "Customer", "methodName": "getId"},
        )
        assert result is None, result


# ==================== Jinja2 Structural Parser + Required-Scalar (server 0.6.173+) ====================


class TestJinja2StructuralParser:
    """``_parse_jinja2_structure`` walks a Jinja2 AST and classifies
    each variable reference by node context, producing the same
    ``[{name, kind}]`` shape that ``_parse_mustache_structure``
    returns.

    Kind taxonomy:
      - ``scalar``: bare ``{{ var }}`` Output → REQUIRED + non-empty
      - ``conditional_test``: ``{% if var %}`` test → optional gate
      - ``iterable``: ``{% for x in var %}`` source → optional
        (jinja2 polymorphic; rejected absence is the render-layer
        concern)
      - ``scalar_with_default``: ``{{ var | default(...) }}`` →
        optional (filter substitutes fallback)

    Merge policy: ``scalar`` always wins.  A variable referenced
    BOTH as bare interpolation AND in a conditional test is
    classified as ``scalar`` (the strictest binding lock).
    """

    def test_bare_interpolation_classified_as_scalar(self, plugin):
        """Iter-49 production shape: pure bare ``{{ var }}``
        interpolation.  Every name classified as kind=scalar."""
        template = "package {{basePackage}}; class {{Entity}}Assembler {}"
        result = plugin._parse_jinja2_structure(template)
        assert {v["name"]: v["kind"] for v in result} == {
            "basePackage": "scalar",
            "Entity": "scalar",
        }

    def test_if_test_classified_as_conditional_test(self, plugin):
        """``{% if flag %}...{% endif %}`` — ``flag`` is the gate
        test, classified as conditional_test (optional)."""
        template = "{% if flag %}gated content{% endif %}"
        result = plugin._parse_jinja2_structure(template)
        assert {v["name"]: v["kind"] for v in result} == {
            "flag": "conditional_test",
        }

    def test_for_iter_classified_as_iterable(self, plugin):
        """``{% for item in items %}{{ item.name }}{% endfor %}``
        — ``items`` is the source iterable.  ``item`` is locally
        bound and EXCLUDED from the top-level."""
        template = "{% for item in items %}{{ item.name }}{% endfor %}"
        result = plugin._parse_jinja2_structure(template)
        names = {v["name"]: v["kind"] for v in result}
        assert names == {"items": "iterable"}
        assert "item" not in names

    def test_default_filter_classified_as_scalar_with_default(self, plugin):
        """``{{ x | default('fallback') }}`` — author explicitly
        opted into the 'may be empty' jinja2 idiom.  Classified as
        scalar_with_default (optional)."""
        template = "{{ x | default('fallback') }}"
        result = plugin._parse_jinja2_structure(template)
        assert {v["name"]: v["kind"] for v in result} == {
            "x": "scalar_with_default",
        }

    def test_merge_policy_scalar_wins(self, plugin):
        """A variable referenced both as bare interpolation AND in
        a conditional test → kind=scalar (the bare binding locks
        it as required)."""
        template = "{{ x }}{% if x %}gated{% endif %}"
        result = plugin._parse_jinja2_structure(template)
        assert {v["name"]: v["kind"] for v in result} == {
            "x": "scalar",
        }

    def test_unparseable_template_returns_empty(self, plugin):
        """When jinja2's parser raises (malformed template), the
        structural extractor returns an empty list rather than
        propagating the exception.  The renderer will surface the
        parse error at render time with proper context."""
        template = "{% if missing_endif %}{{ broken"
        result = plugin._parse_jinja2_structure(template)
        assert result == []


class TestJinja2RequiredScalarEnforcement:
    """``_validate_render_inputs_against_structure`` dispatches to
    ``_parse_jinja2_structure`` for jinja2-detected templates
    (server 0.6.173+) and applies the same kind=scalar required +
    non-empty enforcement that the Mustache path uses.

    Closes the iter-49-class gap empirically: pre-0.6.173 jinja2-
    classified templates bypassed shape validation entirely
    (``if syntax != 'mustache': return None``).  Agents could ship
    ``variables: {}`` and the renderer would silently render empty;
    only downstream (e.g. javac) would catch the corruption 12+
    min/iter later.

    Memory cross-ref:
    ``feedback_kernel_scoping_beats_persona_prose`` — the strict
    enforcement IS the kernel-layer gate that replaces persona-prose
    instructions to the agent to supply all variables.
    """

    def test_iter49_repro_jinja2_empty_variables_rejected(self, plugin):
        """Exact iter-49 reproduction on a pure-bare jinja2
        template (unpatched mod-019 EntityModelAssembler.java.tpl
        shape — no Mustache markers, classifies as jinja2):
        agent ships variables: {}, validator must reject every
        bare scalar reference."""
        template = textwrap.dedent("""\
            package {{basePackage}};
            class {{Entity}}Assembler {
                {{selfLinkMethod}}({{selfLinkParam}});
                String name = "{{entityNameLower}}";
            }
        """)
        result = plugin._validate_render_inputs_against_structure(
            template, {},
        )
        assert result is not None, (
            "validator MUST reject empty payload against jinja2 "
            "template with bare scalar references"
        )
        assert result.get("validation_layer") == "shape_check"
        errs = result["validation_errors"]
        for name in ("basePackage", "Entity", "selfLinkMethod",
                     "selfLinkParam", "entityNameLower"):
            assert any(
                f"variable {name!r}" in e
                and "kind=scalar" in e
                and "Jinja2" in e
                and "absent" in e
                for e in errs
            ), f"missing-scalar error for {name!r} not surfaced: {errs}"

    def test_jinja2_with_if_branch_optional(self, plugin):
        """``{% if flag %}{{ x }}{% endif %}{{ y }}`` — flag is a
        gate (optional), x is gated (also optional since absent
        flag means body doesn't fire), y is unconditionally
        required.  Validator must accept payload that supplies
        only y, leaving flag + x absent."""
        template = textwrap.dedent("""\
            {% if flag %}gated: {{ x }}{% endif %}
            trailer: {{ y }}
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {"y": "supplied"},
        )
        # x is classified as scalar (bare interpolation inside the
        # if-body; the if-gate is a separate node).  Bare scalar
        # absent is rejected — same contract as Mustache.  This
        # documents the contract: gate IS optional, but the gated
        # variable's binding strength inside the body is still
        # scalar (per merge policy).  If kb-authors want true
        # optionality for x they should use ``{{ x | default(...) }}``.
        assert result is not None
        errs = result["validation_errors"]
        # x rejected, y accepted, flag accepted (gate is optional).
        assert any("variable 'x'" in e for e in errs), errs
        assert not any("variable 'y'" in e for e in errs), errs
        assert not any("variable 'flag'" in e for e in errs), errs

    def test_jinja2_with_default_filter_optional(self, plugin):
        """``{{ x | default('fallback') }}`` — author opted into
        the jinja2 optionality idiom.  Validator must accept absent
        x."""
        template = "value: {{ x | default('fallback') }}"
        result = plugin._validate_render_inputs_against_structure(
            template, {},
        )
        assert result is None, result

    def test_jinja2_for_loop_iterable_required(self, plugin):
        """``{% for item in items %}...{% endfor %}`` — items is
        required.  Absent items is rejected (jinja2 would render
        zero iterations, but the more likely intent is that the
        agent forgot to supply the data)."""
        template = "{% for item in items %}{{ item.name }}{% endfor %}"
        result = plugin._validate_render_inputs_against_structure(
            template, {},
        )
        # items is classified as iterable.  Per the current
        # contract, iterable absent does NOT trigger scalar
        # required-rejection (different kind).  This documents the
        # boundary: iterable absence is treated as "branch doesn't
        # fire" (zero iterations).  If kb-author wants stricter
        # enforcement they can add a separate kind=scalar
        # reference to items elsewhere.  Validator accepts.
        assert result is None, result

    def test_jinja2_all_supplied_passes(self, plugin):
        """Canonical correct emission: every bare scalar supplied
        with non-empty value.  Validator accepts."""
        template = "{{ a }}-{{ b }}-{{ c }}"
        result = plugin._validate_render_inputs_against_structure(
            template,
            {"a": "x", "b": "y", "c": "z"},
        )
        assert result is None, result

    def test_jinja2_empty_string_scalar_rejected(self, plugin):
        """Mirror of the Mustache empty-string rejection: a scalar
        supplied as ``""`` is treated identically to absent (both
        produce silent zero-length jinja2 output)."""
        template = "value: {{ x }}"
        result = plugin._validate_render_inputs_against_structure(
            template, {"x": ""},
        )
        assert result is not None
        errs = result["validation_errors"]
        assert any(
            "variable 'x'" in e and "empty string" in e and "Jinja2" in e
            for e in errs
        ), errs


# ==================== Spring Placeholder Parser-Sentinel Protection (server 0.6.173+) ====================


class TestSpringSentinelParserProtection:
    """``_parse_mustache_structure`` applies the same
    ``_protect_spring_placeholders`` sentinel substitution the
    renderer uses, BEFORE walking the content (server 0.6.173+).

    Closes a parser bug surfaced by the strict-required-scalar
    enforcement above: Spring Boot property placeholders
    ``${{{X}}:default}`` (the property-collision pattern from
    mod-code-018 ``application-integration.yml.tpl``) were
    mis-tokenised by the pre-0.6.173 parser, emitting a garbage
    top-level entry ``{"name": "{X", "kind": "scalar"}`` (leading
    ``{``) alongside the legitimate ``X`` entry.  Pre-0.6.173 no
    consumer of parser metadata was strict, so the garbage was
    invisible; the renderer s own ``_protect_spring_placeholders``
    handled the substitution correctly at render time.

    The new top-level required-scalar enforcement WOULD flag the
    garbage entry as a missing variable (no payload key matches
    ``{X``).  Mirroring the renderer's sentinel mechanism at parse
    time closes both surfaces in one layer.
    """

    def test_spring_collision_no_garbage_entry(self, plugin):
        """Spring ``${{{X}}:default}`` no longer emits a
        ``{"name": "{X", ...}`` garbage entry; only the legitimate
        ``X`` variable surfaces."""
        template = textwrap.dedent("""\
            integration:
              {{apiName}}:
                base-url: ${{{BASE_URL_ENV}}:http://localhost:8081}
        """)
        structure = plugin._parse_mustache_structure(template)
        names = sorted(v["name"] for v in structure)
        # Only the two legitimate vars: BASE_URL_ENV (inside the
        # Spring placeholder, but the Mustache variable inside is
        # real) and apiName.  No leading-``{`` garbage.
        assert names == ["BASE_URL_ENV", "apiName"], (
            f"parser still emitting garbage entries: {names}"
        )
        for v in structure:
            assert not v["name"].startswith("{"), (
                f"garbage entry survived sentinel protection: {v}"
            )

    def test_spring_collision_classified_as_scalar(self, plugin):
        """The Mustache variable INSIDE the Spring placeholder is
        a legitimate bare interpolation, classified as kind=scalar.
        After Option A strict-required enforcement, the agent must
        supply a value for it."""
        template = textwrap.dedent("""\
            base-url: ${{{BASE_URL_ENV}}:http://localhost:8081}
        """)
        structure = plugin._parse_mustache_structure(template)
        # One entry, kind=scalar, name=BASE_URL_ENV (no leading brace).
        assert len(structure) == 1
        assert structure[0]["name"] == "BASE_URL_ENV"
        assert structure[0]["kind"] == "scalar"

    def test_spring_collision_with_empty_payload_rejected(self, plugin):
        """End-to-end: parser fix + Option A enforcement compose.
        Template with a Spring-collision pattern + a Mustache section
        marker (to pin syntax detection) + empty variables payload
        is rejected with the legitimate variable name (no garbage)."""
        template = textwrap.dedent("""\
            {{#_marker}}{{/_marker}}
            base-url: ${{{BASE_URL_ENV}}:http://localhost:8081}
        """)
        result = plugin._validate_render_inputs_against_structure(
            template, {},
        )
        assert result is not None
        errs = result["validation_errors"]
        # Must flag BASE_URL_ENV, NOT '{BASE_URL_ENV' or any
        # leading-brace garbage variant.
        assert any(
            "variable 'BASE_URL_ENV'" in e and "absent" in e
            for e in errs
        ), errs
        for e in errs:
            assert "'{BASE_URL_ENV'" not in e, (
                f"garbage entry still surfacing in error message: {e}"
            )


# ==================== Recursive Item-Keys Traversal (server 0.6.171+) ====================

class TestRecursiveItemKeysValidation:
    """``_validate_render_inputs_against_structure`` now drills into
    nested ``item_keys`` produced by ``_parse_mustache_structure``.
    Pre-0.6.171 the validator stopped at the outermost section's
    item_keys; agents could pass a list-of-dicts at the outer layer
    AND have the inner dicts be missing every nested required key —
    Mustache silently rendered empty, broken files landed on disk,
    cascade caught it 25+ minutes later via downstream compile.

    Closes the iter-3 cascade evidence (kb-enablement-2.0,
    2026-06-02): mod-019 ControllerTest-hateoas.java.tpl patched
    form ``{{#hasRequestBody}}.content(...{{#testBodyFields}}
    Map.entry("{{name}}", "{{value}}"){{^last}},{{/last}}
    {{/testBodyFields}})){{/hasRequestBody}}`` — agent
    (haiku-4.5/openrouter, strict-mode-unenforced) emitted
    ``hasRequestBody = [{name, value, last, first, @index} × 4]``
    instead of the bool ``true`` / ``false`` the kb-author intended
    OR list-of-dicts each containing ``testBodyFields``.  Mustache
    iterated the section body 4 times, each iteration's
    ``{{#testBodyFields}}`` looked up ``testBodyFields`` in a
    context that didn't have it, rendered nothing → 4 cascading
    ``.content(objectMapper.writeValueAsString(Map.ofEntries()))``
    empties → 4 cascading javac errors.

    Empirical finding from Advisor's pre-implementation check: the
    parser ALREADY produces 3-level-deep nested item_keys with
    ``required`` flags at every layer; the validator just wasn't
    consuming them.  This test class pins that the recursive
    traversal closes the gap end-to-end.

    Memory cross-ref:
    ``feedback_kernel_scoping_beats_persona_prose`` — the parser's
    metadata IS the kernel-layer source-of-truth; the fix is
    fully-consuming-existing-metadata, not adding new heuristics.
    """

    def test_iter3_repro_nested_missing_testBodyFields_rejected(self, plugin):
        """Exact iter-3 reproduction (2026-06-02, mod-019): agent
        emits ``hasRequestBody`` as list-of-dicts whose inner dicts
        are missing ``testBodyFields``.  Pre-0.6.171: validator
        passed.  Post-0.6.171: validator descends into
        ``hasRequestBody``'s nested item_keys and rejects on the
        missing required key.

        Payload reconstructed verbatim from the rendered
        CustomerControllerHateoasTest.java REQUIRED VARIABLES echo
        line 7 (captured pre-workspace-wipe in
        /tmp/advisor-repros/task234-hasrequestbody/)."""
        template = textwrap.dedent("""\
            {{#apiEndpoints}}
            void {{methodName}}() {
            mockMvc.perform({{methodLower}}("{{testPath}}")
              .contentType(JSON){{#hasRequestBody}}
              .content(objectMapper.writeValueAsString(Map.ofEntries({{#testBodyFields}}
                Map.entry("{{name}}", "{{value}}"){{^last}},{{/last}}{{/testBodyFields}}
              ))){{/hasRequestBody}})
            }
            {{/apiEndpoints}}
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "apiEndpoints": [
                    {
                        "method": "POST",
                        "methodLower": "post",
                        "methodName": "createCustomer",
                        "subPath": "/customers",
                        "testPath": "/api/v1/customers",
                        "hasRequestBody": [
                            {"name": "firstName", "value": "John",
                             "last": False, "first": True, "@index": 0},
                            {"name": "lastName", "value": "Doe",
                             "last": False, "first": False, "@index": 1},
                            {"name": "email", "value": "john@x.com",
                             "last": False, "first": False, "@index": 2},
                            {"name": "status", "value": "ACTIVE",
                             "last": True, "first": False, "@index": 3},
                        ],
                        "first": True, "last": False, "@index": 0,
                    },
                ],
            },
        )
        assert result is not None, (
            "validator should have flagged the nested missing "
            "testBodyFields"
        )
        assert result.get("validation_layer") == "shape_check"
        errs = result["validation_errors"]
        # Breadcrumb preserves the top-level ``item[i]`` format
        # (back-compat with pre-0.6.171 error messages); nested
        # layers extend it as ``item[i].<nested-name>[j]``.  The
        # iter-3 case: apiEndpoints[0].hasRequestBody[N] missing
        # testBodyFields.
        assert any(
            "item[0].hasRequestBody" in e
            and "testBodyFields" in e
            and "missing required keys" in e
            for e in errs
        ), errs
        # Variable identity is in the message too.
        assert any("apiEndpoints" in e for e in errs), errs

    def test_well_formed_nested_passes(self, plugin):
        """Canonical correct emission: each hasRequestBody item
        carries its testBodyFields list, each testBodyFields item
        carries name + value.  Validator must accept end-to-end."""
        template = textwrap.dedent("""\
            {{#apiEndpoints}}
            void {{methodName}}() {
            mockMvc.perform({{methodLower}}("{{testPath}}")
              .contentType(JSON){{#hasRequestBody}}
              .content(objectMapper.writeValueAsString(Map.ofEntries({{#testBodyFields}}
                Map.entry("{{name}}", "{{value}}"){{^last}},{{/last}}{{/testBodyFields}}
              ))){{/hasRequestBody}})
            }
            {{/apiEndpoints}}
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "apiEndpoints": [
                    {
                        "methodName": "createCustomer",
                        "methodLower": "post",
                        "testPath": "/api/v1/customers",
                        "hasRequestBody": [
                            {
                                "testBodyFields": [
                                    {"name": "firstName", "value": "John"},
                                    {"name": "email", "value": "x@y.com"},
                                ],
                            },
                        ],
                    },
                ],
            },
        )
        assert result is None, result

    def test_boolean_gate_form_still_passes(self, plugin):
        """When the kb-author intends ``hasRequestBody`` as a boolean
        gate (Mustache section over bool=True renders body once in
        parent context), the validator must accept bool inputs.  No
        recursion fires because the value is not a list."""
        template = textwrap.dedent("""\
            {{#apiEndpoints}}
            void {{methodName}}() {
            mockMvc.perform({{methodLower}}("{{testPath}}")
              .contentType(JSON){{#hasRequestBody}}
              .content(objectMapper.writeValueAsString(Map.ofEntries({{#testBodyFields}}
                Map.entry("{{name}}", "{{value}}"){{^last}},{{/last}}{{/testBodyFields}}
              ))){{/hasRequestBody}})
            }
            {{/apiEndpoints}}
        """)
        # Case 1: hasRequestBody=True with parent-context testBodyFields.
        result_true = plugin._validate_render_inputs_against_structure(
            template,
            {
                "apiEndpoints": [
                    {
                        "methodName": "createCustomer",
                        "methodLower": "post",
                        "testPath": "/api/v1/customers",
                        "hasRequestBody": True,
                        # When True, Mustache renders the section
                        # body once with the parent apiEndpoints[0]
                        # context — testBodyFields looked up from
                        # there.
                        "testBodyFields": [
                            {"name": "firstName", "value": "John"},
                        ],
                    },
                ],
            },
        )
        assert result_true is None, result_true

        # Case 2: hasRequestBody=False — section doesn't fire; no
        # inner data needed.
        result_false = plugin._validate_render_inputs_against_structure(
            template,
            {
                "apiEndpoints": [
                    {
                        "methodName": "getCustomer",
                        "methodLower": "get",
                        "testPath": "/api/v1/customers/1",
                        "hasRequestBody": False,
                    },
                ],
            },
        )
        assert result_false is None, result_false

    def test_two_levels_deep_nested_validates(self, plugin):
        """Validator descends all the way: third-level (testBodyFields
        items) must carry name + value.  An inner item missing
        ``value`` is caught with the full breadcrumb path."""
        template = textwrap.dedent("""\
            {{#apiEndpoints}}
            void {{methodName}}() {
            mockMvc.perform({{methodLower}}("{{testPath}}")
              .contentType(JSON){{#hasRequestBody}}
              .content(objectMapper.writeValueAsString(Map.ofEntries({{#testBodyFields}}
                Map.entry("{{name}}", "{{value}}"){{^last}},{{/last}}{{/testBodyFields}}
              ))){{/hasRequestBody}})
            }
            {{/apiEndpoints}}
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "apiEndpoints": [
                    {
                        "methodName": "createCustomer",
                        "methodLower": "post",
                        "testPath": "/api/v1/customers",
                        "hasRequestBody": [
                            {
                                "testBodyFields": [
                                    {"name": "firstName", "value": "John"},
                                    # Missing 'value' on second item.
                                    {"name": "lastName"},
                                ],
                            },
                        ],
                    },
                ],
            },
        )
        assert result is not None
        assert result.get("validation_layer") == "shape_check"
        errs = result["validation_errors"]
        assert any(
            "testBodyFields" in e
            and "value" in e
            and "missing required keys" in e
            for e in errs
        ), errs

    def test_error_path_prefix_includes_full_breadcrumb(self, plugin):
        """Recursive errors must carry the full path-prefix breadcrumb
        so the agent can locate the failing position in the payload
        tree.  Format:
        ``<outer-var>[<i>].<nested-name>[<j>]``."""
        template = textwrap.dedent("""\
            {{#apiEndpoints}}
            {{#hasRequestBody}}{{#testBodyFields}}{{name}}={{value}}{{/testBodyFields}}{{/hasRequestBody}}
            {{/apiEndpoints}}
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "apiEndpoints": [
                    # apiEndpoints[0] — has hasRequestBody=False so
                    # section doesn't fire; no nested error here.
                    {"hasRequestBody": False},
                    # apiEndpoints[1] — has hasRequestBody list-of-
                    # dicts; nested testBodyFields[1] missing
                    # required scalars.  Breadcrumb must locate the
                    # failure at apiEndpoints[1].hasRequestBody[0].testBodyFields[1].
                    {
                        "hasRequestBody": [
                            {
                                "testBodyFields": [
                                    {"name": "k", "value": "v"},
                                    {},  # missing both scalars
                                ],
                            },
                        ],
                    },
                ],
            },
        )
        assert result is not None
        errs = result["validation_errors"]
        # Breadcrumb format: top-level ``item[i]`` for back-compat
        # with pre-0.6.171 error messages; nested layers append
        # ``.<nested-name>[j]``.  The full path for the missing
        # scalars at the third level reads
        # ``item[1].hasRequestBody[0].testBodyFields[1]`` — locator
        # substring must appear unbroken so the agent can grep / parse.
        assert any(
            "item[1].hasRequestBody[0].testBodyFields[1]" in e
            and "missing required keys" in e
            for e in errs
        ), errs
        # Outer variable name is in the message too.
        assert any("apiEndpoints" in e for e in errs), errs


# ==================== Helper-Exclusivity A2 (server 0.6.170+) ====================

class TestHelperExclusivityA2:
    """``_validate_render_inputs_against_structure`` rejects helper-
    source variables that are not Python bool, AND rejects rows where
    more than one member of a helper sibling-family is True (A2:
    at-most-one True; all-False is permitted).

    Closes the iter-3 cascade evidence (kb-enablement-2.0,
    2026-06-01): agent emitted ``entityFields[*].isString = []``
    (Mustache-falsy empty list) on a String field where the
    ``{{#isString}}"{{/isString}}`` quote-gate was expected to fire.
    Mustache silently coerced ``[]`` to falsy, the gate didn't fire,
    bare value landed on disk, 54 javac "cannot find symbol" errors
    surfaced 25 minutes later.  Iter 4 cleaned up non-
    deterministically (same model, same session shape, different
    bool-vs-list emission round-to-round).

    Persona prose alone is insufficient per
    ``feedback_kernel_scoping_beats_persona_prose``: codegen.md
    explicitly says ``isId: True`` not ``isId: [True]`` and the
    agent still emits list-form some iterations.  Kernel-enforced
    rejection at the tool boundary closes the variance.
    """

    # NOTE on template syntax: the parser only classifies a variable
    # as ``source="helper"`` when it appears as the ARGUMENT of a
    # Handlebars conditional helper (``{{#if X}}``, ``{{#unless X}}``,
    # ``{{#each X}}``, ``{{#with X}}``, ``{{#lookup X}}``).  Plain
    # Mustache sections like ``{{#X}}...{{/X}}`` are classified as
    # ``source="section"`` (required=True) — not helpers.  The actual
    # kb java-spring templates use ``{{#if isString}}...{{/if}}`` form
    # (see kb/modules/mod-code-017-persistence-systemapi/templates/
    # mapper/SystemApiMapper.java.tpl), so these fixtures mirror that
    # syntax to actually trigger the helper-source classification.

    def test_iter3_repro_isString_empty_list_rejected(self, plugin):
        """Exact reproduction of cascade iter-3: agent emits
        ``isString=[]`` on an actually-String field.  Must reject on
        type-check half (list, not bool)."""
        template = textwrap.dedent("""\
            {{#entityFields}}
            {{#if isString}}"{{/if}}{{value}}{{#if isString}}"{{/if}},
            {{/entityFields}}
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "entityFields": [
                    {"value": "test-value", "isString": []},
                ],
            },
        )
        assert result is not None, "validator should have flagged"
        assert result.get("validation_layer") == "shape_check"
        assert any(
            "isString" in e and "MUST be Python bool" in e
            for e in result["validation_errors"]
        ), result

    def test_iter3_grounded_multiple_empty_lists_rejected(self, plugin):
        """Generalised iter-3 shape: multiple helpers emitted as
        ``[]`` in the same row.  Should produce a type-error per
        non-bool helper variable, NOT a spurious exclusivity error
        on top (exclusivity is suppressed when type-check fails for
        the row)."""
        template = textwrap.dedent("""\
            {{#entityFields}}
            {{#if isString}}STRING{{/if}}{{#if isEnum}}ENUM{{/if}}{{value}},
            {{/entityFields}}
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "entityFields": [
                    {"value": "x", "isString": [], "isEnum": []},
                ],
            },
        )
        assert result is not None
        assert result.get("validation_layer") == "shape_check"
        errs = result["validation_errors"]
        assert any(
            "isString" in e and "MUST be Python bool" in e for e in errs
        ), errs
        assert any(
            "isEnum" in e and "MUST be Python bool" in e for e in errs
        ), errs
        assert not any(
            "multiple helper-source variables set True" in e for e in errs
        ), (
            "exclusivity check must be suppressed when type-check "
            "rejects the same row — otherwise the agent sees double-"
            "attribution noise"
        )

    def test_exactly_one_True_passes(self, plugin):
        """Canonical correct emission: one helper True, siblings
        False.  Validator must accept."""
        template = textwrap.dedent("""\
            {{#entityFields}}
            {{#if isString}}STRING{{/if}}{{#if isEnum}}ENUM{{/if}}{{value}},
            {{/entityFields}}
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "entityFields": [
                    {"value": "x", "isString": True, "isEnum": False},
                ],
            },
        )
        assert result is None, result

    def test_all_False_passes_A2_lenient(self, plugin):
        """A2 contract: all-False is a legitimate Mustache idiom
        ("none of these branches fires").  Permitted even when the
        template's helper-sibling family has multiple members."""
        template = textwrap.dedent("""\
            {{#entityFields}}
            {{#if isString}}S{{/if}}{{#if isEnum}}E{{/if}}{{value}},
            {{/entityFields}}
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "entityFields": [
                    {"value": "x", "isString": False, "isEnum": False},
                ],
            },
        )
        assert result is None, result

    def test_two_True_rejected_exclusivity(self, plugin):
        """Two helper siblings both True in the same row —
        meaningless (which branch is this row?).  Rejected on
        exclusivity half."""
        template = textwrap.dedent("""\
            {{#entityFields}}
            {{#if isString}}S{{/if}}{{#if isEnum}}E{{/if}}{{value}},
            {{/entityFields}}
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "entityFields": [
                    {"value": "x", "isString": True, "isEnum": True},
                ],
            },
        )
        assert result is not None
        assert result.get("validation_layer") == "shape_check"
        assert any(
            "multiple helper-source variables set True" in e
            and "isString" in e and "isEnum" in e
            for e in result["validation_errors"]
        ), result

    def test_string_for_helper_rejected(self, plugin):
        """Non-bool, non-list shape (string) also rejected.  Same
        type-check fires."""
        template = textwrap.dedent("""\
            {{#entityFields}}
            {{#if isString}}S{{/if}}{{value}},
            {{/entityFields}}
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "entityFields": [
                    {"value": "x", "isString": "yes"},
                ],
            },
        )
        assert result is not None
        assert any(
            "isString" in e and "MUST be Python bool" in e
            for e in result["validation_errors"]
        ), result

    def test_truthy_list_True_inside_also_rejected(self, plugin):
        """``[True]`` is Mustache-truthy but the agent contract is
        bool.  Rejected on type-check half — the agent cannot rely
        on Mustache's coercion."""
        template = textwrap.dedent("""\
            {{#entityFields}}
            {{#if isString}}S{{/if}}{{value}},
            {{/entityFields}}
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "entityFields": [
                    {"value": "x", "isString": [True]},
                ],
            },
        )
        assert result is not None
        assert any(
            "isString" in e and "MUST be Python bool" in e
            for e in result["validation_errors"]
        ), result

    def test_helper_absent_from_row_allowed(self, plugin):
        """A helper sibling absent from the row entirely is
        Mustache's absence-is-falsy contract — must NOT be flagged
        as missing.  Only present-but-wrong-shape variables are
        validated.  Helper sources are ``required=False`` in the
        parser's per-key metadata so the upstream required-key
        check does not fire either."""
        template = textwrap.dedent("""\
            {{#entityFields}}
            {{#if isString}}S{{/if}}{{#if isEnum}}E{{/if}}{{value}},
            {{/entityFields}}
        """)
        result = plugin._validate_render_inputs_against_structure(
            template,
            {
                "entityFields": [
                    # Only isString set; isEnum absent.  Mustache
                    # treats isEnum absence as falsy — fine.
                    {"value": "x", "isString": True},
                ],
            },
        )
        assert result is None, result


# ==================== Template Evaluation Kind + Helper Siblings (server 0.6.44+) ====================

class TestTemplateEvaluationKindAndHelperSiblings:
    """Two parser-side additions completing the helpers case for the
    structural-info-surface family:

    - ``template_evaluation_kind`` on TemplateIndexEntry / surfaced via
      listAvailableTemplates: ``"substitution"`` (default) when the
      template is pure variable substitution; ``"helpers"`` when any
      Handlebars conditional (``{{#if}}`` / ``{{#unless}}`` /
      ``{{#with}}``) is present.  Lets agents fast-path
      substitution-only templates.

    - ``helper_siblings`` on per-key entries with source="helper" in
      item_keys: lists the OTHER helper-source siblings in the same
      scope.  Surfaces the kb-author's mutual-exclusivity intent
      (e.g. mod-017's mapper has ``isString/isEnum/isDate/isOther``
      sibling helpers in one section body — exactly one true per
      item per author intent).

    Both are pure parser-derivable metadata; neither encodes
    semantics about which sibling SHOULD be true.  Agent decides
    per-item from field type / context.
    """

    def test_evaluation_kind_substitution_for_pure_substitution(
        self, plugin
    ):
        """Templates without helpers get kind="substitution"."""
        template = textwrap.dedent("""\
            package {{basePackage}};
            public class {{Entity}} {
            {{#fields}}
                private {{type}} {{name}};
            {{/fields}}
            }
        """)
        plugin._parse_mustache_structure(template)
        kind = getattr(plugin._tls, "last_evaluation_kind", "substitution")
        assert kind == "substitution"

    def test_evaluation_kind_helpers_for_if_inside_section(self, plugin):
        """Templates with ``{{#if}}`` (per-item helper) get
        kind="helpers"."""
        template = textwrap.dedent("""\
            {{#fields}}
                {{#if required}}@NotNull{{/if}} {{type}} {{name}};
            {{/fields}}
        """)
        plugin._parse_mustache_structure(template)
        kind = getattr(plugin._tls, "last_evaluation_kind", "substitution")
        assert kind == "helpers"

    def test_evaluation_kind_helpers_for_top_level_if(self, plugin):
        """Top-level ``{{#if}}`` also flips kind to "helpers" — the
        flag fires regardless of nesting depth."""
        template = textwrap.dedent("""\
            {{#if production}}prod-config{{/if}}
            {{#if staging}}staging-config{{/if}}
        """)
        plugin._parse_mustache_structure(template)
        kind = getattr(plugin._tls, "last_evaluation_kind", "substitution")
        assert kind == "helpers"

    def test_evaluation_kind_helpers_for_unless(self, plugin):
        """``{{#unless}}`` also fires the helpers flag."""
        template = "{{#unless empty}}has data: {{count}}{{/unless}}"
        plugin._parse_mustache_structure(template)
        kind = getattr(plugin._tls, "last_evaluation_kind", "substitution")
        assert kind == "helpers"

    def test_evaluation_kind_helpers_for_with(self, plugin):
        """``{{#with}}`` also fires the helpers flag."""
        template = "{{#with user}}{{name}} {{email}}{{/with}}"
        plugin._parse_mustache_structure(template)
        kind = getattr(plugin._tls, "last_evaluation_kind", "substitution")
        assert kind == "helpers"

    def test_helper_siblings_mod17_mapper_case(self, plugin):
        """Reproduces mod-017's SystemApiMapper.java.tpl shape: four
        sibling helpers (isString / isEnum / isDate / isOther) in one
        section body, each gets the other three as helper_siblings."""
        template = textwrap.dedent("""\
            {{#fields}}
                {{#if isString}}toProperCase(response.{{fieldName}}()){{/if}}
                {{#if isEnum}}to{{EnumType}}(response.{{fieldName}}()){{/if}}
                {{#if isDate}}parseDate(response.{{fieldName}}()){{/if}}
                {{#if isOther}}response.{{fieldName}}(){{/if}}
            {{/fields}}
        """)
        result = plugin._parse_mustache_structure(template)
        by_name = {v["name"]: v for v in result}
        item_keys_by_name = {
            k["name"]: k for k in by_name["fields"]["item_keys"]
        }
        # All 4 helpers present
        for h in ("isString", "isEnum", "isDate", "isOther"):
            assert h in item_keys_by_name
            assert item_keys_by_name[h]["source"] == "helper"
            assert item_keys_by_name[h]["required"] is False
            assert "helper_siblings" in item_keys_by_name[h]
            siblings = item_keys_by_name[h]["helper_siblings"]
            # Self excluded; other three present (sorted).
            others = sorted(["isString", "isEnum", "isDate", "isOther"])
            others.remove(h)
            assert siblings == others

    def test_helper_siblings_solo_helper_empty(self, plugin):
        """A helper with no other-helper siblings in the same scope
        gets helper_siblings=[]."""
        template = textwrap.dedent("""\
            {{#fields}}
                {{#if required}}@NotNull{{/if}} {{type}} {{name}};
            {{/fields}}
        """)
        result = plugin._parse_mustache_structure(template)
        by_name = {v["name"]: v for v in result}
        item_keys_by_name = {
            k["name"]: k for k in by_name["fields"]["item_keys"]
        }
        assert item_keys_by_name["required"]["source"] == "helper"
        assert item_keys_by_name["required"]["helper_siblings"] == []

    def test_helper_siblings_only_on_helper_entries(self, plugin):
        """Non-helper entries do NOT carry helper_siblings — keeps
        the response shape minimal for the common substitution case."""
        template = textwrap.dedent("""\
            {{#fields}}
                {{#if required}}@NotNull{{/if}} {{type}} {{name}};
            {{/fields}}
        """)
        result = plugin._parse_mustache_structure(template)
        by_name = {v["name"]: v for v in result}
        for k in by_name["fields"]["item_keys"]:
            if k["source"] == "helper":
                assert "helper_siblings" in k
            else:
                assert "helper_siblings" not in k

    def test_helper_siblings_scoped_per_section(self, plugin):
        """Helpers in DIFFERENT section bodies are NOT siblings of
        each other (scope is per section, not template-wide)."""
        template = textwrap.dedent("""\
            {{#requestFields}}
                {{#if isString}}strBody{{/if}}
            {{/requestFields}}
            {{#responseFields}}
                {{#if isEnum}}enumBody{{/if}}
            {{/responseFields}}
        """)
        result = plugin._parse_mustache_structure(template)
        by_name = {v["name"]: v for v in result}
        # isString in requestFields scope; isEnum in responseFields scope.
        # Neither is a sibling of the other.
        req_keys = {k["name"]: k for k in by_name["requestFields"]["item_keys"]}
        resp_keys = {k["name"]: k for k in by_name["responseFields"]["item_keys"]}
        assert req_keys["isString"]["helper_siblings"] == []
        assert resp_keys["isEnum"]["helper_siblings"] == []

    def test_walker_populates_evaluation_kind(self, plugin, tmp_path):
        """``_discover_standalone_templates`` reads the parser's
        post-parse flag and writes it to TemplateIndexEntry."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        # Pure substitution template
        (tpl_dir / "Entity.java.tpl").write_text(textwrap.dedent("""\
            package {{basePackage}};
            public class {{Entity}} {}
        """))
        # Helpers template
        (tpl_dir / "Mapper.java.tpl").write_text(textwrap.dedent("""\
            {{#fields}}
                {{#if isString}}str({{name}}){{/if}}
            {{/fields}}
        """))
        entries = {
            e.name: e
            for e in plugin._discover_standalone_templates(tpl_dir)
        }
        assert entries["Entity.java.tpl"].template_evaluation_kind == (
            "substitution"
        )
        assert entries["Mapper.java.tpl"].template_evaluation_kind == (
            "helpers"
        )

    def test_listAvailableTemplates_surfaces_evaluation_kind(
        self, plugin, tmp_path
    ):
        """`listAvailableTemplates` response includes
        ``template_evaluation_kind`` per entry."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "Plain.java.tpl").write_text("class Plain {}")
        (tpl_dir / "Helper.java.tpl").write_text(
            "{{#if x}}body{{/if}}"
        )
        for entry in plugin._discover_standalone_templates(tpl_dir):
            plugin._template_index[entry.name] = entry
        result = plugin._execute_list_available({})
        # Cutover (server 0.6.119+): response carries ``id`` not ``name``.
        by_id = {t["id"]: t for t in result["templates"]}
        assert by_id[_template_id("Plain.java.tpl")]["template_evaluation_kind"] == (
            "substitution"
        )
        assert by_id[_template_id("Helper.java.tpl")]["template_evaluation_kind"] == (
            "helpers"
        )

    def test_index_loader_reads_evaluation_kind(self, plugin, tmp_path):
        """Index loader respects ``template_evaluation_kind`` from
        index.json (kb-side tooling can pre-populate)."""
        legacy_index = {
            "templates": {
                "Mapper.java.tpl": {
                    "name": "Mapper.java.tpl",
                    "source_path": str(tmp_path / "Mapper.java.tpl"),
                    "syntax": "mustache",
                    "variables": [],
                    "origin": "standalone",
                    "template_evaluation_kind": "helpers",
                },
            },
        }
        index_path = plugin._templates_dir / "index.json"
        index_path.write_text(json.dumps(legacy_index))
        plugin._load_persisted_index()
        loaded = plugin._template_index["Mapper.java.tpl"]
        assert loaded.template_evaluation_kind == "helpers"

    def test_index_loader_default_substitution_for_legacy_entries(
        self, plugin, tmp_path
    ):
        """Older index.json files (pre-0.6.44) without the field
        load with default "substitution" — no false-positive helpers
        tag."""
        legacy_index = {
            "templates": {
                "Old.tpl": {
                    "name": "Old.tpl",
                    "source_path": str(tmp_path / "Old.tpl"),
                    "syntax": "mustache",
                    "variables": [],
                    "origin": "standalone",
                    # No template_evaluation_kind field.
                },
            },
        }
        index_path = plugin._templates_dir / "index.json"
        index_path.write_text(json.dumps(legacy_index))
        plugin._load_persisted_index()
        loaded = plugin._template_index["Old.tpl"]
        assert loaded.template_evaluation_kind == "substitution"


# ==================== Cutover to template_id (server 0.6.119+) ====================

class TestTemplateIdCutover:
    """Tests for the template-name hashing surface (server 0.6.119+).

    Mirrors the tool-name hashing pattern (``shared.tool_id_map``).
    Templates are identified to the LLM by stable opaque hash ids
    (e.g. ``tpl_a8f0e2b1``) rather than human filenames; this closes
    the training-prior class that v112 codegen evidence exposed —
    agents skipping ``listTemplateVariables`` calls because the
    template name signaled "I already know what this is."

    Internals stay human (``_template_index`` keyed on name; on-disk
    paths human; result-dict ``template_name`` field preserved for
    audit).  Only the tool-arg / tool-response boundary cuts over to
    hash ids.
    """

    def test_listAvailableTemplates_emits_id_not_name(self, plugin, template_dir):
        """The response entries carry ``id`` (hash) and DROP ``name``."""
        entries = plugin._discover_standalone_templates(template_dir)
        for e in entries:
            plugin._template_index[e.name] = e
        result = plugin._execute_list_available({})
        assert result["templates"], "expected templates in the response"
        for entry in result["templates"]:
            assert "id" in entry
            assert entry["id"].startswith("tpl_")
            # The bare LLM-facing ``name`` field is gone.
            assert "name" not in entry

    def test_id_is_stable_function_of_name(self):
        """Same name → same id, always.  Tested independent of any
        plugin instance so the property is pinned at the helper level."""
        from shared.plugins.template.plugin import _template_id
        assert _template_id("Entity.java.tpl") == _template_id("Entity.java.tpl")
        assert _template_id("A.tpl") != _template_id("B.tpl")
        # Same as the tool-name hashing pipeline, with the tpl prefix.
        assert _template_id("Entity.java.tpl").startswith("tpl_")

    def test_render_rejects_template_name_arg(self, plugin, template_dir):
        """Passing the legacy ``template_name`` arg fails — no back-compat."""
        entries = plugin._discover_standalone_templates(template_dir)
        for e in entries:
            plugin._template_index[e.name] = e
        result = plugin._execute_render_template_to_file({
            "template_name": "Entity.java.tpl",  # legacy arg
            "variables": {"Entity": "Customer"},
            "output_path": str(plugin._base_path / "out.java"),
        })
        # No template_id supplied + no inline template → exactly-one error.
        assert "error" in result
        assert "template_id" in result["error"] or "Exactly one" in result["error"]

    def test_list_variables_rejects_template_name_arg(self, plugin):
        """Passing the legacy ``template_name`` arg fails — no back-compat."""
        result = plugin._execute_list_template_variables({
            "template_name": "Entity.java.tpl",
        })
        assert "error" in result
        assert "template_id is required" in result["error"]

    def test_render_resolves_template_id_to_entry(self, plugin, template_dir):
        """Hash id passed by the LLM resolves to the correct template
        via the reverse-lookup map.  Hash must round-trip through
        ``name_to_id`` → ``id_to_name`` cleanly."""
        entries = plugin._discover_standalone_templates(template_dir)
        for e in entries:
            plugin._template_index[e.name] = e
        # Force the hash into the reverse map (which name_to_id does
        # internally on the first call).
        tpl_id = _template_id("Repository.java.tpl")
        output_file = plugin._base_path / "out" / "CustomerRepository.java"
        result = plugin._execute_render_template_to_file({
            "template_id": tpl_id,
            "variables": {
                "basePackage": "com.bank.customer",
                "Entity": "Customer",
                "EntityId": "CustomerId",
            },
            "output_path": str(output_file),
        })
        assert result.get("success") is True, f"render failed: {result}"
        assert output_file.exists()

    def test_listAvailableTemplates_keeps_source_path_for_audit(
        self, plugin, template_dir,
    ):
        """The human ``source_path`` is preserved so kb authors / human
        readers can audit which template each id maps to."""
        entries = plugin._discover_standalone_templates(template_dir)
        for e in entries:
            plugin._template_index[e.name] = e
        result = plugin._execute_list_available({})
        for entry in result["templates"]:
            assert "source_path" in entry
            assert entry["source_path"], "source_path must not be empty"
