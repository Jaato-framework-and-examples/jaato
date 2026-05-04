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
)


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
            assert "name" in t
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
            "template_name": "Repository.java.tpl",
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
            "template_name": "NonExistent.java.tpl",
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
            "template_name": "Repository.java.tpl",
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
        assert api["item_keys"] == ["methodName", "path"]

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
        assert items["item_keys"] == ["name"]
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
        assert "isVoid" in api["item_keys"]
        assert "methodName" in api["item_keys"]
        assert "returnType" in api["item_keys"]

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
        for required in ("controllerSignature", "returnType", "serviceCallArgs", "isVoid"):
            assert required in api["item_keys"], (
                f"{required} missing from apiEndpoints.item_keys: {api['item_keys']}"
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
        assert "fieldName" in fields["item_keys"]
        assert "type" in fields["item_keys"]

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
            "template_name": "test.java.tpl",
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
        assert "Entity" in fields["item_keys"]
        assert "name" in fields["item_keys"]

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
        assert rows["item_keys"] == ["cell"]

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
            "template_name": "Repository.java.tpl",
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

    def test_missing_top_level_variable_skipped_by_validator(self, plugin):
        """Variables NOT provided are skipped by the SHAPE validator
        (the absent-variable check is a separate layer inside
        ``_render_template``).  This test isolates the validator: when
        a key is absent, the shape check returns ``None`` and the error
        — if any — comes from the render layer, not from this layer.
        """
        template = "Hello {{name}}, age {{age}}"
        result = plugin._validate_render_inputs_against_structure(
            template, {"name": "Alice"}
        )
        # Validator must NOT object to absent 'age'; that's the
        # render layer's concern.
        assert result is None

    def test_jinja2_path_skipped(self, plugin):
        """Jinja2 templates skip shape validation (the validator is
        Mustache-only; Jinja2 kind detection is a follow-up)."""
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
                "template_name": name,
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
            "template_name": "Entity.java.tpl",
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
            "template_name": "X.java.tpl",
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
            "template_name": "NoDir.java.tpl",
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
        by_name = {t["name"]: t for t in result["templates"]}
        assert by_name["Has.tpl"]["output_path_template"] == (
            "declared/{{x}}.java"
        )
        assert by_name["Lacks.tpl"]["output_path_template"] == ""

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
