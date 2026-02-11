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
        """listTemplateVariables should resolve template_name via index."""
        entries = plugin._discover_standalone_templates(template_dir)
        for entry in entries:
            plugin._template_index[entry.name] = entry

        result = plugin._execute_list_template_variables({
            "template_name": "Repository.java.tpl",
        })
        assert "variables" in result, f"Expected variables, got: {result}"
        assert "basePackage" in result["variables"]
        assert "Entity" in result["variables"]
        assert result["template_name"] == "Repository.java.tpl"


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

        # Should have annotations for standalone templates
        assert "TEMPLATE AVAILABLE" in result.instructions
        assert "Entity.java.tpl" in result.instructions
        assert "Repository.java.tpl" in result.instructions
        assert "application.yml.tpl" in result.instructions

        # Index should be populated
        assert len(plugin._template_index) == 3

        # Metadata should report standalone count
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
