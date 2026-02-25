"""Tests for ReferenceContents model and reference-context enrichment.

Tests the contents property on ReferenceSource, the reference-context
annotations in enrich_tool_result, and the
file_belongs_to_reference_with_templates public API.
"""

import os
import tempfile
from pathlib import Path

import pytest

from ..models import ReferenceSource, ReferenceContents, SourceType, InjectionMode
from ..plugin import create_plugin
from shared.path_utils import normalize_for_comparison


class TestReferenceContents:
    """Tests for the ReferenceContents dataclass."""

    def test_default_all_none(self):
        contents = ReferenceContents()
        assert contents.templates is None
        assert contents.validation is None
        assert contents.policies is None
        assert contents.scripts is None
        assert contents.has_any() is False

    def test_has_any_with_templates(self):
        contents = ReferenceContents(templates="templates/")
        assert contents.has_any() is True

    def test_has_any_with_all(self):
        contents = ReferenceContents(
            templates="templates/",
            validation="validation/",
            policies="policies/",
            scripts="scripts/",
        )
        assert contents.has_any() is True

    def test_to_dict(self):
        contents = ReferenceContents(templates="templates/", validation="validation/")
        d = contents.to_dict()
        assert d == {
            "templates": "templates/",
            "validation": "validation/",
            "policies": None,
            "scripts": None,
        }

    def test_from_dict_none(self):
        contents = ReferenceContents.from_dict(None)
        assert contents.has_any() is False

    def test_from_dict_empty(self):
        contents = ReferenceContents.from_dict({})
        assert contents.has_any() is False

    def test_from_dict_partial(self):
        contents = ReferenceContents.from_dict({"templates": "tpl/"})
        assert contents.templates == "tpl/"
        assert contents.validation is None

    def test_from_dict_ignores_extra_keys(self):
        contents = ReferenceContents.from_dict({"templates": "t/", "unknown": "x"})
        assert contents.templates == "t/"


class TestReferenceSourceContents:
    """Tests for contents field on ReferenceSource."""

    def test_default_contents(self):
        source = ReferenceSource(
            id="test", name="Test", description="Test ref",
            type=SourceType.LOCAL, mode=InjectionMode.SELECTABLE,
            path="/tmp/test",
        )
        assert source.contents.has_any() is False

    def test_to_dict_includes_contents_when_present(self):
        source = ReferenceSource(
            id="test", name="Test", description="Test ref",
            type=SourceType.LOCAL, mode=InjectionMode.SELECTABLE,
            path="/tmp/test",
            contents=ReferenceContents(templates="templates/"),
        )
        d = source.to_dict()
        assert "contents" in d
        assert d["contents"]["templates"] == "templates/"

    def test_to_dict_omits_contents_when_empty(self):
        source = ReferenceSource(
            id="test", name="Test", description="Test ref",
            type=SourceType.LOCAL, mode=InjectionMode.SELECTABLE,
            path="/tmp/test",
        )
        d = source.to_dict()
        assert "contents" not in d

    def test_from_dict_with_contents(self):
        data = {
            "id": "mod-001",
            "name": "Module 001",
            "type": "local",
            "path": "/tmp/mod-001",
            "contents": {
                "templates": "templates/",
                "validation": "validation/",
                "policies": None,
                "scripts": None,
            },
        }
        source = ReferenceSource.from_dict(data)
        assert source.contents.templates == "templates/"
        assert source.contents.validation == "validation/"
        assert source.contents.policies is None

    def test_from_dict_without_contents(self):
        data = {
            "id": "mod-002",
            "name": "Module 002",
            "type": "local",
            "path": "/tmp/mod-002",
        }
        source = ReferenceSource.from_dict(data)
        assert source.contents.has_any() is False


class TestFileBelongsToReferenceWithTemplates:
    """Tests for file_belongs_to_reference_with_templates."""

    def test_file_inside_reference_with_templates(self):
        """File inside a reference that has contents.templates returns True."""
        plugin = create_plugin()

        source = ReferenceSource(
            id="mod-001", name="Module 001", description="Test",
            type=SourceType.LOCAL, mode=InjectionMode.SELECTABLE,
            path="/tmp/mod-001",
            resolved_path="/tmp/mod-001",
            contents=ReferenceContents(templates="templates/"),
        )
        plugin._sources = [source]
        plugin._preselected_paths = {
            normalize_for_comparison("/tmp/mod-001"): ("mod-001", "Module 001")
        }

        assert plugin.file_belongs_to_reference_with_templates(
            "/tmp/mod-001/MODULE.md"
        ) is True

    def test_file_inside_reference_without_templates(self):
        """File inside a reference without contents.templates returns False."""
        plugin = create_plugin()

        source = ReferenceSource(
            id="adr-001", name="ADR 001", description="Test",
            type=SourceType.LOCAL, mode=InjectionMode.SELECTABLE,
            path="/tmp/adr-001",
            resolved_path="/tmp/adr-001",
        )
        plugin._sources = [source]
        plugin._preselected_paths = {
            normalize_for_comparison("/tmp/adr-001"): ("adr-001", "ADR 001")
        }

        assert plugin.file_belongs_to_reference_with_templates(
            "/tmp/adr-001/ADR.md"
        ) is False

    def test_file_not_in_any_reference(self):
        """File not inside any reference returns False."""
        plugin = create_plugin()
        plugin._preselected_paths = {
            normalize_for_comparison("/tmp/mod-001"): ("mod-001", "Module 001")
        }
        plugin._sources = [
            ReferenceSource(
                id="mod-001", name="Module 001", description="Test",
                type=SourceType.LOCAL, mode=InjectionMode.SELECTABLE,
                resolved_path="/tmp/mod-001",
                contents=ReferenceContents(templates="templates/"),
            )
        ]

        assert plugin.file_belongs_to_reference_with_templates(
            "/tmp/other/file.md"
        ) is False

    def test_empty_preselected_paths(self):
        """No preselected paths returns False."""
        plugin = create_plugin()
        plugin._preselected_paths = {}

        assert plugin.file_belongs_to_reference_with_templates(
            "/tmp/any/file.md"
        ) is False


class TestIsRootMarkdownRead:
    """Tests for _is_root_markdown_read method."""

    def test_root_markdown_detected(self):
        """A .md file directly in the reference root is detected."""
        plugin = create_plugin()
        plugin._sources = [
            ReferenceSource(
                id="mod-001", name="Module 001", description="Test",
                type=SourceType.LOCAL, mode=InjectionMode.SELECTABLE,
                resolved_path="/tmp/mod-001",
            )
        ]

        assert plugin._is_root_markdown_read(
            "mod-001", {"path": "/tmp/mod-001/MODULE.md"}
        ) is True

    def test_nested_markdown_not_detected(self):
        """A .md file inside a subfolder is NOT a root markdown."""
        plugin = create_plugin()
        plugin._sources = [
            ReferenceSource(
                id="mod-001", name="Module 001", description="Test",
                type=SourceType.LOCAL, mode=InjectionMode.SELECTABLE,
                resolved_path="/tmp/mod-001",
            )
        ]

        assert plugin._is_root_markdown_read(
            "mod-001", {"path": "/tmp/mod-001/templates/README.md"}
        ) is False

    def test_non_markdown_not_detected(self):
        """A non-markdown file in the root is not detected."""
        plugin = create_plugin()
        plugin._sources = [
            ReferenceSource(
                id="mod-001", name="Module 001", description="Test",
                type=SourceType.LOCAL, mode=InjectionMode.SELECTABLE,
                resolved_path="/tmp/mod-001",
            )
        ]

        assert plugin._is_root_markdown_read(
            "mod-001", {"path": "/tmp/mod-001/config.json"}
        ) is False


class TestBuildContentsAnnotation:
    """Tests for _build_contents_annotation method."""

    def test_no_annotation_when_no_contents(self):
        """No annotation when reference has no contents."""
        plugin = create_plugin()
        plugin._sources = [
            ReferenceSource(
                id="adr-001", name="ADR 001", description="Test",
                type=SourceType.LOCAL, mode=InjectionMode.SELECTABLE,
                resolved_path="/tmp/adr-001",
            )
        ]

        result = plugin._build_contents_annotation(
            "adr-001", {"path": "/tmp/adr-001/ADR.md"}
        )
        assert result is None

    def test_no_annotation_when_not_root_markdown(self):
        """No annotation when file is not a root markdown."""
        plugin = create_plugin()
        plugin._sources = [
            ReferenceSource(
                id="mod-001", name="Module 001", description="Test",
                type=SourceType.LOCAL, mode=InjectionMode.SELECTABLE,
                resolved_path="/tmp/mod-001",
                contents=ReferenceContents(templates="templates/"),
            )
        ]

        result = plugin._build_contents_annotation(
            "mod-001", {"path": "/tmp/mod-001/templates/Application.java.tpl"}
        )
        assert result is None

    def test_annotation_with_templates(self, tmp_path):
        """Annotation includes template IDs when templates subfolder exists."""
        # Create a temporary templates subfolder with .tpl files
        mod_dir = tmp_path / "mod-001"
        mod_dir.mkdir()
        tpl_dir = mod_dir / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "Application.java.tpl").write_text("{{name}}")
        (tpl_dir / "Entity.java.tpl").write_text("{{entity}}")
        (mod_dir / "MODULE.md").write_text("# Module")

        plugin = create_plugin()
        plugin._sources = [
            ReferenceSource(
                id="mod-001", name="Module 001", description="Test",
                type=SourceType.LOCAL, mode=InjectionMode.SELECTABLE,
                resolved_path=str(mod_dir),
                contents=ReferenceContents(templates="templates/"),
            )
        ]

        result = plugin._build_contents_annotation(
            "mod-001", {"path": str(mod_dir / "MODULE.md")}
        )
        assert result is not None
        assert "Mandatory Templates" in result
        assert "renderTemplateToFile" in result
        assert "Application.java.tpl" in result
        assert "Entity.java.tpl" in result

    def test_annotation_with_validation(self, tmp_path):
        """Annotation includes validation scripts."""
        mod_dir = tmp_path / "mod-001"
        mod_dir.mkdir()
        val_dir = mod_dir / "validation"
        val_dir.mkdir()
        (val_dir / "check.sh").write_text("#!/bin/bash")
        (mod_dir / "MODULE.md").write_text("# Module")

        plugin = create_plugin()
        plugin._sources = [
            ReferenceSource(
                id="mod-001", name="Module 001", description="Test",
                type=SourceType.LOCAL, mode=InjectionMode.SELECTABLE,
                resolved_path=str(mod_dir),
                contents=ReferenceContents(validation="validation/"),
            )
        ]

        result = plugin._build_contents_annotation(
            "mod-001", {"path": str(mod_dir / "MODULE.md")}
        )
        assert result is not None
        assert "Post-Implementation Validation" in result
        assert "MUST run" in result
        assert "check.sh" in result

    def test_annotation_with_policies(self, tmp_path):
        """Annotation includes policy files."""
        mod_dir = tmp_path / "mod-001"
        mod_dir.mkdir()
        pol_dir = mod_dir / "policies"
        pol_dir.mkdir()
        (pol_dir / "naming.md").write_text("# Naming conventions")
        (mod_dir / "MODULE.md").write_text("# Module")

        plugin = create_plugin()
        plugin._sources = [
            ReferenceSource(
                id="mod-001", name="Module 001", description="Test",
                type=SourceType.LOCAL, mode=InjectionMode.SELECTABLE,
                resolved_path=str(mod_dir),
                contents=ReferenceContents(policies="policies/"),
            )
        ]

        result = plugin._build_contents_annotation(
            "mod-001", {"path": str(mod_dir / "MODULE.md")}
        )
        assert result is not None
        assert "Implementation Policies" in result
        assert "naming.md" in result

    def test_annotation_with_scripts(self, tmp_path):
        """Annotation includes helper scripts."""
        mod_dir = tmp_path / "mod-001"
        mod_dir.mkdir()
        scr_dir = mod_dir / "scripts"
        scr_dir.mkdir()
        (scr_dir / "generate-dto.sh").write_text("#!/bin/bash")
        (mod_dir / "MODULE.md").write_text("# Module")

        plugin = create_plugin()
        plugin._sources = [
            ReferenceSource(
                id="mod-001", name="Module 001", description="Test",
                type=SourceType.LOCAL, mode=InjectionMode.SELECTABLE,
                resolved_path=str(mod_dir),
                contents=ReferenceContents(scripts="scripts/"),
            )
        ]

        result = plugin._build_contents_annotation(
            "mod-001", {"path": str(mod_dir / "MODULE.md")}
        )
        assert result is not None
        assert "Helper Scripts" in result
        assert "generate-dto.sh" in result

    def test_annotation_all_content_types(self, tmp_path):
        """Annotation includes all four content types."""
        mod_dir = tmp_path / "mod-001"
        mod_dir.mkdir()
        (mod_dir / "templates").mkdir()
        (mod_dir / "templates" / "App.java.tpl").write_text("{{name}}")
        (mod_dir / "validation").mkdir()
        (mod_dir / "validation" / "check.sh").write_text("#!/bin/bash")
        (mod_dir / "policies").mkdir()
        (mod_dir / "policies" / "rules.md").write_text("# Rules")
        (mod_dir / "scripts").mkdir()
        (mod_dir / "scripts" / "helper.sh").write_text("#!/bin/bash")
        (mod_dir / "MODULE.md").write_text("# Module")

        plugin = create_plugin()
        plugin._sources = [
            ReferenceSource(
                id="mod-001", name="Module 001", description="Test",
                type=SourceType.LOCAL, mode=InjectionMode.SELECTABLE,
                resolved_path=str(mod_dir),
                contents=ReferenceContents(
                    templates="templates/",
                    validation="validation/",
                    policies="policies/",
                    scripts="scripts/",
                ),
            )
        ]

        result = plugin._build_contents_annotation(
            "mod-001", {"path": str(mod_dir / "MODULE.md")}
        )
        assert result is not None
        assert "Mandatory Templates" in result
        assert "Post-Implementation Validation" in result
        assert "Implementation Policies" in result
        assert "Helper Scripts" in result
