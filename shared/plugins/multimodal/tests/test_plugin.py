"""Tests for the multimodal plugin."""

import os
import tempfile
from pathlib import Path
import pytest

from ..plugin import MultimodalPlugin, create_plugin, IMAGE_EXTENSIONS


class TestMultimodalPluginInitialization:
    """Tests for plugin initialization."""

    def test_create_plugin_factory(self):
        plugin = create_plugin()
        assert isinstance(plugin, MultimodalPlugin)

    def test_plugin_name(self):
        plugin = MultimodalPlugin()
        assert plugin.name == "multimodal"

    def test_initialize_without_config(self):
        plugin = MultimodalPlugin()
        plugin.initialize()
        assert plugin._initialized is True
        assert plugin._base_path == Path.cwd()
        assert plugin._max_image_size_mb == 10.0

    def test_initialize_with_base_path(self):
        plugin = MultimodalPlugin()
        plugin.initialize({"base_path": "/custom/path"})
        assert plugin._initialized is True
        assert plugin._base_path == Path("/custom/path")

    def test_initialize_with_max_size(self):
        plugin = MultimodalPlugin()
        plugin.initialize({"max_image_size_mb": 5.0})
        assert plugin._max_image_size_mb == 5.0

    def test_shutdown(self):
        plugin = MultimodalPlugin()
        plugin.initialize()
        plugin._detected_images["test.png"] = Path("/test.png")
        plugin.shutdown()

        assert plugin._initialized is False
        assert plugin._detected_images == {}


class TestMultimodalPluginModelRequirements:
    """Tests for model requirements."""

    def test_get_model_requirements(self):
        plugin = MultimodalPlugin()
        requirements = plugin.get_model_requirements()

        assert requirements is not None
        assert isinstance(requirements, list)
        assert len(requirements) > 0
        # Should require Gemini 3+
        assert any("gemini-3" in r for r in requirements)

    def test_model_requirements_patterns(self):
        plugin = MultimodalPlugin()
        requirements = plugin.get_model_requirements()

        # Check specific patterns
        assert "gemini-3-pro*" in requirements
        assert "gemini-3.5-*" in requirements


class TestMultimodalPluginPromptEnrichment:
    """Tests for prompt enrichment."""

    def test_subscribes_to_prompt_enrichment(self):
        plugin = MultimodalPlugin()
        assert plugin.subscribes_to_prompt_enrichment() is True

    def test_enrich_prompt_no_references(self):
        plugin = MultimodalPlugin()
        plugin.initialize()

        result = plugin.enrich_prompt("Hello, how are you?")

        assert result.prompt == "Hello, how are you?"
        assert result.metadata == {}

    def test_enrich_prompt_with_non_image_reference(self):
        plugin = MultimodalPlugin()
        plugin.initialize()

        result = plugin.enrich_prompt("Check @script.py for errors")

        # Non-image file should not trigger enrichment
        assert result.prompt == "Check @script.py for errors"
        assert result.metadata == {}

    def test_enrich_prompt_with_nonexistent_image(self):
        plugin = MultimodalPlugin()
        plugin.initialize()

        result = plugin.enrich_prompt("What's in @nonexistent.png?")

        # File doesn't exist, should not trigger enrichment
        assert result.prompt == "What's in @nonexistent.png?"
        assert result.metadata == {}

    def test_enrich_prompt_with_existing_image(self):
        """Test enrichment with an existing image file."""
        plugin = MultimodalPlugin()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test image file
            img_path = Path(tmpdir) / "test.png"
            img_path.write_bytes(b"fake png data")

            plugin.initialize({"base_path": tmpdir})

            result = plugin.enrich_prompt("What's in @test.png?")

            # Should have enriched the prompt
            assert "viewImage" in result.prompt
            assert "test.png" in result.prompt
            assert result.metadata.get("detected_images") is not None
            assert "test.png" in result.metadata["detected_images"]

    def test_enrich_prompt_preserves_at_reference(self):
        """Verify plugin does NOT remove @ from references."""
        plugin = MultimodalPlugin()

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "photo.jpg"
            img_path.write_bytes(b"fake jpg data")

            plugin.initialize({"base_path": tmpdir})

            result = plugin.enrich_prompt("Look at @photo.jpg please")

            # The @photo.jpg should still be in the prompt
            # (framework removes it later)
            assert "@photo.jpg" in result.prompt

    def test_enrich_prompt_multiple_images(self):
        """Test enrichment with multiple image references."""
        plugin = MultimodalPlugin()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple test images
            (Path(tmpdir) / "a.png").write_bytes(b"png1")
            (Path(tmpdir) / "b.jpg").write_bytes(b"jpg1")

            plugin.initialize({"base_path": tmpdir})

            result = plugin.enrich_prompt("Compare @a.png with @b.jpg")

            assert result.metadata.get("image_count") == 2
            assert "a.png" in result.metadata["detected_images"]
            assert "b.jpg" in result.metadata["detected_images"]


class TestMultimodalPluginFunctionDeclarations:
    """Tests for function declarations."""

    def test_get_function_declarations(self):
        plugin = MultimodalPlugin()
        declarations = plugin.get_function_declarations()

        assert len(declarations) == 1
        assert declarations[0].name == "viewImage"

    def test_view_image_schema(self):
        plugin = MultimodalPlugin()
        declarations = plugin.get_function_declarations()
        view_image = declarations[0]
        schema = view_image.parameters_json_schema

        assert schema["type"] == "object"
        assert "path" in schema["properties"]
        assert "path" in schema["required"]

    def test_view_image_description(self):
        plugin = MultimodalPlugin()
        declarations = plugin.get_function_declarations()
        view_image = declarations[0]

        assert "View" in view_image.description
        assert "image" in view_image.description.lower()


class TestMultimodalPluginExecutors:
    """Tests for executor mapping."""

    def test_get_executors(self):
        plugin = MultimodalPlugin()
        executors = plugin.get_executors()

        assert "viewImage" in executors
        assert callable(executors["viewImage"])


class TestMultimodalPluginExecution:
    """Tests for viewImage execution."""

    def test_execute_missing_path(self):
        plugin = MultimodalPlugin()
        plugin.initialize()

        result = plugin._execute_view_image({})

        assert "error" in result
        assert "path is required" in result["error"]

    def test_execute_nonexistent_file(self):
        plugin = MultimodalPlugin()
        plugin.initialize()

        result = plugin._execute_view_image({"path": "/nonexistent/image.png"})

        assert "error" in result
        assert "not found" in result["error"]

    def test_execute_existing_image(self):
        """Test successful image loading."""
        plugin = MultimodalPlugin()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test image file
            img_path = Path(tmpdir) / "test.png"
            img_data = b"\x89PNG\r\n\x1a\n" + b"fake png content"
            img_path.write_bytes(img_data)

            plugin.initialize({"base_path": tmpdir})

            result = plugin._execute_view_image({"path": "test.png"})

            assert result.get("_multimodal") is True
            assert result.get("_multimodal_type") == "image"
            assert result.get("image_data") == img_data
            assert result.get("mime_type") == "image/png"
            assert result.get("display_name") == "test.png"
            assert result.get("size_bytes") == len(img_data)

    def test_execute_file_too_large(self):
        """Test rejection of files exceeding size limit."""
        plugin = MultimodalPlugin()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a large file (exceeds 10MB default limit)
            img_path = Path(tmpdir) / "large.png"
            img_path.write_bytes(b"x" * (11 * 1024 * 1024))  # 11 MB

            plugin.initialize({"base_path": tmpdir})

            result = plugin._execute_view_image({"path": "large.png"})

            assert "error" in result
            assert "too large" in result["error"]

    def test_execute_with_custom_size_limit(self):
        """Test custom size limit configuration."""
        plugin = MultimodalPlugin()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a 2MB file
            img_path = Path(tmpdir) / "medium.png"
            img_path.write_bytes(b"x" * (2 * 1024 * 1024))

            # Set 1MB limit - should reject
            plugin.initialize({"base_path": tmpdir, "max_image_size_mb": 1.0})

            result = plugin._execute_view_image({"path": "medium.png"})

            assert "error" in result
            assert "too large" in result["error"]


class TestMultimodalPluginAutoApproval:
    """Tests for auto-approval settings."""

    def test_get_auto_approved_tools(self):
        plugin = MultimodalPlugin()
        auto_approved = plugin.get_auto_approved_tools()

        # viewImage only reads files, should be auto-approved
        assert "viewImage" in auto_approved


class TestMultimodalPluginSystemInstructions:
    """Tests for system instructions."""

    def test_get_system_instructions(self):
        plugin = MultimodalPlugin()
        instructions = plugin.get_system_instructions()

        # Should return None as instructions are added dynamically
        assert instructions is None


class TestImageExtensions:
    """Tests for recognized image extensions."""

    def test_common_extensions(self):
        assert ".png" in IMAGE_EXTENSIONS
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS
        assert ".gif" in IMAGE_EXTENSIONS
        assert ".webp" in IMAGE_EXTENSIONS

    def test_case_handling(self):
        # Extensions should be lowercase
        for ext in IMAGE_EXTENSIONS:
            assert ext == ext.lower()
            assert ext.startswith(".")
