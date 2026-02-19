# shared/plugins/formatter_pipeline/tests/test_registry.py
"""Tests for FormatterRegistry."""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from tempfile import NamedTemporaryFile

from ..registry import (
    FormatterRegistry,
    create_registry,
    DEFAULT_FORMATTERS,
    KNOWN_FORMATTERS,
)


class MockFormatter:
    """Mock formatter for testing."""

    def __init__(self, name="mock", priority=50):
        self._name = name
        self._priority = priority
        self._initialized = False
        self._config = {}

    @property
    def name(self):
        return self._name

    @property
    def priority(self):
        return self._priority

    def initialize(self, config):
        self._initialized = True
        self._config = config

    def process_chunk(self, chunk):
        yield chunk

    def flush(self):
        return iter([])

    def reset(self):
        pass


class MockFormatterWithWiring:
    """Mock formatter that requires wiring."""

    def __init__(self):
        self._name = "mock_wired"
        self._priority = 30
        self._wired = False
        self._lsp_plugin = None

    @property
    def name(self):
        return self._name

    @property
    def priority(self):
        return self._priority

    def wire_dependencies(self, tool_registry):
        lsp = tool_registry.get_plugin("lsp") if tool_registry else None
        if lsp:
            self._lsp_plugin = lsp
            self._wired = True
            return True
        return False

    def initialize(self, config):
        pass

    def process_chunk(self, chunk):
        yield chunk

    def flush(self):
        return iter([])

    def reset(self):
        pass


class TestFormatterRegistry:
    """Tests for FormatterRegistry."""

    def test_create_registry(self):
        """Should create a registry instance."""
        registry = create_registry()
        assert isinstance(registry, FormatterRegistry)

    def test_discover_known_formatters(self):
        """Should discover formatters that can be imported."""
        registry = create_registry()
        discovered = registry.discover()

        # Should find at least some formatters
        assert len(discovered) > 0

        # All discovered should be in KNOWN_FORMATTERS
        for name in discovered:
            assert name in KNOWN_FORMATTERS

    def test_list_available(self):
        """Should list discovered and custom formatters."""
        registry = create_registry()
        registry.discover()

        # Add a custom formatter
        mock = MockFormatter()
        registry.register_custom("custom", mock)

        available = registry.list_available()

        assert "custom" in available

    def test_use_defaults(self):
        """Should set default configuration."""
        registry = create_registry()
        registry.use_defaults()

        assert registry._config == DEFAULT_FORMATTERS

    def test_load_config_from_file(self, tmp_path):
        """Should load configuration from JSON file."""
        config = {
            "formatters": [
                {"name": "test_formatter", "enabled": True},
            ]
        }

        config_file = tmp_path / "formatters.json"
        config_file.write_text(json.dumps(config))

        registry = create_registry()
        result = registry.load_config(str(config_file))

        assert result is True
        assert registry._config == config["formatters"]

    def test_load_config_missing_file(self):
        """Should return False for missing config file."""
        registry = create_registry()
        result = registry.load_config("/nonexistent/path.json")

        assert result is False

    def test_load_config_from_dict(self):
        """Should load configuration from dictionary."""
        config = {
            "formatters": [
                {"name": "test", "enabled": True},
            ]
        }

        registry = create_registry()
        registry.load_config_from_dict(config)

        assert registry._config == config["formatters"]

    def test_register_custom_formatter(self):
        """Should register custom formatter instance."""
        registry = create_registry()
        mock = MockFormatter()

        registry.register_custom("my_formatter", mock)

        assert "my_formatter" in registry._custom_formatters
        assert registry._custom_formatters["my_formatter"] is mock

    def test_set_tool_registry(self):
        """Should store tool registry for wiring."""
        registry = create_registry()
        mock_tool_registry = Mock()

        registry.set_tool_registry(mock_tool_registry)

        assert registry._tool_registry is mock_tool_registry


class TestFormatterRegistryPipeline:
    """Tests for pipeline creation."""

    def test_create_pipeline_with_custom_formatter(self):
        """Should include custom formatters in pipeline."""
        registry = create_registry()
        mock = MockFormatter("custom", 25)
        registry.register_custom("custom", mock)

        # Use empty config so only custom formatter is added
        registry._config = []

        pipeline = registry.create_pipeline()

        formatters = pipeline.list_formatters()
        assert "custom" in formatters

    def test_create_pipeline_respects_enabled_false(self):
        """Should skip formatters with enabled: false."""
        registry = create_registry()

        mock_enabled = MockFormatter("enabled", 10)
        mock_disabled = MockFormatter("disabled", 20)

        registry.register_custom("enabled", mock_enabled)
        registry.register_custom("disabled", mock_disabled)

        registry._config = [
            {"name": "enabled", "enabled": True},
            {"name": "disabled", "enabled": False},
        ]

        pipeline = registry.create_pipeline()
        formatters = pipeline.list_formatters()

        assert "enabled" in formatters
        assert "disabled" not in formatters

    def test_create_pipeline_applies_priority_override(self):
        """Should apply priority override from config."""
        registry = create_registry()

        mock = MockFormatter("test", 50)  # Default priority 50
        registry.register_custom("test", mock)

        registry._config = [
            {"name": "test", "priority": 10},  # Override to 10
        ]

        pipeline = registry.create_pipeline()

        # Check the formatter's priority was updated
        assert mock._priority == 10

    def test_create_pipeline_passes_config_to_formatter(self):
        """Should pass config to formatter's initialize."""
        registry = create_registry()

        mock = MockFormatter("test", 50)
        registry.register_custom("test", mock)

        registry._config = [
            {"name": "test", "config": {"option1": "value1"}},
        ]

        pipeline = registry.create_pipeline()

        assert mock._initialized is True
        assert mock._config == {"option1": "value1"}


class TestFormatterRegistryWiring:
    """Tests for formatter dependency wiring."""

    def test_formatter_wiring_with_tool_registry(self):
        """Should wire formatter with tool registry."""
        registry = create_registry()

        # Mock tool registry with LSP plugin
        mock_tool_registry = Mock()
        mock_lsp = Mock()
        mock_tool_registry.get_plugin.return_value = mock_lsp

        registry.set_tool_registry(mock_tool_registry)

        # Create formatter that needs wiring
        formatter = MockFormatterWithWiring()
        registry.register_custom("wired", formatter)
        registry._config = [{"name": "wired"}]

        pipeline = registry.create_pipeline()

        # Formatter should be wired
        assert formatter._wired is True
        assert formatter._lsp_plugin is mock_lsp

    def test_formatter_skipped_when_wiring_fails(self):
        """Should skip formatter when wiring returns False."""
        registry = create_registry()

        # Mock tool registry WITHOUT LSP plugin
        mock_tool_registry = Mock()
        mock_tool_registry.get_plugin.return_value = None

        registry.set_tool_registry(mock_tool_registry)

        # Create formatter that needs wiring (will fail)
        formatter = MockFormatterWithWiring()
        registry.register_custom("wired", formatter)
        registry._config = [{"name": "wired"}]

        pipeline = registry.create_pipeline()

        # Formatter should NOT be in pipeline
        assert "wired" not in pipeline.list_formatters()

    def test_formatter_skipped_when_no_tool_registry(self):
        """Should skip formatter needing wiring when no tool registry."""
        registry = create_registry()

        # No tool registry set
        formatter = MockFormatterWithWiring()
        registry.register_custom("wired", formatter)
        registry._config = [{"name": "wired"}]

        pipeline = registry.create_pipeline()

        # Formatter should NOT be in pipeline
        assert "wired" not in pipeline.list_formatters()
