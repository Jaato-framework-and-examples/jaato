"""Tests for the mermaid formatter plugin."""

import os
import pytest
from unittest.mock import patch, MagicMock

from ..plugin import MermaidFormatterPlugin, create_plugin


class TestPluginProperties:
    """Tests for plugin metadata."""

    def test_name(self):
        plugin = MermaidFormatterPlugin()
        assert plugin.name == "mermaid_formatter"

    def test_default_priority(self):
        plugin = MermaidFormatterPlugin()
        assert plugin.priority == 28

    def test_priority_after_diff_before_code_block(self):
        plugin = MermaidFormatterPlugin()
        assert 20 < plugin.priority < 40

    def test_create_plugin_factory(self):
        plugin = create_plugin()
        assert isinstance(plugin, MermaidFormatterPlugin)


class TestInitialization:
    """Tests for plugin initialization and configuration."""

    def test_default_config(self):
        plugin = MermaidFormatterPlugin()
        plugin.initialize()
        assert plugin._theme == "default"
        assert plugin._scale == 2
        assert plugin._enabled is True

    def test_custom_config(self):
        plugin = MermaidFormatterPlugin()
        plugin.initialize({
            "theme": "dark",
            "scale": 3,
            "priority": 30,
            "background": "transparent",
        })
        assert plugin._theme == "dark"
        assert plugin._scale == 3
        assert plugin._priority == 30
        assert plugin._background == "transparent"

    def test_env_var_theme(self, monkeypatch):
        monkeypatch.setenv("JAATO_MERMAID_THEME", "forest")
        plugin = MermaidFormatterPlugin()
        plugin.initialize()
        assert plugin._theme == "forest"

    def test_env_var_scale(self, monkeypatch):
        monkeypatch.setenv("JAATO_MERMAID_SCALE", "4")
        plugin = MermaidFormatterPlugin()
        plugin.initialize()
        assert plugin._scale == 4

    def test_env_var_scale_invalid(self, monkeypatch):
        monkeypatch.setenv("JAATO_MERMAID_SCALE", "not_a_number")
        plugin = MermaidFormatterPlugin()
        plugin.initialize()
        assert plugin._scale == 2  # Default preserved

    def test_env_var_backend_off(self, monkeypatch):
        monkeypatch.setenv("JAATO_MERMAID_BACKEND", "off")
        plugin = MermaidFormatterPlugin()
        plugin.initialize()
        assert plugin._enabled is False

    def test_set_console_width(self):
        plugin = MermaidFormatterPlugin()
        plugin.set_console_width(120)
        assert plugin._console_width == 120

    def test_set_console_width_minimum(self):
        plugin = MermaidFormatterPlugin()
        plugin.set_console_width(5)
        assert plugin._console_width == 20  # Clamped to minimum


class TestBlockDetection:
    """Tests for mermaid code block detection in streaming output."""

    def test_passthrough_regular_text(self):
        plugin = MermaidFormatterPlugin()
        result = list(plugin.process_chunk("Hello world"))
        assert result == ["Hello world"]

    def test_passthrough_non_mermaid_code_block(self):
        plugin = MermaidFormatterPlugin()
        result = list(plugin.process_chunk("```python\nprint('hi')\n```"))
        # Trailing ``` held back (could be start of ```mermaid), flush releases
        result.extend(plugin.flush())
        assert "".join(result) == "```python\nprint('hi')\n```"

    def test_detect_mermaid_block_single_chunk(self):
        plugin = MermaidFormatterPlugin()
        plugin._enabled = False  # Disable rendering, just test detection passthrough
        source = "```mermaid\ngraph TD\n    A-->B\n```"
        result = list(plugin.process_chunk(source))
        # When disabled, passes through unchanged
        assert "".join(result) == source

    @patch("shared.plugins.mermaid_formatter.plugin.renderer")
    def test_detect_mermaid_block_renders(self, mock_renderer):
        """When renderer returns None, falls back to code block."""
        mock_renderer.render.return_value = None
        plugin = MermaidFormatterPlugin()
        plugin.initialize()

        chunks = list(plugin.process_chunk("before\n```mermaid\ngraph TD\n    A-->B\n```\nafter"))
        output = "".join(chunks)

        # Should contain the fallback hint
        assert "rendering unavailable" in output
        # Should contain the source for code_block_formatter
        assert "graph TD" in output
        # Should contain surrounding text
        assert "before" in output
        assert "after" in output

    def test_streaming_detection_across_chunks(self):
        """Test mermaid block detection across multiple streamed chunks."""
        plugin = MermaidFormatterPlugin()
        plugin._enabled = False  # Just test passthrough

        all_output = []
        all_output.extend(plugin.process_chunk("Hello "))
        all_output.extend(plugin.process_chunk("```mer"))
        all_output.extend(plugin.process_chunk("maid\ngraph TD\n"))
        all_output.extend(plugin.process_chunk("    A-->B\n``"))
        all_output.extend(plugin.process_chunk("`\nDone"))
        all_output.extend(plugin.flush())

        output = "".join(all_output)
        assert "Hello " in output
        assert "Done" in output

    @patch("shared.plugins.mermaid_formatter.plugin.renderer")
    def test_text_before_mermaid_block_yielded_immediately(self, mock_renderer):
        mock_renderer.render.return_value = None
        plugin = MermaidFormatterPlugin()
        plugin.initialize()

        chunks = list(plugin.process_chunk("prefix text\n```mermaid\ngraph TD\n    A-->B\n```"))
        # First chunk should be the prefix text before the mermaid block
        assert chunks[0] == "prefix text\n"

    @patch("shared.plugins.mermaid_formatter.plugin.renderer")
    def test_multiple_mermaid_blocks(self, mock_renderer):
        """Test handling multiple mermaid blocks in sequence."""
        mock_renderer.render.return_value = None
        plugin = MermaidFormatterPlugin()
        plugin.initialize()

        source = "text1\n```mermaid\ngraph TD\n    A-->B\n```\nmiddle\n```mermaid\ngraph LR\n    C-->D\n```\nend"
        chunks = list(plugin.process_chunk(source))
        output = "".join(chunks)

        assert "text1" in output
        assert "middle" in output
        assert "end" in output
        assert "A-->B" in output
        assert "C-->D" in output


class TestFlushAndReset:
    """Tests for flush and reset behavior."""

    @patch("shared.plugins.mermaid_formatter.plugin.renderer")
    def test_flush_incomplete_mermaid_block(self, mock_renderer):
        """Incomplete block at turn end should be flushed."""
        mock_renderer.render.return_value = None
        plugin = MermaidFormatterPlugin()
        plugin.initialize()

        # Start a mermaid block but don't close it
        list(plugin.process_chunk("```mermaid\ngraph TD\n    A-->B"))

        # Flush should emit the incomplete content
        flushed = list(plugin.flush())
        output = "".join(flushed)
        assert "graph TD" in output

    def test_flush_regular_text(self):
        """Flush should emit buffered regular text."""
        plugin = MermaidFormatterPlugin()

        # Partial match at end causes buffering
        list(plugin.process_chunk("hello `"))
        flushed = list(plugin.flush())
        assert "".join(flushed) == "`"

    def test_flush_empty(self):
        """Flush with no buffer should yield nothing."""
        plugin = MermaidFormatterPlugin()
        flushed = list(plugin.flush())
        assert flushed == []

    def test_reset_clears_state(self):
        """Reset should clear all streaming state."""
        plugin = MermaidFormatterPlugin()

        # Build up some state
        list(plugin.process_chunk("```mermaid\ngraph TD"))
        assert plugin._in_mermaid_block is True
        assert plugin._buffer != ""

        plugin.reset()
        assert plugin._in_mermaid_block is False
        assert plugin._buffer == ""

    def test_reset_allows_reuse(self):
        """Plugin should work correctly after reset."""
        plugin = MermaidFormatterPlugin()

        # First turn
        list(plugin.process_chunk("```mermaid\nincomplete"))
        plugin.reset()

        # Second turn - should work normally
        result = list(plugin.process_chunk("clean text"))
        assert result == ["clean text"]


class TestSystemInstructions:
    """Tests for get_system_instructions()."""

    @patch("shared.plugins.mermaid_formatter.plugin.renderer")
    def test_returns_instructions_when_enabled_and_available(self, mock_renderer):
        mock_renderer.is_renderer_available.return_value = True
        plugin = MermaidFormatterPlugin()
        plugin.initialize()

        instr = plugin.get_system_instructions()
        assert instr is not None
        assert "mermaid" in instr.lower()

    @patch("shared.plugins.mermaid_formatter.plugin.renderer")
    def test_returns_none_when_disabled(self, mock_renderer):
        mock_renderer.is_renderer_available.return_value = True
        plugin = MermaidFormatterPlugin()
        plugin.initialize()
        plugin._enabled = False

        assert plugin.get_system_instructions() is None

    @patch("shared.plugins.mermaid_formatter.plugin.renderer")
    def test_returns_none_when_renderer_unavailable(self, mock_renderer):
        mock_renderer.is_renderer_available.return_value = False
        plugin = MermaidFormatterPlugin()
        plugin.initialize()

        assert plugin.get_system_instructions() is None

    @patch("shared.plugins.mermaid_formatter.plugin.renderer")
    def test_disabled_via_env_var(self, mock_renderer, monkeypatch):
        mock_renderer.is_renderer_available.return_value = True
        monkeypatch.setenv("JAATO_MERMAID_BACKEND", "off")
        plugin = MermaidFormatterPlugin()
        plugin.initialize()

        assert plugin.get_system_instructions() is None

    @patch("shared.plugins.mermaid_formatter.plugin.renderer")
    def test_mentions_diagram_types(self, mock_renderer):
        """Instructions should hint at useful diagram types."""
        mock_renderer.is_renderer_available.return_value = True
        plugin = MermaidFormatterPlugin()
        plugin.initialize()

        instr = plugin.get_system_instructions()
        assert "flow" in instr.lower() or "sequence" in instr.lower()


class TestFallbackRendering:
    """Tests for fallback when no renderer is available."""

    @patch("shared.plugins.mermaid_formatter.plugin.renderer")
    def test_fallback_shows_hint(self, mock_renderer):
        mock_renderer.render.return_value = None
        plugin = MermaidFormatterPlugin()
        plugin.initialize()

        chunks = list(plugin.process_chunk("```mermaid\ngraph TD\n    A-->B\n```"))
        output = "".join(chunks)

        assert "rendering unavailable" in output
        assert "mermaid-cli" in output or "JAATO_KROKI_URL" in output

    @patch("shared.plugins.mermaid_formatter.plugin.renderer")
    def test_fallback_preserves_source(self, mock_renderer):
        mock_renderer.render.return_value = None
        plugin = MermaidFormatterPlugin()
        plugin.initialize()

        chunks = list(plugin.process_chunk("```mermaid\ngraph TD\n    A-->B\n```"))
        output = "".join(chunks)

        # Source should be wrapped in ```mermaid for code_block_formatter
        assert "```mermaid" in output
        assert "graph TD" in output
        assert "A-->B" in output


class TestArtifactSaving:
    """Tests for PNG artifact file saving."""

    @patch("shared.plugins.mermaid_formatter.plugin.renderer")
    @patch("shared.plugins.mermaid_formatter.plugin.select_backend")
    def test_saves_artifact(self, mock_select_backend, mock_renderer, tmp_path):
        # Setup
        mock_renderer.render.return_value = b"\x89PNG fake data"
        mock_backend = MagicMock()
        mock_backend.render.return_value = "[rendered]\n"
        mock_select_backend.return_value = mock_backend

        plugin = MermaidFormatterPlugin()
        plugin.initialize()
        plugin._artifact_dir = str(tmp_path)

        chunks = list(plugin.process_chunk("```mermaid\ngraph TD\n    A-->B\n```"))
        output = "".join(chunks)

        # Check artifact was saved
        saved_files = list(tmp_path.glob("mermaid_*.png"))
        assert len(saved_files) == 1
        assert saved_files[0].read_bytes() == b"\x89PNG fake data"

        # Check output references the artifact
        assert "saved:" in output

    @patch("shared.plugins.mermaid_formatter.plugin.renderer")
    @patch("shared.plugins.mermaid_formatter.plugin.select_backend")
    def test_artifact_counter_increments(self, mock_select_backend, mock_renderer, tmp_path):
        mock_renderer.render.return_value = b"\x89PNG fake data"
        mock_backend = MagicMock()
        mock_backend.render.return_value = "[rendered]\n"
        mock_select_backend.return_value = mock_backend

        plugin = MermaidFormatterPlugin()
        plugin.initialize()
        plugin._artifact_dir = str(tmp_path)

        # Render two diagrams
        list(plugin.process_chunk("```mermaid\nA\n```"))
        list(plugin.process_chunk("```mermaid\nB\n```"))

        saved_files = sorted(tmp_path.glob("mermaid_*.png"))
        assert len(saved_files) == 2
        assert saved_files[0].name == "mermaid_001.png"
        assert saved_files[1].name == "mermaid_002.png"
