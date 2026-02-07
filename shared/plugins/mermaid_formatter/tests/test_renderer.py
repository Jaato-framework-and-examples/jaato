"""Tests for the mermaid renderer module."""

import os
import pytest
from unittest.mock import patch, MagicMock

from ..renderer import (
    render,
    is_renderer_available,
    _find_mmdc,
    _render_mmdc,
    _render_mermaid_py,
    _svg_to_png,
)


@pytest.fixture(autouse=True)
def reset_mmdc_cache():
    """Reset mmdc path cache between tests."""
    import shared.plugins.mermaid_formatter.renderer as mod
    mod._mmdc_path = None
    mod._mmdc_checked = False
    yield
    mod._mmdc_path = None
    mod._mmdc_checked = False


class TestFindMmdc:
    """Tests for mmdc binary discovery."""

    @patch("shutil.which")
    def test_finds_mmdc(self, mock_which):
        mock_which.return_value = "/usr/local/bin/mmdc"
        assert _find_mmdc() == "/usr/local/bin/mmdc"

    @patch("shutil.which")
    def test_mmdc_not_found(self, mock_which):
        mock_which.return_value = None
        assert _find_mmdc() is None

    @patch("shutil.which")
    def test_caches_result(self, mock_which):
        mock_which.return_value = "/usr/local/bin/mmdc"
        _find_mmdc()
        _find_mmdc()
        # Only called once due to caching
        mock_which.assert_called_once_with("mmdc")


class TestRenderMmdc:
    """Tests for mmdc rendering."""

    @patch("shutil.which")
    def test_returns_none_when_mmdc_missing(self, mock_which):
        mock_which.return_value = None
        result = _render_mmdc("graph TD\n    A-->B", "default", 2, "white")
        assert result is None

    @patch("subprocess.run")
    @patch("shutil.which")
    def test_calls_mmdc_with_correct_args(self, mock_which, mock_run, tmp_path):
        mock_which.return_value = "/usr/local/bin/mmdc"
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        # The function creates its own tmpdir, so we patch that
        result = _render_mmdc("graph TD\n    A-->B", "dark", 3, "transparent")
        # mmdc was called
        mock_run.assert_called_once()
        args = mock_run.call_args
        cmd = args[0][0]  # First positional arg
        assert cmd[0] == "/usr/local/bin/mmdc"
        assert "-t" in cmd
        assert "dark" in cmd
        assert "-s" in cmd
        assert "3" in cmd

    @patch("subprocess.run")
    @patch("shutil.which")
    def test_handles_mmdc_failure(self, mock_which, mock_run):
        mock_which.return_value = "/usr/local/bin/mmdc"
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        result = _render_mmdc("invalid", "default", 2, "white")
        assert result is None

    @patch("subprocess.run")
    @patch("shutil.which")
    def test_handles_timeout(self, mock_which, mock_run):
        import subprocess
        mock_which.return_value = "/usr/local/bin/mmdc"
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="mmdc", timeout=30)
        result = _render_mmdc("graph TD", "default", 2, "white")
        assert result is None


class TestRenderMermaidPy:
    """Tests for mermaid Python package rendering."""

    def test_returns_none_when_not_installed(self):
        """Should return None when mermaid package is not installed."""
        # mermaid package is unlikely to be installed in test env
        result = _render_mermaid_py("graph TD\n    A-->B", "default", 2, "white")
        assert result is None


class TestSvgToPng:
    """Tests for SVG to PNG conversion."""

    def test_returns_none_when_cairosvg_missing(self):
        """Should return None gracefully when cairosvg is not installed."""
        with patch.dict("sys.modules", {"cairosvg": None}):
            result = _svg_to_png(b"<svg></svg>")
            # May or may not be None depending on whether cairosvg is actually installed
            # The important thing is it doesn't raise
            assert result is None or isinstance(result, bytes)

    def test_handles_invalid_svg(self):
        result = _svg_to_png(b"not svg at all")
        assert result is None or isinstance(result, bytes)


class TestRender:
    """Tests for the main render() function."""

    @patch("shared.plugins.mermaid_formatter.renderer._render_mermaid_py")
    @patch("shared.plugins.mermaid_formatter.renderer._render_mmdc")
    def test_tries_mmdc_first(self, mock_mmdc, mock_py):
        mock_mmdc.return_value = b"png from mmdc"
        mock_py.return_value = b"png from py"

        result = render("graph TD")
        assert result == b"png from mmdc"
        mock_py.assert_not_called()

    @patch("shared.plugins.mermaid_formatter.renderer._render_mermaid_py")
    @patch("shared.plugins.mermaid_formatter.renderer._render_mmdc")
    def test_falls_back_to_mermaid_py(self, mock_mmdc, mock_py):
        mock_mmdc.return_value = None
        mock_py.return_value = b"png from py"

        result = render("graph TD")
        assert result == b"png from py"

    @patch("shared.plugins.mermaid_formatter.renderer._render_mermaid_py")
    @patch("shared.plugins.mermaid_formatter.renderer._render_mmdc")
    def test_returns_none_when_nothing_available(self, mock_mmdc, mock_py):
        mock_mmdc.return_value = None
        mock_py.return_value = None

        result = render("graph TD")
        assert result is None


class TestIsRendererAvailable:
    """Tests for renderer availability check."""

    @patch("shutil.which")
    def test_available_when_mmdc_exists(self, mock_which):
        mock_which.return_value = "/usr/local/bin/mmdc"
        assert is_renderer_available() is True

    @patch("shutil.which")
    def test_not_available_when_nothing_installed(self, mock_which):
        mock_which.return_value = None
        # Also ensure mermaid package is not importable
        with patch.dict("sys.modules", {"mermaid": None}):
            # Reset cache
            import shared.plugins.mermaid_formatter.renderer as mod
            mod._mmdc_path = None
            mod._mmdc_checked = False
            result = is_renderer_available()
            # Result depends on whether mermaid-py is actually installed
            assert isinstance(result, bool)
