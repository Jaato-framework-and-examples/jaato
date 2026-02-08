"""Tests for the mermaid renderer module."""

import os
import pytest
from unittest.mock import patch, MagicMock

from ..renderer import (
    render,
    RenderResult,
    is_renderer_available,
    _find_mmdc,
    _render_mmdc,
    _render_kroki,
    _check_kroki,
    _get_kroki_url,
    _extract_kroki_error,
    _extract_mmdc_error,
)


@pytest.fixture(autouse=True)
def reset_caches():
    """Reset renderer caches between tests."""
    import shared.plugins.mermaid_formatter.renderer as mod
    mod._mmdc_path = None
    mod._mmdc_checked = False
    mod._kroki_available = None
    yield
    mod._mmdc_path = None
    mod._mmdc_checked = False
    mod._kroki_available = None


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
    def test_returns_empty_when_mmdc_missing(self, mock_which):
        mock_which.return_value = None
        result = _render_mmdc("graph TD\n    A-->B", "default", 2, "white")
        assert result.png is None
        assert result.error is None

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
    def test_handles_mmdc_failure_with_error(self, mock_which, mock_run):
        mock_which.return_value = "/usr/local/bin/mmdc"
        mock_run.return_value = MagicMock(returncode=1, stderr="Error: Parse error on line 1")
        result = _render_mmdc("invalid", "default", 2, "white")
        assert result.png is None
        assert result.error is not None
        assert "error" in result.error.lower()

    @patch("subprocess.run")
    @patch("shutil.which")
    def test_handles_timeout(self, mock_which, mock_run):
        import subprocess
        mock_which.return_value = "/usr/local/bin/mmdc"
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="mmdc", timeout=30)
        result = _render_mmdc("graph TD", "default", 2, "white")
        assert result.png is None


class TestGetKrokiUrl:
    """Tests for kroki URL configuration."""

    def test_default_url(self):
        with patch.dict(os.environ, {}, clear=False):
            # Remove JAATO_KROKI_URL if present
            os.environ.pop("JAATO_KROKI_URL", None)
            assert _get_kroki_url() == "https://kroki.io"

    def test_custom_url(self, monkeypatch):
        monkeypatch.setenv("JAATO_KROKI_URL", "http://localhost:8000")
        assert _get_kroki_url() == "http://localhost:8000"

    def test_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("JAATO_KROKI_URL", "http://localhost:8000/")
        assert _get_kroki_url() == "http://localhost:8000"


class TestCheckKroki:
    """Tests for kroki reachability check."""

    @patch("shared.plugins.mermaid_formatter.renderer.get_url_opener")
    def test_available_when_200(self, mock_get_opener):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp
        mock_get_opener.return_value = mock_opener

        assert _check_kroki() is True

    @patch("shared.plugins.mermaid_formatter.renderer.get_url_opener")
    def test_unavailable_on_error(self, mock_get_opener):
        mock_opener = MagicMock()
        mock_opener.open.side_effect = Exception("connection refused")
        mock_get_opener.return_value = mock_opener
        assert _check_kroki() is False

    @patch("shared.plugins.mermaid_formatter.renderer.get_url_opener")
    def test_caches_result(self, mock_get_opener):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp
        mock_get_opener.return_value = mock_opener

        _check_kroki()
        _check_kroki()
        # Only called once due to caching
        mock_opener.open.assert_called_once()

    @patch("shared.plugins.mermaid_formatter.renderer.get_url_opener")
    def test_sends_test_diagram(self, mock_get_opener):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp
        mock_get_opener.return_value = mock_opener

        _check_kroki()

        # Verify the request was a POST with diagram data
        req = mock_opener.open.call_args[0][0]
        assert req.method == "POST"
        assert req.data == b"graph TD\n    A-->B"
        assert req.get_header("Content-type") == "text/plain"


class TestRenderKroki:
    """Tests for kroki.io rendering."""

    @patch("shared.plugins.mermaid_formatter.renderer.get_url_opener")
    def test_returns_png_on_success(self, mock_get_opener):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"\x89PNG fake data"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp
        mock_get_opener.return_value = mock_opener

        result = _render_kroki("graph TD\n    A-->B", "default")
        assert result.png == b"\x89PNG fake data"
        assert result.error is None

    @patch("shared.plugins.mermaid_formatter.renderer.get_url_opener")
    def test_returns_empty_on_network_error(self, mock_get_opener):
        mock_opener = MagicMock()
        mock_opener.open.side_effect = Exception("timeout")
        mock_get_opener.return_value = mock_opener
        result = _render_kroki("graph TD", "default")
        assert result.png is None
        assert result.error is None

    @patch("shared.plugins.mermaid_formatter.renderer.get_url_opener")
    def test_returns_error_on_400(self, mock_get_opener):
        import urllib.error
        err = urllib.error.HTTPError(
            "https://kroki.io/mermaid/png", 400, "Bad Request", {},
            MagicMock(read=MagicMock(return_value=b"Error 400: SyntaxError: Parse error on line 3"))
        )
        err.read = MagicMock(return_value=b"Error 400: SyntaxError: Parse error on line 3")
        mock_opener = MagicMock()
        mock_opener.open.side_effect = err
        mock_get_opener.return_value = mock_opener
        result = _render_kroki("bad diagram", "default")
        assert result.png is None
        assert result.error is not None
        assert "SyntaxError" in result.error

    @patch("shared.plugins.mermaid_formatter.renderer.get_url_opener")
    def test_injects_theme_directive(self, mock_get_opener):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"\x89PNG"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp
        mock_get_opener.return_value = mock_opener

        _render_kroki("graph TD\n    A-->B", "dark")

        req = mock_opener.open.call_args[0][0]
        body = req.data.decode("utf-8")
        assert "%%{init:" in body
        assert "'theme': 'dark'" in body

    @patch("shared.plugins.mermaid_formatter.renderer.get_url_opener")
    def test_skips_theme_injection_when_already_present(self, mock_get_opener):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"\x89PNG"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp
        mock_get_opener.return_value = mock_opener

        source = "%%{init: {'theme': 'forest'}}%%\ngraph TD\n    A-->B"
        _render_kroki(source, "dark")

        req = mock_opener.open.call_args[0][0]
        body = req.data.decode("utf-8")
        # Should not double-inject theme
        assert body.count("%%{init:") == 1

    @patch("shared.plugins.mermaid_formatter.renderer.get_url_opener")
    def test_no_theme_injection_for_default(self, mock_get_opener):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"\x89PNG"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp
        mock_get_opener.return_value = mock_opener

        _render_kroki("graph TD\n    A-->B", "default")

        req = mock_opener.open.call_args[0][0]
        body = req.data.decode("utf-8")
        assert "%%{init:" not in body

    @patch("shared.plugins.mermaid_formatter.renderer.get_url_opener")
    def test_uses_custom_kroki_url(self, mock_get_opener, monkeypatch):
        monkeypatch.setenv("JAATO_KROKI_URL", "http://localhost:8000")

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"\x89PNG"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp
        mock_get_opener.return_value = mock_opener

        _render_kroki("graph TD", "default")

        req = mock_opener.open.call_args[0][0]
        assert req.full_url == "http://localhost:8000/mermaid/png"


class TestRender:
    """Tests for the main render() function."""

    @patch("shared.plugins.mermaid_formatter.renderer._render_kroki")
    @patch("shared.plugins.mermaid_formatter.renderer._render_mmdc")
    def test_tries_mmdc_first(self, mock_mmdc, mock_kroki):
        mock_mmdc.return_value = RenderResult(png=b"png from mmdc")
        mock_kroki.return_value = RenderResult(png=b"png from kroki")

        result = render("graph TD")
        assert result.png == b"png from mmdc"
        mock_kroki.assert_not_called()

    @patch("shared.plugins.mermaid_formatter.renderer._render_kroki")
    @patch("shared.plugins.mermaid_formatter.renderer._render_mmdc")
    def test_falls_back_to_kroki(self, mock_mmdc, mock_kroki):
        mock_mmdc.return_value = RenderResult()
        mock_kroki.return_value = RenderResult(png=b"png from kroki")

        result = render("graph TD")
        assert result.png == b"png from kroki"

    @patch("shared.plugins.mermaid_formatter.renderer._render_kroki")
    @patch("shared.plugins.mermaid_formatter.renderer._render_mmdc")
    def test_returns_empty_when_nothing_available(self, mock_mmdc, mock_kroki):
        mock_mmdc.return_value = RenderResult()
        mock_kroki.return_value = RenderResult()

        result = render("graph TD")
        assert result.png is None
        assert result.error is None

    @patch("shared.plugins.mermaid_formatter.renderer._render_kroki")
    @patch("shared.plugins.mermaid_formatter.renderer._render_mmdc")
    def test_propagates_syntax_error(self, mock_mmdc, mock_kroki):
        """When mmdc not available and kroki returns syntax error, propagate it."""
        mock_mmdc.return_value = RenderResult()
        mock_kroki.return_value = RenderResult(error="SyntaxError: bad")

        result = render("graph TD")
        assert result.png is None
        assert result.error == "SyntaxError: bad"


class TestIsRendererAvailable:
    """Tests for renderer availability check."""

    @patch("shutil.which")
    def test_available_when_mmdc_exists(self, mock_which):
        mock_which.return_value = "/usr/local/bin/mmdc"
        assert is_renderer_available() is True

    @patch("shared.plugins.mermaid_formatter.renderer._check_kroki")
    @patch("shutil.which")
    def test_checks_kroki_when_no_mmdc(self, mock_which, mock_check_kroki):
        mock_which.return_value = None
        mock_check_kroki.return_value = True
        assert is_renderer_available() is True

    @patch("shared.plugins.mermaid_formatter.renderer._check_kroki")
    @patch("shutil.which")
    def test_unavailable_when_nothing_works(self, mock_which, mock_check_kroki):
        mock_which.return_value = None
        mock_check_kroki.return_value = False
        assert is_renderer_available() is False

    @patch("shutil.which")
    def test_mmdc_skips_kroki_check(self, mock_which):
        """When mmdc is found, kroki check is not performed."""
        mock_which.return_value = "/usr/local/bin/mmdc"

        with patch("shared.plugins.mermaid_formatter.renderer._check_kroki") as mock_kroki:
            is_renderer_available()
            mock_kroki.assert_not_called()


class TestExtractKrokiError:
    """Tests for kroki 400 error extraction."""

    def test_strips_error_prefix(self):
        body = "Error 400: SyntaxError: Parse error on line 3"
        result = _extract_kroki_error(body)
        assert result == "SyntaxError: Parse error on line 3"

    def test_strips_stack_trace(self):
        body = (
            "Error 400: SyntaxError: Parse error\n"
            "Expecting 'SQE', got 'PS'\n"
            "    at Worker.convert (file:///usr/local/kroki/src/worker.js:44:15)\n"
            "    at async file:///usr/local/kroki/src/index.js:31:28"
        )
        result = _extract_kroki_error(body)
        assert "SyntaxError" in result
        assert "Expecting" in result
        assert "Worker.convert" not in result

    def test_empty_body(self):
        assert _extract_kroki_error("") is None
        assert _extract_kroki_error(None) is None


class TestExtractMmdcError:
    """Tests for mmdc stderr error extraction."""

    def test_extracts_error_line(self):
        stderr = "Processing: input.mmd\nError: Parse error on line 5\nDone"
        result = _extract_mmdc_error(stderr)
        assert "Error" in result
        assert "Parse error" in result

    def test_empty_stderr(self):
        assert _extract_mmdc_error("") is None
        assert _extract_mmdc_error(None) is None
