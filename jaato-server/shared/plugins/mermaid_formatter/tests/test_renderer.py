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


def _mock_httpx_client(status_code=200, content=b"", text=""):
    """Create a mock httpx client context manager with a configured response.

    Args:
        status_code: HTTP status code for the response.
        content: Raw bytes for response.content.
        text: Text for response.text.

    Returns:
        Mock that can be used as get_httpx_client return value.
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.content = content
    mock_response.text = text
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    return mock_client


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

    @patch("shared.plugins.mermaid_formatter.renderer.get_httpx_client")
    def test_available_when_200(self, mock_get_client):
        mock_client = _mock_httpx_client(status_code=200)
        mock_get_client.return_value = mock_client

        assert _check_kroki() is True

    @patch("shared.plugins.mermaid_formatter.renderer.get_httpx_client")
    def test_unavailable_on_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = Exception("connection refused")
        mock_get_client.return_value = mock_client
        assert _check_kroki() is False

    @patch("shared.plugins.mermaid_formatter.renderer.get_httpx_client")
    def test_caches_result(self, mock_get_client):
        mock_client = _mock_httpx_client(status_code=200)
        mock_get_client.return_value = mock_client

        _check_kroki()
        _check_kroki()
        # Only called once due to caching (get_httpx_client called once)
        mock_get_client.assert_called_once()

    @patch("shared.plugins.mermaid_formatter.renderer.get_httpx_client")
    def test_sends_test_diagram(self, mock_get_client):
        mock_client = _mock_httpx_client(status_code=200)
        mock_get_client.return_value = mock_client

        _check_kroki()

        # Verify the request was a POST with diagram data
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["content"] == b"graph TD\n    A-->B"
        assert call_kwargs[1]["headers"]["Content-Type"] == "text/plain"


class TestRenderKroki:
    """Tests for kroki.io rendering."""

    @patch("shared.plugins.mermaid_formatter.renderer.get_httpx_client")
    def test_returns_png_on_success(self, mock_get_client):
        mock_client = _mock_httpx_client(status_code=200, content=b"\x89PNG fake data")
        mock_get_client.return_value = mock_client

        result = _render_kroki("graph TD\n    A-->B", "default")
        assert result.png == b"\x89PNG fake data"
        assert result.error is None

    @patch("shared.plugins.mermaid_formatter.renderer.get_httpx_client")
    def test_returns_empty_on_network_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = Exception("timeout")
        mock_get_client.return_value = mock_client

        result = _render_kroki("graph TD", "default")
        assert result.png is None
        assert result.error is None

    @patch("shared.plugins.mermaid_formatter.renderer.get_httpx_client")
    def test_returns_error_on_400(self, mock_get_client):
        mock_client = _mock_httpx_client(
            status_code=400,
            text="Error 400: SyntaxError: Parse error on line 3",
        )
        mock_get_client.return_value = mock_client

        result = _render_kroki("bad diagram", "default")
        assert result.png is None
        assert result.error is not None
        assert "SyntaxError" in result.error

    @patch("shared.plugins.mermaid_formatter.renderer.get_httpx_client")
    def test_injects_theme_directive(self, mock_get_client):
        mock_client = _mock_httpx_client(status_code=200, content=b"\x89PNG")
        mock_get_client.return_value = mock_client

        _render_kroki("graph TD\n    A-->B", "dark")

        call_kwargs = mock_client.post.call_args
        body = call_kwargs[1]["content"].decode("utf-8")
        assert "%%{init:" in body
        assert "'theme': 'dark'" in body

    @patch("shared.plugins.mermaid_formatter.renderer.get_httpx_client")
    def test_skips_theme_injection_when_already_present(self, mock_get_client):
        mock_client = _mock_httpx_client(status_code=200, content=b"\x89PNG")
        mock_get_client.return_value = mock_client

        source = "%%{init: {'theme': 'forest'}}%%\ngraph TD\n    A-->B"
        _render_kroki(source, "dark")

        call_kwargs = mock_client.post.call_args
        body = call_kwargs[1]["content"].decode("utf-8")
        # Should not double-inject theme
        assert body.count("%%{init:") == 1

    @patch("shared.plugins.mermaid_formatter.renderer.get_httpx_client")
    def test_no_theme_injection_for_default(self, mock_get_client):
        mock_client = _mock_httpx_client(status_code=200, content=b"\x89PNG")
        mock_get_client.return_value = mock_client

        _render_kroki("graph TD\n    A-->B", "default")

        call_kwargs = mock_client.post.call_args
        body = call_kwargs[1]["content"].decode("utf-8")
        assert "%%{init:" not in body

    @patch("shared.plugins.mermaid_formatter.renderer.get_httpx_client")
    def test_uses_custom_kroki_url(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("JAATO_KROKI_URL", "http://localhost:8000")

        mock_client = _mock_httpx_client(status_code=200, content=b"\x89PNG")
        mock_get_client.return_value = mock_client

        _render_kroki("graph TD", "default")

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://localhost:8000/mermaid/png"


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
