"""Tests for the mermaid renderer module."""

import os
import pytest
from unittest.mock import patch, MagicMock

from ..renderer import (
    render,
    is_renderer_available,
    _find_mmdc,
    _render_mmdc,
    _render_kroki,
    _check_kroki,
    _get_kroki_url,
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

    @patch("urllib.request.urlopen")
    def test_available_when_200(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        assert _check_kroki() is True

    @patch("urllib.request.urlopen")
    def test_unavailable_on_error(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("connection refused")
        assert _check_kroki() is False

    @patch("urllib.request.urlopen")
    def test_caches_result(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        _check_kroki()
        _check_kroki()
        # Only called once due to caching
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_sends_test_diagram(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        _check_kroki()

        # Verify the request was a POST with diagram data
        req = mock_urlopen.call_args[0][0]
        assert req.method == "POST"
        assert req.data == b"graph TD\n    A-->B"
        assert req.get_header("Content-type") == "text/plain"


class TestRenderKroki:
    """Tests for kroki.io rendering."""

    @patch("urllib.request.urlopen")
    def test_returns_png_on_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"\x89PNG fake data"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _render_kroki("graph TD\n    A-->B", "default")
        assert result == b"\x89PNG fake data"

    @patch("urllib.request.urlopen")
    def test_returns_none_on_error(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("timeout")
        result = _render_kroki("graph TD", "default")
        assert result is None

    @patch("urllib.request.urlopen")
    def test_injects_theme_directive(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"\x89PNG"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        _render_kroki("graph TD\n    A-->B", "dark")

        req = mock_urlopen.call_args[0][0]
        body = req.data.decode("utf-8")
        assert "%%{init:" in body
        assert "'theme': 'dark'" in body

    @patch("urllib.request.urlopen")
    def test_skips_theme_injection_when_already_present(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"\x89PNG"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        source = "%%{init: {'theme': 'forest'}}%%\ngraph TD\n    A-->B"
        _render_kroki(source, "dark")

        req = mock_urlopen.call_args[0][0]
        body = req.data.decode("utf-8")
        # Should not double-inject theme
        assert body.count("%%{init:") == 1

    @patch("urllib.request.urlopen")
    def test_no_theme_injection_for_default(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"\x89PNG"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        _render_kroki("graph TD\n    A-->B", "default")

        req = mock_urlopen.call_args[0][0]
        body = req.data.decode("utf-8")
        assert "%%{init:" not in body

    @patch("urllib.request.urlopen")
    def test_uses_custom_kroki_url(self, mock_urlopen, monkeypatch):
        monkeypatch.setenv("JAATO_KROKI_URL", "http://localhost:8000")

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"\x89PNG"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        _render_kroki("graph TD", "default")

        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://localhost:8000/mermaid/png"


class TestRender:
    """Tests for the main render() function."""

    @patch("shared.plugins.mermaid_formatter.renderer._render_kroki")
    @patch("shared.plugins.mermaid_formatter.renderer._render_mmdc")
    def test_tries_mmdc_first(self, mock_mmdc, mock_kroki):
        mock_mmdc.return_value = b"png from mmdc"
        mock_kroki.return_value = b"png from kroki"

        result = render("graph TD")
        assert result == b"png from mmdc"
        mock_kroki.assert_not_called()

    @patch("shared.plugins.mermaid_formatter.renderer._render_kroki")
    @patch("shared.plugins.mermaid_formatter.renderer._render_mmdc")
    def test_falls_back_to_kroki(self, mock_mmdc, mock_kroki):
        mock_mmdc.return_value = None
        mock_kroki.return_value = b"png from kroki"

        result = render("graph TD")
        assert result == b"png from kroki"

    @patch("shared.plugins.mermaid_formatter.renderer._render_kroki")
    @patch("shared.plugins.mermaid_formatter.renderer._render_mmdc")
    def test_returns_none_when_nothing_available(self, mock_mmdc, mock_kroki):
        mock_mmdc.return_value = None
        mock_kroki.return_value = None

        result = render("graph TD")
        assert result is None


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
