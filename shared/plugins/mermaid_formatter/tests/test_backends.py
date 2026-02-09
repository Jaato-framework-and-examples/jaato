"""Tests for terminal graphics backends."""

import base64
import os
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO

from ..backends import select_backend, _create_backend
from ..backends.kitty import KittyBackend
from ..backends.iterm import ITermBackend
from ..backends.sixel import SixelBackend


def _make_test_png(width=4, height=4) -> bytes:
    """Create a minimal valid PNG for testing."""
    try:
        from PIL import Image
        img = Image.new("RGB", (width, height), color=(255, 0, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        pytest.skip("Pillow required for backend tests")


class TestBackendSelection:
    """Tests for auto-selection of graphics backend."""

    def test_default_is_none(self, monkeypatch):
        """When no graphics protocol detected, returns None."""
        monkeypatch.delenv("JAATO_MERMAID_BACKEND", raising=False)
        monkeypatch.delenv("JAATO_GRAPHICS_PROTOCOL", raising=False)
        # Ensure terminal_caps detects no graphics
        with patch("shared.plugins.mermaid_formatter.backends.detect_terminal_caps") as mock:
            mock.return_value = {"graphics": None}
            backend = select_backend()
            assert backend is None

    def test_env_override_kitty(self, monkeypatch):
        monkeypatch.setenv("JAATO_MERMAID_BACKEND", "kitty")
        backend = select_backend()
        assert isinstance(backend, KittyBackend)

    def test_env_override_iterm(self, monkeypatch):
        monkeypatch.setenv("JAATO_MERMAID_BACKEND", "iterm")
        backend = select_backend()
        assert isinstance(backend, ITermBackend)

    def test_env_override_sixel(self, monkeypatch):
        monkeypatch.setenv("JAATO_MERMAID_BACKEND", "sixel")
        backend = select_backend()
        assert isinstance(backend, SixelBackend)

    def test_env_override_off(self, monkeypatch):
        """When backend is 'off', returns None."""
        monkeypatch.setenv("JAATO_MERMAID_BACKEND", "off")
        backend = select_backend()
        assert backend is None

    def test_env_override_unknown(self, monkeypatch):
        """Unknown backend name falls through to auto-detect."""
        monkeypatch.setenv("JAATO_MERMAID_BACKEND", "ascii")
        with patch("shared.plugins.mermaid_formatter.backends.detect_terminal_caps") as mock:
            mock.return_value = {"graphics": None}
            backend = select_backend()
            assert backend is None

    def test_auto_detect_kitty(self, monkeypatch):
        monkeypatch.delenv("JAATO_MERMAID_BACKEND", raising=False)
        with patch("shared.plugins.mermaid_formatter.backends.detect_terminal_caps") as mock:
            mock.return_value = {"graphics": "kitty"}
            backend = select_backend()
            assert isinstance(backend, KittyBackend)

    def test_auto_detect_iterm(self, monkeypatch):
        monkeypatch.delenv("JAATO_MERMAID_BACKEND", raising=False)
        with patch("shared.plugins.mermaid_formatter.backends.detect_terminal_caps") as mock:
            mock.return_value = {"graphics": "iterm"}
            backend = select_backend()
            assert isinstance(backend, ITermBackend)

    def test_auto_detect_sixel(self, monkeypatch):
        monkeypatch.delenv("JAATO_MERMAID_BACKEND", raising=False)
        with patch("shared.plugins.mermaid_formatter.backends.detect_terminal_caps") as mock:
            mock.return_value = {"graphics": "sixel"}
            backend = select_backend()
            assert isinstance(backend, SixelBackend)


class TestKittyBackend:
    """Tests for Kitty graphics protocol backend."""

    def test_name(self):
        backend = KittyBackend()
        assert backend.name == "kitty"

    def test_render_produces_escape_sequences(self):
        png_data = _make_test_png()
        backend = KittyBackend(max_width=40)
        result = backend.render(png_data)

        # Should start with Kitty graphics protocol escape
        assert "\x1b_G" in result
        # Should end with string terminator
        assert "\x1b\\" in result
        # Should contain base64 data
        assert "f=100" in result  # PNG format
        assert "a=T" in result    # Transmit and display

    def test_render_chunked_for_large_images(self):
        # Create a larger image that requires multiple chunks
        png_data = _make_test_png(100, 100)
        backend = KittyBackend(max_width=80)
        result = backend.render(png_data)

        # Should contain multiple escape sequences for chunks
        chunk_count = result.count("\x1b_G")
        assert chunk_count >= 1

        # First chunk has m=1 (more) or m=0 (last), last has m=0
        assert "m=0" in result


class TestITermBackend:
    """Tests for iTerm2 inline image protocol backend."""

    def test_name(self):
        backend = ITermBackend()
        assert backend.name == "iterm"

    def test_render_produces_osc_1337(self):
        png_data = _make_test_png()
        backend = ITermBackend(max_width=40)
        result = backend.render(png_data)

        # Should contain iTerm2 escape sequence
        assert "\x1b]1337;File=" in result
        assert "inline=1" in result
        assert "preserveAspectRatio=1" in result
        # Should end with BEL
        assert "\x07" in result

    def test_render_includes_size(self):
        png_data = _make_test_png()
        backend = ITermBackend(max_width=40)
        result = backend.render(png_data)

        assert f"size={len(png_data)}" in result


class TestSixelBackend:
    """Tests for Sixel graphics protocol backend."""

    def test_name(self):
        backend = SixelBackend()
        assert backend.name == "sixel"

    def test_render_produces_dcs_sequence(self):
        png_data = _make_test_png()
        backend = SixelBackend(max_width=40)
        result = backend.render(png_data)

        # Should start with DCS (Device Control String)
        assert "\x1bP" in result
        # Should end with ST (String Terminator)
        assert "\x1b\\" in result
        # Should contain color definitions
        assert "#" in result

    def test_rle_encode_basic(self):
        """Test RLE encoding of sixel data."""
        data = [0, 0, 0, 0, 0, 1, 1, 2]
        result = SixelBackend._rle_encode(data)
        # 5x char(0+63)='?' then 2x char(1+63)='@' then 1x char(2+63)='A'
        assert "!5?" in result
        assert "@@" in result
        assert "A" in result

    def test_rle_encode_empty(self):
        assert SixelBackend._rle_encode([]) == ""

    def test_rle_encode_no_runs(self):
        data = [0, 1, 2, 3]
        result = SixelBackend._rle_encode(data)
        assert "?" in result  # char(0+63)
        assert "@" in result  # char(1+63)
        assert "A" in result  # char(2+63)
        assert "B" in result  # char(3+63)


