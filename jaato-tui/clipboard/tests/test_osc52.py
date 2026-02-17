"""Tests for OSC 52 clipboard provider."""

import base64
import sys
from pathlib import Path

import pytest

# Add parent directories to path for imports
_repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_repo_root / "jaato-tui"))

from clipboard.osc52 import (
    OSC52_MAX_BYTES,
    OSC52_SCREEN_MAX_BYTES,
    _truncate_utf8_safe,
    OSC52Provider,
)


class TestTruncateUtf8Safe:
    """Tests for the _truncate_utf8_safe helper function."""

    def test_no_truncation_needed(self):
        """Text under limit is returned unchanged."""
        text = "Hello, World!"
        result = _truncate_utf8_safe(text, 100)
        assert result == text

    def test_exact_fit(self):
        """Text that exactly fits is returned unchanged."""
        text = "12345"
        result = _truncate_utf8_safe(text, 5)
        assert result == text

    def test_ascii_truncation(self):
        """ASCII text is truncated correctly."""
        text = "Hello, World!"
        result = _truncate_utf8_safe(text, 5)
        assert result == "Hello"
        assert len(result.encode("utf-8")) <= 5

    def test_2byte_char_not_split(self):
        """2-byte UTF-8 characters (e.g., é) are not split."""
        # 'é' is 2 bytes in UTF-8 (0xC3 0xA9)
        text = "café"
        # "caf" = 3 bytes, "é" = 2 bytes, total = 5 bytes
        # Truncating at 4 bytes should give "caf", not corrupt data
        result = _truncate_utf8_safe(text, 4)
        assert result == "caf"
        assert len(result.encode("utf-8")) <= 4

    def test_3byte_char_not_split(self):
        """3-byte UTF-8 characters (e.g., 中) are not split."""
        # '中' is 3 bytes in UTF-8 (0xE4 0xB8 0xAD)
        text = "a中b"
        # "a" = 1 byte, "中" = 3 bytes, "b" = 1 byte
        # Truncating at 3 bytes should give "a", not corrupt data
        result = _truncate_utf8_safe(text, 3)
        assert result == "a"
        # Truncating at 4 bytes should give "a中"
        result = _truncate_utf8_safe(text, 4)
        assert result == "a中"

    def test_4byte_char_not_split(self):
        """4-byte UTF-8 characters (e.g., emoji) are not split."""
        # '😀' is 4 bytes in UTF-8 (0xF0 0x9F 0x98 0x80)
        text = "a😀b"
        # "a" = 1 byte, "😀" = 4 bytes, "b" = 1 byte
        # Truncating at 4 bytes should give "a", not corrupt data
        result = _truncate_utf8_safe(text, 4)
        assert result == "a"
        # Truncating at 5 bytes should give "a😀"
        result = _truncate_utf8_safe(text, 5)
        assert result == "a😀"

    def test_multiple_multibyte_chars(self):
        """Multiple multi-byte characters are handled correctly."""
        text = "日本語"  # 9 bytes total (3 chars × 3 bytes each)
        result = _truncate_utf8_safe(text, 6)
        assert result == "日本"
        result = _truncate_utf8_safe(text, 3)
        assert result == "日"

    def test_empty_result_when_first_char_too_large(self):
        """Returns empty string when first char doesn't fit."""
        text = "😀hello"  # First char is 4 bytes
        result = _truncate_utf8_safe(text, 3)
        assert result == ""

    def test_empty_input(self):
        """Empty input returns empty output."""
        result = _truncate_utf8_safe("", 100)
        assert result == ""

    def test_zero_max_bytes(self):
        """Zero max_bytes returns empty string."""
        result = _truncate_utf8_safe("hello", 0)
        assert result == ""


class TestBufferSizeCalculation:
    """Tests to verify the buffer size calculation doesn't overflow."""

    def test_max_bytes_formula_never_overflows(self):
        """The formula (max // 4) * 3 never produces overflow after base64."""
        # Test the standard limit
        max_text_bytes = (OSC52_MAX_BYTES // 4) * 3
        # Create text at exactly max_text_bytes
        text = "a" * max_text_bytes
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        assert len(encoded) <= OSC52_MAX_BYTES

    def test_old_formula_would_overflow(self):
        """Demonstrate that the old formula could overflow."""
        # Old formula: (max * 3) // 4
        old_max_text_bytes = (OSC52_MAX_BYTES * 3) // 4  # = 56245
        text = "a" * old_max_text_bytes
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        # This would exceed the limit!
        assert len(encoded) > OSC52_MAX_BYTES

    def test_tmux_uses_standard_limit(self):
        """Tmux uses the standard limit — no DCS passthrough needed.

        tmux natively intercepts OSC 52 via set-clipboard, so we send
        raw sequences without DCS wrapping and can use the full limit.
        """
        max_text_bytes = (OSC52_MAX_BYTES // 4) * 3
        text = "a" * max_text_bytes
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        assert len(encoded) <= OSC52_MAX_BYTES

    def test_screen_uses_conservative_limit(self):
        """Screen uses a conservative limit similar to tmux."""
        max_text_bytes = (OSC52_SCREEN_MAX_BYTES // 4) * 3
        text = "a" * max_text_bytes
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        assert len(encoded) <= OSC52_SCREEN_MAX_BYTES


class TestOSC52Provider:
    """Tests for the OSC52Provider class."""

    def test_empty_text_returns_false(self, monkeypatch):
        """Empty text returns False without writing anything."""
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.setenv("TERM", "xterm")
        provider = OSC52Provider()
        assert provider.copy("") is False

    def test_name_property_standard(self, monkeypatch):
        """Name property returns 'OSC 52' for standard terminal."""
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.setenv("TERM", "xterm")
        provider = OSC52Provider()
        assert provider.name == "OSC 52"

    def test_name_property_tmux(self, monkeypatch):
        """Name property returns 'OSC 52 (tmux)' when in tmux."""
        monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,12345,0")
        monkeypatch.setenv("TERM", "screen")
        provider = OSC52Provider()
        assert provider.name == "OSC 52 (tmux)"

    def test_name_property_screen(self, monkeypatch):
        """Name property returns 'OSC 52 (screen)' when in screen."""
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.setenv("TERM", "screen-256color")
        provider = OSC52Provider()
        assert provider.name == "OSC 52 (screen)"

    def test_tmux_not_detected_as_screen(self, monkeypatch):
        """When TMUX is set and TERM=screen, detect as tmux not screen.

        tmux typically sets TERM=screen or TERM=screen-256color, but when
        TMUX env var is present, it's tmux — not GNU screen.
        """
        monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,12345,0")
        monkeypatch.setenv("TERM", "screen-256color")
        provider = OSC52Provider()
        assert provider._in_tmux is True
        assert provider._in_screen is False
        assert provider.name == "OSC 52 (tmux)"

    def test_tmux_sends_raw_osc52(self, monkeypatch):
        """In tmux, OSC 52 is sent raw without DCS passthrough wrapping.

        tmux natively intercepts OSC 52 via set-clipboard, so DCS
        passthrough wrapping is unnecessary and imposes size limits.
        """
        monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,12345,0")
        monkeypatch.setenv("TERM", "screen-256color")

        provider = OSC52Provider()

        written_data = []

        def mock_open(path, mode="r", *args, **kwargs):
            if path == "/dev/tty":
                from io import StringIO

                class MockTTY(StringIO):
                    def write(self, data):
                        written_data.append(data)
                        return len(data)

                    def flush(self):
                        pass

                return MockTTY()
            raise OSError("unexpected open")

        monkeypatch.setattr("builtins.open", mock_open)

        provider.copy("hello")
        assert len(written_data) == 1
        seq = written_data[0]

        # Must be raw OSC 52, NOT wrapped in DCS passthrough
        assert seq.startswith("\x1b]52;c;")
        assert seq.endswith("\x1b\\")
        # Must NOT contain DCS passthrough markers
        assert "\x1bPtmux;" not in seq

    def test_tmux_large_text_uses_full_limit(self, monkeypatch):
        """In tmux, large text uses the standard limit, not a conservative one."""
        monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,12345,0")
        monkeypatch.setenv("TERM", "screen-256color")

        provider = OSC52Provider()

        written_data = []

        def mock_open(path, mode="r", *args, **kwargs):
            if path == "/dev/tty":
                from io import StringIO

                class MockTTY(StringIO):
                    def write(self, data):
                        written_data.append(data)
                        return len(data)

                    def flush(self):
                        pass

                return MockTTY()
            raise OSError("unexpected open")

        monkeypatch.setattr("builtins.open", mock_open)

        # 20KB of text — would have been truncated by the old 16KB (base64) limit
        large_text = "x" * 20000
        provider.copy(large_text)
        assert len(written_data) == 1

        seq = written_data[0]
        base64_part = seq[7:-2]  # Remove \x1b]52;c; prefix and \x1b\ suffix
        decoded = base64.b64decode(base64_part).decode("utf-8")
        # All 20KB should be preserved (well under the 56KB standard limit)
        assert decoded == large_text

    def test_large_text_truncated_correctly(self, monkeypatch, tmp_path):
        """Large text is truncated without buffer overflow."""
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.setenv("TERM", "xterm")

        # Create a mock /dev/tty
        mock_tty = tmp_path / "mock_tty"
        mock_tty.touch()

        provider = OSC52Provider()

        # Create text larger than the limit
        large_text = "x" * 100000

        # Patch open to capture what would be written
        written_data = []
        original_open = open

        def mock_open(path, mode="r", *args, **kwargs):
            if path == "/dev/tty":
                from io import StringIO

                class MockTTY(StringIO):
                    def write(self, data):
                        written_data.append(data)
                        return len(data)

                    def flush(self):
                        pass

                return MockTTY()
            return original_open(path, mode, *args, **kwargs)

        monkeypatch.setattr("builtins.open", mock_open)

        result = provider.copy(large_text)
        assert result is True
        assert len(written_data) == 1

        # Extract the base64 portion and verify it's under the limit
        osc_seq = written_data[0]
        # Format: \x1b]52;c;{base64}\x1b\\
        assert osc_seq.startswith("\x1b]52;c;")
        assert osc_seq.endswith("\x1b\\")
        base64_part = osc_seq[7:-2]  # Remove prefix and suffix
        assert len(base64_part) <= OSC52_MAX_BYTES
