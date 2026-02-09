"""Tests for shared terminal capability detection."""

import os
import sys
import pytest

from shared.terminal_caps import (
    detect,
    invalidate_cache,
    _detect,
    _detect_multiplexer,
    _detect_color_depth,
    _detect_emulator,
    _detect_graphics,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Ensure each test starts with a clean cache."""
    invalidate_cache()
    yield
    invalidate_cache()


class TestCaching:
    """Tests for process-wide caching."""

    def test_detect_returns_dict(self):
        result = detect()
        assert isinstance(result, dict)

    def test_detect_is_cached(self):
        first = detect()
        second = detect()
        assert first is second  # Same object, not just equal

    def test_invalidate_cache_forces_redetection(self):
        first = detect()
        invalidate_cache()
        second = detect()
        # After invalidation, should be a new dict (may be equal but not same object)
        assert first is not second

    def test_detect_matches_raw_detect(self, monkeypatch):
        """Cached result should match a fresh detection."""
        result = detect()
        invalidate_cache()
        fresh = _detect()
        assert result == fresh


class TestMultiplexerDetection:
    """Tests for multiplexer detection."""

    def test_tmux_detected(self, monkeypatch):
        monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,12345,0")
        monkeypatch.delenv("STY", raising=False)
        assert _detect_multiplexer("xterm-256color") == "tmux"

    def test_screen_detected_via_sty(self, monkeypatch):
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.setenv("STY", "12345.pts-0.hostname")
        assert _detect_multiplexer("xterm") == "screen"

    def test_screen_detected_via_term(self, monkeypatch):
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.delenv("STY", raising=False)
        assert _detect_multiplexer("screen.xterm-256color") == "screen"

    def test_no_multiplexer(self, monkeypatch):
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.delenv("STY", raising=False)
        assert _detect_multiplexer("xterm-256color") is None

    def test_tmux_takes_priority_over_screen_term(self, monkeypatch):
        monkeypatch.setenv("TMUX", "/tmp/tmux")
        monkeypatch.delenv("STY", raising=False)
        assert _detect_multiplexer("screen") == "tmux"


class TestColorDepthDetection:
    """Tests for color depth detection."""

    def test_non_interactive_is_none(self):
        assert _detect_color_depth(False, "xterm-256color", "truecolor") == "none"

    def test_truecolor_via_colorterm(self):
        assert _detect_color_depth(True, "xterm", "truecolor") == "24bit"

    def test_24bit_via_colorterm(self):
        assert _detect_color_depth(True, "xterm", "24bit") == "24bit"

    def test_256color_via_term(self):
        assert _detect_color_depth(True, "xterm-256color", None) == "256"

    def test_basic_color(self):
        assert _detect_color_depth(True, "xterm", None) == "basic"

    def test_dumb_terminal(self):
        assert _detect_color_depth(True, "dumb", None) == "none"

    def test_no_term(self):
        assert _detect_color_depth(True, None, None) == "none"


class TestEmulatorDetection:
    """Tests for terminal emulator detection."""

    def test_term_program_used(self):
        assert _detect_emulator("iTerm.app", "xterm-256color") == "iTerm.app"

    def test_kitty_detected(self):
        assert _detect_emulator("kitty", "xterm-kitty") == "kitty"

    def test_xterm_compatible_fallback(self):
        assert _detect_emulator(None, "xterm-256color") == "xterm-compatible"

    def test_linux_console_fallback(self):
        assert _detect_emulator(None, "linux") == "linux-console"

    def test_no_emulator(self):
        assert _detect_emulator(None, None) is None


class TestGraphicsDetection:
    """Tests for graphics protocol detection."""

    def test_kitty_terminal(self):
        assert _detect_graphics("kitty", "xterm-kitty", None, True) == "kitty"

    def test_ghostty_terminal(self):
        assert _detect_graphics("ghostty", "xterm-ghostty", None, True) == "kitty"

    def test_iterm_terminal(self):
        assert _detect_graphics("iTerm.app", "xterm-256color", None, True) == "iterm"

    def test_wezterm_terminal(self):
        assert _detect_graphics("WezTerm", "xterm-256color", None, True) == "iterm"

    def test_mintty_terminal(self):
        assert _detect_graphics("mintty", "xterm-256color", None, True) == "iterm"

    def test_foot_terminal(self):
        assert _detect_graphics(None, "foot", None, True) == "sixel"

    def test_foot_extra_terminal(self):
        assert _detect_graphics(None, "foot-extra", None, True) == "sixel"

    def test_mlterm_terminal(self):
        assert _detect_graphics("mlterm", "mlterm", None, True) == "sixel"

    def test_multiplexer_disables_graphics(self):
        assert _detect_graphics("kitty", "xterm-kitty", "tmux", True) is None

    def test_screen_disables_graphics(self):
        assert _detect_graphics("iTerm.app", "screen", "screen", True) is None

    def test_unknown_terminal_no_graphics(self):
        assert _detect_graphics("some-terminal", "xterm", None, True) is None

    def test_no_terminal_no_graphics(self):
        assert _detect_graphics(None, None, None, True) is None

    def test_non_interactive_disables_graphics(self):
        assert _detect_graphics("kitty", "xterm-kitty", None, False) is None

    def test_env_override_kitty(self, monkeypatch):
        monkeypatch.setenv("JAATO_GRAPHICS_PROTOCOL", "kitty")
        # Override wins even for unknown terminal
        assert _detect_graphics(None, "dumb", None, False) == "kitty"

    def test_env_override_iterm(self, monkeypatch):
        monkeypatch.setenv("JAATO_GRAPHICS_PROTOCOL", "iterm")
        assert _detect_graphics(None, None, None, False) == "iterm"

    def test_env_override_sixel(self, monkeypatch):
        monkeypatch.setenv("JAATO_GRAPHICS_PROTOCOL", "sixel")
        assert _detect_graphics(None, None, None, False) == "sixel"

    def test_env_override_none(self, monkeypatch):
        monkeypatch.setenv("JAATO_GRAPHICS_PROTOCOL", "none")
        # "none" disables graphics even for kitty
        assert _detect_graphics("kitty", "xterm-kitty", None, True) is None

    def test_env_override_beats_multiplexer(self, monkeypatch):
        monkeypatch.setenv("JAATO_GRAPHICS_PROTOCOL", "kitty")
        # Override wins even in tmux
        assert _detect_graphics("kitty", "xterm-kitty", "tmux", True) == "kitty"

    def test_env_override_beats_non_interactive(self, monkeypatch):
        monkeypatch.setenv("JAATO_GRAPHICS_PROTOCOL", "sixel")
        # Override wins even when non-interactive
        assert _detect_graphics(None, None, None, False) == "sixel"


class TestFullDetection:
    """Tests for the full detect() function output."""

    def test_all_keys_present(self, monkeypatch):
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.delenv("STY", raising=False)
        result = detect()

        assert "interactive" in result
        assert "term" in result
        assert "term_program" in result
        assert "colorterm" in result
        assert "multiplexer" in result
        assert "color_depth" in result
        assert "emulator" in result
        assert "graphics" in result

    def test_kitty_full_detection(self, monkeypatch):
        monkeypatch.setenv("TERM", "xterm-kitty")
        monkeypatch.setenv("TERM_PROGRAM", "kitty")
        monkeypatch.setenv("COLORTERM", "truecolor")
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.delenv("STY", raising=False)
        monkeypatch.delenv("JAATO_GRAPHICS_PROTOCOL", raising=False)
        result = detect()

        assert result["emulator"] == "kitty"
        # In non-interactive test environments, color_depth and graphics
        # are "none"/None since isatty() is False
        if result["interactive"]:
            assert result["color_depth"] == "24bit"
            assert result["graphics"] == "kitty"
        else:
            assert result["color_depth"] == "none"
            assert result["graphics"] is None

    def test_tmux_full_detection(self, monkeypatch):
        monkeypatch.setenv("TERM", "screen-256color")
        monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,12345,0")
        monkeypatch.delenv("JAATO_GRAPHICS_PROTOCOL", raising=False)
        result = detect()

        assert result["multiplexer"] == "tmux"
        assert result["graphics"] is None  # Disabled in tmux
