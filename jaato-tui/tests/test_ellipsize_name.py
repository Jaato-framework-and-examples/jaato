"""Tests for the ellipsize_name utility function."""

import pytest

from ui_utils import ellipsize_name


class TestEllipsizeName:
    """Tests for middle-ellipsis name truncation."""

    def test_short_name_unchanged(self):
        """Names shorter than max_width should not be modified."""
        assert ellipsize_name("agent-1", 15) == "agent-1"

    def test_exact_length_unchanged(self):
        """Names exactly at max_width should not be modified."""
        assert ellipsize_name("abcde", 5) == "abcde"

    def test_middle_ellipsis_basic(self):
        """Long names should be ellipsized in the middle."""
        result = ellipsize_name("validator-tier3-secondary", 15)
        assert len(result) == 15
        assert "…" in result
        # Head comes from the start, tail from the end
        assert result.startswith("validat") or result.startswith("valida")
        assert result.endswith("econdary") or result.endswith("condary")

    def test_preserves_start_and_end(self):
        """Both the beginning and end of the name should be visible."""
        name = "research-agent-tier3"
        result = ellipsize_name(name, 12)
        assert result.startswith(name[:3])  # at least first 3 chars
        assert result.endswith(name[-3:])  # at least last 3 chars
        assert len(result) == 12

    def test_bias_toward_tail(self):
        """Tail should get slightly more characters than head on odd splits."""
        # max_width=8, ellipsis="…" (1 char) → 7 available → tail=4, head=3
        result = ellipsize_name("abcdefghij", 8)
        assert result == "abc…ghij"

    def test_even_split(self):
        """Even available space should split equally."""
        # max_width=7, ellipsis="…" (1 char) → 6 available → tail=3, head=3
        result = ellipsize_name("abcdefghij", 7)
        assert result == "abc…hij"

    def test_empty_string(self):
        """Empty string should be returned as-is."""
        assert ellipsize_name("", 10) == ""

    def test_zero_max_width(self):
        """Zero max_width should return the original name."""
        assert ellipsize_name("hello", 0) == "hello"

    def test_very_small_max_width(self):
        """Very small max_width that can't fit ellipsis + 2 chars
        should hard-truncate."""
        # "…" is 1 char, so need at least 3 (1 + 2) for middle ellipsis
        result = ellipsize_name("abcdefgh", 2)
        assert result == "ab"
        assert len(result) == 2

    def test_minimum_ellipsis_width(self):
        """Minimum width for middle ellipsis is len(ellipsis) + 2."""
        # "…" = 1 char, so min is 3: "a…h"
        result = ellipsize_name("abcdefgh", 3)
        assert result == "a…h"
        assert len(result) == 3

    def test_custom_ellipsis(self):
        """Custom ellipsis string should work."""
        result = ellipsize_name("abcdefghij", 8, ellipsis="...")
        # "..." = 3 chars, 8 - 3 = 5 available → tail=3, head=2
        assert result == "ab...hij"
        assert len(result) == 8

    def test_realistic_agent_id(self):
        """Realistic agent IDs like those from the tab bar."""
        # "validator-tier3" is 15 chars, fits in 15
        assert ellipsize_name("validator-tier3", 15) == "validator-tier3"

        # "validator-tier3-extra" is 20 chars, needs truncation at 15
        result = ellipsize_name("validator-tier3-extra", 15)
        assert len(result) == 15
        assert "…" in result
        assert result.endswith("extra")  # suffix preserved

    def test_single_char_name(self):
        """Single character names should be unchanged."""
        assert ellipsize_name("a", 5) == "a"
        assert ellipsize_name("a", 1) == "a"

    def test_none_returns_none(self):
        """None input should return None."""
        assert ellipsize_name(None, 10) is None
