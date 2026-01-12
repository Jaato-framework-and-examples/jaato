"""Tests for theme configuration and styling."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Adjust path for imports when running from project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from theme import (
    StyleSpec,
    ThemeConfig,
    load_theme,
    validate_theme,
    is_hex_color,
    BUILTIN_THEMES,
    DEFAULT_PALETTE,
    DEFAULT_SEMANTIC_STYLES,
    get_semantic_style_names,
    get_palette_color_names,
)


class TestIsHexColor:
    """Tests for hex color validation."""

    def test_valid_hex_colors(self):
        """Test valid hex colors are recognized."""
        assert is_hex_color("#000000")
        assert is_hex_color("#ffffff")
        assert is_hex_color("#FFFFFF")
        assert is_hex_color("#5fd7ff")
        assert is_hex_color("#AbCdEf")

    def test_invalid_hex_colors(self):
        """Test invalid hex colors are rejected."""
        assert not is_hex_color("000000")  # Missing #
        assert not is_hex_color("#fff")     # Too short
        assert not is_hex_color("#fffffff")  # Too long
        assert not is_hex_color("#gggggg")  # Invalid chars
        assert not is_hex_color("primary")  # Palette name
        assert not is_hex_color("")


class TestStyleSpec:
    """Tests for StyleSpec dataclass."""

    def test_default_values(self):
        """Test StyleSpec has correct defaults."""
        spec = StyleSpec()
        assert spec.fg is None
        assert spec.bg is None
        assert spec.bold is False
        assert spec.italic is False
        assert spec.dim is False
        assert spec.underline is False

    def test_custom_values(self):
        """Test StyleSpec with custom values."""
        spec = StyleSpec(
            fg="#ff0000",
            bg="#000000",
            bold=True,
            italic=True,
            dim=True,
            underline=True,
        )
        assert spec.fg == "#ff0000"
        assert spec.bg == "#000000"
        assert spec.bold is True
        assert spec.italic is True
        assert spec.dim is True
        assert spec.underline is True

    def test_to_prompt_toolkit_hex_colors(self):
        """Test conversion to prompt_toolkit style with hex colors."""
        spec = StyleSpec(fg="#5fd7ff", bg="#333333", bold=True)
        result = spec.to_prompt_toolkit({})

        assert "bg:#333333" in result
        assert "#5fd7ff" in result
        assert "bold" in result

    def test_to_prompt_toolkit_palette_reference(self):
        """Test conversion resolves palette references."""
        palette = {"primary": "#00ff00", "surface": "#1a1a1a"}
        spec = StyleSpec(fg="primary", bg="surface", italic=True)
        result = spec.to_prompt_toolkit(palette)

        assert "bg:#1a1a1a" in result
        assert "#00ff00" in result
        assert "italic" in result

    def test_to_prompt_toolkit_minimal(self):
        """Test conversion with minimal spec."""
        spec = StyleSpec(fg="#ffffff")
        result = spec.to_prompt_toolkit({})
        assert result == "#ffffff"

    def test_to_rich_hex_colors(self):
        """Test conversion to Rich style with hex colors."""
        spec = StyleSpec(fg="#5fd7ff", bg="#333333", bold=True)
        result = spec.to_rich({})

        assert "bold" in result
        assert "#5fd7ff" in result
        assert "on #333333" in result

    def test_to_rich_palette_reference(self):
        """Test Rich conversion resolves palette references."""
        palette = {"error": "#ff0000"}
        spec = StyleSpec(fg="error", dim=True)
        result = spec.to_rich(palette)

        assert "dim" in result
        assert "#ff0000" in result

    def test_to_rich_modifiers_order(self):
        """Test Rich style has modifiers before colors."""
        spec = StyleSpec(fg="#ffffff", bold=True, italic=True)
        result = spec.to_rich({})

        # Modifiers should come before color
        parts = result.split()
        color_idx = parts.index("#ffffff")
        assert "bold" in parts[:color_idx]
        assert "italic" in parts[:color_idx]

    def test_from_dict_all_fields(self):
        """Test StyleSpec.from_dict with all fields."""
        data = {
            "fg": "#ff0000",
            "bg": "#00ff00",
            "bold": True,
            "italic": True,
            "dim": True,
            "underline": True,
        }
        spec = StyleSpec.from_dict(data)

        assert spec.fg == "#ff0000"
        assert spec.bg == "#00ff00"
        assert spec.bold is True
        assert spec.italic is True
        assert spec.dim is True
        assert spec.underline is True

    def test_from_dict_partial(self):
        """Test StyleSpec.from_dict with partial fields uses defaults."""
        data = {"fg": "primary", "bold": True}
        spec = StyleSpec.from_dict(data)

        assert spec.fg == "primary"
        assert spec.bg is None
        assert spec.bold is True
        assert spec.italic is False

    def test_to_dict_non_defaults_only(self):
        """Test to_dict only includes non-default values."""
        spec = StyleSpec(fg="#ffffff", bold=True)
        result = spec.to_dict()

        assert result == {"fg": "#ffffff", "bold": True}
        assert "bg" not in result
        assert "italic" not in result

    def test_to_dict_empty_for_defaults(self):
        """Test to_dict returns empty dict for default spec."""
        spec = StyleSpec()
        result = spec.to_dict()
        assert result == {}


class TestThemeConfig:
    """Tests for ThemeConfig dataclass."""

    def test_default_values(self):
        """Test ThemeConfig has correct defaults."""
        config = ThemeConfig()

        assert config.name == "dark"
        assert config.colors == DEFAULT_PALETTE
        assert len(config.semantic) > 0
        assert not config.is_modified

    def test_resolve_color_hex(self):
        """Test resolve_color passes through hex values."""
        config = ThemeConfig()
        result = config.resolve_color("#ff0000")
        assert result == "#ff0000"

    def test_resolve_color_palette(self):
        """Test resolve_color looks up palette names."""
        config = ThemeConfig()
        result = config.resolve_color("primary")
        assert result == config.colors["primary"]

    def test_resolve_color_unknown(self):
        """Test resolve_color returns input for unknown names."""
        config = ThemeConfig()
        result = config.resolve_color("unknown_color")
        assert result == "unknown_color"

    def test_get_rich_style(self):
        """Test get_rich_style returns correct Rich style string."""
        config = ThemeConfig()
        # user_header should have fg=success, bold=True
        result = config.get_rich_style("user_header")

        assert "bold" in result
        # Should contain the resolved success color
        assert config.colors["success"] in result or "#" in result

    def test_get_rich_style_unknown(self):
        """Test get_rich_style returns empty for unknown style."""
        config = ThemeConfig()
        result = config.get_rich_style("nonexistent_style")
        assert result == ""

    def test_get_prompt_toolkit_style(self):
        """Test get_prompt_toolkit_style returns Style object."""
        config = ThemeConfig()
        style = config.get_prompt_toolkit_style()

        # Should be a prompt_toolkit Style
        from prompt_toolkit.styles import Style
        assert isinstance(style, Style)

    def test_get_color(self):
        """Test get_color returns palette color."""
        config = ThemeConfig()
        result = config.get_color("primary")
        assert result == DEFAULT_PALETTE["primary"]

    def test_get_color_unknown(self):
        """Test get_color returns white for unknown."""
        config = ThemeConfig()
        result = config.get_color("nonexistent")
        assert result == "#ffffff"

    def test_set_color_valid(self):
        """Test set_color with valid hex color."""
        config = ThemeConfig()
        result = config.set_color("primary", "#aabbcc")

        assert result is True
        assert config.colors["primary"] == "#aabbcc"
        assert config.is_modified

    def test_set_color_invalid_hex(self):
        """Test set_color rejects invalid hex."""
        config = ThemeConfig()
        original = config.colors["primary"]
        result = config.set_color("primary", "not-a-color")

        assert result is False
        assert config.colors["primary"] == original

    def test_set_color_unknown_name(self):
        """Test set_color rejects unknown color names."""
        config = ThemeConfig()
        result = config.set_color("unknown_color", "#aabbcc")
        assert result is False

    def test_set_semantic_style(self):
        """Test set_semantic_style."""
        config = ThemeConfig()
        new_spec = StyleSpec(fg="#ff0000", bold=True)
        result = config.set_semantic_style("user_header", new_spec)

        assert result is True
        assert config.semantic["user_header"].fg == "#ff0000"
        assert config.is_modified

    def test_set_semantic_style_unknown(self):
        """Test set_semantic_style rejects unknown styles."""
        config = ThemeConfig()
        new_spec = StyleSpec(fg="#ff0000")
        result = config.set_semantic_style("unknown_style", new_spec)
        assert result is False

    def test_from_dict_complete(self):
        """Test ThemeConfig.from_dict with complete data."""
        data = {
            "name": "custom",
            "description": "My theme",
            "colors": {
                "primary": "#aabbcc",
            },
            "semantic": {
                "user_header": {"fg": "primary", "bold": True},
            },
        }
        config = ThemeConfig.from_dict(data)

        assert config.name == "custom"
        assert config.description == "My theme"
        assert config.colors["primary"] == "#aabbcc"
        assert config.semantic["user_header"].fg == "primary"
        assert config.semantic["user_header"].bold is True

    def test_from_dict_preserves_defaults(self):
        """Test from_dict preserves defaults for missing fields."""
        data = {"name": "minimal"}
        config = ThemeConfig.from_dict(data)

        # Should have all default colors
        assert "primary" in config.colors
        assert "success" in config.colors

        # Should have all default semantic styles
        assert "user_header" in config.semantic
        assert "tool_output" in config.semantic

    def test_from_file_valid(self):
        """Test loading theme from valid JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            theme_data = {
                "name": "test-theme",
                "colors": {"primary": "#123456"},
            }
            theme_path = Path(temp_dir) / "theme.json"
            with open(theme_path, "w") as f:
                json.dump(theme_data, f)

            config = ThemeConfig.from_file(theme_path)

            assert config is not None
            assert config.name == "test-theme"
            assert config.colors["primary"] == "#123456"
            assert config.source_path == str(theme_path)

    def test_from_file_missing(self):
        """Test from_file returns None for missing file."""
        config = ThemeConfig.from_file(Path("/nonexistent/theme.json"))
        assert config is None

    def test_from_file_invalid_json(self):
        """Test from_file returns None for invalid JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            theme_path = Path(temp_dir) / "theme.json"
            with open(theme_path, "w") as f:
                f.write("{ invalid json }")

            config = ThemeConfig.from_file(theme_path)
            assert config is None

    def test_save_and_load(self):
        """Test saving and loading theme preserves data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ThemeConfig(name="saved-theme")
            config.set_color("primary", "#abcdef")
            config.set_semantic_style("user_header", StyleSpec(fg="#ff0000", italic=True))

            save_path = Path(temp_dir) / "saved.json"
            result = config.save(save_path)

            assert result is True
            assert save_path.exists()
            assert not config.is_modified  # Should be cleared

            # Load and verify
            loaded = ThemeConfig.from_file(save_path)
            assert loaded is not None
            assert loaded.name == "saved-theme"
            assert loaded.colors["primary"] == "#abcdef"
            assert loaded.semantic["user_header"].fg == "#ff0000"
            assert loaded.semantic["user_header"].italic is True

    def test_to_dict(self):
        """Test to_dict exports configuration."""
        config = ThemeConfig(name="export-test", description="Test theme")
        result = config.to_dict()

        assert result["name"] == "export-test"
        assert result["description"] == "Test theme"
        assert result["version"] == "1.0"
        assert "colors" in result
        assert "semantic" in result

    def test_copy(self):
        """Test copy creates independent copy."""
        original = ThemeConfig(name="original")
        original.set_color("primary", "#111111")

        copy = original.copy()
        copy.set_color("primary", "#222222")

        assert original.colors["primary"] == "#111111"
        assert copy.colors["primary"] == "#222222"


class TestLoadTheme:
    """Tests for load_theme function."""

    def test_env_var_preset_dark(self):
        """Test JAATO_THEME=dark selects dark preset."""
        with patch.dict(os.environ, {"JAATO_THEME": "dark"}):
            config = load_theme()
            assert config.name == "dark"

    def test_env_var_preset_light(self):
        """Test JAATO_THEME=light selects light preset."""
        with patch.dict(os.environ, {"JAATO_THEME": "light"}):
            config = load_theme()
            assert config.name == "light"

    def test_env_var_preset_high_contrast(self):
        """Test JAATO_THEME=high-contrast selects high-contrast preset."""
        with patch.dict(os.environ, {"JAATO_THEME": "high-contrast"}):
            config = load_theme()
            assert config.name == "high-contrast"

    def test_env_var_unknown_preset(self):
        """Test unknown JAATO_THEME falls through to file lookup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"JAATO_THEME": "unknown"}):
                # Should fall back to default since no files exist
                config = load_theme(
                    project_path=str(Path(temp_dir) / "project.json"),
                    user_path=str(Path(temp_dir) / "user.json"),
                )
                assert config.name == "dark"  # Default

    def test_project_config_priority(self):
        """Test project config takes priority over user config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create project config
            project_path = Path(temp_dir) / "project" / "theme.json"
            project_path.parent.mkdir()
            with open(project_path, "w") as f:
                json.dump({"name": "project-theme"}, f)

            # Create user config
            user_path = Path(temp_dir) / "user" / "theme.json"
            user_path.parent.mkdir()
            with open(user_path, "w") as f:
                json.dump({"name": "user-theme"}, f)

            # Clear env var
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("JAATO_THEME", None)
                config = load_theme(
                    project_path=str(project_path),
                    user_path=str(user_path),
                )

            assert config.name == "project-theme"

    def test_user_config_fallback(self):
        """Test user config is used when project config missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # No project config

            # Create user config
            user_path = Path(temp_dir) / "user" / "theme.json"
            user_path.parent.mkdir()
            with open(user_path, "w") as f:
                json.dump({"name": "user-theme"}, f)

            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("JAATO_THEME", None)
                config = load_theme(
                    project_path=str(Path(temp_dir) / "nonexistent.json"),
                    user_path=str(user_path),
                )

            assert config.name == "user-theme"

    def test_default_fallback(self):
        """Test default theme when no config found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("JAATO_THEME", None)
                config = load_theme(
                    project_path=str(Path(temp_dir) / "project.json"),
                    user_path=str(Path(temp_dir) / "user.json"),
                )

            assert config.name == "dark"


class TestValidateTheme:
    """Tests for validate_theme function."""

    def test_valid_theme(self):
        """Test validation passes for valid theme."""
        data = {
            "version": "1.0",
            "colors": {"primary": "#aabbcc"},
            "semantic": {"user_header": {"fg": "primary", "bold": True}},
        }
        is_valid, errors = validate_theme(data)

        assert is_valid is True
        assert len(errors) == 0

    def test_invalid_version(self):
        """Test validation fails for unsupported version."""
        data = {"version": "2.0"}
        is_valid, errors = validate_theme(data)

        assert is_valid is False
        assert any("version" in e.lower() for e in errors)

    def test_invalid_hex_color(self):
        """Test validation fails for invalid hex color."""
        data = {"colors": {"primary": "not-a-color"}}
        is_valid, errors = validate_theme(data)

        assert is_valid is False
        assert any("primary" in e for e in errors)

    def test_unknown_palette_color(self):
        """Test validation fails for unknown palette color."""
        data = {"colors": {"unknown_color": "#aabbcc"}}
        is_valid, errors = validate_theme(data)

        assert is_valid is False
        assert any("unknown_color" in e for e in errors)

    def test_unknown_semantic_style(self):
        """Test validation fails for unknown semantic style."""
        data = {"semantic": {"unknown_style": {"fg": "#ffffff"}}}
        is_valid, errors = validate_theme(data)

        assert is_valid is False
        assert any("unknown_style" in e for e in errors)

    def test_invalid_semantic_color_reference(self):
        """Test validation fails for invalid color reference in semantic."""
        data = {"semantic": {"user_header": {"fg": "nonexistent_color"}}}
        is_valid, errors = validate_theme(data)

        assert is_valid is False
        assert any("user_header" in e for e in errors)


class TestBuiltinThemes:
    """Tests for built-in theme presets."""

    def test_dark_theme_exists(self):
        """Test dark theme is available."""
        assert "dark" in BUILTIN_THEMES
        theme = BUILTIN_THEMES["dark"]
        assert theme.name == "dark"

    def test_light_theme_exists(self):
        """Test light theme is available."""
        assert "light" in BUILTIN_THEMES
        theme = BUILTIN_THEMES["light"]
        assert theme.name == "light"

    def test_high_contrast_theme_exists(self):
        """Test high-contrast theme is available."""
        assert "high-contrast" in BUILTIN_THEMES
        theme = BUILTIN_THEMES["high-contrast"]
        assert theme.name == "high-contrast"

    def test_themes_have_all_palette_colors(self):
        """Test all themes have all palette colors defined."""
        for name, theme in BUILTIN_THEMES.items():
            for color in get_palette_color_names():
                assert color in theme.colors, f"{name} missing {color}"

    def test_themes_have_all_semantic_styles(self):
        """Test all themes have all semantic styles defined."""
        for name, theme in BUILTIN_THEMES.items():
            for style in get_semantic_style_names():
                assert style in theme.semantic, f"{name} missing {style}"


class TestHelperFunctions:
    """Tests for module helper functions."""

    def test_get_semantic_style_names(self):
        """Test get_semantic_style_names returns sorted list."""
        names = get_semantic_style_names()
        assert len(names) > 0
        assert names == sorted(names)
        assert "user_header" in names
        assert "tool_output" in names

    def test_get_palette_color_names(self):
        """Test get_palette_color_names returns ordered list."""
        names = get_palette_color_names()
        assert len(names) == 11
        assert "primary" in names
        assert "success" in names
        assert "text" in names
