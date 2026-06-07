"""Tests for the required ``plugins:`` key on profiles (server 0.6.x+).

Pins the contract introduced 2026-06-07 in response to the vLLM smoke
investigation:

- ``plugins:`` MUST be present on every workspace / user / premium
  profile and every inline session spec.
- Absence → clear validation error naming the missing key.
- ``plugins: []`` (explicit empty) → the minimal framework set
  (permission, reliability, lifecycle); no other tool plugins are
  exposed.

Paired with the unconditional ``kwargs["tools"] = self._profile.plugins``
pass-through in ``server/core.py:_apply_profile_overrides`` — together
they eliminate the ambiguity where an absent ``plugins:`` was silently
expanded to "all exposed plugins".
"""

from pathlib import Path

import json
import pytest


def _write_profile(dir_path: Path, name: str, data: dict) -> None:
    """Write a JSON profile to ``dir_path / f"{name}.json"``."""
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{name}.json").write_text(json.dumps(data))


# ============================================================
# discover_profiles — workspace / user tier (file-based)
# ============================================================


class TestWorkspaceProfilePluginsRequired:
    """Pin: workspace-tier profiles missing the ``plugins`` key are
    rejected with a clear error string referencing the key name."""

    def test_profile_without_plugins_key_is_rejected(self, tmp_path: Path):
        from shared.plugins.subagent.config import discover_profiles

        _write_profile(
            tmp_path / ".jaato" / "profiles",
            "no-plugins-key",
            {
                "name": "no-plugins-key",
                "description": "Missing the plugins: key entirely.",
                "model": "test-model",
                # NOTE: no 'plugins' key.
            },
        )
        result = discover_profiles(
            ".jaato/profiles",
            base_path=str(tmp_path),
        )

        assert "no-plugins-key" not in result.profiles, (
            "Profile missing 'plugins:' must NOT be returned in the "
            "profiles dict"
        )
        assert "no-plugins-key" in result.errors
        err = result.errors["no-plugins-key"]
        assert "'plugins'" in err
        assert "required" in err.lower()

    def test_profile_with_explicit_empty_plugins_is_accepted(
        self, tmp_path: Path,
    ):
        """``plugins: []`` is the canonical "minimal framework set" shape
        and MUST be accepted without error."""
        from shared.plugins.subagent.config import discover_profiles

        _write_profile(
            tmp_path / ".jaato" / "profiles",
            "empty-plugins",
            {
                "name": "empty-plugins",
                "description": "Explicit empty plugins list.",
                "model": "test-model",
                "plugins": [],
            },
        )
        result = discover_profiles(
            ".jaato/profiles",
            base_path=str(tmp_path),
        )

        assert "empty-plugins" in result.profiles
        profile = result.profiles["empty-plugins"]
        assert profile.plugins == []
        # No error recorded.
        assert "empty-plugins" not in result.errors

    def test_profile_with_plugin_list_is_accepted(self, tmp_path: Path):
        from shared.plugins.subagent.config import discover_profiles

        _write_profile(
            tmp_path / ".jaato" / "profiles",
            "named-plugins",
            {
                "name": "named-plugins",
                "description": "Explicit plugin list.",
                "model": "test-model",
                "plugins": ["cli", "todo"],
            },
        )
        result = discover_profiles(
            ".jaato/profiles",
            base_path=str(tmp_path),
        )

        assert "named-plugins" in result.profiles
        profile = result.profiles["named-plugins"]
        assert profile.plugins == ["cli", "todo"]

    def test_one_bad_profile_does_not_block_other_profiles(
        self, tmp_path: Path,
    ):
        """A profile missing ``plugins:`` is rejected, but its siblings
        still load — pin the per-profile error isolation."""
        from shared.plugins.subagent.config import discover_profiles

        profiles_dir = tmp_path / ".jaato" / "profiles"
        _write_profile(profiles_dir, "good", {
            "name": "good",
            "model": "test-model",
            "plugins": [],
        })
        _write_profile(profiles_dir, "bad", {
            "name": "bad",
            "model": "test-model",
            # missing plugins
        })

        result = discover_profiles(
            ".jaato/profiles",
            base_path=str(tmp_path),
        )

        assert "good" in result.profiles
        assert "bad" not in result.profiles
        assert "bad" in result.errors


# ============================================================
# build_inline_profile — IPC / API inline specs
# ============================================================


class TestInlineSpecPluginsRequired:
    """Pin: inline session specs (sent over IPC / WS) without
    ``plugins`` raise ``ValueError`` immediately."""

    def test_inline_spec_without_plugins_raises(self):
        from shared.plugins.subagent.config import build_inline_profile

        with pytest.raises(ValueError, match="plugins"):
            build_inline_profile({
                "name": "no-plugins-key",
                "model": "test-model",
            })

    def test_inline_spec_with_empty_plugins_succeeds(self):
        from shared.plugins.subagent.config import build_inline_profile

        profile = build_inline_profile({
            "name": "empty-plugins",
            "model": "test-model",
            "plugins": [],
        })
        assert profile.plugins == []

    def test_inline_spec_with_plugin_list_succeeds(self):
        from shared.plugins.subagent.config import build_inline_profile

        profile = build_inline_profile({
            "name": "named",
            "model": "test-model",
            "plugins": ["cli"],
        })
        assert profile.plugins == ["cli"]
