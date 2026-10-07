"""Tests for SubagentProfile.quirks field (server 0.6.194+).

Pins:

- ``quirks`` is an optional top-level profile field; absence yields an
  empty dict, not None.
- YAML parsing accepts ``quirks: {key: value, ...}`` and round-trips
  the keys + values into ``SubagentProfile.quirks``.
- Non-dict ``quirks:`` values fall back to an empty dict (lenient
  parse — operator typos surface elsewhere via the provider's
  unknown-key warning).
- Inheritance merges quirks dict-union with child-wins-on-collision
  semantics — matches how plugin_configs / env are merged.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ..config import (
    SubagentProfile,
    build_inline_profile,
    discover_profiles,
    resolve_profiles,
)


# ----------------------------------------------------------------------
# Dataclass defaults
# ----------------------------------------------------------------------


class TestSubagentProfileQuirksDefault:
    def test_quirks_defaults_to_empty_dict(self) -> None:
        p = SubagentProfile(name="x", description="x")
        assert p.quirks == {}
        # Independent default — mutating one profile's quirks must not
        # affect another's (field(default_factory=dict)).
        p.quirks["foo"] = "bar"
        p2 = SubagentProfile(name="y", description="y")
        assert p2.quirks == {}


# ----------------------------------------------------------------------
# YAML / JSON parse — direct construction via _build_profile_from_data
# ----------------------------------------------------------------------


class TestQuirksParsing:
    def _build(self, data: dict) -> SubagentProfile:
        data = {"plugins": [], **data}
        return build_inline_profile(data, name="test")

    def test_no_quirks_key_yields_empty_dict(self) -> None:
        p = self._build({})
        assert p.quirks == {}

    def test_quirks_key_with_dict_value_parses(self) -> None:
        p = self._build({"quirks": {
            "coerce_typed_tool_args": True,
            "another_quirk": "some-value",
        }})
        assert p.quirks == {
            "coerce_typed_tool_args": True,
            "another_quirk": "some-value",
        }

    def test_quirks_with_non_string_keys_stringified(self) -> None:
        """YAML may produce int keys (e.g. ``42: true``) — config layer
        coerces to str."""
        p = self._build({"quirks": {42: True}})
        assert p.quirks == {"42": True}

    @pytest.mark.parametrize("bad", [None, [], "true", 42, True])
    def test_non_dict_quirks_falls_back_to_empty(self, bad) -> None:
        p = self._build({"quirks": bad})
        assert p.quirks == {}


# ----------------------------------------------------------------------
# Inheritance merge — quirks is dict-union, child wins on collision
# ----------------------------------------------------------------------


class TestQuirksInheritance:
    def test_child_only(self) -> None:
        parent = SubagentProfile(name="p", description="p")
        child = SubagentProfile(
            name="c", description="c",
            quirks={"coerce_typed_tool_args": True},
        )
        from ..config import _merge_profiles
        merged = _merge_profiles("c", [parent], child, {})
        assert merged is not None
        assert merged.quirks == {"coerce_typed_tool_args": True}

    def test_parent_only(self) -> None:
        parent = SubagentProfile(
            name="p", description="p",
            quirks={"some_quirk": "from-parent"},
        )
        child = SubagentProfile(name="c", description="c")
        from ..config import _merge_profiles
        merged = _merge_profiles("c", [parent], child, {})
        assert merged is not None
        assert merged.quirks == {"some_quirk": "from-parent"}

    def test_parent_and_child_union(self) -> None:
        """Disjoint keys: union — both flow into the resolved profile."""
        parent = SubagentProfile(
            name="p", description="p",
            quirks={"parent_quirk": True},
        )
        child = SubagentProfile(
            name="c", description="c",
            quirks={"child_quirk": "value"},
        )
        from ..config import _merge_profiles
        merged = _merge_profiles("c", [parent], child, {})
        assert merged is not None
        assert merged.quirks == {
            "parent_quirk": True,
            "child_quirk": "value",
        }

    def test_child_wins_on_key_collision(self) -> None:
        """Same key declared by parent and child: child wins.  Mirrors
        plugin_configs / env merge semantics so a child can disable
        a parent's quirk with ``key: false``."""
        parent = SubagentProfile(
            name="p", description="p",
            quirks={"coerce_typed_tool_args": True},
        )
        child = SubagentProfile(
            name="c", description="c",
            quirks={"coerce_typed_tool_args": False},
        )
        from ..config import _merge_profiles
        merged = _merge_profiles("c", [parent], child, {})
        assert merged is not None
        assert merged.quirks == {"coerce_typed_tool_args": False}

    def test_multi_parent_resolution_left_to_right(self) -> None:
        """When two parents declare the same quirk key with different
        values, the merge applies them in inheritance order — later
        parents overwrite earlier.  Child still wins overall."""
        p1 = SubagentProfile(
            name="p1", description="p1",
            quirks={"q": "from-p1", "shared": "p1"},
        )
        p2 = SubagentProfile(
            name="p2", description="p2",
            quirks={"q": "from-p2"},
        )
        child = SubagentProfile(
            name="c", description="c",
            quirks={"shared": "child"},
        )
        from ..config import _merge_profiles
        merged = _merge_profiles("c", [p1, p2], child, {})
        assert merged is not None
        # p2 overwrote p1's "q"; child overwrote "shared".
        assert merged.quirks == {"q": "from-p2", "shared": "child"}


# ----------------------------------------------------------------------
# End-to-end: YAML profile on disk → discovery → SubagentProfile.quirks
# ----------------------------------------------------------------------


class TestQuirksDiscoveryEndToEnd:
    def test_yaml_profile_with_quirks_round_trips(self, tmp_path: Path) -> None:
        """A profile YAML file with a top-level ``quirks:`` section
        parses through discover_profiles() into the dataclass field."""
        profile_dir = tmp_path / "profiles"
        profile_dir.mkdir()
        (profile_dir / "test-quirks.yaml").write_text(
            "name: test-quirks\n"
            "description: test profile with quirks\n"
            "model: foo\n"
            "provider: vllm\n"
            "plugins: []\n"
            "quirks:\n"
            "  coerce_typed_tool_args: true\n"
        )
        result = discover_profiles(str(profile_dir))
        assert result.errors == {}, f"unexpected errors: {result.errors}"
        assert "test-quirks" in result.profiles
        p = result.profiles["test-quirks"]
        assert p.quirks == {"coerce_typed_tool_args": True}
