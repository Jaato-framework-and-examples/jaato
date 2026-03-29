"""Tests for profile inheritance resolution."""

import pytest

from shared.plugins.subagent.config import (
    SubagentProfile,
    GCProfileConfig,
    _normalize_inherits,
    resolve_profiles,
)


class TestNormalizeInherits:
    """Tests for _normalize_inherits helper."""

    def test_none(self):
        assert _normalize_inherits(None) is None

    def test_single_string(self):
        assert _normalize_inherits("readonly") == ["readonly"]

    def test_list(self):
        assert _normalize_inherits(["a", "b"]) == ["a", "b"]

    def test_empty_list(self):
        assert _normalize_inherits([]) == []

    def test_non_string(self):
        assert _normalize_inherits(42) is None


class TestResolveSingleInheritance:
    """Tests for single-parent inheritance."""

    def test_child_gets_parent_plugins(self):
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base",
                plugins=["cli", "memory"],
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                plugins=["web_search"],
                inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].plugins == ["cli", "memory", "web_search"]
        assert resolved["child"].inherits is None  # flattened

    def test_child_gets_parent_env(self):
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base",
                env={"API_KEY": "123"},
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                env={"OTHER": "456"},
                inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].env == {"API_KEY": "123", "OTHER": "456"}

    def test_child_overrides_parent_env(self):
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base",
                env={"KEY": "old"},
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                env={"KEY": "new"},
                inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].env["KEY"] == "new"

    def test_child_inherits_model(self):
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base",
                model="gemini-2.5-flash",
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].model == "gemini-2.5-flash"

    def test_child_overrides_model(self):
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base",
                model="gemini-2.5-flash",
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                model="claude-sonnet-4-20250514",
                inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].model == "claude-sonnet-4-20250514"

    def test_system_instructions_concatenated(self):
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base",
                system_instructions="You are helpful.",
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                system_instructions="Cite sources.",
                inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].system_instructions == "You are helpful.\n\nCite sources."

    def test_description_not_inherited(self):
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base description",
            ),
            "child": SubagentProfile(
                name="child", description="Child description",
                inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].description == "Child description"

    def test_plugin_configs_deep_merge(self):
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base",
                plugin_configs={"cli": {"timeout": 30}, "memory": {"path": "/tmp"}},
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                plugin_configs={"cli": {"max_output": 5000}},
                inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].plugin_configs["cli"] == {"timeout": 30, "max_output": 5000}
        assert resolved["child"].plugin_configs["memory"] == {"path": "/tmp"}

    def test_preloaded_plugins_union(self):
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base",
                preloaded_plugins={"cli", "memory"},
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                preloaded_plugins={"todo"},
                inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].preloaded_plugins == {"cli", "memory", "todo"}


class TestResolveMultipleInheritance:
    """Tests for multiple-parent inheritance."""

    def test_plugins_union_from_two_parents(self):
        profiles = {
            "readonly": SubagentProfile(
                name="readonly", description="Read-only",
                plugins=["filesystem_query", "memory"],
            ),
            "web": SubagentProfile(
                name="web", description="Web",
                plugins=["web_search"],
            ),
            "researcher": SubagentProfile(
                name="researcher", description="Researcher",
                inherits=["readonly", "web"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["researcher"].plugins == ["filesystem_query", "memory", "web_search"]

    def test_scalar_conflict_between_parents_is_error(self):
        profiles = {
            "fast": SubagentProfile(
                name="fast", description="Fast",
                model="claude-haiku-4-5-20251001",
            ),
            "slow": SubagentProfile(
                name="slow", description="Slow",
                model="claude-sonnet-4-20250514",
            ),
            "broken": SubagentProfile(
                name="broken", description="Broken",
                inherits=["fast", "slow"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert "broken" in errors
        assert "model" in errors["broken"]

    def test_scalar_conflict_resolved_by_child_override(self):
        profiles = {
            "fast": SubagentProfile(
                name="fast", description="Fast",
                model="claude-haiku-4-5-20251001",
            ),
            "slow": SubagentProfile(
                name="slow", description="Slow",
                model="claude-sonnet-4-20250514",
            ),
            "fixed": SubagentProfile(
                name="fixed", description="Fixed",
                model="claude-opus-4-20250514",
                inherits=["fast", "slow"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["fixed"].model == "claude-opus-4-20250514"

    def test_env_conflict_between_parents_is_error(self):
        profiles = {
            "a": SubagentProfile(
                name="a", description="A",
                env={"KEY": "val_a"},
            ),
            "b": SubagentProfile(
                name="b", description="B",
                env={"KEY": "val_b"},
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                inherits=["a", "b"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert "child" in errors
        assert "env" in errors["child"]

    def test_env_conflict_resolved_by_child(self):
        profiles = {
            "a": SubagentProfile(
                name="a", description="A",
                env={"KEY": "val_a"},
            ),
            "b": SubagentProfile(
                name="b", description="B",
                env={"KEY": "val_b"},
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                env={"KEY": "val_child"},
                inherits=["a", "b"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].env["KEY"] == "val_child"

    def test_max_turns_uses_minimum(self):
        profiles = {
            "a": SubagentProfile(
                name="a", description="A",
                max_turns=5,
            ),
            "b": SubagentProfile(
                name="b", description="B",
                max_turns=20,
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                inherits=["a", "b"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].max_turns == 5


class TestResolveDeepChain:
    """Tests for multi-level inheritance chains."""

    def test_three_level_chain(self):
        profiles = {
            "grandparent": SubagentProfile(
                name="grandparent", description="GP",
                plugins=["cli"],
                env={"LEVEL": "0"},
            ),
            "parent": SubagentProfile(
                name="parent", description="P",
                plugins=["memory"],
                env={"LEVEL": "1"},
                inherits=["grandparent"],
            ),
            "child": SubagentProfile(
                name="child", description="C",
                plugins=["todo"],
                env={"LEVEL": "2"},
                inherits=["parent"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].plugins == ["cli", "memory", "todo"]
        assert resolved["child"].env["LEVEL"] == "2"  # child wins

    def test_system_instructions_concatenation_order(self):
        profiles = {
            "gp": SubagentProfile(
                name="gp", description="GP",
                system_instructions="Rule 1.",
            ),
            "parent": SubagentProfile(
                name="parent", description="P",
                system_instructions="Rule 2.",
                inherits=["gp"],
            ),
            "child": SubagentProfile(
                name="child", description="C",
                system_instructions="Rule 3.",
                inherits=["parent"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].system_instructions == "Rule 1.\n\nRule 2.\n\nRule 3."


class TestCycleDetection:
    """Tests for inheritance cycle detection."""

    def test_direct_cycle(self):
        profiles = {
            "a": SubagentProfile(
                name="a", description="A",
                inherits=["b"],
            ),
            "b": SubagentProfile(
                name="b", description="B",
                inherits=["a"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert errors  # At least one cycle error

    def test_indirect_cycle(self):
        profiles = {
            "a": SubagentProfile(name="a", description="A", inherits=["b"]),
            "b": SubagentProfile(name="b", description="B", inherits=["c"]),
            "c": SubagentProfile(name="c", description="C", inherits=["a"]),
        }
        resolved, errors = resolve_profiles(profiles)
        assert errors

    def test_self_reference(self):
        profiles = {
            "a": SubagentProfile(name="a", description="A", inherits=["a"]),
        }
        resolved, errors = resolve_profiles(profiles)
        assert "a" in errors
        assert "a" not in resolved


class TestMissingParent:
    """Tests for missing parent references."""

    def test_missing_parent(self):
        profiles = {
            "child": SubagentProfile(
                name="child", description="Child",
                inherits=["nonexistent"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert "child" in errors
        assert "nonexistent" in errors["child"]

    def test_valid_profiles_still_resolve(self):
        """Profiles without broken inheritance should still resolve."""
        profiles = {
            "good": SubagentProfile(name="good", description="Good"),
            "broken": SubagentProfile(
                name="broken", description="Broken",
                inherits=["nonexistent"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert "good" in resolved
        assert "broken" in errors


class TestNoInheritance:
    """Tests for profiles without inheritance."""

    def test_profiles_without_inherits_pass_through(self):
        profiles = {
            "a": SubagentProfile(name="a", description="A", plugins=["cli"]),
            "b": SubagentProfile(name="b", description="B", plugins=["memory"]),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["a"].plugins == ["cli"]
        assert resolved["b"].plugins == ["memory"]

    def test_empty_dict(self):
        resolved, errors = resolve_profiles({})
        assert not errors
        assert resolved == {}
