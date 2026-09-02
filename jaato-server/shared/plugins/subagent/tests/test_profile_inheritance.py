"""Tests for profile inheritance resolution."""

import pytest

from shared.plugins.subagent.config import (
    CompletionProcessor,
    SubagentProfile,
    GCProfileConfig,
    _normalize_inherits,
    _normalize_suppress_inherited_processors,
    _parse_completion_processors,
    resolve_profiles,
)
from shared.runtime_limits import RuntimeLimits


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


class TestSuppressBaseInstructionsMerge:
    """``suppress_base_instructions`` merges by UNION across the chain — a
    piece dropped by any layer stays dropped (a minimalist parent can't be
    silently overridden)."""

    def test_union_of_parent_and_child_pieces(self):
        profiles = {
            # Parent drops only the framework constants.
            "base": SubagentProfile(
                name="base", description="Base",
                suppress_base_instructions={"constants": True},
            ),
            # Child drops only the disk layer.
            "child": SubagentProfile(
                name="child", description="Child",
                suppress_base_instructions={"disk": True},
                inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        # Union: both pieces suppressed.
        assert resolved["child"].suppress_base_instructions == frozenset(
            {"disk", "constants"}
        )

    def test_child_empty_still_inherits_parent_suppression(self):
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base",
                suppress_base_instructions=True,  # -> {disk, constants}
            ),
            "child": SubagentProfile(
                name="child", description="Child", inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].suppress_base_instructions == frozenset(
            {"disk", "constants"}
        )


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

    def test_child_inherits_runtime_limits(self):
        limits = RuntimeLimits(memory_max_mb=512, pids_max=256)
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base",
                runtime_limits=limits,
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].runtime_limits == limits

    def test_child_overrides_runtime_limits(self):
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base",
                runtime_limits=RuntimeLimits(memory_max_mb=512),
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                runtime_limits=RuntimeLimits(memory_max_mb=2048),
                inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].runtime_limits.memory_max_mb == 2048

    def test_runtime_limits_conflict_between_parents_is_error(self):
        # Two parents declaring different limits without child override
        # must produce a conflict (scalar-override semantics).
        profiles = {
            "p1": SubagentProfile(
                name="p1", description="",
                runtime_limits=RuntimeLimits(memory_max_mb=512),
            ),
            "p2": SubagentProfile(
                name="p2", description="",
                runtime_limits=RuntimeLimits(memory_max_mb=2048),
            ),
            "child": SubagentProfile(
                name="child", description="",
                inherits=["p1", "p2"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert "child" in errors
        assert "runtime_limits" in errors["child"]

    def test_runtime_limits_parents_agree_no_conflict(self):
        # Two parents declaring the *same* limits should merge without
        # conflict — frozen dataclass equality drives the agreement check.
        limits = RuntimeLimits(memory_max_mb=512, pids_max=256)
        profiles = {
            "p1": SubagentProfile(
                name="p1", description="", runtime_limits=limits,
            ),
            "p2": SubagentProfile(
                name="p2", description="", runtime_limits=limits,
            ),
            "child": SubagentProfile(
                name="child", description="",
                inherits=["p1", "p2"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].runtime_limits == limits

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

    def test_tool_scopes_per_plugin_override(self):
        # Parent scopes file_edit + cli; child re-scopes file_edit (wins)
        # and leaves cli untouched (parent's scope survives).
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base",
                tool_scopes={"file_edit": ["readFile"], "cli": ["run"]},
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                tool_scopes={"file_edit": ["readFile", "writeFile"]},
                inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].tool_scopes == {
            "file_edit": ["readFile", "writeFile"],  # child wins
            "cli": ["run"],                            # parent survives
        }

    def test_tool_scopes_empty_when_unset(self):
        profiles = {
            "base": SubagentProfile(name="base", description="Base"),
            "child": SubagentProfile(
                name="child", description="Child", inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].tool_scopes == {}


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


class TestCompletionPayloadSchemaInheritance:
    """Scalar-override semantics for completion_payload_schema."""

    SAMPLE = {"type": "object", "properties": {"summary": {"type": "string"}}}
    OTHER = {"type": "object", "properties": {"foo": {"type": "string"}}}

    def test_child_inherits_parent_schema(self):
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base",
                completion_payload_schema=self.SAMPLE,
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].completion_payload_schema == self.SAMPLE

    def test_child_overrides_parent_schema(self):
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base",
                completion_payload_schema=self.SAMPLE,
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                completion_payload_schema=self.OTHER,
                inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].completion_payload_schema == self.OTHER

    def test_two_parents_disagree_is_conflict(self):
        profiles = {
            "p1": SubagentProfile(
                name="p1", description="P1",
                completion_payload_schema=self.SAMPLE,
            ),
            "p2": SubagentProfile(
                name="p2", description="P2",
                completion_payload_schema=self.OTHER,
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                inherits=["p1", "p2"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert "child" in errors
        assert "completion_payload_schema" in errors["child"]

    def test_two_parents_disagree_child_overrides_resolves(self):
        profiles = {
            "p1": SubagentProfile(
                name="p1", description="P1",
                completion_payload_schema=self.SAMPLE,
            ),
            "p2": SubagentProfile(
                name="p2", description="P2",
                completion_payload_schema=self.OTHER,
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                completion_payload_schema=self.SAMPLE,
                inherits=["p1", "p2"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].completion_payload_schema == self.SAMPLE

    def test_path_string_inherited(self):
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base",
                completion_payload_schema="triage.json",
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].completion_payload_schema == "triage.json"

    def test_no_schema_anywhere_stays_none(self):
        profiles = {
            "base": SubagentProfile(name="base", description="Base"),
            "child": SubagentProfile(
                name="child", description="Child",
                inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].completion_payload_schema is None


class TestApparmorInheritance:
    """PR-A (2026-05-14): ``apparmor`` follows OR semantics across the
    inheritance chain.  Once any layer (parents or child) sets it True,
    the resolved profile is confined.  Rationale matches
    ``suppress_base_instructions``: a security primitive shouldn't be
    silently downgradeable by an inheritor.
    """

    def test_default_unset_resolves_false(self):
        profiles = {
            "base": SubagentProfile(name="base", description="Base"),
            "child": SubagentProfile(
                name="child", description="Child", inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].apparmor is False

    def test_parent_true_propagates_to_child(self):
        """Parent declares apparmor=True; child without an explicit
        value inherits it (the security gradient flows downward)."""
        profiles = {
            "secure_base": SubagentProfile(
                name="secure_base", description="Base", apparmor=True,
            ),
            "child": SubagentProfile(
                name="child", description="Child", inherits=["secure_base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].apparmor is True

    def test_child_true_with_unconfined_parent_still_true(self):
        """Child opts in explicitly even if parents didn't."""
        profiles = {
            "loose": SubagentProfile(name="loose", description="Loose"),
            "child": SubagentProfile(
                name="child", description="Child",
                inherits=["loose"], apparmor=True,
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].apparmor is True

    def test_child_cannot_disable_parent_confinement(self):
        """OR semantics: an inheritor setting apparmor=False
        cannot silently downgrade a confined parent.  This matches
        ``suppress_base_instructions``'s rationale — security
        primitives are one-way (off → on, never back).  An inheritor
        that genuinely wants unconfined operation should not inherit
        from a confined parent."""
        profiles = {
            "confined": SubagentProfile(
                name="confined", description="Confined", apparmor=True,
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                inherits=["confined"], apparmor=False,
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        # Parent's True wins via OR semantics.
        assert resolved["child"].apparmor is True

    def test_multi_parent_any_true_wins(self):
        """One confined parent + one unconfined parent → resolved True."""
        profiles = {
            "p_confined": SubagentProfile(
                name="p_confined", description="P1", apparmor=True,
            ),
            "p_loose": SubagentProfile(
                name="p_loose", description="P2", apparmor=False,
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                inherits=["p_confined", "p_loose"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].apparmor is True


class TestApparmorFragmentsInheritance:
    """Piece 1 (2026-05-14): ``apparmor_fragments`` follows
    CHILD-REPLACES-PARENT semantics across the inheritance chain.
    Distinct from ``apparmor`` (bool, OR-merged) above: this is the
    list of *which* fragments to compose, and replace lets cascade
    authors SCOPE DOWN from a broader parent set (least-privilege).
    """

    def test_absent_everywhere_resolves_none(self):
        """No declaration anywhere → None.  Workspace-default
        "compose all fragments" applies at render time
        (back-compat for non-cascade workspaces)."""
        profiles = {
            "base": SubagentProfile(name="base", description="Base"),
            "child": SubagentProfile(
                name="child", description="Child", inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].apparmor_fragments is None

    def test_parent_declaration_propagates_to_child(self):
        """Parent declares a list; child inherits it verbatim when
        child doesn't override.  Pattern for cascade base profile
        declaring the cross-stage commons."""
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base",
                apparmor_fragments=["jaato-kb-enablement-2"],
            ),
            "child": SubagentProfile(
                name="child", description="Child", inherits=["base"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].apparmor_fragments == ["jaato-kb-enablement-2"]

    def test_child_replaces_parent_list(self):
        """Child's list REPLACES parent's — does not union.  This is
        the load-bearing semantic for least-privilege scoping:
        host_validator declares ``[host_validator]`` and gets ONLY
        that fragment, not the union with the parent's set."""
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base",
                apparmor_fragments=["jaato-kb-enablement-2"],
            ),
            "host_validator": SubagentProfile(
                name="host_validator", description="Host validator",
                inherits=["base"],
                apparmor_fragments=["host_validator"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        # NOT the union with parent's ["jaato-kb-enablement-2"].
        assert resolved["host_validator"].apparmor_fragments == ["host_validator"]

    def test_child_empty_list_replaces_parent(self):
        """Explicit ``apparmor_fragments: []`` on the child is the
        maximally locked-down stage — composes NO fragments, even
        if the parent declared some."""
        profiles = {
            "base": SubagentProfile(
                name="base", description="Base",
                apparmor_fragments=["jaato-kb-enablement-2"],
            ),
            "locked_down": SubagentProfile(
                name="locked_down", description="Locked down",
                inherits=["base"],
                apparmor_fragments=[],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["locked_down"].apparmor_fragments == []

    def test_cascade_regression_pin(self):
        """The load-bearing least-privilege contract for cascades
        (peer's 2026-05-14 design ask).  Setup mirrors the
        kb-enablement-2.0 cascade:

        - ``_base`` declares ``[jaato-kb-enablement-2]`` (commons)
        - ``host_validator`` overrides to ``[host_validator]``
          (its stack-specific binary-exec grants)
        - ``codegen`` overrides to ``[]`` (no fragments — pure
          read of workspace tree)

        Assert that the resolved fragments for ``codegen`` do
        NOT include ``host_validator`` — i.e. the binary-exec
        grants stay scoped to the host_validator stage.  Without
        replace semantics (e.g. if we'd union'd), codegen would
        inherit the base's set, and the leak that motivated this
        PR persists.  This test is the contract."""
        profiles = {
            "_base": SubagentProfile(
                name="_base", description="Base",
                apparmor_fragments=["jaato-kb-enablement-2"],
            ),
            "host_validator": SubagentProfile(
                name="host_validator", description="Host validator",
                inherits=["_base"],
                apparmor_fragments=["host_validator"],
            ),
            "codegen": SubagentProfile(
                name="codegen", description="Codegen",
                inherits=["_base"],
                apparmor_fragments=[],
            ),
            "transform": SubagentProfile(
                name="transform", description="Transform",
                inherits=["_base"],
                # No override — inherits base's [jaato-kb-enablement-2]
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        # host_validator: scoped to its own stage.
        assert resolved["host_validator"].apparmor_fragments == ["host_validator"]
        # codegen: locked down completely.
        assert resolved["codegen"].apparmor_fragments == []
        # transform: inherits base's commons (no host_validator leak).
        assert resolved["transform"].apparmor_fragments == ["jaato-kb-enablement-2"]
        # Sanity: NONE of the children that overrode inherit
        # host_validator's grants.
        assert "host_validator" not in (resolved["codegen"].apparmor_fragments or [])
        assert "host_validator" not in (resolved["transform"].apparmor_fragments or [])

    def test_multi_parent_first_declaration_wins(self):
        """Two parents both declare; resolution walks parents in
        order and uses the first non-None.  The current merge order
        is parents-in-declared-order; child inheriting from
        [p1, p2] picks p1's value when both declare.  Pin so any
        future reordering surfaces here."""
        profiles = {
            "p1": SubagentProfile(
                name="p1", description="P1",
                apparmor_fragments=["a", "b"],
            ),
            "p2": SubagentProfile(
                name="p2", description="P2",
                apparmor_fragments=["c", "d"],
            ),
            "child": SubagentProfile(
                name="child", description="Child",
                inherits=["p1", "p2"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        # p1 declared first → its value wins.
        assert resolved["child"].apparmor_fragments == ["a", "b"]


class TestCompletionProcessorsInheritance:
    """#791: ``completion_processors`` concatenate, and
    ``suppress_inherited_processors`` is the one way to scope that down.

    Before the opt-out existed, a child whose stage genuinely completed
    differently had to stop inheriting altogether — and silently lost
    ``budget_control`` / ``max_turns`` / ``runtime_limits`` / ``env`` /
    ``plugin_configs`` in the process.  ``test_the_wrong_turn_...`` below
    pins the cost that made this worth a field.
    """

    ACCEPT = CompletionProcessor(
        script="scripts/processors/accept.py", name="acceptance")
    AUDIT = CompletionProcessor(script="scripts/processors/audit.py")

    def _base(self, **kwargs):
        return SubagentProfile(
            name="base", description="Base",
            completion_processors=[self.ACCEPT, self.AUDIT],
            **kwargs,
        )

    def test_concatenates_parent_then_child(self):
        own = CompletionProcessor(script="scripts/processors/own.py")
        profiles = {
            "base": self._base(),
            "child": SubagentProfile(
                name="child", description="Child", inherits=["base"],
                completion_processors=[own],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].completion_processors == [
            self.ACCEPT, self.AUDIT, own]

    def test_empty_child_list_does_not_clear_the_parents(self):
        """The measured surprise in #791: `[]` adds nothing, it is not a reset."""
        profiles = {
            "base": self._base(),
            "child": SubagentProfile(
                name="child", description="Child", inherits=["base"],
                completion_processors=[],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].completion_processors == [self.ACCEPT, self.AUDIT]

    def test_suppress_by_name(self):
        profiles = {
            "base": self._base(),
            "child": SubagentProfile(
                name="child", description="Child", inherits=["base"],
                suppress_inherited_processors=["acceptance"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].completion_processors == [self.AUDIT]

    def test_suppress_by_script_path(self):
        """A processor answers to its script path too, named or not."""
        profiles = {
            "base": self._base(),
            "child": SubagentProfile(
                name="child", description="Child", inherits=["base"],
                suppress_inherited_processors=[
                    "scripts/processors/accept.py",
                    "scripts/processors/audit.py",
                ],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].completion_processors == []

    def test_suppression_keeps_everything_else_the_base_declared(self):
        """The whole point: decline ONE processor, keep the ceilings."""
        from shared.budget_control import BudgetControlConfig
        profiles = {
            "base": self._base(
                max_turns=4,
                env={"STAGE": "worker"},
                budget_control=BudgetControlConfig(limits={"usd": 1.5}),
            ),
            "child": SubagentProfile(
                name="child", description="Child", inherits=["base"],
                suppress_inherited_processors=["acceptance"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        child = resolved["child"]
        assert child.completion_processors == [self.AUDIT]
        assert child.max_turns == 4
        assert child.env == {"STAGE": "worker"}
        assert child.budget_control.limits == {"usd": 1.5}

    def test_the_wrong_turn_it_replaces_loses_every_ceiling(self):
        """Pin the cost of the only pre-#791 escape: not inheriting at all."""
        from shared.budget_control import BudgetControlConfig
        profiles = {
            "base": self._base(
                max_turns=4,
                env={"STAGE": "worker"},
                budget_control=BudgetControlConfig(limits={"usd": 1.5}),
            ),
            # Same intent as the suppression above, expressed by declaring
            # the profile from scratch.
            "child": SubagentProfile(name="child", description="Child"),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        child = resolved["child"]
        assert child.completion_processors == []
        assert child.budget_control is None      # no cost ceiling
        assert child.max_turns == 10             # and the default is LOOSER
        assert child.env == {}

    def test_suppression_does_not_touch_the_childs_own_processors(self):
        """Scoped to what the PARENTS contributed: a child re-declaring the
        same script keeps its own copy (different `output`, say)."""
        own = CompletionProcessor(
            script="scripts/processors/accept.py",
            name="acceptance",
            output="out/{case_id}/accept.json",
        )
        profiles = {
            "base": self._base(),
            "child": SubagentProfile(
                name="child", description="Child", inherits=["base"],
                completion_processors=[own],
                suppress_inherited_processors=["acceptance"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].completion_processors == [self.AUDIT, own]

    def test_unmatched_suppression_is_a_load_error(self):
        """A stale entry means the base renamed the processor and this child
        is running one it declared it did not want — say so, loudly."""
        profiles = {
            "base": self._base(),
            "child": SubagentProfile(
                name="child", description="Child", inherits=["base"],
                suppress_inherited_processors=["acceptence"],  # typo
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert "child" not in resolved
        assert "acceptence" in errors["child"]
        # names what WAS available, so a rename is told apart from a typo
        assert "acceptance" in errors["child"]

    def test_suppression_is_not_inherited_further(self):
        """Consumed at the merge that resolves it: a grandchild neither
        re-applies an ancestor's suppression nor trips its stale-entry error
        on a processor that is already gone."""
        profiles = {
            "base": self._base(),
            "mid": SubagentProfile(
                name="mid", description="Mid", inherits=["base"],
                suppress_inherited_processors=["acceptance"],
            ),
            "leaf": SubagentProfile(
                name="leaf", description="Leaf", inherits=["mid"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["mid"].suppress_inherited_processors == []
        assert resolved["leaf"].completion_processors == [self.AUDIT]

    def test_suppresses_across_multiple_parents(self):
        other = CompletionProcessor(script="scripts/processors/other.py")
        profiles = {
            "p1": SubagentProfile(
                name="p1", description="P1",
                completion_processors=[self.ACCEPT]),
            "p2": SubagentProfile(
                name="p2", description="P2",
                completion_processors=[other]),
            "child": SubagentProfile(
                name="child", description="Child", inherits=["p1", "p2"],
                suppress_inherited_processors=["scripts/processors/other.py"],
            ),
        }
        resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["child"].completion_processors == [self.ACCEPT]

    def test_no_inherits_is_an_inert_no_op_with_a_warning(self, caplog):
        """Nothing to suppress without a parent — say so rather than letting
        the key look effective."""
        profiles = {
            "solo": SubagentProfile(
                name="solo", description="Solo",
                completion_processors=[self.ACCEPT],
                suppress_inherited_processors=["acceptance"],
            ),
        }
        with caplog.at_level("WARNING"):
            resolved, errors = resolve_profiles(profiles)
        assert not errors
        assert resolved["solo"].completion_processors == [self.ACCEPT]
        assert "does not inherit" in caplog.text


class TestParseSuppressInheritedProcessors:
    """Authoring-surface normalization for the #791 opt-out."""

    def test_absent_is_empty(self):
        assert _normalize_suppress_inherited_processors(None) == []

    def test_single_string(self):
        assert _normalize_suppress_inherited_processors("acceptance") == [
            "acceptance"]

    def test_list(self):
        assert _normalize_suppress_inherited_processors(
            ["a", "b"]) == ["a", "b"]

    def test_non_string_entries_are_coerced_not_dropped(self):
        """Coerced so the merge's "matched nothing" error catches them —
        dropping quietly would leave a processor the author declined."""
        assert _normalize_suppress_inherited_processors([42]) == ["42"]

    def test_wrong_type_is_ignored(self):
        assert _normalize_suppress_inherited_processors({"a": 1}) == []


class TestParseCompletionProcessorName:
    """``name:`` is the stable identity a suppression matches (#791)."""

    def test_name_parsed_and_stripped(self):
        [proc] = _parse_completion_processors(
            [{"script": "a.py", "name": "  acceptance  "}])
        assert proc.name == "acceptance"
        assert proc.identity == "acceptance"

    def test_identity_falls_back_to_script(self):
        [proc] = _parse_completion_processors([{"script": "a.py"}])
        assert proc.name is None
        assert proc.identity == "a.py"

    def test_invalid_name_ignored_entry_survives(self):
        [proc] = _parse_completion_processors(
            [{"script": "a.py", "name": 7}])
        assert proc.name is None
        assert proc.script == "a.py"
