"""Tests for instruction_budget module."""

import pytest
from shared.instruction_budget import (
    InstructionSource,
    GCPolicy,
    SourceEntry,
    InstructionBudget,
    ConversationTurnType,
    PluginToolType,
    SystemChildType,
    DEFAULT_SOURCE_POLICIES,
    DEFAULT_TURN_POLICIES,
    DEFAULT_TOOL_POLICIES,
    DEFAULT_SYSTEM_POLICIES,
    GC_POLICY_INDICATORS,
    estimate_tokens,
)


class TestGCPolicy:
    """Tests for GCPolicy enum and indicators."""

    def test_all_policies_have_indicators(self):
        """Every GC policy should have a UI indicator."""
        for policy in GCPolicy:
            assert policy in GC_POLICY_INDICATORS
            assert len(GC_POLICY_INDICATORS[policy]) > 0

    def test_policy_values(self):
        """Policy values should match expected strings."""
        assert GCPolicy.LOCKED.value == "locked"
        assert GCPolicy.PRESERVABLE.value == "preservable"
        assert GCPolicy.PARTIAL.value == "partial"
        assert GCPolicy.EPHEMERAL.value == "ephemeral"


class TestInstructionSource:
    """Tests for InstructionSource enum."""

    def test_all_sources_have_default_policies(self):
        """Every source should have a default GC policy."""
        for source in InstructionSource:
            assert source in DEFAULT_SOURCE_POLICIES

    def test_source_values(self):
        """Source values should match expected strings."""
        assert InstructionSource.SYSTEM.value == "system"
        assert InstructionSource.PLUGIN.value == "plugin"
        assert InstructionSource.ENRICHMENT.value == "enrichment"
        assert InstructionSource.CONVERSATION.value == "conversation"


class TestSourceEntry:
    """Tests for SourceEntry dataclass."""

    def test_leaf_entry_total_tokens(self):
        """Leaf entry total_tokens should return its own tokens."""
        entry = SourceEntry(
            source=InstructionSource.SYSTEM,
            tokens=500,
            gc_policy=GCPolicy.LOCKED,
        )
        assert entry.total_tokens() == 500

    def test_parent_entry_total_tokens(self):
        """Parent entry total_tokens should sum children."""
        entry = SourceEntry(
            source=InstructionSource.PLUGIN,
            tokens=0,  # Parent has no direct tokens
            gc_policy=GCPolicy.PARTIAL,
            children={
                "cli": SourceEntry(
                    source=InstructionSource.PLUGIN,
                    tokens=800,
                    gc_policy=GCPolicy.LOCKED,
                ),
                "web_search": SourceEntry(
                    source=InstructionSource.PLUGIN,
                    tokens=200,
                    gc_policy=GCPolicy.EPHEMERAL,
                ),
            },
        )
        assert entry.total_tokens() == 1000

    def test_locked_entry_gc_eligible_zero(self):
        """LOCKED entries should have zero GC-eligible tokens."""
        entry = SourceEntry(
            source=InstructionSource.SYSTEM,
            tokens=500,
            gc_policy=GCPolicy.LOCKED,
        )
        assert entry.gc_eligible_tokens() == 0

    def test_ephemeral_entry_gc_eligible_all(self):
        """EPHEMERAL entries should have all tokens GC-eligible."""
        entry = SourceEntry(
            source=InstructionSource.ENRICHMENT,
            tokens=300,
            gc_policy=GCPolicy.EPHEMERAL,
        )
        assert entry.gc_eligible_tokens() == 300

    def test_partial_entry_gc_eligible_recurses(self):
        """PARTIAL entries should recurse to children for GC eligibility."""
        entry = SourceEntry(
            source=InstructionSource.PLUGIN,
            tokens=0,
            gc_policy=GCPolicy.PARTIAL,
            children={
                "cli": SourceEntry(
                    source=InstructionSource.PLUGIN,
                    tokens=800,
                    gc_policy=GCPolicy.LOCKED,
                ),
                "web_search": SourceEntry(
                    source=InstructionSource.PLUGIN,
                    tokens=200,
                    gc_policy=GCPolicy.EPHEMERAL,
                ),
            },
        )
        # Only web_search (200) is GC-eligible
        assert entry.gc_eligible_tokens() == 200

    def test_locked_tokens(self):
        """locked_tokens should count LOCKED entries."""
        entry = SourceEntry(
            source=InstructionSource.PLUGIN,
            tokens=0,
            gc_policy=GCPolicy.PARTIAL,
            children={
                "cli": SourceEntry(
                    source=InstructionSource.PLUGIN,
                    tokens=800,
                    gc_policy=GCPolicy.LOCKED,
                ),
                "web_search": SourceEntry(
                    source=InstructionSource.PLUGIN,
                    tokens=200,
                    gc_policy=GCPolicy.EPHEMERAL,
                ),
            },
        )
        assert entry.locked_tokens() == 800

    def test_preservable_tokens(self):
        """preservable_tokens should count PRESERVABLE entries."""
        entry = SourceEntry(
            source=InstructionSource.CONVERSATION,
            tokens=0,
            gc_policy=GCPolicy.PARTIAL,
            children={
                "turn_0": SourceEntry(
                    source=InstructionSource.CONVERSATION,
                    tokens=50,
                    gc_policy=GCPolicy.LOCKED,
                ),
                "clarify_a": SourceEntry(
                    source=InstructionSource.CONVERSATION,
                    tokens=100,
                    gc_policy=GCPolicy.PRESERVABLE,
                ),
                "working": SourceEntry(
                    source=InstructionSource.CONVERSATION,
                    tokens=2000,
                    gc_policy=GCPolicy.EPHEMERAL,
                ),
            },
        )
        assert entry.preservable_tokens() == 100

    def test_indicator(self):
        """indicator() should return the correct UI symbol."""
        assert SourceEntry(
            source=InstructionSource.SYSTEM,
            tokens=100,
            gc_policy=GCPolicy.LOCKED,
        ).indicator() == "🔒"

        assert SourceEntry(
            source=InstructionSource.CONVERSATION,
            tokens=100,
            gc_policy=GCPolicy.PRESERVABLE,
        ).indicator() == "◑"

    def test_to_dict(self):
        """to_dict should produce serializable output."""
        entry = SourceEntry(
            source=InstructionSource.SYSTEM,
            tokens=500,
            gc_policy=GCPolicy.LOCKED,
            label="System Instructions",
        )
        d = entry.to_dict()
        assert d["source"] == "system"
        assert d["tokens"] == 500
        assert d["total_tokens"] == 500
        assert d["gc_policy"] == "locked"
        assert d["gc_eligible_tokens"] == 0
        assert d["label"] == "System Instructions"
        assert d["indicator"] == "🔒"

    def test_to_dict_with_children(self):
        """to_dict should include children recursively."""
        entry = SourceEntry(
            source=InstructionSource.PLUGIN,
            tokens=0,
            gc_policy=GCPolicy.PARTIAL,
            children={
                "cli": SourceEntry(
                    source=InstructionSource.PLUGIN,
                    tokens=800,
                    gc_policy=GCPolicy.LOCKED,
                ),
            },
        )
        d = entry.to_dict()
        assert "children" in d
        assert "cli" in d["children"]
        assert d["children"]["cli"]["tokens"] == 800


class TestInstructionBudget:
    """Tests for InstructionBudget dataclass."""

    def test_empty_budget(self):
        """Empty budget should have zero tokens."""
        budget = InstructionBudget()
        assert budget.total_tokens() == 0
        assert budget.gc_eligible_tokens() == 0
        assert budget.utilization_percent() == 0.0

    def test_set_entry(self):
        """set_entry should create entries with defaults."""
        budget = InstructionBudget()
        entry = budget.set_entry(InstructionSource.SYSTEM, tokens=500)

        assert entry.source == InstructionSource.SYSTEM
        assert entry.tokens == 500
        assert entry.gc_policy == GCPolicy.LOCKED  # Default for SYSTEM
        assert budget.total_tokens() == 500

    def test_set_entry_custom_policy(self):
        """set_entry should accept custom GC policy."""
        budget = InstructionBudget()
        entry = budget.set_entry(
            InstructionSource.PLUGIN,
            tokens=200,
            gc_policy=GCPolicy.PRESERVABLE,
        )
        assert entry.gc_policy == GCPolicy.PRESERVABLE

    def test_get_entry(self):
        """get_entry should return existing entry or None."""
        budget = InstructionBudget()
        budget.set_entry(InstructionSource.SYSTEM, tokens=500)

        assert budget.get_entry(InstructionSource.SYSTEM) is not None
        assert budget.get_entry(InstructionSource.PLUGIN) is None

    def test_update_tokens(self):
        """update_tokens should modify existing entry."""
        budget = InstructionBudget()
        budget.set_entry(InstructionSource.SYSTEM, tokens=500)
        budget.update_tokens(InstructionSource.SYSTEM, 600)

        assert budget.get_entry(InstructionSource.SYSTEM).tokens == 600

    def test_add_child(self):
        """add_child should add child to parent entry."""
        budget = InstructionBudget()
        budget.set_entry(InstructionSource.PLUGIN, tokens=0)
        child = budget.add_child(
            InstructionSource.PLUGIN,
            child_key="cli",
            tokens=800,
            gc_policy=GCPolicy.LOCKED,
            label="CLI Tool",
        )

        assert child is not None
        assert child.tokens == 800
        assert child.label == "CLI Tool"
        assert budget.total_tokens() == 800

    def test_add_child_to_missing_parent(self):
        """add_child should return None if parent doesn't exist."""
        budget = InstructionBudget()
        child = budget.add_child(
            InstructionSource.PLUGIN,
            child_key="cli",
            tokens=800,
            gc_policy=GCPolicy.LOCKED,
        )
        assert child is None

    def test_remove_child(self):
        """remove_child should remove child from parent."""
        budget = InstructionBudget()
        budget.set_entry(InstructionSource.PLUGIN, tokens=0)
        budget.add_child(
            InstructionSource.PLUGIN,
            child_key="cli",
            tokens=800,
            gc_policy=GCPolicy.LOCKED,
        )

        assert budget.remove_child(InstructionSource.PLUGIN, "cli") is True
        assert budget.total_tokens() == 0
        assert budget.remove_child(InstructionSource.PLUGIN, "cli") is False

    def test_utilization_percent(self):
        """utilization_percent should calculate correctly."""
        budget = InstructionBudget(context_limit=100_000)
        budget.set_entry(InstructionSource.SYSTEM, tokens=1000)
        budget.set_entry(InstructionSource.CONVERSATION, tokens=9000)

        assert budget.utilization_percent() == 10.0

    def test_available_tokens(self):
        """available_tokens should return remaining space."""
        budget = InstructionBudget(context_limit=100_000)
        budget.set_entry(InstructionSource.SYSTEM, tokens=10_000)

        assert budget.available_tokens() == 90_000

    def test_gc_headroom_percent(self):
        """gc_headroom_percent should show reclaimable percentage."""
        budget = InstructionBudget(context_limit=100_000)
        budget.set_entry(InstructionSource.SYSTEM, tokens=1000)  # LOCKED
        budget.set_entry(InstructionSource.ENRICHMENT, tokens=500)  # EPHEMERAL

        # 500 out of 100_000 = 0.5%
        assert budget.gc_headroom_percent() == 0.5

    def test_snapshot(self):
        """snapshot should produce complete serializable output."""
        budget = InstructionBudget(
            session_id="session-123",
            agent_id="main",
            agent_type="main",
            context_limit=128_000,
        )
        budget.set_entry(InstructionSource.SYSTEM, tokens=500)
        budget.set_entry(InstructionSource.PLUGIN, tokens=0)
        budget.add_child(
            InstructionSource.PLUGIN,
            child_key="cli",
            tokens=800,
            gc_policy=GCPolicy.LOCKED,
        )

        snap = budget.snapshot()

        assert snap["session_id"] == "session-123"
        assert snap["agent_id"] == "main"
        assert snap["agent_type"] == "main"
        assert snap["context_limit"] == 128_000
        assert snap["total_tokens"] == 1300
        assert "entries" in snap
        assert "system" in snap["entries"]
        assert "plugin" in snap["entries"]
        assert "children" in snap["entries"]["plugin"]

    def test_create_default(self):
        """create_default should initialize all sources."""
        budget = InstructionBudget.create_default(
            session_id="session-456",
            agent_id="explore-1",
            agent_type="explore",
            context_limit=200_000,
        )

        assert budget.session_id == "session-456"
        assert budget.agent_id == "explore-1"
        assert budget.agent_type == "explore"
        assert budget.context_limit == 200_000
        for source in InstructionSource:
            assert budget.get_entry(source) is not None
            assert budget.get_entry(source).tokens == 0

    def test_create_default_agent_type_defaults_to_agent_id(self):
        """create_default should default agent_type to agent_id if not provided."""
        budget = InstructionBudget.create_default(
            session_id="session-789",
            agent_id="custom-agent",
        )

        assert budget.agent_type == "custom-agent"


class TestConversationTurnType:
    """Tests for ConversationTurnType enum."""

    def test_all_turn_types_have_default_policies(self):
        """Every turn type should have a default GC policy."""
        for turn_type in ConversationTurnType:
            assert turn_type in DEFAULT_TURN_POLICIES

    def test_expected_policies(self):
        """Turn types should have expected default policies."""
        assert DEFAULT_TURN_POLICIES[ConversationTurnType.ORIGINAL_REQUEST] == GCPolicy.LOCKED
        assert DEFAULT_TURN_POLICIES[ConversationTurnType.CLARIFICATION_Q] == GCPolicy.PRESERVABLE
        assert DEFAULT_TURN_POLICIES[ConversationTurnType.CLARIFICATION_A] == GCPolicy.PRESERVABLE
        assert DEFAULT_TURN_POLICIES[ConversationTurnType.TURN_SUMMARY] == GCPolicy.PRESERVABLE
        assert DEFAULT_TURN_POLICIES[ConversationTurnType.WORKING] == GCPolicy.EPHEMERAL


class TestPluginToolType:
    """Tests for PluginToolType enum."""

    def test_all_tool_types_have_default_policies(self):
        """Every tool type should have a default GC policy."""
        for tool_type in PluginToolType:
            assert tool_type in DEFAULT_TOOL_POLICIES

    def test_expected_policies(self):
        """Tool types should have expected default policies."""
        assert DEFAULT_TOOL_POLICIES[PluginToolType.CORE] == GCPolicy.LOCKED
        assert DEFAULT_TOOL_POLICIES[PluginToolType.DISCOVERABLE] == GCPolicy.EPHEMERAL


class TestSystemChildType:
    """Tests for SystemChildType enum."""

    def test_all_child_types_have_default_policies(self):
        """Every system child type should have a default GC policy."""
        for child_type in SystemChildType:
            assert child_type in DEFAULT_SYSTEM_POLICIES

    def test_expected_policies(self):
        """System child types should have expected default policies."""
        assert DEFAULT_SYSTEM_POLICIES[SystemChildType.BASE] == GCPolicy.LOCKED
        assert DEFAULT_SYSTEM_POLICIES[SystemChildType.CLIENT] == GCPolicy.LOCKED
        assert DEFAULT_SYSTEM_POLICIES[SystemChildType.FRAMEWORK] == GCPolicy.LOCKED

    def test_child_type_values(self):
        """System child type values should match expected strings."""
        assert SystemChildType.BASE.value == "base"
        assert SystemChildType.CLIENT.value == "client"
        assert SystemChildType.FRAMEWORK.value == "framework"


class TestEstimateTokens:
    """Tests for estimate_tokens helper."""

    def test_empty_string(self):
        """Empty string should return 0."""
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0

    def test_default_ratio(self):
        """Default ratio is 4 chars per token."""
        assert estimate_tokens("a" * 100) == 25
        assert estimate_tokens("a" * 400) == 100

    def test_custom_ratio(self):
        """Custom chars_per_token should work."""
        assert estimate_tokens("a" * 100, chars_per_token=2.0) == 50
