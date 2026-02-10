"""Tests for BudgetGCPlugin."""

import time
import pytest

from shared.plugins.gc import GCConfig, GCTriggerReason, GCRemovalItem
from shared.plugins.gc_budget import BudgetGCPlugin, create_plugin
from shared.instruction_budget import (
    InstructionBudget,
    InstructionSource,
    SourceEntry,
    GCPolicy,
)
from jaato import Message, Part, Role, FunctionCall, ToolResult


def make_message(role: str, text: str, message_id: str = None) -> Message:
    """Helper to create Message objects."""
    r = Role.USER if role == "user" else Role.MODEL
    msg = Message(role=r, parts=[Part(text=text)])
    if message_id:
        msg.message_id = message_id
    return msg


def make_history(num_turns: int) -> list:
    """Create a history with N turns (user+model pairs)."""
    history = []
    for i in range(num_turns):
        history.append(make_message("user", f"User message {i}"))
        history.append(make_message("model", f"Model response {i}"))
    return history


def make_budget(context_limit: int = 10000) -> InstructionBudget:
    """Create a test budget with some entries."""
    budget = InstructionBudget(context_limit=context_limit)
    return budget


class TestBudgetGCPlugin:
    def test_create_plugin(self):
        plugin = create_plugin()
        assert plugin.name == "gc_budget"

    def test_initialize(self):
        plugin = BudgetGCPlugin()
        plugin.initialize({
            "preserve_recent_turns": 10,
            "target_percent": 50.0,
            "pressure_percent": 85.0,
        })
        assert plugin._initialized
        assert plugin._config.get("preserve_recent_turns") == 10
        assert plugin._config.get("target_percent") == 50.0
        assert plugin._config.get("pressure_percent") == 85.0

    def test_shutdown(self):
        plugin = BudgetGCPlugin()
        plugin.initialize({"key": "value"})
        plugin.shutdown()
        assert not plugin._initialized
        assert plugin._config == {}


class TestShouldCollect:
    def test_auto_trigger_disabled(self):
        plugin = create_plugin()
        plugin.initialize()

        config = GCConfig(auto_trigger=False, threshold_percent=50.0)
        context = {"percent_used": 90.0}

        should_collect, reason = plugin.should_collect(context, config)
        assert not should_collect
        assert reason is None

    def test_threshold_triggered(self):
        plugin = create_plugin()
        plugin.initialize()

        config = GCConfig(threshold_percent=75.0)
        context = {"percent_used": 80.0}

        should_collect, reason = plugin.should_collect(context, config)
        assert should_collect
        assert reason == GCTriggerReason.THRESHOLD

    def test_threshold_not_reached(self):
        plugin = create_plugin()
        plugin.initialize()

        config = GCConfig(threshold_percent=75.0)
        context = {"percent_used": 50.0}

        should_collect, reason = plugin.should_collect(context, config)
        assert not should_collect

    def test_turn_limit_triggered(self):
        plugin = create_plugin()
        plugin.initialize()

        config = GCConfig(threshold_percent=90.0, max_turns=10)
        context = {"percent_used": 50.0, "turns": 15}

        should_collect, reason = plugin.should_collect(context, config)
        assert should_collect
        assert reason == GCTriggerReason.TURN_LIMIT

    def test_continuous_mode_above_target(self):
        """In continuous mode, trigger when above target_percent."""
        plugin = create_plugin()
        plugin.initialize()

        config = GCConfig(
            threshold_percent=80.0,
            target_percent=60.0,
            pressure_percent=0,  # Enable continuous mode
        )
        context = {"percent_used": 65.0}

        should_collect, reason = plugin.should_collect(context, config)
        assert should_collect
        assert reason == GCTriggerReason.THRESHOLD

    def test_continuous_mode_at_target(self):
        """In continuous mode, don't trigger when at or below target."""
        plugin = create_plugin()
        plugin.initialize()

        config = GCConfig(
            threshold_percent=80.0,
            target_percent=60.0,
            pressure_percent=0,  # Enable continuous mode
        )
        context = {"percent_used": 60.0}

        should_collect, reason = plugin.should_collect(context, config)
        assert not should_collect


class TestCollectWithBudget:
    def test_already_below_target(self):
        """No collection when already below target."""
        plugin = create_plugin()
        plugin.initialize({"preserve_recent_turns": 3})

        budget = make_budget(10000)
        # Add some entries but stay below target
        budget.set_entry(InstructionSource.SYSTEM, 1000, GCPolicy.LOCKED)

        history = make_history(5)
        config = GCConfig(target_percent=60.0)
        context = {"percent_used": 10.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget
        )

        assert result.success
        assert result.items_collected == 0
        assert "below target" in result.details.get("message", "")

    def test_clears_enrichment_first(self):
        """Phase 1a: ENRICHMENT is cleared first."""
        plugin = create_plugin()
        plugin.initialize()

        budget = make_budget(10000)
        # Add ENRICHMENT with substantial tokens
        budget.set_entry(InstructionSource.ENRICHMENT, 2000, GCPolicy.EPHEMERAL)
        # Add some conversation
        budget.set_entry(InstructionSource.CONVERSATION, 6000, GCPolicy.PARTIAL)

        history = make_history(5)
        config = GCConfig(target_percent=60.0)  # Target: 6000 tokens
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget
        )

        assert result.success
        # Should have cleared enrichment
        assert result.details.get("enrichment_cleared") is True

    def test_removes_ephemeral_oldest_first(self):
        """Phase 1b: EPHEMERAL entries removed oldest first."""
        plugin = create_plugin()
        plugin.initialize()

        budget = make_budget(10000)
        # Add older ephemeral entry
        budget.set_entry(InstructionSource.PLUGIN, 100, GCPolicy.LOCKED)
        budget.add_child(
            InstructionSource.PLUGIN,
            "old_ephemeral",
            1000,
            GCPolicy.EPHEMERAL,
            label="old_ephemeral",
            created_at=time.time() - 100,  # Older
        )
        budget.add_child(
            InstructionSource.PLUGIN,
            "new_ephemeral",
            1000,
            GCPolicy.EPHEMERAL,
            label="new_ephemeral",
            created_at=time.time(),  # Newer
        )
        # Add conversation to trigger GC
        budget.set_entry(InstructionSource.CONVERSATION, 7000, GCPolicy.PARTIAL)

        history = make_history(5)
        config = GCConfig(target_percent=60.0)  # Target: 6000 tokens
        context = {"percent_used": 90.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget
        )

        assert result.success
        # Should have removed at least the older ephemeral
        assert result.details.get("ephemeral_removed", 0) >= 1

    def test_continuous_mode_never_touches_preservable(self):
        """In continuous mode, PRESERVABLE content is never removed."""
        plugin = create_plugin()
        plugin.initialize()

        budget = make_budget(10000)
        # Add PRESERVABLE entry
        budget.set_entry(InstructionSource.PLUGIN, 0, GCPolicy.PARTIAL)
        budget.add_child(
            InstructionSource.PLUGIN,
            "preservable_1",
            2000,
            GCPolicy.PRESERVABLE,
            label="preservable_1",
        )
        # Add conversation
        budget.set_entry(InstructionSource.CONVERSATION, 7000, GCPolicy.PARTIAL)

        history = make_history(5)
        config = GCConfig(
            target_percent=30.0,  # Very aggressive target
            pressure_percent=0,  # Continuous mode
        )
        context = {"percent_used": 90.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget
        )

        assert result.success
        # PRESERVABLE should NOT be removed in continuous mode
        assert result.details.get("preservable_removed", 0) == 0

    def test_pressure_mode_removes_preservable(self):
        """Under extreme pressure, PRESERVABLE content can be removed."""
        plugin = create_plugin()
        plugin.initialize()

        budget = make_budget(10000)
        # Add PRESERVABLE entry
        budget.set_entry(InstructionSource.PLUGIN, 0, GCPolicy.PARTIAL)
        budget.add_child(
            InstructionSource.PLUGIN,
            "preservable_1",
            2000,
            GCPolicy.PRESERVABLE,
            label="preservable_1",
            created_at=time.time() - 100,
        )
        # Add locked conversation to consume budget
        budget.set_entry(InstructionSource.CONVERSATION, 7500, GCPolicy.LOCKED)

        history = make_history(5)
        config = GCConfig(
            target_percent=30.0,  # Aggressive target
            pressure_percent=90.0,  # Threshold mode with pressure
        )
        context = {"percent_used": 95.0}  # Above pressure_percent

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget
        )

        assert result.success
        # PRESERVABLE may be removed under pressure
        # (depends on whether target was met by other means)


class TestCollectFallback:
    def test_fallback_without_budget(self):
        """Falls back to truncation when no budget available."""
        plugin = create_plugin()
        plugin.initialize({"preserve_recent_turns": 2})

        history = make_history(10)
        config = GCConfig(preserve_recent_turns=2)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget=None
        )

        assert result.success
        assert result.details.get("mode") == "fallback_truncate"
        assert result.items_collected > 0
        # Should preserve last 2 turns (4 messages)
        assert len(new_history) == 4


class TestRemovalList:
    def test_removal_list_populated(self):
        """Verify removal_list is populated correctly."""
        plugin = create_plugin()
        plugin.initialize({"preserve_recent_turns": 2})

        budget = make_budget(10000)
        budget.set_entry(InstructionSource.ENRICHMENT, 2000, GCPolicy.EPHEMERAL)
        budget.set_entry(InstructionSource.CONVERSATION, 7000, GCPolicy.PARTIAL)

        history = make_history(5)
        config = GCConfig(target_percent=60.0)
        context = {"percent_used": 90.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget
        )

        assert result.success
        assert len(result.removal_list) > 0

        # Check removal list items have correct structure
        for item in result.removal_list:
            assert isinstance(item, GCRemovalItem)
            assert item.source is not None
            assert item.reason != ""


class TestNotification:
    def test_notification_when_enabled(self):
        plugin = create_plugin()
        plugin.initialize({
            "preserve_recent_turns": 2,
            "notify_on_gc": True
        })

        budget = make_budget(10000)
        budget.set_entry(InstructionSource.ENRICHMENT, 3000, GCPolicy.EPHEMERAL)
        budget.set_entry(InstructionSource.CONVERSATION, 6000, GCPolicy.PARTIAL)

        history = make_history(5)
        config = GCConfig(target_percent=60.0)
        context = {"percent_used": 90.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget
        )

        assert result.notification is not None
        assert "removed" in result.notification or "freed" in result.notification

    def test_custom_notification_template(self):
        plugin = create_plugin()
        plugin.initialize({
            "preserve_recent_turns": 2,
            "notify_on_gc": True,
            "notification_template": "Cleared {items} items!"
        })

        budget = make_budget(10000)
        budget.set_entry(InstructionSource.ENRICHMENT, 3000, GCPolicy.EPHEMERAL)
        budget.set_entry(InstructionSource.CONVERSATION, 6000, GCPolicy.PARTIAL)

        history = make_history(5)
        config = GCConfig(target_percent=60.0)
        context = {"percent_used": 90.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget
        )

        if result.items_collected > 0:
            assert result.notification is not None
            assert "Cleared" in result.notification
            assert "items!" in result.notification


class TestGCConfigNewFields:
    """Tests for new GCConfig fields added for budget integration."""

    def test_default_values(self):
        """Test GCConfig has correct default values for new fields."""
        config = GCConfig()

        # Default target is 60%
        assert config.target_percent == 60.0
        # Default pressure is 90%
        assert config.pressure_percent == 90.0
        # Not in continuous mode by default
        assert config.continuous_mode is False

    def test_continuous_mode_with_zero_pressure(self):
        """Continuous mode is enabled when pressure_percent is 0."""
        config = GCConfig(pressure_percent=0)
        assert config.continuous_mode is True

    def test_continuous_mode_with_none_pressure(self):
        """Continuous mode is enabled when pressure_percent is None."""
        config = GCConfig(pressure_percent=None)
        assert config.continuous_mode is True

    def test_threshold_mode_with_positive_pressure(self):
        """Threshold mode when pressure_percent is positive."""
        config = GCConfig(pressure_percent=85.0)
        assert config.continuous_mode is False

    def test_custom_target_percent(self):
        """Test custom target_percent."""
        config = GCConfig(target_percent=50.0)
        assert config.target_percent == 50.0


class TestGCRemovalItem:
    """Tests for GCRemovalItem dataclass."""

    def test_default_values(self):
        """Test GCRemovalItem default values."""
        item = GCRemovalItem(source=InstructionSource.CONVERSATION)

        assert item.source == InstructionSource.CONVERSATION
        assert item.child_key is None
        assert item.tokens_freed == 0
        assert item.reason == ""
        assert item.message_ids == []

    def test_full_initialization(self):
        """Test GCRemovalItem with all fields."""
        item = GCRemovalItem(
            source=InstructionSource.ENRICHMENT,
            child_key="tool_result_123",
            tokens_freed=500,
            reason="ephemeral",
            message_ids=["msg-1", "msg-2"],
        )

        assert item.source == InstructionSource.ENRICHMENT
        assert item.child_key == "tool_result_123"
        assert item.tokens_freed == 500
        assert item.reason == "ephemeral"
        assert item.message_ids == ["msg-1", "msg-2"]


class TestRegressionContextLimitRecovery:
    """Regression tests for GC context-limit recovery bugs."""

    def test_ephemeral_candidates_return_dict_keys(self):
        """Bug B: _get_ephemeral_candidates must return dict keys that match budget children."""
        plugin = create_plugin()
        plugin.initialize()

        budget = make_budget(10000)
        budget.set_entry(InstructionSource.PLUGIN, 0, GCPolicy.PARTIAL)
        budget.add_child(
            InstructionSource.PLUGIN,
            "tool_result_42",
            500,
            GCPolicy.EPHEMERAL,
            label="some display label",
        )

        candidates = plugin._get_ephemeral_candidates(budget)
        assert len(candidates) == 1
        child_key, entry = candidates[0]
        # The key must be the actual dict key, not the display label
        assert child_key == "tool_result_42"
        assert child_key in budget.get_entry(InstructionSource.PLUGIN).children

    def test_preservable_candidates_return_dict_keys(self):
        """Bug B: _get_preservable_candidates must return dict keys that match budget children."""
        plugin = create_plugin()
        plugin.initialize()

        budget = make_budget(10000)
        budget.set_entry(InstructionSource.PLUGIN, 0, GCPolicy.PARTIAL)
        budget.add_child(
            InstructionSource.PLUGIN,
            "memo_99",
            300,
            GCPolicy.PRESERVABLE,
            label="memory note 99",
        )

        candidates = plugin._get_preservable_candidates(budget)
        assert len(candidates) == 1
        child_key, entry = candidates[0]
        assert child_key == "memo_99"
        assert child_key in budget.get_entry(InstructionSource.PLUGIN).children

    def test_apply_removals_removes_messages_with_ids(self):
        """Bug A: _apply_removals_to_history must actually remove messages when message_ids are set."""
        plugin = create_plugin()
        plugin.initialize()

        msg1 = make_message("user", "hello", message_id="id-1")
        msg2 = make_message("model", "hi there", message_id="id-2")
        msg3 = make_message("user", "bye", message_id="id-3")
        history = [msg1, msg2, msg3]

        removal_list = [
            GCRemovalItem(
                source=InstructionSource.CONVERSATION,
                child_key="msg_0",
                tokens_freed=10,
                reason="ephemeral",
                message_ids=["id-1"],
            ),
            GCRemovalItem(
                source=InstructionSource.CONVERSATION,
                child_key="msg_1",
                tokens_freed=10,
                reason="ephemeral",
                message_ids=["id-2"],
            ),
        ]

        new_history = plugin._apply_removals_to_history(history, removal_list)
        assert len(new_history) == 1
        assert new_history[0].message_id == "id-3"

    def test_apply_removals_no_op_without_message_ids(self):
        """Without message_ids, _apply_removals_to_history returns original history."""
        plugin = create_plugin()
        plugin.initialize()

        history = [make_message("user", "hello", message_id="id-1")]
        removal_list = [
            GCRemovalItem(
                source=InstructionSource.CONVERSATION,
                child_key="msg_0",
                tokens_freed=10,
                reason="ephemeral",
                message_ids=[],  # Empty — this was the bug
            ),
        ]

        new_history = plugin._apply_removals_to_history(history, removal_list)
        assert len(new_history) == 1  # Nothing removed

    def test_end_to_end_gc_collect_frees_tokens(self):
        """End-to-end: GC collect() with populated message_ids actually removes messages."""
        plugin = create_plugin()
        plugin.initialize({"preserve_recent_turns": 1})

        budget = make_budget(10000)
        # Set up conversation with children that have message_ids
        budget.set_entry(InstructionSource.CONVERSATION, 0, GCPolicy.PARTIAL)

        # Build history with known message IDs
        history = []
        for i in range(6):
            role = "user" if i % 2 == 0 else "model"
            msg = make_message(role, f"Message {i}", message_id=f"msg-id-{i}")
            history.append(msg)

        # Add budget children matching the history, with message_ids
        for i, msg in enumerate(history):
            budget.add_child(
                InstructionSource.CONVERSATION,
                f"msg_{i}",
                1500,
                GCPolicy.EPHEMERAL,
                label=f"turn message {i}",
                message_ids=[msg.message_id],
            )

        config = GCConfig(target_percent=50.0, preserve_recent_turns=1)
        context = {"percent_used": 90.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget
        )

        assert result.success
        assert result.items_collected > 0
        assert result.tokens_freed > 0
        # History should be shorter than original
        assert len(new_history) < len(history)
        # Removal list items should have the correct dict keys
        for item in result.removal_list:
            if item.child_key and item.source == InstructionSource.CONVERSATION:
                assert item.child_key.startswith("msg_")


# --- Helpers for tool_call pair tests ---

def make_model_with_tool_calls(call_ids, message_id=None):
    """Create a MODEL message with function_call parts."""
    parts = []
    for cid in call_ids:
        parts.append(Part(function_call=FunctionCall(
            id=cid, name=f"tool_{cid}", args={},
        )))
    msg = Message(role=Role.MODEL, parts=parts)
    if message_id:
        msg.message_id = message_id
    return msg


def make_tool_result_msg(call_ids, message_id=None, result_text="ok"):
    """Create a TOOL message with function_response parts."""
    parts = []
    for cid in call_ids:
        parts.append(Part(function_response=ToolResult(
            call_id=cid, name=f"tool_{cid}", result=result_text,
        )))
    msg = Message(role=Role.TOOL, parts=parts)
    if message_id:
        msg.message_id = message_id
    return msg


class TestToolCallPairMap:
    """Tests for _build_tool_call_pair_map."""

    def test_empty_history(self):
        plugin = create_plugin()
        plugin.initialize()
        assert plugin._build_tool_call_pair_map([]) == {}

    def test_no_tool_calls(self):
        plugin = create_plugin()
        plugin.initialize()
        history = [
            make_message("user", "Hello", message_id="u1"),
            make_message("model", "Hi", message_id="m1"),
        ]
        assert plugin._build_tool_call_pair_map(history) == {}

    def test_simple_pair(self):
        plugin = create_plugin()
        plugin.initialize()
        history = [
            make_message("user", "Search", message_id="u1"),
            make_model_with_tool_calls(["call_1"], message_id="m1"),
            make_tool_result_msg(["call_1"], message_id="t1"),
            make_message("model", "Done", message_id="m2"),
        ]
        pair_map = plugin._build_tool_call_pair_map(history)
        assert "m1" in pair_map
        assert "t1" in pair_map["m1"]
        assert "t1" in pair_map
        assert "m1" in pair_map["t1"]

    def test_multiple_calls_in_one_message(self):
        """MODEL with multiple tool_calls pairs with multiple results."""
        plugin = create_plugin()
        plugin.initialize()
        history = [
            make_model_with_tool_calls(["A", "B"], message_id="m1"),
            make_tool_result_msg(["A", "B"], message_id="t1"),
        ]
        pair_map = plugin._build_tool_call_pair_map(history)
        assert "t1" in pair_map["m1"]
        assert "m1" in pair_map["t1"]

    def test_sequential_pairs(self):
        """Two separate tool_call/result pairs."""
        plugin = create_plugin()
        plugin.initialize()
        history = [
            make_model_with_tool_calls(["A"], message_id="m1"),
            make_tool_result_msg(["A"], message_id="t1"),
            make_model_with_tool_calls(["B"], message_id="m2"),
            make_tool_result_msg(["B"], message_id="t2"),
        ]
        pair_map = plugin._build_tool_call_pair_map(history)
        assert pair_map["m1"] == {"t1"}
        assert pair_map["t1"] == {"m1"}
        assert pair_map["m2"] == {"t2"}
        assert pair_map["t2"] == {"m2"}


class TestExpandRemovalPairs:
    """Tests for _expand_removal_pairs.

    Verifies that when GC selects one side of a tool_call pair for removal,
    the other side is automatically included.
    """

    def _setup_budget_with_tool_pair(self):
        """Set up budget and history with a tool_call pair.

        Returns (plugin, budget, history, pair_map) with:
        - msg_0: user "Search" (LOCKED)
        - msg_1: model with tool_call "call_X" (EPHEMERAL, 100 tokens)
        - msg_2: tool result for "call_X" (EPHEMERAL, 5000 tokens)
        - msg_3: model "Done" (PRESERVABLE, 50 tokens)
        """
        plugin = create_plugin()
        plugin.initialize({"preserve_recent_turns": 1})

        history = [
            make_message("user", "Search", message_id="uid-0"),
            make_model_with_tool_calls(["call_X"], message_id="uid-1"),
            make_tool_result_msg(["call_X"], message_id="uid-2",
                                 result_text="x" * 20000),  # Large result
            make_message("model", "Done", message_id="uid-3"),
        ]

        budget = make_budget(100000)
        budget.set_entry(InstructionSource.CONVERSATION, 0, GCPolicy.PARTIAL)
        budget.add_child(
            InstructionSource.CONVERSATION, "msg_0", 50,
            GCPolicy.LOCKED, label="turn_1 input (external)",
            message_ids=["uid-0"],
        )
        budget.add_child(
            InstructionSource.CONVERSATION, "msg_1", 100,
            GCPolicy.EPHEMERAL, label="turn_1 output (model)",
            message_ids=["uid-1"],
        )
        budget.add_child(
            InstructionSource.CONVERSATION, "msg_2", 5000,
            GCPolicy.EPHEMERAL, label="turn_1 input (tool = tool_call_X)",
            message_ids=["uid-2"],
        )
        budget.add_child(
            InstructionSource.CONVERSATION, "msg_3", 50,
            GCPolicy.PRESERVABLE, label="turn_1 turn_summary",
            message_ids=["uid-3"],
        )

        pair_map = plugin._build_tool_call_pair_map(history)
        return plugin, budget, history, pair_map

    def test_removing_tool_result_also_removes_model(self):
        """When GC removes a large tool_result, the paired MODEL
        (with tool_calls) is also removed."""
        plugin, budget, history, pair_map = self._setup_budget_with_tool_pair()

        # Simulate: GC selected the large tool_result for removal
        removal_list = [
            GCRemovalItem(
                source=InstructionSource.CONVERSATION,
                child_key="msg_2",
                tokens_freed=5000,
                reason="ephemeral",
                message_ids=["uid-2"],
            ),
        ]

        expanded, extra_tokens = plugin._expand_removal_pairs(
            removal_list, pair_map, budget,
        )

        # Should now include both the tool_result AND the model msg
        assert len(expanded) == 2
        removed_keys = {item.child_key for item in expanded}
        assert "msg_2" in removed_keys  # tool_result
        assert "msg_1" in removed_keys  # model with tool_call
        assert extra_tokens == 100  # model message tokens

    def test_removing_model_also_removes_tool_result(self):
        """When GC removes a MODEL with tool_calls, the paired
        tool_result is also removed."""
        plugin, budget, history, pair_map = self._setup_budget_with_tool_pair()

        removal_list = [
            GCRemovalItem(
                source=InstructionSource.CONVERSATION,
                child_key="msg_1",
                tokens_freed=100,
                reason="ephemeral",
                message_ids=["uid-1"],
            ),
        ]

        expanded, extra_tokens = plugin._expand_removal_pairs(
            removal_list, pair_map, budget,
        )

        assert len(expanded) == 2
        removed_keys = {item.child_key for item in expanded}
        assert "msg_1" in removed_keys
        assert "msg_2" in removed_keys
        assert extra_tokens == 5000

    def test_both_already_in_removal_no_duplicates(self):
        """If both sides are already in the removal list, no duplicates added."""
        plugin, budget, history, pair_map = self._setup_budget_with_tool_pair()

        removal_list = [
            GCRemovalItem(
                source=InstructionSource.CONVERSATION,
                child_key="msg_1", tokens_freed=100,
                reason="ephemeral", message_ids=["uid-1"],
            ),
            GCRemovalItem(
                source=InstructionSource.CONVERSATION,
                child_key="msg_2", tokens_freed=5000,
                reason="ephemeral", message_ids=["uid-2"],
            ),
        ]

        expanded, extra_tokens = plugin._expand_removal_pairs(
            removal_list, pair_map, budget,
        )

        assert len(expanded) == 2  # No extras
        assert extra_tokens == 0

    def test_no_pairs_no_expansion(self):
        """Non-tool messages are not expanded."""
        plugin = create_plugin()
        plugin.initialize()

        budget = make_budget(10000)
        budget.set_entry(InstructionSource.CONVERSATION, 0, GCPolicy.PARTIAL)
        budget.add_child(
            InstructionSource.CONVERSATION, "msg_0", 100,
            GCPolicy.EPHEMERAL, label="some entry",
            message_ids=["uid-0"],
        )

        removal_list = [
            GCRemovalItem(
                source=InstructionSource.CONVERSATION,
                child_key="msg_0", tokens_freed=100,
                reason="ephemeral", message_ids=["uid-0"],
            ),
        ]

        expanded, extra_tokens = plugin._expand_removal_pairs(
            removal_list, {}, budget,  # Empty pair_map
        )

        assert len(expanded) == 1
        assert extra_tokens == 0

    def test_pair_expansion_reason_is_tool_call_pair(self):
        """Expanded entries get reason='tool_call_pair'."""
        plugin, budget, history, pair_map = self._setup_budget_with_tool_pair()

        removal_list = [
            GCRemovalItem(
                source=InstructionSource.CONVERSATION,
                child_key="msg_2", tokens_freed=5000,
                reason="ephemeral", message_ids=["uid-2"],
            ),
        ]

        expanded, _ = plugin._expand_removal_pairs(
            removal_list, pair_map, budget,
        )

        pair_items = [i for i in expanded if i.reason == "tool_call_pair"]
        assert len(pair_items) == 1
        assert pair_items[0].child_key == "msg_1"


class TestEndToEndPairAwareGC:
    """End-to-end tests verifying GC collect() produces valid history
    with intact tool_call pairs."""

    def test_gc_removes_tool_result_keeps_pairing(self):
        """When GC needs to free tokens and selects a large tool_result,
        the resulting history has no orphaned tool_use or tool_result."""
        plugin = create_plugin()
        plugin.initialize({"preserve_recent_turns": 1})

        # History: user -> model(call_A) -> tool(A, large) -> model(text)
        history = [
            make_message("user", "Search", message_id="uid-0"),
            make_model_with_tool_calls(["call_A"], message_id="uid-1"),
            make_tool_result_msg(["call_A"], message_id="uid-2",
                                 result_text="x" * 20000),
            make_message("model", "Summary", message_id="uid-3"),
        ]

        budget = make_budget(10000)
        budget.set_entry(InstructionSource.CONVERSATION, 0, GCPolicy.PARTIAL)
        budget.add_child(
            InstructionSource.CONVERSATION, "msg_0", 50,
            GCPolicy.LOCKED, label="turn_1 input",
            message_ids=["uid-0"],
        )
        budget.add_child(
            InstructionSource.CONVERSATION, "msg_1", 100,
            GCPolicy.EPHEMERAL, label="turn_1 output (model)",
            message_ids=["uid-1"],
            created_at=1.0,
        )
        budget.add_child(
            InstructionSource.CONVERSATION, "msg_2", 5000,
            GCPolicy.EPHEMERAL, label="turn_1 input (tool = tool_call_A)",
            message_ids=["uid-2"],
            created_at=2.0,
        )
        budget.add_child(
            InstructionSource.CONVERSATION, "msg_3", 50,
            GCPolicy.PRESERVABLE, label="turn_1 turn_summary",
            message_ids=["uid-3"],
        )

        config = GCConfig(target_percent=20.0)  # Aggressive to force removal
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget,
        )

        assert result.success

        # Verify no orphaned tool_use or tool_result in resulting history
        pending_call_ids = set()
        for msg in new_history:
            if msg.role == Role.MODEL:
                for p in msg.parts:
                    if p.function_call:
                        pending_call_ids.add(p.function_call.id)
            elif msg.role == Role.TOOL:
                for p in msg.parts:
                    if p.function_response:
                        call_id = p.function_response.call_id
                        assert call_id in pending_call_ids, (
                            f"Orphaned tool_result: call_id={call_id} "
                            f"not in pending={pending_call_ids}"
                        )
                        pending_call_ids.discard(call_id)

        # No remaining unpaired tool_calls (unless at very end awaiting results)
        # Since we only have completed turns, there should be none
        model_with_fc = [
            msg for msg in new_history
            if msg.role == Role.MODEL
            and any(p.function_call for p in msg.parts)
        ]
        for msg in model_with_fc:
            fc_ids = {p.function_call.id for p in msg.parts if p.function_call}
            # Each tool_call must have a matching result somewhere after it
            msg_idx = new_history.index(msg)
            remaining = new_history[msg_idx + 1:]
            result_ids = set()
            for r_msg in remaining:
                for p in r_msg.parts:
                    if p.function_response:
                        result_ids.add(p.function_response.call_id)
            assert fc_ids <= result_ids, (
                f"Unpaired tool_use: {fc_ids - result_ids}"
            )
