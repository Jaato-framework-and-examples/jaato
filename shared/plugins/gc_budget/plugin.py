"""Budget GC Plugin - Policy-aware garbage collection.

This plugin implements an intelligent GC strategy that uses the InstructionBudget
to make removal decisions based on GC policies:
- LOCKED: Never removed
- PRESERVABLE: Only removed under extreme pressure
- PARTIAL: Container with mixed children
- EPHEMERAL: First candidates for removal

Supports two modes:
- Threshold Mode (default): GC triggers when usage >= threshold_percent
- Continuous Mode: GC runs after every turn if usage > target_percent
  (enabled by setting pressure_percent to 0 or None)
"""

import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..model_provider.types import Message

from ..gc import (
    GCConfig,
    GCPlugin,
    GCRemovalItem,
    GCResult,
    GCTriggerReason,
    Turn,
    create_gc_notification_message,
    estimate_history_tokens,
    flatten_turns,
    get_preserved_indices,
    split_into_turns,
)

from ...instruction_budget import (
    InstructionBudget,
    InstructionSource,
    GCPolicy,
    SourceEntry,
)

from shared.trace import trace as _trace_write


class BudgetGCPlugin:
    """GC plugin that uses budget policies to make smart removal decisions.

    This policy-aware strategy removes content in priority order:
    1a. ENRICHMENT (bulk clear - regenerated each turn)
    1b. Other EPHEMERAL entries (oldest first)
    2.  PARTIAL conversation turns (oldest first, respecting preserve_recent_turns)
    3.  PRESERVABLE entries (only under extreme pressure > pressure_percent)
    4.  LOCKED entries (never removed)

    Configuration options (via initialize()):
        preserve_recent_turns: Number of recent turns to always keep
        target_percent: Target usage after GC (default: 60.0)
        pressure_percent: When to touch PRESERVABLE (default: 90.0, 0 = continuous mode)
        notify_on_gc: Whether to inject notification message (default: False)
        notification_template: Custom notification message template

    Continuous Mode:
        Set pressure_percent to 0 to enable continuous mode:
        - GC runs after every turn if usage > target_percent
        - PRESERVABLE content is never touched
        - Provides predictable, gradual trimming

    Example:
        plugin = BudgetGCPlugin()
        plugin.initialize({
            "preserve_recent_turns": 5,
            "target_percent": 60.0,
            "pressure_percent": 90.0,  # or 0 for continuous mode
            "notify_on_gc": True
        })
        client.set_gc_plugin(plugin, GCConfig(threshold_percent=80.0))
    """

    def __init__(self):
        self._initialized = False
        self._config: Dict[str, Any] = {}
        self._agent_name: Optional[str] = None

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        _trace_write("GC_BUDGET", msg)

    @property
    def name(self) -> str:
        """Plugin identifier."""
        return "gc_budget"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Optional configuration dict with:
                - preserve_recent_turns: int - Number of recent turns to keep
                - target_percent: float - Target usage after GC (default: 60.0)
                - pressure_percent: float - When to touch PRESERVABLE (0 = continuous)
                - notify_on_gc: bool - Inject notification message (default: False)
                - notification_template: str - Custom notification template
        """
        self._config = config or {}
        self._agent_name = self._config.get("agent_name")
        self._initialized = True
        preserve = self._config.get("preserve_recent_turns", "default")
        target = self._config.get("target_percent", 60.0)
        pressure = self._config.get("pressure_percent", 90.0)
        continuous = pressure == 0 or pressure is None
        self._trace(
            f"initialize: preserve_recent_turns={preserve}, target={target}%, "
            f"pressure={pressure}%, continuous_mode={continuous}"
        )

    def shutdown(self) -> None:
        """Clean up resources."""
        self._trace("shutdown")
        self._config = {}
        self._initialized = False

    def should_collect(
        self,
        context_usage: Dict[str, Any],
        config: GCConfig
    ) -> Tuple[bool, Optional[GCTriggerReason]]:
        """Check if garbage collection should be triggered.

        In continuous mode (pressure_percent=0), triggers if usage > target_percent.
        In threshold mode, triggers if usage >= threshold_percent.

        Args:
            context_usage: Current context window usage stats.
            config: GC configuration with thresholds.

        Returns:
            Tuple of (should_collect, reason).
        """
        if not config.auto_trigger:
            return False, None

        percent_used = context_usage.get('percent_used', 0)

        # Continuous mode: collect if above target (checked after every turn)
        if config.continuous_mode:
            if percent_used > config.target_percent:
                self._trace(
                    f"should_collect: continuous mode triggered "
                    f"({percent_used:.1f}% > {config.target_percent}%)"
                )
                return True, GCTriggerReason.THRESHOLD
            self._trace(
                f"should_collect: continuous mode - no collection needed "
                f"({percent_used:.1f}% <= {config.target_percent}%)"
            )
            return False, None

        # Threshold mode: collect when threshold exceeded
        if percent_used >= config.threshold_percent:
            self._trace(
                f"should_collect: threshold triggered "
                f"({percent_used:.1f}% >= {config.threshold_percent}%)"
            )
            return True, GCTriggerReason.THRESHOLD

        # Check turn limit
        if config.max_turns is not None:
            turns = context_usage.get('turns', 0)
            if turns >= config.max_turns:
                self._trace(
                    f"should_collect: triggered by turn_limit ({turns} >= {config.max_turns})"
                )
                return True, GCTriggerReason.TURN_LIMIT

        self._trace(
            f"should_collect: not triggered - "
            f"usage={percent_used:.1f}% < threshold={config.threshold_percent}%"
        )
        return False, None

    def collect(
        self,
        history: List[Message],
        context_usage: Dict[str, Any],
        config: GCConfig,
        reason: GCTriggerReason,
        budget: Optional[InstructionBudget] = None,
    ) -> Tuple[List[Message], GCResult]:
        """Perform policy-aware garbage collection.

        Uses the budget's GC policies to prioritize what to remove:
        1a. ENRICHMENT (bulk clear)
        1b. Other EPHEMERAL entries (oldest first)
        2.  PARTIAL conversation turns (oldest first)
        3.  PRESERVABLE (only if usage > pressure_percent)

        Args:
            history: Current conversation history.
            context_usage: Current context window usage stats.
            config: GC configuration.
            reason: Why this collection was triggered.
            budget: Instruction budget for policy-aware decisions.

        Returns:
            Tuple of (new_history, result) with removal_list for budget sync.
        """
        self._trace(f"collect: reason={reason.value}, history_len={len(history)}")
        tokens_before = estimate_history_tokens(history)

        # Fall back to simple turn-based collection if no budget available
        if not budget:
            self._trace("collect: no budget available, falling back to turn-based GC")
            return self._fallback_truncate(history, context_usage, config, reason)

        # Calculate target tokens
        current_tokens = budget.total_tokens()
        target_tokens = int(budget.context_limit * config.target_percent / 100)
        tokens_to_free = current_tokens - target_tokens

        if tokens_to_free <= 0:
            self._trace(
                f"collect: already at or below target "
                f"({current_tokens} <= {target_tokens})"
            )
            return history, GCResult(
                success=True,
                items_collected=0,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                plugin_name=self.name,
                trigger_reason=reason,
                details={"message": "Already below target, no collection needed"}
            )

        removal_list: List[GCRemovalItem] = []
        tokens_freed = 0
        percent_used = context_usage.get('percent_used', 0)

        # Phase 1a: Bulk clear ENRICHMENT (always ephemeral, regenerated each turn)
        enrichment_entry = budget.get_entry(InstructionSource.ENRICHMENT)
        if enrichment_entry and enrichment_entry.total_tokens() > 0:
            enrichment_tokens = enrichment_entry.total_tokens()
            removal_list.append(GCRemovalItem(
                source=InstructionSource.ENRICHMENT,
                child_key=None,  # Bulk clear entire source
                tokens_freed=enrichment_tokens,
                reason="enrichment_bulk_clear",
            ))
            tokens_freed += enrichment_tokens
            self._trace(f"collect: phase 1a - cleared enrichment ({enrichment_tokens} tokens)")

        # Phase 1b: Remove other EPHEMERAL entries (oldest first)
        if tokens_freed < tokens_to_free:
            ephemeral_candidates = self._get_ephemeral_candidates(budget)
            # Sort by created_at (oldest first)
            sorted_candidates = sorted(
                ephemeral_candidates,
                key=lambda e: e.created_at or 0
            )

            for entry in sorted_candidates:
                if tokens_freed >= tokens_to_free:
                    break
                entry_tokens = entry.total_tokens()
                removal_list.append(GCRemovalItem(
                    source=entry.source,
                    child_key=entry.label,
                    tokens_freed=entry_tokens,
                    reason="ephemeral",
                    message_ids=entry.message_ids,
                ))
                tokens_freed += entry_tokens
                self._trace(
                    f"collect: phase 1b - removed ephemeral '{entry.label}' "
                    f"({entry_tokens} tokens)"
                )

        # Phase 2: Remove old PARTIAL conversation turns
        if tokens_freed < tokens_to_free:
            turn_candidates = self._get_partial_turn_candidates(budget, config)
            for turn_key, entry in turn_candidates:
                if tokens_freed >= tokens_to_free:
                    break
                entry_tokens = entry.total_tokens()
                removal_list.append(GCRemovalItem(
                    source=InstructionSource.CONVERSATION,
                    child_key=turn_key,
                    tokens_freed=entry_tokens,
                    reason="partial_turn",
                    message_ids=entry.message_ids,
                ))
                tokens_freed += entry_tokens
                self._trace(
                    f"collect: phase 2 - removed turn '{turn_key}' "
                    f"({entry_tokens} tokens)"
                )

        # Phase 3: PRESERVABLE (only in threshold mode when usage > pressure_percent)
        # In continuous mode (pressure_percent=0/None), PRESERVABLE is never touched
        if (tokens_freed < tokens_to_free
                and config.pressure_percent
                and percent_used >= config.pressure_percent):
            self._trace(
                f"collect: phase 3 - extreme pressure mode "
                f"({percent_used:.1f}% >= {config.pressure_percent}%)"
            )
            preservable_candidates = self._get_preservable_candidates(budget)
            for entry in preservable_candidates:
                if tokens_freed >= tokens_to_free:
                    break
                entry_tokens = entry.total_tokens()
                removal_list.append(GCRemovalItem(
                    source=entry.source,
                    child_key=entry.label,
                    tokens_freed=entry_tokens,
                    reason="preservable_under_pressure",
                    message_ids=entry.message_ids,
                ))
                tokens_freed += entry_tokens
                self._trace(
                    f"collect: phase 3 - removed preservable '{entry.label}' "
                    f"({entry_tokens} tokens)"
                )

        # Apply removals to history using message IDs
        new_history = self._apply_removals_to_history(history, removal_list)
        tokens_after = estimate_history_tokens(new_history)

        # Build result
        result = GCResult(
            success=True,
            items_collected=len(removal_list),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            plugin_name=self.name,
            trigger_reason=reason,
            removal_list=removal_list,
            details={
                "target_tokens": target_tokens,
                "tokens_to_free": tokens_to_free,
                "tokens_freed": tokens_freed,
                "enrichment_cleared": any(
                    r.reason == "enrichment_bulk_clear" for r in removal_list
                ),
                "ephemeral_removed": sum(
                    1 for r in removal_list if r.reason == "ephemeral"
                ),
                "partial_removed": sum(
                    1 for r in removal_list if r.reason == "partial_turn"
                ),
                "preservable_removed": sum(
                    1 for r in removal_list if r.reason == "preservable_under_pressure"
                ),
            }
        )
        self._trace(
            f"collect: completed - removed {len(removal_list)} items, "
            f"freed {tokens_freed} tokens ({tokens_before} -> {tokens_after})"
        )

        # Add notification if configured
        if self._config.get('notify_on_gc', False) and result.items_collected > 0:
            template = self._config.get(
                'notification_template',
                "Context cleaned: removed {items} items, freed {freed} tokens."
            )
            notification = template.format(
                items=result.items_collected,
                freed=result.tokens_freed,
            )
            result.notification = notification
            notification_content = create_gc_notification_message(notification)
            new_history = [notification_content] + new_history

        return new_history, result

    def _get_ephemeral_candidates(
        self,
        budget: InstructionBudget,
    ) -> List[SourceEntry]:
        """Get EPHEMERAL entries eligible for removal (excluding ENRICHMENT).

        Args:
            budget: The instruction budget to search.

        Returns:
            List of EPHEMERAL entries sorted by creation time (oldest first).
        """
        candidates: List[SourceEntry] = []

        for source, entry in budget.entries.items():
            # Skip ENRICHMENT (handled separately as bulk clear)
            if source == InstructionSource.ENRICHMENT:
                continue

            # Check children for EPHEMERAL entries
            for child_key, child in entry.children.items():
                if child.gc_policy == GCPolicy.EPHEMERAL:
                    candidates.append(child)

        return candidates

    def _get_partial_turn_candidates(
        self,
        budget: InstructionBudget,
        config: GCConfig,
    ) -> List[Tuple[str, SourceEntry]]:
        """Get PARTIAL conversation turns eligible for removal.

        Respects preserve_recent_turns and pinned_turn_indices.

        Args:
            budget: The instruction budget to search.
            config: GC configuration with preservation settings.

        Returns:
            List of (turn_key, entry) tuples for removable turns, oldest first.
        """
        conv_entry = budget.get_entry(InstructionSource.CONVERSATION)
        if not conv_entry:
            return []

        # Get all turn children
        turn_entries: List[Tuple[str, SourceEntry]] = []
        for child_key, child in conv_entry.children.items():
            # Only consider PARTIAL or EPHEMERAL turns (not LOCKED or PRESERVABLE)
            if child.gc_policy in (GCPolicy.PARTIAL, GCPolicy.EPHEMERAL):
                # Skip summaries (they're PRESERVABLE and should not be here)
                if child_key.startswith("gc_summary_"):
                    continue
                turn_entries.append((child_key, child))

        # Sort by creation time (oldest first)
        turn_entries.sort(key=lambda x: x[1].created_at or 0)

        # Determine preservation count
        preserve_count = self._config.get(
            'preserve_recent_turns',
            config.preserve_recent_turns
        )

        # Get preserved indices (most recent N)
        total_turns = len(turn_entries)
        preserved_indices = get_preserved_indices(
            total_turns,
            preserve_count,
            config.pinned_turn_indices
        )

        # Filter out preserved turns
        candidates: List[Tuple[str, SourceEntry]] = []
        for i, (turn_key, entry) in enumerate(turn_entries):
            if i not in preserved_indices:
                candidates.append((turn_key, entry))

        return candidates

    def _get_preservable_candidates(
        self,
        budget: InstructionBudget,
    ) -> List[SourceEntry]:
        """Get PRESERVABLE entries (only for extreme pressure mode).

        Args:
            budget: The instruction budget to search.

        Returns:
            List of PRESERVABLE entries sorted by creation time (oldest first).
        """
        candidates: List[SourceEntry] = []

        for source, entry in budget.entries.items():
            # Check children for PRESERVABLE entries
            for child_key, child in entry.children.items():
                if child.gc_policy == GCPolicy.PRESERVABLE:
                    candidates.append(child)

        # Sort by creation time (oldest first)
        candidates.sort(key=lambda e: e.created_at or 0)
        return candidates

    def _apply_removals_to_history(
        self,
        history: List[Message],
        removal_list: List[GCRemovalItem],
    ) -> List[Message]:
        """Remove messages from history based on removal list.

        Uses message IDs for precise removal.

        Args:
            history: Current conversation history.
            removal_list: Items to remove.

        Returns:
            New history with removed messages filtered out.
        """
        # Collect all message IDs to remove
        ids_to_remove: set = set()
        for item in removal_list:
            if item.message_ids:
                ids_to_remove.update(item.message_ids)

        if not ids_to_remove:
            # No message IDs to remove, fall back to keeping all
            return history

        # Filter history
        return [msg for msg in history if msg.message_id not in ids_to_remove]

    def _fallback_truncate(
        self,
        history: List[Message],
        context_usage: Dict[str, Any],
        config: GCConfig,
        reason: GCTriggerReason,
    ) -> Tuple[List[Message], GCResult]:
        """Fallback to simple truncation when budget is not available.

        Args:
            history: Current conversation history.
            context_usage: Current context window usage stats.
            config: GC configuration.
            reason: Why this collection was triggered.

        Returns:
            Tuple of (new_history, result).
        """
        tokens_before = estimate_history_tokens(history)
        turns = split_into_turns(history)
        total_turns = len(turns)

        preserve_count = self._config.get(
            'preserve_recent_turns',
            config.preserve_recent_turns
        )

        preserved_indices = get_preserved_indices(
            total_turns,
            preserve_count,
            config.pinned_turn_indices
        )

        if len(preserved_indices) >= total_turns:
            return history, GCResult(
                success=True,
                items_collected=0,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                plugin_name=self.name,
                trigger_reason=reason,
                details={"message": "All turns preserved (fallback mode)"}
            )

        kept_turns: List[Turn] = []
        removal_list: List[GCRemovalItem] = []

        for turn in turns:
            if turn.index in preserved_indices:
                kept_turns.append(turn)
            else:
                message_ids = [msg.message_id for msg in turn.contents]
                removal_list.append(GCRemovalItem(
                    source=InstructionSource.CONVERSATION,
                    child_key=f"turn_{turn.index}",
                    tokens_freed=turn.estimated_tokens,
                    reason="truncated_fallback",
                    message_ids=message_ids,
                ))

        new_history = flatten_turns(kept_turns)
        tokens_after = estimate_history_tokens(new_history)

        return new_history, GCResult(
            success=True,
            items_collected=len(removal_list),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            plugin_name=self.name,
            trigger_reason=reason,
            removal_list=removal_list,
            details={
                "mode": "fallback_truncate",
                "turns_before": total_turns,
                "turns_after": len(kept_turns),
            }
        )


def create_plugin() -> BudgetGCPlugin:
    """Factory function to create a BudgetGCPlugin instance."""
    return BudgetGCPlugin()
