"""Truncate GC Plugin - Simple turn-based garbage collection.

This plugin implements the simplest GC strategy: remove oldest turns
while preserving the most recent N turns. No summarization, minimal
overhead, fast execution.

Similar to Java's simple heap compaction - just remove the oldest data.
"""

from typing import Any, Dict, List, Optional, Tuple

from google.genai import types

from ..gc import (
    GCConfig,
    GCPlugin,
    GCResult,
    GCTriggerReason,
    Turn,
    create_gc_notification_content,
    estimate_history_tokens,
    flatten_turns,
    get_preserved_indices,
    split_into_turns,
)


class TruncateGCPlugin:
    """GC plugin that removes oldest turns to free context space.

    This is the simplest and fastest GC strategy:
    - Splits history into turns
    - Removes oldest turns beyond the preservation limit
    - Keeps recent N turns intact

    Configuration options (via initialize()):
        preserve_recent_turns: Override default from GCConfig
        notify_on_gc: Whether to inject notification message (default: False)
        notification_template: Custom notification message template

    Example:
        plugin = TruncateGCPlugin()
        plugin.initialize({
            "preserve_recent_turns": 10,
            "notify_on_gc": True
        })
        client.set_gc_plugin(plugin, GCConfig(threshold_percent=75.0))
    """

    def __init__(self):
        self._initialized = False
        self._config: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Plugin identifier."""
        return "gc_truncate"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Optional configuration dict with:
                - preserve_recent_turns: int - Override preservation count
                - notify_on_gc: bool - Inject notification message (default: False)
                - notification_template: str - Custom notification template
        """
        self._config = config or {}
        self._initialized = True

    def shutdown(self) -> None:
        """Clean up resources."""
        self._config = {}
        self._initialized = False

    def should_collect(
        self,
        context_usage: Dict[str, Any],
        config: GCConfig
    ) -> Tuple[bool, Optional[GCTriggerReason]]:
        """Check if garbage collection should be triggered.

        Triggers based on:
        1. Context usage exceeding threshold percentage
        2. Turn count exceeding max_turns limit

        Args:
            context_usage: Current context window usage stats.
            config: GC configuration with thresholds.

        Returns:
            Tuple of (should_collect, reason).
        """
        if not config.auto_trigger:
            return False, None

        # Check threshold percentage
        percent_used = context_usage.get('percent_used', 0)
        if percent_used >= config.threshold_percent:
            return True, GCTriggerReason.THRESHOLD

        # Check turn limit
        if config.max_turns is not None:
            turns = context_usage.get('turns', 0)
            if turns >= config.max_turns:
                return True, GCTriggerReason.TURN_LIMIT

        return False, None

    def collect(
        self,
        history: List[types.Content],
        context_usage: Dict[str, Any],
        config: GCConfig,
        reason: GCTriggerReason
    ) -> Tuple[List[types.Content], GCResult]:
        """Perform garbage collection by truncating oldest turns.

        Args:
            history: Current conversation history.
            context_usage: Current context window usage stats.
            config: GC configuration.
            reason: Why this collection was triggered.

        Returns:
            Tuple of (new_history, result).
        """
        tokens_before = estimate_history_tokens(history)

        # Split into turns
        turns = split_into_turns(history)
        total_turns = len(turns)

        # Determine preservation count (plugin config overrides GCConfig)
        preserve_count = self._config.get(
            'preserve_recent_turns',
            config.preserve_recent_turns
        )

        # Get indices to preserve
        preserved_indices = get_preserved_indices(
            total_turns,
            preserve_count,
            config.pinned_turn_indices
        )

        # Nothing to collect if all turns are preserved
        if len(preserved_indices) >= total_turns:
            return history, GCResult(
                success=True,
                items_collected=0,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                plugin_name=self.name,
                trigger_reason=reason,
                details={"message": "All turns preserved, nothing to collect"}
            )

        # Filter turns - keep only preserved ones
        kept_turns: List[Turn] = []
        removed_count = 0

        for turn in turns:
            if turn.index in preserved_indices:
                kept_turns.append(turn)
            else:
                removed_count += 1

        # Flatten back to history
        new_history = flatten_turns(kept_turns)
        tokens_after = estimate_history_tokens(new_history)

        # Build result
        result = GCResult(
            success=True,
            items_collected=removed_count,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            plugin_name=self.name,
            trigger_reason=reason,
            details={
                "turns_before": total_turns,
                "turns_after": len(kept_turns),
                "preserve_count": preserve_count,
                "preserved_indices": list(preserved_indices),
            }
        )

        # Add notification if configured
        if self._config.get('notify_on_gc', False):
            template = self._config.get(
                'notification_template',
                "Context cleaned: removed {removed} old turns, kept {kept} recent turns."
            )
            notification = template.format(
                removed=removed_count,
                kept=len(kept_turns),
                tokens_freed=tokens_before - tokens_after
            )
            result.notification = notification

            # Prepend notification to history
            notification_content = create_gc_notification_content(notification)
            new_history = [notification_content] + new_history

        return new_history, result


def create_plugin() -> TruncateGCPlugin:
    """Factory function to create a TruncateGCPlugin instance."""
    return TruncateGCPlugin()
