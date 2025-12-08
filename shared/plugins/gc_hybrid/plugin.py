"""Hybrid GC Plugin - Generational garbage collection.

This plugin implements a hybrid GC strategy inspired by Java's generational GC:
- Young generation (recent turns): Always preserved intact
- Old generation (middle-aged turns): Summarized for compression
- Ancient (very old turns): Truncated/removed entirely

This provides a balance between context preservation and token efficiency.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from ..model_provider.types import Message

from ..gc import (
    GCConfig,
    GCPlugin,
    GCResult,
    GCTriggerReason,
    Turn,
    create_summary_content,
    estimate_history_tokens,
    flatten_turns,
    split_into_turns,
)


class HybridGCPlugin:
    """GC plugin that combines truncation and summarization.

    This generational strategy:
    - Preserves recent turns intact (young generation)
    - Summarizes middle-aged turns (old generation)
    - Truncates very old turns (ancient generation)

    Configuration options (via initialize()):
        preserve_recent_turns: Number of recent turns to keep intact (default: 5)
        summarize_middle_turns: Number of middle turns to summarize (default: 10)
        summarizer: Callable (str) -> str for summarization (required for summarization)
        notify_on_gc: Whether to inject notification message (default: False)
        notification_template: Custom notification message template

    When summarizer is not provided, this behaves like truncation but with
    configurable preservation of more recent turns.

    Example:
        plugin = HybridGCPlugin()
        plugin.initialize({
            "preserve_recent_turns": 5,
            "summarize_middle_turns": 15,
            "summarizer": my_summarize_function
        })
        client.set_gc_plugin(plugin, GCConfig(threshold_percent=75.0))

    Turn layout after GC:
        [Summary of ancient+middle turns] + [preserved recent turns]

        Or without summarizer:
        [preserved recent turns] (ancient and middle turns truncated)
    """

    def __init__(self):
        self._initialized = False
        self._config: Dict[str, Any] = {}
        self._summarizer: Optional[Callable[[str], str]] = None

    @property
    def name(self) -> str:
        """Plugin identifier."""
        return "gc_hybrid"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Optional configuration dict with:
                - preserve_recent_turns: int - Recent turns to keep intact
                - summarize_middle_turns: int - Middle turns to summarize
                - summarizer: Callable[[str], str] - Summary generator function
                - notify_on_gc: bool - Inject notification message
                - notification_template: str - Custom notification template
        """
        self._config = config or {}
        self._summarizer = self._config.get('summarizer')
        self._initialized = True

    def shutdown(self) -> None:
        """Clean up resources."""
        self._config = {}
        self._summarizer = None
        self._initialized = False

    def should_collect(
        self,
        context_usage: Dict[str, Any],
        config: GCConfig
    ) -> Tuple[bool, Optional[GCTriggerReason]]:
        """Check if garbage collection should be triggered.

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
        history: List[Message],
        context_usage: Dict[str, Any],
        config: GCConfig,
        reason: GCTriggerReason
    ) -> Tuple[List[Message], GCResult]:
        """Perform hybrid garbage collection.

        The history is divided into three regions:
        1. Recent (young): Last N turns, always preserved
        2. Middle (old): Next M turns, summarized if summarizer available
        3. Ancient: Remaining turns, always truncated

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

        # Get configuration
        preserve_recent = self._config.get(
            'preserve_recent_turns',
            config.preserve_recent_turns
        )
        summarize_middle = self._config.get('summarize_middle_turns', 10)

        # Nothing to collect if we have fewer turns than preservation count
        if total_turns <= preserve_recent:
            return history, GCResult(
                success=True,
                items_collected=0,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                plugin_name=self.name,
                trigger_reason=reason,
                details={"message": "Not enough turns to collect"}
            )

        # Divide turns into generations
        # Recent: last preserve_recent turns
        # Middle: next summarize_middle turns before recent
        # Ancient: everything before middle

        recent_start = max(0, total_turns - preserve_recent)
        middle_start = max(0, recent_start - summarize_middle)

        ancient_turns = turns[:middle_start]
        middle_turns = turns[middle_start:recent_start]
        recent_turns = turns[recent_start:]

        # Process based on what we have
        new_history_parts: List[Message] = []
        turns_truncated = len(ancient_turns)
        turns_summarized = 0
        summary_text = ""

        # If we have middle turns and a summarizer, summarize them
        turns_to_summarize = ancient_turns + middle_turns
        if turns_to_summarize and self._summarizer:
            conversation_text = self._format_turns_for_summary(turns_to_summarize)

            try:
                summary_text = self._summarizer(conversation_text)
                summary_content = create_summary_content(summary_text)
                new_history_parts.append(summary_content)
                turns_summarized = len(turns_to_summarize)
                turns_truncated = 0  # Everything was summarized, not truncated
            except Exception as e:
                # Summarization failed - fall back to truncation
                return history, GCResult(
                    success=False,
                    items_collected=0,
                    tokens_before=tokens_before,
                    tokens_after=tokens_before,
                    plugin_name=self.name,
                    trigger_reason=reason,
                    error=f"Summarization failed: {str(e)}"
                )
        elif turns_to_summarize:
            # No summarizer - just truncate
            turns_truncated = len(turns_to_summarize)

        # Add recent turns
        new_history_parts.extend(flatten_turns(recent_turns))

        tokens_after = estimate_history_tokens(new_history_parts)

        # Build result
        result = GCResult(
            success=True,
            items_collected=turns_truncated + turns_summarized,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            plugin_name=self.name,
            trigger_reason=reason,
            details={
                "turns_before": total_turns,
                "turns_after": len(recent_turns) + (1 if summary_text else 0),
                "turns_truncated": turns_truncated,
                "turns_summarized": turns_summarized,
                "preserve_recent": preserve_recent,
                "summarize_middle": summarize_middle,
                "had_summarizer": self._summarizer is not None,
            }
        )

        # Add notification if configured
        if self._config.get('notify_on_gc', False):
            template = self._config.get(
                'notification_template',
                "Context cleaned: {truncated} turns removed, "
                "{summarized} turns summarized, {kept} recent turns preserved."
            )
            notification = template.format(
                truncated=turns_truncated,
                summarized=turns_summarized,
                kept=len(recent_turns),
                tokens_freed=tokens_before - tokens_after
            )
            result.notification = notification

        return new_history_parts, result

    def _format_turns_for_summary(self, turns: List[Turn]) -> str:
        """Format turns into a text string for summarization.

        Args:
            turns: List of turns to format.

        Returns:
            Formatted conversation text.
        """
        lines: List[str] = []

        for turn in turns:
            for content in turn.contents:
                role = content.role.upper() if content.role else "UNKNOWN"

                if content.parts:
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            lines.append(f"{role}: {part.text}")
                        elif hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            lines.append(f"{role}: [Called {fc.name}]")
                        elif hasattr(part, 'function_response') and part.function_response:
                            fr = part.function_response
                            lines.append(f"{role}: [Response from {fr.name}]")

        return "\n".join(lines)


def create_plugin() -> HybridGCPlugin:
    """Factory function to create a HybridGCPlugin instance."""
    return HybridGCPlugin()
