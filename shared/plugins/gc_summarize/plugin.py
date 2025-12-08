"""Summarize GC Plugin - Compression-based garbage collection.

This plugin implements a summarization GC strategy: compress old turns
into a summary rather than removing them entirely. Preserves context
information while freeing token space.

Similar to Java's generational GC - old content gets compacted into
a more efficient representation.
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
    get_preserved_indices,
    split_into_turns,
)


# Default summarization prompt template
DEFAULT_SUMMARIZE_PROMPT = """Summarize the following conversation history concisely.
Focus on key information, decisions made, and important context.
Keep the summary brief but preserve essential details.

Conversation to summarize:
{conversation}

Provide a concise summary:"""


class SummarizeGCPlugin:
    """GC plugin that compresses old turns into summaries.

    This strategy preserves context information while freeing token space:
    - Splits history into turns
    - Identifies turns to be collected (oldest beyond preservation limit)
    - Generates a summary of collected turns
    - Replaces collected turns with the summary

    Configuration options (via initialize()):
        preserve_recent_turns: Override default from GCConfig
        summarize_prompt: Custom prompt template for summarization
        summarizer: Custom callable (str) -> str for summarization
        max_summary_tokens: Target max tokens for summary (default: 500)
        notify_on_gc: Whether to inject notification message (default: False)
        notification_template: Custom notification message template

    Example:
        plugin = SummarizeGCPlugin()
        plugin.initialize({
            "preserve_recent_turns": 10,
            "summarizer": my_summarize_function,
            "notify_on_gc": True
        })
        client.set_gc_plugin(plugin, GCConfig(threshold_percent=75.0))
    """

    def __init__(self):
        self._initialized = False
        self._config: Dict[str, Any] = {}
        self._summarizer: Optional[Callable[[str], str]] = None

    @property
    def name(self) -> str:
        """Plugin identifier."""
        return "gc_summarize"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Optional configuration dict with:
                - preserve_recent_turns: int - Override preservation count
                - summarize_prompt: str - Custom summarization prompt template
                - summarizer: Callable[[str], str] - Function to generate summaries
                - max_summary_tokens: int - Target max tokens for summary
                - notify_on_gc: bool - Inject notification message (default: False)
                - notification_template: str - Custom notification template

        Note:
            A summarizer function MUST be provided for this plugin to work.
            The summarizer should accept a conversation string and return a summary.
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
        history: List[Message],
        context_usage: Dict[str, Any],
        config: GCConfig,
        reason: GCTriggerReason
    ) -> Tuple[List[Message], GCResult]:
        """Perform garbage collection by summarizing oldest turns.

        Args:
            history: Current conversation history.
            context_usage: Current context window usage stats.
            config: GC configuration.
            reason: Why this collection was triggered.

        Returns:
            Tuple of (new_history, result).
        """
        tokens_before = estimate_history_tokens(history)

        # Check if summarizer is available
        if self._summarizer is None:
            return history, GCResult(
                success=False,
                items_collected=0,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                plugin_name=self.name,
                trigger_reason=reason,
                error="No summarizer function configured. "
                      "Provide a 'summarizer' callable in plugin config."
            )

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

        # Separate turns into to-summarize and to-preserve
        turns_to_summarize: List[Turn] = []
        turns_to_preserve: List[Turn] = []

        for turn in turns:
            if turn.index in preserved_indices:
                turns_to_preserve.append(turn)
            else:
                turns_to_summarize.append(turn)

        # Generate summary of old turns
        conversation_text = self._format_turns_for_summary(turns_to_summarize)

        try:
            summary_text = self._summarizer(conversation_text)
        except Exception as e:
            return history, GCResult(
                success=False,
                items_collected=0,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                plugin_name=self.name,
                trigger_reason=reason,
                error=f"Summarization failed: {str(e)}"
            )

        # Create summary content
        summary_content = create_summary_content(summary_text)

        # Build new history: summary + preserved turns
        new_history = [summary_content] + flatten_turns(turns_to_preserve)
        tokens_after = estimate_history_tokens(new_history)

        # Build result
        result = GCResult(
            success=True,
            items_collected=len(turns_to_summarize),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            plugin_name=self.name,
            trigger_reason=reason,
            details={
                "turns_before": total_turns,
                "turns_after": len(turns_to_preserve) + 1,  # +1 for summary
                "turns_summarized": len(turns_to_summarize),
                "preserve_count": preserve_count,
                "summary_length": len(summary_text),
            }
        )

        # Add notification if configured
        if self._config.get('notify_on_gc', False):
            template = self._config.get(
                'notification_template',
                "Context cleaned: summarized {removed} old turns into context summary."
            )
            notification = template.format(
                removed=len(turns_to_summarize),
                kept=len(turns_to_preserve),
                tokens_freed=tokens_before - tokens_after
            )
            result.notification = notification

        return new_history, result

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

    def get_summarize_prompt(self, conversation: str) -> str:
        """Get the prompt for summarization.

        Args:
            conversation: The conversation text to summarize.

        Returns:
            Formatted summarization prompt.
        """
        template = self._config.get('summarize_prompt', DEFAULT_SUMMARIZE_PROMPT)
        return template.format(conversation=conversation)


def create_plugin() -> SummarizeGCPlugin:
    """Factory function to create a SummarizeGCPlugin instance."""
    return SummarizeGCPlugin()
