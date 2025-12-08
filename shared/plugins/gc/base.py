"""Base types and protocol for Context Garbage Collection plugins.

This module defines the interface that all GC strategy plugins must implement,
along with supporting types for configuration, results, and trigger reasons.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from ..model_provider.types import Message


class GCTriggerReason(Enum):
    """Reason why garbage collection was triggered."""

    THRESHOLD = "threshold"      # Context usage exceeded threshold percentage
    MANUAL = "manual"            # Explicitly requested by caller
    TURN_LIMIT = "turn_limit"    # Maximum turn count exceeded
    PRE_MESSAGE = "pre_message"  # Triggered before sending a message


@dataclass
class GCResult:
    """Result of a garbage collection operation.

    Provides detailed information about what was collected and the outcome.
    """

    success: bool
    """Whether the GC operation completed successfully."""

    items_collected: int
    """Number of Content items removed or modified."""

    tokens_before: int
    """Estimated token count before GC."""

    tokens_after: int
    """Estimated token count after GC."""

    plugin_name: str
    """Name of the GC plugin that performed the collection."""

    trigger_reason: GCTriggerReason
    """What triggered this GC operation."""

    notification: Optional[str] = None
    """Optional message to inject into history to notify the model of GC."""

    details: Dict[str, Any] = field(default_factory=dict)
    """Plugin-specific details about the collection."""

    error: Optional[str] = None
    """Error message if the operation failed."""

    @property
    def tokens_freed(self) -> int:
        """Calculate tokens freed by this GC operation."""
        return max(0, self.tokens_before - self.tokens_after)


@dataclass
class GCConfig:
    """Configuration for context garbage collection.

    Controls when GC triggers and what content to preserve.
    """

    # Trigger settings
    threshold_percent: float = 80.0
    """Trigger GC when context usage exceeds this percentage."""

    max_turns: Optional[int] = None
    """Trigger GC when turn count exceeds this limit (None = no limit)."""

    auto_trigger: bool = True
    """Whether to automatically trigger GC based on thresholds."""

    check_before_send: bool = True
    """Whether to check and possibly trigger GC before each send_message."""

    # Preservation settings
    preserve_recent_turns: int = 5
    """Number of recent turns to always preserve."""

    pinned_turn_indices: List[int] = field(default_factory=list)
    """Specific turn indices to never remove (0-indexed)."""

    # Plugin-specific configuration
    plugin_config: Dict[str, Any] = field(default_factory=dict)
    """Additional configuration passed to the GC plugin."""


@runtime_checkable
class GCPlugin(Protocol):
    """Protocol for Context Garbage Collection strategy plugins.

    GC plugins implement different strategies for managing conversation
    history to prevent context window overflow. Each plugin can implement
    its own approach (truncation, summarization, hybrid, etc.).

    This follows the same pattern as PermissionPlugin - JaatoClient accepts
    any plugin implementing this interface via set_gc_plugin().

    Example implementation:
        class TruncateGCPlugin:
            @property
            def name(self) -> str:
                return "gc_truncate"

            def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
                self._config = config or {}

            def shutdown(self) -> None:
                pass

            def should_collect(self, context_usage, config):
                percent = context_usage.get('percent_used', 0)
                return percent >= config.threshold_percent, GCTriggerReason.THRESHOLD

            def collect(self, history, context_usage, config, reason):
                # Implement truncation logic
                ...
    """

    @property
    def name(self) -> str:
        """Unique identifier for this GC plugin (e.g., 'gc_truncate')."""
        ...

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Plugin-specific configuration dictionary.
        """
        ...

    def shutdown(self) -> None:
        """Clean up any resources held by the plugin."""
        ...

    def should_collect(
        self,
        context_usage: Dict[str, Any],
        config: GCConfig
    ) -> Tuple[bool, Optional[GCTriggerReason]]:
        """Check if garbage collection should be triggered.

        Args:
            context_usage: Current context window usage from JaatoClient.get_context_usage().
                Contains: model, context_limit, total_tokens, prompt_tokens,
                output_tokens, turns, percent_used, tokens_remaining.
            config: GC configuration with thresholds and preservation settings.

        Returns:
            Tuple of (should_collect: bool, reason: GCTriggerReason or None).
        """
        ...

    def collect(
        self,
        history: List[Message],
        context_usage: Dict[str, Any],
        config: GCConfig,
        reason: GCTriggerReason
    ) -> Tuple[List[Message], GCResult]:
        """Perform garbage collection on the conversation history.

        Args:
            history: Current conversation history as list of Message objects.
            context_usage: Current context window usage statistics.
            config: GC configuration with thresholds and preservation settings.
            reason: The reason this collection was triggered.

        Returns:
            Tuple of (new_history: List[Message], result: GCResult).
            The new_history should be a modified copy, not the original.
        """
        ...
