"""Base types and protocol for Context Garbage Collection plugins.

This module defines the interface that all GC strategy plugins must implement,
along with supporting types for configuration, results, and trigger reasons.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from ..model_provider.types import Message

if TYPE_CHECKING:
    from shared.instruction_budget import InstructionBudget, InstructionSource


class GCTriggerReason(Enum):
    """Reason why garbage collection was triggered."""

    THRESHOLD = "threshold"      # Context usage exceeded threshold percentage
    MANUAL = "manual"            # Explicitly requested by caller
    TURN_LIMIT = "turn_limit"    # Maximum turn count exceeded
    PRE_MESSAGE = "pre_message"  # Triggered before sending a message
    CONTEXT_LIMIT = "context_limit"  # Model rejected request due to context limit exceeded


@dataclass
class GCRemovalItem:
    """Describes a single item to be removed from the instruction budget.

    Used by GC plugins to report what was removed, so the session can
    synchronize the budget with the actual history changes.

    Attributes:
        source: The instruction source type (CONVERSATION, ENRICHMENT, etc.).
        child_key: Specific child key to remove (None = remove entire source).
        tokens_freed: Estimated tokens freed by removing this item.
        reason: Description of why this item was removed.
        message_ids: Message IDs that were removed (for history sync).
    """
    source: "InstructionSource"
    child_key: Optional[str] = None
    tokens_freed: int = 0
    reason: str = ""
    message_ids: List[str] = field(default_factory=list)


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

    removal_list: List[GCRemovalItem] = field(default_factory=list)
    """Structured list of items removed for budget synchronization.

    Each item describes what was removed from the budget, allowing the
    session to sync the InstructionBudget with the actual history changes.
    """

    @property
    def tokens_freed(self) -> int:
        """Calculate tokens freed by this GC operation."""
        return max(0, self.tokens_before - self.tokens_after)


def _get_pressure_percent() -> Optional[float]:
    """Get pressure_percent from environment, returning None if 0."""
    value = float(os.getenv('JAATO_GC_PRESSURE', '90.0'))
    return value if value > 0 else None


@dataclass
class GCConfig:
    """Configuration for context garbage collection.

    Controls when GC triggers and what content to preserve.

    Supports two operating modes:
    - Threshold Mode (default): GC triggers when usage >= threshold_percent
    - Continuous Mode: GC runs after every turn if usage > target_percent
      (enabled by setting pressure_percent to 0 or None)
    """

    # Trigger settings
    threshold_percent: float = field(
        default_factory=lambda: float(os.getenv('JAATO_GC_THRESHOLD', '80.0'))
    )
    """Trigger GC when context usage exceeds this percentage.

    Can be overridden via JAATO_GC_THRESHOLD environment variable.
    Ignored in continuous mode (when pressure_percent is 0 or None).
    """

    target_percent: float = field(
        default_factory=lambda: float(os.getenv('JAATO_GC_TARGET', '60.0'))
    )
    """Target context usage after GC. GC will try to reach this level.

    Can be overridden via JAATO_GC_TARGET environment variable.
    """

    pressure_percent: Optional[float] = field(default_factory=_get_pressure_percent)
    """When usage exceeds this, PRESERVABLE content may be collected.

    Can be overridden via JAATO_GC_PRESSURE environment variable.

    When 0 or None: enables continuous GC mode - GC runs after every turn
    if usage exceeds target_percent (threshold_percent is ignored).
    Same priority order (EPHEMERAL → PARTIAL → PRESERVABLE) still applies,
    but PRESERVABLE content is never touched in continuous mode.
    """

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

    @property
    def continuous_mode(self) -> bool:
        """True if continuous GC is enabled (pressure_percent is 0 or None).

        In continuous mode:
        - GC runs after every turn if usage > target_percent
        - threshold_percent is ignored
        - PRESERVABLE content is never touched
        """
        return not self.pressure_percent


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
        reason: GCTriggerReason,
        budget: Optional["InstructionBudget"] = None,
    ) -> Tuple[List[Message], GCResult]:
        """Perform garbage collection on the conversation history.

        Args:
            history: Current conversation history as list of Message objects.
            context_usage: Current context window usage statistics.
            config: GC configuration with thresholds and preservation settings.
            reason: The reason this collection was triggered.
            budget: Optional InstructionBudget for policy-aware GC decisions.
                If provided, plugins can use GC policies (LOCKED, PRESERVABLE,
                EPHEMERAL) to make smarter removal decisions.

        Returns:
            Tuple of (new_history: List[Message], result: GCResult).
            The new_history should be a modified copy, not the original.
            The result.removal_list should contain GCRemovalItem entries
            describing what was removed, so the session can sync the budget.
        """
        ...
