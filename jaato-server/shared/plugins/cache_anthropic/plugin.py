"""Anthropic Cache Plugin — explicit breakpoint caching for Claude models.

This plugin implements Anthropic's explicit cache control breakpoints,
placing up to 3 of the 4 available breakpoints on:
  - BP1: System instruction (LOCKED in budget — always stable)
  - BP2: Last tool definition (core tools are LOCKED in budget)
  - BP3: History breakpoint (budget-aware PRESERVABLE/EPHEMERAL boundary)

Key improvements over the previous provider-internal implementation:
  1. Budget-aware BP3 — uses InstructionBudget GC policies instead of
     the simpler ``cache_exclude_recent_turns`` turn-counting heuristic.
  2. GC coordination — ``on_gc_result()`` tracks when GC removes
     PRESERVABLE content that may invalidate the cached prefix.
  3. Decoupled from provider inheritance — ZhipuAIProvider and
     OllamaProvider no longer need ``_enable_caching = False`` hacks.

Cache breakpoint placement:
    ┌─────────────────────────────────────┐
    │  System instruction                 │ ← BP1
    ├─────────────────────────────────────┤
    │  Tool definitions (sorted by name)  │ ← BP2
    ├─────────────────────────────────────┤
    │  LOCKED/PRESERVABLE history turns   │ ← BP3 at policy boundary
    ├─────────────────────────────────────┤
    │  EPHEMERAL turns (recent work)      │ ← Not cached
    │  Latest user message                │
    └─────────────────────────────────────┘
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from shared.instruction_budget import InstructionBudget
    from shared.plugins.gc.base import GCResult
    from jaato_sdk.plugins.model_provider.types import TokenUsage

logger = logging.getLogger(__name__)

# Anthropic requires minimum token thresholds for caching to be effective.
# Content below these thresholds is silently ignored by the API.
CACHE_MIN_TOKENS_SONNET = 1024
CACHE_MIN_TOKENS_OTHER = 2048

# Default number of recent turns to exclude from history caching
# when no InstructionBudget is available (fallback to turn-counting).
DEFAULT_CACHE_EXCLUDE_RECENT_TURNS = 2


class AnthropicCachePlugin:
    """Explicit breakpoint caching for Anthropic Claude models.

    This plugin manages cache_control annotations on system, tool, and
    message content blocks.  It is budget-aware: when an InstructionBudget
    is available, BP3 (history breakpoint) is placed at the
    PRESERVABLE/EPHEMERAL boundary instead of using a fixed turn count.

    Lifecycle:
        1. ``initialize(config)`` — reads enable_caching, cache_ttl, etc.
        2. ``set_budget(budget)`` — receives InstructionBudget updates
        3. Per-turn: ``prepare_request(system, tools, messages)``
        4. Per-turn: ``extract_cache_usage(usage)``
        5. After GC: ``on_gc_result(result)``
        6. ``shutdown()`` — no-op (no persistent resources)

    Attributes:
        _enabled: Whether caching is active.
        _cache_ttl: Cache TTL ("5m" or "1h").
        _cache_history: Whether to place BP3 on message history.
        _cache_exclude_recent_turns: Fallback turn count for BP3
            when no budget is available.
        _enforce_min_tokens: Whether to check minimum token thresholds.
        _model_name: Current model name (for threshold selection).
        _budget: Current InstructionBudget (set via set_budget).
        _prefix_invalidated: Set to True when GC removes PRESERVABLE
            content, indicating the cached prefix may be stale.
        _total_cache_read_tokens: Session-level aggregate of cache reads.
        _total_cache_creation_tokens: Session-level aggregate of cache writes.
        _gc_invalidation_count: Number of GC operations that may have
            invalidated the cached prefix.
    """

    def __init__(self):
        """Initialize with defaults (not yet configured)."""
        self._enabled: bool = False
        self._cache_ttl: str = "5m"
        self._cache_history: bool = True
        self._cache_exclude_recent_turns: int = DEFAULT_CACHE_EXCLUDE_RECENT_TURNS
        self._enforce_min_tokens: bool = True
        self._model_name: Optional[str] = None

        # Budget reference (set via set_budget)
        self._budget: Optional["InstructionBudget"] = None

        # Tracking state
        self._prefix_invalidated: bool = False
        self._total_cache_read_tokens: int = 0
        self._total_cache_creation_tokens: int = 0
        self._gc_invalidation_count: int = 0

    # ==================== Protocol Properties ====================

    @property
    def name(self) -> str:
        """Plugin identifier."""
        return "cache_anthropic"

    @property
    def provider_name(self) -> str:
        """Serves the Anthropic provider."""
        return "anthropic"

    @property
    def compatible_models(self) -> List[str]:
        """Glob patterns for compatible Claude models."""
        return [
            "claude-sonnet-4*",
            "claude-opus-4*",
            "claude-haiku-4*",
            "claude-3-5-*",
            "claude-3-7-*",
        ]

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with provider config extras.

        Reads the same configuration keys that AnthropicProvider previously
        handled internally.

        Args:
            config: Dict with optional keys:
                - enable_caching (bool): Enable/disable caching.
                - cache_ttl (str): "5m" or "1h".
                - cache_history (bool): Whether to cache history (BP3).
                - cache_exclude_recent_turns (int): Fallback turn count.
                - cache_min_tokens (bool): Enforce min token threshold.
                - model_name (str): Current model for threshold selection.
        """
        if config is None:
            config = {}

        from shared.plugins.model_provider.anthropic.env import resolve_enable_caching
        self._enabled = config.get("enable_caching", resolve_enable_caching())
        self._cache_ttl = config.get("cache_ttl", "5m")
        self._cache_history = config.get("cache_history", True)
        self._cache_exclude_recent_turns = config.get(
            "cache_exclude_recent_turns", DEFAULT_CACHE_EXCLUDE_RECENT_TURNS
        )
        self._enforce_min_tokens = config.get("cache_min_tokens", True)
        self._model_name = config.get("model_name")

    def shutdown(self) -> None:
        """Clean up resources (no-op — no persistent resources)."""
        pass

    # ==================== Budget ====================

    def set_budget(self, budget: "InstructionBudget") -> None:
        """Receive an updated InstructionBudget.

        The plugin reads GC policy assignments from this budget in
        ``prepare_request()`` to decide BP3 placement.

        Args:
            budget: The current instruction budget.
        """
        self._budget = budget

    def set_model_name(self, model_name: str) -> None:
        """Update the model name (affects min-token threshold selection).

        Args:
            model_name: The model ID (e.g., 'claude-sonnet-4-20250514').
        """
        self._model_name = model_name

    # ==================== Pre-Request Annotation ====================

    def prepare_request(
        self,
        system: Any,
        tools: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Inject cache_control annotations on system, tools, and messages.

        If caching is disabled, returns inputs unmodified.

        Args:
            system: System instruction in Anthropic API format
                (list of content blocks or str).
            tools: Tool definitions in Anthropic API format (list of dicts).
            messages: Message history in Anthropic API format.

        Returns:
            Dict with ``system``, ``tools``, ``messages``, and optionally
            ``cache_breakpoint_index`` (int, -1 means no history BP).
        """
        if not self._enabled:
            return {
                "system": system,
                "tools": tools,
                "messages": messages,
                "cache_breakpoint_index": -1,
            }

        cache_type = {"type": "ephemeral"}

        # BP1: System instruction
        annotated_system = self._inject_system_breakpoint(system, cache_type)

        # BP2: Tool definitions (sort by name for cache stability)
        annotated_tools = self._inject_tool_breakpoint(tools, cache_type)

        # BP3: History breakpoint
        bp_index = self._compute_history_breakpoint()

        # Reset prefix invalidation flag after using it
        self._prefix_invalidated = False

        return {
            "system": annotated_system,
            "tools": annotated_tools,
            "messages": messages,
            "cache_breakpoint_index": bp_index,
        }

    def _inject_system_breakpoint(
        self,
        system: Any,
        cache_type: Dict[str, str],
    ) -> Any:
        """Add cache_control to the system instruction (BP1).

        Args:
            system: System in Anthropic format (list of content blocks).
            cache_type: The cache_control dict (e.g., {"type": "ephemeral"}).

        Returns:
            Annotated system (same structure with cache_control added).
        """
        if not system:
            return system

        # system can be a list of content blocks or a string
        if isinstance(system, list) and len(system) > 0:
            # Annotate the last block (Anthropic requires breakpoint on last
            # block in a logical group)
            last_block = system[-1]
            text = last_block.get("text", "")
            if text and self._should_cache_content(text):
                # Deep copy to avoid mutating caller's data
                system = [dict(b) for b in system]
                system[-1] = dict(system[-1])
                system[-1]["cache_control"] = cache_type

        return system

    def _inject_tool_breakpoint(
        self,
        tools: List[Dict[str, Any]],
        cache_type: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Sort tools by name and add cache_control to the last one (BP2).

        Sorting ensures consistent ordering across sessions, preventing
        cache invalidation due to tool registration order changes.

        Args:
            tools: Tool definitions in Anthropic API format.
            cache_type: The cache_control dict.

        Returns:
            Sorted, annotated tools list.
        """
        if not tools:
            return tools

        # Sort by name for cache stability
        sorted_tools = sorted(tools, key=lambda t: t.get("name", ""))

        # Estimate combined size for threshold check
        tools_json = json.dumps(sorted_tools)
        if self._should_cache_content(tools_json):
            # Deep copy last tool to add cache_control
            sorted_tools = [dict(t) for t in sorted_tools]
            sorted_tools[-1] = dict(sorted_tools[-1])
            sorted_tools[-1]["cache_control"] = cache_type

        return sorted_tools

    def _compute_history_breakpoint(self) -> int:
        """Compute the optimal history message index for BP3.

        Uses budget-aware policy boundary when available, falling back
        to the recency-based turn count heuristic.

        Returns:
            Message index for the breakpoint, or -1 to skip.
        """
        if not self._cache_history:
            return -1

        # Try budget-aware placement first
        if self._budget:
            bp = self._find_policy_boundary()
            if bp >= 0:
                return bp

        # Fallback: recency-based (original provider logic)
        return -1  # Caller uses the existing recency logic as fallback

    def _find_policy_boundary(self) -> int:
        """Find the PRESERVABLE/EPHEMERAL boundary in conversation turns.

        Walks the InstructionBudget's CONVERSATION children to find the
        last LOCKED or PRESERVABLE turn.  This is the optimal BP3 position:
        everything before it is stable (won't be GC'd except under extreme
        pressure), everything after is EPHEMERAL.

        Returns:
            The message index to place the breakpoint on, or -1 if no
            suitable boundary is found.
        """
        from shared.instruction_budget import InstructionSource, GCPolicy

        if not self._budget:
            return -1

        conv_entry = self._budget.get_entry(InstructionSource.CONVERSATION)
        if not conv_entry or not conv_entry.children:
            return -1

        # Find the last stable child (LOCKED or PRESERVABLE)
        last_stable_message_ids: List[str] = []
        for _key, child in conv_entry.children.items():
            policy = child.effective_gc_policy()
            if policy in (GCPolicy.LOCKED, GCPolicy.PRESERVABLE):
                if child.message_ids:
                    last_stable_message_ids = child.message_ids

        if not last_stable_message_ids:
            return -1

        # Return the index of the last message ID from the last stable child.
        # The caller will map this to the actual message list position.
        # For now, we encode the message ID; the provider will resolve it.
        # Store it as metadata so the provider can look it up.
        # Actually, we need to return an integer index.  Since we don't have
        # the message list here, we store the boundary info and the provider
        # uses _compute_history_cache_breakpoint as fallback.
        # For the budget-aware path, we need the provider to look up message
        # positions.  We'll return a sentinel that tells the provider to use
        # the budget-based boundary.

        # Implementation note: The budget children track message_ids, but we
        # need to map those to positions in the Anthropic-format message list.
        # Since we don't have that mapping here, we store the last stable
        # message_id and let the provider resolve it via the messages list.
        # For now, return -2 as a sentinel meaning "use budget boundary"
        # and store the target message_id.
        self._budget_bp3_message_id = (
            last_stable_message_ids[-1] if last_stable_message_ids else None
        )
        return -2  # Sentinel: budget-based breakpoint

    # ==================== Post-Response ====================

    def extract_cache_usage(self, usage: "TokenUsage") -> None:
        """Track session-level cache metrics.

        Args:
            usage: Token usage from the provider response.
        """
        if usage.cache_read_tokens is not None and usage.cache_read_tokens > 0:
            self._total_cache_read_tokens += usage.cache_read_tokens
        if usage.cache_creation_tokens is not None and usage.cache_creation_tokens > 0:
            self._total_cache_creation_tokens += usage.cache_creation_tokens

    # ==================== GC Coordination ====================

    def on_gc_result(self, result: "GCResult") -> None:
        """Track when GC may invalidate the cached prefix.

        Only PRESERVABLE removal risks disrupting the cached prefix.
        EPHEMERAL and PARTIAL removal is safe — that content was outside
        the cached prefix (or excluded by the breakpoint).

        Args:
            result: The GC result with removal details.
        """
        preservable_removed = result.details.get("preservable_removed", 0)
        if preservable_removed > 0:
            self._prefix_invalidated = True
            self._gc_invalidation_count += 1
            logger.debug(
                "Cache prefix may be invalidated: GC removed %d PRESERVABLE items",
                preservable_removed,
            )

    # ==================== Threshold Checking ====================

    def _should_cache_content(self, content: str) -> bool:
        """Check if content meets minimum token threshold for caching.

        Anthropic silently ignores cache_control on content below the
        minimum threshold, so adding it would waste the breakpoint.

        Args:
            content: Text content to check.

        Returns:
            True if content is large enough to benefit from caching.
        """
        if not self._enforce_min_tokens:
            return True
        min_tokens = self._get_cache_min_tokens()
        estimated = self._estimate_tokens(content)
        return estimated >= min_tokens

    def _get_cache_min_tokens(self) -> int:
        """Get minimum token threshold based on model.

        Returns:
            1024 for Claude 3.5 Sonnet, 2048 for other models.
        """
        if self._model_name and "sonnet" in self._model_name.lower() and "3-5" in self._model_name:
            return CACHE_MIN_TOKENS_SONNET
        return CACHE_MIN_TOKENS_OTHER

    @staticmethod
    def _estimate_tokens(content: str) -> int:
        """Estimate token count (~4 chars per token).

        Args:
            content: Text to estimate.

        Returns:
            Estimated token count.
        """
        return len(content) // 4

    # ==================== Metrics ====================

    @property
    def total_cache_read_tokens(self) -> int:
        """Session-level total cache read tokens (90% savings)."""
        return self._total_cache_read_tokens

    @property
    def total_cache_creation_tokens(self) -> int:
        """Session-level total cache creation tokens (1.25x write cost)."""
        return self._total_cache_creation_tokens

    @property
    def gc_invalidation_count(self) -> int:
        """Number of GC ops that may have invalidated cached prefix."""
        return self._gc_invalidation_count

    @property
    def prefix_invalidated(self) -> bool:
        """Whether GC recently removed PRESERVABLE content."""
        return self._prefix_invalidated


def create_plugin() -> AnthropicCachePlugin:
    """Factory function for plugin discovery."""
    return AnthropicCachePlugin()
