"""ZhipuAI Cache Plugin — monitoring-only for implicit caching.

ZhipuAI uses fully automatic/implicit caching.  The system computes
input message content and identifies prefixes identical or highly
similar to previous requests.  No code changes or annotations are
required.

This plugin:
  - Does NOT inject cache_control annotations (``prepare_request`` is a passthrough)
  - Tracks cache hit metrics from ``usage.cache_read_tokens``
  - Monitors GC operations that may disrupt the implicit cache prefix
  - Provides session-level cache hit rate aggregates

Usage data appears in ``usage.prompt_tokens_details.cached_tokens``
(OpenAI-compatible format), mapped to ``TokenUsage.cache_read_tokens``
by the ZhipuAI provider's response parser.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from shared.instruction_budget import InstructionBudget
    from shared.plugins.gc.base import GCResult
    from jaato_sdk.plugins.model_provider.types import TokenUsage

logger = logging.getLogger(__name__)


class ZhipuAICachePlugin:
    """Monitoring-only cache plugin for ZhipuAI's implicit caching.

    Since ZhipuAI caching is fully automatic, this plugin does not
    modify requests.  It provides:
      - Session-level cache read token tracking
      - GC invalidation counting (for observability)
      - Advisory logging when GC disrupts the cached prefix

    Attributes:
        _total_cache_read_tokens: Session-level aggregate of cache reads.
        _gc_invalidation_count: Number of GC ops that removed content.
        _budget: Current InstructionBudget (informational).
    """

    def __init__(self):
        """Initialize with defaults."""
        self._total_cache_read_tokens: int = 0
        self._gc_invalidation_count: int = 0
        self._budget: Optional["InstructionBudget"] = None

    # ==================== Protocol Properties ====================

    @property
    def name(self) -> str:
        """Plugin identifier."""
        return "cache_zhipuai"

    @property
    def provider_name(self) -> str:
        """Serves the ZhipuAI provider."""
        return "zhipuai"

    @property
    def compatible_models(self) -> List[str]:
        """Glob patterns for compatible GLM models."""
        return ["glm-5*", "glm-4.7*", "glm-4.6*", "glm-4.5*"]

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize (no configuration needed for monitoring-only plugin).

        Args:
            config: Ignored — ZhipuAI caching has no configurable parameters.
        """
        pass

    def shutdown(self) -> None:
        """Clean up resources (no-op)."""
        pass

    # ==================== Budget ====================

    def set_budget(self, budget: "InstructionBudget") -> None:
        """Receive InstructionBudget updates (informational).

        The budget is stored but not used for request annotation
        since ZhipuAI caching is implicit.

        Args:
            budget: The current instruction budget.
        """
        self._budget = budget

    # ==================== Pre-Request (Passthrough) ====================

    def prepare_request(
        self,
        system: Any,
        tools: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Pass through without modification — ZhipuAI caching is implicit.

        Args:
            system: System instruction in API format.
            tools: Tool definitions.
            messages: Message history.

        Returns:
            Unmodified inputs.
        """
        return {
            "system": system,
            "tools": tools,
            "messages": messages,
            "cache_breakpoint_index": -1,
        }

    # ==================== Post-Response ====================

    def extract_cache_usage(self, usage: "TokenUsage") -> None:
        """Track session-level cache metrics from ZhipuAI responses.

        Args:
            usage: Token usage from the provider response.
        """
        if usage.cache_read_tokens is not None and usage.cache_read_tokens > 0:
            self._total_cache_read_tokens += usage.cache_read_tokens

    # ==================== GC Coordination ====================

    def on_gc_result(self, result: "GCResult") -> None:
        """Track when GC potentially breaks the implicit cache prefix.

        Any content removal may disrupt the implicit prefix match.

        Args:
            result: The GC result.
        """
        if result.tokens_freed > 0:
            self._gc_invalidation_count += 1
            logger.debug(
                "ZhipuAI implicit cache prefix may be disrupted: "
                "GC freed %d tokens",
                result.tokens_freed,
            )

    # ==================== Metrics ====================

    @property
    def total_cache_read_tokens(self) -> int:
        """Session-level total cache read tokens."""
        return self._total_cache_read_tokens

    @property
    def gc_invalidation_count(self) -> int:
        """Number of GC ops that may have disrupted the cache prefix."""
        return self._gc_invalidation_count


def create_plugin() -> ZhipuAICachePlugin:
    """Factory function for plugin discovery."""
    return ZhipuAICachePlugin()
