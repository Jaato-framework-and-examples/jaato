"""Google GenAI Cache Plugin — monitoring for implicit context caching.

Google GenAI (Gemini) provides automatic context caching where the
system detects repeated input prefixes across requests and serves
them from cache at reduced cost.  Cache hit data appears in
``usage_metadata.cached_content_token_count``, which the
``GoogleGenAIProvider`` maps to ``TokenUsage.cache_read_tokens``.

This plugin:
  - Does NOT inject cache control annotations (``prepare_request`` is
    a passthrough)
  - Tracks cache hit metrics from ``usage.cache_read_tokens``
  - Monitors GC operations that may disrupt the implicit cache prefix
  - Provides session-level cache hit rate aggregates

Future enhancement: Active caching via the ``google.genai.caching``
API would allow explicit prefix pinning through ``CachedContent``
objects.  This requires extending the ``prepare_request()`` return
dict with a ``cached_content`` key that the provider resolves to a
``CachedContent`` name reference in the ``GenerateContentConfig``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from shared.instruction_budget import InstructionBudget
    from shared.plugins.gc.base import GCResult
    from jaato_sdk.plugins.model_provider.types import TokenUsage

logger = logging.getLogger(__name__)


class GoogleGenAICachePlugin:
    """Monitoring-only cache plugin for Google GenAI's implicit caching.

    Since Google GenAI caching is automatic for supported models, this
    plugin does not modify requests.  It provides:
      - Session-level cache read token tracking
      - GC invalidation counting (for observability)
      - Advisory logging when GC disrupts the cached prefix

    Attributes:
        _total_cache_read_tokens: Session-level aggregate of cache reads.
        _gc_invalidation_count: Number of GC ops that removed content,
            potentially disrupting the implicit cache prefix.
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
        return "cache_google_genai"

    @property
    def provider_name(self) -> str:
        """Serves the Google GenAI provider."""
        return "google_genai"

    @property
    def compatible_models(self) -> List[str]:
        """Glob patterns for compatible Gemini models.

        Context caching is available for Gemini 1.5+ models.
        """
        return ["gemini-1.5*", "gemini-2.0*", "gemini-2.5*", "gemini-3*"]

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize (no configuration needed for monitoring-only plugin).

        Args:
            config: Ignored — Google GenAI implicit caching has no
                configurable parameters at the plugin level.
        """
        pass

    def shutdown(self) -> None:
        """Clean up resources (no-op)."""
        pass

    # ==================== Budget ====================

    def set_budget(self, budget: "InstructionBudget") -> None:
        """Receive InstructionBudget updates (informational).

        The budget is stored but not used for request annotation
        since Google GenAI caching is implicit.

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
        """Pass through without modification — caching is implicit.

        Args:
            system: System instruction in API format.
            tools: Tool definitions.
            messages: Message history.

        Returns:
            Unmodified inputs with ``cache_breakpoint_index: -1``.
        """
        return {
            "system": system,
            "tools": tools,
            "messages": messages,
            "cache_breakpoint_index": -1,
        }

    # ==================== Post-Response ====================

    def extract_cache_usage(self, usage: "TokenUsage") -> None:
        """Track session-level cache metrics from Google GenAI responses.

        The ``GoogleGenAIProvider`` populates ``cache_read_tokens``
        from the SDK's ``usage_metadata.cached_content_token_count``.

        Args:
            usage: Token usage from the provider response.
        """
        if usage.cache_read_tokens is not None and usage.cache_read_tokens > 0:
            self._total_cache_read_tokens += usage.cache_read_tokens

    # ==================== GC Coordination ====================

    def on_gc_result(self, result: "GCResult") -> None:
        """Track when GC potentially breaks the implicit cache prefix.

        Any content removal may disrupt the implicit prefix match,
        causing cache misses in subsequent requests.

        Args:
            result: The GC result with information about removed content.
        """
        if result.tokens_freed > 0:
            self._gc_invalidation_count += 1
            logger.debug(
                "Google GenAI implicit cache prefix may be disrupted: "
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


def create_plugin() -> GoogleGenAICachePlugin:
    """Factory function for plugin discovery."""
    return GoogleGenAICachePlugin()
