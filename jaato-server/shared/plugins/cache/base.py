"""Base types and protocol for Cache Control plugins.

Cache plugins manage provider-specific prompt caching strategies.
They are CONSUMERS of the InstructionBudget's GC policies -- they read
LOCKED/PRESERVABLE/EPHEMERAL assignments to decide where to place cache
breakpoints, what content is worth the write premium, and what to skip.

Each provider has different caching mechanics (explicit breakpoints vs
implicit/automatic), so cache plugins are provider-specific.  The protocol
defines a common interface for:
- Pre-request annotation (``prepare_request``)
- Post-response metric extraction (``extract_cache_usage``)
- GC-cache coordination (``on_gc_result``)
- Budget-aware breakpoint placement (``set_budget``)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from shared.instruction_budget import InstructionBudget
    from shared.plugins.gc.base import GCResult
    from jaato_sdk.plugins.model_provider.types import TokenUsage


@runtime_checkable
class CachePlugin(Protocol):
    """Protocol for provider-specific cache control plugins.

    Cache plugins are CONSUMERS of the InstructionBudget's GC policies.
    They read the LOCKED/PRESERVABLE/EPHEMERAL assignments to make
    caching decisions -- they don't invent their own content lifecycle model.

    Lifecycle:
        1. ``initialize(config)`` -- called once with provider config extras
        2. ``set_budget(budget)`` -- called whenever the InstructionBudget updates
        3. Per-turn:
           a. ``prepare_request(...)`` -- annotate request before sending
           b. ``extract_cache_usage(usage)`` -- extract metrics from response
        4. After GC: ``on_gc_result(result)`` -- track prefix invalidation
        5. ``shutdown()`` -- clean up

    Note: Methods are synchronous, matching the GCPlugin protocol pattern.
    """

    @property
    def name(self) -> str:
        """Plugin identifier (e.g., 'cache_anthropic', 'cache_zhipuai')."""
        ...

    @property
    def provider_name(self) -> str:
        """Which provider this plugin serves (e.g., 'anthropic', 'zhipuai').

        Used by the session to match cache plugins to the active provider.
        """
        ...

    @property
    def compatible_models(self) -> List[str]:
        """Glob patterns for compatible models (e.g., ['claude-sonnet-4*']).

        Currently informational; matching is done by ``provider_name``.
        """
        ...

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with provider-specific configuration.

        Args:
            config: Provider config extras (e.g., enable_caching, cache_ttl).
        """
        ...

    def shutdown(self) -> None:
        """Clean up resources."""
        ...

    def set_budget(self, budget: "InstructionBudget") -> None:
        """Receive the current InstructionBudget.

        Called whenever the budget updates (after populate, after GC, etc.).
        The plugin stores the reference and reads GC policies in
        ``prepare_request()`` to decide breakpoint placement.

        Args:
            budget: The current instruction budget with GC policy assignments.
        """
        ...

    def prepare_request(
        self,
        system: Any,
        tools: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Annotate the API request with cache control markers.

        Called BEFORE the LLM request is sent, after the provider has
        converted system/tools/messages to its native API format.

        The plugin reads its stored InstructionBudget (from ``set_budget``)
        to decide where to place breakpoints based on GC policies.

        For explicit-caching providers (Anthropic), this injects
        ``cache_control`` dicts on content blocks.  For implicit-caching
        providers (ZhipuAI), this is a no-op passthrough.

        Args:
            system: System instruction in provider API format
                (e.g., list of content blocks for Anthropic).
            tools: Tool definitions in provider API format.
            messages: Message history in provider API format.

        Returns:
            Dict with keys ``system``, ``tools``, ``messages`` --
            the annotated versions.  May also include
            ``cache_breakpoint_index`` for history breakpoint placement.
        """
        ...

    def extract_cache_usage(self, usage: "TokenUsage") -> None:
        """Extract cache metrics from the response.

        Called AFTER the LLM response is received.  The ``TokenUsage``
        already contains ``cache_read_tokens`` and ``cache_creation_tokens``
        populated by the provider's converter.  This hook lets the plugin
        track session-level cache hit rate aggregates.

        Args:
            usage: Token usage from the provider response.
        """
        ...

    def on_gc_result(self, result: "GCResult") -> None:
        """Handle GC collection notification.

        Called AFTER GC collects content from the conversation.
        The cache plugin uses this to track prefix invalidation
        and adjust breakpoint strategy for subsequent requests.

        Args:
            result: The GC result with removal details.
        """
        ...
