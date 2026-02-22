"""Google GenAI Cache Plugin — explicit context caching via CachedContent.

Google GenAI supports explicit context caching through the
``google.genai.caching`` API.  The plugin creates a ``CachedContent``
object containing the session's system instruction and tool definitions,
then returns the cache name so the provider can reference it in
subsequent ``GenerateContentConfig`` requests.

Cache lifecycle:
  1. ``initialize()`` — reads ``enable_caching`` from provider config
  2. ``set_client()`` — receives the ``genai.Client`` for API calls
  3. ``prepare_request()`` — lazily creates/reuses ``CachedContent``;
     returns ``cached_content`` key in result dict when active
  4. ``on_gc_result()`` — tracks prefix invalidation (informational)
  5. ``shutdown()`` — deletes the ``CachedContent`` if one exists

When caching is disabled (default) or the client is unavailable, the
plugin falls back to monitoring-only mode (passthrough for requests,
metrics tracking from ``TokenUsage.cache_read_tokens``).

Token usage data appears in ``usage_metadata.cached_content_token_count``
which the ``GoogleGenAIProvider`` maps to ``TokenUsage.cache_read_tokens``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from shared.instruction_budget import InstructionBudget
    from shared.plugins.gc.base import GCResult
    from jaato_sdk.plugins.model_provider.types import TokenUsage

logger = logging.getLogger(__name__)

# Rough chars-per-token estimate for threshold gating.
# Google requires ~32 768 tokens minimum for context caching.
_MIN_CACHE_TOKENS = 32_768
_CHARS_PER_TOKEN = 4


class GoogleGenAICachePlugin:
    """Cache plugin for Google GenAI with explicit CachedContent support.

    When ``enable_caching`` is ``True`` in provider config and a client
    is attached via ``set_client()``, the plugin creates a server-side
    ``CachedContent`` object containing the session's system instruction
    and tool definitions.  On each ``prepare_request()`` call, it returns
    the cache name in the ``cached_content`` result key so the provider
    can skip re-sending system/tools and instead reference the cached
    prefix.

    When caching is disabled or the content is below the minimum token
    threshold (~32 768 tokens), the plugin operates in monitoring-only
    mode — it still tracks ``cache_read_tokens`` metrics but does not
    create ``CachedContent`` objects.

    Attributes:
        _client: Google GenAI client for ``caches.create/delete`` calls.
        _model_name: Model name for cache creation (e.g. ``gemini-2.5-flash``).
        _enabled: Whether explicit caching is enabled via config.
        _cache_ttl: TTL string for cached content (default ``"3600s"`` = 1 hour).
        _cached_content_name: Name of the active ``CachedContent``, or ``None``.
        _content_hash: SHA-256 of the cached system + tools for change detection.
        _total_cache_read_tokens: Session-level aggregate of cache reads.
        _total_cache_creation_tokens: Session-level aggregate of cache writes.
        _gc_invalidation_count: Number of GC ops that removed content.
        _budget: Current InstructionBudget (informational).
    """

    def __init__(self):
        """Initialize with defaults (caching disabled until configured)."""
        # Client and model (set externally)
        self._client: Any = None
        self._model_name: Optional[str] = None

        # Cache state
        self._enabled: bool = False
        self._cache_ttl: str = "3600s"
        self._cached_content_name: Optional[str] = None
        self._content_hash: Optional[str] = None

        # Metrics
        self._total_cache_read_tokens: int = 0
        self._total_cache_creation_tokens: int = 0
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
        """Initialize with provider-specific configuration.

        Args:
            config: Provider config extras.  Recognised keys:

                - ``enable_caching`` (bool): Enable explicit context caching.
                  Defaults to ``False``.
                - ``cache_ttl`` (str): TTL for cached content, e.g. ``"3600s"``
                  (1 hour) or ``"1800s"`` (30 min).  Defaults to ``"3600s"``.
                - ``model_name`` (str): Model identifier, injected by session
                  wiring.
        """
        if not config:
            return
        self._enabled = bool(config.get("enable_caching", False))
        self._cache_ttl = config.get("cache_ttl", "3600s")
        self._model_name = config.get("model_name")

    def shutdown(self) -> None:
        """Delete the active CachedContent (if any) and release resources."""
        self._delete_cached_content()

    # ==================== Client ====================

    def set_client(self, client: Any) -> None:
        """Receive the Google GenAI client for cache management API calls.

        Called by ``GoogleGenAIProvider.set_cache_plugin()`` after the
        provider is initialized.

        Args:
            client: A ``google.genai.Client`` instance.
        """
        self._client = client

    # ==================== Budget ====================

    def set_budget(self, budget: "InstructionBudget") -> None:
        """Receive InstructionBudget updates.

        The budget is stored for future budget-aware caching decisions
        (e.g. caching PRESERVABLE conversation prefix).

        Args:
            budget: The current instruction budget.
        """
        self._budget = budget

    # ==================== Pre-Request ====================

    def prepare_request(
        self,
        system: Any,
        tools: List[Any],
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create or reuse a CachedContent and return its name.

        When caching is active, computes a hash of *system + tools*.
        If the hash differs from the previous call (content changed),
        the old cache is deleted and a new one created.  The cache
        name is returned in the ``cached_content`` key so the provider
        can include it in ``GenerateContentConfig``.

        When caching is inactive (disabled, no client, content too small),
        returns the inputs unchanged (monitoring-only passthrough).

        Args:
            system: System instruction (string or structured).
            tools: Tool definitions (``ToolSchema`` objects or dicts).
            messages: Message history (unused — history is managed by
                the chat object).

        Returns:
            Dict with ``system``, ``tools``, ``messages``,
            ``cache_breakpoint_index``, and optionally ``cached_content``
            (the ``CachedContent.name`` string).
        """
        base = {
            "system": system,
            "tools": tools,
            "messages": messages,
            "cache_breakpoint_index": -1,
        }

        if not self._enabled or not self._client or not self._model_name:
            return base

        # Check if content is large enough to warrant caching
        if not self._meets_token_threshold(system, tools):
            return base

        # Detect content changes via hash
        new_hash = self._compute_content_hash(system, tools)
        if new_hash != self._content_hash:
            self._delete_cached_content()
            self._create_cached_content(system, tools)
            self._content_hash = new_hash

        if self._cached_content_name:
            base["cached_content"] = self._cached_content_name

        return base

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
        if (
            hasattr(usage, "cache_creation_tokens")
            and usage.cache_creation_tokens is not None
            and usage.cache_creation_tokens > 0
        ):
            self._total_cache_creation_tokens += usage.cache_creation_tokens

    # ==================== GC Coordination ====================

    def on_gc_result(self, result: "GCResult") -> None:
        """Track when GC potentially disrupts the cached prefix.

        Content removal may invalidate the implicit or explicit cache.
        This is primarily informational — the explicit ``CachedContent``
        stores system + tools which are not affected by conversation GC.

        Args:
            result: The GC result with information about removed content.
        """
        if result.tokens_freed > 0:
            self._gc_invalidation_count += 1
            logger.debug(
                "Google GenAI cache: GC freed %d tokens "
                "(explicit CachedContent unaffected — only system+tools cached)",
                result.tokens_freed,
            )

    # ==================== Metrics ====================

    @property
    def total_cache_read_tokens(self) -> int:
        """Session-level total cache read tokens."""
        return self._total_cache_read_tokens

    @property
    def total_cache_creation_tokens(self) -> int:
        """Session-level total cache creation tokens."""
        return self._total_cache_creation_tokens

    @property
    def gc_invalidation_count(self) -> int:
        """Number of GC ops that may have disrupted the cache prefix."""
        return self._gc_invalidation_count

    @property
    def cached_content_name(self) -> Optional[str]:
        """Name of the active CachedContent, or None."""
        return self._cached_content_name

    # ==================== Internal Helpers ====================

    def _meets_token_threshold(self, system: Any, tools: List[Any]) -> bool:
        """Estimate whether content is large enough for caching.

        Google requires ~32 768 tokens minimum for context caching.
        Uses a rough chars-per-token heuristic to avoid an API round-trip.

        Args:
            system: System instruction.
            tools: Tool definitions.

        Returns:
            True if estimated token count meets the minimum threshold.
        """
        char_count = len(str(system)) if system else 0
        for tool in (tools or []):
            char_count += len(str(tool))
        estimated_tokens = char_count // _CHARS_PER_TOKEN
        return estimated_tokens >= _MIN_CACHE_TOKENS

    def _compute_content_hash(self, system: Any, tools: List[Any]) -> str:
        """Compute a stable hash of system + tools for change detection.

        Args:
            system: System instruction.
            tools: Tool definitions.

        Returns:
            SHA-256 hex digest.
        """
        hasher = hashlib.sha256()
        hasher.update(str(system).encode("utf-8", errors="replace"))
        for tool in (tools or []):
            hasher.update(str(tool).encode("utf-8", errors="replace"))
        return hasher.hexdigest()

    def _create_cached_content(self, system: Any, tools: List[Any]) -> None:
        """Create a CachedContent on the server with system + tools.

        On failure (API error, quota, content too small for server-side
        validation), logs a debug message and leaves
        ``_cached_content_name`` as ``None`` so subsequent requests
        fall through to non-cached mode.

        Args:
            system: System instruction (string).
            tools: Tool definitions (``ToolSchema`` objects or dicts).
        """
        try:
            from google.genai import types as genai_types

            sdk_tools = self._convert_tools_to_sdk(tools)

            config = genai_types.CreateCachedContentConfig(
                system_instruction=str(system) if system else None,
                tools=sdk_tools,
                ttl=self._cache_ttl,
                display_name="jaato_session_cache",
            )

            cached = self._client.caches.create(
                model=self._model_name,
                config=config,
            )
            self._cached_content_name = cached.name
            logger.debug(
                "Google GenAI cache: created CachedContent %s (model=%s, ttl=%s)",
                cached.name,
                self._model_name,
                self._cache_ttl,
            )
        except Exception as e:
            logger.debug("Google GenAI cache: creation failed: %s", e)
            self._cached_content_name = None

    def _delete_cached_content(self) -> None:
        """Delete the active CachedContent from the server.

        Silently ignores errors (cache may have already expired).
        """
        if not self._cached_content_name or not self._client:
            return
        try:
            self._client.caches.delete(name=self._cached_content_name)
            logger.debug(
                "Google GenAI cache: deleted CachedContent %s",
                self._cached_content_name,
            )
        except Exception as e:
            logger.debug(
                "Google GenAI cache: delete failed (may have expired): %s", e
            )
        self._cached_content_name = None

    @staticmethod
    def _convert_tools_to_sdk(tools: List[Any]) -> Optional[List[Any]]:
        """Convert tool definitions to SDK Tool format for cache creation.

        Handles both ``ToolSchema`` objects (with ``name``, ``description``,
        ``parameters`` attributes) and raw dicts.

        Args:
            tools: Tool definitions in either format.

        Returns:
            List containing a single ``types.Tool`` with all function
            declarations, or ``None`` if no tools.
        """
        if not tools:
            return None
        try:
            from google.genai import types as genai_types

            func_decls = []
            for tool in tools:
                if hasattr(tool, "name") and hasattr(tool, "description"):
                    # ToolSchema object
                    params = getattr(tool, "parameters", None) or getattr(
                        tool, "input_schema", None
                    )
                    func_decls.append(
                        genai_types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description or "",
                            parameters_json_schema=params,
                        )
                    )
                elif isinstance(tool, dict):
                    # Raw dict format
                    params = tool.get("parameters") or tool.get("input_schema")
                    func_decls.append(
                        genai_types.FunctionDeclaration(
                            name=tool.get("name", ""),
                            description=tool.get("description", ""),
                            parameters_json_schema=params,
                        )
                    )
            if func_decls:
                return [genai_types.Tool(function_declarations=func_decls)]
        except Exception as e:
            logger.debug("Google GenAI cache: tool conversion failed: %s", e)
        return None


def create_plugin() -> GoogleGenAICachePlugin:
    """Factory function for plugin discovery."""
    return GoogleGenAICachePlugin()
