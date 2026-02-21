# jaato Cache Plugin System — Design Document

## Overview

This document captures the design for a provider-specific prompt caching system for the jaato agentic AI framework. The cache plugins leverage KV caching (Key-Value caching in the transformer attention mechanism) offered by LLM providers to reduce token costs by up to 90% and latency by up to 85%.

The central design principle is that **cache plugins are consumers of the GC Budget plugin's content map**. The GC Budget plugin already classifies every content block in the conversation as `locked`, `conditional`, or `ephemeral` — the cache plugin reads this same taxonomy to decide where to place cache breakpoints, what's worth the write premium, and what to skip.

---

## Background: What Is KV Caching?

LLM inference works by feeding the entire prompt through the model on every iteration. The transformer's attention mechanism computes K (Key) and V (Value) matrices from the input embeddings (`embeddings × WK` and `embeddings × WV`). These matrices don't change for tokens already processed.

Providers cache these K and V matrices between requests. When a new request shares a prefix with a previously processed prompt, the provider reuses the cached KV matrices instead of recomputing them. This is a **prefix match** — the cache is useful when the beginning of your prompt is identical across requests, and partial matches still benefit from the cached portion.

Key properties:

- **What's cached**: The K and V matrices (intermediate tensor representations), not raw text or responses.
- **Match type**: Prefix-based. The longer the identical prefix, the bigger the savings.
- **Independence from output**: Caching doesn't affect generation. Same prompt with cache hit produces same distribution of outputs. Temperature, top_p, top_k don't invalidate caches.
- **Provider-specific mechanics**: Each provider implements caching differently (automatic vs explicit, different TTLs, different pricing).

Reference: [ngrok blog — Prompt caching: 10x cheaper LLM tokens, but how?](https://ngrok.com/blog/prompt-caching/)

---

## The Core Tension: GC vs Cache

The GC Budget plugin and the cache system have a natural tension:

- **GC wants to mutate the prefix** to reclaim token budget (truncate, summarize, compact).
- **Caching wants prefix stability** to maximize cache hits and save cost.

The resolution is coordination: the cache plugin reads the GC Budget's content categorization to understand which content is stable (and worth caching) vs volatile (and not worth the write premium), and the GC Budget notifies the cache plugin when it collects content that might invalidate the cached prefix.

---

## GC Categories as Cache Priorities

The GC Budget plugin categorizes every content block in the conversation. This categorization maps directly to cache priority:

| GC Category | GC Behavior | Cache Implication |
|---|---|---|
| **locked** | Never collected, always in context | **Highest cache priority** — stable prefix. System prompt, persistent tool schemas, core instructions. Always worth caching because guaranteed present on every request. |
| **conditional** | Collected only when conditions are met | **Cacheable but volatile** — tool results that matter until superseded, conversation context relevant until a topic shift. Cache it, but track when conditions trigger collection to anticipate invalidation. |
| **ephemeral** | Freely collected when budget is tight | **Don't waste cache budget** — intermediate reasoning, exploratory tool calls, verbose outputs. The GC will likely collect these before the cache TTL expires anyway. |

---

## Provider Comparison

### ZhipuAI (Z.AI) — Implicit/Automatic Caching

- **Mechanism**: Fully automatic. The system computes input message content and identifies prefixes identical or highly similar to previous requests. No code changes required.
- **Configuration**: None. No `cache_control` parameters or breakpoints.
- **Supported models**: All GLM models (GLM-5, GLM-4.7, GLM-4.6, GLM-4.5 series).
- **Monitoring**: `usage.prompt_tokens_details.cached_tokens` in the response.
- **API format**: OpenAI-compatible (`/v4/chat/completions`).
- **Pricing**: Cached tokens billed at lower prices (details in Z.AI pricing docs).
- **Cache lifetime**: Not explicitly documented; automatic eviction.

Reference: [Z.AI Context Caching Documentation](https://docs.z.ai/guides/capabilities/cache)

### Anthropic — Explicit Cache Control

- **Mechanism**: Manual placement of `cache_control` breakpoints on content blocks, or automatic mode with a single top-level `cache_control`.
- **Supported models**: Claude Sonnet 4/4.5, Claude Opus 4/4.5, Claude Haiku 4.5.
- **Max breakpoints**: 4 per request.
- **Cache processing order**: `tools → system → messages`.
- **Min tokens**: 1024 (Sonnet/Opus), 2048 (Haiku).
- **Default TTL**: 5 minutes, refreshed on each hit at no extra cost.
- **Extended TTL**: 1 hour (costs 2× base write price).
- **Cache write cost**: 1.25× base input token price.
- **Cache read cost**: 0.10× base input token price (90% savings).
- **Monitoring**: `cache_creation_input_tokens` and `cache_read_input_tokens` in the response usage object.

Anthropic offers two modes:

**Automatic caching** — single top-level `cache_control` on the request body. The system caches everything up to the last cacheable block and the breakpoint moves forward automatically as conversations grow.

**Explicit breakpoints** — place `cache_control: {"type": "ephemeral"}` on individual content blocks for fine-grained control. Up to 4 breakpoints, processed in the order tools → system → messages.

Reference: [Anthropic Prompt Caching Documentation](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)

### Cache Invalidation (Both Providers)

Things that break the cached prefix:

- Any mutation to the system prompt (even whitespace changes).
- Changing `tool_choice` between calls (Anthropic).
- Adding/removing images mid-conversation (Anthropic).
- Reordering JSON keys in tool schemas (some languages randomize key order — Go, Swift).
- Summarizing/truncating conversation history instead of sending the full prefix.
- GC collecting content within the cached prefix region.

---

## Plugin Architecture

```
plugins/
├── cache/                          # Base cache infrastructure
│   ├── __init__.py
│   ├── plugin.py                   # CacheControlPlugin protocol + registry
│   ├── metrics.py                  # CacheMetrics dataclass + aggregation
│   └── events.py                   # CACHE_HIT, CACHE_MISS, CACHE_STATS events
│
├── cache_anthropic/                # Provider-specific
│   ├── __init__.py
│   ├── plugin.py                   # AnthropicCachePlugin
│   ├── breakpoint_strategy.py      # Breakpoint placement logic
│   └── compatible_models.py        # ["claude-sonnet-4-*", "claude-opus-4-*", ...]
│
├── cache_zhipuai/                  # Provider-specific
│   ├── __init__.py
│   ├── plugin.py                   # ZhipuAICachePlugin
│   └── compatible_models.py        # ["glm-5", "glm-4.7", "glm-4.6", ...]
│
├── cache_vertex/                   # Future provider
│   └── ...
│
└── gc/                             # Existing GC Budget plugin, gets new hook
    ├── __init__.py
    ├── plugin.py                   # Modified: notifies cache plugin after collection
    └── strategies.py
```

### Relationship Diagram

```
                     ┌─────────────────────────────┐
                     │      GC Budget Plugin        │
                     │                              │
                     │  Content Registry:           │
                     │  ┌────────┬────────┬──────┐  │
                     │  │ block  │ tokens │ cat  │  │
                     │  ├────────┼────────┼──────┤  │
                     │  │ sys    │ 2,100  │ 🔒   │  │
                     │  │ tools  │ 1,800  │ 🔒   │  │
                     │  │ turn1  │   340  │ 🔶   │  │
                     │  │ tool_r │   890  │ 🔶   │  │
                     │  │ turn2  │   210  │ 🔶   │  │
                     │  │ turn3  │   180  │ 💨   │  │
                     │  │ tool_r │   450  │ 💨   │  │
                     │  │ turn4  │   290  │ 🔶   │  │
                     │  └────────┴────────┴──────┘  │
                     │  🔒=locked 🔶=conditional     │
                     │  💨=ephemeral                 │
                     │                              │
                     │  Budget: 6,260/32,000 tokens │
                     └──────────┬────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │ reads           │ reads            │ notifies
              ▼                 ▼                  ▼
   ┌──────────────────┐  ┌───────────────┐  ┌──────────────┐
   │ Cache Plugin      │  │ GC Collector  │  │ Event Bus    │
   │ (provider-specific)│  │ (budget-based)│  │              │
   │                   │  │               │  │ → TUI        │
   │ "Where do I place │  │ "What can I   │  │ → OTel       │
   │  breakpoints?"    │  │  safely drop?" │  │ → Telegram   │
   └──────────────────┘  └───────────────┘  └──────────────┘
```

---

## Protocol Definition

### GC Categories (from existing GC Budget plugin)

```python
from enum import Enum

class GCCategory(Enum):
    LOCKED = "locked"           # Never collected
    CONDITIONAL = "conditional" # Collected when conditions met
    EPHEMERAL = "ephemeral"     # Freely collectible
```

### Content Map (from existing GC Budget plugin)

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContentBlock:
    """A block in the GC Budget's content registry."""
    block_id: str               # e.g., "system", "tools", "turn_3", "tool_result_5"
    gc_category: GCCategory
    token_count: int
    position: int               # Index in the message sequence
    turn_number: Optional[int] = None
    tool_name: Optional[str] = None
    condition: Optional[str] = None  # For conditional: what triggers collection


@dataclass
class ContentMap:
    """The GC Budget's view of what's in context and how it's categorized."""
    blocks: list[ContentBlock]
    total_tokens: int
    budget_limit: int

    @property
    def locked_tokens(self) -> int:
        return sum(b.token_count for b in self.blocks
                   if b.gc_category == GCCategory.LOCKED)

    @property
    def conditional_tokens(self) -> int:
        return sum(b.token_count for b in self.blocks
                   if b.gc_category == GCCategory.CONDITIONAL)

    @property
    def ephemeral_tokens(self) -> int:
        return sum(b.token_count for b in self.blocks
                   if b.gc_category == GCCategory.EPHEMERAL)

    def stable_prefix_boundary(self) -> int:
        """
        Find the last position where all content up to that point
        is either LOCKED or CONDITIONAL (not yet triggered).
        This is the natural cache breakpoint.
        """
        boundary = 0
        for block in self.blocks:
            if block.gc_category == GCCategory.EPHEMERAL:
                break
            boundary = block.position + 1
        return boundary

    def locked_boundary(self) -> int:
        """Position after the last locked block — the most stable prefix."""
        boundary = 0
        for block in self.blocks:
            if block.gc_category == GCCategory.LOCKED:
                boundary = block.position + 1
        return boundary
```

### Cache Metrics

```python
import time
from dataclasses import dataclass, field


@dataclass
class CacheTurnMetrics:
    """Metrics for a single LLM call."""
    timestamp: float = field(default_factory=time.time)
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    uncached_tokens: int = 0
    total_input_tokens: int = 0
    # GC-informed metrics
    locked_tokens_cached: int = 0
    conditional_tokens_cached: int = 0
    ephemeral_tokens_skipped: int = 0

    @property
    def hit_rate(self) -> float:
        return (self.cache_read_tokens / self.total_input_tokens
                if self.total_input_tokens > 0 else 0.0)

    @property
    def cache_efficiency(self) -> float:
        """
        How much of the cached content is actually stable (locked)?
        High efficiency = we're caching the right stuff.
        Low efficiency = we're caching ephemeral content that
        will get GC'd before the cache TTL expires.
        """
        total_cached = self.cache_read_tokens + self.cache_write_tokens
        if total_cached == 0:
            return 0.0
        return self.locked_tokens_cached / total_cached


@dataclass
class CacheSessionMetrics:
    """Aggregated metrics across a session."""
    turns: list[CacheTurnMetrics] = field(default_factory=list)
    gc_invalidations: int = 0  # Times GC broke our cache

    @property
    def cumulative_hit_rate(self) -> float:
        total_read = sum(t.cache_read_tokens for t in self.turns)
        total_input = sum(t.total_input_tokens for t in self.turns)
        return total_read / total_input if total_input > 0 else 0.0

    @property
    def avg_cache_efficiency(self) -> float:
        efficiencies = [t.cache_efficiency for t in self.turns
                        if t.cache_efficiency > 0]
        return sum(efficiencies) / len(efficiencies) if efficiencies else 0.0
```

### Cache Callback Type

```python
from typing import Callable, Awaitable

CacheCallback = Callable[[CacheTurnMetrics, CacheSessionMetrics], Awaitable[None]]
```

### Cache Control Plugin Protocol

```python
from abc import ABC, abstractmethod


class CacheControlPlugin(ABC):
    """
    Base protocol for provider-specific cache control plugins.

    Plugins declare model compatibility via compatible_models.
    The plugin registry activates the right cache plugin based
    on the current session's model provider.

    The key design principle: cache plugins are CONSUMERS of the
    GC Budget's content map. They read the locked/conditional/ephemeral
    categorization to make caching decisions — they don't invent
    their own content lifecycle model.
    """

    _callbacks: list[CacheCallback] = []

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """e.g., 'anthropic', 'zhipuai', 'vertex_ai'"""
        ...

    @property
    @abstractmethod
    def compatible_models(self) -> list[str]:
        """Glob patterns: ['claude-sonnet-4-*', 'claude-opus-4-*']"""
        ...

    @abstractmethod
    async def prepare_request(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | list[dict],
        content_map: ContentMap,
    ) -> dict:
        """
        Hook called BEFORE the LLM request is sent.

        Inject cache control annotations informed by the GC Budget's
        content map. The content_map tells us what's locked, conditional,
        and ephemeral — so we know where to place breakpoints.

        Returns a modified request dict with provider-specific cache
        annotations (or unmodified for implicit caching providers).
        """
        ...

    @abstractmethod
    async def extract_metrics(
        self,
        response: dict,
        content_map: ContentMap,
    ) -> CacheTurnMetrics:
        """
        Hook called AFTER the LLM response is received.

        Extracts provider-specific cache metrics from the response
        usage object and correlates them with GC categories.
        """
        ...

    @abstractmethod
    async def on_gc_collection(
        self,
        collected_blocks: list[ContentBlock],
        remaining_map: ContentMap,
    ) -> None:
        """
        Hook called BY the GC Budget plugin AFTER it collects blocks.

        The cache plugin uses this to track invalidations and
        adjust its strategy for the next request.
        """
        ...

    # ── Callback registration for clients ──

    def on_cache_metrics(self, callback: CacheCallback):
        """Register a callback for cache metrics updates."""
        self._callbacks.append(callback)

    async def _emit_metrics(self, turn: CacheTurnMetrics,
                             session: CacheSessionMetrics):
        """Notify all registered callbacks."""
        for cb in self._callbacks:
            await cb(turn, session)
```

---

## Provider Implementations

### Anthropic — Explicit Breakpoints Informed by GC Categories

```python
class AnthropicCachePlugin(CacheControlPlugin):

    provider_name = "anthropic"
    compatible_models = [
        "claude-sonnet-4-*", "claude-sonnet-4.5-*",
        "claude-opus-4-*", "claude-opus-4.5-*",
        "claude-haiku-4.5-*",
    ]

    def __init__(self, config: dict):
        self._ttl = config.get("cache_ttl", "5m")  # "5m" or "1h"
        self._session_metrics = CacheSessionMetrics()
        self._last_gc_invalidated = False

    async def prepare_request(self, messages, tools, system,
                                content_map: ContentMap) -> dict:
        cache_type = {"type": "ephemeral"}
        if self._ttl == "1h":
            cache_type["ttl"] = "1h"

        # ── Strategy: place breakpoints at GC category boundaries ──
        #
        # With 4 breakpoints available, allocate them based on
        # the content map's lifecycle annotations:
        #
        #   BP1: End of LOCKED system content (most stable)
        #   BP2: End of LOCKED tool definitions
        #   BP3: End of the CONDITIONAL zone (conversation prefix
        #         that's stable until conditions trigger)
        #   BP4: Penultimate message (for multi-turn cache hits)
        #
        # Never waste a breakpoint inside EPHEMERAL content —
        # the GC will eat it before the cache TTL expires.

        breakpoints_placed = 0
        max_breakpoints = 4

        # BP1: System prompt — always locked, always cache
        modified_system = self._inject_breakpoint(system, cache_type)
        breakpoints_placed += 1

        # BP2: Tool definitions — locked for session duration
        locked_tools = [
            b for b in content_map.blocks
            if b.tool_name and b.gc_category == GCCategory.LOCKED
        ]
        modified_tools = tools
        if locked_tools and breakpoints_placed < max_breakpoints:
            modified_tools = self._inject_tool_breakpoint(tools, cache_type)
            breakpoints_placed += 1

        # BP3 & BP4: Messages — use content_map to find the right spots
        modified_messages = self._inject_message_breakpoints_from_map(
            messages, content_map, cache_type,
            remaining_breakpoints=max_breakpoints - breakpoints_placed
        )

        return {
            "system": modified_system,
            "tools": modified_tools,
            "messages": modified_messages,
        }

    def _inject_message_breakpoints_from_map(
        self, messages, content_map, cache_type, remaining_breakpoints
    ):
        """
        Place message breakpoints at GC category transition points.

        The conversation looks like this in the content map:

          [LOCKED...] [CONDITIONAL...] [EPHEMERAL...] [new turn]
                    ^                ^
                    BP3              BP4 (if conditional zone is large)

        We want breakpoints at the boundary between categories,
        because that's where the prefix stability changes.
        """
        if remaining_breakpoints <= 0:
            return messages

        modified = [msg.copy() for msg in messages]

        # Find message-level blocks from the content map
        msg_blocks = [
            b for b in content_map.blocks
            if b.turn_number is not None
        ]

        # Strategy: work backwards from the end
        # - Skip the last message (that's the new user input, uncached)
        # - Find the last conditional-or-locked block
        # - Place a breakpoint there
        last_stable_idx = None
        penultimate_idx = len(messages) - 2 if len(messages) >= 2 else None

        for block in reversed(msg_blocks):
            if block.gc_category in (GCCategory.LOCKED, GCCategory.CONDITIONAL):
                if block.position < len(messages) - 1:
                    last_stable_idx = block.position
                    break

        # Place BP at the last stable message
        if last_stable_idx is not None and remaining_breakpoints > 0:
            modified[last_stable_idx] = self._add_cache_control(
                modified[last_stable_idx], cache_type
            )
            remaining_breakpoints -= 1

        # If we have another BP available and the penultimate message
        # is different from the stable boundary, place it there too
        if (penultimate_idx is not None
            and penultimate_idx != last_stable_idx
            and remaining_breakpoints > 0):
            modified[penultimate_idx] = self._add_cache_control(
                modified[penultimate_idx], cache_type
            )

        return modified

    async def extract_metrics(self, response, content_map) -> CacheTurnMetrics:
        usage = response.get("usage", {})

        turn = CacheTurnMetrics(
            cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            uncached_tokens=usage.get("input_tokens", 0),
            total_input_tokens=(
                usage.get("cache_creation_input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
                + usage.get("input_tokens", 0)
            ),
            locked_tokens_cached=content_map.locked_tokens,
            conditional_tokens_cached=content_map.conditional_tokens,
            ephemeral_tokens_skipped=content_map.ephemeral_tokens,
        )

        if self._last_gc_invalidated:
            self._session_metrics.gc_invalidations += 1
            self._last_gc_invalidated = False

        self._session_metrics.turns.append(turn)
        await self._emit_metrics(turn, self._session_metrics)
        return turn

    async def on_gc_collection(self, collected_blocks, remaining_map):
        """Track when GC breaks our cached prefix."""
        had_locked_or_conditional = any(
            b.gc_category in (GCCategory.LOCKED, GCCategory.CONDITIONAL)
            for b in collected_blocks
        )
        if had_locked_or_conditional:
            self._last_gc_invalidated = True
```

### ZhipuAI — Implicit Caching (Monitor + Advise)

```python
class ZhipuAICachePlugin(CacheControlPlugin):

    provider_name = "zhipuai"
    compatible_models = [
        "glm-5", "glm-4.7", "glm-4.6", "glm-4.5",
        "glm-4.7-*", "glm-4.6-*", "glm-4.5-*",
    ]

    def __init__(self, config: dict):
        self._session_metrics = CacheSessionMetrics()

    async def prepare_request(self, messages, tools, system,
                                content_map: ContentMap) -> dict:
        # ZhipuAI caching is implicit — no annotations to inject.
        # BUT we can use the content_map to verify that the message
        # ordering preserves prefix stability: locked content first,
        # then conditional, then ephemeral.
        #
        # This is a no-op if the GC Budget plugin already
        # maintains this ordering (which it should).
        return {
            "system": system,
            "tools": tools,
            "messages": messages,
        }

    async def extract_metrics(self, response, content_map) -> CacheTurnMetrics:
        usage = response.get("usage", {})
        details = usage.get("prompt_tokens_details", {})
        cached = details.get("cached_tokens", 0)
        total_prompt = usage.get("prompt_tokens", 0)

        turn = CacheTurnMetrics(
            cache_write_tokens=0,  # ZhipuAI doesn't separate write cost
            cache_read_tokens=cached,
            uncached_tokens=total_prompt - cached,
            total_input_tokens=total_prompt,
            locked_tokens_cached=min(cached, content_map.locked_tokens),
            conditional_tokens_cached=max(
                0, cached - content_map.locked_tokens
            ),
            ephemeral_tokens_skipped=content_map.ephemeral_tokens,
        )

        self._session_metrics.turns.append(turn)
        await self._emit_metrics(turn, self._session_metrics)
        return turn

    async def on_gc_collection(self, collected_blocks, remaining_map):
        """Track when GC breaks the implicit cache prefix."""
        had_stable = any(
            b.gc_category != GCCategory.EPHEMERAL
            for b in collected_blocks
        )
        if had_stable:
            self._session_metrics.gc_invalidations += 1
```

---

## GC Budget Plugin Integration

The existing GC Budget plugin requires a small addition: notifying the active cache plugin after collection. The collection logic itself remains unchanged — it still collects ephemeral first, then conditional (if conditions are met), and never touches locked content.

```python
class GCBudgetPlugin:

    async def collect(self, session: Session):
        """Budget-based collection respecting content categories."""
        content_map = self._build_content_map(session)

        if content_map.total_tokens < self._budget_threshold:
            return

        # Determine what to collect (existing logic):
        # 1. Ephemeral first (free to collect)
        # 2. Conditional next (if conditions are met)
        # 3. Locked — NEVER (by definition)
        collectible = self._select_collectible(content_map)

        if not collectible:
            return  # Nothing to collect, we're budget-constrained

        # Execute collection
        self._remove_blocks(session, collectible)

        # ── NEW: Notify cache plugin ──
        cache_plugin = self._registry.get_active_cache_plugin(session)
        if cache_plugin:
            remaining_map = self._build_content_map(session)
            await cache_plugin.on_gc_collection(
                collected_blocks=collectible,
                remaining_map=remaining_map,
            )

        # Emit event for clients
        self._emit_event("gc.budget.collected", {
            "blocks_collected": len(collectible),
            "tokens_freed": sum(b.token_count for b in collectible),
            "categories": {
                cat.value: sum(
                    b.token_count for b in collectible
                    if b.gc_category == cat
                )
                for cat in GCCategory
            },
            "remaining_budget": remaining_map.total_tokens,
        })
```

---

## Event Protocol Integration

For non-callback consumers (OTel, logging, Telegram client), the cache plugin emits standard jaato events:

```python
# New event types in cache/events.py

CACHE_TURN_METRICS = "cache.turn_metrics"           # Per-LLM-call
CACHE_SESSION_STATS = "cache.session_stats"         # Periodic / on-demand
CACHE_PREFIX_INVALIDATED = "cache.prefix_invalidated"  # When GC breaks cache
```

This means:

- The **TUI client** can register a callback via `on_cache_metrics()` for inline display.
- The **Telegram client** can subscribe to `CACHE_TURN_METRICS` events and render a compact status.
- The **OTel integration** can export them as `jaato.cache.hit_rate` and `jaato.cache.cost_savings_pct` gauge metrics.
- Any future client gets cache visibility for free via the event bus.

---

## Client Callback — TUI Example

```python
async def display_cache_metrics(
    turn: CacheTurnMetrics,
    session: CacheSessionMetrics
):
    """Rich TUI display showing cache-GC correlation."""
    # Hit rate bar
    bar_len = 20
    filled = int(turn.hit_rate * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    # Efficiency indicator (how much of cache is stable content)
    eff = turn.cache_efficiency
    eff_icon = "🟢" if eff > 0.7 else "🟡" if eff > 0.4 else "🔴"

    print(f"  Cache [{bar}] {turn.hit_rate:.0%} hit "
          f"| {eff_icon} {eff:.0%} efficiency")
    print(f"    🔒 {turn.locked_tokens_cached:,} locked "
          f"| 🔶 {turn.conditional_tokens_cached:,} conditional "
          f"| 💨 {turn.ephemeral_tokens_skipped:,} skipped")

    if session.gc_invalidations > 0:
        print(f"    ⚠️  {session.gc_invalidations} GC invalidation(s) "
              f"this session")

    print(f"  Session: {session.cumulative_hit_rate:.0%} avg hit "
          f"| {session.avg_cache_efficiency:.0%} avg efficiency")
```

### Registration During Client Init

```python
cache_plugin = registry.get_active_cache_plugin(session)
if cache_plugin:
    cache_plugin.on_cache_metrics(display_cache_metrics)
```

### Example TUI Output — Normal Operation

```
You: Run the CICS security assessment against the test region

  Cache [████████████████████] 97% hit | 🟢 89% efficiency
    🔒 3,900 locked | 🔶 1,240 conditional | 💨 450 skipped
  Session: 91% avg hit | 🟢 85% avg efficiency

Agent: Executing APT scan against CICSTS01...
```

### Example TUI Output — After GC Invalidation

```
You: Now compare with the production region results

  Cache [██████░░░░░░░░░░░░░░] 30% hit | 🟡 45% efficiency
    🔒 3,900 locked | 🔶 0 conditional | 💨 0 skipped
    ⚠️  1 GC invalidation(s) this session
  Session: 78% avg hit | 🟢 72% avg efficiency

Agent: Rebuilding context for production comparison...
```

---

## Prompt Structure for Optimal Caching

The cache plugin works best when the prompt follows this structure, which the GC Budget plugin's content ordering naturally produces:

```
┌─────────────────────────────────────┐
│  LOCKED (highest cache priority)    │  ← Cached, 90% cheaper
│  - System prompt                    │     BP1 (Anthropic)
│  - Tool definitions / schemas       │     BP2 (Anthropic)
│  - Core orchestration rules         │
│  - Few-shot examples                │
├─────────────────────────────────────┤
│  CONDITIONAL (cacheable, volatile)  │  ← Cached until GC triggers
│  - Important tool results           │     BP3 (Anthropic)
│  - Conversation history (stable)    │
│  - Reference documents              │
├─────────────────────────────────────┤
│  EPHEMERAL (skip caching)           │  ← Full price, but short-lived
│  - Intermediate reasoning           │     No breakpoint here
│  - Exploratory tool calls           │
│  - Verbose outputs                  │
├─────────────────────────────────────┤
│  NEW TURN (always uncached)         │  ← Current user input
│  - Latest user message              │     BP4 at penultimate msg
└─────────────────────────────────────┘
```

---

## Synergy with Terse Prompting

Terse prompting (60-70% token reduction for AI-to-AI communication) and KV caching are complementary but work on different axes:

- **Terse prompting** reduces the total number of tokens, saving cost linearly.
- **KV caching** reduces the compute cost per cached token by ~90%, saving on repeated prefixes.

The optimal strategy for jaato:

- Use **natural/verbose language** for the **LOCKED prefix** (system prompt, tool schemas). It's cheap when cached and benefits from clarity.
- Use **terse prompting** for the **EPHEMERAL and CONDITIONAL** parts (per-turn agent instructions, dynamic context). These are full-price tokens that benefit from compression.
- The combination can yield compounding savings beyond either technique alone.

---

## Anthropic-Specific Considerations

### The 1-Hour TTL

For jaato's longer agentic sessions (tool-heavy workflows that run 10+ minutes with multiple steps), the default 5-minute TTL can expire between steps. The 1-hour TTL costs more per cache write (2× vs 1.25× base) but avoids repeated cache misses during extended workflows.

Configuration recommendation: use `"5m"` for interactive sessions with frequent turns, `"1h"` for batch or long-running autonomous agent tasks.

### Breakpoint Allocation Strategy

With only 4 breakpoints available, allocation matters:

| Breakpoint | Target | Rationale |
|---|---|---|
| BP1 | System prompt | Most stable, highest reuse across all turns |
| BP2 | Last tool definition | Stable within session, changes only on plugin reconfiguration |
| BP3 | Last CONDITIONAL message | Marks the end of the "stable conversation prefix" as informed by GC Budget |
| BP4 | Penultimate user message | Captures the growing conversation for multi-turn cache hits |

Never place a breakpoint inside EPHEMERAL content — the GC will collect it before the cache TTL expires, wasting the write premium.

---

## ZhipuAI-Specific Considerations

### Implicit Caching — Zero Configuration

ZhipuAI's implicit caching means the cache plugin for ZhipuAI is primarily a **monitoring and advisory** role:

- **Monitoring**: Extract `cached_tokens` from the response and correlate with GC categories.
- **Advisory**: Ensure the GC Budget plugin maintains message ordering that preserves prefix stability. If the plugin detects that the GC is about to disrupt a highly-cached prefix, it can signal this through the metrics.

### API Compatibility

ZhipuAI uses OpenAI-compatible API format (`/v4/chat/completions`). If jaato already has an OpenAI-compatible client path, ZhipuAI caching works out of the box with no additional integration.

---

## Design Rationale: One Taxonomy, Two Consumers

The key architectural advantage of building cache plugins on top of the GC Budget's content categorization is that **there is no duplicate lifecycle model**. The GC Budget plugin already understands:

- Which tool results matter (locked).
- Which are useful until superseded (conditional).
- Which are intermediate noise (ephemeral).

The cache plugin reads that same classification to make its own decisions. Adding a new provider's cache plugin requires only implementing `prepare_request`, `extract_metrics`, and `on_gc_collection` — the content lifecycle intelligence comes for free from the GC Budget.

This also means that when tools declare their GC category (e.g., a tool result marked as `gc_locked` because it contains critical reference data), the cache plugin automatically treats it as high-priority cached content without any additional configuration.
