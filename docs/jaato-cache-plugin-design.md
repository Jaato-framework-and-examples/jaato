# jaato Cache Plugin System — Design Document

## Overview

This document captures the design for a provider-specific prompt caching system for the jaato agentic AI framework. The cache plugins leverage KV caching (Key-Value caching in the transformer attention mechanism) offered by LLM providers to reduce token costs by up to 90% and latency by up to 85%.

The central design principle is that **cache plugins are consumers of the InstructionBudget's GC policies**. The `InstructionBudget` already classifies every source entry using `GCPolicy` — `LOCKED`, `PRESERVABLE`, `PARTIAL`, `EPHEMERAL`, or `CONDITIONAL` — and the cache plugin reads this same taxonomy to decide where to place cache breakpoints, what's worth the write premium, and what to skip.

### Implementation Status

**Variant A (provider delegates to cache plugin) is implemented** as of 2026-02-22. The `CachePlugin` protocol, `AnthropicCachePlugin`, and `ZhipuAICachePlugin` are live. Cache metrics are threaded through the full pipeline to TUI clients. See the [Impact Analysis](design/cache-plugin-sequencing-impact-analysis.md) for detailed progress.

**Variant B (session orchestrates)** is not started — blocked on session-owned history Phase 2.

The sections below retain the original design rationale and motivation. Where behavior has been implemented, it reflects the actual implementation rather than the proposal.

### Original Motivation

Prompt caching for Anthropic was originally implemented **inside `AnthropicProvider`** (`jaato-server/shared/plugins/model_provider/anthropic/provider.py`). The cache logic was tightly coupled to the provider class, creating problems:

1. **Inheritance coupling**: `ZhipuAIProvider` inherits from `AnthropicProvider` and had to hardcode `self._enable_caching = False` to suppress the parent's Anthropic-specific breakpoint logic. ZhipuAI has its own implicit caching mechanism that's incompatible with Anthropic's explicit breakpoints.
2. **No budget awareness**: The original Anthropic implementation used `cache_exclude_recent_turns` (a simple turn count) to decide what to cache, rather than consulting the `InstructionBudget`'s GC policy assignments which already know which content is LOCKED, PRESERVABLE, or EPHEMERAL.
3. **Not extensible**: Adding caching for a new provider required either modifying the provider class or creating another inheritance workaround.

Cache logic has since been extracted into separate plugins that consume the `InstructionBudget`. The legacy code path remains in `AnthropicProvider` as a fallback when no cache plugin is attached.

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

The GC system and the cache system have a natural tension:

- **GC wants to mutate the prefix** to reclaim token budget (truncate, summarize, compact).
- **Caching wants prefix stability** to maximize cache hits and save cost.

The resolution is coordination: the cache plugin reads the `InstructionBudget`'s GC policy assignments to understand which content is stable (and worth caching) vs volatile (and not worth the write premium), and the GC system returns a `GCResult` with a `removal_list` of `GCRemovalItem` entries that describe what was collected and which message IDs were affected.

---

## GC Policies as Cache Priorities

The `InstructionBudget` assigns a `GCPolicy` to every `SourceEntry`. These policies map directly to cache priority:

| GC Policy | GC Behavior | Cache Implication |
|---|---|---|
| **LOCKED** | Never GC'd — essential for operation | **Highest cache priority** — stable prefix. System prompt, core tool schemas, original request. Always worth caching because guaranteed present on every request. |
| **PRESERVABLE** | Prefer to keep, GC only under extreme pressure | **High cache priority** — clarification Q&A, turn summaries. Stable unless context is critically full. Worth caching with the understanding it may be evicted under `pressure_percent`. |
| **PARTIAL** | Container with mixed children | **Varies by child** — the `PLUGIN` and `CONVERSATION` sources are typically `PARTIAL`, meaning their children have heterogeneous policies. Cache decisions should be made per-child. |
| **EPHEMERAL** | Can be fully GC'd | **Don't waste cache budget** — intermediate reasoning, discoverable tool schemas, enrichment content. The GC will likely collect these before the cache TTL expires. |
| **CONDITIONAL** | Delegated to plugin evaluation | **Cacheable but volatile** — content whose collectibility depends on runtime conditions evaluated by a plugin. |

### Actual GC Policy Assignments (from `instruction_budget.py`)

```python
# instruction_budget.py — DEFAULT_SOURCE_POLICIES
InstructionSource.SYSTEM:       GCPolicy.LOCKED     # All children LOCKED
InstructionSource.PLUGIN:       GCPolicy.PARTIAL     # Mixed: core=LOCKED, discoverable=EPHEMERAL
InstructionSource.ENRICHMENT:   GCPolicy.EPHEMERAL
InstructionSource.CONVERSATION: GCPolicy.PARTIAL     # Mixed: original_request=LOCKED, working=EPHEMERAL
InstructionSource.THINKING:     GCPolicy.EPHEMERAL   # Output tokens, not context

# Conversation turn types (DEFAULT_TURN_POLICIES)
ConversationTurnType.ORIGINAL_REQUEST:  GCPolicy.LOCKED
ConversationTurnType.CLARIFICATION_Q:   GCPolicy.PRESERVABLE
ConversationTurnType.CLARIFICATION_A:   GCPolicy.PRESERVABLE
ConversationTurnType.TURN_SUMMARY:      GCPolicy.PRESERVABLE
ConversationTurnType.WORKING:           GCPolicy.EPHEMERAL

# Plugin tool types (DEFAULT_TOOL_POLICIES)
PluginToolType.CORE:          GCPolicy.LOCKED
PluginToolType.DISCOVERABLE:  GCPolicy.EPHEMERAL
```

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

### Anthropic — Explicit Cache Control (Implemented)

- **Mechanism**: Manual placement of `cache_control` breakpoints on content blocks, or automatic mode with a single top-level `cache_control`.
- **Supported models**: Claude Sonnet 4/4.5, Claude Opus 4/4.5, Claude Haiku 4.5.
- **Max breakpoints**: 4 per request (`MAX_CACHE_BREAKPOINTS = 4`).
- **Cache processing order**: `tools → system → messages`.
- **Min tokens**: 1024 for Sonnet 3.5 (`CACHE_MIN_TOKENS_SONNET`), 2048 for others (`CACHE_MIN_TOKENS_OTHER`).
- **Default TTL**: 5 minutes, refreshed on each hit at no extra cost.
- **Extended TTL**: 1 hour (costs 2x base write price).
- **Cache write cost**: 1.25x base input token price.
- **Cache read cost**: 0.10x base input token price (90% savings).
- **Monitoring**: `cache_creation_input_tokens` and `cache_read_input_tokens` in the response usage object, surfaced through `TokenUsage.cache_creation_tokens` and `TokenUsage.cache_read_tokens`.

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

## Legacy Architecture: Provider-Internal Caching

> **Note:** This section describes the **legacy** cache architecture that existed before Variant A was implemented. The provider-internal cache logic is retained as a fallback path in `AnthropicProvider` when no `CachePlugin` is attached, but new deployments use the plugin-based architecture described in [Proposed Design](#proposed-design-cache-plugin-protocol) below.

### Architecture

Caching for Anthropic is currently implemented **inside `AnthropicProvider`** rather than as a separate plugin. The provider owns the cache breakpoint strategy because it has direct access to the system instruction, tools, and message history at request construction time.

> **Cross-reference:** The [Session-Owned History Impact Analysis](design/session-owned-history-impact-analysis.md) proposes moving history ownership from providers to the session. Under that proposal, the provider would receive messages as a parameter to a stateless `complete()` method rather than maintaining `self._history`. This strengthens the case for extracting cache logic into plugins, since the provider would no longer "own" the data that drives breakpoint decisions. See [Impact on Cache Plugin Integration](#impact-of-session-owned-history) below.

Key files:
- `jaato-server/shared/plugins/model_provider/anthropic/provider.py` — Cache configuration, breakpoint placement, threshold checking
- `jaato-server/shared/plugins/model_provider/anthropic/converters.py` — `messages_to_anthropic(cache_breakpoint_index=...)` applies history cache breakpoint
- `jaato-server/shared/plugins/model_provider/anthropic/env.py` — `resolve_enable_caching()` reads `JAATO_ANTHROPIC_ENABLE_CACHING`
- `jaato-sdk/jaato_sdk/plugins/model_provider/types.py` — `TokenUsage` dataclass with `cache_read_tokens` and `cache_creation_tokens` fields

### The Inheritance Problem

Two providers inherit from `AnthropicProvider` and must both suppress its cache logic:

**`ZhipuAIProvider`** inherits from `AnthropicProvider` because Z.AI exposes an Anthropic-compatible API — all message handling, streaming, and converters are reused. But the parent's cache code is Anthropic-specific (explicit breakpoints with `cache_control` annotations), while ZhipuAI uses fully automatic/implicit caching that requires no annotations at all.

**`OllamaProvider`** also inherits from `AnthropicProvider` because Ollama exposes an Anthropic-compatible API locally. Ollama doesn't support prompt caching at all.

Both suppress the parent's cache logic by hardcoding:

```python
# ZhipuAIProvider.__init__()
self._enable_caching = False  # Caching may not be supported by Zhipu AI's API

# OllamaProvider.__init__()
self._enable_caching = False  # Ollama doesn't support prompt caching
self._enable_thinking = False
```

This means ZhipuAI sessions get **no cache monitoring at all** — even though Z.AI does support implicit caching and reports `cached_tokens` in responses. The provider can't enable the parent's cache logic (wrong mechanism) and can't implement its own (the parent's `_build_api_kwargs()` already handles breakpoint injection).

Extracting cache logic into a plugin would let each provider get the right caching strategy without inheritance conflicts — and would let `AnthropicProvider` drop its cache code entirely, simplifying the class that two other providers inherit from.

### Configuration

Caching is configured through `ProviderConfig.extra`:

```python
# AnthropicProvider.initialize(config)
self._enable_caching = config.extra.get("enable_caching", resolve_enable_caching())
self._cache_ttl = config.extra.get("cache_ttl", "5m")
self._cache_history = config.extra.get("cache_history", True)
self._cache_exclude_recent_turns = config.extra.get(
    "cache_exclude_recent_turns", DEFAULT_CACHE_EXCLUDE_RECENT_TURNS  # 2
)
self._cache_min_tokens = config.extra.get("cache_min_tokens", True)
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_caching` | bool | `False` (or `JAATO_ANTHROPIC_ENABLE_CACHING`) | Enable prompt caching |
| `cache_ttl` | str | `"5m"` | Cache TTL (`"5m"` or `"1h"`) |
| `cache_history` | bool | `True` | Cache historical messages (breakpoint #3) |
| `cache_exclude_recent_turns` | int | `2` | Recent turns to exclude from history caching |
| `cache_min_tokens` | bool | `True` | Enforce minimum token threshold per content block |

### Breakpoint Strategy (`_build_api_kwargs`)

The provider places up to 3 of the 4 available breakpoints:

```
┌─────────────────────────────────────────┐
│  System instruction                     │ ← BP1: cache_control on text block
│  (combined with CLAUDE_CODE_IDENTITY    │
│   when using OAuth)                     │
├─────────────────────────────────────────┤
│  Tool definitions                       │ ← BP2: cache_control on last tool
│  (sorted by name for consistency)       │
├─────────────────────────────────────────┤
│  Historical messages                    │ ← BP3: cache_control on computed
│  (older, stable turns)                  │    breakpoint index
├─────────────────────────────────────────┤
│  Recent turns (excluded from caching)   │ ← Not cached (too volatile)
│  Latest user message                    │
└─────────────────────────────────────────┘
```

**BP1 — System instruction** (`_build_api_kwargs`):
```python
system_block = {"type": "text", "text": self._system_instruction}
if self._enable_caching and self._should_cache_content(self._system_instruction):
    system_block["cache_control"] = {"type": "ephemeral"}
kwargs["system"] = [system_block]
```

**BP2 — Tool definitions** (`_build_api_kwargs`):
```python
# Sort tools by name for consistent ordering (improves cache hits)
anthropic_tools = sorted(anthropic_tools, key=lambda t: t["name"])
if self._enable_caching and len(anthropic_tools) > 0:
    tools_json = json.dumps(anthropic_tools)
    if self._should_cache_content(tools_json):
        anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
```

**BP3 — History breakpoint** (`_compute_history_cache_breakpoint`):
```python
def _compute_history_cache_breakpoint(self) -> int:
    """Find the last assistant message before the 'exclude recent' window."""
    # Walk backward from end of history
    # Skip the last N user messages (cache_exclude_recent_turns)
    # Return index of the last MODEL message before that boundary
    # Returns -1 if caching disabled or history too short
```

The breakpoint index is passed to `messages_to_anthropic(cache_breakpoint_index=idx)`, which adds `cache_control` to the last content block of the message at that index.

### Threshold Checking

Content must meet minimum token thresholds to be worth caching:

```python
def _should_cache_content(self, content: str) -> bool:
    if not self._cache_min_tokens:
        return True
    min_tokens = self._get_cache_min_tokens()  # 1024 (Sonnet 3.5) or 2048 (others)
    estimated = self._estimate_tokens(content)  # len(content) // 4
    return estimated >= min_tokens
```

### Token Usage Reporting

Cache metrics flow through the standard `TokenUsage` dataclass:

```python
# jaato_sdk/plugins/model_provider/types.py
@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: Optional[int] = None       # Tokens read from cache (90% savings)
    cache_creation_tokens: Optional[int] = None    # Tokens written to cache (1.25x cost)
    reasoning_tokens: Optional[int] = None
    thinking_tokens: Optional[int] = None
```

Extracted from Anthropic responses in `converters.py`:

```python
# extract_usage_from_response()
cache_creation = getattr(resp_usage, "cache_creation_input_tokens", None)
cache_read = getattr(resp_usage, "cache_read_input_tokens", None)
if cache_creation is not None and cache_creation > 0:
    usage.cache_creation_tokens = cache_creation
if cache_read is not None and cache_read > 0:
    usage.cache_read_tokens = cache_read
```

Cache tokens are tracked via OpenTelemetry in `JaatoSession`:

```python
if response.usage.cache_read_tokens is not None:
    llm_telemetry.set_attribute("gen_ai.usage.cache_read_tokens", ...)
if response.usage.cache_creation_tokens is not None:
    llm_telemetry.set_attribute("gen_ai.usage.cache_creation_tokens", ...)
```

Cache metrics are also threaded through the full event pipeline to clients (implemented 2026-02-22):

```
ProviderResponse.usage.cache_read_tokens / .cache_creation_tokens
  → JaatoSession._accumulate_turn_tokens() → turn_data['cache_read'] / ['cache_creation']
  → on_agent_turn_completed callback
  → TurnCompletedEvent / TurnProgressEvent (jaato_sdk/events.py)
  → JaatoServer (server/core.py) → event forwarding
  → TUI rich_client.py:
      - Per-turn summary: "─── cache hit: 85%" after each turn
      - /history command: cache hit % per turn + session total
```

---

## GC System Architecture (Actual)

### Types and Interfaces

The GC system is built on these actual types (not the `GCCategory`/`ContentMap`/`ContentBlock` types from the previous revision, which do not exist):

**`InstructionBudget`** (`shared/instruction_budget.py`) — Tracks token usage by `InstructionSource`:

```python
class InstructionSource(Enum):
    SYSTEM = "system"           # System instructions (children: base, client, framework)
    PLUGIN = "plugin"           # Plugin instructions (children: per-tool)
    ENRICHMENT = "enrichment"   # Prompt enrichment pipeline additions
    CONVERSATION = "conversation"  # Message history (children: per-turn)
    THINKING = "thinking"       # Extended thinking output tokens

class GCPolicy(Enum):
    LOCKED = "locked"           # Never GC — essential for operation
    PRESERVABLE = "preservable" # Prefer to keep, GC only under extreme pressure
    PARTIAL = "partial"         # Container with mixed children
    EPHEMERAL = "ephemeral"     # Can be fully GC'd
    CONDITIONAL = "conditional" # Delegated to plugin evaluation
```

**`SourceEntry`** — A single instruction source with its token count, GC policy, and children:

```python
@dataclass
class SourceEntry:
    source: InstructionSource
    tokens: int
    gc_policy: GCPolicy
    label: Optional[str] = None
    children: Dict[str, "SourceEntry"] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[float] = field(default_factory=time.time)
    message_ids: List[str] = field(default_factory=list)

    def total_tokens(self) -> int: ...
    def gc_eligible_tokens(self) -> int: ...
    def locked_tokens(self) -> int: ...
    def preservable_tokens(self) -> int: ...
    def effective_gc_policy(self) -> GCPolicy: ...
```

**`InstructionBudget`** — Aggregates all sources:

```python
@dataclass
class InstructionBudget:
    session_id: str = ""
    agent_id: str = "main"
    agent_type: Optional[str] = None
    entries: Dict[InstructionSource, SourceEntry] = field(default_factory=dict)
    context_limit: int = 128_000

    def total_tokens(self) -> int: ...         # Excludes THINKING
    def gc_eligible_tokens(self) -> int: ...
    def locked_tokens(self) -> int: ...
    def preservable_tokens(self) -> int: ...
    def utilization_percent(self) -> float: ...
    def available_tokens(self) -> int: ...
    def gc_headroom_percent(self) -> float: ...
```

### GC Plugin Protocol

```python
# shared/plugins/gc/base.py
@runtime_checkable
class GCPlugin(Protocol):
    @property
    def name(self) -> str: ...

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None: ...
    def shutdown(self) -> None: ...

    def should_collect(
        self,
        context_usage: Dict[str, Any],
        config: GCConfig
    ) -> Tuple[bool, Optional[GCTriggerReason]]: ...

    def collect(
        self,
        history: List[Message],
        context_usage: Dict[str, Any],
        config: GCConfig,
        reason: GCTriggerReason,
        budget: Optional[InstructionBudget] = None,
    ) -> Tuple[List[Message], GCResult]: ...
```

Key observations:
- **Synchronous** — `collect()` is not `async`
- **Takes `List[Message]`** — not a `ContentMap`, not a session object
- **Returns `Tuple[List[Message], GCResult]`** — the new history and a result with `removal_list: List[GCRemovalItem]`
- **Budget is optional** — passed when available for policy-aware decisions

### GC Plugin Implementations

| Plugin | Location | Strategy | Budget-Aware |
|--------|----------|----------|:------------:|
| **`gc_budget`** | `shared/plugins/gc_budget/plugin.py` | **Policy-aware phased collection using InstructionBudget** | **Yes** |
| `gc_truncate` | `shared/plugins/gc_truncate/plugin.py` | Remove oldest turns, keep recent N | No |
| `gc_summarize` | `shared/plugins/gc_summarize/plugin.py` | Compress old turns into summary | No |
| `gc_hybrid` | `shared/plugins/gc_hybrid/plugin.py` | Generational: truncate ancient, summarize middle, preserve recent | No |

**`gc_budget` is the primary GC plugin** — it is the only one that uses the `InstructionBudget`'s `GCPolicy` assignments to make removal decisions, which makes it the most relevant to cache coordination. The simpler plugins (`gc_truncate`, `gc_summarize`, `gc_hybrid`) use turn-based heuristics without consulting GC policies.

All plugins follow the same protocol:
1. `should_collect()` checks `context_usage['percent_used']` against `config.threshold_percent`
2. `collect()` determines which content to remove and returns `(new_history, GCResult)` with `GCRemovalItem` entries for budget sync

### GCConfig

```python
@dataclass
class GCConfig:
    threshold_percent: float = 80.0   # Trigger GC when usage exceeds this
    target_percent: float = 60.0      # Target usage after GC
    pressure_percent: Optional[float] = 90.0  # When PRESERVABLE may be collected
    max_turns: Optional[int] = None
    auto_trigger: bool = True
    check_before_send: bool = True
    preserve_recent_turns: int = 5
    pinned_turn_indices: List[int] = field(default_factory=list)
    plugin_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def continuous_mode(self) -> bool:
        """True when pressure_percent is 0 or None (GC every turn)."""
        return not self.pressure_percent
```

### GCResult

```python
@dataclass
class GCResult:
    success: bool
    items_collected: int
    tokens_before: int
    tokens_after: int
    plugin_name: str
    trigger_reason: GCTriggerReason
    notification: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    removal_list: List[GCRemovalItem] = field(default_factory=list)

    @property
    def tokens_freed(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)

@dataclass
class GCRemovalItem:
    source: InstructionSource
    child_key: Optional[str] = None
    tokens_freed: int = 0
    reason: str = ""
    message_ids: List[str] = field(default_factory=list)
```

### BudgetGCPlugin — Policy-Aware Collection (`gc_budget`)

The `BudgetGCPlugin` (`shared/plugins/gc_budget/plugin.py`) is the primary GC plugin and the most relevant to cache coordination because it makes removal decisions based on the `InstructionBudget`'s `GCPolicy` assignments.

**Phased removal priority:**

```
Phase 1a: ENRICHMENT (bulk clear — regenerated each turn)
Phase 1b: Other EPHEMERAL entries (oldest first, by created_at)
Phase 2:  PARTIAL conversation turns (oldest first, respecting preserve_recent_turns)
Phase 3:  PRESERVABLE entries (only when usage > pressure_percent)
          Never touched in continuous mode.
Never:    LOCKED entries
```

Each phase stops as soon as `tokens_freed >= tokens_to_free` (i.e., when the target utilization is reached).

**Tool call pair expansion:** When removing a tool_result message, the plugin also removes the paired MODEL message containing the corresponding tool_call (and vice versa) to prevent orphaned `function_call`/`function_response` messages that providers would reject.

**CONDITIONAL policy evaluation:** For entries with `GCPolicy.CONDITIONAL`, the plugin delegates to the owning tool plugin's `evaluate_gc_policy()` method via the `PluginRegistry`, falling back to `EPHEMERAL` if no evaluator is available.

**Dual operating modes:**

| Mode | Trigger | PRESERVABLE | Configuration |
|------|---------|-------------|---------------|
| **Threshold** (default) | `usage >= threshold_percent` | Touched when `usage >= pressure_percent` | `pressure_percent > 0` (e.g., 90.0) |
| **Continuous** | `usage > target_percent` (every turn) | Never touched | `pressure_percent = 0` or `None` |

**Fallback:** When no `InstructionBudget` is available, `gc_budget` falls back to simple turn-based truncation (identical to `gc_truncate`).

**Cache implications:** Because `gc_budget` removes content in policy priority order (EPHEMERAL first, LOCKED never), the cached prefix (LOCKED system + LOCKED tools + PRESERVABLE history) remains stable through most GC operations. Only under extreme pressure (Phase 3) would the cached prefix be disrupted.

### Session-GC Integration

GC is integrated into `JaatoSession` at three points:

1. **Pre-send check** — `_maybe_collect_before_send()` runs before each `send_message()` if `gc_config.check_before_send` is `True`
2. **Proactive streaming check** — During streaming, a wrapped usage callback monitors `percent_used` and sets `_gc_threshold_crossed` if the threshold is exceeded
3. **Post-turn collection** — After the turn completes, if `_gc_threshold_crossed` was set, `_maybe_collect_after_turn()` triggers GC

```python
# JaatoSession
def set_gc_plugin(self, plugin: GCPlugin, config: Optional[GCConfig] = None) -> None:
    self._gc_plugin = plugin
    self._gc_config = config or GCConfig()
```

---

## Relationship Diagrams

### Legacy Architecture (cache logic inside provider, pre-Variant A)

```
┌───────────────────────────────────────────────────────┐
│                   InstructionBudget                    │
│                                                       │
│  ┌──────────────┬────────┬─────────────────────────┐  │
│  │ Source        │ Tokens │ GCPolicy                │  │
│  ├──────────────┼────────┼─────────────────────────┤  │
│  │ SYSTEM        │ 2,100  │ LOCKED 🔒              │  │
│  │ PLUGIN        │ 1,800  │ PARTIAL ◐              │  │
│  │ ENRICHMENT    │   150  │ EPHEMERAL ○            │  │
│  │ CONVERSATION  │ 3,200  │ PARTIAL ◐              │  │
│  │ THINKING      │ 1,200  │ EPHEMERAL ○            │  │
│  └──────────────┴────────┴─────────────────────────┘  │
└──────────┬──────────────────┬─────────────────────────┘
           │                   │ reads policies
           │ NOT used          ▼
           │            ┌──────────────────────────────┐
           │            │ BudgetGCPlugin (gc_budget)   │
           │            └──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│ AnthropicProvider                                │
│   cache logic INSIDE provider:                   │
│   _build_api_kwargs() ← breakpoint placement     │
│   _compute_history_cache_breakpoint()            │
│   _should_cache_content()                        │
│                                                  │
│   ❌ Does NOT read InstructionBudget policies    │
│   ❌ Uses simple cache_exclude_recent_turns      │
│                                                  │
├──────────────────────────────────────────────────┤
│ ZhipuAIProvider (inherits AnthropicProvider)     │
│   ❌ _enable_caching = False  (hardcoded)        │
│   ❌ No cache monitoring at all                  │
│   ❌ Can't use Z.AI's implicit caching           │
├──────────────────────────────────────────────────┤
│ OllamaProvider (inherits AnthropicProvider)      │
│   ❌ _enable_caching = False  (hardcoded)        │
│   (correct — Ollama has no caching)              │
└─────────────────────────────────────────────────┘
```

### Current Architecture — Variant A (cache logic in plugins, implemented)

```
┌───────────────────────────────────────────────────────┐
│                   InstructionBudget                    │
│                                                       │
│  ┌──────────────┬────────┬─────────────────────────┐  │
│  │ Source        │ Tokens │ GCPolicy                │  │
│  ├──────────────┼────────┼─────────────────────────┤  │
│  │ SYSTEM        │ 2,100  │ LOCKED 🔒              │  │
│  │ PLUGIN        │ 1,800  │ PARTIAL ◐              │  │
│  │ ENRICHMENT    │   150  │ EPHEMERAL ○            │  │
│  │ CONVERSATION  │ 3,200  │ PARTIAL ◐              │  │
│  │ THINKING      │ 1,200  │ EPHEMERAL ○            │  │
│  └──────────────┴────────┴─────────────────────────┘  │
└────────┬──────────────┬──────────────┬────────────────┘
         │ reads         │ reads         │ reads
         ▼               ▼               ▼
┌─────────────────┐ ┌──────────────┐ ┌───────────────────┐
│ BudgetGCPlugin  │ │ Anthropic    │ │ ZhipuAI           │
│ (gc_budget)     │ │ CachePlugin  │ │ CachePlugin       │
│                 │ │              │ │                   │
│ phased removal  │ │ explicit BPs │ │ implicit caching  │
│ 1a→1b→2→3      │ │ budget-aware │ │ monitoring only   │
│                 │ │ breakpoint   │ │ cache hit rate    │
│ returns         │ │ placement    │ │ tracking          │
│ GCResult ───────┼→│ on_gc_result │ │ on_gc_result      │
│                 │ │              │ │                   │
└─────────────────┘ └──────┬───────┘ └─────────┬─────────┘
                           │                    │
             ┌─────────────┴─────────────┐      │
             │ AnthropicProvider          │      │
             │ (no cache logic — cleaned) │      │
             ├───────────────────────────┤      │
             │ ZhipuAIProvider (inherits) │◄─────┘
             │ (no _enable_caching hack)  │
             ├───────────────────────────┤
             │ OllamaProvider (inherits)  │  (no cache plugin
             │ (no suppression needed)    │   needed — correct)
             └───────────────────────────┘

Standalone providers (no caching, no inheritance conflict):
  GoogleGenAIProvider, AntigravityProvider,
  ClaudeCLIProvider, GitHubModelsProvider
  → Cache plugin can be added later without modifying the provider
```

---

## Cache Plugin Protocol (Implemented)

> **Status:** Implemented as Variant A (2026-02-22). The protocol, both plugin implementations, session wiring, and metrics pipeline are live.

Cache logic has been **extracted from provider classes** into separate plugins that:

1. Are **provider-specific** — each plugin knows how its provider's caching works (explicit breakpoints vs implicit)
2. Are **budget-aware** — plugins consume `InstructionBudget` GC policies to make smarter decisions
3. Are **GC-coordinated** — plugins receive `GCResult` notifications to track invalidations
4. **Decouple caching from provider inheritance** — `ZhipuAIProvider` can inherit `AnthropicProvider` for API compatibility without inheriting the wrong cache strategy

### Cache as a Fifth Plugin Kind

Cache plugins introduce a new `PLUGIN_KIND = "cache"`, following the same discovery pattern as the existing four kinds:

| Kind | Protocol | Entry Point Group | Example |
|------|----------|-------------------|---------|
| `"tool"` | `ToolPlugin` | `jaato.plugins` | `cli`, `file_edit`, `todo` |
| `"gc"` | `GCPlugin` | `jaato.gc_plugins` | `gc_truncate`, `gc_budget` |
| `"model_provider"` | `ModelProviderPlugin` | (import-based) | `anthropic`, `google_genai` |
| `"session"` | (session protocol) | (import-based) | session persistence |
| **`"cache"`** | **`CachePlugin`** | **`jaato.cache_plugins`** | **`cache_anthropic`, `cache_zhipuai`** |

Each cache plugin follows the standard plugin scaffolding:

```python
# shared/plugins/cache_anthropic/__init__.py
PLUGIN_KIND = "cache"
from .plugin import AnthropicCachePlugin, create_plugin
__all__ = ["PLUGIN_KIND", "AnthropicCachePlugin", "create_plugin"]

# shared/plugins/cache_anthropic/plugin.py
class AnthropicCachePlugin:
    ...

def create_plugin() -> AnthropicCachePlugin:
    return AnthropicCachePlugin()
```

```toml
# pyproject.toml
[project.entry-points."jaato.cache_plugins"]
cache_anthropic = "shared.plugins.cache_anthropic:create_plugin"
cache_zhipuai = "shared.plugins.cache_zhipuai:create_plugin"
```

**Selection:** The session (or runtime) selects the cache plugin by matching the active provider's name against `CachePlugin.provider_name`. If no cache plugin matches the current provider, no caching is applied — this is the correct behavior for providers like Ollama, ClaudeCLI, and Antigravity that don't support prompt caching.

### Provider Coverage

| Provider | Inheritance | Cache Plugin | Behavior |
|----------|-------------|--------------|----------|
| **AnthropicProvider** | Base class | `cache_anthropic` | Explicit breakpoints, budget-aware BP3 |
| **ZhipuAIProvider** | inherits Anthropic | `cache_zhipuai` | Monitoring only (implicit caching) |
| **OllamaProvider** | inherits Anthropic | None | No caching (correct) |
| **GoogleGenAIProvider** | Standalone | `cache_google_genai` (future) | Google context caching (when added) |
| **AntigravityProvider** | Standalone | None (for now) | No caching API exposed |
| **ClaudeCLIProvider** | Standalone | None | CLI handles caching internally |
| **GitHubModelsProvider** | Standalone | None (for now) | Depends on underlying model provider |

After extraction, `AnthropicProvider` drops its `_enable_caching`, `_cache_history`, `_cache_exclude_recent_turns`, and `_cache_min_tokens` fields, plus `_build_api_kwargs()`'s breakpoint injection, `_compute_history_cache_breakpoint()`, and `_should_cache_content()`. The child providers (`ZhipuAIProvider`, `OllamaProvider`) no longer need to hardcode `self._enable_caching = False`.

### Protocol Definition

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class CachePlugin(Protocol):
    """Protocol for provider-specific cache control plugins.

    Cache plugins are CONSUMERS of the InstructionBudget's GC policies.
    They read the LOCKED/PRESERVABLE/EPHEMERAL assignments to make
    caching decisions — they don't invent their own content lifecycle model.

    Note: Methods are synchronous, matching the GCPlugin protocol pattern.
    """

    @property
    def name(self) -> str:
        """Plugin identifier (e.g., 'cache_anthropic', 'cache_zhipuai')."""
        ...

    @property
    def provider_name(self) -> str:
        """Which provider this plugin serves (e.g., 'anthropic', 'zhipuai')."""
        ...

    @property
    def compatible_models(self) -> list[str]:
        """Glob patterns for compatible models."""
        ...

    def initialize(self, config: dict | None = None) -> None:
        """Initialize with provider-specific configuration."""
        ...

    def shutdown(self) -> None:
        """Clean up resources."""
        ...

    def prepare_request(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | list[dict],
        budget: InstructionBudget | None = None,
    ) -> dict:
        """Hook called BEFORE the LLM request is sent.

        Inject cache control annotations informed by the InstructionBudget's
        GC policies. The budget tells us what's LOCKED, PRESERVABLE, and
        EPHEMERAL — so we know where to place breakpoints.

        Returns a modified request dict with provider-specific cache
        annotations (or unmodified for implicit caching providers).
        """
        ...

    def extract_cache_usage(self, usage: "TokenUsage") -> None:
        """Hook called AFTER the LLM response is received.

        Extracts cache-specific metrics from the TokenUsage dataclass.
        The TokenUsage already contains cache_read_tokens and
        cache_creation_tokens populated by the provider's converter.
        """
        ...

    def on_gc_result(self, result: GCResult) -> None:
        """Hook called AFTER GC collects content.

        The cache plugin uses this to track invalidations and
        adjust its strategy for the next request.
        """
        ...
```

### Legacy vs Plugin Architecture

| Aspect | Legacy (provider-internal) | Plugin (implemented) |
|---|---|---|
| **Location** | Inside `AnthropicProvider._build_api_kwargs()` | Separate `cache_anthropic/plugin.py` |
| **Budget awareness** | No — uses `cache_exclude_recent_turns` (turn count) | Yes — reads `GCPolicy` from `InstructionBudget` |
| **GC coordination** | None — doesn't know when GC runs | `on_gc_result()` receives `GCResult` after each collection |
| **ZhipuAI support** | Disabled (`_enable_caching = False`) | Separate `cache_zhipuai/plugin.py` for monitoring |
| **New provider** | Modify provider class or override methods | Add a new plugin, no provider changes |
| **Metrics** | OTel attributes only | OTel + session-level aggregates + `on_gc_result` tracking + TUI per-turn display |

### File Structure (Implemented)

```
plugins/
├── model_provider/
│   ├── anthropic/
│   │   ├── provider.py          # AnthropicProvider (delegates to CachePlugin when attached, legacy fallback otherwise)
│   │   ├── converters.py        # messages_to_anthropic(cache_breakpoint_index=...)
│   │   └── env.py               # resolve_enable_caching()
│   └── ...
│
├── gc/                           # GC infrastructure
│   ├── __init__.py               # GCPlugin protocol, discover_gc_plugins(), etc.
│   ├── base.py                   # GCPlugin, GCConfig, GCResult, GCRemovalItem, GCTriggerReason
│   └── utils.py                  # Turn, split_into_turns(), ensure_tool_call_integrity()
│
├── gc_budget/                    # PRIMARY: Policy-aware GC using InstructionBudget
│   ├── __init__.py
│   ├── plugin.py                 # BudgetGCPlugin (phased removal, pair expansion)
│   └── tests/test_budget_gc.py   # Comprehensive test suite
│
├── gc_truncate/plugin.py         # TruncateGCPlugin (simple turn truncation)
├── gc_summarize/plugin.py        # SummarizeGCPlugin (summarization-based)
├── gc_hybrid/plugin.py           # HybridGCPlugin (generational: truncate + summarize)
│
├── cache_anthropic/              # Extracted from AnthropicProvider (IMPLEMENTED)
│   ├── __init__.py               # PLUGIN_KIND = "cache", create_plugin()
│   ├── plugin.py                 # AnthropicCachePlugin (breakpoints + budget) — 473 LOC
│   └── tests/test_plugin.py      # 31 tests
│
├── cache_zhipuai/                # ZhipuAI cache monitoring plugin (IMPLEMENTED)
│   ├── __init__.py               # PLUGIN_KIND = "cache", create_plugin()
│   ├── plugin.py                 # ZhipuAICachePlugin (monitoring only) — 166 LOC
│   └── tests/test_plugin.py      # 21 tests
│
└── cache_vertex/                 # Future: Vertex AI cache plugin
    └── ...
```

### Anthropic Cache Plugin — Budget-Aware Breakpoints

Extracted from the current `AnthropicProvider._build_api_kwargs()` logic, but enhanced with `InstructionBudget` awareness:

```python
class AnthropicCachePlugin:
    """Explicit breakpoint caching for Anthropic, informed by GC policies.

    Extracted from AnthropicProvider so that:
    1. ZhipuAIProvider doesn't inherit Anthropic-specific cache logic
    2. Breakpoint placement can use InstructionBudget policies instead
       of the simpler cache_exclude_recent_turns heuristic
    3. GC invalidations are tracked via on_gc_result()
    """

    @property
    def name(self) -> str:
        return "cache_anthropic"

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def compatible_models(self) -> list[str]:
        return ["claude-sonnet-4*", "claude-opus-4*", "claude-haiku-4.5*"]

    def prepare_request(self, messages, tools, system, budget=None) -> dict:
        cache_type = {"type": "ephemeral"}

        # BP1: System instruction (LOCKED in budget — always stable)
        modified_system = self._inject_system_breakpoint(system, cache_type)

        # BP2: Last tool definition (core tools are LOCKED in budget)
        # Sort by name for consistent ordering across sessions
        modified_tools = self._inject_tool_breakpoint(tools, cache_type)

        # BP3: History — use budget policies instead of turn counting
        if budget:
            # Place breakpoint at the boundary between PRESERVABLE
            # and EPHEMERAL conversation turns. This is more precise
            # than cache_exclude_recent_turns because the budget knows
            # which turns are stable (clarification Q&A, summaries)
            # vs volatile (working output).
            bp_index = self._find_policy_boundary(budget)
        else:
            # Fallback: use the existing turn-counting heuristic
            bp_index = self._compute_history_breakpoint_by_recency()

        modified_messages = self._inject_history_breakpoint(
            messages, bp_index, cache_type
        )

        return {
            "system": modified_system,
            "tools": modified_tools,
            "messages": modified_messages,
        }

    def _find_policy_boundary(self, budget: InstructionBudget) -> int:
        """Find the last LOCKED or PRESERVABLE conversation turn.

        This is the optimal cache breakpoint: everything before it
        is stable (won't be GC'd except under extreme pressure),
        everything after it is EPHEMERAL (may be GC'd any time).
        """
        conv = budget.get_entry(InstructionSource.CONVERSATION)
        if not conv or not conv.children:
            return -1

        last_stable = -1
        for key, child in conv.children.items():
            policy = child.effective_gc_policy()
            if policy in (GCPolicy.LOCKED, GCPolicy.PRESERVABLE):
                # Track this as a candidate breakpoint
                last_stable = child  # resolve to message index
        return last_stable

    def on_gc_result(self, result: GCResult) -> None:
        # Only PRESERVABLE removal risks disrupting the cached prefix
        if result.details.get("preservable_removed", 0) > 0:
            self._prefix_invalidated = True
```

### ZhipuAI Cache Plugin — Implicit Caching Monitor

ZhipuAI's caching is fully automatic. The plugin is monitoring-only — it doesn't inject annotations, but it tracks cache hit rates and GC invalidations:

```python
class ZhipuAICachePlugin:
    """Monitoring-only cache plugin for ZhipuAI's implicit caching."""

    @property
    def name(self) -> str:
        return "cache_zhipuai"

    @property
    def provider_name(self) -> str:
        return "zhipuai"

    @property
    def compatible_models(self) -> list[str]:
        return ["glm-5", "glm-4.7*", "glm-4.6*", "glm-4.5*"]

    def prepare_request(self, messages, tools, system, budget=None) -> dict:
        # ZhipuAI caching is implicit — no annotations needed.
        return {"system": system, "tools": tools, "messages": messages}

    def extract_cache_usage(self, usage: TokenUsage) -> None:
        # TokenUsage.cache_read_tokens is already populated
        # by the ZhipuAI provider's response parser.
        # This hook can track session-level aggregates.
        pass

    def on_gc_result(self, result: GCResult) -> None:
        # Track when GC potentially breaks the implicit cache prefix.
        if result.tokens_freed > 0:
            self._gc_invalidation_count += 1
```

---

## Integration: How Provider, Cache Plugin, and GC Plugin Interact

### The Wiring Problem

The current architecture creates a challenge for cache plugin integration:

1. **The provider owns the API call.** `_build_api_kwargs()` is called inside `AnthropicProvider.send_message()` on every request. The session calls `provider.send_message(text)` and never sees the kwargs.
2. **The provider maintains its own history.** `self._history`, `self._system_instruction`, and `self._tools` live inside the provider. The session only interacts with history via `provider.get_history()` (read) and `_create_provider_session(history)` (replace).
3. **GC talks to the session, not the provider.** After GC, the session calls `self.reset_session(new_history)` which calls `provider.create_session(system_instruction, tools, history=new_history)` — a full replacement.

> **Cross-reference:** These three constraints are specific to the current provider-owned history architecture. The [Session-Owned History proposal](design/session-owned-history-impact-analysis.md) would change all three: (1) the session would call `provider.complete(messages, system, tools, ...)` instead of `provider.send_message(text)`, giving the session control over all inputs; (2) history would live in `session._messages`, not `provider._history`; (3) GC would slice `session._messages` directly without going through `provider.create_session()`. This opens a second integration path — see [Impact on Cache Plugin Integration](#impact-of-session-owned-history) and the [Session-Orchestration variant](#variant-b-session-orchestrates-future) below.

```
Current data flow (per turn):

    JaatoSession                        AnthropicProvider
    ────────────                        ──────────────────
    send_message(prompt, callback)
      │
      ├─ _maybe_collect_before_send()   ← GC check (may reset_session)
      │
      ├─ _run_chat_loop(prompt)
      │    │
      │    └──────────────────────────→ send_message(text)
      │                                   │
      │                                   ├─ _history.append(user_msg)
      │                                   ├─ _compute_history_cache_breakpoint()
      │                                   ├─ messages_to_anthropic(cache_breakpoint_index=bp)
      │                                   ├─ _build_api_kwargs()  ← cache logic HERE
      │                                   ├─ client.messages.create(...)
      │                                   ├─ _add_response_to_history()
      │                                   │
      │    ◄──────────────────────────── return ProviderResponse
      │                                   │
      │    (ProviderResponse.usage has     │
      │     cache_read_tokens,             │
      │     cache_creation_tokens)         │
      │
      ├─ _maybe_collect_after_turn()    ← GC check (may reset_session)
      │
      └─ return response
```

### Integration: Provider Delegates to Cache Plugin

There are two integration architectures for cache plugins. **Variant A is implemented.** Variant B becomes viable when session-owned history Phase 2 lands.

#### Variant A: Provider Delegates Internally (Implemented)

Under the current provider-owned history model, the provider has an **optional `CachePlugin` slot**. The provider still owns the API call, but delegates cache annotation decisions to the plugin when one is attached. When no plugin is attached, the legacy provider-internal cache logic is used as a fallback.

```
Variant A data flow (provider-owned history):

    JaatoSession                 AnthropicProvider              CachePlugin
    ────────────                 ──────────────────             ────────────
    configure()
      │
      ├─ create provider
      ├─ discover cache plugin (by provider_name match)
      ├─ provider.set_cache_plugin(plugin)  ──────────→ self._cache_plugin = plugin
      │
    send_message(prompt)
      │
      ├─ _run_chat_loop()
      │    │
      │    └─────────────────→ send_message(text)
      │                          │
      │                          ├─ _build_api_kwargs()
      │                          │    │
      │                          │    ├─ if self._cache_plugin:
      │                          │    │    plugin.prepare_request(  ◄──── budget-aware
      │                          │    │      system, tools, messages,     breakpoint
      │                          │    │      budget=...)                  placement
      │                          │    │    → modified system/tools/messages
      │                          │    │
      │                          │    └─ else: current cache logic (fallback)
      │                          │
      │                          ├─ client.messages.create(...)
      │                          │
      │    ◄─────────────────── return response (with TokenUsage)
      │
      ├─ extract cache metrics from response.usage
      │    │
      │    └─ cache_plugin.extract_cache_usage(usage) ──────→ track hit rate
      │
      ├─ _maybe_collect_after_turn()
      │    │
      │    └─ gc_plugin.collect(history, ..., budget)
      │         │
      │         └─ returns GCResult
      │              │
      │              └─ cache_plugin.on_gc_result(result) ──→ track invalidation
      │
      └─ return response
```

#### Variant B: Session Orchestrates (Not Started — Blocked on Session-Owned History) {#variant-b-session-orchestrates-future}

When the [Session-Owned History proposal](design/session-owned-history-impact-analysis.md) is implemented (specifically Phase 2+, where providers expose a stateless `complete()` method), a second integration path becomes viable: the **session orchestrates cache annotations** before calling the provider. The A→B transition cost is ~21 lines of throwaway code (~3% of Variant A). See the [Impact Analysis](design/cache-plugin-sequencing-impact-analysis.md) for the detailed breakdown.

```
Variant B data flow (session-owned history, Phase 2+):

    JaatoSession                 AnthropicProvider              CachePlugin
    ────────────                 ──────────────────             ────────────
    configure()
      │
      ├─ create provider
      ├─ discover cache plugin (by provider_name match)
      ├─ cache_plugin.initialize(config)
      ├─ self._cache_plugin = cache_plugin  ← session holds plugin directly
      │
    send_message(prompt)
      │
      ├─ _maybe_collect_before_send()   ← GC slices self._messages directly
      │
      ├─ self._messages.append(user_msg)
      │
      ├─ _run_chat_loop()
      │    │
      │    └─────────────────→ complete(self._messages, system, tools, ...)
      │                          │
      │                          ├─ convert messages to Anthropic format
      │                          ├─ cache_plugin.prepare_request(  ◄──── provider calls
      │                          │    api_messages, api_tools,           plugin with
      │                          │    api_system, budget)                API-format data
      │                          ├─ client.messages.create(...)
      │                          │
      │    ◄─────────────────── return ProviderResponse
      │
      ├─ self._messages.append(response_msg)
      │
      ├─ cache_plugin.extract_cache_usage(response.usage)
      │
      ├─ _maybe_collect_after_turn()
      │    │
      │    ├─ gc_plugin.collect(self._messages, ..., budget)
      │    │    → slices self._messages in place (no reset_session)
      │    │
      │    └─ cache_plugin.on_gc_result(result)
      │
      └─ return response
```

**Key differences from Variant A:**

| Aspect | Variant A (provider delegates) | Variant B (session orchestrates) |
|--------|-------------------------------|----------------------------------|
| **History ownership** | Provider (`self._history`) | Session (`self._messages`) |
| **Who calls `prepare_request`** | Provider, inside `_build_api_kwargs()` | Provider, inside `complete()` — data comes from parameters, not internal state |
| **Provider protocol change** | Add `set_cache_plugin()` method | No protocol change needed — plugin is wired to session |
| **GC flow** | `reset_session()` → `provider.create_session(history)` | `self._messages.replace(new)` — no provider call |
| **Provider statefulness** | Stateful (owns history + cache plugin) | Stateless w.r.t. history; may still hold cache plugin reference |

**Why the provider should still call `prepare_request()` in both variants:** Cache annotations are format-specific — Anthropic's `cache_control` dicts are injected into Anthropic-format message/tool/system structures, not generic `Message` objects. The provider converts to API format first, then calls the cache plugin. Having the session inject annotations before conversion would couple it to a specific provider's wire format, violating the provider abstraction.

**Status:** Variant A is implemented. When session-owned history is adopted later, the cache plugin protocol (`CachePlugin.prepare_request()`) requires **no changes** — only the caller context shifts from provider-internal state to provider-received parameters. The A→B migration is mechanical (~2 hours).

### Provider Protocol Extension

> **Note:** This extension is needed for [Variant A](#variant-a-provider-delegates-internally-current-architecture) (current architecture). Under [Variant B](#variant-b-session-orchestrates-future) (session-owned history), the provider protocol may not need this method — the session holds the cache plugin directly and the provider receives pre-annotated data or calls the plugin using parameters it received. However, even under Variant B, having the provider call `prepare_request()` internally is cleaner because annotations are format-specific (see [Variant B rationale](#variant-b-session-orchestrates-future)).

Add an optional method to `ModelProviderPlugin`:

```python
# shared/plugins/model_provider/base.py

class ModelProviderPlugin(Protocol):
    # ... existing methods ...

    def set_cache_plugin(self, plugin: "CachePlugin") -> None:
        """Attach a cache control plugin.

        When set, the provider delegates cache annotation decisions
        (breakpoint placement, threshold checks) to this plugin
        instead of using provider-internal logic.

        This decouples cache strategy from provider implementation,
        allowing ZhipuAIProvider and OllamaProvider to inherit from
        AnthropicProvider without inheriting the wrong cache logic.
        """
        ...
```

Providers that don't support caching ignore this (duck typing — the method is optional on the protocol). `AnthropicProvider` implements it:

```python
# AnthropicProvider
def set_cache_plugin(self, plugin: CachePlugin) -> None:
    self._cache_plugin = plugin

def _build_api_kwargs(self, response_schema=None) -> dict:
    kwargs = {}

    if self._cache_plugin:
        # Delegate to plugin — it gets the budget and decides breakpoints
        cache_result = self._cache_plugin.prepare_request(
            system=self._system_instruction,
            tools=self._tools,
            messages=self._history,
            budget=self._instruction_budget,  # if available
        )
        kwargs["system"] = cache_result["system"]
        kwargs["tools"] = cache_result["tools"]
        # messages handled via cache_breakpoint_index from plugin
    else:
        # Existing logic (backwards compatible, no plugin attached)
        # ... current _build_api_kwargs code ...

    return kwargs
```

Under session-owned history (Variant B), the same delegation pattern works but with parameters instead of internal state:

```python
# AnthropicProvider.complete() — Variant B (session-owned history)
def complete(self, messages, system_instruction, tools, ...) -> ProviderResponse:
    # Convert to Anthropic format
    api_messages = messages_to_anthropic(messages)
    api_tools = tools_to_anthropic(tools)
    api_system = [{"type": "text", "text": system_instruction}]

    if self._cache_plugin:
        # Same delegation, but data comes from parameters, not self._history
        cache_result = self._cache_plugin.prepare_request(
            system=api_system,
            tools=api_tools,
            messages=api_messages,
            budget=...,  # from plugin's own state (set_budget)
        )
        api_system = cache_result["system"]
        api_tools = cache_result["tools"]
        api_messages = cache_result["messages"]

    response = self._client.messages.create(
        messages=api_messages, system=api_system, tools=api_tools, ...
    )
    return self._convert_response(response)
```

### Session Wiring (Implemented)

`JaatoSession.configure()` discovers and attaches the cache plugin:

```python
# JaatoSession.configure() — Variant A (implemented)
def _wire_cache_plugin(self) -> None:
    """Discover and attach the cache plugin matching the active provider."""
    if not self._provider:
        return

    provider_name = self._provider.name
    cache_plugin = self._runtime.get_cache_plugin(provider_name)

    if cache_plugin:
        cache_plugin.initialize(self._provider_config.extra)

        # Provider delegates cache decisions to plugin
        if hasattr(self._provider, 'set_cache_plugin'):
            self._provider.set_cache_plugin(cache_plugin)

        self._cache_plugin = cache_plugin
```

Under session-owned history (Variant B), the wiring simplifies — no provider involvement:

```python
# JaatoSession.configure() — Variant B (session-owned history)
def _wire_cache_plugin(self) -> None:
    """Discover and attach the cache plugin matching the active provider."""
    if not self._provider:
        return

    cache_plugin = self._runtime.get_cache_plugin(self._provider.name)
    if cache_plugin:
        cache_plugin.initialize(self._provider_config.extra)
        self._cache_plugin = cache_plugin
        # No provider.set_cache_plugin() needed — session orchestrates
```

After GC runs, the session notifies the cache plugin. This pattern is **identical in both variants** — cache invalidation tracking is always session-level:

```python
# JaatoSession._maybe_collect_after_turn()
new_history, result = self._gc_plugin.collect(...)
if result.success:
    # Variant A: self.reset_session(new_history)
    #   → provider.create_session(system, tools, history=new_history)
    # Variant B: self._history.replace(new_history)
    #   → no provider call needed
    self._apply_gc_removal_list(result)

    # Notify cache plugin about what GC removed (both variants)
    if self._cache_plugin:
        self._cache_plugin.on_gc_result(result)
```

### Session-Level vs. Provider-Level Cache Orchestration

An alternative to provider delegation is to have the session orchestrate cache annotations directly. Under the **current** provider-owned history architecture, this is impractical:

1. **The session calls `provider.send_message(text)`** — it doesn't see or control the API kwargs. The provider builds and sends the request internally.
2. **The provider owns the Anthropic SDK client** — only the provider can add `cache_control` annotations to the request.
3. **History is inside the provider** — the cache plugin needs access to the current history to compute breakpoints, and only the provider has it at call time.

Under the current architecture, the delegation pattern (provider calls plugin) is the right approach.

> **Cross-reference:** Under the [Session-Owned History proposal](design/session-owned-history-impact-analysis.md), arguments 1 and 3 are **invalidated**: the session would call `provider.complete(messages, ...)` (controlling all inputs) and history would live in `session._messages` (accessible to the cache plugin at any time). Argument 2 partially survives — the Anthropic SDK client remains on the provider — but the real constraint is **format-specificity**, not SDK ownership: `cache_control` annotations target Anthropic-format dicts, so the provider should still call `prepare_request()` after converting messages to API format, regardless of who owns the history.
>
> The practical recommendation is the same in both architectures: the **provider calls the cache plugin** during request construction, because that's when format-specific annotation is natural. The difference is whether the provider reads from internal state (Variant A) or from parameters (Variant B).

### InstructionBudget Access

The cache plugin needs the `InstructionBudget` to make policy-aware decisions. Two options:

**Option A: Provider passes it.** The session sets the budget on the provider (`provider.set_instruction_budget(budget)`), and the provider passes it to the cache plugin in `prepare_request()`.

**Option B: Plugin receives it directly.** The session sets the budget on the cache plugin (`cache_plugin.set_budget(budget)`), and the plugin reads it in `prepare_request()` without the provider being involved.

Option B is simpler — it avoids adding budget awareness to the `ModelProviderPlugin` protocol, which would affect all 7 providers.

> **Cross-reference:** Option B becomes even more natural under the [Session-Owned History proposal](design/session-owned-history-impact-analysis.md). In that architecture, the provider is meant to be stateless with respect to messages — adding a budget reference to it would re-introduce state and work against the design goal. Option B keeps the budget flow entirely at the session level, consistent with session-owned history's principle that the session orchestrates all context management.

```python
# JaatoSession - after budget update
if self._cache_plugin:
    self._cache_plugin.set_budget(self._instruction_budget)
```

### Complete Lifecycle

#### Variant A — Provider-Owned History (Implemented)

```
Session starts:
  1. JaatoSession.configure()
       → creates provider
       → discovers cache plugin (by provider_name match)
       → cache_plugin.initialize(config)
       → cache_plugin.set_budget(instruction_budget)
       → provider.set_cache_plugin(cache_plugin)

Each turn:
  2. JaatoSession.send_message(prompt)
       → provider.send_message(text)
           → provider._build_api_kwargs()
               → cache_plugin.prepare_request(system, tools, messages)
                   → reads budget GC policies for BP3 placement
                   → returns annotated system/tools/messages
           → API call with cache annotations
           → returns ProviderResponse (with TokenUsage)
       → cache_plugin.extract_cache_usage(response.usage)

GC triggers:
  3. JaatoSession._maybe_collect_after_turn()
       → gc_plugin.collect(history, ..., budget)
           → returns (new_history, GCResult)
       → session.reset_session(new_history)
           → provider.create_session(system, tools, history=new_history)
       → cache_plugin.on_gc_result(result)
           → tracks prefix invalidation
       → cache_plugin.set_budget(updated_budget)

Budget changes:
  4. JaatoSession._emit_instruction_budget_update()
       → cache_plugin.set_budget(updated_budget)
```

#### Variant B — Session-Owned History (Not Started — After Phase 2+ Migration)

> **Cross-reference:** See [Session-Owned History Impact Analysis](design/session-owned-history-impact-analysis.md), Phases 2-5.

```
Session starts:
  1. JaatoSession.configure()
       → creates provider
       → discovers cache plugin (by provider_name match)
       → cache_plugin.initialize(config)
       → cache_plugin.set_budget(instruction_budget)
       → self._cache_plugin = cache_plugin  (no provider.set_cache_plugin needed)

Each turn:
  2. JaatoSession.send_message(prompt)
       → self._messages.append(user_msg)
       → provider.complete(self._messages, system, tools, ...)
           → converts to API format
           → cache_plugin.prepare_request(api_system, api_tools, api_messages)
               → reads budget GC policies for BP3 placement
               → returns annotated api_system/api_tools/api_messages
           → API call with cache annotations
           → returns ProviderResponse (with TokenUsage)
       → self._messages.append(response_msg)
       → cache_plugin.extract_cache_usage(response.usage)

GC triggers:
  3. JaatoSession._maybe_collect_after_turn()
       → gc_plugin.collect(self._messages, ..., budget)
           → returns (new_messages, GCResult)
       → self._history.replace(new_messages)  (no provider.create_session call)
       → cache_plugin.on_gc_result(result)
           → tracks prefix invalidation
       → cache_plugin.set_budget(updated_budget)

Budget changes:
  4. JaatoSession._emit_instruction_budget_update()
       → cache_plugin.set_budget(updated_budget)
```

**Key simplifications in Variant B:**
- No `provider.set_cache_plugin()` — session holds the plugin directly
- No `provider.create_session(history=...)` after GC — session mutates its own list
- The `CachePlugin` protocol is **identical** in both variants — only the orchestration changes

---

## GC-Cache Coordination

### How `gc_budget` Preserves Cached Prefixes

The `BudgetGCPlugin`'s phased removal strategy naturally preserves the cached prefix:

1. **Phase 1 (ENRICHMENT + EPHEMERAL)** removes content that was never worth caching — enrichment is regenerated each turn, and ephemeral content (working turns, discoverable tool schemas) would have been excluded from cache breakpoints anyway.

2. **Phase 2 (PARTIAL conversation turns)** removes older working turns while respecting `preserve_recent_turns`. Since the Anthropic provider's `cache_exclude_recent_turns` (default: 2) already excludes recent turns from BP3, these two settings work in concert — the cache doesn't cover the most volatile turns, and GC doesn't touch the most recent ones.

3. **Phase 3 (PRESERVABLE)** is the only phase that may disrupt the cached prefix. It only runs when `usage >= pressure_percent` (default: 90%) and is completely disabled in continuous mode. When it does run, PRESERVABLE content (clarification Q&A, turn summaries) may be removed, which could invalidate BP3's cached history.

4. **LOCKED content is never removed** — system instructions (BP1) and core tool schemas (BP2) are always stable.

### How the InstructionBudget Informs Caching

```python
budget = session._instruction_budget

# What's safe to cache (will never be GC'd)?
locked = budget.locked_tokens()

# What might survive but could be collected under pressure?
preservable = budget.preservable_tokens()

# How much room does GC have before touching PRESERVABLE?
headroom = budget.gc_headroom_percent()

# What's the overall utilization?
utilization = budget.utilization_percent()
```

For Anthropic's breakpoint strategy, the key insight is:

1. **BP1 (system)** and **BP2 (tools)** correspond to `InstructionSource.SYSTEM` (LOCKED) and `InstructionSource.PLUGIN` (core children are LOCKED) — these are always stable
2. **BP3 (history)** should be placed at the boundary between LOCKED/PRESERVABLE turns and EPHEMERAL turns in `InstructionSource.CONVERSATION`'s children
3. Recent turns (excluded via `cache_exclude_recent_turns`) are the most volatile and should not be cached

### How GC Results Could Notify Cache

After GC runs, `JaatoSession` applies the result:

```python
new_history, result = self._gc_plugin.collect(
    history, context_usage, self._gc_config, reason,
    budget=self._instruction_budget,
)
# result.removal_list contains GCRemovalItem entries
# Each entry has: source, child_key, tokens_freed, reason, message_ids
```

The `GCResult.details` dict from `gc_budget` includes a breakdown by phase:

```python
details = {
    "enrichment_cleared": True,              # Phase 1a ran
    "ephemeral_removed": 3,                  # Phase 1b: 3 entries removed
    "partial_removed": 2,                    # Phase 2: 2 turns removed
    "preservable_removed": 0,               # Phase 3: nothing under pressure
}
```

A future cache plugin could inspect this to determine cache impact:

```python
def on_gc_result(self, result: GCResult) -> None:
    # Only PRESERVABLE removal risks disrupting the cached prefix
    if result.details.get("preservable_removed", 0) > 0:
        self._prefix_may_be_invalidated = True
    # EPHEMERAL and PARTIAL removal is safe — that content was
    # outside the cached prefix (or excluded by cache_exclude_recent_turns)
```

---

## Prompt Structure for Optimal Caching

The prompt naturally follows this structure, which aligns with both Anthropic's prefix-based caching and GC policy assignments:

```
┌─────────────────────────────────────┐
│  LOCKED (highest cache priority)    │  ← Cached, 90% cheaper
│  - System instructions (SYSTEM)     │     BP1 (Anthropic)
│  - Core tool schemas (PLUGIN/core)  │     BP2 (Anthropic)
│  - Original request (CONVERSATION)  │
├─────────────────────────────────────┤
│  PRESERVABLE (cache-worthy)         │  ← Cached until pressure_percent
│  - Clarification Q&A               │     BP3 at boundary
│  - Turn summaries                   │
├─────────────────────────────────────┤
│  EPHEMERAL (skip caching)           │  ← Full price, short-lived
│  - Working turns                    │     No breakpoint here
│  - Enrichment content               │
│  - Discoverable tool schemas        │
├─────────────────────────────────────┤
│  Recent turns (not cached)          │  ← Excluded by
│  - Latest user message              │     cache_exclude_recent_turns
└─────────────────────────────────────┘
```

---

## Synergy with Terse Prompting

Terse prompting (60-70% token reduction for AI-to-AI communication) and KV caching are complementary but work on different axes:

- **Terse prompting** reduces the total number of tokens, saving cost linearly.
- **KV caching** reduces the compute cost per cached token by ~90%, saving on repeated prefixes.

The optimal strategy for jaato:

- Use **natural/verbose language** for the **LOCKED prefix** (system prompt, tool schemas). It's cheap when cached and benefits from clarity.
- Use **terse prompting** for the **EPHEMERAL** parts (per-turn working output, dynamic context). These are full-price tokens that benefit from compression.
- The combination can yield compounding savings beyond either technique alone.

---

## Anthropic-Specific Considerations

### The 1-Hour TTL

For jaato's longer agentic sessions (tool-heavy workflows that run 10+ minutes with multiple steps), the default 5-minute TTL can expire between steps. The 1-hour TTL costs more per cache write (2x vs 1.25x base) but avoids repeated cache misses during extended workflows.

Configuration: `config.extra["cache_ttl"] = "1h"` — currently stored on the provider but only `"5m"` (ephemeral) cache_control type is emitted. Extended TTL support would require emitting `{"type": "ephemeral", "ttl": "1h"}`.

### Breakpoint Allocation Strategy

With 4 breakpoints available, the current allocation and proposed improvement:

| Breakpoint | Target | Current (provider-internal) | Proposed (cache plugin) |
|---|---|---|---|
| BP1 | System instruction | cache_control on system text block | Same |
| BP2 | Last tool definition | cache_control on `tools[-1]` (sorted) | Same |
| BP3 | Historical messages | Last MODEL before `cache_exclude_recent_turns` window | Last LOCKED/PRESERVABLE turn boundary from `InstructionBudget` |
| BP4 | (Reserved) | Not used | Not used |

The key improvement is BP3: the current implementation uses a simple turn count (`cache_exclude_recent_turns = 2`) which doesn't know whether those turns are important or not. The proposed plugin would use the budget's GC policies to place the breakpoint at the PRESERVABLE/EPHEMERAL boundary — caching exactly the content that `gc_budget` won't remove.

### Tool Sorting for Cache Stability

The provider sorts tools by name before sending to Anthropic:

```python
anthropic_tools = sorted(anthropic_tools, key=lambda t: t["name"])
```

This prevents cache invalidation when tool registration order varies between sessions or when deferred tool loading changes the order dynamically.

---

## Provider-Specific Considerations

### Inheritance Cleanup: AnthropicProvider (Implemented)

After extraction, `AnthropicProvider` was **refactored** rather than just stripped. The cache logic was extracted into `AnthropicCachePlugin`, and `_build_api_kwargs()` was split into composable methods (`_build_system_blocks`, `_build_tool_list`, `_apply_legacy_cache_annotations`). The provider gained `set_cache_plugin()` for plugin attachment and delegates to `CachePlugin.prepare_request()` when a plugin is attached.

| Extracted from provider | Destination in `cache_anthropic` plugin |
|---|---|
| `_enable_caching`, `_cache_ttl` | `AnthropicCachePlugin._ttl` |
| `_cache_history`, `_cache_exclude_recent_turns` | Replaced by budget-aware `_find_policy_boundary()` |
| `_cache_min_tokens` | `AnthropicCachePlugin._should_cache_content()` |
| `_build_api_kwargs()` breakpoint injection | `prepare_request()` |
| `_compute_history_cache_breakpoint()` | `_find_policy_boundary()` (budget-aware) + `_compute_history_breakpoint_by_recency()` (fallback) |
| `_should_cache_content()`, `_get_cache_min_tokens()`, `_estimate_tokens()` | Moved as-is |

**Legacy fallback retained:** The original cache code path remains in `AnthropicProvider` for backward compatibility when no `CachePlugin` is attached. This means `_enable_caching`, `_cache_history`, etc. still exist on the provider — they're used only in the fallback path.

**Child provider `_enable_caching = False` retained:** `ZhipuAIProvider` and `OllamaProvider` still set `_enable_caching = False`, but it's now documented as "legacy annotation suppression" — it disables the parent's fallback cache path so the attached plugin (or no plugin, for Ollama) handles caching instead. This flag becomes unnecessary once the legacy fallback is removed from `AnthropicProvider`.

### ZhipuAI: Implicit Caching (Implemented)

`ZhipuAIProvider` inherits `AnthropicProvider` for API compatibility (Z.AI exposes an Anthropic-compatible endpoint). The separate `ZhipuAICachePlugin` is now implemented and provides:

- **Monitoring**: Extracts `cached_tokens` from the response `TokenUsage` and correlates with GC policy assignments from the `InstructionBudget`.
- **Advisory**: Detects when GC disrupts a cached prefix and tracks invalidation count via `on_gc_result()`.
- **No annotation injection**: `prepare_request()` is a pass-through since Z.AI caching is fully automatic.
- **Independence from parent**: Cache logic is in the plugin, not the provider inheritance chain.

Cache tokens appear in `usage.prompt_tokens_details.cached_tokens` (OpenAI-compatible format) and are mapped to `TokenUsage.cache_read_tokens` by the provider's response parser.

### Ollama: No Caching (No Change Needed)

`OllamaProvider` also inherits `AnthropicProvider`. Ollama doesn't support prompt caching. No `CachePlugin` matches the provider name `"ollama"`, so no caching is applied — correct behavior. The `_enable_caching = False` flag is retained to suppress the parent's legacy fallback path.

### Google GenAI: Future Context Caching

`GoogleGenAIProvider` is standalone (no inheritance). Google offers explicit context caching for Gemini models (different mechanism from Anthropic's — cached content is created via a separate API call and referenced by name). A future `cache_google_genai/` plugin could implement this without modifying `GoogleGenAIProvider`.

### Other Providers

- **AntigravityProvider**: Standalone. No caching API known. No plugin needed.
- **ClaudeCLIProvider**: Standalone. The CLI handles its own caching internally. No plugin needed.
- **GitHubModelsProvider**: Standalone. Caching depends on the underlying model provider (OpenAI, Anthropic, Google). A plugin could be added if GitHub Models exposes cache tokens in responses.

---

## Impact of Session-Owned History {#impact-of-session-owned-history}

The [Session-Owned History Impact Analysis](design/session-owned-history-impact-analysis.md) proposes inverting message history ownership from providers to the session. This section summarizes how that proposal affects the cache plugin design.

### Summary of the Session-Owned History Proposal

The proposal proceeds in 5 phases:

1. **Phase 1:** Introduce `SessionHistory` wrapper — session holds canonical copy, still syncs to provider
2. **Phase 2:** Add stateless `complete(messages, system, tools, ...) -> ProviderResponse` to providers
3. **Phase 3:** Migrate all providers to `complete()`
4. **Phase 4:** Remove legacy `create_session()`, `get_history()`, `send_message()` from protocol
5. **Phase 5:** Simplify session internals — GC operates on `self._messages` directly

### Impact Assessment by Cache Doc Component

| Component | Impact | Notes |
|-----------|:------:|-------|
| `CachePlugin` protocol (`prepare_request`, `on_gc_result`, etc.) | **None** | Signature takes explicit parameters, not provider state. Works in both architectures. |
| `_find_policy_boundary(budget)` logic | **None** | Operates on `InstructionBudget`, which is orthogonal to history ownership. |
| GC-Cache coordination (`on_gc_result`) | **None** | Cache invalidation tracking is session-level in both variants. |
| `InstructionBudget` access (Option B) | **None** | Already recommended; becomes even more natural under session-owned history. |
| Provider-level delegation (Variant A) | **Low** | Still works — provider calls `prepare_request()` using received parameters instead of `self._history`. |
| `set_cache_plugin()` on provider protocol | **Medium** | May become unnecessary if session orchestrates. Provider can still hold plugin reference via constructor. |
| Integration data flow diagrams | **Medium** | New Variant B diagrams needed (added above in [Proposed Integration](#proposed-integration-provider-delegates-to-cache-plugin)). |
| "Why Not Session-Level Interception?" section | **High** | 2 of 3 arguments invalidated. Rewritten as [balanced comparison](#session-level-vs-provider-level-cache-orchestration) above. |
| GC flow (`reset_session` → `create_session`) | **High** | Eliminated under Phase 4+. GC mutates `session._messages` directly. |

### What Stays the Same

The core cache plugin design is **robust to the session-owned history change**:

- The `CachePlugin` protocol requires no signature changes
- Budget-aware breakpoint placement (`_find_policy_boundary`) is unaffected
- GC-cache coordination via `on_gc_result()` works identically
- The "One Taxonomy, Two Consumers" principle applies regardless of history ownership
- All provider-specific cache plugins (Anthropic, ZhipuAI) are unaffected
- Prompt structure for optimal caching is unchanged

### What Changes

The **integration wiring** simplifies under session-owned history:

1. **No `provider.set_cache_plugin()` needed** — session holds the plugin
2. **No `provider.create_session(history)` after GC** — session mutates its own list
3. **Provider is stateless** w.r.t. history — receives messages as parameter to `complete()`
4. **Cache plugin wiring is purely session-level** — no duck-typing checks on provider

### Sequencing Status

**Variant A (cache plugin first) was implemented** on 2026-02-22. The next step is the A→B transition, which becomes possible when session-owned history Phase 2 lands and Anthropic gets a stateless `complete()` method. The transition is ~2 hours of mechanical work with no design decisions required.

See the [Impact Analysis](design/cache-plugin-sequencing-impact-analysis.md) for the full implementation progress and A→B transition breakdown.

---

## Design Rationale: One Taxonomy, Two Consumers

The key architectural advantage of building cache decisions on the `InstructionBudget`'s GC policy assignments is that **there is no duplicate lifecycle model**. The `InstructionBudget` already understands:

- Which content is essential and permanent (LOCKED) — system instructions, core tools, original request.
- Which content is important but expendable under pressure (PRESERVABLE) — clarification history, turn summaries.
- Which content is freely collectible (EPHEMERAL) — working turns, enrichment, discoverable tool schemas.

The `BudgetGCPlugin` and the proposed cache plugins would both read these same policy assignments. The GC plugin uses them to decide removal priority; the cache plugin uses them to decide breakpoint placement. Neither system invents its own content lifecycle model.

When tools declare their GC policy (e.g., a core tool marked LOCKED via `DEFAULT_TOOL_POLICIES[PluginToolType.CORE]`), `gc_budget` will never remove it and the cache system automatically treats it as high-priority cached content — without any cross-system configuration.

This also means improvements to GC policy assignment (such as per-tool reliability policies or dynamic CONDITIONAL evaluation via `evaluate_gc_policy()`) automatically improve both GC behavior and cache breakpoint placement.
