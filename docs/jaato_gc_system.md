# JAATO Context Garbage Collection System

## Executive Summary

JAATO implements a **pluggable context garbage collection (GC) system** that prevents context window overflow during long-running agentic sessions. Four strategy plugins — `gc_truncate`, `gc_summarize`, `gc_hybrid`, and `gc_budget` — share a common `GCPlugin` protocol defined in `shared/plugins/gc/base.py`. The most advanced strategy, **`gc_budget`**, uses the `InstructionBudget` to make **policy-aware removal decisions** across a five-tier priority system (ENRICHMENT → EPHEMERAL → PARTIAL → PRESERVABLE → LOCKED) and supports a **continuous collection mode** that trims context after every turn rather than waiting for a threshold breach. The session integrates with all plugins through **proactive threshold monitoring during streaming**, **pre-send checks**, and **automatic budget synchronization** after each collection.

---

## Part 1: The Context Window Problem

### Why Context GC Is Needed

Agentic sessions produce large volumes of context: tool call arguments, tool results, intermediate reasoning, enrichment data, and accumulated conversation history. Without management, this context eventually exceeds the model's context window, causing request failures.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTEXT GROWTH OVER TIME                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Token Usage                                                         │
│  ▲                                                                   │
│  │                                          ╭─── Context Limit ───── │
│  │                                    ╭─────╯     (128K tokens)      │
│  │                               ╭────╯                              │
│  │                          ╭────╯    ← Without GC: crash            │
│  │                     ╭────╯                                        │
│  │                ╭────╯                                             │
│  │           ╭────╯                                                  │
│  │      ╭────╯       ╭──── Threshold (80%) ─── GC triggers here     │
│  │ ╭────╯       ╭────╯                                               │
│  │─╯       ╭────╯         ← With GC: usage drops to target (60%)    │
│  │    ╭────╯    ╭─────╮        then resumes growing                  │
│  │────╯    ╭────╯     ╰────╮                                        │
│  │    ╭────╯               ╰────╮     Sawtooth pattern               │
│  │────╯                         ╰────                                │
│  └──────────────────────────────────────────────────────────────►    │
│                         Conversation Turns                           │
│                                                                      │
│  Challenges:                                                         │
│  1. Context grows with every tool call and model response            │
│  2. Not all context is equally important (enrichment vs user query)  │
│  3. Aggressive removal loses coherence; conservative risks overflow  │
│  4. Budget must stay synchronized with actual history after removal  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: GC Plugin Architecture

### The Plugin Protocol

All four GC strategies implement the `GCPlugin` protocol defined in `shared/plugins/gc/base.py`. This protocol-based design allows the session to swap strategies without code changes.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GC PLUGIN ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  shared/plugins/gc/base.py                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ GCPlugin (Protocol)                                          │   │
│  │                                                               │   │
│  │ @property name → str                                          │   │
│  │ initialize(config) → None                                     │   │
│  │ shutdown() → None                                             │   │
│  │ should_collect(context_usage, config) → (bool, reason?)       │   │
│  │ collect(history, usage, config, reason, budget?) → (hist, res)│   │
│  └──────────────┬──────────────┬──────────────┬─────────────────┘   │
│                  │              │              │                      │
│       ┌──────────┴──┐  ┌───────┴──────┐  ┌───┴──────────────┐      │
│       │ gc_truncate  │  │ gc_summarize │  │ gc_hybrid        │      │
│       │              │  │              │  │ (generational)   │      │
│       │ Remove oldest│  │ Compress old │  │ Ancient:truncate │      │
│       │ turns        │  │ into summary │  │ Middle:summarize │      │
│       │              │  │              │  │ Recent:preserve  │      │
│       └──────────────┘  └──────────────┘  └──────────────────┘      │
│                                                                      │
│       ┌─────────────────────────────────────────────────────────┐   │
│       │ gc_budget (policy-aware)                                │   │
│       │                                                          │   │
│       │ Uses InstructionBudget GC policies for smart decisions   │   │
│       │ Five-tier removal priority                               │   │
│       │ Supports continuous collection mode                      │   │
│       │ Budget-synchronized removal via GCRemovalItem list       │   │
│       └─────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Supporting Types:                                                   │
│  ─────────────────                                                   │
│  GCConfig         Thresholds, preservation settings, mode control    │
│  GCResult         Outcome: tokens freed, removal list, details       │
│  GCRemovalItem    What was removed (source, child_key, message_ids)  │
│  GCTriggerReason  Why collected (THRESHOLD, MANUAL, TURN_LIMIT,      │
│                   PRE_MESSAGE)                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Plugin Discovery

GC plugins are registered as Python entry points in `pyproject.toml` and discovered at runtime:

```toml
[project.entry-points."jaato.gc_plugins"]
gc_truncate  = "shared.plugins.gc_truncate:create_plugin"
gc_summarize = "shared.plugins.gc_summarize:create_plugin"
gc_hybrid    = "shared.plugins.gc_hybrid:create_plugin"
gc_budget    = "shared.plugins.gc_budget:create_plugin"
```

`discover_gc_plugins()` finds all registered plugins; `load_gc_plugin(name, config)` instantiates and initializes one by name.

---

## Part 3: The Four Strategies Compared

### Strategy Comparison

| Aspect | gc_truncate | gc_summarize | gc_hybrid | gc_budget |
|--------|-------------|--------------|-----------|-----------|
| **Approach** | Remove oldest turns | Compress old turns into summary | Generational: truncate ancient, summarize middle, preserve recent | Policy-aware: remove by GC policy tier |
| **Context preservation** | None for removed turns | Summary retains key points | Partial (summary for middle, none for ancient) | Varies by content importance |
| **Requires summarizer** | No | Yes | Optional (degrades to truncation) | No |
| **Budget-aware** | No | No | No | Yes |
| **Continuous mode** | No | No | No | Yes |
| **Content discrimination** | All turns equal | All turns equal | Age-based tiers | Policy-based tiers |
| **Overhead** | Minimal | LLM call for summary | LLM call if summarizer present | Policy evaluation per entry |
| **Best for** | Simple sessions, low overhead | Long sessions needing context continuity | Balanced approach | Enterprise, long-running agents |

### Strategy Decision Flow

```
Which GC strategy?
    │
    ├── Need simplest/fastest? → gc_truncate
    │
    ├── Need context preservation? → gc_summarize
    │   (requires summarizer function)
    │
    ├── Need balanced approach? → gc_hybrid
    │   (generational: truncate + summarize)
    │
    └── Need intelligent decisions? → gc_budget
        ├── Budget-aware removal by policy
        ├── Content importance matters
        └── Continuous mode for predictable trimming
```

---

## Part 4: The Instruction Budget and GC Policies

### GC Policy Tiers

The `InstructionBudget` (defined in `shared/instruction_budget.py`) assigns a `GCPolicy` to every tracked instruction source. These policies determine removal priority during budget-aware garbage collection.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GC POLICY TIERS                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  GCPolicy.LOCKED  🔒                                                │
│  ─────────────────                                                   │
│  Never removed under any circumstances.                              │
│  • System instructions (base, client, framework)                     │
│  • User's original request                                           │
│  • Core tool schemas                                                 │
│                                                                      │
│  GCPolicy.PRESERVABLE  ◑                                            │
│  ──────────────────────                                              │
│  Removed only under extreme pressure (usage > pressure_percent).     │
│  Never touched in continuous mode.                                   │
│  • Clarification Q&A pairs                                           │
│  • Turn summaries                                                    │
│  • GC summary messages (gc_summary_1, gc_summary_2, ...)            │
│                                                                      │
│  GCPolicy.PARTIAL  ◐                                                │
│  ─────────────────                                                   │
│  Container with mixed children — some removable, some not.           │
│  • CONVERSATION source (contains LOCKED + EPHEMERAL turns)           │
│  • PLUGIN source (contains LOCKED core + EPHEMERAL discoverable)     │
│                                                                      │
│  GCPolicy.EPHEMERAL  ○                                              │
│  ────────────────────                                                │
│  First candidates for removal. Regenerated or non-essential.         │
│  • ENRICHMENT source (regenerated each turn)                         │
│  • Discoverable tool schemas                                         │
│  • Working/verbose tool output                                       │
│                                                                      │
│  Removal Priority:  EPHEMERAL → PARTIAL → PRESERVABLE → [LOCKED]   │
│                     (first)      (middle)   (last resort)  (never)   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Default Policy Assignments

| Source | Default Policy | Children |
|--------|---------------|----------|
| **SYSTEM** | LOCKED | base=LOCKED, client=LOCKED, framework=LOCKED |
| **PLUGIN** | PARTIAL | core tools=LOCKED, discoverable tools=EPHEMERAL |
| **ENRICHMENT** | EPHEMERAL | (regenerated each turn) |
| **CONVERSATION** | PARTIAL | original_request=LOCKED, clarification=PRESERVABLE, working=EPHEMERAL |

### SourceEntry Structure

Each budget entry tracks tokens, policy, creation time, and message IDs for precise removal:

```python
@dataclass
class SourceEntry:
    source: InstructionSource       # SYSTEM, PLUGIN, ENRICHMENT, CONVERSATION
    tokens: int                     # Direct token count (excluding children)
    gc_policy: GCPolicy             # LOCKED, PRESERVABLE, PARTIAL, EPHEMERAL
    label: Optional[str]            # Display label
    children: Dict[str, SourceEntry]  # Nested entries
    metadata: Dict[str, Any]
    created_at: Optional[float]     # Unix timestamp (for age-based ordering)
    message_ids: List[str]          # For GC history sync
```

---

## Part 5: The gc_budget Plugin — Five-Phase Collection

### How gc_budget Removes Content

When `gc_budget` collects, it executes up to four removal phases in strict priority order, stopping as soon as enough tokens have been freed to reach `target_percent`:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    gc_budget FIVE-TIER REMOVAL PRIORITY              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 1a: ENRICHMENT (Bulk Clear)                                   │
│  ─────────────────────────────────                                   │
│  Clear entire ENRICHMENT source at once.                             │
│  Always first — enrichment is regenerated every turn.                │
│  No message-level granularity needed; bulk clear.                    │
│                                                                      │
│      ┌─ tokens_to_free ──────────────────────────────────────┐      │
│      │████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│      │
│      │ enrichment │          remaining to free                │      │
│      └────────────┴──────────────────────────────────────────┘      │
│                                                                      │
│  Phase 1b: Other EPHEMERAL Entries (Oldest First)                    │
│  ────────────────────────────────────────────────                    │
│  Remove EPHEMERAL entries from PLUGIN and CONVERSATION.              │
│  Sorted by created_at timestamp — oldest removed first.              │
│  Discoverable tool schemas, verbose working output.                  │
│                                                                      │
│      ┌─ tokens_to_free ──────────────────────────────────────┐      │
│      │████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│      │
│      │ enrichment │ ephemeral  │     remaining to free        │      │
│      └────────────┴────────────┴─────────────────────────────┘      │
│                                                                      │
│  Phase 2: PARTIAL Conversation Turns (Oldest First)                  │
│  ──────────────────────────────────────────────────                  │
│  Remove old conversation turns with PARTIAL or EPHEMERAL policy.     │
│  Respects preserve_recent_turns and pinned_turn_indices.             │
│  Skips gc_summary_* entries (those are PRESERVABLE).                 │
│                                                                      │
│      ┌─ tokens_to_free ──────────────────────────────────────┐      │
│      │██████████████████████████████████████░░░░░░░░░░░░░░░░░│      │
│      │ enrichment │ ephemeral  │ partial    │  remaining      │      │
│      └────────────┴────────────┴────────────┴────────────────┘      │
│                                                                      │
│  Phase 3: PRESERVABLE (Extreme Pressure Only)                        │
│  ────────────────────────────────────────────                        │
│  Only if usage >= pressure_percent.                                  │
│  Never touched in continuous mode (pressure_percent=0/None).         │
│  Removes clarification pairs, turn summaries, GC summaries.         │
│  Oldest first by created_at.                                         │
│                                                                      │
│      ┌─ tokens_to_free ──────────────────────────────────────┐      │
│      │██████████████████████████████████████████████████████░░│      │
│      │ enrichment │ ephemeral  │ partial    │ preservable │   │      │
│      └────────────┴────────────┴────────────┴─────────────┘  │      │
│                                                                      │
│  LOCKED: Never Removed                                               │
│  ─────────────────────                                               │
│  System instructions, user's original request, core tool schemas.    │
│  Protected regardless of pressure level.                             │
│                                                                      │
│  Each phase stops as soon as tokens_freed >= tokens_to_free.         │
│  If one phase frees enough, later phases are skipped entirely.       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Message-Level Removal

After computing the removal list, `gc_budget` filters the conversation history by message ID:

```python
def _apply_removals_to_history(self, history, removal_list):
    ids_to_remove = set()
    for item in removal_list:
        if item.message_ids:
            ids_to_remove.update(item.message_ids)
    return [msg for msg in history if msg.message_id not in ids_to_remove]
```

This precise removal — rather than turn-index-based slicing — ensures that only the entries identified by the budget are removed from the actual conversation history.

---

## Part 6: Continuous Collection Mode

### Threshold Mode vs Continuous Mode

`gc_budget` supports two operating modes controlled by the `pressure_percent` setting:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OPERATING MODES                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  THRESHOLD MODE (default: pressure_percent > 0)                      │
│  ──────────────────────────────────────────────                      │
│                                                                      │
│  Token Usage                                                         │
│  ▲         pressure_percent (90%)  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   │
│  │                                                                   │
│  │    threshold_percent (80%)  ── GC triggers ──────────────────    │
│  │         ╭────╮                       ╭────╮                      │
│  │    ╭────╯    ╰────╮             ╭────╯    ╰────╮                 │
│  │────╯              ╰─────── ─────╯              ╰────────        │
│  │    target_percent (60%)  ── GC targets ──────────────────        │
│  │                                                                   │
│  └─────────────────────────────────────────────────────────►        │
│              Turns                                                   │
│                                                                      │
│  Behavior:                                                           │
│  • GC triggers when usage >= threshold_percent (80%)                 │
│  • Frees tokens until usage reaches target_percent (60%)             │
│  • PRESERVABLE touched only if usage >= pressure_percent (90%)       │
│  • Large swings between threshold and target (sawtooth)              │
│                                                                      │
│                                                                      │
│  CONTINUOUS MODE (pressure_percent = 0 or None)                      │
│  ──────────────────────────────────────────────                      │
│                                                                      │
│  Token Usage                                                         │
│  ▲                                                                   │
│  │                                                                   │
│  │    target_percent (60%)  ── GC targets ──────────────────        │
│  │    ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮          │
│  │────╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯──        │
│  │                                                                   │
│  │                                                                   │
│  └─────────────────────────────────────────────────────────►        │
│              Turns                                                   │
│                                                                      │
│  Behavior:                                                           │
│  • GC runs after every turn if usage > target_percent                │
│  • threshold_percent is ignored                                      │
│  • PRESERVABLE content is never touched                              │
│  • Small, predictable trims each turn (gentle sawtooth)              │
│  • Ideal for long-running sessions with stable context needs         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Configuration

| Setting | Threshold Mode | Continuous Mode |
|---------|---------------|-----------------|
| `pressure_percent` | > 0 (default: 90.0) | 0 or None |
| `threshold_percent` | Trigger point (default: 80.0) | Ignored |
| `target_percent` | Target after GC (default: 60.0) | Target after GC |
| GC frequency | Occasional, large collections | Every turn, small collections |
| PRESERVABLE content | Touched under extreme pressure | Never touched |

### GCConfig Properties

```python
@dataclass
class GCConfig:
    threshold_percent: float   # JAATO_GC_THRESHOLD (default: 80.0)
    target_percent: float      # JAATO_GC_TARGET (default: 60.0)
    pressure_percent: float    # JAATO_GC_PRESSURE (default: 90.0, 0=continuous)
    preserve_recent_turns: int # Always keep last N turns (default: 5)
    pinned_turn_indices: List[int]  # Specific turns to never remove
    check_before_send: bool    # Pre-send GC check (default: True)
    auto_trigger: bool         # Automatic triggering (default: True)

    @property
    def continuous_mode(self) -> bool:
        return not self.pressure_percent  # True when 0 or None
```

---

## Part 7: Proactive GC Integration

### Session-Level GC Lifecycle

The GC system integrates with `JaatoSession` at three points during each turn:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GC INTEGRATION IN SESSION LIFECYCLE               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  session.send_message(prompt)                                        │
│      │                                                               │
│      ├── 1. PRE-SEND CHECK                                           │
│      │   if gc_config.check_before_send:                             │
│      │       should, reason = plugin.should_collect(usage, config)    │
│      │       if should:                                               │
│      │           new_hist, result = plugin.collect(...)               │
│      │           _apply_gc_removal_list(result)                      │
│      │           _emit_instruction_budget_update()                   │
│      │                                                               │
│      ├── 2. STREAMING THRESHOLD MONITOR                              │
│      │   Provider streams response tokens...                         │
│      │   ┌─ on each usage update during streaming ─────────────┐    │
│      │   │ percent = (total_tokens / context_limit) * 100       │    │
│      │   │ if percent >= threshold and not already_crossed:      │    │
│      │   │     _gc_threshold_crossed = True                      │    │
│      │   │     on_gc_threshold(percent, threshold)  ← callback   │    │
│      │   └──────────────────────────────────────────────────────┘    │
│      │                                                               │
│      ├── 3. POST-TURN COLLECTION                                     │
│      │   if _gc_threshold_crossed:                                   │
│      │       new_hist, result = plugin.collect(...)                   │
│      │       update history                                          │
│      │       _apply_gc_removal_list(result)                          │
│      │       _emit_instruction_budget_update()                       │
│      │                                                               │
│      └── Return response                                             │
│                                                                      │
│  GC can also be triggered manually:                                  │
│  session.manual_gc() → forces collection with MANUAL reason          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Threshold Callback

The `on_gc_threshold` callback allows the UI to display warnings when context pressure rises:

```python
def on_gc_threshold_callback(percent_used: float, threshold: float):
    # UI can show warning notification
    print(f"Context pressure: {percent_used:.1f}% >= {threshold}%")

session.send_message(
    prompt,
    on_output=output_callback,
    on_gc_threshold=on_gc_threshold_callback,
)
```

---

## Part 8: Budget Synchronization

### The Synchronization Problem

When GC removes messages from history, the `InstructionBudget` must be updated to reflect the new token counts. Without synchronization, the budget would report stale values and GC would not trigger correctly on subsequent turns.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BUDGET SYNCHRONIZATION FLOW                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. GC Plugin produces removal_list                                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ removal_list = [                                              │   │
│  │   GCRemovalItem(source=ENRICHMENT, child_key=None,            │   │
│  │                 tokens_freed=2000, reason="enrichment_bulk")   │   │
│  │   GCRemovalItem(source=CONVERSATION, child_key="turn_3",      │   │
│  │                 tokens_freed=800, message_ids=["msg-7","msg-8"])│  │
│  │   GCRemovalItem(source=CONVERSATION, child_key="turn_4",      │   │
│  │                 tokens_freed=1200, message_ids=["msg-9","msg-10"])│ │
│  │ ]                                                              │   │
│  └────────────────────────────┬─────────────────────────────────┘   │
│                                │                                     │
│  2. Session applies removals   ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ _apply_gc_removal_list(result):                               │   │
│  │                                                                │   │
│  │   For each GCRemovalItem:                                      │   │
│  │     if child_key:                                              │   │
│  │       budget.remove_child(source, child_key)                   │   │
│  │     else:  (bulk clear)                                        │   │
│  │       entry.tokens = 0                                         │   │
│  │       entry.children.clear()                                   │   │
│  │                                                                │   │
│  │   If summary was created:                                      │   │
│  │     budget.add_child(CONVERSATION, "gc_summary_N",             │   │
│  │       tokens=summary_tokens, gc_policy=PRESERVABLE)            │   │
│  └────────────────────────────┬─────────────────────────────────┘   │
│                                │                                     │
│  3. Emit budget update         ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ _emit_instruction_budget_update():                             │   │
│  │   snapshot = budget.snapshot()                                  │   │
│  │   → on_instruction_budget_updated(snapshot)  ← callback        │   │
│  │   → ui_hooks.on_agent_instruction_budget_updated(...)          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  The budget is now consistent with the actual conversation history.  │
│  Next should_collect() call will use accurate utilization_percent.    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Budget as Single Source of Truth

The session's `get_context_usage()` uses the `InstructionBudget` as its sole data source:

```python
def get_context_usage(self) -> Dict[str, Any]:
    total_tokens = self._instruction_budget.total_tokens()
    context_limit = self._instruction_budget.context_limit
    percent_used = self._instruction_budget.utilization_percent()
    return {
        'context_limit': context_limit,
        'total_tokens': total_tokens,
        'percent_used': percent_used,
        'tokens_remaining': self._instruction_budget.available_tokens(),
    }
```

This ensures GC triggering decisions are based on accurate, budget-tracked token counts rather than estimates.

---

## Part 9: Conversation Coherence Preservation

### Turn Preservation Mechanisms

All four GC plugins respect turn-level preservation to maintain conversation coherence:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COHERENCE PRESERVATION                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Turn history:                                                       │
│  [turn_0] [turn_1] [turn_2] ... [turn_N-5] [turn_N-4] ... [turn_N] │
│   ↑ pinned                       ↑───── preserve_recent_turns ─────↑│
│                                                                      │
│  Mechanism 1: preserve_recent_turns (default: 5)                     │
│  ─────────────────────────────────────────────────                   │
│  Last N turns are never removed by any plugin.                       │
│  Ensures the model has recent conversation context.                  │
│                                                                      │
│  Mechanism 2: pinned_turn_indices                                    │
│  ────────────────────────────────                                    │
│  Specific turn indices (0-based) that are never removed.             │
│  Example: pin turn 0 to always keep the initial user request.        │
│                                                                      │
│  Mechanism 3: GC Policy (gc_budget only)                             │
│  ────────────────────────────────────────                            │
│  LOCKED entries are never removed regardless of other settings.      │
│  The user's original_request is LOCKED by default.                   │
│                                                                      │
│  Mechanism 4: Summary Chain                                          │
│  ──────────────────────────────                                      │
│  When gc_summarize or gc_hybrid creates summaries, those summaries   │
│  become PRESERVABLE budget entries (gc_summary_1, gc_summary_2...).  │
│  They bridge removed history to current context.                     │
│  gc_budget skips gc_summary_* entries in Phase 2 (PARTIAL removal).  │
│                                                                      │
│  Mechanism 5: Turn Boundaries                                        │
│  ────────────────────────────                                        │
│  GC operates on turn boundaries, not arbitrary message boundaries.   │
│  A turn is the atomic unit: user message + model response + tool     │
│  calls. No partial turn removal occurs.                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 10: Configuration and Deployment

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JAATO_GC_THRESHOLD` | `80.0` | Trigger GC when usage exceeds this % |
| `JAATO_GC_TARGET` | `60.0` | Target usage % after GC |
| `JAATO_GC_PRESSURE` | `90.0` | Touch PRESERVABLE above this % (0 = continuous mode) |

### Programmatic Configuration

```python
from shared.plugins.gc import load_gc_plugin, GCConfig

# Threshold mode (default)
gc_plugin = load_gc_plugin('gc_budget', {
    'preserve_recent_turns': 5,
    'target_percent': 60.0,
    'pressure_percent': 90.0,
    'notify_on_gc': True,
})
config = GCConfig(threshold_percent=80.0)
client.set_gc_plugin(gc_plugin, config)

# Continuous mode
gc_plugin = load_gc_plugin('gc_budget', {
    'preserve_recent_turns': 5,
    'target_percent': 60.0,
    'pressure_percent': 0,  # enables continuous mode
})
config = GCConfig(pressure_percent=None)  # or 0
client.set_gc_plugin(gc_plugin, config)
```

### Subagent GC Profiles

GC can be configured per subagent via `.jaato/profiles/*.json`:

```json
{
  "name": "research",
  "model": "gemini-2.5-flash",
  "gc": {
    "type": "gc_budget",
    "threshold_percent": 75.0,
    "target_percent": 50.0,
    "pressure_percent": 0,
    "preserve_recent_turns": 3
  }
}
```

The `GCProfileConfig` dataclass in `shared/plugins/subagent/config.py` maps these settings to `GCConfig` and plugin initialization.

---

## Part 11: GC Result and Observability

### GCResult Structure

Every collection returns a `GCResult` with detailed outcome information:

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether the operation completed |
| `items_collected` | `int` | Number of items removed |
| `tokens_before` | `int` | Token count before GC |
| `tokens_after` | `int` | Token count after GC |
| `tokens_freed` | `int` | Property: `tokens_before - tokens_after` |
| `plugin_name` | `str` | Which plugin performed the collection |
| `trigger_reason` | `GCTriggerReason` | Why GC was triggered |
| `removal_list` | `List[GCRemovalItem]` | Items removed for budget sync |
| `notification` | `Optional[str]` | Optional model-visible notification |
| `details` | `Dict` | Plugin-specific details |
| `error` | `Optional[str]` | Error message if failed |

### gc_budget Result Details

The `gc_budget` plugin includes phase breakdown in the `details` dict:

```python
details = {
    "target_tokens": 76800,       # Target after GC
    "tokens_to_free": 25600,      # How much needed to free
    "tokens_freed": 26100,        # How much actually freed
    "enrichment_cleared": True,   # Phase 1a ran
    "ephemeral_removed": 3,       # Phase 1b: 3 entries removed
    "partial_removed": 5,         # Phase 2: 5 turns removed
    "preservable_removed": 0,     # Phase 3: not needed
}
```

---

## Part 12: Fallback Behavior

### gc_budget Without a Budget

When `gc_budget.collect()` is called without an `InstructionBudget` (e.g., during early session setup before the budget is initialized), it falls back to simple turn-based truncation identical to `gc_truncate`:

```
gc_budget.collect(history, usage, config, reason, budget=None)
    │
    └── _fallback_truncate(history, usage, config, reason)
            │
            ├── Split history into turns
            ├── Get preserved indices (recent + pinned)
            ├── Remove non-preserved turns
            └── Return (new_history, GCResult with mode="fallback_truncate")
```

This ensures `gc_budget` remains operational even in edge cases where the budget is unavailable.

---

## Part 13: File Structure

```
shared/plugins/gc/
├── base.py                  # GCPlugin protocol, GCConfig, GCResult, GCRemovalItem
├── utils.py                 # Turn splitting, token estimation, message utilities
├── __init__.py              # Plugin discovery, load functions
└── tests/

shared/plugins/gc_truncate/
├── plugin.py                # TruncateGCPlugin — remove oldest turns
└── __init__.py

shared/plugins/gc_summarize/
├── plugin.py                # SummarizeGCPlugin — compress old turns
└── __init__.py

shared/plugins/gc_hybrid/
├── plugin.py                # HybridGCPlugin — generational (truncate + summarize)
└── __init__.py

shared/plugins/gc_budget/
├── plugin.py                # BudgetGCPlugin — policy-aware, continuous mode
├── __init__.py
└── tests/

shared/instruction_budget.py # InstructionBudget, SourceEntry, GCPolicy, defaults
shared/jaato_session.py      # GC integration: threshold monitor, pre-send, post-turn
```

---

## Part 14: Related Documentation

| Document | Focus |
|----------|-------|
| [jaato_model_harness.md](jaato_model_harness.md) | Overall harness architecture (GC as one of the three harness layers) |
| [jaato_instruction_sources.md](jaato_instruction_sources.md) | Instruction source assembly and token budgets |
| [jaato_subagent_architecture.md](jaato_subagent_architecture.md) | Per-subagent GC profiles via SubagentConfig |
| [jaato_opentelemetry.md](jaato_opentelemetry.md) | GC operations appear in OTel spans |

---

## Part 15: Color Coding Suggestion for Infographic

- **Blue:** Configuration layer (GCConfig, environment variables, thresholds)
- **Green:** GC plugins (gc_truncate, gc_summarize, gc_hybrid, gc_budget)
- **Orange:** Instruction Budget and GC policies (LOCKED, PRESERVABLE, PARTIAL, EPHEMERAL)
- **Red:** Removal phases (enrichment clear, ephemeral removal, partial turns, preservable under pressure)
- **Purple:** Session integration (pre-send check, streaming monitor, post-turn collection)
- **Gray:** Data flow arrows (history → GC plugin → new history, removal_list → budget sync)
- **Yellow:** Mode indicators (threshold mode vs continuous mode, sawtooth patterns)
