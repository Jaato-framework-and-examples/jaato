# Design — Cache-Aware GC (Option 2)

**Status:** Backlog (LOW priority)
**Author:** Analysis from 2026-04-08 conversation
**Related commits:**
- `1d281032` — telemetry for the GC ↔ cache plugin coordination (the verification baseline for this design)

## Context

Jaato's `gc_budget` plugin and provider cache plugins (`cache_anthropic`,
`cache_zhipuai`, `cache_google_genai`) work in parallel today, with no
explicit coordination beyond a one-way `on_gc_result()` callback that
only flips `_prefix_invalidated = True` when GC removes PRESERVABLE
content.

Provider caching is **prefix-hashed**: a cache hit requires the request
bytes to match the cached request byte-for-byte from position 0 up to
the cache point. If GC removes any content from before the cache
breakpoint, the prefix changes and the cache misses for everything
after that point.

`cache_anthropic` places its history breakpoint (BP3) at the **last
LOCKED/PRESERVABLE conversation child**. The intent is "everything
before the breakpoint is stable, everything after is EPHEMERAL." But
this assumption isn't enforced — `gc_budget` is free to remove
EPHEMERAL content from anywhere in the history, including positions
before the breakpoint.

### The actual problem

A typical multi-turn conversation:

```
[User1: LOCKED] → [working1: EPHEMERAL] → [summary1: PRESERVABLE]
[User2: LOCKED] → [working2: EPHEMERAL] → [summary2: PRESERVABLE]
[User3: LOCKED] → [working3: EPHEMERAL] → [summary3: PRESERVABLE]   ← BP3 here
```

When `gc_budget` runs Phase 1b (remove oldest EPHEMERAL first), it
removes `working1`. The new history shifts: the bytes at the position
where `working1` lived now contain `summary1` instead. The cached
prefix is invalidated from that position onwards — including BP3 at
`summary3`.

### Empirical observation

After commit `1d281032` added telemetry for this dance, baseline
measurements show **~98% cache hit rate** in normal operation. The
concern is real but the impact in practice is small — most sessions
don't trigger GC frequently enough to expose the issue. **This design
is parked at LOW priority unless a future workload starts seeing
significant cache misses on turns following GC operations.**

## Goals

1. When GC runs, removed content should NOT come from before the cache
   breakpoint, so the cached prefix survives across turns.
2. The current "EPHEMERAL/PARTIAL" semantics should still apply — GC
   still removes the oldest, largest, lowest-policy content, just
   constrained to the post-breakpoint region.
3. When the post-breakpoint region doesn't contain enough freeable
   content, GC must escalate gracefully — the cache break should be
   explicit and observable, not silent.
4. Existing GC plugins (`gc_truncate`, `gc_summarize`, `gc_hybrid`)
   should continue to work without changes. Only `gc_budget` opts in.
5. Cache plugins that don't expose an anchor (implicit caching like
   ZhipuAI, system-only caching like Google GenAI) should not be
   affected — `cache_anchor_message_id` returns `None`, GC behaves as
   today.

## Non-goals

- Making cache anchor placement smarter (the cache plugin's existing
  policy boundary logic stays unchanged).
- Automatically adjusting `pressure_percent` based on cache state.
- Cross-provider cache portability.

## API changes

### Cache plugin protocol

Already added in commit `1d281032` for telemetry purposes:

```python
def get_cache_anchor_message_id(self) -> Optional[str]:
    """Return the message_id at which the currently-cached prefix ends.

    Returns None when caching is disabled, before the first request,
    or when the cache plugin uses a strategy that doesn't have a
    history-level anchor (implicit caching, system-only caching).
    """
```

Implemented by:
- `cache_anthropic` → returns `_budget_bp3_message_id` (or `None`)
- `cache_zhipuai` → returns `None` (implicit)
- `cache_google_genai` → returns `None` (system+tools only)

### GC plugin protocol

Add an optional parameter to `GCPlugin.collect()`:

```python
def collect(
    self,
    history: List[Message],
    context_usage: Dict[str, Any],
    config: GCConfig,
    reason: GCTriggerReason,
    budget: Optional[InstructionBudget] = None,
    cache_anchor_message_id: Optional[str] = None,  # NEW
) -> Tuple[List[Message], GCResult]:
```

Default `None` keeps every existing caller and plugin working unchanged.

### Session glue

```python
anchor = None
if self._cache_plugin and hasattr(self._cache_plugin, "get_cache_anchor_message_id"):
    anchor = self._cache_plugin.get_cache_anchor_message_id()

new_history, result = self._gc_plugin.collect(
    history, usage, config, reason,
    budget=self._instruction_budget,
    cache_anchor_message_id=anchor,
)
```

## `gc_budget` implementation

### Building the protected set

```python
def _build_protected_message_ids(
    self, history: List[Message], anchor_id: Optional[str]
) -> Set[str]:
    """Return the set of message_ids GC must not remove.

    Walks history forward until the anchor message is reached.
    Returns an empty set if the anchor is None or not found in
    history (e.g. stale anchor after a session reset).
    """
    if not anchor_id:
        return set()
    protected: Set[str] = set()
    for msg in history:
        protected.add(msg.message_id)
        if msg.message_id == anchor_id:
            return protected
    return set()  # Anchor not found — graceful degradation
```

### Candidate filter

```python
def _is_safe_to_remove(
    self, entry: BudgetEntry, protected: Set[str]
) -> bool:
    """An entry is safe iff none of its message_ids are protected."""
    if not entry.message_ids:
        return True  # No message bounds → conservative allow (legacy)
    return not any(mid in protected for mid in entry.message_ids)
```

### Phase-by-phase impact

| Phase | Cache impact | New behavior |
|---|---|---|
| **1a — ENRICHMENT bulk-clear** | None (pure bookkeeping, no message mutation) | Unchanged |
| **1b — EPHEMERAL** | Removes messages → may break cache | Filter candidates by `_is_safe_to_remove()` |
| **2 — PARTIAL turns** | Removes messages → may break cache | Filter candidates by `_is_safe_to_remove()` |
| **3 — PRESERVABLE under pressure** | Always invalidates cache | Ignores protection — explicit escape valve |

ENRICHMENT (Phase 1a) is verified safe: it only resets a budget
tracking entry (`entry.tokens = 0; entry.children.clear()`) and never
mutates message bodies in history. Past tool results are immutable
once written.

### Tool call pair expansion

`_expand_removal_pairs()` requires careful handling. Today it runs
**after** the phases build the removal list. With protection, we need
to know about pair conflicts **before** committing a candidate, because
otherwise we'd have to retroactively cancel removals.

**Solution:** hoist `_build_tool_call_pair_map(history)` to the top of
`collect()`. The candidate safety check then becomes:

```python
def _is_safe_to_remove(
    self, entry: BudgetEntry, protected: Set[str], pair_map: Dict[str, List[str]]
) -> bool:
    if not entry.message_ids:
        return True
    # Check own message_ids
    if any(mid in protected for mid in entry.message_ids):
        return False
    # Check paired messages (tool_use ↔ tool_result partners)
    for mid in entry.message_ids:
        for partner_id in pair_map.get(mid, []):
            if partner_id in protected:
                return False
    return True
```

This handles all four pair sub-cases:

| tool_use position | tool_result position | Action |
|---|---|---|
| Protected | Protected | Skip both — cached, untouched |
| Protected | Free | Cannot remove tool_result alone (would orphan protected tool_use). Skip. |
| Free | Protected | Cannot remove tool_use alone (would orphan protected tool_result). Skip. |
| Free | Free | Remove both — current behavior |

### Escalation strategy

The hardest design decision: what happens when the post-protected
region doesn't contain enough freeable content?

**Strategy: pressure-aware drop**

```python
# After Phases 1b and 2
if tokens_freed < tokens_to_free and protected:
    # Couldn't free enough while preserving cache
    if percent_used >= config.pressure_percent or not config.pressure_percent:
        # Under real pressure or in continuous mode — break cache
        self._trace("collect: cache protection insufficient, dropping protection")
        result.details["escalated"] = True
        protected = set()
        # Re-run Phase 1b/2 without protection
        ... (recurse or repeat phase logic)
```

The `pressure_percent` threshold already exists for Phase 3
(PRESERVABLE removal) and represents the "I'd rather break things than
OOM" point. Reusing it for the protection-drop decision keeps the
escalation semantics consistent: at high pressure, both protection and
PRESERVABLE preservation are sacrificed.

The trace and the `escalated=True` detail flag make the cache break
**observable** — the telemetry already added in `1d281032` will
surface it as `gc.escalated=true` on the GC span.

### Stale anchor handling

If the anchor `message_id` is no longer in history (e.g. it was
removed by manual reset or a previous GC pass), `_build_protected_message_ids`
returns an empty set when the loop completes without finding the
anchor. GC behaves as if no protection was requested. ✓

### Cold start handling

On the first turn, `cache_anthropic` hasn't placed BP3 yet, so
`get_cache_anchor_message_id()` returns `None`. The protected set is
empty. GC behaves as today. ✓

## Cache plugin lifecycle

Verified in the conversation that produced this design:

```
1. Session: _maybe_collect_before_send()        ← GC may fire here
2. Session: provider.complete()
   ├── 2a. cache_plugin.prepare_request()
   │       ├── Walks budget conversation children
   │       ├── Sets self._budget_bp3_message_id = ...
   │       └── Returns annotated system + tools
   ├── 2b. Provider sends request
   └── 2c. Provider receives response
3. Session: tools execute → tool results appended to history
4. Session: _execute_tools_and_continue() → loop back to step 1
```

**Critical observation:** at step 1 (GC time), `_budget_bp3_message_id`
holds the breakpoint from the **last** request — exactly what's
currently cached on the provider side. The next request hasn't
recomputed it yet. So reading the anchor at GC time gives the right
value: "the prefix that's currently cached, which we want to preserve."

The only window where the anchor would be stale is **inside step 2**,
between `prepare_request()` overwriting it and the API call sending it
out. GC never fires inside step 2 — only between turns at step 1.

## Verification

The telemetry baseline added in commit `1d281032` is the verification
infrastructure. After implementing this design, query Phoenix/Jaeger
for:

- **Before:** `count(turn) where cache.outcome=miss after gc_span`
  → ~2% in current measurements
- **After:** Same query → should approach 0%
- **`gc.escalated=true` rate** → should be rare (only under real
  pressure); high rates indicate the protection is too aggressive
- **`cache.gc_invalidation_count` per session** → should plateau
  unless Phase 3 fires

## Risks

1. **Tool integrity refactor blast radius**: hoisting
   `_build_tool_call_pair_map()` before the phases changes the order
   of operations. Existing tests around tool integrity must pass.

2. **Anchor desync**: if the cache plugin recomputes the anchor between
   two requests (e.g. config change, provider switch), the anchor
   might point to a position that no longer represents what was
   actually cached. Mitigation: cache plugin should snapshot the anchor
   used in the **last successful request** rather than the one being
   computed for the next. Currently the anchor is overwritten at step
   2a — this would need to change to a "last-used" snapshot for full
   correctness.

3. **Insufficient escalation**: when escalation triggers, we re-run
   Phases 1b/2 without protection. This might still not free enough,
   pushing into Phase 3 (PRESERVABLE removal). Phase 3 wasn't designed
   to be a frequent fallback. Worth measuring `gc.escalated` rate
   after deployment.

4. **Per-provider cache semantics divergence**: Anthropic's explicit
   BP3, Z.AI's implicit caching, Google GenAI's CachedContent all have
   different semantics. The protected region from one provider's
   anchor doesn't apply to another. Solution is already in place:
   each cache plugin defines its own `get_cache_anchor_message_id()`,
   and providers without an anchor return `None`.

## Cost estimate

| Change | Lines | Complexity |
|---|---|---|
| `gc_budget` — protected set + filters | ~30 | Low |
| `gc_budget` — pair-aware safety check | ~10 | Low |
| `gc_budget` — escalation logic | ~20 | Medium |
| Pair map hoisting refactor | ~10 | Medium (test risk) |
| Session glue (1 site, both GC paths share `_apply_gc_removal_list`) | ~5 | Low |
| GC protocol parameter | ~1 | Low |
| Tests — protected EPHEMERAL filter | ~30 | Low |
| Tests — pair conflict cases (4 sub-cases) | ~50 | Medium |
| Tests — escalation under pressure | ~30 | Low |
| Tests — stale anchor / cold start | ~20 | Low |

Total: ~70 lines of production code + ~150 lines of tests, no protocol
breaking changes.

## Decision log

**2026-04-08:** Parked at LOW priority. Telemetry baseline shows ~98%
cache hit rate already; the concern was bigger than the actual problem.
Implement only if a future workload starts seeing significant cache
misses on turns following GC operations. The telemetry from commit
`1d281032` will surface the regression if it appears.
