# Impact Analysis: Cache Plugin Extraction (Variant A) → Session-Owned History (Variant B)

**Date:** 2026-02-22
**Status:** Analysis
**Depends on:**
- [Cache Plugin Design](../jaato-cache-plugin-design.md) — Variants A and B
- [Session-Owned History Impact Analysis](session-owned-history-impact-analysis.md) — 5-phase migration

## Executive Summary

This document analyzes the impact of implementing cache plugin extraction using **Variant A** (provider delegates, current architecture) as a first step, then transitioning to **Variant B** (session orchestrates) when session-owned history lands.

**Verdict:** The sequencing is sound. The throwaway cost of the A→B transition is **~20 lines of glue code**. The core CachePlugin protocol, all plugin implementations, budget wiring, and GC-cache coordination survive the transition unchanged. Variant A delivers immediate value (inheritance cleanup, budget-aware breakpoints, extensibility) without blocking or being blocked by the session-owned history migration.

---

## 1. What Variant A Requires

### 1.1 New Files

| Path | Purpose | Estimated LOC |
|------|---------|:---:|
| `shared/plugins/cache/__init__.py` | Package init | 5 |
| `shared/plugins/cache/base.py` | `CachePlugin` protocol/ABC | ~80 |
| `shared/plugins/cache_anthropic/__init__.py` | Package init | 5 |
| `shared/plugins/cache_anthropic/plugin.py` | `AnthropicCachePlugin` implementation | ~250 |
| `shared/plugins/cache_anthropic/tests/` | Unit tests | ~200 |
| `shared/plugins/cache_zhipuai/plugin.py` | ZhipuAI monitoring-only plugin | ~80 |

### 1.2 Modifications to Existing Files

#### `AnthropicProvider` (`shared/plugins/model_provider/anthropic/provider.py`)

**What moves out** (~186 lines removed from provider):

| Code | Lines | Destination |
|------|:---:|-------------|
| `_enable_caching`, `_cache_ttl`, `_cache_history`, `_cache_exclude_recent_turns`, `_cache_min_tokens` | 10 | `AnthropicCachePlugin` config |
| Cache config resolution in `initialize()` | 23 | `AnthropicCachePlugin.initialize()` |
| `_compute_history_cache_breakpoint()` | 39 | `AnthropicCachePlugin._find_policy_boundary()` (budget-aware rewrite) |
| `_should_cache_content()` | 14 | `AnthropicCachePlugin._should_cache_content()` |
| `_get_cache_min_tokens()` | 12 | `AnthropicCachePlugin._get_cache_min_tokens()` |
| `_estimate_tokens()` | 7 | `AnthropicCachePlugin._estimate_tokens()` |
| Cache breakpoint injection in `_build_api_kwargs()` | ~40 | `AnthropicCachePlugin.prepare_request()` |
| Cache breakpoint calls in 5× `send_*` methods | 10 | Removed (plugin handles inside `prepare_request`) |

**What gets added** (~15 lines):

```python
# New attribute
self._cache_plugin: Optional["CachePlugin"] = None

# New method
def set_cache_plugin(self, plugin: "CachePlugin") -> None:
    self._cache_plugin = plugin

# Modified _build_api_kwargs() — delegation branch
if self._cache_plugin:
    cache_result = self._cache_plugin.prepare_request(
        system=..., tools=..., messages=..., budget=...
    )
    kwargs["system"] = cache_result["system"]
    kwargs["tools"] = cache_result["tools"]
    # messages handled via cache_result["cache_breakpoint_index"]
else:
    # Existing logic preserved as fallback
```

**Net effect on provider:** ~170 lines removed. The provider sheds cache policy logic while retaining the API call orchestration. `_build_api_kwargs()` shrinks from 73 lines to ~40 lines.

#### `JaatoSession` (`shared/jaato_session.py`)

**Added** (~30 lines):

```python
# Attribute
self._cache_plugin: Optional[CachePlugin] = None

# Wiring method
def _wire_cache_plugin(self) -> None:
    provider_name = self._provider.name
    cache_plugin = self._runtime.get_cache_plugin(provider_name)
    if cache_plugin:
        cache_plugin.initialize(self._provider_config.extra)
        if hasattr(self._provider, 'set_cache_plugin'):
            self._provider.set_cache_plugin(cache_plugin)
        self._cache_plugin = cache_plugin

# Budget notification (after budget updates)
if self._cache_plugin:
    self._cache_plugin.set_budget(self._instruction_budget)

# GC notification (after GC runs)
if self._cache_plugin:
    self._cache_plugin.on_gc_result(result)
```

#### `ModelProviderPlugin` Protocol (`shared/plugins/model_provider/types.py`)

**Added** (~5 lines): Optional `set_cache_plugin()` method. Since it's duck-typed (checked via `hasattr`), providers that don't support caching don't need to implement it.

#### `PluginRegistry` / `JaatoRuntime`

**Added** (~20 lines): `get_cache_plugin(provider_name)` discovery method that matches cache plugins by their `provider_name` property.

#### `ZhipuAIProvider`, `OllamaProvider`

**Removed** (~3 lines each): Delete `self._enable_caching = False` hack. With cache logic extracted, there's nothing to suppress.

### 1.3 Variant A Test Coverage

| Test | Purpose |
|------|---------|
| `test_prepare_request_breakpoints` | Verify BP1/BP2/BP3 placement on annotated output |
| `test_budget_aware_bp3` | BP3 at PRESERVABLE/EPHEMERAL boundary instead of recency |
| `test_threshold_check` | Content below min tokens doesn't get `cache_control` |
| `test_tool_sorting` | Tools sorted by name for cache stability |
| `test_on_gc_result_tracking` | `on_gc_result()` tracks prefix invalidation |
| `test_extract_cache_usage` | `cache_read_tokens`/`cache_creation_tokens` extracted |
| `test_no_plugin_fallback` | Provider without plugin uses existing logic |
| `test_zhipuai_monitoring` | ZhipuAI plugin extracts `cached_tokens` without annotation |
| Integration: cache hit verification | End-to-end: verify `cache_read_tokens > 0` after warm-up |

### 1.4 Total Variant A Effort

| Category | Lines | Notes |
|----------|:---:|-------|
| New plugin code | ~420 | Protocol + Anthropic + ZhipuAI implementations |
| Provider modification | ~185 | ~170 removed, ~15 added (net -155) |
| Session modification | ~30 | Wiring, budget notification, GC notification |
| Runtime/registry modification | ~20 | Discovery method |
| Tests | ~200 | Unit + integration |
| **Total new/modified** | **~670** | |
| **Net LOC change** | **~+315** | New plugin code outweighs provider reduction |

---

## 2. What the A→B Transition Requires

The A→B transition happens when session-owned history Phase 2 lands (stateless `complete()` on the Anthropic provider). At that point:

### 2.1 Changes Required

| Component | Variant A State | Variant B Change | Lines Changed |
|-----------|----------------|------------------|:---:|
| `provider.set_cache_plugin()` | Protocol method, implemented by `AnthropicProvider` | **Removed** from protocol. Session holds plugin directly. | -8 |
| `AnthropicProvider._cache_plugin` attribute | Set via `set_cache_plugin()` | Received via `complete()` parameters or kept as constructor arg | ~3 |
| `_build_api_kwargs()` delegation | Reads from `self._history`, `self._system_instruction`, `self._tools` | Moves into `complete()`, reads from method parameters | ~15 relocated |
| `JaatoSession._wire_cache_plugin()` | `hasattr` check + `provider.set_cache_plugin(plugin)` | Direct `self._cache_plugin = plugin` (no provider call) | -3 |
| `JaatoSession` GC flow | `reset_session(new_history)` → `provider.create_session(history=...)` | `self._history.replace(new_messages)` (no provider call) | ~5 |
| `CachePlugin` protocol | `prepare_request(system, tools, messages, budget)` | **Unchanged** | 0 |
| `AnthropicCachePlugin` implementation | All breakpoint logic | **Unchanged** | 0 |
| Budget wiring (Option B) | `cache_plugin.set_budget(budget)` | **Unchanged** | 0 |
| GC notification | `cache_plugin.on_gc_result(result)` | **Unchanged** | 0 |
| `ZhipuAICachePlugin` | Monitoring implementation | **Unchanged** | 0 |
| Tests | Plugin unit tests | ~5 tests need caller context updates | ~30 |

### 2.2 What Gets Thrown Away

**Throwaway code from Variant A:**

1. `set_cache_plugin()` on `ModelProviderPlugin` protocol — **~5 lines** (method signature + docstring)
2. `set_cache_plugin()` implementation on `AnthropicProvider` — **~3 lines** (method body)
3. `hasattr(self._provider, 'set_cache_plugin')` check in session wiring — **~3 lines**
4. `self._cache_plugin` attribute on provider — **~2 lines** (init + type annotation)
5. Delegation logic in `_build_api_kwargs()` reading from `self._` fields — **~8 lines** (relocated to `complete()` reading from parameters)

**Total throwaway: ~21 lines of glue code.** This represents ~3% of the Variant A implementation.

### 2.3 What Survives Unchanged

| Component | LOC | % of Variant A |
|-----------|:---:|:-:|
| `CachePlugin` protocol (base.py) | ~80 | 100% |
| `AnthropicCachePlugin` (plugin.py) | ~250 | 100% |
| `ZhipuAICachePlugin` | ~80 | 100% |
| Budget wiring (set_budget) | ~5 | 100% |
| GC notification (on_gc_result) | ~5 | 100% |
| Session-level cache metrics extraction | ~5 | 100% |
| Plugin tests (core logic) | ~170 | 85% |
| **Total surviving** | **~595** | **~97%** |

### 2.4 A→B Transition Effort

| Category | Lines Changed | Time |
|----------|:---:|:---:|
| Remove `set_cache_plugin` from protocol + provider | -13 | 15 min |
| Relocate delegation from `_build_api_kwargs` to `complete()` | ~15 | 30 min |
| Simplify session wiring | -3 | 10 min |
| Update affected tests | ~30 | 30 min |
| Verify cache hits still work (integration test) | 0 | 30 min |
| **Total** | **~60 lines touched** | **~2 hours** |

---

## 3. Dependency and Sequencing Analysis

### 3.1 Independence

Variant A and session-owned history are **fully independent** — neither blocks the other:

```
Timeline option 1: A first, then session-owned history
──────────────────────────────────────────────────────

  Variant A ─────────┐
  (cache extraction)  │
                      │  A→B transition (~2 hours)
                      │  ↓
  SOH Phase 1 ──── Phase 2 ──── Phase 3 ──── Phase 4 ──── Phase 5
  (SessionHistory)  (complete())  (all provs)  (remove     (simplify)
                     ↑                          legacy)
                     │
                     A→B can happen here
                     (when Anthropic gets complete())


Timeline option 2: Session-owned history first
──────────────────────────────────────────────

  SOH Phase 1 ── Phase 2 ── Phase 3 ── Phase 4 ── Phase 5
                                                    │
                                                    └─ Cache plugin extraction
                                                       (use Variant B directly,
                                                        skip Variant A entirely)


Timeline option 3: Parallel development
───────────────────────────────────────

  Variant A ─────────────────────────────┐
                                          │ merge
  SOH Phase 1 ── Phase 2 ───────────────┤
                                          │
                                A→B transition (~2 hours)
```

### 3.2 Optimal Sequencing

**Recommended: Option 1 (A first).** Rationale:

1. **Immediate value.** Variant A delivers inheritance cleanup (ZhipuAI/Ollama no longer need `_enable_caching = False`), budget-aware breakpoints (InstructionBudget GC policies drive BP3 instead of blind `cache_exclude_recent_turns`), and provider extensibility (Google GenAI context caching becomes possible).

2. **Low throwaway.** Only ~21 lines (~3%) of Variant A code is discarded during A→B. The core design — protocol, plugin implementations, budget wiring, GC coordination — is explicitly designed to survive the transition.

3. **A→B is mechanical.** The transition is a ~2-hour task that can be bundled with session-owned history Phase 2 at near-zero incremental cost. No design decisions, no protocol changes, no behavioral changes — just moving the call site from provider-internal state to provider-received parameters.

4. **Session-owned history has a longer timeline.** The 5-phase migration touches all 6 providers and ~2350 LOC. Waiting for it to complete before extracting cache logic would delay cache improvements unnecessarily.

### 3.3 When Exactly to Execute A→B

The A→B transition becomes possible at **session-owned history Phase 2** (when `AnthropicProvider` gains `complete()`). It becomes mandatory at **Phase 4** (when `set_cache_plugin()` would be removed along with other legacy provider methods).

**Recommended trigger:** Bundle A→B with Phase 2 for Anthropic. The `complete()` method implementation naturally includes relocating the cache plugin delegation from `_build_api_kwargs()` to `complete()`. Doing it as part of the same PR keeps the transition atomic.

---

## 4. Risk Analysis

### 4.1 Variant A Risks

| Risk | Severity | Likelihood | Mitigation |
|------|:---:|:---:|------------|
| Cache hit regression after extraction | HIGH | LOW | Integration test: verify `cache_read_tokens > 0` before/after. The `prepare_request()` output is functionally identical to current `_build_api_kwargs()` output. |
| Protocol pollution (`set_cache_plugin` on all providers) | LOW | N/A | Duck-typed via `hasattr` — only Anthropic implements it. Other providers are unaffected. Throwaway is ~5 lines. |
| ZhipuAI inheritance breakage | MEDIUM | LOW | Extraction *fixes* the inheritance problem. Removing cache members from parent simplifies child. |
| Budget-aware BP3 differs from recency-based BP3 | MEDIUM | MEDIUM | The new BP3 placement may differ from the old one. This is intentional — budget-aware placement should be *better* — but could cause short-term cache miss increases while the new breakpoint warms up. Monitor `cache_read_tokens` during rollout. |

### 4.2 A→B Transition Risks

| Risk | Severity | Likelihood | Mitigation |
|------|:---:|:---:|------------|
| Cache breaks during `complete()` migration | MEDIUM | LOW | `prepare_request()` signature is parameter-based — works identically whether called from `_build_api_kwargs()` or `complete()`. Same inputs, same outputs. |
| GC-cache coordination breaks | LOW | VERY LOW | `on_gc_result()` is session-level in both variants. The GC notification path doesn't touch the provider at all. |
| Forgotten `set_cache_plugin` removal | LOW | LOW | Phase 4 removes all legacy provider methods. Any surviving `set_cache_plugin` references would cause a test failure. |
| Parallel development conflicts | MEDIUM | MEDIUM | If Variant A and session-owned history are developed in parallel (Timeline option 3), merge conflicts in `_build_api_kwargs()` are likely. Mitigated by: the provider-side changes are small and well-isolated. |

### 4.3 Risk Comparison: A-first vs. Skip-to-B

| Approach | Total Risk | Reasoning |
|----------|:---:|-----------|
| **A first, then B** | LOW | Two small, well-defined changes. Each is independently testable. |
| **Skip to B directly** | MEDIUM | Must implement cache plugin + session-owned history + cache wiring simultaneously. Larger blast radius, harder to isolate regressions. |
| **Parallel A + SOH** | MEDIUM | Merge conflicts in provider code. Manageable but requires coordination. |

---

## 5. Detailed Data Flow Comparison

### 5.1 Variant A: Per-Turn Flow

```
JaatoSession                     AnthropicProvider                 AnthropicCachePlugin
────────────                     ──────────────────                ────────────────────
send_message(prompt)
  │
  ├─ _run_chat_loop()
  │    │
  │    └──────────────────────→ send_message(text)
  │                               │
  │                               ├─ self._history.append(user_msg)
  │                               ├─ validate_tool_use_pairing(self._history)
  │                               │
  │                               ├─ _build_api_kwargs()
  │                               │    │
  │                               │    ├─ convert system/tools to API format
  │                               │    │
  │                               │    └─ self._cache_plugin.prepare_request(
  │                               │         api_system, api_tools,          ───→ compute BP1, BP2, BP3
  │                               │         api_messages, budget)                using InstructionBudget
  │                               │         → returns annotated dicts       ←── GC policies
  │                               │
  │                               ├─ messages_to_anthropic(self._history,
  │                               │    cache_breakpoint_index=result["bp3_index"])
  │                               │
  │                               ├─ client.messages.create(...)
  │                               ├─ self._add_response_to_history(response)
  │                               │
  │    ◄────────────────────── return ProviderResponse
  │
  ├─ cache_plugin.extract_cache_usage(response.usage) ──────────→ track hit rate
  │
  ├─ _maybe_collect_after_turn()
  │    ├─ gc_plugin.collect(history, ..., budget)
  │    │    → (new_history, GCResult)
  │    ├─ self.reset_session(new_history)
  │    │    → provider.create_session(system, tools, history=new_history)
  │    └─ cache_plugin.on_gc_result(result) ─────────────────────→ track invalidation
  │
  └─ return response
```

### 5.2 Variant B: Per-Turn Flow (After Session-Owned History Phase 2+)

```
JaatoSession                     AnthropicProvider                 AnthropicCachePlugin
────────────                     ──────────────────                ────────────────────
send_message(prompt)
  │
  ├─ self._messages.append(user_msg)
  ├─ validate_tool_use_pairing(self._messages)  ← moved to session layer
  │
  ├─ _run_chat_loop()
  │    │
  │    └──────────────────────→ complete(self._messages, system, tools, ...)
  │                               │
  │                               ├─ convert messages/system/tools to API format
  │                               │
  │                               ├─ self._cache_plugin.prepare_request(   (or received
  │                               │    api_system, api_tools,               from session)
  │                               │    api_messages, budget)          ───→ same computation
  │                               │    → returns annotated dicts      ←── same result
  │                               │
  │                               ├─ client.messages.create(...)
  │                               │
  │    ◄────────────────────── return ProviderResponse
  │
  ├─ self._messages.append(response_msg)  ← session appends, not provider
  │
  ├─ cache_plugin.extract_cache_usage(response.usage) ──────────→ unchanged
  │
  ├─ _maybe_collect_after_turn()
  │    ├─ gc_plugin.collect(self._messages, ..., budget)
  │    │    → (new_messages, GCResult)
  │    ├─ self._history.replace(new_messages)  ← no provider.create_session()
  │    └─ cache_plugin.on_gc_result(result) ─────────────────────→ unchanged
  │
  └─ return response
```

### 5.3 What Changes Between the Two Flows

| Step | Variant A | Variant B | Plugin Impact |
|------|-----------|-----------|:---:|
| User message append | Provider (`self._history.append`) | Session (`self._messages.append`) | None |
| Tool-use validation | Provider (inside `send_message`) | Session (before `complete()`) | None |
| Cache plugin call site | Provider `_build_api_kwargs()` | Provider `complete()` | None — same `prepare_request()` call |
| Data source for `prepare_request` | Provider internal state (`self._history`) | Provider method parameter (`messages`) | None — same data, different source |
| Response append | Provider (`_add_response_to_history`) | Session (`self._messages.append`) | None |
| GC application | `reset_session()` → `create_session()` | `self._history.replace()` | None — `on_gc_result()` is unchanged |
| Cache plugin wiring | `provider.set_cache_plugin(plugin)` | `self._cache_plugin = plugin` | None — plugin is the same object |

**The cache plugin is called with identical inputs and produces identical outputs in both variants.** The only thing that changes is the orchestration layer above it.

---

## 6. Provider Protocol Impact

### 6.1 Variant A: Protocol Addition

```python
# Added to ModelProviderPlugin (duck-typed, optional)
def set_cache_plugin(self, plugin: "CachePlugin") -> None: ...
```

**Providers affected:** Only `AnthropicProvider` implements it. `ZhipuAIProvider` and `OllamaProvider` inherit it (but their matched cache plugins do different things). `GoogleGenAIProvider`, `AntigravityProvider`, `ClaudeCLIProvider`, `GitHubModelsProvider` — unaffected.

### 6.2 Variant B: Protocol Removal

`set_cache_plugin()` is removed from the protocol. The cache plugin slot moves to the session.

**Impact on providers:** Remove `set_cache_plugin()` method and `self._cache_plugin` attribute from `AnthropicProvider`. ~8 lines.

### 6.3 Net Protocol Churn

```
Variant A:  +set_cache_plugin()  (optional, duck-typed)
   ↓
Variant B:  -set_cache_plugin()

Net after both:  No protocol change relative to today.
```

The protocol returns to its original state. The temporary addition is small enough (one optional method) that the churn is negligible.

---

## 7. InstructionBudget Integration (Unchanged Across Variants)

The `InstructionBudget` integration is the primary motivator for cache plugin extraction and is **completely unaffected** by the A→B transition:

```python
# CachePlugin.prepare_request() — same in both variants
def prepare_request(self, system, tools, messages, budget=None):
    if budget:
        # Use GC policies for BP3 placement
        bp3_index = self._find_policy_boundary(budget)
    else:
        # Fallback to recency-based (current behavior)
        bp3_index = self._find_recency_boundary(messages)
    ...
```

The `_find_policy_boundary()` method reads `InstructionBudget` GC policy assignments to place BP3 at the PRESERVABLE/EPHEMERAL boundary. This logic is orthogonal to who owns history or how the cache plugin is wired.

**GC-cache coordination via `on_gc_result()`** is also unchanged — it tracks whether GC removed PRESERVABLE content (which may invalidate the cached prefix) and adjusts breakpoint strategy accordingly.

---

## 8. Summary

### Decision Matrix

| Factor | A-first + B-later | B-only (skip A) | A-only (no B) |
|--------|:---:|:---:|:---:|
| Time to first value | **Fast** (A is independent) | Slow (blocked by SOH Phases 1-2) | **Fast** |
| Total throwaway | ~21 lines (3%) | 0 | 0 |
| Protocol churn | +1 method, then -1 | 0 | +1 method (permanent) |
| Implementation complexity | Two small changes | One larger change | One small change |
| Risk of cache regression | Low (two isolated steps) | Medium (combined blast radius) | Low |
| Final architecture quality | Clean (session orchestrates) | Clean | Less clean (provider holds cache plugin permanently) |
| Blocks session-owned history? | No | N/A | No |

### Recommendation

**Implement Variant A now.** The ~21 lines of throwaway code (~3% of total) is a trivial cost for:

- Immediate inheritance cleanup (ZhipuAI/Ollama)
- Budget-aware breakpoint placement (up to 90% cost reduction from better BP3)
- Provider extensibility (Google GenAI context caching becomes possible)
- Clean separation of concerns (cache policy vs. API call orchestration)

Bundle the A→B transition with session-owned history Phase 2 when Anthropic gets `complete()`. The transition is ~2 hours of mechanical work with no design decisions required.
