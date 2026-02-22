# Cache Plugin: Pending Implementation Plan

**Date:** 2026-02-22
**Branch:** `claude/plan-cache-plugin-impl-6Zrnd`
**Context:** [Cache Plugin Design](../jaato-cache-plugin-design.md) | [Sequencing Impact Analysis](cache-plugin-sequencing-impact-analysis.md)

## Current Status

**Variant A is fully implemented.** The `CachePlugin` protocol, `AnthropicCachePlugin`, `ZhipuAICachePlugin`, session wiring, budget forwarding, GC notification, and full cache metrics pipeline (provider → session → server → TUI) are all live with 57 passing tests.

## Pending Work Items

### Tier 1: Actionable Now (No Blockers)

#### 1.1 — Create Session-Owned History Impact Analysis Document

The design doc references `docs/design/session-owned-history-impact-analysis.md` but this file **does not exist**. It's the prerequisite design document for the 5-phase session-owned history migration, which in turn unblocks Variant B.

**Scope:** Design document covering:
- Phase 1: Introduce `SessionHistory` wrapper (session holds canonical copy, still syncs to provider)
- Phase 2: Add stateless `complete(messages, system, tools, ...) -> ProviderResponse` to providers
- Phase 3: Migrate all 6 providers to `complete()`
- Phase 4: Remove legacy `create_session()`, `get_history()`, `send_message()` from protocol
- Phase 5: Simplify session internals (GC operates on `self._messages` directly)

**Impact:** ~2350 LOC across all 6 providers (per design doc estimates). This is a design-only deliverable.

#### 1.2 — Legacy Fallback Removal from AnthropicProvider

The legacy cache code path (`_apply_legacy_cache_annotations()`, `_enable_caching`, `_cache_history`, `_cache_exclude_recent_turns`, `_cache_min_tokens`) is retained in `AnthropicProvider` as a fallback when no `CachePlugin` is attached. Once all deployments use plugin-based caching, this fallback can be removed.

**Files to modify:**

| File | Change |
|------|--------|
| `anthropic/provider.py` | Remove `_apply_legacy_cache_annotations()`, `_compute_history_cache_breakpoint()` (legacy path), `_should_cache_content()`, `_get_cache_min_tokens()`, `_estimate_tokens()`. Remove legacy cache config fields from `initialize()`. Simplify `_build_api_kwargs()` to always delegate to plugin. |
| `zhipuai/provider.py` | Remove `_enable_caching = False` (no legacy path to suppress). |
| `ollama/provider.py` | Remove `_enable_caching = False` (no legacy path to suppress). |
| `anthropic/env.py` | Remove `resolve_enable_caching()` if only used by legacy path. |
| Tests | Update any tests that rely on the legacy path. |

**Prerequisites:**
- Confidence that all deployments use cache plugin (not legacy path)
- Or: make cache plugin the only path (require plugin attachment for caching)

**Estimated scope:** ~150 lines removed from provider, ~10 lines simplified in session.

#### 1.3 — Extended TTL Support for Anthropic

The `AnthropicCachePlugin` accepts `cache_ttl` config (`"5m"` or `"1h"`) but currently only emits `{"type": "ephemeral"}` cache_control. For long agentic sessions (10+ minutes with tool-heavy workflows), the 5-minute TTL can expire between steps.

**Change:** When `cache_ttl == "1h"`, emit `{"type": "ephemeral", "ttl": "1h"}` instead of plain `{"type": "ephemeral"}`.

**Files:** `cache_anthropic/plugin.py` — modify `_make_cache_control()` or equivalent.

**Estimated scope:** ~5 lines.

#### 1.4 — Google GenAI Context Caching Plugin (New) — **COMPLETED**

**Status:** Implemented as monitoring-only plugin (Pattern 2, like ZhipuAI) with 31 passing tests.

The `GoogleGenAICachePlugin` tracks cache hit metrics from `cached_content_token_count` (already extracted by `GoogleGenAIProvider` into `TokenUsage.cache_read_tokens`) and monitors GC invalidations. It does not modify requests since Google GenAI uses implicit prefix caching.

**Files created:**

| Path | Purpose |
|------|---------|
| `shared/plugins/cache_google_genai/__init__.py` | Package init, `PLUGIN_KIND = "cache"`, `create_plugin()` |
| `shared/plugins/cache_google_genai/plugin.py` | `GoogleGenAICachePlugin` — monitoring-only implementation |
| `shared/plugins/cache_google_genai/tests/test_plugin.py` | 31 unit tests (protocol compliance, passthrough, metrics, GC) |

**Entry point:** Added to `pyproject.toml` under `[project.entry-points."jaato.cache_plugins"]`.

**Future enhancement:** Active caching via the `google.genai.caching` API (create/manage `CachedContent` objects for explicit prefix pinning). This would require extending the `prepare_request()` return dict with a `cached_content` key and managing `CachedContent` lifecycle (create, TTL, invalidation).

---

### Tier 2: Blocked on Session-Owned History Phase 2

#### 2.1 — A→B Transition (Cache Plugin Variant B)

When session-owned history Phase 2 lands (Anthropic gets stateless `complete()`), the cache plugin wiring transitions from provider-delegates to session-orchestrates.

**Changes required:**

| Item | Lines | Description |
|------|:---:|-------------|
| Remove `set_cache_plugin()` from protocol + `AnthropicProvider` | ~13 | Drop optional method from `ModelProviderPlugin` and implementation |
| Relocate delegation from `_build_api_kwargs` to `complete()` | ~15 | `complete()` calls `prepare_request()` using received parameters |
| Simplify session wiring | ~3 | Remove `hasattr(self._provider, 'set_cache_plugin')` check |
| Remove `_enable_caching = False` from ZhipuAI/Ollama | ~6 | No legacy path to suppress |
| Update affected tests | ~30 | Test caller context changes |
| Integration test: verify cache hits | — | End-to-end `cache_read_tokens > 0` check |

**Blocking dependency:** Session-owned history Phase 2 (stateless `complete()` on `AnthropicProvider`).

**Estimated scope:** ~60 lines touched, mechanical work.

---

### Tier 3: Blocked on Session-Owned History Completion (Phases 1-5)

#### 3.1 — Session-Owned History Migration

The full 5-phase migration that moves history ownership from providers to the session:

| Phase | Description | Scope |
|-------|-------------|-------|
| 1 | `SessionHistory` wrapper — session holds canonical copy | Foundation |
| 2 | Stateless `complete()` on providers | **Unblocks A→B** |
| 3 | Migrate all 6 providers to `complete()` | Bulk migration |
| 4 | Remove legacy protocol methods | Cleanup |
| 5 | Simplify session internals | GC operates on `self._messages` directly |

**Total estimated scope:** ~2350 LOC across all providers.

**This is a standalone initiative** — the cache plugin Variant A works independently and delivers value today. The session-owned history migration is motivated by broader architectural goals (provider statelessness, simplified GC flow, cleaner subagent spawning) beyond just cache plugin improvement.

---

## Recommended Sequencing

```
COMPLETED
├── 1.2 Legacy fallback removal ✓
├── 1.3 Extended TTL support ✓
└── 1.4 Google GenAI cache plugin (monitoring-only) ✓

REMAINING (no blockers)
└── 1.1 Session-Owned History design doc (prerequisite for Phases 1-5)

AFTER Session-Owned History Phase 2
└── 2.1 A→B transition (~60 lines, mechanical)

AFTER Session-Owned History Phase 4+
└── Full legacy cleanup (any remaining provider-level cache code)
```

## Remaining Decision

**1.1 (SOH design doc):** The only remaining Tier 1 item. Unblocks the entire Tier 2/3 pipeline but is design work, not implementation.
