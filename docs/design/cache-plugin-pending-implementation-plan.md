# Cache Plugin: Pending Implementation Plan

**Date:** 2026-02-22
**Branch:** `claude/plan-cache-plugin-impl-6Zrnd`
**Context:** [Cache Plugin Design](../jaato-cache-plugin-design.md) | [Sequencing Impact Analysis](cache-plugin-sequencing-impact-analysis.md)

## Current Status

**Variant A is fully implemented.** The `CachePlugin` protocol, `AnthropicCachePlugin`, `ZhipuAICachePlugin`, session wiring, budget forwarding, GC notification, and full cache metrics pipeline (provider → session → server → TUI) are all live with 57 passing tests.

## Pending Work Items

### Tier 1: Actionable Now (No Blockers)

#### 1.1 — Create Session-Owned History Impact Analysis Document — **COMPLETED**

**Status:** Created on 2026-02-22. See [session-owned-history-impact-analysis.md](session-owned-history-impact-analysis.md).

The document covers:
- 5-phase migration plan (SessionHistory wrapper → stateless `complete()` → legacy removal → simplification)
- Per-provider impact analysis across all 7 providers (~1455 LOC affected)
- Unified `complete()` method design (combines batch + streaming, replaces 7 protocol methods)
- Impact on dependent systems (cache plugin A→B, GC, subagents, session persistence, telemetry)
- Risk analysis (GoogleGenAI SDK bypass, Claude CLI delegated mode, dual code paths)
- Dependency graph and sequencing

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

**Status:** Implemented with active `CachedContent` support — 61 passing tests.

The `GoogleGenAICachePlugin` supports two modes:

1. **Active caching** (`enable_caching: true`): Creates a server-side `CachedContent` object containing system instruction + tool definitions via `client.caches.create()`. Returns `cached_content` name in `prepare_request()` result so the provider references the cached prefix in `GenerateContentConfig`. Content changes are detected via SHA-256 hash; stale caches are automatically deleted and recreated. Minimum ~32K token threshold prevents wasteful caching of small prompts.

2. **Monitoring-only** (default): Tracks `cache_read_tokens` metrics from `cached_content_token_count` and GC invalidation counts without modifying requests.

**Provider integration:** `GoogleGenAIProvider.set_cache_plugin()` passes the `genai.Client` to the plugin. `_get_cached_content_config()` builds a `GenerateContentConfig` with `cached_content` reference, injected into all 5 send methods.

**Files created/modified:**

| Path | Purpose |
|------|---------|
| `shared/plugins/cache_google_genai/__init__.py` | Package init, `PLUGIN_KIND = "cache"`, `create_plugin()` |
| `shared/plugins/cache_google_genai/plugin.py` | `GoogleGenAICachePlugin` — active + monitoring modes |
| `shared/plugins/cache_google_genai/tests/test_plugin.py` | 61 unit tests |
| `model_provider/google_genai/provider.py` | `set_cache_plugin()`, `_get_cached_content_config()`, 5 send method integrations |

**Entry point:** Added to `pyproject.toml` under `[project.entry-points."jaato.cache_plugins"]`.

**Configuration** (via `ProviderConfig.extra`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_caching` | bool | `False` | Enable explicit CachedContent creation |
| `cache_ttl` | str | `"3600s"` | TTL for cached content (e.g. `"1800s"`) |

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
├── 1.1 Session-Owned History design doc ✓ (2026-02-22)
├── 1.2 Legacy fallback removal ✓
├── 1.3 Extended TTL support ✓
└── 1.4 Google GenAI cache plugin ✓

NEXT: Session-Owned History Phase 1 (SessionHistory wrapper, no blockers)
└── Phase 2 (Anthropic complete(), unblocks A→B)
    └── 2.1 A→B transition (~60 lines, mechanical)

AFTER Session-Owned History Phase 4+
└── Full legacy cleanup (any remaining provider-level cache code)
```

## Status

**All Tier 1 items are complete.** The design document for session-owned history ([session-owned-history-impact-analysis.md](session-owned-history-impact-analysis.md)) was created on 2026-02-22, covering the full 5-phase migration plan, per-provider impact analysis, dependent system impacts, and risk analysis.

**Next actionable work:** Session-Owned History Phase 1 (`SessionHistory` wrapper), which has no blockers and is a low-risk behavioral no-op that establishes the foundation for subsequent phases.
