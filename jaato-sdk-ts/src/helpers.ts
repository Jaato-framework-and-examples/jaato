// Convenience helpers over the wire-protocol event types.
//
// Mirror of jaato-sdk/jaato_sdk/helpers.py (Python).  Both are
// hand-written — they aren't part of the events.py → events.ts
// codegen surface, but they share the same contract so a TS client
// and a Python client computing the same metric get identical
// numbers.

import type { TurnCompletedEvent, TurnProgressEvent } from "./events.js";

/**
 * Wire-protocol events that carry prompt-cache token counts.
 */
export type CacheReportingEvent = TurnCompletedEvent | TurnProgressEvent;

/**
 * Cache hit % for a turn — share of paid input tokens served from cache.
 *
 * The denominator is `cache_read_tokens + prompt_tokens` (the
 * "non-creation" input).  Anthropic's `input_tokens` /
 * pydantic's `prompt_tokens` is the *new uncached* input — it
 * excludes both `cache_read_tokens` and `cache_creation_tokens`,
 * so the natural hit-rate denominator is "what was the model
 * paying for this turn that *could* have been a cache hit", which
 * is `cache_read` (the actual hit) plus `prompt` (the new input
 * that wasn't a hit).  `cache_creation_tokens` is excluded because
 * it represents *new* content being written to cache — it's not a
 * hit on prior content, just future infrastructure.
 *
 * @returns
 * `null` when the provider does not report cache stats (i.e.
 * `cache_read_tokens` is null/undefined).  Treating "no cache info"
 * as "0% hit" would be misleading — clients should handle the null
 * case explicitly (e.g. omit the cache-hit line from a status bar,
 * rather than show "0%").
 *
 * `0.0` when `cache_read_tokens` is reported as 0 (the provider
 * supports caching but this turn had no hits) — this is a real
 * measurement, distinct from "no support".
 *
 * Otherwise the percentage in the range `[0.0, 100.0]`.
 */
export function computeCacheHitPercent(event: CacheReportingEvent): number | null {
  const usage = event.usage;
  if (usage == null || usage.cache_read_tokens == null) {
    return null;
  }
  // prompt_tokens has a default of 0 on the Python side, so undefined
  // here is the same as 0 — treat them uniformly.
  const promptTokens = usage.prompt_tokens ?? 0;
  const total = usage.cache_read_tokens + promptTokens;
  if (total === 0) {
    return 0.0;
  }
  return (usage.cache_read_tokens / total) * 100.0;
}
