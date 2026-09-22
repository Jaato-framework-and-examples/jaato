/**
 * What the Instructions panel says about GC (#1190), as plain strings.
 *
 * Three facts, and the wording keeps them apart because they are easy to
 * confuse: the POLICY in force (``gc.config``: strategy, when it triggers,
 * how far it collects), the last PASS (``gc`` phase ``completed``: when, and
 * what it freed), and the per-layer GC icons, which describe what each layer
 * ALLOWS to be collected, not what recently was.
 */
import type { GcState } from "@/store/types";

export function formatTokens(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + "M";
  if (n >= 10_000) return (n / 1000).toFixed(1) + "k";
  if (n >= 1_000) return (n / 1000).toFixed(1) + "k";
  return String(n);
}

/** "just now", "12 min ago", "3 h ago", "2 d ago". */
export function formatAgo(at: number, now: number): string {
  const s = Math.max(0, Math.round((now - at) / 1000));
  if (s < 45) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 48) return `${h} h ago`;
  return `${Math.round(h / 24)} d ago`;
}

/**
 * The policy in one line.  ``null`` when the daemon has not said -- which is
 * not the same as a session with NO strategy, which says so, because a
 * session that never collects is the state an operator most needs to see.
 */
export function describeGcPolicy(config: GcState["config"]): string | null {
  if (!config) return null;
  if (!config.strategy) return "GC: none configured — this session's history is never collected";
  const parts = [`GC: ${config.strategy}`];
  if (config.continuous) {
    parts.push(config.targetPercent != null ? `after every turn above ${config.targetPercent}%` : "after every turn");
  } else {
    if (config.threshold != null) parts.push(`runs at ${config.threshold}%`);
    if (config.targetPercent != null) parts.push(`down to ${config.targetPercent}%`);
  }
  return parts.join(" · ");
}

/** The last pass in one line, or what is known instead of one. */
export function describeLastPass(gc: GcState | undefined, now: number): string {
  if (gc?.running) return "collecting…";
  const p = gc?.lastPass;
  if (!p) return "no GC pass reported yet";
  const when = formatAgo(p.at, now);
  if (!p.success) return `last GC ${when} failed${p.error ? `: ${p.error}` : ""}`;
  return p.tokensFreed != null ? `last GC ${when} · freed ${formatTokens(p.tokensFreed)} tokens` : `last GC ${when}`;
}

/** The detail a hover shows: absolute time, trigger, before -> after. */
export function lastPassTitle(p: NonNullable<GcState["lastPass"]>): string {
  const bits = [new Date(p.at).toLocaleString()];
  if (p.trigger) bits.push(`trigger: ${p.trigger}`);
  if (p.strategy) bits.push(`strategy: ${p.strategy}`);
  if (p.tokensBefore != null && p.tokensAfter != null) bits.push(`${formatTokens(p.tokensBefore)} → ${formatTokens(p.tokensAfter)} tokens`);
  return bits.join(" · ");
}

/** One line per GC policy glyph, for its tooltip: what the layer ALLOWS. */
export const POLICY_EXPLAINED: Record<string, string> = {
  locked: "Pinned: GC never reclaims this layer",
  preservable: "Kept unless context pressure is severe",
  partial: "Partly eligible: GC may reclaim some of this layer",
  ephemeral: "Freely collected: GC may reclaim all of this layer",
  conditional: "Collected only when a condition holds",
};
