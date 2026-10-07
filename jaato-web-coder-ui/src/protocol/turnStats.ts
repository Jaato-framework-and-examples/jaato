/**
 * What ``TurnCompletedEvent.function_calls`` actually is.
 *
 * On the wire it is a LIST of per-call records the session accumulates
 * while the turn runs -- ``{name, start_time, end_time, duration_seconds}``
 * (``JaatoSession`` appends one per executed call) -- not a count.  The
 * budget panel read it as a number and interpolated the array into its
 * "Last turn" line, which the browser rendered as
 * ``[object Object],[object Object]``; the mock daemon sent an integer,
 * so the e2e suite never saw the real shape.  Same pattern as the
 * clarification and permission cards: a client vocabulary the daemon
 * never spoke, certified by a mock written to the client.
 *
 * ``summarizeToolCalls`` reduces the list to what the panel wants -- how
 * many, and which tools, grouped and counted -- and tolerates a bare
 * number so an older or synthetic producer still yields a count.
 */

export interface ToolCallSummary {
  /** How many calls the turn made. */
  count: number;
  /** ``name`` → how many times it was called, in first-seen order. */
  byName: Array<[string, number]>;
}

export function summarizeToolCalls(raw: unknown): ToolCallSummary | null {
  if (raw == null) return null;
  if (typeof raw === "number") return Number.isFinite(raw) ? { count: raw, byName: [] } : null;
  if (!Array.isArray(raw)) return null;
  const counts = new Map<string, number>();
  for (const entry of raw) {
    const name = entry && typeof entry === "object" && typeof (entry as { name?: unknown }).name === "string"
      ? (entry as { name: string }).name
      : typeof entry === "string" ? entry : "?";
    counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  return { count: raw.length, byName: [...counts.entries()] };
}

/** ``3 tool calls (readFile ×2, cli_based_tool)``; ``no tool calls`` for an empty turn. */
export function describeToolCalls(summary: ToolCallSummary): string {
  if (summary.count === 0) return "no tool calls";
  const head = `${summary.count} tool call${summary.count === 1 ? "" : "s"}`;
  if (summary.byName.length === 0) return head;
  const names = summary.byName.map(([n, k]) => (k > 1 ? `${n} ×${k}` : n)).join(", ");
  return `${head} (${names})`;
}
