/** Context-window and token usage — the TUI's budget panel + `context` command. */
import { useJaato } from "@/store/store";

function fmt(n: number | null | undefined): string {
  if (n == null) return "–";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + "M";
  if (n >= 10_000) return (n / 1000).toFixed(1) + "k";
  return String(n);
}

export function BudgetPanel({ agentId }: { agentId: string }) {
  const ctx = useJaato((s) => s.context[agentId]);
  if (!ctx) return <div className="p-3 text-xs text-text-muted italic">No usage reported yet.</div>;
  const pct = ctx.percentUsed ?? (ctx.contextLimit && ctx.usage.total_tokens ? (ctx.usage.total_tokens / ctx.contextLimit) * 100 : null);
  const u = ctx.usage;
  const rows: [string, string][] = [
    ["prompt", fmt(u.prompt_tokens)],
    ["output", fmt(u.output_tokens)],
    ["cache read", fmt(u.cache_read_tokens)],
    ["cache write", fmt(u.cache_creation_tokens)],
    ["reasoning", fmt(u.reasoning_tokens ?? u.thinking_tokens)],
    ["total", fmt(u.total_tokens)],
    ["limit", fmt(ctx.contextLimit)],
    ["remaining", fmt(ctx.tokensRemaining)],
    ["turns", fmt(ctx.turns)],
  ];
  if (u.cost_usd != null) rows.push(["cost", `$${Number(u.cost_usd).toFixed(4)}`]);
  return (
    <div className="p-3 text-[13px]">
      <div className="flex items-baseline justify-between mb-1"><span className="font-semibold">Context</span><span className="text-xs text-text-muted">{pct != null ? `${pct.toFixed(1)}%` : ""}</span></div>
      <div className="h-1.5 rounded bg-surface mb-3 overflow-hidden">
        <div className={`h-full ${pct != null && pct > 80 ? "bg-error" : pct != null && pct > 60 ? "bg-warning" : "bg-primary"}`} style={{ width: `${Math.min(100, pct ?? 0)}%` }} />
      </div>
      <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-0.5 font-mono text-xs">
        {rows.map(([k, v]) => <div key={k} className="contents"><dt className="text-text-muted">{k}</dt><dd className="text-right">{v}</dd></div>)}
      </dl>
      {ctx.lastTurn && (
        <div className="mt-3 text-xs text-text-muted">
          Last turn{ctx.lastTurn.turnNumber != null ? ` #${ctx.lastTurn.turnNumber}` : ""}: {ctx.lastTurn.durationSeconds != null ? `${ctx.lastTurn.durationSeconds.toFixed(1)}s` : ""}{ctx.lastTurn.functionCalls != null ? ` · ${ctx.lastTurn.functionCalls} tool calls` : ""}{ctx.lastTurn.finishReason ? ` · ${ctx.lastTurn.finishReason}` : ""}
        </div>
      )}
    </div>
  );
}
