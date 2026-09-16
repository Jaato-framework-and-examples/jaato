/** Context-window and token usage — the TUI's budget panel + `context` command, as a rail section. */
import { describeToolCalls } from "@/protocol/turnStats";
import { useJaato } from "@/store/store";

function fmt(n: number | null | undefined): string {
  if (n == null) return "–";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + "M";
  if (n >= 10_000) return (n / 1000).toFixed(1) + "k";
  return String(n);
}

export function BudgetPanel({ agentId }: { agentId: string }) {
  const ctx = useJaato((s) => s.context[agentId]);
  if (!ctx) return <div className="px-3.5 py-3 text-xs text-text-muted italic">No usage reported yet.</div>;
  const pct = ctx.percentUsed ?? (ctx.contextLimit && ctx.usage.total_tokens ? (ctx.usage.total_tokens / ctx.contextLimit) * 100 : null);
  const u = ctx.usage;
  const rows: [string, string][] = [
    ["prompt", fmt(u.prompt_tokens)],
    ["output", fmt(u.output_tokens)],
    ["cache read", fmt(u.cache_read_tokens)],
    ["cache write", fmt(u.cache_creation_tokens)],
    ["reasoning", fmt(u.reasoning_tokens ?? u.thinking_tokens)],
    ["total / limit", `${fmt(u.total_tokens)} / ${fmt(ctx.contextLimit)}`],
    ["remaining", fmt(ctx.tokensRemaining)],
    ["turns", fmt(ctx.turns)],
  ];
  if (u.cost_usd != null) rows.push(["cost", `$${Number(u.cost_usd).toFixed(4)}`]);
  return (
    <div className="px-3.5 py-3 text-[13px]">
      <div className="flex items-baseline justify-between mb-1.5"><span className="display text-[16px]">Context</span><span className="font-mono text-[11px] text-text-muted">{pct != null ? `${pct.toFixed(1)}%` : ""}</span></div>
      <div className="bar mb-3">
        <span className={pct != null && pct > 80 ? "bg-error" : pct != null && pct > 60 ? "bg-warning" : ""} style={{ width: `${Math.min(100, pct ?? 0)}%` }} />
      </div>
      <dl className="kv m-0">
        {rows.map(([k, v]) => <div key={k} className="contents"><dt>{k}</dt><dd className="m-0">{v}</dd></div>)}
      </dl>
      {ctx.lastTurn && (
        <div className="mt-3 text-xs text-text-muted">
          Last turn{ctx.lastTurn.turnNumber != null ? ` #${ctx.lastTurn.turnNumber}` : ""}: {ctx.lastTurn.durationSeconds != null ? `${ctx.lastTurn.durationSeconds.toFixed(1)}s` : ""}{ctx.lastTurn.toolCalls ? ` · ${describeToolCalls(ctx.lastTurn.toolCalls)}` : ""}{ctx.lastTurn.finishReason ? ` · ${ctx.lastTurn.finishReason}` : ""}
        </div>
      )}
    </div>
  );
}
