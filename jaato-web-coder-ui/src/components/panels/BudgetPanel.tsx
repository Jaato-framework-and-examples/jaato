/**
 * The rail's Budget section: the TUI's budget panel (Ctrl+B) over its
 * `context` command, in that order.
 *
 * These answer two different questions and the TUI keeps them apart, which
 * is what this section did not: it rendered the context readout alone under
 * a heading that said Budget.  A budget is what the window is spent ON --
 * by instruction source, with each layer's GC policy -- and it was on the
 * wire the whole time as `InstructionBudgetEvent` and read by nobody.
 */
import { describeToolCalls } from "@/protocol/turnStats";
import type { BudgetEntry, BudgetState } from "@/store/types";
import { useJaato } from "@/store/store";

function fmt(n: number | null | undefined): string {
  if (n == null) return "–";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + "M";
  if (n >= 10_000) return (n / 1000).toFixed(1) + "k";
  return String(n);
}

/**
 * The source layers, in the order the TUI lists them.
 *
 * A source the daemon reports and this table does not name is still shown,
 * after these -- the snapshot is the source of truth for WHAT exists, and a
 * hardcoded list deciding that would hide a layer added later.
 */
const SOURCE_ORDER = ["system", "plugin", "enrichment", "conversation", "thinking"];
const SOURCE_NAMES: Record<string, string> = {
  system: "System",
  plugin: "Plugin",
  enrichment: "Enrichment",
  conversation: "Conversation",
  thinking: "Thinking",
};

/** The daemon's own glyph, falling back to one per policy — never invented per row. */
const POLICY_GLYPH: Record<string, string> = {
  locked: "\u{1F512}",
  preservable: "◑",
  partial: "◐",
  ephemeral: "○",
  conditional: "◌",
};

/** `locked` is the one a reader acts on differently, so it is the one that is coloured. */
function policyClass(policy: string | undefined): string {
  if (policy === "locked") return "text-warning";
  if (policy === "ephemeral") return "text-text-muted";
  return "";
}

function sourceOrder(entries: Record<string, BudgetEntry>): [string, BudgetEntry][] {
  const known = SOURCE_ORDER.filter((k) => entries[k] != null);
  const rest = Object.keys(entries).filter((k) => !SOURCE_ORDER.includes(k)).sort();
  return [...known, ...rest].flatMap((k) => {
    const e = entries[k];
    return e ? ([[k, e]] as [string, BudgetEntry][]) : [];
  });
}

function label(key: string, entry: BudgetEntry): string {
  return entry.label ?? SOURCE_NAMES[key] ?? key;
}

/** What a row displays: the entry plus its children, which is what it costs. */
function rowTokens(entry: BudgetEntry): number {
  return entry.total_tokens ?? entry.tokens ?? 0;
}

function BudgetRow({
  name, entry, denominator, depth, expanded, onToggle,
}: {
  name: string;
  entry: BudgetEntry;
  denominator: number;
  depth: number;
  expanded?: boolean;
  onToggle?: () => void;
}) {
  const tokens = rowTokens(entry);
  const pct = denominator > 0 ? (tokens / denominator) * 100 : 0;
  const glyph = entry.indicator ?? POLICY_GLYPH[entry.gc_policy ?? ""] ?? "";
  const children = entry.children ?? {};
  const drillable = onToggle != null && Object.keys(children).length > 0;
  const Row = drillable ? "button" : "div";
  return (
    <Row
      {...(drillable
        ? { type: "button" as const, onClick: onToggle, "aria-expanded": expanded }
        : {})}
      className={
        "w-full text-left grid grid-cols-[1fr_auto_auto] items-baseline gap-x-2 py-0.5 " +
        (drillable ? "hover:bg-surface cursor-pointer" : "")
      }
      style={{ paddingLeft: depth * 12 }}
    >
      <span className="truncate">
        {drillable ? <span className="text-text-muted mr-1">{expanded ? "▾" : "▸"}</span> : null}
        {name}
      </span>
      <span className="font-mono text-[11px] tabular-nums">{fmt(tokens)}</span>
      <span className={"text-[11px] " + policyClass(entry.gc_policy)} title={entry.gc_policy ?? ""}>
        {glyph}
      </span>
      <span className="col-span-3 bar h-[3px]">
        <span style={{ width: `${Math.min(100, pct)}%` }} />
      </span>
    </Row>
  );
}

function BudgetTable({ budget, expanded, onToggle }: {
  budget: BudgetState;
  expanded: string[];
  onToggle: (source: string) => void;
}) {
  // Against the context limit when the daemon reported one, so the bars read
  // as a share of the WINDOW; against the largest row otherwise, which at
  // least keeps them comparable with each other.
  const total = budget.totalTokens ?? 0;
  const rows = sourceOrder(budget.entries);
  const denominator =
    budget.contextLimit && budget.contextLimit > 0
      ? budget.contextLimit
      : Math.max(total, ...rows.map(([, e]) => rowTokens(e)), 1);
  return (
    <div className="text-[13px]">
      {rows.map(([key, entry]) => {
        const open = expanded.includes(key);
        const children = entry.children ?? {};
        return (
          <div key={key}>
            <BudgetRow
              name={label(key, entry)}
              entry={entry}
              denominator={denominator}
              depth={0}
              expanded={open}
              onToggle={() => onToggle(key)}
            />
            {open
              ? Object.entries(children)
                  .sort((a, b) => rowTokens(b[1]) - rowTokens(a[1]))
                  .map(([childKey, child]) => (
                    <BudgetRow
                      key={childKey}
                      name={child.label ?? childKey}
                      entry={child}
                      denominator={denominator}
                      depth={1}
                    />
                  ))
              : null}
          </div>
        );
      })}
    </div>
  );
}

export function BudgetPanel({ agentId }: { agentId: string }) {
  const ctx = useJaato((s) => s.context[agentId]);
  const budget = useJaato((s) => s.budget[agentId]);
  const expanded = useJaato((s) => s.budgetExpanded);
  const toggleSource = useJaato((s) => s.toggleBudgetSource);
  if (!ctx && !budget) {
    return <div className="px-3.5 py-3 text-xs text-text-muted italic">No usage reported yet.</div>;
  }
  const pct =
    ctx?.percentUsed ??
    budget?.utilizationPercent ??
    (ctx?.contextLimit && ctx.usage.total_tokens ? (ctx.usage.total_tokens / ctx.contextLimit) * 100 : null);
  const u = ctx?.usage ?? {};
  const rows: [string, string][] = [
    ["prompt", fmt(u.prompt_tokens)],
    ["output", fmt(u.output_tokens)],
    ["cache read", fmt(u.cache_read_tokens)],
    ["cache write", fmt(u.cache_creation_tokens)],
    ["reasoning", fmt(u.reasoning_tokens ?? u.thinking_tokens)],
    ["total / limit", `${fmt(u.total_tokens)} / ${fmt(ctx?.contextLimit)}`],
    ["remaining", fmt(ctx?.tokensRemaining)],
    ["turns", fmt(ctx?.turns)],
  ];
  if (u.cost_usd != null) rows.push(["cost", `$${Number(u.cost_usd).toFixed(4)}`]);
  return (
    <div className="px-3.5 py-3 text-[13px]">
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="display text-[16px]">Instructions</span>
        <span className="font-mono text-[11px] text-text-muted">
          {budget ? `${fmt(budget.totalTokens)} tracked` : ""}
        </span>
      </div>
      {budget ? (
        <>
          <BudgetTable budget={budget} expanded={expanded} onToggle={toggleSource} />
          <div className="mt-2 text-[11px] text-text-muted">
            {POLICY_GLYPH.locked} never collected &middot; {POLICY_GLYPH.partial} partly &middot;{" "}
            {POLICY_GLYPH.ephemeral} freely
          </div>
        </>
      ) : (
        /* The daemon reports this only once a session has an InstructionBudget;
           saying which panel is empty beats an unexplained gap. */
        <div className="text-xs text-text-muted italic">No instruction budget reported yet.</div>
      )}

      <div className="flex items-baseline justify-between mt-4 mb-1.5">
        <span className="display text-[16px]">Context</span>
        <span className="font-mono text-[11px] text-text-muted">{pct != null ? `${pct.toFixed(1)}%` : ""}</span>
      </div>
      <div className="bar mb-3">
        <span
          className={pct != null && pct > 80 ? "bg-error" : pct != null && pct > 60 ? "bg-warning" : ""}
          style={{ width: `${Math.min(100, pct ?? 0)}%` }}
        />
      </div>
      <dl className="kv m-0">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt>{k}</dt>
            <dd className="m-0">{v}</dd>
          </div>
        ))}
      </dl>
      {ctx?.lastTurn && (
        <div className="mt-3 text-xs text-text-muted">
          Last turn{ctx.lastTurn.turnNumber != null ? ` #${ctx.lastTurn.turnNumber}` : ""}:{" "}
          {ctx.lastTurn.durationSeconds != null ? `${ctx.lastTurn.durationSeconds.toFixed(1)}s` : ""}
          {ctx.lastTurn.toolCalls ? ` · ${describeToolCalls(ctx.lastTurn.toolCalls)}` : ""}
          {ctx.lastTurn.finishReason ? ` · ${ctx.lastTurn.finishReason}` : ""}
        </div>
      )}
    </div>
  );
}
