/** The agent's plan (todo plugin), with the TUI's status vocabulary. */
import { useJaato } from "@/store/store";

const GLYPH: Record<string, [string, string]> = {
  pending: ["○", "text-text-muted"],
  in_progress: ["◐", "text-primary"],
  active: ["◐", "text-primary"],
  completed: ["●", "text-success"],
  done: ["●", "text-success"],
  failed: ["✗", "text-error"],
  skipped: ["–", "text-warning"],
  cancelled: ["–", "text-warning"],
  blocked: ["⊘", "text-accent"],
};

export function PlanPanel({ agentId }: { agentId: string }) {
  const plan = useJaato((s) => s.plan[agentId]);
  if (!plan) return <div className="p-3 text-xs text-text-muted italic">No plan yet.</div>;
  const total = plan.steps.length;
  const done = plan.steps.filter((s) => s.status === "completed" || s.status === "done").length;
  return (
    <div className="p-3 text-[13px]">
      <div className="flex items-baseline justify-between mb-2">
        <div className="font-semibold">{plan.name}</div>
        <div className="text-xs text-text-muted">{done}/{total}</div>
      </div>
      <div className="h-1 rounded bg-surface mb-3 overflow-hidden"><div className="h-full bg-success" style={{ width: total ? `${(done / total) * 100}%` : 0 }} /></div>
      <ol className="space-y-1.5">
        {[...plan.steps].sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0)).map((st, i) => {
          const [g, cls] = GLYPH[st.status ?? "pending"] ?? GLYPH.pending!;
          return (
            <li key={st.step_id ?? i} className="flex gap-2">
              <span className={`${cls} w-4 shrink-0`}>{g}</span>
              <div className="min-w-0">
                <div className={st.status === "in_progress" || st.status === "active" ? "font-medium" : st.status === "completed" ? "text-text-muted" : ""}>{st.content}</div>
                {st.result && <div className="text-xs text-success/80 whitespace-pre-wrap">↳ {st.result}</div>}
                {st.error && <div className="text-xs text-error/90 whitespace-pre-wrap">↳ {st.error}</div>}
                {st.blocked_by && st.blocked_by.length > 0 && <div className="text-xs text-accent">blocked by {st.blocked_by.join(", ")}</div>}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
