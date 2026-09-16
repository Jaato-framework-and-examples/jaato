/** The agent's plan (todo plugin), with the TUI's status vocabulary, as a rail section (design frame 04). */
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
  blocked: ["⊘", "text-warning"],
};

/** ``done/total`` for the rail header, or nothing before a plan exists. */
export function planProgress(plan: { steps: { status?: string }[] } | undefined): string | null {
  if (!plan) return null;
  const total = plan.steps.length;
  const done = plan.steps.filter((s) => s.status === "completed" || s.status === "done").length;
  return `${done}/${total}`;
}

export function PlanPanel({ agentId }: { agentId: string }) {
  const plan = useJaato((s) => s.plan[agentId]);
  if (!plan) return <div className="px-3.5 py-3 text-xs text-text-muted italic">No plan yet.</div>;
  const total = plan.steps.length;
  const done = plan.steps.filter((s) => s.status === "completed" || s.status === "done").length;
  return (
    <div className="px-3.5 py-3 text-[13.5px]">
      <div className="display text-[16px] mb-2">{plan.name}</div>
      <div className="bar mb-2.5"><span className="bg-success" style={{ width: total ? `${(done / total) * 100}%` : 0 }} /></div>
      <ol className="space-y-1.5 list-none m-0 p-0">
        {[...plan.steps].sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0)).map((st, i) => {
          const [g, cls] = GLYPH[st.status ?? "pending"] ?? GLYPH.pending!;
          return (
            <li key={st.step_id ?? i} className="flex gap-2.5">
              <span className={`${cls} w-3 shrink-0`}>{g}</span>
              <div className="min-w-0">
                <div className={st.status === "in_progress" || st.status === "active" ? "font-medium" : st.status === "completed" || st.status === "done" ? "text-text-muted" : ""}>{st.content}</div>
                {st.result && <div className="text-xs text-success/80 whitespace-pre-wrap">↳ {st.result}</div>}
                {st.error && <div className="text-xs text-error/90 whitespace-pre-wrap">↳ {st.error}</div>}
                {st.blocked_by && st.blocked_by.length > 0 && <div className="text-xs text-warning">blocked by {st.blocked_by.join(", ")}</div>}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
