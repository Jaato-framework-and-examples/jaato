/**
 * One tab per agent (main + subagents) as cells of the session header,
 * like the TUI's agent tab bar (Ctrl+A cycles).  A tab is a status glyph
 * and the agent's name in the chrome face; the selected one sits on the
 * surface with a steel rule under it -- a warning rule when that agent is
 * waiting on a permission, so the tab says what the status bar's count
 * says.  Always rendered, ``main`` included, because the header is where
 * the session's identity lives (design frame 04).
 */
import { useJaato } from "@/store/store";

const STATUS_GLYPH: Record<string, [string, string]> = {
  processing: ["●", "text-primary pulse"],
  running: ["●", "text-primary pulse"],
  awaiting_permission: ["⚠", "text-warning"],
  awaiting_clarification: ["?", "text-steel"],
  finished: ["✓", "text-text-muted"],
  completed: ["✓", "text-text-muted"],
  error: ["✗", "text-error"],
  idle: ["○", "text-text-muted"],
};

export function AgentTabs() {
  const agents = useJaato((s) => s.agents);
  const order = useJaato((s) => s.agentOrder);
  const selected = useJaato((s) => s.selectedAgentId);
  const select = useJaato((s) => s.selectAgent);
  const pendingPerms = useJaato((s) => s.permissions);
  return (
    <div role="tablist" className="flex items-stretch overflow-x-auto" title="Ctrl+A selects the next agent">
      {order.map((id) => {
        const a = agents[id];
        if (!a) return null;
        const hasPerm = pendingPerms.some((p) => p.agentId === id);
        const [g, cls] = hasPerm ? STATUS_GLYPH.awaiting_permission! : (STATUS_GLYPH[a.status] ?? STATUS_GLYPH.idle!);
        const current = id === selected;
        return (
          <button
            key={id}
            role="tab"
            aria-selected={current}
            onClick={() => select(id)}
            className={`flex items-center gap-2 px-3.5 border-r hairline whitespace-nowrap ${current ? `bg-surface text-text shadow-[inset_0_-2px_0_var(--c-steel)] ${hasPerm ? "shadow-[inset_0_-2px_0_var(--c-warning)]" : ""}` : "text-text-muted hover:text-text"}`}
            title={a.profile ? `${a.type} · ${a.profile}` : a.type}
          >
            <span className={cls}>{g}</span>
            <span className={`chrome ${current ? "" : "font-medium"}`}>{a.name}</span>
          </button>
        );
      })}
    </div>
  );
}
