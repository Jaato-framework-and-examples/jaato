/** One tab per agent (main + subagents), with a status glyph, like the TUI's agent tab bar (Ctrl+A cycles). */
import { useJaato } from "@/store/store";

const STATUS_GLYPH: Record<string, [string, string]> = {
  processing: ["●", "text-primary pulse"],
  running: ["●", "text-primary pulse"],
  awaiting_permission: ["⚠", "text-warning"],
  awaiting_clarification: ["?", "text-primary"],
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
  if (order.length <= 1 && !pendingPerms.length) return null;
  return (
    <div role="tablist" className="flex items-center gap-1 px-3 py-1 border-b hairline overflow-x-auto text-xs">
      {order.map((id) => {
        const a = agents[id];
        if (!a) return null;
        const hasPerm = pendingPerms.some((p) => p.agentId === id);
        const [g, cls] = hasPerm ? STATUS_GLYPH.awaiting_permission! : (STATUS_GLYPH[a.status] ?? STATUS_GLYPH.idle!);
        return (
          <button
            key={id}
            role="tab"
            aria-selected={id === selected}
            onClick={() => select(id)}
            className={`px-2.5 py-1 rounded-md flex items-center gap-1.5 whitespace-nowrap ${id === selected ? "bg-surface text-text" : "text-text-muted hover:text-text"}`}
            title={a.profile ? `${a.type} · ${a.profile}` : a.type}
          >
            <span className={cls}>{g}</span>
            <span className={id === selected ? "font-semibold" : ""}>{a.name}</span>
          </button>
        );
      })}
      <span className="flex-1" />
      <span className="text-text-muted hidden md:inline"><kbd>Ctrl</kbd>+<kbd>A</kbd> next agent</span>
    </div>
  );
}
