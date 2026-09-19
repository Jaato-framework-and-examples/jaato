/**
 * One tab per agent (main + subagents) as cells of the session header,
 * like the TUI's agent tab bar (Ctrl+A cycles).  A tab is a status glyph
 * and the agent's name in the chrome face; the selected one sits on the
 * surface with a steel rule under it -- a warning rule when that agent is
 * waiting on a permission, so the tab says what the status bar's count
 * says.  Always rendered, ``main`` included, because the header is where
 * the session's identity lives (design frame 04).
 *
 * The glyph is the agent's PHASE where it has one and the daemon's status
 * otherwise, which is what makes a tab the place to notice an agent you
 * are not looking at.  The status map used to key on ``processing`` /
 * ``running`` / ``awaiting_permission`` / ``finished`` -- four words no
 * daemon emits -- so every tab fell through to the idle glyph whatever the
 * agent was doing.  The vocabulary here is now the one in
 * ``jaato_sdk/events.py``, and the TUI's ``agent_tab_bar.py`` is the
 * reference for which symbol means what.
 */
import { useShallow } from "zustand/react/shallow";
import { useJaato } from "@/store/store";
import { agentPhase, phaseLabel, type AgentPhase } from "@/store/phase";

/** The daemon's own vocabulary, for an agent that is not busy. */
const STATUS_GLYPH: Record<string, [string, string]> = {
  idle: ["○", "text-text-muted"],
  done: ["✓", "text-text-muted"],
  error: ["✗", "text-error"],
  cancelled: ["⊘", "text-text-muted"],
};

/** What the agent is doing, when it is doing something. */
const PHASE_GLYPH: Record<Exclude<AgentPhase["kind"], "idle">, [string, string]> = {
  sending: ["●", "text-text-muted pulse"],
  thinking: ["●", "text-primary pulse"],
  tool: ["▸", "text-steel pulse"],
  waiting: ["⚠", "text-warning"],
};

function AgentTab({ id }: { id: string }) {
  const agent = useJaato((s) => s.agents[id]);
  const selected = useJaato((s) => s.selectedAgentId === id);
  const select = useJaato((s) => s.selectAgent);
  const phase = useJaato(useShallow((s) => agentPhase(s, id)));
  if (!agent) return null;
  const waiting = phase.kind === "waiting";
  const [glyph, cls] = phase.kind === "idle"
    ? STATUS_GLYPH[agent.status] ?? STATUS_GLYPH.idle!
    : PHASE_GLYPH[phase.kind];
  const what = phase.kind === "idle" ? agent.status : phaseLabel(phase);
  return (
    <button
      role="tab"
      aria-selected={selected}
      onClick={() => select(id)}
      className={`flex items-center gap-2 px-3.5 border-r hairline whitespace-nowrap ${selected ? `bg-surface text-text shadow-[inset_0_-2px_0_var(--c-steel)] ${waiting ? "shadow-[inset_0_-2px_0_var(--c-warning)]" : ""}` : "text-text-muted hover:text-text"}`}
      title={`${agent.profile ? `${agent.type} · ${agent.profile}` : agent.type} — ${what}`}
    >
      <span className={cls}>{glyph}</span>
      <span className={`chrome ${selected ? "" : "font-medium"}`}>{agent.name}</span>
    </button>
  );
}

export function AgentTabs() {
  const order = useJaato((s) => s.agentOrder);
  return (
    <div role="tablist" className="flex items-stretch overflow-x-auto" title="Ctrl+A selects the next agent">
      {order.map((id) => <AgentTab key={id} id={id} />)}
    </div>
  );
}
