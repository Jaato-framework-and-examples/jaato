/**
 * One tab per agent (main + subagents) as cells of the session header,
 * like the TUI's agent tab bar (⌘/Ctrl+K then [ / ] steps, 1-9 jumps
 * straight to a tab -- #1304 §7's leader, replacing the direct Ctrl+A
 * this bar used to cycle forward on).  A tab is a status glyph
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
 *
 * Stall detection (#1304 §3): a ``thinking``/``sending`` agent this
 * session has heard nothing from for the configured threshold draws
 * amber, with a live "no output Ns" clock -- the one glyph state
 * ``PHASE_GLYPH`` does not own outright, since a stall is a property of
 * silence WITHIN a phase, not a phase of its own (``store/phase.ts``).
 * The clock ticks off the shared ``hooks/useTick`` 1 Hz interval, the same
 * one ``PhaseLine`` and the parent-transcript stall banner use, so the
 * three cannot drift about what "now" is.
 *
 * ``main`` included, and tab-switching is never automatic: nothing here
 * calls ``selectAgent`` except the click handler below, so a stalled or
 * newly-created background agent is announced (its glyph, and the
 * attention banner in ``SessionScreen``) rather than switched to.
 */
import { useShallow } from "zustand/react/shallow";
import { useJaato } from "@/store/store";
import { agentPhase, phaseLabel, stalled, type AgentPhase } from "@/store/phase";
import { useTick } from "@/hooks/useTick";

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
  const lastEventAt = useJaato((s) => s.lastEventAt[id]);
  const thresholdMs = useJaato((s) => s.stallThresholdMs);
  const ctx = useJaato((s) => s.context[id]);
  // Ticks only while there is something to stall — an idle tab costs no
  // per-second re-render.
  const now = useTick(phase.kind === "thinking" || phase.kind === "sending");
  if (!agent) return null;
  const waiting = phase.kind === "waiting";
  const stall = stalled(phase, lastEventAt, thresholdMs, now);
  const [glyph, cls] = stall
    ? ["●", "text-warning pulse"]
    : phase.kind === "idle"
      ? STATUS_GLYPH[agent.status] ?? STATUS_GLYPH.idle!
      : PHASE_GLYPH[phase.kind];
  const what = stall ? `Stalled — no output ${Math.round(stall.silentForMs / 1000)}s` : phase.kind === "idle" ? agent.status : phaseLabel(phase);
  const pct = ctx?.percentUsed;
  return (
    <button
      role="tab"
      aria-selected={selected}
      onClick={() => select(id)}
      className={`flex items-center gap-2 px-3.5 border-r hairline whitespace-nowrap ${selected ? `bg-surface text-text shadow-[inset_0_-2px_0_var(--c-steel)] ${waiting ? "shadow-[inset_0_-2px_0_var(--c-warning)]" : stall ? "shadow-[inset_0_-2px_0_var(--c-warning)]" : ""}` : "text-text-muted hover:text-text"}`}
      title={`${agent.profile ? `${agent.type} · ${agent.profile}` : agent.type} — ${what}`}
      data-testid={`agent-tab-${id}`}
      data-stalled={stall ? "true" : undefined}
    >
      <span className={cls}>{glyph}</span>
      <span className={`chrome ${selected ? "" : "font-medium"}`}>{agent.name}</span>
      {stall && <span className="chrome-xs text-warning font-mono" data-testid="agent-tab-stall-clock">{Math.round(stall.silentForMs / 1000)}s</span>}
      {/* The context meter, kept on subagent tabs too (§4): previously
          shown only for the selected agent in ``SessionHeader``. */}
      {pct != null && (
        <span className="font-mono text-[10px] text-text-muted" title="Context window used">{pct.toFixed(0)}%</span>
      )}
    </button>
  );
}

export function AgentTabs() {
  const order = useJaato((s) => s.agentOrder);
  return (
    <div role="tablist" className="flex items-stretch overflow-x-auto" title="⌘/Ctrl+K then [ or ] steps to the previous/next agent, 1-9 jumps to a tab">
      {order.map((id) => <AgentTab key={id} id={id} />)}
    </div>
  );
}
