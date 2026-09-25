/**
 * The line above the composer that says what the agent is doing.
 *
 * It used to say one thing -- "Agent working" -- and to say it from a
 * boolean that was true for about one round trip per turn and never true
 * at all for a turn this client did not start (see ``store/phase.ts``).
 * It now names the phase, and for the two phases that can last a while it
 * carries an elapsed clock, because "Running cli_based_tool — 2:14" is the
 * difference between a session that is working and one that is wedged.
 *
 * The clock ticks via ``hooks/useTick`` rather than in the store: a second
 * hand is presentation, and putting it in the store would commit React
 * once a second for every subscriber of every slice.  ``AgentTabs`` and the
 * stall banner share the same hook for the same reason.
 */
import { useShallow } from "zustand/react/shallow";
import { useJaato } from "@/store/store";
import { agentPhase, phaseLabel, type AgentPhase } from "@/store/phase";
import { useTick } from "@/hooks/useTick";

/** ``m:ss`` past a minute, ``Ns`` below it — the shape a duration is read at. */
export function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  if (total < 60) return `${total}s`;
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

const GLYPH: Record<AgentPhase["kind"], [string, string]> = {
  idle: ["○", "text-text-muted"],
  sending: ["●", "text-text-muted pulse"],
  thinking: ["●", "text-primary pulse"],
  tool: ["▸", "text-steel pulse"],
  waiting: ["⚠", "text-warning"],
};

export function PhaseLine({ agentId }: { agentId: string }) {
  // ``useShallow``: the phase is derived, so it is a new object on every
  // store update; zustand v5 requires a stable reference from a selector.
  const phase = useJaato(useShallow((s) => agentPhase(s, agentId)));
  const now = useTick(phase.kind === "thinking" || phase.kind === "tool" || phase.kind === "waiting");
  if (phase.kind === "idle") return null;
  const [glyph, cls] = GLYPH[phase.kind];
  // ``sending`` has no clock: it lasts one round trip, and a "0s" that never
  // moves reads as a stall rather than as the handover it is.
  const elapsed = phase.kind === "sending" ? null : formatElapsed(now - phase.since);
  return (
    <div
      role="status"
      aria-live="polite"
      className="kicker kicker-muted text-[12px] tracking-[0.1em] mb-1.5 flex items-center gap-2"
    >
      <span className={cls}>{glyph}</span>
      <span>{phaseLabel(phase)}</span>
      {elapsed && <span className="font-mono normal-case tracking-normal text-text-muted">{elapsed}</span>}
      <span className="text-divider">·</span>
      {phase.kind === "waiting" ? (
        <span>answer above to continue</span>
      ) : (
        <span>
          type to queue a follow-up · <span className="font-mono normal-case tracking-normal">stop</span> or Ctrl+C to interrupt
        </span>
      )}
    </div>
  );
}
