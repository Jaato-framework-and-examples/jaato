/**
 * A background agent needing you or stalled (#1304 §3, §4), as a banner
 * strip above the transcript, one row per agent that is NOT the one
 * currently selected — the selected agent's own prompts and its tab's
 * stall glyph are already on screen, so duplicating them here would say
 * the same thing twice.  Never auto-switches tabs: every action here is
 * explicit (Open tab / Nudge / Cancel), never a `selectAgent` call of its
 * own accord.
 *
 * **Nudge-channel finding** (read before touching this file, and see the
 * PR description for #1304 phase 2a for the full writeup): `injectPrompt`
 * (`jaato-sdk-ts` `client.ts`) carries **no** `agent_id` field.  The
 * daemon's `InjectPromptRequest` handler (`session_manager.py`) resolves
 * `session_id = self._client_to_session[client_id]` — the PARENT/top-level
 * session this WS connection is attached to — and calls
 * `deliver_prompt_to_session`, with no mechanism to target a specific
 * agent WITHIN that session's event stream.  The web client's "agent
 * tabs" (main + subagents) all live inside ONE `sessionId` on the wire;
 * subagents are surfaced purely via `agent_id`-tagged notification
 * frames, never as separate client-visible sessions.  So Nudge can only
 * ever reach the MAIN agent's driving turn.
 *
 * Offering "Nudge" for a stalled SUBAGENT would either nudge the wrong
 * agent (the main one) or nothing at all, while the button claims to
 * nudge the agent named on the row — indistinguishable, from the user's
 * side, from doing the right thing.  So it is scoped rather than wired
 * to a target it cannot reach: enabled only when `agentId === MAIN_AGENT`,
 * and rendered disabled with an explanatory tooltip for every subagent
 * row.  Inventing a new server verb to target a subagent is explicitly
 * out of scope for this PR (see the task brief) — that is a `jaato-server`
 * change and belongs, if it happens, in its own issue.
 *
 * Cancel has no such limitation: `client.stop(agentId)` (confirmed in
 * `jaato-sdk-ts/src/client.ts` ~L695 — `{type: STOP, agent_id: agentId ??
 * null}`) already carries an `agent_id` and stops exactly that agent, so
 * Cancel is offered for every row.
 */
import { useMemo } from "react";
import { useJaato, MAIN_AGENT, type JaatoState } from "@/store/store";
import { agentPhase, phaseLabel, stalled } from "@/store/phase";
import { useTick } from "@/hooks/useTick";
import { getClient, isConnected } from "@/sdk/connection";

interface AttentionRow {
  id: string;
  name: string;
  kind: "waiting" | "stalled";
  detail: string;
}

function attentionRows(s: JaatoState, now: number): AttentionRow[] {
  const out: AttentionRow[] = [];
  for (const id of s.agentOrder) {
    if (id === s.selectedAgentId) continue;
    const agent = s.agents[id];
    if (!agent) continue;
    const phase = agentPhase(s, id);
    if (phase.kind === "waiting") {
      out.push({ id, name: agent.name, kind: "waiting", detail: phaseLabel(phase) });
      continue;
    }
    const stall = stalled(phase, s.lastEventAt[id], s.stallThresholdMs, now);
    if (stall) out.push({ id, name: agent.name, kind: "stalled", detail: `no output ${Math.max(1, Math.round(stall.silentForMs / 1000))}s` });
  }
  return out;
}

const NUDGE_TEXT = "(nudge) Still there? Please continue with the task, or say what is blocking you.";

function AttentionRowView({ row }: { row: AttentionRow }) {
  const selectAgent = useJaato((s) => s.selectAgent);
  const canNudge = row.id === MAIN_AGENT;
  const nudge = () => {
    if (!isConnected()) return;
    getClient().injectPrompt(NUDGE_TEXT, "user").catch(() => undefined);
  };
  const cancel = () => {
    if (!isConnected()) return;
    getClient().stop(row.id).catch(() => undefined);
  };
  return (
    <div
      role="alert"
      data-testid={`attention-banner-${row.id}`}
      className={`flex items-center gap-3 px-3.5 py-1.5 border hairline text-[13px] ${row.kind === "stalled" ? "text-warning border-warning/40" : "text-warning border-warning/40"}`}
    >
      <span aria-hidden="true">{row.kind === "waiting" ? "⚠" : "●"}</span>
      <span className="flex-1 truncate">
        <span className="font-medium">{row.name}</span>
        <span className="text-text-muted"> — {row.kind === "waiting" ? "needs you" : "stalled"} — {row.detail}</span>
      </span>
      <button type="button" onClick={() => selectAgent(row.id)} className="btn btn-sm btn-quiet">Open tab</button>
      <button
        type="button"
        onClick={nudge}
        disabled={!canNudge}
        title={canNudge ? "Send a nudge to keep this agent going" : "Nudge can only reach the main agent's turn — the daemon has no way to target a subagent's inject (see this file's docstring)"}
        className="btn btn-sm btn-quiet disabled:opacity-40 disabled:cursor-not-allowed"
      >
        Nudge
      </button>
      <button type="button" onClick={cancel} className="btn btn-sm btn-quiet hover:text-error">Cancel</button>
    </div>
  );
}

export function AttentionBanners() {
  // Ticks only while a stall could exist to notice going stale; the
  // 2s cadence is plenty for a "no output Ns" line nobody needs to the
  // second.
  const now = useTick(true, 2000);
  // ``attentionRows`` builds a FRESH array of fresh row objects every call
  // -- it has to, ``silentForMs`` grows with ``now``.  Feeding that straight
  // through a ``useShallow`` zustand selector is unstable: ``useShallow``
  // compares the array's ELEMENTS by reference, and a freshly-constructed
  // object is never ``===`` its predecessor, so ``getSnapshot`` never
  // converges and React tears the tree down with "Maximum update depth
  // exceeded" the instant a row exists (reproduced live: a stalled
  // subagent crashed the whole session view, not just this component).
  // ``useJaato()`` with no selector returns the store's own state object,
  // which IS referentially stable between renders until something actually
  // ``set()``s -- so the derivation reads the store here and is memoized
  // separately, off ``now`` and that one stable reference, rather than
  // being asked to stand in for zustand's own equality check.
  const s = useJaato();
  const rows = useMemo(() => attentionRows(s, now), [s, now]);
  if (rows.length === 0) return null;
  return (
    <div className="flex flex-col gap-1 px-5 pt-2" aria-label="Background agents needing attention">
      {rows.map((r) => <AttentionRowView key={r.id} row={r} />)}
    </div>
  );
}
