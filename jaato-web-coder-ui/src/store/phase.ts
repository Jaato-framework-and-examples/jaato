/**
 * What an agent is doing right now, derived rather than stored.
 *
 * The client used to keep one boolean per agent, ``processing``, written
 * from four places and read from four more -- and the write that mattered
 * tested for a status vocabulary the daemon does not have.
 * ``AgentStatusChangedEvent.status`` is one of ``active`` / ``idle`` /
 * ``done`` / ``error`` / ``cancelled`` (``jaato_sdk/events.py``, and every
 * emitter in ``server/core.py`` and the subagent plugin); the store tested
 * ``status === "processing" || status === "running"``.  Nothing upstream
 * produces either word.  The only producer was the client itself, which
 * dispatched a fabricated ``status: "processing"`` at its own composer
 * before awaiting ``sendMessage`` -- so the flag was true for exactly as
 * long as it took the daemon's real ``active`` to arrive and, being an
 * unconditional assignment, turn it back OFF.  A turn the composer did not
 * start (an attach to a running session, a subagent, ``session.wake``, an
 * injected prompt, a reconnect mid-turn) never set it at all.
 *
 * Deriving removes the class: there is no second copy of "is it busy" to
 * fall out of step with the events.  The phase is a function of facts the
 * store already holds for other reasons -- the daemon's word for the
 * agent's status, the pending-prompt lists the prompts themselves render
 * from, and the tool blocks in the transcript.
 *
 * It is also self-healing where a stored flag was not.  A tool block whose
 * ``tool.call_end`` never arrived (a reconnect mid-call) cannot pin the
 * indicator, because the ``tool`` phase is reachable only while the agent
 * is busy by the daemon's own account.
 */
import type { JaatoState } from "./store";

/**
 * The statuses the daemon actually emits.  ``cancelled`` comes from the
 * subagent plugin, the rest from ``server/core.py``.  Kept as a named set
 * so a reader can see the whole vocabulary in one place, and so a test can
 * assert the client answers for each of them.
 */
export const DAEMON_STATUSES = ["active", "idle", "done", "error", "cancelled"] as const;

/** The one status that means "this agent is taking a turn right now". */
export const BUSY_STATUS = "active";

export type WaitingOn = "permission" | "clarification" | "reference";

export type AgentPhase =
  /** Nothing in flight: the composer's turn. */
  | { kind: "idle" }
  /** The composer has sent and the daemon has not yet spoken about it. */
  | { kind: "sending"; since: number }
  /** Busy, with no tool call open: the model is generating. */
  | { kind: "thinking"; since: number }
  /** Busy, with at least one tool call open. */
  | { kind: "tool"; since: number; toolName: string; running: number }
  /** Blocked on a person: a prompt is on screen and nothing moves until it is answered. */
  | { kind: "waiting"; since: number; on: WaitingOn; toolName?: string };

const IDLE: AgentPhase = { kind: "idle" };

/**
 * The oldest tool call still open for ``agentId``, or ``null``.
 *
 * Oldest rather than newest: with a parallel batch in flight the useful
 * elapsed time is the one the turn has been in tools, and the newest call
 * would restart the clock on every fan-out.
 */
function openTools(s: JaatoState, agentId: string): { first: { toolName: string; startedAt: number }; count: number } | null {
  let first: { toolName: string; startedAt: number } | null = null;
  let count = 0;
  for (const b of s.blocks[agentId] ?? []) {
    if (b.kind !== "tool" || b.status !== "running") continue;
    count += 1;
    if (!first || b.startedAt < first.startedAt) first = { toolName: b.toolName, startedAt: b.startedAt };
  }
  return first ? { first, count } : null;
}

/** The pending prompt this agent is blocked on, in the order a person sees them. */
function waitingOn(s: JaatoState, agentId: string): { on: WaitingOn; toolName?: string } | null {
  const p = s.permissions.find((x) => x.agentId === agentId);
  if (p) return { on: "permission", toolName: p.toolName };
  const c = s.clarifications.find((x) => x.agentId === agentId);
  if (c) return { on: "clarification", toolName: c.toolName || undefined };
  if (s.referenceSelections.some((x) => x.agentId === agentId)) return { on: "reference" };
  return null;
}

/**
 * The agent's phase.  Priority is by what a person needs to know first:
 * a prompt waiting on them outranks anything the machine is doing, and a
 * named tool outranks the generic "thinking" it is a part of.
 */
export function agentPhase(s: JaatoState, agentId: string): AgentPhase {
  // A prompt on screen is a fact about this agent whatever else is true of
  // it, and it is the one phase a person has to act on.
  const w = waitingOn(s, agentId);
  if (w) return { kind: "waiting", since: s.busySince[agentId] ?? Date.now(), ...w };

  // ``busySince`` is the busy predicate, not the status string: it is
  // stamped by the two events that START a turn (the daemon's ``active``,
  // the composer's send) and dropped by every event that ENDS one -- a
  // non-active status, ``turn.completed``, ``agent.completed``,
  // ``agent.error``.  So a tool call whose ``call_end`` was lost cannot
  // keep the indicator alive past the turn that opened it.
  const since = s.busySince[agentId];
  if (since === undefined) return IDLE;

  const tools = openTools(s, agentId);
  if (tools) return { kind: "tool", since: tools.first.startedAt, toolName: tools.first.toolName, running: tools.count };
  // ``active`` is the daemon's word; without it the only thing that made
  // this agent busy is the composer, still waiting to be answered.
  if (s.agents[agentId]?.status === BUSY_STATUS) return { kind: "thinking", since };
  return { kind: "sending", since };
}

/** Is this agent doing anything?  The successor of ``processing[agentId]``. */
export function isBusy(s: JaatoState, agentId: string): boolean {
  return agentPhase(s, agentId).kind !== "idle";
}

/** Is ANY agent of this session doing anything? (the exit question's input). */
export function anyBusy(s: JaatoState): boolean {
  return s.agentOrder.some((id) => isBusy(s, id));
}

/**
 * Stall detection (#1304 §3): an agent is STALLED when it is
 * ``thinking``/``sending`` -- generating, or waiting for the daemon's
 * first word about a send -- and nothing has been heard from it for the
 * configured threshold.  Deliberately does NOT apply to ``tool`` (a
 * running tool call is known activity, not silence) or ``waiting`` (that
 * is already the "needs you" state, and outranks stalled).
 *
 * ``DEFAULT_STALL_THRESHOLD_MS`` is the issue's default (60s); the knob is
 * a store field (``stallThresholdMs``) clamped to the issue's declared
 * range (30s-300s) by the SETTER (``setStallThreshold`` in ``store.ts``) --
 * this module clamps too, defensively, so a value written some other way
 * (a test, a stale ``localStorage`` entry) cannot produce a threshold
 * outside the documented range.
 */
export const DEFAULT_STALL_THRESHOLD_MS = 60_000;
export const MIN_STALL_THRESHOLD_MS = 30_000;
export const MAX_STALL_THRESHOLD_MS = 300_000;

export function clampStallThreshold(ms: number): number {
  if (!Number.isFinite(ms)) return DEFAULT_STALL_THRESHOLD_MS;
  return Math.min(MAX_STALL_THRESHOLD_MS, Math.max(MIN_STALL_THRESHOLD_MS, ms));
}

export interface StallInfo {
  /** How long since the last event this agent was heard from. */
  silentForMs: number;
  /** The threshold that was crossed, after clamping. */
  thresholdMs: number;
}

/**
 * The pure predicate: given a phase already computed (``agentPhase``), the
 * agent's ``lastEventAt`` stamp (``undefined`` if nothing has been heard at
 * all, in which case the phase's own ``since`` is the fallback -- the
 * moment it started being busy), the configured threshold and the current
 * clock, decide whether this reads as a stall.
 *
 * Framework-free on purpose: ``AgentTabs`` and the parent-transcript banner
 * both need this against a live-ticking ``now`` they own themselves (see
 * ``hooks/useTick``), so it takes exactly the values it needs rather than
 * the whole store -- a ``useJaato`` selector re-run every tick would work,
 * but coupling the predicate to ``JaatoState`` makes it untestable without
 * one.
 */
export function stalled(phase: AgentPhase, lastEventAt: number | undefined, thresholdMs: number, now: number): StallInfo | null {
  if (phase.kind !== "thinking" && phase.kind !== "sending") return null;
  const threshold = clampStallThreshold(thresholdMs);
  const last = lastEventAt ?? phase.since;
  const silentForMs = now - last;
  return silentForMs >= threshold ? { silentForMs, thresholdMs: threshold } : null;
}

/** ``stalled`` computed straight off a ``JaatoState``, for a caller (a test,
 *  a non-React helper) that already has one and does not want to derive the
 *  phase itself. */
export function stallInfo(s: JaatoState, agentId: string, now: number = Date.now()): StallInfo | null {
  return stalled(agentPhase(s, agentId), s.lastEventAt[agentId], s.stallThresholdMs, now);
}

/** One line of chrome copy for a phase: what is happening, without the elapsed time. */
export function phaseLabel(p: AgentPhase): string {
  switch (p.kind) {
    case "sending": return "Sending";
    case "thinking": return "Thinking";
    case "tool": return p.running > 1 ? `Running ${p.toolName} +${p.running - 1} more` : `Running ${p.toolName}`;
    case "waiting":
      if (p.on === "permission") return p.toolName ? `Waiting for you — permission for ${p.toolName}` : "Waiting for you — permission";
      if (p.on === "clarification") return "Waiting for you — clarification";
      return "Waiting for you — reference";
    default: return "Idle";
  }
}
