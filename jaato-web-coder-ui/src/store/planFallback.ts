/**
 * "If ``setStepStatus`` fails, the plan panel shows the step's real state
 * from the subagent's status" (#1304 §3).
 *
 * A plan is per-agent (``s.plan[agentId]``), so the ordinary case is the
 * agent updating its OWN plan by calling ``setStepStatus`` as it works.
 * When that call fails -- a tool row for ``setStepStatus`` with
 * ``status: "error"`` in this agent's transcript -- the step the call was
 * trying to update can be left showing a stale status (``in_progress``,
 * say) forever, because nothing else ever retries the write.  The agent's
 * OWN lifecycle status is a fact the panel already has and the failed
 * write cannot desynchronise: once the daemon reports the agent ``done``
 * or ``error``, that is the true outcome of its last step, whatever the
 * plan document still says.
 *
 * Deliberately narrow: only the LAST step (by ``sequence``) is ever
 * overridden, and only when it is not already in a terminal state -- an
 * earlier step's stale status is not this fallback's business, and a step
 * already marked ``completed``/``failed``/``cancelled``/``skipped`` is not
 * stale, it is the record.
 */
import type { PlanState, PlanStep, ToolBlock } from "./types";

const TERMINAL_STEP_STATUSES = new Set(["completed", "done", "failed", "cancelled", "skipped"]);

/** True when this agent's transcript shows a ``setStepStatus`` call that failed. */
export function hasFailedSetStepStatus(blocks: readonly { kind: string }[]): boolean {
  return (blocks as ToolBlock[]).some((b) => b.kind === "tool" && b.toolName === "setStepStatus" && b.status === "error");
}

/** The step ``sequence`` puts last, or ``null`` for an empty plan. */
function lastStep(steps: readonly PlanStep[]): PlanStep | null {
  if (steps.length === 0) return null;
  return [...steps].sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0))[steps.length - 1]!;
}

export interface StepFallback {
  stepId: string | undefined;
  /** The status to SHOW for this step, overriding the plan document's own. */
  status: "completed" | "failed";
  /** Why: shown as a note under the step, so the override does not read as the model's own report. */
  note: string;
}

/**
 * The override to apply, or ``null`` when none is warranted: the agent is
 * still busy, no ``setStepStatus`` call failed, the plan is empty, or its
 * last step is already terminal.
 */
export function planFallback(plan: PlanState | undefined, agentStatus: string, setStepStatusFailed: boolean): StepFallback | null {
  if (!plan || !setStepStatusFailed) return null;
  if (agentStatus !== "done" && agentStatus !== "error") return null;
  const step = lastStep(plan.steps);
  if (!step || TERMINAL_STEP_STATUSES.has(step.status ?? "")) return null;
  return agentStatus === "done"
    ? { stepId: step.step_id, status: "completed", note: "shown from the agent's own status — its last update to this step failed" }
    : { stepId: step.step_id, status: "failed", note: "shown from the agent's own status (it ended in error) — its last update to this step failed" };
}

/** ``plan.steps`` with the fallback applied, if any -- what the panel renders. */
export function stepsWithFallback(plan: PlanState | undefined, fallback: StepFallback | null): PlanStep[] {
  if (!plan) return [];
  if (!fallback) return plan.steps;
  return plan.steps.map((st) => (st.step_id === fallback.stepId ? { ...st, status: fallback.status, error: fallback.status === "failed" ? (st.error ?? fallback.note) : st.error, __fallbackNote: fallback.note } : st));
}
