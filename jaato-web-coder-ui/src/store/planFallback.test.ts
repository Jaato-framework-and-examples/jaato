import { describe, expect, it } from "vitest";
import { hasFailedSetStepStatus, planFallback, stepsWithFallback } from "./planFallback";
import type { PlanState, ToolBlock } from "./types";

const tb = (over: Partial<ToolBlock>): ToolBlock => ({ id: "t1", kind: "tool", agentId: "main", callId: "c1", toolName: "setStepStatus", args: {}, output: "", media: [], status: "success", expanded: false, startedAt: 0, ...over });

describe("hasFailedSetStepStatus", () => {
  it("is false with no setStepStatus call at all", () => {
    expect(hasFailedSetStepStatus([{ kind: "text", id: "x", text: "hi", streaming: false } as never])).toBe(false);
  });
  it("is false when setStepStatus succeeded", () => {
    expect(hasFailedSetStepStatus([tb({ status: "success" })])).toBe(false);
  });
  it("is true when a setStepStatus call errored", () => {
    expect(hasFailedSetStepStatus([tb({ status: "error" })])).toBe(true);
  });
  it("ignores a differently-named tool's error", () => {
    expect(hasFailedSetStepStatus([tb({ toolName: "writeNewFile", status: "error" })])).toBe(false);
  });
});

function plan(steps: PlanState["steps"]): PlanState {
  return { name: "Plan", steps };
}

describe("planFallback — #1304 §3", () => {
  it("is null while the agent is still busy, even with a failed call", () => {
    expect(planFallback(plan([{ step_id: "s1", content: "x", status: "in_progress", sequence: 0 }]), "active", true)).toBeNull();
  });
  it("is null when nothing failed", () => {
    expect(planFallback(plan([{ step_id: "s1", content: "x", status: "in_progress", sequence: 0 }]), "done", false)).toBeNull();
  });
  it("is null for an empty plan", () => {
    expect(planFallback(undefined, "done", true)).toBeNull();
    expect(planFallback(plan([]), "done", true)).toBeNull();
  });
  it("is null when the last step is already terminal -- not stale, it is the record", () => {
    expect(planFallback(plan([{ step_id: "s1", content: "x", status: "completed", sequence: 0 }]), "done", true)).toBeNull();
    expect(planFallback(plan([{ step_id: "s1", content: "x", status: "failed", sequence: 0 }]), "error", true)).toBeNull();
  });
  it("overrides only the LAST step (by sequence), to completed on a done agent", () => {
    const p = plan([
      { step_id: "s1", content: "a", status: "completed", sequence: 0 },
      { step_id: "s2", content: "b", status: "in_progress", sequence: 1 },
    ]);
    const fb = planFallback(p, "done", true);
    expect(fb).toMatchObject({ stepId: "s2", status: "completed" });
  });
  it("overrides to failed on an agent that ended in error", () => {
    const p = plan([{ step_id: "s1", content: "a", status: "in_progress", sequence: 0 }]);
    const fb = planFallback(p, "error", true);
    expect(fb).toMatchObject({ stepId: "s1", status: "failed" });
  });
  it("an earlier step's stale status is not this fallback's business", () => {
    const p = plan([
      { step_id: "s1", content: "a", status: "in_progress", sequence: 0 },
      { step_id: "s2", content: "b", status: "completed", sequence: 1 },
    ]);
    expect(planFallback(p, "done", true)).toBeNull();
  });
});

describe("stepsWithFallback", () => {
  it("returns the plan's own steps unchanged when there is no fallback", () => {
    const p = plan([{ step_id: "s1", content: "a", status: "in_progress", sequence: 0 }]);
    expect(stepsWithFallback(p, null)).toBe(p.steps);
  });
  it("applies the override to exactly the named step", () => {
    const p = plan([
      { step_id: "s1", content: "a", status: "completed", sequence: 0 },
      { step_id: "s2", content: "b", status: "in_progress", sequence: 1 },
    ]);
    const steps = stepsWithFallback(p, { stepId: "s2", status: "completed", note: "shown from status" });
    expect(steps.find((s) => s.step_id === "s2")?.status).toBe("completed");
    expect(steps.find((s) => s.step_id === "s1")?.status).toBe("completed");
  });
});
