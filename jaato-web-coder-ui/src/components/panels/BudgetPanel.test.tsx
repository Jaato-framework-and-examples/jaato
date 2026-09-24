/**
 * The Budget section shows a budget.
 *
 * Reported from a deployed client: "this is not the same budget panel as the
 * TUI".  It was not — the section rendered the context readout (how FULL the
 * window is) under a heading that said Budget, while the breakdown of what
 * the window is spent ON rode the wire as `InstructionBudgetEvent` and was
 * read by nobody: neither `INSTRUCTION_BUDGET_UPDATED` nor its snapshot
 * appeared anywhere under `src/`.
 *
 * Two of these are negative controls, and they are the load-bearing ones: a
 * panel that blanked itself on an empty snapshot, or that invented a policy
 * glyph the daemon did not send, would pass every positive case here.
 */
import { beforeEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { JaatoEvent } from "@jaato/sdk";
import { BudgetPanel } from "./BudgetPanel";
import { POLICY_EXPLAINED } from "@/protocol/gc";
import { useJaato, MAIN_AGENT } from "@/store/store";

const ev = (o: Record<string, unknown>) => o as unknown as JaatoEvent;

/** A snapshot shaped like `InstructionBudget.snapshot()` serialises one. */
const SNAPSHOT = {
  session_id: "20260921_100000",
  agent_id: "main",
  context_limit: 200_000,
  total_tokens: 33_000,
  utilization_percent: 16.5,
  locked_tokens: 9_000,
  gc_eligible_tokens: 20_000,
  entries: {
    system: { source: "system", tokens: 9_000, total_tokens: 9_000, gc_policy: "locked", indicator: "\u{1F512}" },
    plugin: {
      source: "plugin", tokens: 0, total_tokens: 12_000, gc_policy: "partial", indicator: "◐",
      children: {
        cli: { source: "plugin", tokens: 4_000, total_tokens: 4_000, gc_policy: "locked", label: "cli" },
        memory: { source: "plugin", tokens: 8_000, total_tokens: 8_000, gc_policy: "ephemeral", label: "memory" },
      },
    },
    conversation: { source: "conversation", tokens: 12_000, total_tokens: 12_000, gc_policy: "ephemeral", indicator: "○" },
  },
};

function budgetEvent(over: Record<string, unknown> = {}) {
  return ev({ type: "instruction_budget.updated", agent_id: "main", budget_snapshot: { ...SNAPSHOT, ...over } });
}

beforeEach(() => {
  cleanup();
  useJaato.getState().resetSessionState();
  // The drill-down is rail state, not session state, so resetSessionState
  // does not clear it -- a leaked expansion would make the next case's
  // "not showing yet" assertion pass or fail by test order.
  for (const k of useJaato.getState().budgetExpanded) useJaato.getState().toggleBudgetSource(k);
});

describe("the store reads the instruction budget", () => {
  it("keeps the daemon's snapshot by agent", () => {
    useJaato.getState().dispatch([budgetEvent()]);
    const b = useJaato.getState().budget[MAIN_AGENT]!;
    expect(b.contextLimit).toBe(200_000);
    expect(b.totalTokens).toBe(33_000);
    expect(Object.keys(b.entries)).toEqual(["system", "plugin", "conversation"]);
    expect(b.entries.plugin!.children!.memory!.total_tokens).toBe(8_000);
  });

  it("IGNORES a snapshot with no entries rather than blanking the panel", () => {
    useJaato.getState().dispatch([budgetEvent()]);
    useJaato.getState().dispatch([budgetEvent({ entries: {} })]);
    expect(Object.keys(useJaato.getState().budget[MAIN_AGENT]!.entries)).toHaveLength(3);
  });
});

describe("the panel", () => {
  it("lists the source layers with what each costs", () => {
    useJaato.getState().dispatch([budgetEvent()]);
    render(<BudgetPanel agentId={MAIN_AGENT} />);
    expect(screen.getByText("System")).toBeTruthy();
    expect(screen.getByText("Plugin")).toBeTruthy();
    expect(screen.getByText("Conversation")).toBeTruthy();
    // 12_000 renders through the same `fmt` the context rows use.
    expect(screen.getAllByText("12.0k").length).toBeGreaterThan(0);
  });

  it("drills into a source that has children, and only that one", () => {
    useJaato.getState().dispatch([budgetEvent()]);
    render(<BudgetPanel agentId={MAIN_AGENT} />);
    expect(screen.queryByText("memory")).toBeNull();
    // `system` has no children, so it is not a control at all.
    expect(screen.getByText("System").closest("button")).toBeNull();
    const plugin = screen.getByText("Plugin").closest("button")!;
    expect(plugin.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(plugin);
    expect(screen.getByText("memory")).toBeTruthy();
    expect(screen.getByText("cli")).toBeTruthy();
  });

  it("renders the daemon's own glyph, never one of its own", () => {
    useJaato.getState().dispatch([
      budgetEvent({ entries: { system: { source: "system", total_tokens: 1, gc_policy: "locked", indicator: "!" } } }),
    ]);
    render(<BudgetPanel agentId={MAIN_AGENT} />);
    // The daemon said "!", so "!" is what shows -- the local POLICY_GLYPH
    // table is a fallback for a snapshot that carries none, not an override.
    // The row's glyph carries the policy explained (#1190); it comes before
    // the legend, which carries the same tooltip.
    expect(screen.getAllByTitle(POLICY_EXPLAINED.locked!)[0]!.textContent).toBe("!");
  });

  it("still shows the context readout beneath it", () => {
    useJaato.getState().dispatch([
      budgetEvent(),
      ev({ type: "context.updated", agent_id: "main", usage: { total_tokens: 33_000, prompt_tokens: 32_900 },
           context_limit: 200_000, percent_used: 16.5, turns: 4 }),
    ]);
    render(<BudgetPanel agentId={MAIN_AGENT} />);
    expect(screen.getByText("Context")).toBeTruthy();
    expect(screen.getByText("16.5%")).toBeTruthy();
    expect(screen.getByText("remaining")).toBeTruthy();
  });

  it("says WHICH readout is missing when the daemon has reported no budget", () => {
    useJaato.getState().dispatch([
      ev({ type: "context.updated", agent_id: "main", usage: { total_tokens: 10 }, context_limit: 1000, percent_used: 1 }),
    ]);
    render(<BudgetPanel agentId={MAIN_AGENT} />);
    expect(screen.getByText(/No instruction budget reported yet/)).toBeTruthy();
    expect(screen.getByText("Context")).toBeTruthy();
  });
});

describe("the GC line (#1190)", () => {
  const PASS_AT = "2026-09-22T14:32:00.000Z";

  it("shows the last pass with its OWN time and what it freed", () => {
    useJaato.getState().dispatch([
      budgetEvent(),
      ev({ type: "gc", agent_id: "main", phase: "started", strategy: "budget" }),
      ev({ type: "gc", agent_id: "main", phase: "completed", success: true, tokens_freed: 14_200, tokens_before: 90_000, tokens_after: 75_800, trigger_reason: "threshold", timestamp: PASS_AT }),
    ]);
    const pass = useJaato.getState().gc[MAIN_AGENT]!.lastPass!;
    // The event's timestamp, not the moment it was reduced: a replay on attach
    // carries the pass's own time, and "when did GC last run" answered with
    // the attach time would be a readout that lies.
    expect(pass.at).toBe(Date.parse(PASS_AT));
    render(<BudgetPanel agentId={MAIN_AGENT} />);
    expect(screen.getByTestId("gc-summary").textContent).toMatch(/last GC .* · freed 14\.2k tokens/);
  });

  it("says a pass is running between started and completed", () => {
    useJaato.getState().dispatch([budgetEvent(), ev({ type: "gc", agent_id: "main", phase: "started" })]);
    render(<BudgetPanel agentId={MAIN_AGENT} />);
    expect(screen.getByTestId("gc-summary").textContent).toContain("collecting");
  });

  it("shows the policy in force, and a session with none as a warning, not as nothing", () => {
    useJaato.getState().dispatch([budgetEvent(), ev({ type: "gc.config", agent_id: "main", strategy: "budget", threshold: 80, target_percent: 60 })]);
    const { unmount } = render(<BudgetPanel agentId={MAIN_AGENT} />);
    expect(screen.getByTestId("gc-summary").textContent).toContain("GC: budget · runs at 80% · down to 60%");
    unmount();
    useJaato.getState().dispatch([ev({ type: "gc.config", agent_id: "main", strategy: null })]);
    render(<BudgetPanel agentId={MAIN_AGENT} />);
    expect(screen.getByTestId("gc-summary").textContent).toContain("GC: none configured");
  });

  it("the control: with nothing reported, it says so rather than 'never collected'", () => {
    useJaato.getState().dispatch([budgetEvent()]);
    render(<BudgetPanel agentId={MAIN_AGENT} />);
    const text = screen.getByTestId("gc-summary").textContent ?? "";
    expect(text).toContain("no GC pass reported yet");
    expect(text).not.toContain("GC:");
  });
});
