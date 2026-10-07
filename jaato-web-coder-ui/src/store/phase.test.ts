import { describe, expect, it } from "vitest";
import { clampStallThreshold, DEFAULT_STALL_THRESHOLD_MS, MAX_STALL_THRESHOLD_MS, MIN_STALL_THRESHOLD_MS, stalled, type AgentPhase } from "./phase";

describe("clampStallThreshold", () => {
  it("clamps to the documented 30s-300s range", () => {
    expect(clampStallThreshold(1000)).toBe(MIN_STALL_THRESHOLD_MS);
    expect(clampStallThreshold(10_000_000)).toBe(MAX_STALL_THRESHOLD_MS);
    expect(clampStallThreshold(60_000)).toBe(60_000);
  });
  it("falls back to the default for a non-finite value", () => {
    expect(clampStallThreshold(NaN)).toBe(DEFAULT_STALL_THRESHOLD_MS);
    expect(clampStallThreshold(Infinity)).toBe(DEFAULT_STALL_THRESHOLD_MS);
  });
});

describe("stalled — #1304 §3", () => {
  const thinking: AgentPhase = { kind: "thinking", since: 0 };
  const sending: AgentPhase = { kind: "sending", since: 0 };
  const tool: AgentPhase = { kind: "tool", since: 0, toolName: "cli_based_tool", running: 1 };
  const waiting: AgentPhase = { kind: "waiting", since: 0, on: "permission" };
  const idle: AgentPhase = { kind: "idle" };

  it("fires for thinking/sending once the threshold has passed since the last event", () => {
    expect(stalled(thinking, 0, 60_000, 60_000)).toMatchObject({ silentForMs: 60_000, thresholdMs: 60_000 });
    expect(stalled(sending, 0, 60_000, 60_000)).toMatchObject({ silentForMs: 60_000 });
  });
  it("does not fire before the threshold", () => {
    expect(stalled(thinking, 0, 60_000, 59_999)).toBeNull();
  });
  it("never fires for tool or waiting -- a running tool is activity, waiting already outranks it", () => {
    expect(stalled(tool, 0, 60_000, 999_999)).toBeNull();
    expect(stalled(waiting, 0, 60_000, 999_999)).toBeNull();
  });
  it("never fires for idle", () => {
    expect(stalled(idle, 0, 60_000, 999_999)).toBeNull();
  });
  it("falls back to the phase's own 'since' when no event has been observed at all", () => {
    const p: AgentPhase = { kind: "thinking", since: 1000 };
    expect(stalled(p, undefined, 60_000, 61_000)).toMatchObject({ silentForMs: 60_000 });
    expect(stalled(p, undefined, 60_000, 60_999)).toBeNull();
  });
  it("clamps a threshold outside the documented range before comparing", () => {
    // A threshold below the 30s floor cannot make a stall fire early.
    expect(stalled(thinking, 0, 1000, 10_000)).toBeNull();
    expect(stalled(thinking, 0, 1000, MIN_STALL_THRESHOLD_MS)).toMatchObject({ thresholdMs: MIN_STALL_THRESHOLD_MS });
  });
});
