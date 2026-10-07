import { describe, expect, it } from "vitest";
import { describeToolCalls, summarizeToolCalls } from "./turnStats";

describe("summarizeToolCalls", () => {
  it("reads the per-call records the daemon actually sends", () => {
    const s = summarizeToolCalls([
      { name: "readFile", start_time: "t0", end_time: "t1", duration_seconds: 0.2 },
      { name: "cli_based_tool", start_time: "t1", end_time: "t2", duration_seconds: 1.4 },
      { name: "readFile", start_time: "t2", end_time: "t3", duration_seconds: 0.1 },
    ])!;
    expect(s.count).toBe(3);
    expect(s.byName).toEqual([["readFile", 2], ["cli_based_tool", 1]]);
    expect(describeToolCalls(s)).toBe("3 tool calls (readFile ×2, cli_based_tool)");
  });

  it("still accepts a bare count and says nothing for an absent field", () => {
    expect(describeToolCalls(summarizeToolCalls(1)!)).toBe("1 tool call");
    expect(describeToolCalls(summarizeToolCalls([])!)).toBe("no tool calls");
    expect(summarizeToolCalls(undefined)).toBeNull();
    expect(summarizeToolCalls(null)).toBeNull();
    expect(summarizeToolCalls("nope")).toBeNull();
  });
});
