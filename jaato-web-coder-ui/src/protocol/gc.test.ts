import { describe, expect, it } from "vitest";
import { describeGcPolicy, describeLastPass, formatAgo } from "./gc";

describe("describeGcPolicy", () => {
  it("renders a threshold policy and a continuous one differently", () => {
    expect(describeGcPolicy({ strategy: "budget", threshold: 80, targetPercent: 60, continuous: false })).toBe("GC: budget · runs at 80% · down to 60%");
    expect(describeGcPolicy({ strategy: "hybrid", threshold: 80, targetPercent: 50, continuous: true })).toBe("GC: hybrid · after every turn above 50%");
  });
  it("tells 'not said' (null) from 'no strategy' (a sentence)", () => {
    expect(describeGcPolicy(undefined)).toBeNull();
    expect(describeGcPolicy({ strategy: null, threshold: null, targetPercent: null, continuous: false })).toMatch(/none configured/);
  });
});

describe("describeLastPass", () => {
  const now = Date.parse("2026-09-22T15:00:00Z");
  const pass = { at: Date.parse("2026-09-22T14:48:00Z"), success: true, tokensFreed: 900, tokensBefore: null, tokensAfter: null, trigger: null, strategy: null, error: null };
  it("says how long ago and how much", () => {
    expect(describeLastPass({ lastPass: pass }, now)).toBe("last GC 12 min ago · freed 900 tokens");
  });
  it("says a failed pass failed", () => {
    expect(describeLastPass({ lastPass: { ...pass, success: false, error: "boom" } }, now)).toBe("last GC 12 min ago failed: boom");
  });
});

it("formatAgo", () => {
  const now = 1_000_000_000;
  expect(formatAgo(now - 10_000, now)).toBe("just now");
  expect(formatAgo(now - 5 * 3600_000, now)).toBe("5 h ago");
});
