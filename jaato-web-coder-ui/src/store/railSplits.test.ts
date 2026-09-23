/**
 * The rail's split arithmetic (#1244).  The load-bearing properties the issue
 * names: moving a boundary changes EXACTLY two shares (split-pane — the others
 * don't move), the per-section minimum is respected, opening or closing a
 * section redistributes its share, and an unknown or removed section id in
 * saved state is ignored.
 */
import { describe, expect, it } from "vitest";
import {
  RAIL_SECTION_MIN_PX,
  boundaryPercent,
  moveBoundary,
  resetPair,
  sanitizeSplits,
  sharesFor,
  type RailSectionId,
} from "./railSplits";

describe("moveBoundary changes exactly the two neighbours", () => {
  it("moves height between the pair and leaves every other section put", () => {
    // Four open sections, equal weights.  Drag the plan/budget boundary down;
    // files and sessions must not move.
    const before = { plan: 1, budget: 1, files: 1, sessions: 1 };
    const after = moveBoundary(before, "plan", "budget", 200, 200, 40);
    expect(after.files).toBe(before.files);
    expect(after.sessions).toBe(before.sessions);
    // The pair's combined weight is preserved.
    expect(after.plan! + after.budget!).toBeCloseTo(before.plan + before.budget, 10);
    // Dragging DOWN grew the section above.
    expect(after.plan!).toBeGreaterThan(before.plan);
    expect(after.budget!).toBeLessThan(before.budget);
  });

  it("only two keys differ from the input", () => {
    const before = { plan: 2, budget: 1, files: 3, sessions: 1 };
    const after = moveBoundary(before, "budget", "files", 100, 300, -50);
    const changed = (Object.keys(after) as RailSectionId[]).filter((k) => after[k] !== before[k as RailSectionId]);
    expect(changed.sort()).toEqual(["budget", "files"]);
  });
});

describe("the per-section minimum is respected", () => {
  it("clamps a drag that would collapse the section below", () => {
    // Push the boundary far past the bottom neighbour's floor.
    const after = moveBoundary({ plan: 1, budget: 1 }, "plan", "budget", 300, 100, 10_000, RAIL_SECTION_MIN_PX);
    // above ends at combinedPx - minPx = 400 - 72 = 328 of 400.
    const combined = after.plan! + after.budget!;
    expect(combined).toBeCloseTo(2, 10);
    expect(after.plan! / combined).toBeCloseTo(328 / 400, 6);
    expect(after.budget! / combined).toBeCloseTo(72 / 400, 6);
  });

  it("refuses to move a pair with no room for two minimums", () => {
    const before = { plan: 1, budget: 1 };
    // combinedPx (100) < 2 * minPx (144): unchanged.
    expect(moveBoundary(before, "plan", "budget", 50, 50, 40)).toBe(before);
  });

  it("a degenerate measurement is returned unchanged", () => {
    const before = { plan: 1, budget: 1 };
    expect(moveBoundary(before, "plan", "budget", 0, 0, 40)).toBe(before);
  });
});

describe("sharesFor redistributes on open and close", () => {
  it("normalises to fractions over the open set", () => {
    const s = sharesFor({ plan: 1, budget: 1 }, ["plan", "budget"]);
    expect(s.plan).toBeCloseTo(0.5, 10);
    expect(s.budget).toBeCloseTo(0.5, 10);
  });

  it("opening a section shrinks the others proportionally", () => {
    const two = sharesFor({}, ["plan", "budget"]);
    const three = sharesFor({}, ["plan", "budget", "files"]);
    expect(two.plan).toBeCloseTo(0.5, 10);
    expect(three.plan).toBeCloseTo(1 / 3, 10);
    // plan and budget shrank by the same factor; their ratio is unchanged.
    expect(three.plan! / three.budget!).toBeCloseTo(two.plan! / two.budget!, 10);
  });

  it("closing a section hands its share back to the rest proportionally", () => {
    const weights = { plan: 2, budget: 1, files: 1 };
    const three = sharesFor(weights, ["plan", "budget", "files"]);
    const two = sharesFor(weights, ["plan", "budget"]);
    // plan:budget ratio survives the close (files' share redistributed by weight).
    expect(two.plan! / two.budget!).toBeCloseTo(three.plan! / three.budget!, 10);
    expect(two.plan! + two.budget!).toBeCloseTo(1, 10);
  });

  it("an empty open set yields no shares", () => {
    expect(sharesFor({ plan: 1 }, [])).toEqual({});
  });

  it("a section with no remembered weight opens at an equal share", () => {
    // budget has a saved weight, plan does not: plan opens at the default (1).
    const s = sharesFor({ budget: 1 }, ["plan", "budget"]);
    expect(s.plan).toBeCloseTo(0.5, 10);
  });
});

describe("resetPair levels a pair", () => {
  it("splits the pair's combined weight in half and leaves others alone", () => {
    const after = resetPair({ plan: 3, budget: 1, files: 5 }, "plan", "budget");
    expect(after.plan).toBe(2);
    expect(after.budget).toBe(2);
    expect(after.files).toBe(5);
  });
});

describe("saved state with an unknown or removed id is ignored", () => {
  it("drops ids that are not rail sections", () => {
    const clean = sanitizeSplits({ plan: 1, budget: 2, gone: 4, tools: 9 });
    expect(clean).toEqual({ plan: 1, budget: 2 });
  });

  it("drops non-positive and non-finite weights", () => {
    expect(sanitizeSplits({ plan: 0, budget: -1, files: NaN, sessions: 2 })).toEqual({ sessions: 2 });
  });

  it("a removed id in saved state does not scramble the open sections", () => {
    // ``budget`` was persisted but is not open now; ``sharesFor`` ignores it.
    const s = sharesFor({ plan: 1, budget: 5 }, ["plan", "files"]);
    expect(s.plan).toBeCloseTo(0.5, 10);
    expect(s.files).toBeCloseTo(0.5, 10);
    expect(s.budget).toBeUndefined();
  });

  it("non-object saved state is empty", () => {
    expect(sanitizeSplits(null)).toEqual({});
    expect(sanitizeSplits("nonsense")).toEqual({});
  });
});

describe("boundaryPercent is the pair's above-share", () => {
  it("reports the above section's percent of the pair", () => {
    expect(boundaryPercent({ plan: 3, budget: 1 }, "plan", "budget")).toBe(75);
    expect(boundaryPercent({ plan: 1, budget: 1 }, "plan", "budget")).toBe(50);
  });
});
