/**
 * The minimap's geometry: how an item becomes bars, how a laid-out item
 * is placed at its slot, and how a point on the rail maps back to a
 * scroll position and a turn. The rail component is layout and a canvas;
 * these are the parts that can be wrong in a way a person notices (a
 * tap that lands on the wrong turn, a red failure drawn green).
 */
import { describe, expect, it } from "vitest";
import type { ToolBlock } from "@/store/types";
import type { TranscriptItem } from "@/store/transcript";
import { layoutItem, layoutSignature, placeItems, scrollTopFor, scrubLabel, turnAt, wrapWords, type Measure } from "./minimap";

/** Every character is 10px wide, whatever the font. */
const measure: Measure = (text) => text.length * 10;

function tool(status: ToolBlock["status"], extra: Partial<ToolBlock> = {}): ToolBlock {
  return { id: `b-${status}`, kind: "tool", agentId: "main", callId: "c", toolName: "run", args: { command: "ls -la" }, status, startedAt: 0, output: "", media: [], expanded: false, ...extra };
}

describe("wrapWords", () => {
  it("puts one bar per word and wraps greedily at the width", () => {
    const r = wrapWords("aa bbb cc", { x0: 0, y0: 0, maxWidth: 60, font: "f", lineHeight: 20, tone: "ink" }, measure);
    // "aa" (20) + space (10) + "bbb" (30) = 60 fits; "cc" wraps.
    expect(r.bars.map((b) => [b.x, b.y, b.w])).toEqual([[0, 5, 20], [30, 5, 30], [0, 25, 20]]);
    expect(r.bars.every((b) => b.h === 10)).toBe(true);
    expect(r.bottom).toBe(40);
  });

  it("keeps a hard newline and an empty line, and cuts rather than wraps under nowrap", () => {
    const r = wrapWords("a\n\nb", { x0: 0, y0: 0, maxWidth: 100, font: "f", lineHeight: 10, tone: "ink" }, measure);
    expect(r.bars.map((b) => b.y)).toEqual([2.5, 22.5]);
    const cut = wrapWords("aaaa bbbb cccc", { x0: 0, y0: 0, maxWidth: 90, font: "f", lineHeight: 10, tone: "ink", nowrap: true }, measure);
    expect(cut.bars).toHaveLength(2);
    expect(cut.bottom).toBe(10);
  });
});

describe("layoutItem", () => {
  it("draws a failed tool row's glyph in the error tone and a finished one in success", () => {
    const row = (s: ToolBlock["status"], err = false): TranscriptItem => ({ kind: "toolGroup", id: "t", agentId: "main", mode: "row", toolClass: "other", calls: [tool(s, { isErrorResult: err })] });
    expect(layoutItem(row("error"), 600, measure).bars[0]!.tone).toBe("error");
    expect(layoutItem(row("success", true), 600, measure).bars[0]!.tone).toBe("error");
    expect(layoutItem(row("success"), 600, measure).bars[0]!.tone).toBe("success");
    expect(layoutItem(row("running"), 600, measure).bars[0]!.tone).toBe("muted");
    // one line: glyph, name, arguments
    const bars = layoutItem(row("success"), 600, measure).bars;
    expect(new Set(bars.map((b) => b.y)).size).toBe(1);
    expect(bars.map((b) => b.tone)).toEqual(["success", "ink", "muted", "muted"]);
  });

  it("marks a user message as a card, right-aligned like the pane", () => {
    const layout = layoutItem({ kind: "userMessage", id: "u", agentId: "main", text: "hi there", turn: 1 }, 600, measure);
    expect(layout.card).toBe(true);
    const right = Math.max(...layout.bars.map((b) => b.x + b.w));
    expect(right).toBeGreaterThan(550);
  });

  it("draws fenced code muted and prose in ink", () => {
    const layout = layoutItem({ kind: "assistantText", id: "a", agentId: "main", text: "see\n```\ncode here\n```\ndone" }, 600, measure);
    expect(layout.bars.map((b) => b.tone)).toEqual(["ink", "muted", "muted", "ink"]);
  });
});

describe("layoutSignature", () => {
  it("changes when a streaming text grows or a tool finishes, and not otherwise", () => {
    const a: TranscriptItem = { kind: "assistantText", id: "a", agentId: "main", text: "abc" };
    expect(layoutSignature(a, 600)).toBe(layoutSignature({ ...a }, 600));
    expect(layoutSignature(a, 600)).not.toBe(layoutSignature({ ...a, text: "abcd" }, 600));
    expect(layoutSignature(a, 600)).not.toBe(layoutSignature(a, 500));
    const t = (s: ToolBlock["status"]): TranscriptItem => ({ kind: "toolGroup", id: "t", agentId: "main", mode: "row", toolClass: "other", calls: [tool(s)] });
    expect(layoutSignature(t("running"), 600)).not.toBe(layoutSignature(t("success"), 600));
  });
});

describe("placeItems", () => {
  it("offsets each layout by its slot and squeezes one taller than its slot", () => {
    const tall = { bars: [{ x: 0, y: 80, w: 5, h: 20, tone: "ink" as const }], height: 200, card: true };
    const short = { bars: [{ x: 0, y: 10, w: 5, h: 4, tone: "ink" as const }], height: 30, card: false };
    const out = placeItems([tall, short], [{ start: 0, size: 100 }, { start: 100, size: 64 }], 12);
    expect(out.bars[0]).toMatchObject({ y: 12 + 40, h: 10 });
    expect(out.bars[1]).toMatchObject({ y: 12 + 100 + 10, h: 4 });
    expect(out.cards).toEqual([{ y: 12, h: 100 }]);
  });
});

describe("scrolling from the rail", () => {
  it("centres the tapped point, clamped to the scroll range", () => {
    expect(scrollTopFor(0.5, 1000, 200)).toBe(400);
    expect(scrollTopFor(0, 1000, 200)).toBe(0);
    expect(scrollTopFor(1, 1000, 200)).toBe(800);
    expect(scrollTopFor(1.4, 1000, 200)).toBe(800);
  });

  it("finds the last turn whose top is at or above the point", () => {
    const turns = [{ top: 0, turn: 1 }, { top: 300, turn: 2 }, { top: 700, turn: 3 }];
    expect(turnAt(turns, 299)?.turn).toBe(1);
    expect(turnAt(turns, 300)?.turn).toBe(2);
    expect(turnAt(turns, 5000)?.turn).toBe(3);
    expect(turnAt([{ top: 50, turn: 1 }], 10)).toBeNull();
  });

  it("labels a turn with two digits and the first line of its message", () => {
    expect(scrubLabel(5, "\n  Is the subagent stuck?\nmore")).toEqual({ number: "05", text: "Is the subagent stuck?" });
  });
});
