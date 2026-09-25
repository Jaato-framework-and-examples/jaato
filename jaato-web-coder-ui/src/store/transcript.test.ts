import { describe, expect, it } from "vitest";
import { buildTranscript, firstSentence, thinkingElapsedSeconds, type ThinkingClock, type ToolGroupItem } from "./transcript";
import type { OutputBlock, ToolBlock, ToolStatus } from "./types";

const AGENT = "main";
let seq = 0;
const nextId = () => `b${++seq}`;

function user(text: string): OutputBlock {
  return { id: nextId(), kind: "user", agentId: AGENT, text, echoed: true };
}
function text(source: string, body: string): OutputBlock {
  return { id: nextId(), kind: "text", agentId: AGENT, source, text: body };
}
function tool(toolName: string, status: ToolStatus = "success", extra: Partial<ToolBlock> = {}): ToolBlock {
  return {
    id: nextId(), kind: "tool", agentId: AGENT, callId: nextId(), toolName, args: {}, status,
    startedAt: Date.now(), output: "", media: [], expanded: false, ...extra,
  };
}
function system(body: string, style = "info"): OutputBlock {
  return { id: nextId(), kind: "system", agentId: AGENT, text: body, style };
}

describe("buildTranscript", () => {
  it("numbers user turns starting at 1", () => {
    const items = buildTranscript([user("hi"), text("model", "hello"), user("again")], {});
    const turns = items.filter((i) => i.kind === "userMessage").map((i) => i.turn);
    expect(turns).toEqual([1, 2]);
  });

  it("splits thinking, assistant text and other sources into their own kinds", () => {
    const items = buildTranscript([text("thinking", "considering..."), text("model", "the answer"), text("permission", "note")], {});
    expect(items.map((i) => i.kind)).toEqual(["thinking", "assistantText", "systemNote"]);
    expect((items[2] as { source?: string }).source).toBe("permission");
  });

  it("promotes an error/warning system block to a banner, and leaves the rest as a note", () => {
    const items = buildTranscript([system("boom", "error"), system("careful", "warning"), system("fyi", "hint")], {});
    expect(items.map((i) => i.kind)).toEqual(["banner", "banner", "systemNote"]);
  });

  it("marks only the tail thinking block as streaming", () => {
    const blocks = [text("thinking", "one")];
    const [tip] = buildTranscript(blocks, {});
    expect(tip!.kind).toBe("thinking");
    expect((tip as { streaming: boolean }).streaming).toBe(true);

    const superseded = [...blocks, text("model", "done")];
    const [still] = buildTranscript(superseded, {});
    expect((still as { streaming: boolean }).streaming).toBe(false);
  });

  it("remembers when a thinking block was first seen across calls, and gives an unknown clock nothing", () => {
    const clock: ThinkingClock = new Map();
    const blocks = [text("thinking", "one")];
    const first = buildTranscript(blocks, {}, clock, 1_000);
    const later = buildTranscript(blocks, {}, clock, 55_000);
    expect((first[0] as { startedAt: number | null }).startedAt).toBe(1_000);
    expect((later[0] as { startedAt: number | null }).startedAt).toBe(1_000);
    expect(thinkingElapsedSeconds(later[0] as never, 55_000)).toBe(54);

    const fresh = buildTranscript(blocks, {});
    expect(thinkingElapsedSeconds(fresh[0] as never)).not.toBeNull();
    const noClockGiven = buildTranscript(blocks, {}, new Map(), 0);
    expect((noClockGiven[0] as { startedAt: number | null }).startedAt).toBe(0);
  });

  it("the fib.py shape: 6 housekeeping calls fold to one line, a write and an exec stay full rows, and a failure that recovered becomes one muted line", () => {
    const blocks: OutputBlock[] = [
      tool("createPlan"),
      tool("list_tools"),
      tool("list_tools"),
      tool("list_tools"),
      tool("subscribeToEvents"),
      tool("get_tool_schemas"),
      tool("writeNewFile", "success", { args: { path: "fib.py", content: "print(1)" } }),
      tool("cli_based_tool", "success", { args: { command: "python fib.py" } }),
      tool("setStepStatus", "error"),
      tool("setStepStatus", "success"),
    ];
    const items = buildTranscript(blocks, {});
    const groups = items.filter((i): i is ToolGroupItem => i.kind === "toolGroup");
    expect(groups).toHaveLength(4);

    const fold = groups.find((g) => g.mode === "fold")!;
    expect(fold.toolClass).toBe("housekeeping");
    expect(fold.calls).toHaveLength(6);
    expect(fold.label).toContain("6 internal calls");
    expect(fold.label).toContain("list_tools ×3");

    const rows = groups.filter((g) => g.mode === "row");
    expect(rows.map((r) => r.calls[0]!.toolName).sort()).toEqual(["cli_based_tool", "writeNewFile"]);

    const recovered = groups.find((g) => g.mode === "recovered")!;
    expect(recovered.calls.map((c) => c.status)).toEqual(["error", "success"]);
    expect(recovered.calls.every((c) => c.toolName === "setStepStatus")).toBe(true);
  });

  it("resolves a hashed tool id before classifying and folding it", () => {
    const blocks: OutputBlock[] = [tool("t_deadbeef"), tool("t_deadbeef")];
    const items = buildTranscript(blocks, { t_deadbeef: "createPlan" });
    const [group] = items.filter((i): i is ToolGroupItem => i.kind === "toolGroup");
    expect(group!.mode).toBe("fold");
    expect(group!.label).toContain("createPlan ×2");
  });

  it("prefers a call's own server-reported tool_class over the client table (jaato/#1304 phase 3)", () => {
    // "store_memory" has no opinion in the client table (classifyTool would
    // answer "other") -- the daemon's answer is what must win, so this
    // call folds into the housekeeping run rather than getting its own row.
    const blocks: OutputBlock[] = [
      tool("createPlan"),
      tool("store_memory", "success", { toolClass: "housekeeping" }),
    ];
    const items = buildTranscript(blocks, {});
    const groups = items.filter((i): i is ToolGroupItem => i.kind === "toolGroup");
    expect(groups).toHaveLength(1);
    expect(groups[0]!.mode).toBe("fold");
    expect(groups[0]!.toolClass).toBe("housekeeping");
    expect(groups[0]!.calls).toHaveLength(2);
  });

  it("falls back to the client table for a call the daemon reported no class for", () => {
    const blocks: OutputBlock[] = [tool("store_memory")];
    const items = buildTranscript(blocks, {});
    const [group] = items.filter((i): i is ToolGroupItem => i.kind === "toolGroup");
    expect(group!.mode).toBe("row");
    expect(group!.toolClass).toBe("other");
  });

  it("only pairs a failure with a LATER success of the SAME tool, not a different one", () => {
    const blocks: OutputBlock[] = [tool("readFile", "error"), tool("writeNewFile", "success"), tool("readFile", "success")];
    const items = buildTranscript(blocks, {});
    const groups = items.filter((i): i is ToolGroupItem => i.kind === "toolGroup");
    expect(groups).toHaveLength(2);
    expect(groups.find((g) => g.mode === "recovered")).toBeTruthy();
    expect(groups.find((g) => g.mode === "row" && g.calls[0]!.toolName === "writeNewFile")).toBeTruthy();
  });

  it("a failure that never recovers stays its own row, not folded away", () => {
    const blocks: OutputBlock[] = [tool("readFile", "error")];
    const items = buildTranscript(blocks, {});
    expect(items).toHaveLength(1);
    expect((items[0] as ToolGroupItem).mode).toBe("row");
    expect((items[0] as ToolGroupItem).calls[0]!.status).toBe("error");
  });

  it("keeps two separate tool runs apart when text falls between them", () => {
    const blocks: OutputBlock[] = [tool("readFile"), text("model", "ok"), tool("readFile")];
    const items = buildTranscript(blocks, {});
    expect(items.map((i) => i.kind)).toEqual(["toolGroup", "assistantText", "toolGroup"]);
  });
});

describe("firstSentence", () => {
  it("stops at the first sentence terminator", () => {
    expect(firstSentence("Considering the options. Then a second thought.")).toBe("Considering the options.");
  });

  it("caps a sentence with no terminator", () => {
    expect(firstSentence("x".repeat(200), 10)).toBe("xxxxxxxxx…");
  });

  it("falls back to the trimmed text when nothing matches", () => {
    expect(firstSentence("   ")).toBe("");
  });
});
