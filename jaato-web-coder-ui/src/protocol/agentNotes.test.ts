/**
 * Output meant for the agent: which sources collapse, how a run of them
 * reads as one line, and how a history page keeps a subagent's report
 * out of the user's turns.
 */
import { describe, expect, it } from "vitest";
import { buildTranscript } from "@/store/transcript";
import type { OutputBlock } from "@/store/types";
import { collapsedSummary, reportHeaders } from "./agentNotes";
import { historyPageBlocks } from "./history";

let seq = 0;
const nextId = () => `n${++seq}`;
const text = (source: string, body: string): OutputBlock => ({ id: nextId(), kind: "text", agentId: "main", source, text: body });
const user = (body: string): OutputBlock => ({ id: nextId(), kind: "user", agentId: "main", text: body, echoed: true });

const DONE = "[SUBAGENT agent_id=subagent_1 event=COMPLETED]\nFound the bug in parser.py.";
const ASK = "[SUBAGENT agent_id=subagent_2 event=CLARIFICATION_REQUESTED]\nWhich branch?";

describe("reportHeaders", () => {
  it("reads every header in a batch, event or forwarded source", () => {
    expect(reportHeaders(`${DONE}\n\n[SUBAGENT agent_id=s3 source=model]\nhi`)).toEqual([
      { agentId: "subagent_1", event: "COMPLETED", source: "" },
      { agentId: "s3", event: "", source: "model" },
    ]);
  });
});

describe("collapsedSummary", () => {
  it("names the subagent as the tabs do, with its event and first line", () => {
    expect(collapsedSummary("child", [DONE], { subagent_1: "researcher" }))
      .toBe("Subagent · researcher completed · Found the bug in parser.py.");
  });

  it("counts a run of reports", () => {
    expect(collapsedSummary("child", [DONE, ASK])).toBe(
      "Subagent · 2 reports · subagent_1 completed, subagent_2 clarification requested · Which branch?");
  });

  it("shows a plan run by its last line", () => {
    expect(collapsedSummary("plan", ["Plan created: Fix it"])).toBe("Plan · Plan created: Fix it");
    expect(collapsedSummary("plan", ["Plan created: Fix it", "[1] Run probes: done"])).toBe("Plan · 2 updates · [1] Run probes: done");
  });
});

describe("the transcript", () => {
  it("folds a run of one collapsed source into one item that takes no turn", () => {
    const items = buildTranscript([user("go"), text("child", DONE), text("child", ASK), text("plan", "Plan created: x"), text("model", "ok"), user("next")], {});
    expect(items.map((i) => i.kind)).toEqual(["userMessage", "collapsedNote", "collapsedNote", "assistantText", "userMessage"]);
    const [, child, plan] = items;
    expect(child).toMatchObject({ source: "child", texts: [DONE, ASK] });
    expect(plan).toMatchObject({ source: "plan", texts: ["Plan created: x"] });
    expect(items.filter((i) => i.kind === "userMessage").map((i) => (i as { turn: number }).turn)).toEqual([1, 2]);
  });
});

describe("a history page", () => {
  it("draws a unit the server marks as a subagent report as a report, not a user turn", () => {
    const blocks = historyPageBlocks([
      { kind: "user", text: "go" },
      { kind: "user", text: DONE, origin: "subagent" },
    ], "main", nextId, false);
    expect(blocks.map((b) => [b.kind, b.kind === "text" ? b.source : ""])).toEqual([["user", ""], ["text", "child"]]);
  });
});
