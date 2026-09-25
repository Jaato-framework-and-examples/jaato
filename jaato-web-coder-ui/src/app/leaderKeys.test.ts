import { describe, expect, it } from "vitest";
import { agentForDigit, resolveLeaderKey } from "./leaderKeys";

const ctx = (over: Partial<Parameters<typeof resolveLeaderKey>[1]> = {}) => ({
  agentOrder: ["main"],
  selectedAgentId: "main",
  runningToolCallIds: [],
  popupCallId: null,
  ...over,
});

describe("resolveLeaderKey: the rail toggles", () => {
  it("P/B/F map to the Plan/Budget/Files rail sections, case-insensitively", () => {
    expect(resolveLeaderKey("p", ctx())).toEqual({ kind: "setActivePanel", id: "plan" });
    expect(resolveLeaderKey("B", ctx())).toEqual({ kind: "setActivePanel", id: "budget" });
    expect(resolveLeaderKey("f", ctx())).toEqual({ kind: "setActivePanel", id: "files" });
  });
  it("T toggles tool-call boxes", () => {
    expect(resolveLeaderKey("t", ctx())).toEqual({ kind: "toggleTools" });
  });
});

describe("resolveLeaderKey: agent tabs", () => {
  const three = ["main", "sub-a", "sub-b"];
  it("] steps to the next tab, wrapping", () => {
    expect(resolveLeaderKey("]", ctx({ agentOrder: three, selectedAgentId: "main" }))).toEqual({ kind: "selectAgent", id: "sub-a" });
    expect(resolveLeaderKey("]", ctx({ agentOrder: three, selectedAgentId: "sub-b" }))).toEqual({ kind: "selectAgent", id: "main" });
  });
  it("[ steps to the previous tab, wrapping the other way", () => {
    expect(resolveLeaderKey("[", ctx({ agentOrder: three, selectedAgentId: "main" }))).toEqual({ kind: "selectAgent", id: "sub-b" });
    expect(resolveLeaderKey("[", ctx({ agentOrder: three, selectedAgentId: "sub-a" }))).toEqual({ kind: "selectAgent", id: "main" });
  });
  it("stepping with only one tab does nothing (not one leader key falling through to the next)", () => {
    expect(resolveLeaderKey("]", ctx({ agentOrder: ["main"] }))).toBeNull();
    expect(resolveLeaderKey("[", ctx({ agentOrder: ["main"] }))).toBeNull();
  });
  it("1-9 jump straight to a tab; a digit past the end of the strip is unhandled", () => {
    expect(resolveLeaderKey("2", ctx({ agentOrder: three }))).toEqual({ kind: "selectAgent", id: "sub-a" });
    expect(resolveLeaderKey("1", ctx({ agentOrder: three }))).toEqual({ kind: "selectAgent", id: "main" });
    expect(resolveLeaderKey("9", ctx({ agentOrder: three }))).toBeNull();
  });
  it("agentForDigit is 1-based and answers null past the strip", () => {
    expect(agentForDigit(three, 1)).toBe("main");
    expect(agentForDigit(three, 3)).toBe("sub-b");
    expect(agentForDigit(three, 4)).toBeNull();
  });
});

describe("resolveLeaderKey: the live-output popup", () => {
  it("O steps to the next running tool, wrapping, when nothing is pinned yet", () => {
    expect(resolveLeaderKey("o", ctx({ runningToolCallIds: ["a", "b"], popupCallId: null }))).toEqual({ kind: "cyclePopup", callId: "a" });
  });
  it("O steps forward from the pinned call, wrapping to the first", () => {
    expect(resolveLeaderKey("o", ctx({ runningToolCallIds: ["a", "b"], popupCallId: "a" }))).toEqual({ kind: "cyclePopup", callId: "b" });
    expect(resolveLeaderKey("o", ctx({ runningToolCallIds: ["a", "b"], popupCallId: "b" }))).toEqual({ kind: "cyclePopup", callId: "a" });
  });
  it("O with no running tool does nothing", () => {
    expect(resolveLeaderKey("o", ctx({ runningToolCallIds: [] }))).toBeNull();
  });
});

describe("resolveLeaderKey: everything else falls through", () => {
  it("a key outside the vocabulary is unhandled, for the palette to treat as search text", () => {
    for (const k of ["a", "x", "z", "!", " "]) expect(resolveLeaderKey(k, ctx())).toBeNull();
  });
});
