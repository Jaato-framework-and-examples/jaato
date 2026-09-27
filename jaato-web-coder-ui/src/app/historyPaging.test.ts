/**
 * Scroll-back through the daemon's paged history (protocol 1.28): the
 * unit -> block conversion, the reducer's replace-vs-prepend rule, and the
 * request's guards.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { JaatoEvent } from "@jaato/sdk";
import { MAIN_AGENT, useJaato } from "@/store/store";
import { historyPageBlocks } from "@/protocol/history";

const ev = (o: Record<string, unknown>) => o as unknown as JaatoEvent;
let n = 0;
const nextId = () => `t${++n}`;

const units = [
  { id: "0:a", kind: "user", group: "m0", turn: 0, text: "hi" },
  { id: "1:b", kind: "thinking", group: "m1p0t", turn: 0, text: "hmm" },
  { id: "2:c", kind: "model", group: "m1p0", turn: 0, text: "Intro.\n\n" },
  { id: "3:d", kind: "model", group: "m1p0", turn: 0, text: "<j-code language=\"py\">\n</j-code>\n" },
  { id: "4:e", kind: "tools", group: "m1c", turn: 0, tools: [
    { call_id: "c1", tool_name: "readFile", tool_args: { path: "x" }, success: true },
    { call_id: "c2", tool_name: "cli_based_tool", tool_args: {}, success: false },
    { call_id: "c3", tool_name: "grep", tool_args: {}, success: null },
  ] },
  { id: "5:f", kind: "model", group: "m3p0", turn: 0, text: "Done." },
];

describe("historyPageBlocks", () => {
  it("joins one text part's segments, keeps thinking apart, and draws tool outcomes", () => {
    const blocks = historyPageBlocks(units, MAIN_AGENT, nextId, false);
    expect(blocks.map((b) => b.kind)).toEqual(["user", "text", "text", "tool", "tool", "tool", "text"]);
    const [user, thinking, reply] = blocks;
    expect(user).toMatchObject({ kind: "user", text: "hi", echoed: true });
    expect(thinking).toMatchObject({ source: "thinking", text: "hmm" });
    expect(reply).toMatchObject({ source: "model", text: "Intro.\n\n<j-code language=\"py\">\n</j-code>\n" });
    const tools = blocks.filter((b) => b.kind === "tool");
    expect(tools.map((t) => (t.kind === "tool" ? t.status : ""))).toEqual(["success", "error", "success"]);
    // a failed call opens so its failure is visible, as the live tree does
    expect(tools.map((t) => (t.kind === "tool" ? t.expanded : null))).toEqual([false, true, false]);
  });
});

describe("reduce — history.page", () => {
  beforeEach(() => useJaato.getState().resetSessionState());

  it("the attach page replaces the transcript; an older page is prepended", () => {
    const d = useJaato.getState().dispatch;
    d([ev({ type: "agent.output", agent_id: "main", source: "model", text: "stale", mode: "write" })]);
    d([ev({ type: "history.page", agent_id: "main", request_id: "", units: units.slice(4), before: "4:e", has_more: true })]);
    let s = useJaato.getState();
    expect(s.blocks[MAIN_AGENT]!.some((b) => b.kind === "text" && b.text === "stale")).toBe(false);
    expect(s.historyPaging[MAIN_AGENT]).toMatchObject({ before: "4:e", hasMore: true, loading: false });

    d([ev({ type: "history.page", agent_id: "main", request_id: "hist-1", units: units.slice(0, 4), before: "", has_more: false })]);
    s = useJaato.getState();
    const kinds = s.blocks[MAIN_AGENT]!.map((b) => b.kind);
    expect(kinds[0]).toBe("user");
    expect(kinds[kinds.length - 1]).toBe("text");
    expect(s.historyPaging[MAIN_AGENT]).toMatchObject({ hasMore: false });
  });

  it("a stale answer stops scroll-back and says so, changing nothing on screen", () => {
    const d = useJaato.getState().dispatch;
    d([ev({ type: "history.page", agent_id: "main", request_id: "", units: units.slice(5), before: "5:f", has_more: true })]);
    const before = useJaato.getState().blocks[MAIN_AGENT];
    d([ev({ type: "history.page", agent_id: "main", request_id: "hist-2", units: [], stale: true })]);
    const s = useJaato.getState();
    expect(s.blocks[MAIN_AGENT]).toBe(before);
    expect(s.historyPaging[MAIN_AGENT]).toMatchObject({ hasMore: false });
    expect(s.historyPaging[MAIN_AGENT]!.error).toMatch(/changed/);
  });

  it("a page ends a pending replay-mode history.request", () => {
    useJaato.getState().setHistoryMode("replay");
    useJaato.getState().dispatch([ev({ type: "history.page", agent_id: "main", request_id: "", units: [] })]);
    expect(useJaato.getState().historyMode).toBe("listing");
  });
});

describe("loadOlderHistory", () => {
  beforeEach(() => useJaato.getState().resetSessionState());

  it("asks with the stored cursor, once, and records a failure instead of throwing", async () => {
    const requestHistoryPage = vi.fn().mockRejectedValue(new Error("boom"));
    vi.doMock("@/sdk/connection", () => ({ getClient: () => ({ requestHistoryPage }), isConnected: () => true }));
    vi.resetModules();
    const { useJaato: store } = await import("@/store/store");
    const { loadOlderHistory } = await import("./historyPaging");
    store.setState((s) => ({ connection: { ...s.connection, protocolVersion: "1.28" } }));
    store.getState().setHistoryPaging(MAIN_AGENT, { before: "7:x", hasMore: true });
    await Promise.all([loadOlderHistory(), loadOlderHistory()]);
    expect(requestHistoryPage).toHaveBeenCalledTimes(1);
    expect(requestHistoryPage.mock.calls[0]![0]).toMatchObject({ agentId: MAIN_AGENT, before: "7:x" });
    expect(store.getState().historyPaging[MAIN_AGENT]).toMatchObject({ loading: false, error: "boom" });
    vi.doUnmock("@/sdk/connection");
  });

  it("does nothing against a daemon below 1.28", async () => {
    const requestHistoryPage = vi.fn();
    vi.doMock("@/sdk/connection", () => ({ getClient: () => ({ requestHistoryPage }), isConnected: () => true }));
    vi.resetModules();
    const { useJaato: store } = await import("@/store/store");
    const { loadOlderHistory } = await import("./historyPaging");
    store.setState((s) => ({ connection: { ...s.connection, protocolVersion: "1.27" } }));
    store.getState().setHistoryPaging(MAIN_AGENT, { before: "7:x", hasMore: true });
    await loadOlderHistory();
    expect(requestHistoryPage).not.toHaveBeenCalled();
    vi.doUnmock("@/sdk/connection");
  });
});
