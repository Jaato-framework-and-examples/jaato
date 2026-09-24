/**
 * The Memories rail's data module (#1232).
 *
 * Pinned here, each a way the section could mislead:
 *   • a failed list is REPORTED and keeps the rows it had -- an empty list
 *     reads as "nothing remembered";
 *   • an answer for a session this client has left is dropped;
 *   • a daemon below 1.22 is "unsupported", not "empty";
 *   • a refusal from the owner gate is shown in words, and the list is
 *     re-read afterwards (the daemon's store is the truth);
 *   • an edit sends only what CHANGED;
 *   • the refresh triggers: a successful memory write in any agent, the
 *     ``memory`` command's push, a session appearing -- and not our own
 *     correlated answers or unrelated tools.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useJaato } from "@/store/store";
import type { MemoryRow } from "@/store/types";

const listMemories = vi.fn();
const getMemory = vi.fn();
const updateMemory = vi.fn();
const approveMemory = vi.fn();
const dismissMemory = vi.fn();
const deleteMemory = vi.fn();
let protocol = "1.22";
let connected = true;

vi.mock("@/sdk/connection", () => ({
  isConnected: () => connected,
  getClient: () => ({
    get serverProtocolVersion() { return protocol; },
    listMemories, getMemory, updateMemory, approveMemory, dismissMemory, deleteMemory,
  }),
}));

const mem = await import("@/app/memories");

const row = (id: string, extra: Partial<MemoryRow> = {}): MemoryRow => ({
  id, description: `about ${id}`, tags: ["alpha", "beta"], maturity: "raw", tier: "workspace", ...extra,
});

beforeEach(() => {
  // Disconnected while the session is set, so the module's own "a session
  // appeared" trigger does not fire a timed refresh into the test.
  connected = false;
  useJaato.getState().resetSessionState();
  useJaato.setState({ sessionId: "s1" });
  for (const f of [listMemories, getMemory, updateMemory, approveMemory, dismissMemory, deleteMemory]) f.mockReset();
  protocol = "1.22";
  connected = true;
});

describe("the list", () => {
  it("loads rows and the owner gate's answer", async () => {
    listMemories.mockResolvedValue({ ok: true, memories: [row("a"), row("b", { maturity: "validated" })], may_curate: true });
    await mem.refreshMemories();
    const m = useJaato.getState().memories;
    expect(m.status).toBe("loaded");
    expect(m.rows.map((r) => r.id)).toEqual(["a", "b"]);
    expect(m.mayCurate).toBe(true);
    expect(mem.memoriesSummary(m)).toBe("2 memories · 1 unvetted");
  });

  it("a failed read is reported and does not empty the list", async () => {
    listMemories.mockResolvedValueOnce({ ok: true, memories: [row("a")], may_curate: false });
    await mem.refreshMemories();
    listMemories.mockResolvedValueOnce({ ok: false, category: "runner_unreachable", error: "timed out", memories: [] });
    await mem.refreshMemories();
    const m = useJaato.getState().memories;
    expect(m.status).toBe("error");
    expect(m.error).toContain("runner did not answer");
    expect(m.rows.map((r) => r.id)).toEqual(["a"]);
  });

  it("an answer for a session this client has left is dropped", async () => {
    let resolve!: (v: unknown) => void;
    listMemories.mockReturnValue(new Promise((r) => { resolve = r; }));
    const asked = mem.refreshMemories();
    connected = false;
    useJaato.getState().resetSessionState();
    useJaato.setState({ sessionId: "s2" });
    connected = true;
    resolve({ ok: true, memories: [row("old")], may_curate: true });
    await asked;
    expect(useJaato.getState().memories.rows).toEqual([]);
  });

  it("a daemon below 1.22 is unsupported, not empty, and is never asked", async () => {
    protocol = "1.21";
    await mem.refreshMemories();
    expect(useJaato.getState().memories.status).toBe("unsupported");
    expect(listMemories).not.toHaveBeenCalled();
    expect(mem.memoriesSummary(useJaato.getState().memories)).toBeNull();
  });

  it("the this-session filter keeps what was written or used here", () => {
    const rows = [row("w", { written_this_session: true }), row("r", { retrieved_this_session: true }), row("x")];
    expect(mem.visibleMemories({ rows, thisSessionOnly: false }).length).toBe(3);
    expect(mem.visibleMemories({ rows, thisSessionOnly: true }).map((r) => r.id)).toEqual(["w", "r"]);
  });
});

describe("the actions", () => {
  beforeEach(async () => {
    listMemories.mockResolvedValue({ ok: true, memories: [row("a")], may_curate: true });
    await mem.refreshMemories();
    listMemories.mockClear();
  });

  it("an owner-gate refusal is shown in words and the list is re-read", async () => {
    approveMemory.mockResolvedValue({ ok: false, category: "not_owner", error: "no" });
    const ok = await mem.approveMemory("a");
    expect(ok).toBe(false);
    const m = useJaato.getState().memories;
    expect(m.notice).toEqual({ text: expect.stringContaining("Only the owner"), error: true });
    expect(m.busy).toEqual({});
    expect(listMemories).toHaveBeenCalledTimes(1);
  });

  it("dismiss and remove call their verbs and report success", async () => {
    dismissMemory.mockResolvedValue({ ok: true });
    deleteMemory.mockResolvedValue({ ok: true });
    expect(await mem.dismissMemory("a")).toBe(true);
    expect(dismissMemory).toHaveBeenCalledWith("a");
    expect(await mem.removeMemory("a")).toBe(true);
    expect(deleteMemory).toHaveBeenCalledWith("a");
    expect(useJaato.getState().memories.notice).toEqual({ text: "Removed." });
  });

  it("expanding fetches the content once", async () => {
    getMemory.mockResolvedValue({ ok: true, memory: { id: "a", content: "the body", evidence: "seen" } });
    await mem.toggleMemory("a");
    expect(useJaato.getState().memories.details.a).toEqual({ state: "loaded", content: "the body", evidence: "seen" });
    await mem.toggleMemory("a");
    await mem.toggleMemory("a");
    expect(getMemory).toHaveBeenCalledTimes(1);
  });

  it("an edit sends only the fields that changed", async () => {
    getMemory.mockResolvedValue({ ok: true, memory: { id: "a", content: "the body" } });
    updateMemory.mockResolvedValue({ ok: true });
    await mem.startEditMemory("a");
    mem.setMemoryDraft("a", { description: "sharper", tags: "alpha, beta" });
    expect(await mem.saveEditMemory("a")).toBe(true);
    expect(updateMemory).toHaveBeenCalledWith("a", { description: "sharper" });
    expect(useJaato.getState().memories.editing.a).toBeUndefined();
  });

  it("a refused edit keeps the form open with the draft", async () => {
    getMemory.mockResolvedValue({ ok: true, memory: { id: "a", content: "the body" } });
    updateMemory.mockResolvedValue({ ok: false, category: "invalid", error: "description cannot be empty" });
    await mem.startEditMemory("a");
    mem.setMemoryDraft("a", { description: " " });
    expect(await mem.saveEditMemory("a")).toBe(false);
    expect(useJaato.getState().memories.editing.a?.description).toBe(" ");
    expect(useJaato.getState().memories.notice?.text).toContain("description cannot be empty");
  });
});

describe("what triggers a refresh", () => {
  it("a successful memory write in any agent", () => {
    expect(mem.triggersMemoryRefresh({ type: "tool.call_end", tool_name: "store_memory", success: true })).toBe(true);
    expect(mem.triggersMemoryRefresh({ type: "tool.call_end", tool_name: "delete_memory", success: true })).toBe(true);
    expect(mem.triggersMemoryRefresh({ type: "tool.call_end", tool_name: "store_memory", success: false })).toBe(false);
    expect(mem.triggersMemoryRefresh({ type: "tool.call_end", tool_name: "store_memory", success: true, is_error_result: true })).toBe(false);
    expect(mem.triggersMemoryRefresh({ type: "tool.call_end", tool_name: "retrieve_memories", success: true })).toBe(false);
  });

  it("a hashed tool id resolves through the registry", () => {
    expect(mem.triggersMemoryRefresh({ type: "tool.call_end", tool_name: "t_1a2b3c4d", success: true }, { t_1a2b3c4d: "update_memory" })).toBe(true);
  });

  it("the memory command's push, but not our own correlated answers", () => {
    expect(mem.triggersMemoryRefresh({ type: "memory.list" })).toBe(true);
    expect(mem.triggersMemoryRefresh({ type: "memory.list", request_id: "mem-1" })).toBe(false);
  });

  it("a session.info naming a session", () => {
    expect(mem.triggersMemoryRefresh({ type: "session.info", session_id: "s1" })).toBe(true);
    expect(mem.triggersMemoryRefresh({ type: "session.info" })).toBe(false);
  });
});
