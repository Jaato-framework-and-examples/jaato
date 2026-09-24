/**
 * The Memories panel (#1232): what it shows, and when it offers to change it.
 *
 * Three facts the markup must carry: a raw memory says **unvetted** in
 * words (colour alone is not a marker); the curation buttons appear only
 * when the daemon said this connection may curate -- the same gate it
 * enforces; and a row written or used in this session says so.  Remove is a
 * two-step action, so one stray tap on a touch screen deletes nothing.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useJaato, emptyMemories } from "@/store/store";
import type { MemoryRow } from "@/store/types";

const removeMemory = vi.fn(async (_id: string) => true);
const approveMemory = vi.fn(async (_id: string) => true);
vi.mock("@/app/memories", async (orig) => ({
  ...(await orig<typeof import("@/app/memories")>()),
  removeMemory: (id: string) => removeMemory(id),
  approveMemory: (id: string) => approveMemory(id),
  refreshMemories: vi.fn(async () => undefined),
}));

const { MemoriesPanel } = await import("./MemoriesPanel");

const row = (id: string, extra: Partial<MemoryRow> = {}): MemoryRow => ({
  id, description: `about ${id}`, tags: ["alpha"], maturity: "raw", tier: "workspace", ...extra,
});

function load(rows: MemoryRow[], mayCurate: boolean) {
  useJaato.setState({ sessionId: "s1", memories: { ...emptyMemories(), rows, status: "loaded", mayCurate } });
}

beforeEach(() => { removeMemory.mockClear(); approveMemory.mockClear(); });
afterEach(() => { cleanup(); });

describe("MemoriesPanel", () => {
  it("marks a raw memory unvetted in words, and names its tier", () => {
    load([row("a"), row("b", { maturity: "validated", tier: "global" })], false);
    render(<MemoriesPanel />);
    const rows = screen.getAllByTestId("memory-row");
    expect(rows[0]!.textContent).toContain("unvetted");
    expect(rows[0]!.textContent).toContain("workspace");
    expect(rows[1]!.textContent).not.toContain("unvetted");
    expect(rows[1]!.textContent).toContain("global");
  });

  it("offers no curation to a connection the daemon said may not curate", () => {
    load([row("a")], false);
    render(<MemoriesPanel />);
    expect(screen.queryByRole("button", { name: /Approve memory/ })).toBeNull();
    expect(screen.queryByRole("button", { name: /Remove memory/ })).toBeNull();
    expect(screen.getByText(/Only the owner/)).toBeTruthy();
  });

  it("offers approve on a raw memory and not on an approved one", () => {
    load([row("a"), row("b", { maturity: "validated" })], true);
    render(<MemoriesPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Approve memory about a" }));
    expect(approveMemory).toHaveBeenCalledWith("a");
    expect(screen.queryByRole("button", { name: "Approve memory about b" })).toBeNull();
  });

  it("remove asks for confirmation before it deletes", () => {
    load([row("a")], true);
    render(<MemoriesPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Remove memory about a" }));
    expect(removeMemory).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Confirm remove memory about a" }));
    expect(removeMemory).toHaveBeenCalledWith("a");
  });

  it("says which rows this session wrote or used, and can narrow to them", () => {
    load([row("w", { written_this_session: true }), row("x")], false);
    render(<MemoriesPanel />);
    expect(screen.getByText("written in this session")).toBeTruthy();
    fireEvent.click(screen.getByRole("checkbox", { name: /This session only/ }));
    expect(screen.getAllByTestId("memory-row").length).toBe(1);
  });

  it("a store that could not be read is an error, not 'nothing remembered'", () => {
    useJaato.setState({ sessionId: "s1", memories: { ...emptyMemories(), status: "error", error: "runner gone" } });
    render(<MemoriesPanel />);
    expect(screen.getByRole("alert").textContent).toContain("runner gone");
    expect(screen.queryByText("Nothing remembered yet.")).toBeNull();
  });
});
