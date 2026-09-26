/**
 * The session board's discard is inline and state-aware, and a card leaves
 * only when the daemon says the session is gone.  The two properties worth
 * pinning are the ones a lazy implementation gets wrong: the message names
 * what THIS state loses, and a refused discard keeps the card.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { SessionSummary } from "@/protocol/sessions";

const deleteSession = vi.fn();
vi.mock("@/app/sessionDelete", () => ({
  deleteSession: (id: string) => deleteSession(id),
  isGone: (a: { kind: string }) => a.kind === "deleted" || a.kind === "missing",
}));
const ensureSessions = vi.fn(async () => undefined);
vi.mock("@/app/actions", () => ({ ensureSessions: () => ensureSessions() }));

import { SessionBoard, discardMessage, stateLine } from "./SessionBoard";
import { useJaato } from "@/store/store";

const base: Omit<SessionSummary, "id"> = {
  name: "", description: "Refactor parser", provider: "minimax", model: "M3", isLoaded: true, isCurrent: false,
  clientCount: 1, turnCount: 41, workspacePath: "/ws/one", profile: "skill-020", lastActivity: "", isProcessing: false,
};
const sessions: SessionSummary[] = [
  { ...base, id: "20260926_140512", awaiting: "permission", awaitingSince: new Date(Date.now() - 4 * 60000).toISOString() },
  { ...base, id: "20260926_091043" },
  { ...base, id: "20260925_173010", isLoaded: false, clientCount: 0 },
];

beforeEach(() => { useJaato.setState({ notes: {}, noteStatus: {} }); deleteSession.mockReset(); ensureSessions.mockClear(); });
afterEach(() => cleanup());

describe("SessionBoard", () => {
  it("puts each session in its column with a count, and attaching is a real button", () => {
    const onAttach = vi.fn();
    render(<SessionBoard sessions={sessions} onAttach={onAttach} />);
    for (const [col, id] of [["Waiting on you", "20260926_140512"], ["Awake", "20260926_091043"], ["Sleeping", "20260925_173010"]] as const) {
      const region = screen.getByRole("region", { name: col });
      expect(within(region).getByText(id)).toBeInTheDocument();
    }
    expect(screen.getByText("waiting 4 min: permission")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Attach session 20260925_173010" }));
    expect(onAttach).toHaveBeenCalledWith("20260925_173010");
  });

  it("confirms one card at a time, with the state's own message", () => {
    render(<SessionBoard sessions={sessions} onAttach={() => undefined} />);
    fireEvent.click(screen.getByRole("button", { name: "Discard session 20260926_140512…" }));
    expect(screen.getByText(discardMessage("20260926_140512", "waiting"))).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Discard session 20260925_173010…" }));
    expect(screen.queryByText(discardMessage("20260926_140512", "waiting"))).toBeNull();
    expect(screen.getByText("Delete 20260925_173010 and its saved history?")).toBeInTheDocument();
  });

  it("a refused discard keeps the card and says why; a confirmed one refreshes the list", async () => {
    deleteSession.mockResolvedValueOnce({ kind: "refused", text: "not yours" });
    render(<SessionBoard sessions={sessions} onAttach={() => undefined} />);
    fireEvent.click(screen.getByRole("button", { name: "Discard session 20260926_091043…" }));
    fireEvent.click(screen.getByRole("button", { name: "Discard session 20260926_091043" }));
    await waitFor(() => expect(screen.getByText("not yours")).toBeInTheDocument());
    expect(ensureSessions).not.toHaveBeenCalled();

    deleteSession.mockResolvedValueOnce({ kind: "deleted", text: "ok" });
    fireEvent.click(screen.getByRole("button", { name: "Discard session 20260926_091043" }));
    await waitFor(() => expect(ensureSessions).toHaveBeenCalled());
    expect(deleteSession).toHaveBeenLastCalledWith("20260926_091043");
  });
});

describe("stateLine", () => {
  it("says who is attached to an awake session and that a sleeping one is on disk", () => {
    expect(stateLine(sessions[1]!, "awake")).toBe("awake · 1 client attached");
    expect(stateLine({ ...sessions[1]!, isProcessing: true, clientCount: 0 }, "awake")).toBe("awake · working · no client attached");
    expect(stateLine(sessions[2]!, "sleeping")).toBe("sleeping · on disk");
  });
});
