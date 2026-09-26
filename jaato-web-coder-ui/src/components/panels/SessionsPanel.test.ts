/**
 * Two rules about a session the daemon says is waiting on a person.
 *
 * Both are about not inventing a fact the daemon declined to state, which
 * is what the protocol's own wording asks of a reader: an absent
 * ``awaiting_since`` is NOT MEASURED, never "just now", and an absent
 * ``awaiting`` is "nothing is waiting as far as this daemon says", never a
 * positive no.
 */
import { describe, expect, it } from "vitest";
import { awaitingLabel, notedSummary } from "./SessionsPanel";
import type { SessionSummary } from "@/protocol/sessions";

const sess = (over: Partial<SessionSummary>): SessionSummary => ({
  id: "s1", name: "", description: "", provider: "", model: "",
  isLoaded: true, isCurrent: false, clientCount: 0, turnCount: 0,
  workspacePath: "/ws/one", profile: "", lastActivity: "", isProcessing: false, ...over,
});

describe("awaitingLabel", () => {
  const now = Date.parse("2026-09-19T12:30:00Z");

  it("renders how long somebody has been kept waiting", () => {
    expect(awaitingLabel("permission", "2026-09-19T12:26:00Z", now)).toBe("waiting 4 min: permission");
    expect(awaitingLabel("clarification", "2026-09-19T10:05:00Z", now)).toBe("waiting 2h 25m: clarification");
  });

  it("an absent timestamp drops the duration rather than reading as zero", () => {
    // The daemon documents an absent `awaiting_since` as not measured.
    // "waiting just now" would be a claim it declined to make; the kind is
    // the actionable half and survives on its own.
    expect(awaitingLabel("permission", undefined, now)).toBe("waiting: permission");
    expect(awaitingLabel("permission", "not a date", now)).toBe("waiting: permission");
  });
});

describe("the Sessions header", () => {
  it("counts what needs a person ahead of what carries a note", () => {
    const sessions = [sess({ id: "a", awaiting: "permission" }), sess({ id: "b" }), sess({ id: "c" })];
    const notes = { b: { text: "re-run e2e" }, c: { text: "ask about grace" } };
    expect(notedSummary(sessions, notes)).toBe("1 waiting on you");
  });

  it("falls back to the note count when nothing is blocked", () => {
    const sessions = [sess({ id: "a" }), sess({ id: "b" })];
    expect(notedSummary(sessions, { b: { text: "re-run e2e" } })).toBe("1 of 2 noted");
    expect(notedSummary([], {})).toBeNull();
  });
});
