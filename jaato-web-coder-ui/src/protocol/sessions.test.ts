import { describe, expect, it } from "vitest";
import { formatSessionList, normalizeSessionList, sessionBoard, sessionIdCompletions, sessionsInWorkspace, wantsSessionIds } from "./sessions";

const RAW = [
  { id: "20260916_1", description: "fix the panel", model_provider: "anthropic", model_name: "claude", is_loaded: true, is_current: true, client_count: 1, turn_count: 4, workspace_path: "/srv/ws/project-a" },
  { id: "20260915_9", name: "old", is_loaded: false, workspace_path: "/srv/ws/project-b" },
  { nope: true },
];

describe("session listing", () => {
  it("reads the daemon's free-form dicts and drops entries with no id", () => {
    const list = normalizeSessionList(RAW);
    expect(list.map((s) => s.id)).toEqual(["20260916_1", "20260915_9"]);
    expect(list[0]).toMatchObject({ isCurrent: true, isLoaded: true, turnCount: 4, provider: "anthropic" });
  });
  it("renders the TUI's listing: marker, id, description, model, counts, workspace, legend", () => {
    const text = formatSessionList(normalizeSessionList(RAW));
    expect(text).toContain("▶ 20260916_1 - fix the panel [anthropic/claude], 1 client(s), 4 turns");
    expect(text).toContain("      /srv/ws/project-a");
    expect(text).toContain("○ 20260915_9 - old");
    expect(text).toContain("▶ current  ● loaded  ○ on disk");
    expect(formatSessionList([])).toContain("No sessions available");
  });
});

describe("session id completion", () => {
  const sessions = normalizeSessionList(RAW);
  it("proposes ids after `session attach `, filtered by the partial", () => {
    expect(sessionIdCompletions("session attach ", sessions)!.map((c) => c.insert)).toEqual(["session attach 20260916_1", "session attach 20260915_9"]);
    expect(sessionIdCompletions("session attach 202609_1", sessions)).toEqual([]);
    expect(sessionIdCompletions("session attach 20260915", sessions)!.map((c) => c.label)).toEqual(["20260915_9"]);
    expect(sessionIdCompletions("resume 2026", sessions)!.length).toBe(2);
  });
  it("is not in play before the third word or after it", () => {
    expect(sessionIdCompletions("session att", sessions)).toBeNull();
    expect(sessionIdCompletions("session attach 20260916_1 more", sessions)).toBeNull();
    expect(sessionIdCompletions("model x", sessions)).toBeNull();
    expect(wantsSessionIds("session attach ")).toBe(true);
    expect(wantsSessionIds("session list")).toBe(false);
  });
});

describe("sessions of a workspace", () => {
  const sessions = normalizeSessionList(RAW);
  it("matches by the daemon's path when known, else by the trailing directory name", () => {
    expect(sessionsInWorkspace(sessions, { name: "project-a", path: "/srv/ws/project-a/" }).map((s) => s.id)).toEqual(["20260916_1"]);
    expect(sessionsInWorkspace(sessions, { name: "project-b" }).map((s) => s.id)).toEqual(["20260915_9"]);
    expect(sessionsInWorkspace(sessions, undefined)).toEqual([]);
  });
});

describe("sessionBoard", () => {
  const base = { name: "", description: "", provider: "", model: "", isCurrent: false, clientCount: 0, turnCount: 0, workspacePath: "/w", profile: "", isProcessing: false };
  it("sorts waiting longest-first and the rest most-recent-first", () => {
    const now = "2026-09-26T12:00:00Z";
    const b = sessionBoard([
      { ...base, id: "w-new", isLoaded: true, awaiting: "permission", awaitingSince: "2026-09-26T11:58:00Z", lastActivity: now },
      { ...base, id: "w-old", isLoaded: true, awaiting: "clarification", awaitingSince: "2026-09-26T11:40:00Z", lastActivity: now },
      { ...base, id: "w-undated", isLoaded: true, awaiting: "permission", lastActivity: now },
      { ...base, id: "a1", isLoaded: true, lastActivity: "2026-09-26T09:00:00Z" },
      { ...base, id: "a2", isLoaded: true, lastActivity: "2026-09-26T10:00:00Z" },
      { ...base, id: "s1", isLoaded: false, lastActivity: "2026-09-24T10:00:00Z" },
      { ...base, id: "s2", isLoaded: false, lastActivity: "2026-09-25T10:00:00Z" },
    ]);
    expect(b.waiting.map((s) => s.id)).toEqual(["w-old", "w-new", "w-undated"]);
    expect(b.awake.map((s) => s.id)).toEqual(["a2", "a1"]);
    expect(b.sleeping.map((s) => s.id)).toEqual(["s2", "s1"]);
  });
});
