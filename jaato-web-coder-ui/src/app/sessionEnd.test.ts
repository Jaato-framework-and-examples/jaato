/**
 * End keeps the session on a daemon that lists finished sessions (1.29),
 * and deletes it on one that does not.  The version gate is the property
 * worth pinning: ending on an older daemon would leave a session nobody can
 * tell from a sleeping one, and deleting on a newer one throws away what
 * the Finished column exists to keep.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EventTypeValue } from "@jaato/sdk";
import { END_CONFIRM_GRACE_MS, endSession, finishSupported } from "./sessionEnd";
import { exitOptions } from "./exitChoice";
import { useJaato } from "@/store/store";

const ID = "20260927_101500";
type Handler = (ev: Record<string, unknown>) => void;
const subs: Record<string, Handler[]> = {};
const ended: number[] = [];

vi.mock("@/sdk/connection", () => ({
  getClient: () => ({
    subscribe: (type: string, fn: Handler) => {
      (subs[type] ??= []).push(fn);
      return () => { subs[type] = (subs[type] ?? []).filter((f) => f !== fn); };
    },
    endSession: async () => { ended.push(1); },
  }),
  disconnect: async () => undefined,
  isConnected: () => true,
}));

function emit(type: string, ev: Record<string, unknown>) {
  for (const fn of [...(subs[type] ?? [])]) fn(ev);
}

beforeEach(() => {
  vi.useFakeTimers();
  for (const k of Object.keys(subs)) delete subs[k];
  ended.length = 0;
});
afterEach(() => { vi.useRealTimers(); });

describe("finishSupported", () => {
  it("is the 1.29 floor", () => {
    expect(finishSupported("1.29")).toBe(true);
    expect(finishSupported("1.30")).toBe(true);
    expect(finishSupported("1.28")).toBe(false);
    expect(finishSupported(null)).toBe(false);
  });
});

describe("endSession: the daemon's terminal settles it", () => {
  it("sends session.end and answers with the terminal's reason", async () => {
    const p = endSession(ID);
    await Promise.resolve();
    emit(EventTypeValue.SESSION_TERMINATED, { session_id: "someone_else", reason: "natural" });
    emit(EventTypeValue.SESSION_TERMINATED, { session_id: ID, reason: "client_request" });
    await expect(p).resolves.toEqual({ kind: "ended", reason: "client_request" });
    expect(ended).toHaveLength(1);
  });

  it("gives up after the grace when the daemon says nothing", async () => {
    const p = endSession(ID);
    await Promise.resolve();
    vi.advanceTimersByTime(END_CONFIRM_GRACE_MS);
    await expect(p).resolves.toEqual({ kind: "silent" });
  });
});

describe("the exit plate says what End will do", () => {
  it("keeps on 1.29, deletes below", () => {
    const keep = exitOptions(false, true).find((o) => o.key === "e")!;
    const del = exitOptions(false, false).find((o) => o.key === "e")!;
    expect(keep.description).toContain("Finished");
    expect(del.description).toContain("Delete");
  });

  it("reads the connected daemon's version by default", () => {
    useJaato.setState({ connection: { ...useJaato.getState().connection, protocolVersion: "1.29" } });
    expect(exitOptions(false).find((o) => o.key === "e")!.description).toContain("Finished");
  });
});
