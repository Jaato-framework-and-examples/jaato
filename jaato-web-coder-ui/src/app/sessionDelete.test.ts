/**
 * A note must not outlive its session — and must not die before one.
 *
 * Reported from a deployed client: the rail offered a note under a session
 * its owner had deleted.  Notes live where the daemon cannot see them, so
 * nothing removed one when its session went away.
 *
 * The rule under test is narrow on purpose.  Pruning notes whose session is
 * absent from the listing would be the obvious fix and is wrong: that
 * listing NARROWS (a workspace deselected, an identity not yet resolved),
 * and a note is the only copy of what you meant to do next.  So a note goes
 * when the daemon SAYS the session is gone, and at no other time — which
 * makes the two negative cases here the load-bearing ones.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EventTypeValue } from "@jaato/sdk";
import { DELETE_CONFIRM_GRACE_MS, deleteSession } from "./sessionDelete";
import { useJaato } from "@/store/store";

const ID = "20260917_120657";

type Handler = (ev: Record<string, unknown>) => void;
const subs: Record<string, Handler[]> = {};
const deleted: string[] = [];

vi.mock("@/sdk/connection", () => ({
  getClient: () => ({
    subscribe: (type: string, fn: Handler) => {
      (subs[type] ??= []).push(fn);
      return () => { subs[type] = (subs[type] ?? []).filter((f) => f !== fn); };
    },
    deleteSession: async (id: string) => { deleted.push(id); },
  }),
}));

function emit(type: string, ev: Record<string, unknown>) {
  for (const fn of [...(subs[type] ?? [])]) fn(ev);
}

/** The BFF's note routes, so the real ``notesApi`` path is the one exercised. */
function serveNotes() {
  const removed: string[] = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    if (init?.method === "DELETE") { removed.push(String(url).split("/").pop()!); return new Response(null, { status: 204 }); }
    return new Response(JSON.stringify({ notes: [] }), { status: 200 });
  }));
  return removed;
}

beforeEach(() => {
  vi.useFakeTimers();
  for (const k of Object.keys(subs)) delete subs[k];
  deleted.length = 0;
  useJaato.setState({
    notesUrl: "./api/notes",
    notes: { [ID]: { sessionId: ID, text: "esta sesión va del acuerdo", updatedAt: "now" } },
    noteStatus: {},
  });
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

/** Run ``deleteSession`` and answer it with one event, after the send. */
async function run(answer: () => void) {
  const p = deleteSession(ID);
  await Promise.resolve();
  answer();
  return p;
}

describe("deleteSession: the daemon's word decides", () => {
  it("forgets the note when the daemon says it deleted the session", async () => {
    const removed = serveNotes();
    const answer = await run(() => emit(EventTypeValue.SYSTEM_MESSAGE, { message: `Session '${ID}' deleted.` }));

    expect(deleted).toEqual([ID]);
    expect(answer.kind).toBe("deleted");
    expect(removed).toEqual([ID]);
    expect(useJaato.getState().notes[ID]).toBeUndefined();
  });

  it("forgets it when the daemon says there was no such session", async () => {
    // The reported state exactly: a note for a session that is already
    // gone.  "not found" is as good as "deleted" for the note.
    const removed = serveNotes();
    const answer = await run(() => emit(EventTypeValue.SYSTEM_MESSAGE, { message: `Session '${ID}' not found.` }));

    expect(answer.kind).toBe("missing");
    expect(removed).toEqual([ID]);
    expect(useJaato.getState().notes[ID]).toBeUndefined();
  });

  it("KEEPS it when the delete is refused — the session is still there", async () => {
    const removed = serveNotes();
    const answer = await run(() => emit(EventTypeValue.ERROR, {
      error: `session.delete: ${ID} is not one of your sessions`,
      error_type: "SessionError",
    }));

    expect(answer.kind).toBe("refused");
    expect(removed).toEqual([]);
    expect(useJaato.getState().notes[ID]?.text).toBe("esta sesión va del acuerdo");
  });

  it("KEEPS it when nobody answered — silence is not evidence", async () => {
    const removed = serveNotes();
    const p = deleteSession(ID);
    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(DELETE_CONFIRM_GRACE_MS + 10);

    expect((await p).kind).toBe("silent");
    expect(removed).toEqual([]);
    expect(useJaato.getState().notes[ID]?.text).toBe("esta sesión va del acuerdo");
  });

  it("ignores an answer about a different session", async () => {
    const removed = serveNotes();
    const p = deleteSession(ID);
    await Promise.resolve();
    emit(EventTypeValue.SYSTEM_MESSAGE, { message: "Session '20260101_000000' deleted." });
    await vi.advanceTimersByTimeAsync(DELETE_CONFIRM_GRACE_MS + 10);

    expect((await p).kind).toBe("silent");
    expect(removed).toEqual([]);
  });

  it("still forgets the note when the note store refuses the delete", async () => {
    // Best effort by design: the session is gone either way, so a 500 from
    // the notes backend must not leave the row rendering under a session
    // that no longer exists.
    vi.stubGlobal("fetch", vi.fn(async (_u: string, init?: RequestInit) =>
      init?.method === "DELETE" ? new Response("no", { status: 500 })
        : new Response(JSON.stringify({ notes: [] }), { status: 200 })));

    await run(() => emit(EventTypeValue.SYSTEM_MESSAGE, { message: `Session '${ID}' deleted.` }));
    expect(useJaato.getState().notes[ID]).toBeUndefined();
  });
});
