import { beforeEach, describe, expect, it, vi } from "vitest";
import { MAX_NOTE_CHARS, localNotesApi, noteFirstLine, normalizeNote, notesApi } from "./notes";
import { SignInRequiredError } from "./tickets";

const ok = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

describe("notes API (backend)", () => {
  it("lists, and tolerates junk in the answer rather than throwing at the page", async () => {
    const f = vi.fn().mockResolvedValue(ok({ notes: [{ sessionId: "s1", text: "a", updatedAt: "t" }, null, { nope: 1 }] }));
    const api = notesApi("./api/notes", f as unknown as typeof fetch);
    expect(api.scope).toBe("backend");
    expect(await api.list()).toEqual([{ sessionId: "s1", text: "a", updatedAt: "t" }]);
  });

  it("PUTs to the session's own url, with the text normalised the way the server will", async () => {
    const f = vi.fn().mockResolvedValue(ok({ note: { sessionId: "s 1", text: "a", updatedAt: "t" } }));
    await notesApi("./api/notes", f as unknown as typeof fetch).put("s 1", "  a\r\n  ");
    const [url, init] = f.mock.calls[0]!;
    expect(url).toBe("./api/notes/s%201");
    expect(init.method).toBe("PUT");
    expect(JSON.parse(init.body as string)).toEqual({ text: "a" });
  });

  it("an emptied note answers null: clearing the box IS deleting", async () => {
    const f = vi.fn().mockResolvedValue(ok({ note: null }));
    expect(await notesApi("./api/notes", f as unknown as typeof fetch).put("s1", "")).toBeNull();
  });

  it("401 is SignInRequiredError, not a generic failure", async () => {
    // BFF-side storage means the cookie can expire mid-session, and "signed
    // out" is a different thing to tell somebody than "could not save".
    const f = vi.fn().mockResolvedValue(new Response("{}", { status: 401 }));
    await expect(notesApi("./api/notes", f as unknown as typeof fetch).put("s1", "x")).rejects.toBeInstanceOf(SignInRequiredError);
  });

  it("a 404 on DELETE is not an error: forgetting a note nobody stored is done", async () => {
    const f = vi.fn().mockResolvedValue(new Response("", { status: 404 }));
    await expect(notesApi("./api/notes", f as unknown as typeof fetch).remove("s1")).resolves.toBeUndefined();
  });
});

describe("notes API (this browser)", () => {
  let store: Storage;
  beforeEach(() => {
    const data = new Map<string, string>();
    store = {
      getItem: (k: string) => data.get(k) ?? null,
      setItem: (k: string, v: string) => void data.set(k, v),
      removeItem: (k: string) => void data.delete(k),
      clear: () => data.clear(),
      key: () => null,
      length: 0,
    } as unknown as Storage;
  });

  it("round-trips, and says it is local so the UI can say so too", async () => {
    const api = localNotesApi(store);
    expect(api.scope).toBe("local");
    await api.put("s1", "remember this");
    expect((await api.list()).map((n) => n.text)).toEqual(["remember this"]);
    await api.put("s1", "  ");
    expect(await api.list()).toEqual([]);
  });

  it("a browser that refuses to store reports it rather than taking the page down", async () => {
    // localStorage throws in a private window and with site data blocked;
    // a note that cannot be saved must say so.
    const hostile = { ...store, setItem: () => { throw new Error("nope"); } } as unknown as Storage;
    await expect(localNotesApi(hostile).put("s1", "x")).rejects.toThrow(/refused to store/);
    expect(await localNotesApi(null).list()).toEqual([]);
  });
});

describe("note text", () => {
  it("normalises exactly as the server does, so the two cannot disagree about what was saved", () => {
    expect(normalizeNote("  a\r\nb  \n")).toBe("a\nb");
    expect(normalizeNote("x".repeat(MAX_NOTE_CHARS + 50)).length).toBe(MAX_NOTE_CHARS);
  });
  it("the one line a row has space for is the first non-empty one", () => {
    expect(noteFirstLine("\n\n  ask about the grace period\nthen re-run e2e")).toBe("ask about the grace period");
    expect(noteFirstLine(undefined)).toBe("");
    expect(noteFirstLine("   ")).toBe("");
  });
});
