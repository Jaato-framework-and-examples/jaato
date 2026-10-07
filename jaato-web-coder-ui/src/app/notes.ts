/**
 * A note you write to your future self about one session.
 *
 * What it is for.  Running several sessions at once, you lose track of what
 * each one still needs from YOU.  A day later the picker says what the
 * session *is* and nothing about what you were going to do next; "waiting on
 * an answer about the grace period, then re-run the e2e suite" lives in your
 * head or nowhere.  Two session fields already exist and neither is this:
 * ``name`` is create-only (there is no rename verb anywhere in the tree), and
 * ``description`` is written by the MODEL, so a note put there is overwritten
 * by the next turn.  The picker already tells you what the agent thought the
 * session was about; this is what you meant to do about it.
 *
 * Where it lives, and why not the daemon.  ``jaato-web-coder-server`` keeps
 * it per signed-in user (``notesUrl`` in ``config.json``, beside
 * ``credentialsUrl``), the call ``app/credentials.ts`` already made: the
 * daemon knows users only as ``app:user``, a ``session.note`` verb would sit
 * in the protocol for one client, and the TUI would carry it and never call
 * it.  It also makes "for the human" mean what it says -- a note in the
 * workspace tree would be ``readFile``-reachable by the agent, so it would
 * only be *not injected into the prompt* rather than unreachable.  The model
 * never sees this.  Neither does the daemon.
 *
 * Without a backend.  A local ``npx @jaato/web-coder-ui`` against a daemon
 * has no server at all, so notes go to ``localStorage`` instead.  That is
 * worth having and worth SAYING: they are then per browser, and "next time I
 * look" from another device loses them.  ``NotesApi.scope`` is what the UI
 * renders that from -- a silent fallback would be a promise the storage does
 * not keep.
 *
 * The session id is opaque to the backend: it stores text under a key and
 * the page joins that key against the listing it already holds from
 * ``session.list``.  Nothing here models a session either.
 */
import { SignInRequiredError } from "./tickets";

/** The cap the BFF enforces; the editor shows a counter as you approach it. */
export const MAX_NOTE_CHARS = 2000;

const LOCAL_KEY = "jaato.web-coder.notes.v1";

export interface SessionNote {
  sessionId: string;
  text: string;
  updatedAt: string;
}

/**
 * Where notes are kept, so the UI can say so.
 *
 * - ``backend`` — the signed-in user's, shared across their devices
 * - ``local`` — this browser only (no sign-in backend configured)
 */
export type NoteScope = "backend" | "local";

export interface NotesApi {
  readonly scope: NoteScope;
  list(): Promise<SessionNote[]>;
  /** Write one.  Empty text forgets it, and the answer is ``null`` — clearing the box IS deleting. */
  put(sessionId: string, text: string): Promise<SessionNote | null>;
  remove(sessionId: string): Promise<void>;
}

function noteUrl(base: string, sessionId: string): string {
  const [path, query = ""] = base.split(/\?(.*)/s, 2);
  return `${path}/${encodeURIComponent(sessionId)}${query ? `?${query}` : ""}`;
}

async function failure(res: Response, what: string): Promise<Error> {
  // A backend note means the cookie can expire mid-session, and "signed out"
  // is a different thing to tell somebody than "could not save".
  if (res.status === 401) return new SignInRequiredError("./auth/login");
  let detail = "";
  try { detail = String(((await res.json()) as { error?: unknown }).error ?? ""); } catch { /* not JSON */ }
  return new Error(`${what}: HTTP ${res.status}${detail ? ` (${detail})` : ""}`);
}

/** Trim and bound exactly as the BFF does, so the two cannot disagree about what was saved. */
export function normalizeNote(text: string): string {
  return text.replace(/\r\n?/g, "\n").replace(/[^\S\n\t]+$/gm, "").trim().slice(0, MAX_NOTE_CHARS);
}

export function notesApi(notesUrl: string, fetchImpl: typeof fetch = fetch): NotesApi {
  const common: RequestInit = { credentials: "same-origin", cache: "no-store" };
  return {
    scope: "backend",
    async list() {
      const res = await fetchImpl(notesUrl, { ...common, headers: { Accept: "application/json" } });
      if (!res.ok) throw await failure(res, "Loading your notes failed");
      const body = (await res.json()) as { notes?: unknown };
      return Array.isArray(body.notes) ? (body.notes as SessionNote[]).filter((n) => n && typeof n.sessionId === "string") : [];
    },
    async put(sessionId, text) {
      const res = await fetchImpl(noteUrl(notesUrl, sessionId), {
        ...common, method: "PUT",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify({ text: normalizeNote(text) }),
      });
      if (!res.ok) throw await failure(res, "Saving the note failed");
      const body = (await res.json()) as { note?: SessionNote | null };
      return body.note ?? null;
    },
    async remove(sessionId) {
      const res = await fetchImpl(noteUrl(notesUrl, sessionId), { ...common, method: "DELETE" });
      if (!res.ok && res.status !== 404) throw await failure(res, "Deleting the note failed");
    },
  };
}

/**
 * The no-backend fallback.  Same interface, so every caller is written once;
 * only ``scope`` differs, and that is what the UI tells the person.
 *
 * Every read and write is wrapped: ``localStorage`` throws in a private
 * window and with site data blocked, and a note that cannot be saved must
 * report that rather than take the page down.
 */
export function localNotesApi(storage: Storage | null = typeof localStorage === "undefined" ? null : localStorage): NotesApi {
  const read = (): Record<string, SessionNote> => {
    try {
      const raw = storage?.getItem(LOCAL_KEY);
      const parsed = raw ? (JSON.parse(raw) as unknown) : null;
      return parsed && typeof parsed === "object" ? (parsed as Record<string, SessionNote>) : {};
    } catch { return {}; }
  };
  const write = (all: Record<string, SessionNote>): void => {
    if (!storage) throw new Error("This browser is not storing notes (private window, or site data blocked)");
    try { storage.setItem(LOCAL_KEY, JSON.stringify(all)); }
    catch { throw new Error("This browser refused to store the note (private window, or storage full)"); }
  };
  return {
    scope: "local",
    async list() {
      return Object.values(read()).filter((n) => n && typeof n.sessionId === "string");
    },
    async put(sessionId, text) {
      const t = normalizeNote(text);
      const all = read();
      if (!t) { delete all[sessionId]; write(all); return null; }
      const note: SessionNote = { sessionId, text: t, updatedAt: new Date().toISOString() };
      all[sessionId] = note;
      write(all);
      return note;
    },
    async remove(sessionId) {
      const all = read();
      if (!(sessionId in all)) return;
      delete all[sessionId];
      write(all);
    },
  };
}

/** The one line a note contributes to a row that has no space for more. */
export function noteFirstLine(text: string | undefined): string {
  if (!text) return "";
  return text.split("\n").map((l) => l.trim()).find((l) => l.length > 0) ?? "";
}
