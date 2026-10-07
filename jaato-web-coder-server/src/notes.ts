/**
 * A note the signed-in person wrote to their future self about one
 * session, kept by THIS application and by nothing else.
 *
 * Why here.  Two session fields already exist and neither is this.
 * ``Session.name`` is free text and create-only -- there is no rename verb
 * anywhere in the tree -- which is the wrong moment, because you learn what
 * a session needs ten turns in.  ``Session.description`` is written by the
 * MODEL (the ``session`` plugin lets an agent name its own session, and the
 * picker shows it as the row's name), so a note written there is overwritten
 * by the next turn.  What the picker tells you today is what the AGENT
 * thought the session was about; what is missing is what YOU meant to do
 * about it.
 *
 * Why not the daemon.  This is the call ``credentials.ts`` already made, and
 * every clause transfers: a note is per signed-in user, is never read by the
 * model, is never read by the daemon, and a ``session.note`` verb would sit
 * in the protocol for exactly one client while the TUI carried it and never
 * called it.  It also removes the one wart a daemon-side design has -- a
 * note stored in the workspace tree is ``readFile``-reachable by the agent,
 * so "for the human" would mean *not injected into the prompt* rather than
 * *unreachable*.  Here it is genuinely unreachable, which is what the
 * feature means.
 *
 * The session id is OPAQUE.  This server never learns what a session is: it
 * stores text under a key, and the browser joins that key against the
 * listing it already holds from ``session.list``.  The id is validated for
 * shape and length only, so a junk key cannot grow the file without bound.
 *
 * Trust posture.  The same envelope as ``credentials.ts`` -- AES-256-GCM
 * under a key derived from a 0600 secret file -- rather than a second
 * storage posture in one process.  A note is not a secret in the way an API
 * key is, but a note about client work is not obviously less sensitive than
 * an API key *label*, which that store already encrypts.  The AAD binds each
 * ciphertext to its owner and session, so editing the plaintext metadata to
 * re-attribute a note makes it undecryptable rather than somebody else's.
 *
 * Owner is the OIDC ``sub``, not the display claim: ``subject_claim`` is
 * configurable and a display name can be re-mapped, while ``sub`` is what
 * the issuer promises to keep stable.  A consequence worth stating because
 * it does not follow from the storage design: a note on a session you did
 * not create is YOUR note about their session.
 *
 * Lifecycle mirrors ``FileCredentialStore``: load once at construction, hold
 * in memory, rewrite the whole file atomically on every mutation (temp file
 * + rename, mode 0600), one process owning the file.
 */
import { createCipheriv, createDecipheriv, hkdfSync, randomBytes } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

/** What a client sees -- which here is everything, because the text IS the feature. */
export interface NoteEntry {
  sessionId: string;
  text: string;
  updatedAt: string;
}

export class NoteError extends Error {
  override name = "NoteError";
  constructor(message: string, readonly status: 400 | 404 | 409 = 400) { super(message); }
}

export interface NoteStore {
  list(owner: string): NoteEntry[];
  /** Upsert.  Empty text REMOVES the note and returns ``null`` -- clearing the box is forgetting. */
  put(owner: string, sessionId: string, text: string): NoteEntry | null;
  remove(owner: string, sessionId: string): boolean;
}

/**
 * Daemon session ids are timestamp-shaped (``20260903_084517``), but this
 * server does not model sessions and must not assume that: the pattern
 * bounds the key rather than describing it.
 */
const SESSION_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
/** Long enough for "waiting on X, then re-run Y" and a few checkboxes; short enough to stay a sticky note. */
export const MAX_NOTE_CHARS = 2000;
/** Per owner.  Notes outlive the sessions they name (see ``stale entries`` in the design), so this is the only bound. */
export const MAX_NOTES_PER_OWNER = 200;

interface StoredNote {
  owner: string;
  sessionId: string;
  updatedAt: string;
  /** base64url: 12-byte nonce, 16-byte tag, ciphertext. */
  nonce: string;
  tag: string;
  ciphertext: string;
}

interface StoreFile { version: 1; notes: StoredNote[] }

export function validateSessionId(sessionId: unknown): string {
  if (typeof sessionId !== "string" || !SESSION_ID_RE.test(sessionId)) {
    throw new NoteError("session id must be 1..128 characters of letters, digits, '.', '_', ':' or '-'");
  }
  return sessionId;
}

/**
 * Notes are multi-line by design (``- [ ]`` checkboxes come free that way),
 * so newlines and tabs survive; other control characters do not, since the
 * value is rendered into a textarea and logged nowhere.
 */
export function validateNote(text: unknown): string {
  if (typeof text !== "string") throw new NoteError("note must be a string");
  const t = text.replace(/\r\n?/g, "\n").replace(/[^\S\n\t]+$/gm, "").trim();
  if (t.length > MAX_NOTE_CHARS) throw new NoteError(`note longer than ${MAX_NOTE_CHARS} characters`);
  if (/[\p{Cc}]/u.test(t.replace(/[\n\t]/g, ""))) throw new NoteError("note must not contain control characters");
  return t;
}

const b64 = (b: Buffer) => b.toString("base64url");
const unb64 = (s: string) => Buffer.from(s, "base64url");

export class FileNoteStore implements NoteStore {
  private readonly _key: Buffer;
  private _notes: StoredNote[] = [];

  /**
   * @param path   the JSON file; created on first write, directory too
   * @param secret the operator's key material (a 0600 secret file's content, >= 32 chars);
   *               the AES key is derived with HKDF, under an info string of this store's
   *               own so one secret file can serve both stores without sharing a key
   */
  constructor(private readonly path: string, secret: string, private readonly now: () => Date = () => new Date()) {
    if (secret.length < 32) throw new NoteError("note key must be at least 32 characters");
    this._key = Buffer.from(hkdfSync("sha256", secret, "", "jaato-web-coder notes v1", 32));
    this._load();
  }

  private _load(): void {
    if (!existsSync(this.path)) return;
    const raw = JSON.parse(readFileSync(this.path, "utf8")) as Partial<StoreFile>;
    if (raw.version !== 1 || !Array.isArray(raw.notes)) throw new NoteError(`${this.path}: not a note store (version ${String(raw.version)})`);
    this._notes = raw.notes;
  }

  private _save(): void {
    mkdirSync(dirname(this.path), { recursive: true, mode: 0o700 });
    const tmp = join(dirname(this.path), `.${randomBytes(6).toString("hex")}.tmp`);
    const body: StoreFile = { version: 1, notes: this._notes };
    writeFileSync(tmp, JSON.stringify(body, null, 1) + "\n", { mode: 0o600 });
    renameSync(tmp, this.path);
  }

  private _aad(owner: string, sessionId: string): Buffer {
    return Buffer.from(`${owner}\0${sessionId}`, "utf8");
  }

  private _decrypt(n: StoredNote): string {
    const d = createDecipheriv("aes-256-gcm", this._key, unb64(n.nonce));
    d.setAAD(this._aad(n.owner, n.sessionId));
    d.setAuthTag(unb64(n.tag));
    return Buffer.concat([d.update(unb64(n.ciphertext)), d.final()]).toString("utf8");
  }

  list(owner: string): NoteEntry[] {
    return this._notes
      .filter((n) => n.owner === owner)
      .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
      .map((n) => ({ sessionId: n.sessionId, text: this._decrypt(n), updatedAt: n.updatedAt }));
  }

  put(owner: string, sessionId: string, text: string): NoteEntry | null {
    if (!owner) throw new NoteError("owner required");
    validateSessionId(sessionId);
    const t = validateNote(text);
    // An emptied box is a forgotten note: the editor debounce-saves whatever
    // is in it, so clearing one has to reach the store as a removal rather
    // than as an empty row that keeps counting against the cap.
    if (!t) { this.remove(owner, sessionId); return null; }

    const existing = this._notes.find((n) => n.owner === owner && n.sessionId === sessionId);
    if (!existing && this._notes.filter((n) => n.owner === owner).length >= MAX_NOTES_PER_OWNER) {
      throw new NoteError(`at most ${MAX_NOTES_PER_OWNER} notes per user; delete one first`, 409);
    }
    const nonce = randomBytes(12);
    const c = createCipheriv("aes-256-gcm", this._key, nonce);
    c.setAAD(this._aad(owner, sessionId));
    const ciphertext = Buffer.concat([c.update(Buffer.from(t, "utf8")), c.final()]);
    const updatedAt = this.now().toISOString();
    const stored: StoredNote = { owner, sessionId, updatedAt, nonce: b64(nonce), tag: b64(c.getAuthTag()), ciphertext: b64(ciphertext) };
    if (existing) Object.assign(existing, stored);
    else this._notes.push(stored);
    this._save();
    return { sessionId, text: t, updatedAt };
  }

  remove(owner: string, sessionId: string): boolean {
    const before = this._notes.length;
    this._notes = this._notes.filter((n) => !(n.owner === owner && n.sessionId === sessionId));
    if (this._notes.length === before) return false;
    this._save();
    return true;
  }
}
