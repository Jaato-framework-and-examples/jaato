/**
 * Cookie sessions: an opaque id in an HMAC-signed cookie, state in memory.
 *
 * The cookie carries ``<id>.<hmac-sha256(id)>``; the browser holds nothing
 * it can read a fact from, and a forged or truncated cookie fails the
 * signature before any lookup.  State lives in this process, so a restart
 * signs everyone out — the accepted trade for a first version (a shared
 * store is a one-interface change here), and consistent with the daemon's
 * in-memory ticket registry, which a restart also empties.
 *
 * What a session records is what the ticket route and both logout paths
 * need: the ``user`` the daemon will be told, the OIDC ``sub`` and ``sid``
 * (so a back-channel logout token naming either can find it) and the raw
 * ID token (for RP-initiated logout's ``id_token_hint``).
 */
import { createHmac, randomBytes, timingSafeEqual } from "node:crypto";

export interface SessionRecord {
  id: string;
  /** What ``ticket.bind`` is told; the daemon qualifies it with the app id. */
  user: string;
  sub: string;
  sid?: string;
  idToken?: string;
  createdAt: number;
  expiresAt: number;
}

export class SessionStore {
  private readonly _sessions = new Map<string, SessionRecord>();

  constructor(
    private readonly _secret: string,
    private readonly _ttlSeconds: number,
    private readonly _now: () => number = () => Date.now(),
  ) {}

  private _sign(id: string): string {
    return createHmac("sha256", this._secret).update(id).digest("base64url");
  }

  /** Create a session; returns the record and the cookie VALUE to set. */
  create(fields: Omit<SessionRecord, "id" | "createdAt" | "expiresAt">): { record: SessionRecord; cookieValue: string } {
    const id = randomBytes(32).toString("base64url");
    const now = this._now();
    const record: SessionRecord = { id, createdAt: now, expiresAt: now + this._ttlSeconds * 1000, ...fields };
    this._sessions.set(id, record);
    return { record, cookieValue: `${id}.${this._sign(id)}` };
  }

  /** Resolve a cookie value to a live session, or ``null`` for anything else. */
  resolve(cookieValue: string | undefined): SessionRecord | null {
    if (!cookieValue) return null;
    const dot = cookieValue.lastIndexOf(".");
    if (dot <= 0) return null;
    const id = cookieValue.slice(0, dot);
    const sig = cookieValue.slice(dot + 1);
    const expected = this._sign(id);
    const a = Buffer.from(sig), b = Buffer.from(expected);
    if (a.length !== b.length || !timingSafeEqual(a, b)) return null;
    const rec = this._sessions.get(id);
    if (!rec) return null;
    if (rec.expiresAt <= this._now()) { this._sessions.delete(id); return null; }
    return rec;
  }

  delete(id: string): boolean {
    return this._sessions.delete(id);
  }

  /** End every session for an OIDC subject or session id (back-channel logout). Returns the users affected. */
  deleteMatching(match: { sub?: string; sid?: string }): SessionRecord[] {
    const gone: SessionRecord[] = [];
    for (const rec of this._sessions.values()) {
      if ((match.sid && rec.sid === match.sid) || (match.sub && rec.sub === match.sub)) {
        this._sessions.delete(rec.id);
        gone.push(rec);
      }
    }
    return gone;
  }

  /** Drop expired records; call periodically. */
  sweep(): number {
    const now = this._now();
    let n = 0;
    for (const [id, rec] of this._sessions) if (rec.expiresAt <= now) { this._sessions.delete(id); n++; }
    return n;
  }

  get size(): number { return this._sessions.size; }
}

/** Parse a ``Cookie`` header into a map (first occurrence wins). */
export function parseCookies(header: string | undefined): Map<string, string> {
  const out = new Map<string, string>();
  if (!header) return out;
  for (const part of header.split(";")) {
    const i = part.indexOf("=");
    if (i < 0) continue;
    const k = part.slice(0, i).trim();
    if (k && !out.has(k)) out.set(k, part.slice(i + 1).trim());
  }
  return out;
}

/** ``Set-Cookie`` for the session: HttpOnly, SameSite=Lax (the OIDC callback is a top-level navigation), Secure on https. */
export function sessionCookie(name: string, value: string, opts: { secure: boolean; maxAgeSeconds?: number; path?: string }): string {
  const parts = [`${name}=${value}`, `Path=${opts.path ?? "/"}`, "HttpOnly", "SameSite=Lax"];
  if (opts.secure) parts.push("Secure");
  if (opts.maxAgeSeconds !== undefined) parts.push(`Max-Age=${Math.max(0, Math.floor(opts.maxAgeSeconds))}`);
  return parts.join("; ");
}
