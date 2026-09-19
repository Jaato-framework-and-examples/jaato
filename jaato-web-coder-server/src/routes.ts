/**
 * The HTTP surface of jaato-web-coder-server.
 *
 * | Route | Method | Does |
 * |---|---|---|
 * | ``/config.json`` | GET | tells the bundle where the daemon is and where to ask for tickets |
 * | ``/auth/login`` | GET | starts the OIDC login (302 to the issuer) |
 * | ``/auth/callback`` | GET | finishes it, sets the session cookie, 302 to ``/`` |
 * | ``/auth/backchannel-logout`` | POST | OIDC back-channel logout from the issuer |
 * | ``/api/session`` | GET | 200 ``{user}`` or 401 |
 * | ``/api/ticket`` | POST | mints one single-use ticket for the signed-in user (same-origin only) |
 * | ``/api/logout`` | GET | ends the session, revokes tickets, redirects through the issuer's logout |
 * | ``/api/credentials[?provider=]`` | GET | the signed-in user's stored provider keys: labels and hints, never secrets |
 * | ``/api/credentials`` | POST | store one ``{provider, secret, label?}`` (same-origin only) |
 * | ``/api/credentials/<id>/reveal`` | POST | the secret behind one entry, for the page to forward to the daemon (same-origin only) |
 * | ``/api/credentials/<id>`` | DELETE | forget one entry (same-origin only) |
 * | ``/api/notes`` | GET | every note this user has written, ``{sessionId, text, updatedAt}`` |
 * | ``/api/notes/<session_id>`` | PUT | write one (same-origin only); an empty body forgets it |
 * | ``/api/notes/<session_id>`` | DELETE | forget one (same-origin only) |
 * | anything else | GET | the bundle (``@jaato/web-coder-ui``'s static handler) |
 *
 * The four credential routes exist only when the config carries a
 * ``credentials:`` block (``src/credentials.ts``); otherwise they are 404
 * like any other unknown ``/api/`` path and ``config.json`` names no
 * ``credentialsUrl``.  The three note routes answer to a ``notes:`` block
 * (``src/notes.ts``) the same way, and their absence is not an error the
 * page reports: with no ``notesUrl`` it keeps notes in ``localStorage``
 * instead, and says which it is doing.
 *
 * Every route is relative to the mount point: the bundle asks for
 * ``./config.json`` and ``./api/ticket`` relative to itself, so serving the
 * whole thing under ``/`` behind a reverse proxy is what this assumes.
 */
import type { IncomingMessage, ServerResponse } from "node:http";
import { createStaticHandler, DIST_DIR } from "@jaato/web-coder-ui";
import type { ServerConfig } from "./config.js";
import { BindChannel, BindRefusedError, BindUnavailableError } from "./bind-channel.js";
import { CredentialError, type CredentialStore, validateProvider } from "./credentials.js";
import { NoteError, type NoteStore, validateSessionId } from "./notes.js";
import { type IdentityProvider, SignInRefusedError } from "./auth/identity.js";
import { parseCookies, sessionCookie, SessionStore } from "./session.js";

export interface RouterDeps {
  config: ServerConfig;
  idp: IdentityProvider;
  sessions: SessionStore;
  bind: BindChannel;
  /** The per-user key store; absent = the credential routes do not exist. */
  credentials?: CredentialStore;
  /** The per-user session-note store; absent = the note routes do not exist. */
  notes?: NoteStore;
  /** Serve the bundle from here; defaults to the installed @jaato/web-coder-ui dist. */
  distDir?: string;
  log?: (msg: string) => void;
}

type Handler = (req: IncomingMessage, res: ServerResponse) => Promise<void>;

interface PendingLogin { pending: Record<string, string>; createdAt: number }
const PENDING_LOGIN_TTL_MS = 10 * 60 * 1000;

function json(res: ServerResponse, status: number, body: unknown, headers: Record<string, string> = {}): void {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", ...headers });
  res.end(JSON.stringify(body));
}

function text(res: ServerResponse, status: number, body: string, headers: Record<string, string> = {}): void {
  res.writeHead(status, { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store", ...headers });
  res.end(body);
}

function redirect(res: ServerResponse, location: string, headers: Record<string, string> = {}): void {
  res.writeHead(302, { Location: location, "Cache-Control": "no-store", ...headers });
  res.end();
}

/**
 * CSRF guard for the state-changing routes.  A browser sends
 * ``Sec-Fetch-Site`` on every request; ``same-origin`` is what the page's
 * own fetch carries.  ``none`` is a user-typed navigation (allowed for
 * logout, which is a link).  Older browsers without the header are judged
 * by ``Origin`` against ``public_url``.
 */
export function isSameOrigin(req: IncomingMessage, publicUrl: string, allowNavigation = false): boolean {
  const sfs = req.headers["sec-fetch-site"];
  if (typeof sfs === "string") return sfs === "same-origin" || (allowNavigation && sfs === "none");
  const origin = req.headers.origin;
  if (typeof origin === "string") return origin === new URL(publicUrl).origin;
  const referer = req.headers.referer;
  if (typeof referer === "string") { try { return new URL(referer).origin === new URL(publicUrl).origin; } catch { return false; } }
  return allowNavigation;
}

async function readBody(req: IncomingMessage, limit = 64 * 1024): Promise<string> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const c of req) {
    size += (c as Buffer).length;
    if (size > limit) throw new Error("body too large");
    chunks.push(c as Buffer);
  }
  return Buffer.concat(chunks).toString("utf8");
}

export function createRouter(deps: RouterDeps): Handler {
  const { config, idp, sessions, bind, credentials, notes } = deps;
  const log = deps.log ?? (() => undefined);
  const secure = config.publicUrl.startsWith("https://");
  const pendingLogins = new Map<string, PendingLogin>();
  const redirectUri = `${config.publicUrl}/auth/callback`;
  const cookieName = config.session.cookieName;

  const staticHandler = createStaticHandler({
    root: deps.distDir ?? DIST_DIR,
    config: {
      daemon: config.daemon.url, ticketUrl: "./api/ticket", loginUrl: "./auth/login", autoConnect: true,
      // Named only when the store exists: the bundle shows the combobox iff this key is present.
      ...(credentials ? { credentialsUrl: "./api/credentials" } : {}),
      // Likewise: named only when the store exists, so the page knows
      // whether notes are shared across devices or local to this browser.
      ...(notes ? { notesUrl: "./api/notes" } : {}),
    },
    allowedHosts: null,
  }) as Handler;

  /**
   * ``/api/credentials`` and below.  Every route needs the cookie; the
   * mutating ones and ``reveal`` need same-origin too, so a cross-site page
   * cannot read a secret or plant one.  Owner is the session's OIDC ``sub``.
   */
  const handleCredentials = async (req: IncomingMessage, res: ServerResponse, url: URL, method: string, rest: string[]): Promise<void> => {
    if (!credentials) return text(res, 404, "not found");
    const s = sessions.resolve(parseCookies(req.headers.cookie).get(cookieName));
    if (!s) return json(res, 401, { error: "not signed in", loginUrl: "./auth/login" });
    const sameOrigin = () => isSameOrigin(req, config.publicUrl);
    try {
      if (rest.length === 0) {
        if (method === "GET") {
          const provider = url.searchParams.get("provider");
          return json(res, 200, { entries: credentials.list(s.sub, provider === null ? undefined : validateProvider(provider)) });
        }
        if (method === "POST") {
          if (!sameOrigin()) return json(res, 403, { error: "cross-site request refused" });
          let body: Record<string, unknown>;
          try { body = JSON.parse(await readBody(req)) as Record<string, unknown>; }
          catch { return json(res, 400, { error: "body must be JSON" }); }
          if (!body || typeof body !== "object") return json(res, 400, { error: "body must be a JSON object" });
          const entry = credentials.add(s.sub, validateProvider(body.provider), body.secret as string, body.label as string | undefined);
          log(`credential stored for ${s.user}: provider=${entry.provider} id=${entry.id}`);
          return json(res, 201, { entry });
        }
        return text(res, 405, "GET or POST", { Allow: "GET, POST" });
      }
      const id = rest[0]!;
      if (rest.length === 1 && method === "DELETE") {
        if (!sameOrigin()) return json(res, 403, { error: "cross-site request refused" });
        if (!credentials.remove(s.sub, id)) return json(res, 404, { error: "no such credential" });
        log(`credential deleted for ${s.user}: id=${id}`);
        res.writeHead(204, { "Cache-Control": "no-store" }); res.end();
        return;
      }
      if (rest.length === 2 && rest[1] === "reveal" && method === "POST") {
        if (!sameOrigin()) return json(res, 403, { error: "cross-site request refused" });
        const secret = credentials.reveal(s.sub, id);
        if (secret === null) return json(res, 404, { error: "no such credential" });
        log(`credential revealed to ${s.user}: id=${id}`);
        return json(res, 200, { secret });
      }
      return text(res, 404, "not found");
    } catch (e) {
      if (e instanceof CredentialError) return json(res, e.status, { error: e.message });
      throw e;
    }
  };

  /**
   * ``/api/notes`` and below.  Same shape as the credential handler: the
   * cookie for every route, same-origin for the mutating ones, owner is the
   * session's OIDC ``sub``.  Unlike a credential, the TEXT comes back on
   * ``GET`` -- reading it is the whole feature.
   */
  const handleNotes = async (req: IncomingMessage, res: ServerResponse, method: string, rest: string[]): Promise<void> => {
    if (!notes) return text(res, 404, "not found");
    const s = sessions.resolve(parseCookies(req.headers.cookie).get(cookieName));
    if (!s) return json(res, 401, { error: "not signed in", loginUrl: "./auth/login" });
    const sameOrigin = () => isSameOrigin(req, config.publicUrl);
    try {
      if (rest.length === 0) {
        if (method === "GET") return json(res, 200, { notes: notes.list(s.sub) });
        return text(res, 405, "GET", { Allow: "GET" });
      }
      if (rest.length !== 1) return text(res, 404, "not found");
      const sessionId = validateSessionId(rest[0]);
      if (method === "PUT") {
        if (!sameOrigin()) return json(res, 403, { error: "cross-site request refused" });
        let body: Record<string, unknown>;
        try { body = JSON.parse(await readBody(req)) as Record<string, unknown>; }
        catch { return json(res, 400, { error: "body must be JSON" }); }
        if (!body || typeof body !== "object") return json(res, 400, { error: "body must be a JSON object" });
        const note = notes.put(s.sub, sessionId, body.text as string);
        // The note's TEXT is never logged: it is the one thing here written
        // for a person and read by nobody else, and a log is somebody else.
        log(note ? `note saved for ${s.user}: session=${sessionId}` : `note cleared for ${s.user}: session=${sessionId}`);
        return json(res, 200, { note });
      }
      if (method === "DELETE") {
        if (!sameOrigin()) return json(res, 403, { error: "cross-site request refused" });
        // Idempotent on purpose: ExitPrompt's "End session" deletes the note
        // beside the session, and a session that never had one is not an error.
        notes.remove(s.sub, sessionId);
        log(`note deleted for ${s.user}: session=${sessionId}`);
        res.writeHead(204, { "Cache-Control": "no-store" }); res.end();
        return;
      }
      return text(res, 405, "PUT or DELETE", { Allow: "PUT, DELETE" });
    } catch (e) {
      if (e instanceof NoteError) return json(res, e.status, { error: e.message });
      throw e;
    }
  };

  const currentSession = (req: IncomingMessage) => sessions.resolve(parseCookies(req.headers.cookie).get(cookieName));
  const clearCookie = () => sessionCookie(cookieName, "", { secure, maxAgeSeconds: 0 });

  const sweepPending = () => {
    const now = Date.now();
    for (const [k, v] of pendingLogins) if (now - v.createdAt > PENDING_LOGIN_TTL_MS) pendingLogins.delete(k);
  };

  return async (req, res) => {
    const url = new URL(req.url ?? "/", config.publicUrl);
    const path = url.pathname;
    const method = req.method ?? "GET";

    try {
      if (path === "/auth/login" && method === "GET") {
        sweepPending();
        const start = await idp.startLogin(redirectUri);
        pendingLogins.set(start.state, { pending: start.pending, createdAt: Date.now() });
        return redirect(res, start.url);
      }

      if (path === "/auth/callback" && method === "GET") {
        const state = url.searchParams.get("state") ?? "";
        const entry = pendingLogins.get(state);
        pendingLogins.delete(state);
        if (!entry) return text(res, 400, "Sign-in state is missing or expired. Start again from the application.");
        let user;
        try {
          user = await idp.completeLogin(url, entry.pending, redirectUri);
        } catch (e) {
          if (e instanceof SignInRefusedError) { log(`sign-in refused: ${e.message}`); return text(res, 403, `Sign-in refused: ${e.message}`); }
          log(`sign-in failed: ${(e as Error).message}`);
          return text(res, 400, "Sign-in failed. Start again from the application.");
        }
        const { cookieValue } = sessions.create({ user: user.user, sub: user.sub, sid: user.sid, idToken: user.idToken });
        log(`signed in: ${user.user}`);
        return redirect(res, `${config.publicUrl}/`, { "Set-Cookie": sessionCookie(cookieName, cookieValue, { secure, maxAgeSeconds: config.session.ttlSeconds }) });
      }

      if (path === "/auth/backchannel-logout" && method === "POST") {
        const body = new URLSearchParams(await readBody(req));
        const token = body.get("logout_token");
        if (!token) return text(res, 400, "logout_token required");
        let lt;
        try { lt = await idp.verifyLogoutToken(token); }
        catch (e) { log(`back-channel logout refused: ${(e as Error).message}`); return text(res, 400, "invalid logout token"); }
        const gone = sessions.deleteMatching(lt);
        const users = new Set(gone.map((r) => r.user));
        for (const u of users) { try { await bind.revokeUser(u); } catch (e) { log(`revoke after back-channel logout failed for ${u}: ${(e as Error).message}`); } }
        log(`back-channel logout: ${gone.length} session(s) ended`);
        return text(res, 200, "", { "Cache-Control": "no-store" });
      }

      if (path === "/api/session" && method === "GET") {
        const s = currentSession(req);
        if (!s) return json(res, 401, { error: "not signed in", loginUrl: "./auth/login" });
        return json(res, 200, { user: s.user, expiresAt: new Date(s.expiresAt).toISOString() });
      }

      if (path === "/api/ticket") {
        if (method !== "POST") return text(res, 405, "POST only", { Allow: "POST" });
        const s = currentSession(req);
        if (!s) return json(res, 401, { error: "not signed in", loginUrl: "./auth/login" });
        if (!isSameOrigin(req, config.publicUrl)) return json(res, 403, { error: "cross-site request refused" });
        try {
          const t = await bind.bind(s.user, config.ticket.ttlSeconds);
          log(`ticket for ${t.qualified}`);
          return json(res, 200, { ticket: t.ticket, daemon: config.daemon.url, expiresAt: t.expiresAt, qualified: t.qualified });
        } catch (e) {
          if (e instanceof BindUnavailableError) { log(`ticket unavailable: ${e.message}`); return json(res, 503, { error: "the daemon cannot issue a ticket right now" }, { "Retry-After": "5" }); }
          if (e instanceof BindRefusedError) {
            if (e.status === "capacity") return json(res, 503, { error: "the daemon is at its ticket ceiling" }, { "Retry-After": "30" });
            log(`ticket refused: ${e.message}`);
            return json(res, 500, { error: `ticket refused: ${e.status}` });
          }
          throw e;
        }
      }

      if (path === "/api/logout" && method === "GET") {
        const s = currentSession(req);
        if (!isSameOrigin(req, config.publicUrl, true)) return text(res, 403, "cross-site request refused");
        if (s) {
          sessions.delete(s.id);
          try { await bind.revokeUser(s.user); } catch (e) { log(`revoke on logout failed for ${s.user}: ${(e as Error).message}`); }
          log(`signed out: ${s.user}`);
        }
        const end = idp.endSessionUrl(s?.idToken, `${config.publicUrl}/`);
        return redirect(res, end ?? `${config.publicUrl}/`, { "Set-Cookie": clearCookie() });
      }

      if (path === "/api/credentials" || path.startsWith("/api/credentials/")) {
        return await handleCredentials(req, res, url, method, path.slice("/api/credentials".length).split("/").filter(Boolean));
      }

      if (path === "/api/notes" || path.startsWith("/api/notes/")) {
        return await handleNotes(req, res, method, path.slice("/api/notes".length).split("/").filter(Boolean));
      }

      if (path.startsWith("/api/") || path.startsWith("/auth/")) return text(res, 404, "not found");

      return await staticHandler(req, res);
    } catch (e) {
      log(`unhandled error on ${method} ${path}: ${(e as Error).stack ?? String(e)}`);
      if (!res.headersSent) text(res, 500, "internal error");
      else res.end();
    }
  };
}
