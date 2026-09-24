/**
 * Shared test doubles: a mock daemon that speaks the protocol-1.10 ticket
 * verbs over ``ws``, a fake identity provider, and a config factory that
 * writes real 0600 secret files into a temp dir (the loader refuses
 * anything looser, so the tests exercise that path rather than bypass it).
 */
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { WebSocketServer, type WebSocket } from "ws";
import { configFromObject, type ServerConfig } from "../src/config.js";
import type { AuthenticatedUser, IdentityProvider, LoginStart, LogoutToken } from "../src/auth/identity.js";
import { SignInRefusedError } from "../src/auth/identity.js";

export const APP_CREDENTIAL = "app-credential-0123456789abcdefghijklmnopqrstuv";

export interface MockDaemon {
  url: string;
  port: number;
  binds: Array<{ user: string; ttl_seconds: number; single_use: boolean; auth: string | undefined }>;
  revokes: Array<{ user?: string; ticket?: string }>;
  /** Users the daemon was asked to reload (``secret.reload``, #1226 §6.4). */
  reloads: string[];
  /** How many loaded sessions to report reloaded; default 0. */
  nextReloadCount: number;
  /** Answer the next bind with this status instead of "bound". */
  nextBindStatus: string | null;
  authHeaders: Array<string | undefined>;
  /** Send a ``secret.resolve`` down the app-credential connection and await its ``secret.resolve.result``. */
  askSecretResolve(req: { request_id: string; user: string; workspace: string; name: string }): Promise<Record<string, unknown>>;
  close(): Promise<void>;
}

/** A daemon that accepts an app-credential bind channel and mints "ticket-N". */
export async function startMockDaemon(): Promise<MockDaemon> {
  const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  await new Promise<void>((r) => wss.on("listening", r));
  const port = (wss.address() as { port: number }).port;
  let minted = 0;
  let appSock: WebSocket | null = null;
  const resolvePending = new Map<string, (ev: Record<string, unknown>) => void>();
  const d: MockDaemon = {
    url: `ws://127.0.0.1:${port}`, port, binds: [], revokes: [], reloads: [], nextReloadCount: 0,
    nextBindStatus: null, authHeaders: [],
    askSecretResolve(reqEv) {
      if (!appSock) return Promise.reject(new Error("no app-credential connection to ask"));
      return new Promise((resolve) => {
        resolvePending.set(reqEv.request_id, resolve);
        appSock!.send(JSON.stringify({ type: "secret.resolve", ...reqEv }));
      });
    },
    close: () => new Promise((r) => wss.close(() => r())),
  };
  wss.on("connection", (sock: WebSocket, req) => {
    const auth = req.headers.authorization;
    d.authHeaders.push(auth);
    const isApp = auth === `Bearer ${APP_CREDENTIAL}`;
    if (isApp) appSock = sock;
    sock.send(JSON.stringify({ type: "connected", timestamp: new Date().toISOString(), protocol_version: "1.10", server_info: { client_id: "client_1", server_version: "0.9.0" } }));
    sock.on("message", (raw) => {
      const ev = JSON.parse(String(raw));
      if (ev.type === "ticket.bind") {
        if (!isApp) return sock.send(JSON.stringify({ type: "ticket.bind.result", request_id: ev.request_id, status: "denied", ticket: "" }));
        d.binds.push({ user: ev.user, ttl_seconds: ev.ttl_seconds, single_use: ev.single_use, auth });
        const status = d.nextBindStatus ?? "bound"; d.nextBindStatus = null;
        if (status !== "bound") return sock.send(JSON.stringify({ type: "ticket.bind.result", request_id: ev.request_id, status, ticket: "", detail: `mock ${status}` }));
        minted += 1;
        return sock.send(JSON.stringify({ type: "ticket.bind.result", request_id: ev.request_id, status: "bound", ticket: `ticket-${minted}`, qualified: `jaato-web-coder:${ev.user}`, app_id: "jaato-web-coder", expires_at: new Date(Date.now() + ev.ttl_seconds * 1000).toISOString() }));
      }
      if (ev.type === "ticket.revoke") {
        d.revokes.push({ user: ev.user, ticket: ev.ticket });
        return sock.send(JSON.stringify({ type: "ticket.revoke.result", request_id: ev.request_id, status: "not_found", revoked: 0 }));
      }
      if (ev.type === "secret.reload") {
        if (!isApp) return sock.send(JSON.stringify({ type: "secret.reload.result", request_id: ev.request_id, status: "denied", reloaded: 0 }));
        d.reloads.push(ev.user);
        const reloaded = d.nextReloadCount; d.nextReloadCount = 0;
        return sock.send(JSON.stringify({ type: "secret.reload.result", request_id: ev.request_id, status: "ok", reloaded }));
      }
      // The application's answer to a daemon-initiated secret.resolve.
      if (ev.type === "secret.resolve.result") {
        const cb = resolvePending.get(ev.request_id);
        if (cb) { resolvePending.delete(ev.request_id); cb(ev); }
        return;
      }
      if (isApp) return sock.send(JSON.stringify({ type: "error", error: `'${ev.type}' is not available on an application-credential connection`, error_type: "UsageError", recoverable: true }));
    });
  });
  return d;
}

/** An identity provider whose callback trusts a ``?user=`` parameter — the routes, not OIDC, are under test. */
export class FakeIdp implements IdentityProvider {
  logins: LoginStart[] = [];
  logoutTokens = new Map<string, LogoutToken>();
  refuse: string | null = null;
  async startLogin(redirectUri: string): Promise<LoginStart> {
    const state = `state-${this.logins.length + 1}`;
    const s = { url: `https://idp.test/authorize?redirect_uri=${encodeURIComponent(redirectUri)}&state=${state}`, state, pending: { state, nonce: "n" } };
    this.logins.push(s);
    return s;
  }
  async completeLogin(cb: URL, pending: Record<string, string>): Promise<AuthenticatedUser> {
    if (cb.searchParams.get("state") !== pending.state) throw new Error("state mismatch");
    if (this.refuse) throw new SignInRefusedError(this.refuse);
    const user = cb.searchParams.get("user") ?? "alice";
    return { sub: `sub-${user}`, user, sid: `sid-${user}`, idToken: `idtoken-${user}` };
  }
  endSessionUrl(idToken: string | undefined, post: string): string | null {
    return `https://idp.test/logout?post_logout_redirect_uri=${encodeURIComponent(post)}${idToken ? `&id_token_hint=${idToken}` : ""}`;
  }
  async verifyLogoutToken(token: string): Promise<LogoutToken> {
    const lt = this.logoutTokens.get(token);
    if (!lt) throw new Error("bad logout token");
    return lt;
  }
}

export function writeSecrets(dir = mkdtempSync(join(tmpdir(), "jwcs-"))): { dir: string } {
  writeFileSync(join(dir, "app.credential"), APP_CREDENTIAL + "\n", { mode: 0o600 });
  writeFileSync(join(dir, "oidc.secret"), "client-secret\n", { mode: 0o600 });
  writeFileSync(join(dir, "session.secret"), "s".repeat(48) + "\n", { mode: 0o600 });
  writeFileSync(join(dir, "credentials.key"), "c".repeat(48) + "\n", { mode: 0o600 });
  writeFileSync(join(dir, "github.key"), "g".repeat(48) + "\n", { mode: 0o600 });
  writeFileSync(join(dir, "github-app.secret"), "gh-client-secret\n", { mode: 0o600 });
  return { dir };
}

export function testConfig(daemonUrl: string, publicUrl: string, overrides: Record<string, unknown> = {}): ServerConfig {
  const { dir } = writeSecrets();
  return configFromObject({
    listen: "127.0.0.1:0",
    public_url: publicUrl,
    daemon: { url: daemonUrl, app_id: "jaato-web-coder", app_credential_file: "app.credential" },
    auth: { kind: "oidc", oidc: { issuer: "https://idp.test/realms/jaato-web-coder-shell", client_id: "jaato-web-coder", client_secret_file: "oidc.secret" } },
    session: { secret_file: "session.secret", ttl: "1h" },
    ticket: { ttl_seconds: 60 },
    ...overrides,
  }, dir);
}
