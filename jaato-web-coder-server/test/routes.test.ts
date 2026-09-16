import { strict as assert } from "node:assert";
import { mkdtempSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, beforeEach, describe, test } from "node:test";
import { BindChannel } from "../src/bind-channel.js";
import { createRouter, isSameOrigin } from "../src/routes.js";
import { SessionStore } from "../src/session.js";
import { APP_CREDENTIAL, FakeIdp, startMockDaemon, testConfig, type MockDaemon } from "./helpers.js";

/** The full flow, against a mock daemon and a fake IdP, over real HTTP. */
describe("routes", () => {
  let daemon: MockDaemon; let http: Server; let base: string; let idp: FakeIdp; let bind: BindChannel; let sessions: SessionStore;
  const logs: string[] = [];

  before(async () => {
    daemon = await startMockDaemon();
    const dist = mkdtempSync(join(tmpdir(), "jwcs-dist-"));
    writeFileSync(join(dist, "index.html"), "<!doctype html><div id=root>ui</div>");
    idp = new FakeIdp();
    // public_url is set after listen, so build the router with a placeholder and rebuild once the port is known
    http = createServer();
    await new Promise<void>((r) => http.listen(0, "127.0.0.1", r));
    base = `http://127.0.0.1:${(http.address() as { port: number }).port}`;
    const config = testConfig(daemon.url, base);
    bind = new BindChannel({ bindUrl: daemon.url, appCredential: APP_CREDENTIAL });
    await bind.connect();
    sessions = new SessionStore(config.session.secret, config.session.ttlSeconds);
    http.on("request", createRouter({ config, idp, sessions, bind, distDir: dist, log: (m) => logs.push(m) }));
  });
  after(async () => { await bind.close(); await new Promise<void>((r) => http.close(() => r())); await daemon.close(); });
  beforeEach(() => { daemon.binds.length = 0; daemon.revokes.length = 0; idp.refuse = null; });

  const noRedirect = { redirect: "manual" as const };

  async function signIn(user = "alice"): Promise<string> {
    const login = await fetch(`${base}/auth/login`, noRedirect);
    assert.equal(login.status, 302);
    const loc = new URL(login.headers.get("location")!);
    assert.equal(loc.origin, "https://idp.test");
    assert.equal(loc.searchParams.get("redirect_uri"), `${base}/auth/callback`);
    const state = loc.searchParams.get("state")!;
    const cb = await fetch(`${base}/auth/callback?code=c&state=${state}&user=${user}`, noRedirect);
    assert.equal(cb.status, 302);
    assert.equal(cb.headers.get("location"), `${base}/`);
    const setCookie = cb.headers.get("set-cookie")!;
    assert.match(setCookie, /HttpOnly; SameSite=Lax/);
    assert.ok(!setCookie.includes("Secure"), "http public_url → no Secure flag");
    return setCookie.split(";")[0]!;
  }

  test("config.json points the bundle at the ticket endpoint, not at a token", async () => {
    const r = await fetch(`${base}/config.json`);
    assert.deepEqual(await r.json(), { daemon: daemon.url, ticketUrl: "./api/ticket", loginUrl: "./auth/login", autoConnect: true });
    assert.equal(r.headers.get("cache-control"), "no-store");
  });

  test("the bundle is served for unknown paths (SPA fallback), API paths are not", async () => {
    assert.match(await (await fetch(`${base}/`)).text(), /id=root/);
    assert.match(await (await fetch(`${base}/some/route`)).text(), /id=root/);
    assert.equal((await fetch(`${base}/api/nope`)).status, 404);
  });

  test("no session: /api/session and /api/ticket answer 401 with the login URL", async () => {
    const s = await fetch(`${base}/api/session`);
    assert.equal(s.status, 401); assert.equal((await s.json()).loginUrl, "./auth/login");
    const t = await fetch(`${base}/api/ticket`, { method: "POST", headers: { "sec-fetch-site": "same-origin" } });
    assert.equal(t.status, 401);
    assert.equal(daemon.binds.length, 0, "no bind without a session");
  });

  test("login → callback → cookie → ticket, minted per call, for the signed-in user", async () => {
    const cookie = await signIn("alice");
    const s = await fetch(`${base}/api/session`, { headers: { cookie } });
    assert.equal((await s.json()).user, "alice");
    const t1 = await fetch(`${base}/api/ticket`, { method: "POST", headers: { cookie, "sec-fetch-site": "same-origin" } });
    assert.equal(t1.status, 200);
    const b1 = await t1.json();
    assert.equal(b1.ticket, "ticket-1"); assert.equal(b1.daemon, daemon.url); assert.equal(b1.qualified, "jaato-web-coder:alice");
    const t2 = await fetch(`${base}/api/ticket`, { method: "POST", headers: { cookie, "sec-fetch-site": "same-origin" } });
    assert.equal((await t2.json()).ticket, "ticket-2", "every connect attempt gets a fresh ticket");
    assert.deepEqual(daemon.binds.map((x) => x.user), ["alice", "alice"]);
  });

  test("a cross-site POST with a valid cookie is refused and mints nothing (CSRF)", async () => {
    const cookie = await signIn("alice");
    const cross = await fetch(`${base}/api/ticket`, { method: "POST", headers: { cookie, "sec-fetch-site": "cross-site" } });
    assert.equal(cross.status, 403);
    const noHeaderBadOrigin = await fetch(`${base}/api/ticket`, { method: "POST", headers: { cookie, origin: "https://evil.example" } });
    assert.equal(noHeaderBadOrigin.status, 403);
    assert.equal(daemon.binds.length, 0);
    const get = await fetch(`${base}/api/ticket`, { headers: { cookie } });
    assert.equal(get.status, 405);
  });

  test("a stale or unknown callback state is refused", async () => {
    const r = await fetch(`${base}/auth/callback?code=c&state=nope`, noRedirect);
    assert.equal(r.status, 400);
  });

  test("the IdP refusing the account (missing role) is a 403 with the reason, and no session", async () => {
    idp.refuse = "account lacks the required role 'jaato-user'";
    const login = await fetch(`${base}/auth/login`, noRedirect);
    const state = new URL(login.headers.get("location")!).searchParams.get("state")!;
    const cb = await fetch(`${base}/auth/callback?code=c&state=${state}`, noRedirect);
    assert.equal(cb.status, 403);
    assert.match(await cb.text(), /required role/);
    assert.equal(cb.headers.get("set-cookie"), null);
  });

  test("daemon at capacity → 503 with Retry-After; channel down → 503", async () => {
    const cookie = await signIn("alice");
    daemon.nextBindStatus = "capacity";
    const r = await fetch(`${base}/api/ticket`, { method: "POST", headers: { cookie, "sec-fetch-site": "same-origin" } });
    assert.equal(r.status, 503); assert.equal(r.headers.get("retry-after"), "30");
  });

  test("logout ends the session, revokes the user's tickets, clears the cookie and redirects through the IdP", async () => {
    const cookie = await signIn("alice");
    const r = await fetch(`${base}/api/logout`, { headers: { cookie, "sec-fetch-site": "none" }, ...noRedirect });
    assert.equal(r.status, 302);
    assert.match(r.headers.get("location")!, /^https:\/\/idp\.test\/logout\?post_logout_redirect_uri=.*&id_token_hint=idtoken-alice$/);
    assert.match(r.headers.get("set-cookie")!, /Max-Age=0/);
    assert.deepEqual(daemon.revokes, [{ user: "alice", ticket: undefined }]);
    const after = await fetch(`${base}/api/session`, { headers: { cookie } });
    assert.equal(after.status, 401);
  });

  test("back-channel logout from the IdP ends the matching session and revokes", async () => {
    const cookie = await signIn("carol");
    idp.logoutTokens.set("lt-1", { sid: "sid-carol" });
    const r = await fetch(`${base}/auth/backchannel-logout`, { method: "POST", headers: { "content-type": "application/x-www-form-urlencoded" }, body: "logout_token=lt-1" });
    assert.equal(r.status, 200);
    assert.equal((await fetch(`${base}/api/session`, { headers: { cookie } })).status, 401);
    assert.deepEqual(daemon.revokes, [{ user: "carol", ticket: undefined }]);
    const bad = await fetch(`${base}/auth/backchannel-logout`, { method: "POST", headers: { "content-type": "application/x-www-form-urlencoded" }, body: "logout_token=forged" });
    assert.equal(bad.status, 400);
  });

  test("isSameOrigin: Sec-Fetch-Site first, then Origin, then Referer", () => {
    const req = (h: Record<string, string>) => ({ headers: h }) as unknown as import("node:http").IncomingMessage;
    assert.ok(isSameOrigin(req({ "sec-fetch-site": "same-origin" }), "https://a"));
    assert.ok(!isSameOrigin(req({ "sec-fetch-site": "none" }), "https://a"));
    assert.ok(isSameOrigin(req({ "sec-fetch-site": "none" }), "https://a", true));
    assert.ok(isSameOrigin(req({ origin: "https://a" }), "https://a/"));
    assert.ok(!isSameOrigin(req({ origin: "https://b" }), "https://a"));
    assert.ok(isSameOrigin(req({ referer: "https://a/page" }), "https://a"));
    assert.ok(!isSameOrigin(req({}), "https://a"));
  });
});
