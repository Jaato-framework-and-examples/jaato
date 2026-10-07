import { strict as assert } from "node:assert";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, beforeEach, describe, test } from "node:test";
import { BindChannel } from "../src/bind-channel.js";
import { FileGitHubStore } from "../src/github-store.js";
import { GitHubService, GH_TOKEN_ENV_VALUE } from "../src/github.js";
import { createRouter } from "../src/routes.js";
import { SessionStore } from "../src/session.js";
import { APP_CREDENTIAL, FakeIdp, startMockDaemon, testConfig, type MockDaemon } from "./helpers.js";
import { FakeGitHubApi } from "./github-fakes.js";

/** The GitHub connect / bind / disconnect routes, over real HTTP against a mock daemon + fake IdP + fake GitHub. */
describe("github routes", () => {
  let daemon: MockDaemon; let http: Server; let base: string; let idp: FakeIdp; let bind: BindChannel;
  let sessions: SessionStore; let api: FakeGitHubApi; let github: GitHubService; let wsRoot: string;

  before(async () => {
    daemon = await startMockDaemon();
    const dist = mkdtempSync(join(tmpdir(), "jwcs-dist-"));
    writeFileSync(join(dist, "index.html"), "<!doctype html><div id=root>ui</div>");
    idp = new FakeIdp();
    http = createServer();
    await new Promise<void>((r) => http.listen(0, "127.0.0.1", r));
    base = `http://127.0.0.1:${(http.address() as { port: number }).port}`;
    const config = testConfig(daemon.url, base);
    bind = new BindChannel({ bindUrl: daemon.url, appCredential: APP_CREDENTIAL });
    await bind.connect();
    sessions = new SessionStore(config.session.secret, config.session.ttlSeconds);
    wsRoot = mkdtempSync(join(tmpdir(), "jwcs-wsroot-"));
    const store = new FileGitHubStore(join(mkdtempSync(join(tmpdir(), "jwcs-gh-")), "github.json"), "g".repeat(48));
    api = new FakeGitHubApi();
    github = new GitHubService({ store, api, reloader: bind, workspaceRoot: wsRoot });
    // The daemon asks this app to resolve app://github over the bind channel;
    // server.ts wires this, so the routes test must too, or askSecretResolve hangs.
    bind.attachSecretResolver(github.resolveSecret);
    http.on("request", createRouter({ config, idp, sessions, bind, github, distDir: dist, log: () => undefined }));
  });
  after(async () => { await bind.close(); await new Promise<void>((r) => http.close(() => r())); await daemon.close(); });
  beforeEach(() => { daemon.reloads.length = 0; idp.refuse = null; });

  const noRedirect = { redirect: "manual" as const };

  async function signIn(user = "alice"): Promise<string> {
    const login = await fetch(`${base}/auth/login`, noRedirect);
    const state = new URL(login.headers.get("location")!).searchParams.get("state")!;
    const cb = await fetch(`${base}/auth/callback?code=c&state=${state}&user=${user}`, noRedirect);
    return cb.headers.get("set-cookie")!.split(";")[0]!;
  }

  /** Run the GitHub App OAuth round-trip; returns nothing but leaves the grant stored. */
  async function connectGitHub(cookie: string): Promise<void> {
    const login = await fetch(`${base}/auth/github/login`, { headers: { cookie }, ...noRedirect });
    assert.equal(login.status, 302);
    const loc = new URL(login.headers.get("location")!);
    assert.equal(loc.origin, "https://github.com");
    const state = loc.searchParams.get("state")!;
    const cb = await fetch(`${base}/auth/github/callback?code=abc&state=${state}`, noRedirect);
    assert.equal(cb.status, 302);
    assert.equal(cb.headers.get("location"), `${base}/?github=connected`);
  }

  test("connect → accounts lists the login and installations, never a token", async () => {
    const cookie = await signIn("alice");
    await connectGitHub(cookie);
    const r = await fetch(`${base}/api/github/accounts`, { headers: { cookie } });
    const body = await r.json() as { accounts: Array<{ login: string; isDefault: boolean; installations: unknown[] }> };
    assert.equal(body.accounts.length, 1);
    assert.equal(body.accounts[0]!.login, "alice");
    assert.equal(body.accounts[0]!.isDefault, true);
    assert.deepEqual(body.accounts[0]!.installations, [{ id: 7, account: "acme" }]);
    assert.ok(!JSON.stringify(body).includes("access-"), "no access token in the response");
    assert.ok(!JSON.stringify(body).includes("refresh-"), "no refresh token in the response");
  });

  test("there is NO reveal route for github", async () => {
    const cookie = await signIn("alice");
    const r = await fetch(`${base}/api/github/accounts/anything/reveal`, { method: "POST", headers: { cookie, "sec-fetch-site": "same-origin" } });
    assert.equal(r.status, 404);
  });

  test("bind writes GH_TOKEN into the workspace .env, seeds .gitconfig, and reloads the user's sessions", async () => {
    const cookie = await signIn("alice");
    await connectGitHub(cookie);
    const accounts = await (await fetch(`${base}/api/github/accounts`, { headers: { cookie } })).json() as { accounts: Array<{ id: string }> };
    const id = accounts.accounts[0]!.id;
    const ws = join(wsRoot, "proj");
    mkdirSync(ws, { recursive: true });
    daemon.nextReloadCount = 1;
    const r = await fetch(`${base}/api/github/bind`, {
      method: "POST", headers: { cookie, "sec-fetch-site": "same-origin", "content-type": "application/json" },
      body: JSON.stringify({ workspace: ws, account_id: id }),
    });
    assert.equal(r.status, 200);
    const body = await r.json() as { binding: string; envWritten: boolean; gitconfigSeeded: boolean; reloaded: number };
    assert.equal(body.binding, "set");
    assert.equal(body.envWritten, true);
    assert.equal(body.gitconfigSeeded, true);
    assert.equal(body.reloaded, 1);
    assert.equal(readFileSync(join(ws, ".env"), "utf8").trim(), `GH_TOKEN=${GH_TOKEN_ENV_VALUE}`);
    assert.deepEqual(daemon.reloads, ["alice"]);
    // The bound token is now resolvable by the daemon over secret.resolve.
    const resolved = await daemon.askSecretResolve({ request_id: "rr-1", user: "alice", workspace: ws, name: "github" });
    assert.equal(resolved.status, "ok");
    assert.ok(String(resolved.value).startsWith("access-"));
  });

  test("another user's workspace resolves to that OTHER user's token, or none", async () => {
    // alice connected + bound above; bob has not.
    const ws = join(wsRoot, "proj");
    const forBob = await daemon.askSecretResolve({ request_id: "rr-2", user: "bob", workspace: ws, name: "github" });
    assert.equal(forBob.status, "not_found");
    assert.match(String(forBob.detail), /not_connected|not_bound/);
  });

  test("disconnect revokes at GitHub, drops the binding, and reloads; the next resolve is not-connected", async () => {
    const cookie = await signIn("carol");
    await connectGitHub(cookie);
    const accounts = await (await fetch(`${base}/api/github/accounts`, { headers: { cookie } })).json() as { accounts: Array<{ id: string }> };
    const id = accounts.accounts[0]!.id;
    const ws = join(wsRoot, "carol-proj");
    mkdirSync(ws, { recursive: true });
    await fetch(`${base}/api/github/bind`, { method: "POST", headers: { cookie, "sec-fetch-site": "same-origin", "content-type": "application/json" }, body: JSON.stringify({ workspace: ws, account_id: id }) });
    daemon.reloads.length = 0;
    const before = api.revokes.length;
    const r = await fetch(`${base}/api/github/disconnect`, { method: "POST", headers: { cookie, "sec-fetch-site": "same-origin", "content-type": "application/json" }, body: JSON.stringify({ id }) });
    assert.equal(r.status, 200);
    assert.equal(api.revokes.length, before + 1, "revoked at GitHub");
    assert.deepEqual(daemon.reloads, ["carol"]);
    const resolved = await daemon.askSecretResolve({ request_id: "rr-3", user: "carol", workspace: ws, name: "github" });
    assert.match(String(resolved.detail), /not_connected/);
  });

  test("the mutating routes refuse a cross-site request and an unauthenticated one", async () => {
    const cookie = await signIn("dave");
    // cross-site (no Sec-Fetch-Site: same-origin, no matching Origin)
    const xsite = await fetch(`${base}/api/github/bind`, { method: "POST", headers: { cookie, "content-type": "application/json", origin: "https://evil.example" }, body: JSON.stringify({ workspace: "/x", account_id: null }) });
    assert.equal(xsite.status, 403);
    // unauthenticated
    const anon = await fetch(`${base}/api/github/accounts`);
    assert.equal(anon.status, 401);
  });

  test("the github routes 404 when the service is not configured", async () => {
    const dist = mkdtempSync(join(tmpdir(), "jwcs-dist2-"));
    writeFileSync(join(dist, "index.html"), "x");
    const config = testConfig(daemon.url, base);
    const router = createRouter({ config, idp, sessions, bind, distDir: dist, log: () => undefined }); // no github
    const srv = createServer(router);
    await new Promise<void>((r) => srv.listen(0, "127.0.0.1", r));
    const b2 = `http://127.0.0.1:${(srv.address() as { port: number }).port}`;
    const cookie = await signIn("erin");
    const r = await fetch(`${b2}/api/github/accounts`, { headers: { cookie } });
    assert.equal(r.status, 404);
    await new Promise<void>((r) => srv.close(() => r()));
  });

  test("repos lists across two installations, de-duplicated and sorted, with no token in the body", async () => {
    api.identity = { ...api.identity, installations: [{ id: 7, account: "acme" }, { id: 8, account: "alice" }] };
    api.installationRepos.set(7, [
      { fullName: "acme/api", private: true, defaultBranch: "main", pushedAt: "2026-09-01T00:00:00Z" },
      { fullName: "acme/shared", private: false, defaultBranch: "trunk", pushedAt: "2026-09-20T00:00:00Z" },
    ]);
    api.installationRepos.set(8, [
      { fullName: "acme/shared", private: false, defaultBranch: "trunk", pushedAt: "2026-09-20T00:00:00Z" },
      { fullName: "alice/dots", private: false, defaultBranch: "master" },
      { fullName: "alice/blog", private: false, defaultBranch: "main", pushedAt: "2026-09-10T00:00:00Z" },
    ]);
    try {
      const cookie = await signIn("frank");
      await connectGitHub(cookie);
      const before = api.repoCalls.length;
      const r = await fetch(`${base}/api/github/repos`, { headers: { cookie } });
      assert.equal(r.status, 200);
      const raw = await r.text();
      const body = JSON.parse(raw) as { account: { id: string; login: string }; repos: Array<{ fullName: string; private: boolean; defaultBranch: string; pushedAt?: string }> };
      assert.equal(body.account.login, "alice");
      assert.deepEqual(body.repos.map((x) => x.fullName), ["acme/shared", "alice/blog", "acme/api", "alice/dots"]);
      assert.equal(body.repos.find((x) => x.fullName === "acme/api")!.private, true);
      assert.equal(api.repoCalls.length - before, 2, "one listing per installation");
      assert.ok(api.repoCalls.slice(before).every((c) => c.token.startsWith("access-")), "listed with the user's token");
      assert.ok(!raw.includes("access-") && !raw.includes("refresh-"), "no token in the response");
      // Cached: a second read inside the window does not reach GitHub.
      const again = await fetch(`${base}/api/github/repos?account=${encodeURIComponent(body.account.id)}`, { headers: { cookie } });
      assert.equal(again.status, 200);
      assert.equal(api.repoCalls.length - before, 2, "served from the cache");
      // Branches: default branch taken from the cached listing; still no token.
      api.branches.set("acme/shared", ["trunk", "feature/x"]);
      const b = await fetch(`${base}/api/github/branches?repo=acme/shared`, { headers: { cookie } });
      assert.equal(b.status, 200);
      const braw = await b.text();
      assert.deepEqual(JSON.parse(braw), { repo: "acme/shared", defaultBranch: "trunk", branches: ["trunk", "feature/x"] });
      assert.ok(!braw.includes("access-") && !braw.includes("refresh-"), "no token in the branches response");
    } finally {
      api.identity = { ...api.identity, installations: [{ id: 7, account: "acme" }] };
      api.installationRepos.clear();
      api.branches.clear();
    }
  });

  test("repos and branches answer 404 when no GitHub account is connected, and 401 when not signed in", async () => {
    const cookie = await signIn("gina");
    const r = await fetch(`${base}/api/github/repos`, { headers: { cookie } });
    assert.equal(r.status, 404);
    assert.deepEqual(await r.json(), { error: "no GitHub account connected" });
    const b = await fetch(`${base}/api/github/branches?repo=a/b`, { headers: { cookie } });
    assert.equal(b.status, 404);
    assert.equal((await fetch(`${base}/api/github/repos`)).status, 401);
    assert.equal((await fetch(`${base}/api/github/branches?repo=a/b`)).status, 401);
  });

  test("repos refuses an account that is not the user's own", async () => {
    const cookie = await signIn("hank");
    await connectGitHub(cookie);
    const r = await fetch(`${base}/api/github/repos?account=not-mine`, { headers: { cookie } });
    assert.equal(r.status, 404);
  });

  test("branches validates repo before reaching GitHub", async () => {
    const cookie = await signIn("ivy");
    await connectGitHub(cookie);
    const before = api.branchCalls.length;
    for (const bad of ["", "noslash", "a/b/c", "../etc", "a/..", "a b/c", "owner/na%2Fme", "owner/"]) {
      const r = await fetch(`${base}/api/github/branches?repo=${encodeURIComponent(bad)}`, { headers: { cookie } });
      assert.equal(r.status, 400, `refused ${JSON.stringify(bad)}`);
    }
    assert.equal(api.branchCalls.length, before, "no GitHub call for an invalid repo");
  });
});
