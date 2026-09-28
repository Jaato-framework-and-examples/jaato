import { strict as assert } from "node:assert";
import { mkdtempSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, describe, test } from "node:test";
import { BindChannel } from "../src/bind-channel.js";
import { EnvironmentService } from "../src/environment/service.js";
import { FileEnvironmentStore } from "../src/environment/store.js";
import { createRouter } from "../src/routes.js";
import { SessionStore } from "../src/session.js";
import { APP_CREDENTIAL, FakeIdp, startMockDaemon, testConfig, type MockDaemon } from "./helpers.js";

/** The ``/api/environment`` routes over real HTTP, with ownership asked of a mock daemon over the real bind channel. */
describe("environment routes", () => {
  let daemon: MockDaemon; let http: Server; let base: string; let bind: BindChannel;
  // A workspace this process cannot see: the routes must not need to.
  const ws = "/root/.jaato/workspaces/alice-1";

  before(async () => {
    daemon = await startMockDaemon({ protocolVersion: "1.30" });
    const dist = mkdtempSync(join(tmpdir(), "jwcs-dist-"));
    writeFileSync(join(dist, "index.html"), "<!doctype html>");
    http = createServer();
    await new Promise<void>((r) => http.listen(0, "127.0.0.1", r));
    base = `http://127.0.0.1:${(http.address() as { port: number }).port}`;
    const config = testConfig(daemon.url, base);
    bind = new BindChannel({ bindUrl: daemon.url, appCredential: APP_CREDENTIAL });
    await bind.connect();
    daemon.owners.set(ws, "alice");
    const env = new EnvironmentService({
      tools: { node: ["22"] }, servers: {}, paranoid: false, installTimeoutSeconds: 900,
      store: new FileEnvironmentStore(join(mkdtempSync(join(tmpdir(), "jwcs-st-")), "st.json")), ownership: bind,
    });
    const sessions = new SessionStore(config.session.secret, config.session.ttlSeconds);
    http.on("request", createRouter({ config, idp: new FakeIdp(), sessions, bind, environment: env, distDir: dist, log: () => undefined }));
  });
  after(async () => { await bind.close(); await new Promise<void>((r) => http.close(() => r())); await daemon.close(); });

  async function signIn(user: string): Promise<string> {
    const login = await fetch(`${base}/auth/login`, { redirect: "manual" });
    const state = new URL(login.headers.get("location")!).searchParams.get("state")!;
    const cb = await fetch(`${base}/auth/callback?code=c&state=${state}&user=${user}`, { redirect: "manual" });
    return cb.headers.get("set-cookie")!.split(";")[0]!;
  }
  const post = (cookie: string, leaf: string, body: unknown, sameOrigin = true) => fetch(`${base}/api/environment/${leaf}`, {
    method: "POST", headers: { cookie, "content-type": "application/json", ...(sameOrigin ? { "sec-fetch-site": "same-origin" } : { "sec-fetch-site": "cross-site" }) }, body: JSON.stringify(body),
  });
  const status = (cookie: string) => fetch(`${base}/api/environment?workspace=${encodeURIComponent(ws)}`, { headers: { cookie } });

  test("config.json names the environment URL", async () => {
    const r = await fetch(`${base}/config.json`);
    assert.equal((await r.json() as { environmentUrl?: string }).environmentUrl, "./api/environment");
  });

  test("the owner gets the allow-list and the offer for a workspace this process cannot read", async () => {
    const cookie = await signIn("alice");
    const st = await (await status(cookie)).json() as { allowed: Array<{ tool: string }>; offer: { path: string; content: string } };
    assert.deepEqual(st.allowed.map((a) => a.tool), ["node"]);
    assert.equal(JSON.parse(st.offer.content).toolchains[0].tool, "node");
  });

  test("a decline is remembered and undone", async () => {
    const cookie = await signIn("alice");
    assert.equal((await post(cookie, "decline", { workspace: ws, tool: "node" })).status, 200);
    assert.deepEqual((await (await status(cookie)).json() as { declined: string[] }).declined, ["node"]);
    assert.equal((await post(cookie, "undecline", { workspace: ws, tool: "node" })).status, 200);
    assert.deepEqual((await (await status(cookie)).json() as { declined: string[] }).declined, []);
  });

  test("another user is refused as if the workspace did not exist", async () => {
    const cookie = await signIn("bob");
    assert.equal((await status(cookie)).status, 404);
    assert.equal((await post(cookie, "decline", { workspace: ws, tool: "node" })).status, 404);
  });

  test("cross-site, malformed and removed requests are refused", async () => {
    const cookie = await signIn("alice");
    assert.equal((await post(cookie, "decline", { workspace: ws, tool: "node" }, false)).status, 403);
    assert.equal((await post(cookie, "decline", { workspace: ws, tool: "rust" })).status, 400);
    assert.equal((await post(cookie, "bind", { workspace: ws, tool: "node", version: "22" })).status, 404, "installs are the plugin's, not a route");
    assert.equal((await fetch(`${base}/api/environment?workspace=x`)).status, 401);
  });
});
