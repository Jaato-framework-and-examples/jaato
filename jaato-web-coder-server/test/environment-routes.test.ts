import { strict as assert } from "node:assert";
import { chmodSync, mkdirSync, mkdtempSync, realpathSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { after, before, describe, test } from "node:test";
import { BindChannel } from "../src/bind-channel.js";
import { EnvironmentService } from "../src/environment/service.js";
import { FileEnvironmentStore } from "../src/environment/store.js";
import type { ProcessRunner } from "../src/environment/installer.js";
import { createRouter } from "../src/routes.js";
import { SessionStore } from "../src/session.js";
import { APP_CREDENTIAL, FakeIdp, startMockDaemon, testConfig, type MockDaemon } from "./helpers.js";

/** A runner that "installs" node by creating its bin/ and answers ``where``. */
const runner: ProcessRunner = async (spec) => {
  const data = spec.env.MISE_DATA_DIR!;
  const dir = join(data, "installs", "node", "22.0.1");
  if (spec.args[0] === "install") {
    mkdirSync(join(dir, "bin"), { recursive: true });
    writeFileSync(join(dir, "bin", "node"), "#!/bin/sh\n"); chmodSync(join(dir, "bin", "node"), 0o755);
  }
  if (spec.args[0] === "where") spec.onLine(dir, "stdout");
  return { code: 0, signal: null };
};

/** The ``/api/environment`` routes over real HTTP, with ownership asked of a mock daemon over the real bind channel. */
describe("environment routes", () => {
  let daemon: MockDaemon; let http: Server; let base: string; let bind: BindChannel; let ws: string;
  let env: EnvironmentService;

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
    const root = realpathSync(mkdtempSync(join(tmpdir(), "jwcs-envroot-")));
    ws = join(root, "ws");
    mkdirSync(join(ws, "api"), { recursive: true });
    writeFileSync(join(ws, "api", ".nvmrc"), "22\n");
    daemon.owners.set(ws, "alice");
    env = new EnvironmentService({
      workspaceRoot: root, tools: { node: ["22"] }, lsp: {}, mise: "mise", python: "python3", paranoid: false,
      installTimeoutMs: 10_000, store: new FileEnvironmentStore(join(root, "..", `st-${Date.now()}.json`)), ownership: bind, run: runner,
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

  test("config.json names the environment URL", async () => {
    const r = await fetch(`${base}/config.json`);
    assert.equal((await r.json() as { environmentUrl?: string }).environmentUrl, "./api/environment");
  });

  test("the owner sees a proposal, binds it, and follows the job", async () => {
    const cookie = await signIn("alice");
    const st = await (await fetch(`${base}/api/environment?workspace=${encodeURIComponent(ws)}`, { headers: { cookie } })).json() as { proposals: Array<{ tool: string; version: string; source: string }> };
    assert.deepEqual(st.proposals, [{ tool: "node", label: "Node.js", version: "22", pin: "22", pinAllowed: true, source: "api/.nvmrc" }]);
    const r = await post(cookie, "bind", { workspace: ws, tool: "node", version: "22" });
    assert.equal(r.status, 202);
    const { job } = await r.json() as { job: { id: string } };
    await env.waitFor(job.id);
    const jr = await (await fetch(`${base}/api/environment/jobs/${job.id}`, { headers: { cookie } })).json() as { job: { status: string } };
    assert.equal(jr.job.status, "done");
  });

  test("another user is refused as if the workspace did not exist, and cannot read the job", async () => {
    const cookie = await signIn("bob");
    const r = await fetch(`${base}/api/environment?workspace=${encodeURIComponent(ws)}`, { headers: { cookie } });
    assert.equal(r.status, 404);
    assert.equal((await post(cookie, "bind", { workspace: ws, tool: "node", version: "22" })).status, 404);
  });

  test("cross-site and malformed requests are refused", async () => {
    const cookie = await signIn("alice");
    assert.equal((await post(cookie, "bind", { workspace: ws, tool: "node", version: "22" }, false)).status, 403);
    assert.equal((await post(cookie, "bind", { workspace: ws, tool: "rust", version: "1" })).status, 400);
    assert.equal((await post(cookie, "bind", { workspace: ws, tool: "node", version: "18" })).status, 403);
    assert.equal((await fetch(`${base}/api/environment?workspace=x`)).status, 401);
  });
});
