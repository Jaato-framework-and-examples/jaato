/**
 * Against the REAL daemon (protocol 1.10, #1074): spawn `python -m server
 * --web-socket … --ws-app-credentials …`, open the bind channel with the
 * app credential, mint a ticket, connect as a browser would with
 * `?token=<ticket>`, and prove the ticket is single-use.
 *
 * Runs when a Python with jaato-server is available: `JAATO_DAEMON_PYTHON`
 * or the repo's `.venv/bin/python`; skips otherwise (the Node CI job has no
 * Python), so `npm test` stays green everywhere and `npm run test:daemon`
 * is the explicit local check.
 */
import { strict as assert } from "node:assert";
import { spawn, type ChildProcess } from "node:child_process";
import { existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { createServer, Socket } from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { after, before, describe, test } from "node:test";
import { JaatoClient } from "@jaato/sdk";
import { BindChannel, BindRefusedError } from "../src/bind-channel.js";
import { APP_CREDENTIAL } from "./helpers.js";

const PYTHON = process.env.JAATO_DAEMON_PYTHON ?? resolve(import.meta.dirname, "../../.venv/bin/python");
const available = existsSync(PYTHON);

async function freePort(): Promise<number> {
  return new Promise((ok) => { const s = createServer(); s.listen(0, "127.0.0.1", () => { const p = (s.address() as { port: number }).port; s.close(() => ok(p)); }); });
}

async function waitForPort(port: number, proc: ChildProcess, ms = 60_000): Promise<void> {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    if (proc.exitCode !== null) throw new Error(`daemon exited with ${proc.exitCode}`);
    const ok = await new Promise<boolean>((r) => { const s = new Socket(); s.once("connect", () => { s.destroy(); r(true); }); s.once("error", () => r(false)); s.connect(port, "127.0.0.1"); });
    if (ok) return;
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error("daemon did not open its port");
}

describe("against the real daemon", { skip: available ? false : `no Python with jaato-server at ${PYTHON}` }, () => {
  let proc: ChildProcess; let port: number; let stderr = "";

  before(async () => {
    port = await freePort();
    const dir = mkdtempSync(join(tmpdir(), "jwcs-daemon-"));
    writeFileSync(join(dir, "ws-apps.json"), JSON.stringify({ "jaato-web-coder": APP_CREDENTIAL }), { mode: 0o600 });
    proc = spawn(PYTHON, ["-m", "server", "--web-socket", `127.0.0.1:${port}`, "--ws-app-credentials", join(dir, "ws-apps.json"), "--pid-file", join(dir, "daemon.pid")],
      { cwd: dir, env: { ...process.env, HOME: dir, JAATO_RUNNER_POOL_ENABLED: "false" }, stdio: ["ignore", "pipe", "pipe"] });
    proc.stderr!.on("data", (c) => { stderr += c; });
    proc.stdout!.on("data", (c) => { stderr += c; });
    try { await waitForPort(port, proc); } catch (e) { throw new Error(`${(e as Error).message}\n--- daemon output ---\n${stderr.slice(-4000)}`); }
  });
  after(async () => { proc?.kill("SIGTERM"); await new Promise((r) => setTimeout(r, 500)); if (proc?.exitCode === null) proc.kill("SIGKILL"); });

  test("bind channel → ticket → attributed browser connection; the ticket is single-use", async () => {
    const ch = new BindChannel({ bindUrl: `ws://127.0.0.1:${port}`, appCredential: APP_CREDENTIAL });
    await ch.connect();
    const t = await ch.bind("alice", 60);
    assert.equal(t.qualified, "jaato-web-coder:alice");
    assert.equal(t.appId, "jaato-web-coder");
    assert.ok(t.ticket.length >= 32);

    // The browser's connection: ?token=<ticket>, an ordinary client with presentation config.
    const browser = new JaatoClient({ url: `ws://127.0.0.1:${port}`, token: t.ticket, recovery: { autoReconnect: false }, clientConfig: { presentation: { client_type: "web" } } });
    await browser.connect();
    assert.ok(browser.serverVersion, "handshake completed");
    await browser.close();

    // Replaying the consumed ticket must be refused at the Upgrade.
    const replay = new JaatoClient({ url: `ws://127.0.0.1:${port}`, token: t.ticket, recovery: { autoReconnect: false } });
    await assert.rejects(replay.connect(), /1008|auth|closed/i);

    // The app credential itself cannot open a session: any non-ticket frame is refused.
    // (The bind channel sends none; a bind with a wrong credential is `denied`.)
    const stranger = new BindChannel({ bindUrl: `ws://127.0.0.1:${port}`, appCredential: "wrong-credential-0000000000000000000000" });
    await assert.rejects(stranger.connect(), /1008|auth|closed/i);

    // Logout after a completed login revokes nothing — and says so honestly.
    assert.deepEqual(await ch.revokeUser("alice"), { status: "not_found", revoked: 0 });
    // A ticket minted and NOT presented is revocable.
    await ch.bind("bob", 60);
    assert.deepEqual(await ch.revokeUser("bob"), { status: "revoked", revoked: 1 });
    // ttl outside the daemon's range is `invalid`.
    await assert.rejects(ch.bind("alice", 0), (e: unknown) => e instanceof BindRefusedError && e.status === "invalid");
    await ch.close();
  });
});
