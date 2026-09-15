/**
 * Tests for the launcher: ``node --test bin/`` (no dist/ needed — a
 * throwaway root stands in for the bundle).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { request } from "node:http";
import { createStaticServer, isLoopbackHost, parseArgs, resolveToken } from "./jaato-web.js";

/** ``fetch`` refuses a caller-supplied ``Host``; plain ``http`` does not. */
function getWithHost(port, path, host) {
  return new Promise((ok, fail) => {
    request({ host: "127.0.0.1", port, path, headers: { host } }, (res) => {
      let body = "";
      res.on("data", (c) => (body += c));
      res.on("end", () => ok({ status: res.statusCode, body }));
    }).on("error", fail).end();
  });
}

function fakeDist() {
  const root = mkdtempSync(join(tmpdir(), "jaato-web-"));
  mkdirSync(join(root, "assets"));
  writeFileSync(join(root, "index.html"), "<!doctype html><div id=root></div>");
  writeFileSync(join(root, "assets", "index-abc.js"), "console.log(1)");
  writeFileSync(join(root, "secret.txt"), "not served outside root");
  return root;
}

async function listen(server) {
  await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
  const { port } = server.address();
  return { port, base: `http://127.0.0.1:${port}` };
}

test("parseArgs: defaults, --key=value and --key value forms, validation", () => {
  const d = parseArgs([]);
  assert.equal(d.daemon, "ws://127.0.0.1:8080");
  assert.equal(d.host, "127.0.0.1");
  assert.equal(d.port, 5180);
  assert.equal(d.open, true);

  const o = parseArgs(["--daemon=ws://box:1", "--port", "0", "--no-open", "--token", "abc", "--host", "0.0.0.0"]);
  assert.equal(o.daemon, "ws://box:1");
  assert.equal(o.port, 0);
  assert.equal(o.open, false);
  assert.equal(o.token, "abc");
  assert.equal(o.host, "0.0.0.0");

  assert.throws(() => parseArgs(["--port", "70000"]), /--port/);
  assert.throws(() => parseArgs(["--bogus"]), /unknown option --bogus/);
  assert.throws(() => parseArgs(["--daemon"]), /needs a value/);
});

test("isLoopbackHost", () => {
  for (const h of ["127.0.0.1", "127.9.9.9", "localhost", "LOCALHOST", "::1", "[::1]"]) assert.ok(isLoopbackHost(h), h);
  for (const h of ["0.0.0.0", "10.0.0.5", "build-box", "::"]) assert.ok(!isLoopbackHost(h), h);
});

test("resolveToken: flag > file > default file for a loopback daemon > none", () => {
  const fsMock = { readFile: (p) => `  tok-from-${p}\n`, exists: () => true };
  assert.deepEqual(resolveToken({ noToken: true, daemon: "ws://127.0.0.1:8080" }, fsMock), { token: undefined, source: "disabled" });
  assert.equal(resolveToken({ token: "x", daemon: "ws://127.0.0.1:8080" }, fsMock).token, "x");
  assert.equal(resolveToken({ tokenFile: "/p/t", daemon: "ws://remote:8080" }, fsMock).token, "tok-from-/p/t");

  const local = resolveToken({ daemon: "ws://localhost:8080" }, fsMock);
  assert.match(local.source, /ws\.token$/);
  assert.match(local.token, /^tok-from-/);

  // The default file is this machine's daemon's secret: never sent for a remote daemon.
  assert.deepEqual(resolveToken({ daemon: "ws://build-box:8080" }, fsMock), { token: undefined, source: "none" });
  // ...and not when it does not exist.
  assert.deepEqual(resolveToken({ daemon: "ws://127.0.0.1:8080" }, { ...fsMock, exists: () => false }), { token: undefined, source: "none" });
});

test("static server: index, SPA fallback, hashed assets, config.json, traversal", async () => {
  const root = fakeDist();
  const server = createStaticServer({ root, config: { daemon: "ws://d:1", token: "T", autoConnect: true }, allowedHosts: null });
  const { base } = await listen(server);
  try {
    let r = await fetch(base + "/");
    assert.equal(r.status, 200);
    assert.match(r.headers.get("content-type"), /text\/html/);
    assert.equal(r.headers.get("cache-control"), "no-cache");
    assert.equal(r.headers.get("x-content-type-options"), "nosniff");

    r = await fetch(base + "/workspaces/anything");
    assert.equal(r.status, 200);
    assert.match(await r.text(), /id=root/);

    r = await fetch(base + "/assets/index-abc.js");
    assert.match(r.headers.get("content-type"), /javascript/);
    assert.match(r.headers.get("cache-control"), /immutable/);

    r = await fetch(base + "/config.json");
    assert.equal(r.headers.get("cache-control"), "no-store");
    assert.deepEqual(await r.json(), { daemon: "ws://d:1", token: "T", autoConnect: true });

    r = await fetch(base + "/..%2f..%2fetc%2fpasswd");
    assert.ok(r.status === 200 || r.status === 403);
    assert.doesNotMatch(await r.text(), /root:x:/);

    r = await fetch(base + "/", { method: "POST" });
    assert.equal(r.status, 405);

    r = await fetch(base + "/", { method: "HEAD" });
    assert.equal(r.status, 200);
  } finally {
    server.close();
  }
});

test("static server: a Host header we did not bind is refused (DNS rebinding)", async () => {
  const root = fakeDist();
  const allowed = new Set();
  const server = createStaticServer({ root, config: { daemon: "ws://d:1", token: "T" }, allowedHosts: allowed });
  const { base, port } = await listen(server);
  allowed.add(`127.0.0.1:${port}`);
  try {
    let r = await fetch(base + "/config.json");
    assert.equal(r.status, 200);
    const bad = await getWithHost(port, "/config.json", `attacker.example:${port}`);
    assert.equal(bad.status, 421);
    assert.doesNotMatch(bad.body, /"T"/);
  } finally {
    server.close();
  }
});
