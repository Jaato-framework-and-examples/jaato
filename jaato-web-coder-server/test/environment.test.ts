/**
 * The environment bootstrap's policy side (#1344): the allow-list, the
 * offer the page stages, the declines and the ownership gate.  Nothing here
 * installs or reads a workspace: the web coder's toolchains plugin does that
 * in the session's runner, and its own tests (``plugin/tests``) cover it.
 */
import { strict as assert } from "node:assert";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { describe, test } from "node:test";
import { TOOL_IDS } from "../src/environment/catalog.js";
import { EnvironmentService, type WorkspaceOwnership } from "../src/environment/service.js";
import { FileEnvironmentStore } from "../src/environment/store.js";
import { managedContent, ownerMarker, writeManagedFile } from "../src/managed-files.js";
import { parseEnvironment, ConfigError } from "../src/config.js";

const plainWrite = (path: string, body: string, mode: number) => { mkdirSync(dirname(path), { recursive: true }); writeFileSync(path, body, { mode }); };

function put(path: string, body: string): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, body);
}

const owner = (answer: boolean | null): WorkspaceOwnership => ({ owns: async () => answer });

function service(opts: Partial<ConstructorParameters<typeof EnvironmentService>[0]> = {}): EnvironmentService {
  return new EnvironmentService({
    tools: { node: ["22", "20"], go: ["1.23"], java: ["21", "temurin-17"], maven: ["3.9.9"], gradle: ["8.10"] },
    servers: { basedpyright: { version: "1.31.6" }, gopls: { version: "v0.20.0" }, jdtls: { version: "1.40.0", java: "21", max_heap: "768m" } },
    paranoid: false, installTimeoutSeconds: 900,
    store: new FileEnvironmentStore(join(mkdtempSync(join(tmpdir(), "jwcs-env-")), "state.json")),
    ownership: owner(true), ...opts,
  });
}

describe("managed files — per-format markers", () => {
  test("hash marker round-trips and a user file without it is kept", () => {
    const dir = mkdtempSync(join(tmpdir(), "jwcs-mf-"));
    const file = { relativePath: "a/config.toml", markerId: "toolchains", version: 1, body: "[tools]\n", format: "hash" as const, generated: true };
    const c = managedContent(file);
    assert.ok(c.startsWith("# jaato-managed: toolchains v1 "));
    assert.deepEqual(ownerMarker(c, "hash"), { markerId: "toolchains", version: 1 });
    put(join(dir, "a/config.toml"), "[tools]\nnode = \"18\"\n");
    assert.equal(writeManagedFile(dir, file, plainWrite).action, "skipped-user-file");
  });

  test("json marker is a key; a generated file refreshes on a content change at the same version", () => {
    const dir = mkdtempSync(join(tmpdir(), "jwcs-mf-"));
    const f = (servers: object) => ({ relativePath: ".lsp.json", markerId: "lsp", version: 1, body: JSON.stringify({ languageServers: servers }), format: "json" as const, generated: true });
    assert.equal(writeManagedFile(dir, f({ a: 1 }), plainWrite).action, "written");
    const parsed = JSON.parse(readFileSync(join(dir, ".lsp.json"), "utf8"));
    assert.equal(parsed._jaato_managed, "lsp v1");
    assert.deepEqual(parsed.languageServers, { a: 1 });
    assert.equal(writeManagedFile(dir, f({ a: 1 }), plainWrite).action, "unchanged");
    assert.deepEqual(writeManagedFile(dir, f({ b: 2 }), plainWrite), { path: ".lsp.json", action: "written", reason: "content-changed" });
    // The user's own .lsp.json (no marker key) is theirs.
    writeFileSync(join(dir, ".lsp.json"), JSON.stringify({ languageServers: { mine: {} } }));
    assert.equal(writeManagedFile(dir, f({ c: 3 }), plainWrite).action, "skipped-user-file");
    // A file that does not parse is not ours either.
    writeFileSync(join(dir, ".lsp.json"), "{ not json");
    assert.equal(writeManagedFile(dir, f({ c: 3 }), plainWrite).action, "skipped-user-file");
  });

  test("a SHIPPED (non-generated) file keeps the same-version no-op rule", () => {
    const dir = mkdtempSync(join(tmpdir(), "jwcs-mf-"));
    const f = (body: string) => ({ relativePath: "x.md", markerId: "x", version: 1, body });
    writeManagedFile(dir, f("one"), plainWrite);
    assert.equal(writeManagedFile(dir, f("two"), plainWrite).action, "unchanged");
  });
});

describe("EnvironmentService", () => {
  test("status: the allow-list, the declines and the offer, with no workspace access", async () => {
    const svc = service();
    const ws = "/srv/nowhere/that/exists/ws1";
    await svc.decline("sub", "alice", ws, "go");
    const st = await svc.status("sub", "alice", ws);
    assert.deepEqual(st.allowed.map((a) => a.tool), ["python", "node", "go", "java", "maven", "gradle"], "catalog order; python because basedpyright is pinned");
    assert.deepEqual(st.allowed.find((a) => a.tool === "java")!.server, { id: "jdtls", version: "1.40.0" });
    assert.deepEqual(st.declined, ["go"]);
    const offer = JSON.parse(st.offer.content);
    assert.equal(st.offer.path, ".jaato/toolchain-offer.json");
    assert.equal(offer._jaato_managed, "toolchain-offer v2");
    assert.equal(offer.schema, 2);
    assert.deepEqual(offer.toolchains.find((t: { tool: string }) => t.tool === "java"), { tool: "java", label: "Java", versions: ["21", "temurin-17"] });
    assert.deepEqual(offer.servers.jdtls, { version: "1.40.0", java: "21", max_heap: "768m" });
    assert.deepEqual(offer.install, { timeout_seconds: 900, paranoid: false });
    await svc.undecline("sub", "alice", ws, "go");
    assert.deepEqual((await svc.status("sub", "alice", ws)).declined, []);
  });

  test("a pinned server whose toolchain is not allowed is not offered", async () => {
    const svc = service({ tools: { node: ["22"] }, servers: { gopls: { version: "v0.20.0" }, "typescript-language-server": { version: "4.4.0", typescript: "5.9.3" } } });
    const offer = JSON.parse((await svc.status("sub", "alice", "/w")).offer.content);
    assert.deepEqual(Object.keys(offer.servers), ["typescript-language-server"]);
  });

  test("a server that allows nothing offers nothing", async () => {
    const offer = JSON.parse((await service({ tools: {}, servers: {} }).status("sub", "alice", "/w")).offer.content);
    assert.deepEqual(offer.toolchains, []);
  });

  test("the offer is exactly what the plugin's fixture says it reads", async () => {
    // plugin/tests/fixtures/toolchain-offer.json is also read by the Python plugin's tests: one contract, two sides.
    const svc = service({ tools: { node: ["22", "20"], java: ["21", "temurin-17"], maven: ["3.9.9"] }, servers: { jdtls: { version: "1.40.0", java: "21", max_heap: "1G" } } });
    const fixture = readFileSync(new URL("../plugin/tests/fixtures/toolchain-offer.json", import.meta.url), "utf8");
    assert.deepEqual(JSON.parse((await svc.status("sub", "alice", "/w")).offer.content), JSON.parse(fixture));
  });

  test("the toolchain ids agree with the plugin's catalog", () => {
    const py = readFileSync(new URL("../plugin/jaato_web_coder_toolchains/catalog.py", import.meta.url), "utf8");
    const ids = [...py.matchAll(/^    Toolchain\("([a-z]+)"/gm)].map((m) => m[1]);
    assert.deepEqual(ids, TOOL_IDS);
  });

  test("another user's workspace, a relative path and an unreachable daemon are refused", async () => {
    await assert.rejects(service({ ownership: owner(false) }).status("sub", "bob", "/w"), (e: Error & { status?: number }) => e.status === 404);
    await assert.rejects(service({ ownership: owner(null) }).status("sub", "alice", "/w"), (e: Error & { status?: number }) => e.status === 503);
    await assert.rejects(service().status("sub", "alice", "w"), /absolute path/);
  });
});

describe("config: the environment block", () => {
  test("parses, and refuses what it cannot install", () => {
    const e = parseEnvironment({ tools: { node: [22, "20.1"] }, lsp: { basedpyright: "1.31.6" } }, "/etc/x")!;
    assert.deepEqual(e.tools, { node: ["22", "20.1"] });
    assert.deepEqual(e.servers, { basedpyright: { version: "1.31.6" } });
    assert.equal(e.installTimeoutSeconds, 900);
    assert.equal(parseEnvironment(undefined, "/"), undefined);
    assert.throws(() => parseEnvironment({ tools: { rust: ["1"] } }, "/"), /not a toolchain/);
    assert.throws(() => parseEnvironment({ tools: { node: ["22; rm -rf /"] } }, "/"), /not a version/);
    assert.throws(() => parseEnvironment({ lsp: { "typescript-language-server": "4" } }, "/"), /typescript-language-server\.typescript is required/);
    assert.throws(() => parseEnvironment({ install_timeout: "5s" }, "/"), /between 30s and 2h/);
  });

  test("keys for installing here are refused now that installs run in the runner", () => {
    assert.throws(() => parseEnvironment({ workspace_root: "/w" }, "/"), /workspace_root is no longer used/);
    assert.throws(() => parseEnvironment({ mise: "/usr/bin/mise" }, "/"), /mise is no longer used: mise runs in the session's runner/);
    assert.throws(() => parseEnvironment({ python: "python3" }, "/"), /python is no longer used/);
  });

  test("a server's settings live under its lsp entry; a toolchain's under its tools entry", () => {
    const j = parseEnvironment({
      tools: { java: { versions: ["temurin-21", 17] }, maven: ["3.9.9"], gradle: "8.10" },
      lsp: { gopls: "v0.20.0", "typescript-language-server": { version: "4.4.0", typescript: "5.9.3" }, jdtls: { version: "1.40.0", java: 25, max_heap: "2G", mirror: "https://mirror.example/jdtls" } },
    }, "/")!;
    assert.deepEqual(j.tools, { java: ["temurin-21", "17"], maven: ["3.9.9"], gradle: ["8.10"] });
    assert.deepEqual(j.servers, {
      gopls: { version: "v0.20.0" },
      "typescript-language-server": { version: "4.4.0", typescript: "5.9.3" },
      jdtls: { version: "1.40.0", java: "25", max_heap: "2G", mirror: "https://mirror.example/jdtls" },
    });
    assert.deepEqual(parseEnvironment({ lsp: { jdtls: { version: "1.40.0" } } }, "/")!.servers.jdtls, { version: "1.40.0", java: "21", max_heap: "1G" });
    assert.deepEqual(parseEnvironment({ tools: { node: { versions: ["22"] } } }, "/")!.tools, parseEnvironment({ tools: { node: ["22"] } }, "/")!.tools);

    const bad = (x: Record<string, unknown>) => () => parseEnvironment(x, "/");
    assert.throws(bad({ lsp: { jdtls: { version: "1.40.0", max_heap: "lots" } } }), /lsp\.jdtls\.max_heap: 'lots' is not a JVM heap size/);
    assert.throws(bad({ lsp: { jdtls: { version: "1.40.0", mirror: "http://mirror.example" } } }), /https/);
    assert.throws(bad({ lsp: { jdtls: { java: "21" } } }), /lsp\.jdtls\.version is required/);
    assert.throws(bad({ lsp: { gopls: { version: "v0.20.0", typescript: "5" } } }), /lsp\.gopls\.typescript: unknown key \(one of version\)/);
    assert.throws(bad({ tools: { node: { versions: ["22"], mirror: "x" } } }), /tools\.node\.mirror: unknown key/);
    assert.throws(bad({ typescript_version: "5.9.3" }), /typescript_version has moved to environment\.lsp\.typescript-language-server\.typescript/);
    assert.throws(bad({ jdtls_max_heap: "2G" }), /jdtls_max_heap has moved to environment\.lsp\.jdtls\.max_heap/);
  });
});
