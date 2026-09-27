/**
 * The environment bootstrap (#1344): detection, the managed files, the
 * install job and the ownership gate.
 *
 * Installs run through a fake {@link ProcessRunner} that does what mise, pip
 * and npm would do to the filesystem (create an install directory with a
 * ``bin/``), so the links, the manifest and the ``.lsp.json`` are exercised
 * for real without a network.  Whether mise then produces binaries a
 * CONFINED session can run is phase 0 of the epic and needs an enforcing
 * host; nothing here claims it.
 */
import { strict as assert } from "node:assert";
import { chmodSync, existsSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, readlinkSync, realpathSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { describe, test } from "node:test";
import { matchAllowedVersion } from "../src/environment/catalog.js";
import { detectWorkspace } from "../src/environment/detect.js";
import { findRepoGuidance, repoGuidanceFile } from "../src/environment/guidance.js";
import { readManifest } from "../src/environment/files.js";
import { EnvironmentError, EnvironmentService, type WorkspaceOwnership } from "../src/environment/service.js";
import { FileEnvironmentStore } from "../src/environment/store.js";
import type { ProcessRunner, RunSpec } from "../src/environment/installer.js";
import { managedContent, ownerMarker, writeManagedFile } from "../src/managed-files.js";
import { parseEnvironment, ConfigError } from "../src/config.js";

const plainWrite = (path: string, body: string, mode: number) => { mkdirSync(dirname(path), { recursive: true }); writeFileSync(path, body, { mode }); };

function exe(path: string, body = "#!/bin/sh\n"): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, body);
  chmodSync(path, 0o755);
}

function put(path: string, body: string): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, body);
}

/** A root with one workspace in it, both real paths. */
function workspaceRoot(): { root: string; ws: string } {
  const root = realpathSync(mkdtempSync(join(tmpdir(), "jwcs-env-")));
  const ws = join(root, "ws1");
  mkdirSync(join(ws, ".home"), { recursive: true });
  return { root, ws };
}

/** What mise / pip / npm / go would do to the filesystem, recorded. */
function fakeRunner(opts: { failOn?: string; hang?: string } = {}): { run: ProcessRunner; calls: RunSpec[] } {
  const calls: RunSpec[] = [];
  const run: ProcessRunner = async (spec) => {
    calls.push(spec);
    const line = [spec.command, ...spec.args].join(" ");
    if (opts.failOn && line.includes(opts.failOn)) { spec.onLine("boom: no such version", "stderr"); return { code: 1, signal: null }; }
    if (opts.hang && line.includes(opts.hang)) {
      await new Promise<void>((r) => spec.signal.addEventListener("abort", () => r(), { once: true }));
      return { code: null, signal: "SIGTERM" };
    }
    const data = spec.env.MISE_DATA_DIR!;
    if (spec.args[0] === "install" && spec.args[1]?.includes("@") && spec.command.endsWith("mise")) {
      const [tool, ver] = spec.args[1].split("@") as [string, string];
      const dir = join(data, "installs", tool, `${ver}.0.1`);
      for (const b of tool === "node" ? ["node", "npm", "npx"] : tool === "go" ? ["go", "gofmt"] : [tool]) exe(join(dir, "bin", b));
      put(join(dir, "bin", "README"), "not executable\n");
      spec.onLine(`${tool}@${ver} installed`, "stderr");
    } else if (spec.args[0] === "where") {
      const [tool, ver] = spec.args[1]!.split("@") as [string, string];
      spec.onLine(join(data, "installs", tool, `${ver}.0.1`), "stdout");
    } else if (spec.args[0] === "-m" && spec.args[1] === "venv") {
      exe(join(spec.args[2]!, "bin", "python"));
    } else if (spec.args.includes("pip")) {
      exe(join(dirname(spec.command), "basedpyright-langserver"));
    } else if (spec.args[0] === "install" && spec.args.includes("-g")) {
      const prefix = spec.args[spec.args.indexOf("--prefix") + 1]!;
      put(join(prefix, "lib", "node_modules", "typescript-language-server", "lib", "cli.mjs"), "// cli\n");
    } else if (spec.args[0] === "install" && spec.args[1]?.startsWith("golang.org/")) {
      exe(join(spec.env.GOBIN!, "gopls"));
    }
    return { code: 0, signal: null };
  };
  return { run, calls };
}

const owner = (answer: boolean | null): WorkspaceOwnership & { asked: string[] } => {
  const asked: string[] = [];
  return { asked, owns: async (user, ws) => { asked.push(`${user} ${ws}`); return answer; } };
};

function service(root: string, runner: ProcessRunner, ownership: WorkspaceOwnership = owner(true)) {
  return new EnvironmentService({
    workspaceRoot: root,
    tools: { node: ["22", "20"], go: ["1.23"] },
    lsp: { basedpyright: "1.31.6", "typescript-language-server": "4.4.0", gopls: "v0.20.0" },
    typescriptVersion: "5.9.3",
    mise: "/usr/bin/mise", python: "python3", paranoid: false, installTimeoutMs: 60_000,
    store: new FileEnvironmentStore(join(root, "..", `state-${Math.random()}.json`)),
    ownership, run: runner,
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

describe("detection", () => {
  test("pins come from the repository's own files, root and immediate subdirectories", () => {
    const { ws } = workspaceRoot();
    put(join(ws, "api", ".nvmrc"), "v22.11.0\n");
    put(join(ws, "api", "package.json"), JSON.stringify({ engines: { node: ">=18" } }));
    put(join(ws, "svc", "go.mod"), "module x\n\ngo 1.23.4\n");
    put(join(ws, "py", "pyproject.toml"), "[project]\n");
    put(join(ws, "deep", "nested", "go.mod"), "go 1.1\n");
    put(join(ws, "node_modules", "package.json"), "{}");
    const found = detectWorkspace(ws);
    const byTool = Object.fromEntries(found.map((d) => [d.tool, d]));
    assert.deepEqual(byTool.node, { tool: "node", pin: "v22.11.0", source: "api/.nvmrc" });
    assert.deepEqual(byTool.go, { tool: "go", pin: "1.23.4", source: "svc/go.mod" });
    assert.equal(byTool.python?.source, "py/pyproject.toml");
    assert.equal(found.length, 3);
  });

  test("a pin maps to the most specific allowed version at a component boundary", () => {
    assert.equal(matchAllowedVersion("v22.11.0", ["20", "22", "22.11"]), "22.11");
    assert.equal(matchAllowedVersion(">=22", ["22"]), "22");
    assert.equal(matchAllowedVersion("220", ["22"]), null);
    assert.equal(matchAllowedVersion("18", ["22"]), null);
  });
});

describe("repository guidance pointer", () => {
  test("names subdirectory guidance files and never their contents; the root is the framework's", () => {
    const { ws } = workspaceRoot();
    put(join(ws, "AGENTS.md"), "root guidance\n");
    put(join(ws, "api", "AGENTS.md"), "SENTINEL-DO-NOT-INLINE\n");
    put(join(ws, "api", ".github", "copilot-instructions.md"), "x\n");
    put(join(ws, "web", "CONTRIBUTING.md"), "x\n");
    const paths = findRepoGuidance(ws);
    assert.deepEqual(paths, ["api/AGENTS.md", "api/.github/copilot-instructions.md", "web/CONTRIBUTING.md"]);
    const body = managedContent(repoGuidanceFile(paths)!);
    assert.ok(body.startsWith("<!-- jaato-managed: repo-guidance v1"));
    assert.ok(body.includes("- `api/AGENTS.md`"));
    assert.ok(!body.includes("SENTINEL-DO-NOT-INLINE"));
    assert.ok(!body.includes("- `AGENTS.md`"), "the root file is left to the framework (#1347)");
    assert.equal(repoGuidanceFile([]), null);
  });
});

describe("EnvironmentService", () => {
  test("status: proposals exclude what is bound or declined, and follow the allow-list", async () => {
    const { root, ws } = workspaceRoot();
    put(join(ws, "a", ".nvmrc"), "22\n");
    put(join(ws, "a", "Cargo.toml"), "[package]\n");
    put(join(ws, "b", ".go-version"), "1.19\n");
    const svc = service(root, fakeRunner().run);
    const st = await svc.status("sub", "alice", ws);
    const node = st.proposals.find((p) => p.tool === "node")!;
    assert.equal(node.version, "22");
    assert.equal(node.pinAllowed, true);
    const go = st.proposals.find((p) => p.tool === "go")!;
    assert.equal(go.pinAllowed, false, "1.19 is not on the operator's list");
    assert.equal(go.version, "1.23");
    await svc.decline("sub", "alice", ws, "go");
    const after = await svc.status("sub", "alice", ws);
    assert.deepEqual(after.proposals.map((p) => p.tool), ["node"]);
    assert.deepEqual(after.declined, ["go"]);
    // Another user of the same workspace was not asked.
    assert.deepEqual((await svc.status("sub2", "bob", ws)).proposals.map((p) => p.tool).sort(), ["go", "node"]);
  });

  test("bind node: mise install, links, the pinned server, and the four managed files", async () => {
    const { root, ws } = workspaceRoot();
    const { run, calls } = fakeRunner();
    const svc = service(root, run);
    const job = await svc.bind("sub", "alice", ws, "node", "22");
    assert.equal(job.status, "running");
    const done = await svc.waitFor(job.id);
    assert.equal(done?.status, "done", done?.error);

    // The links are relative, point into the workspace's mise data, and skip a non-executable.
    const link = join(ws, ".home/.local/bin/node");
    assert.ok(lstatSync(link).isSymbolicLink());
    assert.ok(!readlinkSync(link).startsWith("/"));
    assert.ok(realpathSync(link).startsWith(join(ws, ".home/.local/share/mise/installs/node/")));
    assert.ok(!existsSync(join(ws, ".home/.local/bin/README")));

    // mise ran with its directories under the workspace HOME, a clean env, and no project config above .home.
    const install = calls.find((c) => c.args[0] === "install" && c.args[1] === "node@22")!;
    assert.equal(install.env.HOME, join(ws, ".home"));
    assert.equal(install.env.MISE_DATA_DIR, join(ws, ".home/.local/share/mise"));
    assert.equal(install.env.MISE_CEILING_PATHS, join(ws, ".home"));
    assert.equal(install.cwd, join(ws, ".home"));
    assert.equal(install.env.GITHUB_TOKEN, undefined);

    // The language server was installed with the linked npm into jaato-lsp.
    const npm = calls.find((c) => c.command.endsWith(".home/.local/bin/npm"))!;
    assert.ok(npm.args.includes("typescript-language-server@4.4.0"));
    assert.ok(npm.args.includes("typescript@5.9.3"));

    const lsp = JSON.parse(readFileSync(join(ws, ".lsp.json"), "utf8"));
    assert.equal(lsp._jaato_managed, "lsp v1");
    assert.equal(lsp.languageServers.typescript.command, join(ws, ".home/.local/bin/node"));
    assert.ok(lsp.languageServers.typescript.args[0].endsWith("typescript-language-server/lib/cli.mjs"));

    const toml = readFileSync(join(ws, ".home/.config/mise/config.toml"), "utf8");
    assert.ok(toml.startsWith("# jaato-managed: toolchains v1"));
    assert.ok(toml.includes('node = "22"'));

    const md = readFileSync(join(ws, ".jaato/instructions/45-environment.md"), "utf8");
    assert.ok(md.includes("Node.js 22"));
    assert.ok(md.includes('get_environment(aspect="runtime")'));

    const manifest = readManifest(ws);
    assert.deepEqual(manifest.toolchains.map((t) => [t.tool, t.version, t.bin]), [["node", "22", ["node", "npm", "npx"]]]);
    assert.equal(JSON.parse(readFileSync(join(ws, ".jaato/environment.json"), "utf8"))._jaato_managed, "environment v1");

    const st = await svc.status("sub", "alice", ws);
    assert.deepEqual(st.toolchains.map((t) => t.tool), ["node"]);
    assert.equal(st.job?.status, "done");
  });

  test("bind python installs basedpyright into its own venv and writes a python server", async () => {
    const { root, ws } = workspaceRoot();
    const svc = service(root, fakeRunner().run);
    const done = await svc.waitFor((await svc.bind("sub", "alice", ws, "python", "system")).id);
    assert.equal(done?.status, "done", done?.error);
    const lsp = JSON.parse(readFileSync(join(ws, ".lsp.json"), "utf8"));
    assert.equal(lsp.languageServers.python.command, join(ws, ".home/.local/share/jaato-lsp/basedpyright/bin/basedpyright-langserver"));
  });

  test("a failed install writes no manifest entry and says why", async () => {
    const { root, ws } = workspaceRoot();
    const svc = service(root, fakeRunner({ failOn: "install node@22" }).run);
    const done = await svc.waitFor((await svc.bind("sub", "alice", ws, "node", "22")).id);
    assert.equal(done?.status, "failed");
    assert.match(done!.error!, /installing node@22 failed \(exit 1\): boom/);
    assert.deepEqual(readManifest(ws).toolchains, []);
    assert.ok(!existsSync(join(ws, ".lsp.json")));
  });

  test("cancel stops the job and leaves no binding", async () => {
    const { root, ws } = workspaceRoot();
    const svc = service(root, fakeRunner({ hang: "install node@22" }).run);
    const job = await svc.bind("sub", "alice", ws, "node", "22");
    await assert.rejects(svc.bind("sub", "alice", ws, "go", "1.23"), (e) => e instanceof EnvironmentError && e.status === 409);
    assert.equal(svc.cancel("other-sub", job.id), null, "another user cannot cancel it");
    svc.cancel("sub", job.id);
    const done = await svc.waitFor(job.id);
    assert.equal(done?.status, "cancelled");
    assert.deepEqual(readManifest(ws).toolchains, []);
  });

  test("unbind removes our links and the managed files, and keeps a file that is not ours", async () => {
    const { root, ws } = workspaceRoot();
    const svc = service(root, fakeRunner().run);
    await svc.waitFor((await svc.bind("sub", "alice", ws, "go", "1.23")).id);
    assert.ok(existsSync(join(ws, ".home/.local/bin/gopls")));
    // A user's own binary with a name the next bind would want.
    exe(join(ws, ".home/.local/bin/mine"));
    const r = await svc.unbind("sub", "alice", ws, "go");
    assert.deepEqual(r.removed.sort(), ["go", "gofmt", "gopls"]);
    assert.ok(existsSync(join(ws, ".home/.local/bin/mine")));
    assert.ok(!existsSync(join(ws, ".lsp.json")));
    assert.ok(!existsSync(join(ws, ".jaato/instructions/45-environment.md")));
    assert.ok(!existsSync(join(ws, ".jaato/environment.json")));
    // The installed data is kept for a rebind.
    assert.ok(existsSync(join(ws, ".home/.local/share/mise/installs/go")));
  });

  test("a link the user made is never replaced", async () => {
    const { root, ws } = workspaceRoot();
    mkdirSync(join(ws, ".home/.local/bin"), { recursive: true });
    exe(join(ws, "own-node"));
    symlinkSync(join(ws, "own-node"), join(ws, ".home/.local/bin/node"));
    const svc = service(root, fakeRunner().run);
    const done = await svc.waitFor((await svc.bind("sub", "alice", ws, "node", "22")).id);
    assert.equal(realpathSync(join(ws, ".home/.local/bin/node")), join(ws, "own-node"));
    assert.ok(done!.log.some((l) => l.includes("kept .home/.local/bin/node")));
  });

  test("off-list versions, other users' workspaces and an unreachable daemon are refused before anything runs", async () => {
    const { root, ws } = workspaceRoot();
    const { run, calls } = fakeRunner();
    await assert.rejects(service(root, run).bind("sub", "alice", ws, "node", "18"), (e) => e instanceof EnvironmentError && e.status === 403);
    await assert.rejects(service(root, run, owner(false)).bind("sub", "alice", ws, "node", "22"), (e) => e instanceof EnvironmentError && e.status === 404);
    await assert.rejects(service(root, run, owner(null)).bind("sub", "alice", ws, "node", "22"), (e) => e instanceof EnvironmentError && e.status === 503);
    await assert.rejects(service(root, run).status("sub", "alice", tmpdir()), (e) => e instanceof EnvironmentError && e.status === 404);
    // A symlink under the root that leaves it is judged by its target.
    symlinkSync(tmpdir(), join(root, "escape"));
    await assert.rejects(service(root, run).status("sub", "alice", join(root, "escape")), (e) => e instanceof EnvironmentError && e.status === 404);
    assert.equal(calls.length, 0);
  });

  test("refresh writes the pointer, then removes it when nothing is left", async () => {
    const { root, ws } = workspaceRoot();
    put(join(ws, "api", "AGENTS.md"), "x\n");
    const svc = service(root, fakeRunner().run);
    assert.equal((await svc.refreshGuidance("sub", "alice", ws)).outcome, "written");
    assert.ok(readFileSync(join(ws, ".jaato/instructions/30-repo-guidance.md"), "utf8").includes("`api/AGENTS.md`"));
    assert.equal((await svc.refreshGuidance("sub", "alice", ws)).outcome, "unchanged");
    writeFileSync(join(ws, "api", "AGENTS.md"), "");
    const { rmSync } = await import("node:fs");
    rmSync(join(ws, "api", "AGENTS.md"));
    assert.equal((await svc.refreshGuidance("sub", "alice", ws)).outcome, "removed");
  });
});

describe("config: the environment block", () => {
  test("parses, and refuses what it cannot install", () => {
    const e = parseEnvironment({ workspace_root: "/srv/ws", tools: { node: [22, "20.1"] }, lsp: { basedpyright: "1.31.6" } }, "/etc/x")!;
    assert.deepEqual(e.tools, { node: ["22", "20.1"] });
    assert.equal(e.mise, "mise");
    assert.equal(e.installTimeoutSeconds, 900);
    assert.equal(parseEnvironment(undefined, "/"), undefined);
    assert.throws(() => parseEnvironment({ tools: {} }, "/"), ConfigError);
    assert.throws(() => parseEnvironment({ workspace_root: "/w", tools: { rust: ["1"] } }, "/"), /not a toolchain/);
    assert.throws(() => parseEnvironment({ workspace_root: "/w", tools: { node: ["22; rm -rf /"] } }, "/"), /not a version/);
    assert.throws(() => parseEnvironment({ workspace_root: "/w", lsp: { "typescript-language-server": "4" } }, "/"), /typescript_version is required/);
  });
});
