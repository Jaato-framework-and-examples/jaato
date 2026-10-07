import { strict as assert } from "node:assert";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { describe, test } from "node:test";
import { FileGitHubStore } from "../src/github-store.js";
import { GitHubService } from "../src/github.js";
import {
  GITHUB_GUIDANCE_MARKER_ID,
  GITHUB_GUIDANCE_PATH,
  GITHUB_GUIDANCE_VERSION,
  githubGuidanceFile,
} from "../src/github-guidance.js";
import { managedContent, managedMarker } from "../src/managed-files.js";
import { FakeGitHubApi, FakeReloader } from "./github-fakes.js";

const KEY = "g".repeat(40);
const storePath = () => join(mkdtempSync(join(tmpdir(), "jwcs-ghsvc-")), "github.json");

function service(opts: { workspaceRoot?: string } = {}) {
  const store = new FileGitHubStore(storePath(), KEY);
  const api = new FakeGitHubApi();
  const reloader = new FakeReloader();
  const svc = new GitHubService({ store, api, reloader, workspaceRoot: opts.workspaceRoot, marginSeconds: 300 });
  return { store, api, reloader, svc };
}

function workspaceUnder(root: string): string {
  const ws = join(root, "proj");
  mkdirSync(ws, { recursive: true });
  return ws;
}

const guidancePath = (ws: string) => join(ws, GITHUB_GUIDANCE_PATH);

describe("github bind — the working-guidance file (#1246)", () => {
  test("bind writes 40-github.md with our marker; guidanceWritten is true", async () => {
    const root = mkdtempSync(join(tmpdir(), "jwcs-ws-"));
    const ws = workspaceUnder(root);
    const { svc } = service({ workspaceRoot: root });
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    const result = await svc.bind("sub-alice", "alice", ws, account.id);
    assert.equal(result.guidanceWritten, true);
    assert.equal(result.note, undefined, "a clean write is silent");
    const body = readFileSync(guidancePath(ws), "utf8");
    assert.equal(body, managedContent(githubGuidanceFile()), "one source of the bytes");
    assert.ok(body.startsWith(managedMarker(GITHUB_GUIDANCE_MARKER_ID, GITHUB_GUIDANCE_VERSION)));
    assert.match(body, /Work in your own git worktree/);
    assert.match(body, /Never force-push/);
  });

  test("a marker-deleted file is left alone, with a note; guidanceWritten is false", async () => {
    const root = mkdtempSync(join(tmpdir(), "jwcs-ws-"));
    const ws = workspaceUnder(root);
    const mine = "# my own GitHub notes, marker deleted\n";
    mkdirSync(dirname(guidancePath(ws)), { recursive: true });
    writeFileSync(guidancePath(ws), mine);
    const { svc } = service({ workspaceRoot: root });
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    const result = await svc.bind("sub-alice", "alice", ws, account.id);
    assert.equal(result.guidanceWritten, false);
    assert.ok(result.note && /kept your own/.test(result.note), "the skip is never silent");
    assert.equal(readFileSync(guidancePath(ws), "utf8"), mine, "the user's file is byte-for-byte intact");
  });

  test("a version bump overwrites an unedited on-disk copy", async () => {
    const root = mkdtempSync(join(tmpdir(), "jwcs-ws-"));
    const ws = workspaceUnder(root);
    // An older shipped version on disk (marker v0, still ours).
    mkdirSync(dirname(guidancePath(ws)), { recursive: true });
    writeFileSync(guidancePath(ws), `${managedMarker(GITHUB_GUIDANCE_MARKER_ID, 0)}\n# stale shipped guidance\n`);
    const { svc } = service({ workspaceRoot: root });
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    const result = await svc.bind("sub-alice", "alice", ws, account.id);
    assert.equal(result.guidanceWritten, true);
    assert.ok(result.note && /refreshed/.test(result.note));
    assert.equal(readFileSync(guidancePath(ws), "utf8"), managedContent(githubGuidanceFile()));
  });

  test("bind-to-none removes the file", async () => {
    const root = mkdtempSync(join(tmpdir(), "jwcs-ws-"));
    const ws = workspaceUnder(root);
    const { svc, store } = service({ workspaceRoot: root });
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    await svc.bind("sub-alice", "alice", ws, account.id);
    assert.ok(existsSync(guidancePath(ws)), "written on bind");
    store.bind("sub-alice", "alice", ws, account.id);
    const result = await svc.bind("sub-alice", "alice", ws, null);
    assert.equal(result.binding, "cleared");
    assert.ok(!existsSync(guidancePath(ws)), "removed on bind-to-none");
  });

  test("bind-to-none leaves a user-replaced file in place, with a note", async () => {
    const root = mkdtempSync(join(tmpdir(), "jwcs-ws-"));
    const ws = workspaceUnder(root);
    const mine = "# my own file after I deleted the marker\n";
    mkdirSync(dirname(guidancePath(ws)), { recursive: true });
    writeFileSync(guidancePath(ws), mine);
    const { svc } = service({ workspaceRoot: root });
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    const result = await svc.bind("sub-alice", "alice", ws, null);
    assert.ok(result.note && /left your own/.test(result.note));
    assert.equal(readFileSync(guidancePath(ws), "utf8"), mine);
  });

  test("a workspace outside workspace_root is skipped and nothing is written there", async () => {
    const root = mkdtempSync(join(tmpdir(), "jwcs-ws-"));
    const outside = mkdtempSync(join(tmpdir(), "jwcs-outside-"));
    const { svc } = service({ workspaceRoot: root });
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    const result = await svc.bind("sub-alice", "alice", outside, account.id);
    assert.equal(result.guidanceWritten, false);
    assert.ok(result.note && /workspace_root/.test(result.note));
    assert.ok(!existsSync(guidancePath(outside)), "nothing written outside the root");
  });

  test("with NO workspace_root the guidance write is skipped (split-host: nothing here yet)", async () => {
    const dir = mkdtempSync(join(tmpdir(), "jwcs-ws-"));
    const { svc } = service(); // no workspaceRoot
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    const result = await svc.bind("sub-alice", "alice", dir, account.id);
    assert.equal(result.guidanceWritten, false);
    assert.ok(!existsSync(guidancePath(dir)));
  });
});

describe("github guidance — the example workspace ships the current file (drift guard)", () => {
  test("examples/gh/.jaato/instructions/40-github.md equals the generated content", () => {
    const example = join(dirname(new URL(import.meta.url).pathname), "..", "examples", "gh", GITHUB_GUIDANCE_PATH);
    assert.ok(existsSync(example), `expected the example guidance file at ${example}`);
    assert.equal(
      readFileSync(example, "utf8"),
      managedContent(githubGuidanceFile()),
      "the hand-shipped example file has drifted from src/github-guidance.ts — regenerate it (and bump GITHUB_GUIDANCE_VERSION if the content changed)",
    );
  });
});
