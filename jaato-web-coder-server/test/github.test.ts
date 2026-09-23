import { strict as assert } from "node:assert";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, test } from "node:test";
import { FileGitHubStore } from "../src/github-store.js";
import { GitHubService, GH_TOKEN_ENV_VALUE, upsertEnvLine, renderGitConfig, tokenSetToSecret } from "../src/github.js";
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

/** Seed a connected + bound account and return its id. */
async function connectAndBind(svc: GitHubService, store: FileGitHubStore, workspace: string): Promise<string> {
  const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
  store.bind("sub-alice", "alice", workspace, account.id);
  return account.id;
}

describe("pure helpers", () => {
  test("upsertEnvLine adds, replaces and removes GH_TOKEN, preserving other lines", () => {
    assert.equal(upsertEnvLine("", "GH_TOKEN", "app://github"), "GH_TOKEN=app://github\n");
    assert.equal(upsertEnvLine("A=1\n", "GH_TOKEN", "app://github"), "A=1\nGH_TOKEN=app://github\n");
    assert.equal(upsertEnvLine("A=1\nGH_TOKEN=old\nB=2\n", "GH_TOKEN", "app://github"), "A=1\nB=2\nGH_TOKEN=app://github\n");
    assert.equal(upsertEnvLine("A=1\nexport GH_TOKEN=old\n", "GH_TOKEN", null), "A=1\n");
    assert.equal(upsertEnvLine("GH_TOKEN=old\n", "GH_TOKEN", null), "");
  });

  test("renderGitConfig carries the identity and the gh credential helper", () => {
    const c = renderGitConfig({ name: "Alice Example", email: "1+alice@users.noreply.github.com", gitHost: "github.com" });
    assert.match(c, /\[user\]/);
    assert.match(c, /name = Alice Example/);
    assert.match(c, /email = 1\+alice@users\.noreply\.github\.com/);
    assert.match(c, /\[credential "https:\/\/github\.com"\]/);
    assert.match(c, /helper = !gh auth git-credential/);
  });

  test("tokenSetToSecret carries the previous refresh token when a response omits one", () => {
    const prev = { refreshToken: "r-old", refreshExpiresAt: 999 };
    const s = tokenSetToSecret({ accessToken: "a", accessExpiresInSeconds: 100 }, 0, prev);
    assert.equal(s.refreshToken, "r-old");
    assert.equal(s.accessExpiresAt, 100_000);
  });
});

describe("github service — connect and mint", () => {
  test("completeConnect exchanges the code, fetches identity and stores the grant", async () => {
    const { svc, store, api } = service();
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    assert.equal(api.exchanges, 1);
    assert.equal(account.login, "alice");
    assert.equal(account.noreplyEmail, "4242+alice@users.noreply.github.com");
    assert.deepEqual(account.installations, [{ id: 7, account: "acme" }]);
    assert.equal(store.secretOf(account.id)?.refreshToken, "refresh-1");
  });

  test("a fresh cached access token is REUSED, not refreshed", async () => {
    const { svc, store, api } = service();
    const id = await connectAndBind(svc, store, "/ws/one");
    const out = await svc.resolveSecret({ user: "alice", workspace: "/ws/one", name: "github" });
    assert.equal(out.status, "ok");
    assert.equal(out.value, "access-1");
    assert.equal(api.refreshes, 0, "the cached token was still fresh");
  });

  test("an expired access token is refreshed and the rotated refresh token is persisted", async () => {
    const { svc, store, api } = service();
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    store.bind("sub-alice", "alice", "/ws/one", account.id);
    // Age the access token past the margin.
    store.updateSecret(account.id, { ...store.secretOf(account.id)!, accessExpiresAt: Date.now() + 10_000 });
    const out = await svc.resolveSecret({ user: "alice", workspace: "/ws/one", name: "github" });
    assert.equal(out.status, "ok");
    assert.equal(api.refreshes, 1);
    // The store now holds the NEW rotated refresh token, not the one we presented.
    assert.match(store.secretOf(account.id)!.refreshToken!, /^refresh-\d+-from-refresh-1$/);
  });
});

describe("github service — the #683 rotation lock", () => {
  test("two concurrent resolves for one grant collapse to a SINGLE refresh (re-read after acquiring)", async () => {
    const { svc, store, api } = service();
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    store.bind("sub-alice", "alice", "/ws/one", account.id);
    store.updateSecret(account.id, { ...store.secretOf(account.id)!, accessExpiresAt: Date.now() + 10_000 }); // stale
    const [a, b] = await Promise.all([
      svc.resolveSecret({ user: "alice", workspace: "/ws/one", name: "github" }),
      svc.resolveSecret({ user: "alice", workspace: "/ws/one", name: "github" }),
    ]);
    assert.equal(a.status, "ok");
    assert.equal(b.status, "ok");
    assert.equal(api.refreshes, 1, "the waiter adopted the winner's token instead of refreshing again");
    assert.equal(a.value, b.value, "both got the same token");
  });
});

describe("github service — secret.resolve statuses", () => {
  test("ok for a bound account; the wire status is ok with a value", async () => {
    const { svc, store } = service();
    await connectAndBind(svc, store, "/ws/one");
    const out = await svc.resolveSecret({ user: "alice", workspace: "/ws/one", name: "github" });
    assert.equal(out.status, "ok");
    assert.ok(out.value);
  });

  test("not_connected: a user with no grant → not_found, detail names it", async () => {
    const { svc } = service();
    const out = await svc.resolveSecret({ user: "nobody", workspace: "/ws/one", name: "github" });
    assert.equal(out.status, "not_found");
    assert.match(out.detail!, /not_connected/);
  });

  test("not_bound: connected but this workspace has no binding → not_found, detail names it", async () => {
    const { svc } = service();
    await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb"); // connected, not bound
    const out = await svc.resolveSecret({ user: "alice", workspace: "/ws/unbound", name: "github" });
    assert.equal(out.status, "not_found");
    assert.match(out.detail!, /not_bound/);
  });

  test("revoked: a dead refresh token → denied, detail names it", async () => {
    const { svc, store, api } = service();
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    store.bind("sub-alice", "alice", "/ws/one", account.id);
    store.updateSecret(account.id, { ...store.secretOf(account.id)!, accessExpiresAt: Date.now() + 10_000 }); // stale, forces refresh
    api.nextRefreshRevoked = true;
    const out = await svc.resolveSecret({ user: "alice", workspace: "/ws/one", name: "github" });
    assert.equal(out.status, "denied");
    assert.match(out.detail!, /revoked/);
  });

  test("a transient refresh failure → error, not denied", async () => {
    const { svc, store, api } = service();
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    store.bind("sub-alice", "alice", "/ws/one", account.id);
    store.updateSecret(account.id, { ...store.secretOf(account.id)!, accessExpiresAt: Date.now() + 10_000 });
    api.nextRefreshError = "503 from GitHub";
    const out = await svc.resolveSecret({ user: "alice", workspace: "/ws/one", name: "github" });
    assert.equal(out.status, "error");
  });

  test("a non-github reference is declined (this app resolves only app://github)", async () => {
    const { svc } = service();
    const out = await svc.resolveSecret({ user: "alice", workspace: "/ws/one", name: "gitlab" });
    assert.equal(out.status, "not_found");
  });
});

describe("github service — disconnect", () => {
  test("disconnect revokes at GitHub and reloads the owner's sessions", async () => {
    const { svc, store, api, reloader } = service();
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    store.bind("sub-alice", "alice", "/ws/one", account.id);
    await svc.disconnect("sub-alice", "alice", account.id);
    assert.equal(api.revokes.length, 1, "grant revoked at GitHub");
    assert.deepEqual(reloader.calls, ["alice"], "the owner's sessions were reloaded");
    assert.equal(store.accountById(account.id), null);
    // The next resolve for that workspace now reports not-connected.
    const out = await svc.resolveSecret({ user: "alice", workspace: "/ws/one", name: "github" });
    assert.match(out.detail!, /not_connected/);
  });

  test("disconnecting an account of another owner is refused", async () => {
    const { svc } = service();
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    await assert.rejects(svc.disconnect("sub-bob", "bob", account.id), /no such connected account/);
  });
});

describe("github service — bind writes the workspace", () => {
  test("bind writes GH_TOKEN=app://github and seeds .home/.gitconfig, contained to workspace_root", async () => {
    const root = mkdtempSync(join(tmpdir(), "jwcs-ws-"));
    const ws = join(root, "proj");
    mkdirSync(ws, { recursive: true });
    const { svc, store } = service({ workspaceRoot: root });
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    const result = await svc.bind("sub-alice", "alice", ws, account.id);
    assert.equal(result.binding, "set");
    assert.equal(result.envWritten, true);
    assert.equal(result.gitconfigSeeded, true);
    assert.equal(readFileSync(join(ws, ".env"), "utf8").trim(), `GH_TOKEN=${GH_TOKEN_ENV_VALUE}`);
    const gitconfig = readFileSync(join(ws, ".home", ".gitconfig"), "utf8");
    assert.match(gitconfig, /email = 4242\+alice@users\.noreply\.github\.com/);
    assert.match(gitconfig, /helper = !gh auth git-credential/);
    assert.equal(store.bindingFor("alice", ws), account.id);
  });

  test("bind-to-none removes GH_TOKEN from .env and clears the binding", async () => {
    const root = mkdtempSync(join(tmpdir(), "jwcs-ws-"));
    const ws = join(root, "proj");
    mkdirSync(ws, { recursive: true });
    writeFileSync(join(ws, ".env"), `MODEL_NAME=x\nGH_TOKEN=${GH_TOKEN_ENV_VALUE}\n`);
    const { svc, store } = service({ workspaceRoot: root });
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    store.bind("sub-alice", "alice", ws, account.id);
    const result = await svc.bind("sub-alice", "alice", ws, null);
    assert.equal(result.binding, "cleared");
    assert.equal(readFileSync(join(ws, ".env"), "utf8"), "MODEL_NAME=x\n");
    assert.equal(store.bindingFor("alice", ws), null);
  });

  test("a workspace outside workspace_root records the binding but skips the .env write, and says so", async () => {
    const root = mkdtempSync(join(tmpdir(), "jwcs-ws-"));
    const outside = mkdtempSync(join(tmpdir(), "jwcs-outside-"));
    const { svc, store } = service({ workspaceRoot: root });
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    const result = await svc.bind("sub-alice", "alice", outside, account.id);
    assert.equal(result.binding, "set", "the binding is still recorded — it is what secret.resolve reads");
    assert.equal(result.envWritten, false);
    assert.ok(result.note && /workspace_root/.test(result.note));
    assert.ok(!existsSync(join(outside, ".env")), "nothing written outside the root");
    assert.equal(store.bindingFor("alice", outside), account.id);
  });

  test("with NO workspace_root the binding is recorded and the .env write is skipped with a note", async () => {
    const dir = mkdtempSync(join(tmpdir(), "jwcs-ws-"));
    const { svc } = service(); // no workspaceRoot
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    const result = await svc.bind("sub-alice", "alice", dir, account.id);
    assert.equal(result.envWritten, false);
    assert.ok(result.note && /no workspace_root configured/.test(result.note));
    assert.ok(!existsSync(join(dir, ".env")));
  });
});
