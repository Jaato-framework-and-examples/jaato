import { strict as assert } from "node:assert";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, test } from "node:test";
import { FileGitHubStore } from "../src/github-store.js";
import { GitHubService, GH_TOKEN_ENV_VALUE, upsertEnvLine, renderGitConfig, tokenSetToSecret } from "../src/github.js";
import { FakeGitHubApi, FakeReloader, FakeWorkspaceWriter } from "./github-fakes.js";

const KEY = "g".repeat(40);
const storePath = () => join(mkdtempSync(join(tmpdir(), "jwcs-ghsvc-")), "github.json");

function service(opts: { workspaceRoot?: string; writer?: FakeWorkspaceWriter } = {}) {
  const store = new FileGitHubStore(storePath(), KEY);
  const api = new FakeGitHubApi();
  const reloader = new FakeReloader();
  const svc = new GitHubService({ store, api, reloader, workspaceRoot: opts.workspaceRoot, workspaceWriter: opts.writer, marginSeconds: 300 });
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
    assert.ok(result.note && /no workspace_root/.test(result.note));
    assert.match(result.note!, /will not get GH_TOKEN/, "the note says what the skip costs");
    assert.doesNotMatch(result.note!, /config\.update/, "no promise of a write path that does not exist");
    assert.ok(!existsSync(join(dir, ".env")));
  });
});

describe("github service — the daemon writes what this server cannot reach", () => {
  test("with no workspace_root, bind asks the daemon for the reference, the gitconfig and the guidance", async () => {
    const writer = new FakeWorkspaceWriter();
    const { svc } = service({ writer });
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    const result = await svc.bind("sub-alice", "alice", "/root/.jaato/workspaces/proj", account.id);
    assert.equal(writer.calls.length, 1);
    const call = writer.calls[0]!;
    assert.equal(call.user, "alice");
    assert.equal(call.workspace, "/root/.jaato/workspaces/proj");
    assert.deepEqual(call.request.env, { GH_TOKEN: GH_TOKEN_ENV_VALUE });
    assert.deepEqual(call.request.files.map((f) => f.path), [".home/.gitconfig", ".jaato/instructions/40-github.md"]);
    assert.equal(call.request.files[1]!.managed_by, "github-guidance");
    assert.equal(result.envWritten, true);
    assert.equal(result.gitconfigSeeded, true);
    assert.equal(result.note, undefined);
  });

  test("bind-to-none asks the daemon to remove the reference and the managed guidance", async () => {
    const writer = new FakeWorkspaceWriter();
    const { svc, store } = service({ writer });
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    store.bind("sub-alice", "alice", "/w", account.id);
    const result = await svc.bind("sub-alice", "alice", "/w", null);
    const req = writer.calls[0]!.request;
    assert.deepEqual(req.env, { GH_TOKEN: null });
    assert.deepEqual(req.files, [{ path: ".jaato/instructions/40-github.md", content: null, managed_by: "github-guidance" }]);
    assert.equal(result.envWritten, true);
  });

  test("a daemon refusal is a note, never a silent success", async () => {
    const writer = new FakeWorkspaceWriter();
    writer.answer = { status: "not_found", env: {}, files: [], detail: "no such workspace for this user" };
    const { svc } = service({ writer });
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    const result = await svc.bind("sub-alice", "alice", "/w", account.id);
    assert.equal(result.envWritten, false);
    assert.match(result.note!, /not_found: no such workspace for this user/);
  });

  test("a daemon below 1.30 is not asked, and the note says so", async () => {
    const writer = new FakeWorkspaceWriter();
    writer.supported = false;
    const { svc } = service({ writer });
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    const result = await svc.bind("sub-alice", "alice", "/w", account.id);
    assert.equal(writer.calls.length, 0);
    assert.match(result.note!, /workspace\.app_write/);
  });

  test("resync writes every recorded binding and reloads only the users whose .env changed", async () => {
    const writer = new FakeWorkspaceWriter();
    const { svc, store, reloader } = service({ writer });
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    store.bind("sub-alice", "alice", "/w1", account.id);
    store.bind("sub-alice", "alice", "/w2", account.id);
    const first = await svc.resyncWorkspaces();
    assert.equal(first.bindings, 2);
    assert.equal(first.changed, 2);
    assert.deepEqual(writer.calls.map((c) => c.workspace), ["/w1", "/w2"]);
    assert.deepEqual(reloader.calls, ["alice"], "one reload per user, not per workspace");
    writer.answer = { status: "ok", env: { GH_TOKEN: "unchanged" }, files: [], detail: undefined };
    const again = await svc.resyncWorkspaces();
    assert.equal(again.changed, 0);
    assert.deepEqual(reloader.calls, ["alice"], "an unchanged resync reloads nothing");
  });

  test("resync drops a binding the daemon has no workspace for, and keeps the rest", async () => {
    const writer = new FakeWorkspaceWriter();
    const { svc, store } = service({ writer });
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    store.bind("sub-alice", "alice", "/gone", account.id);
    store.bind("sub-alice", "alice", "/live", account.id);
    const real = writer.writeWorkspace.bind(writer);
    writer.writeWorkspace = async (user, workspace, request) => workspace === "/gone"
      ? { status: "not_found", env: {}, files: [], detail: "no such workspace for this user" }
      : real(user, workspace, request);
    const result = await svc.resyncWorkspaces();
    assert.equal(result.removed, 1);
    assert.deepEqual(store.allBindings().map((b) => b.workspace), ["/live"]);
    assert.equal(result.notes.length, 0, "a dropped binding is not also reported as a refusal");
  });

  test("resync drops a binding keyed by a name, without asking the daemon", async () => {
    const writer = new FakeWorkspaceWriter();
    const { svc, store } = service({ writer });
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    store.bind("sub-alice", "alice", "legacy-name", account.id);
    const result = await svc.resyncWorkspaces();
    assert.equal(result.removed, 1);
    assert.equal(writer.calls.length, 0);
    assert.deepEqual(store.allBindings(), []);
  });

  test("resync keeps a binding the daemon refused for another reason", async () => {
    const writer = new FakeWorkspaceWriter();
    writer.answer = { status: "error", env: {}, files: [], detail: "disk full" };
    const { svc, store } = service({ writer });
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    store.bind("sub-alice", "alice", "/w", account.id);
    const result = await svc.resyncWorkspaces();
    assert.equal(result.removed, 0);
    assert.equal(store.allBindings().length, 1);
  });

  test("an explicit bind that the daemon answers not_found keeps the binding and says why", async () => {
    const writer = new FakeWorkspaceWriter();
    writer.answer = { status: "not_found", env: {}, files: [], detail: "no such workspace for this user" };
    const { svc, store } = service({ writer });
    const account = await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    const result = await svc.bind("sub-alice", "alice", "/w", account.id);
    assert.match(result.note!, /not_found/);
    assert.equal(store.allBindings().length, 1);
  });
});

describe("github service — which repositories the App can write to", () => {
  test("the listing names where the App is installed and links to installing it", async () => {
    const { svc, api } = service();
    api.identity = { ...api.identity, installations: [{ id: 7, account: "alice", appSlug: "jaato-web-coder" }] };
    api.installationRepos.set(7, [{ fullName: "alice/dots", private: false, defaultBranch: "main" }]);
    await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    const l = await svc.listRepos("sub-alice");
    assert.deepEqual(l.installedOn, ["alice"]);
    assert.equal(l.truncated, false);
    assert.equal(l.installUrl, "https://github.com/apps/jaato-web-coder/installations/new");
    // An org repository the App is not installed on is absent: the listing
    // is complete, so its absence means the token cannot write there.
    assert.ok(!l.repos.some((r) => r.fullName.startsWith("Jaato-framework-and-examples/")));
  });

  test("a configured slug wins, and a GHE web host is used for the link", async () => {
    const store = new FileGitHubStore(storePath(), KEY);
    const api = new FakeGitHubApi();
    const svc = new GitHubService({ store, api, reloader: new FakeReloader(), appSlug: "configured", webBaseUrl: "https://ghe.example/" });
    api.installationRepos.set(7, []);
    await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    assert.equal((await svc.listRepos("sub-alice")).installUrl, "https://ghe.example/apps/configured/installations/new");
  });

  test("an installation that failed to list makes the listing incomplete, a removed one does not", async () => {
    const { svc, api } = service();
    api.identity = { ...api.identity, installations: [{ id: 7, account: "acme" }, { id: 8, account: "alice" }] };
    api.installationRepos.set(7, [{ fullName: "acme/api", private: true, defaultBranch: "main" }]);
    await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    // 8 answers 404 while GitHub still reports it: unknown, so incomplete.
    assert.equal((await svc.listRepos("sub-alice")).truncated, true);
  });

  test("a stored installation GitHub no longer reports is not asked and does not make the listing incomplete", async () => {
    const store = new FileGitHubStore(storePath(), KEY);
    const api = new FakeGitHubApi();
    const svc = new GitHubService({ store, api, reloader: new FakeReloader() });
    api.identity = { ...api.identity, installations: [{ id: 7, account: "acme" }, { id: 9, account: "gone" }] };
    await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    api.identity = { ...api.identity, installations: [{ id: 7, account: "acme" }] };
    api.installationRepos.set(7, [{ fullName: "acme/api", private: true, defaultBranch: "main" }]);
    const l = await svc.listRepos("sub-alice");
    assert.equal(l.truncated, false);
    assert.deepEqual(l.installedOn, ["acme"]);
    assert.ok(!api.repoCalls.some((c) => c.installationId === 9));
  });
});

describe("github service — repo autocomplete", () => {
  test("owner/partial searches that owner, and flags what the App cannot write to", async () => {
    const { svc, api } = service();
    api.identity = { ...api.identity, installations: [{ id: 7, account: "alice" }] };
    api.installationRepos.set(7, [{ fullName: "alice/jaato-notes", private: false, defaultBranch: "main" }]);
    api.searchable = [
      { fullName: "Jaato-framework-and-examples/jaato", private: false, defaultBranch: "main" },
      { fullName: "alice/jaato-notes", private: false, defaultBranch: "main" },
    ];
    await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    const org = await svc.searchRepos("sub-alice", "Jaato-framework-and-examples/jaa");
    assert.deepEqual(org.repos.map((r) => [r.fullName, r.appCanWrite]), [["Jaato-framework-and-examples/jaato", false]]);
    assert.equal(api.searchCalls.at(-1)!.query, "jaa in:name user:Jaato-framework-and-examples");
    const word = await svc.searchRepos("sub-alice", "jaato");
    assert.deepEqual(word.repos.map((r) => [r.fullName, r.appCanWrite]), [["Jaato-framework-and-examples/jaato", false], ["alice/jaato-notes", true]]);
  });

  test("a term too short or outside the name alphabet does not reach GitHub", async () => {
    const { svc, api } = service();
    api.installationRepos.set(7, []);
    await svc.completeConnect("sub-alice", "alice", "code", "https://app/cb");
    for (const t of ["", "j", "a b", "x/../y", "user:evil"]) assert.deepEqual((await svc.searchRepos("sub-alice", t)).repos, []);
    assert.equal(api.searchCalls.length, 0);
  });
});
