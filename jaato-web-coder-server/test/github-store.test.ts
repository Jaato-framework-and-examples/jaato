import { strict as assert } from "node:assert";
import { mkdtempSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, test } from "node:test";
import { FileGitHubStore, GitHubStoreError, type GrantInput } from "../src/github-store.js";

const KEY = "g".repeat(40);
const fresh = () => join(mkdtempSync(join(tmpdir(), "jwcs-gh-")), "sub", "github.json");

const grant = (login = "alice", id = 1, refreshToken = "refresh-1"): GrantInput => ({
  login, githubId: id, name: `${login} Example`, noreplyEmail: `${id}+${login}@users.noreply.github.com`,
  installations: [{ id: 9, account: "acme" }],
  secret: { refreshToken, refreshExpiresAt: Date.now() + 1e9, accessToken: "access-1", accessExpiresAt: Date.now() + 1e7 },
});

describe("github store", () => {
  test("connect → list shows non-secret metadata, never the token; the file holds no token", () => {
    const path = fresh();
    const store = new FileGitHubStore(path, KEY, () => new Date("2026-09-23T10:00:00Z"));
    const a = store.connect("sub-alice", "alice", grant("alice", 1, "refresh-SECRET-token"));
    assert.equal(a.login, "alice");
    assert.equal(a.isDefault, true, "first account is default");
    assert.deepEqual(store.listAccounts("sub-alice").map((x) => x.login), ["alice"]);
    const onDisk = readFileSync(path, "utf8");
    assert.ok(!onDisk.includes("refresh-SECRET-token"), "refresh token must not be in the file");
    assert.ok(!onDisk.includes("access-1"), "access token must not be in the file");
    assert.equal(statSync(path).mode & 0o777, 0o600);
    // The secret round-trips through decryption.
    assert.equal(store.secretOf(a.id)?.refreshToken, "refresh-SECRET-token");
  });

  test("owners are isolated: another sub sees nothing and cannot disconnect", () => {
    const store = new FileGitHubStore(fresh(), KEY);
    const a = store.connect("sub-alice", "alice", grant());
    assert.deepEqual(store.listAccounts("sub-bob"), []);
    assert.equal(store.disconnect("sub-bob", a.id), null);
    assert.equal(store.accountById(a.id)?.login, "alice");
  });

  test("re-connecting the same github id updates in place (fresh tokens), not a duplicate row", () => {
    const store = new FileGitHubStore(fresh(), KEY);
    const a = store.connect("sub-alice", "alice", grant("alice", 1, "refresh-old"));
    const b = store.connect("sub-alice", "alice", grant("alice", 1, "refresh-new"));
    assert.equal(a.id, b.id, "same grant id");
    assert.equal(store.listAccounts("sub-alice").length, 1);
    assert.equal(store.secretOf(a.id)?.refreshToken, "refresh-new");
  });

  test("a second account is not default; setDefault flips exactly one", () => {
    const store = new FileGitHubStore(fresh(), KEY);
    const a = store.connect("sub-alice", "alice", grant("alice", 1));
    const b = store.connect("sub-alice", "alice", grant("alice-work", 2));
    assert.equal(store.listAccounts("sub-alice").find((x) => x.id === b.id)?.isDefault, false);
    assert.equal(store.setDefault("sub-alice", b.id), true);
    const accts = store.listAccounts("sub-alice");
    assert.equal(accts.find((x) => x.id === a.id)?.isDefault, false);
    assert.equal(accts.find((x) => x.id === b.id)?.isDefault, true);
  });

  test("bindings key on the DAEMON-facing user; resolve finds the grant by id without a sub", () => {
    const store = new FileGitHubStore(fresh(), KEY);
    const a = store.connect("sub-alice", "alice", grant());
    store.bind("sub-alice", "alice", "/ws/one", a.id);
    // The resolve path knows only (user, workspace):
    assert.equal(store.bindingFor("alice", "/ws/one"), a.id);
    assert.equal(store.bindingFor("alice", "/ws/other"), null);
    assert.equal(store.bindingFor("bob", "/ws/one"), null);
    assert.ok(store.userHasAnyAccount("alice"));
    assert.ok(!store.userHasAnyAccount("bob"));
    assert.deepEqual(store.listBindings("sub-alice"), [{ workspace: "/ws/one", accountId: a.id }]);
  });

  test("binding to a foreign or unknown account is refused", () => {
    const store = new FileGitHubStore(fresh(), KEY);
    const a = store.connect("sub-alice", "alice", grant());
    assert.throws(() => store.bind("sub-bob", "bob", "/ws/one", a.id), GitHubStoreError);
    assert.throws(() => store.bind("sub-alice", "alice", "/ws/one", "no-such-id"), GitHubStoreError);
  });

  test("disconnect removes the grant AND every binding pointing at it; default falls to the oldest remaining", () => {
    const store = new FileGitHubStore(fresh(), KEY);
    const a = store.connect("sub-alice", "alice", grant("alice", 1));
    const b = store.connect("sub-alice", "alice", grant("alice-work", 2));
    store.bind("sub-alice", "alice", "/ws/one", a.id);
    const removed = store.disconnect("sub-alice", a.id);
    assert.equal(removed?.account.login, "alice");
    assert.equal(removed?.user, "alice");
    assert.equal(store.accountById(a.id), null);
    assert.equal(store.bindingFor("alice", "/ws/one"), null, "binding dropped with its grant");
    assert.equal(store.listAccounts("sub-alice").find((x) => x.id === b.id)?.isDefault, true, "the survivor becomes default");
  });

  test("clearing a binding (grantId null) removes it", () => {
    const store = new FileGitHubStore(fresh(), KEY);
    const a = store.connect("sub-alice", "alice", grant());
    store.bind("sub-alice", "alice", "/ws/one", a.id);
    store.bind("sub-alice", "alice", "/ws/one", null);
    assert.equal(store.bindingFor("alice", "/ws/one"), null);
  });

  test("editing the plaintext owner to re-attribute a grant makes it undecryptable, not somebody else's", () => {
    const path = fresh();
    const store = new FileGitHubStore(path, KEY);
    const a = store.connect("sub-alice", "alice", grant("alice", 1, "refresh-SECRET"));
    const raw = JSON.parse(readFileSync(path, "utf8"));
    raw.grants[0].owner = "sub-bob";
    writeFileSync(path, JSON.stringify(raw));
    const tampered = new FileGitHubStore(path, KEY);
    // The AAD binds the ciphertext to owner+id, so decryption fails rather than
    // handing bob alice's token.
    assert.throws(() => tampered.secretOf(a.id));
  });
});
