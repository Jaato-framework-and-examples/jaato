import { strict as assert } from "node:assert";
import { mkdtempSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, test } from "node:test";
import { CredentialError, FileCredentialStore, autoLabel, secretHint } from "../src/credentials.js";

const KEY = "k".repeat(40);
const fresh = () => join(mkdtempSync(join(tmpdir(), "jwcs-cred-")), "sub", "credentials.json");

describe("credential store", () => {
  test("hint and auto label: the tail of a long secret, nothing for a short one", () => {
    assert.equal(secretHint("sk-abcdefghijklmnop"), "mnop");
    assert.equal(secretHint("short"), "");
    assert.equal(autoLabel("zhipuai", "sk-abcdefghijklmnop"), "zhipuai …mnop");
    assert.equal(autoLabel("zhipuai", "short"), "zhipuai");
  });

  test("add → list shows label and hint, never the secret; reveal returns it; the file holds no plaintext", () => {
    const path = fresh();
    const store = new FileCredentialStore(path, KEY, () => new Date("2026-09-16T10:00:00Z"));
    const e = store.add("sub-alice", "zhipuai", "sk-abcdefghijklmnop", "work");
    assert.deepEqual(store.list("sub-alice"), [{ id: e.id, provider: "zhipuai", label: "work", hint: "mnop", createdAt: "2026-09-16T10:00:00.000Z" }]);
    assert.equal(store.reveal("sub-alice", e.id), "sk-abcdefghijklmnop");
    const onDisk = readFileSync(path, "utf8");
    assert.ok(!onDisk.includes("sk-abcdefghijklmnop"), "secret must not be in the file");
    assert.ok(!onDisk.includes("abcdefghijkl"), "not even a fragment");
    assert.equal(statSync(path).mode & 0o777, 0o600);
  });

  test("owners are isolated: another sub sees nothing, cannot reveal, cannot delete", () => {
    const store = new FileCredentialStore(fresh(), KEY);
    const e = store.add("sub-alice", "zhipuai", "sk-abcdefghijklmnop");
    assert.deepEqual(store.list("sub-bob"), []);
    assert.equal(store.reveal("sub-bob", e.id), null);
    assert.equal(store.remove("sub-bob", e.id), false);
    assert.equal(store.reveal("sub-alice", e.id), "sk-abcdefghijklmnop");
  });

  test("the same secret stored twice for one owner and provider is one entry (a re-label updates it)", () => {
    const store = new FileCredentialStore(fresh(), KEY);
    const a = store.add("s", "zhipuai", "sk-abcdefghijklmnop");
    const b = store.add("s", "zhipuai", "sk-abcdefghijklmnop", "renamed");
    assert.equal(a.id, b.id);
    assert.equal(store.list("s").length, 1);
    assert.equal(store.list("s")[0]!.label, "renamed");
    // A different provider with the same bytes is a different entry.
    const c = store.add("s", "openrouter", "sk-abcdefghijklmnop");
    assert.notEqual(c.id, a.id);
    assert.deepEqual(store.list("s", "openrouter").map((x) => x.id), [c.id]);
  });

  test("entries survive a restart with the same key and are unreadable with another", () => {
    const path = fresh();
    const id = new FileCredentialStore(path, KEY).add("s", "zhipuai", "sk-abcdefghijklmnop").id;
    assert.equal(new FileCredentialStore(path, KEY).reveal("s", id), "sk-abcdefghijklmnop");
    assert.throws(() => new FileCredentialStore(path, "x".repeat(40)).reveal("s", id), /Unsupported state|unable to authenticate/);
  });

  test("re-attributing an entry by editing the file makes it undecryptable rather than somebody else's", () => {
    const path = fresh();
    const store = new FileCredentialStore(path, KEY);
    const id = store.add("sub-alice", "zhipuai", "sk-abcdefghijklmnop").id;
    const raw = JSON.parse(readFileSync(path, "utf8"));
    raw.entries[0].owner = "sub-bob";
    writeFileSync(path, JSON.stringify(raw));
    const reopened = new FileCredentialStore(path, KEY);
    assert.throws(() => reopened.reveal("sub-bob", id), /Unsupported state|unable to authenticate/);
  });

  test("validation: provider shape, empty or multi-line secret, oversize label, short key, per-owner ceiling", () => {
    const store = new FileCredentialStore(fresh(), KEY);
    assert.throws(() => store.add("s", "Zhipu AI", "sk-abcdefghijklmnop"), CredentialError);
    assert.throws(() => store.add("s", "zhipuai", "   "), /empty/);
    assert.throws(() => store.add("s", "zhipuai", "a\nb"), /one line/);
    assert.throws(() => store.add("s", "zhipuai", "sk-abcdefghijklmnop", "x".repeat(65)), /longer/);
    assert.throws(() => new FileCredentialStore(fresh(), "short"), /at least 32/);
    for (let i = 0; i < 50; i++) store.add("s", "zhipuai", `sk-${String(i).padStart(20, "0")}`);
    assert.throws(() => store.add("s", "zhipuai", "sk-one-too-many-0000000"), (e: unknown) => e instanceof CredentialError && e.status === 409);
    // Another owner is not bounded by the first one's ceiling.
    store.add("t", "zhipuai", "sk-abcdefghijklmnop");
  });

  test("remove drops the entry and persists the removal", () => {
    const path = fresh();
    const store = new FileCredentialStore(path, KEY);
    const id = store.add("s", "zhipuai", "sk-abcdefghijklmnop").id;
    assert.equal(store.remove("s", id), true);
    assert.equal(store.remove("s", id), false);
    assert.deepEqual(new FileCredentialStore(path, KEY).list("s"), []);
  });

  test("the 'github' provider is refused here — its token never reaches the browser (no /reveal)", () => {
    const store = new FileCredentialStore(fresh(), KEY);
    assert.throws(() => store.add("s", "github", "gho_abcdefghijklmnop"), (e: unknown) => e instanceof CredentialError && /GitHub connect flow/.test(e.message));
    // A github secret cannot enter the store, so there is nothing for reveal to return.
    assert.deepEqual(store.list("s", undefined), []);
  });
});
