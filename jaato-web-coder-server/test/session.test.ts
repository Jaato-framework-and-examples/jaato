import { strict as assert } from "node:assert";
import { describe, test } from "node:test";
import { parseCookies, sessionCookie, SessionStore } from "../src/session.js";

describe("SessionStore", () => {
  test("round-trips a signed cookie and rejects a tampered or foreign one", () => {
    const store = new SessionStore("k".repeat(32), 3600);
    const { record, cookieValue } = store.create({ user: "alice", sub: "s1", sid: "sid1", idToken: "t" });
    assert.equal(store.resolve(cookieValue)?.id, record.id);
    const [id, sig] = cookieValue.split(".");
    assert.equal(store.resolve(`${id}.${sig!.slice(0, -1)}x`), null, "tampered signature");
    assert.equal(store.resolve(`${id}x.${sig}`), null, "tampered id");
    assert.equal(store.resolve(undefined), null);
    assert.equal(store.resolve("garbage"), null);
    const other = new SessionStore("z".repeat(32), 3600);
    assert.equal(other.resolve(cookieValue), null, "another secret");
  });

  test("expires by the injected clock and sweeps", () => {
    let now = 1_000_000;
    const store = new SessionStore("k".repeat(32), 10, () => now);
    const { cookieValue } = store.create({ user: "a", sub: "s" });
    now += 9_000; assert.ok(store.resolve(cookieValue));
    now += 2_000; assert.equal(store.resolve(cookieValue), null);
    store.create({ user: "b", sub: "s2" }); now += 20_000;
    assert.equal(store.sweep(), 1);
    assert.equal(store.size, 0);
  });

  test("deleteMatching ends every session of a subject or an OIDC sid (back-channel logout)", () => {
    const store = new SessionStore("k".repeat(32), 3600);
    store.create({ user: "alice", sub: "s1", sid: "sidA" });
    store.create({ user: "alice", sub: "s1", sid: "sidB" });
    store.create({ user: "bob", sub: "s2", sid: "sidC" });
    assert.equal(store.deleteMatching({ sid: "sidB" }).length, 1);
    assert.equal(store.deleteMatching({ sub: "s1" }).length, 1);
    assert.equal(store.size, 1);
  });

  test("cookie helpers", () => {
    assert.equal(parseCookies("a=1; b=2; a=3").get("a"), "1");
    const c = sessionCookie("n", "v", { secure: true, maxAgeSeconds: 60 });
    assert.match(c, /^n=v; Path=\/; HttpOnly; SameSite=Lax; Secure; Max-Age=60$/);
    assert.ok(!sessionCookie("n", "v", { secure: false }).includes("Secure"));
  });
});
