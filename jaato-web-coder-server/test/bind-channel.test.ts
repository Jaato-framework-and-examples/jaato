import { strict as assert } from "node:assert";
import { afterEach, beforeEach, describe, test } from "node:test";
import { BindChannel, BindRefusedError, BindUnavailableError } from "../src/bind-channel.js";
import { APP_CREDENTIAL, startMockDaemon, type MockDaemon } from "./helpers.js";

describe("BindChannel", () => {
  let daemon: MockDaemon;
  beforeEach(async () => { daemon = await startMockDaemon(); });
  afterEach(async () => { await daemon.close(); });

  test("authenticates with the app credential in the Authorization header and mints per request", async () => {
    const ch = new BindChannel({ bindUrl: daemon.url, appCredential: APP_CREDENTIAL });
    await ch.connect();
    assert.deepEqual(daemon.authHeaders, [`Bearer ${APP_CREDENTIAL}`]);
    const [a, b] = await Promise.all([ch.bind("alice", 60), ch.bind("bob", 60)]);
    assert.equal(a.ticket, "ticket-1"); assert.equal(a.qualified, "jaato-web-coder:alice"); assert.equal(a.appId, "jaato-web-coder");
    assert.equal(b.ticket, "ticket-2");
    assert.deepEqual(daemon.binds.map((x) => [x.user, x.ttl_seconds, x.single_use]), [["alice", 60, true], ["bob", 60, true]]);
    await ch.close();
  });

  test("a non-bound status is a BindRefusedError carrying the daemon's status", async () => {
    const ch = new BindChannel({ bindUrl: daemon.url, appCredential: APP_CREDENTIAL });
    await ch.connect();
    daemon.nextBindStatus = "capacity";
    await assert.rejects(ch.bind("alice", 60), (e: unknown) => e instanceof BindRefusedError && e.status === "capacity");
    await ch.close();
  });

  test("the wrong credential is denied by the daemon, never a ticket", async () => {
    const ch = new BindChannel({ bindUrl: daemon.url, appCredential: "not-the-credential-00000000000000000" });
    await ch.connect();
    await assert.rejects(ch.bind("alice", 60), (e: unknown) => e instanceof BindRefusedError && e.status === "denied");
    await ch.close();
  });

  test("revokeUser sends the user form and reports the daemon's answer", async () => {
    const ch = new BindChannel({ bindUrl: daemon.url, appCredential: APP_CREDENTIAL });
    await ch.connect();
    assert.deepEqual(await ch.revokeUser("alice"), { status: "not_found", revoked: 0 });
    assert.deepEqual(daemon.revokes, [{ user: "alice", ticket: undefined }]);
    await ch.close();
  });

  test("a bind with the channel down is unavailable, not an error page", async () => {
    const ch = new BindChannel({ bindUrl: daemon.url, appCredential: APP_CREDENTIAL, timeoutMs: 200 });
    await assert.rejects(ch.bind("alice", 60), BindUnavailableError);
    await ch.close().catch(() => undefined);
  });

  test("reloadUser sends secret.reload and reports the daemon's answer (#1226 §6.4)", async () => {
    const ch = new BindChannel({ bindUrl: daemon.url, appCredential: APP_CREDENTIAL });
    await ch.connect();
    daemon.nextReloadCount = 2;
    assert.deepEqual(await ch.reloadUser("alice"), { status: "ok", reloaded: 2 });
    assert.deepEqual(daemon.reloads, ["alice"]);
    await ch.close();
  });

  test("attachSecretResolver answers a daemon-initiated secret.resolve, correlated by request_id", async () => {
    const ch = new BindChannel({ bindUrl: daemon.url, appCredential: APP_CREDENTIAL });
    await ch.connect();
    const seen: Array<{ user: string; workspace: string; name: string }> = [];
    ch.attachSecretResolver(async (req) => { seen.push(req); return { value: "the-token", expiresAt: "2026-01-01T00:00:00Z" }; });
    // Give the subscription a tick to register before the daemon asks.
    await new Promise((r) => setTimeout(r, 20));
    const answer = await daemon.askSecretResolve({ request_id: "r-1", user: "alice", workspace: "/ws/one", name: "github" });
    assert.equal(answer.status, "ok");
    assert.equal(answer.value, "the-token");
    assert.equal(answer.expires_at, "2026-01-01T00:00:00Z");
    assert.deepEqual(seen, [{ user: "alice", workspace: "/ws/one", name: "github" }]);
    await ch.close();
  });

  test("a resolver that declines answers not_found with no value", async () => {
    const ch = new BindChannel({ bindUrl: daemon.url, appCredential: APP_CREDENTIAL });
    await ch.connect();
    ch.attachSecretResolver(async () => ({ status: "not_found", detail: "not_bound" }));
    await new Promise((r) => setTimeout(r, 20));
    const answer = await daemon.askSecretResolve({ request_id: "r-2", user: "bob", workspace: "/ws/x", name: "github" });
    assert.equal(answer.status, "not_found");
    assert.equal(answer.value, undefined);
    await ch.close();
  });
});
