import { strict as assert } from "node:assert";
import { chmodSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, test } from "node:test";
import { ConfigError, configFromObject, loadConfig, parseDuration, parseListen } from "../src/config.js";
import { testConfig, writeSecrets } from "./helpers.js";

describe("config", () => {
  test("a complete config loads with the documented defaults", () => {
    const c = testConfig("ws://127.0.0.1:1", "https://jaato.example.org");
    assert.equal(c.daemon.bindUrl, c.daemon.url);
    assert.equal(c.auth.oidc.subjectClaim, "preferred_username");
    assert.deepEqual(c.auth.oidc.scopes, ["openid", "profile"]);
    assert.equal(c.session.cookieName, "jaato_web_coder_session");
    assert.equal(c.session.ttlSeconds, 3600);
    assert.equal(c.ticket.ttlSeconds, 60);
    assert.equal(c.mode, "direct");
  });

  test("a secret file readable by others is refused, not read", () => {
    const { dir } = writeSecrets();
    chmodSync(join(dir, "app.credential"), 0o644);
    assert.throws(() => configFromObject({
      listen: ":1", public_url: "https://x", daemon: { url: "ws://d", app_id: "a", app_credential_file: "app.credential" },
      auth: { oidc: { issuer: "https://i", client_id: "c", client_secret_file: "oidc.secret" } }, session: { secret_file: "session.secret" },
    }, dir), /readable by group\/others/);
  });

  test("plain-http back channel is accepted on loopback only; app_id may not carry ':'", () => {
    assert.throws(() => testConfig("ws://d", "https://x", { auth: { oidc: { issuer: "https://i", backchannel_url: "http://10.0.0.5:8180", client_id: "c", client_secret_file: "oidc.secret" } } }), /loopback/);
    const ok = testConfig("ws://d", "https://x", { auth: { oidc: { issuer: "https://i", backchannel_url: "http://127.0.0.1:8180", client_id: "c", client_secret_file: "oidc.secret" } } });
    assert.equal(ok.auth.oidc.backchannelUrl, "http://127.0.0.1:8180");
    assert.throws(() => testConfig("ws://d", "https://x", { daemon: { url: "ws://d", app_id: "bad:id", app_credential_file: "app.credential" } }), /app_id/);
  });

  test("ticket ttl outside the daemon's 1..3600 is refused here rather than by the daemon later", () => {
    assert.throws(() => testConfig("ws://d", "https://x", { ticket: { ttl_seconds: 0 } }), ConfigError);
    assert.throws(() => testConfig("ws://d", "https://x", { ticket: { ttl_seconds: 4000 } }), ConfigError);
  });

  test("durations and listen addresses", () => {
    assert.equal(parseDuration("8h", "x"), 28800);
    assert.equal(parseDuration("30m", "x"), 1800);
    assert.equal(parseDuration("45s", "x"), 45);
    assert.equal(parseDuration(90, "x"), 90);
    assert.throws(() => parseDuration("soon", "x"), ConfigError);
    assert.deepEqual(parseListen(":8443"), { host: "127.0.0.1", port: 8443 });
    assert.deepEqual(parseListen("0.0.0.0:80"), { host: "0.0.0.0", port: 80 });
    assert.deepEqual(parseListen("[::1]:9"), { host: "::1", port: 9 });
  });

  test("loadConfig reads YAML with paths relative to the file", () => {
    const { dir } = writeSecrets();
    writeFileSync(join(dir, "server.yaml"), `
public_url: https://jaato.example.org
daemon: {url: wss://jaato.example.org/daemon, bind_url: ws://127.0.0.1:8080, app_id: jaato-web-coder, app_credential_file: app.credential}
auth: {kind: oidc, oidc: {issuer: https://jaato.example.org/auth/realms/jaato-web-coder-shell, client_id: jaato-web-coder, client_secret_file: oidc.secret, required_role: jaato-user}}
session: {secret_file: session.secret}
`);
    const c = loadConfig(join(dir, "server.yaml"));
    assert.equal(c.daemon.bindUrl, "ws://127.0.0.1:8080");
    assert.equal(c.auth.oidc.requiredRole, "jaato-user");
    assert.equal(c.session.ttlSeconds, 8 * 3600);
    const empty = mkdtempSync(join(tmpdir(), "jwcs-empty-"));
    assert.throws(() => loadConfig(join(empty, "nope.yaml")), /cannot read config/);
  });
});
