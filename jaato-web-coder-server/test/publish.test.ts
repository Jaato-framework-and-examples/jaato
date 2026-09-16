import { strict as assert } from "node:assert";
import { execFileSync } from "node:child_process";
import { cpSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, test } from "node:test";
// The script is plain ESM JavaScript; import it as such.
// @ts-expect-error — no declarations for the script module
import { preparedManifest } from "../scripts/prepare-publish.mjs";

const ROOT = resolve(import.meta.dirname, "..");

describe("publishing", () => {
  test("the development manifest is private and links its siblings with file: (so nothing can be published by accident)", () => {
    const pkg = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
    assert.equal(pkg.private, true);
    assert.match(pkg.dependencies["@jaato/sdk"], /^file:/);
    assert.match(pkg.dependencies["@jaato/web-coder-ui"], /^file:/);
    assert.equal(pkg.scripts.prepack, "node scripts/check-publishable.mjs");
  });

  test("preparedManifest rewrites file: links to the siblings' caret ranges and drops private, nothing else", () => {
    const pkg = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
    const out = preparedManifest(pkg, (rel: string) => ({ "../jaato-sdk-ts": "0.6.0", "../jaato-web-coder-ui": "0.1.0" })[rel]!);
    assert.equal(out.private, undefined);
    assert.equal(out.dependencies["@jaato/sdk"], "^0.6.0");
    assert.equal(out.dependencies["@jaato/web-coder-ui"], "^0.1.0");
    assert.equal(out.dependencies["openid-client"], pkg.dependencies["openid-client"], "non-link deps untouched");
    assert.deepEqual(out.bin, pkg.bin);
    assert.equal(out.name, "@jaato/web-coder-server");
    assert.throws(() => preparedManifest(pkg, () => "not-a-version"), /semver/);
  });

  test("the real siblings declare semver versions (a dry run against the checkout succeeds)", () => {
    const text = execFileSync(process.execPath, [join(ROOT, "scripts/prepare-publish.mjs"), "--dry-run"], { encoding: "utf8" });
    const out = JSON.parse(text);
    assert.match(out.dependencies["@jaato/sdk"], /^\^\d+\.\d+\.\d+/);
    assert.match(out.dependencies["@jaato/web-coder-ui"], /^\^\d+\.\d+\.\d+/);
  });

  test("prepack refuses the development manifest and accepts a prepared one", () => {
    const dir = mkdtempSync(join(tmpdir(), "jwcs-pack-"));
    cpSync(join(ROOT, "package.json"), join(dir, "package.json"));
    cpSync(join(ROOT, "scripts"), join(dir, "scripts"), { recursive: true });
    const run = () => execFileSync(process.execPath, [join(dir, "scripts/check-publishable.mjs")], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
    assert.throws(run, /file:/);
    // prepare-publish resolves the siblings relative to ITS OWN location, so run the checkout's copy against the temp manifest
    execFileSync(process.execPath, [join(ROOT, "scripts/prepare-publish.mjs"), "--package-json", join(dir, "package.json")], { encoding: "utf8" });
    // prepared but not built: the other guard fires
    assert.throws(run, /dist\/cli\.js/);
  });
});
