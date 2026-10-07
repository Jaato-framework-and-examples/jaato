#!/usr/bin/env node
/**
 * Turn the development manifest into the one npm may publish.
 *
 * In the checkout, package.json depends on the sibling packages through
 * ``file:`` links (so one `npm install` wires everything without anything
 * being on npm) and is ``private`` (so nobody publishes that manifest by
 * accident).  Neither may reach the registry.  This script, run by the
 * publish workflow AFTER the build, rewrites the two dependencies to the
 * caret range of the version each sibling checkout declares and drops
 * ``private``.  It changes nothing else, and it changes nothing in git:
 * the workflow's checkout is discarded.
 *
 *   node scripts/prepare-publish.mjs [--dry-run] [--package-json PATH]
 *
 * ``--dry-run`` prints the resulting manifest and writes nothing.  The
 * REGISTRY check (are those versions on npm?) is the workflow's job, since
 * the answer depends on where you are publishing to.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

/** Which local sibling each publishable dependency is linked from. */
export const SIBLINGS = {
  "@jaato/sdk": "../jaato-sdk-ts",
  "@jaato/web-coder-ui": "../jaato-web-coder-ui",
};

export function preparedManifest(manifest, readSibling = defaultReadSibling) {
  const out = JSON.parse(JSON.stringify(manifest));
  delete out.private;
  for (const [name, spec] of Object.entries(out.dependencies ?? {})) {
    if (!String(spec).startsWith("file:")) continue;
    const rel = SIBLINGS[name];
    if (!rel) throw new Error(`dependency ${name} is a file: link but has no known sibling; add it to SIBLINGS`);
    const version = readSibling(rel);
    if (!/^\d+\.\d+\.\d+/.test(version)) throw new Error(`sibling ${rel} declares no semver version (${version})`);
    out.dependencies[name] = `^${version}`;
  }
  return out;
}

function defaultReadSibling(rel) {
  return JSON.parse(readFileSync(join(HERE, "..", rel, "package.json"), "utf8")).version;
}

function main(argv) {
  const dry = argv.includes("--dry-run");
  const i = argv.indexOf("--package-json");
  const path = i >= 0 ? resolve(argv[i + 1]) : join(HERE, "..", "package.json");
  const manifest = JSON.parse(readFileSync(path, "utf8"));
  const prepared = preparedManifest(manifest);
  const text = JSON.stringify(prepared, null, 2) + "\n";
  if (dry) { process.stdout.write(text); return; }
  writeFileSync(path, text);
  process.stdout.write(`prepared ${path}: ${Object.entries(prepared.dependencies).map(([k, v]) => `${k}@${v}`).join(", ")}\n`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main(process.argv.slice(2));
