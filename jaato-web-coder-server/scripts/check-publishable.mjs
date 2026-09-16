// ``prepack`` guard: the tarball must never carry a ``file:`` dependency
// (unresolvable for anyone who installs it) or a stale build.  On the
// development checkout this FAILS by design — run scripts/prepare-publish.mjs
// first, which is what the publish workflow does.
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
const problems = [];
for (const [name, spec] of Object.entries(pkg.dependencies ?? {})) {
  if (String(spec).startsWith("file:")) problems.push(`dependency ${name} is ${spec}; run scripts/prepare-publish.mjs`);
}
if (!existsSync(join(root, "dist", "cli.js"))) problems.push("dist/cli.js is missing; run `npm run build`");
if (problems.length) {
  console.error("jaato-web-coder-server is not publishable:\n  - " + problems.join("\n  - "));
  process.exit(1);
}
