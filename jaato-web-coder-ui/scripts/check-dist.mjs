// ``prepack`` guard: ``npm pack`` / ``npm publish`` must never ship an
// empty or stale bundle.  The build is deliberately NOT run here (it
// needs the sibling ../jaato-sdk-ts checkout); the publish workflow runs
// it explicitly, and a local packer is told what to do.
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const dist = join(dirname(fileURLToPath(import.meta.url)), "..", "dist", "index.html");
if (!existsSync(dist)) {
  console.error("jaato-web-coder-ui: dist/index.html is missing — run `npm run build` before packing");
  process.exit(1);
}
