/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath } from "node:url";
import { readFileSync } from "node:fs";
import { execSync } from "node:child_process";

// What this bundle speaks, stamped at build time.  The SDK is compiled IN
// from ../jaato-sdk-ts/src, so nothing at runtime could otherwise say
// which SDK revision or protocol floor a deployed UI carries — and
// "this UI against that daemon" is the first question a connection
// failure raises.  Shown on the connect screen and the status bar, and
// written to dist/build-info.json for the launcher's --version.
function buildInfo() {
  const here = (rel: string) => fileURLToPath(new URL(rel, import.meta.url));
  const ui = JSON.parse(readFileSync(here("./package.json"), "utf8")) as { version: string };
  const sdk = JSON.parse(readFileSync(here("../jaato-sdk-ts/package.json"), "utf8")) as { version: string };
  const client = readFileSync(here("../jaato-sdk-ts/src/client.ts"), "utf8");
  const protocolMin = /MIN_PROTOCOL_VERSION\s*=\s*"([^"]+)"/.exec(client)?.[1] ?? "unknown";
  let commit = "unknown";
  try { commit = execSync("git rev-parse --short HEAD", { stdio: ["ignore", "pipe", "ignore"] }).toString().trim(); } catch { /* not a checkout */ }
  return { ui: ui.version, sdk: sdk.version, protocolMin, commit, builtAt: new Date().toISOString() };
}
const BUILD = buildInfo();

// The SDK is consumed from its TypeScript source so the client never
// depends on a prebuilt ``dist/`` — ``npm run build`` in jaato-web-coder-ui is
// self-sufficient, and a protocol change in jaato-sdk-ts is picked up
// on the next HMR tick.
const sdkSrc = fileURLToPath(new URL("../jaato-sdk-ts/src/index.ts", import.meta.url));

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    {
      name: "jaato-build-info",
      apply: "build",
      generateBundle() {
        this.emitFile({ type: "asset", fileName: "build-info.json", source: JSON.stringify(BUILD, null, 2) + "\n" });
      },
    },
  ],
  define: { __JAATO_BUILD__: JSON.stringify(BUILD) },
  resolve: {
    alias: {
      "@jaato/sdk": sdkSrc,
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    // The SDK source and the TUI's theme JSONs live in sibling packages.
    fs: { allow: [".."] },
    // ``VITE_WS_URL`` overrides the daemon address; by default the dev
    // server proxies ``/ws`` to a local daemon started with
    // ``python -m server --web-socket :8080`` so the page is same-origin.
    proxy: {
      "/ws": {
        target: process.env.JAATO_WS_TARGET ?? "ws://127.0.0.1:8080",
        ws: true,
        rewrite: () => "/",
      },
    },
  },
  // Relative asset URLs: the bundle is served from a local port by the
  // ``jaato-web-coder-ui`` launcher (``bin/jaato-web-coder-ui.js``) or from any path an
  // operator mounts ``dist/`` under, so ``index.html`` must not assume it
  // sits at ``/``.
  base: "./",
  // "hidden": the map is written for anyone debugging a checkout build but
  // the bundle carries no sourceMappingURL, so the npm package can leave
  // the 1.9 MB map out (package.json `files`) without a dangling reference.
  build: { sourcemap: "hidden" },
  test: {
    pool: "threads",
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
