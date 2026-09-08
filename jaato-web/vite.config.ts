/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath } from "node:url";

// The SDK is consumed from its TypeScript source so the client never
// depends on a prebuilt ``dist/`` — ``npm run build`` in jaato-web is
// self-sufficient, and a protocol change in jaato-sdk-ts is picked up
// on the next HMR tick.
const sdkSrc = fileURLToPath(new URL("../jaato-sdk-ts/src/index.ts", import.meta.url));

export default defineConfig({
  plugins: [react(), tailwindcss()],
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
  build: { sourcemap: true },
  test: {
    pool: "threads",
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
