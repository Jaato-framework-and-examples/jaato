import { existsSync } from "node:fs";
import { defineConfig } from "@playwright/test";

// Prefer an explicitly provided Chromium (CI images and the remote dev
// container ship one under /opt/pw-browsers) over Playwright's own download.
const chromium = process.env.PLAYWRIGHT_CHROMIUM
  ?? (existsSync("/opt/pw-browsers/chromium") ? "/opt/pw-browsers/chromium" : undefined);

/**
 * End-to-end smoke against the scripted mock daemon (``mock/daemon.ts``)
 * and the Vite dev server.  Both are started here; the page connects
 * straight to the mock's port so no proxy is involved.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:5199",
    headless: true,
    launchOptions: chromium ? { executablePath: chromium } : {},
  },
  webServer: [
    { command: "MOCK_PORT=8097 MOCK_SPEED=0 node --import tsx mock/daemon.ts", port: 8097, reuseExistingServer: false, timeout: 30_000 },
    { command: "./node_modules/.bin/vite --port 5199 --strictPort", port: 5199, reuseExistingServer: false, timeout: 60_000 },
  ],
});
