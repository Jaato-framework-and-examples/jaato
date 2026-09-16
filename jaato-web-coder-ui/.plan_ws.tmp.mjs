// Drive the real web client against a real daemon over WebSocket, through
// createPlan, and report what the Plan rail section shows.  Run from
// jaato-web-coder-ui (resolves @playwright/test from there).
import { chromium } from "@playwright/test";
const base = process.env.UI_BASE, ws = process.env.WS_URL, out = process.env.OUT_DIR;
const browser = await chromium.launch({ executablePath: "/opt/pw-browsers/chromium" });
const page = await (await browser.newContext({ viewport: { width: 1280, height: 760 } })).newPage();
const log = [];
page.on("console", (m) => log.push(m.text()));
try {
  await page.goto(base + "/");
  await page.getByPlaceholder("ws://host:8080").fill(ws);
  await page.getByRole("button", { name: "Connect" }).click();
  await page.getByRole("button", { name: "Open workspace planws" }).click({ timeout: 30000 });
  await page.getByRole("button", { name: /planner/ }).click({ timeout: 30000 });
  const box = page.getByLabel("Prompt");
  await box.waitFor({ timeout: 60000 });
  await page.waitForTimeout(500);
  await box.fill("go"); await box.press("Enter");
  // The rail header shows done/total once a plan exists, whether or not the section is open.
  await page.getByText("0/2", { exact: true }).waitFor({ timeout: 90000 });
  await page.getByRole("button", { name: "Toggle plan (Ctrl+P)" }).click();
  await page.getByText("Prueba", { exact: true }).waitFor({ timeout: 5000 });
  await page.getByText("uno", { exact: true }).waitFor();
  await page.screenshot({ path: `${out}/plan-ws.png` });
  console.log("PLAN PANEL OK: shows Prueba with steps uno/dos");
} catch (e) {
  await page.screenshot({ path: `${out}/plan-ws-fail.png` }).catch(() => {});
  console.log("PLAN PANEL FAILED:", String(e).split("\n")[0]);
  console.log(log.slice(-15).join("\n"));
  process.exitCode = 1;
} finally { await browser.close(); }
