// Open the spike page in headless Chromium and print the worker's log.
// Usage: node drive.mjs [http://127.0.0.1:8766]
// Needs Playwright; set PLAYWRIGHT_MODULE / CHROMIUM_PATH to override.
const mod = process.env.PLAYWRIGHT_MODULE || "playwright";
const { chromium } = await import(mod);
const opts = process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {};
const browser = await chromium.launch(opts);
const page = await browser.newPage();
await page.goto((process.argv[2] || "http://127.0.0.1:8766") + "/web/index.html" + (process.env.QS || ""));
await page.waitForFunction(() => window.__done, null, { timeout: 600000, polling: 1000 });
for (const line of await page.evaluate(() => window.__logs)) console.log(line);
await browser.close();
