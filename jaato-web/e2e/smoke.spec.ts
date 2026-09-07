import { expect, test, type Page } from "@playwright/test";

const WS = "ws://127.0.0.1:8097";

async function openSession(page: Page) {
  await page.goto("/");
  await page.getByPlaceholder("ws://host:8080").fill(WS);
  await page.getByRole("button", { name: "Connect" }).click();
  await page.getByRole("button", { name: /default/ }).click();
  await expect(page.getByText("Connected to the mock daemon")).toBeVisible();
}

const composer = (page: Page) => page.getByLabel("Prompt");

test("connects, creates a session, streams a reply with code and a table", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("code");
  await composer(page).press("Enter");
  await expect(page.locator(".tok-keyword", { hasText: "def" })).toBeVisible();
  await expect(page.locator("table.j-table th", { hasText: "Latency" })).toBeVisible();
  // Entities inside tokens are unescaped, tags never leak as text.
  await expect(page.locator(".code-block", { hasText: "<3" })).toBeVisible();
  await expect(page.getByText("<j-code")).toHaveCount(0);
});

test("bare-word commands: proposal, Tab completion, Enter runs the command", async ({ page }) => {
  await openSession(page);
  const box = composer(page);
  await box.fill("mo");
  const listbox = page.getByRole("listbox", { name: "Command proposals" });
  await expect(listbox).toBeVisible();
  await expect(listbox.getByRole("option", { name: /model/ })).toBeVisible();
  await box.press("Tab");
  await expect(box).toHaveValue("model ");
  await box.type("mock-2");
  await expect(page.getByText("runs command")).toBeVisible();
  await box.press("Enter");
  await expect(page.getByText("Model switched to mock-2")).toBeVisible();
});

test("Escape on the proposal sends the word verbatim as a message", async ({ page }) => {
  await openSession(page);
  const box = composer(page);
  await box.fill("model");
  await expect(page.getByRole("listbox", { name: "Command proposals" })).toBeVisible();
  await box.press("Escape");
  await expect(page.getByRole("listbox", { name: "Command proposals" })).toHaveCount(0);
  await box.type(" this is broken");
  await expect(page.getByText("Sending as text")).toBeVisible();
  await box.press("Enter");
  // The mock echoes messages; a command would have said "Model switched".
  await expect(page.getByText("You said:")).toBeVisible();
  await expect(page.getByText("Model switched")).toHaveCount(0);
});

test("a non-command sentence that starts with a command word still routes as the TUI does", async ({ page }) => {
  await openSession(page);
  const box = composer(page);
  await box.fill("help me write a test");
  await expect(page.getByText("runs command")).toHaveCount(0);
  await box.press("Enter");
  await expect(page.getByText("You said:")).toBeVisible();
});

test("tool calls stream into a collapsible block and update the plan panel", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("tool");
  await composer(page).press("Enter");
  const block = page.getByRole("button", { name: /run_command/ }).first();
  await expect(block).toBeVisible();
  await expect(page.getByText("README.md")).toBeVisible();
  await page.getByRole("button", { name: "Toggle plan (Ctrl+P)" }).click();
  await expect(page.getByText("Task plan")).toBeVisible();
  await expect(page.getByText("List the directory")).toBeVisible();
});

test("permission prompt shows the diff and the typed key answers it", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("permit");
  await composer(page).press("Enter");
  await expect(page.getByText("Permission requested for")).toBeVisible();
  await expect(page.locator(".diff-add", { hasText: "print('hi')" })).toBeVisible();
  await expect(page.getByText("outside the sandbox")).toBeVisible();
  await composer(page).fill("y");
  await composer(page).press("Enter");
  await expect(page.getByText("Written (you answered")).toBeVisible();
  await page.getByRole("button", { name: "Toggle workspace changes (Alt+W)" }).click();
  await expect(page.getByText("~ app.py")).toBeVisible();
});

test("batch clarification walks its questions and replies once", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("ask");
  await composer(page).press("Enter");
  await expect(page.getByText("Which framework should the client use?")).toBeVisible();
  await page.getByRole("button", { name: /Svelte 5/ }).click();
  await expect(page.getByText("Anything else I should know?")).toBeVisible();
  await composer(page).fill("nothing");
  await composer(page).press("Enter");
  await expect(page.locator("p", { hasText: /you chose/ })).toContainText("Svelte 5");
});

test("subagents get their own tab", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("subagent");
  await composer(page).press("Enter");
  const tab = page.getByRole("tab", { name: /researcher/ });
  await expect(tab).toBeVisible();
  await tab.click();
  await expect(page.getByRole("heading", { name: "Research notes" })).toBeVisible();
});
