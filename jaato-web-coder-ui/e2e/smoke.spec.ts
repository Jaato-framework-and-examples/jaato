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
  await box.pressSequentially("mock-2");
  // Assert the ARGUMENT, not just the hint.  "runs command" is already
  // visible from the bare "model ", so it says nothing about whether the
  // argument arrived -- it cannot separate a scrambled line from a good
  // one, and the failure then reads as a missing message rather than as
  // the mistyped command it is.
  await expect(box).toHaveValue("model mock-2");
  await expect(page.getByText("runs command")).toBeVisible();
  await box.press("Enter");
  await expect(page.getByText("Model switched to mock-2")).toBeVisible();
});

test("accepting a completion never moves the caret into what is typed next", async ({ page }) => {
  // Accepting a completion sets the text now and has to place the caret
  // after it; doing that from a requestAnimationFrame callback left a
  // window one frame wide in which the user is already typing the
  // argument.  When the frame landed mid-word the caret jumped back to
  // the end of the completed word and the rest of the argument was
  // inserted there -- "model " + "mock-2" submitted as "model ck-2mo",
  // so the daemon switched to a model nobody asked for.
  //
  // Occasional in CI (it needs the frame to land between two
  // keystrokes), deterministic here: rAF is delayed past the typing so
  // the bad frame is guaranteed to land in the middle.  Against a caret
  // applied in a layout effect there is no frame to miss and the delay
  // is inert, which is the point -- this fails only if the scheduling
  // goes back.
  await page.addInitScript(() => {
    const real = window.requestAnimationFrame.bind(window);
    window.requestAnimationFrame = ((cb: FrameRequestCallback) =>
      real(() => window.setTimeout(() => cb(performance.now()), 250))) as typeof window.requestAnimationFrame;
  });
  await openSession(page);
  const box = composer(page);
  await box.fill("mo");
  await expect(page.getByRole("listbox", { name: "Command proposals" })).toBeVisible();
  await box.press("Tab");
  await expect(box).toHaveValue("model ");
  await box.pressSequentially("mock-2", { delay: 100 });
  await expect(box).toHaveValue("model mock-2");
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

test("a permission ASK with no prompt content falls back to the tool arguments", async ({ page }) => {
  // A tool whose plugin renders no display info: prompt_lines and warnings are
  // null on the wire.  The card must still say what is being asked.
  await openSession(page);
  await composer(page).fill("permit-bare");
  await composer(page).press("Enter");
  await expect(page.getByText("Permission requested for")).toBeVisible();
  await expect(page.getByRole("group", { name: /Permission request/ }).getByText("src/app.py")).toBeVisible();
  await expect(page.locator(".diff-add")).toHaveCount(0);
  await page.getByRole("button", { name: /^y yes$/ }).click();
  await expect(page.getByText("Written (you answered")).toBeVisible();
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

test("a launcher config.json pre-fills the form and connects on its own", async ({ page }) => {
  // The Vite dev server answers /config.json with index.html; stand in for
  // ``bin/jaato-web-coder-ui.js`` by serving what it would.
  await page.route("**/config.json", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ daemon: WS, autoConnect: true }) }),
  );
  await page.goto("/");
  await expect(page.getByRole("button", { name: /default/ })).toBeVisible();
  await page.getByRole("button", { name: /default/ }).click();
  await expect(page.getByText("Connected to the mock daemon")).toBeVisible();
});

test("a ticketUrl config mints a fresh ticket per connection and connects (#1074)", async ({ page }) => {
  let minted = 0;
  await page.route("**/config.json", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ daemon: WS, ticketUrl: "/api/ticket", autoConnect: true }) }),
  );
  await page.route("**/api/ticket", (route) => {
    expect(route.request().method()).toBe("POST");
    minted += 1;
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ticket: `t-${minted}` }) });
  });
  await page.goto("/");
  await expect(page.getByRole("button", { name: /default/ })).toBeVisible();
  expect(minted).toBe(1);
  // We are past the welcome screen: the credential was the backend's to mint.
  await expect(page.getByText(/Sign in to open your coding environment/)).toHaveCount(0);
});

test("a 401 from the ticket endpoint offers Sign in instead of an error", async ({ page }) => {
  await page.route("**/config.json", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ daemon: WS, ticketUrl: "/api/ticket", loginUrl: "/auth/login", autoConnect: true }) }),
  );
  await page.route("**/api/ticket", (route) => route.fulfill({ status: 401, contentType: "application/json", body: "{}" }));
  await page.goto("/");
  const signIn = page.getByRole("link", { name: "Sign in" });
  await expect(signIn).toBeVisible();
  await expect(signIn).toHaveAttribute("href", "/auth/login");
  await expect(page.getByText(/Sign in to open your coding environment/)).toBeVisible();
  // Nothing to type: no token field, no daemon address in the way of the one thing to do.
  await expect(page.getByLabel(/Bearer token/)).toHaveCount(0);
  await expect(page.getByPlaceholder("ws://host:8080")).toHaveCount(0);
});

// ── Sign in first, as the TUI allows ────────────────────────────────────

test("sign in from the picker with no session: the daemon's auth.setup offer opens the session", async ({ page }) => {
  await page.goto("/");
  await page.getByPlaceholder("ws://host:8080").fill(WS);
  await page.getByRole("button", { name: "Connect" }).click();
  // The picker lists the daemon's auth commands beside the profiles.
  await page.getByRole("button", { name: "mock-auth login" }).click();
  await expect(page.getByText("Authenticated as tester@example.com")).toBeVisible();
  const card = page.getByRole("group", { name: "Post-auth setup" });
  await expect(card.getByText("Signed in to Mock Provider. Open a session with it?")).toBeVisible();
  // No workspace on this daemon → nothing to persist to, so no checkbox.
  await expect(card.getByRole("checkbox")).toHaveCount(0);
  await card.getByLabel("Model").selectOption("mock-2");
  await card.getByRole("button", { name: "Open session" }).click();
  await expect(page.getByText("Session created with mock / mock-2")).toBeVisible();
  // And the session is live: the prompt works.
  await composer(page).fill("code");
  await composer(page).press("Enter");
  await expect(page.locator("table.j-table th", { hasText: "Latency" })).toBeVisible();
});

test("declining the auth.setup offer leaves the prompt usable with no session", async ({ page }) => {
  await page.goto("/");
  await page.getByPlaceholder("ws://host:8080").fill(WS);
  await page.getByRole("button", { name: "Connect" }).click();
  await page.getByRole("button", { name: "mock-auth login" }).click();
  const card = page.getByRole("group", { name: "Post-auth setup" });
  await card.getByRole("button", { name: "Not now" }).click();
  await expect(card).toHaveCount(0);
  await expect(page.getByText("No model selected, skipping session setup.")).toBeVisible();
  // Daemon commands still run with no session, as in the TUI.
  await composer(page).fill("mock-auth status");
  await composer(page).press("Enter");
  await expect(page.getByText("mock-auth: login | logout | status")).toBeVisible();
});

test("workspace mode: an unconfigured workspace opens straight into the session picker", async ({ page }) => {
  await page.goto("/");
  await page.getByPlaceholder("ws://host:8080").fill("ws://127.0.0.1:8098");
  await page.getByRole("button", { name: "Connect" }).click();
  await expect(page.getByText("Workspaces", { exact: true })).toBeVisible();
  await expect(page.getByText("no provider in .env")).toBeVisible();
  // Not a provider form: the picker, exactly as for a configured one.
  await page.getByRole("button", { name: "Open workspace project-b" }).click();
  await expect(page.getByText("New session")).toBeVisible();
  await page.getByRole("button", { name: "mock-auth login" }).click();
  const card = page.getByRole("group", { name: "Post-auth setup" });
  // A workspace exists here, so the offer can persist to its .env, on by default.
  await expect(card.getByRole("checkbox")).toBeChecked();
  await card.getByRole("button", { name: "Open session" }).click();
  await expect(page.getByText("Saved JAATO_PROVIDER=mock and MODEL_NAME=mock-1 to .env")).toBeVisible();
  await expect(page.getByText("Session created with mock / mock-1")).toBeVisible();
});

test("workspace mode: the manual provider form is a disclosure, not a gate", async ({ page }) => {
  await page.goto("/");
  await page.getByPlaceholder("ws://host:8080").fill("ws://127.0.0.1:8098");
  await page.getByRole("button", { name: "Connect" }).click();
  await page.getByRole("button", { name: "Configure workspace project-b" }).click();
  const form = page.getByRole("group", { name: "Manual provider configuration" }).or(page.getByLabel("Manual provider configuration"));
  await expect(form.getByText("missing: provider, api_key")).toBeVisible();
  await expect(form.locator("select option")).toHaveCount(4); // — + the daemon's three
  await form.getByRole("button", { name: "Open session →" }).click();
  await expect(page.getByText("New session")).toBeVisible();
});
