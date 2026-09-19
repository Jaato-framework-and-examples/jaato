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
  // The TUI's Ctrl+T: the status-bar toggle expands every tool block and
  // collapses them again, instead of flipping a flag nothing reads.
  const toolBlock = page.locator("[data-testid=tool-block] [aria-expanded]").first();
  await page.getByRole("button", { name: "Toggle tool call boxes (Ctrl+T)" }).click();
  await expect(toolBlock).toHaveAttribute("aria-expanded", "true");
  await page.getByRole("button", { name: "Toggle tool call boxes (Ctrl+T)" }).click();
  await expect(toolBlock).toHaveAttribute("aria-expanded", "false");
});

test("the status bar shows the permission default policy the daemon reports", async ({ page }) => {
  await openSession(page);
  await expect(page.getByTestId("permission-status")).toHaveText(/permissions\s+ask/);
});

test("`session list` prints the daemon's listing; `session attach` completes ids and replays the conversation", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("session list");
  await composer(page).press("Enter");
  await expect(page.getByText("▶ current  ● loaded  ○ on disk")).toBeVisible();
  await expect(page.getByText(/● 20260916_090000 - fix the budget panel \[anthropic\/claude-sonnet-4\]/)).toBeVisible();

  // Third-level completion: the ids the daemon listed, filtered as you type.
  await composer(page).fill("session attach 2026091");
  const listbox = page.getByRole("listbox", { name: "Command proposals" });
  await expect(listbox.getByRole("option", { name: /20260916_090000/ })).toBeVisible();
  await expect(listbox.getByRole("option", { name: /20260915_170000/ })).toBeVisible();
  await composer(page).fill("session attach 20260916");
  await expect(listbox.getByRole("option")).toHaveCount(1);
  await composer(page).press("Tab");
  await expect(composer(page)).toHaveValue("session attach 20260916_090000 ");
  await composer(page).press("Enter");

  // The pane is the attached session's: its conversation, its permission policy.
  await expect(page.getByText("what are those [object Object] in the budget panel?")).toBeVisible();
  await expect(page.getByText("The panel reads function_calls as a number; it is a list of records.")).toBeVisible();
  await expect(page.getByText("Connected to the mock daemon")).toHaveCount(0);
  await expect(page.getByTestId("permission-status")).toHaveText(/permissions\s+allow/);
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
  // The daemon's key is ``status``: a created file renders as ``+``.
  await expect(page.getByText("+ session.log")).toBeVisible();
});

test("files panel: hide drops an entry from the view, show-hidden brings it back, ignore toggles .gitignore", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("permit");
  await composer(page).press("Enter");
  await expect(page.getByText("Permission requested for")).toBeVisible();
  await composer(page).fill("y");
  await composer(page).press("Enter");
  await expect(page.getByText("Written (you answered")).toBeVisible();
  await page.getByRole("button", { name: "Toggle workspace changes (Alt+W)" }).click();
  const panel = page.getByRole("region", { name: "Files" });
  await expect(panel.getByText("~ app.py")).toBeVisible();

  // hide (the TUI's ``h``): client-side, the entry leaves the view.
  await panel.getByRole("button", { name: "Hide src/app.py", exact: true }).click();
  await expect(panel.getByText("~ app.py")).toHaveCount(0);
  await expect(panel.getByText("1 hidden")).toBeVisible();
  // a whole directory hides everything under it
  await panel.getByRole("button", { name: "Hide .jaato/", exact: true }).click();
  await expect(panel.getByText("+ session.log")).toHaveCount(0);
  await expect(panel.getByText("2 hidden")).toBeVisible();

  // show hidden: back, dimmed, with the H marker, and unhide works.
  await panel.getByRole("button", { name: "show hidden" }).click();
  await expect(panel.getByText("~ app.py")).toBeVisible();
  await expect(panel.locator("[data-hidden]")).toHaveCount(4); // .jaato/, logs/, session.log, app.py
  await panel.getByRole("button", { name: "Unhide src/app.py", exact: true }).click();
  await panel.getByRole("button", { name: "hide hidden" }).click();
  await expect(panel.getByText("~ app.py")).toBeVisible();

  // ignore (the TUI's ``i``): through the daemon, whose answer is the notice.
  await panel.getByRole("button", { name: "Add src/app.py to .gitignore" }).click();
  await expect(panel.getByRole("status")).toHaveText("src/app.py added to .gitignore");
  await panel.getByRole("button", { name: "Remove src/app.py from .gitignore" }).click();
  await expect(panel.getByRole("status")).toHaveText("src/app.py removed from .gitignore");
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
  await page.getByRole("button", { name: /^yes y$/ }).click();
  await expect(page.getByText("Written (you answered")).toBeVisible();
});

test("the agent tab says what that agent is doing, including a prompt waiting on you", async ({ page }) => {
  // The glyph is the whole point of a tab you are NOT looking at, and it
  // is keyed on the daemon's own status vocabulary (active | idle | done |
  // error | cancelled).  It used to key on four words no daemon emits, so
  // every tab read "idle" whatever the agent was doing -- and the mock
  // emitted one of those invented words, which is why this suite was green.
  await openSession(page);
  const main = page.getByRole("tab", { name: "main" });
  await expect(main).toHaveAttribute("title", /idle/);
  await composer(page).fill("permit");
  await composer(page).press("Enter");
  await expect(main).toHaveAttribute("title", /Waiting for you/);
  await page.getByRole("button", { name: /^yes y$/ }).click();
  await expect(main).toHaveAttribute("title", /idle/);
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

test("workspace mode: a configured workspace reopens with its sessions and no provider or .env question", async ({ page }) => {
  await page.goto("/");
  await page.getByPlaceholder("ws://host:8080").fill("ws://127.0.0.1:8098");
  await page.getByRole("button", { name: "Connect" }).click();
  await page.getByRole("button", { name: "Open workspace project-a" }).click();
  const card = page.getByText("Workspace", { exact: false }).first();
  await expect(card).toBeVisible();
  await expect(page.getByText("anthropic / claude-sonnet-4").first()).toBeVisible();
  // No sign-in row, no .env talk: the workspace already binds a provider.
  await expect(page.getByText("No provider configured yet?")).toHaveCount(0);
  // Its previous session is offered for resuming, and resuming replays it.
  await page.getByRole("button", { name: "Resume session 20260916_090000" }).click();
  await expect(page.getByText("The panel reads function_calls as a number; it is a list of records.")).toBeVisible();
});

test("workspace mode: a workspace is deleted after confirmation, and a refusal is shown", async ({ page }) => {
  await page.goto("/");
  await page.getByPlaceholder("ws://host:8080").fill("ws://127.0.0.1:8098");
  await page.getByRole("button", { name: "Connect" }).click();
  await expect(page.getByRole("button", { name: "Open workspace project-b" })).toBeVisible();
  // The daemon says who owns what it lists.
  await expect(page.getByTitle("Owned by mock:tester")).toBeVisible();
  // Cancel leaves everything as it was.
  await page.getByRole("button", { name: "Delete workspace project-b" }).click();
  await page.getByRole("button", { name: "Cancel delete" }).click();
  await expect(page.getByRole("button", { name: "Open workspace project-b" })).toBeVisible();
  // A refusal (loaded sessions) is reported, and the row stays.
  await page.getByRole("button", { name: "Delete workspace project-a" }).click();
  await page.getByRole("button", { name: "Confirm delete workspace project-a" }).click();
  await expect(page.getByRole("status")).toContainText("has 1 loaded session(s)");
  await expect(page.getByRole("button", { name: "Open workspace project-a" })).toBeVisible();
  // The confirmed delete removes the row.
  await page.getByRole("button", { name: "Delete workspace project-b" }).click();
  await page.getByRole("button", { name: "Confirm delete workspace project-b" }).click();
  await expect(page.getByRole("status")).toHaveText("Workspace project-b deleted");
  await expect(page.getByRole("button", { name: "Open workspace project-b" })).toHaveCount(0);
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
  // The picker, headed by the workspace; unconfigured, so the sign-in row stays.
  await expect(page.getByTestId("session-picker")).toContainText("project-b");
  await expect(page.getByLabel("Sign in to a provider")).toBeVisible();
});

// ── Keys the sign-in backend remembers (the configure form's combobox) ──

const WS_WORKSPACES = "ws://127.0.0.1:8098";

/** A backend with a key store: config.json names credentialsUrl and the routes answer for one stored key. */
async function backendWithKeyStore(page: import("@playwright/test").Page, entries: Array<{ id: string; provider: string; label: string; hint: string }>) {
  const calls: Array<{ method: string; url: string; body?: unknown }> = [];
  await page.route("**/config.json", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ daemon: WS_WORKSPACES, ticketUrl: "/api/ticket", credentialsUrl: "/api/credentials", autoConnect: true }) }),
  );
  await page.route("**/api/ticket", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ticket: "t-1" }) }));
  await page.route("**/api/credentials**", (route) => {
    const req = route.request();
    const url = new URL(req.url());
    const rec = { method: req.method(), url: url.pathname + url.search, body: req.postDataJSON() as unknown };
    calls.push(rec);
    if (req.method() === "GET") {
      const provider = url.searchParams.get("provider");
      return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ entries: entries.filter((e) => e.provider === provider).map((e) => ({ ...e, createdAt: "2026-09-16T10:00:00Z" })) }) });
    }
    if (url.pathname.endsWith("/reveal")) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ secret: "sk-revealed-0000mnop" }) });
    if (req.method() === "DELETE") { entries.splice(0, entries.length, ...entries.filter((e) => !url.pathname.endsWith(`/${e.id}`))); return route.fulfill({ status: 204 }); }
    const body = req.postDataJSON() as { provider: string; label?: string; secret: string };
    const entry = { id: "new-1", provider: body.provider, label: body.label ?? `${body.provider} …${body.secret.slice(-4)}`, hint: body.secret.slice(-4) };
    entries.push(entry);
    return route.fulfill({ status: 201, contentType: "application/json", body: JSON.stringify({ entry: { ...entry, createdAt: "2026-09-16T11:00:00Z" } }) });
  });
  return calls;
}

test("a stored key is offered for the provider, preselected, and revealed only when applied", async ({ page }) => {
  const calls = await backendWithKeyStore(page, [{ id: "k1", provider: "anthropic", label: "work", hint: "mnop" }]);
  await page.goto("/");
  await page.getByRole("button", { name: "Configure workspace project-b" }).click();
  const form = page.getByLabel("Manual provider configuration");
  await form.getByLabel("Provider").selectOption("anthropic");
  const picker = form.getByLabel("API key");
  // The newest stored key for this provider is the default; the secret never came down.
  await expect(picker).toHaveValue("k1");
  await expect(form.getByRole("option", { name: "work (…mnop)" })).toHaveCount(1);
  expect(calls.some((c) => c.url.includes("/reveal"))).toBe(false);
  // Another provider has no stored key: the list says so and offers a new one.
  await form.getByLabel("Provider").selectOption("openrouter");
  await expect(picker).toHaveValue("");
  await expect(form.getByRole("option", { name: "— no stored key —" })).toHaveCount(1);
  await form.getByLabel("Provider").selectOption("anthropic");
  await expect(picker).toHaveValue("k1");
  // Applying reveals it once and forwards it to the daemon as the api_key.
  await form.getByRole("button", { name: "Save configuration" }).click();
  await expect(page.getByText("configured", { exact: true })).toBeVisible();
  expect(calls.filter((c) => c.url === "/api/credentials/k1/reveal" && c.method === "POST")).toHaveLength(1);
});

test("a new key is stored under a label before it is applied, and a stored one can be forgotten", async ({ page }) => {
  const calls = await backendWithKeyStore(page, [{ id: "k1", provider: "anthropic", label: "work", hint: "mnop" }]);
  await page.goto("/");
  await page.getByRole("button", { name: "Configure workspace project-b" }).click();
  const form = page.getByLabel("Manual provider configuration");
  await form.getByLabel("Provider").selectOption("anthropic");
  await expect(form.getByLabel("API key")).toHaveValue("k1");
  await form.getByRole("button", { name: "Forget stored key work" }).click();
  await expect(form.getByRole("option", { name: "work (…mnop)" })).toHaveCount(0);
  expect(calls.some((c) => c.method === "DELETE" && c.url === "/api/credentials/k1")).toBe(true);
  await form.getByLabel("API key").selectOption("__new__");
  await form.getByLabel("New API key").fill("sk-typed-000000wxyz");
  await form.getByLabel("Key label").fill("personal");
  await form.getByRole("button", { name: "Save configuration" }).click();
  await expect(page.getByText("configured", { exact: true })).toBeVisible();
  const stored = calls.find((c) => c.method === "POST" && c.url === "/api/credentials");
  expect(stored?.body).toEqual({ provider: "anthropic", secret: "sk-typed-000000wxyz", label: "personal" });
  // The typed key is now a stored one: the list carries it and it is the selection.
  await expect(form.getByRole("option", { name: "personal (…wxyz)" })).toHaveCount(1);
  await expect(form.getByLabel("API key")).toHaveValue("new-1");
});

test("without a key store the configure form keeps its plain key field", async ({ page }) => {
  await page.goto("/");
  await page.getByPlaceholder("ws://host:8080").fill(WS_WORKSPACES);
  await page.getByRole("button", { name: "Connect" }).click();
  await page.getByRole("button", { name: "Configure workspace project-b" }).click();
  const form = page.getByLabel("Manual provider configuration");
  await expect(form.getByLabel("API key")).toHaveAttribute("type", "password");
  await expect(form.getByTestId("credential-picker")).toHaveCount(0);
});

// ── Leaving: Exit in the chat, Sign out on the workspace list ─────────

const EXIT = "Exit (detach from or end the session)";

test("the status bar's Exit asks first; Detach leaves like the exit command, and an autoConnect page does not connect straight back", async ({ page }) => {
  await page.route("**/config.json", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ daemon: WS, autoConnect: true }) }),
  );
  await page.goto("/");
  await page.getByRole("button", { name: /default/ }).click();
  await expect(page.getByText("Connected to the mock daemon")).toBeVisible();
  await page.getByRole("button", { name: EXIT }).click();
  // The TUI's question, with its idle option set, and the composer captures the answer.
  const plate = page.getByRole("group", { name: "Exit options" });
  await expect(plate).toBeVisible();
  await expect(plate.getByRole("button")).toHaveText([/Detach\s+d/, /End session\s+e/, /Return\s+r/]);
  await expect(composer(page)).toHaveAttribute("placeholder", /Exit: d · e · r/);
  await plate.getByRole("button", { name: /Detach/ }).click();
  // Back on the connect screen, and staying there: the page waits for a click.
  const open = page.getByRole("button", { name: "Open my environment" });
  await expect(open).toBeVisible();
  await page.waitForTimeout(700);
  await expect(open).toBeVisible();
  await expect(page.getByRole("button", { name: /default/ })).toHaveCount(0);
  // The click reconnects (the mark was spent); a fresh load would have connected on its own again.
  await open.click();
  await expect(open).toHaveCount(0);
  await expect(page.getByRole("button", { name: EXIT }).or(page.getByTestId("session-picker"))).toBeVisible();
});

test("a typed r (or Escape) returns to the session; the exit command opens the same question", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("exit");
  await composer(page).press("Enter");
  const plate = page.getByRole("group", { name: "Exit options" });
  await expect(plate).toBeVisible();
  await composer(page).fill("r");
  await composer(page).press("Enter");
  await expect(plate).toHaveCount(0);
  await expect(page.getByText("Returning to session.")).toBeVisible();
  // Still in the session: the prompt works as before.
  await composer(page).fill("code");
  await composer(page).press("Enter");
  await expect(page.locator(".tok-keyword", { hasText: "def" })).toBeVisible();
  await page.getByRole("button", { name: EXIT }).click();
  await expect(plate).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(plate).toHaveCount(0);
});

test("with a turn in flight the question offers Cancel task and exit first, and it stops the turn before leaving", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("hang");
  await composer(page).press("Enter");
  await expect(page.getByRole("status")).toContainText("Thinking");
  await page.getByRole("button", { name: EXIT }).click();
  const plate = page.getByRole("group", { name: "Exit options" });
  await expect(plate.getByText("Task in progress")).toBeVisible();
  await expect(plate.getByRole("button")).toHaveText([/Cancel task and exit\s+c/, /Detach\s+d/, /End session\s+e/, /Return\s+r/]);
  await composer(page).fill("c");
  await composer(page).press("Enter");
  await expect(page.getByRole("button", { name: "Connect" })).toBeVisible();
});

test("End session deletes the session and, in workspace mode, lands on the workspace list with the workspace still there", async ({ page }) => {
  await page.goto("/");
  await page.getByPlaceholder("ws://host:8080").fill(WS_WORKSPACES);
  await page.getByRole("button", { name: "Connect" }).click();
  await page.getByRole("button", { name: "Open workspace project-b" }).click();
  await page.getByRole("button", { name: /default/ }).click();
  await expect(page.getByText("Connected to the mock daemon")).toBeVisible();
  await page.getByRole("button", { name: EXIT }).click();
  await page.getByRole("group", { name: "Exit options" }).getByRole("button", { name: /End session/ }).click();
  // Not the connect screen: the connection is kept and the list is where the
  // workspace, which the deletion never touched, is picked again.
  await expect(page.getByText("Workspaces", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open workspace project-b" })).toBeVisible();
  // ``exact``: the list's own way out is "Disconnect", which a substring match would count.
  await expect(page.getByRole("button", { name: "Connect", exact: true })).toHaveCount(0);
});

test("End session on a single-workspace daemon disconnects like Detach", async ({ page }) => {
  await openSession(page);
  await page.getByRole("button", { name: EXIT }).click();
  await composer(page).fill("e");
  await composer(page).press("Enter");
  await expect(page.getByRole("button", { name: "Connect" })).toBeVisible();
});

test("a new session starts on an empty pane", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("fail");
  await composer(page).press("Enter");
  await expect(page.getByText("The command failed; see the tool block.")).toBeVisible();
  // Leave and open another one.  What the session just left said must not
  // become the top of the next one's transcript: the errors of a failed
  // attempt above a "Session created" line read as the new session's own.
  await page.getByRole("button", { name: EXIT }).click();
  await page.getByRole("group", { name: "Exit options" }).getByRole("button", { name: /Detach/ }).click();
  await page.getByRole("button", { name: "Connect" }).click();
  // The picker, not the transcript of the session just detached from: the
  // connection that held it is gone, so this client holds no session.
  await expect(page.getByTestId("session-picker")).toBeVisible();
  await expect(page.getByText("The command failed; see the tool block.")).toHaveCount(0);
  await page.getByRole("button", { name: /default/ }).click();
  await expect(page.getByText("Connected to the mock daemon")).toBeVisible();
  await expect(page.getByRole("button", { name: /run_command/ })).toHaveCount(0);
});

test("a reconnect re-selects the workspace, so a file attached after it still lands", async ({ page }) => {
  await page.goto("/");
  await page.getByPlaceholder("ws://host:8080").fill(WS_WORKSPACES);
  await page.getByRole("button", { name: "Connect" }).click();
  await page.getByRole("button", { name: "Open workspace project-a" }).click();
  await page.getByRole("button", { name: /default/ }).click();
  await expect(page.getByText("Connected to the mock daemon")).toBeVisible();
  // The socket drops.  The daemon forgets this connection's workspace and
  // detaches its session; the SDK reconnects as a new client, and the store
  // still names both -- which is what used to refuse the next staged file
  // with ``No workspace selected for client …``.
  await composer(page).fill("mock-drop");
  await composer(page).press("Enter");
  await expect(page.getByText("Dropping the connection.")).toBeVisible();
  await expect(page.getByText(/^reconnecting/)).toBeVisible();
  await expect(page.getByText("connected", { exact: true })).toBeVisible();
  await page.getByLabel("Attach files").setInputFiles([{ name: "after.txt", mimeType: "text/plain", buffer: Buffer.from("x") }]);
  await expect(page.getByText("Staged into the workspace: after.txt")).toBeVisible();
});

test("the workspace list says who is signed in and offers the backend's Sign out", async ({ page }) => {
  await page.route("**/config.json", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ daemon: WS_WORKSPACES, ticketUrl: "/api/ticket", autoConnect: true }) }),
  );
  await page.route("**/api/ticket", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ticket: "t-1" }) }));
  await page.route("**/api/session", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ user: "alice" }) }));
  await page.goto("/");
  await expect(page.getByText("Workspaces", { exact: true })).toBeVisible();
  await expect(page.getByText("Signed in as")).toBeVisible();
  await expect(page.getByText("alice", { exact: true })).toBeVisible();
  const signOut = page.getByRole("link", { name: "Sign out" });
  await expect(signOut).toHaveAttribute("href", "/api/logout");
  // No backend: nothing to sign out of, so the way out is the connection itself.
  await expect(page.getByRole("button", { name: "Disconnect" })).toHaveCount(0);
});

test("without a backend the workspace list offers Disconnect, which is the exit command", async ({ page }) => {
  await page.goto("/");
  await page.getByPlaceholder("ws://host:8080").fill(WS_WORKSPACES);
  await page.getByRole("button", { name: "Connect" }).click();
  await expect(page.getByRole("link", { name: "Sign out" })).toHaveCount(0);
  await page.getByRole("button", { name: "Disconnect" }).click();
  await expect(page.getByRole("button", { name: "Connect" })).toBeVisible();
});

test("the rail's drag handle resizes it, by pointer and by keyboard", async ({ page }) => {
  await openSession(page);
  const rail = page.getByRole("complementary", { name: "Session rail" });
  const handle = page.getByRole("separator", { name: "Resize the session rail" });
  expect(Math.round((await rail.boundingBox())!.width)).toBe(300);
  const box = (await handle.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + 200);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 - 120, box.y + 200, { steps: 6 });
  await page.mouse.up();
  expect(Math.round((await rail.boundingBox())!.width)).toBe(420);
  await handle.focus();
  await page.keyboard.press("ArrowRight");
  expect(Math.round((await rail.boundingBox())!.width)).toBe(404);
  // Remembered per browser: a new session in the same tab reopens at that width.
  await openSession(page);
  expect(Math.round((await page.getByRole("complementary", { name: "Session rail" }).boundingBox())!.width)).toBe(404);
});

test("files attached in the composer are staged into the workspace, listed in Files, and named by the next message", async ({ page }) => {
  await openSession(page);
  // A pick through the strip's hidden input (a drop or a paste reach the same call).
  await page.getByLabel("Attach files").setInputFiles([
    { name: "notes.md", mimeType: "text/markdown", buffer: Buffer.from("# notes\nhello\n") },
    { name: "data.bin", mimeType: "application/octet-stream", buffer: Buffer.from([1, 2, 3, 4]) },
  ]);
  const strip = page.getByRole("group", { name: "Attached files" });
  await expect(strip.getByText("notes.md")).toBeVisible();
  await expect(strip.locator("li[data-status=staged]")).toHaveCount(2);
  await expect(page.getByText("Staged into the workspace: notes.md, data.bin")).toBeVisible();
  // The daemon's file monitor reports them, so the Files panel lists them.
  await page.getByRole("button", { name: "Toggle workspace changes (Alt+W)" }).click();
  const panel = page.getByRole("region", { name: "Files" });
  await expect(panel.getByText("+ notes.md")).toBeVisible();
  await expect(panel.getByText("+ data.bin")).toBeVisible();
  // The next message names them and the strip is cleared.
  await composer(page).fill("summarise the notes");
  await composer(page).press("Enter");
  await expect(page.getByText("Attached files, staged in the workspace: notes.md, data.bin")).toBeVisible();
  await expect(strip).toHaveCount(0);
});

test("a file refused by the daemon shows the daemon's reason on its chip", async ({ page }) => {
  await openSession(page);
  // Over the daemon's per-file cap: refused by the client's precheck with the daemon's own words, and never sent.
  await page.getByLabel("Attach files").setInputFiles([{ name: "big.bin", mimeType: "application/octet-stream", buffer: Buffer.alloc(11 * 1024 * 1024) }]);
  const strip = page.getByRole("group", { name: "Attached files" });
  await expect(strip.locator("li[data-status=failed]")).toHaveCount(1);
  await expect(strip.getByText(/per-file cap/)).toBeVisible();
  await expect(page.getByText(/Staged into the workspace/)).toHaveCount(0);
  // The failed chip goes with the next send, and the prompt gains no footer for it.
  await composer(page).fill("hello");
  await composer(page).press("Enter");
  await expect(strip).toHaveCount(0);
  await expect(page.getByText(/Attached files, staged/)).toHaveCount(0);
});

test("files attached on the session picker are in the workspace when the session opens", async ({ page }) => {
  await page.goto("/");
  await page.getByPlaceholder("ws://host:8080").fill(WS);
  await page.getByRole("button", { name: "Connect" }).click();
  const strip = page.getByRole("group", { name: "Attached files" });
  await expect(strip).toBeVisible();
  await page.getByLabel("Attach files").setInputFiles([{ name: "brief.txt", mimeType: "text/plain", buffer: Buffer.from("do the thing") }]);
  // No workspace yet on this daemon: the file waits for the session.
  await expect(strip.locator("li[data-status=queued]")).toHaveCount(1);
  await page.getByRole("button", { name: /default/ }).click();
  await expect(page.getByText("Connected to the mock daemon")).toBeVisible();
  await expect(page.getByText("Staged into the workspace: brief.txt")).toBeVisible();
  await page.getByRole("button", { name: "Toggle workspace changes (Alt+W)" }).click();
  await expect(page.getByRole("region", { name: "Files" }).getByText("+ brief.txt")).toBeVisible();
});

test("a note written on the exit plate survives Escape, is kept, and is the rail's copy too", async ({ page }) => {
  // The whole loop in a real browser, with no BFF -- which is the shape a
  // local `npx @jaato/web-coder-ui` has, so the store behind it is this
  // browser's and the UI has to SAY so rather than imply a shared one.
  await openSession(page);
  await page.getByRole("button", { name: EXIT }).click();
  const plate = page.getByRole("group", { name: "Exit options" });
  const field = plate.getByLabel("Note to self");
  await field.fill("waiting on the grace period answer, then re-run e2e");
  // Escape inside the field leaves the field, never the session: it used to
  // answer `r` unconditionally and take the half-typed note with it.
  await field.press("Escape");
  await expect(plate).toBeVisible();
  await expect(field).toHaveValue(/grace period/);
  await plate.getByRole("button", { name: /Return/ }).click();
  await expect(plate).toHaveCount(0);

  // Same note, read from the rail -- one store, four mount points.
  await page.getByRole("button", { name: "Toggle your sessions and their notes" }).click();
  const rail = page.getByRole("region", { name: "Sessions" });
  await expect(rail.getByLabel("This session")).toHaveValue(/grace period/);
  await expect(rail.getByText("Notes are kept in this browser only", { exact: false })).toBeVisible();
});
