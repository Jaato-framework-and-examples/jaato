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
  await page.getByRole("button", { name: "Open Plan" }).click();
  await expect(page.getByText("Task plan")).toBeVisible();
  await expect(page.getByText("List the directory")).toBeVisible();
  // The TUI's Ctrl+T: the status-bar toggle expands every tool block and
  // collapses them again, instead of flipping a flag nothing reads.
  const toolBlock = page.locator("[data-testid=tool-block] [aria-expanded]").first();
  await page.getByRole("button", { name: /Toggle tool call boxes/ }).click();
  await expect(toolBlock).toHaveAttribute("aria-expanded", "true");
  await page.getByRole("button", { name: /Toggle tool call boxes/ }).click();
  await expect(toolBlock).toHaveAttribute("aria-expanded", "false");
});

test("a notebook cell renders as a cell, not as its <nb-row> tags (#1193)", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("run a notebook cell");
  await composer(page).press("Enter");
  await expect(page.getByText("The cell raised a ZeroDivisionError.")).toBeVisible();
  const cell = page.locator("[data-testid=tool-block] .nb-cells");
  await expect(cell).toBeVisible();
  await expect(cell.locator(".nb-label")).toHaveText(["In [1]:", "Out [1]:", "Err [1]:"]);
  await expect(cell.locator('.nb-row[data-nb-type="input"] pre.code-block')).toContainText("1/0");
  await expect(cell.locator('.nb-row[data-nb-type="error"] pre.nb-out')).toContainText('File "<cell>", line 2');
  // The reported leak: no wrapper tag reaches the page as text.
  await expect(page.getByText("<nb-row")).toHaveCount(0);
  await expect(page.getByText("</nb-row>")).toHaveCount(0);
});

test("an early-exit notebook error renders as a cell too", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("show an early notebook exit");
  await composer(page).press("Enter");
  const label = page.locator("[data-testid=tool-block] .nb-label");
  await expect(label).toHaveText("Err:");
  await expect(page.locator("[data-testid=tool-block] pre.nb-out")).toHaveText("No code provided");
  await expect(page.getByText("<nb-row")).toHaveCount(0);
});

test("the status bar shows the permission default policy the daemon reports", async ({ page }) => {
  await openSession(page);
  await expect(page.getByTestId("permission-status")).toHaveText(/permissions\s+ask/);
});

test("the permissions plate changes the policy and the readout follows it", async ({ page }) => {
  // The segment is a CONTROL, so the loop is the thing worth asserting:
  // click a default, the daemon applies it and re-emits its status, the
  // bar reads the new one back.  It was broken in the daemon for as long
  // as the segment existed -- it read a copy of the policy that nothing
  // updated, so `permissions default deny` left the bar saying `ask` --
  // and no test here could see it, because the mock was answering in the
  // shape the daemon was SUPPOSED to and the daemon was not.
  await openSession(page);
  await expect(page.getByTestId("permission-status")).toHaveText(/permissions\s+ask/);

  await page.getByTestId("permission-status").click();
  const plate = page.getByRole("dialog", { name: /permission/i });
  await expect(plate).toBeVisible();
  await plate.getByRole("button", { name: /^deny$/i }).click();
  await expect(page.getByTestId("permission-status")).toHaveText(/permissions\s+deny/);

  // Suspension outranks the default in the rendering because it does in
  // the daemon: while prompting is suspended no policy is consulted.
  await page.getByTestId("permission-status").click();
  await page.getByRole("dialog", { name: /permission/i })
    .getByRole("button", { name: /until idle/i }).click();
  await expect(page.getByTestId("permission-status")).toHaveText(/allow\s+\(idle\)/);
});

test("`session list` prints the daemon's listing; `session attach` completes ids and replays the conversation", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("session list");
  await composer(page).press("Enter");
  await expect(page.getByText("▶ current  ● loaded  ○ on disk")).toBeVisible();
  // The waiting marker sits between the description and the model, because
  // #1138 added `awaiting` to the listing precisely so this command answers
  // "which of these wants me" and not only "which of these exist".
  await expect(page.getByText(/● 20260916_090000 - fix the budget panel \[waiting: permission\] \[anthropic\/claude-sonnet-4\]/)).toBeVisible();

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
  await page.getByRole("button", { name: "Open Files" }).click();
  await expect(page.getByText("~ app.py")).toBeVisible();
  // ``.jaato/`` is hidden BY DEFAULT (#1304 §5, a view filter): the panel's
  // own metadata directory does not show without asking for it.
  await expect(page.getByText("+ session.log")).toHaveCount(0);
  await page.getByRole("region", { name: "Files" }).getByRole("button", { name: "show hidden" }).click();
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
  await page.getByRole("button", { name: "Open Files" }).click();
  const panel = page.getByRole("region", { name: "Files" });
  await expect(panel.getByText("~ app.py")).toBeVisible();
  // ``.jaato/`` is hidden BY DEFAULT (#1304 §5) -- a view filter, not
  // ``.gitignore``: it counts toward "hidden" and its row does not even
  // render before "show hidden", with no click needed to get there.
  await expect(panel.getByText("+ session.log")).toHaveCount(0);
  await expect(panel.getByText("1 hidden")).toBeVisible();

  // hide (the TUI's ``h``): client-side, the entry leaves the view -- ON
  // TOP of the default hide, so the count is now two.
  await panel.getByRole("button", { name: "Hide src/app.py", exact: true }).click();
  await expect(panel.getByText("~ app.py")).toHaveCount(0);
  await expect(panel.getByText("2 hidden")).toBeVisible();

  // show hidden: both come back, dimmed, with the H marker, and unhide
  // works for the explicit one.
  await panel.getByRole("button", { name: "show hidden" }).click();
  await expect(panel.getByText("~ app.py")).toBeVisible();
  await expect(panel.getByText("+ session.log")).toBeVisible();
  await expect(panel.locator("[data-hidden]")).toHaveCount(4); // .jaato/, logs/, session.log, app.py
  await panel.getByRole("button", { name: "Unhide src/app.py", exact: true }).click();
  await panel.getByRole("button", { name: "hide hidden" }).click();
  await expect(panel.getByText("~ app.py")).toBeVisible();
  await expect(panel.getByText("+ session.log")).toHaveCount(0);

  // collapse (the TUI's Left/Right): the arrow folds a directory to one
  // line that says how many files it holds, and unfolds it again.
  await panel.getByRole("button", { name: "Collapse src/", exact: true }).click();
  await expect(panel.getByText("~ app.py")).toHaveCount(0);
  await expect(panel.getByRole("button", { name: "Expand src/", exact: true })).toHaveAttribute("aria-expanded", "false");
  await expect(panel.getByRole("button", { name: "Expand src/", exact: true })).toContainText("(1)");
  await panel.getByRole("button", { name: "Expand src/", exact: true }).click();
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

test("the regrouped permission card: 'Allow for…' grants a scoped duration, and a granted write's diff shows inline with Open diff (jaato/#1304 §2, phase 3)", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("permit");
  await composer(page).press("Enter");
  await expect(page.getByText("Permission requested for")).toBeVisible();
  // The risk tag + plain question -- computed from the daemon's own
  // tool_class ("write"), not the client's fallback table.
  await expect(page.getByText("Changes a file")).toBeVisible();
  await expect(page.getByText("Let the agent change this file?")).toBeVisible();
  // Allow / Deny are unchanged -- and the durations are NOT top-level
  // buttons, they live in the dropdown.
  await expect(page.getByRole("button", { name: /^turn t$/ })).toHaveCount(0);
  const dropdown = page.getByRole("button", { name: /Allow for/ });
  await dropdown.click();
  await expect(page.getByRole("menu", { name: /Allow for a scope/ })).toBeVisible();
  await page.getByRole("menuitem", { name: /^turn t$/ }).click();
  await expect(page.getByText("Written (you answered")).toBeVisible();
  // The tool row's own diff (tool.call_end, phase 3) -- always shown for
  // a write, not behind the row's own expand toggle -- with "Open diff"
  // since the mock's diff runs past the client's 6-line preview cap.
  const row = page.getByTestId("tool-block").filter({ hasText: "write_file" });
  await expect(row.locator(".diff-add", { hasText: "print('hi')" })).toBeVisible();
  const openDiff = row.getByRole("button", { name: "Open diff" });
  await expect(openDiff).toBeVisible();
  await expect(row.getByText("# trailer")).toHaveCount(0);
  await openDiff.click();
  await expect(row.getByRole("button", { name: "Collapse diff" })).toBeVisible();
  await expect(row.getByText("# trailer")).toBeVisible();
});

test("the regrouped permission card: a note sent with Allow reaches the transcript as feedback, and hidden options are behind More…", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("permit");
  await composer(page).press("Enter");
  await expect(page.getByText("Permission requested for")).toBeVisible();
  // once/edit are hidden until "More…" is opened.
  await expect(page.getByRole("button", { name: /^once once$/ })).toHaveCount(0);
  await page.getByText("More…").click();
  await expect(page.getByRole("button", { name: /^once once$/ })).toBeVisible();
  await page.getByText("Fewer options").click();
  await expect(page.getByRole("button", { name: /^once once$/ })).toHaveCount(0);
  // A note turns Allow into "yc:<text>" -- feedback the model reads back,
  // not the plain "y" key.
  await page.getByText("+ Add a note").click();
  await page.getByPlaceholder(/the model reads this back/).fill("looks fine, double check the perms");
  await page.getByRole("button", { name: /^yes y$/ }).click();
  await expect(page.getByText(/you answered.*yc:looks fine/)).toBeVisible();
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

test("long clarification choices wrap inside the plate, not off its edge (#1245)", async ({ page }) => {
  // Reported with a screenshot: ~300-char choices rendered as one uppercase
  // non-wrapping line running past the plate, across the transcript and over
  // the rail.  An unlayered ``.btn { white-space: nowrap; text-transform:
  // uppercase }`` outranked the button's ``normal-case`` / ``whitespace-normal``
  // utilities.  Measured, not styled: each choice's box must lie inside the
  // clarification plate AND inside the transcript column, and the text must
  // not be uppercase-transformed.
  await page.setViewportSize({ width: 1280, height: 800 });
  await openSession(page);
  await composer(page).fill("ask long");
  await composer(page).press("Enter");
  await expect(page.getByText("Which framework should the client use?")).toBeVisible();
  const plate = page.getByRole("group", { name: "Clarification" });
  // The plate's footer "cancel" is also a <button>, so filter to the choices.
  const buttons = plate.getByRole("button").filter({ hasNotText: /^cancel$/ });
  const count = await buttons.count();
  expect(count).toBe(3);
  const plateBox = (await plate.boundingBox())!;
  const main = (await page.locator("main").boundingBox())!;
  for (let i = 0; i < count; i++) {
    const b = buttons.nth(i);
    const box = (await b.boundingBox())!;
    // Inside the plate's box (a nowrap line overflows it to the right).
    expect(box.x).toBeGreaterThanOrEqual(plateBox.x - 1);
    expect(box.x + box.width).toBeLessThanOrEqual(plateBox.x + plateBox.width + 1);
    // Inside the transcript column — so it cannot cross onto the rail.
    expect(box.x + box.width).toBeLessThanOrEqual(main.x + main.width + 1);
    // The label is body text, not chrome: no uppercase transform.
    expect(await b.evaluate((el) => getComputedStyle(el).textTransform)).not.toBe("uppercase");
  }
});

test("subagents get their own tab, named by the daemon's agent_name", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("subagent");
  await composer(page).press("Enter");
  const tab = page.getByRole("tab", { name: /researcher/ });
  await expect(tab).toBeVisible();
  await tab.click();
  await expect(page.getByRole("heading", { name: "Research notes" })).toBeVisible();
});

test("one name everywhere (#1304 §4): the parent's own text resolves the raw subagent id to its display name", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("subagent");
  await composer(page).press("Enter");
  // The mock's parent text literally says "(id: sub-XXXXXX)"; the client
  // resolves it, at render time, to the name the tab already carries.
  await expect(page.getByText(/Delegating to a researcher subagent \(id: researcher\)/)).toBeVisible();
  await expect(page.getByText(/sub-[0-9a-f]{6}/)).toHaveCount(0);
});

test("a stalled agent shows amber on its tab and a parent-transcript banner; Cancel stops exactly that agent (#1304 §3, §4)", async ({ page }) => {
  // The real 30s-300s range (``store/phase.ts``) is re-enforced INSIDE
  // ``stalled()`` itself, defensively, so a test cannot shortcut it by
  // writing a smaller value to the store -- the documented default
  // (60s) is exercised for real by fast-forwarding Playwright's clock
  // past it, rather than by fabricating a threshold nothing configures.
  //
  // Installed BEFORE ``openSession``, deliberately: ``AttentionBanners``
  // mounts as soon as the session view does, and its own ``useTick``
  // interval is a plain ``window.setInterval`` created at that mount.  A
  // clock installed only after ``openSession`` leaves that one interval
  // on the REAL, pre-install timer (Playwright's fake-timer swap affects
  // calls made after install, not ones already scheduled) -- so a later
  // ``fastForward`` would advance every OTHER component's tick (each
  // agent tab's own ``useTick`` mounts later, once the subagent exists)
  // while this one keeps waiting on real wall-clock seconds that never
  // arrive inside the test's assertion window.  Installing first means
  // every interval the session view creates is fake-clock-driven alike.
  await page.clock.install();
  await openSession(page);
  await composer(page).fill("stall subagent");
  await composer(page).press("Enter");
  await expect(page.getByText(/Delegated to a background worker/)).toBeVisible();

  const tab = page.getByTestId(/^agent-tab-sub-/);
  await expect(tab).toBeVisible();
  await expect(tab).not.toHaveAttribute("data-stalled");
  await page.clock.fastForward("01:01");
  await expect(tab).toHaveAttribute("data-stalled", "true");
  await expect(tab.getByTestId("agent-tab-stall-clock")).toBeVisible();

  const banner = page.locator('[data-testid^="attention-banner-sub-"]');
  await expect(banner).toBeVisible();
  await expect(banner).toContainText("stalled");
  // Nudge can only ever reach the main agent (see AttentionBanners.tsx's
  // docstring for the finding) -- for a subagent row it is disabled, not
  // silently wired to the wrong target.
  await expect(banner.getByRole("button", { name: "Nudge" })).toBeDisabled();

  await banner.getByRole("button", { name: "Cancel" }).click();
  // The stalled SUBAGENT stops -- not the whole session, and not the
  // main agent, which never took a turn here.
  await expect(page.getByText(/Stopped sub-/)).toBeVisible();
  await expect(tab).not.toHaveAttribute("data-stalled");
  await expect(banner).toHaveCount(0);
  await expect(composer(page)).toBeEnabled();
});

test("the icon rail replaces the accordion: one panel is open at a time, and Plan may be pinned beside it (#1304 §5)", async ({ page }) => {
  await openSession(page);
  await expect(page.getByRole("region", { name: "Plan" })).toHaveCount(0);
  await page.getByRole("button", { name: "Open Plan" }).click();
  await expect(page.getByRole("region", { name: "Plan" })).toBeVisible();
  // Switching to a different badge replaces the panel -- unlike the old
  // accordion, where any number of sections could be open at once.
  await page.getByRole("button", { name: "Open Files" }).click();
  await expect(page.getByRole("region", { name: "Plan" })).toHaveCount(0);
  await expect(page.getByRole("region", { name: "Files" })).toBeVisible();
  // Pinning Plan is the one way to see two panels together.
  await page.getByRole("button", { name: "Open Plan" }).click();
  await page.getByRole("button", { name: "pin", exact: true }).click();
  await page.getByRole("button", { name: "Open Sessions" }).click();
  await expect(page.getByRole("region", { name: "Plan" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Sessions" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Files" })).toHaveCount(0);
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
  // Targeted by ROLE: the chip carried `btn btn-steel` styling as a plain
  // span for two releases, so it looked like this control and was not one.
  await page.getByRole("button", { name: "Attach session 20260916_090000" }).click();
  await expect(page.getByText("The panel reads function_calls as a number; it is a list of records.")).toBeVisible();
});

test("workspace mode: the session picker goes back to the workspace list", async ({ page }) => {
  // Reported from a deployed client: a workspace opened by mistake could
  // only be left by ending a session or leaving the daemon, because
  // WorkspaceScreen routes forward and nothing routed back.
  await page.goto("/");
  await page.getByPlaceholder("ws://host:8080").fill("ws://127.0.0.1:8098");
  await page.getByRole("button", { name: "Connect" }).click();
  await page.getByRole("button", { name: "Open workspace project-a" }).click();
  await expect(page.getByTestId("session-picker")).toBeVisible();

  await page.getByRole("button", { name: "Workspaces" }).click();

  // The list, with both workspaces still there — going back selects nothing
  // and destroys nothing, so the other one is one click away.
  await expect(page.getByRole("button", { name: "Open workspace project-a" })).toBeVisible();
  await page.getByRole("button", { name: "Open workspace project-b" }).click();
  await expect(page.getByTestId("session-picker")).toBeVisible();
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

test("the Files panel's reset shows only later changes, and survives a reconnect (#1189)", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("please touch old.py kept.py");
  await composer(page).press("Enter");
  await expect(page.getByText("Touched old.py, kept.py.")).toBeVisible();
  await page.getByRole("button", { name: "Open Files" }).click();
  const panel = page.getByRole("region", { name: "Files" });
  await expect(panel.getByText("~ old.py")).toBeVisible();

  await panel.getByRole("button", { name: "reset", exact: true }).click();
  await expect(panel.getByText("No files changed since the reset.")).toBeVisible();

  // kept.py was already listed: touching it again is what the reset is for.
  await composer(page).fill("please touch kept.py new.py");
  await composer(page).press("Enter");
  await expect(page.getByText("Touched kept.py, new.py.")).toBeVisible();
  await expect(panel.getByText("~ kept.py")).toBeVisible();
  await expect(panel.getByText("~ new.py")).toBeVisible();
  await expect(panel.getByText("~ old.py")).toHaveCount(0);

  // The reconnect's snapshot replaces the list wholesale; the numbering on
  // it is what lets the reset survive.
  await composer(page).fill("mock-drop");
  await composer(page).press("Enter");
  await expect(page.getByText(/^reconnecting/)).toBeVisible();
  await expect(page.getByText("connected", { exact: true })).toBeVisible();
  await expect(panel.getByText("~ new.py")).toBeVisible();
  await expect(panel.getByText("~ old.py")).toHaveCount(0);

  await panel.getByRole("button", { name: "show everything" }).click();
  await expect(panel.getByText("~ old.py")).toBeVisible();
});

test("a hashed category id in a tool call is shown by its name", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("please discover tools");
  await composer(page).press("Enter");
  await expect(page.getByText("I have a system category.")).toBeVisible();
  // list_tools is a housekeeping call (#1304 phase 1) and folds into a
  // one-line summary; expand it to reach the resolved call detail.
  const fold = page.getByRole("button", { name: /list_tools/ });
  await fold.click();
  const row = page.locator("[data-testid=tool-block]").filter({ hasText: "list_tools" });
  // The mapping arrived after the call: the row resolves when it does.
  await expect(row).toContainText("category_id=system");
  await expect(row).not.toContainText("c_bbc5e661");
});

test("the Instructions panel says when GC last ran, what it freed, and the policy -- to a tab that attached later too (#1190)", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("please collect garbage");
  await composer(page).press("Enter");
  await expect(page.getByText("Collected.")).toBeVisible();
  const budgetToggle = page.getByRole("button", { name: /Open Budget/ });
  if (await budgetToggle.count()) await budgetToggle.click();
  const gc = page.getByTestId("gc-summary");
  // The mock stamps the pass 12 minutes in the past: the panel must show
  // the pass's own time, not the moment the event arrived.
  await expect(gc).toContainText("last GC 12 min ago · freed 14.2k tokens");
  await expect(gc).toContainText("GC: budget · runs at 80% · down to 60%");

  // A second tab attaching to the same session never saw the pass.  Only
  // the daemon's replay can tell it -- a reconnect of THIS tab would not
  // prove that, since it keeps what the tab already knew.
  const sessionId = await page.locator("span[title]").filter({ hasText: /^session / }).first().getAttribute("title");
  expect(sessionId).toBeTruthy();
  const other = await page.context().newPage();
  await other.goto("/");
  await other.getByPlaceholder("ws://host:8080").fill(WS);
  await other.getByRole("button", { name: "Connect" }).click();
  await other.getByRole("button", { name: "Go to the prompt without a session" }).click();
  await composer(other).fill(`session attach ${sessionId}`);
  await composer(other).press("Enter");
  const otherToggle = other.getByRole("button", { name: /Open Budget/ });
  if (await otherToggle.count()) await otherToggle.click();
  await expect(other.getByTestId("gc-summary")).toContainText("last GC 12 min ago · freed 14.2k tokens");
  await expect(other.getByTestId("gc-summary")).toContainText("GC: budget");
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
  expect(Math.round((await rail.boundingBox())!.width)).toBe(400);
  const box = (await handle.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + 200);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 - 120, box.y + 200, { steps: 6 });
  await page.mouse.up();
  expect(Math.round((await rail.boundingBox())!.width)).toBe(520);
  await handle.focus();
  await page.keyboard.press("ArrowRight");
  expect(Math.round((await rail.boundingBox())!.width)).toBe(504);
  // Remembered per browser: a new session in the same tab reopens at that width.
  await openSession(page);
  expect(Math.round((await page.getByRole("complementary", { name: "Session rail" }).boundingBox())!.width)).toBe(504);
});

test("dragging the boundary between two rail sections moves height between them, and survives a reload (#1244)", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await openSession(page);
  // The icon rail shows ONE panel at a time (#1304 §5) except Plan, which
  // may be pinned open beside whichever other panel is active.
  await page.getByRole("button", { name: "Open Plan" }).click();
  await page.getByRole("button", { name: "pin", exact: true }).click();
  await page.getByRole("button", { name: "Open Sessions" }).click();

  const plan = page.getByRole("region", { name: "Plan" });
  const sessions = page.getByRole("region", { name: "Sessions" });
  const rail = page.locator("[data-rail]");
  const handle = page.getByRole("separator", { name: "Resize between Plan and Sessions" });

  const planBefore = (await plan.boundingBox())!;
  const sessionsBefore = (await sessions.boundingBox())!;

  // Measured from the page, not from styles.  Drag the boundary UP, so Plan
  // shrinks and Sessions grows by the same amount.
  const hb = (await handle.boundingBox())!;
  await page.mouse.move(hb.x + hb.width / 2, hb.y + hb.height / 2);
  await page.mouse.down();
  await page.mouse.move(hb.x + hb.width / 2, hb.y + hb.height / 2 - 120, { steps: 6 });
  await page.mouse.up();

  const planAfter = (await plan.boundingBox())!;
  const sessionsAfter = (await sessions.boundingBox())!;
  expect(planAfter.height).toBeLessThan(planBefore.height - 40);
  expect(sessionsAfter.height).toBeGreaterThan(sessionsBefore.height + 40);
  // Split-pane: what one loses the other gains.
  const shrank = planBefore.height - planAfter.height;
  const grew = sessionsAfter.height - sessionsBefore.height;
  expect(Math.abs(shrank - grew)).toBeLessThan(2);

  // The rail as a whole does not scroll: the open sections divide its height.
  expect(await rail.evaluate((el) => el.scrollHeight - el.clientHeight)).toBeLessThanOrEqual(1);

  // Remembered per browser: reload, reopen the two sections, the proportion holds.
  const ratio = planAfter.height / sessionsAfter.height;
  await page.reload();
  await page.getByPlaceholder("ws://host:8080").fill("ws://127.0.0.1:8097");
  await page.getByRole("button", { name: "Connect" }).click();
  await page.getByRole("button", { name: /default/ }).click();
  await expect(page.getByText("Connected to the mock daemon")).toBeVisible();
  // Pin state is per session-screen mount, not remembered across a reload
  // -- only the SPLIT ratio (``ui.railSplits``) is, which is what this
  // test is actually pinning down below.
  await page.getByRole("button", { name: "Open Plan" }).click();
  await page.getByRole("button", { name: "pin", exact: true }).click();
  await page.getByRole("button", { name: "Open Sessions" }).click();
  const planReload = (await page.getByRole("region", { name: "Plan" }).boundingBox())!;
  const sessionsReload = (await page.getByRole("region", { name: "Sessions" }).boundingBox())!;
  expect(planReload.height / sessionsReload.height).toBeCloseTo(ratio, 1);
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
  await page.getByRole("button", { name: "Open Files" }).click();
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

// The reported PDF: 1.4 MB.  Before the daemon set a message limit,
// ``websockets`` applied 1 MiB and closed the connection on this file's
// binary frame, and the chip spun for two minutes.  The mock enforces and
// advertises the daemon's limit, so this stages or fails loudly.
test("a file over 1 MiB stages instead of hanging", async ({ page }) => {
  await openSession(page);
  await page.getByLabel("Attach files").setInputFiles([{ name: "report.pdf", mimeType: "application/pdf", buffer: Buffer.alloc(1_400_000, 7) }]);
  const strip = page.getByRole("group", { name: "Attached files" });
  await expect(strip.locator("li[data-status=staged]")).toHaveCount(1);
  await expect(page.getByText("Staged into the workspace: report.pdf")).toBeVisible();
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
  await page.getByRole("button", { name: "Open Files" }).click();
  await expect(page.getByRole("region", { name: "Files" }).getByText("+ brief.txt")).toBeVisible();
});

test("an attached file does not follow you into another session (#1250)", async ({ page }) => {
  // The reported bug: uploads were one flat global list, so a file attached
  // in session A sat above session B's composer too.  They are scoped to the
  // session now, so switching away hides them.
  await openSession(page);
  await page.getByLabel("Attach files").setInputFiles([
    { name: "switchme.txt", mimeType: "text/plain", buffer: Buffer.from("mine only") },
  ]);
  const strip = page.getByRole("group", { name: "Attached files" });
  await expect(strip.getByText("switchme.txt")).toBeVisible();
  await expect(strip.locator("li[data-status=staged]")).toHaveCount(1);

  // Switch to an unrelated session: the file belongs to the one it was
  // attached in, so the strip in this one does not show it.
  await composer(page).fill("session attach 20260916_090000");
  await composer(page).press("Enter");
  await expect(page.getByText("switchme.txt")).toHaveCount(0);
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
  await page.getByRole("button", { name: "Open Sessions" }).click();
  const rail = page.getByRole("region", { name: "Sessions" });
  await expect(rail.getByLabel("This session")).toHaveValue(/grace period/);
  await expect(rail.getByText("Notes are kept in this browser only", { exact: false })).toBeVisible();
});

test("a session blocked on a person says so in the rail, from another session", async ({ page }) => {
  // The one fact the rail exists for that a note cannot supply: prompt
  // events reach only that session's attached clients, so working in one
  // session is exactly when you cannot otherwise learn another wants you.
  await openSession(page);
  await page.getByRole("button", { name: "Open Sessions" }).click();
  const rail = page.getByRole("region", { name: "Sessions" });
  await expect(rail.getByText(/waiting 4 min: permission/)).toBeVisible();
  // And the header counts what needs a person ahead of what carries a note.
  await expect(page.getByText("1 waiting on you")).toBeVisible();
});

test("deleting a session forgets the note written about it", async ({ page }) => {
  // The reported state: the rail offered `✎ …` under a session its owner
  // had deleted.  A note is keyed by session id and stored where the daemon
  // cannot see it, so nothing removed one when its session went away.
  await openSession(page);
  await page.getByRole("button", { name: "Open Sessions" }).click();
  const rail = page.getByRole("region", { name: "Sessions" });

  const row = rail.getByRole("button", { name: "Edit your note about session 20260915_170000" });
  await row.click();
  await rail.getByRole("textbox", { name: "Note about session 20260915_170000" }).fill("ask about the grace period");
  await expect(rail.getByText("✎ ask about the grace period")).toBeVisible();

  // The OTHER delete route -- the one that had no forget at all.
  await composer(page).fill("session delete 20260915_170000");
  await composer(page).press("Enter");

  // The row goes because the daemon really removed the record, and the
  // listing is re-asked; that is the refresh, not the forget.
  await expect(rail.getByText("20260915_170000")).toHaveCount(0);

  // The forget is asserted in the STORE, because the row disappearing takes
  // the note's line with it whether or not anything forgot anything -- a
  // first draft of this case passed with the forget deleted.  Without a BFF
  // the store is this browser's ``localStorage``, which outlives the row.
  await expect
    .poll(() => page.evaluate(() => localStorage.getItem("jaato.web-coder.notes.v1") ?? ""))
    .not.toContain("grace period");
});

test("the live tool-output popup floats inside the transcript, not off its edge", async ({ page }) => {
  // Reported with a screenshot of the CLI's popup cut off on the left.  An
  // unlayered ``.plate { position: relative }`` outranked Tailwind's
  // ``absolute``, so the popup sat in normal flow and ``right-5`` pushed it
  // off the left edge.  Measured, not styled: the box must lie inside the
  // transcript column and above the composer.
  await page.setViewportSize({ width: 1280, height: 800 });
  await openSession(page);
  await composer(page).fill("live");
  await composer(page).press("Enter");
  const popup = page.getByRole("dialog", { name: "Live tool output" });
  await expect(popup).toContainText("src/slow.test.ts");
  const box = (await popup.boundingBox())!;
  const main = (await page.locator("main").boundingBox())!;
  const input = (await composer(page).boundingBox())!;
  expect(box.x).toBeGreaterThanOrEqual(main.x);
  expect(box.x + box.width).toBeLessThanOrEqual(main.x + main.width);
  expect(box.y).toBeGreaterThanOrEqual(main.y);
  expect(box.y + box.height).toBeLessThanOrEqual(input.y);
  // Anchored to the right, as the design draws it -- in flow it hugs the left.
  expect(main.x + main.width - (box.x + box.width)).toBeLessThan(40);
});

test("the command proposals float above the composer instead of pushing the layout", async ({ page }) => {
  // The same unlayered rule cancelled this list's ``absolute bottom-full``:
  // it was laid out in flow, so the composer's strip grew upward and the
  // transcript shrank by the list's height every time a proposal appeared.
  // The input itself does not move (it is pinned to the bottom), which is
  // why the strip's TOP is what is measured.
  await openSession(page);
  const strip = page.locator("main > div.border-t");
  const before = (await strip.boundingBox())!;
  await composer(page).fill("mo");
  const listbox = page.getByRole("listbox", { name: "Command proposals" });
  await expect(listbox).toBeVisible();
  const list = (await listbox.boundingBox())!;
  const after = (await strip.boundingBox())!;
  expect(after.y).toBe(before.y);
  expect(list.y + list.height).toBeLessThanOrEqual((await composer(page).boundingBox())!.y);
});

test("a file in the Files panel downloads when its name is clicked (protocol 1.20)", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("please touch out/report.txt");
  await composer(page).press("Enter");
  await expect(page.getByText("Touched out/report.txt.")).toBeVisible();
  await page.getByRole("button", { name: "Open Files" }).click();
  const panel = page.getByRole("region", { name: "Files" });
  const downloading = page.waitForEvent("download");
  await panel.getByRole("button", { name: "Download out/report.txt", exact: true }).click();
  const download = await downloading;
  expect(download.suggestedFilename()).toBe("report.txt");
  const body = await (await download.createReadStream()).toArray();
  expect(Buffer.concat(body).toString()).toBe("mock content of out/report.txt\n");
});

test("the model offers a file with offer_download and the chat draws a button that downloads it", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("please offer out/report.txt");
  await composer(page).press("Enter");
  await expect(page.getByText("Here it is -- use the button above.")).toBeVisible();
  const chip = page.getByTestId("tool-block").getByRole("button", { name: "Download out/report.txt", exact: true });
  const downloading = page.waitForEvent("download");
  await chip.click();
  const download = await downloading;
  expect(download.suggestedFilename()).toBe("report.txt");

  // A file that must not leave is refused to the MODEL, and no button is drawn.
  await composer(page).fill("please offer .env");
  await composer(page).press("Enter");
  await expect(page.getByText("I could not offer it: .env: holds credentials")).toBeVisible();
  await expect(page.getByRole("button", { name: "Download .env", exact: true })).toHaveCount(0);
});

test("the jaato-sdk skill is bootstrapped into the workspace on session start (#1263)", async ({ page }) => {
  // On session start the client asks the daemon to run
  // ``jaato-scaffold integration claude-code --refresh`` into its
  // workspace; the daemon writes the skill and its monitor reports the
  // files, so the Files panel lists ``.claude/skills/jaato-sdk/SKILL.md``
  // and the notice names the version the copy was stamped with.
  await openSession(page);
  await page.getByRole("button", { name: "Open Files" }).click();
  const panel = page.getByRole("region", { name: "Files" });
  await expect(
    panel.getByRole("button", { name: "Download .claude/skills/jaato-sdk/SKILL.md", exact: true }),
  ).toBeVisible();
  await expect(page.getByText(/claude-code skill installed \(jaato-server mock-0\.0\.1\)/)).toBeVisible();
});

test("memories rail lists the store, re-lists on a store_memory, and approves and removes (#1232)", async ({ page }) => {
  await openSession(page);
  await page.getByRole("button", { name: "Open Memories" }).click();
  const panel = page.getByRole("region", { name: "Memories" });

  // The seeded store: one raw (unvetted), two approved, one of them global.
  await expect(panel.getByTestId("memory-row")).toHaveCount(3);
  await expect(page.getByRole("button", { name: "Close Memories" })).toContainText("3 memories · 1 unvetted");
  const raw = panel.locator('[data-memory-id="mem_raw_1"]');
  await expect(raw).toContainText("unvetted");
  await expect(panel.locator('[data-memory-id="mem_global_1"]')).toContainText("global");

  // Nothing is printed into the transcript: the verbs are quiet.
  await expect(page.getByText("mock: executed memory")).toHaveCount(0);

  // Expanding fetches the content the list does not carry.
  await raw.getByRole("button", { name: /^Show memory/ }).click();
  await expect(raw.getByTestId("memory-content")).toHaveText("Run pnpm install; npm install breaks the lockfile.");

  // A successful store_memory in the conversation re-lists, and the new
  // memory says it was written here.
  await composer(page).fill("please remember the api is versioned");
  await composer(page).press("Enter");
  await expect(panel.getByTestId("memory-row")).toHaveCount(4);
  await expect(panel.getByText("written in this session")).toBeVisible();
  await expect(panel.getByText("used in this session")).toBeVisible();
  // The filter keeps what was written OR retrieved here: the new memory,
  // and the seeded one this session's retrieval surfaced.
  await panel.getByRole("checkbox", { name: /This session only/ }).check();
  await expect(panel.getByTestId("memory-row")).toHaveCount(2);
  await panel.getByRole("checkbox", { name: /This session only/ }).uncheck();

  // Approve: the unvetted marker goes, and the daemon's store is re-read.
  await raw.getByRole("button", { name: /^Approve memory/ }).click();
  await expect(panel.getByRole("status")).toHaveText("Approved.");
  await expect(raw).not.toContainText("unvetted");

  // Remove is two steps.
  await raw.getByRole("button", { name: /^Remove memory/ }).click();
  await raw.getByRole("button", { name: /^Confirm remove memory/ }).click();
  await expect(panel.getByRole("status")).toHaveText("Removed.");
  await expect(panel.locator('[data-memory-id="mem_raw_1"]')).toHaveCount(0);
});

test("diagnostics rail shows the record and a live re-check, distinctly, and can be refused (#1294)", async ({ page }) => {
  await openSession(page);
  await page.getByRole("button", { name: "Open Diagnostics" }).click();
  const panel = page.getByRole("region", { name: "Diagnostics" });

  // Opening the section asks once, with no prompt typed -- the live-view
  // entry point is a click, not a hidden key combination.
  const record = panel.getByRole("region", { name: "Session record" });
  const live = panel.getByRole("region", { name: "Live confinement check" });
  await expect(record).toBeVisible();
  await expect(live).toBeVisible();
  await expect(record).toContainText("apparmor");
  await expect(live).toContainText("Enforced");

  // Nothing is printed into the transcript: the verb is quiet.
  await expect(page.getByText("mock: executed session.diagnostics")).toHaveCount(0);

  // Nothing here offers to export, download, or copy the result.
  await expect(panel.getByRole("button", { name: /export/i })).toHaveCount(0);
  await expect(panel.getByRole("button", { name: /download/i })).toHaveCount(0);
  await expect(panel.getByRole("button", { name: /copy/i })).toHaveCount(0);

  // Arm a refusal, then explicitly re-check -- the daemon's owner gate,
  // rendered in words, replacing nothing already on screen until it
  // answers.
  await composer(page).fill("diag refuse");
  await composer(page).press("Enter");
  await panel.getByRole("button", { name: "Re-check now" }).click();
  await expect(page.getByRole("alert")).toContainText("Only the owner");
});

test("the command palette: Ctrl/⌘+K opens it, search filters, Enter runs, Escape closes (#1304 §6)", async ({ page }) => {
  await openSession(page);
  const dialog = page.getByRole("dialog", { name: "Command palette" });
  await page.keyboard.press("Control+k");
  await expect(dialog).toBeVisible();
  const search = page.getByLabel("Search commands");
  await expect(search).toBeFocused();

  // Filters the same ``command.list`` the composer's own proposals
  // complete from -- there is no second source of names.
  await search.fill("permissions");
  await expect(dialog.getByRole("option", { name: /permissions status/ })).toBeVisible();
  await expect(dialog.getByRole("option", { name: /^model/ })).toHaveCount(0);

  // Escape closes without running anything.
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(page.getByText("mock: executed")).toHaveCount(0);

  // Reopen, filter, run by clicking -- the same path Enter takes.
  await page.keyboard.press("Control+k");
  await page.getByLabel("Search commands").fill("permissions status");
  await dialog.getByRole("option", { name: /permissions status/ }).click();
  await expect(dialog).toHaveCount(0);
  // ``permissions <anything>`` is handled by the mock's own dedicated
  // branch (the one the permission-status plate's round trip depends on),
  // which reports ``mock: permissions <args>`` rather than falling
  // through to the generic ``mock: executed <cmd> <args>`` echo.
  await expect(page.getByText("mock: permissions status")).toBeVisible();
});

test("§6: help opens the palette instead of dumping into the transcript", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("help");
  await composer(page).press("Enter");
  await expect(page.getByRole("dialog", { name: "Command palette" })).toBeVisible();
  // The old ~500-line dump is gone.
  await expect(page.getByText(/Keys: Ctrl\+P/)).toHaveCount(0);
});

test("the leader chord replaces the direct Ctrl+P/B/T/A/O bindings (#1304 §7)", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("tool");
  await composer(page).press("Enter");
  await expect(page.getByRole("button", { name: /run_command/ }).first()).toBeVisible();

  // A bare Ctrl+P is not this app's -- it changes nothing here (it falls
  // through to whatever the browser does with it; see
  // ``useKeyboardShortcuts.test.ts`` for the assertion that it is never
  // even ``preventDefault``ed).
  await page.keyboard.press("Control+p");
  await expect(page.getByText("Task plan")).toHaveCount(0);

  // K then P: the leader opens the palette and, on the very first
  // keystroke with the search box still empty, applies the quick action
  // and closes -- no visible list in between.
  await page.keyboard.press("Control+k");
  await expect(page.getByRole("dialog", { name: "Command palette" })).toBeVisible();
  await page.keyboard.press("p");
  await expect(page.getByRole("dialog", { name: "Command palette" })).toHaveCount(0);
  await expect(page.getByText("Task plan")).toBeVisible();
  await expect(page.getByText("List the directory")).toBeVisible();
});

test("a session that fails to bootstrap turns the status bar red (#1304 §6)", async ({ page }) => {
  await page.goto("/");
  await page.getByPlaceholder("ws://host:8080").fill(WS);
  await page.getByRole("button", { name: "Connect" }).click();
  // A status bar exists even with no session yet -- ``fault`` is what
  // turns it red, not ``sessionId``.
  await page.getByRole("button", { name: /bootstrap-fail/ }).click();
  const dot = page.locator(".bg-error").first();
  await expect(page.getByText("no session")).toBeVisible();
  await expect(dot).toBeVisible();
  // The tooltip is on the indicator's OUTER span (dot + text together, one
  // hover target), not on the "no session" text node itself.
  await expect(page.locator("span", { hasText: "no session" }).first()).toHaveAttribute(
    "title",
    /RunnerBootstrapFailed: Runner bootstrap failed/,
  );
});

test("at 375px the session screen does not overflow horizontally (#1304 §8)", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 700 });
  await openSession(page);
  await composer(page).fill("code");
  await composer(page).press("Enter");
  await expect(page.locator("table.j-table th", { hasText: "Latency" })).toBeVisible();
  const { docW, winW } = await page.evaluate(() => ({
    docW: document.documentElement.scrollWidth,
    winW: window.innerWidth,
  }));
  expect(docW).toBe(winW);
});

test("files panel: a markdown file is rendered, raw on request, and its relative links open in the viewer", async ({ page }) => {
  await openSession(page);
  await composer(page).fill("write markdown docs");
  await composer(page).press("Enter");
  await expect(page.getByText("Wrote the docs.")).toBeVisible();
  await page.getByRole("button", { name: "Open Files" }).click();
  const panel = page.getByRole("region", { name: "Files" });
  await panel.getByRole("button", { name: "View docs/README.md", exact: true }).click();

  const viewer = panel.getByTestId("file-viewer");
  await expect(viewer.getByRole("heading", { name: "Project guide" })).toBeVisible();
  await expect(viewer.getByRole("table")).toBeVisible();
  await expect(viewer.getByRole("link", { name: "our site" })).toHaveAttribute("target", "_blank");
  // The document's raw <script> is dropped, never executed.
  expect(await page.evaluate(() => (window as unknown as { __pwned?: boolean }).__pwned)).toBeUndefined();

  // raw shows the markdown as written; rendered goes back.
  await viewer.getByRole("button", { name: "raw" }).click();
  await expect(viewer.getByText("# Project guide", { exact: false })).toBeVisible();
  await viewer.getByRole("button", { name: "rendered" }).click();

  // A relative link opens the linked file in the same viewer; back returns.
  await viewer.getByRole("link", { name: "the setup steps" }).click();
  await expect(viewer.getByRole("heading", { name: "Setup" })).toBeVisible();
  await expect(viewer.getByRole("checkbox")).toHaveCount(2);
  await viewer.getByRole("button", { name: "Back to docs/README.md" }).click();
  await expect(viewer.getByRole("heading", { name: "Project guide" })).toBeVisible();
});
