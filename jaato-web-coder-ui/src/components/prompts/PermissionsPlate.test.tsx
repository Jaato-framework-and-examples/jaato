/**
 * The status bar's permission segment is a control, not a readout.
 *
 * It already reported the effective default and named the command in its
 * tooltip, so the reading and the doing were in different places and only
 * the reading was on screen.
 *
 * Two rules carry the design, and both are asserted rather than described:
 *
 *  - **Every action goes through `submitInput`**, the path a typed command
 *    takes.  A button speaking to the daemon directly would change the
 *    session's permission posture with no record in the transcript of who
 *    asked for it -- and would be a second way to express `permissions`
 *    that can drift from the first.
 *  - **Dismissing is never a decision.**  This is a popover over the page,
 *    not a question blocking the turn, so Escape and a click outside close
 *    it and run nothing.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

/**
 * Click the way a browser does: mousedown, then click.
 *
 * ``fireEvent.click`` dispatches only the last of the three, so a test
 * using it cannot see the outside-click listener at all -- and the
 * deferral that stops the OPENING click from immediately closing the
 * plate would be untested.  Verified: with the deferral removed, the
 * dismissal case below fails.
 */
function realClick(el: Element): void {
  fireEvent.mouseDown(el);
  fireEvent.click(el);
}
import { StatusBar } from "@/components/layout/StatusBar";
import { useJaato } from "@/store/store";

const submitted: string[] = [];

vi.mock("@/app/actions", () => ({
  submitInput: (text: string) => { submitted.push(text); return Promise.resolve(); },
}));
vi.mock("@/app/exitChoice", () => ({ requestExit: () => Promise.resolve() }));

function withStatus(effectiveDefault: string, suspensionScope: string | null = null) {
  useJaato.setState({ permissionStatus: { effectiveDefault, suspensionScope } });
}

beforeEach(() => {
  submitted.length = 0;
  useJaato.getState().resetSessionState();
});
afterEach(cleanup);

describe("the permission segment", () => {
  it("is a button that opens the plate", () => {
    withStatus("ask");
    render(<StatusBar />);
    const seg = screen.getByTestId("permission-status");
    expect(seg.tagName).toBe("BUTTON");
    expect(seg.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByRole("dialog")).toBeNull();
    realClick(seg);
    expect(screen.getByRole("dialog", { name: "Session permissions" })).toBeTruthy();
  });

  it("is not rendered at all before the daemon has reported a policy", () => {
    render(<StatusBar />);
    expect(screen.queryByTestId("permission-status")).toBeNull();
  });
});

describe("the plate runs the command", () => {
  it("sets the default through submitInput, the same path a typed command takes", () => {
    withStatus("ask");
    render(<StatusBar />);
    realClick(screen.getByTestId("permission-status"));
    fireEvent.click(screen.getByRole("button", { name: "Deny" }));
    expect(submitted).toEqual(["permissions default deny"]);
    // ...and it closes, so the transcript the daemon answers into is visible.
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("marks the policy in force, and offers the other two", () => {
    withStatus("allow");
    render(<StatusBar />);
    realClick(screen.getByTestId("permission-status"));
    expect(screen.getByRole("button", { name: "Allow" }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByRole("button", { name: "Ask" }).getAttribute("aria-pressed")).toBe("false");
  });

  it("offers the two suspend scopes the daemon accepts", () => {
    withStatus("ask");
    render(<StatusBar />);
    realClick(screen.getByTestId("permission-status"));
    fireEvent.click(screen.getByRole("button", { name: "This turn" }));
    expect(submitted).toEqual(["permissions suspend --turn"]);
  });

  it("offers Resume instead once prompting IS suspended, and says the default is moot", () => {
    withStatus("ask", "turn");
    render(<StatusBar />);
    realClick(screen.getByTestId("permission-status"));
    expect(screen.queryByRole("button", { name: "This turn" })).toBeNull();
    expect(screen.getByText(/Prompting is suspended/)).toBeTruthy();
    // No policy reads as in force while it is not being consulted.
    for (const name of ["Ask", "Allow", "Deny"]) {
      expect(screen.getByRole("button", { name }).getAttribute("aria-pressed")).toBe("false");
    }
    fireEvent.click(screen.getByRole("button", { name: "Resume prompting" }));
    expect(submitted).toEqual(["permissions resume"]);
  });
});

describe("dismissing is never a decision", () => {
  it("Escape closes and runs nothing", () => {
    withStatus("ask");
    render(<StatusBar />);
    realClick(screen.getByTestId("permission-status"));
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(submitted).toEqual([]);
  });

  it("clicking the segment again CLOSES it, rather than closing and reopening", async () => {
    // The segment is outside the plate, so the outside-click listener sees
    // its mousedown too -- and a close that lands before the toggle's own
    // click would simply reopen.  This is the case that decides whether
    // the listener may look at the segment at all.
    withStatus("ask");
    render(<StatusBar />);
    const seg = screen.getByTestId("permission-status");
    realClick(seg);
    await new Promise((r) => setTimeout(r, 0));
    realClick(seg);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("a click outside closes and runs nothing", async () => {
    withStatus("ask");
    render(<StatusBar />);
    realClick(screen.getByTestId("permission-status"));
    // The outside-click listener is attached on a deferred tick, so the
    // click that OPENED the plate cannot immediately close it again.
    await new Promise((r) => setTimeout(r, 0));
    fireEvent.mouseDown(document.body);
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(submitted).toEqual([]);
  });
});
