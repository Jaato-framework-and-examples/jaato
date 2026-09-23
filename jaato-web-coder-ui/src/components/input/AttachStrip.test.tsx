/**
 * The attachment strip's two behaviours under test here:
 *
 *  - **A staging progress bar (#1248/#1250 status work).**  While a file is
 *    ``staging`` the row carries an indeterminate bar; once it settles to
 *    ``staged`` (✓) or ``failed`` (✗) — including the #1248 staging timeout,
 *    which resolves to ``failed`` — the bar is gone.  No byte-level progress
 *    is available from the transport, so it is indeterminate rather than a
 *    faked percentage.
 *  - **Scope filtering (#1250).**  An upload carries the session it was
 *    staged into; the strip shows only the active context's uploads, so a
 *    file attached in session A is not shown above session B's composer, and
 *    is shown again on returning to A.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { AttachStrip } from "./AttachStrip";
import { useJaato } from "@/store/store";
import type { StagedUpload } from "@/store/types";

function seed(uploads: StagedUpload[], sessionId: string | null = null) {
  useJaato.setState({ uploads, sessionId });
}

beforeEach(() => {
  useJaato.getState().resetSessionState();
  useJaato.setState({ uploads: [], sessionId: null });
});
afterEach(cleanup);

describe("AttachStrip — staging progress bar", () => {
  it("shows the bar while a file is staging", () => {
    seed([{ id: "s1", path: "a.txt", size: 5, status: "staging", scope: "" }]);
    render(<AttachStrip always hint="" />);
    expect(screen.getByTestId("staging-bar")).toBeTruthy();
  });

  it("shows no bar once the file is staged", () => {
    seed([{ id: "s1", path: "a.txt", size: 5, status: "staged", scope: "" }]);
    render(<AttachStrip always hint="" />);
    expect(screen.queryByTestId("staging-bar")).toBeNull();
  });

  it("shows no bar once the file has failed (the #1248 timeout lands here)", () => {
    seed([{ id: "s1", path: "a.txt", size: 5, status: "failed", scope: "", error: "stageFiles: no workspace.files.staged response after 120000 ms" }]);
    render(<AttachStrip always hint="" />);
    expect(screen.queryByTestId("staging-bar")).toBeNull();
    expect(screen.getByText(/no workspace\.files\.staged response/)).toBeTruthy();
  });

  it("shows no bar for a queued file", () => {
    seed([{ id: "s1", path: "a.txt", size: 5, status: "queued", scope: "" }]);
    render(<AttachStrip always hint="" />);
    expect(screen.queryByTestId("staging-bar")).toBeNull();
  });
});

describe("AttachStrip — scope filtering (#1250)", () => {
  it("hides a session A upload while session B is active, and shows it on return", () => {
    const upload: StagedUpload = { id: "a1", path: "sessionA.txt", size: 5, status: "staged", scope: "session:A" };

    // Active session B: A's upload is not shown.
    seed([upload], "B");
    const { unmount } = render(<AttachStrip always hint="" />);
    expect(screen.queryByText("sessionA.txt")).toBeNull();
    unmount();
    cleanup();

    // Back on A: it is shown again (the store never wiped it).
    seed([upload], "A");
    render(<AttachStrip always hint="" />);
    expect(screen.getByText("sessionA.txt")).toBeTruthy();
  });

  it("shows a picker (unscoped) upload while no session is active", () => {
    seed([{ id: "pk", path: "picked.txt", size: 5, status: "queued", scope: "" }], null);
    render(<AttachStrip always hint="" />);
    expect(screen.getByText("picked.txt")).toBeTruthy();
  });
});
