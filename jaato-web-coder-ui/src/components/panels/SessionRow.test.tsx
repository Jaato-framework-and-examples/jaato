/**
 * Which element in a session row is the control.
 *
 * This row shipped a regression that no screenshot and no existing test
 * could see.  The whole row used to be a ``button`` with ``Attach`` as a
 * decorative ``span`` inside it, so clicking the chip worked by bubbling.
 * Adding the note pencil forced the row out of a ``button`` -- a button
 * cannot nest inside a button -- and the chip became a SIBLING of the
 * clickable element while keeping its ``btn btn-steel`` styling.  It looked
 * exactly as before and did nothing.
 *
 * The e2e suite stayed green throughout, and could not have caught it: it
 * resumed by ROLE and accessible name, which lived on the row's text, while
 * the chip was ``aria-hidden`` and therefore had no role to match.  A test
 * that asserts *where the click goes* is the only kind that fails on this,
 * which is why these three exist when the rest of the component is markup.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { SessionRow } from "./SessionsPanel";
import { useJaato } from "@/store/store";
import type { SessionSummary } from "@/protocol/sessions";

const ID = "20260919_120000";

const sess: SessionSummary = {
  id: ID, name: "", description: "fix the budget panel", provider: "anthropic",
  model: "claude-sonnet-4", isLoaded: true, isCurrent: false, clientCount: 0,
  turnCount: 3, workspacePath: "/ws/one",
};

beforeEach(() => { useJaato.setState({ notes: {}, noteStatus: {} }); });
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("SessionRow: the control is the chip, not the row", () => {
  it("Attach is a real button, and clicking it resumes that session", () => {
    const onAttach = vi.fn();
    render(<SessionRow sess={sess} onAttach={onAttach} />);

    // By ROLE: a span wearing `btn btn-steel` does not satisfy this, which
    // is the whole regression. The accessible name contains the visible
    // word (WCAG 2.5.3), so a voice user asking for "Attach" reaches it.
    const attach = screen.getByRole("button", { name: `Attach session ${ID}` });
    fireEvent.click(attach);

    expect(onAttach).toHaveBeenCalledWith(ID);
    expect(onAttach).toHaveBeenCalledTimes(1);
  });

  it("the row's text is inert — it is not a control and has none above it", () => {
    render(<SessionRow sess={sess} onAttach={vi.fn()} />);

    // The mirror image of the defect: the id used to BE the button.
    expect(screen.getByText(ID).closest("button")).toBeNull();
    // Only two controls in the row, and neither is the text.
    const names = screen.getAllByRole("button").map((b) => b.getAttribute("aria-label"));
    expect(names).toEqual([`Edit your note about session ${ID}`, `Attach session ${ID}`]);
  });

  it("offers no Attach where resuming is not on offer (the rail)", () => {
    // The control: without this, a chip rendered unconditionally would pass
    // the first case while being wrong in the one place it must not appear.
    render(<SessionRow sess={sess} showWorkspace />);
    expect(screen.queryByRole("button", { name: /^Attach session/ })).toBeNull();
  });
});
