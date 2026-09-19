/**
 * Two decisions the exit plate makes about the note field, each of which
 * would be invisible in a screenshot and expensive to get wrong.
 *
 * Everything else about this component is markup, and a test asserting
 * markup only pins the markup.  These two are the ones where a plausible
 * implementation loses what somebody typed:
 *
 * - **Escape inside the field must not answer the prompt.**  Escape answered
 *   ``r`` unconditionally, so pressing it to dismiss the textarea discarded
 *   a half-written note AND left the session.  It now blurs first.
 * - **A failed save must keep the plate open.**  The whole reason the save
 *   is awaited before the exit action is that detaching anyway loses the
 *   note silently -- which is the one outcome this feature cannot have.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { ExitPrompt } from "./ExitPrompt";
import { exitOptions } from "@/app/exitChoice";
import { useJaato } from "@/store/store";

const SESSION = "20260919_120000";

/**
 * The note API the hook resolves is chosen from ``notesUrl``; pointing it at
 * a backend and stubbing ``fetch`` is how a save is made to fail without
 * reaching into the hook, so the test exercises the real path the browser
 * takes.
 */
function serveNotes(put: (text: string) => Promise<Response>) {
  vi.stubGlobal("fetch", vi.fn(async (_url: string, init?: RequestInit) => {
    if (init?.method === "PUT") return put(JSON.parse(String(init.body)).text as string);
    return new Response(JSON.stringify({ notes: [] }), { status: 200 });
  }));
}

const okPut = async (text: string) =>
  new Response(JSON.stringify({ note: { sessionId: SESSION, text, updatedAt: "now" } }), { status: 200 });

beforeEach(() => {
  useJaato.setState({ sessionId: SESSION, notesUrl: "./api/notes", notes: {}, noteStatus: {} });
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

function renderPrompt(onAnswer = vi.fn()) {
  const x = { running: false, options: exitOptions(false), focus: 0 };
  render(<ExitPrompt x={x} onAnswer={onAnswer} />);
  return onAnswer;
}

describe("ExitPrompt and the note field", () => {
  it("Escape inside the field blurs it and does NOT answer the prompt", async () => {
    serveNotes(okPut);
    const onAnswer = renderPrompt();
    const field = screen.getByRole("textbox");
    field.focus();
    fireEvent.change(field, { target: { value: "ask about the grace period" } });

    // Fired ON the field so it bubbles to the window listener carrying the
    // field as ``event.target`` -- which is the whole thing the guard reads.
    fireEvent.keyDown(field, { key: "Escape" });
    expect(onAnswer).not.toHaveBeenCalled();
    expect(document.activeElement).not.toBe(field);
    // And what was typed is still there to be saved.
    expect((field as HTMLTextAreaElement).value).toBe("ask about the grace period");

    // A second Escape, now outside the field, returns as it always did.
    fireEvent.keyDown(document.body, { key: "Escape" });
    await waitFor(() => expect(onAnswer).toHaveBeenCalledWith("r"));
  });

  it("a failed save keeps the plate open: the answer is not forwarded", async () => {
    serveNotes(async () => new Response("no", { status: 500 }));
    const onAnswer = renderPrompt();
    const field = screen.getByRole("textbox");
    fireEvent.change(field, { target: { value: "re-run the e2e suite" } });

    fireEvent.click(screen.getByRole("button", { name: /Detach/i }));

    await screen.findByRole("alert");
    expect(onAnswer).not.toHaveBeenCalled();
    // The note is still in the box -- the only copy of it that exists.
    expect((field as HTMLTextAreaElement).value).toBe("re-run the e2e suite");
  });

  it("a save that succeeds forwards the answer, once", async () => {
    // The control: without this, a component that never answers would pass
    // both cases above.
    serveNotes(okPut);
    const onAnswer = renderPrompt();
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "hand off to review" } });
    fireEvent.click(screen.getByRole("button", { name: /Detach/i }));
    await waitFor(() => expect(onAnswer).toHaveBeenCalledWith("d"));
    expect(onAnswer).toHaveBeenCalledTimes(1);
    expect(useJaato.getState().notes[SESSION]?.text).toBe("hand off to review");
  });
});
