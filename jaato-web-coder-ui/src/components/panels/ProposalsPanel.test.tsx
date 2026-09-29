/**
 * The Proposals panel: reference claims agents proposed, and curating them.
 *
 * Facts the markup must carry: every proposal says **unreviewed** in words;
 * whether a person approved it at the prompt is shown; a claim the daemon
 * says cannot be promoted shows WHY and its Promote is disabled; the
 * buttons appear only when the daemon said this connection may curate; and
 * Dismiss is two-step.  Model-written text is shown as text, not markup.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useJaato, emptyReferenceClaims } from "@/store/store";
import type { ReferenceClaimRow } from "@/store/types";

const promote = vi.fn(async (_id: string) => true);
const dismiss = vi.fn(async (_id: string) => true);
vi.mock("@/app/referenceClaims", async (orig) => ({
  ...(await orig<typeof import("@/app/referenceClaims")>()),
  promoteReferenceClaim: (id: string) => promote(id),
  dismissReferenceClaim: (id: string) => dismiss(id),
  refreshReferenceClaims: vi.fn(async () => undefined),
}));

const { ProposalsPanel } = await import("./ProposalsPanel");

const row = (id: string, extra: Partial<ReferenceClaimRow> = {}): ReferenceClaimRow => ({
  claim_id: `c-${id}`, id, name: `Doc ${id}`, description: "", tags: ["ops"],
  type: "local", path: `docs/${id}.md`, problems: [], ...extra,
});

function load(rows: ReferenceClaimRow[], mayCurate: boolean) {
  useJaato.setState({ sessionId: "s1", referenceClaims: { ...emptyReferenceClaims(), rows, status: "loaded", mayCurate } });
}

beforeEach(() => { promote.mockClear(); dismiss.mockClear(); });
afterEach(() => { cleanup(); });

describe("ProposalsPanel", () => {
  it("marks every proposal unreviewed in words, with its path", () => {
    load([row("a")], false);
    render(<ProposalsPanel />);
    const r = screen.getByTestId("proposal-row");
    expect(r.textContent).toContain("unreviewed");
    expect(r.textContent).toContain("docs/a.md");
  });

  it("offers no curation to a connection the daemon said may not curate", () => {
    load([row("a")], false);
    render(<ProposalsPanel />);
    expect(screen.queryByRole("button", { name: /Promote proposal/ })).toBeNull();
    expect(screen.getByText(/Only the owner/)).toBeTruthy();
  });

  it("promotes on click", () => {
    load([row("a")], true);
    render(<ProposalsPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Promote proposal Doc a" }));
    expect(promote).toHaveBeenCalledWith("c-a");
  });

  it("a claim promotion would refuse says why and cannot be promoted", () => {
    load([row("a", { problems: ["'a' is already in the catalog"] })], true);
    render(<ProposalsPanel />);
    expect(screen.getByText("'a' is already in the catalog")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Promote proposal Doc a" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("dismiss asks for confirmation first", () => {
    load([row("a")], true);
    render(<ProposalsPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Dismiss proposal Doc a" }));
    expect(dismiss).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Confirm dismiss proposal Doc a" }));
    expect(dismiss).toHaveBeenCalledWith("c-a");
  });

  it("shows who approved it at the prompt, and model text as text", () => {
    load([row("a", {
      type: "inline", path: undefined, content: "<b>steps</b>",
      origin: { generated_by: { agent_id: "writer" }, witnessed_by: { user: "acme:bob" } },
    })], false);
    render(<ProposalsPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Show proposal Doc a" }));
    expect(screen.getByText(/approved at the prompt by acme:bob/)).toBeTruthy();
    expect(screen.getByTestId("proposal-content").textContent).toBe("<b>steps</b>");
    expect(screen.getByTestId("proposal-content").querySelector("b")).toBeNull();
  });

  it("names files it could not show", () => {
    useJaato.setState({ sessionId: "s1", referenceClaims: { ...emptyReferenceClaims(), status: "loaded", unreadable: ["x.json"] } });
    render(<ProposalsPanel />);
    expect(screen.getByText(/could not be shown: x.json/)).toBeTruthy();
    expect(screen.getByText("Nothing proposed.")).toBeTruthy();
  });
});
