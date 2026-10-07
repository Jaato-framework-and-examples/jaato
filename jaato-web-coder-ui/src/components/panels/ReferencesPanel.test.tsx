/**
 * The References panel: the catalog, its links both ways, and the links
 * editor.
 *
 * Facts the markup must carry: a dangling edge and a ``supersedes`` are
 * marked in words; Edit links appears only when the daemon said this
 * connection may curate, and never on an id declared in two files; the
 * editor adds, changes and removes rows in the draft; Save hands the draft
 * to the module.  Catalog text is shown as text, not markup.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { useJaato, emptyReferenceCatalog } from "@/store/store";
import type { ReferenceCatalogRow } from "@/store/types";

const save = vi.fn(async () => true);
vi.mock("@/app/referenceCatalog", async (orig) => ({
  ...(await orig<typeof import("@/app/referenceCatalog")>()),
  saveDraftLinks: () => save(),
  refreshReferenceCatalog: vi.fn(async () => undefined),
}));

const { ReferencesPanel } = await import("./ReferencesPanel");

const ref = (id: string, extra: Partial<ReferenceCatalogRow> = {}): ReferenceCatalogRow => ({
  id, name: `Ref ${id}`, description: "", bundle: "", file: `.jaato/references/${id}.json`,
  links: [], linked_from: [], ...extra,
});

function load(rows: ReferenceCatalogRow[], mayCurate: boolean) {
  useJaato.setState({ sessionId: "s1", referenceCatalog: { ...emptyReferenceCatalog(), rows, status: "loaded", mayCurate } });
}

beforeEach(() => { save.mockClear(); });
afterEach(() => { cleanup(); });

describe("ReferencesPanel", () => {
  it("marks a dangling edge and a supersedes in words", () => {
    load([ref("a", { links: [{ to: "gone", rel: "elaborates", dangling: true }, { to: "b", rel: "supersedes" }] }), ref("b")], false);
    render(<ReferencesPanel />);
    const links = within(screen.getAllByTestId("reference-row")[0]!).getByTestId("reference-links");
    expect(links.textContent).toContain("elaborates gone (not in this catalog)");
    expect(links.textContent).toContain("supersedes b (requests for it get this one)");
  });

  it("shows who links to a reference when expanded", () => {
    load([ref("a", { linked_from: [{ from: "c", rel: "depends-on" }] })], false);
    render(<ReferencesPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Show reference Ref a" }));
    expect(screen.getByLabelText("Links to Ref a").textContent).toContain("c (depends on)");
  });

  it("offers no editor to a connection that may not curate", () => {
    load([ref("a")], false);
    render(<ReferencesPanel />);
    expect(screen.queryByRole("button", { name: /Edit links/ })).toBeNull();
    expect(screen.getByText(/Only the owner/)).toBeTruthy();
  });

  it("offers no editor on an id declared in two files", () => {
    load([ref("a", { duplicate_id: true })], true);
    render(<ReferencesPanel />);
    expect(screen.queryByRole("button", { name: /Edit links/ })).toBeNull();
    expect(screen.getByText(/declared in two files/)).toBeTruthy();
  });

  it("edits the draft and saves it", () => {
    load([ref("a", { links: [{ to: "b", rel: "elaborates" }] }), ref("b")], true);
    render(<ReferencesPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Edit links of Ref a" }));
    fireEvent.change(screen.getByLabelText("Relation of link 1"), { target: { value: "supersedes" } });
    fireEvent.click(screen.getByRole("button", { name: "Add link to Ref a" }));
    fireEvent.change(screen.getByLabelText("Target of link 2"), { target: { value: "c" } });
    fireEvent.change(screen.getByLabelText("Note of link 2"), { target: { value: "see also" } });
    expect(useJaato.getState().referenceCatalog.editing?.links).toEqual([
      { to: "b", rel: "supersedes" }, { to: "c", rel: "elaborates", note: "see also" }]);
    fireEvent.click(screen.getByRole("button", { name: "Remove link 1" }));
    expect(useJaato.getState().referenceCatalog.editing?.links).toEqual([{ to: "c", rel: "elaborates", note: "see also" }]);
    fireEvent.click(screen.getByRole("button", { name: "Save links of Ref a" }));
    expect(save).toHaveBeenCalled();
  });

  it("cancel leaves the catalog links as they were", () => {
    load([ref("a", { links: [{ to: "b", rel: "elaborates" }] }), ref("b")], true);
    render(<ReferencesPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Edit links of Ref a" }));
    fireEvent.click(screen.getByRole("button", { name: "Remove link 1" }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel editing Ref a" }));
    expect(useJaato.getState().referenceCatalog.editing).toBeNull();
    expect(screen.getByTestId("reference-links").textContent).toContain("elaborates b");
  });

  it("renders catalog text as text", () => {
    load([ref("a", { name: "<b>bold</b>" })], false);
    render(<ReferencesPanel />);
    expect(screen.getByText("<b>bold</b>")).toBeTruthy();
  });
});
