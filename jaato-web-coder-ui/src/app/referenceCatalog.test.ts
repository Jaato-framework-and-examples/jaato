/**
 * The References rail's data module: the workspace catalog and its links.
 *
 * Pinned here, each a way the section could mislead or lose work:
 *   • a failed list is REPORTED and keeps the rows it had;
 *   • a daemon below 1.33 is "unsupported" and is never asked;
 *   • a save sends the WHOLE draft, trimmed, and closes the editor only on
 *     success -- a refusal keeps what was typed and says why in words;
 *   • a draft the daemon would refuse is caught before anything is sent;
 *   • the daemon's warnings reach the notice, and a save re-lists;
 *   • a dangling edge is what the badge flags.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useJaato } from "@/store/store";
import type { ReferenceCatalogRow } from "@/store/types";

const listReferenceCatalog = vi.fn();
const updateReferenceLinks = vi.fn();
let protocol = "1.33";
let connected = true;

vi.mock("@/sdk/connection", () => ({
  isConnected: () => connected,
  getClient: () => ({
    get serverProtocolVersion() { return protocol; },
    listReferenceCatalog, updateReferenceLinks,
  }),
}));

const rk = await import("@/app/referenceCatalog");

const ref = (id: string, extra: Partial<ReferenceCatalogRow> = {}): ReferenceCatalogRow => ({
  id, name: `Ref ${id}`, description: "", bundle: "", file: `.jaato/references/${id}.json`,
  links: [], linked_from: [], ...extra,
});

beforeEach(() => {
  connected = false;
  useJaato.getState().resetSessionState();
  useJaato.setState({ sessionId: "s1" });
  listReferenceCatalog.mockReset();
  updateReferenceLinks.mockReset();
  protocol = "1.33";
  connected = true;
});

describe("the list", () => {
  it("loads rows and the owner gate's answer", async () => {
    listReferenceCatalog.mockResolvedValue({ ok: true, references: [ref("a"), ref("b")], unreadable: [], may_curate: true });
    await rk.refreshReferenceCatalog();
    const r = useJaato.getState().referenceCatalog;
    expect(r.status).toBe("loaded");
    expect(r.rows.map((x) => x.id)).toEqual(["a", "b"]);
    expect(r.mayCurate).toBe(true);
    expect(rk.referencesSummary(r)).toBe("2");
  });

  it("a failed read is reported and does not empty the list", async () => {
    listReferenceCatalog.mockResolvedValueOnce({ ok: true, references: [ref("a")] });
    await rk.refreshReferenceCatalog();
    listReferenceCatalog.mockResolvedValueOnce({ ok: false, category: "unsafe_path", error: "leaves the workspace" });
    await rk.refreshReferenceCatalog();
    const r = useJaato.getState().referenceCatalog;
    expect(r.status).toBe("error");
    expect(r.error).toContain("leaves the workspace");
    expect(r.rows.map((x) => x.id)).toEqual(["a"]);
  });

  it("an older daemon is unsupported and never asked", async () => {
    protocol = "1.32";
    await rk.refreshReferenceCatalog();
    expect(useJaato.getState().referenceCatalog.status).toBe("unsupported");
    expect(listReferenceCatalog).not.toHaveBeenCalled();
  });

  it("the badge flags a dangling edge", () => {
    expect(rk.referencesSummary({ rows: [ref("a", { links: [{ to: "x", rel: "elaborates", dangling: true }] })] })).toBe("!");
    expect(rk.referencesSummary({ rows: [] })).toBeNull();
  });
});

describe("the draft", () => {
  it("catches what the daemon would refuse", () => {
    expect(rk.draftProblem("a", [{ to: " ", rel: "elaborates" }])).toMatch(/target/);
    expect(rk.draftProblem("a", [{ to: "a", rel: "elaborates" }])).toMatch(/itself/);
    expect(rk.draftProblem("a", [{ to: "b", rel: "elaborates" }, { to: "b ", rel: "elaborates" }])).toMatch(/twice/);
    expect(rk.draftProblem("a", [{ to: "b c", rel: "elaborates" }])).toMatch(/one id/);
    expect(rk.draftProblem("a", [{ to: "b", rel: "supersedes" }, { to: "b", rel: "elaborates" }])).toBeNull();
  });

  it("is sent trimmed, an empty note left out", () => {
    expect(rk.draftToWire([{ to: " b ", rel: "depends-on", note: "  " }, { to: "c", rel: "elaborates", note: " why " }]))
      .toEqual([{ to: "b", rel: "depends-on" }, { to: "c", rel: "elaborates", note: "why" }]);
  });
});

describe("saving", () => {
  beforeEach(async () => {
    listReferenceCatalog.mockResolvedValue({ ok: true, references: [ref("a", { links: [{ to: "b", rel: "elaborates" }] }), ref("b")], may_curate: true });
    await rk.refreshReferenceCatalog();
    rk.startEditingLinks(useJaato.getState().referenceCatalog.rows[0]!);
  });

  it("sends the whole list, closes the editor, reports warnings and re-lists", async () => {
    rk.setDraftLinks([{ to: "b", rel: "supersedes" }, { to: "later", rel: "depends-on" }]);
    updateReferenceLinks.mockResolvedValue({ ok: true, links: [], warnings: ["links to 'later' (depends-on), which is not in this workspace's catalog"] });
    listReferenceCatalog.mockClear();
    expect(await rk.saveDraftLinks()).toBe(true);
    expect(updateReferenceLinks).toHaveBeenCalledWith("a", [{ to: "b", rel: "supersedes" }, { to: "later", rel: "depends-on" }]);
    const r = useJaato.getState().referenceCatalog;
    expect(r.editing).toBeNull();
    expect(r.notice?.warning).toBe(true);
    expect(r.notice?.text).toContain("later");
    expect(listReferenceCatalog).toHaveBeenCalled();
  });

  it("a refusal keeps the draft and says why in words", async () => {
    rk.setDraftLinks([{ to: "b", rel: "supersedes" }]);
    updateReferenceLinks.mockResolvedValue({ ok: false, category: "not_owner", error: "x" });
    expect(await rk.saveDraftLinks()).toBe(false);
    const r = useJaato.getState().referenceCatalog;
    expect(r.editing?.links).toEqual([{ to: "b", rel: "supersedes" }]);
    expect(r.notice?.text).toMatch(/Only the owner/);
    expect(r.busy).toEqual({});
  });

  it("a draft with a problem is not sent", async () => {
    rk.setDraftLinks([{ to: "a", rel: "elaborates" }]);
    expect(await rk.saveDraftLinks()).toBe(false);
    expect(updateReferenceLinks).not.toHaveBeenCalled();
    expect(useJaato.getState().referenceCatalog.notice?.error).toBe(true);
  });
});
