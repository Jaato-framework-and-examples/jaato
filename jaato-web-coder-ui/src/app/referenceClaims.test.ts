/**
 * The Proposals rail's data module: reference claims agents proposed.
 *
 * Pinned here, each a way the section could mislead:
 *   • a failed list is REPORTED and keeps the rows it had -- an empty list
 *     reads as "nothing proposed";
 *   • an answer for a session this client has left is dropped;
 *   • a daemon below 1.32 is "unsupported", not "empty", and is never asked;
 *   • a refusal from the owner gate is shown in words, and the list is
 *     re-read afterwards;
 *   • the origin line says whether a person approved the proposal, and
 *     never invents who;
 *   • the refresh triggers: a successful proposeReference in any agent, a
 *     session appearing -- and not a failed proposal or another tool.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useJaato } from "@/store/store";
import type { ReferenceClaimRow } from "@/store/types";

const listReferenceClaims = vi.fn();
const promoteReferenceClaim = vi.fn();
const dismissReferenceClaim = vi.fn();
let protocol = "1.32";
let connected = true;

vi.mock("@/sdk/connection", () => ({
  isConnected: () => connected,
  getClient: () => ({
    get serverProtocolVersion() { return protocol; },
    listReferenceClaims, promoteReferenceClaim, dismissReferenceClaim,
  }),
}));

const rc = await import("@/app/referenceClaims");

const row = (id: string, extra: Partial<ReferenceClaimRow> = {}): ReferenceClaimRow => ({
  claim_id: `c-${id}`, id, name: `about ${id}`, description: "", tags: ["ops"],
  type: "inline", content: "x", problems: [], ...extra,
});

beforeEach(() => {
  connected = false;
  useJaato.getState().resetSessionState();
  useJaato.setState({ sessionId: "s1" });
  for (const f of [listReferenceClaims, promoteReferenceClaim, dismissReferenceClaim]) f.mockReset();
  protocol = "1.32";
  connected = true;
});

describe("the list", () => {
  it("loads rows, unreadable names and the owner gate's answer", async () => {
    listReferenceClaims.mockResolvedValue({ ok: true, claims: [row("a"), row("b")], unreadable: ["x.json"], may_curate: true });
    await rc.refreshReferenceClaims();
    const r = useJaato.getState().referenceClaims;
    expect(r.status).toBe("loaded");
    expect(r.rows.map((x) => x.id)).toEqual(["a", "b"]);
    expect(r.unreadable).toEqual(["x.json"]);
    expect(r.mayCurate).toBe(true);
    expect(rc.proposalsSummary(r)).toBe("2");
  });

  it("a failed read is reported and does not empty the list", async () => {
    listReferenceClaims.mockResolvedValueOnce({ ok: true, claims: [row("a")], may_curate: false });
    await rc.refreshReferenceClaims();
    listReferenceClaims.mockResolvedValueOnce({ ok: false, category: "unsafe_path", error: "leaves the workspace", claims: [] });
    await rc.refreshReferenceClaims();
    const r = useJaato.getState().referenceClaims;
    expect(r.status).toBe("error");
    expect(r.error).toContain("leaves the workspace");
    expect(r.rows.map((x) => x.id)).toEqual(["a"]);
  });

  it("an answer for a session this client has left is dropped", async () => {
    let resolve!: (v: unknown) => void;
    listReferenceClaims.mockReturnValue(new Promise((r) => { resolve = r; }));
    const asked = rc.refreshReferenceClaims();
    connected = false;
    useJaato.getState().resetSessionState();
    useJaato.setState({ sessionId: "s2" });
    connected = true;
    resolve({ ok: true, claims: [row("old")], may_curate: true });
    await asked;
    expect(useJaato.getState().referenceClaims.rows).toEqual([]);
  });

  it("a daemon below 1.32 is unsupported, not empty, and is never asked", async () => {
    protocol = "1.31";
    await rc.refreshReferenceClaims();
    expect(useJaato.getState().referenceClaims.status).toBe("unsupported");
    expect(listReferenceClaims).not.toHaveBeenCalled();
  });
});

describe("curation", () => {
  it("a refusal is said in words and the list is re-read", async () => {
    promoteReferenceClaim.mockResolvedValue({ ok: false, category: "not_owner", error: "only the owner" });
    listReferenceClaims.mockResolvedValue({ ok: true, claims: [row("a")], may_curate: false });
    const ok = await rc.promoteReferenceClaim("c-a");
    expect(ok).toBe(false);
    const r = useJaato.getState().referenceClaims;
    expect(r.notice).toEqual({ text: "Only the owner of this workspace can promote or dismiss proposals.", error: true });
    expect(listReferenceClaims).toHaveBeenCalledTimes(1);
    expect(r.busy).toEqual({});
  });

  it("a promotion names the catalog id it became", async () => {
    promoteReferenceClaim.mockResolvedValue({ ok: true, reference_id: "runbook" });
    listReferenceClaims.mockResolvedValue({ ok: true, claims: [], may_curate: true });
    expect(await rc.promoteReferenceClaim("c-a")).toBe(true);
    expect(useJaato.getState().referenceClaims.notice?.text).toBe("Promoted into the catalog as runbook.");
  });

  it("dismiss calls the dismiss verb", async () => {
    dismissReferenceClaim.mockResolvedValue({ ok: true });
    listReferenceClaims.mockResolvedValue({ ok: true, claims: [], may_curate: true });
    await rc.dismissReferenceClaim("c-a");
    expect(dismissReferenceClaim).toHaveBeenCalledWith("c-a");
    expect(useJaato.getState().referenceClaims.notice?.text).toBe("Dismissed.");
  });
});

describe("the origin line", () => {
  it("names who approved it at the prompt", () => {
    expect(rc.describeClaimOrigin({
      generated_by: { agent_id: "writer", provider: "p", model: "m" },
      created_by: "acme:alice",
      witnessed_by: { via: "permission-prompt", method: "user_approved", user: "acme:bob" },
    })).toBe("agent writer (p/m) for acme:alice · approved at the prompt by acme:bob");
  });

  it("says nobody was asked, and invents no agent", () => {
    expect(rc.describeClaimOrigin({ kind: "agent" })).toBe("agent unknown · not approved at a prompt");
    expect(rc.describeClaimOrigin(null)).toBe("origin not recorded");
  });
});

describe("triggers", () => {
  const end = (tool_name: string, extra: Record<string, unknown> = {}) =>
    ({ type: "tool.call_end", tool_name, success: true, ...extra });

  it("a successful proposal in any agent refreshes; a failed one and other tools do not", () => {
    expect(rc.triggersReferenceClaimsRefresh(end("proposeReference"))).toBe(true);
    expect(rc.triggersReferenceClaimsRefresh(end("proposeReference", { success: false }))).toBe(false);
    expect(rc.triggersReferenceClaimsRefresh(end("proposeReference", { is_error_result: true }))).toBe(false);
    expect(rc.triggersReferenceClaimsRefresh(end("store_memory"))).toBe(false);
  });

  it("resolves the hashed wire id a tool call carries", () => {
    expect(rc.triggersReferenceClaimsRefresh(end("t_1234abcd"), { t_1234abcd: "proposeReference" })).toBe(true);
  });

  it("a session.info naming a session refreshes", () => {
    expect(rc.triggersReferenceClaimsRefresh({ type: "session.info", session_id: "s1" })).toBe(true);
    expect(rc.triggersReferenceClaimsRefresh({ type: "session.info" })).toBe(false);
  });
});
