/**
 * The rail's Proposals section: the reference CLAIMS agents proposed in the
 * session's workspace, and -- for the workspace owner -- promoting one into
 * the catalog or dismissing it.
 *
 * An agent's ``proposeReference`` writes a claim under
 * ``.jaato/references-claims/``, never a catalog entry: a confined runner
 * cannot write the catalog.  The daemon lists the claims
 * (``listReferenceClaims``) and does the write (``promoteReferenceClaim`` /
 * ``dismissReferenceClaim``), all three correlated request/answer pairs of
 * protocol 1.33 that print nothing to the transcript.
 *
 * WHO MAY PROMOTE is the daemon's decision: the list carries ``may_curate``
 * and a refused action answers ``category: "not_owner"``.  The panel hides
 * the buttons on the first and reports the second.
 *
 * WHEN THE LIST IS ASKED FOR (the Memories rail's triggers, and one more):
 *   • a session appears, and every ``session.info`` for it;
 *   • a ``proposeReference`` call ends successfully, in ANY agent;
 *   • the window regains focus;
 *   • after each of the rail's own actions.
 * Requests are debounced into one and a stale answer is dropped.  A failed
 * ask is reported and keeps the rows it had.
 */
import { EventTypeValue, MIN_REFERENCE_CURATION_PROTOCOL, isProtocolCompatible, type JaatoClient, type JaatoEvent } from "@jaato/sdk";
import { useJaato } from "@/store/store";
import { getClient, isConnected } from "@/sdk/connection";
import type { ReferenceBundleOption, ReferenceClaimRow, ReferenceClaimsState } from "@/store/types";
import { scheduleReferenceCatalogRefresh } from "@/app/referenceCatalog";

/** The tool whose success means a claim was written. */
export const PROPOSE_TOOL = "proposeReference";

/** How long a burst of triggers waits before becoming one list request. */
export const REFRESH_DEBOUNCE_MS = 150;

type Patch = Partial<ReferenceClaimsState> | ((r: ReferenceClaimsState) => Partial<ReferenceClaimsState>);
const patch = (p: Patch) => useJaato.getState().patchReferenceClaims(p);

/** Whether a daemon speaking ``protocol`` serves the reference-claim verbs. */
export function servesReferenceClaims(protocol: string | null | undefined): boolean {
  return !!protocol && isProtocolCompatible(protocol, MIN_REFERENCE_CURATION_PROTOCOL);
}

/** What a refusal category means to the person who clicked. */
export function refusalText(category: string | undefined, error: string | undefined): string {
  switch (category) {
    case "not_owner": return "Only the owner of this workspace can promote or dismiss proposals.";
    case "not_found": return "That proposal no longer exists -- the list has been refreshed.";
    case "collision": return "A reference with that id is already in the catalog.";
    case "unknown_bundle": return `There is no such bundle: ${error || "the list has been refreshed."}`;
    case "invalid_claim": return `The proposal cannot be promoted: ${error || "it is not valid any more."}`;
    case "unsafe_path": return `Refused: ${error || "a path leaves the workspace."}`;
    case "no_workspace": return "This connection has no workspace.";
    default: return error || "The daemon refused the change.";
  }
}

/**
 * The section's badge value: ``2``.  ``null`` when nothing is proposed or
 * nothing has been listed -- an unasked directory is not an empty one.
 */
export function proposalsSummary(r: Pick<ReferenceClaimsState, "rows">): string | null {
  return r.rows.length ? String(r.rows.length) : null;
}

/**
 * One line naming who proposed a claim and who approved it:
 * ``agent writer (openrouter/m) for acme:alice · approved at the prompt by acme:bob``.
 * Only what the claim recorded; a missing field is left out, never guessed.
 */
export function describeClaimOrigin(origin: Record<string, unknown> | null | undefined): string {
  if (!origin) return "origin not recorded";
  const gen = (origin.generated_by ?? {}) as Record<string, unknown>;
  const parts: string[] = [];
  const binding = [gen.provider, gen.model].filter(Boolean).join("/");
  parts.push(`agent ${gen.agent_id ? String(gen.agent_id) : "unknown"}${binding ? ` (${binding})` : ""}`);
  if (origin.created_by) parts[0] += ` for ${String(origin.created_by)}`;
  const witness = origin.witnessed_by as Record<string, unknown> | undefined;
  if (witness) {
    const who = witness.user ?? witness.approver;
    parts.push(who ? `approved at the prompt by ${String(who)}` : "approved at the prompt");
  } else {
    parts.push("not approved at a prompt");
  }
  return parts.join(" · ");
}

/**
 * The template a proposed page was rendered from, as the daemon last
 * checked it: ``{ text: "from template runbook.md.tpl", edited: false }``,
 * or ``null`` when the claim records no render (a page written some other
 * way, or proposed inline).  ``edited`` means the file no longer matches
 * the render, so the curator is reading something the template did not
 * produce on its own.
 */
export function describeRenderedFrom(
  origin: Record<string, unknown> | null | undefined,
): { text: string; edited: boolean } | null {
  const rendered = origin?.rendered_from as Record<string, unknown> | undefined;
  if (!rendered || typeof rendered !== "object") return null;
  const name = typeof rendered.template === "string" && rendered.template ? ` ${rendered.template}` : "";
  const edited = rendered.edited_after_render === true;
  return { text: `from template${name}${edited ? ", edited since" : ""}`, edited };
}

/**
 * One declared edge in words, as a curator reads it before promoting:
 * ``supersedes adr-1 — rewritten``.  ``supersedes`` is the one with a
 * consequence past the listing (a request for the target is routed to this
 * reference once promoted), so {@link isRoutingLink} lets the panel mark it.
 */
export function describeClaimLink(link: { to: string; rel: string; note?: string }): string {
  const rel = link.rel.replace(/-/g, " ");
  return link.note ? `${rel} ${link.to} — ${link.note}` : `${rel} ${link.to}`;
}

/**
 * The caveat a promotion carries about its destination bundle's vector
 * index, or ``null`` when there is none to report.  ``updated`` / ``clean``
 * / ``none`` (no index) say nothing; anything else means the reference is
 * in the catalog but similarity matching cannot find it yet.
 */
export function reconcileCaveat(reconcile: string | undefined, detail: string | undefined): string | null {
  if (!reconcile || reconcile === "updated" || reconcile === "clean" || reconcile === "none") return null;
  const why = detail ? ` (${detail})` : "";
  switch (reconcile) {
    case "busy": return `The bundle's vector index is being updated by something else, so this reference has no vector yet${why}.`;
    case "unavailable": return `The bundle's vector index was not updated${why}; similarity matching will not find this reference until it is.`;
    default: return `Updating the bundle's vector index failed${why}; similarity matching will not find this reference until it is.`;
  }
}

/** Where a promotion wrote, in words: ``the catalog`` or ``bundle ops``. */
export function promotionTarget(bundle: string | undefined): string {
  return bundle ? `bundle ${bundle}` : "the catalog";
}

/** Whether promoting a claim with this edge changes what OTHER requests get. */
export function isRoutingLink(link: { rel: string }): boolean {
  return link.rel === "supersedes";
}

// ── the list ────────────────────────────────────────────────────────────

let generation = 0;

/** Ask the daemon for the list now, keeping the rows it has while asking. */
export async function refreshReferenceClaims(): Promise<void> {
  if (!isConnected()) return;
  const sessionId = useJaato.getState().sessionId;
  if (!sessionId) return;
  const client = getClient();
  if (!servesReferenceClaims(client.serverProtocolVersion)) {
    patch({ status: "unsupported", error: null });
    return;
  }
  const gen = ++generation;
  patch((r) => (r.status === "idle" || r.status === "unsupported" ? { status: "loading" } : {}));
  const current = () => gen === generation && useJaato.getState().sessionId === sessionId;
  try {
    const answer = await client.listReferenceClaims();
    if (!current()) return;
    if (answer.ok === false) {
      patch({ status: "error", error: refusalText(String(answer.category ?? ""), String(answer.error ?? "")) });
      return;
    }
    const rows = (answer.claims ?? []) as unknown as ReferenceClaimRow[];
    const ids = new Set(rows.map((r) => r.claim_id));
    patch((r) => ({
      rows,
      status: "loaded",
      error: null,
      unreadable: (answer.unreadable ?? []).map(String),
      mayCurate: typeof answer.may_curate === "boolean" ? answer.may_curate : null,
      expanded: r.expanded && ids.has(r.expanded) ? r.expanded : null,
      bundles: ((answer as { bundles?: unknown }).bundles ?? []) as ReferenceBundleOption[],
    }));
  } catch (err) {
    if (!current()) return;
    patch({ status: "error", error: err instanceof Error ? err.message : String(err) });
  }
}

let pending: ReturnType<typeof setTimeout> | null = null;

/** Coalesce a burst of triggers into one {@link refreshReferenceClaims}. */
export function scheduleReferenceClaimsRefresh(delayMs: number = REFRESH_DEBOUNCE_MS): void {
  if (pending) clearTimeout(pending);
  pending = setTimeout(() => {
    pending = null;
    void refreshReferenceClaims();
  }, delayMs);
}

/** Show or hide a claim's details. */
export function toggleReferenceClaim(claimId: string): void {
  patch((r) => ({ expanded: r.expanded === claimId ? null : claimId }));
}

function withoutKey<T>(rec: Record<string, T>, key: string): Record<string, T> {
  if (!(key in rec)) return rec;
  const next = { ...rec };
  delete next[key];
  return next;
}

/** Run one curation verb, report its outcome, and re-list. */
async function curate(
  claimId: string,
  action: "promote" | "dismiss",
  run: (client: JaatoClient) => Promise<{
    ok?: boolean; category?: string; error?: string; reference_id?: string;
    bundle?: string; reconcile?: string; reconcile_detail?: string;
  }>,
): Promise<boolean> {
  patch((r) => ({ busy: { ...r.busy, [claimId]: action }, notice: null }));
  let ok = false;
  try {
    const answer = await run(getClient());
    ok = answer.ok !== false;
    const caveat = action === "promote" ? reconcileCaveat(answer.reconcile, answer.reconcile_detail) : null;
    const done = action === "promote"
      ? `Promoted into ${promotionTarget(answer.bundle)} as ${answer.reference_id || "a reference"}.${caveat ? ` ${caveat}` : ""}`
      : "Dismissed.";
    patch({ notice: ok ? { text: done, warning: !!caveat } : { text: refusalText(answer.category, answer.error), error: true } });
  } catch (err) {
    patch({ notice: { text: err instanceof Error ? err.message : String(err), error: true } });
  } finally {
    patch((r) => ({ busy: withoutKey(r.busy, claimId) }));
  }
  await refreshReferenceClaims();
  // A promotion adds a catalog entry: the References section re-lists.
  if (ok && action === "promote") scheduleReferenceCatalogRefresh();
  return ok;
}

/**
 * Promote: the daemon writes the catalog entry and removes the claim.
 * ``bundle`` names a sub-bundle from the listing (``""``: the catalog root);
 * an indexed one is reconciled by the daemon, which may take a while the
 * first time the session loads its embedding model.
 */
export function promoteReferenceClaim(claimId: string, bundle = ""): Promise<boolean> {
  return curate(claimId, "promote", (c) => c.promoteReferenceClaim(claimId, bundle ? { bundle } : {}));
}

/** Dismiss: the claim file is removed; the catalog is not touched. */
export function dismissReferenceClaim(claimId: string): Promise<boolean> {
  return curate(claimId, "dismiss", (c) => c.dismissReferenceClaim(claimId));
}

// ── triggers ────────────────────────────────────────────────────────────

type WireEvent = { type?: string; session_id?: string; tool_name?: string; success?: boolean; is_error_result?: boolean };

/** Whether ``ev`` means the claims may have changed (or a session appeared). */
export function triggersReferenceClaimsRefresh(ev: WireEvent, toolIdNames: Record<string, string> = {}): boolean {
  switch (ev.type) {
    case EventTypeValue.SESSION_INFO:
      return !!ev.session_id;
    case EventTypeValue.TOOL_CALL_END: {
      const name = (ev.tool_name && toolIdNames[ev.tool_name]) || ev.tool_name || "";
      return name === PROPOSE_TOOL && ev.success !== false && ev.is_error_result !== true;
    }
    default:
      return false;
  }
}

/** Wire the event triggers onto one client (called beside ``wireMemoryRail``). */
export function wireReferenceClaimsRail(client: JaatoClient): void {
  client.subscribeAll((raw: JaatoEvent) => {
    if (triggersReferenceClaimsRefresh(raw as unknown as WireEvent, useJaato.getState().toolIdNames)) {
      scheduleReferenceClaimsRefresh();
    }
  });
}

let installed = false;

/** A session appearing, and the window regaining focus.  Idempotent; self-installed. */
export function installReferenceClaimsRail(): void {
  if (installed) return;
  installed = true;
  let lastSession: string | null | undefined;
  useJaato.subscribe((st) => {
    if (st.sessionId === lastSession) return;
    lastSession = st.sessionId;
    if (st.sessionId && isConnected()) scheduleReferenceClaimsRefresh(0);
  });
  if (typeof window !== "undefined") {
    window.addEventListener("focus", () => {
      if (isConnected() && useJaato.getState().sessionId) scheduleReferenceClaimsRefresh();
    });
  }
}

installReferenceClaimsRail();
