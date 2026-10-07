/**
 * The rail's References section: the session workspace's reference CATALOG
 * with its typed links both ways, and -- for the workspace owner -- editing
 * one reference's links.
 *
 * A link (``{to, rel, note?}``) reaches the catalog by hand or by promoting
 * an agent's proposal.  Changing it afterwards needs the daemon, because a
 * confined runner cannot write ``.jaato/references/``: ``listReferenceCatalog``
 * reads it and ``updateReferenceLinks`` rewrites one reference's ``links``
 * key, both correlated request/answer pairs of protocol 1.33 that print
 * nothing to the transcript.
 *
 * WHO MAY EDIT is the daemon's decision (``may_curate``, and a refused save
 * answers ``not_owner``), as in the Proposals section.  A save sends the
 * WHOLE new list; the daemon validates it (an unknown ``rel``, an edge to
 * the reference itself) and answers with the links as written and any
 * warnings (a target the catalog does not hold, a second ``supersedes`` of
 * one target), which do not block the save.
 *
 * WHEN THE LIST IS ASKED FOR: a session appearing, a successful promotion
 * (the Proposals section calls {@link scheduleReferenceCatalogRefresh}),
 * the window regaining focus, and after each save.  A stale answer is
 * dropped and a failed ask keeps the rows it had.
 */
import { MIN_REFERENCE_CURATION_PROTOCOL, isProtocolCompatible } from "@jaato/sdk";
import { useJaato } from "@/store/store";
import { getClient, isConnected } from "@/sdk/connection";
import type { ReferenceCatalogRow, ReferenceCatalogState, ReferenceClaimLink } from "@/store/types";

/** The relations a link may declare, in the order the editor offers them. */
// Mirrors links.LINK_RELS (jaato-server references plugin); the daemon refuses any other.
export const LINK_RELS = ["depends-on", "elaborates", "see-also", "supersedes", "contradicts"] as const;

/** What each relation does, for the editor's hint. */
export const REL_EFFECT: Record<(typeof LINK_RELS)[number], string> = {
  "depends-on": "selecting this one also brings in the target",
  elaborates: "the target is offered beside this one",
  "see-also": "the target is offered beside this one (a neighbouring concern)",
  supersedes: "requests for the target get this one instead",
  contradicts: "listed only",
};

/** How long a burst of triggers waits before becoming one list request. */
export const REFRESH_DEBOUNCE_MS = 150;

type Patch = Partial<ReferenceCatalogState> | ((r: ReferenceCatalogState) => Partial<ReferenceCatalogState>);
const patch = (p: Patch) => useJaato.getState().patchReferenceCatalog(p);

/** Whether a daemon speaking ``protocol`` serves the catalog verbs. */
export function servesReferenceCatalog(protocol: string | null | undefined): boolean {
  return !!protocol && isProtocolCompatible(protocol, MIN_REFERENCE_CURATION_PROTOCOL);
}

/** What a refusal category means to the person who clicked Save. */
export function linksRefusalText(category: string | undefined, error: string | undefined): string {
  switch (category) {
    case "not_owner": return "Only the owner of this workspace can change references' links.";
    case "invalid_links": return `Not saved: ${error || "a link is not valid."}`;
    case "not_found": return "That reference is no longer in the catalog -- the list has been refreshed.";
    case "ambiguous": return `Not saved: ${error || "the id is declared in two files."}`;
    case "unsafe_path": return `Refused: ${error || "a path leaves the workspace."}`;
    case "no_workspace": return "This connection has no workspace.";
    default: return error || "The daemon refused the change.";
  }
}

/** The section's badge: how many references hold a dangling edge, else the count. */
export function referencesSummary(r: Pick<ReferenceCatalogState, "rows">): string | null {
  if (!r.rows.length) return null;
  const dangling = r.rows.filter((row) => row.links.some((l) => l.dangling)).length;
  return dangling ? "!" : String(r.rows.length);
}

/**
 * Why a draft cannot be saved, checked before anything is sent (the daemon
 * checks again): an empty target, a target equal to the reference, a
 * duplicate ``(to, rel)``.  ``null`` when the draft may be saved.
 */
export function draftProblem(id: string, links: ReferenceClaimLink[]): string | null {
  const seen = new Set<string>();
  for (const l of links) {
    const to = l.to.trim();
    if (!to) return "Every link needs a target id.";
    if (/\s/.test(to)) return `'${to}' is not one id.`;
    if (to === id) return "A reference cannot link to itself.";
    const key = `${l.rel}:${to}`;
    if (seen.has(key)) return `'${l.rel} ${to}' is listed twice.`;
    seen.add(key);
  }
  return null;
}

/** The draft as the wire takes it: trimmed, an empty note left out. */
export function draftToWire(links: ReferenceClaimLink[]): ReferenceClaimLink[] {
  return links.map((l) => {
    const note = (l.note ?? "").trim();
    return note ? { to: l.to.trim(), rel: l.rel, note } : { to: l.to.trim(), rel: l.rel };
  });
}

// ── the list ────────────────────────────────────────────────────────────

let generation = 0;

/** Ask the daemon for the catalog now, keeping the rows it has while asking. */
export async function refreshReferenceCatalog(): Promise<void> {
  if (!isConnected()) return;
  const sessionId = useJaato.getState().sessionId;
  if (!sessionId) return;
  const client = getClient();
  if (!servesReferenceCatalog(client.serverProtocolVersion)) {
    patch({ status: "unsupported", error: null });
    return;
  }
  const gen = ++generation;
  patch((r) => (r.status === "idle" || r.status === "unsupported" ? { status: "loading" } : {}));
  const current = () => gen === generation && useJaato.getState().sessionId === sessionId;
  try {
    const answer = await client.listReferenceCatalog();
    if (!current()) return;
    if (answer.ok === false) {
      patch({ status: "error", error: linksRefusalText(String(answer.category ?? ""), String(answer.error ?? "")) });
      return;
    }
    const rows = (answer.references ?? []) as unknown as ReferenceCatalogRow[];
    const ids = new Set(rows.map((r) => r.id));
    patch((r) => ({
      rows,
      status: "loaded",
      error: null,
      unreadable: (answer.unreadable ?? []).map(String),
      mayCurate: typeof answer.may_curate === "boolean" ? answer.may_curate : null,
      expanded: r.expanded && ids.has(r.expanded) ? r.expanded : null,
      editing: r.editing && ids.has(r.editing.id) ? r.editing : null,
    }));
  } catch (err) {
    if (!current()) return;
    patch({ status: "error", error: err instanceof Error ? err.message : String(err) });
  }
}

let pending: ReturnType<typeof setTimeout> | null = null;

/** Coalesce a burst of triggers into one {@link refreshReferenceCatalog}. */
export function scheduleReferenceCatalogRefresh(delayMs: number = REFRESH_DEBOUNCE_MS): void {
  if (pending) clearTimeout(pending);
  pending = setTimeout(() => {
    pending = null;
    void refreshReferenceCatalog();
  }, delayMs);
}

/** Show or hide a reference's details. */
export function toggleReference(id: string): void {
  patch((r) => ({ expanded: r.expanded === id ? null : id }));
}

// ── editing ─────────────────────────────────────────────────────────────

/** Start editing ``row``'s links from what the catalog holds. */
export function startEditingLinks(row: ReferenceCatalogRow): void {
  patch({
    editing: { id: row.id, links: row.links.map((l) => (l.note ? { to: l.to, rel: l.rel, note: l.note } : { to: l.to, rel: l.rel })) },
    expanded: row.id,
    notice: null,
  });
}

/** Leave the editor without saving. */
export function cancelEditingLinks(): void {
  patch({ editing: null });
}

/** Replace the draft (the panel's one writer of it). */
export function setDraftLinks(links: ReferenceClaimLink[]): void {
  patch((r) => (r.editing ? { editing: { id: r.editing.id, links } } : {}));
}

/**
 * Save the draft: the daemon replaces the reference's ``links`` and answers
 * with what it wrote.  The editor closes on success and stays open with the
 * reason on a refusal, so nothing typed is lost.
 */
export async function saveDraftLinks(): Promise<boolean> {
  const editing = useJaato.getState().referenceCatalog.editing;
  if (!editing) return false;
  const problem = draftProblem(editing.id, editing.links);
  if (problem) {
    patch({ notice: { text: problem, error: true } });
    return false;
  }
  const id = editing.id;
  patch((r) => ({ busy: { ...r.busy, [id]: true }, notice: null }));
  let ok = false;
  try {
    const answer = await getClient().updateReferenceLinks(id, draftToWire(editing.links));
    ok = answer.ok !== false;
    const warnings = (answer.warnings ?? []).map(String);
    patch(ok
      ? { editing: null, notice: { text: warnings.length ? `Saved. ${warnings.join(" ")}` : "Saved.", warning: warnings.length > 0 } }
      : { notice: { text: linksRefusalText(answer.category, answer.error), error: true } });
  } catch (err) {
    patch({ notice: { text: err instanceof Error ? err.message : String(err), error: true } });
  } finally {
    patch((r) => {
      const busy = { ...r.busy };
      delete busy[id];
      return { busy };
    });
  }
  await refreshReferenceCatalog();
  return ok;
}

// ── triggers ────────────────────────────────────────────────────────────

let installed = false;

/** A session appearing, and the window regaining focus.  Idempotent; self-installed. */
export function installReferenceCatalogRail(): void {
  if (installed) return;
  installed = true;
  let lastSession: string | null | undefined;
  useJaato.subscribe((st) => {
    if (st.sessionId === lastSession) return;
    lastSession = st.sessionId;
    if (st.sessionId && isConnected()) scheduleReferenceCatalogRefresh(0);
  });
  if (typeof window !== "undefined") {
    window.addEventListener("focus", () => {
      if (isConnected() && useJaato.getState().sessionId) scheduleReferenceCatalogRefresh();
    });
  }
}

installReferenceCatalogRail();
