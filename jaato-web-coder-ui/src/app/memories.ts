/**
 * The rail's Memories section (#1232): what the session's memory store
 * holds, and -- for the workspace owner -- approving, dismissing, editing
 * and removing what is in it.
 *
 * Everything goes through the daemon's QUIET memory verbs (protocol 1.22):
 * ``listMemories`` / ``getMemory`` / ``updateMemory`` / ``deleteMemory``,
 * request/result pairs that write nothing to the transcript.  The ``memory``
 * command would print a listing into the conversation every time the rail
 * refreshed, and ``memory edit`` opens ``$EDITOR`` on the daemon's host,
 * which a browser cannot drive.
 *
 * WHO MAY CHANGE WHAT is the daemon's decision, never this module's: the
 * list answer carries ``may_curate`` and a refused mutation answers
 * ``category: "not_owner"``.  The panel hides the buttons on the first and
 * reports the second, but a client that showed them anyway would change
 * nothing.
 *
 * WHEN THE LIST IS ASKED FOR, per the issue:
 *   • a session appears (attach, create) and every ``session.info`` for it,
 *     which covers the re-attach after a reconnect;
 *   • a ``store_memory`` / ``update_memory`` / ``delete_memory`` call ends
 *     successfully, in ANY agent -- a subagent's write lands in the same
 *     store;
 *   • the ``memory`` command pushes its unsolicited ``memory.list`` (a
 *     ``memory delete`` typed in the composer changes the store too);
 *   • the window regains focus;
 *   • after each of the rail's own actions.
 * Requests are debounced into one and a stale answer (for a session this
 * client has since left, or overtaken by a newer ask) is dropped.
 *
 * A failed ask is REPORTED and keeps the rows it had: an empty list is how
 * "nothing remembered" looks, and a store that could not be read is not
 * that.
 */
import { EventTypeValue, MIN_MEMORY_VERBS_PROTOCOL, isProtocolCompatible, type JaatoClient, type JaatoEvent } from "@jaato/sdk";
import { useJaato } from "@/store/store";
import { getClient, isConnected } from "@/sdk/connection";
import type { MemoriesState, MemoryDraft, MemoryRow } from "@/store/types";

/** The memory plugin's tools whose success means the store changed. */
export const MEMORY_WRITE_TOOLS: ReadonlySet<string> = new Set(["store_memory", "update_memory", "delete_memory"]);

/** How long a burst of triggers waits before becoming one list request. */
export const REFRESH_DEBOUNCE_MS = 150;

/** Maturities a person has approved -- the rest are the curator's queue. */
export const CURATED = new Set(["validated", "escalated"]);

type Patch = Partial<MemoriesState> | ((m: MemoriesState) => Partial<MemoriesState>);
const patch = (p: Patch) => useJaato.getState().patchMemories(p);

/** Whether a daemon speaking ``protocol`` serves the memory verbs. */
export function servesMemories(protocol: string | null | undefined): boolean {
  return !!protocol && isProtocolCompatible(protocol, MIN_MEMORY_VERBS_PROTOCOL);
}

/** What a refusal category means to the person who clicked. */
export function refusalText(category: string | undefined, error: string | undefined): string {
  switch (category) {
    case "not_owner": return "Only the owner of this workspace can change its memories.";
    case "not_found": return "That memory no longer exists -- the list has been refreshed.";
    case "no_plugin": return "This session does not enable the memory plugin.";
    case "no_session": return "No session is attached.";
    case "runner_unreachable": return `The session's runner did not answer${error ? `: ${error}` : "."}`;
    default: return error || "The daemon refused the change.";
  }
}

/** Rows the "this session" filter keeps: written or retrieved here. */
export function visibleMemories(m: Pick<MemoriesState, "rows" | "thisSessionOnly">): MemoryRow[] {
  if (!m.thisSessionOnly) return m.rows;
  return m.rows.filter((r) => r.written_this_session || r.retrieved_this_session);
}

/**
 * The section header's value: ``12 memories · 3 unvetted``.  ``null`` when
 * nothing has been listed -- an unasked store is not an empty one.
 */
export function memoriesSummary(m: Pick<MemoriesState, "rows" | "status">): string | null {
  if (m.status !== "loaded" && m.rows.length === 0) return null;
  const n = m.rows.length;
  const raw = m.rows.filter((r) => r.maturity === "raw").length;
  const head = `${n} ${n === 1 ? "memory" : "memories"}`;
  return raw ? `${head} · ${raw} unvetted` : head;
}

// ── the list ────────────────────────────────────────────────────────────

let generation = 0;

/**
 * Ask the daemon for the list now.  Keeps the rows it has while asking,
 * so a refresh never blanks the section.
 */
export async function refreshMemories(): Promise<void> {
  if (!isConnected()) return;
  const sessionId = useJaato.getState().sessionId;
  if (!sessionId) return;
  const client = getClient();
  if (!servesMemories(client.serverProtocolVersion)) {
    patch({ status: "unsupported", error: null });
    return;
  }
  const gen = ++generation;
  patch((m) => (m.status === "idle" || m.status === "unsupported" ? { status: "loading" } : {}));
  const current = () => gen === generation && useJaato.getState().sessionId === sessionId;
  try {
    const answer = await client.listMemories();
    if (!current()) return;
    if (answer.ok === false) {
      patch({ status: "error", error: refusalText(String(answer.category ?? ""), String(answer.error ?? "")) });
      return;
    }
    const rows = (answer.memories ?? []) as unknown as MemoryRow[];
    const ids = new Set(rows.map((r) => r.id));
    patch((m) => ({
      rows,
      status: "loaded",
      error: null,
      mayCurate: typeof answer.may_curate === "boolean" ? answer.may_curate : null,
      expanded: m.expanded && ids.has(m.expanded) ? m.expanded : null,
    }));
  } catch (err) {
    if (!current()) return;
    patch({ status: "error", error: err instanceof Error ? err.message : String(err) });
  }
}

let pending: ReturnType<typeof setTimeout> | null = null;

/** Coalesce a burst of triggers into one {@link refreshMemories}. */
export function scheduleMemoryRefresh(delayMs: number = REFRESH_DEBOUNCE_MS): void {
  if (pending) clearTimeout(pending);
  pending = setTimeout(() => {
    pending = null;
    void refreshMemories();
  }, delayMs);
}

// ── one row ─────────────────────────────────────────────────────────────

/** Show or hide a row's content, fetching it the first time. */
export async function toggleMemory(id: string): Promise<void> {
  const m = useJaato.getState().memories;
  if (m.expanded === id) {
    patch({ expanded: null });
    return;
  }
  patch({ expanded: id });
  if (m.details[id]?.state === "loaded") return;
  await loadDetail(id);
}

async function loadDetail(id: string): Promise<void> {
  patch((m) => ({ details: { ...m.details, [id]: { state: "loading" } } }));
  try {
    const answer = await getClient().getMemory(id);
    const memory = (answer.memory ?? null) as { content?: string; evidence?: string | null } | null;
    const detail = answer.ok === false || !memory
      ? { state: "error" as const, message: refusalText(String(answer.category ?? ""), String(answer.error ?? "")) }
      : { state: "loaded" as const, content: String(memory.content ?? ""), evidence: memory.evidence ?? null };
    patch((m) => ({ details: { ...m.details, [id]: detail } }));
  } catch (err) {
    patch((m) => ({ details: { ...m.details, [id]: { state: "error", message: err instanceof Error ? err.message : String(err) } } }));
  }
}

function withoutKey<T>(rec: Record<string, T>, key: string): Record<string, T> {
  if (!(key in rec)) return rec;
  const next = { ...rec };
  delete next[key];
  return next;
}

/**
 * Run one mutation, mark the row busy while it runs, report the outcome,
 * and re-list -- the daemon's store is the truth, not a local splice.
 */
async function mutate(
  id: string,
  action: string,
  run: (client: JaatoClient) => Promise<{ ok?: boolean; category?: string; error?: string }>,
  done: string,
): Promise<boolean> {
  patch((m) => ({ busy: { ...m.busy, [id]: action }, notice: null }));
  let ok = false;
  try {
    const answer = await run(getClient());
    ok = answer.ok !== false;
    patch({ notice: ok ? { text: done } : { text: refusalText(answer.category, answer.error), error: true } });
  } catch (err) {
    patch({ notice: { text: err instanceof Error ? err.message : String(err), error: true } });
  } finally {
    patch((m) => ({ busy: withoutKey(m.busy, id), details: withoutKey(m.details, id) }));
  }
  await refreshMemories();
  return ok;
}

/** Approve: the memory leaves the queue and becomes retrievable (``validated``). */
export function approveMemory(id: string): Promise<boolean> {
  return mutate(id, "approve", (c) => c.approveMemory(id), "Approved.");
}

/** Dismiss: rejected, and gone from the store -- the queue keeps no trace. */
export function dismissMemory(id: string): Promise<boolean> {
  return mutate(id, "dismiss", (c) => c.dismissMemory(id), "Dismissed.");
}

/** Remove, through the plugin's own delete path. */
export function removeMemory(id: string): Promise<boolean> {
  return mutate(id, "remove", (c) => c.deleteMemory(id), "Removed.");
}

// ── the edit form ───────────────────────────────────────────────────────

/** Open the edit form, seeded from the row and its content (fetched when needed). */
export async function startEditMemory(id: string): Promise<void> {
  const st = useJaato.getState().memories;
  const row = st.rows.find((r) => r.id === id);
  if (!row) return;
  if (st.details[id]?.state !== "loaded") await loadDetail(id);
  const detail = useJaato.getState().memories.details[id];
  const content = detail?.state === "loaded" ? detail.content : "";
  patch((m) => ({ editing: { ...m.editing, [id]: { description: row.description, content, tags: row.tags.join(", ") } } }));
}

export function setMemoryDraft(id: string, draft: Partial<MemoryDraft>): void {
  patch((m) => (m.editing[id] ? { editing: { ...m.editing, [id]: { ...m.editing[id]!, ...draft } } } : {}));
}

export function cancelEditMemory(id: string): void {
  patch((m) => ({ editing: withoutKey(m.editing, id) }));
}

/** ``"alpha, beta ,,gamma"`` -> ``["alpha", "beta", "gamma"]``. */
export function parseTags(text: string): string[] {
  return text.split(",").map((t) => t.trim()).filter(Boolean);
}

/**
 * Send the form.  Only the fields that CHANGED are sent, so an edit to the
 * description does not rewrite the content with whatever the form happened
 * to hold.  The form stays open on a refusal, with the draft, because what
 * was typed is then the only copy.
 */
export async function saveEditMemory(id: string): Promise<boolean> {
  const st = useJaato.getState().memories;
  const draft = st.editing[id];
  const row = st.rows.find((r) => r.id === id);
  if (!draft || !row) return false;
  const detail = st.details[id];
  const fields: { description?: string; content?: string; tags?: string[] } = {};
  if (draft.description !== row.description) fields.description = draft.description;
  const tags = parseTags(draft.tags);
  if (tags.join("\u0000") !== row.tags.join("\u0000")) fields.tags = tags;
  if (detail?.state !== "loaded" || draft.content !== detail.content) fields.content = draft.content;
  if (Object.keys(fields).length === 0) {
    cancelEditMemory(id);
    return true;
  }
  const ok = await mutate(id, "edit", (c) => c.updateMemory(id, fields), "Saved.");
  if (ok) cancelEditMemory(id);
  return ok;
}

export function setMemoriesThisSessionOnly(on: boolean): void {
  patch({ thisSessionOnly: on });
}

// ── triggers ────────────────────────────────────────────────────────────

type WireEvent = { type?: string; session_id?: string; tool_name?: string; success?: boolean; is_error_result?: boolean; request_id?: string };

/** Whether ``ev`` means the store may have changed (or a session appeared). */
export function triggersMemoryRefresh(ev: WireEvent, toolIdNames: Record<string, string> = {}): boolean {
  switch (ev.type) {
    case EventTypeValue.SESSION_INFO:
      return !!ev.session_id;
    case EventTypeValue.TOOL_CALL_END: {
      const name = (ev.tool_name && toolIdNames[ev.tool_name]) || ev.tool_name || "";
      return MEMORY_WRITE_TOOLS.has(name) && ev.success !== false && ev.is_error_result !== true;
    }
    case EventTypeValue.MEMORY_LIST:
      // The ``memory`` command's unsolicited push.  Our own answers carry a
      // request_id and are consumed by the call that asked.
      return !ev.request_id;
    default:
      return false;
  }
}

/**
 * Wire the refresh triggers onto one client.  Called by the connection
 * adapter beside ``wireDownloadTool``, so each client gets its own
 * subscription and a replaced client leaves none behind.
 */
export function wireMemoryRail(client: JaatoClient): void {
  client.subscribeAll((raw: JaatoEvent) => {
    if (triggersMemoryRefresh(raw as unknown as WireEvent, useJaato.getState().toolIdNames)) scheduleMemoryRefresh();
  });
}

let installed = false;

/**
 * The client-independent triggers: a session appearing in the store, and
 * the window regaining focus.  Idempotent; self-installed on import.
 */
export function installMemoryRail(): void {
  if (installed) return;
  installed = true;
  let lastSession: string | null | undefined;
  useJaato.subscribe((st) => {
    if (st.sessionId === lastSession) return;
    lastSession = st.sessionId;
    if (st.sessionId && isConnected()) scheduleMemoryRefresh(0);
  });
  if (typeof window !== "undefined") {
    window.addEventListener("focus", () => {
      if (isConnected() && useJaato.getState().sessionId) scheduleMemoryRefresh();
    });
  }
}

installMemoryRail();
