/**
 * The rail's Memories section (#1232): what the session's memory store
 * holds, raw and curated, workspace and global -- and, for the workspace
 * owner, the curation the TUI does with ``memory edit``.
 *
 * The data and every action live in ``app/memories.ts``; this component
 * draws the store slice and calls those functions.  What it shows:
 *
 *   • one row per memory -- a maturity glyph, the description, the tags,
 *     which TIER it lives in (``workspace`` / ``global``), and an
 *     **unvetted** marker on a raw one, because a memory nobody approved is
 *     the one a person needs to look at;
 *   • rows written or retrieved in THIS session are highlighted and say so,
 *     and a toggle narrows the list to them;
 *   • expanding a row fetches its content (the list carries none);
 *   • Approve / Dismiss / Edit / Remove -- drawn only when the daemon says
 *     this connection may curate (``may_curate``), which is the owner gate
 *     the daemon also ENFORCES.  Hiding the buttons is courtesy; refusing
 *     is the daemon's.
 *
 * Every control is a real ``button`` with an accessible name that contains
 * its visible word (the lesson ``SessionRow`` records), and none is
 * hover-gated -- a touch screen has no hover.
 */
import { useEffect, useState } from "react";
import { useJaato } from "@/store/store";
import type { MemoryRow } from "@/store/types";
import {
  CURATED,
  approveMemory,
  cancelEditMemory,
  dismissMemory,
  refreshMemories,
  removeMemory,
  saveEditMemory,
  setMemoriesThisSessionOnly,
  setMemoryDraft,
  startEditMemory,
  toggleMemory,
  visibleMemories,
} from "@/app/memories";

/** The maturity glyph: filled once a person approved it. */
export function maturityGlyph(maturity: string): string {
  switch (maturity) {
    case "validated": return "●";
    case "escalated": return "▲";
    case "dismissed": return "×";
    default: return "○";
  }
}

/** Who approved it, from ``curated_by``: the person, else the model, else "unrecorded". */
export function curatorLabel(curatedBy: Record<string, unknown> | null | undefined): string | null {
  if (!curatedBy) return null;
  const who = curatedBy.user ?? curatedBy.agent ?? curatedBy.model;
  return who ? String(who) : "an approver not recorded";
}

function day(ts: string | null | undefined): string {
  if (!ts) return "";
  const d = new Date(ts);
  return Number.isNaN(d.getTime()) ? String(ts) : d.toISOString().slice(0, 10);
}

function MemoryDetails({ row }: { row: MemoryRow }) {
  const detail = useJaato((s) => s.memories.details[row.id]);
  const model = row.generated_by ? String(row.generated_by.model ?? row.generated_by.provider ?? "") : "";
  const curator = curatorLabel(row.curated_by);
  return (
    <div className="pl-6 pb-2 flex flex-col gap-1.5">
      {!detail || detail.state === "loading" ? (
        <p className="m-0 text-text-muted">Loading…</p>
      ) : detail.state === "error" ? (
        <p className="m-0 text-error">{detail.message}</p>
      ) : (
        <>
          <pre className="m-0 whitespace-pre-wrap font-mono text-[12px]" data-testid="memory-content">{detail.content}</pre>
          {detail.evidence && <p className="m-0 text-[12px] text-text-muted">Evidence: {detail.evidence}</p>}
        </>
      )}
      <p className="m-0 font-mono text-[11px] text-text-muted">
        {[
          row.timestamp ? `written ${day(row.timestamp)}` : "",
          row.source_agent ? `by ${row.source_agent}` : "",
          model ? `model ${model}` : "",
          row.usage_count ? `used ${row.usage_count}×` : "",
          row.last_accessed ? `last ${day(row.last_accessed)}` : "",
          curator ? `approved by ${curator}` : "",
        ].filter(Boolean).join(" · ")}
      </p>
    </div>
  );
}

function MemoryEditForm({ id }: { id: string }) {
  const draft = useJaato((s) => s.memories.editing[id]);
  const busy = useJaato((s) => s.memories.busy[id]);
  if (!draft) return null;
  const field = "w-full border hairline bg-bg px-2 py-1 font-mono text-[12px]";
  return (
    <form
      className="pl-6 pb-2.5 flex flex-col gap-1.5"
      aria-label={`Edit memory ${id}`}
      onSubmit={(e) => { e.preventDefault(); void saveEditMemory(id); }}
    >
      <label className="kicker kicker-muted text-[11px]">Description
        <input className={field} value={draft.description} onChange={(e) => setMemoryDraft(id, { description: e.target.value })} />
      </label>
      <label className="kicker kicker-muted text-[11px]">Content
        <textarea className={field} rows={5} value={draft.content} onChange={(e) => setMemoryDraft(id, { content: e.target.value })} />
      </label>
      <label className="kicker kicker-muted text-[11px]">Tags (comma-separated)
        <input className={field} value={draft.tags} onChange={(e) => setMemoryDraft(id, { tags: e.target.value })} />
      </label>
      <div className="flex gap-2">
        <button type="submit" disabled={!!busy} className="btn btn-sm btn-steel">Save</button>
        <button type="button" onClick={() => cancelEditMemory(id)} className="btn btn-sm btn-quiet">Cancel</button>
      </div>
    </form>
  );
}

export function MemoryRowView({ row, mayCurate }: { row: MemoryRow; mayCurate: boolean }) {
  const expanded = useJaato((s) => s.memories.expanded === row.id);
  const editing = useJaato((s) => !!s.memories.editing[row.id]);
  const busy = useJaato((s) => s.memories.busy[row.id]);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const raw = row.maturity === "raw";
  const here = row.written_this_session ? "written in this session" : row.retrieved_this_session ? "used in this session" : "";
  return (
    <div className={`border-t hairline ${here ? "border-l-2 border-l-steel pl-1.5" : ""}`} data-testid="memory-row" data-memory-id={row.id}>
      <div className="flex gap-2 py-2 items-start">
        <span className={`pt-0.5 ${CURATED.has(row.maturity) ? "text-success" : raw ? "text-warning" : "text-text-muted"}`} title={row.maturity} aria-hidden="true">{maturityGlyph(row.maturity)}</span>
        <div className="min-w-0 flex-1">
          <span className="block text-[13px]">{row.description}</span>
          <span className="block font-mono text-[11px] text-text-muted truncate">
            {raw && <span className="text-warning uppercase tracking-[0.08em] mr-1.5">unvetted</span>}
            <span className="mr-1.5">{row.tier}</span>
            {row.tags.map((t) => `#${t}`).join(" ")}
          </span>
          {here && <span className="block text-[11px] text-steel">{here}</span>}
        </div>
        <button
          type="button"
          onClick={() => { void toggleMemory(row.id); }}
          aria-expanded={expanded}
          aria-label={`${expanded ? "Hide" : "Show"} memory ${row.description}`}
          className="btn btn-sm btn-quiet self-center shrink-0"
        >
          {expanded ? "Hide" : "Show"}
        </button>
      </div>
      {expanded && <MemoryDetails row={row} />}
      {mayCurate && (
        <div className="pl-6 pb-2 flex flex-wrap gap-1.5" aria-label={`Curate memory ${row.id}`}>
          {!CURATED.has(row.maturity) && (
            <button type="button" disabled={!!busy} onClick={() => { void approveMemory(row.id); }} className="btn btn-sm btn-steel" aria-label={`Approve memory ${row.description}`}>Approve</button>
          )}
          <button type="button" disabled={!!busy} onClick={() => { void dismissMemory(row.id); }} className="btn btn-sm btn-quiet" aria-label={`Dismiss memory ${row.description}`}>Dismiss</button>
          <button type="button" disabled={!!busy || editing} onClick={() => { void startEditMemory(row.id); }} className="btn btn-sm btn-quiet" aria-label={`Edit memory ${row.description}`}>Edit</button>
          {confirmRemove ? (
            <>
              <button type="button" disabled={!!busy} onClick={() => { setConfirmRemove(false); void removeMemory(row.id); }} className="btn btn-sm btn-quiet text-error" aria-label={`Confirm remove memory ${row.description}`}>Confirm remove</button>
              <button type="button" onClick={() => setConfirmRemove(false)} className="btn btn-sm btn-quiet">Keep</button>
            </>
          ) : (
            <button type="button" disabled={!!busy} onClick={() => setConfirmRemove(true)} className="btn btn-sm btn-quiet hover:text-error" aria-label={`Remove memory ${row.description}`}>Remove</button>
          )}
          {busy && <span className="font-mono text-[11px] text-text-muted self-center">{busy}…</span>}
        </div>
      )}
      {editing && <MemoryEditForm id={row.id} />}
    </div>
  );
}

/**
 * The section body.  Opening the section asks when nothing has been asked
 * yet -- the list is normally fetched on attach, and this covers a daemon
 * that had not answered by the time the section opened.
 */
export function MemoriesPanel() {
  const m = useJaato((s) => s.memories);
  const sessionId = useJaato((s) => s.sessionId);
  useEffect(() => {
    if (sessionId && useJaato.getState().memories.status === "idle") void refreshMemories();
  }, [sessionId]);
  const rows = visibleMemories(m);
  const mayCurate = m.mayCurate === true;
  if (!sessionId) return <p className="m-0 px-3.5 py-2 text-[13px] text-text-muted">No session attached.</p>;
  if (m.status === "unsupported") {
    return <p className="m-0 px-3.5 py-2 text-[13px] text-text-muted">This daemon is older than protocol 1.22 and cannot list memories quietly. The <span className="font-mono">memory list</span> command still works.</p>;
  }
  return (
    <div className="px-3.5 py-2 text-[13px]">
      <div className="flex items-center gap-2 pb-1.5">
        <label className="flex items-center gap-1.5 text-[12px]">
          <input type="checkbox" checked={m.thisSessionOnly} onChange={(e) => setMemoriesThisSessionOnly(e.target.checked)} />
          This session only
        </label>
        <span className="flex-1" />
        <button type="button" onClick={() => { void refreshMemories(); }} className="btn btn-sm btn-quiet" aria-label="Refresh memories">Refresh</button>
      </div>
      {m.status === "error" && <p className="m-0 py-1 text-error" role="alert">Could not read the memory store: {m.error}</p>}
      {m.notice && <p className={`m-0 py-1 ${m.notice.error ? "text-error" : "text-text-muted"}`} role="status">{m.notice.text}</p>}
      {m.mayCurate === false && m.rows.length > 0 && (
        <p className="m-0 py-1 text-[11px] text-text-muted">Only the owner of this workspace can approve, edit or remove these.</p>
      )}
      {m.status === "loading" && m.rows.length === 0 ? (
        <p className="m-0 py-2 text-text-muted">Loading…</p>
      ) : rows.length === 0 ? (
        <p className="m-0 py-2 text-text-muted">{m.thisSessionOnly && m.rows.length > 0 ? "Nothing written or used in this session yet." : m.status === "loaded" ? "Nothing remembered yet." : ""}</p>
      ) : (
        rows.map((row) => <MemoryRowView key={row.id} row={row} mayCurate={mayCurate} />)
      )}
    </div>
  );
}
