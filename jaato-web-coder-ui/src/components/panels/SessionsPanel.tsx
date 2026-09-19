/**
 * Which of my sessions needs me — and the note that says why.
 *
 * The picker is not a session list: ``SessionScreen`` shows it only when no
 * session is held (which is why a Detach and a reconnect used to open
 * straight into a dead transcript), and the ``session list`` command renders
 * inert monospace text into the transcript, where it scrolls away carrying
 * no affordances.  So there was no surface where you could survey your
 * sessions and act on them WHILE working in one, which is exactly the
 * reported situation: several running, wanting to know which needs you.
 *
 * The rail is the right home -- persistent, resizable, and already the place
 * for "a list of things with per-row actions" (Files).  Its data is free:
 * the store holds ``sessions``, refreshed by ``SessionListEvent``, and
 * ``sessionListSilent`` exists precisely to refresh without printing a
 * listing nobody typed.
 *
 * The listing is CROSS-WORKSPACE (the store's own comment says so), so a row
 * carries its workspace: two ``20260917_090428``-shaped ids from different
 * trees would otherwise read as the same session.
 */
import { useEffect, useState } from "react";
import { ensureSessions } from "@/app/actions";
import { noteFirstLine } from "@/app/notes";
import { useNote } from "@/app/useNote";
import { useJaato } from "@/store/store";
import type { SessionSummary } from "@/protocol/sessions";
import { NoteEditor } from "./NoteEditor";

/** The last path segment, which is what tells two same-shaped ids apart. */
export function workspaceLabel(workspacePath: string): string {
  const parts = workspacePath.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] ?? "";
}

export interface SessionRowProps {
  sess: SessionSummary;
  /** The big target: resume this session.  Omitted where that is not offered. */
  onAttach?: (id: string) => void;
  /** Show which tree it lives in — wanted in the rail, redundant in a workspace-filtered picker. */
  showWorkspace?: boolean;
}

/**
 * One session as a row, with its note.
 *
 * The note gets its OWN line rather than a fourth ``·`` segment: that line
 * is already the agent's voice (``description``), and telling yours apart
 * from its is the whole point.
 *
 * A button cannot nest inside a button, so the row is a ``div`` holding the
 * attach button and the pencil as SIBLINGS.  The pencil is always drawn,
 * never hover-gated -- a touch screen has no hover, the lesson the Files
 * panel's ``hide`` / ``ignore`` actions already taught.
 */
export function SessionRow({ sess, onAttach, showWorkspace = false }: SessionRowProps) {
  const [open, setOpen] = useState(false);
  const note = useJaato((s) => s.notes[sess.id]?.text);
  const first = noteFirstLine(note);
  const ws = showWorkspace ? workspaceLabel(sess.workspacePath) : "";
  const subtitle = [sess.description || sess.name, ws, sess.provider ? `${sess.provider}/${sess.model}` : "", sess.turnCount ? `${sess.turnCount} turns` : ""].filter(Boolean).join(" · ");
  return (
    <div className="border-t hairline">
      <div className="flex gap-3 py-2.5 items-start">
        <span className={`pt-0.5 ${sess.isLoaded ? "text-success" : "text-text-muted"}`} aria-hidden="true">{sess.isLoaded ? "●" : "○"}</span>
        {/* The row's own aria-label used to override its content, so a screen
            reader was told the id and nothing else.  The label names the
            action; the text under it is readable in its own right. */}
        <button
          type="button"
          onClick={() => onAttach?.(sess.id)}
          disabled={!onAttach}
          aria-label={onAttach ? `Resume session ${sess.id}` : undefined}
          className="min-w-0 flex-1 text-left disabled:cursor-default"
        >
          <span className="block font-mono text-[13px]">{sess.id}</span>
          {subtitle && <span className="block text-[13px] text-text-muted truncate">{subtitle}</span>}
          {first && <span className="block chrome text-[13px] truncate" title={note}>✎ {first}</span>}
        </button>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-label={`${open ? "Close" : "Edit"} your note about session ${sess.id}`}
          className="btn btn-sm btn-quiet self-center shrink-0"
        >
          {first ? "✎" : "✎ Add a note"}
        </button>
        {onAttach && <span className={`btn btn-sm self-center shrink-0 ${sess.isLoaded ? "btn-steel" : "btn-quiet"}`} aria-hidden="true">Attach</span>}
      </div>
      {open && <div className="pb-2.5"><NoteEditor sessionId={sess.id} rows={3} /></div>}
    </div>
  );
}

/**
 * The rail's Sessions section: this session first with its note expanded --
 * the common case keeps a dedicated field instead of being buried in a list
 * -- then every other session the daemon told us about.
 */
export function SessionsPanel() {
  // Opening the section is the ask: refresh silently, which is exactly what
  // ``sessionListSilent`` exists for -- a listing nobody typed must not be
  // printed into the transcript.
  useEffect(() => { void ensureSessions(); }, []);
  const sessions = useJaato((s) => s.sessions);
  const current = useJaato((s) => s.sessionId);
  const scope = useNote(current).scope;
  const others = sessions.filter((s) => s.id !== current);
  return (
    <div className="px-3.5 py-2 text-[13px]">
      {current ? (
        <div className="pb-2">
          <NoteEditor sessionId={current} rows={3} label="This session" />
        </div>
      ) : (
        <p className="m-0 py-2 text-text-muted">No session attached.</p>
      )}
      {others.length > 0 && <div className="kicker kicker-muted text-[11px] pt-1">Other sessions</div>}
      {others.map((sess) => <SessionRow key={sess.id} sess={sess} showWorkspace />)}
      {scope === "local" && (
        // Said rather than assumed: without a sign-in backend these live in
        // this browser, so "next time I look" from another device loses them.
        <p className="m-0 pt-3 text-[11px] text-text-muted">Notes are kept in this browser only — a sign-in backend shares them across your devices.</p>
      )}
    </div>
  );
}

/** ``2 of 5 noted`` for the section header's ``value`` slot, or ``null`` when there is nothing to count. */
export function notedSummary(sessions: SessionSummary[], notes: Record<string, { text: string }>): string | null {
  if (sessions.length === 0) return null;
  const noted = sessions.filter((s) => noteFirstLine(notes[s.id]?.text)).length;
  return `${noted} of ${sessions.length} noted`;
}
