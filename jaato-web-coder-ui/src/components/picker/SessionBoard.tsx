/**
 * Columns 1–3 of the session picker (design 2a): the workspace's existing
 * sessions by state -- Waiting on you, Awake, Sleeping -- as cards.
 *
 * Every session is ATTACHED to; there is no "Resume".  Attaching to a
 * sleeping session is what wakes it, which the column's hint says so the
 * verb does not have to.
 *
 * Discard is inline, never a modal: the card's footer turns into a pale red
 * strip naming what is lost for THAT state (an open prompt cancelled,
 * running work stopped, or just the saved history).  One card confirms at a
 * time.  It is ``session.delete`` through ``app/sessionDelete.ts`` -- the
 * daemon stops a loaded session before deleting its record -- and the card
 * leaves only when the daemon SAYS the session is gone; a refusal or a
 * silent daemon is shown in the strip, and the card stays.
 */
import { useState } from "react";
import { deleteSession, isGone } from "@/app/sessionDelete";
import { ensureSessions } from "@/app/actions";
import { noteFirstLine } from "@/app/notes";
import { useJaato } from "@/store/store";
import { sessionBoard, type SessionColumn, type SessionSummary } from "@/protocol/sessions";
import { awaitingLabel } from "@/components/panels/SessionsPanel";
import { NoteEditor } from "@/components/panels/NoteEditor";

interface ColumnSpec {
  id: SessionColumn;
  title: string;
  hint: string;
  /** The state colour: header rule, title and each card's state line. */
  tone: string;
  rule: string;
}

export const BOARD_COLUMNS: ColumnSpec[] = [
  { id: "waiting", title: "Waiting on you", hint: "A prompt is open", tone: "text-warning", rule: "border-warning" },
  { id: "awake", title: "Awake", hint: "Loaded in the daemon", tone: "text-success", rule: "border-success" },
  { id: "sleeping", title: "Sleeping", hint: "Saved on disk; attaching wakes it", tone: "text-text-muted", rule: "border-[color:var(--c-divider)]" },
];

/** The coloured state line of a card. */
export function stateLine(s: SessionSummary, col: SessionColumn, now: number = Date.now()): string {
  if (col === "waiting") return awaitingLabel(s.awaiting ?? "prompt", s.awaitingSince, now);
  if (col === "awake") {
    const clients = s.clientCount === 1 ? "1 client attached" : s.clientCount > 1 ? `${s.clientCount} clients attached` : "no client attached";
    return `awake · ${s.isProcessing ? "working · " : ""}${clients}`;
  }
  return "sleeping · on disk";
}

/** What Discard says it will do, by state. */
export function discardMessage(id: string, col: SessionColumn): string {
  if (col === "waiting") return `Stop and discard ${id}? Its open prompt is cancelled and the saved history is deleted.`;
  if (col === "awake") return `Stop and discard ${id}? Any running work stops and the saved history is deleted.`;
  return `Delete ${id} and its saved history?`;
}

function SessionCard({ sess, col, tone, confirming, onConfirm, onAttach }: {
  sess: SessionSummary;
  col: SessionColumn;
  tone: string;
  confirming: boolean;
  onConfirm: (id: string | null) => void;
  onAttach: (id: string) => void;
}) {
  const note = useJaato((s) => s.notes[sess.id]?.text);
  const first = noteFirstLine(note);
  const [noteOpen, setNoteOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const describe = [sess.description || sess.name, sess.turnCount ? `${sess.turnCount} turns` : ""].filter((x) => x && x !== sess.id).join(" · ");
  const binding = [sess.profile, sess.provider ? `${sess.provider}/${sess.model}` : ""].filter(Boolean).join(" · ");
  const discard = async () => {
    setBusy(true);
    setError("");
    const answer = await deleteSession(sess.id);
    setBusy(false);
    if (isGone(answer)) {
      onConfirm(null);
      await ensureSessions();
      return;
    }
    setError(answer.kind === "silent" ? "The daemon did not confirm the discard." : answer.text || "The daemon refused the discard.");
  };
  return (
    <li className="border hairline bg-surface flex flex-col" data-testid="session-card" data-session={sess.id}>
      <div className="px-3.5 pt-3 pb-2.5 flex flex-col gap-0.5">
        <span className="font-mono text-[13px]">{sess.id}</span>
        <span className={`text-[13px] ${tone}`}>{stateLine(sess, col)}</span>
        {describe && <span className="text-[13px] text-text-muted">{describe}</span>}
        {binding && <span className="font-mono text-[11px] text-text-muted break-words">{binding}</span>}
        {first && <span className="chrome text-[13px] mt-1 truncate" title={note}>✎ {first}</span>}
      </div>
      {confirming ? (
        <div className="tint-error border-t border-error px-3.5 py-2.5 flex flex-col gap-2" role="group" aria-label={`Confirm discarding session ${sess.id}`}>
          <span className="text-[13px]">{discardMessage(sess.id, col)}</span>
          {error && <span className="text-[12px] text-error">{error}</span>}
          <span className="flex gap-2">
            <button type="button" disabled={busy} onClick={discard} className="btn btn-sm btn-danger-solid" aria-label={`Discard session ${sess.id}`}>Discard</button>
            <button type="button" disabled={busy} onClick={() => onConfirm(null)} className="btn btn-sm">Cancel</button>
          </span>
        </div>
      ) : (
        <div className="flex border-t hairline">
          <button
            type="button"
            onClick={() => onAttach(sess.id)}
            aria-label={`Attach session ${sess.id}`}
            className={`flex-1 text-left px-3.5 py-2 chrome chrome-sm hover:bg-tint ${col === "sleeping" ? "text-text-muted" : "text-steel"}`}
          >
            Attach <span aria-hidden="true">→</span>
          </button>
          <button
            type="button"
            onClick={() => setNoteOpen((v) => !v)}
            aria-expanded={noteOpen}
            aria-label={`${noteOpen ? "Close" : first ? "Edit" : "Add"} your note about session ${sess.id}`}
            title={first ? "Edit note" : "Add a note"}
            className="px-3.5 py-2 border-l hairline text-text-muted hover:bg-tint hover:text-text"
          >
            ✎
          </button>
          <button
            type="button"
            onClick={() => onConfirm(sess.id)}
            aria-label={`Discard session ${sess.id}…`}
            className="px-3.5 py-2 border-l hairline chrome chrome-sm text-text-muted hover:bg-tint hover:text-error"
          >
            Discard
          </button>
        </div>
      )}
      {noteOpen && <div className="px-3.5 pb-3 border-t hairline pt-2"><NoteEditor sessionId={sess.id} rows={3} /></div>}
    </li>
  );
}

/** One column header: condensed title + count, a 2px rule in the state colour. */
export function ColumnHeader({ title, count, tone, rule }: { title: string; count?: number; tone: string; rule: string }) {
  return (
    <div className={`px-4 py-3 border-b-2 ${rule} flex items-baseline gap-2.5`}>
      <span className={`chrome ${tone}`}>{title}</span>
      {count != null && <span className="font-mono text-[11px] text-text-muted">{count}</span>}
    </div>
  );
}

/**
 * The three session columns.  Rendered as grid children of the picker's
 * four-column body (the New session column is the fourth), so each column
 * carries its own left rule rather than the grid drawing them.
 */
export function SessionBoard({ sessions, onAttach }: { sessions: SessionSummary[]; onAttach: (id: string) => void }) {
  const [confirming, setConfirming] = useState<string | null>(null);
  const board = sessionBoard(sessions);
  return (
    <>
      {BOARD_COLUMNS.map((c) => (
        <section key={c.id} aria-label={c.title} data-column={c.id} className="flex flex-col min-w-0 border-b min-[900px]:border-b-0 min-[900px]:border-r hairline">
          <ColumnHeader title={c.title} count={board[c.id].length} tone={c.tone} rule={c.rule} />
          <div className="px-4 py-3 flex flex-col gap-2.5">
            <span className="text-[13px] text-text-muted">{c.hint}</span>
            <ul className="flex flex-col gap-2.5 m-0 p-0 list-none">
              {board[c.id].map((s) => (
                <SessionCard key={s.id} sess={s} col={c.id} tone={c.tone} confirming={confirming === s.id} onConfirm={setConfirming} onAttach={onAttach} />
              ))}
            </ul>
            {board[c.id].length === 0 && <span className="text-[12px] text-text-muted italic">None</span>}
          </div>
        </section>
      ))}
    </>
  );
}
