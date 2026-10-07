/**
 * The one note editor, mounted wherever a note is written.
 *
 * Four surfaces touch a note -- the exit prompt, the resume picker's row,
 * the Sessions rail for this session and for another one -- so the textarea,
 * the counter, the status line and the placeholder are here rather than in
 * each of them (``app/useNote.ts`` holds the behaviour behind it).
 *
 * The placeholder is the most valuable string in the feature.  An empty box
 * gets skipped; a question gets answered.
 */
import { useNote, type NoteHandle } from "@/app/useNote";
import type { NoteStatus } from "@/store/store";

export const NOTE_PLACEHOLDER = "What should you pick up next time?";
/** Below this the counter is noise; near the cap it is the only thing that explains a refusal. */
const COUNTER_FROM = 200;

/** ``saved 12:04`` and its five siblings, in ONE dialect. */
export function noteStatusLabel(status: NoteStatus, scope: "backend" | "local"): string {
  switch (status.state) {
    case "saving": return "saving…";
    case "saved": return `saved ${new Date(status.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
    case "error": return "could not save — retry";
    // A cookie that expired mid-session is not a generic failure, and saying
    // so is the difference between retrying forever and signing in.
    case "signed-out": return "signed out — sign in to save";
    default: return scope === "local" ? "this browser only" : "";
  }
}

export interface NoteEditorProps {
  sessionId: string;
  /** Rows for the textarea; the rail gives it more than a picker row does. */
  rows?: number;
  /** Rendered above the field, when the surface has room for it. */
  label?: string;
  autoFocus?: boolean;
  className?: string;
  /** Lifted so a caller that must await the save before acting (the exit prompt) shares this instance. */
  handle?: NoteHandle;
}

export function NoteEditor({ sessionId, rows = 3, label, autoFocus = false, className = "", handle }: NoteEditorProps) {
  // A caller that needs ``flush`` passes its own handle so the two are the
  // SAME note rather than two hooks racing each other over one session id.
  const own = useNote(sessionId);
  const note = handle ?? own;
  const status = noteStatusLabel(note.status, note.scope);
  const failed = note.status.state === "error" || note.status.state === "signed-out";
  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      {label && <span className="kicker kicker-muted text-[11px]">{label}</span>}
      <textarea
        value={note.text}
        rows={rows}
        autoFocus={autoFocus}
        onChange={(e) => note.change(e.target.value)}
        onBlur={() => { void note.flush().catch(() => undefined); }}
        placeholder={NOTE_PLACEHOLDER}
        aria-label={label ?? `Note about session ${sessionId}`}
        className="w-full bg-transparent border hairline px-2 py-1.5 text-[13px] leading-snug resize-y focus:outline-none focus:border-[var(--c-steel)]"
      />
      <div className="flex items-center gap-2 text-[11px] text-text-muted">
        <span className={failed ? "text-warning" : ""}>{status}</span>
        <span className="flex-1" />
        {note.remaining <= COUNTER_FROM && <span className="font-mono">{note.remaining}</span>}
      </div>
    </div>
  );
}
