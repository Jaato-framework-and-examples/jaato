/**
 * The exit choice as a plate (``app/exitChoice.ts``): the TUI's
 * "Exit options:" / "Task in progress. What would you like to do?" and
 * its lettered answers, drawn like the permission plate so the two read
 * as the same kind of question.  The focused option is the one solid
 * button, Return sits apart at the right edge, every button shows its
 * key, Tab cycles focus, and the composer forwards typed keys.
 *
 * It also carries the note field, because this is the instant you know what
 * needs doing next -- which is what makes the field get filled rather than
 * being an empty box nobody visits.  A textarea in a plate whose whole
 * interaction model is keystrokes collides four ways, and each is handled
 * here rather than left to chance:
 *
 * | Key | |
 * |---|---|
 * | a letter (``d``/``e``/``r``) | free: the composer forwards keys only while IT has focus, and the textarea takes them |
 * | ``Tab`` | already guarded by ``inField`` -- cycling options from inside the field would be a trap |
 * | ``Escape`` | guarded too: it used to answer ``r`` unconditionally, discarding a half-typed note.  First Escape blurs, second returns |
 * | ``Enter`` | must insert a newline, so this plate is never ``as="form"`` |
 *
 * And the correctness bit that outranks all of it: **the save completes
 * before the exit action runs.**  A failed ``PUT`` (backend down, cookie
 * expired) while the client detaches anyway loses the note silently, so a
 * failure keeps the plate open with the reason instead of exiting.
 */
import { useEffect, useState } from "react";
import type { ExitChoice } from "@/store/types";
import { useJaato } from "@/store/store";
import { useNote } from "@/app/useNote";
import { Plate } from "@/components/layout/Plate";
import { NoteEditor, noteStatusLabel } from "@/components/panels/NoteEditor";

export function ExitPrompt({ x, onAnswer }: { x: ExitChoice; onAnswer: (key: string) => void }) {
  const focus = useJaato((s) => s.focusExitChoice);
  const sessionId = useJaato((s) => s.sessionId);
  const note = useNote(sessionId);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);

  // A note about a session you are DELETING is an orphan in the store, so
  // the field goes away when that is the answer in focus; the forgetting
  // happens once the daemon confirms (``app/sessionDelete.ts``).
  const ending = x.options[x.focus]?.key === "e";

  const answer = async (key: string): Promise<void> => {
    if (busy) return;
    setBusy(true);
    setFailed(null);
    try {
      if (key !== "e" && key !== "r") {
        await note.flush();
      }
      // ``e`` neither saves nor deletes here.  Not saving is the point --
      // a draft about a session being deleted has nowhere to go.  Deleting
      // used to happen here and was wrong twice: it ran BEFORE the daemon
      // was asked, so a refused delete took the note and left the session,
      // and it left the OTHER delete route (``session delete <id>``) with
      // no forget at all.  ``sessionDelete.ts`` now owns both.
    } catch {
      setBusy(false);
      setFailed(noteStatusLabel(note.status, note.scope) || "could not save — retry");
      return;
    }
    setBusy(false);
    onAnswer(key);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      const inField = !!t && (t.tagName === "TEXTAREA" || t.tagName === "INPUT");
      if (e.key === "Tab" && !inField) {
        e.preventDefault();
        const n = x.options.length || 1;
        focus((x.focus + (e.shiftKey ? -1 : 1) + n) % n);
      } else if (e.key === "Escape") {
        // Inside the field, Escape leaves the field: it must not discard
        // what was typed.  A second Escape, now outside, returns.
        if (inField) { e.preventDefault(); t?.blur(); return; }
        e.preventDefault();
        void answer("r");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const button = (o: ExitChoice["options"][number], i: number) => (
    <button
      key={o.key}
      type="button"
      onClick={() => { void answer(o.key); }}
      disabled={busy}
      title={o.description}
      className={`btn ${i === x.focus ? "btn-primary" : o.key === "e" ? "btn-danger" : ""}`}
      aria-current={i === x.focus ? "true" : undefined}
    >
      {o.label} <span className="key">{o.key}</span>
    </button>
  );
  const back = x.options.findIndex((o) => o.key === "r");
  return (
    <Plate edge="warning" className="my-3" role="group" aria-label="Exit options">
      <div className="px-3.5 py-2.5 flex items-center gap-3 border-b hairline">
        <span className="text-warning">⚠</span>
        <span className="kicker text-warning">{x.running ? "Task in progress" : "Exit options"}</span>
        <span className="text-[13px] text-text-muted">{x.running ? "What would you like to do?" : "What should become of the session?"}</span>
      </div>
      <ul className="px-3.5 py-2 m-0 list-none text-[13px] text-text-muted flex flex-col gap-0.5">
        {x.options.map((o) => (
          <li key={o.key}><span className="font-mono text-text">[{o.key}]</span> {o.label} <span>— {o.description}</span></li>
        ))}
      </ul>
      {sessionId && (
        <div className="px-3.5 py-2.5 border-t hairline">
          {ending ? (
            <p className="m-0 text-[12px] text-text-muted">Ending the session also forgets your note about it.</p>
          ) : (
            <NoteEditor sessionId={sessionId} handle={note} rows={2} label="Note to self" />
          )}
        </div>
      )}
      {failed && <p className="px-3.5 pb-2 m-0 text-[12px] text-warning" role="alert">Your note was not saved ({failed}). Nothing has been closed — try again, or clear the note to leave anyway.</p>}
      <div className="px-3.5 py-2.5 flex items-center gap-2 border-t hairline">
        {x.options.map((o, i) => (o.key === "r" ? null : button(o, i)))}
        <span className="flex-1" />
        {back >= 0 && button(x.options[back]!, back)}
      </div>
    </Plate>
  );
}
