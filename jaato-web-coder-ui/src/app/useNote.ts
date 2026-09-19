/**
 * One note, one behaviour, four mount points.
 *
 * The note a person writes about a session is reachable from four places --
 * the exit prompt, the resume picker's row, the Sessions rail for THIS
 * session, and the same rail for another one.  The debounce, the save, the
 * cap, the error rendering and the placeholder therefore live here rather
 * than in each, or there would be four behaviours and four bugs, and
 * ``saved 12:04`` would appear in three dialects.
 *
 * Which store is behind it is decided once, from ``notesUrl``: the signed-in
 * backend's, or this browser's ``localStorage`` when the page was served
 * without one.  ``scope`` is returned so the UI can SAY which -- a silent
 * local fallback is a promise the storage does not keep, because "next time
 * I look" from another device would lose them.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useJaato } from "@/store/store";
import type { NoteStatus } from "@/store/store";
import { MAX_NOTE_CHARS, localNotesApi, normalizeNote, notesApi, type NoteScope, type NotesApi } from "./notes";
import { SignInRequiredError } from "./tickets";

/** Long enough not to save every keystroke, short enough that a quick exit still catches it. */
export const NOTE_DEBOUNCE_MS = 800;

/** One API for the whole page, rebuilt only when the backend URL changes. */
export function useNotesApi(): NotesApi {
  const notesUrl = useJaato((s) => s.notesUrl);
  return useMemo(() => (notesUrl ? notesApi(notesUrl) : localNotesApi()), [notesUrl]);
}

/**
 * Load every note this person has written, once the page knows where from.
 *
 * Failure is deliberately quiet: notes are an aid, and a page that refuses
 * to show a session picker because a note listing 500'd has traded the
 * feature's value for its cost.  A write failure is NOT quiet -- see below.
 */
export function useLoadNotes(): void {
  const api = useNotesApi();
  const setNotes = useJaato((s) => s.setNotes);
  useEffect(() => {
    let cancelled = false;
    void api.list().then(
      (list) => { if (!cancelled) setNotes(list); },
      () => undefined,
    );
    return () => { cancelled = true; };
  }, [api, setNotes]);
}

export interface NoteHandle {
  /** What the textarea shows: the local draft while typing, the stored text otherwise. */
  text: string;
  status: NoteStatus;
  scope: NoteScope;
  /** Remaining characters, for the counter the editor shows only near the cap. */
  remaining: number;
  /** Type into it; saves after ``NOTE_DEBOUNCE_MS``. */
  change: (text: string) => void;
  /**
   * Save NOW and resolve when the store has it.
   *
   * The exit prompt awaits this before it acts: a failed ``PUT`` while the
   * client detaches anyway loses the note silently, which is the one outcome
   * the feature cannot have.  Rejects on failure so the caller can stay open.
   */
  flush: () => Promise<void>;
  /** Forget it — what "End session" does beside deleting the session. */
  remove: () => Promise<void>;
}

const IDLE: NoteStatus = { state: "idle" };

export function useNote(sessionId: string | null | undefined): NoteHandle {
  const api = useNotesApi();
  const stored = useJaato((s) => (sessionId ? s.notes[sessionId]?.text : undefined));
  const status = useJaato((s) => (sessionId ? s.noteStatus[sessionId] : undefined)) ?? IDLE;
  const setNote = useJaato((s) => s.setNote);
  const setNoteStatus = useJaato((s) => s.setNoteStatus);

  // A draft exists only while typing.  ``null`` means "show what is stored",
  // so a note saved from ANOTHER mount point appears here without a reload.
  const [draft, setDraft] = useState<string | null>(null);
  const timer = useRef<number | null>(null);
  const pending = useRef<string | null>(null);

  const clearTimer = () => {
    if (timer.current !== null) { window.clearTimeout(timer.current); timer.current = null; }
  };
  useEffect(() => clearTimer, []);
  // Switching sessions drops a draft that belonged to the previous one.
  useEffect(() => { setDraft(null); pending.current = null; clearTimer(); }, [sessionId]);

  const save = useCallback(async (text: string): Promise<void> => {
    if (!sessionId) return;
    pending.current = null;
    setNoteStatus(sessionId, { state: "saving" });
    try {
      const note = await api.put(sessionId, text);
      setNote(sessionId, note);
      setDraft(null);
      setNoteStatus(sessionId, { state: "saved", at: Date.now() });
    } catch (e) {
      // Never clear the draft on failure: what the person typed is the only
      // copy left, and the textarea is where it still is.
      if (e instanceof SignInRequiredError) setNoteStatus(sessionId, { state: "signed-out", loginUrl: e.loginUrl });
      else setNoteStatus(sessionId, { state: "error", message: e instanceof Error ? e.message : String(e) });
      throw e;
    }
  }, [api, sessionId, setNote, setNoteStatus]);

  const change = useCallback((text: string) => {
    const bounded = text.slice(0, MAX_NOTE_CHARS);
    setDraft(bounded);
    pending.current = bounded;
    clearTimer();
    timer.current = window.setTimeout(() => { timer.current = null; void save(bounded).catch(() => undefined); }, NOTE_DEBOUNCE_MS);
  }, [save]);

  const flush = useCallback(async () => {
    clearTimer();
    const text = pending.current;
    // Nothing typed since the last save, or nothing that changed it: the
    // caller still gets a resolved promise, so "await before you exit" is
    // unconditional at every call site rather than a branch each one repeats.
    if (text === null || normalizeNote(text) === normalizeNote(stored ?? "")) { pending.current = null; return; }
    await save(text);
  }, [save, stored]);

  const remove = useCallback(async () => {
    if (!sessionId) return;
    clearTimer();
    pending.current = null;
    await api.remove(sessionId);
    setNote(sessionId, null);
    setDraft(null);
    setNoteStatus(sessionId, IDLE);
  }, [api, sessionId, setNote, setNoteStatus]);

  const text = draft ?? stored ?? "";
  return { text, status, scope: api.scope, remaining: MAX_NOTE_CHARS - text.length, change, flush, remove };
}
