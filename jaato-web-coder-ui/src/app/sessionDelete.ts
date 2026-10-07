/**
 * Deleting a session, and forgetting what you wrote about it.
 *
 * A note is keyed by session id and is stored where the daemon cannot see
 * it (the BFF, or this browser).  So nothing removed one when its session
 * went away, and the rail kept offering ``✎ ESTA SESIÓN VA DEL ACORD…``
 * under a session that no longer existed.  That is not cosmetic: a note is
 * what you meant to do NEXT, and one attached to a session you cannot open
 * is an instruction to yourself that can never be carried out.
 *
 * **Only positive evidence removes a note.**  The tempting rule -- prune
 * notes whose session is absent from the listing -- is wrong here, and this
 * very defect is what shows why: that listing is entitlement-scoped and it
 * NARROWS (a workspace deselected, an identity not yet resolved, a daemon
 * that answered 2 where the snapshot said 169).  Pruning on absence would
 * destroy a note because a workspace was momentarily out of view.  A note
 * is forgotten when the daemon SAYS the session was deleted, and at no
 * other time.
 *
 * **One place, because there are two routes.**  ``End session`` on the exit
 * plate and ``session delete <id>`` typed into the composer both delete;
 * only the first used to touch the note, and it did so BEFORE the daemon
 * was asked -- so a delete the daemon refuses (a foreign session, since
 * that verb is now inside the #1113 boundary) took the note with it and
 * left the session.  Both routes come here now, and the forget happens
 * after the answer.
 */
import { EventTypeValue } from "@jaato/sdk";
import { getClient } from "@/sdk/connection";
import { useJaato } from "@/store/store";
import { localNotesApi, notesApi } from "./notes";

/** How long a delete waits for the daemon's confirmation before giving up on one. */
export const DELETE_CONFIRM_GRACE_MS = 4000;

export type DeleteAnswer = {
  /**
   * - ``deleted`` / ``missing`` — the daemon spoke: the session is gone.
   * - ``refused`` — outside this caller's boundary; it still exists.
   * - ``silent`` — nobody answered inside the grace; nothing is known.
   */
  kind: "deleted" | "missing" | "refused" | "silent";
  text: string;
};

/** The two outcomes that mean "there is no such session any more". */
export function isGone(answer: DeleteAnswer): boolean {
  return answer.kind === "deleted" || answer.kind === "missing";
}

/**
 * The daemon answers ``session.delete`` with a ``system.message`` —
 * ``Session '<id>' deleted.`` or ``Session '<id>' not found.`` — and, for a
 * loaded session, also tells attached clients ``Session deleted: <name>``.
 * A session outside the caller's boundary is refused with a
 * ``SessionError`` naming the verb instead.  The first of those about THIS
 * session settles the wait; the grace settles it for a daemon that says
 * nothing.
 */
export function awaitDeleteAnswer(sessionId: string): Promise<DeleteAnswer> {
  return new Promise((resolve) => {
    const client = getClient();
    let done = false;
    const subs: Array<() => void> = [];
    const finish = (a: DeleteAnswer) => {
      if (done) return;
      done = true;
      for (const off of subs) off();
      clearTimeout(timer);
      resolve(a);
    };
    subs.push(client.subscribe(EventTypeValue.SYSTEM_MESSAGE, (ev) => {
      const text = String((ev as { message?: unknown }).message ?? "");
      if (text.includes(`'${sessionId}' not found`)) finish({ kind: "missing", text });
      else if (text.includes(`'${sessionId}' deleted`) || text.startsWith("Session deleted:")) finish({ kind: "deleted", text });
    }));
    subs.push(client.subscribe(EventTypeValue.ERROR, (ev) => {
      const e = ev as { error?: unknown; error_type?: unknown };
      if (String(e.error_type ?? "") !== "SessionError") return;
      const text = String(e.error ?? "");
      if (text.includes(sessionId)) finish({ kind: "refused", text });
    }));
    const timer = setTimeout(() => finish({ kind: "silent", text: "" }), DELETE_CONFIRM_GRACE_MS);
  });
}

/**
 * Forget the note about a session that is gone.
 *
 * Best effort by design: the session has already been deleted, so blocking
 * or reporting on a failed note DELETE would trade the trivial failure for
 * the expensive one.  It is idempotent, which is what lets the caller run
 * it without first asking whether a note exists.
 */
export async function forgetNote(sessionId: string): Promise<void> {
  const st = useJaato.getState();
  const api = st.notesUrl ? notesApi(st.notesUrl) : localNotesApi();
  try {
    await api.remove(sessionId);
  } catch {
    /* the session is gone either way */
  }
  st.setNote(sessionId, null);
  st.setNoteStatus(sessionId, { state: "idle" });
}

/**
 * Ask the daemon to delete *sessionId*, wait for its word, and forget the
 * note if it says the session is gone.
 *
 * Returns what the daemon said so the caller can render it — ``End
 * session`` reports a ``missing`` as a warning, and a ``refused`` is the
 * one outcome after which nothing local should change.
 */
export async function deleteSession(sessionId: string): Promise<DeleteAnswer> {
  const answered = awaitDeleteAnswer(sessionId);
  await getClient().deleteSession(sessionId);
  const answer = await answered;
  if (isGone(answer)) await forgetNote(sessionId);
  return answer;
}
