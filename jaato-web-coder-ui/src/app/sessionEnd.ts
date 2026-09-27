/**
 * Ending a session and keeping it: the web client's End on a 1.29 daemon.
 *
 * End used to be ``session.delete``: the record and its history were gone,
 * so a session a person had chosen to end could not be looked at again.
 * From protocol 1.29 the daemon marks a session FINISHED when it is ended
 * (``session.end``), when its agent completes, or when its budget stops it,
 * and the session picker lists it in a Finished column.  So End sends
 * ``session.end`` there and Discard stays the way to delete.
 *
 * Below 1.29 ``session.end`` marks nothing, and the ended session would come
 * back as an ordinary sleeping one; there End keeps deleting, as it always
 * did (``exitChoice.ts``).
 *
 * **End waits for the daemon's word.**  ``session.end`` is answered by a
 * ``SessionTerminatedEvent`` for that session (reason ``client_request``,
 * or ``stopped`` when a turn was cancelled).  A daemon that says nothing is
 * given a bounded grace, and the caller leaves either way: unlike a
 * refused delete, there is no answer to ``session.end`` after which the
 * session is still running as far as the person asked.
 *
 * The note is KEPT: a finished session is still listed, and a note is what
 * you meant to do next with it.  Only a confirmed delete forgets one.
 */
import { EventTypeValue, MIN_SESSION_FINISH_PROTOCOL, isProtocolCompatible } from "@jaato/sdk";
import { getClient } from "@/sdk/connection";
import { useJaato } from "@/store/store";

/** How long End waits for the daemon's terminal before leaving anyway. */
export const END_CONFIRM_GRACE_MS = 4000;

/** Does the connected daemon keep an ended session listed as finished? */
export function finishSupported(protocolVersion?: string | null): boolean {
  const v = protocolVersion ?? useJaato.getState().connection.protocolVersion;
  return !!v && isProtocolCompatible(v, MIN_SESSION_FINISH_PROTOCOL);
}

export type EndAnswer =
  /** The daemon confirmed; ``reason`` is the terminal's. */
  | { kind: "ended"; reason: string }
  /** Nothing arrived inside the grace. */
  | { kind: "silent" };

/** Resolve on the first ``session.terminated`` naming *sessionId*, or the grace. */
export function awaitEndAnswer(sessionId: string): Promise<EndAnswer> {
  return new Promise((resolve) => {
    const client = getClient();
    let done = false;
    const finish = (a: EndAnswer) => {
      if (done) return;
      done = true;
      off();
      clearTimeout(timer);
      resolve(a);
    };
    const off = client.subscribe(EventTypeValue.SESSION_TERMINATED, (ev) => {
      const e = ev as { session_id?: unknown; reason?: unknown };
      // An event naming no session is accepted: a daemon predating the
      // stamp sends it unattributed, and this connection is attached to
      // the session being ended.
      if (e.session_id && String(e.session_id) !== sessionId) return;
      finish({ kind: "ended", reason: String(e.reason ?? "") });
    });
    const timer = setTimeout(() => finish({ kind: "silent" }), END_CONFIRM_GRACE_MS);
  });
}

/** Send ``session.end`` for the attached *sessionId* and wait for the daemon's word. */
export async function endSession(sessionId: string): Promise<EndAnswer> {
  const answered = awaitEndAnswer(sessionId);
  await getClient().endSession();
  return answered;
}
