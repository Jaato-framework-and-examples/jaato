/**
 * The TUI's exit confirmation, ported.
 *
 * The TUI's ``exit`` does not leave at once: it asks what should become of
 * the session, because "exit" means three different things to a session
 * that lives on the daemon — leave it running for later (detach), stop
 * the turn and leave it (cancel), or delete it (end).  The web client
 * used to detach unconditionally, which was the safe reading but left
 * the other two without a button.  This module holds the question and
 * its answers; the store holds the open question (``exitChoice``), the
 * prompt component draws it, and the composer forwards a typed key here
 * exactly as it does for a permission prompt.
 *
 * Two facts the TUI does not have to decide, and this client does:
 *
 * - **End session lands on the workspace list in workspace mode.**  The
 *   TUI is a process, so "end" is also "quit".  Here the connection is
 *   worth keeping: the session is gone but its workspace directory is
 *   not (``session.delete`` removes the session's record and memory and
 *   never touches the tree it ran in), so the list is where the person
 *   picks it, or another, again.  On a single-workspace daemon there is
 *   no list, and End disconnects like Detach.
 * - **End waits for the daemon's word before leaving.**  ``session.delete``
 *   is answered by a ``system.message`` naming the outcome, and leaving
 *   before it arrives would report a deletion nobody has confirmed.  A
 *   daemon that answers nothing is given a bounded grace.  The waiting,
 *   and the forgetting of the note that answer licenses, live in
 *   ``sessionDelete.ts`` -- ``session delete <id>`` deletes too, and one
 *   of the two routes touching the note is how a note outlives a
 *   session.
 */
import type { ExitOption } from "@/store/types";
import { useJaato } from "@/store/store";
import { anyBusy } from "@/store/phase";
import { disconnect, getClient, isConnected } from "@/sdk/connection";
import { markExited } from "./exitIntent";
import { deleteSession } from "./sessionDelete";

/**
 * The options the TUI offers, in its order, for a session with a turn in
 * flight (``[c/d/e/r]``) or idle (``[d/e/r]``).  Labels are the TUI's
 * own words; ``r`` is the last option either way so Tab from the
 * default (``c`` or ``d``) reaches the refusals in the same order.
 */
export function exitOptions(running: boolean): ExitOption[] {
  const detach: ExitOption = running
    ? { key: "d", label: "Detach", description: "The task continues in the background; reconnect later" }
    : { key: "d", label: "Detach", description: "Keep the session on the daemon; reconnect later" };
  const end: ExitOption = running
    ? { key: "e", label: "End session", description: "Cancel the task and delete the session" }
    : { key: "e", label: "End session", description: "Delete the session from the daemon" };
  const back: ExitOption = { key: "r", label: "Return", description: "Return to the session" };
  return running
    ? [{ key: "c", label: "Cancel task and exit", description: "Stop the turn, keep the session on the daemon" }, detach, end, back]
    : [detach, end, back];
}

/** Where End session goes once the daemon has answered. */
export function endDestination(workspaceMode: "unknown" | "enabled" | "disabled"): "workspaces" | "connect" {
  return workspaceMode === "enabled" ? "workspaces" : "connect";
}

/**
 * Open the question.  With no session attached there is nothing to ask
 * about, and the ``exit`` command means what it always meant: detach.
 */
export async function requestExit(): Promise<void> {
  const st = useJaato.getState();
  if (!st.sessionId) return detach();
  const running = anyBusy(st);
  st.openExitChoice(running, exitOptions(running));
}

/**
 * Answer the open question with a key.  Anything that is not one of the
 * offered keys is Return, as the TUI reads any other input.
 */
export async function answerExit(key: string): Promise<void> {
  const st = useJaato.getState();
  const choice = st.exitChoice;
  if (!choice) return;
  const k = key.trim().toLowerCase() || choice.options[choice.focus]?.key || "r";
  st.closeExitChoice();
  const known = choice.options.some((o) => o.key === k);
  if (!known || k === "r") {
    st.addSystemBlock(st.selectedAgentId, "Returning to session.", "dim");
    return;
  }
  if (k === "c") {
    // The TUI's [c]: stop the turn, leave the session for later.
    try { await getClient().stop(); } catch { /* the disconnect below drops it anyway */ }
    return detach();
  }
  if (k === "d") return detach();
  return endSession(choice.running);
}

/** The exit command as it was: leave the daemon, keep the session, show the connect screen. */
async function detach(): Promise<void> {
  markExited();
  await disconnect();
  useJaato.getState().setScreen("connect");
}

/**
 * The TUI's [e]: stop a running turn, delete the session, and leave — to
 * the workspace list where there is one, else the connect screen.
 */
async function endSession(running: boolean): Promise<void> {
  const st = useJaato.getState();
  const sessionId = st.sessionId;
  if (!sessionId || !isConnected()) return detach();
  if (running) {
    try { await getClient().stop(); } catch { /* deletion still proceeds */ }
  }
  const answer = await deleteSession(sessionId);
  const fresh = useJaato.getState();
  if (answer.kind === "refused") {
    // The one answer after which nothing local may change: the session is
    // still there, so leaving would report a deletion that did not happen.
    fresh.addSystemBlock(fresh.selectedAgentId, answer.text, "warning");
    return;
  }
  if (answer.kind === "missing") {
    // The daemon says there was nothing to delete; the session is gone
    // either way, so the destination is the same.
    fresh.addSystemBlock(fresh.selectedAgentId, answer.text, "warning");
  }
  fresh.resetSessionState();
  const to = endDestination(fresh.workspace.mode);
  if (to === "connect") return detach();
  fresh.setScreen("workspaces");
}
