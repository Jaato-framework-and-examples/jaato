/**
 * The 26px foot of the session screen (design frame 04): what the daemon
 * reports about this connection in monospace on the left, the rail's
 * section toggles in the chrome face on the right, and ``exit``.  The
 * session's identity (workspace, model, context) lives in the header,
 * not here; what is left is the connection, the versions, the session id
 * and the permission default -- with a count of prompts waiting, since a
 * waiting prompt is the one thing worth reading off the foot of the page.
 */
import { requestExit } from "@/app/exitChoice";
import { BUILD, buildLine } from "@/app/buildInfo";
import { useJaato } from "@/store/store";

export function StatusBar() {
  const conn = useJaato((s) => s.connection);
  const sessionId = useJaato((s) => s.sessionId);
  const permStatus = useJaato((s) => s.permissionStatus);
  const waiting = useJaato((s) => s.permissions.length);
  const toggle = useJaato((s) => s.toggleUi);
  const setToolsExpanded = useJaato((s) => s.setToolsExpanded);
  const ui = useJaato((s) => s.ui);
  const dotCls = conn.phase === "connected" ? "bg-success" : conn.phase === "reconnecting" || conn.phase === "connecting" ? "bg-warning pulse" : "bg-error";
  const tab = (on: boolean) => `chrome-sm font-heading font-medium uppercase tracking-[0.12em] ${on ? "text-steel" : "text-text-muted hover:text-steel"}`;
  return (
    <div className="flex items-center gap-4 px-4 h-[26px] text-[11px] bg-surface border-t hairline font-mono text-text-muted select-none shrink-0">
      <span className="flex items-center gap-1.5 text-text"><span className={`inline-block w-1.5 h-1.5 ${dotCls}`} />{conn.phase}{conn.attempt ? ` #${conn.attempt}` : ""}</span>
      <span title={buildLine()}>{conn.serverVersion ? `server ${conn.serverVersion} · ` : ""}ui {BUILD.ui}</span>
      {sessionId && <span title={sessionId}>session {sessionId.slice(0, 8)}</span>}
      {permStatus && (
        <span title="Permission default policy (permissions default allow|deny|ask)" data-testid="permission-status" className={waiting ? "text-warning" : ""}>
          <span>permissions </span>
          {permStatus.suspensionScope ? (
            <span className="text-success">allow <span className="text-text-muted">({permStatus.suspensionScope})</span></span>
          ) : (
            <span className={waiting ? "" : permStatus.effectiveDefault === "deny" ? "text-warning" : permStatus.effectiveDefault === "allow" ? "text-success" : "text-text"}>{permStatus.effectiveDefault}</span>
          )}
          {waiting > 0 && <span> · {waiting} waiting</span>}
        </span>
      )}
      <span className="flex-1" />
      <button type="button" onClick={() => toggle("showPlan")} className={tab(ui.showPlan)} title="Toggle plan (Ctrl+P)" aria-label="Toggle plan (Ctrl+P)">Plan</button>
      <button type="button" onClick={() => toggle("showBudget")} className={tab(ui.showBudget)} title="Toggle budget (Ctrl+B)" aria-label="Toggle budget (Ctrl+B)">Budget</button>
      <button type="button" onClick={() => toggle("showWorkspace")} className={tab(ui.showWorkspace)} title="Toggle workspace changes (Alt+W)" aria-label="Toggle workspace changes (Alt+W)">Files</button>
      {/* No shortcut letter: Ctrl+P/T/R/F/G/W are spoken for, and the free
          set should be checked rather than guessed. */}
      <button type="button" onClick={() => toggle("showSessions")} className={tab(ui.showSessions)} title="Toggle your sessions and their notes" aria-label="Toggle your sessions and their notes">Sessions</button>
      <button type="button" onClick={() => setToolsExpanded(!ui.showTools)} className={tab(ui.showTools)} title={ui.showTools ? "Tool call boxes expanded — click to collapse them (Ctrl+T)" : "Tool call boxes collapsed — click to expand them (Ctrl+T)"} aria-label="Toggle tool call boxes (Ctrl+T)">Tools</button>
      {/* The ``exit`` command as a button: asks what becomes of the session -- detach, end, or cancel the task -- before leaving. */}
      <button type="button" onClick={() => { requestExit().catch(() => undefined); }} className={`${tab(false)} hover:text-error`} title="Leave: detach from the session, or end it (the exit command)" aria-label="Exit (detach from or end the session)">Exit</button>
    </div>
  );
}
