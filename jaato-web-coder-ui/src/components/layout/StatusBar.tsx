/**
 * The 26px foot of the session screen (design frame 04): what the daemon
 * reports about this connection in monospace on the left, the tool-call
 * expand toggle and ``exit`` in the chrome face on the right.  The
 * session's identity (workspace, model, context) lives in the header, and
 * Plan/Files/Budget/Sessions/Memories/Diagnostics live behind the 56px
 * icon rail (#1304 §5) rather than as duplicate buttons here -- what is
 * left is the connection, the versions, the session id and the
 * permission default -- with a count of prompts waiting, since a waiting
 * prompt is the one thing worth reading off the foot of the page.
 */
import { useRef, useState } from "react";
import { requestExit } from "@/app/exitChoice";
import { BUILD, buildLine } from "@/app/buildInfo";
import { PermissionsPlate } from "@/components/prompts/PermissionsPlate";
import { useJaato } from "@/store/store";

export function StatusBar() {
  const conn = useJaato((s) => s.connection);
  const sessionId = useJaato((s) => s.sessionId);
  const fault = useJaato((s) => s.sessionFault);
  const permStatus = useJaato((s) => s.permissionStatus);
  const waiting = useJaato((s) => s.permissions.length);
  const setToolsExpanded = useJaato((s) => s.setToolsExpanded);
  const ui = useJaato((s) => s.ui);
  const [perms, setPerms] = useState(false);
  const permsBtn = useRef<HTMLButtonElement>(null);
  // A non-recoverable ErrorEvent (RunnerBootstrapFailed among its causes,
  // ``protocol/sessionFault.ts``) means the SESSION never came up, which
  // the WebSocket transport's own phase has no way to say -- it stays
  // "connected" throughout, correctly, and used to be the only thing this
  // line read.  ``fault`` outranks the transport phase here on purpose.
  const dotCls = fault ? "bg-error" : conn.phase === "connected" ? "bg-success" : conn.phase === "reconnecting" || conn.phase === "connecting" ? "bg-warning pulse" : "bg-error";
  const tab = (on: boolean) => `chrome-sm font-heading font-medium uppercase tracking-[0.12em] ${on ? "text-steel" : "text-text-muted hover:text-steel"}`;
  return (
    <div className="relative flex items-center gap-4 px-4 h-[26px] text-[11px] bg-surface border-t hairline font-mono text-text-muted select-none shrink-0">
      <span className="flex items-center gap-1.5 text-text" title={fault ? `${fault.errorType}: ${fault.message}` : undefined}>
        <span className={`inline-block w-1.5 h-1.5 shrink-0 ${dotCls}`} />
        {fault ? <span className="text-error">no session</span> : <>{conn.phase}{conn.attempt ? ` #${conn.attempt}` : ""}</>}
      </span>
      <span title={buildLine()}>{conn.serverVersion ? `server ${conn.serverVersion} · ` : ""}ui {BUILD.ui}</span>
      {sessionId && <span title={sessionId}>session {sessionId.slice(0, 8)}</span>}
      {/* Reporting the policy and being able to change it used to be in
          different places, and only the reading was on screen.  The
          segment keeps its wording and becomes the control. */}
      {permStatus && (
        <button
          ref={permsBtn}
          type="button"
          onClick={() => setPerms((o) => !o)}
          aria-expanded={perms}
          aria-haspopup="dialog"
          title="Session permissions — click to change the default, suspend prompting, or show the policy"
          data-testid="permission-status"
          className={`font-mono hover:text-steel ${waiting ? "text-warning" : ""}`}
        >
          <span>permissions </span>
          {permStatus.suspensionScope ? (
            <span className="text-success">allow <span className="text-text-muted">({permStatus.suspensionScope})</span></span>
          ) : (
            <span className={waiting ? "" : permStatus.effectiveDefault === "deny" ? "text-warning" : permStatus.effectiveDefault === "allow" ? "text-success" : "text-text"}>{permStatus.effectiveDefault}</span>
          )}
          {waiting > 0 && <span> · {waiting} waiting</span>}
        </button>
      )}
      {perms && <PermissionsPlate onClose={() => setPerms(false)} anchor={permsBtn} />}
      <span className="flex-1 min-w-2" />
      {/* A fixed cluster of buttons does not fit a 375px bar beside the
          connection info, and this row is not allowed to wrap (the bar's
          whole contract is one line).  ``min-w-0`` lets the CLUSTER shrink
          below its own content width instead of forcing the bar -- and the
          page under it -- wider than the viewport; ``overflow-x-auto`` is
          where the difference goes instead of off the edge. */}
      <div className="flex items-center gap-4 min-w-0 overflow-x-auto">
        <button type="button" onClick={() => setToolsExpanded(!ui.showTools)} className={tab(ui.showTools)} title={ui.showTools ? "Tool call boxes expanded — click to collapse them (⌘/Ctrl+K then T)" : "Tool call boxes collapsed — click to expand them (⌘/Ctrl+K then T)"} aria-label="Toggle tool call boxes (⌘/Ctrl+K then T)">Tools</button>
        {/* The ``exit`` command as a button: asks what becomes of the session -- detach, end, or cancel the task -- before leaving. */}
        <button type="button" onClick={() => { requestExit().catch(() => undefined); }} className={`${tab(false)} hover:text-error shrink-0`} title="Leave: detach from the session, or end it (the exit command)" aria-label="Exit (detach from or end the session)">Exit</button>
      </div>
    </div>
  );
}
