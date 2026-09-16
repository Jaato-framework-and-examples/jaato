import { exitToConnect } from "@/app/actions";
import { BUILD, buildLine } from "@/app/buildInfo";
import { useJaato } from "@/store/store";

export function StatusBar() {
  const conn = useJaato((s) => s.connection);
  const session = useJaato((s) => s.session);
  const sessionId = useJaato((s) => s.sessionId);
  const ws = useJaato((s) => s.workspace.selected);
  const selected = useJaato((s) => s.selectedAgentId);
  const ctx = useJaato((s) => s.context[selected]);
  const permStatus = useJaato((s) => s.permissionStatus);
  const toggle = useJaato((s) => s.toggleUi);
  const setToolsExpanded = useJaato((s) => s.setToolsExpanded);
  const ui = useJaato((s) => s.ui);
  const dot = conn.phase === "connected" ? "bg-success" : conn.phase === "reconnecting" || conn.phase === "connecting" ? "bg-warning pulse" : "bg-error";
  const pct = ctx?.percentUsed;
  return (
    <div className="flex items-center gap-3 px-3 h-7 text-[11px] surface-2 border-t hairline font-mono select-none">
      <span className="flex items-center gap-1.5"><span className={`inline-block w-2 h-2 rounded-full ${dot}`} />{conn.phase}{conn.attempt ? ` #${conn.attempt}` : ""}</span>
      {conn.serverVersion && <span className="text-text-muted">server {conn.serverVersion}</span>}
      <span className="text-text-muted" title={buildLine()}>ui {BUILD.ui}</span>
      {ws && <span className="text-accent">{ws}</span>}
      {sessionId && <span className="text-text-muted" title={sessionId}>session {sessionId.slice(0, 8)}</span>}
      {(session.provider || session.model) && <span>{[session.provider, session.model].filter(Boolean).join(" / ")}</span>}
      {session.profile && <span className="text-text-muted">profile {session.profile}</span>}
      {pct != null && <span className={pct > 80 ? "text-error" : pct > 60 ? "text-warning" : ""}>ctx {pct.toFixed(0)}%</span>}
      {permStatus && (
        <span title="Permission default policy (permissions default allow|deny|ask)" data-testid="permission-status">
          <span className="text-text-muted">permissions </span>
          {permStatus.suspensionScope ? (
            <span className="text-success">allow <span className="text-text-muted">({permStatus.suspensionScope})</span></span>
          ) : (
            <span className={permStatus.effectiveDefault === "deny" ? "text-warning" : permStatus.effectiveDefault === "allow" ? "text-success" : ""}>{permStatus.effectiveDefault}</span>
          )}
        </span>
      )}
      <span className="flex-1" />
      <button type="button" onClick={() => toggle("showPlan")} className={ui.showPlan ? "text-primary" : "text-text-muted hover:text-text"} title="Toggle plan (Ctrl+P)" aria-label="Toggle plan (Ctrl+P)">plan</button>
      <button type="button" onClick={() => toggle("showBudget")} className={ui.showBudget ? "text-primary" : "text-text-muted hover:text-text"} title="Toggle budget (Ctrl+B)" aria-label="Toggle budget (Ctrl+B)">budget</button>
      <button type="button" onClick={() => toggle("showWorkspace")} className={ui.showWorkspace ? "text-primary" : "text-text-muted hover:text-text"} title="Toggle workspace changes (Alt+W)" aria-label="Toggle workspace changes (Alt+W)">files</button>
      <button type="button" onClick={() => setToolsExpanded(!ui.showTools)} className={ui.showTools ? "text-primary" : "text-text-muted hover:text-text"} title={ui.showTools ? "Tool call boxes expanded — click to collapse them (Ctrl+T)" : "Tool call boxes collapsed — click to expand them (Ctrl+T)"} aria-label="Toggle tool call boxes (Ctrl+T)">tools</button>
      {/* The ``exit`` command as a button: detach, back to the connect screen; the session stays on the daemon. */}
      <button type="button" onClick={() => { exitToConnect().catch(() => undefined); }} className="text-text-muted hover:text-error" title="Detach from the session and return to the connect screen (the exit command)" aria-label="Exit (detach from the session)">exit</button>
    </div>
  );
}
