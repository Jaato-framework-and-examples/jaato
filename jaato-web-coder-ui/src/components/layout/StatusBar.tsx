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
  const ui = useJaato((s) => s.ui);
  const dot = conn.phase === "connected" ? "bg-success" : conn.phase === "reconnecting" || conn.phase === "connecting" ? "bg-warning pulse" : "bg-error";
  const pct = ctx?.percentUsed;
  return (
    <div className="flex items-center gap-3 px-3 h-7 text-[11px] surface-2 border-t hairline font-mono select-none">
      <span className="flex items-center gap-1.5"><span className={`inline-block w-2 h-2 rounded-full ${dot}`} />{conn.phase}{conn.attempt ? ` #${conn.attempt}` : ""}</span>
      {conn.serverVersion && <span className="text-text-muted">server {conn.serverVersion}</span>}
      {ws && <span className="text-accent">{ws}</span>}
      {sessionId && <span className="text-text-muted" title={sessionId}>session {sessionId.slice(0, 8)}</span>}
      {(session.provider || session.model) && <span>{[session.provider, session.model].filter(Boolean).join(" / ")}</span>}
      {session.profile && <span className="text-text-muted">profile {session.profile}</span>}
      {pct != null && <span className={pct > 80 ? "text-error" : pct > 60 ? "text-warning" : ""}>ctx {pct.toFixed(0)}%</span>}
      {permStatus && <span className="text-warning">{permStatus}</span>}
      <span className="flex-1" />
      <button type="button" onClick={() => toggle("showPlan")} className={ui.showPlan ? "text-primary" : "text-text-muted hover:text-text"} title="Toggle plan (Ctrl+P)" aria-label="Toggle plan (Ctrl+P)">plan</button>
      <button type="button" onClick={() => toggle("showBudget")} className={ui.showBudget ? "text-primary" : "text-text-muted hover:text-text"} title="Toggle budget (Ctrl+B)" aria-label="Toggle budget (Ctrl+B)">budget</button>
      <button type="button" onClick={() => toggle("showWorkspace")} className={ui.showWorkspace ? "text-primary" : "text-text-muted hover:text-text"} title="Toggle workspace changes (Alt+W)" aria-label="Toggle workspace changes (Alt+W)">files</button>
      <button type="button" onClick={() => toggle("showTools")} className={ui.showTools ? "text-primary" : "text-text-muted hover:text-text"} title="Expand/collapse tool output (Ctrl+T)" aria-label="Expand/collapse tool output (Ctrl+T)">tools</button>
    </div>
  );
}
