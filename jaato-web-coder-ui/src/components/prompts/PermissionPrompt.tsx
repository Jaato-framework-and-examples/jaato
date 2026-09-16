/**
 * Inline permission card.  Mirrors the TUI's prompt: the tool and its
 * arguments (or the server's pre-formatted ``prompt_lines`` — a diff for
 * file edits), any analyzer warnings, then the response options.  Each
 * option carries the key the daemon expects back (``y``, ``n``, ``a``,
 * ``t``, ``i``, ``once``, ``never``, ``all`` …); Tab cycles focus, the
 * key or Enter answers, and the composer forwards typed keys too.
 */
import { useEffect } from "react";
import type { PendingPermission } from "@/store/types";
import { useJaato } from "@/store/store";
import { DiffLines } from "@/components/output/JMarkup";

export function PermissionPrompt({ p, onRespond }: { p: PendingPermission; onRespond: (key: string) => void }) {
  const focus = useJaato((s) => s.focusPermission);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      const inField = t && (t.tagName === "TEXTAREA" || t.tagName === "INPUT");
      if (e.key === "Tab" && !inField) {
        e.preventDefault();
        const n = p.options.length || 1;
        focus(p.requestId, (p.focus + (e.shiftKey ? -1 : 1) + n) % n);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [p, focus]);

  const argEntries = Object.entries(p.toolArgs ?? {});
  const level = p.warningLevel ?? "warning";
  return (
    <div className="my-2 rounded-lg border border-warning/60 surface-1 overflow-hidden" role="group" aria-label={`Permission request for ${p.toolName}`}>
      <div className="px-3 py-1.5 flex items-center gap-2 border-b hairline">
        <span className="text-warning">⚠</span>
        <span className="text-sm">Permission requested for <span className="font-mono text-accent font-semibold">{p.toolName}</span></span>
        {p.agentId !== "main" && <span className="text-xs text-text-muted">· {p.agentId}</span>}
      </div>
      {p.promptLines.length > 0 ? (
        p.formatHint === "diff" ? <DiffLines lines={p.promptLines} /> : <pre className="code-block whitespace-pre-wrap p-2 max-h-[40vh] overflow-auto">{p.promptLines.join("\n")}</pre>
      ) : argEntries.length > 0 ? (
        <dl className="px-3 py-2 grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-[13px] font-mono max-h-[40vh] overflow-auto">
          {argEntries.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-text-muted">{k}</dt>
              <dd className="whitespace-pre-wrap break-words">{typeof v === "string" ? v : JSON.stringify(v, null, 2)}</dd>
            </div>
          ))}
        </dl>
      ) : null}
      {p.warnings && (
        <div className={`px-3 py-1.5 text-xs whitespace-pre-wrap border-t hairline ${level === "error" ? "text-error" : level === "info" ? "text-text-muted" : "text-warning"}`}>{p.warnings}</div>
      )}
      <div className="px-3 py-2 flex flex-wrap gap-2 border-t hairline">
        {p.options.map((o, i) => (
          <button
            key={o.key + i}
            type="button"
            onClick={() => onRespond(o.key)}
            title={o.description ?? o.action ?? undefined}
            className={`px-2.5 py-1 rounded-md text-xs border font-mono ${i === p.focus ? "border-primary text-primary bg-surface" : "hairline text-text hover:bg-surface"}`}
          >
            <span className="text-accent">{o.key}</span> {o.label}
          </button>
        ))}
        {p.options.length === 0 && ["y", "n"].map((k) => (
          <button key={k} type="button" onClick={() => onRespond(k)} className="px-2.5 py-1 rounded-md text-xs border hairline font-mono hover:bg-surface">{k}</button>
        ))}
      </div>
    </div>
  );
}
