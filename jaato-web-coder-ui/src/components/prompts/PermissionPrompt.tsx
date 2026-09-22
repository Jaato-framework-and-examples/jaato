/**
 * Inline permission plate (design frame 05): a full-width, warning-edged
 * plate carrying the tool and its arguments (or the server's
 * pre-formatted ``prompt_lines`` — a diff for file edits, at full
 * measure), any analyzer warnings, then the response options.  Each
 * option carries the key the daemon expects back (``y``, ``n``, ``a``,
 * ``t``, ``i``, ``once``, ``never``, ``all`` …); the focused option is
 * the one solid button, the refusals sit apart at the right edge, and
 * every button shows its key so the keyboard route is never a secret.
 * Tab cycles focus, the key or Enter answers, and the composer forwards
 * typed keys too.
 */
import { useEffect } from "react";
import type { PendingPermission } from "@/store/types";
import { useJaato } from "@/store/store";
import { DiffLines } from "@/components/output/JMarkup";
import { Plate } from "@/components/layout/Plate";
import { resolveToolArgs } from "@/protocol/toolIds";

/** A refusal, from what the daemon labelled it: ``deny``, ``never``, ``no``. */
function isRefusal(o: { key: string; label?: string; action?: string | null }): boolean {
  return /\b(deny|denied|never|reject|no)\b/i.test(`${o.label ?? ""} ${o.action ?? ""}`) || o.key === "n";
}

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

  const toolIdNames = useJaato((s) => s.toolIdNames);
  const argEntries = Object.entries(resolveToolArgs(p.toolArgs ?? {}, toolIdNames));
  const level = p.warningLevel ?? "warning";
  const options = p.options.length ? p.options : [{ key: "y", label: "yes" }, { key: "n", label: "no" }];
  const grants = options.map((o, i) => [o, i] as const).filter(([o]) => !isRefusal(o));
  const refusals = options.map((o, i) => [o, i] as const).filter(([o]) => isRefusal(o));
  const button = ([o, i]: readonly [{ key: string; label?: string; description?: string | null; action?: string | null }, number]) => (
    <button
      key={o.key + i}
      type="button"
      onClick={() => onRespond(o.key)}
      title={o.description ?? o.action ?? undefined}
      className={`btn ${i === p.focus ? "btn-primary" : isRefusal(o) ? "btn-danger" : ""}`}
      aria-current={i === p.focus ? "true" : undefined}
    >
      {o.label ?? o.key} <span className="key">{o.key}</span>
    </button>
  );
  return (
    <Plate edge="warning" className="my-3" role="group" aria-label={`Permission request for ${p.toolName}`}>
      <div className="px-3.5 py-2.5 flex items-center gap-3 border-b hairline">
        <span className="text-warning">⚠</span>
        <span className="kicker text-warning">Permission requested for</span>
        <span className="font-mono text-sm">{p.toolName}</span>
        <span className="flex-1" />
        <span className="font-mono text-[11px] text-text-muted">agent {p.agentId}</span>
      </div>
      {p.promptLines.length > 0 ? (
        p.formatHint === "diff" ? <DiffLines lines={p.promptLines} /> : <pre className="code-block whitespace-pre-wrap px-3.5 py-2.5 max-h-[40vh] overflow-auto">{p.promptLines.join("\n")}</pre>
      ) : argEntries.length > 0 ? (
        <dl className="px-3.5 py-2.5 grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-[13px] font-mono max-h-[40vh] overflow-auto m-0">
          {argEntries.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-text-muted">{k}</dt>
              <dd className="whitespace-pre-wrap break-words m-0">{typeof v === "string" ? v : JSON.stringify(v, null, 2)}</dd>
            </div>
          ))}
        </dl>
      ) : null}
      {p.warnings && (
        <div className={`px-3.5 py-2 text-[13px] whitespace-pre-wrap border-t hairline ${level === "error" ? "text-error" : level === "info" ? "text-text-muted" : "text-warning"}`}>{p.warnings}</div>
      )}
      <div className="px-3.5 py-3 flex flex-wrap items-center gap-2.5 border-t hairline">
        {grants.map(button)}
        <span className="flex-1" />
        {refusals.map(button)}
      </div>
    </Plate>
  );
}
