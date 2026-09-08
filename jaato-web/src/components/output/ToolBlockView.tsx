/**
 * One tool call as a collapsible block — the browser counterpart of the
 * TUI's ``ToolBlock`` rendering: a header line (status glyph, tool
 * name, a one-line argument summary, duration) that toggles the live /
 * final output underneath.  Failed calls open by default; successful
 * ones honour the server's ``show_output`` hint.
 */
import { memo } from "react";
import type { ToolBlock } from "@/store/types";
import { useJaato } from "@/store/store";
import { MediaView } from "./MediaView";
import { JMarkup } from "./JMarkup";

export function summarizeArgs(args: Record<string, unknown>, max = 110): string {
  const parts: string[] = [];
  for (const [k, v] of Object.entries(args)) {
    let s: string;
    if (typeof v === "string") s = v;
    else if (v === null || v === undefined) continue;
    else s = JSON.stringify(v);
    s = s.replace(/\s+/g, " ");
    parts.push(`${k}=${s.length > 60 ? s.slice(0, 57) + "…" : s}`);
  }
  const joined = parts.join("  ");
  return joined.length > max ? joined.slice(0, max - 1) + "…" : joined;
}

function StatusGlyph({ status }: { status: ToolBlock["status"] }) {
  if (status === "running") return <span className="text-primary pulse" aria-label="running">●</span>;
  if (status === "success") return <span className="text-success" aria-label="succeeded">✓</span>;
  return <span className="text-error" aria-label="failed">✗</span>;
}

export const ToolBlockView = memo(function ToolBlockView({ block }: { block: ToolBlock }) {
  const toggle = useJaato((s) => s.toggleTool);
  const setPopup = useJaato((s) => s.setPopup);
  const hasBody = block.output.length > 0 || block.media.length > 0 || !!block.errorMessage;
  return (
    <div className={`my-1 rounded-md border hairline ${block.status === "error" ? "border-error/50" : ""} surface-1`}>
      <button
        type="button"
        onClick={() => toggle(block.agentId, block.id)}
        className="w-full flex items-center gap-2 px-2 py-1 text-left font-mono text-[12.5px] hover:bg-surface/60"
        aria-expanded={block.expanded}
      >
        <span className="w-3 text-text-muted">{block.expanded ? "▾" : "▸"}</span>
        <StatusGlyph status={block.status} />
        <span className="text-accent font-semibold">{block.toolName}</span>
        <span className="text-text-muted truncate flex-1">{summarizeArgs(block.args)}</span>
        {block.backgrounded && <span className="text-[10px] uppercase text-warning">bg</span>}
        {block.status === "running" && block.output && (
          <span
            role="button"
            tabIndex={0}
            className="text-[11px] text-text-muted hover:text-primary"
            onClick={(e) => { e.stopPropagation(); setPopup(block.callId); }}
            onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); setPopup(block.callId); } }}
            title="Follow live output"
          >
            follow
          </span>
        )}
        {typeof block.durationSeconds === "number" && (
          <span className="text-[11px] text-text-muted">{block.durationSeconds.toFixed(block.durationSeconds < 10 ? 2 : 1)}s</span>
        )}
      </button>
      {block.expanded && hasBody && (
        <div className="border-t hairline">
          {block.errorMessage && <div className="px-2 py-1 text-error text-xs whitespace-pre-wrap">{block.errorMessage}</div>}
          {block.output && (
            block.output.includes("<j-") ? (
              <div className="px-2 py-1 text-[13px]"><JMarkup text={block.output} /></div>
            ) : (
              <pre className="code-block whitespace-pre-wrap break-words p-2 max-h-[60vh] overflow-auto">{block.output}</pre>
            )
          )}
          <MediaView items={block.media} />
        </div>
      )}
    </div>
  );
});
