/**
 * One tool call as a row — the browser counterpart of the TUI's
 * ``ToolBlock`` rendering, drawn the way design frame 04 draws it: a
 * status glyph, the tool's name in the chrome face on a fixed column,
 * its arguments in monospace on the rest of the line, the duration at
 * the right edge; a running call sits on a steel tint and offers
 * ``Follow`` (the live-output popup).  The row toggles the live / final
 * output underneath, in a ground plate indented under the name.  Failed
 * calls open by default; successful ones honour the server's
 * ``show_output`` hint.
 *
 * A successful ``offer_download`` call (the host tool the model uses to
 * hand the user a file, ``app/downloads.ts``) also draws a download
 * button under its row, always visible: the button IS the tool's output,
 * so it cannot sit behind the expand toggle.
 */
import { memo, useState } from "react";
import type { ToolBlock } from "@/store/types";
import { useJaato } from "@/store/store";
import { Plate } from "@/components/layout/Plate";
import { MediaView } from "./MediaView";
import { hasServerMarkup, JMarkup } from "./JMarkup";
import { resolveToolArgs } from "@/protocol/toolIds";
import { downloadWorkspaceFile, OFFER_DOWNLOAD_TOOL } from "@/app/downloads";

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

/**
 * The button an ``offer_download`` call leaves in the chat.  The bytes are
 * fetched when it is clicked, not when it is drawn, so a file that changed
 * or vanished since the offer is reported here, on the button.
 */
export function DownloadChip({ path, label }: { path: string; label?: string }) {
  const [state, setState] = useState<{ busy: boolean; error: string | null }>({ busy: false, error: null });
  const name = path.split("/").pop() || path;
  const onClick = async () => {
    setState({ busy: true, error: null });
    const error = await downloadWorkspaceFile(path);
    setState({ busy: false, error });
  };
  return (
    <div className="ml-6 mt-1 mb-2 flex items-center gap-3 flex-wrap">
      <button type="button" className="btn btn-steel" onClick={() => { void onClick(); }} disabled={state.busy} aria-label={`Download ${path}`} title={path}>
        {state.busy ? "Downloading…" : `⤓ ${label || name}`}
      </button>
      {state.error && <span role="status" className="text-xs text-error">{name}: {state.error}</span>}
    </div>
  );
}

export const ToolBlockView = memo(function ToolBlockView({ block }: { block: ToolBlock }) {
  const toggle = useJaato((s) => s.toggleTool);
  // Hashed tool / category ids in the arguments, shown by name (protocol/toolIds.ts).
  const toolIdNames = useJaato((s) => s.toolIdNames);
  const setPopup = useJaato((s) => s.setPopup);
  const hasBody = block.output.length > 0 || block.media.length > 0 || !!block.errorMessage;
  const offered = (toolIdNames[block.toolName] ?? block.toolName) === OFFER_DOWNLOAD_TOOL && block.status === "success" && typeof block.args.path === "string" ? block.args.path : null;
  const running = block.status === "running";
  return (
    <div data-testid="tool-block" className="border-t hairline">
      <button
        type="button"
        onClick={() => toggle(block.agentId, block.id)}
        className={`w-full flex items-center gap-3 py-1.5 pr-1 text-left ${running ? "bg-steel/8" : "hover:bg-tint/70"}`}
        aria-expanded={block.expanded}
      >
        <span className="w-3 shrink-0 text-center">
          <StatusGlyph status={block.status} />
        </span>
        <span className={`chrome w-[120px] shrink-0 truncate ${block.status === "error" ? "text-error" : ""}`}>{block.toolName}</span>
        <span className="font-mono text-xs text-text-muted truncate flex-1">{summarizeArgs(resolveToolArgs(block.args, toolIdNames))}</span>
        {block.backgrounded && <span className="kicker kicker-muted text-[10px]">bg</span>}
        {running && block.output ? (
          <span
            role="button"
            tabIndex={0}
            className="chrome-sm font-heading font-medium uppercase tracking-[0.08em] text-steel w-[52px] text-right"
            onClick={(e) => { e.stopPropagation(); setPopup(block.callId); }}
            onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); setPopup(block.callId); } }}
            title="Follow live output"
          >
            Follow
          </span>
        ) : typeof block.durationSeconds === "number" ? (
          <span className="font-mono text-[11px] text-text-muted w-[52px] text-right">{block.durationSeconds.toFixed(block.durationSeconds < 10 ? 2 : 1)}s</span>
        ) : (
          <span className="w-[52px]" />
        )}
      </button>
      {offered && <DownloadChip path={offered} label={typeof block.args.label === "string" ? block.args.label : undefined} />}
      {block.expanded && hasBody && (
        <Plate ground corners="two" edge={block.status === "error" ? "error" : "hairline"} className="ml-6 mt-0.5 mb-2.5">
          {block.errorMessage && <div className="px-3 py-1.5 text-error text-xs whitespace-pre-wrap">{block.errorMessage}</div>}
          {block.output && (
            hasServerMarkup(block.output) ? (
              <div className="px-3 py-1.5 text-[13px]"><JMarkup text={block.output} /></div>
            ) : (
              <pre className="code-block whitespace-pre-wrap break-words px-3 py-2 max-h-[60vh] overflow-auto">{block.output}</pre>
            )
          )}
          <MediaView items={block.media} />
        </Plate>
      )}
    </div>
  );
});
