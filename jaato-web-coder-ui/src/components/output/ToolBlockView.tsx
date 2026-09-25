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
 * ``toolClass`` (jaato/#1304 §1) adds two class-specific extras, both
 * built from what the CLIENT already has -- the call's own arguments and
 * its accumulated output -- because neither a ``diff`` field on
 * ``tool.call_end`` nor a workspace diff viewer exists yet (see
 * ``protocol/toolPreview.ts``'s own docstring for that gap):
 *
 * - ``write``: an inline diff/content preview, always shown (not behind
 *   the expand toggle -- a write is exactly the row a reviewer wants to
 *   see without a click), with an "Open diff" action that downloads the
 *   current file until a real diff viewer exists.
 * - ``exec``: the command itself as the row's title (rather than a
 *   generic argument summary) and, while collapsed, a trailing-lines
 *   preview of its output with a "Copy output" action that copies the
 *   WHOLE accumulated output, not just the preview.
 *
 * The caller (``ToolGroupView``, for a ``"row"``-mode group) already
 * knows the class from the resolved display name and passes it in;
 * ``toolClass`` is re-derived here only as a fallback for a caller (a
 * test, a future direct use) that does not.
 *
 * A successful ``offer_download`` call (the host tool the model uses to
 * hand the user a file, ``app/downloads.ts``) also draws a download
 * button under its row, always visible: the button IS the tool's output,
 * so it cannot sit behind the expand toggle.
 */
import { memo, useState, type MouseEvent } from "react";
import type { ToolBlock } from "@/store/types";
import { useJaato } from "@/store/store";
import { Plate } from "@/components/layout/Plate";
import { MediaView } from "./MediaView";
import { hasServerMarkup, JMarkup, DiffLines } from "./JMarkup";
import { resolveToolArgs } from "@/protocol/toolIds";
import { downloadWorkspaceFile, OFFER_DOWNLOAD_TOOL } from "@/app/downloads";
import { classifyTool, type ToolClass } from "@/protocol/toolClass";
import { diffPreviewForCall, execOutputPreview, execTitle } from "@/protocol/toolPreview";

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

export const ToolBlockView = memo(function ToolBlockView({ block, toolClass }: { block: ToolBlock; toolClass?: ToolClass }) {
  const toggle = useJaato((s) => s.toggleTool);
  // Hashed tool / category ids in the arguments, shown by name (protocol/toolIds.ts).
  const toolIdNames = useJaato((s) => s.toolIdNames);
  const setPopup = useJaato((s) => s.setPopup);
  const displayName = toolIdNames[block.toolName] ?? block.toolName;
  const cls = toolClass ?? classifyTool(displayName);
  const resolvedArgs = resolveToolArgs(block.args, toolIdNames);
  const hasBody = block.output.length > 0 || block.media.length > 0 || !!block.errorMessage;
  const offered = displayName === OFFER_DOWNLOAD_TOOL && block.status === "success" && typeof block.args.path === "string" ? block.args.path : null;
  const running = block.status === "running";

  const diff = cls === "write" ? diffPreviewForCall(displayName, resolvedArgs) : null;
  const hasDiffPreview = !!diff && (diff.diffLines !== null || diff.text !== null);
  const execTail = cls === "exec" && block.output && !block.expanded ? execOutputPreview(block.output) : null;
  const title = cls === "exec" ? execTitle(displayName, resolvedArgs) : null;
  const copyOutput = (e: MouseEvent) => { e.stopPropagation(); navigator.clipboard?.writeText(block.output).catch(() => undefined); };

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
        <span className={`chrome w-[120px] shrink-0 truncate ${block.status === "error" ? "text-error" : ""}`}>{displayName}</span>
        {/* ``truncate`` alone does nothing on a flex item: its default
            ``min-width: auto`` keeps the item from shrinking below its own
            text, so a long argument summary widened the row (and the tool
            row's ancestors have no scroll of their own to absorb it) instead
            of eliding. ``min-w-0`` is what lets ``flex-1`` actually win. */}
        <span className="font-mono text-xs text-text-muted truncate flex-1 min-w-0">{title ?? summarizeArgs(resolvedArgs)}</span>
        {block.backgrounded && <span className="kicker kicker-muted text-[10px]">bg</span>}
        {cls === "exec" && block.output && (
          <span role="button" tabIndex={0} className="chrome-sm font-heading uppercase tracking-[0.08em] text-text-muted hover:text-steel shrink-0" onClick={copyOutput} onKeyDown={(e) => { if (e.key === "Enter") copyOutput(e as unknown as MouseEvent); }} title="Copy the whole output">
            copy
          </span>
        )}
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
      {hasDiffPreview && (
        <div className="ml-6 mr-1 mb-1.5">
          <div className="flex items-center justify-between px-0.5 pb-0.5">
            <span className="font-mono text-[10px] text-text-muted truncate">{diff!.path ?? ""}{diff!.truncated ? " (truncated)" : ""}</span>
            {diff!.path && (
              <button type="button" className="link text-[11px] shrink-0" onClick={(e) => { e.stopPropagation(); void downloadWorkspaceFile(diff!.path!); }} title="No diff viewer exists yet -- this downloads the current file">
                Open diff
              </button>
            )}
          </div>
          {diff!.diffLines ? <DiffLines lines={diff!.diffLines} /> : <pre className="code-block whitespace-pre-wrap break-words px-2 py-1.5">{diff!.text}</pre>}
        </div>
      )}
      {execTail && (
        <pre className="ml-6 mr-1 mb-1.5 code-block whitespace-pre-wrap break-words px-2 py-1.5 text-text-muted">{execTail.text}</pre>
      )}
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
