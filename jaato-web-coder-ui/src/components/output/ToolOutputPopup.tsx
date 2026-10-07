/**
 * Floating live-output plate for running tools — the TUI's
 * ``tool_output_popup``: appears when a running tool starts producing
 * output, tails it, offers tabs across concurrent running tools, and
 * dismisses when the pinned tool finishes (unless it belongs to a
 * continuation group, e.g. an interactive shell, which keeps it open).
 * Drawn as a steel-edged ground plate over the composer (design frame
 * 04): ``Live · <tool>`` for the pinned one, the other running tools as
 * quiet kickers beside it.
 *
 * Stepping to the next running tool used to be a direct Ctrl+O listener
 * owned by this component; it is now the leader's ``O``
 * (``app/leaderKeys.ts``, struck through the command palette), and the
 * running set it steps over is ``store.runningToolCallIds`` -- the ONE
 * definition, so the popup and the leader cannot disagree about which
 * tools are running or in what order.
 */
import { useEffect, useMemo, useRef } from "react";
import { runningToolCallIds, useJaato } from "@/store/store";
import type { ToolBlock } from "@/store/types";
import { Plate } from "@/components/layout/Plate";

export function ToolOutputPopup({ agentId }: { agentId: string }) {
  const blocks = useJaato((s) => s.blocks[agentId]);
  const popupCallId = useJaato((s) => s.ui.popupCallId);
  const setPopup = useJaato((s) => s.setPopup);
  const runningIds = useMemo(() => runningToolCallIds({ blocks: { [agentId]: blocks ?? [] } }, agentId), [blocks, agentId]);
  const running = useMemo(
    () => runningIds.map((id) => (blocks ?? []).find((b): b is ToolBlock => b.kind === "tool" && b.callId === id)).filter((b): b is ToolBlock => b != null),
    [runningIds, blocks],
  );
  const pinned = running.find((b) => b.callId === popupCallId) ?? null;
  const preRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    const el = preRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [pinned?.output.length]);

  if (!pinned) return null;
  return (
    <Plate ground edge="steel" className="absolute right-5 bottom-[104px] w-[min(560px,70%)] max-h-[45vh] flex flex-col shadow-lg z-20" role="dialog" aria-label="Live tool output">
      <div className="flex items-center gap-2.5 px-2.5 py-1.5 border-b hairline">
        {running.map((b) => (
          <button
            key={b.callId}
            type="button"
            onClick={() => setPopup(b.callId)}
            className={`kicker tracking-[0.12em] text-[12px] ${b.callId === pinned.callId ? "" : "kicker-muted hover:text-text"}`}
          >
            {b.callId === pinned.callId ? "Live · " : ""}{b.toolName}
          </button>
        ))}
        <span className="flex-1" />
        <span className="font-mono text-[10px] text-text-muted hidden md:inline">⌘/Ctrl+K then O — next</span>
        <button type="button" className="text-text-muted hover:text-text" onClick={() => setPopup(null)} aria-label="Close">✕</button>
      </div>
      <pre ref={preRef} className="code-block whitespace-pre-wrap break-words px-3 py-2.5 overflow-auto flex-1">{pinned.output}</pre>
    </Plate>
  );
}
