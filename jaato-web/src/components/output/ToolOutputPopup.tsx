/**
 * Floating live-output panel for running tools — the TUI's
 * ``tool_output_popup``: appears when a running tool starts producing
 * output, tails it, offers tabs across concurrent running tools, and
 * dismisses when the pinned tool finishes (unless it belongs to a
 * continuation group, e.g. an interactive shell, which keeps it open).
 */
import { useEffect, useMemo, useRef } from "react";
import { useJaato } from "@/store/store";
import type { ToolBlock } from "@/store/types";

export function ToolOutputPopup({ agentId }: { agentId: string }) {
  const blocks = useJaato((s) => s.blocks[agentId]);
  const popupCallId = useJaato((s) => s.ui.popupCallId);
  const setPopup = useJaato((s) => s.setPopup);
  const running = useMemo(() => (blocks ?? []).filter((b): b is ToolBlock => b.kind === "tool" && b.status === "running" && b.output.length > 0), [blocks]);
  const pinned = running.find((b) => b.callId === popupCallId) ?? null;
  const preRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    const el = preRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [pinned?.output.length]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key.toLowerCase() === "o" && running.length) {
        e.preventDefault();
        const i = running.findIndex((b) => b.callId === popupCallId);
        setPopup(running[(i + 1) % running.length]!.callId);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [running, popupCallId, setPopup]);

  if (!pinned) return null;
  return (
    <div className="absolute right-4 bottom-4 w-[min(46rem,60vw)] max-h-[45vh] flex flex-col rounded-lg border border-primary/50 bg-bg shadow-xl z-20" role="dialog" aria-label="Live tool output">
      <div className="flex items-center gap-1 px-2 py-1 border-b hairline text-xs">
        {running.map((b) => (
          <button
            key={b.callId}
            type="button"
            onClick={() => setPopup(b.callId)}
            className={`px-2 py-0.5 rounded font-mono ${b.callId === pinned.callId ? "bg-surface text-primary" : "text-text-muted hover:text-text"}`}
          >
            ● {b.toolName}
          </button>
        ))}
        <span className="flex-1" />
        <span className="text-text-muted hidden md:inline"><kbd>Ctrl</kbd>+<kbd>O</kbd> next</span>
        <button type="button" className="ml-2 text-text-muted hover:text-text" onClick={() => setPopup(null)} aria-label="Close">✕</button>
      </div>
      <pre ref={preRef} className="code-block whitespace-pre-wrap break-words p-2 overflow-auto flex-1">{pinned.output}</pre>
    </div>
  );
}
