/**
 * The conversation column for one agent: a virtualised, auto-following
 * list of blocks.  Following stops when the user scrolls up and resumes
 * when they return to the bottom (or press End), like a terminal.
 *
 * The pane also numbers the turns: the n-th user message in the column
 * is turn n, and ``UserBlockView`` prints it in the gutter.
 */
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { selectBlocks, useJaato } from "@/store/store";
import { BlockView } from "./Blocks";

export function OutputPane({ agentId }: { agentId: string }) {
  const blocks = useJaato(selectBlocks(agentId));
  const parentRef = useRef<HTMLDivElement>(null);
  const [follow, setFollow] = useState(true);

  const turnOf = useMemo(() => {
    const m = new Map<string, number>();
    let n = 0;
    for (const b of blocks) if (b.kind === "user") m.set(b.id, ++n);
    return m;
  }, [blocks]);

  const virtualizer = useVirtualizer({
    count: blocks.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 64,
    overscan: 8,
    getItemKey: (i) => blocks[i]?.id ?? i,
  });

  const lastLen = useRef(0);
  const lastTail = useRef<string | null>(null);
  const tail = blocks[blocks.length - 1];
  const tailSig = tail ? `${tail.id}:${tail.kind === "text" ? tail.text.length : tail.kind === "tool" ? tail.output.length + (tail.expanded ? 1 : 0) : 0}` : null;

  useLayoutEffect(() => {
    if (!follow) return;
    if (blocks.length !== lastLen.current || tailSig !== lastTail.current) {
      lastLen.current = blocks.length;
      lastTail.current = tailSig;
      const el = parentRef.current;
      if (el) requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
    }
  }, [blocks.length, tailSig, follow]);

  useEffect(() => {
    const el = parentRef.current;
    if (!el) return;
    const onScroll = () => {
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
      setFollow(atBottom);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      const inField = t && (t.tagName === "TEXTAREA" || t.tagName === "INPUT");
      if (inField && e.key !== "End" && e.key !== "Home") return;
      const el = parentRef.current;
      if (!el) return;
      if (e.key === "End" && (e.ctrlKey || !inField)) { el.scrollTop = el.scrollHeight; setFollow(true); }
      if (e.key === "Home" && (e.ctrlKey || !inField)) { el.scrollTop = 0; setFollow(false); }
      if (!inField && e.key === "PageDown") el.scrollTop += el.clientHeight * 0.9;
      if (!inField && e.key === "PageUp") el.scrollTop -= el.clientHeight * 0.9;
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const items = virtualizer.getVirtualItems();
  return (
    <div ref={parentRef} className="relative flex-1 overflow-y-auto px-5 py-3" data-testid="output-pane">
      {blocks.length === 0 && (
        <div className="h-full flex items-center justify-center text-text-muted text-sm select-none">
          <div className="text-center max-w-[52ch] space-y-2">
            <div className="display text-[34px] text-text">jaato</div>
            <div>Type a message, or a command such as <kbd>help</kbd>, <kbd>tools</kbd>, <kbd>model</kbd>.</div>
            <div>Commands are plain words — the composer proposes them as you type; <kbd>Esc</kbd> dismisses the proposal to send the word verbatim.</div>
          </div>
        </div>
      )}
      <div style={{ height: virtualizer.getTotalSize(), position: "relative", width: "100%" }}>
        {items.map((vi) => {
          const block = blocks[vi.index]!;
          return (
            <div
              key={vi.key}
              data-index={vi.index}
              ref={virtualizer.measureElement}
              style={{ position: "absolute", top: 0, left: 0, width: "100%", transform: `translateY(${vi.start}px)` }}
            >
              <BlockView block={block} turn={turnOf.get(block.id)} />
            </div>
          );
        })}
      </div>
      {!follow && blocks.length > 0 && (
        <button
          type="button"
          className="sticky bottom-3 left-1/2 -translate-x-1/2 btn btn-sm bg-surface text-text-muted shadow-md"
          onClick={() => { const el = parentRef.current; if (el) { el.scrollTop = el.scrollHeight; } setFollow(true); }}
        >
          ↓ follow output
        </button>
      )}
    </div>
  );
}
