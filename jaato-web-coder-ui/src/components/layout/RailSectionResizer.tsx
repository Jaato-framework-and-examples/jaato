import { useCallback, useRef } from "react";
import { useJaato } from "@/store/store";
import {
  boundaryPercent,
  moveBoundary,
  resetPair,
  type RailSectionId,
} from "@/store/railSplits";

/**
 * The horizontal drag handle between two adjacent OPEN rail sections.
 *
 * A thin strip that sits on the boundary between the section above and the
 * section below.  Dragging it moves height between those two neighbours only
 * (split-pane behaviour — the others don't move); as a focusable
 * ``separator`` the arrow keys move it, and a double-click levels the pair.
 * The split weights live in the store (``ui.railSplits``) and are remembered
 * per browser, so a reopened section returns to the size it was left at.
 *
 * The pixel geometry the arithmetic needs is measured from the two sections'
 * rendered heights (via their ``data-rail-section`` markers) at the moment of
 * the drag, so a fraction is only ever derived against real layout.  Pointer
 * events carry ``touch-action: none`` so a finger drags the boundary instead
 * of scrolling the rail.
 */
export function RailSectionResizer({
  aboveId,
  belowId,
  aboveTitle,
  belowTitle,
}: {
  aboveId: RailSectionId;
  belowId: RailSectionId;
  aboveTitle: string;
  belowTitle: string;
}) {
  const splits = useJaato((s) => s.ui.railSplits);
  const setSplits = useJaato((s) => s.setRailSplits);
  const drag = useRef<{ startY: number; pxAbove: number; pxBelow: number; splits: typeof splits } | null>(null);

  // The section heights on either side of this handle, read from the rail the
  // handle is mounted in so a second rail elsewhere on the page cannot answer.
  const neighbourHeights = useCallback((el: HTMLElement): { above: number; below: number } => {
    const rail = el.closest('[data-rail]');
    const measure = (id: RailSectionId) => {
      const node = rail?.querySelector<HTMLElement>(`[data-rail-section="${id}"]`);
      return node ? node.getBoundingClientRect().height : 0;
    };
    return { above: measure(aboveId), below: measure(belowId) };
  }, [aboveId, belowId]);

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 && e.pointerType === "mouse") return;
    const { above, below } = neighbourHeights(e.currentTarget);
    drag.current = { startY: e.clientY, pxAbove: above, pxBelow: below, splits: useJaato.getState().ui.railSplits };
    e.currentTarget.setPointerCapture(e.pointerId);
    e.preventDefault();
  }, [neighbourHeights]);

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d) return;
    // Dragging DOWN grows the section above and shrinks the one below.
    setSplits(moveBoundary(d.splits, aboveId, belowId, d.pxAbove, d.pxBelow, e.clientY - d.startY));
  }, [setSplits, aboveId, belowId]);

  const onPointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    drag.current = null;
    try { e.currentTarget.releasePointerCapture(e.pointerId); } catch { /* already released */ }
  }, []);

  const onKeyDown = useCallback((e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
    const { above, below } = neighbourHeights(e.currentTarget);
    const step = (e.shiftKey ? 48 : 16) * (e.key === "ArrowDown" ? 1 : -1);
    setSplits(moveBoundary(useJaato.getState().ui.railSplits, aboveId, belowId, above, below, step));
    e.preventDefault();
  }, [neighbourHeights, setSplits, aboveId, belowId]);

  const onDoubleClick = useCallback(() => {
    setSplits(resetPair(useJaato.getState().ui.railSplits, aboveId, belowId));
  }, [setSplits, aboveId, belowId]);

  return (
    <div
      role="separator"
      aria-orientation="horizontal"
      aria-label={`Resize between ${aboveTitle} and ${belowTitle}`}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={boundaryPercent(splits, aboveId, belowId)}
      tabIndex={0}
      title="Drag to resize; double-click to reset"
      className="shrink-0 h-[7px] -my-[3px] z-10 cursor-row-resize select-none outline-none hover:bg-steel/40 focus-visible:bg-steel/60 active:bg-steel/60"
      style={{ touchAction: "none" }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onKeyDown={onKeyDown}
      onDoubleClick={onDoubleClick}
    />
  );
}
