import { useCallback, useRef } from "react";
import { useJaato } from "@/store/store";
import { RAIL_WIDTH_MAX, RAIL_WIDTH_MIN } from "@/store/railWidth";

/**
 * The drag handle between the conversation pane and the session rail.
 *
 * A 7px strip on the rail's left edge: pointer-drag (mouse, pen or touch —
 * pointer events, with ``touch-action: none`` so a finger drags the rail
 * instead of scrolling the page) resizes the rail, and as a focusable
 * ``separator`` the arrow keys move it 16px at a time. The width lives in
 * the store (``ui.railWidth``) and is remembered per browser, so the rail
 * reopens at the size it was left.
 */
export function RailResizer() {
  const width = useJaato((s) => s.ui.railWidth);
  const setWidth = useJaato((s) => s.setRailWidth);
  const drag = useRef<{ startX: number; startW: number } | null>(null);

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 && e.pointerType === "mouse") return;
    drag.current = { startX: e.clientX, startW: width };
    e.currentTarget.setPointerCapture(e.pointerId);
    e.preventDefault();
  }, [width]);
  const onPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d) return;
    // The rail sits to the right, so moving the pointer left widens it.
    setWidth(d.startW + (d.startX - e.clientX));
  }, [setWidth]);
  const onPointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    drag.current = null;
    try { e.currentTarget.releasePointerCapture(e.pointerId); } catch { /* already released */ }
  }, []);
  const onKeyDown = useCallback((e: React.KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 64 : 16;
    if (e.key === "ArrowLeft") { setWidth(width + step); e.preventDefault(); }
    else if (e.key === "ArrowRight") { setWidth(width - step); e.preventDefault(); }
    else if (e.key === "Home") { setWidth(RAIL_WIDTH_MAX); e.preventDefault(); }
    else if (e.key === "End") { setWidth(RAIL_WIDTH_MIN); e.preventDefault(); }
  }, [setWidth, width]);

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize the session rail"
      aria-valuemin={RAIL_WIDTH_MIN}
      aria-valuemax={RAIL_WIDTH_MAX}
      aria-valuenow={width}
      tabIndex={0}
      title="Drag to resize the rail"
      className="hidden md:block shrink-0 w-[7px] -mr-[4px] -ml-[3px] z-10 cursor-col-resize select-none outline-none hover:bg-steel/40 focus-visible:bg-steel/60 active:bg-steel/60"
      style={{ touchAction: "none" }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onKeyDown={onKeyDown}
    />
  );
}
