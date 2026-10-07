/**
 * The conversation minimap rail (design handoff "Conversation minimap
 * rail", option 1a): a 64px strip on the right of the transcript that
 * draws a miniature of it -- every word a short bar at its position and
 * in its colour, user messages over white cards -- with a frame showing
 * the part on screen. It replaces the pane's native scrollbar, which is
 * too thin to grab on a touch tablet.
 *
 * **Drawing.** One ``<canvas>`` holds the miniature. It is computed from
 * the transcript MODEL (``minimap.ts``), not from the DOM, because the
 * pane is virtualised and an off-screen item has no DOM. Each item's
 * layout is cached by ``layoutSignature`` (id, width, and what the item
 * draws), so a streaming tail re-lays out only itself; the canvas is
 * repainted, coalesced to one frame, when the items, the virtualizer's
 * measurements, the scroll height, the widths or the theme change. The
 * viewport frame, the "earlier history" rule and the scrub label are DOM
 * overlays, so scrolling moves the frame without repainting the canvas.
 * The backing store is scaled by ``devicePixelRatio`` so 1px bars stay
 * crisp.
 *
 * **Interaction.** A pointer down anywhere on the rail (mouse, touch or
 * pen) captures the pointer and jumps: the tapped point is centred on
 * screen. Moving while captured scrubs continuously, and a label left of
 * the rail names the turn under the pointer. Release hides it. The rail
 * only ever sets the scroller's ``scrollTop``, so follow mode and
 * history paging react through the pane's own scroll handler, exactly as
 * for a wheel scroll. It is not a tab stop; the pane's keys still work.
 *
 * **Fonts.** Word widths are measured with a canvas in the fonts the pane
 * renders in. Barlow is self-hosted and loads after first paint, so the
 * width cache is dropped and the rail repainted when a font finishes
 * loading.
 */
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type PointerEvent, type RefObject } from "react";
import type { TranscriptItem } from "@/store/transcript";
import { useJaato } from "@/store/store";
import {
  TRACK_INSET_X, TRACK_INSET_Y, layoutItem, layoutSignature, placeItems, scrollTopFor, scrubLabel, turnAt,
  type ItemLayout, type ItemSlot, type Measure, type MinimapTone,
} from "./minimap";

/** The rail's width, in CSS pixels. */
export const RAIL_WIDTH = 64;
/** The scroller's horizontal padding (``px-5``), subtracted to get the content width. */
const SCROLLER_PAD_X = 20;
/** The viewport frame's minimum height. */
const MIN_VIEWPORT = 14;

const TONE_VARS: Record<MinimapTone, string> = {
  ink: "--c-text",
  muted: "--c-text-muted",
  accent: "--c-steel",
  success: "--c-success",
  error: "--c-error",
  warning: "--c-warning",
};

let measureCtx: CanvasRenderingContext2D | null | undefined;
const widthCache = new Map<string, number>();

/** Canvas text measurement with a per-(font, word) cache; a crude estimate where there is no canvas. */
const measureText: Measure = (text, font) => {
  const key = `${font}\u0000${text}`;
  const hit = widthCache.get(key);
  if (hit !== undefined) return hit;
  if (measureCtx === undefined) {
    try { measureCtx = document.createElement("canvas").getContext("2d"); } catch { measureCtx = null; }
  }
  let w: number;
  if (measureCtx) {
    measureCtx.font = font;
    w = measureCtx.measureText(text).width;
  } else {
    const px = Number(/(\d+(?:\.\d+)?)px/.exec(font)?.[1] ?? 14);
    w = text.length * px * 0.55;
  }
  widthCache.set(key, w);
  return w;
};

interface Metrics {
  scrollTop: number;
  scrollHeight: number;
  clientHeight: number;
  contentWidth: number;
  listTop: number;
}

function readMetrics(scroller: HTMLElement, list: HTMLElement | null): Metrics {
  const listTop = list ? list.getBoundingClientRect().top - scroller.getBoundingClientRect().top + scroller.scrollTop : 0;
  return {
    scrollTop: scroller.scrollTop,
    scrollHeight: Math.max(1, scroller.scrollHeight),
    clientHeight: scroller.clientHeight,
    contentWidth: Math.max(1, scroller.clientWidth - SCROLLER_PAD_X * 2),
    listTop,
  };
}

function sameMetrics(a: Metrics, b: Metrics): boolean {
  return a.scrollTop === b.scrollTop && a.scrollHeight === b.scrollHeight && a.clientHeight === b.clientHeight
    && a.contentWidth === b.contentWidth && a.listTop === b.listTop;
}

export interface MinimapRailProps {
  /** The transcript scroller the rail drives and mirrors. */
  scrollerRef: RefObject<HTMLDivElement | null>;
  /** The virtualised list inside it, whose top offsets every item. */
  listRef: RefObject<HTMLDivElement | null>;
  items: readonly TranscriptItem[];
  /** The virtualizer's measurements, index-aligned with ``items``. */
  slots: readonly ItemSlot[];
  /** Earlier history exists that is not loaded: a dashed rule marks the top of the track. */
  hasMore: boolean;
}

export function MinimapRail({ scrollerRef, listRef, items, slots, hasMore }: MinimapRailProps) {
  const railRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const layouts = useRef(new Map<string, ItemLayout>());
  const theme = useJaato((s) => s.ui.theme);
  const [metrics, setMetrics] = useState<Metrics>({ scrollTop: 0, scrollHeight: 1, clientHeight: 0, contentWidth: 1, listTop: 0 });
  const [railHeight, setRailHeight] = useState(0);
  const [fontEpoch, setFontEpoch] = useState(0);
  const [drag, setDrag] = useState<{ y: number; turn: number; text: string } | null>(null);

  // Scroll and resize update the metrics, at most once a frame.
  useEffect(() => {
    const scroller = scrollerRef.current;
    const rail = railRef.current;
    if (!scroller || !rail) return;
    let frame = 0;
    const update = () => {
      frame = 0;
      const next = readMetrics(scroller, listRef.current);
      setMetrics((prev) => (sameMetrics(prev, next) ? prev : next));
      setRailHeight(rail.clientHeight);
    };
    const schedule = () => { if (!frame) frame = requestAnimationFrame(update); };
    update();
    scroller.addEventListener("scroll", schedule, { passive: true });
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(schedule) : null;
    ro?.observe(scroller);
    ro?.observe(rail);
    if (listRef.current) ro?.observe(listRef.current);
    return () => {
      scroller.removeEventListener("scroll", schedule);
      ro?.disconnect();
      if (frame) cancelAnimationFrame(frame);
    };
  }, [scrollerRef, listRef]);

  // Content changes (new items, new measurements) change the scroll height
  // without a scroll or resize event on the scroller, so re-read after render.
  useLayoutEffect(() => {
    const scroller = scrollerRef.current;
    if (!scroller) return;
    const next = readMetrics(scroller, listRef.current);
    setMetrics((prev) => (sameMetrics(prev, next) ? prev : next));
  }, [items, slots, scrollerRef, listRef]);

  // A web font that finishes loading changes every width: drop the caches.
  useEffect(() => {
    const fonts = typeof document !== "undefined" ? document.fonts : undefined;
    if (!fonts?.addEventListener) return;
    const onLoaded = () => { widthCache.clear(); layouts.current.clear(); setFontEpoch((n) => n + 1); };
    fonts.addEventListener("loadingdone", onLoaded);
    return () => fonts.removeEventListener("loadingdone", onLoaded);
  }, []);

  // Lay out every item, reusing cached layouts; only changed items are recomputed.
  const placed = useMemo(() => {
    const width = metrics.contentWidth;
    const cache = layouts.current;
    const used = new Set<string>();
    const itemLayouts = items.map((item) => {
      const key = layoutSignature(item, width);
      used.add(key);
      let layout = cache.get(key);
      if (!layout) {
        layout = layoutItem(item, width, measureText);
        cache.set(key, layout);
      }
      return layout;
    });
    for (const key of cache.keys()) if (!used.has(key)) cache.delete(key);
    return placeItems(itemLayouts, slots, metrics.listTop);
    // fontEpoch: a font load invalidated the cache contents.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items, slots, metrics.contentWidth, metrics.listTop, fontEpoch]);

  const turns = useMemo(() => {
    const out: { top: number; turn: number; text: string }[] = [];
    items.forEach((item, i) => {
      const slot = slots[i];
      if (item.kind === "userMessage" && slot) out.push({ top: metrics.listTop + slot.start, turn: item.turn, text: item.text });
    });
    return out;
  }, [items, slots, metrics.listTop]);

  // Paint the miniature: once per frame at most, and only when content changes.
  useEffect(() => {
    const canvas = canvasRef.current;
    const rail = railRef.current;
    if (!canvas || !rail) return;
    const frame = requestAnimationFrame(() => {
      const cssW = rail.clientWidth;
      const cssH = rail.clientHeight;
      if (!cssW || !cssH) return;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
      let ctx: CanvasRenderingContext2D | null = null;
      try { ctx = canvas.getContext("2d"); } catch { ctx = null; }
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, cssH);
      const root = getComputedStyle(document.documentElement);
      const color = (v: string) => root.getPropertyValue(v).trim();
      const trackW = cssW - TRACK_INSET_X * 2;
      const trackH = cssH - TRACK_INSET_Y * 2;
      const sy = trackH / metrics.scrollHeight;
      const sx = trackW / metrics.contentWidth;

      // User-message cards: surface, full track width plus 2px each side, a hairline edge.
      ctx.fillStyle = color("--c-surface") || "#fff";
      ctx.strokeStyle = color("--c-divider") || "rgba(29,31,32,.12)";
      ctx.lineWidth = 1;
      for (const c of placed.cards) {
        const y = TRACK_INSET_Y + c.y * sy;
        const h = Math.max(1, c.h * sy);
        ctx.fillRect(TRACK_INSET_X - 2, y, trackW + 4, h);
        ctx.strokeRect(TRACK_INSET_X - 2 + 0.5, y + 0.5, trackW + 3, Math.max(0, h - 1));
      }

      // Word bars, grouped by tone so each colour is set once.
      ctx.globalAlpha = 0.85;
      const byTone = new Map<MinimapTone, typeof placed.bars>();
      for (const b of placed.bars) {
        const list = byTone.get(b.tone);
        if (list) list.push(b); else byTone.set(b.tone, [b]);
      }
      for (const [tone, bars] of byTone) {
        ctx.fillStyle = color(TONE_VARS[tone]) || "#1d1f20";
        for (const b of bars) {
          const x = TRACK_INSET_X + Math.min(trackW, b.x * sx);
          const w = Math.max(1, Math.min(trackW - (x - TRACK_INSET_X), b.w * sx));
          ctx.fillRect(x, TRACK_INSET_Y + b.y * sy, w, Math.max(1, b.h * sy));
        }
      }
      ctx.globalAlpha = 1;
    });
    return () => cancelAnimationFrame(frame);
  }, [placed, metrics.scrollHeight, metrics.contentWidth, railHeight, theme]);

  const scrub = useCallback((clientY: number) => {
    const scroller = scrollerRef.current;
    const rail = railRef.current;
    if (!scroller || !rail) return;
    const rect = rail.getBoundingClientRect();
    const trackH = Math.max(1, rect.height - TRACK_INSET_Y * 2);
    const f = Math.min(1, Math.max(0, (clientY - rect.top - TRACK_INSET_Y) / trackH));
    const sh = scroller.scrollHeight;
    scroller.scrollTop = scrollTopFor(f, sh, scroller.clientHeight);
    const t = turnAt(turns, f * sh);
    setDrag(t ? { y: clientY - rect.top, turn: t.turn, text: t.text } : { y: clientY - rect.top, turn: 0, text: "" });
  }, [scrollerRef, turns]);

  const dragging = useRef(false);
  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    dragging.current = true;
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* a synthetic pointer */ }
    scrub(e.clientY);
  };
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => { if (dragging.current) scrub(e.clientY); };
  const onPointerEnd = (e: PointerEvent<HTMLDivElement>) => {
    dragging.current = false;
    try { e.currentTarget.releasePointerCapture(e.pointerId); } catch { /* not captured */ }
    setDrag(null);
  };

  const trackH = Math.max(0, railHeight - TRACK_INSET_Y * 2);
  const viewTop = TRACK_INSET_Y + (metrics.scrollTop / metrics.scrollHeight) * trackH;
  const viewH = Math.max(MIN_VIEWPORT, Math.min(trackH, (metrics.clientHeight / metrics.scrollHeight) * trackH));
  const label = drag && drag.turn > 0 ? scrubLabel(drag.turn, drag.text) : null;

  return (
    <div
      ref={railRef}
      className="minimap-rail relative shrink-0 border-l hairline select-none"
      style={{ width: RAIL_WIDTH, touchAction: "none", cursor: drag ? "grabbing" : "grab" }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerEnd}
      onPointerCancel={onPointerEnd}
      data-testid="minimap-rail"
      aria-hidden="true"
    >
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
      {hasMore && (
        <div className="absolute pointer-events-none border-t border-dashed" style={{ top: TRACK_INSET_Y, left: TRACK_INSET_X, right: TRACK_INSET_X, borderColor: "color-mix(in srgb, var(--c-text) 35%, transparent)" }} data-testid="minimap-more" />
      )}
      <div
        className="absolute pointer-events-none box-border"
        style={{
          top: viewTop,
          height: viewH,
          left: TRACK_INSET_X - 4,
          right: TRACK_INSET_X - 4,
          border: "2px solid var(--c-steel)",
          background: "color-mix(in srgb, var(--c-steel) 12%, transparent)",
        }}
        data-testid="minimap-viewport"
      />
      {label && (
        <div
          className="absolute pointer-events-none whitespace-nowrap overflow-hidden text-ellipsis z-20 font-sans text-[12px]"
          style={{
            right: "calc(100% + 8px)",
            top: drag!.y,
            transform: "translateY(-50%)",
            maxWidth: 320,
            padding: "6px 10px",
            background: "var(--c-text)",
            color: "var(--c-bg)",
          }}
          data-testid="minimap-label"
        >
          <span className="font-mono" style={{ color: "color-mix(in srgb, var(--c-steel) 45%, var(--c-bg))" }}>{label.number}</span> {label.text}
        </div>
      )}
    </div>
  );
}
