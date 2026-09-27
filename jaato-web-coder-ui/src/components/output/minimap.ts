/**
 * The conversation minimap's geometry (design handoff "Conversation
 * minimap rail", option 1a): pure functions from the transcript model
 * (``store/transcript.ts``) to the bars ``MinimapRail`` draws, and from a
 * point on the rail back to a scroll position and a turn.
 *
 * **Laid out from the model, never from the DOM.** ``OutputPane`` is
 * virtualised, so an item scrolled out of view has no DOM to measure.
 * Each item is laid out here against the content width with the font it
 * renders in (greedy wrap on spaces, word widths from ``measure``), and
 * placed at the offset the virtualizer holds for it: exact once the item
 * has been measured, the 64px estimate before that. The result is close
 * to what the pane draws, not a pixel mirror; the rail is for
 * orientation.
 *
 * **Colours are keys, not values.** A bar carries a ``MinimapTone``; the
 * rail resolves the tones to the theme's CSS variables once per draw,
 * so every theme works and nothing reads computed styles per word.
 *
 * Coordinates: an ``ItemLayout`` is in CSS pixels relative to the item's
 * own top-left. ``placeItems`` maps them into CONTENT coordinates (the
 * scroller's scroll space), and the rail maps content to track pixels.
 */
import type { TranscriptItem } from "@/store/transcript";

/** What colour a bar is drawn in; the rail maps each to a theme variable. */
export type MinimapTone = "ink" | "muted" | "accent" | "success" | "error" | "warning";

/** One word, as a bar. */
export interface WordBar {
  x: number;
  y: number;
  w: number;
  h: number;
  tone: MinimapTone;
}

/** One transcript item, laid out. ``card`` marks a user message (drawn over a surface card). */
export interface ItemLayout {
  bars: WordBar[];
  height: number;
  card: boolean;
}

/** Measures ``text`` in ``font`` (a CSS font shorthand), in CSS pixels. */
export type Measure = (text: string, font: string) => number;

/** The fonts the pane renders in (theme.css, Blocks.tsx, ToolGroupView.tsx). */
export const FONTS = {
  prose: { font: "15px Barlow", lineHeight: 15 * 1.55 },
  user: { font: "15px Barlow", lineHeight: 15 * 1.5 },
  small: { font: "13px Barlow", lineHeight: 13 * 1.5 },
  note: { font: "italic 12px Barlow", lineHeight: 12 * 1.5 },
  kicker: { font: "10px ui-monospace, monospace", lineHeight: 14 },
  mono: { font: "12px ui-monospace, monospace", lineHeight: 12 * 1.55 },
} as const;

/** The user card's gutter (the ``T<n>`` column plus its gap) and its max share of the rest. */
const USER_GUTTER = 48;
const USER_MAX_SHARE = 0.76;
const USER_PAD_X = 12;

/**
 * Lays ``text`` out as word bars, greedily wrapped at ``maxWidth``,
 * starting at ``(x0, y0)``. A hard newline starts a line; an empty line
 * still takes its height. Returns the bars and the y below the last line.
 * With ``nowrap`` a line is cut at ``maxWidth`` instead of wrapping.
 */
export function wrapWords(
  text: string,
  opts: { x0: number; y0: number; maxWidth: number; font: string; lineHeight: number; tone: MinimapTone; nowrap?: boolean },
  measure: Measure,
): { bars: WordBar[]; bottom: number } {
  const { x0, maxWidth, font, lineHeight, tone } = opts;
  const bars: WordBar[] = [];
  const space = measure(" ", font);
  let y = opts.y0;
  for (const line of text.split("\n")) {
    let x = 0;
    for (const word of line.split(/\s+/)) {
      if (!word) continue;
      const w = Math.min(measure(word, font), maxWidth);
      if (x > 0 && x + w > maxWidth) {
        if (opts.nowrap) break;
        y += lineHeight;
        x = 0;
      }
      bars.push({ x: x0 + x, y: y + lineHeight * 0.25, w, h: lineHeight * 0.5, tone });
      x += w + space;
    }
    y += lineHeight;
  }
  return { bars, bottom: y };
}

function toneForStyle(style: string): MinimapTone {
  if (style === "error") return "error";
  if (style === "warning") return "warning";
  return "muted";
}

/** A tool call's one-line summary: its arguments, as the row shows them. */
function argsText(args: Record<string, unknown>): string {
  return Object.values(args)
    .map((v) => (typeof v === "string" ? v : JSON.stringify(v)))
    .join(" ")
    .replace(/\s+/g, " ")
    .slice(0, 400);
}

/** The prose column's max width: ``.prose-j`` caps it at 74ch. */
function proseWidth(width: number, measure: Measure): number {
  return Math.min(width, 74 * measure("0", FONTS.prose.font));
}

/** Prose with fenced code: code lines are monospace, muted, and not wrapped. */
function layoutProse(text: string, width: number, measure: Measure, y0: number): { bars: WordBar[]; bottom: number } {
  const bars: WordBar[] = [];
  let y = y0;
  let inFence = false;
  let buffer: string[] = [];
  const flushProse = () => {
    if (!buffer.length) return;
    const r = wrapWords(buffer.join("\n"), { x0: 0, y0: y, maxWidth: proseWidth(width, measure), ...FONTS.prose, tone: "ink" }, measure);
    bars.push(...r.bars);
    y = r.bottom;
    buffer = [];
  };
  for (const line of text.split("\n")) {
    if (/^\s*(```|~~~)/.test(line)) {
      flushProse();
      inFence = !inFence;
      y += FONTS.mono.lineHeight * 0.5;
      continue;
    }
    if (inFence) {
      const r = wrapWords(line, { x0: 10, y0: y, maxWidth: width - 20, ...FONTS.mono, tone: "muted", nowrap: true }, measure);
      bars.push(...r.bars);
      y = r.bottom;
    } else {
      buffer.push(line);
    }
  }
  flushProse();
  return { bars, bottom: y };
}

/**
 * Lays one transcript item out against ``width`` (the scroller's content
 * width). Each kind mirrors its view in ``Blocks.tsx`` /
 * ``ToolGroupView.tsx``: the user card sits right of the ``T<n>`` gutter
 * and is at most 76% wide; a tool row is one line (glyph, name,
 * arguments); reasoning is its one-line summary.
 */
export function layoutItem(item: TranscriptItem, width: number, measure: Measure): ItemLayout {
  switch (item.kind) {
    case "userMessage": {
      const maxInner = Math.max(40, (width - USER_GUTTER) * USER_MAX_SHARE - USER_PAD_X * 2);
      const longest = Math.max(...item.text.split("\n").map((l) => measure(l, FONTS.user.font)), 0);
      const inner = Math.min(maxInner, longest);
      const x0 = width - inner - USER_PAD_X;
      const r = wrapWords(item.text, { x0, y0: 18, maxWidth: inner, ...FONTS.user, tone: "ink" }, measure);
      return { bars: r.bars, height: r.bottom + 18, card: true };
    }
    case "assistantText": {
      const r = layoutProse(item.text, width, measure, 6);
      return { bars: r.bars, height: r.bottom + 6, card: false };
    }
    case "thinking": {
      const first = item.text.split("\n").find((l) => l.trim()) ?? "Thinking";
      const r = wrapWords(first, { x0: 18, y0: 4, maxWidth: width - 18, ...FONTS.note, tone: "muted", nowrap: true }, measure);
      return { bars: r.bars, height: r.bottom + 4, card: false };
    }
    case "toolGroup": {
      const lh = FONTS.mono.lineHeight;
      const failed = item.mode === "row" && item.calls.some((c) => c.status === "error" || c.isErrorResult);
      const running = item.mode === "row" && item.calls.some((c) => c.status === "running");
      const glyphTone: MinimapTone = failed ? "error" : running || item.mode !== "row" ? "muted" : "success";
      const glyphW = measure("✓", FONTS.mono.font);
      const bars: WordBar[] = [{ x: 0, y: 6 + lh * 0.25, w: glyphW, h: lh * 0.5, tone: glyphTone }];
      if (item.mode === "row") {
        const call = item.calls[0]!;
        const name = wrapWords(call.toolName, { x0: 24, y0: 6, maxWidth: width - 24, ...FONTS.mono, tone: "ink", nowrap: true }, measure);
        bars.push(...name.bars);
        const nameEnd = name.bars.reduce((m, b) => Math.max(m, b.x + b.w), 24);
        const args = wrapWords(argsText(call.args), { x0: nameEnd + 12, y0: 6, maxWidth: Math.max(0, width - nameEnd - 12 - 60), ...FONTS.mono, tone: "muted", nowrap: true }, measure);
        bars.push(...args.bars);
      } else {
        const label = item.label ?? item.calls.map((c) => c.toolName).join(" ");
        bars.push(...wrapWords(label, { x0: 24, y0: 6, maxWidth: width - 84, ...FONTS.mono, tone: "muted", nowrap: true }, measure).bars);
      }
      return { bars, height: lh + 12, card: false };
    }
    case "banner":
    case "systemNote": {
      const pad = item.kind === "banner" ? 12 : 0;
      const r = wrapWords(item.text, { x0: pad, y0: 4, maxWidth: width - pad * 2, ...FONTS.small, tone: toneForStyle(item.style) }, measure);
      return { bars: r.bars, height: r.bottom + 4, card: false };
    }
  }
}

/** A cache key for an item's layout: it changes when what the item draws does. */
export function layoutSignature(item: TranscriptItem, width: number): string {
  switch (item.kind) {
    case "toolGroup":
      return `${item.id}|${width}|${item.mode}|${item.calls.map((c) => `${c.id}:${c.status}:${c.isErrorResult ? 1 : 0}`).join(",")}|${item.label ?? ""}`;
    case "userMessage":
    case "assistantText":
    case "thinking":
    case "banner":
    case "systemNote":
      return `${item.id}|${width}|${item.text.length}`;
  }
}

/** Where an item sits in content space: the virtualizer's start/size, plus the list's offset in the scroller. */
export interface ItemSlot {
  start: number;
  size: number;
}

/** A bar or card in CONTENT coordinates (the scroller's scroll space). */
export interface PlacedBar extends WordBar {}
export interface PlacedCard {
  y: number;
  h: number;
}

/**
 * Places each item's layout at its slot. A layout taller than its slot
 * (a wrap that came out longer than the real rendering) is squeezed to
 * fit, so no item's bars spill into its neighbour's. A user message's
 * card spans its slot.
 */
export function placeItems(
  layouts: readonly ItemLayout[],
  slots: readonly ItemSlot[],
  listTop: number,
): { bars: PlacedBar[]; cards: PlacedCard[] } {
  const bars: PlacedBar[] = [];
  const cards: PlacedCard[] = [];
  layouts.forEach((layout, i) => {
    const slot = slots[i];
    if (!slot) return;
    const top = listTop + slot.start;
    const scale = layout.height > slot.size && layout.height > 0 ? slot.size / layout.height : 1;
    if (layout.card) cards.push({ y: top, h: slot.size });
    for (const b of layout.bars) bars.push({ ...b, y: top + b.y * scale, h: b.h * scale });
  });
  return { bars, cards };
}

/** The rail's inset track (handoff: 8px top/bottom, 6px left/right). */
export const TRACK_INSET_Y = 8;
export const TRACK_INSET_X = 6;

/**
 * The scroll position for a pointer at fraction ``f`` of the track: the
 * tapped point ends up centred on screen (handoff: jump, not smooth).
 */
export function scrollTopFor(f: number, scrollHeight: number, clientHeight: number): number {
  const clamped = Math.min(1, Math.max(0, f));
  const target = clamped * scrollHeight - clientHeight / 2;
  return Math.max(0, Math.min(scrollHeight - clientHeight, target));
}

/**
 * The turn under a content offset: the LAST user message whose top is at
 * or above ``contentY``. ``turns`` is in order, each with its content
 * top. Returns ``null`` above the first user message.
 */
export function turnAt<T extends { top: number }>(turns: readonly T[], contentY: number): T | null {
  let found: T | null = null;
  for (const t of turns) {
    if (t.top <= contentY) found = t;
    else break;
  }
  return found;
}

/** The scrub label: the two-digit turn, then the first line of that turn's message. */
export function scrubLabel(turn: number, text: string): { number: string; text: string } {
  const first = text.split("\n").find((l) => l.trim()) ?? "";
  return { number: String(turn).padStart(2, "0"), text: first.trim() };
}
