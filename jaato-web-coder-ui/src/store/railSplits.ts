/**
 * How the rail's open sections (Plan, Budget, Files, Sessions, Memories) share its
 * height, and the per-browser memory of it.
 *
 * A section's size is a **weight**, not a pixel count: the open sections'
 * weights are normalised to fractions of the rail's flexible height at render
 * time (``sharesFor``), so they survive a window resize and a change in how
 * many sections are open.  The weight is remembered per section id, so
 * reopening a section restores the size it was left at, and adding a section
 * later cannot scramble another's saved size.  A per-viewer convenience, so it
 * lives in ``localStorage`` beside ``ui.railWidth`` and every read and write is
 * wrapped: storage can be absent or throw, and the rail must still render.
 *
 * The arithmetic is deliberately pure and framework-free so the split
 * behaviour can be unit-tested without a DOM: ``moveBoundary`` changes exactly
 * two neighbours' weights and preserves their sum (split-pane behaviour — the
 * others don't move), ``resetPair`` levels a pair, and ``sharesFor`` is the one
 * place open/close redistribution happens.
 */
export const RAIL_SECTION_IDS = ["plan", "budget", "files", "sessions", "memories"] as const;
export type RailSectionId = (typeof RAIL_SECTION_IDS)[number];

/** Weight per section id.  Absent = the default weight; a section not open is
 *  simply left out of ``sharesFor``'s open set (its remembered weight stays). */
export type RailSplits = Partial<Record<RailSectionId, number>>;

/** Minimum height of a section's content region, so a drag can't collapse a
 *  section to nothing — collapsing stays the header's job.  Roughly a header's
 *  worth of a few rows. */
export const RAIL_SECTION_MIN_PX = 72;

/** The share a section takes when it has no remembered weight — equal footing
 *  with its already-open neighbours. */
export const RAIL_DEFAULT_WEIGHT = 1;

const STORAGE_KEY = "jaato.railSplits";
const KNOWN = new Set<string>(RAIL_SECTION_IDS);

function weightOf(splits: RailSplits, id: RailSectionId): number {
  const w = splits[id];
  return typeof w === "number" && Number.isFinite(w) && w > 0 ? w : RAIL_DEFAULT_WEIGHT;
}

/**
 * Normalised fractions (summing to 1) over the OPEN sections only.
 *
 * This is where opening and closing a section redistributes its share: an id
 * added to ``openIds`` shrinks the others proportionally, an id removed hands
 * its share back to them proportionally, and neither touches a section's
 * remembered weight.  An unknown or removed id in ``splits`` never appears in
 * ``openIds``, so it is ignored.
 */
export function sharesFor(splits: RailSplits, openIds: RailSectionId[]): Record<string, number> {
  const out: Record<string, number> = {};
  if (openIds.length === 0) return out;
  let total = 0;
  for (const id of openIds) total += weightOf(splits, id);
  if (!(total > 0)) {
    for (const id of openIds) out[id] = 1 / openIds.length;
    return out;
  }
  for (const id of openIds) out[id] = weightOf(splits, id) / total;
  return out;
}

/**
 * Move the boundary between two adjacent open sections by ``deltaPx`` pixels.
 *
 * Only the two neighbours' weights change and their combined weight is
 * preserved, so every other open section keeps its height.  The move is
 * clamped so neither neighbour falls below ``minPx``; a pair with no room to
 * move (or a degenerate measurement) is returned unchanged.  ``pxAbove`` /
 * ``pxBelow`` are the neighbours' measured heights at the drag's reference
 * point, so the pixel delta is translated into weight against real geometry.
 */
export function moveBoundary(
  splits: RailSplits,
  aboveId: RailSectionId,
  belowId: RailSectionId,
  pxAbove: number,
  pxBelow: number,
  deltaPx: number,
  minPx = RAIL_SECTION_MIN_PX,
): RailSplits {
  const combinedPx = pxAbove + pxBelow;
  if (!(combinedPx > 0) || combinedPx < minPx * 2) return splits;
  const wAbove = weightOf(splits, aboveId);
  const wBelow = weightOf(splits, belowId);
  const combinedW = wAbove + wBelow;
  const newPxAbove = Math.min(combinedPx - minPx, Math.max(minPx, pxAbove + deltaPx));
  const nextAbove = (combinedW * newPxAbove) / combinedPx;
  return { ...splits, [aboveId]: nextAbove, [belowId]: combinedW - nextAbove };
}

/** Reset a pair to equal shares (double-click), leaving every other section
 *  untouched. */
export function resetPair(splits: RailSplits, aboveId: RailSectionId, belowId: RailSectionId): RailSplits {
  const half = (weightOf(splits, aboveId) + weightOf(splits, belowId)) / 2;
  return { ...splits, [aboveId]: half, [belowId]: half };
}

/** Percent of a pair's combined weight the ABOVE section holds — the boundary's
 *  ``aria-valuenow``. */
export function boundaryPercent(splits: RailSplits, aboveId: RailSectionId, belowId: RailSectionId): number {
  const wAbove = weightOf(splits, aboveId);
  const combined = wAbove + weightOf(splits, belowId);
  return combined > 0 ? Math.round((wAbove / combined) * 100) : 50;
}

/** Keep only known section ids carrying a finite positive weight — a saved map
 *  with an unknown or removed section id is not scrambled, that id is dropped. */
export function sanitizeSplits(raw: unknown): RailSplits {
  const out: RailSplits = {};
  if (!raw || typeof raw !== "object") return out;
  for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
    if (KNOWN.has(k) && typeof v === "number" && Number.isFinite(v) && v > 0) {
      out[k as RailSectionId] = v;
    }
  }
  return out;
}

export function loadRailSplits(): RailSplits {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v == null ? {} : sanitizeSplits(JSON.parse(v));
  } catch {
    return {};
  }
}

export function saveRailSplits(splits: RailSplits): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(splits));
  } catch {
    /* storage unavailable */
  }
}
