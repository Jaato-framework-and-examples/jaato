/**
 * The session rail's width: bounds, and the per-browser memory of it.
 *
 * A per-viewer convenience, so it lives in ``localStorage`` like the theme
 * and the last daemon URL, and every read and write is wrapped: storage can
 * be absent or throw (a private window, cleared site data), and the rail
 * must still render at its default.
 */
export const RAIL_WIDTH_DEFAULT = 400;
export const RAIL_WIDTH_MIN = 220;
export const RAIL_WIDTH_MAX = 720;
const STORAGE_KEY = "jaato.railWidth";

export function clampRailWidth(w: number): number {
  if (!Number.isFinite(w)) return RAIL_WIDTH_DEFAULT;
  return Math.min(RAIL_WIDTH_MAX, Math.max(RAIL_WIDTH_MIN, Math.round(w)));
}

export function loadRailWidth(): number {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v == null ? RAIL_WIDTH_DEFAULT : clampRailWidth(Number(v));
  } catch {
    return RAIL_WIDTH_DEFAULT;
  }
}

export function saveRailWidth(w: number): void {
  try { localStorage.setItem(STORAGE_KEY, String(w)); } catch { /* storage unavailable */ }
}
