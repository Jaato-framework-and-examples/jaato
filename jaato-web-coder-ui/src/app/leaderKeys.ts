/**
 * The Ctrl/⌘+K leader's quick actions (#1304 §7).
 *
 * Ctrl+P / Ctrl+B / Ctrl+T / Ctrl+A / Ctrl+O used to be bound directly, and
 * every one of them is a browser shortcut first: Ctrl+P prints, Ctrl+B
 * toggles the bookmarks bar, Ctrl+T opens a new tab (a page cannot
 * intercept that one -- Chrome just ignores ``preventDefault`` on it, so
 * the old binding was already inert there and only ever fired in tests),
 * Ctrl+A selects all, Ctrl+O opens a file. Moving them behind a leader
 * frees the browser's own bindings; this module is what "K then <letter>"
 * resolves to, once ``CommandPalette`` (which the leader also opens) sees
 * the next bare keystroke land on an empty search box.
 *
 * The mapping, and why each key means what it does:
 *
 *   P / B / F   -> the Plan / Budget / Files rail sections (``Ctrl+P`` /
 *                  ``Ctrl+B`` opened Plan/Budget directly; ``F`` is new --
 *                  Files answered to ``Alt+W`` before, which stays bound
 *                  (it never conflicted with anything on this list), and
 *                  now has a leader form too, for one vocabulary across
 *                  all three rail toggles
 *   T           -> tool-call boxes expand/collapse (``Ctrl+T`` did this)
 *   [ / ]       -> the previous / next agent tab -- what ``Ctrl+A``'s
 *                  forward cycle did, split into two directions now that
 *                  a chord can spare two keys for it
 *   1-9         -> jump straight to agent tab N (main is always 1)
 *   O           -> step the live tool-output popup to the next running
 *                  tool (``Ctrl+O`` did this, from a listener local to
 *                  ``ToolOutputPopup`` that is now gone -- there is one
 *                  leader, not one leader plus a component keeping its
 *                  own direct binding alive)
 *
 * Pure and DOM-free: ``useKeyboardShortcuts`` and ``CommandPalette`` both
 * call ``resolveLeaderKey`` and apply whatever ``LeaderEffect`` it
 * returns, so the mapping is unit-testable with no keyboard event and no
 * store at all.
 */

export type LeaderEffect =
  | { kind: "toggleUi"; key: "showPlan" | "showBudget" | "showWorkspace" }
  | { kind: "toggleTools" }
  | { kind: "selectAgent"; id: string }
  | { kind: "cyclePopup"; callId: string };

export interface LeaderContext {
  /** ``s.agentOrder`` -- tab 1 is ``agentOrder[0]`` (main), and so on. */
  agentOrder: string[];
  selectedAgentId: string;
  /**
   * Running tool call ids for the SELECTED agent, in the order the live-
   * output popup's tab strip renders them (``store.runningToolCallIds``,
   * the one definition the popup itself now reads too).
   */
  runningToolCallIds: string[];
  /** ``s.ui.popupCallId`` -- the call currently pinned in the popup, if any. */
  popupCallId: string | null | undefined;
}

const UI_TOGGLE_KEYS: Record<string, "showPlan" | "showBudget" | "showWorkspace"> = {
  p: "showPlan",
  b: "showBudget",
  f: "showWorkspace",
};

/** The tab ``[`` / ``]`` step to, wrapping -- or ``null`` with fewer than two tabs, where stepping means nothing. */
function stepAgent(order: string[], selected: string, dir: 1 | -1): string | null {
  if (order.length < 2) return null;
  const i = order.indexOf(selected);
  const from = i < 0 ? 0 : i;
  return order[(from + dir + order.length) % order.length] ?? null;
}

/** Which tab a digit key (1-9) names, or ``null`` past the end of the tab strip. */
export function agentForDigit(order: string[], digit: number): string | null {
  return order[digit - 1] ?? null;
}

/**
 * What a bare key means once the leader has fired, or ``null`` when this
 * key is none of the leader's own -- in which case a caller (the palette)
 * treats it as the first character of an ordinary search instead.
 */
export function resolveLeaderKey(key: string, ctx: LeaderContext): LeaderEffect | null {
  const k = key.toLowerCase();
  const uiKey = UI_TOGGLE_KEYS[k];
  if (uiKey) return { kind: "toggleUi", key: uiKey };
  if (k === "t") return { kind: "toggleTools" };
  if (k === "[" || k === "]") {
    const next = stepAgent(ctx.agentOrder, ctx.selectedAgentId, k === "]" ? 1 : -1);
    return next ? { kind: "selectAgent", id: next } : null;
  }
  if (/^[1-9]$/.test(k)) {
    const id = agentForDigit(ctx.agentOrder, Number(k));
    return id ? { kind: "selectAgent", id } : null;
  }
  if (k === "o") {
    if (!ctx.runningToolCallIds.length) return null;
    const i = ctx.popupCallId ? ctx.runningToolCallIds.indexOf(ctx.popupCallId) : -1;
    const next = ctx.runningToolCallIds[(i + 1) % ctx.runningToolCallIds.length];
    return next ? { kind: "cyclePopup", callId: next } : null;
  }
  return null;
}

/** The keys this module answers for, in the order the issue names them -- what the palette's footer hint lists. */
export const LEADER_KEY_HINT = "P · B · F · T · [ / ] · 1-9 · O";
