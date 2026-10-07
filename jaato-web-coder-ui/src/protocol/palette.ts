/**
 * Grouping and filtering for the command palette (#1304 §6), built from
 * ``CommandListEvent`` -- the same ``CommandSpec[]`` the composer's
 * proposal popup already completes from.  There is no second source of
 * command names: the palette is a different PRESENTATION of
 * ``state.commands``, not a rival list somebody has to keep in sync with
 * it.
 *
 * The command list's own shape is the grouping.  A daemon-advertised name
 * is either bare (``help``, ``reset``) or ``"<area> <verb>"``
 * (``"tools list"``, ``"permissions status"``) -- the wire's own way of
 * saying which area a command belongs to, so grouping by a name's first
 * word needs no invented taxonomy and cannot drift from what the daemon
 * actually sent.
 */
import type { CommandSpec } from "./commands";

export interface PaletteGroup {
  area: string;
  items: CommandSpec[];
}

function areaOf(name: string): string {
  return name.trim().split(/\s+/)[0] || name;
}

/**
 * Group commands by area, areas sorted alphabetically.  Within an area,
 * the bare command (``name === area``, e.g. ``"tools"`` beside
 * ``"tools list"``) sorts first -- the umbrella before its verbs -- and
 * the rest sort by name.
 */
export function groupCommands(commands: CommandSpec[]): PaletteGroup[] {
  const byArea = new Map<string, CommandSpec[]>();
  for (const c of commands) {
    if (!c.name) continue;
    const area = areaOf(c.name);
    const list = byArea.get(area);
    if (list) list.push(c); else byArea.set(area, [c]);
  }
  return [...byArea.keys()].sort().map((area) => ({
    area,
    items: (byArea.get(area) ?? []).slice().sort((a, b) => {
      const aBare = a.name === area, bBare = b.name === area;
      if (aBare !== bBare) return aBare ? -1 : 1;
      return a.name.localeCompare(b.name);
    }),
  }));
}

/**
 * Groups whose name or description matches ``query`` (a plain, case-
 * insensitive substring test -- the same rule the composer's own
 * completions use).  An empty query returns every group unchanged, and a
 * group left with no matches is dropped rather than shown empty.
 */
export function filterGroups(groups: PaletteGroup[], query: string): PaletteGroup[] {
  const q = query.trim().toLowerCase();
  if (!q) return groups;
  const out: PaletteGroup[] = [];
  for (const g of groups) {
    const items = g.items.filter((c) => c.name.toLowerCase().includes(q) || (c.description ?? "").toLowerCase().includes(q));
    if (items.length) out.push({ area: g.area, items });
  }
  return out;
}

/** The groups flattened in render order -- what arrow-key navigation counts over, so the index the keyboard moves matches the index the mouse can click. */
export function flattenGroups(groups: PaletteGroup[]): CommandSpec[] {
  return groups.flatMap((g) => g.items);
}
