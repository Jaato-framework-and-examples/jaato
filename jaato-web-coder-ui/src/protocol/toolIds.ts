/**
 * Hashed tool and category ids, shown by the names a person knows.
 *
 * Tool names reach the MODEL as hash-derived ids (``t_a3f2b1c0`` for a tool,
 * ``c_7e4d9f12`` for a category; ``shared/tool_id_map.py``) because
 * upstreams restrict tool-name characters.  So the arguments of a
 * discovery call arrive as ids -- ``list_tools(category_id=c_bbc5e661)`` --
 * which mean nothing to a reader.  The daemon sends the mapping so clients
 * need not reverse-engineer it: ``tools.id_registry`` (``ToolIdRegistryEvent``,
 * the full current set each time, never a delta) and the same map as
 * ``session.info``'s ``tool_id_mappings``.
 *
 * The agreement is that anything user-facing shows the human name, as the
 * TUI does (``jaato-tui/ui_utils.py`` ``resolve_tool_ids``).  This is that
 * function: it walks strings, arrays and plain objects and replaces a value
 * that IS an id; anything else passes through unchanged.  An id the map does
 * not name stays as it is -- showing the id is honest, inventing a name is
 * not.
 *
 * Resolution happens at RENDER time, not when an event is reduced: the
 * mapping can arrive after the call that used an id (it is sent after tool
 * configuration and again when deferred tools activate), and a row rendered
 * from the stored arguments picks up a mapping whenever it arrives.
 */
export function resolveToolIds(value: unknown, names: Readonly<Record<string, string>>): unknown {
  if (typeof value === "string") return Object.prototype.hasOwnProperty.call(names, value) ? names[value] : value;
  if (Array.isArray(value)) return value.map((v) => resolveToolIds(v, names));
  if (value && typeof value === "object" && Object.getPrototypeOf(value) === Object.prototype) {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>).map(([k, v]) => [k, resolveToolIds(v, names)]));
  }
  return value;
}

/** ``resolveToolIds`` over a tool call's argument map. */
export function resolveToolArgs(args: Record<string, unknown>, names: Readonly<Record<string, string>>): Record<string, unknown> {
  return resolveToolIds(args, names) as Record<string, unknown>;
}

/** Read a mapping off an event, keeping only string -> string entries. */
export function toolIdMappings(raw: unknown): Record<string, string> | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(raw as Record<string, unknown>)) if (typeof v === "string") out[k] = v;
  return out;
}
