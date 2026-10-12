// Hashed tool and category ids, shown by the names a person knows (#1650).
//
// Mirror of jaato-sdk/jaato_sdk/tool_ids.py.  Same rules, same names.
//
// Tool names reach the MODEL as hash-derived ids (`t_a3f2b1c0` for a tool,
// `c_7e4d9f12` for a category; jaato_server/shared/tool_id_map.py) because
// upstreams restrict tool-name characters, so the arguments of a discovery
// call arrive as ids -- `list_tools(category_id="c_bbc5e661")`.  The daemon
// sends the mapping as `tools.id_registry` (`ToolIdRegistryEvent`, the full
// current set each time, never a delta) and as `session.info`'s
// `tool_id_mappings`.
//
// Rules:
// - the mapping is replaced wholesale on each `tools.id_registry`, and
//   seeded from a non-empty `session.info.tool_id_mappings`;
// - resolution walks arrays and plain objects and replaces only a string
//   that IS an id in the map; an unknown id is left as it is (showing the id
//   is honest, inventing a name is not), and a string merely containing one
//   is untouched;
// - resolve at RENDER time, not when an event is reduced: the registry can
//   arrive after the call that used an id.

import { EventTypeValue } from "./events.js";
import type { Unsubscribe } from "./event-typing.js";

/** Id -> human name. */
export type ToolIdNameMap = Readonly<Record<string, string>>;

/** Read a mapping off an event field, keeping only string -> string entries. */
export function toolIdMappings(raw: unknown): Record<string, string> | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(raw as Record<string, unknown>)) if (typeof v === "string") out[k] = v;
  return out;
}

/** Replace every string in `value` that IS an id in `names`; returns a new structure. */
export function resolveToolIds(value: unknown, names: ToolIdNameMap): unknown {
  if (typeof value === "string") return Object.prototype.hasOwnProperty.call(names, value) ? names[value] : value;
  if (Array.isArray(value)) return value.map((v) => resolveToolIds(v, names));
  if (value && typeof value === "object" && Object.getPrototypeOf(value) === Object.prototype) {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([k, v]) => [k, resolveToolIds(v, names)]),
    );
  }
  return value;
}

/** `resolveToolIds` over a tool call's argument map. */
export function resolveToolArgs(args: Record<string, unknown>, names: ToolIdNameMap): Record<string, unknown> {
  return resolveToolIds(args, names) as Record<string, unknown>;
}

/** The subset of JaatoClient that `ToolIdNames.track` needs. */
export interface ToolIdEventSource {
  subscribe(eventType: "tools.id_registry" | "session.info", handler: (ev: never) => void): Unsubscribe;
}

/**
 * The latest id -> name mapping, kept current from the event stream.
 *
 * `observe(ev)` with a `tools.id_registry` event REPLACES the map; with a
 * `session.info` whose `tool_id_mappings` is non-empty, replaces it too;
 * anything else is ignored.  `track(client)` subscribes for you.  Read with
 * `resolve(value)` / `name(id)` when rendering.
 */
export class ToolIdNames {
  #names: Record<string, string>;

  constructor(mappings?: Readonly<Record<string, string>> | null) {
    this.#names = toolIdMappings(mappings) ?? {};
  }

  /** A copy of the current map. */
  get mappings(): Record<string, string> {
    return { ...this.#names };
  }

  get size(): number {
    return Object.keys(this.#names).length;
  }

  /** Replace the map wholesale. */
  replace(mappings: Readonly<Record<string, string>>): void {
    this.#names = toolIdMappings(mappings) ?? {};
  }

  /** Update from one wire event; returns whether the map was replaced. */
  observe(ev: unknown): boolean {
    if (!ev || typeof ev !== "object") return false;
    const e = ev as Record<string, unknown>;
    if (e.type === EventTypeValue.TOOL_ID_REGISTRY) {
      const names = toolIdMappings(e.mappings);
      if (!names) return false;
      this.#names = names;
      return true;
    }
    if (e.type === EventTypeValue.SESSION_INFO) {
      const names = toolIdMappings(e.tool_id_mappings);
      if (!names || !Object.keys(names).length) return false;
      this.#names = names;
      return true;
    }
    return false;
  }

  /** The name for `value` when it is a known id, else `value`. */
  name(value: string): string {
    return Object.prototype.hasOwnProperty.call(this.#names, value) ? this.#names[value] : value;
  }

  /** `resolveToolIds` over the current map. */
  resolve(value: unknown): unknown {
    return resolveToolIds(value, this.#names);
  }

  /** Keep the map current from a client; one unsubscribe for both event types. */
  track(client: ToolIdEventSource): Unsubscribe {
    const handler = (ev: never) => { this.observe(ev); };
    const unsubs = [
      client.subscribe(EventTypeValue.TOOL_ID_REGISTRY, handler),
      client.subscribe(EventTypeValue.SESSION_INFO, handler),
    ];
    return () => { for (const u of unsubs) u(); };
  }
}
