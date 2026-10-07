/**
 * "One name everywhere" (#1304 §4): the agent tab already shows a
 * subagent's daemon-reported ``agent_name`` (``FIB-REVIEWER``), while the
 * PARENT's own text routinely quotes the raw id it spawned
 * ("Subagent spawned (id: subagent_1)") -- a model has no reason to know
 * the display name a later ``AGENT_CREATED`` event assigns.  So a reader
 * sees two names for one agent, a tab over from each other.
 *
 * Mirrors ``protocol/toolIds.ts``'s pattern: resolved at RENDER time from
 * a map the store already keeps (``s.agents``), never written back into a
 * block's stored text -- the mapping can arrive (or change) after the text
 * that mentions the id was already stored, and a run rendered from that
 * text picks up the name whenever it lands.
 */
import type { Agent } from "@/store/types";

/** id -> its daemon-reported name, for every agent whose name differs from its id. */
export function agentNameMap(agents: Readonly<Record<string, Pick<Agent, "id" | "name">>>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [id, a] of Object.entries(agents)) {
    if (a.name && a.name !== id) out[id] = a.name;
  }
  return out;
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Replace a raw agent id mentioned in free text with its display name,
 * matched on a word boundary so a name is never rewritten inside a longer
 * word it merely happens to be a substring of.  Longest ids are matched
 * first, so one id that is a prefix of another (``sub`` / ``sub_2``)
 * cannot steal a shorter match out from under the longer one.
 */
export function resolveAgentIdsInText(text: string, names: Readonly<Record<string, string>>): string {
  const ids = Object.keys(names);
  if (ids.length === 0 || !text) return text;
  const sorted = [...ids].sort((a, b) => b.length - a.length);
  const re = new RegExp(`\\b(?:${sorted.map(escapeRegExp).join("|")})\\b`, "g");
  return text.replace(re, (m) => names[m] ?? m);
}
