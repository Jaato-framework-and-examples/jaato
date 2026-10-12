// Hashed tool/category id resolution (#1650): the rules the TUI and the web
// coder agreed on, now in the SDK.  Mirror of jaato_sdk/tests/test_tool_ids.py.

import { strict as assert } from "node:assert";
import { test } from "node:test";

import type { JaatoClient } from "./client.js";
import { ToolIdNames, resolveToolArgs, resolveToolIds, toolIdMappings } from "./toolIds.js";

const NAMES = { c_bbc5e661: "filesystem", t_c162b0f2: "readFile" };

test("resolves a string that IS an id, through arrays and objects", () => {
  assert.deepEqual(
    resolveToolArgs({ category_id: "c_bbc5e661", tool_ids: ["t_c162b0f2", "t_ffffffff"], n: 3 }, NAMES),
    { category_id: "filesystem", tool_ids: ["readFile", "t_ffffffff"], n: 3 },
  );
});

test("an unknown id and a string merely containing an id are left alone", () => {
  assert.equal(resolveToolIds("t_00000000", NAMES), "t_00000000");
  assert.equal(resolveToolIds("see t_c162b0f2", NAMES), "see t_c162b0f2");
  // Keys are argument names, never resolved; prototype names are not ids.
  assert.deepEqual(resolveToolIds({ t_c162b0f2: 1 }, NAMES), { t_c162b0f2: 1 });
  assert.equal(resolveToolIds("toString", NAMES), "toString");
});

test("toolIdMappings keeps string -> string entries only", () => {
  assert.deepEqual(toolIdMappings({ c_1: "system", bad: 3 }), { c_1: "system" });
  assert.equal(toolIdMappings(null), null);
  assert.equal(toolIdMappings([]), null);
});

test("tools.id_registry replaces the map wholesale", () => {
  const n = new ToolIdNames({ t_old: "old" });
  assert.equal(n.observe({ type: "tools.id_registry", mappings: NAMES }), true);
  assert.deepEqual(n.mappings, NAMES);
  assert.equal(n.name("t_old"), "t_old");
  assert.equal(n.observe({ type: "tools.id_registry", mappings: {} }), true);
  assert.equal(n.size, 0);
});

test("session.info seeds the map only when it carries mappings", () => {
  const n = new ToolIdNames();
  assert.equal(n.observe({ type: "session.info", tool_id_mappings: NAMES }), true);
  assert.equal(n.observe({ type: "session.info", tool_id_mappings: {} }), false);
  assert.equal(n.observe({ type: "session.info" }), false);
  assert.equal(n.observe({ type: "agent.output", mappings: { x: "y" } }), false);
  assert.equal(n.name("t_c162b0f2"), "readFile");
});

test("resolution happens at read time, so a late registry still applies", () => {
  const n = new ToolIdNames();
  const args = { tool_ids: ["t_c162b0f2"] };
  assert.deepEqual(n.resolve(args), args);
  n.observe({ type: "tools.id_registry", mappings: NAMES });
  assert.deepEqual(n.resolve(args), { tool_ids: ["readFile"] });
  assert.deepEqual(args, { tool_ids: ["t_c162b0f2"] }, "input is not mutated");
});

test("track subscribes to both event types and unsubscribes both", () => {
  const handlers = new Map<string, (ev: never) => void>();
  const client = {
    subscribe(type: "tools.id_registry" | "session.info", h: (ev: never) => void) {
      handlers.set(type, h);
      return () => { handlers.delete(type); };
    },
  };
  const n = new ToolIdNames();
  const off = n.track(client);
  assert.deepEqual([...handlers.keys()].sort(), ["session.info", "tools.id_registry"]);
  handlers.get("tools.id_registry")!({ type: "tools.id_registry", mappings: NAMES } as never);
  assert.equal(n.name("c_bbc5e661"), "filesystem");
  off();
  assert.equal(handlers.size, 0);
});

// Type-level: a real JaatoClient is accepted by track().  Never called.
export function _typecheckTrack(client: JaatoClient): () => void {
  return new ToolIdNames().track(client);
}
