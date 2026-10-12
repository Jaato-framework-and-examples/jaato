/**
 * Hashed tool and category ids, shown by the names a person knows.
 *
 * The rules (replace the map wholesale on ``tools.id_registry``, seed from a
 * non-empty ``session.info.tool_id_mappings``, replace only a string that IS
 * a known id, resolve at render time) live in the SDK since #1650
 * (``jaato-sdk-ts/src/toolIds.ts``, mirror of ``jaato_sdk/tool_ids.py``), so
 * this client and the TUI share one copy.  The store keeps the map as plain
 * state (``toolIdNames``) and components resolve against it when rendering.
 */
export { resolveToolArgs, resolveToolIds, toolIdMappings } from "@jaato/sdk";
