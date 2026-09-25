/**
 * Coarse classes for tool calls, so the transcript can decide how much
 * weight to give a row (jaato/#1304 §1).  Client-side table for now --
 * the issue's own server-changes list still asks for an optional
 * ``tool_class`` on ``tool.call_start`` / ``permission.requested``, which
 * is not built (Phase 3).  Until then this is the one place the mapping
 * lives, so the transcript view model and anything else that wants it
 * (a future permission-card regroup) read the same table rather than
 * growing a second opinion.
 *
 * Classification is by the tool's DISPLAY name -- the caller resolves a
 * hashed ``t_xxxxxxxx`` id (``protocol/toolIds.ts``) before asking here,
 * because the id carries no information about what the tool does.
 */

export type ToolClass = "housekeeping" | "write" | "exec" | "read" | "agent" | "other";

/**
 * The issue's own housekeeping list, verbatim: plan bookkeeping plus the
 * discovery/subscription calls a model makes before it does anything a
 * person cares about.
 */
const HOUSEKEEPING_TOOLS = new Set([
  "createPlan",
  "startPlan",
  "setStepStatus",
  "completePlan",
  "list_tools",
  "get_tool_schemas",
  "listReferences",
  "list_subagent_profiles",
  "subscribeToEvents",
]);

/** ``file_edit`` -- anything that changes bytes on disk. */
const WRITE_TOOLS = new Set([
  "writeNewFile",
  "updateFile",
  "removeFile",
  "moveFile",
  "renameFile",
  "multiFileEdit",
  "findAndReplace",
  "restoreFile",
  "undoFileChange",
]);

/** A subprocess, a shell session, or a notebook cell -- code that RUNS. */
const EXEC_TOOLS = new Set([
  "cli_based_tool",
  "shell_spawn",
  "shell_input",
  "shell_read",
  "shell_control",
  "shell_close",
  "notebook_execute",
]);

/** Looks without changing anything. */
const READ_TOOLS = new Set([
  "readFile",
  "glob_files",
  "grep_content",
  "listBackups",
  "shell_list",
  "retrieve_memories",
  "selectReferences",
]);

/** Spawns, messages or lists other sessions/subagents. */
const AGENT_TOOLS = new Set([
  "spawn_subagent",
  "close_subagent",
  "cancel_subagent",
  "send_to_subagent",
  "list_subagents",
  "send_to_sibling",
  "list_siblings",
  "send_to_session",
  "list_group_sessions",
]);

const TABLE: Record<string, ToolClass> = Object.fromEntries([
  ...[...HOUSEKEEPING_TOOLS].map((n) => [n, "housekeeping"] as const),
  ...[...WRITE_TOOLS].map((n) => [n, "write"] as const),
  ...[...EXEC_TOOLS].map((n) => [n, "exec"] as const),
  ...[...READ_TOOLS].map((n) => [n, "read"] as const),
  ...[...AGENT_TOOLS].map((n) => [n, "agent"] as const),
]);

/**
 * A tool's class by its DISPLAY name.  An unrecognised name -- an MCP
 * tool (``mcp__server__tool``), a plugin this table has no opinion about
 * (``store_memory``, ``web_search``, ``call_service``, ...) -- is
 * ``"other"`` rather than a guess: the table is a source of classes,
 * never of invented ones.
 */
export function classifyTool(displayName: string): ToolClass {
  return TABLE[displayName] ?? "other";
}
