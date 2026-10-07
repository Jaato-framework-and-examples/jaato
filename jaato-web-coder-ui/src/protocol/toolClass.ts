/**
 * Coarse classes for tool calls, so the transcript can decide how much
 * weight to give a row (jaato/#1304 §1).  ``tool.call_start`` and
 * ``permission.requested`` now carry an OPTIONAL server-computed
 * ``tool_class`` (Phase 3, ``jaato_server.shared.tool_classification``), and
 * ``resolveToolClass`` below prefers it -- but this table stays as the
 * fallback for a daemon that predates the field, and as the one place
 * the mapping lives client-side so nothing needs a second opinion.
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

/** The closed vocabulary a server-computed ``tool_class`` may name -- the
 * same set ``jaato_server.shared.tool_classification.TOOL_CLASSES`` declares. */
const KNOWN_CLASSES: ReadonlySet<string> = new Set([
  "housekeeping", "write", "exec", "read", "agent", "other",
]);

/**
 * A tool's class, preferring the DAEMON's own answer over this table.
 *
 * ``serverClass`` is ``ToolCallStartEvent.tool_class`` /
 * ``PermissionRequestedEvent.tool_class`` (jaato/#1304 phase 3) -- ``null``
 * or ``undefined`` means "not reported" (an older daemon), in which case
 * this falls back to ``classifyTool(displayName)`` exactly as before the
 * field existed.  A value outside the closed vocabulary -- which should
 * never happen, but a client must not trust the wire blindly -- is
 * treated the same way: fall back rather than render an invented class.
 */
export function resolveToolClass(displayName: string, serverClass?: string | null): ToolClass {
  if (serverClass && KNOWN_CLASSES.has(serverClass)) return serverClass as ToolClass;
  return classifyTool(displayName);
}
