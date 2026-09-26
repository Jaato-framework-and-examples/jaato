/**
 * Best-effort previews for a tool row, from data the CLIENT already has
 * -- the call's own arguments and its accumulated output.
 *
 * ``updateFile`` / ``writeNewFile`` now carry a REAL server-computed
 * ``diff`` on ``tool.call_end`` (jaato/#1304 phase 3,
 * ``jaato_server.shared.plugins.file_edit.diff_utils``), and
 * ``ToolBlockView`` prefers it -- ``diffPreviewForCall`` below is read
 * only as the fallback for an older daemon, or for a write-shaped tool
 * this codebase has no server diff for at all (``findAndReplace``,
 * ``moveFile`` / ``renameFile``, ``restoreFile`` / ``undoFileChange``,
 * ``removeFile``).  For a TARGETED edit (``updateFile`` with
 * ``old``/``new``) the call's own arguments already ARE a before/after
 * pair, so a small diff-shaped preview is honest to build from them.  A
 * full rewrite (``writeNewFile``, or ``updateFile`` in its whole-file
 * ``new_content`` mode) has no "before" the client holds, so the
 * fallback preview is the new content's head, unmarked as a diff.
 *
 * No diff VIEWER exists anywhere in this client yet (the Files panel's
 * file names only download, ``docs/sdk-file-staging.md``), so the
 * fallback's "Open diff" stays wired to the download affordance;
 * ``ToolBlockView``'s server-diff path instead EXPANDS the diff inline,
 * since the daemon already sent the whole (capped) text.
 */

import { isMarkdownPath, resolveWorkspacePath } from "./workspacePaths";

const MAX_PREVIEW_LINES = 6;
const MAX_PREVIEW_CHARS = 480;

function truncateLines(text: string, maxLines = MAX_PREVIEW_LINES): { text: string; truncated: boolean } {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const truncated = lines.length > maxLines || text.length > MAX_PREVIEW_CHARS;
  let out = lines.slice(0, maxLines).join("\n");
  if (out.length > MAX_PREVIEW_CHARS) out = out.slice(0, MAX_PREVIEW_CHARS);
  return { text: out, truncated };
}

export interface ServerDiffPreview {
  /** Every line of the server's diff, split for ``DiffLines``. */
  fullLines: string[];
  /** The first ``MAX_PREVIEW_LINES`` of ``fullLines`` -- what shows before "Open diff". */
  previewLines: string[];
  /** Whether ``previewLines`` is shorter than ``fullLines`` -- an "Open diff" affordance is needed. */
  needsExpand: boolean;
}

/**
 * Splits a server-computed unified diff (``ToolCallEndEvent.diff``) into a
 * short preview and the full text, so a long diff shows a few lines
 * before asking for a click rather than filling the transcript.  This is
 * a CLIENT-side display cap on top of the server's own truncation
 * (``ToolCallEndEvent.diff_truncated`` / ``DEFAULT_MAX_LINES``) -- the
 * two are independent and both may fire on the same call.
 */
export function splitServerDiff(diff: string, maxPreviewLines = MAX_PREVIEW_LINES): ServerDiffPreview {
  const fullLines = diff.replace(/\r\n/g, "\n").split("\n");
  const needsExpand = fullLines.length > maxPreviewLines;
  return { fullLines, previewLines: fullLines.slice(0, maxPreviewLines), needsExpand };
}

export interface DiffPreview {
  /** Unified-diff-shaped lines (``+``/``-`` prefixed) when a before/after pair was available. */
  diffLines: string[] | null;
  /** A plain text preview when only the new content is known (a full write/rewrite). */
  text: string | null;
  truncated: boolean;
  /** The path this preview is about, when the call named one. */
  path: string | null;
}

const NO_PREVIEW: DiffPreview = { diffLines: null, text: null, truncated: false, path: null };

function asDiffLines(before: string, after: string): string[] {
  // A minimal, line-oriented diff -- not an LCS/Myers diff (no daemon
  // ``diff`` field exists to check one against yet, #1304 Phase 3), just
  // the shape a reviewer expects: what left, what arrived.
  const beforeLines = before.replace(/\r\n/g, "\n").split("\n");
  const afterLines = after.replace(/\r\n/g, "\n").split("\n");
  const out: string[] = [];
  for (const l of beforeLines) out.push(`-${l}`);
  for (const l of afterLines) out.push(`+${l}`);
  return out.slice(0, MAX_PREVIEW_LINES * 2);
}

/** A diff/content preview for a ``write``-class tool call, from its own arguments. */
export function diffPreviewForCall(toolName: string, args: Record<string, unknown>): DiffPreview {
  const path = typeof args.path === "string" ? args.path : null;
  const str = (k: string): string | null => (typeof args[k] === "string" ? (args[k] as string) : null);

  if (toolName === "updateFile") {
    const oldText = str("old");
    const newText = str("new");
    if (oldText != null && newText != null) {
      const diffLines = asDiffLines(oldText, newText);
      return { diffLines, text: null, truncated: diffLines.length >= MAX_PREVIEW_LINES * 2, path };
    }
    const whole = str("new_content");
    if (whole != null) {
      const { text, truncated } = truncateLines(whole);
      return { diffLines: null, text, truncated, path };
    }
    return { ...NO_PREVIEW, path };
  }

  if (toolName === "findAndReplace") {
    const pattern = str("pattern") ?? str("find");
    const replacement = str("replacement") ?? str("replace");
    if (pattern != null && replacement != null) {
      return { diffLines: asDiffLines(pattern, replacement), text: null, truncated: false, path };
    }
    return { ...NO_PREVIEW, path };
  }

  if (toolName === "writeNewFile") {
    const content = str("content");
    if (content != null) {
      const { text, truncated } = truncateLines(content);
      return { diffLines: null, text, truncated, path };
    }
    return { ...NO_PREVIEW, path };
  }

  if (toolName === "removeFile") return { diffLines: null, text: `removed ${path ?? "(unknown path)"}`, truncated: false, path };
  if (toolName === "moveFile" || toolName === "renameFile") {
    const to = str("new_path") ?? str("to") ?? str("destination");
    return { diffLines: null, text: `${path ?? "(unknown path)"} → ${to ?? "(unknown)"}`, truncated: false, path };
  }
  if (toolName === "restoreFile" || toolName === "undoFileChange") {
    return { diffLines: null, text: `restored ${path ?? "(unknown path)"}`, truncated: false, path };
  }

  return { ...NO_PREVIEW, path };
}

/** A trailing-lines preview of a running/finished exec tool's accumulated output. */
export function execOutputPreview(output: string): { text: string; truncated: boolean } {
  const lines = output.replace(/\r\n/g, "\n").split("\n").filter((l, i, arr) => !(i === arr.length - 1 && l === ""));
  const truncated = lines.length > MAX_PREVIEW_LINES;
  const tail = lines.slice(-MAX_PREVIEW_LINES).join("\n");
  return { text: tail, truncated };
}

/** The command/title line for an ``exec``-class call: what a person reads as "what ran". */
export function execTitle(toolName: string, args: Record<string, unknown>): string {
  const command = args.command;
  if (typeof command === "string" && command.trim()) return command;
  const text = args.text;
  if (typeof text === "string" && text.trim()) return text;
  return toolName;
}

/**
 * The markdown files a successful ``file_edit`` call left on disk, for the
 * tool row's ``view`` button (the Files panel's rendered markdown viewer).
 *
 * Read from the call's own arguments, plus ``serverPath`` -- the path the
 * daemon reported on ``tool.call_end`` -- when there is one:
 *
 * - ``writeNewFile`` / ``updateFile`` / ``restoreFile`` / ``undoFileChange``: ``path``.
 * - ``moveFile`` / ``renameFile``: ``destination_path`` (the source is gone).
 * - ``multiFileEdit``: every ``edit`` / ``create`` operation's ``path`` and
 *   every ``rename``'s ``to``; a ``delete`` leaves nothing to view.
 *
 * An absolute path counts when it lies under ``workspaceRoot`` (the
 * session's ``workspace_path``) and is made relative to it.
 *
 * ``removeFile`` yields nothing, and so does any other tool.  Order is
 * first-seen, normalised (``./a.md`` is ``a.md``), duplicates dropped; an
 * absolute path or one that climbs out of the workspace is not offered.
 */
export function markdownPathsForCall(toolName: string, args: Record<string, unknown>, serverPath?: string | null, workspaceRoot?: string | null): string[] {
  const out: string[] = [];
  const root = workspaceRoot ? workspaceRoot.replace(/\/+$/, "") + "/" : null;
  const add = (p: unknown) => {
    if (typeof p !== "string" || !isMarkdownPath(p)) return;
    let rel = p;
    if (rel.startsWith("/")) {
      // An absolute path is offered only when it lies under the session's
      // own workspace; anything else is not a workspace-relative file.
      if (!root || !rel.startsWith(root)) return;
      rel = rel.slice(root.length);
    }
    const norm = resolveWorkspacePath(rel, ""); // ``./a.md`` and ``a.md`` are one file
    if (norm && !out.includes(norm)) out.push(norm);
  };
  switch (toolName) {
    case "writeNewFile":
    case "updateFile":
    case "restoreFile":
    case "undoFileChange":
      add(serverPath);
      add(args.path);
      break;
    case "moveFile":
    case "renameFile":
      add(args.destination_path);
      break;
    case "multiFileEdit":
      if (Array.isArray(args.operations)) {
        for (const op of args.operations as Record<string, unknown>[]) {
          if (!op || typeof op !== "object") continue;
          if (op.action === "rename") add(op.to);
          else if (op.action === "edit" || op.action === "create") add(op.path);
        }
      }
      break;
    default:
      break;
  }
  return out;
}
