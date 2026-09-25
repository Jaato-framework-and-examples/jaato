/**
 * Best-effort previews for a tool row, from data the CLIENT already has
 * -- the call's own arguments and its accumulated output.  Two gaps this
 * intentionally does not paper over:
 *
 * - jaato/#1304's own "Server changes needed" list still asks for a
 *   ``diff`` (unified, capped) field on ``tool.call_end`` for file
 *   writers, not shipped (Phase 3).  Until then there is no unified diff
 *   to show, only the arguments the model itself passed -- which, for a
 *   TARGETED edit (``updateFile`` with ``old``/``new``,
 *   ``findAndReplace``, one entry of ``multiFileEdit``), already ARE a
 *   before/after pair, so a small diff-shaped preview is honest to build
 *   from them.  A full rewrite (``writeNewFile``, or ``updateFile`` in
 *   its whole-file ``new_content`` mode) has no "before" the client
 *   holds, so the preview is the new content's head, unmarked as a diff.
 * - No diff VIEWER exists anywhere in this client yet (the Files panel's
 *   file names only download, ``docs/sdk-file-staging.md``). "Open diff"
 *   is therefore wired to the same download affordance until one is
 *   built -- see the PR description for this gap.
 */

const MAX_PREVIEW_LINES = 6;
const MAX_PREVIEW_CHARS = 480;

function truncateLines(text: string, maxLines = MAX_PREVIEW_LINES): { text: string; truncated: boolean } {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const truncated = lines.length > maxLines || text.length > MAX_PREVIEW_CHARS;
  let out = lines.slice(0, maxLines).join("\n");
  if (out.length > MAX_PREVIEW_CHARS) out = out.slice(0, MAX_PREVIEW_CHARS);
  return { text: out, truncated };
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
