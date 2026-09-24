/**
 * Files attached from the browser are STAGED into the session's workspace
 * on the daemon (``StageFilesRequest``, ``docs/sdk-file-staging.md``) —
 * they land on disk where the agent's tools can read them, not in the
 * model's context.  This module holds the pure rules of that flow; the
 * wire half lives in ``app/staging.ts``.
 *
 *  • A file's workspace path is its own name (or the path a dropped
 *    directory gave it) under the folder the user chose, and it may never
 *    leave the workspace: a name that climbs (``..``) or is absolute is
 *    refused HERE, with the same words the daemon would answer with, so
 *    no bytes are sent for it.
 *  • The daemon caps one file at 10 MB and one request at 50 MB
 *    (``DEFAULT_STAGE_PER_FILE_LIMIT`` / ``DEFAULT_STAGE_TOTAL_LIMIT`` in
 *    ``server/websocket.py``), and a file travels as ONE WebSocket
 *    message, so it is also capped by the daemon's message limit.  A
 *    daemon advertises all three in its handshake (``serverLimits`` on
 *    the SDK client); one that advertises nothing enforces 1 MiB per
 *    message.  The limits are applied before sending: an over-size
 *    message is not refused politely, the daemon closes the connection.
 *  • When the message is sent, the prompt gains one trailing line naming
 *    the files staged during its composition, so the model knows where
 *    they are; the TUI convention for pointing at a workspace file is
 *    ``@path``, and the footer spells the paths the way that sigil wants.
 */

export const STAGE_PER_FILE_LIMIT = 10 * 1024 * 1024;
export const STAGE_TOTAL_LIMIT = 50 * 1024 * 1024;

/** A folder typed by the user, as a workspace-relative prefix (``""`` = the root). */
export function normalizeFolder(folder: string): string {
  return folder
    .trim()
    .replace(/\\/g, "/")
    .split("/")
    .filter((seg) => seg && seg !== ".")
    .join("/");
}

/**
 * The workspace-relative path one file is staged at.  ``relativePath`` is
 * what a dropped or picked directory reports (``webkitRelativePath``);
 * empty for a plain file.  Returns ``null`` for a name the daemon would
 * refuse as unsafe, so the caller marks it failed without sending it.
 */
export function stagedName(fileName: string, relativePath: string, folder: string): string | null {
  const own = (relativePath || fileName).replace(/\\/g, "/");
  const parts = own.split("/").filter((seg) => seg && seg !== ".");
  const dir = normalizeFolder(folder);
  if (!parts.length) return null;
  const all = [...(dir ? dir.split("/") : []), ...parts];
  if (all.some((seg) => seg === "..")) return null;
  if (own.startsWith("/") || /^[a-zA-Z]:/.test(own)) return null;
  return all.join("/");
}

/** Human size for a chip: ``1.2 MB``, ``48 kB``, ``312 B``. */
export function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} kB`;
  return `${bytes} B`;
}

/** The caps a request is judged against; the SDK's ``ServerLimits`` fits. */
export interface SizeLimits {
  stagePerFileLimit: number;
  stageTotalLimit: number;
  /** ``false`` when the daemon advertised nothing and these are the legacy values. */
  advertised?: boolean;
}

export interface SizeVerdict {
  /** ``null`` when the file may be sent; otherwise the reason, in the daemon's words. */
  reason: string | null;
}

/**
 * Apply the daemon's caps before a request is built: a file over the
 * per-file cap is refused on its own; a batch over the total cap refuses
 * every file in it, since the daemon answers the whole request that way.
 */
export function checkSizes(sizes: number[], limits?: SizeLimits | null): SizeVerdict[] {
  const perFile = limits?.stagePerFileLimit ?? STAGE_PER_FILE_LIMIT;
  const cap = limits?.stageTotalLimit ?? STAGE_TOTAL_LIMIT;
  const total = sizes.reduce((a, b) => a + Math.max(b, 0), 0);
  if (total > cap) {
    const reason = `declared total ${total} bytes exceeds cap ${cap}`;
    return sizes.map(() => ({ reason }));
  }
  // A daemon that advertised no limits is an older one whose real cap is
  // its 1 MiB message size, not the staging cap it would quote: say so,
  // since the remedy (upgrade the daemon) is not what "exceeds cap" implies.
  const legacy = limits != null && limits.advertised === false;
  return sizes.map((size) => ({
    reason: size <= perFile
      ? null
      : legacy
        ? `${formatSize(size)} is over this daemon's ${formatSize(perFile)} message limit; upgrade jaato-server to attach larger files`
        : `declared size ${size} bytes exceeds per-file cap ${perFile}`,
  }));
}

/**
 * The line appended to a prompt for the files staged while it was being
 * written.  Empty when nothing was staged, so a prompt with no attachments
 * is sent byte-for-byte as typed.
 */
export function attachmentFooter(paths: string[]): string {
  if (!paths.length) return "";
  const refs = paths.map((p) => (/\s/.test(p) ? `"${p}"` : p));
  return `\n\nAttached files, staged in the workspace: ${refs.join(", ")}`;
}
