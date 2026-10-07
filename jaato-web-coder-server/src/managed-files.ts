/**
 * Application-managed workspace files: the generic half of shipping working
 * guidance into a bound workspace (docs/design/github-workspace-guidance.md
 * §5, §7 Q5).
 *
 * The application already writes content into a workspace two ways —
 * ``.env`` and ``.home/.gitconfig`` at bind time (``src/github.ts``).  A
 * *managed file* is a third: a file the application OWNS in the workspace, so
 * it may refresh it when its shipped content changes, but must never clobber a
 * copy the user made their own.  A machine-readable marker on the first line
 * carries the ownership and the version:
 *
 * ```
 * <!-- jaato-managed: github-guidance v1 — delete this line to keep your own edits -->
 * ```
 *
 * The write / refresh rules (§5), all decided here so a marker author cannot
 * get them wrong:
 *
 * | On disk | ``writeManagedFile`` |
 * |---|---|
 * | absent | write |
 * | our marker, SAME version | no-op |
 * | our marker, DIFFERENT version | overwrite |
 * | present WITHOUT our marker | **skip** — the user replaced it; record it, never silent |
 *
 * On bind-to-none the file is removed the way ``GH_TOKEN`` is removed from
 * ``.env`` — but only when it still carries our marker: a copy the user made
 * their own (marker deleted) is theirs and is left in place.
 *
 * This module is deliberately GENERIC — it knows nothing about GitHub.  The
 * content, the marker id and the version are a {@link ManagedFile} the caller
 * supplies (``src/github-guidance.ts`` builds the one GitHub instance), so a
 * future forge (GitLab) registers its own file rather than re-architecting
 * this.  Containment (``_resolveWorkspace``, symlinks resolved) and the
 * temp-file-plus-rename ``atomicWrite`` stay in the caller: this module is
 * handed an already-contained absolute workspace directory and the ``write``
 * primitive, so it has no filesystem policy of its own beyond ``mkdir`` +
 * ``read`` + ``rm``.
 */
import { existsSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";

/**
 * How the ownership marker is spelled, per file type:
 *
 * | Format | Marker | For |
 * |---|---|---|
 * | ``html`` (default) | ``<!-- jaato-managed: <id> vN — … -->`` on the first line | Markdown instruction files |
 * | ``hash`` | ``# jaato-managed: <id> vN — …`` on the first line | TOML, shell, ``.gitconfig``-style files |
 * | ``json`` | a top-level ``"_jaato_managed": "<id> vN"`` key | ``.lsp.json``, the environment manifest |
 *
 * JSON has no comments, so its marker is a key the reader of the file
 * ignores (the ``lsp`` plugin reads only ``languageServers``).  A JSON body
 * that does not parse, or parses to something other than an object, carries
 * no marker: it is the user's file.
 */
export type ManagedFormat = "html" | "hash" | "json";

/** The key a JSON managed file carries its marker under. */
export const JSON_MARKER_KEY = "_jaato_managed";

/** A file the application owns in a workspace: a relative path, a marker id, a version and the body (marker excluded). */
export interface ManagedFile {
  /** Workspace-relative path, e.g. ``.jaato/instructions/40-github.md``.  Never absolute, never ``..``-bearing. */
  relativePath: string;
  /** The token in the ``jaato-managed:`` marker; identifies which application owns the file (``github-guidance``). */
  markerId: string;
  /** A monotonically increasing integer; bumped when the shipped {@link ManagedFile.body} changes. */
  version: number;
  /**
   * The file's content WITHOUT the marker; {@link managedContent} adds it.
   * For ``json``, a JSON OBJECT text; the marker key is inserted first.
   */
  body: string;
  /** How the marker is spelled; default ``html``. */
  format?: ManagedFormat;
  /**
   * The body is DATA the application derives (a server table, a manifest),
   * not shipped prose: a marked copy is refreshed whenever its content
   * differs, not only when the version changes.  A shipped file (``false``,
   * the default) keeps the §5 rule: same version is a no-op, so an edit made
   * with the marker left in place survives until the next version bump.
   */
  generated?: boolean;
}

/** What {@link writeManagedFile} did, so a caller can surface a skip / overwrite and never claim a silent write. */
export type ManagedWriteOutcome =
  | { path: string; action: "written"; reason: "absent" | "version-changed" | "content-changed"; previousVersion?: number }
  | { path: string; action: "unchanged" }
  | { path: string; action: "skipped-user-file" }
  | { path: string; action: "error"; message: string };

/** What {@link removeManagedFile} did on bind-to-none. */
export type ManagedRemoveOutcome =
  | { path: string; action: "removed" }
  | { path: string; action: "absent" }
  | { path: string; action: "skipped-user-file" }
  | { path: string; action: "error"; message: string };

/** The write primitive the caller supplies: an atomic (temp-file + rename) write of ``body`` to an absolute ``path`` at ``mode``. */
export type AtomicWrite = (path: string, body: string, mode: number) => void;

/** The exact marker line for ``markerId`` at ``version`` — the first line of a managed file. */
export function managedMarker(markerId: string, version: number, format: ManagedFormat = "html"): string {
  if (format === "hash") return `# jaato-managed: ${markerId} v${version} — delete this line to keep your own edits`;
  if (format === "json") return `${markerId} v${version}`;
  return `<!-- jaato-managed: ${markerId} v${version} — delete this line to keep your own edits -->`;
}

/** The full on-disk content: the marker, then the body (a trailing newline is ensured). */
export function managedContent(file: ManagedFile): string {
  const format = file.format ?? "html";
  if (format === "json") {
    const parsed = JSON.parse(file.body) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error(`managed file ${file.relativePath}: a json body must be an object`);
    }
    const rest = { ...(parsed as Record<string, unknown>) };
    delete rest[JSON_MARKER_KEY];
    return `${JSON.stringify({ [JSON_MARKER_KEY]: managedMarker(file.markerId, file.version, "json"), ...rest }, null, 2)}\n`;
  }
  const body = file.body.endsWith("\n") ? file.body : `${file.body}\n`;
  return `${managedMarker(file.markerId, file.version, format)}\n${body}`;
}

// The marker on the FIRST line: ``jaato-managed: <id> v<n>`` followed by anything.
const MARKER_RE = /^<!--\s*jaato-managed:\s*(\S+)\s+v(\d+)\b.*-->\s*$/;
const HASH_MARKER_RE = /^#\s*jaato-managed:\s*(\S+)\s+v(\d+)\b/;
const JSON_MARKER_RE = /^(\S+)\s+v(\d+)$/;

/** Parse a managed marker from a line, or ``null`` when the line is not one of ours. */
export function parseMarker(line: string, format: ManagedFormat = "html"): { markerId: string; version: number } | null {
  const m = (format === "hash" ? HASH_MARKER_RE : format === "json" ? JSON_MARKER_RE : MARKER_RE).exec(line.trim());
  if (!m) return null;
  return { markerId: m[1]!, version: Number(m[2]) };
}

/** The marker an existing body carries, or ``null`` — the ownership question, asked of an existing body. */
export function ownerMarker(body: string, format: ManagedFormat = "html"): { markerId: string; version: number } | null {
  if (format === "json") {
    let parsed: unknown;
    try { parsed = JSON.parse(body); } catch { return null; }
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
    const v = (parsed as Record<string, unknown>)[JSON_MARKER_KEY];
    return typeof v === "string" ? parseMarker(v, "json") : null;
  }
  const nl = body.indexOf("\n");
  return parseMarker(nl === -1 ? body : body.slice(0, nl), format);
}

/**
 * Write (or refresh, or leave) ``file`` under the already-contained absolute
 * ``workspaceDir``, per the §5 rules.  ``write`` is the caller's atomic write;
 * ``log`` receives any warning.  Never raises — an I/O failure becomes an
 * ``error`` outcome the caller can surface.
 */
export function writeManagedFile(
  workspaceDir: string,
  file: ManagedFile,
  write: AtomicWrite,
  log: (msg: string) => void = () => undefined,
): ManagedWriteOutcome {
  const dest = join(workspaceDir, file.relativePath);
  try {
    if (existsSync(dest)) {
      const current = readFileSync(dest, "utf8");
      const marker = ownerMarker(current, file.format);
      if (!marker || marker.markerId !== file.markerId) {
        // The user replaced our file (deleted the marker, or it was never
        // ours).  Their file, left as-is — never a silent skip.
        return { path: file.relativePath, action: "skipped-user-file" };
      }
      if (marker.version === file.version && (!file.generated || current === managedContent(file))) {
        return { path: file.relativePath, action: "unchanged" };
      }
      mkdirSync(dirname(dest), { recursive: true });
      write(dest, managedContent(file), 0o644);
      if (marker.version === file.version) return { path: file.relativePath, action: "written", reason: "content-changed" };
      return { path: file.relativePath, action: "written", reason: "version-changed", previousVersion: marker.version };
    }
    mkdirSync(dirname(dest), { recursive: true });
    write(dest, managedContent(file), 0o644);
    return { path: file.relativePath, action: "written", reason: "absent" };
  } catch (e) {
    const message = (e as Error).message;
    log(`managed file ${file.relativePath}: could not write: ${message}`);
    return { path: file.relativePath, action: "error", message };
  }
}

/**
 * Remove ``file`` from ``workspaceDir`` on bind-to-none — but only when it
 * still carries our marker.  A copy the user made their own (marker deleted)
 * is theirs and is left in place, with a ``skipped-user-file`` outcome so the
 * caller can say so.
 */
export function removeManagedFile(
  workspaceDir: string,
  file: ManagedFile,
  log: (msg: string) => void = () => undefined,
): ManagedRemoveOutcome {
  const dest = join(workspaceDir, file.relativePath);
  try {
    if (!existsSync(dest)) return { path: file.relativePath, action: "absent" };
    const marker = ownerMarker(readFileSync(dest, "utf8"), file.format);
    if (!marker || marker.markerId !== file.markerId) {
      return { path: file.relativePath, action: "skipped-user-file" };
    }
    rmSync(dest);
    return { path: file.relativePath, action: "removed" };
  } catch (e) {
    const message = (e as Error).message;
    log(`managed file ${file.relativePath}: could not remove: ${message}`);
    return { path: file.relativePath, action: "error", message };
  }
}
