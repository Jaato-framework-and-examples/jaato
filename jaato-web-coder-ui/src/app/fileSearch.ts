/**
 * Finding any file in the workspace by name (protocol 1.32).
 *
 * The Files panel lists what CHANGED.  A file nobody touched this session,
 * one the user hid, or one git ignores never appears there, so the panel's
 * finder asks the daemon (``JaatoClient.searchWorkspaceFiles``), which walks
 * the whole workspace -- dotfiles, gitignored paths and hidden entries
 * included.  What is searched is the daemon's decision
 * (``server/workspace_file_search.py``); this module only decides when to
 * ask and how to word the answer.
 */
import { isProtocolCompatible, MIN_FILE_SEARCH_PROTOCOL, type WorkspaceFilesSearchResultEvent } from "@jaato/sdk";

/** Fewest characters worth a search: one letter matches most of a tree. */
export const MIN_QUERY_CHARS = 2;
/** Pause after the last keystroke before asking. */
export const SEARCH_DEBOUNCE_MS = 250;

/** One match as the daemon sends it. */
export interface FileMatch {
  path: string;
  size: number;
  credential: boolean;
}

/** True when a daemon speaking ``protocolVersion`` serves ``workspace.files.search``. */
export function servesFileSearch(protocolVersion: string | null | undefined): boolean {
  return !!protocolVersion && isProtocolCompatible(protocolVersion, MIN_FILE_SEARCH_PROTOCOL);
}

/** True when ``query`` is long enough to send. */
export function isSearchable(query: string): boolean {
  return query.trim().length >= MIN_QUERY_CHARS;
}

/** The matches of an answer, read defensively (a row missing ``path`` is dropped). */
export function matchesOf(answer: Pick<WorkspaceFilesSearchResultEvent, "matches">): FileMatch[] {
  const rows = Array.isArray(answer.matches) ? answer.matches : [];
  return rows.flatMap((r) => {
    const row = r as Partial<FileMatch>;
    if (typeof row.path !== "string" || !row.path) return [];
    return [{ path: row.path, size: typeof row.size === "number" ? row.size : 0, credential: row.credential === true }];
  });
}

/**
 * The line under the results.  A walk that stopped early is said, because
 * "no match" from a partial search is not "no such file".
 */
export function summaryText(answer: Pick<WorkspaceFilesSearchResultEvent, "total" | "truncated">, shown: number): string {
  const total = typeof answer.total === "number" ? answer.total : shown;
  const head = total === 0 ? "No matching file" : shown < total ? `${shown} of ${total} matches` : `${total} match${total === 1 ? "" : "es"}`;
  return answer.truncated ? `${head} (the search stopped early; narrow it to see more)` : head;
}

/** A size as a reader wants it. */
export function sizeText(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
