/**
 * The workspace verbs of protocol 1.27, as the web client reads them.
 *
 * - ``WorkspaceInfo.sources`` -- the git checkouts in a workspace, derived
 *   by the daemon from ``.git`` on disk (not declared anywhere), so the
 *   Workspaces table's *Sources* column is true of any workspace, including
 *   one whose repositories were cloned by hand.
 * - ``workspace.inspect`` → ``workspace.inspected`` -- what deleting a
 *   workspace would lose: sessions by state, per-repository uncommitted
 *   files and unpushed commits, size on disk.
 * - ``workspace.clone`` → a stream of ``workspace.clone_progress`` -- one
 *   repository at a time, ``queued`` → ``cloning`` (with a percentage) →
 *   ``checkout`` → ``done`` | ``failed``.
 *
 * Everything here is a pure reading of free-form dicts; nothing trusts a
 * field to be present, because an older daemon sends none of them.
 */

export interface WorkspaceSource {
  forge: string;
  repo: string;
  branch: string;
  path: string;
}

export function normalizeSources(raw: unknown): WorkspaceSource[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((r): r is Record<string, unknown> => !!r && typeof r === "object")
    .map((r) => ({ forge: String(r.forge ?? ""), repo: String(r.repo ?? ""), branch: String(r.branch ?? ""), path: String(r.path ?? "") }))
    .filter((s) => s.repo || s.path);
}

/** ``github: owner/repo@branch`` -- or the directory, for a checkout with no origin. */
export function formatSource(s: WorkspaceSource): string {
  const at = s.branch ? `@${s.branch}` : "";
  if (!s.repo) return `${s.path}${at} (no remote)`;
  return `${s.forge ? `${s.forge}: ` : ""}${s.repo}${at}`;
}

export interface InspectRepo extends WorkspaceSource {
  uncommitted: number | null;
  unpushed: number | null;
  error: string;
}

export interface WorkspaceInspection {
  name: string;
  ok: boolean;
  error: string;
  path: string;
  sizeBytes: number | null;
  sessions: { total: number; waiting: number; awake: number; sleeping: number };
  repos: InspectRepo[];
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

export function normalizeInspection(raw: Record<string, unknown>): WorkspaceInspection {
  const s = (raw.sessions && typeof raw.sessions === "object" ? raw.sessions : {}) as Record<string, unknown>;
  const count = (k: string) => num(s[k]) ?? 0;
  const repos = Array.isArray(raw.repos) ? (raw.repos as Record<string, unknown>[]) : [];
  return {
    name: String(raw.name ?? ""),
    ok: raw.ok !== false,
    error: String(raw.error ?? ""),
    path: String(raw.path ?? ""),
    sizeBytes: num(raw.size_bytes),
    sessions: { total: count("total"), waiting: count("waiting"), awake: count("awake"), sleeping: count("sleeping") },
    repos: repos.map((r) => ({
      ...normalizeSources([r])[0] ?? { forge: "", repo: "", branch: "", path: "" },
      uncommitted: num(r.uncommitted),
      unpushed: num(r.unpushed),
      error: String(r.error ?? ""),
    })),
  };
}

/** ``5 uncommitted files, 1 commit not pushed`` -- or ``""`` when nothing would be lost. */
export function repoLoss(r: InspectRepo): string {
  const parts: string[] = [];
  if (r.uncommitted) parts.push(`${r.uncommitted} uncommitted ${r.uncommitted === 1 ? "file" : "files"}`);
  if (r.unpushed) parts.push(`${r.unpushed} ${r.unpushed === 1 ? "commit" : "commits"} not pushed`);
  return parts.join(", ");
}

/**
 * Whether deleting needs the name typed.  Anything to lose -- a session, a
 * repository -- asks for it; an empty workspace is one click.  A failed
 * inspection asks for it too: not knowing what is there is not the same as
 * knowing nothing is.
 */
export function needsTypedConfirm(i: WorkspaceInspection | null): boolean {
  if (!i || !i.ok) return true;
  return i.sessions.total > 0 || i.repos.length > 0;
}

export function formatBytes(n: number | null): string {
  if (n == null) return "size unknown";
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) { v /= 1024; u++; }
  return `${v >= 10 ? Math.round(v) : v.toFixed(1)} ${units[u]}`;
}

// ── Cloning ──────────────────────────────────────────────────────────

export type CloneState = "queued" | "cloning" | "checkout" | "done" | "failed";

export interface CloneRow {
  repo: string;
  branch: string;
  state: CloneState;
  percent: number;
  error: string;
}

const STATES: CloneState[] = ["queued", "cloning", "checkout", "done", "failed"];

/**
 * Fold one ``workspace.clone_progress`` into the rows.  A row the event
 * names is updated in place; an event naming no repository is a refusal
 * of the whole request and fails every row not already done.
 */
export function applyCloneProgress(rows: CloneRow[], ev: Record<string, unknown>): CloneRow[] {
  const repo = String(ev.repo ?? "");
  const state = STATES.includes(ev.state as CloneState) ? (ev.state as CloneState) : null;
  if (!state) return rows;
  const error = String(ev.error ?? "");
  if (!repo) return rows.map((r) => (r.state === "done" ? r : { ...r, state: "failed", error: error || "the daemon refused the clone" }));
  return rows.map((r) => {
    if (r.repo !== repo) return r;
    const percent = state === "done" ? 100 : num(ev.percent) ?? r.percent;
    // A failure keeps the bar where it stopped.
    return { ...r, state, percent: state === "queued" ? 0 : percent, error: state === "failed" ? error || "clone failed" : "" };
  });
}

export function cloneStatusLine(r: CloneRow): string {
  switch (r.state) {
    case "queued": return "queued";
    case "cloning": return `cloning… ${r.percent}%`;
    case "checkout": return `checking out ${r.branch}`;
    case "done": return `checked out @${r.branch}`;
    case "failed": return `clone failed: ${r.error}`;
  }
}

export const CLONE_GLYPH: Record<CloneState, string> = { queued: "○", cloning: "◐", checkout: "◑", done: "✓", failed: "✕" };

/** The target a repository clones into: ``{workspace}/{repo name}``. */
export function cloneTarget(workspace: string, repo: string): string {
  return `${workspace}/${repo.split("/").pop() ?? repo}`;
}

/** Workspace names the daemon accepts as one flat component. */
export const WORKSPACE_NAME_RE = /^[A-Za-z0-9._-]+$/;

export function validWorkspaceName(name: string): boolean {
  const n = name.trim();
  return WORKSPACE_NAME_RE.test(n) && n !== "." && n !== "..";
}
