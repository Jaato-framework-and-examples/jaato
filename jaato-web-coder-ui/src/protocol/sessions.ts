/**
 * The daemon's session listing, as the web client reads it.
 *
 * ``SessionListEvent.sessions`` (the answer to ``session list``) and
 * ``SessionInfoEvent.sessions`` (the snapshot every session-info carries)
 * are both ``[{id, name, description, model_provider, model_name,
 * is_loaded, is_current, client_count, turn_count, workspace_path, ...}]``
 * — free-form dicts the daemon documents as "client handles formatting".
 * This module is the one reading of that shape: it feeds the ``session
 * list`` listing (the TUI's rendering, ported), the composer's third-level
 * completion for ``session attach <id>`` / ``session delete <id>`` /
 * ``resume <id>``, and the session picker's "resume" list.
 */

export interface SessionSummary {
  id: string;
  name: string;
  description: string;
  provider: string;
  model: string;
  isLoaded: boolean;
  isCurrent: boolean;
  clientCount: number;
  turnCount: number;
  workspacePath: string;
  createdBy?: string;
  /**
   * ``"permission"`` / ``"clarification"`` when that session is blocked on
   * an unanswered human prompt (protocol 1.17), undefined otherwise.
   *
   * It is the ONLY way a client working in session A learns that B wants
   * it: prompt events go to that session's attached clients and a client
   * is attached to one at a time.  Absent means "nothing is waiting, as
   * far as this daemon says" -- never a positive no, since a daemon below
   * 1.17 sends nothing and an unloaded session is never reported.
   */
  awaiting?: string;
  /** When that prompt was raised, ISO-8601 UTC.  Absent means NOT MEASURED, never "just now". */
  awaitingSince?: string;
  /** The profile the session was created from (protocol 1.27), "" when none or not reported. */
  profile: string;
  /** Last activity, ISO-8601; "" when the daemon did not send it. */
  lastActivity: string;
  /** A turn is running right now (loaded sessions only). */
  isProcessing: boolean;
}

export function normalizeSessionSummary(raw: unknown): SessionSummary | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const id = String(o.id ?? o.session_id ?? "");
  if (!id) return null;
  return {
    id,
    name: String(o.name ?? ""),
    description: String(o.description ?? ""),
    provider: String(o.model_provider ?? ""),
    model: String(o.model_name ?? ""),
    isLoaded: o.is_loaded === true,
    isCurrent: o.is_current === true,
    clientCount: Number(o.client_count ?? 0) || 0,
    turnCount: Number(o.turn_count ?? 0) || 0,
    workspacePath: String(o.workspace_path ?? ""),
    createdBy: o.created_by ? String(o.created_by) : undefined,
    awaiting: typeof o.awaiting === "string" && o.awaiting ? o.awaiting : undefined,
    awaitingSince: typeof o.awaiting_since === "string" && o.awaiting_since ? o.awaiting_since : undefined,
    profile: typeof o.profile === "string" ? o.profile : "",
    lastActivity: typeof o.last_activity === "string" ? o.last_activity : "",
    isProcessing: o.is_processing === true,
  };
}

export function normalizeSessionList(raw: unknown): SessionSummary[] {
  if (!Array.isArray(raw)) return [];
  return raw.map(normalizeSessionSummary).filter((s): s is SessionSummary => s !== null);
}

/**
 * One line per session, the TUI's ``session list`` rendering -- plus, on its
 * own ``↳`` line, whatever note THIS person wrote about that session
 * (``app/notes.ts``).  Someone who types the command expects to see it.
 *
 * Read-only by nature, and that is right: this is a snapshot printed into a
 * transcript, not a management surface, and it should not grow affordances.
 */
export function formatSessionList(sessions: SessionSummary[], notes: Record<string, { text: string }> = {}): string {
  if (sessions.length === 0) return "No sessions available.\nUse 'session new' to create one.";
  const lines = ["Sessions:", "  Use 'session attach <id>' to switch sessions", ""];
  for (const s of sessions) {
    const status = s.isCurrent ? "▶" : s.isLoaded ? "●" : "○";
    const desc = s.description || s.name;
    const parts = [
      `  ${status} ${s.id}${desc && desc !== s.id ? ` - ${desc}` : ""}`,
      s.awaiting ? ` [waiting: ${s.awaiting}]` : "",
      s.provider ? ` [${s.provider}/${s.model}]` : "",
      s.clientCount ? `, ${s.clientCount} client(s)` : "",
      s.turnCount ? `, ${s.turnCount} turns` : "",
    ];
    lines.push(parts.join(""));
    if (s.workspacePath) lines.push(`      ${s.workspacePath}`);
    const note = notes[s.id]?.text;
    if (note) for (const line of note.split("\n")) lines.push(`      ↳ ${line}`);
  }
  lines.push("", "  ▶ current  ● loaded  ○ on disk");
  return lines.join("\n");
}

/** Commands whose next word is a session id (the TUI's ``SessionIdCompleter``). */
const SESSION_ID_COMMANDS = ["session attach", "session delete", "resume"];

/**
 * Third-level completions: after ``session attach ␣`` (or ``session delete``,
 * ``resume``) propose the known session ids, filtered by the partial id
 * typed so far.  ``null`` when the caret is not in that position, so the
 * caller falls through to the command completer.
 */
export function sessionIdCompletions(
  textBeforeCaret: string,
  sessions: SessionSummary[],
): { insert: string; label: string; description?: string }[] | null {
  const text = textBeforeCaret.replace(/^\s+/, "");
  const lower = text.toLowerCase();
  for (const cmd of SESSION_ID_COMMANDS) {
    if (!lower.startsWith(cmd + " ")) continue;
    const rest = text.slice(cmd.length + 1);
    if (/\s/.test(rest.trim())) return null; // already past the id
    const partial = rest.trim().toLowerCase();
    return sessions
      .filter((s) => s.id.toLowerCase().startsWith(partial))
      .map((s) => ({
        insert: `${cmd} ${s.id}`,
        label: s.id,
        description: [s.isCurrent ? "current" : s.isLoaded ? "loaded" : "on disk", s.description || s.name, s.provider ? `${s.provider}/${s.model}` : "", s.workspacePath].filter(Boolean).join(" · "),
      }));
  }
  return null;
}

/** Does the caret sit where a session id is expected — i.e. should the list be fetched? */
export function wantsSessionIds(textBeforeCaret: string): boolean {
  const lower = textBeforeCaret.replace(/^\s+/, "").toLowerCase();
  return SESSION_ID_COMMANDS.some((cmd) => lower.startsWith(cmd + " "));
}

/**
 * The sessions that belong to a workspace.  A session records its
 * ``workspace_path`` (absolute, daemon-side); a workspace is known to the
 * client by name and, when the daemon sent it, by path.
 */
export function sessionsInWorkspace(
  sessions: SessionSummary[],
  workspace: { name: string; path?: string | null } | undefined,
): SessionSummary[] {
  if (!workspace) return [];
  return sessions.filter((s) => {
    if (!s.workspacePath) return false;
    if (workspace.path && s.workspacePath.replace(/\/+$/, "") === workspace.path.replace(/\/+$/, "")) return true;
    const tail = s.workspacePath.replace(/\/+$/, "").split("/").pop();
    return tail === workspace.name;
  });
}

/**
 * The session picker's three columns of existing sessions (design 2a).
 *
 * ``waiting`` — a prompt is open (the daemon's ``awaiting``, protocol 1.17);
 * ``awake`` — loaded in the daemon; ``sleeping`` — saved on disk only,
 * where attaching wakes it.  A session waiting on a person is loaded by
 * construction, so the waiting test comes first.
 */
export type SessionColumn = "waiting" | "awake" | "sleeping";

export function sessionColumn(s: SessionSummary): SessionColumn {
  if (s.awaiting) return "waiting";
  return s.isLoaded ? "awake" : "sleeping";
}

function time(iso: string | undefined): number {
  const t = iso ? Date.parse(iso) : NaN;
  return Number.isFinite(t) ? t : 0;
}

/**
 * Sessions grouped by column, ordered the way each column is read:
 * longest waiting first (an unmeasured wait sorts last, since nothing says
 * it is old), then most recent activity first.  Ids break ties -- they are
 * timestamps, so the fallback is still recency.
 */
export function sessionBoard(sessions: SessionSummary[]): Record<SessionColumn, SessionSummary[]> {
  const out: Record<SessionColumn, SessionSummary[]> = { waiting: [], awake: [], sleeping: [] };
  for (const s of sessions) out[sessionColumn(s)].push(s);
  out.waiting.sort((a, b) => (time(a.awaitingSince) || Infinity) - (time(b.awaitingSince) || Infinity) || b.id.localeCompare(a.id));
  const recent = (a: SessionSummary, b: SessionSummary) => time(b.lastActivity) - time(a.lastActivity) || b.id.localeCompare(a.id);
  out.awake.sort(recent);
  out.sleeping.sort(recent);
  return out;
}
