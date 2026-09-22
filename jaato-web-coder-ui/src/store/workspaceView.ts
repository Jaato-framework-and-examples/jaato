/**
 * The Files panel's reset point (#1189): "show only what changes from now
 * on" -- the TUI's ``workspace_clear`` -- kept so it survives a reconnect.
 *
 * The store keeps the FULL list (``workspaceFiles``) and a per-path change
 * number (``workspaceSeqs``); a reset records a mark, and the panel shows the
 * paths whose number is past it.  Keeping the full list rather than
 * deleting from it is what makes "show everything" possible, and it is why a
 * file the agent touches again after a reset reappears: its number moves past
 * the mark whether or not it was in the list before.
 *
 * **Why the daemon numbers changes, rather than this module.**  A reconnect
 * sends a ``workspace.files_snapshot`` the store applies wholesale, and a
 * ``{path, status}`` entry does not say WHEN it changed -- so a mark kept
 * only here was undone by the next reconnect, which a browser does
 * routinely (a tablet sleeps, a network blips).  Since protocol 1.19 every
 * changed event carries the batch's ``seq`` and ``epoch``, and the snapshot
 * carries ``seqs`` (path -> seq of its latest change).
 *
 * **The epoch is what keeps this from failing silently.**  The monitor is
 * rebuilt when the daemon reloads a session, and it counts from 0 again.  A
 * mark of 500 compared against a counter that restarted would hide every new
 * change and leave the panel empty, with nothing saying why.  So a mark
 * belongs to the epoch it was taken in, and a snapshot naming a different
 * epoch VOIDS it -- the full list comes back, and the notice says so.
 *
 * **An older daemon** sends neither field.  Changes are numbered locally
 * instead, which is enough between snapshots, and a snapshot with no epoch
 * voids the mark -- the TUI's behaviour before 1.19, reset until the next
 * snapshot, said out loud rather than silently.
 */

/** A reset point: the epoch it was taken in and the last number seen then. */
export interface WorkspaceReset {
  epoch: string | null;
  seq: number;
}

/** What the store keeps about workspace changes, beyond the path -> status map. */
export interface WorkspaceNumbering {
  files: Record<string, string>;
  seqs: Record<string, number>;
  /** The monitor instance whose numbers ``seqs`` holds; ``null`` = unnumbered daemon. */
  epoch: string | null;
  /** The highest number seen -- what a reset records. */
  seq: number;
  reset: WorkspaceReset | null;
}

/** The outcome of applying one event: the new numbering, and a notice when a mark was voided. */
export interface WorkspaceApplied extends WorkspaceNumbering {
  voided: string | null;
}

function statusOf(o: Record<string, unknown>): string {
  return String(o.status ?? o.change ?? o.type ?? "modified");
}

/** ``workspace.files_changed`` -- one batch, one number. */
export function applyChanged(cur: WorkspaceNumbering, ev: Record<string, unknown>): WorkspaceApplied {
  const numbered = typeof ev.seq === "number" && typeof ev.epoch === "string";
  const epoch = numbered ? (ev.epoch as string) : cur.epoch;
  // A batch from a monitor this client has not seen a snapshot of: the
  // numbers it holds belong to the old one.  Void them and the mark.
  const switched = numbered && cur.epoch !== null && cur.epoch !== epoch;
  const seq = numbered ? (ev.seq as number) : cur.seq + 1;
  const files = { ...cur.files };
  const seqs = switched ? {} : { ...cur.seqs };
  for (const ch of (ev.changes as Record<string, unknown>[] | undefined) ?? []) {
    const p = (ch.path ?? ch.file) as string | undefined;
    if (!p) continue;
    const status = statusOf(ch);
    if (status === "deleted") { delete files[p]; delete seqs[p]; }
    else { files[p] = status; seqs[p] = seq; }
  }
  const voided = switched && cur.reset ? "The reset was dropped: the session was reloaded, so its changes were renumbered." : null;
  return { files, seqs, epoch, seq: Math.max(switched ? 0 : cur.seq, seq), reset: voided ? null : cur.reset, voided };
}

/** ``workspace.files_snapshot`` -- the whole list, replacing what was held. */
export function applySnapshot(cur: WorkspaceNumbering, ev: Record<string, unknown>): WorkspaceApplied {
  const files: Record<string, string> = {};
  const given = (ev.seqs && typeof ev.seqs === "object" ? ev.seqs : {}) as Record<string, unknown>;
  const seqs: Record<string, number> = {};
  for (const f of (ev.files as unknown[] | undefined) ?? []) {
    let p: string | undefined;
    let status = "modified";
    if (typeof f === "string") p = f;
    else if (f && typeof f === "object") {
      const o = f as Record<string, unknown>;
      p = (o.path ?? o.file) as string | undefined;
      status = statusOf(o);
    }
    if (!p || status === "deleted") continue;
    files[p] = status;
    // No number for a path means "changed before this monitor numbered
    // anything" -- older than any mark taken in this epoch.
    seqs[p] = typeof given[p] === "number" ? (given[p] as number) : 0;
  }
  const epoch = typeof ev.epoch === "string" ? ev.epoch : null;
  const seq = typeof ev.seq === "number" ? ev.seq : 0;
  let voided: string | null = null;
  let reset = cur.reset;
  if (reset && epoch === null) {
    voided = "The reset was dropped: this daemon does not number file changes, so a reconnect cannot keep it (it needs protocol 1.19).";
    reset = null;
  } else if (reset && reset.epoch !== epoch) {
    voided = "The reset was dropped: the session was reloaded, so its changes were renumbered.";
    reset = null;
  }
  return { files, seqs, epoch, seq, reset, voided };
}

/** The paths the panel shows: everything, or only what changed past the mark. */
export function visibleFiles(n: Pick<WorkspaceNumbering, "files" | "seqs" | "reset">): Record<string, string> {
  if (!n.reset) return n.files;
  const mark = n.reset.seq;
  const out: Record<string, string> = {};
  for (const [p, status] of Object.entries(n.files)) {
    if ((n.seqs[p] ?? 0) > mark) out[p] = status;
  }
  return out;
}

/** Take a mark now: everything currently known is at or below it. */
export function markReset(n: Pick<WorkspaceNumbering, "epoch" | "seq">): WorkspaceReset {
  return { epoch: n.epoch, seq: n.seq };
}
