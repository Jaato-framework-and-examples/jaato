/**
 * New workspace (design 3b): name it, pick repositories, and have them
 * cloned and checked out BEFORE the session picker opens -- so no session
 * can start in a workspace whose sources are half there.
 *
 * Two phases in one plate:
 *
 *  1. FORM -- name (one flat component: letters, digits, ``.``, ``-``,
 *     ``_``) and the repository picker (``RepoPicker``).
 *  2. PROGRESS -- one row per repository, ``queued`` → ``cloning N%`` →
 *     ``checking out`` → ``checked out`` | ``failed``.  A failed row offers
 *     Retry (a new ``workspace.clone`` for that repository alone) and
 *     "Remove this repo and continue"; the footer offers "Cancel and
 *     delete workspace" until everything is checked out, then "Open
 *     session picker".
 *
 * The order of the requests is load-bearing: the workspace is created,
 * then the user's default GitHub account is bound to it (which writes
 * ``GH_TOKEN=app://github`` into its ``.env``), and only then is the clone
 * asked for -- the daemon resolves the clone's credential the way a session
 * spawned there would, so a private repository cloned before the bind
 * would fail on authentication.
 */
import { useEffect, useRef, useState } from "react";
import { Plate } from "@/components/layout/Plate";
import { RepoPicker, type PickedRepo } from "./RepoPicker";
import { autoBindDefaultGitHubAccount, githubApi } from "@/app/github";
import { cloneIntoWorkspace, createWorkspace, deleteWorkspace } from "@/sdk/connection";
import { applyCloneProgress, CLONE_GLYPH, cloneStatusLine, validWorkspaceName, type CloneRow } from "@/protocol/workspaces";

const TONE: Record<CloneRow["state"], string> = {
  queued: "text-text-muted",
  cloning: "text-steel",
  checkout: "text-steel",
  done: "text-success",
  failed: "text-error",
};
/** Bar fill by state, as a style: ``.bar > span`` is unlayered CSS and a background utility loses to it. */
const BAR: Record<CloneRow["state"], string> = {
  queued: "var(--c-steel)", cloning: "var(--c-steel)", checkout: "var(--c-steel)", done: "var(--c-success)", failed: "var(--c-error)",
};

export function CloneProgress({ name, rows, onRetry, onRemove, onCancel, onOpen, busy, title }: {
  name: string;
  rows: CloneRow[];
  onRetry: (repo: string) => void;
  onRemove: (repo: string) => void;
  onCancel?: () => void;
  onOpen?: () => void;
  busy: boolean;
  title?: { working: string; ready: string };
}) {
  const done = rows.filter((r) => r.state === "done").length;
  const failed = rows.find((r) => r.state === "failed");
  const ready = rows.length > 0 && done === rows.length;
  const heading = title ?? { working: "Creating workspace", ready: "Workspace ready" };
  return (
    <div className="flex flex-col" data-testid="clone-progress">
      <div className="flex flex-wrap items-baseline justify-between gap-3 px-5 py-3.5 border-b hairline">
        <div className="flex items-baseline gap-3">
          <span className={`kicker tracking-[0.16em] ${ready ? "text-success" : ""}`}>{ready ? heading.ready : heading.working}</span>
          <span className="font-mono text-[15px]">{name}</span>
        </div>
        <span className="text-[13px] text-text-muted" role="status">{done} of {rows.length} {rows.length === 1 ? "repository" : "repositories"} checked out</span>
      </div>
      <ul className="m-0 px-5 list-none">
        {rows.map((r) => (
          <li key={r.repo} className="py-3.5 border-b hairline" data-testid="clone-row" data-state={r.state}>
            <div className="flex items-baseline gap-3">
              <span className={`w-4 shrink-0 ${TONE[r.state]}`} aria-hidden="true">{CLONE_GLYPH[r.state]}</span>
              <div className="flex-1 min-w-0 flex flex-col gap-0.5">
                <span className="font-mono text-[13px] break-words">{r.repo}<span className="text-steel">@{r.branch}</span></span>
                <span className={`text-[13px] ${TONE[r.state]}`}>{cloneStatusLine(r)}</span>
              </div>
              {(r.state === "cloning" || r.state === "failed") && <span className="font-mono text-[11px] text-text-muted">{r.percent}%</span>}
            </div>
            <div className="ml-7 mt-2 bar" role="progressbar" aria-valuenow={r.percent} aria-valuemin={0} aria-valuemax={100} aria-label={`${r.repo} progress`}>
              <span style={{ width: `${r.state === "queued" ? 0 : r.percent}%`, background: BAR[r.state] }} />
            </div>
            {r.state === "failed" && (
              <div className="ml-7 mt-2.5 flex flex-wrap gap-2">
                <button type="button" disabled={busy} onClick={() => onRetry(r.repo)} className="btn btn-sm btn-steel">Retry</button>
                <button type="button" disabled={busy} onClick={() => onRemove(r.repo)} className="btn btn-sm btn-quiet">Remove this repo and continue</button>
              </div>
            )}
          </li>
        ))}
      </ul>
      <div className={`flex flex-wrap items-center justify-between gap-3 px-5 py-3.5 border-t hairline ${failed ? "tint-error" : "bg-bg"}`}>
        <span className={`text-[13px] ${failed ? "text-error" : "text-text-muted"}`}>
          {failed ? `${failed.repo} failed. Retry it, remove it, or cancel.`
            : ready ? `All repositories checked out.${onOpen ? " The session picker opens next." : ""}`
            : `Cloning.${onOpen ? " Sessions can start once this finishes." : ""}`}
        </span>
        {ready && onOpen ? (
          <button type="button" onClick={onOpen} className="btn btn-primary gap-6">Open session picker <span aria-hidden="true">→</span></button>
        ) : onCancel ? (
          <button type="button" disabled={busy} onClick={onCancel} className="btn">Cancel and delete workspace</button>
        ) : null}
      </div>
    </div>
  );
}

/** Rows for a clone request, all ``queued`` until the daemon says otherwise. */
export function queuedRows(repos: PickedRepo[]): CloneRow[] {
  return repos.map((r) => ({ repo: r.repo, branch: r.branch || "main", state: "queued", percent: 0, error: "" }));
}

/**
 * Drive one clone batch into ``setRows``; returns the unsubscribe.  Shared
 * by the New workspace plate and the Sources plate.
 */
export function startClone(name: string, rows: CloneRow[], setRows: (fn: (cur: CloneRow[]) => CloneRow[]) => void): () => void {
  return cloneIntoWorkspace(name, rows.map((r) => ({ repo: r.repo, branch: r.branch })), (ev) => setRows((cur) => applyCloneProgress(cur, ev)));
}

export function NewWorkspacePlate({ githubUrl, onCreated, onOpen, onNotice }: {
  githubUrl?: string | null;
  /** The workspace exists (bound, not yet cloned): the table can show it. */
  onCreated: () => void;
  /** Everything is checked out: go to the session picker for ``name``. */
  onOpen: (name: string) => void;
  onNotice: (text: string, error?: boolean) => void;
}) {
  const [name, setName] = useState("");
  const [picked, setPicked] = useState<PickedRepo[]>([]);
  const [phase, setPhase] = useState<"form" | "progress">("form");
  const [created, setCreated] = useState("");
  const [rows, setRows] = useState<CloneRow[]>([]);
  const [busy, setBusy] = useState(false);
  const unsubs = useRef<Array<() => void>>([]);
  useEffect(() => () => { for (const u of unsubs.current) u(); }, []);

  const valid = validWorkspaceName(name);
  const ready = phase === "progress" && rows.length > 0 && rows.every((r) => r.state === "done");

  const clone = (target: string, batch: CloneRow[]) => {
    unsubs.current.push(startClone(target, batch, setRows));
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!valid || busy) return;
    const n = name.trim();
    setBusy(true);
    try {
      const info = await createWorkspace(n);
      if (!info) { onNotice(`Could not create workspace ${n}.`, true); return; }
      onCreated();
      // Bind before cloning: the clone's credential is resolved from the
      // workspace .env, which the bind is what writes.
      if (githubUrl && info.path) {
        try { await autoBindDefaultGitHubAccount(githubApi(githubUrl), info.path); }
        catch (err) { onNotice(`Workspace created, but binding your GitHub account failed: ${err instanceof Error ? err.message : String(err)}`, true); }
      }
      if (picked.length === 0) { onOpen(n); return; }
      const batch = queuedRows(picked);
      setCreated(n);
      setRows(batch);
      setPhase("progress");
      clone(n, batch);
    } finally {
      setBusy(false);
    }
  };

  const retry = (repo: string) => {
    const row = rows.find((r) => r.repo === repo);
    if (!row) return;
    const again: CloneRow = { ...row, state: "queued", percent: 0, error: "" };
    setRows((cur) => cur.map((r) => (r.repo === repo ? again : r)));
    clone(created, [again]);
  };
  const remove = (repo: string) => setRows((cur) => cur.filter((r) => r.repo !== repo));
  const cancel = async () => {
    setBusy(true);
    for (const u of unsubs.current) u();
    unsubs.current = [];
    const answer = await deleteWorkspace(created, { stopSessions: true });
    setBusy(false);
    if (answer && !answer.ok) { onNotice(answer.error || `Could not delete ${created}.`, true); return; }
    onNotice(`Cancelled: deleted ${created}.`);
    setPhase("form");
    setRows([]);
    setCreated("");
  };

  // A ready workspace waits for "Open session picker" -- the button says
  // where it goes.  One whose every repository was REMOVED has nothing left
  // to show here, so it moves on by itself.
  useEffect(() => {
    if (phase === "progress" && rows.length === 0 && created) onOpen(created); // everything was removed
  }, [phase, rows.length, created, onOpen]);

  return (
    <Plate className="w-full max-w-[920px] flex flex-col" aria-label="New workspace" data-testid="new-workspace">
      {phase === "progress" ? (
        <CloneProgress name={created} rows={rows} onRetry={retry} onRemove={remove} onCancel={() => { void cancel(); }} onOpen={ready ? () => onOpen(created) : undefined} busy={busy} />
      ) : (
        <form onSubmit={submit} className="flex flex-col">
          <div className="px-5 py-3.5 border-b hairline"><span className="kicker tracking-[0.16em]">New workspace</span></div>
          <RepoPicker githubUrl={githubUrl} workspace={name.trim()} picked={picked} onChange={setPicked} leading={(
            <div className="flex flex-col gap-1 mb-3">
              <label htmlFor="new-workspace-name" className="text-[12px] text-text-muted">Name</label>
              <input
                id="new-workspace-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="new-workspace-name"
                spellCheck={false}
                autoComplete="off"
                aria-label="New workspace name"
                aria-invalid={name.trim() !== "" && !valid}
                className={`input input-mono ${name.trim() && !valid ? "input-warn" : ""}`}
              />
              {name.trim() && !valid && <span className="text-[12px] text-warning">Letters, digits, “.”, “-” and “_” only.</span>}
            </div>
          )} />
          <div className="flex flex-wrap items-center gap-4 px-5 py-3.5 border-t hairline">
            <button type="submit" disabled={!valid || busy} className="btn btn-primary gap-6">
              {picked.length ? `Create and clone ${picked.length} ${picked.length === 1 ? "repo" : "repos"}` : "Create empty workspace"} <span aria-hidden="true">→</span>
            </button>
            <span className="text-[13px] text-text-muted">Repos are cloned before any session can start.</span>
          </div>
        </form>
      )}
    </Plate>
  );
}
