/**
 * The Workspaces table's inline delete panel (design 3a): a full-width row
 * directly under the workspace being deleted, saying what is lost before
 * anything is.
 *
 * On open it asks the daemon (``workspace.inspect``, protocol 1.27) for the
 * impact -- sessions by state, each repository's uncommitted files and
 * unpushed commits, size on disk -- and renders it in three cells.  Then:
 *
 *  • anything to lose (a session, a repository) → type the name to
 *    confirm; the red button is enabled only on an exact match;
 *  • nothing to lose → one click.
 *
 * A failed or unanswered inspection is treated as "something to lose": not
 * knowing what is there is not the same as knowing nothing is.
 *
 * Deleting sends ``stop_sessions``: the panel has just said open prompts
 * are cancelled and running work stops, so the daemon is asked to do
 * exactly that rather than refuse because sessions are loaded.
 */
import { useEffect, useState } from "react";
import { inspectWorkspace } from "@/sdk/connection";
import { formatBytes, needsTypedConfirm, normalizeInspection, repoLoss, formatSource, type WorkspaceInspection } from "@/protocol/workspaces";

export interface DeletePanelProps {
  name: string;
  /** Absolute path, for the "On disk" cell while the inspection is pending. */
  path?: string | null;
  busy: boolean;
  onDelete: () => void;
  onCancel: () => void;
}

function Cell({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-surface px-4 py-3 flex flex-col gap-1.5 min-w-0">
      <span className="kicker kicker-muted">{title}</span>
      {children}
    </div>
  );
}

function SessionsCell({ i }: { i: WorkspaceInspection }) {
  const s = i.sessions;
  const live = s.waiting + s.awake > 0;
  return (
    <Cell title="Sessions">
      <span className="text-[14px]">{s.total ? `${s.total} ${s.total === 1 ? "session" : "sessions"} deleted` : "No sessions"}</span>
      {s.total > 0 && (
        <span className="font-mono text-[12px] flex flex-wrap gap-x-3">
          {s.waiting > 0 && <span className="text-warning">⚠ {s.waiting} waiting</span>}
          {s.awake > 0 && <span className="text-success">● {s.awake} awake</span>}
          {s.sleeping > 0 && <span className="text-text-muted">○ {s.sleeping} sleeping</span>}
        </span>
      )}
      {live && <span className="text-[12px] text-warning">Open prompts are cancelled. Running work stops.</span>}
    </Cell>
  );
}

function ReposCell({ i }: { i: WorkspaceInspection }) {
  return (
    <Cell title="Repositories">
      {i.repos.length === 0 && <span className="text-[14px] text-text-muted">None</span>}
      {i.repos.map((r) => {
        const loss = repoLoss(r);
        return (
          <div key={r.path || r.repo} className="flex flex-col">
            <span className="font-mono text-[13px] break-words">{formatSource(r).replace(/^[a-z]+: /, "")}</span>
            {r.error ? <span className="text-[12px] text-warning">⚠ status unknown: {r.error}</span>
              : loss ? <span className="text-[12px] text-warning">⚠ {loss}, lost</span>
              : <span className="text-[12px] text-text-muted">clean</span>}
          </div>
        );
      })}
    </Cell>
  );
}

export function DeletePanel({ name, path, busy, onDelete, onCancel }: DeletePanelProps) {
  const [inspection, setInspection] = useState<WorkspaceInspection | null>(null);
  const [pending, setPending] = useState(true);
  const [typed, setTyped] = useState("");
  useEffect(() => {
    let live = true;
    setPending(true);
    setTyped("");
    inspectWorkspace(name).then((raw) => {
      if (!live) return;
      setInspection(raw ? normalizeInspection(raw) : null);
      setPending(false);
    });
    return () => { live = false; };
  }, [name]);

  const typedConfirm = needsTypedConfirm(pending ? null : inspection);
  const matches = typed.trim() === name;
  const canDelete = !busy && !pending && (!typedConfirm || matches);
  const failed = !pending && (!inspection || !inspection.ok);

  return (
    <div className="px-[18px] py-4 flex flex-col gap-3.5 font-sans" role="group" aria-label={`Delete workspace ${name}`} data-testid="delete-panel">
      <div className="flex flex-wrap items-baseline gap-3">
        <span className="chrome text-error">Delete workspace</span>
        <span className="font-mono text-[14px]">{name}</span>
        <span className="text-[13px] text-text-muted">This can't be undone.</span>
      </div>
      {pending ? (
        <div className="text-[13px] text-text-muted" role="status">Checking what this workspace holds…</div>
      ) : failed ? (
        <div className="text-[13px] text-warning" role="status">
          Could not check what this workspace holds{inspection?.error ? `: ${inspection.error}` : " (the daemon did not answer)"}. Anything in it is deleted.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-divider border hairline">
          <SessionsCell i={inspection!} />
          <ReposCell i={inspection!} />
          <Cell title="On disk">
            <span className="text-[14px]">{formatBytes(inspection!.sizeBytes)}</span>
            <span className="font-mono text-[12px] text-text-muted break-all">{inspection!.path || path || ""}</span>
          </Cell>
        </div>
      )}
      {!pending && (typedConfirm ? (
        <div className="flex flex-col gap-2">
          <label htmlFor={`confirm-delete-${name}`} className="text-[13px]">
            Type <span className="font-mono border hairline bg-surface px-1.5">{name}</span> to confirm
          </label>
          <div className="flex flex-wrap gap-2">
            <input
              id={`confirm-delete-${name}`}
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              autoComplete="off"
              spellCheck={false}
              autoCapitalize="off"
              aria-label={`Type ${name} to confirm`}
              className={`input input-mono flex-1 min-w-[12rem] ${typed && !matches ? "input-warn" : ""}`}
            />
            <button type="button" disabled={!canDelete} onClick={onDelete} className="btn btn-danger-solid" aria-label={`Delete workspace ${name} permanently`}>Delete workspace</button>
            <button type="button" disabled={busy} onClick={onCancel} className="btn">Cancel</button>
          </div>
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[13px] flex-1">Nothing to lose: no sessions, no repositories, no staged files.</span>
          <button type="button" disabled={!canDelete} onClick={onDelete} className="btn btn-danger-solid" aria-label={`Delete workspace ${name} permanently`}>Delete workspace</button>
          <button type="button" disabled={busy} onClick={onCancel} className="btn">Cancel</button>
        </div>
      ))}
    </div>
  );
}
