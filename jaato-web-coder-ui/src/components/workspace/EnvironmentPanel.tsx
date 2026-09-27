/**
 * A workspace's toolchains (#1344): what is bound, what its files propose,
 * and the install in progress.  Drawn in two places:
 *
 *   • under the New workspace plate once every repository is checked out
 *     (``scan``), which asks the backend to scan the clones and write the
 *     repository-guidance pointer, then shows the proposals -- the
 *     clone-time chip;
 *   • in the session rail's Toolchains section, with the mid-session hint
 *     (``hint``): a command the last shell run could not find.
 *
 * Nothing is installed without a click.  A proposal offers **Bind** and
 * **Not now**; "Not now" is remembered by the backend for this user and
 * workspace.  An install shows its last output lines and a **Cancel**; a
 * failed one says why and offers **Retry**.  Bound toolchains can be
 * unbound, which removes the links and the managed files but keeps the
 * download, so binding again is quick.
 *
 * Every control is a real ``button`` whose accessible name contains its
 * visible word, and none is hover-gated.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  environmentApi,
  proposalText,
  type EnvironmentJob,
  type EnvironmentStatus,
  type ToolId,
} from "@/app/environment";

/** How often a running install is polled. */
export const JOB_POLL_MS = 1000;

export function EnvironmentPanel({ url, workspace, scan = false, hint, onHintDone, fetchImpl }: {
  url: string;
  /** The workspace's absolute path (what the daemon and the backend know it by). */
  workspace: string;
  /** Scan the workspace (and write the repository-guidance pointer) on mount -- the clone-time case. */
  scan?: boolean;
  /** A missing command the last shell run reported, when a known toolchain provides it. */
  hint?: { command: string; tool: string } | null;
  onHintDone?: () => void;
  fetchImpl?: typeof fetch;
}) {
  const api = useRef(environmentApi(url, fetchImpl)).current;
  const [status, setStatus] = useState<EnvironmentStatus | null>(null);
  const [job, setJob] = useState<EnvironmentJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [pick, setPick] = useState<{ tool: ToolId | ""; version: string }>({ tool: "", version: "" });

  const load = useCallback(async (withScan: boolean) => {
    try {
      const st = withScan ? await api.refresh(workspace) : await api.status(workspace);
      setStatus(st);
      setJob(st.job);
      setError(null);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  }, [api, workspace]);

  useEffect(() => { void load(scan); }, [load, scan]);

  // Poll a running install; reload the status once it ends.
  useEffect(() => {
    if (!job || job.status !== "running") return;
    const t = setTimeout(async () => {
      try {
        const next = await api.job(job.id);
        setJob(next);
        if (next.status !== "running") await load(false);
      } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    }, JOB_POLL_MS);
    return () => clearTimeout(t);
  }, [api, job, load]);

  const act = async (fn: () => Promise<void>) => {
    setBusy(true);
    try { await fn(); setError(null); } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };
  const bind = (tool: ToolId, version: string) => act(async () => {
    setJob(await api.bind(workspace, tool, version));
    if (hint && hint.tool === tool) onHintDone?.();
  });
  const decline = (tool: ToolId) => act(async () => { await api.decline(workspace, tool); await load(false); });
  const unbind = (tool: ToolId) => act(async () => { const r = await api.unbind(workspace, tool); setStatus(r.status); });
  const cancel = () => act(async () => { if (job) setJob(await api.cancel(job.id)); });

  if (!status) {
    return <div className="px-3.5 py-3 text-[13px] text-text-muted" role="status">{error ?? "Reading the workspace's toolchains…"}</div>;
  }
  const running = job?.status === "running";
  const bound = new Set(status.toolchains.map((t) => t.tool));
  const hintTool = hint ? status.allowed.find((a) => a.tool === hint.tool && !bound.has(a.tool)) : undefined;
  const addable = status.allowed.filter((a) => !bound.has(a.tool));
  const pickedAllowed = addable.find((a) => a.tool === pick.tool);

  return (
    <div className="flex flex-col gap-3 px-3.5 py-3 text-[13px]" data-testid="environment-panel">
      {error && <div className="text-error" role="alert">{error}</div>}

      {hintTool && (
        <div className="tint-warning border hairline px-3 py-2 flex flex-col gap-2" data-testid="environment-hint">
          <span><code className="font-mono">{hint!.command}</code> was not found in the last command. Bind {hintTool.label} {hintTool.tool === "python" ? "" : hintTool.versions[0]}?</span>
          <div className="flex gap-2">
            <button type="button" disabled={busy || running} className="btn btn-sm btn-steel" onClick={() => void bind(hintTool.tool, hintTool.versions[0]!)}>Bind {hintTool.label}</button>
            <button type="button" className="btn btn-sm btn-quiet" onClick={() => onHintDone?.()}>Dismiss</button>
          </div>
        </div>
      )}

      {status.proposals.length > 0 && (
        <ul className="m-0 p-0 list-none flex flex-col gap-2" aria-label="Toolchain proposals">
          {status.proposals.map((p) => (
            <li key={p.tool} className="border hairline px-3 py-2 flex flex-col gap-2" data-testid="environment-proposal">
              <span>{proposalText(p)}</span>
              <div className="flex gap-2">
                <button type="button" disabled={busy || running} className="btn btn-sm btn-steel" onClick={() => void bind(p.tool, p.version)}>Bind {p.label}</button>
                <button type="button" disabled={busy} className="btn btn-sm btn-quiet" onClick={() => void decline(p.tool)}>Not now</button>
              </div>
            </li>
          ))}
        </ul>
      )}

      {job && (
        <div className="border hairline px-3 py-2 flex flex-col gap-1.5" data-testid="environment-job" data-state={job.status}>
          <span className={job.status === "failed" ? "text-error" : job.status === "done" ? "text-success" : "text-steel"} role="status">
            {job.status === "running" ? `Installing ${job.tool} ${job.version}…`
              : job.status === "done" ? `${job.tool} ${job.version} installed.`
              : job.status === "cancelled" ? `Install of ${job.tool} ${job.version} cancelled.`
              : `Install of ${job.tool} ${job.version} failed: ${job.error ?? "unknown error"}`}
          </span>
          {job.log.length > 0 && job.status !== "done" && (
            <pre className="m-0 font-mono text-[11px] text-text-muted whitespace-pre-wrap break-words max-h-32 overflow-auto">{job.log.slice(-8).join("\n")}</pre>
          )}
          {job.notes.map((n) => <span key={n} className="text-warning">{n}</span>)}
          <div className="flex gap-2">
            {running && <button type="button" disabled={busy} className="btn btn-sm" onClick={() => void cancel()}>Cancel install</button>}
            {(job.status === "failed" || job.status === "cancelled") && (
              <button type="button" disabled={busy} className="btn btn-sm btn-steel" onClick={() => void bind(job.tool, job.version)}>Retry {job.tool}</button>
            )}
          </div>
        </div>
      )}

      <section aria-label="Bound toolchains" className="flex flex-col gap-1.5">
        <span className="kicker">Bound</span>
        {status.toolchains.length === 0 ? (
          <span className="text-text-muted italic">None yet. {status.allowed.length ? "Bind one below or from a proposal." : "This server offers none."}</span>
        ) : (
          <ul className="m-0 p-0 list-none flex flex-col gap-1">
            {status.toolchains.map((t) => (
              <li key={t.tool} className="flex items-baseline justify-between gap-2" data-testid="environment-bound">
                <span>
                  <span className="font-mono">{t.tool}{t.tool === "python" ? "" : ` ${t.version}`}</span>
                  {t.server && <span className="text-text-muted"> · {t.server.id}</span>}
                </span>
                <button type="button" disabled={busy || running} className="btn btn-sm btn-quiet" onClick={() => void unbind(t.tool)}>Unbind {t.tool}</button>
              </li>
            ))}
          </ul>
        )}
      </section>

      {addable.length > 0 && (
        <form className="flex flex-wrap items-end gap-2" onSubmit={(e) => { e.preventDefault(); if (pickedAllowed && pick.version) void bind(pickedAllowed.tool, pick.version); }}>
          <label className="flex flex-col gap-1">
            <span className="kicker">Toolchain</span>
            <select className="input" value={pick.tool} onChange={(e) => {
              const a = addable.find((x) => x.tool === e.target.value);
              setPick({ tool: (a?.tool ?? "") as ToolId | "", version: a?.versions[0] ?? "" });
            }}>
              <option value="">Choose…</option>
              {addable.map((a) => <option key={a.tool} value={a.tool}>{a.label}</option>)}
            </select>
          </label>
          {pickedAllowed && pickedAllowed.tool !== "python" && (
            <label className="flex flex-col gap-1">
              <span className="kicker">Version</span>
              <select className="input" value={pick.version} onChange={(e) => setPick({ ...pick, version: e.target.value })}>
                {pickedAllowed.versions.map((v) => <option key={v} value={v}>{v}</option>)}
              </select>
            </label>
          )}
          <button type="submit" disabled={busy || running || !pickedAllowed} className="btn btn-sm btn-steel">Bind</button>
        </form>
      )}

      {status.guidance.length > 0 && (
        <section aria-label="Repository guidance" className="flex flex-col gap-1">
          <span className="kicker">Repository guidance</span>
          <span className="text-text-muted">Sessions are pointed at: {status.guidance.map((g) => <code key={g} className="font-mono mr-1">{g}</code>)}</span>
        </section>
      )}
    </div>
  );
}
