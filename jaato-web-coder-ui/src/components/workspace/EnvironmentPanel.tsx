/**
 * A workspace's toolchains (#1344): what is bound, what its repositories
 * propose, and the install in progress.  Drawn in two places:
 *
 *   • under the New workspace plate once every repository is checked out.
 *     There is no session there yet, and installs run in a session's runner,
 *     so a choice is remembered (``app/toolchains.ts``, pending binds) and
 *     bound when the first session in the workspace starts;
 *   • in the session rail's Toolchains section, with the mid-session hint
 *     (``hint``): a command the last shell run could not find.
 *
 * In a session the panel is LIVE: it reads ``.jaato/environment.json``, which
 * the ``web_coder_toolchains`` plugin keeps, through the daemon (polling
 * while a job runs), and acts by sending the plugin's ``toolchain`` command.
 * The sign-in backend only supplies the allow-list and remembers "Not now".
 *
 * Nothing is installed without a click.  Every control is a real ``button``
 * whose accessible name contains its visible word, and none is hover-gated.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { environmentApi, type AllowedTool, type EnvironmentStatus, type ToolId } from "@/app/environment";
import { stageToolchainOffer, currentWorkspacePath } from "@/app/toolchainOffer";
import {
  POLL_MS, addPending, canReadManifest, pendingBinds, readManifest, removePending, toolchainCommand,
  type Manifest, type PendingBind, type Proposal,
} from "@/app/toolchains";
import { useJaato } from "@/store/store";

/** The chip text for a proposal. */
export function proposalText(p: Proposal): string {
  const what = p.tool === "python" ? `${p.label}` : `${p.label} ${p.version}`;
  if (p.pin && !p.pinAllowed) return `${p.label} detected (${p.source} pins ${p.pin}, which this server does not offer). Bind ${p.version} instead?`;
  return `${what} detected (from ${p.source}). Bind it?`;
}

export function EnvironmentPanel({ url, workspace, hint, onHintDone, fetchImpl, pollMs = POLL_MS }: {
  url: string;
  /** The workspace's absolute path (what the daemon and the backend know it by). */
  workspace: string;
  /** A missing command the last shell run reported, when a known toolchain provides it. */
  hint?: { command: string; tool: string } | null;
  onHintDone?: () => void;
  fetchImpl?: typeof fetch;
  pollMs?: number;
}) {
  const api = useRef(environmentApi(url, fetchImpl, (st) => { void stageToolchainOffer(st); })).current;
  const live = useJaato((s) => !!s.sessionId && currentWorkspacePath(s) === workspace) && canReadManifest();
  const [status, setStatus] = useState<EnvironmentStatus | null>(null);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [pending, setPending] = useState<PendingBind[]>(() => pendingBinds(workspace));
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [pick, setPick] = useState<{ tool: ToolId | ""; version: string }>({ tool: "", version: "" });

  const loadStatus = useCallback(async () => {
    try { setStatus(await api.status(workspace)); setError(null); } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  }, [api, workspace]);

  const loadManifest = useCallback(async () => {
    if (!live) return;
    try { setManifest(await readManifest()); } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  }, [live]);

  useEffect(() => { void loadStatus(); }, [loadStatus]);
  useEffect(() => { void loadManifest(); }, [loadManifest]);

  // Follow a running job, and pick up the proposals of a scan that has not landed yet.
  const running = manifest?.job?.status === "running";
  useEffect(() => {
    if (!live || !(running || pending.length)) return;
    const t = setInterval(() => { void loadManifest(); setPending(pendingBinds(workspace)); }, pollMs);
    return () => clearInterval(t);
  }, [live, running, pending.length, loadManifest, workspace, pollMs]);

  const act = async (fn: () => Promise<void>) => {
    setBusy(true);
    try { await fn(); setError(null); } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };
  /** Send a command, then re-read until the manifest reflects it (or a few polls pass). */
  const command = (...args: string[]) => act(async () => {
    await toolchainCommand(...args);
    for (let i = 0; i < 4; i++) { await new Promise((r) => setTimeout(r, pollMs / 2)); await loadManifest(); }
  });
  const bind = (tool: ToolId, version: string) => {
    if (hint && hint.tool === tool) onHintDone?.();
    if (!live) {
      addPending(workspace, tool, version);
      setPending(pendingBinds(workspace));
      return Promise.resolve();
    }
    return act(async () => {
      if (status?.declined.includes(tool)) await api.undecline(workspace, tool);
    }).then(() => command("bind", tool, version));
  };
  const decline = (tool: ToolId) => act(async () => { await api.decline(workspace, tool); await loadStatus(); });

  if (!status) {
    return <div className="px-3.5 py-3 text-[13px] text-text-muted" role="status">{error ?? "Reading the workspace's toolchains…"}</div>;
  }
  const m = manifest ?? { toolchains: [], proposals: [], guidance: [], job: null };
  const job = m.job;
  const bound = new Set(m.toolchains.map((t) => t.tool));
  const pendingTools = new Set(pending.map((p) => p.tool));
  const allowed = (tool: string): AllowedTool | undefined => status.allowed.find((a) => a.tool === tool);
  const hintTool = hint ? status.allowed.find((a) => a.tool === hint.tool && !bound.has(a.tool)) : undefined;
  const proposals = m.proposals.filter((p) => !bound.has(p.tool) && !status.declined.includes(p.tool) && !pendingTools.has(p.tool) && allowed(p.tool));
  const addable = status.allowed.filter((a) => !bound.has(a.tool) && !pendingTools.has(a.tool));
  const pickedAllowed = addable.find((a) => a.tool === pick.tool);

  return (
    <div className="flex flex-col gap-3 px-3.5 py-3 text-[13px]" data-testid="environment-panel" data-live={live ? "true" : "false"}>
      {error && <div className="text-error" role="alert">{error}</div>}
      {!live && (
        <div className="text-text-muted" role="note">
          Toolchains are installed inside a session. Choose them now and they are bound when the first session in this workspace starts.
        </div>
      )}

      {hintTool && (
        <div className="tint-warning border hairline px-3 py-2 flex flex-col gap-2" data-testid="environment-hint">
          <span><code className="font-mono">{hint!.command}</code> was not found in the last command. Bind {hintTool.label} {hintTool.tool === "python" ? "" : hintTool.versions[0]}?</span>
          <div className="flex gap-2">
            <button type="button" disabled={busy || running} className="btn btn-sm btn-steel" onClick={() => void bind(hintTool.tool, hintTool.versions[0]!)}>Bind {hintTool.label}</button>
            <button type="button" className="btn btn-sm btn-quiet" onClick={() => onHintDone?.()}>Dismiss</button>
          </div>
        </div>
      )}

      {proposals.length > 0 && (
        <ul className="m-0 p-0 list-none flex flex-col gap-2" aria-label="Toolchain proposals">
          {proposals.map((p) => (
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

      {job && job.action === "bind" && (
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
            {running && live && <button type="button" disabled={busy} className="btn btn-sm" onClick={() => void command("cancel")}>Cancel install</button>}
            {(job.status === "failed" || job.status === "cancelled") && live && job.version && (
              <button type="button" disabled={busy} className="btn btn-sm btn-steel" onClick={() => void command("bind", job.tool, job.version!)}>Retry {job.tool}</button>
            )}
          </div>
        </div>
      )}

      {pending.length > 0 && (
        <section aria-label="Chosen toolchains" className="flex flex-col gap-1.5">
          <span className="kicker">To bind in the first session</span>
          <ul className="m-0 p-0 list-none flex flex-col gap-1">
            {pending.map((p) => (
              <li key={p.tool} className="flex items-baseline justify-between gap-2" data-testid="environment-pending">
                <span className="font-mono">{p.tool}{p.tool === "python" ? "" : ` ${p.version}`}</span>
                <button type="button" className="btn btn-sm btn-quiet" onClick={() => { removePending(workspace, p.tool); setPending(pendingBinds(workspace)); }}>Remove {p.tool}</button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {live && (
        <section aria-label="Bound toolchains" className="flex flex-col gap-1.5">
          <span className="kicker">Bound</span>
          {m.toolchains.length === 0 ? (
            <span className="text-text-muted italic">None yet. {status.allowed.length ? "Bind one below or from a proposal." : "This server offers none."}</span>
          ) : (
            <ul className="m-0 p-0 list-none flex flex-col gap-1">
              {m.toolchains.map((t) => (
                <li key={t.tool} className="flex items-baseline justify-between gap-2" data-testid="environment-bound">
                  <span>
                    <span className="font-mono">{t.tool}{t.tool === "python" ? "" : ` ${t.version}`}</span>
                    {t.server && <span className="text-text-muted"> · {t.server.id}</span>}
                  </span>
                  <button type="button" disabled={busy || running} className="btn btn-sm btn-quiet" onClick={() => void command("unbind", t.tool)}>Unbind {t.tool}</button>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {addable.length > 0 && (
        <form className="flex flex-wrap items-end gap-2" onSubmit={(e) => { e.preventDefault(); if (pickedAllowed && pick.version) { void bind(pickedAllowed.tool, pick.version); setPick({ tool: "", version: "" }); } }}>
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
          <button type="submit" disabled={busy || running || !pickedAllowed} className="btn btn-sm btn-steel">{live ? "Bind" : "Choose"}</button>
        </form>
      )}

      {live && (
        <div className="flex flex-wrap items-center gap-2">
          <button type="button" disabled={busy || running} className="btn btn-sm btn-quiet" onClick={() => void command("scan")}>Rescan repositories</button>
        </div>
      )}

      {m.guidance.length > 0 && (
        <section aria-label="Repository guidance" className="flex flex-col gap-1">
          <span className="kicker">Repository guidance</span>
          <span className="text-text-muted">Sessions are pointed at: {m.guidance.map((g) => <code key={g} className="font-mono mr-1">{g}</code>)}</span>
        </section>
      )}
    </div>
  );
}
