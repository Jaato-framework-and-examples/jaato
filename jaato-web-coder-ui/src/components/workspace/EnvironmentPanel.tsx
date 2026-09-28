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
 * Layout (the "status strip + tiles" design): a strip that always says what
 * the environment is doing (:func:`stripState`) with its one action, a
 * progress bar and the job's log while an install runs, ask rows for the
 * hint and the proposals, the bound toolchains as tiles with a final
 * "+ Add toolchain" tile that opens a version picker, and a footer.  A
 * finished install is "✓ just now" on its tile, not a banner.
 *
 * Nothing is installed without a click.  Every control is a real ``button``
 * whose accessible name contains its visible word, and none is hover-gated.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { environmentApi, type AllowedTool, type EnvironmentStatus, type ToolId } from "@/app/environment";
import { stageToolchainOffer, currentWorkspacePath } from "@/app/toolchainOffer";
import {
  POLL_MS, addPending, canReadManifest, pendingBinds, readManifest, removePending, toolchainCommand, waitingProposals,
  type Manifest, type PendingBind, type Proposal, type ToolchainJob,
} from "@/app/toolchains";
import { useJaato } from "@/store/store";

/** The ask-row text for a proposal. */
export function proposalText(p: Proposal): string {
  const what = p.tool === "python" ? `${p.label}` : `${p.label} ${p.version}`;
  if (p.pin && !p.pinAllowed) return `${p.label} detected (${p.source} pins ${p.pin}, which this server does not offer). Bind ${p.version} instead?`;
  return `${what} detected in ${p.source}.`;
}

export type StripTone = "running" | "failed" | "waiting" | "neutral" | "ready";

/**
 * What the strip says, first match wins: a running install, one that failed
 * or was cancelled, asks waiting on the user, not yet in a session, nothing
 * bound, ready.  A FINISHED install has no strip state: it is "✓ just now"
 * on its tile.  ``action`` is the strip's one button, shown only when live.
 */
export function stripState(a: {
  job: ToolchainJob | null; label: (tool: string) => string; waiting: number;
  live: boolean; bound: number; pending: number;
}): { tone: StripTone; glyph: string; text: string; action: "cancel" | "retry" | null } {
  const { job } = a;
  const what = job ? `${a.label(job.tool)}${job.tool === "python" || !job.version ? "" : ` ${job.version}`}` : "";
  if (job?.status === "running") {
    return { tone: "running", glyph: "↓", text: `${job.action === "unbind" ? "Unbinding" : "Installing"} ${what}…`, action: "cancel" };
  }
  if (job && (job.status === "failed" || job.status === "cancelled")) {
    const text = job.status === "cancelled" ? `Install of ${what} cancelled.` : `${what} failed: ${job.error ?? "unknown error"}`;
    return { tone: "failed", glyph: "✗", text, action: job.action === "bind" && job.version ? "retry" : null };
  }
  if (a.waiting > 0) return { tone: "waiting", glyph: "!", text: `${a.waiting} suggestion${a.waiting === 1 ? "" : "s"} waiting`, action: null };
  if (!a.live) {
    return a.pending > 0
      ? { tone: "neutral", glyph: "◷", text: `${a.pending} chosen · bound when the first session starts`, action: null }
      : { tone: "neutral", glyph: "○", text: "No toolchains bound", action: null };
  }
  if (a.bound === 0) return { tone: "neutral", glyph: "○", text: "No toolchains bound", action: null };
  return { tone: "ready", glyph: "✓", text: `${a.bound} bound · ready`, action: null };
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
  const setWaiting = useJaato((s) => s.setEnvironmentWaiting);
  const [status, setStatus] = useState<EnvironmentStatus | null>(null);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [pending, setPending] = useState<PendingBind[]>(() => pendingBinds(workspace));
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [adding, setAdding] = useState(false);
  const [picked, setPicked] = useState<Partial<Record<ToolId, string>>>({});

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

  const m = manifest ?? { toolchains: [], proposals: [], guidance: [], job: null };
  const proposals = status ? waitingProposals(m, status, pending) : [];
  // The rail badge counts what this panel would ask, for the session's own workspace.
  useEffect(() => { if (live && status && manifest) setWaiting(proposals.length); }, [live, status, manifest, proposals.length, setWaiting]);

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
  const job = m.job;
  const bound = new Set(m.toolchains.map((t) => t.tool));
  const pendingTools = new Set(pending.map((p) => p.tool));
  const allowed = (tool: string): AllowedTool | undefined => status.allowed.find((a) => a.tool === tool);
  const label = (tool: string) => allowed(tool)?.label ?? tool;
  const hintTool = hint ? status.allowed.find((a) => a.tool === hint.tool && !bound.has(a.tool)) : undefined;
  const addable = status.allowed.filter((a) => !bound.has(a.tool) && !pendingTools.has(a.tool));
  const strip = stripState({
    job, label, waiting: proposals.length + (hintTool ? 1 : 0), live, bound: m.toolchains.length, pending: pending.length,
  });
  const locked = busy || running;
  const versionOf = (tool: string, version: string) => (tool === "python" ? "" : version);

  const tiles = live
    ? m.toolchains.map((t) => ({
        key: t.tool, tool: t.tool, version: t.version, testid: "environment-bound",
        sub: t.server ? `LSP ${t.server.id}` : t.bin.join(" "),
        meta: job?.action === "bind" && job.status === "done" && job.tool === t.tool
          ? <span className="text-success">✓ just now</span> : null,
        action: <button type="button" disabled={locked} className="tc-text-btn" aria-label={`Unbind ${t.tool}`} onClick={() => void command("unbind", t.tool)}>Unbind</button>,
      }))
    : pending.map((p) => {
        const a = allowed(p.tool);
        return {
          key: p.tool, tool: p.tool, version: p.version, testid: "environment-pending",
          sub: a?.server ? `LSP ${a.server.id}` : "",
          meta: <span className="text-text-muted">pending</span>,
          action: <button type="button" className="tc-text-btn" aria-label={`Remove ${p.tool}`} onClick={() => { removePending(workspace, p.tool); setPending(pendingBinds(workspace)); }}>Remove</button>,
        };
      });

  return (
    <div className="flex flex-col text-[13px]" data-testid="environment-panel" data-live={live ? "true" : "false"}>
      <div className={`tc-strip tc-strip-${strip.tone}`} role="status" data-testid="environment-strip" data-tone={strip.tone}>
        <span className="glyph" aria-hidden="true">{strip.glyph}</span>
        <span className="min-w-0 break-words">{strip.text}</span>
        {live && strip.action === "cancel" && (
          <button type="button" disabled={busy} className="tc-strip-action" aria-label="Cancel install" onClick={() => void command("cancel")}>Cancel</button>
        )}
        {live && strip.action === "retry" && job?.version && (
          <button type="button" disabled={busy} className="tc-strip-action" aria-label={`Retry ${job.tool}`} onClick={() => void command("bind", job.tool, job.version!)}>Retry</button>
        )}
      </div>
      {running && <div className="tc-progress" aria-hidden="true"><span /></div>}

      <div className="tc-body flex flex-col gap-3.5 px-4 pt-3.5 pb-4">
        {job && job.status !== "done" && (
          <div className="flex flex-col gap-1.5" data-testid="environment-job" data-state={job.status}>
            {job.log.length > 0 && <pre className="tc-log">{job.log.slice(-8).join("\n")}</pre>}
            {job.notes.map((n) => <span key={n} className="text-warning text-[12px]">{n}</span>)}
          </div>
        )}
        {job?.status === "done" && job.notes.length > 0 && (
          <div className="flex flex-col gap-1" data-testid="environment-job" data-state="done">
            {job.notes.map((n) => <span key={n} className="text-warning text-[12px]">{n}</span>)}
          </div>
        )}

        {error && <div className="text-error text-[13px]" role="alert">{error}</div>}

        {hintTool && (
          <div className="tc-ask tc-ask-hint" data-testid="environment-hint">
            <span className="min-w-0">
              “<code className="font-mono">{hint!.command}</code>” was not found in the last command. {hintTool.label}{hintTool.tool === "python" ? "" : ` ${hintTool.versions[0]}`} provides it.
            </span>
            <span className="flex gap-2">
              <button type="button" disabled={locked} className="tc-primary" onClick={() => void bind(hintTool.tool, hintTool.versions[0]!)}>Bind {hintTool.label}</button>
              <button type="button" className="tc-x" aria-label="Dismiss" onClick={() => onHintDone?.()}>✕</button>
            </span>
          </div>
        )}

        {proposals.length > 0 && (
          <ul className="m-0 p-0 list-none flex flex-col gap-2" aria-label="Toolchain proposals">
            {proposals.map((p) => (
              <li key={p.tool} className="tc-ask tc-ask-proposal" data-testid="environment-proposal">
                <span className="min-w-0">{proposalText(p)}</span>
                <span className="flex gap-2">
                  <button type="button" disabled={locked} className="tc-primary" aria-label={`Bind ${p.label}`} onClick={() => void bind(p.tool, p.version)}>Bind</button>
                  <button type="button" disabled={busy} className="tc-x" aria-label="Not now" onClick={() => void decline(p.tool)}>✕</button>
                </span>
              </li>
            ))}
          </ul>
        )}

        {(tiles.length > 0 || addable.length > 0) && (
          <div className="tc-tiles" aria-label={live ? "Bound toolchains" : "Chosen toolchains"} role="group">
            {tiles.map((t) => (
              <div key={t.key} className="tc-tile" data-testid={t.testid}>
                <div>
                  <span className="tc-tile-label">{label(t.tool)}</span>
                  {versionOf(t.tool, t.version) && <span className="tc-tile-version">{t.version}</span>}
                </div>
                {t.sub && <div className="tc-tile-sub">{t.sub}</div>}
                <div className="tc-tile-foot">
                  <span>{t.meta}</span>
                  {t.action}
                </div>
              </div>
            ))}
            {addable.length > 0 && (
              <button type="button" className="tc-add" aria-expanded={adding} disabled={running} onClick={() => setAdding((v) => !v)}>
                <span className="tc-add-label">+ Add toolchain</span>
                <span className="text-[12px] text-text-muted">{addable.length} offered by this server</span>
              </button>
            )}
          </div>
        )}

        {adding && addable.length > 0 && (
          <div className="tc-picker" aria-label="Add a toolchain" role="group">
            {addable.map((a) => {
              const version = picked[a.tool] ?? a.versions[0] ?? "";
              return (
                <div key={a.tool} className="tc-pick-row">
                  <span className="text-[13px] font-semibold">{a.label}</span>
                  {a.tool === "python" ? <span /> : (
                    <span className="flex flex-wrap gap-1" role="radiogroup" aria-label={`${a.label} version`}>
                      {a.versions.map((v) => (
                        <button key={v} type="button" role="radio" aria-checked={v === version} className="tc-chip"
                          onClick={() => setPicked((cur) => ({ ...cur, [a.tool]: v }))}>{v}</button>
                      ))}
                    </span>
                  )}
                  <button type="button" disabled={locked || !version} className="tc-primary" aria-label={`${live ? "Bind" : "Choose"} ${a.label}`}
                    onClick={() => { void bind(a.tool, version); setAdding(false); }}>{live ? "Bind" : "Choose"}</button>
                </div>
              );
            })}
          </div>
        )}

        <div className="flex items-baseline justify-between gap-3 border-t hairline pt-2.5">
          <span className="text-[12px] text-text-muted min-w-0">
            {status.allowed.length === 0 ? "This server offers none."
              : m.guidance.length > 0 ? <>Sessions read {m.guidance.map((g, i) => <span key={g}>{i ? ", " : ""}<code className="font-mono">{g}</code></span>)}</>
              : !live ? <span role="note">Toolchains install inside a session. Choose them now and they are bound when the first session in this workspace starts.</span>
              : null}
          </span>
          {live && (
            <button type="button" disabled={locked} className="tc-rescan" aria-label="Rescan repositories" onClick={() => void command("scan")}>↻ Rescan repos</button>
          )}
        </div>
      </div>
    </div>
  );
}
