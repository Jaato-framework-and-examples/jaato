/**
 * Workspace-first flow for remote daemons (``--workspace-root``): pick or
 * create the workspace a session runs in, then go to the session picker.
 *
 * Drawn as a table (design 3a): one row per workspace with its SOURCES --
 * the git checkouts the daemon finds in it (protocol 1.27) -- when it was
 * last opened, and three actions: Open, Sources, Delete.  A workspace no
 * longer has a provider and model of its own (a new session gets its
 * binding from a base profile, chosen in the picker), and the table shows
 * no session counts -- those are the picker's business.
 *
 * Delete is an inline impact panel (``components/workspace/DeletePanel``)
 * directly under its row: what is lost -- sessions by state, repositories
 * with uncommitted or unpushed work, size on disk -- and a typed-name
 * confirmation whenever anything is.  Only one is open at a time.
 *
 * New workspace (design 3b, ``NewWorkspacePlate``) creates it, binds the
 * user's GitHub account, and clones the picked repositories before the
 * session picker opens.  ``Sources`` opens a plate for an existing
 * workspace: its checkouts, the bound GitHub account, cloning more
 * repositories, and -- kept, because keys still live in the workspace
 * ``.env`` -- the provider credential form.
 */
import { Fragment, useCallback, useEffect, useRef, useState } from "react";
import { exitToConnect } from "@/app/actions";
import { resolveKeyChoice } from "@/app/sessionKey";
import { Plate } from "@/components/layout/Plate";
import { CredentialPicker, type KeyChoice } from "@/components/workspace/CredentialPicker";
import { GitHubAccountPicker } from "@/components/workspace/GitHubAccountPicker";
import { GitHubConnect } from "@/components/workspace/GitHubConnect";
import { deleteWorkspace, requestWorkspaceList, selectWorkspace, updateConfig } from "@/sdk/connection";
import { DeletePanel } from "@/components/workspace/DeletePanel";
import { CloneProgress, NewWorkspacePlate, queuedRows, startClone } from "@/components/workspace/NewWorkspacePlate";
import { RepoPicker, type PickedRepo } from "@/components/workspace/RepoPicker";
import { formatSource, normalizeSources, type CloneRow } from "@/protocol/workspaces";
import type { WorkspaceInfo } from "@/store/types";
import { useJaato } from "@/store/store";

const TH = "px-3.5 py-2 text-left font-medium border-b hairline kicker kicker-muted";
const TD = "px-3.5 py-2.5 border-b hairline align-middle";

function when(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  const time = d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  return sameDay ? `${time} today` : `${d.toLocaleDateString(undefined, { day: "numeric", month: "short" })} ${time}`;
}

/**
 * The provider credential form -- ``config.update`` writing
 * ``JAATO_PROVIDER`` / ``MODEL_NAME`` / the key to the workspace ``.env``.
 * Collapsed under Sources: the picker chooses a session's model now, but an
 * API key still has to live somewhere, and this is where it did.
 */
function CredentialsForm({ workspace }: { workspace: string }) {
  const cfg = useJaato((s) => (s.workspace.config?.workspace === workspace ? s.workspace.config : undefined));
  const credentialsUrl = useJaato((s) => s.credentialsUrl);
  const setWorkspaceNotice = useJaato((s) => s.setWorkspaceListNotice);
  const [provider, setProvider] = useState(cfg?.provider ?? "");
  const [model, setModel] = useState(cfg?.model ?? "");
  const [keyChoice, setKeyChoice] = useState<KeyChoice>({ kind: "none" });
  const [keyListVersion, setKeyListVersion] = useState(0);
  const [busy, setBusy] = useState(false);
  useEffect(() => { if (cfg) { setProvider(cfg.provider ?? ""); setModel(cfg.model ?? ""); } }, [cfg]);
  if (!cfg) return <div className="text-[13px] text-text-muted">Loading the workspace configuration…</div>;
  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      // The key reaches the daemon the one way it always has -- as ``api_key``
      // on ``config.update`` -- whether typed now or picked from the store.
      const { apiKey, storedId } = await resolveKeyChoice(credentialsUrl, provider, keyChoice, (text) => setWorkspaceNotice({ text, error: true }));
      await updateConfig({ provider: provider || undefined, model: model || undefined, api_key: apiKey });
      if (keyChoice.kind === "new") {
        setKeyChoice(storedId ? { kind: "stored", id: storedId } : { kind: "none" });
        setKeyListVersion((v) => v + 1);
      }
    } catch (err) {
      setWorkspaceNotice({ text: err instanceof Error ? err.message : String(err), error: true });
    } finally { setBusy(false); }
  };
  return (
    <form onSubmit={save} className="flex flex-col gap-3" aria-label="Provider credentials">
      <div className="text-xs text-text-muted">Writes <span className="font-mono">JAATO_PROVIDER</span> / <span className="font-mono">MODEL_NAME</span> and the key to this workspace's <span className="font-mono">.env</span>. The session picker chooses each session's model; this is where a provider key lives.</div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
        <label className="block"><span className="field-label">Provider</span>
          <select value={provider} onChange={(e) => setProvider(e.target.value)} className="input">
            <option value="">—</option>
            {cfg.availableProviders.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </label>
        <label className="block"><span className="field-label">Model</span><input value={model} onChange={(e) => setModel(e.target.value)} className="input input-mono" /></label>
        <CredentialPicker credentialsUrl={credentialsUrl} provider={provider} value={keyChoice} onChange={setKeyChoice} reloadKey={keyListVersion} onError={(text) => setWorkspaceNotice({ text, error: true })} />
      </div>
      <div><button type="submit" disabled={busy} className="btn btn-steel">Save to .env</button></div>
    </form>
  );
}

/**
 * The Sources plate for an existing workspace: the checkouts the daemon
 * found, the GitHub account bound to it, cloning more repositories into it,
 * and the provider credential form.
 */
function SourcesPlate({ w, onClose }: { w: WorkspaceInfo; onClose: () => void }) {
  const githubUrl = useJaato((s) => s.githubUrl);
  const setWorkspaceNotice = useJaato((s) => s.setWorkspaceListNotice);
  const [picked, setPicked] = useState<PickedRepo[]>([]);
  const [rows, setRows] = useState<CloneRow[]>([]);
  const [showKeys, setShowKeys] = useState(false);
  const unsubs = useRef<Array<() => void>>([]);
  useEffect(() => () => { for (const u of unsubs.current) u(); }, []);
  const sources = normalizeSources(w.sources);
  const cloning = rows.length > 0;
  const done = cloning && rows.every((r) => r.state === "done" || r.state === "failed");
  useEffect(() => { if (done) requestWorkspaceList().catch(() => undefined); }, [done]);
  const clone = () => {
    const batch = queuedRows(picked);
    setRows(batch);
    setPicked([]);
    unsubs.current.push(startClone(w.name, batch, setRows));
  };
  const retry = (repo: string) => {
    const row = rows.find((r) => r.repo === repo);
    if (!row) return;
    const again: CloneRow = { ...row, state: "queued", percent: 0, error: "" };
    setRows((cur) => cur.map((r) => (r.repo === repo ? again : r)));
    unsubs.current.push(startClone(w.name, [again], setRows));
  };
  return (
    <Plate role="group" aria-label={`Sources of ${w.name}`} className="flex flex-col">
      <div className="flex items-baseline gap-3 px-5 py-3.5 border-b hairline">
        <span className="kicker tracking-[0.16em]">Sources</span>
        <span className="font-mono text-[16px]">{w.name}</span>
        <span className="flex-1" />
        <button type="button" onClick={onClose} className="btn btn-sm btn-quiet">Close</button>
      </div>
      <div className="px-5 py-4 flex flex-col gap-4">
        <div className="flex flex-col gap-1">
          <span className="text-[12px] text-text-muted">In the workspace now</span>
          {sources.length === 0 ? <span className="font-mono text-[13px] text-text-muted">— empty</span>
            : sources.map((src) => <span key={src.path} className="font-mono text-[13px]">{formatSource(src)} <span className="text-text-muted">→ {src.path === "." ? w.name : `${w.name}/${src.path}`}</span></span>)}
        </div>
        {githubUrl && (
          <GitHubAccountPicker
            githubUrl={githubUrl}
            workspace={w.path ?? w.name}
            reloadKey={0}
            onError={(text) => setWorkspaceNotice({ text, error: true })}
            onNotice={(text) => setWorkspaceNotice({ text })}
          />
        )}
      </div>
      <div className="border-t hairline">
        {cloning ? (
          <CloneProgress
            name={w.name}
            rows={rows}
            onRetry={retry}
            onRemove={(repo) => setRows((cur) => cur.filter((r) => r.repo !== repo))}
            busy={false}
            title={{ working: "Cloning into", ready: "Cloned into" }}
          />
        ) : (
          <>
            <RepoPicker githubUrl={githubUrl} workspace={w.name} picked={picked} onChange={setPicked} />
            <div className="px-5 py-3 border-t hairline">
              <button type="button" disabled={picked.length === 0} onClick={clone} className="btn btn-primary gap-6">Clone {picked.length || ""} {picked.length === 1 ? "repo" : "repos"} <span aria-hidden="true">→</span></button>
            </div>
          </>
        )}
        {done && <div className="px-5 py-2 border-t hairline"><button type="button" onClick={() => setRows([])} className="link text-[13px]">Clone more repositories</button></div>}
      </div>
      <div className="px-5 py-3 border-t hairline flex flex-col gap-3">
        <button type="button" onClick={() => setShowKeys((v) => !v)} aria-expanded={showKeys} className="self-start chrome chrome-sm text-steel">
          <span aria-hidden="true">{showKeys ? "▾" : "▸"}</span> Provider credentials (.env)
        </button>
        {showKeys && <CredentialsForm workspace={w.name} />}
      </div>
    </Plate>
  );
}

export function WorkspaceScreen() {
  const ws = useJaato((s) => s.workspace);
  const setScreen = useJaato((s) => s.setScreen);
  const setWorkspaceNotice = useJaato((s) => s.setWorkspaceListNotice);
  const githubUrl = useJaato((s) => s.githubUrl);
  const githubLoginUrl = useJaato((s) => s.githubLoginUrl);
  const backend = useJaato((s) => s.backend);
  const [busy, setBusy] = useState(false);
  const [sourcesOf, setSourcesOf] = useState<string | null>(null);
  const [confirmingDelete, setConfirmingDelete] = useState<string | null>(null);
  const [showGithub, setShowGithub] = useState(false);
  const [githubReload, setGithubReload] = useState(0);

  useEffect(() => { requestWorkspaceList().catch(() => undefined); }, []);

  const open = useCallback(async (name: string) => {
    setBusy(true);
    try { await selectWorkspace(name); setScreen("session"); } finally { setBusy(false); }
  }, [setScreen]);
  const showSources = async (name: string) => {
    if (sourcesOf === name) { setSourcesOf(null); return; }
    setBusy(true);
    // Selecting loads the workspace's config.status, which the credential form reads.
    try { await selectWorkspace(name); setSourcesOf(name); } finally { setBusy(false); }
  };
  const remove = async (name: string) => {
    setBusy(true);
    try {
      const answer = await deleteWorkspace(name, { stopSessions: true });
      if (!answer) setWorkspaceNotice({ text: `The daemon did not confirm deleting ${name}.`, error: true });
      else if (answer.ok) { setConfirmingDelete(null); if (sourcesOf === name) setSourcesOf(null); }
    } finally { setBusy(false); }
  };

  const COLS = 4;
  const sourcesRow = sourcesOf ? ws.list.find((w) => w.name === sourcesOf) : undefined;
  return (
    <div className="h-full overflow-auto p-6 sm:p-10 flex justify-center">
      <div className="w-full max-w-[1100px] flex flex-col gap-5 [&>*]:shrink-0">
        <header className="flex items-end justify-between gap-4 border-b hairline pb-2.5">
          <div>
            <div className="kicker tracking-[0.16em]">{ws.root ? "Workspace root" : "Pick a workspace"}</div>
            <h1 className="display text-[30px] m-0">Workspaces</h1>
          </div>
          <div className="text-right space-y-0.5">
            <div className="font-mono text-xs text-text-muted">{ws.root ?? "workspace mode"} · {ws.list.length} {ws.list.length === 1 ? "entry" : "entries"}</div>
            <div className="text-xs text-text-muted">
              {backend?.user && <>Signed in as <span className="font-semibold text-text">{backend.user}</span> · </>}
              {githubUrl && <><button type="button" onClick={() => setShowGithub((v) => !v)} className="link" aria-expanded={showGithub}>Connect GitHub</button> · </>}
              {backend ? (
                <a href={backend.logoutUrl} className="link">Sign out</a>
              ) : (
                <button type="button" onClick={() => { exitToConnect().catch(() => undefined); }} className="link">Disconnect</button>
              )}
            </div>
          </div>
        </header>
        {ws.notice && <div role="status" className={`text-xs ${ws.notice.error ? "text-error" : "text-text-muted"}`}>{ws.notice.text}</div>}

        {showGithub && githubUrl && (
          <Plate>
            <GitHubConnect
              githubUrl={githubUrl}
              githubLoginUrl={githubLoginUrl}
              onError={(text) => setWorkspaceNotice({ text, error: true })}
              onNotice={(text) => setWorkspaceNotice({ text })}
              onChanged={() => setGithubReload((v) => v + 1)}
            />
          </Plate>
        )}

        <Plate className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr>
                <th className={TH}>Workspace</th>
                <th className={TH}>Sources</th>
                <th className={TH}>Last opened</th>
                <th className={`${TH} text-right`}>Actions</th>
              </tr>
            </thead>
            <tbody className="font-mono text-[13px]">
              {ws.list.length === 0 && (
                <tr><td colSpan={COLS} className="px-3.5 py-3 font-sans text-sm text-text-muted italic">No workspaces yet — create one below.</td></tr>
              )}
              {ws.list.map((w) => {
                const deleting = confirmingDelete === w.name;
                const sources = normalizeSources(w.sources);
                return (
                  <Fragment key={w.name}>
                    <tr className={deleting ? "tint-error" : ws.selected === w.name ? "tint" : "hover:bg-tint/60"} data-workspace={w.name}>
                      <td className={`${TD} align-top whitespace-nowrap`}>
                        <button type="button" disabled={busy} onClick={() => open(w.name)} className="text-left hover:text-steel" title={w.owner ? `Owned by ${w.owner}` : undefined}>{w.name}</button>
                      </td>
                      <td className={`${TD} align-top`}>
                        {sources.length === 0
                          ? <span className="text-text-muted">— empty</span>
                          : sources.map((src) => <div key={src.path} className="whitespace-nowrap">{formatSource(src)}</div>)}
                      </td>
                      <td className={`${TD} align-top text-text-muted whitespace-nowrap`}>{when(w.last_accessed)}</td>
                      <td className={`${TD} align-top text-right whitespace-nowrap`}>
                        <span className="inline-flex items-center gap-2">
                          <button type="button" disabled={busy} onClick={() => open(w.name)} className="btn btn-sm btn-steel" aria-label={`Open workspace ${w.name}`}>Open</button>
                          <button type="button" disabled={busy} onClick={() => { void showSources(w.name); }} aria-expanded={sourcesOf === w.name} className="btn btn-sm btn-quiet border-transparent" aria-label={`Sources of workspace ${w.name}`}>Sources</button>
                          <button
                            type="button"
                            disabled={busy}
                            onClick={() => setConfirmingDelete(deleting ? null : w.name)}
                            aria-expanded={deleting}
                            className={`btn btn-sm ${deleting ? "btn-danger border-error" : "btn-quiet border-transparent hover:text-error"}`}
                            aria-label={`Delete workspace ${w.name}`}
                          >
                            Delete
                          </button>
                        </span>
                      </td>
                    </tr>
                    {deleting && (
                      <tr className="tint-error">
                        <td colSpan={COLS} className="p-0 border-x border-error border-b-2 border-b-error">
                          <DeletePanel name={w.name} path={w.path} busy={busy} onDelete={() => { void remove(w.name); }} onCancel={() => setConfirmingDelete(null)} />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </Plate>

        {sourcesRow && <SourcesPlate key={sourcesRow.name} w={sourcesRow} onClose={() => setSourcesOf(null)} />}

        <NewWorkspacePlate
          key={githubReload}
          githubUrl={githubUrl}
          onCreated={() => { requestWorkspaceList().catch(() => undefined); }}
          onOpen={(name) => { void open(name); }}
          onNotice={(text, error) => setWorkspaceNotice({ text, error })}
        />
        <div className="flex">
          <span className="flex-1" />
          <button type="button" className="link text-[13px]" onClick={() => setScreen("session")} title="The daemon provisions a fresh workspace for the session under its workspace root">Open a session in a server-provisioned workspace</button>
        </div>
      </div>
    </div>
  );
}
