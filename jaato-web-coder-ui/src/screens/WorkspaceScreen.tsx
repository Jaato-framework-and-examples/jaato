/**
 * Workspace-first flow for remote daemons (``--workspace-root``): pick or
 * create the workspace this session runs in, then go to the session
 * screen.
 *
 * Drawn as a table, not a stack of buttons (design frame 02): one row per
 * workspace with its provider and model, owner, last-opened time and the
 * three actions -- Open, Configure, Delete -- so "which workspaces exist"
 * is answered by scanning columns.  The create row and the way out of the
 * flow sit under it.
 *
 * Nothing here is gated on the workspace being "configured".  The TUI
 * opens with no provider bound and lets you sign in from the prompt
 * (``anthropic-auth login`` and friends), after which the daemon offers
 * to open the session itself -- so an unconfigured workspace is a normal
 * starting point, not a form to fill first.  The daemon's
 * ``config.update`` verb (provider / model / API key written to the
 * workspace ``.env``) stays reachable behind ``Configure`` for the people
 * who prefer it, as a plate under the table; the list the daemon sends
 * for it is what it names.
 */
import { useEffect, useState } from "react";
import { exitToConnect } from "@/app/actions";
import { credentialsApi } from "@/app/credentials";
import { Plate } from "@/components/layout/Plate";
import { CredentialPicker, type KeyChoice } from "@/components/workspace/CredentialPicker";
import { GitHubAccountPicker } from "@/components/workspace/GitHubAccountPicker";
import { GitHubConnect } from "@/components/workspace/GitHubConnect";
import { createWorkspace, deleteWorkspace, requestWorkspaceList, selectWorkspace, updateConfig } from "@/sdk/connection";
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

export function WorkspaceScreen() {
  const ws = useJaato((s) => s.workspace);
  const setScreen = useJaato((s) => s.setScreen);
  const setWorkspaceNotice = useJaato((s) => s.setWorkspaceListNotice);
  const credentialsUrl = useJaato((s) => s.credentialsUrl);
  const githubUrl = useJaato((s) => s.githubUrl);
  const githubLoginUrl = useJaato((s) => s.githubLoginUrl);
  const backend = useJaato((s) => s.backend);
  const [newName, setNewName] = useState("");
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  // The key to apply: a stored entry (revealed at save), a new one (stored
  // at save when a store exists, then applied), or none.
  const [keyChoice, setKeyChoice] = useState<KeyChoice>({ kind: "none" });
  const [keyListVersion, setKeyListVersion] = useState(0); // bumped when a key was just stored, so the picker relists
  const [busy, setBusy] = useState(false);
  const [manual, setManual] = useState<string | null>(null); // workspace whose manual form is open
  const [confirming, setConfirming] = useState<string | null>(null); // workspace whose delete is awaiting confirmation
  const [showGithub, setShowGithub] = useState(false); // the "Connect GitHub" settings plate
  const [githubReload, setGithubReload] = useState(0); // bumped when the account set changed, so the picker relists

  useEffect(() => { requestWorkspaceList().catch(() => undefined); }, []);
  useEffect(() => {
    if (ws.config) { setProvider(ws.config.provider ?? ""); setModel(ws.config.model ?? ""); }
  }, [ws.config]);

  // Select and go: the daemon answers with ``config.status``, which the
  // session screen's status bar reflects; the choice of provider happens
  // there, from a profile, the workspace .env, or an auth command.
  const open = async (name: string) => {
    setBusy(true);
    try { await selectWorkspace(name); setScreen("session"); } finally { setBusy(false); }
  };
  const configure = async (name: string) => {
    setBusy(true);
    try { await selectWorkspace(name); setManual(name); } finally { setBusy(false); }
  };
  // Destructive and daemon-side (the directory and its sessions go), so the
  // row asks first; the daemon's answer lands in ``ws.notice``.
  const remove = async (name: string) => {
    setBusy(true);
    try { await deleteWorkspace(name); } finally { setBusy(false); setConfirming(null); }
  };
  const create = async (e: React.FormEvent) => { e.preventDefault(); if (!newName.trim()) return; await createWorkspace(newName.trim()); setNewName(""); };
  // The key reaches the daemon the one way it always has -- as ``api_key``
  // on ``config.update`` -- whether it was typed now or picked from the
  // store; the daemon knows nothing of the store.
  const resolveApiKey = async (): Promise<{ apiKey?: string; storedId?: string }> => {
    const api = credentialsUrl ? credentialsApi(credentialsUrl) : null;
    if (keyChoice.kind === "stored") return { apiKey: api ? await api.reveal(keyChoice.id) : undefined };
    if (keyChoice.kind === "new" && keyChoice.secret.trim()) {
      const secret = keyChoice.secret.trim();
      let storedId: string | undefined;
      if (api && provider) {
        // Remembered for the next workspace; a failure to remember must not stop this one.
        try { storedId = (await api.add(provider, secret, keyChoice.label.trim() || undefined)).id; }
        catch (err) { setWorkspaceNotice({ text: `Key applied, but not stored for later: ${err instanceof Error ? err.message : String(err)}`, error: true }); }
      }
      return { apiKey: secret, storedId };
    }
    return {};
  };
  const saveConfig = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      const { apiKey, storedId } = await resolveApiKey();
      await updateConfig({ provider: provider || undefined, model: model || undefined, api_key: apiKey });
      if (keyChoice.kind === "new") {
        // The typed key is now a stored one: the picker relists and selects it.
        setKeyChoice(storedId ? { kind: "stored", id: storedId } : { kind: "none" });
        setKeyListVersion((v) => v + 1);
      }
    } catch (err) {
      setWorkspaceNotice({ text: err instanceof Error ? err.message : String(err), error: true });
    } finally { setBusy(false); }
  };

  const cfg = ws.config;
  return (
    <div className="h-full overflow-auto p-6 sm:p-10 flex justify-center">
      <div className="w-full max-w-[880px] flex flex-col gap-5">
        <header className="flex items-end justify-between gap-4 border-b hairline pb-2.5">
          <div>
            <div className="kicker tracking-[0.16em]">{ws.root ? "Workspace root" : "Pick a workspace"}</div>
            <h1 className="display text-[30px] m-0">Workspaces</h1>
          </div>
          {/* Where the list comes from, who it belongs to, and the way out.
              "Sign out" is the backend's logout (it also revokes the daemon
              tickets); without a backend the only thing to leave is the
              connection, which the exit command does. */}
          <div className="text-right space-y-0.5">
            <div className="font-mono text-xs text-text-muted">{ws.root ?? "workspace mode"} · {ws.list.length} {ws.list.length === 1 ? "entry" : "entries"}</div>
            <div className="text-xs text-text-muted">
              {backend?.user && <>Signed in as <span className="font-semibold text-text">{backend.user}</span> · </>}
              {/* Per-user GitHub connection: present only when the backend has a
                  ``github:`` block (``githubUrl`` set). Opens the settings plate below. */}
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
                <th className={TH}>Provider · model</th>
                <th className={TH}>Owner</th>
                <th className={TH}>Last opened</th>
                <th className={`${TH} text-right`}>Actions</th>
              </tr>
            </thead>
            <tbody className="font-mono text-[13px]">
              {ws.list.length === 0 && (
                <tr><td colSpan={5} className="px-3.5 py-3 font-sans text-sm text-text-muted italic">No workspaces yet — create one below.</td></tr>
              )}
              {ws.list.map((w, i) => {
                const last = i === ws.list.length - 1;
                const cell = last ? TD.replace("border-b hairline", "") : TD;
                return (
                  <tr key={w.name} className={ws.selected === w.name ? "tint" : "hover:bg-tint/60"}>
                    <td className={`${cell} whitespace-nowrap`}>
                      <button type="button" disabled={busy} onClick={() => open(w.name)} className="text-left hover:text-steel">
                        <span className={`mr-2 ${w.configured ? "text-success" : "text-warning"}`}>{w.configured ? "●" : "○"}</span>{w.name}
                      </button>
                    </td>
                    <td className={`${cell} ${w.configured ? "" : "text-text-muted"}`}>
                      {w.configured ? [w.provider, w.model].filter(Boolean).join(" / ") : <>no provider in <span className="text-text">.env</span> — sign in from the prompt, or configure</>}
                    </td>
                    <td className={`${cell} text-text-muted`}>{w.owner ? <span title={`Owned by ${w.owner}`}>{w.owner}</span> : "—"}</td>
                    <td className={`${cell} text-text-muted whitespace-nowrap`}>{when(w.last_accessed)}</td>
                    <td className={`${cell} text-right whitespace-nowrap`}>
                      {confirming === w.name ? (
                        <span className="inline-flex items-center gap-2 font-sans text-xs" role="group" aria-label={`Confirm deleting workspace ${w.name}`}>
                          <span className="text-warning">delete {w.name} and its sessions?</span>
                          <button type="button" disabled={busy} onClick={() => remove(w.name)} className="btn btn-sm btn-danger" aria-label={`Confirm delete workspace ${w.name}`}>Delete</button>
                          <button type="button" onClick={() => setConfirming(null)} className="btn btn-sm" aria-label="Cancel delete">Cancel</button>
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-2">
                          <button type="button" disabled={busy} onClick={() => open(w.name)} className={`btn btn-sm ${ws.selected === w.name ? "btn-steel" : "text-steel"}`} aria-label={`Open workspace ${w.name}`}>Open</button>
                          <button type="button" disabled={busy} onClick={() => configure(w.name)} className="btn btn-sm btn-quiet border-transparent" aria-label={`Configure workspace ${w.name}`}>Configure</button>
                          <button type="button" disabled={busy} onClick={() => setConfirming(w.name)} className="btn btn-sm btn-quiet border-transparent hover:text-error" aria-label={`Delete workspace ${w.name}`}>Delete</button>
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Plate>

        <form onSubmit={create} className="flex items-center gap-3">
          <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="new-workspace-name" className="input input-mono flex-1" aria-label="New workspace name" />
          <button type="submit" className="btn">Create</button>
          <span className="flex-1" />
          <button type="button" className="link text-[13px]" onClick={() => setScreen("session")} title="The daemon provisions a fresh workspace for the session under its workspace root">Open a session in a server-provisioned workspace</button>
        </form>

        {cfg && manual === cfg.workspace && (
          <Plate role="group" aria-label="Manual provider configuration" className="flex flex-col">
            <div className="flex items-baseline gap-3 px-5 py-3.5 border-b hairline">
              <span className="kicker tracking-[0.16em]">Configure</span>
              <span className="font-mono text-[16px]">{cfg.workspace}</span>
              <span className="flex-1" />
              <span className="font-mono text-xs text-text-muted">{cfg.configured ? "configured" : `missing: ${cfg.missingFields.join(", ") || "provider"}`}</span>
            </div>
            <form onSubmit={saveConfig} className="p-5 flex flex-col gap-4">
              <div className="text-xs text-text-muted">Writes <span className="font-mono">JAATO_PROVIDER</span> / <span className="font-mono">MODEL_NAME</span> (and the key) to this workspace's <span className="font-mono">.env</span>. Optional: an auth command from the prompt does the same after signing you in.</div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
                <label className="block"><span className="field-label">Provider</span>
                  <select value={provider} onChange={(e) => setProvider(e.target.value)} className="input">
                    <option value="">—</option>
                    {cfg.availableProviders.map((p) => <option key={p} value={p}>{p}</option>)}
                  </select>
                </label>
                <label className="block"><span className="field-label">Model</span><input value={model} onChange={(e) => setModel(e.target.value)} className="input input-mono" /></label>
                <CredentialPicker credentialsUrl={credentialsUrl} provider={provider} value={keyChoice} onChange={setKeyChoice} reloadKey={keyListVersion} onError={(text) => setWorkspaceNotice({ text, error: true })} />
                {/* Binds a connected GitHub account to this workspace (BFF state,
                    not config.update); absent unless the backend has a github: block. */}
                {githubUrl && (
                  <GitHubAccountPicker
                    githubUrl={githubUrl}
                    workspace={cfg.workspace}
                    reloadKey={githubReload}
                    onError={(text) => setWorkspaceNotice({ text, error: true })}
                    onNotice={(text) => setWorkspaceNotice({ text })}
                  />
                )}
              </div>
              <div className="flex items-center gap-2 border-t hairline pt-4">
                <button type="submit" disabled={busy} className="btn btn-steel">Save configuration</button>
                <button type="button" onClick={() => setManual(null)} className="btn btn-quiet">Close</button>
                <span className="flex-1" />
                <button type="button" onClick={() => setScreen("session")} className="btn btn-primary">Open session →</button>
              </div>
            </form>
          </Plate>
        )}
      </div>
    </div>
  );
}
