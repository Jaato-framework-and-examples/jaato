/**
 * Workspace-first flow for remote daemons (``--workspace-root``): pick or
 * create the workspace this session runs in, then go to the session
 * screen.
 *
 * Nothing here is gated on the workspace being "configured".  The TUI
 * opens with no provider bound and lets you sign in from the prompt
 * (``anthropic-auth login`` and friends), after which the daemon offers
 * to open the session itself -- so an unconfigured workspace is a normal
 * starting point, not a form to fill first.  The daemon's
 * ``config.update`` verb (provider / model / API key written to the
 * workspace ``.env``) stays reachable behind a disclosure for the people
 * who prefer it; the list the daemon sends for it is what it names.
 */
import { useEffect, useState } from "react";
import { exitToConnect } from "@/app/actions";
import { credentialsApi } from "@/app/credentials";
import { CredentialPicker, type KeyChoice } from "@/components/workspace/CredentialPicker";
import { createWorkspace, deleteWorkspace, requestWorkspaceList, selectWorkspace, updateConfig } from "@/sdk/connection";
import { useJaato } from "@/store/store";

export function WorkspaceScreen() {
  const ws = useJaato((s) => s.workspace);
  const setScreen = useJaato((s) => s.setScreen);
  const setWorkspaceNotice = useJaato((s) => s.setWorkspaceListNotice);
  const credentialsUrl = useJaato((s) => s.credentialsUrl);
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
    <div className="h-full overflow-auto p-6 flex justify-center">
      <div className="w-full max-w-2xl space-y-6">
        <div className="flex items-start gap-4">
          <div className="flex-1">
            <div className="text-xl font-semibold">Workspaces</div>
            <div className="text-sm text-text-muted">{ws.root ? <span className="font-mono">{ws.root}</span> : "Pick the workspace this session should run in."}</div>
          </div>
          {/* Who this list belongs to, and the way out.  "Sign out" is the backend's
              logout (it also revokes the daemon tickets); without a backend the
              only thing to leave is the connection, which the exit command does. */}
          <div className="text-xs text-text-muted text-right space-y-0.5">
            {backend?.user && <div>Signed in as <span className="font-semibold text-text">{backend.user}</span></div>}
            {backend ? (
              <a href={backend.logoutUrl} className="underline hover:text-text">Sign out</a>
            ) : (
              <button type="button" onClick={() => { exitToConnect().catch(() => undefined); }} className="underline hover:text-text">Disconnect</button>
            )}
          </div>
        </div>
        {ws.notice && <div role="status" className={`text-xs ${ws.notice.error ? "text-error" : "text-text-muted"}`}>{ws.notice.text}</div>}
        <ul className="rounded-xl border hairline surface-1 divide-y divide-[color-mix(in_srgb,var(--c-muted)_35%,transparent)]">
          {ws.list.length === 0 && <li className="px-4 py-3 text-sm text-text-muted italic">No workspaces yet — create one below.</li>}
          {ws.list.map((w) => (
            <li key={w.name} className="flex items-center">
              <button type="button" disabled={busy} onClick={() => open(w.name)} className={`flex-1 text-left px-4 py-2.5 flex items-center gap-3 hover:bg-surface/60 ${ws.selected === w.name ? "bg-surface/60" : ""}`} aria-label={`Open workspace ${w.name}`}>
                <span className={w.configured ? "text-success" : "text-warning"}>{w.configured ? "●" : "○"}</span>
                <span className="font-mono">{w.name}</span>
                <span className="text-xs text-text-muted">{w.configured ? [w.provider, w.model].filter(Boolean).join(" / ") : "no provider in .env — sign in from the prompt, or configure"}</span>
                {w.owner && <span className="text-[10px] text-text-muted border hairline rounded px-1" title={`Owned by ${w.owner}`}>{w.owner}</span>}
                <span className="flex-1" />
                {w.last_accessed && <span className="text-[11px] text-text-muted">{new Date(w.last_accessed).toLocaleString()}</span>}
              </button>
              <button type="button" disabled={busy} onClick={() => configure(w.name)} className="px-3 py-2.5 text-xs text-text-muted hover:text-primary" aria-label={`Configure workspace ${w.name}`}>configure</button>
              {confirming === w.name ? (
                <span className="flex items-center gap-1 pr-2 text-xs" role="group" aria-label={`Confirm deleting workspace ${w.name}`}>
                  <span className="text-warning">delete {w.name} and its sessions?</span>
                  <button type="button" disabled={busy} onClick={() => remove(w.name)} className="rounded px-2 py-1 bg-error/80 text-bg font-semibold" aria-label={`Confirm delete workspace ${w.name}`}>Delete</button>
                  <button type="button" onClick={() => setConfirming(null)} className="rounded px-2 py-1 border hairline" aria-label="Cancel delete">Cancel</button>
                </span>
              ) : (
                <button type="button" disabled={busy} onClick={() => setConfirming(w.name)} className="px-3 py-2.5 text-xs text-text-muted hover:text-error" aria-label={`Delete workspace ${w.name}`}>delete</button>
              )}
            </li>
          ))}
        </ul>
        <form onSubmit={create} className="flex gap-2">
          <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="new-workspace-name" className="flex-1 rounded-md border hairline bg-bg px-2 py-1.5 font-mono text-[13px] outline-none focus:border-primary/60" />
          <button type="submit" className="rounded-md px-3 py-1.5 text-sm bg-surface hover:text-primary">Create</button>
        </form>

        {cfg && manual === cfg.workspace && (
          <div className="rounded-xl border hairline surface-1 p-4 space-y-3" aria-label="Manual provider configuration">
            <div className="font-semibold">{cfg.workspace} <span className="text-xs text-text-muted font-normal">{cfg.configured ? "configured" : `missing: ${cfg.missingFields.join(", ") || "provider"}`}</span></div>
            <div className="text-xs text-text-muted">Writes <span className="font-mono">JAATO_PROVIDER</span> / <span className="font-mono">MODEL_NAME</span> (and the key) to this workspace's <span className="font-mono">.env</span>. Optional: an auth command from the prompt does the same after signing you in.</div>
            <form onSubmit={saveConfig} className="grid grid-cols-1 md:grid-cols-3 gap-2 text-sm">
              <label className="block"><span className="text-text-muted text-xs">Provider</span>
                <select value={provider} onChange={(e) => setProvider(e.target.value)} className="mt-1 w-full rounded-md border hairline bg-bg px-2 py-1.5">
                  <option value="">—</option>
                  {cfg.availableProviders.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
              </label>
              <label className="block"><span className="text-text-muted text-xs">Model</span><input value={model} onChange={(e) => setModel(e.target.value)} className="mt-1 w-full rounded-md border hairline bg-bg px-2 py-1.5 font-mono" /></label>
              <CredentialPicker credentialsUrl={credentialsUrl} provider={provider} value={keyChoice} onChange={setKeyChoice} reloadKey={keyListVersion} onError={(text) => setWorkspaceNotice({ text, error: true })} />
              <div className="md:col-span-3 flex justify-end gap-2">
                <button type="button" onClick={() => setManual(null)} className="rounded-md px-3 py-1.5 text-xs border hairline hover:bg-surface">Close</button>
                <button type="submit" disabled={busy} className="rounded-md px-3 py-1.5 bg-surface hover:text-primary">Save configuration</button>
              </div>
            </form>
            <div className="flex justify-end">
              <button type="button" onClick={() => setScreen("session")} className="rounded-md px-4 py-1.5 bg-primary text-bg font-semibold">Open session →</button>
            </div>
          </div>
        )}
        <div className="text-right"><button type="button" className="text-xs text-text-muted underline" onClick={() => setScreen("session")}>Skip — open a session without picking a workspace</button></div>
      </div>
    </div>
  );
}
