/**
 * Workspace-first flow for remote daemons (``--workspace-root``): pick
 * or create a workspace, complete its provider configuration when the
 * daemon reports it incomplete, then open a session in it.
 */
import { useEffect, useState } from "react";
import { createWorkspace, requestWorkspaceList, selectWorkspace, updateConfig } from "@/sdk/connection";
import { useJaato } from "@/store/store";

export function WorkspaceScreen() {
  const ws = useJaato((s) => s.workspace);
  const setScreen = useJaato((s) => s.setScreen);
  const [newName, setNewName] = useState("");
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => { requestWorkspaceList().catch(() => undefined); }, []);
  useEffect(() => {
    if (ws.config) { setProvider(ws.config.provider ?? ""); setModel(ws.config.model ?? ""); }
  }, [ws.config]);

  const pick = async (name: string) => { setBusy(true); try { await selectWorkspace(name); } finally { setBusy(false); } };
  const create = async (e: React.FormEvent) => { e.preventDefault(); if (!newName.trim()) return; await createWorkspace(newName.trim()); setNewName(""); };
  const saveConfig = async (e: React.FormEvent) => { e.preventDefault(); setBusy(true); try { await updateConfig({ provider: provider || undefined, model: model || undefined, api_key: apiKey || undefined }); setApiKey(""); } finally { setBusy(false); } };

  const cfg = ws.config;
  return (
    <div className="h-full overflow-auto p-6 flex justify-center">
      <div className="w-full max-w-2xl space-y-6">
        <div>
          <div className="text-xl font-semibold">Workspaces</div>
          <div className="text-sm text-text-muted">{ws.root ? <span className="font-mono">{ws.root}</span> : "Select the workspace this session should run in."}</div>
        </div>
        <ul className="rounded-xl border hairline surface-1 divide-y divide-[color-mix(in_srgb,var(--c-muted)_35%,transparent)]">
          {ws.list.length === 0 && <li className="px-4 py-3 text-sm text-text-muted italic">No workspaces yet — create one below.</li>}
          {ws.list.map((w) => (
            <li key={w.name}>
              <button type="button" disabled={busy} onClick={() => pick(w.name)} className={`w-full text-left px-4 py-2.5 flex items-center gap-3 hover:bg-surface/60 ${ws.selected === w.name ? "bg-surface/60" : ""}`}>
                <span className={w.configured ? "text-success" : "text-warning"}>{w.configured ? "●" : "○"}</span>
                <span className="font-mono">{w.name}</span>
                <span className="text-xs text-text-muted">{w.configured ? [w.provider, w.model].filter(Boolean).join(" / ") : "not configured"}</span>
                <span className="flex-1" />
                {w.last_accessed && <span className="text-[11px] text-text-muted">{new Date(w.last_accessed).toLocaleString()}</span>}
              </button>
            </li>
          ))}
        </ul>
        <form onSubmit={create} className="flex gap-2">
          <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="new-workspace-name" className="flex-1 rounded-md border hairline bg-bg px-2 py-1.5 font-mono text-[13px] outline-none focus:border-primary/60" />
          <button type="submit" className="rounded-md px-3 py-1.5 text-sm bg-surface hover:text-primary">Create</button>
        </form>

        {cfg && (
          <div className="rounded-xl border hairline surface-1 p-4 space-y-3">
            <div className="font-semibold">{cfg.workspace} <span className="text-xs text-text-muted font-normal">{cfg.configured ? "configured" : `missing: ${cfg.missingFields.join(", ") || "provider"}`}</span></div>
            {!cfg.configured && (
              <form onSubmit={saveConfig} className="grid grid-cols-1 md:grid-cols-3 gap-2 text-sm">
                <label className="block"><span className="text-text-muted text-xs">Provider</span>
                  <select value={provider} onChange={(e) => setProvider(e.target.value)} className="mt-1 w-full rounded-md border hairline bg-bg px-2 py-1.5">
                    <option value="">—</option>
                    {cfg.availableProviders.map((p) => <option key={p} value={p}>{p}</option>)}
                  </select>
                </label>
                <label className="block"><span className="text-text-muted text-xs">Model</span><input value={model} onChange={(e) => setModel(e.target.value)} className="mt-1 w-full rounded-md border hairline bg-bg px-2 py-1.5 font-mono" /></label>
                <label className="block"><span className="text-text-muted text-xs">API key</span><input value={apiKey} onChange={(e) => setApiKey(e.target.value)} type="password" autoComplete="off" className="mt-1 w-full rounded-md border hairline bg-bg px-2 py-1.5 font-mono" /></label>
                <div className="md:col-span-3 flex justify-end"><button type="submit" disabled={busy} className="rounded-md px-3 py-1.5 bg-surface hover:text-primary">Save configuration</button></div>
              </form>
            )}
            <div className="flex justify-end">
              <button type="button" disabled={!cfg.configured} onClick={() => setScreen("session")} className="rounded-md px-4 py-1.5 bg-primary text-bg font-semibold disabled:opacity-40">Open session →</button>
            </div>
          </div>
        )}
        <div className="text-right"><button type="button" className="text-xs text-text-muted underline" onClick={() => setScreen("session")}>Skip — the daemon has a single workspace</button></div>
      </div>
    </div>
  );
}
