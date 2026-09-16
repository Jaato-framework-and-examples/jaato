/**
 * The card that answers the daemon's ``auth.setup`` offer.
 *
 * Shown after a daemon-level auth command (``<provider>-auth login``)
 * succeeded: the daemon knows the provider and its default models and
 * asks three things the TUI asks one line at a time -- open a session
 * with it (or switch the current one), which model, and whether to
 * persist ``JAATO_PROVIDER`` / ``MODEL_NAME`` to the workspace ``.env``
 * so the next session needs no setup.  One card answers all three.
 */
import { useState } from "react";
import type { PendingPostAuthSetup } from "@/store/types";

export function PostAuthSetupPrompt({ p, onRespond }: {
  p: PendingPostAuthSetup;
  onRespond: (a: { connect: boolean; modelName?: string; persistEnv?: boolean }) => void;
}) {
  const [model, setModel] = useState(p.models[0]?.name ?? "");
  const [persist, setPersist] = useState(Boolean(p.workspacePath));
  const canConnect = model.trim().length > 0;
  const heading = p.hasActiveSession
    ? `Signed in to ${p.providerDisplayName}. Switch this session to it? (currently ${[p.currentProvider, p.currentModel].filter(Boolean).join(" / ")})`
    : `Signed in to ${p.providerDisplayName}. Open a session with it?`;

  return (
    <div className="my-2 rounded-lg border border-primary/50 surface-1 overflow-hidden" role="group" aria-label="Post-auth setup">
      <div className="px-3 py-1.5 border-b hairline text-sm">{heading}</div>
      <form
        className="px-3 py-2 space-y-2 text-sm"
        onSubmit={(e) => { e.preventDefault(); if (canConnect) onRespond({ connect: true, modelName: model.trim(), persistEnv: persist }); }}
      >
        <label className="block">
          <span className="text-xs text-text-muted">Model</span>
          {p.models.length > 0 ? (
            <select aria-label="Model" value={model} onChange={(e) => setModel(e.target.value)} className="mt-1 w-full rounded-md border hairline bg-bg px-2 py-1.5 font-mono">
              {p.models.map((m) => <option key={m.name} value={m.name}>{m.name}{m.description ? ` — ${m.description}` : ""}</option>)}
              <option value="">other…</option>
            </select>
          ) : null}
          {(p.models.length === 0 || model === "") && (
            <input aria-label="Model name" value={model} onChange={(e) => setModel(e.target.value)} placeholder="model name" className="mt-1 w-full rounded-md border hairline bg-bg px-2 py-1.5 font-mono" />
          )}
        </label>
        {p.workspacePath && (
          <label className="flex items-center gap-2 text-xs text-text-muted">
            <input type="checkbox" checked={persist} onChange={(e) => setPersist(e.target.checked)} />
            Save provider and model to <span className="font-mono">{p.workspacePath}/.env</span>
          </label>
        )}
        <div className="flex justify-end gap-2">
          <button type="button" onClick={() => onRespond({ connect: false })} className="rounded-md px-3 py-1.5 text-xs border hairline hover:bg-surface">Not now</button>
          <button type="submit" disabled={!canConnect} className="rounded-md px-3 py-1.5 text-xs bg-primary text-bg font-semibold disabled:opacity-40">{p.hasActiveSession ? "Switch" : "Open session"}</button>
        </div>
      </form>
    </div>
  );
}
