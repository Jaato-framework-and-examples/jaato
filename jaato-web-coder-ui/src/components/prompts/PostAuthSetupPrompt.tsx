/**
 * The plate that answers the daemon's ``auth.setup`` offer.
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
import { Plate } from "@/components/layout/Plate";

export function PostAuthSetupPrompt({ p, onRespond, alreadyConfigured = false }: {
  p: PendingPostAuthSetup;
  onRespond: (a: { connect: boolean; modelName?: string; persistEnv?: boolean }) => void;
  /**
   * The selected workspace's ``.env`` already binds this provider: nothing
   * to persist, so the question is not asked (a reopened workspace must not
   * be asked about its own ``.env`` again).
   */
  alreadyConfigured?: boolean;
}) {
  const [model, setModel] = useState(p.models[0]?.name ?? "");
  const [persist, setPersist] = useState(Boolean(p.workspacePath) && !alreadyConfigured);
  const canConnect = model.trim().length > 0;
  const heading = p.hasActiveSession
    ? `Signed in to ${p.providerDisplayName}. Switch this session to it? (currently ${[p.currentProvider, p.currentModel].filter(Boolean).join(" / ")})`
    : `Signed in to ${p.providerDisplayName}. Open a session with it?`;

  return (
    <Plate edge="steel" className="my-3" role="group" aria-label="Post-auth setup">
      <div className="px-3.5 py-2.5 border-b hairline flex items-baseline gap-3">
        <span className="kicker">Signed in</span>
        <span className="text-[15px]">{heading}</span>
      </div>
      <form
        className="px-3.5 py-3 space-y-3 text-sm"
        onSubmit={(e) => { e.preventDefault(); if (canConnect) onRespond({ connect: true, modelName: model.trim(), persistEnv: persist }); }}
      >
        <label className="block max-w-md">
          <span className="field-label">Model</span>
          {p.models.length > 0 ? (
            <select aria-label="Model" value={model} onChange={(e) => setModel(e.target.value)} className="input input-mono">
              {p.models.map((m) => <option key={m.name} value={m.name}>{m.name}{m.description ? ` — ${m.description}` : ""}</option>)}
              <option value="">other…</option>
            </select>
          ) : null}
          {(p.models.length === 0 || model === "") && (
            <input aria-label="Model name" value={model} onChange={(e) => setModel(e.target.value)} placeholder="model name" className="input input-mono mt-1" />
          )}
        </label>
        {p.workspacePath && !alreadyConfigured && (
          <label className="flex items-center gap-2 text-xs text-text-muted">
            <input type="checkbox" checked={persist} onChange={(e) => setPersist(e.target.checked)} />
            Save provider and model to <span className="font-mono">{p.workspacePath}/.env</span>
          </label>
        )}
        <div className="flex gap-2 pt-1">
          <button type="submit" disabled={!canConnect} className="btn btn-primary">{p.hasActiveSession ? "Switch" : "Open session"}</button>
          <button type="button" onClick={() => onRespond({ connect: false })} className="btn btn-quiet">Not now</button>
        </div>
      </form>
    </Plate>
  );
}
