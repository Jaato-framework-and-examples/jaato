/**
 * Column 4 of the session picker (design 2a / 2a′): start a new session.
 *
 * Four sections, top to bottom: the BASE PROFILE (``default`` selected, the
 * others behind a collapsed toggle), the MODEL (three cases -- see
 * ``app/newSession.ts``), the STAGED FILES and START.
 *
 * Staging here is client-side until Start: nothing is written to the
 * workspace while you are still choosing.  On Start each file is handed to
 * ``app/staging.ts`` (the ``StageFilesRequest`` verb) under its target
 * folder, and ``openSessionWithQueued`` puts them on disk before
 * ``session.new`` when a workspace is selected -- the session starts with
 * them there.  The word is *stage*, never *attach*: attaching is what you
 * do to a session.
 *
 * A file whose target path already exists in the workspace says so before
 * Start (``⚠ replaces …``), from a metadata-only ``workspace.file.fetch``
 * (protocol 1.20).  A daemon below that, or no workspace selected, simply
 * gives no warning -- the check is advisory and never blocks.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { useJaato } from "@/store/store";
import { getClient } from "@/sdk/connection";
import { attachFiles } from "@/app/staging";
import { servesDownloads } from "@/app/downloads";
import { formatSize, stagedName } from "@/protocol/attachments";
import { filesFromTransfer } from "@/components/input/AttachStrip";
import {
  baseProfiles, definesModel, formatBinding, knownModels, knownProviders, modelStatus, modelTag,
  resolveModel, startRequest, startSummary, type BaseProfile, type ModelChoice, type ModelMode,
} from "@/app/newSession";
import { ColumnHeader } from "./SessionBoard";
import { CredentialPicker, type KeyChoice } from "@/components/workspace/CredentialPicker";
import { servesKeyOnly } from "@/app/sessionKey";

/** The API key Start applies for the session's provider, when one was chosen. */
export interface SessionKey {
  provider: string;
  choice: KeyChoice;
}

export interface StagedDraft {
  id: string;
  file: File;
  name: string;
  size: number;
  /** Target folder inside the workspace; "" is the root. */
  dir: string;
}

let draftSeq = 0;

/** The workspace path a draft will land at, or ``null`` when the daemon would refuse the name. */
export function draftTarget(d: StagedDraft): string | null {
  return stagedName(d.name, "", d.dir);
}

/**
 * Which drafts would replace an existing workspace file.  Asked per target
 * path, debounced, and only when the daemon can answer a metadata fetch.
 */
function useExistingTargets(drafts: StagedDraft[]): Record<string, boolean> {
  const selected = useJaato((s) => s.workspace.selected);
  const protocol = useJaato((s) => (s.connection.phase === "connected" ? s.connection.protocolVersion : null));
  const [exists, setExists] = useState<Record<string, boolean>>({});
  const key = drafts.map((d) => draftTarget(d) ?? "").join("\n");
  useEffect(() => {
    if (!selected || !servesDownloads(protocol)) return;
    const targets = [...new Set(drafts.map(draftTarget).filter((t): t is string => Boolean(t)))].filter((t) => !(t in exists));
    if (!targets.length) return;
    let live = true;
    const timer = setTimeout(async () => {
      const found: Record<string, boolean> = {};
      for (const t of targets) {
        try {
          const { event } = await getClient().fetchWorkspaceFile(t, { metadataOnly: true });
          found[t] = event.ok === true;
        } catch {
          found[t] = false;
        }
      }
      if (live) setExists((cur) => ({ ...cur, ...found }));
    }, 350);
    return () => { live = false; clearTimeout(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, selected, protocol]);
  return exists;
}

function ProfileBox({ p }: { p: BaseProfile }) {
  return (
    <div className="border border-steel bg-surface px-3.5 py-2.5 flex flex-col gap-0.5" data-testid="base-profile">
      <span className="font-mono text-[13px] break-words">{p.name}</span>
      {p.description && <span className="text-[12px] text-text-muted">{p.description}</span>}
      <span className="font-mono text-[11px] text-steel">{modelTag(p)}</span>
    </div>
  );
}

function ModelSelects({ providers, models, pick, onPick, warn }: {
  providers: string[];
  models: string[];
  pick: ModelChoice;
  onPick: (c: ModelChoice) => void;
  warn: boolean;
}) {
  const warnBorder = (empty: boolean) => (warn && empty ? "input-warn" : "");
  return (
    <div className="flex flex-col gap-2">
      <select
        value={pick.provider}
        onChange={(e) => onPick({ provider: e.target.value, model: "" })}
        aria-label="Provider"
        className={`input ${warnBorder(!pick.provider)}`}
      >
        <option value="">Select provider…</option>
        {providers.map((p) => <option key={p} value={p}>{p}</option>)}
      </select>
      <input
        value={pick.model}
        onChange={(e) => onPick({ ...pick, model: e.target.value })}
        disabled={!pick.provider}
        list="new-session-models"
        placeholder="Select model…"
        aria-label="Model"
        spellCheck={false}
        autoComplete="off"
        className={`input input-mono ${warnBorder(!pick.model.trim())}`}
      />
      <datalist id="new-session-models">{models.map((m) => <option key={m} value={m} />)}</datalist>
    </div>
  );
}

/** New drafts for ``files``, each going to the workspace root until its folder is edited. */
export function draftsFor(files: File[]): StagedDraft[] {
  return files.map((f) => ({ id: `draft-${++draftSeq}`, file: f, name: f.name, size: f.size, dir: "" }));
}

function StagedFiles({ drafts, setDrafts, add }: { drafts: StagedDraft[]; setDrafts: (fn: (d: StagedDraft[]) => StagedDraft[]) => void; add: (files: File[]) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const exists = useExistingTargets(drafts);
  return (
    <div className="flex flex-col gap-1.5" data-testid="staged-files">
      <div className="flex justify-between text-[12px] text-text-muted">
        <span>Staged files</span>
        <span className="font-mono">{drafts.length || "none"}</span>
      </div>
      <input ref={input} type="file" multiple className="hidden" aria-label="Stage files" onChange={(e) => { add(Array.from(e.target.files ?? [])); e.target.value = ""; }} />
      <ul className="border hairline bg-surface m-0 p-0 list-none flex flex-col">
        {drafts.map((d) => {
          const target = draftTarget(d);
          return (
            <li key={d.id} className="px-3 py-2 border-b hairline flex flex-col gap-1" data-testid="staged-file">
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-[12px] flex-1 min-w-0 truncate" title={d.name}>{d.name}</span>
                <span className="font-mono text-[11px] text-text-muted shrink-0">{formatSize(d.size)}</span>
                <button type="button" onClick={() => setDrafts((cur) => cur.filter((x) => x.id !== d.id))} aria-label={`Remove ${d.name}`} className="text-text-muted hover:text-error px-0.5">×</button>
              </div>
              <label className="flex items-center gap-2">
                <span className="kicker kicker-muted">Into</span>
                <input
                  value={d.dir}
                  onChange={(e) => setDrafts((cur) => cur.map((x) => (x.id === d.id ? { ...x, dir: e.target.value } : x)))}
                  placeholder="workspace root"
                  aria-label={`Folder for ${d.name}`}
                  spellCheck={false}
                  className="input input-mono text-[12px] py-0.5 min-h-0"
                />
              </label>
              {target === null && <span className="text-[12px] text-error">The folder must be a relative path with no “..”.</span>}
              {target && exists[target] && <span className="text-[12px] text-warning">⚠ replaces {target} in the workspace</span>}
            </li>
          );
        })}
        <li>
          <button type="button" onClick={() => input.current?.click()} className="w-full text-left px-3 py-2 chrome chrome-sm text-steel hover:bg-tint">
            + Stage files · drop or choose
          </button>
        </li>
      </ul>
      <span className="text-[12px] text-text-muted">Copied into the workspace when you press Start.</span>
    </div>
  );
}

/**
 * The provider whose API key the session will need: the inherited binding's,
 * or the picked one's.  ``""`` while no provider is known (and for the
 * daemon's .env fallback, whose key is already wherever the .env put it).
 */
export function keyProvider(base: BaseProfile, mode: ModelMode, pick: ModelChoice): string {
  if (definesModel(base) && mode === "inherit") return base.provider;
  return pick.provider;
}

export function NewSessionColumn({ onStart }: { onStart: (profile: string | null, model: ModelChoice | undefined, files: StagedDraft[], key?: SessionKey) => void }) {
  const profiles = useJaato((s) => s.profiles);
  const sessions = useJaato((s) => s.sessions);
  const cfg = useJaato((s) => (s.workspace.config && s.workspace.config.workspace === s.workspace.selected ? s.workspace.config : undefined));
  const workspaceMode = useJaato((s) => s.workspace.mode === "enabled");
  const bases = useMemo(() => baseProfiles(profiles, { envFallback: !workspaceMode }), [profiles, workspaceMode]);
  const [baseName, setBaseName] = useState(bases[0]!.name);
  const base = bases.find((b) => b.name === baseName) ?? bases[0]!;
  const [moreOpen, setMoreOpen] = useState(false);
  const [mode, setMode] = useState<ModelMode>("inherit");
  // A workspace whose .env binds a provider/model prefills the pick: the
  // workspace no longer HAS a model, but what it used to say is the best
  // guess at what you want.
  const [pick, setPick] = useState<ModelChoice>({ provider: "", model: "" });
  const prefilled = useRef(false);
  useEffect(() => {
    if (prefilled.current || !cfg?.provider) return;
    prefilled.current = true;
    setPick((cur) => (cur.provider ? cur : { provider: cfg.provider ?? "", model: cfg.model ?? "" }));
  }, [cfg]);
  // The API key list box -- the configure form's control, kept by the
  // redesign.  A key is written to the selected workspace's .env
  // (``config.update`` key_only, 1.26), so it is offered only where there is
  // a workspace to write to and a daemon that writes nothing else.
  const credentialsUrl = useJaato((s) => s.credentialsUrl);
  const selectedWs = useJaato((s) => s.workspace.selected);
  const protocol = useJaato((s) => (s.connection.phase === "connected" ? s.connection.protocolVersion : null));
  const [keyChoice, setKeyChoice] = useState<KeyChoice>({ kind: "none" });
  const [keyError, setKeyError] = useState("");
  const [drafts, setDrafts] = useState<StagedDraft[]>([]);
  const addDrafts = (files: File[]) => { if (files.length) setDrafts((cur) => [...cur, ...draftsFor(files)]); };
  const [dragging, setDragging] = useState(false);

  const seen: ModelChoice[] = useMemo(() => [
    ...profiles.map((p) => ({ provider: String(p.provider ?? ""), model: String(p.model ?? "") })),
    ...sessions.map((s) => ({ provider: s.provider, model: s.model })),
    ...(cfg?.provider ? [{ provider: cfg.provider, model: cfg.model ?? "" }] : []),
  ], [profiles, sessions, cfg]);
  const providers = knownProviders(cfg?.availableProviders ?? [], seen);
  const models = knownModels(pick.provider, seen);

  const resolved = resolveModel(base, mode, pick);
  const status = modelStatus(base, mode, pick);
  const request = startRequest(base, mode, pick);
  const others = bases.filter((b) => b.name !== base.name);
  const badDraft = drafts.some((d) => draftTarget(d) === null);
  const provider = keyProvider(base, mode, pick);
  const offersKey = Boolean(selectedWs && servesKeyOnly(protocol) && provider);
  // A new provider is a new list: the picker preselects its newest stored key.
  useEffect(() => { setKeyChoice({ kind: "none" }); setKeyError(""); }, [provider]);
  const keyIncomplete = keyChoice.kind === "new" && !keyChoice.secret.trim();

  const choose = (name: string) => {
    setBaseName(name);
    setMoreOpen(false);
    setMode("inherit");
  };
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    addDrafts(filesFromTransfer(e.dataTransfer));
  };

  return (
    <section
      aria-label="New session"
      data-column="new"
      className={`flex flex-col min-w-0 bg-bg ${dragging ? "outline outline-2 outline-steel -outline-offset-2" : ""}`}
      onDragOver={(e) => { if (Array.from(e.dataTransfer.types).includes("Files")) { e.preventDefault(); setDragging(true); } }}
      onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragging(false); }}
      onDrop={onDrop}
    >
      <ColumnHeader title="New session" tone="text-steel" rule="border-steel" />
      <div className="px-4 py-3 flex flex-col gap-4">
        <div className="flex flex-col gap-1.5">
          <span className="text-[12px] text-text-muted">Base profile</span>
          <ProfileBox p={base} />
          {others.length > 0 && (
            <button type="button" onClick={() => setMoreOpen((v) => !v)} aria-expanded={moreOpen} className="self-start chrome chrome-sm text-steel mt-0.5">
              <span aria-hidden="true">{moreOpen ? "▾" : "▸"}</span> {moreOpen ? "Hide base profiles" : `${others.length} more base ${others.length === 1 ? "profile" : "profiles"}`}
            </button>
          )}
          {moreOpen && (
            <ul className="border hairline bg-surface m-0 p-0 list-none max-h-[260px] overflow-auto" aria-label="Base profiles">
              {others.map((p) => (
                <li key={p.name}>
                  <button type="button" onClick={() => choose(p.name)} className="w-full text-left px-3 py-2 border-b hairline hover:bg-tint flex flex-col" aria-label={`Use base profile ${p.name}`}>
                    <span className="font-mono text-[12px] break-words">{p.name}</span>
                    <span className={`font-mono text-[11px] ${definesModel(p) ? "text-steel" : "text-text-muted"}`}>{modelTag(p)}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="flex flex-col gap-1.5" data-testid="model-section">
          <span className="text-[12px] text-text-muted">Model <span className={status.warn ? "text-warning" : ""}>· {status.text}</span></span>
          {definesModel(base) ? (
            <>
              <div className="grid grid-cols-2 border hairline" role="group" aria-label="Model source">
                {(["inherit", "override"] as const).map((m) => (
                  <button key={m} type="button" aria-pressed={mode === m} onClick={() => setMode(m)} className={`chrome chrome-sm px-3 py-1.5 text-left ${mode === m ? "bg-steel text-bg" : "bg-surface text-text-muted hover:text-text"}`}>
                    {m === "inherit" ? "Inherit" : "Override"}
                  </button>
                ))}
              </div>
              {mode === "inherit" ? (
                <div className="border hairline bg-surface px-3 py-2 font-mono text-[13px]" aria-label="Inherited model">{formatBinding(base.provider, base.model)}</div>
              ) : (
                <>
                  <ModelSelects providers={providers} models={models} pick={pick} onPick={setPick} warn={false} />
                  <span className="text-[12px] text-text-muted">Replaces {base.model} for this session only; the base profile is unchanged.</span>
                </>
              )}
            </>
          ) : (
            <>
              <ModelSelects providers={providers} models={models} pick={pick} onPick={setPick} warn={!base.isDefault} />
              <span className="text-[12px] text-text-muted">
                {base.envFallback ? "Optional: leave empty to use the daemon's .env provider and model."
                  : base.isDefault ? "The default profile has no model; pick one for this session."
                  : "Required. This base profile doesn't define a model."}
              </span>
            </>
          )}
        </div>

        {offersKey && (
          <div className="flex flex-col gap-1.5" data-testid="api-key-section">
            <CredentialPicker credentialsUrl={credentialsUrl} provider={provider} value={keyChoice} onChange={setKeyChoice} onError={setKeyError} />
            <span className="text-[12px] text-text-muted">
              {keyChoice.kind === "none"
                ? `Uses the ${provider} key the workspace already has, if any.`
                : `Written to this workspace's .env for ${provider} when you press Start.`}
            </span>
            {keyError && <span className="text-[12px] text-error">{keyError}</span>}
          </div>
        )}

        <StagedFiles drafts={drafts} setDrafts={setDrafts} add={addDrafts} />

        <div className="flex flex-col gap-1.5">
          <button
            type="button"
            disabled={!request || badDraft || (offersKey && keyIncomplete)}
            onClick={() => request && onStart(request.profile, request.model, drafts, offersKey && keyChoice.kind !== "none" ? { provider, choice: keyChoice } : undefined)}
            className="btn btn-primary w-full justify-between"
          >
            <span>Start session</span><span aria-hidden="true">→</span>
          </button>
          <span className="font-mono text-[11px] text-text-muted break-words" data-testid="start-summary">{startSummary(base, resolved, drafts.length)}</span>
        </div>
      </div>
    </section>
  );
}

/** Hand the drafts to the staging module, each under its own folder. */
export function stageDrafts(drafts: StagedDraft[]): void {
  const byDir = new Map<string, File[]>();
  for (const d of drafts) {
    const named = d.name === d.file.name ? d.file : new File([d.file], d.name, { type: d.file.type });
    byDir.set(d.dir, [...(byDir.get(d.dir) ?? []), named]);
  }
  for (const [dir, files] of byDir) attachFiles(files, dir);
}
