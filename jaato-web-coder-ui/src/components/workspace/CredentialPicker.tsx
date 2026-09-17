/**
 * The API-key control of the workspace configure form.
 *
 * Served by a backend with a key store (``config.json`` names
 * ``credentialsUrl``), it is a combobox of the keys this user has stored
 * for the selected provider -- label and a masked hint, never the secret --
 * with "New key…" as the last option; choosing it reveals the password
 * field and an optional label.  Served without one, it is that password
 * field alone, exactly what the form had before.  The form owns the
 * choice (:type:`KeyChoice`) and resolves it to a key at save time
 * (``WorkspaceScreen.saveConfig``); this component only lists, picks and
 * forgets.
 *
 * The list reloads when the provider changes, and the most recently stored
 * entry is preselected the first time a list arrives so that "open a new
 * workspace on the same provider" is one click.  A user who has already
 * chosen keeps their choice across reloads.
 */
import { useEffect, useState } from "react";
import { credentialsApi, describeCredential, type StoredCredential } from "@/app/credentials";

/** What the form saves: a stored entry to reveal, a new key to store and apply, or nothing. */
export type KeyChoice =
  | { kind: "none" }
  | { kind: "stored"; id: string }
  | { kind: "new"; secret: string; label: string };

export const NEW_KEY = "__new__";
export const NO_KEY = "";

export function CredentialPicker({ credentialsUrl, provider, value, onChange, onError, reloadKey = 0 }: {
  credentialsUrl: string | null;
  provider: string;
  value: KeyChoice;
  onChange: (c: KeyChoice) => void;
  onError: (message: string) => void;
  /** Bump to reload the list without changing the provider -- after the form stored a new key. */
  reloadKey?: number;
}) {
  const [entries, setEntries] = useState<StoredCredential[]>([]);
  const [loaded, setLoaded] = useState<string | null>(null); // provider the list belongs to
  const [busy, setBusy] = useState(false);
  const api = credentialsUrl ? credentialsApi(credentialsUrl) : null;

  useEffect(() => {
    if (!api || !provider) { setEntries([]); setLoaded(null); return; }
    let cancelled = false;
    api.list(provider).then((list) => {
      if (cancelled) return;
      setEntries(list);
      setLoaded(provider);
      // First list for this provider and nothing chosen yet: offer the newest.
      if (value.kind === "none" && list.length > 0) onChange({ kind: "stored", id: list[list.length - 1]!.id });
      if (value.kind === "stored" && !list.some((e) => e.id === value.id)) onChange({ kind: "none" });
    }).catch((err) => { if (!cancelled) onError(err instanceof Error ? err.message : String(err)); });
    return () => { cancelled = true; };
    // The list is a function of the provider; the choice is the form's.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [credentialsUrl, provider, reloadKey]);

  const forget = async (id: string) => {
    if (!api) return;
    setBusy(true);
    try {
      await api.remove(id);
      setEntries((es) => es.filter((e) => e.id !== id));
      if (value.kind === "stored" && value.id === id) onChange({ kind: "none" });
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally { setBusy(false); }
  };

  const secret = value.kind === "new" ? value.secret : "";
  const label = value.kind === "new" ? value.label : "";
  const setNew = (patch: { secret?: string; label?: string }) => onChange({ kind: "new", secret: patch.secret ?? secret, label: patch.label ?? label });
  const inputClass = "input input-mono mt-1";

  // No store: the plain field the form always had.
  if (!api) {
    return (
      <label className="block"><span className="field-label">API key</span>
        <input aria-label="API key" value={secret} onChange={(e) => setNew({ secret: e.target.value })} type="password" autoComplete="off" className={inputClass} />
      </label>
    );
  }

  const selectValue = value.kind === "stored" ? value.id : value.kind === "new" ? NEW_KEY : NO_KEY;
  const selected = value.kind === "stored" ? entries.find((e) => e.id === value.id) : undefined;
  return (
    <div className="block" data-testid="credential-picker">
      <label className="block"><span className="field-label">API key</span>
        <select
          aria-label="API key"
          value={selectValue}
          disabled={!provider}
          onChange={(e) => {
            const v = e.target.value;
            if (v === NEW_KEY) onChange({ kind: "new", secret: "", label: "" });
            else if (v === NO_KEY) onChange({ kind: "none" });
            else onChange({ kind: "stored", id: v });
          }}
          className="input"
        >
          <option value={NO_KEY}>{loaded === provider && entries.length === 0 ? "— no stored key —" : "—"}</option>
          {entries.map((e) => <option key={e.id} value={e.id}>{describeCredential(e)}</option>)}
          <option value={NEW_KEY}>New key…</option>
        </select>
      </label>
      {selected && (
        <div className="mt-1 flex items-center justify-between text-[11px] text-text-muted">
          <span>stored {new Date(selected.createdAt).toLocaleDateString()}</span>
          <button type="button" disabled={busy} onClick={() => forget(selected.id)} className="link hover:text-error" aria-label={`Forget stored key ${selected.label}`}>forget</button>
        </div>
      )}
      {value.kind === "new" && (
        <>
          <input aria-label="New API key" value={secret} onChange={(e) => setNew({ secret: e.target.value })} type="password" autoComplete="off" placeholder="paste the key" className={inputClass} />
          <input aria-label="Key label" value={label} onChange={(e) => setNew({ label: e.target.value })} placeholder="label (optional) — e.g. work account" className={`${inputClass} text-xs`} />
        </>
      )}
    </div>
  );
}
