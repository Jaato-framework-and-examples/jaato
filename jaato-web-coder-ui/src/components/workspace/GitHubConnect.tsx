/**
 * The "Connect GitHub" settings surface — the per-user half of the GitHub
 * feature, beside the workspace-account dropdown that is the per-workspace
 * half.
 *
 * Served by a backend with a ``github:`` block (``config.json`` names
 * ``githubUrl`` / ``githubLoginUrl``), it lists the GitHub accounts this
 * signed-in user has connected and lets them connect another, choose the
 * default, or disconnect one.  Modelled on the API-key list in the configure
 * form: a same-origin JSON API, a plate of rows, never a secret.
 *
 * The one thing it never does is show or fetch a token: there is no reveal
 * for a GitHub grant (the token travels BFF -> daemon over
 * ``secret.resolve``, never to the browser).  The page handles only the
 * account id, the ``@login``, the installation list and the default flag.
 *
 * Connecting is a full-page navigation to ``githubLoginUrl`` (a 302 to
 * GitHub whose callback lands the browser back on the app), so it is an
 * ordinary link, not a fetch.  Disconnecting is destructive -- it revokes
 * the grant at GitHub and reloads the user's live sessions -- so the row
 * asks first.
 */
import { useEffect, useState } from "react";
import { describeAccount, githubApi, type GitHubAccount } from "@/app/github";

export function GitHubConnect({ githubUrl, githubLoginUrl, onError, onNotice, onChanged }: {
  githubUrl: string;
  githubLoginUrl: string | null;
  onError: (message: string) => void;
  onNotice?: (message: string) => void;
  /** Fired after set-default / disconnect changed the accounts, so a workspace picker relists. */
  onChanged?: () => void;
}) {
  const [accounts, setAccounts] = useState<GitHubAccount[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState<string | null>(null); // account id whose disconnect is awaiting confirmation
  const api = githubApi(githubUrl);

  useEffect(() => {
    let cancelled = false;
    api.listAccounts().then((list) => {
      if (cancelled) return;
      setAccounts(list);
      setLoaded(true);
    }).catch((err) => { if (!cancelled) onError(err instanceof Error ? err.message : String(err)); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [githubUrl]);

  const makeDefault = async (id: string) => {
    setBusy(true);
    try { setAccounts(await api.setDefault(id)); onChanged?.(); }
    catch (err) { onError(err instanceof Error ? err.message : String(err)); }
    finally { setBusy(false); }
  };
  const disconnect = async (id: string) => {
    setBusy(true);
    try {
      const { disconnected, accounts: left } = await api.disconnect(id);
      setAccounts(left);
      onNotice?.(`Disconnected @${disconnected} from GitHub.`);
      onChanged?.();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally { setBusy(false); setConfirming(null); }
  };

  return (
    <div className="flex flex-col" role="group" aria-label="Connect GitHub" data-testid="github-connect">
      <div className="flex items-baseline gap-3 px-5 py-3.5 border-b hairline">
        <span className="kicker tracking-[0.16em]">Connect GitHub</span>
        <span className="flex-1" />
        {githubLoginUrl && (
          <a href={githubLoginUrl} className="btn btn-sm btn-steel" aria-label="Connect a GitHub account">Connect an account</a>
        )}
      </div>
      <div className="p-5 flex flex-col gap-3">
        <div className="text-xs text-text-muted">
          Sessions in a workspace bound to an account act as that GitHub user through <span className="font-mono">gh</span>.
          The token stays on the server — it is never shown here.
        </div>
        {loaded && accounts.length === 0 ? (
          <div className="text-sm text-text-muted italic">No GitHub accounts connected yet.</div>
        ) : (
          <ul className="flex flex-col divide-y hairline">
            {accounts.map((a) => (
              <li key={a.id} className="flex items-center gap-3 py-2.5">
                <div className="flex-1 min-w-0">
                  <div className="font-mono text-sm">
                    {describeAccount(a)}
                    {a.name && <span className="ml-2 text-text-muted font-sans text-xs">{a.name}</span>}
                  </div>
                  <div className="text-[11px] text-text-muted">
                    {a.installations.length
                      ? `installed on ${a.installations.map((i) => i.account).join(", ")}`
                      : "no installations"}
                  </div>
                </div>
                {confirming === a.id ? (
                  <span className="inline-flex items-center gap-2 text-xs" role="group" aria-label={`Confirm disconnecting @${a.login}`}>
                    <span className="text-warning">revoke @{a.login}?</span>
                    <button type="button" disabled={busy} onClick={() => disconnect(a.id)} className="btn btn-sm btn-danger" aria-label={`Confirm disconnect @${a.login}`}>Disconnect</button>
                    <button type="button" onClick={() => setConfirming(null)} className="btn btn-sm" aria-label="Cancel disconnect">Cancel</button>
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-2">
                    {!a.isDefault && (
                      <button type="button" disabled={busy} onClick={() => makeDefault(a.id)} className="btn btn-sm btn-quiet border-transparent text-steel" aria-label={`Make @${a.login} the default`}>Set default</button>
                    )}
                    <button type="button" disabled={busy} onClick={() => setConfirming(a.id)} className="btn btn-sm btn-quiet border-transparent hover:text-error" aria-label={`Disconnect @${a.login}`}>Disconnect</button>
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
