/**
 * The GitHub-account control of the workspace configure form.
 *
 * Served by a backend with a ``github:`` block (``config.json`` names
 * ``githubUrl``), it is a dropdown of the GitHub accounts this user has
 * connected -- ``@login`` and a mark on the default -- with ``— none —`` to
 * clear.  Choosing one binds it to this workspace through the backend at
 * once (``POST /api/github/bind``); the daemon then resolves ``GH_TOKEN=
 * app://github`` for that account on every spawn.  Never a token: the page
 * only ever handles the account id and the ``(workspace -> accountId)``
 * binding, and there is no reveal.
 *
 * Unlike the API-key picker beside it, this one writes on change rather than
 * at save time: a binding is BFF state that does not travel through the
 * daemon's ``config.update`` at all, so folding it into the form's Save would
 * be a second, weaker expression of an action that stands on its own.
 *
 * Served without a ``github:`` block, ``githubUrl`` is ``null`` and the whole
 * control is absent -- the form shows what it always did.  With a block but
 * no connected account, it points at the "Connect GitHub" entry rather than
 * offering an empty dropdown.
 */
import { useEffect, useState } from "react";
import { describeAccount, githubApi, type GitHubAccount } from "@/app/github";

export const GH_NONE = "";

export function GitHubAccountPicker({ githubUrl, workspace, onError, onNotice, reloadKey = 0 }: {
  githubUrl: string | null;
  workspace: string;
  onError: (message: string) => void;
  /** Report what the bind did, so an env-not-written note is never silent. */
  onNotice?: (message: string) => void;
  /** Bump to relist after a connect returned (a new account may now exist). */
  reloadKey?: number;
}) {
  const [accounts, setAccounts] = useState<GitHubAccount[]>([]);
  const [bound, setBound] = useState<string>(GH_NONE); // accountId bound to this workspace, or GH_NONE
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const api = githubUrl ? githubApi(githubUrl) : null;

  useEffect(() => {
    if (!api || !workspace) { setAccounts([]); setBound(GH_NONE); setLoaded(false); return; }
    let cancelled = false;
    Promise.all([api.listAccounts(), api.listBindings()]).then(([accs, binds]) => {
      if (cancelled) return;
      setAccounts(accs);
      const b = binds.find((x) => x.workspace === workspace);
      // Keep the binding only if the account it names still exists.
      setBound(b && accs.some((a) => a.id === b.accountId) ? b.accountId : GH_NONE);
      setLoaded(true);
    }).catch((err) => { if (!cancelled) onError(err instanceof Error ? err.message : String(err)); });
    return () => { cancelled = true; };
    // The list and the binding are a function of the workspace; a reconnect bumps reloadKey.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [githubUrl, workspace, reloadKey]);

  const choose = async (accountId: string) => {
    if (!api) return;
    const previous = bound;
    setBound(accountId);
    setBusy(true);
    try {
      const result = await api.bind(workspace, accountId || null);
      if (result.note) onNotice?.(result.note);
    } catch (err) {
      setBound(previous);
      onError(err instanceof Error ? err.message : String(err));
    } finally { setBusy(false); }
  };

  if (!api) return null;

  return (
    <label className="block" data-testid="github-account-picker">
      <span className="field-label">GitHub account</span>
      {loaded && accounts.length === 0 ? (
        <div className="input flex items-center text-[13px] text-text-muted" aria-label="GitHub account">
          none connected — use “Connect GitHub” above
        </div>
      ) : (
        <select
          aria-label="GitHub account"
          value={bound}
          disabled={busy || !loaded}
          onChange={(e) => choose(e.target.value)}
          className="input"
        >
          <option value={GH_NONE}>— none —</option>
          {accounts.map((a) => <option key={a.id} value={a.id}>{describeAccount(a)}</option>)}
        </select>
      )}
    </label>
  );
}
