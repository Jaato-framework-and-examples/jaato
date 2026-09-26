/**
 * Pick GitHub repositories and a branch for each (design 3b, form phase):
 * a search over the repositories the signed-in user's GitHub App
 * installations can reach, on the left, and the repositories picked, each
 * with its branch and target path, on the right.
 *
 * The listing comes from the web backend (``/api/github/repos``,
 * ``/api/github/branches``), which holds the token; the browser only ever
 * sees names.  Without a backend -- or with no GitHub account connected --
 * a typed ``owner/repo`` can still be added by hand, which clones anything
 * public.
 */
import { useEffect, useMemo, useState, type Dispatch, type ReactNode, type SetStateAction } from "react";
import { githubApi, type GitHubRepo } from "@/app/github";
import { cloneTarget } from "@/protocol/workspaces";

export interface PickedRepo {
  repo: string;
  branch: string;
  private?: boolean;
  /** Branches the backend listed, when it did; free text otherwise. */
  branches?: string[];
}

const REPO_RE = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;

/** The listing, filtered by what was typed (case-insensitive substring on ``owner/repo``). */
export function filterRepos(repos: GitHubRepo[], query: string): GitHubRepo[] {
  const q = query.trim().toLowerCase();
  return q ? repos.filter((r) => r.fullName.toLowerCase().includes(q)) : repos;
}

function useRepoListing(githubUrl: string | null | undefined) {
  const [repos, setRepos] = useState<GitHubRepo[]>([]);
  const [login, setLogin] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!githubUrl) return;
    let live = true;
    setLoading(true);
    githubApi(githubUrl).listRepos().then((l) => {
      if (!live) return;
      setRepos(l.repos);
      setLogin(l.account.login);
      setError("");
    }).catch((err) => {
      if (live) setError(err instanceof Error ? err.message : String(err));
    }).finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [githubUrl]);
  return { repos, login, error, loading };
}

function Check({ on }: { on: boolean }) {
  return (
    <span aria-hidden="true" className={`inline-flex items-center justify-center w-[14px] h-[14px] border text-[10px] leading-none shrink-0 ${on ? "bg-steel border-steel text-bg" : "border-[color:var(--c-text-muted)]"}`}>
      {on ? "✓" : ""}
    </span>
  );
}

export function RepoPicker({ githubUrl, workspace, picked, onChange, leading }: {
  githubUrl?: string | null;
  /** Drawn at the top of the left column (the New workspace form's Name field). */
  leading?: ReactNode;
  workspace: string;
  picked: PickedRepo[];
  /** A state setter: branch lists arrive after the pick, so updates are functional. */
  onChange: Dispatch<SetStateAction<PickedRepo[]>>;
}) {
  const { repos, login, error, loading } = useRepoListing(githubUrl);
  const [query, setQuery] = useState("");
  const shown = useMemo(() => filterRepos(repos, query), [repos, query]);
  const isPicked = (name: string) => picked.some((p) => p.repo === name);
  const typed = query.trim();
  const canAddTyped = REPO_RE.test(typed) && !repos.some((r) => r.fullName.toLowerCase() === typed.toLowerCase()) && !isPicked(typed);

  const loadBranches = (name: string) => {
    if (!githubUrl) return;
    githubApi(githubUrl).listBranches(name).then((b) => {
      onChange((cur) => cur.map((p) => (p.repo === name ? { ...p, branches: b.branches, branch: p.branch || b.defaultBranch || b.branches[0] || "" } : p)));
    }).catch(() => undefined);
  };

  const toggle = (r: GitHubRepo) => {
    if (isPicked(r.fullName)) onChange(picked.filter((p) => p.repo !== r.fullName));
    else {
      onChange([...picked, { repo: r.fullName, branch: r.defaultBranch || "", private: r.private }]);
      loadBranches(r.fullName);
    }
  };
  const addTyped = () => {
    onChange([...picked, { repo: typed, branch: "" }]);
    setQuery("");
    loadBranches(typed);
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-[1fr_1px_1.25fr]">
      <div className="p-5 flex flex-col gap-2 min-w-0">
        {leading}
        <label htmlFor="repo-search" className="text-[12px] text-text-muted">
          {githubUrl ? <>Find repositories on GitHub {login && <span className="font-mono text-text">@{login}</span>}</> : "Add a public GitHub repository"}
        </label>
        <input
          id="repo-search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && canAddTyped) { e.preventDefault(); addTyped(); } }}
          placeholder="owner/repo"
          spellCheck={false}
          autoComplete="off"
          className="input input-mono"
        />
        {error && <span className="text-[12px] text-warning">{error}</span>}
        <ul className="border hairline m-0 p-0 list-none max-h-[230px] overflow-auto" aria-label="Repositories">
          {loading && <li className="px-3 py-2 text-[13px] text-text-muted">Loading repositories…</li>}
          {shown.map((r) => {
            const on = isPicked(r.fullName);
            return (
              <li key={r.fullName}>
                <button type="button" role="checkbox" aria-checked={on} onClick={() => toggle(r)} className={`w-full flex items-center gap-2.5 px-3 py-1.5 border-b hairline text-left ${on ? "tint" : "hover:bg-tint/60"}`}>
                  <Check on={on} />
                  <span className="font-mono text-[12px] flex-1 min-w-0 truncate">{r.fullName}</span>
                  <span className="chrome chrome-sm text-[11px] text-text-muted">{r.private ? "Private" : "Public"}</span>
                </button>
              </li>
            );
          })}
          {canAddTyped && (
            <li>
              <button type="button" onClick={addTyped} className="w-full text-left px-3 py-1.5 chrome chrome-sm text-steel hover:bg-tint">+ Add {typed}</button>
            </li>
          )}
          {!loading && shown.length === 0 && !canAddTyped && (
            <li className="px-3 py-2 text-[13px] text-text-muted">{repos.length || githubUrl ? "No repositories match." : "Type owner/repo to add one."}</li>
          )}
        </ul>
      </div>
      <div className="hidden md:block bg-divider" aria-hidden="true" />
      <div className="p-5 flex flex-col gap-2 min-w-0" aria-label="Repositories to clone">
        <div className="flex justify-between items-baseline border-b hairline pb-2">
          <span className="kicker kicker-muted">Clone into the workspace</span>
          <span className="font-mono text-[12px] text-text-muted">{picked.length}</span>
        </div>
        {picked.length === 0 ? (
          <div className="border border-dashed hairline px-4 py-6 text-[13px] text-text-muted">No repositories picked. The workspace will be created empty.</div>
        ) : (
          <ul className="m-0 p-0 list-none flex flex-col">
            {picked.map((p) => (
              <li key={p.repo} className="flex items-center gap-3 py-2.5 border-b hairline" data-testid="picked-repo">
                <div className="flex-1 min-w-0 flex flex-col">
                  <span className="font-mono text-[13px] truncate">{p.repo}</span>
                  <span className="font-mono text-[11px] text-text-muted truncate">→ {cloneTarget(workspace || "{name}", p.repo)}</span>
                </div>
                {/* A fixed-width wrapper: ``.input`` is unlayered CSS with
                    ``width: 100%``, which a width utility on the field loses to. */}
                <div className="w-[150px] shrink-0">
                {p.branches && p.branches.length > 0 ? (
                  <select value={p.branch} onChange={(e) => onChange(picked.map((x) => (x.repo === p.repo ? { ...x, branch: e.target.value } : x)))} aria-label={`Branch of ${p.repo}`} className="input input-mono">
                    {!p.branches.includes(p.branch) && p.branch && <option value={p.branch}>{p.branch}</option>}
                    {p.branches.map((b) => <option key={b} value={b}>{b}</option>)}
                  </select>
                ) : (
                  <input value={p.branch} onChange={(e) => onChange(picked.map((x) => (x.repo === p.repo ? { ...x, branch: e.target.value } : x)))} placeholder="main" aria-label={`Branch of ${p.repo}`} spellCheck={false} className="input input-mono" />
                )}
                </div>
                <button type="button" onClick={() => onChange(picked.filter((x) => x.repo !== p.repo))} aria-label={`Remove ${p.repo}`} className="text-text-muted hover:text-error px-1">×</button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
