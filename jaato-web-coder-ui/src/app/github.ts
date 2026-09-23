/**
 * The page's side of the sign-in backend's per-user GitHub connection.
 *
 * ``jaato-web-coder-server`` may hold, for a signed-in user, the GitHub App
 * grants they have connected (personal, work) and the ``(workspace ->
 * account)`` bindings that point sessions at one of them.  ``config.json``
 * names where (``githubUrl`` for the JSON API, ``githubLoginUrl`` for the
 * server-side OAuth connect); without them the settings entry and the
 * workspace account dropdown do not appear, and nothing here is called --
 * the same opt-in shape ``app/credentials.ts`` has.
 *
 * Everything is a same-origin fetch carrying the backend's session cookie,
 * like ``app/tickets.ts`` and ``app/credentials.ts``.  The one thing this
 * module deliberately CANNOT do is read a token value: there is no ``reveal``
 * here, because a GitHub token never reaches the browser -- it travels BFF ->
 * daemon over ``secret.resolve``.  The page only ever handles an account
 * ``id``, its GitHub ``login``, the installation list, the default flag, and
 * the ``(workspace -> accountId)`` bindings.  A ``401`` means the backend
 * session is gone; every other failure is reported as what it is.
 *
 * Connecting an account is NOT a fetch: it is a full-page navigation to
 * ``githubLoginUrl`` (a 302 to GitHub, whose callback lands the browser back
 * on the app), so this module does not model it -- the caller navigates.
 */
import { SignInRequiredError } from "./tickets";

/** One GitHub App installation the connected account can reach. */
export interface GitHubInstallation {
  id: number;
  account: string;
}

/** What a client may see about one connected account: never a token. */
export interface GitHubAccount {
  id: string;
  login: string;
  name: string | null;
  noreplyEmail: string;
  installations: GitHubInstallation[];
  isDefault: boolean;
  createdAt: string;
  updatedAt: string;
}

/** A ``(workspace -> account)`` binding, as the configure form prefills from it. */
export interface GitHubBinding {
  workspace: string;
  accountId: string;
}

/** The backend's answer to a ``bind`` -- what it did, so the UI never claims a write that did not happen. */
export interface BindResult {
  binding: "set" | "cleared";
  envWritten: boolean;
  gitconfigSeeded: boolean;
  reloaded: number;
  note?: string;
}

export interface GitHubApi {
  listAccounts(): Promise<GitHubAccount[]>;
  listBindings(): Promise<GitHubBinding[]>;
  /** Make one connected account the default; returns the accounts as they now stand. */
  setDefault(id: string): Promise<GitHubAccount[]>;
  /** Delete + revoke a grant and reload the user's sessions; returns the login dropped and the accounts left. */
  disconnect(id: string): Promise<{ disconnected: string; accounts: GitHubAccount[] }>;
  /** Bind an account to a workspace, or clear it with ``null``. */
  bind(workspace: string, accountId: string | null): Promise<BindResult>;
}

function subUrl(base: string, leaf: string): string {
  const [path = base, query = ""] = base.split(/\?(.*)/s, 2);
  return `${path.replace(/\/$/, "")}/${leaf}${query ? `?${query}` : ""}`;
}

async function failure(res: Response, what: string): Promise<Error> {
  if (res.status === 401) return new SignInRequiredError("./auth/login");
  let detail = "";
  try { detail = String(((await res.json()) as { error?: unknown }).error ?? ""); } catch { /* not JSON */ }
  return new Error(`${what}: HTTP ${res.status}${detail ? ` (${detail})` : ""}`);
}

function accountsOf(body: unknown): GitHubAccount[] {
  const list = (body as { accounts?: unknown }).accounts;
  return Array.isArray(list) ? (list as GitHubAccount[]).filter((a) => a && typeof a.id === "string") : [];
}

export function githubApi(githubUrl: string, fetchImpl: typeof fetch = fetch): GitHubApi {
  const common: RequestInit = { credentials: "same-origin", cache: "no-store" };
  const getInit = { ...common, headers: { Accept: "application/json" } };
  const jsonHeaders = { Accept: "application/json", "Content-Type": "application/json" };
  const post = (leaf: string, body: unknown) =>
    fetchImpl(subUrl(githubUrl, leaf), { ...common, method: "POST", headers: jsonHeaders, body: JSON.stringify(body) });
  return {
    async listAccounts() {
      const res = await fetchImpl(subUrl(githubUrl, "accounts"), getInit);
      if (!res.ok) throw await failure(res, "Listing GitHub accounts failed");
      return accountsOf(await res.json());
    },
    async listBindings() {
      const res = await fetchImpl(subUrl(githubUrl, "bindings"), getInit);
      if (!res.ok) throw await failure(res, "Listing GitHub bindings failed");
      const list = (await res.json() as { bindings?: unknown }).bindings;
      return Array.isArray(list)
        ? (list as GitHubBinding[]).filter((b) => b && typeof b.workspace === "string" && typeof b.accountId === "string")
        : [];
    },
    async setDefault(id) {
      const res = await post("default", { id });
      if (!res.ok) throw await failure(res, "Setting the default GitHub account failed");
      return accountsOf(await res.json());
    },
    async disconnect(id) {
      const res = await post("disconnect", { id });
      if (!res.ok) throw await failure(res, "Disconnecting the GitHub account failed");
      const body = (await res.json()) as { disconnected?: unknown };
      return { disconnected: typeof body.disconnected === "string" ? body.disconnected : "", accounts: accountsOf(body) };
    },
    async bind(workspace, accountId) {
      const res = await post("bind", { workspace, account_id: accountId });
      if (!res.ok) throw await failure(res, "Binding the GitHub account failed");
      const body = (await res.json()) as Partial<BindResult>;
      return {
        binding: body.binding === "set" ? "set" : "cleared",
        envWritten: !!body.envWritten,
        gitconfigSeeded: !!body.gitconfigSeeded,
        reloaded: typeof body.reloaded === "number" ? body.reloaded : 0,
        note: typeof body.note === "string" ? body.note : undefined,
      };
    },
  };
}

/** A connected account's one-line label for an ``<option>`` or a row: ``@login`` and, when it is the default, a mark. */
export function describeAccount(a: GitHubAccount): string {
  return a.isDefault ? `@${a.login} (default)` : `@${a.login}`;
}
