/**
 * The GitHub side of the connect flow: the four calls to github.com and
 * api.github.com this application makes on a user's behalf, behind one
 * injectable interface.
 *
 * Why an interface.  Every call here reaches the open internet, so the
 * credential logic that drives them (``src/github.ts``) must be testable
 * without a network or a registered App.  ``HttpGitHubApi`` is the real
 * implementation over ``fetch``; a test supplies a fake with the same shape.
 * The interface is also the whole GitHub Enterprise seam: the OAuth host and
 * the API host are constructor arguments, so a GHE deployment sets
 * ``oauth_base_url`` / ``api_base_url`` and nothing else changes.
 *
 * What each call is (GitHub App user-to-server authorization):
 *
 * - ``exchangeCode``  — ``POST {oauth}/login/oauth/access_token`` with the
 *   authorization code, yielding an access token (~8h) and, when the App has
 *   token expiration enabled, a refresh token (~6 months).
 * - ``refresh``       — the same endpoint with ``grant_type=refresh_token``.
 *   GitHub ROTATES: the response carries a NEW refresh token and voids the
 *   one presented, which is the whole reason ``src/github.ts`` serialises the
 *   read-refresh-write (#683).
 * - ``revokeGrant``   — ``DELETE {api}/applications/{client_id}/grant`` under
 *   HTTP Basic (client id + secret), which voids every token of the user's
 *   grant.  This is what *Disconnect* does so a live session's next call
 *   fails, rather than the token merely being forgotten here.
 * - ``fetchIdentity`` — ``GET {api}/user`` and ``GET {api}/user/installations``
 *   under the freshly minted access token, for the non-secret ``login`` /
 *   installation list kept beside the grant for display, and the noreply
 *   email seeded into the workspace ``.gitconfig``.
 *
 * - ``listInstallationRepos`` — ``GET {api}/user/installations/{id}/repositories``
 *   (paginated) under the user's access token: the repositories one App
 *   installation lets this user reach, for the *New workspace* picker.
 * - ``listBranches`` — ``GET {api}/repos/{owner}/{repo}/branches`` (paginated),
 *   for the branch chooser beside each picked repository.
 *
 * The two listing calls follow ``Link: rel="next"`` only while it stays on the
 * configured API origin (the bearer token is never sent anywhere else), and
 * stop at a caller-supplied cap so one huge organisation cannot turn a picker
 * refresh into hundreds of requests.
 *
 * The secret NEVER travels to the browser through here: these are all
 * server-to-GitHub calls, driven by ``src/github.ts``, which hands only an
 * access token onward to the daemon over the authenticated bind channel.
 */

/** A GitHub error that means the grant is dead: the user revoked it, or the refresh token expired. */
export class GitHubGrantRevoked extends Error {
  override name = "GitHubGrantRevoked";
}

/** A GitHub call that failed for a transient reason (network, 5xx, rate limit). */
export class GitHubApiError extends Error {
  override name = "GitHubApiError";
}

/** The token pair GitHub returns from an authorization-code exchange or a refresh. */
export interface GitHubTokenSet {
  accessToken: string;
  /** Seconds until the access token expires, when GitHub reports one (App token expiration on). */
  accessExpiresInSeconds?: number;
  /** The refresh token, absent for an App with token expiration OFF (the access token then never expires). */
  refreshToken?: string;
  /** Seconds until the refresh token expires, when reported. */
  refreshExpiresInSeconds?: number;
}

/** One App installation the user can see, for display beside the grant. */
export interface GitHubInstallation {
  id: number;
  account: string;
}

/** The non-secret identity behind an access token. */
export interface GitHubIdentity {
  login: string;
  id: number;
  name: string | null;
  /** GitHub's ``id+login@users.noreply.github.com``, computed here so a private email is never published in a commit. */
  noreplyEmail: string;
  installations: GitHubInstallation[];
}

/** One repository an installation lets the user reach (non-secret, for display). */
export interface GitHubRepo {
  /** ``owner/name``. */
  fullName: string;
  private: boolean;
  defaultBranch: string;
  /** ISO-8601 time of the last push, when GitHub reports one. */
  pushedAt?: string;
}

export interface GitHubApi {
  /** The authorize URL the browser is redirected to (App user-to-server). */
  authorizeUrl(state: string, redirectUri: string): string;
  exchangeCode(code: string, redirectUri: string): Promise<GitHubTokenSet>;
  /** Refresh; GitHub rotates the refresh token, so the caller MUST persist the returned one. */
  refresh(refreshToken: string): Promise<GitHubTokenSet>;
  /** Revoke the user's whole grant at GitHub (best-effort; a already-dead grant is not an error to the caller). */
  revokeGrant(accessToken: string): Promise<void>;
  fetchIdentity(accessToken: string): Promise<GitHubIdentity>;
  /**
   * Every repository installation ``installationId`` lets the token's user
   * reach, following pagination up to ``maxItems``.  A 404 / 403 (an
   * installation that was removed, or one this user no longer sees) is a
   * {@link GitHubApiError}; a 401 is {@link GitHubGrantRevoked}.
   */
  listInstallationRepos(accessToken: string, installationId: number, maxItems: number): Promise<GitHubRepo[]>;
  /** Branch names of ``owner/repo``, following pagination up to ``maxItems``. */
  listBranches(accessToken: string, owner: string, repo: string, maxItems: number): Promise<string[]>;
  /** The git host to seed into ``.gitconfig`` (``github.com`` or the GHE host). */
  readonly gitHost: string;
}

export interface HttpGitHubApiOptions {
  clientId: string;
  clientSecret: string;
  /** Default ``https://github.com``; a GHE host for enterprise. */
  oauthBaseUrl?: string;
  /** Default ``https://api.github.com``; ``https://<ghe-host>/api/v3`` for enterprise. */
  apiBaseUrl?: string;
  /** The noreply email domain; ``users.noreply.github.com`` for github.com. */
  noreplyDomain?: string;
  /** Injectable for tests; defaults to the global ``fetch``. */
  fetchImpl?: typeof fetch;
}

const DEFAULT_OAUTH = "https://github.com";
const DEFAULT_API = "https://api.github.com";
const DEFAULT_NOREPLY = "users.noreply.github.com";

/** The real implementation over ``fetch``. */
export class HttpGitHubApi implements GitHubApi {
  private readonly _clientId: string;
  private readonly _clientSecret: string;
  private readonly _oauth: string;
  private readonly _api: string;
  private readonly _noreplyDomain: string;
  private readonly _fetch: typeof fetch;
  readonly gitHost: string;

  constructor(opts: HttpGitHubApiOptions) {
    this._clientId = opts.clientId;
    this._clientSecret = opts.clientSecret;
    this._oauth = (opts.oauthBaseUrl ?? DEFAULT_OAUTH).replace(/\/+$/, "");
    this._api = (opts.apiBaseUrl ?? DEFAULT_API).replace(/\/+$/, "");
    this._noreplyDomain = opts.noreplyDomain ?? DEFAULT_NOREPLY;
    this._fetch = opts.fetchImpl ?? fetch;
    this.gitHost = new URL(this._oauth).host;
  }

  authorizeUrl(state: string, redirectUri: string): string {
    const u = new URL(`${this._oauth}/login/oauth/authorize`);
    u.searchParams.set("client_id", this._clientId);
    u.searchParams.set("redirect_uri", redirectUri);
    u.searchParams.set("state", state);
    // No `scope`: a GitHub App's reach is the App's installed permissions
    // intersected with the user, not an OAuth scope string.
    return u.toString();
  }

  private async _token(body: Record<string, string>): Promise<GitHubTokenSet> {
    let resp: Response;
    try {
      resp = await this._fetch(`${this._oauth}/login/oauth/access_token`, {
        method: "POST",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify({ client_id: this._clientId, client_secret: this._clientSecret, ...body }),
      });
    } catch (e) {
      throw new GitHubApiError(`GitHub token endpoint unreachable: ${(e as Error).message}`);
    }
    if (!resp.ok) throw new GitHubApiError(`GitHub token endpoint returned ${resp.status}`);
    const data = (await resp.json()) as Record<string, unknown>;
    // GitHub answers 200 with an { error } body for a bad/expired grant.
    if (typeof data.error === "string") {
      const err = data.error;
      const detail = typeof data.error_description === "string" ? data.error_description : err;
      if (err === "bad_refresh_token" || err === "bad_verification_code" || err === "unauthorized" || err.includes("expired")) {
        throw new GitHubGrantRevoked(detail);
      }
      throw new GitHubApiError(detail);
    }
    const accessToken = data.access_token;
    if (typeof accessToken !== "string" || !accessToken) throw new GitHubApiError("GitHub returned no access_token");
    const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
    return {
      accessToken,
      accessExpiresInSeconds: num(data.expires_in),
      refreshToken: typeof data.refresh_token === "string" && data.refresh_token ? data.refresh_token : undefined,
      refreshExpiresInSeconds: num(data.refresh_token_expires_in),
    };
  }

  exchangeCode(code: string, redirectUri: string): Promise<GitHubTokenSet> {
    return this._token({ code, redirect_uri: redirectUri });
  }

  refresh(refreshToken: string): Promise<GitHubTokenSet> {
    return this._token({ grant_type: "refresh_token", refresh_token: refreshToken });
  }

  async revokeGrant(accessToken: string): Promise<void> {
    const basic = Buffer.from(`${this._clientId}:${this._clientSecret}`, "utf8").toString("base64");
    let resp: Response;
    try {
      resp = await this._fetch(`${this._api}/applications/${encodeURIComponent(this._clientId)}/grant`, {
        method: "DELETE",
        headers: { Accept: "application/vnd.github+json", Authorization: `Basic ${basic}`, "Content-Type": "application/json" },
        body: JSON.stringify({ access_token: accessToken }),
      });
    } catch (e) {
      throw new GitHubApiError(`GitHub grant-revoke unreachable: ${(e as Error).message}`);
    }
    // 204 = revoked, 404 = already gone. Both are success from our side.
    if (resp.status !== 204 && resp.status !== 404 && resp.status !== 422) {
      throw new GitHubApiError(`GitHub grant-revoke returned ${resp.status}`);
    }
  }

  private async _get(accessToken: string, path: string): Promise<unknown> {
    let resp: Response;
    try {
      resp = await this._fetch(`${this._api}${path}`, {
        headers: { Accept: "application/vnd.github+json", Authorization: `Bearer ${accessToken}`, "X-GitHub-Api-Version": "2022-11-28" },
      });
    } catch (e) {
      throw new GitHubApiError(`GitHub ${path} unreachable: ${(e as Error).message}`);
    }
    if (resp.status === 401) throw new GitHubGrantRevoked(`GitHub ${path} returned 401`);
    if (!resp.ok) throw new GitHubApiError(`GitHub ${path} returned ${resp.status}`);
    return resp.json();
  }

  async fetchIdentity(accessToken: string): Promise<GitHubIdentity> {
    const user = (await this._get(accessToken, "/user")) as Record<string, unknown>;
    const login = typeof user.login === "string" ? user.login : "";
    const id = typeof user.id === "number" ? user.id : 0;
    if (!login || !id) throw new GitHubApiError("GitHub /user returned no login/id");
    const name = typeof user.name === "string" ? user.name : null;
    const noreplyEmail = `${id}+${login}@${this._noreplyDomain}`;
    let installations: GitHubInstallation[] = [];
    try {
      const inst = (await this._get(accessToken, "/user/installations")) as Record<string, unknown>;
      const arr = Array.isArray(inst.installations) ? inst.installations : [];
      installations = arr
        .map((raw): GitHubInstallation | null => {
          const r = raw as Record<string, unknown>;
          const account = r.account as Record<string, unknown> | undefined;
          const acct = account && typeof account.login === "string" ? account.login : "";
          return typeof r.id === "number" ? { id: r.id, account: acct } : null;
        })
        .filter((x): x is GitHubInstallation => x !== null);
    } catch (e) {
      // The installation list is display-only; a grant with no installations
      // (or a transient failure fetching them) still connects.
      if (e instanceof GitHubGrantRevoked) throw e;
    }
    return { login, id, name, noreplyEmail, installations };
  }

  /**
   * GET every page of a list endpoint, following ``Link: rel="next"`` while it
   * stays on the configured API origin, until ``maxItems`` are collected.
   * ``extract`` pulls the items out of one page's body.
   */
  private async _getPaged<T>(accessToken: string, path: string, maxItems: number, extract: (body: unknown) => T[]): Promise<T[]> {
    const out: T[] = [];
    const apiOrigin = new URL(this._api).origin;
    let url: string | null = `${this._api}${path}`;
    // A hard page bound as well as the item bound: a server that answers
    // empty pages with a next link must not loop forever.
    for (let page = 0; url && out.length < maxItems && page < 100; page += 1) {
      let resp: Response;
      try {
        resp = await this._fetch(url, {
          headers: { Accept: "application/vnd.github+json", Authorization: `Bearer ${accessToken}`, "X-GitHub-Api-Version": "2022-11-28" },
        });
      } catch (e) {
        throw new GitHubApiError(`GitHub ${path} unreachable: ${(e as Error).message}`);
      }
      if (resp.status === 401) throw new GitHubGrantRevoked(`GitHub ${path} returned 401`);
      if (!resp.ok) throw new GitHubApiError(`GitHub ${path} returned ${resp.status}`);
      out.push(...extract(await resp.json()));
      const next = nextLink(resp.headers.get("link"));
      url = next && safeOrigin(next) === apiOrigin ? next : null;
    }
    return out.slice(0, maxItems);
  }

  listInstallationRepos(accessToken: string, installationId: number, maxItems: number): Promise<GitHubRepo[]> {
    const path = `/user/installations/${encodeURIComponent(String(installationId))}/repositories?per_page=100`;
    return this._getPaged(accessToken, path, maxItems, (body) => {
      const arr = Array.isArray((body as Record<string, unknown>)?.repositories) ? (body as { repositories: unknown[] }).repositories : [];
      return arr
        .map((raw): GitHubRepo | null => {
          const r = raw as Record<string, unknown>;
          if (typeof r.full_name !== "string" || !r.full_name) return null;
          return {
            fullName: r.full_name,
            private: r.private === true,
            defaultBranch: typeof r.default_branch === "string" && r.default_branch ? r.default_branch : "main",
            pushedAt: typeof r.pushed_at === "string" && r.pushed_at ? r.pushed_at : undefined,
          };
        })
        .filter((x): x is GitHubRepo => x !== null);
    });
  }

  listBranches(accessToken: string, owner: string, repo: string, maxItems: number): Promise<string[]> {
    const path = `/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/branches?per_page=100`;
    return this._getPaged(accessToken, path, maxItems, (body) =>
      (Array.isArray(body) ? body : [])
        .map((raw) => (raw as Record<string, unknown>).name)
        .filter((n): n is string => typeof n === "string" && n.length > 0));
  }
}

/** The ``rel="next"`` URL of a GitHub ``Link`` header, or ``null``. */
export function nextLink(header: string | null): string | null {
  if (!header) return null;
  for (const part of header.split(",")) {
    const m = /<([^>]+)>\s*;\s*rel="?next"?/.exec(part.trim());
    if (m) return m[1]!;
  }
  return null;
}

function safeOrigin(u: string): string | null {
  try { return new URL(u).origin; } catch { return null; }
}
