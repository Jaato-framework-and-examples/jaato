/**
 * The GitHub grants a signed-in user has connected, and the per-workspace
 * bindings that point at them — kept by THIS application, never by the
 * daemon, in one encrypted file.
 *
 * This is ``credentials.ts`` for a different secret shape.  A provider key is
 * one opaque string; a GitHub App grant is a rotating refresh token plus a
 * cached 8-hour access token plus non-secret display metadata (the ``login``,
 * the installation list, the noreply email seeded into ``.gitconfig``).  So
 * the envelope is the same — AES-256-GCM under a key HKDF-derived from a 0600
 * secret file, the AAD binding each ciphertext to its owner and id, atomic
 * temp-file-plus-rename at mode 0600, one process owning the file — but the
 * plaintext holds the metadata and the ciphertext holds a JSON secret blob.
 *
 * Two keyings, and they are deliberately different, because two different
 * questions ask them.
 *
 * - A **grant** is keyed by the OIDC ``sub`` (``owner``), the stable id the
 *   issuer promises to keep across a display-name remap — the reason
 *   ``credentials.ts`` keys by ``sub``.  Connect, list, disconnect and set-
 *   default all hold an OIDC session, so ``sub`` is in hand.  Each grant also
 *   carries a globally-unique ``id`` and the ``user`` (the ``subject_claim``
 *   the daemon speaks), so a binding can find it without ``sub``.
 * - A **binding** ``(user, workspace) -> grantId`` is keyed by ``user`` — the
 *   identity the DAEMON speaks (``ticket.bind``'s ``user``, and the ``user``
 *   half of a workspace owner ``app:user``).  ``secret.resolve`` arrives with
 *   only ``user`` and ``workspace`` and no OIDC session, so this is the only
 *   keying it can use.  Keying bindings by ``user`` is consistent with the
 *   daemon, whose whole ownership model is ``app:user``.
 *
 * The link between them is the ``grantId`` written into the binding by an
 * authenticated session that held BOTH identities: at bind time this store's
 * caller has verified the grant is owned by the binding writer's ``sub``, so
 * resolving a binding to its grant by id is safe even though the resolve path
 * knows only ``user``.
 */
import { createCipheriv, createDecipheriv, hkdfSync, randomBytes } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import type { GitHubInstallation } from "./github-api.js";

export class GitHubStoreError extends Error {
  override name = "GitHubStoreError";
  constructor(message: string, readonly status: 400 | 404 | 409 = 400) { super(message); }
}

/** The secret half of a grant — encrypted at rest, never leaves the server. */
export interface GrantSecret {
  refreshToken?: string;
  /** Epoch ms; undefined = the refresh token does not expire (or none exists). */
  refreshExpiresAt?: number;
  /** The cached user-to-server access token, reused until near expiry. */
  accessToken?: string;
  /** Epoch ms; undefined = the access token does not expire (App token expiration off). */
  accessExpiresAt?: number;
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

/** A ``(user, workspace) -> grant`` binding, as a client sees it. */
export interface GitHubBinding {
  workspace: string;
  accountId: string;
}

/** Fields a caller supplies when connecting or re-connecting an account. */
export interface GrantInput {
  login: string;
  githubId: number;
  name: string | null;
  noreplyEmail: string;
  installations: GitHubInstallation[];
  secret: GrantSecret;
}

interface StoredGrant extends Omit<GitHubAccount, "isDefault"> {
  owner: string;
  /** The daemon-facing identity (subject_claim), for resolve-time lookup by ``user``. */
  user: string;
  githubId: number;
  isDefault: boolean;
  /** base64url: 12-byte nonce, 16-byte tag, ciphertext (a JSON GrantSecret). */
  nonce: string;
  tag: string;
  ciphertext: string;
}

interface StoredBinding {
  /** The daemon-facing identity the daemon resolves against. */
  user: string;
  /** The owning OIDC sub, recorded for reconciliation; not the lookup key. */
  owner: string;
  workspace: string;
  grantId: string;
  updatedAt: string;
}

interface StoreFile { version: 1; grants: StoredGrant[]; bindings: StoredBinding[] }

export const MAX_ACCOUNTS_PER_OWNER = 20;

const b64 = (b: Buffer) => b.toString("base64url");
const unb64 = (s: string) => Buffer.from(s, "base64url");

export class FileGitHubStore {
  private readonly _key: Buffer;
  private _grants: StoredGrant[] = [];
  private _bindings: StoredBinding[] = [];

  /**
   * @param path   the JSON file; created on first write, directory too
   * @param secret the operator's key material (a 0600 secret file's content, >= 32 chars);
   *               the AES key is derived with HKDF under this store's own info string,
   *               so one secret file can serve this and the credential/note stores
   *               without any of them sharing a key
   */
  constructor(private readonly path: string, secret: string, private readonly now: () => Date = () => new Date()) {
    if (secret.length < 32) throw new GitHubStoreError("github key must be at least 32 characters");
    this._key = Buffer.from(hkdfSync("sha256", secret, "", "jaato-web-coder github v1", 32));
    this._load();
  }

  private _load(): void {
    if (!existsSync(this.path)) return;
    const raw = JSON.parse(readFileSync(this.path, "utf8")) as Partial<StoreFile>;
    if (raw.version !== 1 || !Array.isArray(raw.grants) || !Array.isArray(raw.bindings)) {
      throw new GitHubStoreError(`${this.path}: not a github store (version ${String(raw.version)})`);
    }
    this._grants = raw.grants;
    this._bindings = raw.bindings;
  }

  private _save(): void {
    mkdirSync(dirname(this.path), { recursive: true, mode: 0o700 });
    const tmp = join(dirname(this.path), `.${randomBytes(6).toString("hex")}.tmp`);
    const body: StoreFile = { version: 1, grants: this._grants, bindings: this._bindings };
    writeFileSync(tmp, JSON.stringify(body, null, 1) + "\n", { mode: 0o600 });
    renameSync(tmp, this.path);
  }

  private _aad(owner: string, id: string): Buffer {
    return Buffer.from(`${owner}\0${id}`, "utf8");
  }

  private _encrypt(owner: string, id: string, secret: GrantSecret): Pick<StoredGrant, "nonce" | "tag" | "ciphertext"> {
    const nonce = randomBytes(12);
    const c = createCipheriv("aes-256-gcm", this._key, nonce);
    c.setAAD(this._aad(owner, id));
    const ct = Buffer.concat([c.update(Buffer.from(JSON.stringify(secret), "utf8")), c.final()]);
    return { nonce: b64(nonce), tag: b64(c.getAuthTag()), ciphertext: b64(ct) };
  }

  private _decryptSecret(g: StoredGrant): GrantSecret {
    const d = createDecipheriv("aes-256-gcm", this._key, unb64(g.nonce));
    d.setAAD(this._aad(g.owner, g.id));
    d.setAuthTag(unb64(g.tag));
    return JSON.parse(Buffer.concat([d.update(unb64(g.ciphertext)), d.final()]).toString("utf8")) as GrantSecret;
  }

  private _public(g: StoredGrant): GitHubAccount {
    return {
      id: g.id, login: g.login, name: g.name, noreplyEmail: g.noreplyEmail,
      installations: g.installations, isDefault: g.isDefault, createdAt: g.createdAt, updatedAt: g.updatedAt,
    };
  }

  // ---- grants -------------------------------------------------------------

  /** Accounts a signed-in user has connected, default first, then oldest-first. */
  listAccounts(owner: string): GitHubAccount[] {
    return this._grants
      .filter((g) => g.owner === owner)
      .sort((a, b) => (a.isDefault === b.isDefault ? a.createdAt.localeCompare(b.createdAt) : a.isDefault ? -1 : 1))
      .map((g) => this._public(g));
  }

  /** Whether ``user`` (the daemon-facing identity) has ANY connected account — for ``not_connected``. */
  userHasAnyAccount(user: string): boolean {
    return this._grants.some((g) => g.user === user);
  }

  /**
   * Connect (or re-connect) an account.  A second connect of the SAME GitHub
   * login for one owner UPDATES that grant in place (fresh tokens, refreshed
   * metadata) rather than adding a duplicate row; the first account an owner
   * connects becomes their default.
   */
  connect(owner: string, user: string, input: GrantInput): GitHubAccount {
    if (!owner) throw new GitHubStoreError("owner required");
    if (!user) throw new GitHubStoreError("user required");
    const existing = this._grants.find((g) => g.owner === owner && g.githubId === input.githubId);
    const stamp = this.now().toISOString();
    if (existing) {
      existing.user = user;
      existing.login = input.login;
      existing.name = input.name;
      existing.noreplyEmail = input.noreplyEmail;
      existing.installations = input.installations;
      existing.updatedAt = stamp;
      Object.assign(existing, this._encrypt(owner, existing.id, input.secret));
      this._save();
      return this._public(existing);
    }
    if (this._grants.filter((g) => g.owner === owner).length >= MAX_ACCOUNTS_PER_OWNER) {
      throw new GitHubStoreError(`at most ${MAX_ACCOUNTS_PER_OWNER} GitHub accounts per user; disconnect one first`, 409);
    }
    const id = randomBytes(12).toString("base64url");
    const isDefault = this._grants.filter((g) => g.owner === owner).length === 0;
    const grant: StoredGrant = {
      id, owner, user, login: input.login, githubId: input.githubId, name: input.name,
      noreplyEmail: input.noreplyEmail, installations: input.installations, isDefault,
      createdAt: stamp, updatedAt: stamp, ...this._encrypt(owner, id, input.secret),
    };
    this._grants.push(grant);
    this._save();
    return this._public(grant);
  }

  /** The account matching a grant id, for the resolve path (no ``sub`` in hand). */
  accountById(grantId: string): GitHubAccount | null {
    const g = this._grants.find((x) => x.id === grantId);
    return g ? this._public(g) : null;
  }

  /** The secret behind a grant, by id.  Server-internal only; never a client response. */
  secretOf(grantId: string): GrantSecret | null {
    const g = this._grants.find((x) => x.id === grantId);
    return g ? this._decryptSecret(g) : null;
  }

  /** Replace a grant's secret blob (a token rotation).  By id, because the refresh path holds no ``sub``. */
  updateSecret(grantId: string, secret: GrantSecret): void {
    const g = this._grants.find((x) => x.id === grantId);
    if (!g) throw new GitHubStoreError("no such grant", 404);
    Object.assign(g, this._encrypt(g.owner, g.id, secret));
    g.updatedAt = this.now().toISOString();
    this._save();
  }

  /** Make one account the owner's default; the previous default is cleared. */
  setDefault(owner: string, grantId: string): boolean {
    const mine = this._grants.filter((g) => g.owner === owner);
    if (!mine.some((g) => g.id === grantId)) return false;
    for (const g of mine) g.isDefault = g.id === grantId;
    this._save();
    return true;
  }

  /**
   * Forget a grant and every binding pointing at it.  Returns the removed
   * account (for the disconnect flow, which revokes at GitHub) or ``null``.
   * If the default was removed, the oldest remaining account becomes default.
   */
  disconnect(owner: string, grantId: string): { account: GitHubAccount; user: string } | null {
    const g = this._grants.find((x) => x.owner === owner && x.id === grantId);
    if (!g) return null;
    const account = this._public(g);
    const user = g.user;
    this._grants = this._grants.filter((x) => x.id !== grantId);
    this._bindings = this._bindings.filter((b) => b.grantId !== grantId);
    if (g.isDefault) {
      const rest = this._grants.filter((x) => x.owner === owner).sort((a, b) => a.createdAt.localeCompare(b.createdAt));
      if (rest[0]) rest[0].isDefault = true;
    }
    this._save();
    return { account, user };
  }

  // ---- bindings -----------------------------------------------------------

  /** The bindings a signed-in user has set, for the configure form to prefill. */
  listBindings(owner: string): GitHubBinding[] {
    return this._bindings
      .filter((b) => b.owner === owner)
      .sort((a, b) => a.workspace.localeCompare(b.workspace))
      .map((b) => ({ workspace: b.workspace, accountId: b.grantId }));
  }

  /** The grant id bound to a workspace for the DAEMON-facing ``user``, or ``null``. */
  bindingFor(user: string, workspace: string): string | null {
    const b = this._bindings.find((x) => x.user === user && x.workspace === workspace);
    return b ? b.grantId : null;
  }

  /**
   * Bind ``workspace`` to ``grantId`` (or clear it for ``grantId === null``).
   * The grant must be owned by ``owner``; ``user`` is the daemon-facing
   * identity the resolve path will match, recorded alongside so a
   * display-name remap does not silently break the binding the daemon reads.
   */
  bind(owner: string, user: string, workspace: string, grantId: string | null): void {
    this._bindings = this._bindings.filter((b) => !(b.user === user && b.workspace === workspace));
    if (grantId !== null) {
      const g = this._grants.find((x) => x.id === grantId);
      if (!g || g.owner !== owner) throw new GitHubStoreError("no such connected account", 404);
      this._bindings.push({ user, owner, workspace, grantId, updatedAt: this.now().toISOString() });
    }
    this._save();
  }
}
