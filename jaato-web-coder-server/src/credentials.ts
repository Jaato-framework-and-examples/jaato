/**
 * Provider API keys a signed-in user has used before, kept by THIS
 * application rather than by the daemon.
 *
 * Why here and not in the SDK or the daemon.  The daemon persists a key
 * where ``config.update`` puts it: in the selected workspace's ``.env``,
 * under the provider's first credential variable.  That is the right
 * place for a session to read it from and the wrong place to remember it
 * for the person, so every new workspace asked for the same key again.
 * Who the person is, across workspaces, is a fact only this server holds
 * (the OIDC ``sub`` behind the session cookie), and the daemon's
 * application-credential connection is deliberately unable to drive a
 * session -- so the vault is application state, reached over the same
 * cookie-authenticated, same-origin routes as ``/api/ticket``, and the
 * browser forwards a chosen key to the daemon exactly as it forwards a
 * typed one.  No protocol change, no daemon change.
 *
 * Trust posture, stated rather than implied.  Entries are encrypted at
 * rest (AES-256-GCM, a key derived from a 0600 secret file the operator
 * provisions beside the other secrets), so a copied backup or a
 * world-readable directory does not hand out keys.  The server can
 * decrypt every entry -- it has to, to answer ``reveal`` -- so this is
 * the same posture as the workspace ``.env`` the daemon writes today, and
 * NOT per-user cryptographic isolation.  ``pass``/GnuPG was considered
 * and rejected for this unattended path: gpg-agent's ``max-cache-ttl`` is
 * absolute, so a store that needs a passphrase eventually asks pinentry
 * on a host with nobody at the terminal, and a passphrase-less key
 * protects exactly what a 0600 file protects.
 *
 * Owner is the OIDC ``sub``, not the display claim: ``subject_claim`` is
 * configurable and a display name can be re-mapped, while ``sub`` is what
 * the issuer promises to keep stable.  Nothing about the owner is
 * derivable from the file without the key -- the AAD binds each
 * ciphertext to its owner, provider and id, so editing the plaintext
 * metadata to re-attribute an entry makes it undecryptable rather than
 * somebody else's.
 *
 * Lifecycle.  ``FileCredentialStore`` loads the file once at construction
 * and holds the entries in memory; every mutation rewrites the whole file
 * atomically (temp file + rename, mode 0600).  One process owns the file,
 * the same single-writer assumption ``SessionStore`` makes.
 */
import { createCipheriv, createDecipheriv, hkdfSync, randomBytes, timingSafeEqual } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

/** What a client may see: never the secret. */
export interface CredentialEntry {
  id: string;
  provider: string;
  /** Free text the user chose, or the auto label ``<provider> …<hint>``. */
  label: string;
  /** The last few characters of the secret, so two keys of one provider can be told apart. */
  hint: string;
  createdAt: string;
}

export class CredentialError extends Error {
  override name = "CredentialError";
  constructor(message: string, readonly status: 400 | 404 | 409 = 400) { super(message); }
}

export interface CredentialStore {
  list(owner: string, provider?: string): CredentialEntry[];
  /** Store a secret; the same secret stored twice for one owner and provider is ONE entry. */
  add(owner: string, provider: string, secret: string, label?: string): CredentialEntry;
  /** The secret behind an entry, or ``null`` when the owner has no such entry. */
  reveal(owner: string, id: string): string | null;
  remove(owner: string, id: string): boolean;
}

/** Provider names as the daemon spells them (``zhipuai``, ``google_genai``, ...). */
const PROVIDER_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;
export const MAX_LABEL_CHARS = 64;
export const MAX_SECRET_CHARS = 4096;
/** Per owner; a combobox longer than this is not a combobox. */
export const MAX_ENTRIES_PER_OWNER = 50;

interface StoredEntry extends CredentialEntry {
  owner: string;
  /** base64url: 12-byte nonce, 16-byte tag, ciphertext. */
  nonce: string;
  tag: string;
  ciphertext: string;
}

interface StoreFile { version: 1; entries: StoredEntry[] }

/** ``sk-abc…wxyz`` → ``wxyz``; a secret too short to hint at safely hints nothing. */
export function secretHint(secret: string): string {
  return secret.length >= 12 ? secret.slice(-4) : "";
}

export function autoLabel(provider: string, secret: string): string {
  const h = secretHint(secret);
  return h ? `${provider} …${h}` : provider;
}

export function validateProvider(provider: unknown): string {
  if (typeof provider !== "string" || !PROVIDER_RE.test(provider)) throw new CredentialError("provider must be a daemon provider name (lower-case letters, digits, _ or -)");
  return provider;
}

export function validateSecret(secret: unknown): string {
  if (typeof secret !== "string") throw new CredentialError("secret must be a string");
  const s = secret.trim();
  if (!s) throw new CredentialError("secret must not be empty");
  if (s.length > MAX_SECRET_CHARS) throw new CredentialError(`secret longer than ${MAX_SECRET_CHARS} characters`);
  if (/[\r\n\0]/.test(s)) throw new CredentialError("secret must be one line");
  return s;
}

export function validateLabel(label: unknown, fallback: string): string {
  if (label === undefined || label === null || label === "") return fallback;
  if (typeof label !== "string") throw new CredentialError("label must be a string");
  const l = label.trim().replace(/\s+/g, " ");
  if (!l) return fallback;
  if (l.length > MAX_LABEL_CHARS) throw new CredentialError(`label longer than ${MAX_LABEL_CHARS} characters`);
  if (/[\p{Cc}]/u.test(l)) throw new CredentialError("label must not contain control characters");
  return l;
}

const b64 = (b: Buffer) => b.toString("base64url");
const unb64 = (s: string) => Buffer.from(s, "base64url");

export class FileCredentialStore implements CredentialStore {
  private readonly _key: Buffer;
  private _entries: StoredEntry[] = [];

  /**
   * @param path   the JSON file; created on first write, directory too
   * @param secret the operator's key material (a 0600 secret file's content, >= 32 chars);
   *               the AES key is derived from it with HKDF so the file never holds it raw
   */
  constructor(private readonly path: string, secret: string, private readonly now: () => Date = () => new Date()) {
    if (secret.length < 32) throw new CredentialError("credential key must be at least 32 characters");
    this._key = Buffer.from(hkdfSync("sha256", secret, "", "jaato-web-coder credentials v1", 32));
    this._load();
  }

  private _load(): void {
    if (!existsSync(this.path)) return;
    const raw = JSON.parse(readFileSync(this.path, "utf8")) as Partial<StoreFile>;
    if (raw.version !== 1 || !Array.isArray(raw.entries)) throw new CredentialError(`${this.path}: not a credential store (version ${String(raw.version)})`);
    this._entries = raw.entries;
  }

  private _save(): void {
    mkdirSync(dirname(this.path), { recursive: true, mode: 0o700 });
    const tmp = join(dirname(this.path), `.${randomBytes(6).toString("hex")}.tmp`);
    const body: StoreFile = { version: 1, entries: this._entries };
    writeFileSync(tmp, JSON.stringify(body, null, 1) + "\n", { mode: 0o600 });
    renameSync(tmp, this.path);
  }

  private _aad(owner: string, provider: string, id: string): Buffer {
    return Buffer.from(`${owner}\0${provider}\0${id}`, "utf8");
  }

  private _decrypt(e: StoredEntry): string {
    const d = createDecipheriv("aes-256-gcm", this._key, unb64(e.nonce));
    d.setAAD(this._aad(e.owner, e.provider, e.id));
    d.setAuthTag(unb64(e.tag));
    return Buffer.concat([d.update(unb64(e.ciphertext)), d.final()]).toString("utf8");
  }

  private _public(e: StoredEntry): CredentialEntry {
    return { id: e.id, provider: e.provider, label: e.label, hint: e.hint, createdAt: e.createdAt };
  }

  list(owner: string, provider?: string): CredentialEntry[] {
    return this._entries
      .filter((e) => e.owner === owner && (provider === undefined || e.provider === provider))
      .sort((a, b) => a.createdAt.localeCompare(b.createdAt))
      .map((e) => this._public(e));
  }

  add(owner: string, provider: string, secret: string, label?: string): CredentialEntry {
    if (!owner) throw new CredentialError("owner required");
    validateProvider(provider);
    const s = validateSecret(secret);
    const l = validateLabel(label, autoLabel(provider, s));
    // The same key typed again is the same entry, so the combobox does not
    // grow a row per workspace it was used in.  Compared on the decrypted
    // value in constant time; a store holds tens of entries, not millions.
    const mine = this._entries.filter((e) => e.owner === owner && e.provider === provider);
    const target = Buffer.from(s, "utf8");
    for (const e of mine) {
      const have = Buffer.from(this._decrypt(e), "utf8");
      if (have.length === target.length && timingSafeEqual(have, target)) {
        if (label && e.label !== l) { e.label = l; this._save(); }
        return this._public(e);
      }
    }
    if (this._entries.filter((e) => e.owner === owner).length >= MAX_ENTRIES_PER_OWNER) {
      throw new CredentialError(`at most ${MAX_ENTRIES_PER_OWNER} stored keys per user; delete one first`, 409);
    }
    const id = randomBytes(12).toString("base64url");
    const nonce = randomBytes(12);
    const c = createCipheriv("aes-256-gcm", this._key, nonce);
    c.setAAD(this._aad(owner, provider, id));
    const ciphertext = Buffer.concat([c.update(target), c.final()]);
    const entry: StoredEntry = {
      id, owner, provider, label: l, hint: secretHint(s), createdAt: this.now().toISOString(),
      nonce: b64(nonce), tag: b64(c.getAuthTag()), ciphertext: b64(ciphertext),
    };
    this._entries.push(entry);
    this._save();
    return this._public(entry);
  }

  reveal(owner: string, id: string): string | null {
    const e = this._entries.find((x) => x.owner === owner && x.id === id);
    return e ? this._decrypt(e) : null;
  }

  remove(owner: string, id: string): boolean {
    const before = this._entries.length;
    this._entries = this._entries.filter((x) => !(x.owner === owner && x.id === id));
    if (this._entries.length === before) return false;
    this._save();
    return true;
  }
}
