/**
 * The page's side of the sign-in backend's per-user key store.
 *
 * ``jaato-web-coder-server`` may keep the provider API keys a signed-in
 * user has used before, so a new workspace offers them in a combobox
 * instead of asking for the same key again.  ``config.json`` names where
 * (``credentialsUrl``, beside ``ticketUrl``); without it the configure
 * form shows the plain key field it always had, and nothing here is
 * called.
 *
 * Everything is a same-origin fetch carrying the backend's session
 * cookie, like ``app/tickets.ts``.  The listing carries labels and a
 * masked hint and never a secret; ``revealCredential`` is the one call
 * that returns the key, and the page forwards it to the daemon through
 * ``config.update`` exactly as it forwards a typed one -- the daemon and
 * the SDK know nothing of the store.  A ``401`` means the backend session
 * is gone; every other failure is reported as what it is.
 */
import { SignInRequiredError } from "./tickets";

export interface StoredCredential {
  id: string;
  provider: string;
  label: string;
  /** The last few characters of the key, so two keys of one provider can be told apart. */
  hint: string;
  createdAt: string;
}

export interface CredentialsApi {
  list(provider: string): Promise<StoredCredential[]>;
  add(provider: string, secret: string, label?: string): Promise<StoredCredential>;
  reveal(id: string): Promise<string>;
  remove(id: string): Promise<void>;
}

function entryUrl(base: string, id: string, suffix = ""): string {
  const [path, query = ""] = base.split(/\?(.*)/s, 2);
  return `${path}/${encodeURIComponent(id)}${suffix}${query ? `?${query}` : ""}`;
}

function listUrl(base: string, provider: string): string {
  const u = base.includes("?") ? `${base}&` : `${base}?`;
  return `${u}provider=${encodeURIComponent(provider)}`;
}

async function failure(res: Response, what: string): Promise<Error> {
  if (res.status === 401) return new SignInRequiredError("./auth/login");
  let detail = "";
  try { detail = String(((await res.json()) as { error?: unknown }).error ?? ""); } catch { /* not JSON */ }
  return new Error(`${what}: HTTP ${res.status}${detail ? ` (${detail})` : ""}`);
}

export function credentialsApi(credentialsUrl: string, fetchImpl: typeof fetch = fetch): CredentialsApi {
  const common: RequestInit = { credentials: "same-origin", cache: "no-store" };
  const jsonHeaders = { Accept: "application/json", "Content-Type": "application/json" };
  return {
    async list(provider) {
      const res = await fetchImpl(listUrl(credentialsUrl, provider), { ...common, headers: { Accept: "application/json" } });
      if (!res.ok) throw await failure(res, "Listing stored keys failed");
      const body = (await res.json()) as { entries?: unknown };
      return Array.isArray(body.entries) ? (body.entries as StoredCredential[]).filter((e) => e && typeof e.id === "string") : [];
    },
    async add(provider, secret, label) {
      const res = await fetchImpl(credentialsUrl, { ...common, method: "POST", headers: jsonHeaders, body: JSON.stringify({ provider, secret, label: label || undefined }) });
      if (!res.ok) throw await failure(res, "Storing the key failed");
      const body = (await res.json()) as { entry?: StoredCredential };
      if (!body.entry || typeof body.entry.id !== "string") throw new Error("Storing the key failed: no entry in the answer");
      return body.entry;
    },
    async reveal(id) {
      const res = await fetchImpl(entryUrl(credentialsUrl, id, "/reveal"), { ...common, method: "POST", headers: { Accept: "application/json" } });
      if (!res.ok) throw await failure(res, "Reading the stored key failed");
      const body = (await res.json()) as { secret?: unknown };
      if (typeof body.secret !== "string" || !body.secret) throw new Error("Reading the stored key failed: no secret in the answer");
      return body.secret;
    },
    async remove(id) {
      const res = await fetchImpl(entryUrl(credentialsUrl, id), { ...common, method: "DELETE" });
      if (!res.ok && res.status !== 404) throw await failure(res, "Deleting the stored key failed");
    },
  };
}

/**
 * A stored key's one-line description for a ``<option>``: the label, and
 * the hint when the label does not already end in it.
 */
export function describeCredential(c: StoredCredential): string {
  if (!c.hint || c.label.endsWith(c.hint)) return c.label;
  return `${c.label} (…${c.hint})`;
}
