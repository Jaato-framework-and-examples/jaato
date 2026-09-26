/**
 * The API key a provider is used with, chosen in the list box of stored
 * keys (``components/workspace/CredentialPicker``) -- the same control the
 * workspace configure form has always had, now also in the session
 * picker's New session column (#1332 follow-up: the redesign had dropped
 * it).
 *
 * A key reaches the daemon one way: as ``api_key`` on ``config.update``,
 * written to the selected workspace's ``.env`` under the variable the
 * provider reads.  The picker sends it ``key_only`` (protocol 1.26), which
 * writes the key and nothing else -- the workspace's ``JAATO_PROVIDER`` /
 * ``MODEL_NAME`` stay as they are and no server is bootstrapped -- right
 * before ``session.new``, so the session it starts resolves that key.
 */
import { EventTypeValue, isProtocolCompatible } from "@jaato/sdk";
import { credentialsApi } from "@/app/credentials";
import type { KeyChoice } from "@/components/workspace/CredentialPicker";
import { getClient } from "@/sdk/connection";

/** The protocol that knows ``config.update``'s ``key_only``. */
export const MIN_KEY_ONLY_PROTOCOL = "1.26";

export function servesKeyOnly(protocolVersion: string | null | undefined): boolean {
  return !!protocolVersion && isProtocolCompatible(protocolVersion, MIN_KEY_ONLY_PROTOCOL);
}

/**
 * Turn a choice into the key to send.  A stored entry is revealed now (the
 * secret never sat in the page before); a new one is stored first when a
 * key store exists, so the next workspace offers it.  A failure to STORE
 * does not stop the key being applied -- ``onNotice`` says so.
 */
export async function resolveKeyChoice(
  credentialsUrl: string | null,
  provider: string,
  choice: KeyChoice,
  onNotice: (text: string) => void = () => undefined,
): Promise<{ apiKey?: string; storedId?: string }> {
  const api = credentialsUrl ? credentialsApi(credentialsUrl) : null;
  if (choice.kind === "stored") return { apiKey: api ? await api.reveal(choice.id) : undefined };
  if (choice.kind === "new" && choice.secret.trim()) {
    const secret = choice.secret.trim();
    let storedId: string | undefined;
    if (api && provider) {
      try { storedId = (await api.add(provider, secret, choice.label.trim() || undefined)).id; }
      catch (err) { onNotice(`Key applied, but not stored for later: ${err instanceof Error ? err.message : String(err)}`); }
    }
    return { apiKey: secret, storedId };
  }
  return {};
}

/**
 * Write ``apiKey`` for ``provider`` into the selected workspace's ``.env``
 * and nothing else.  Resolves once the daemon confirms; rejects with the
 * daemon's refusal (a provider that takes no key from the environment, no
 * workspace selected) or on a timeout, so the caller does not start a
 * session on a key that never landed.
 */
export function applySessionKey(provider: string, apiKey: string, timeoutMs = 10_000): Promise<void> {
  const c = getClient();
  return new Promise((resolve, reject) => {
    let done = false;
    const finish = (err?: Error) => {
      if (done) return;
      done = true;
      offOk();
      offErr();
      clearTimeout(timer);
      if (err) reject(err);
      else resolve();
    };
    const offOk = c.subscribe(EventTypeValue.CONFIG_UPDATED, (ev) => {
      const e = ev as unknown as { success?: boolean; error?: string | null };
      finish(e.success === false ? new Error(e.error || "The daemon did not store the API key.") : undefined);
    });
    const offErr = c.subscribe(EventTypeValue.ERROR, (ev) => finish(new Error(String((ev as unknown as { error?: string }).error ?? "The daemon refused the API key."))));
    const timer = setTimeout(() => finish(new Error("The daemon did not confirm the API key.")), timeoutMs);
    c.sendRawEvent({ type: EventTypeValue.CONFIG_UPDATE_REQUEST, provider, api_key: apiKey, key_only: true } as never).catch((err) => finish(err instanceof Error ? err : new Error(String(err))));
  });
}
