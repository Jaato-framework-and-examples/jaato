/**
 * Remember a key typed at the prompt, so the store learns it too.
 *
 * ``<provider>-auth key <secret>`` is the TUI's way of storing an API key
 * daemon-side, and the web prompt sends it the same way.  The daemon then
 * answers with its ``auth.setup`` offer, which names the PROVIDER the key
 * belongs to -- something the command text alone cannot say reliably
 * (``github-auth`` serves ``github_models``).  So the secret is parked
 * when the command is sent and filed into the backend's key store
 * (``app/credentials.ts``) when the offer arrives, under the provider the
 * daemon named.  A typed key that the daemon refused never produces an
 * offer, so it is never stored; a parked secret older than the window is
 * dropped unread.
 *
 * Best effort in both directions: storing is a convenience for the next
 * workspace, so a failure is swallowed rather than shown beside a sign-in
 * that succeeded.
 */
import type { PendingPostAuthSetup } from "@/store/types";
import { useJaato } from "@/store/store";
import { type CredentialsApi, credentialsApi } from "./credentials";

const AUTH_COMMAND_RE = /^[a-z0-9_-]+-auth$/i;
/** How long a parked secret waits for the daemon's offer. */
export const CAPTURE_WINDOW_MS = 120_000;

let pending: { secret: string; at: number } | null = null;

/** Called with the parsed server command as it is sent; parks the secret of a ``*-auth key <secret>``. */
export function noteAuthKeyCommand(command: string | undefined, args: string[] | undefined, now = Date.now()): void {
  if (!command || !AUTH_COMMAND_RE.test(command)) return;
  if ((args?.[0] ?? "").toLowerCase() !== "key") return;
  const secret = args?.[1];
  if (!secret) return;
  pending = { secret, at: now };
  ensureInstalled();
}

/** The parked secret, once, if it is still inside the window. */
export function takePendingAuthKey(now = Date.now()): string | null {
  const p = pending;
  pending = null;
  return p && now - p.at <= CAPTURE_WINDOW_MS ? p.secret : null;
}

/**
 * The daemon's offer arrived: file the parked secret under the provider it
 * names.  Returns what was stored for tests; ``null`` when nothing was.
 */
export async function captureOnPostAuth(postAuth: PendingPostAuthSetup | null, api: CredentialsApi | null, now = Date.now()): Promise<{ provider: string } | null> {
  if (!postAuth || !postAuth.providerName) return null;
  const secret = takePendingAuthKey(now);
  if (!secret || !api) return null;
  try {
    await api.add(postAuth.providerName, secret);
    return { provider: postAuth.providerName };
  } catch {
    return null;
  }
}

let installed = false;
function ensureInstalled(): void {
  if (installed) return;
  installed = true;
  let last = useJaato.getState().postAuth;
  useJaato.subscribe((s) => {
    if (s.postAuth === last) return;
    last = s.postAuth;
    void captureOnPostAuth(s.postAuth, s.credentialsUrl ? credentialsApi(s.credentialsUrl) : null);
  });
}
