/**
 * The page's side of the per-user ticket flow (protocol 1.10, #1074).
 *
 * When the bundle is served by ``jaato-web-coder-server`` (the sign-in and
 * ticket-custody backend, ``docs/design/web-server-bff.md``), the daemon
 * credential is not a token the page holds but a **ticket** the backend
 * mints per connection: single-use, short-lived, bound to the signed-in
 * user.  ``config.json`` names where to ask (``ticketUrl``), and this module
 * turns that into the ``TokenProvider`` the SDK calls before every
 * connection attempt — the initial connect and each reconnect — so the
 * consumed ticket is never replayed.
 *
 * The request is a same-origin ``POST`` carrying the backend's session
 * cookie.  A ``401`` means there is no signed-in session, and the connect
 * screen turns that into a "Sign in" button rather than an error; every
 * other failure is reported as what it is.
 */
import type { TokenProvider } from "@jaato/sdk";

/** The backend answered 401: no session cookie, or it expired. */
export class SignInRequiredError extends Error {
  constructor(public readonly loginUrl: string) {
    super("Sign in required");
    this.name = "SignInRequiredError";
  }
}

export interface TicketProviderOptions {
  ticketUrl: string;
  loginUrl: string;
  fetchImpl?: typeof fetch;
}

/** Shape of a successful ``ticketUrl`` response body. */
interface TicketResponse {
  ticket?: unknown;
}

export function ticketProvider(opts: TicketProviderOptions): TokenProvider {
  const fetchImpl = opts.fetchImpl ?? fetch;
  return async () => {
    const res = await fetchImpl(opts.ticketUrl, {
      method: "POST",
      credentials: "same-origin",
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    if (res.status === 401) throw new SignInRequiredError(opts.loginUrl);
    if (res.status === 503) throw new Error("The sign-in backend cannot issue a ticket right now (daemon at capacity or unreachable); retrying.");
    if (!res.ok) throw new Error(`Ticket request failed: HTTP ${res.status}`);
    const body = (await res.json()) as TicketResponse;
    if (typeof body.ticket !== "string" || !body.ticket) throw new Error("Ticket response carried no ticket");
    return body.ticket;
  };
}
