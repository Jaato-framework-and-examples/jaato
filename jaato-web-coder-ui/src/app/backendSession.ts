/**
 * Who is signed in, as the sign-in backend sees it.
 *
 * ``jaato-web-coder-server`` publishes three sibling endpoints beside the
 * ticket URL it names in ``config.json``: ``./api/session`` (``200 {user}``
 * or ``401``), ``./api/ticket`` and ``./api/logout``
 * (``docs/design/web-server-bff.md`` §5.2).  The config carries only the
 * ticket URL, so the other two are derived from it — a backend mounted
 * under ``/app/`` keeps all three together — unless the config names them
 * outright (``sessionUrl`` / ``logoutUrl``).
 *
 * The greeting is a courtesy, never a gate: the credential that actually
 * opens the connection is the ticket (``app/tickets.ts``), so a session
 * endpoint that is missing, answers HTML (the Vite dev server's SPA
 * fallback) or is unreachable yields ``null`` and the page connects
 * anyway.
 */

/** ``./api/ticket`` → ``./api/session``: same directory, another name. */
export function siblingEndpoint(ticketUrl: string, name: string): string {
  const [path] = ticketUrl.split(/[?#]/, 1);
  const slash = (path ?? "").lastIndexOf("/");
  return slash < 0 ? name : `${(path ?? "").slice(0, slash + 1)}${name}`;
}

export interface SignedInUser {
  user: string;
}

/**
 * ``GET sessionUrl`` with the backend's cookie.  Only a JSON ``200`` whose
 * ``user`` is a non-empty string counts as signed in; everything else —
 * a 401, an HTML fallback, a network error — is ``null``.
 */
export async function fetchSignedInUser(sessionUrl: string, fetchImpl: typeof fetch = fetch): Promise<SignedInUser | null> {
  try {
    const res = await fetchImpl(sessionUrl, { credentials: "same-origin", cache: "no-store", headers: { Accept: "application/json" } });
    if (!res.ok) return null;
    if (!(res.headers.get("content-type") ?? "").includes("application/json")) return null;
    const body = (await res.json()) as { user?: unknown };
    return typeof body.user === "string" && body.user ? { user: body.user } : null;
  } catch {
    return null;
  }
}
