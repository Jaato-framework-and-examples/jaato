/**
 * The bundle's one contact with whatever served it.
 *
 * ``bin/jaato-web.js`` (the ``@jaato/web`` launcher) and anyone hosting
 * ``dist/`` by hand may publish a ``config.json`` next to ``index.html``:
 *
 *     {"daemon": "ws://127.0.0.1:8080", "token": "…", "autoConnect": true}
 *
 * ``daemon`` pre-fills the WebSocket URL, ``token`` the bearer token, and
 * ``autoConnect`` makes the connect screen connect without a click.  A
 * sign-in backend (``jaato-web-server``) writes ``ticketUrl`` instead of
 * ``token``: the page then asks that URL for a fresh per-user ticket before
 * every connection attempt (``app/tickets.ts``), and ``loginUrl`` (default
 * ``./auth/login``) is where a 401 sends the person.  All fields are
 * optional; ``ticketUrl`` wins over ``token`` when both are present.  The file is fetched relative to the page (so a
 * bundle served under ``/app/`` looks for ``/app/config.json``), and
 * anything that is not a JSON document — Vite's dev server answers the
 * path with ``index.html``, a static host with 404 — means "no launcher
 * config", never an error.
 */
export interface LauncherConfig {
  daemon?: string;
  token?: string;
  ticketUrl?: string;
  loginUrl?: string;
  autoConnect?: boolean;
}

export function parseLauncherConfig(raw: unknown): LauncherConfig {
  if (!raw || typeof raw !== "object") return {};
  const o = raw as Record<string, unknown>;
  const out: LauncherConfig = {};
  if (typeof o.daemon === "string" && o.daemon) out.daemon = o.daemon;
  if (typeof o.token === "string" && o.token) out.token = o.token;
  if (typeof o.ticketUrl === "string" && o.ticketUrl) out.ticketUrl = o.ticketUrl;
  if (typeof o.loginUrl === "string" && o.loginUrl) out.loginUrl = o.loginUrl;
  if (typeof o.autoConnect === "boolean") out.autoConnect = o.autoConnect;
  return out;
}

export async function loadLauncherConfig(fetchImpl: typeof fetch = fetch): Promise<LauncherConfig> {
  try {
    const res = await fetchImpl("./config.json", { cache: "no-store" });
    if (!res.ok) return {};
    if (!(res.headers.get("content-type") ?? "").includes("application/json")) return {};
    return parseLauncherConfig(await res.json());
  } catch {
    return {};
  }
}
