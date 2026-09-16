/**
 * The welcome screen: the front door of the jaato web coding environment.
 *
 * What it says depends on who served the bundle, because that decides what
 * the person has to do to get in:
 *
 * - **Sign-in backend** (``ticketUrl`` in ``config.json`` — the
 *   ``jaato-web-coder-server`` deployment).  The credential is a per-user
 *   ticket the backend mints before every connection attempt
 *   (``app/tickets.ts``), so there is nothing to type: the page greets the
 *   signed-in person (``GET`` on the session endpoint,
 *   ``app/backendSession.ts``), connects on its own and moves on.  A 401
 *   from the ticket endpoint means nobody is signed in, and the whole
 *   screen becomes one **Sign in** call to action — a top-level navigation
 *   to ``loginUrl``, so the OIDC redirect can set the cookie; the page
 *   lands back here and connects.  A "Sign out" link goes to the backend's
 *   logout endpoint.
 * - **Launcher** (``daemon`` / ``token`` in ``config.json`` — ``npx
 *   @jaato/web-coder-ui``).  Pre-filled and, with ``autoConnect``,
 *   connected without a click; a failure shows the error and offers to
 *   retry.
 * - **Nothing** (``npm run dev``, or ``dist/`` on a static host).  The only
 *   way in is to say where the daemon is, so the connection form is open:
 *   WebSocket URL (default ``/ws`` on this origin, the dev proxy path) and
 *   bearer token.
 *
 * In the first two cases the URL and token live behind a "Connection
 * details" disclosure: still reachable, since a wrong daemon address is
 * fixed by typing rather than by redeploying, but not the first thing a
 * person sees.  The build stamp is a footer for the person staring at a
 * connection failure, not part of the welcome.
 */
import { useEffect, useRef, useState } from "react";
import type { TokenProvider } from "@jaato/sdk";
import { connect, probeWorkspaceMode } from "@/sdk/connection";
import { useJaato } from "@/store/store";
import { loadLauncherConfig, type LauncherConfig } from "@/app/launcherConfig";
import { SignInRequiredError, ticketProvider } from "@/app/tickets";
import { fetchSignedInUser, siblingEndpoint } from "@/app/backendSession";
import { buildLine } from "@/app/buildInfo";

function defaultUrl(): string {
  const env = (import.meta as unknown as { env: Record<string, string | undefined> }).env.VITE_WS_URL;
  if (env) return env;
  try {
    const saved = localStorage.getItem("jaato.url");
    if (saved) return saved;
  } catch { /* ignore */ }
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws`;
}

/** The sign-in backend's three endpoints, resolved from what ``config.json`` named. */
interface Backend {
  provider: TokenProvider;
  loginUrl: string;
  sessionUrl: string;
  logoutUrl: string;
}

function backendFrom(cfg: LauncherConfig): Backend | null {
  if (!cfg.ticketUrl) return null;
  const loginUrl = cfg.loginUrl ?? "./auth/login";
  return {
    provider: ticketProvider({ ticketUrl: cfg.ticketUrl, loginUrl }),
    loginUrl,
    sessionUrl: cfg.sessionUrl ?? siblingEndpoint(cfg.ticketUrl, "session"),
    logoutUrl: cfg.logoutUrl ?? siblingEndpoint(cfg.ticketUrl, "logout"),
  };
}

const FEATURES: Array<[string, string]> = [
  ["Sessions", "Start from a profile, or pick up a session where you left it."],
  ["Tools you approve", "Shell, files and the web — each call asks before it runs."],
  ["Plans and subagents", "Watch the plan unfold; every subagent gets its own tab."],
];

const inputClass = "mt-1 w-full rounded-md border hairline bg-bg px-2 py-1.5 font-mono text-[13px] outline-none focus:border-primary/60";

export function ConnectScreen() {
  const [url, setUrl] = useState(defaultUrl);
  const [token, setToken] = useState(() => { try { return sessionStorage.getItem("jaato.token") ?? ""; } catch { return ""; } });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // null until config.json has been read: what the screen leads with depends on it.
  const [launcher, setLauncher] = useState<LauncherConfig | null>(null);
  // Set when config.json names a ticketUrl: the backend mints the credential.
  const [backend, setBackend] = useState<Backend | null>(null);
  const [who, setWho] = useState<string | null>(null);
  const [signInUrl, setSignInUrl] = useState<string | null>(null);
  const conn = useJaato((s) => s.connection);
  const setScreen = useJaato((s) => s.setScreen);
  const launched = useRef(false);

  const go = async (e?: React.FormEvent, override?: { url: string; token: string | TokenProvider }) => {
    e?.preventDefault();
    const u = override?.url ?? url;
    const t = override?.token ?? backend?.provider ?? token;
    setBusy(true);
    setError(null);
    setSignInUrl(null);
    try {
      try {
        localStorage.setItem("jaato.url", u);
        // A ticket is single-use and belongs to the backend; only a typed token is worth remembering.
        if (typeof t === "string") sessionStorage.setItem("jaato.token", t);
      } catch { /* ignore */ }
      await connect({ url: u, token: t || undefined });
      const mode = await probeWorkspaceMode();
      setScreen(mode === "enabled" ? "workspaces" : "session");
    } catch (err) {
      if (err instanceof SignInRequiredError) setSignInUrl(err.loginUrl);
      else setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    void loadLauncherConfig().then((cfg) => {
      if (cancelled || launched.current) return;
      launched.current = true;
      setLauncher(cfg);
      if (!cfg.daemon && !cfg.token && !cfg.ticketUrl) return;
      const nextUrl = cfg.daemon ?? url;
      setUrl(nextUrl);
      let credential: string | TokenProvider = cfg.token ?? token;
      const b = backendFrom(cfg);
      if (b) {
        setBackend(b);
        credential = b.provider;
        // A courtesy, not a gate: the ticket decides whether we get in.
        void fetchSignedInUser(b.sessionUrl).then((s) => { if (!cancelled && s) setWho(s.user); });
      } else if (cfg.token) {
        setToken(cfg.token);
      }
      if (cfg.autoConnect && nextUrl) void go(undefined, { url: nextUrl, token: credential });
    });
    return () => { cancelled = true; };
    // Runs once: the launcher config is a property of the page load, not of the form.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Served by a launcher or a backend, the daemon address is theirs to know; the form is a fallback.
  const hosted = !!launcher && !!(launcher.daemon || launcher.token || launcher.ticketUrl);
  const dev = (import.meta as unknown as { env: { DEV?: boolean } }).env.DEV === true;

  return (
    <div className="h-full flex flex-col items-center justify-center p-6 gap-6">
      <div className="w-full max-w-lg text-center space-y-3">
        <div className="inline-flex items-baseline gap-3">
          <span className="text-4xl font-semibold tracking-tight text-primary">jaato</span>
          <span className="text-xs uppercase tracking-widest text-text-muted">web coding environment</span>
        </div>
        <h1 className="text-xl font-medium">Welcome. Your coding agent, in the browser.</h1>
        <p className="text-sm text-text-muted">
          Everything the terminal client does — sessions, tools, plans, permissions — from a tab, on any machine.
        </p>
      </div>

      <form onSubmit={go} className="w-full max-w-lg rounded-xl border hairline surface-1 p-6 space-y-4">
        {launcher === null ? (
          <div className="text-sm text-text-muted text-center">Preparing your environment…</div>
        ) : signInUrl ? (
          <div className="text-center space-y-3">
            <div className="text-base">Sign in to open your coding environment.</div>
            <a href={signInUrl} className="inline-block rounded-md bg-primary text-bg font-semibold px-6 py-2">Sign in</a>
            <div className="text-xs text-text-muted">Your sign-in backend issues a per-user ticket for each connection; nothing to copy or paste.</div>
          </div>
        ) : (who || busy || (hosted && !error)) ? (
          <div className="text-center space-y-1">
            {who && <div className="text-base">Welcome back, <span className="font-semibold">{who}</span>.</div>}
            {busy ? (
              <div className="text-sm text-text-muted">{conn.detail ?? "Opening your coding environment…"}</div>
            ) : hosted && !error ? (
              <div className="text-sm text-text-muted">Ready when you are.</div>
            ) : null}
          </div>
        ) : null}

        {error && <div className="rounded-md border border-error/40 p-3 text-sm text-error whitespace-pre-wrap">{error}</div>}

        {launcher !== null && !signInUrl && (
          <details open={!hosted} className="group text-sm">
            <summary className={`cursor-pointer select-none text-text-muted ${hosted ? "text-xs" : ""}`}>
              {hosted ? "Connection details" : "Where is your daemon?"}
            </summary>
            <div className="mt-3 space-y-3">
              {!hosted && (
                <div className="text-xs text-text-muted">
                  Point the page at a jaato daemon started with <code className="font-mono">--web-socket</code>.
                </div>
              )}
              <label className="block">
                <span className="text-text-muted">WebSocket URL</span>
                <input value={url} onChange={(e) => setUrl(e.target.value)} className={inputClass} placeholder="ws://host:8080" autoFocus={!hosted} />
              </label>
              {!backend && (
                <label className="block">
                  <span className="text-text-muted">Bearer token <span className="opacity-70">(from <code className="font-mono">~/.jaato/ws.token</code>; leave empty for <code className="font-mono">--ws-unsafe-no-auth</code>)</span></span>
                  <input value={token} onChange={(e) => setToken(e.target.value)} type="password" autoComplete="off" className={inputClass} />
                </label>
              )}
              {dev && !hosted && (
                <div className="text-[11px] text-text-muted">
                  Dev tip: <code className="font-mono">npm run dev</code> proxies <code className="font-mono">/ws</code> to <code className="font-mono">ws://127.0.0.1:8080</code>; set <code className="font-mono">JAATO_WS_TARGET</code> to change it.
                </div>
              )}
            </div>
          </details>
        )}

        {launcher !== null && !signInUrl && (
          <button type="submit" disabled={busy || !url} className="w-full rounded-md bg-primary text-bg font-semibold py-2 disabled:opacity-50">
            {busy ? "Connecting…" : error ? "Try again" : hosted ? "Open my environment" : "Connect"}
          </button>
        )}

        {backend && who && (
          <div className="text-center text-xs text-text-muted">
            Not {who}? <a href={backend.logoutUrl} className="underline hover:text-text">Sign out</a>
          </div>
        )}
      </form>

      <div className="w-full max-w-lg grid grid-cols-1 sm:grid-cols-3 gap-3 text-center">
        {FEATURES.map(([title, blurb]) => (
          <div key={title} className="rounded-lg border hairline p-3">
            <div className="text-sm font-medium">{title}</div>
            <div className="mt-1 text-xs text-text-muted">{blurb}</div>
          </div>
        ))}
      </div>

      <div className="text-[11px] text-text-muted font-mono" data-testid="build-info">{buildLine()}</div>
    </div>
  );
}
