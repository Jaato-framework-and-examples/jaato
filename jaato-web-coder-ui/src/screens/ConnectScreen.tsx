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
 * Drawn as one plate in two columns (design frame 01): the left half is
 * what jaato is — brand, headline, the three things it does as a numbered
 * list — and the right half is the one thing to do to get in.  In the
 * first two cases the URL and token live behind a "Daemon settings"
 * disclosure: still reachable, since a wrong daemon address is fixed by
 * typing rather than by redeploying, but not the first thing a person
 * sees.  The build stamp sits at the foot of the right column as a
 * key/value grid, for the person staring at a connection failure.
 */
import { useEffect, useRef, useState } from "react";
import type { TokenProvider } from "@jaato/sdk";
import { connect, probeWorkspaceMode } from "@/sdk/connection";
import { useJaato } from "@/store/store";
import { consumeExited } from "@/app/exitIntent";
import { loadLauncherConfig, type LauncherConfig } from "@/app/launcherConfig";
import { SignInRequiredError, ticketProvider } from "@/app/tickets";
import { fetchSignedInUser, siblingEndpoint } from "@/app/backendSession";
import { BUILD, buildLine } from "@/app/buildInfo";
import { Plate } from "@/components/layout/Plate";

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
      // A property of the page load, like the rest of the launcher config;
      // the configure form reads it from the store.
      useJaato.getState().setCredentialsUrl(cfg.credentialsUrl ?? null);
      if (!cfg.daemon && !cfg.token && !cfg.ticketUrl) return;
      const nextUrl = cfg.daemon ?? url;
      setUrl(nextUrl);
      let credential: string | TokenProvider = cfg.token ?? token;
      const b = backendFrom(cfg);
      if (b) {
        setBackend(b);
        credential = b.provider;
        // The workspace screen's "Sign out" reads this; who is signed in follows once known.
        useJaato.getState().setBackend({ logoutUrl: b.logoutUrl, user: null });
        // A courtesy, not a gate: the ticket decides whether we get in.
        void fetchSignedInUser(b.sessionUrl).then((s) => {
          if (cancelled || !s) return;
          setWho(s.user);
          useJaato.getState().setBackend({ logoutUrl: b.logoutUrl, user: s.user });
        });
      } else if (cfg.token) {
        setToken(cfg.token);
      }
      // A deliberate ``exit`` brought us here: everything above still applies,
      // but connecting again on our own would undo what the person just did.
      if (cfg.autoConnect && nextUrl && !consumeExited()) void go(undefined, { url: nextUrl, token: credential });
    });
    return () => { cancelled = true; };
    // Runs once: the launcher config is a property of the page load, not of the form.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Served by a launcher or a backend, the daemon address is theirs to know; the form is a fallback.
  const hosted = !!launcher && !!(launcher.daemon || launcher.token || launcher.ticketUrl);
  const dev = (import.meta as unknown as { env: { DEV?: boolean } }).env.DEV === true;

  // What the right column is for, in one line.
  const kicker = launcher === null ? "Preparing" : signInUrl ? "Sign in to continue" : backend ? (who ? "Welcome back" : "Your environment") : hosted ? "Your environment" : "Where is your daemon?";

  return (
    <div className="h-full overflow-auto flex items-center justify-center p-6 sm:p-12">
      <Plate className="w-full max-w-[860px] grid grid-cols-1 md:grid-cols-[1fr_1px_1fr]">
        {/* What this is. */}
        <div className="p-7 flex flex-col gap-4">
          <div className="flex items-baseline gap-2.5">
            <span className="display text-[40px] leading-none">jaato</span>
            <span className="kicker tracking-[0.18em]">web coding environment</span>
          </div>
          <h1 className="display text-[26px] leading-[1.15] m-0">Your coding agent, in the browser.</h1>
          <p className="m-0 text-sm text-text-muted max-w-[34ch]">
            Everything the terminal client does — sessions, tools, plans, permissions — from a tab, on any machine.
          </p>
          <ol className="mt-auto pt-4 border-t hairline list-none m-0 p-0">
            {FEATURES.map(([title, blurb], i) => (
              <li key={title} className={`flex gap-3 py-2 ${i < FEATURES.length - 1 ? "border-b hairline" : ""}`}>
                <span className="font-mono text-[11px] text-steel w-[22px] shrink-0 pt-0.5">{String(i + 1).padStart(2, "0")}</span>
                <div>
                  <div className="chrome text-[13px] font-medium">{title}</div>
                  <div className="text-[13px] text-text-muted">{blurb}</div>
                </div>
              </li>
            ))}
          </ol>
        </div>
        <div className="hidden md:block bg-divider" aria-hidden="true" />

        {/* The one thing to do to get in. */}
        <form onSubmit={go} className="p-7 flex flex-col gap-4 border-t md:border-t-0 hairline">
          <div className="kicker tracking-[0.16em]">{kicker}</div>

          {launcher === null ? (
            <div className="text-sm text-text-muted">Preparing your environment…</div>
          ) : signInUrl ? (
            <>
              <div className="text-sm text-text-muted max-w-[36ch]">Sign in to open your coding environment. Sign in with your usual account; nothing to copy or paste.</div>
              <Plate edge="steel" className="p-0" style={{ "--corner-color": "color-mix(in srgb, var(--c-bg) 60%, transparent)" } as React.CSSProperties}>
                <a href={signInUrl} className="btn btn-primary w-full text-[16px] py-2.5 no-underline border-0">Sign in</a>
              </Plate>
              <div className="text-xs text-text-muted">Your sign-in backend issues a per-user ticket for each connection.</div>
            </>
          ) : (who || busy || (hosted && !error)) ? (
            <div className="space-y-1">
              {who && <div className="text-base">Welcome back, <span className="font-semibold">{who}</span>.</div>}
              {busy ? (
                <div className="text-sm text-text-muted">{conn.detail ?? "Opening your coding environment…"}</div>
              ) : hosted && !error ? (
                <div className="text-sm text-text-muted">Ready when you are.</div>
              ) : null}
            </div>
          ) : null}

          {error && <div className="border border-error/40 p-3 text-sm text-error whitespace-pre-wrap" role="alert">{error}</div>}

          {launcher !== null && !signInUrl && (
            <button type="submit" disabled={busy || !url} className="btn btn-primary text-[16px] py-2.5">
              {busy ? "Connecting…" : error ? "Try again" : hosted ? "Open my environment" : "Connect"}
            </button>
          )}

          {launcher !== null && !signInUrl && (
            <details open={!hosted} className="group border-t hairline pt-3 text-sm">
              <summary className="cursor-pointer select-none list-none flex items-baseline gap-2.5 [&::-webkit-details-marker]:hidden">
                <span className="text-text-muted transition-transform group-open:rotate-90">▸</span>
                <span>
                  <span className="kicker kicker-muted tracking-[0.12em] text-[12px] block">Daemon settings</span>
                  <span className="text-[13px] text-text-muted">{hosted ? "WebSocket URL and bearer token — only if the served target is wrong." : "Point the page at a jaato daemon started with --web-socket."}</span>
                </span>
              </summary>
              <div className="mt-3 space-y-3">
                <label className="block">
                  <span className="field-label">WebSocket URL</span>
                  <input value={url} onChange={(e) => setUrl(e.target.value)} className="input input-mono" placeholder="ws://host:8080" autoFocus={!hosted} />
                </label>
                {!backend && (
                  <label className="block">
                    <span className="field-label">Bearer token <span className="opacity-70">(from <code className="font-mono">~/.jaato/ws.token</code>; leave empty for <code className="font-mono">--ws-unsafe-no-auth</code>)</span></span>
                    <input value={token} onChange={(e) => setToken(e.target.value)} type="password" autoComplete="off" className="input input-mono" />
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

          {backend && who && (
            <div className="text-xs text-text-muted">
              Not {who}? <a href={backend.logoutUrl} className="link">Sign out</a>
            </div>
          )}

          {/* The build stamp: what this page speaks, for the person staring at a failure. */}
          <dl className="kv mt-auto pt-3 border-t hairline text-[11px] m-0" data-testid="build-info" title={buildLine()}>
            <dt>ui</dt><dd className="text-text m-0">{BUILD.ui}</dd>
            <dt>sdk</dt><dd className="text-text m-0">@jaato/sdk {BUILD.sdk}</dd>
            <dt>protocol</dt><dd className="text-text m-0">≥ {BUILD.protocolMin}</dd>
            <dt>commit</dt><dd className="text-text m-0">{BUILD.commit}</dd>
          </dl>
        </form>
      </Plate>
    </div>
  );
}
