/**
 * First screen: where is the daemon, and what token does it want.
 * Defaults to the dev proxy path (``/ws`` on this origin) so
 * ``npm run dev`` against ``python -m server --web-socket :8080`` needs
 * no configuration; ``VITE_WS_URL`` or the form override it.
 */
import { useState } from "react";
import { connect, probeWorkspaceMode } from "@/sdk/connection";
import { useJaato } from "@/store/store";

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

export function ConnectScreen() {
  const [url, setUrl] = useState(defaultUrl);
  const [token, setToken] = useState(() => { try { return sessionStorage.getItem("jaato.token") ?? ""; } catch { return ""; } });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const conn = useJaato((s) => s.connection);
  const setScreen = useJaato((s) => s.setScreen);

  const go = async (e?: React.FormEvent) => {
    e?.preventDefault();
    setBusy(true);
    setError(null);
    try {
      try { localStorage.setItem("jaato.url", url); sessionStorage.setItem("jaato.token", token); } catch { /* ignore */ }
      await connect({ url, token: token || undefined });
      const mode = await probeWorkspaceMode();
      setScreen(mode === "enabled" ? "workspaces" : "session");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="h-full flex items-center justify-center p-6">
      <form onSubmit={go} className="w-full max-w-md rounded-xl border hairline surface-1 p-6 space-y-4">
        <div>
          <div className="text-2xl font-semibold">jaato</div>
          <div className="text-sm text-text-muted">Connect to a jaato daemon started with <code className="font-mono">--web-socket</code>.</div>
        </div>
        <label className="block text-sm">
          <span className="text-text-muted">WebSocket URL</span>
          <input value={url} onChange={(e) => setUrl(e.target.value)} className="mt-1 w-full rounded-md border hairline bg-bg px-2 py-1.5 font-mono text-[13px] outline-none focus:border-primary/60" placeholder="ws://host:8080" autoFocus />
        </label>
        <label className="block text-sm">
          <span className="text-text-muted">Bearer token <span className="opacity-70">(from <code className="font-mono">~/.jaato/ws.token</code>; leave empty for <code className="font-mono">--ws-unsafe-no-auth</code>)</span></span>
          <input value={token} onChange={(e) => setToken(e.target.value)} type="password" autoComplete="off" className="mt-1 w-full rounded-md border hairline bg-bg px-2 py-1.5 font-mono text-[13px] outline-none focus:border-primary/60" />
        </label>
        {error && <div className="text-sm text-error whitespace-pre-wrap">{error}</div>}
        {conn.detail && !error && <div className="text-xs text-text-muted">{conn.detail}</div>}
        <button type="submit" disabled={busy || !url} className="w-full rounded-md bg-primary text-bg font-semibold py-1.5 disabled:opacity-50">{busy ? "Connecting…" : "Connect"}</button>
        <div className="text-[11px] text-text-muted">
          Dev tip: <code className="font-mono">npm run dev</code> proxies <code className="font-mono">/ws</code> to <code className="font-mono">ws://127.0.0.1:8080</code>; set <code className="font-mono">JAATO_WS_TARGET</code> to change it.
        </div>
      </form>
    </div>
  );
}
