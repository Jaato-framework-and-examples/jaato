import { useEffect, useRef, useState } from "react";
import { getClient } from "@/sdk/connection";
import { useJaato } from "@/store/store";
import { downloadFromPanel } from "@/app/downloads";
import { isSearchable, matchesOf, SEARCH_DEBOUNCE_MS, servesFileSearch, sizeText, summaryText, type FileMatch } from "@/app/fileSearch";
import { effectiveHidden } from "./WorkspacePanel";

interface Answer { query: string; matches: FileMatch[]; summary: string; error?: string }

const ACTION = "font-heading uppercase tracking-[0.08em] text-[10px] px-1 text-text-muted hover:text-steel";

/**
 * The Files panel's finder: locate ANY file in the workspace by name,
 * changed or not, hidden or not (protocol 1.32).
 *
 * Typing asks the daemon after a short pause; an answer to an older query
 * than the one in the field is dropped, so fast typing never shows stale
 * rows.  Each row offers ``view`` (the panel's viewer) and ``download``;
 * a credential file (a ``.env``, a stored ``*_auth.json``) is listed but
 * offers neither, because the fetch verb refuses it.  A file the panel
 * hides is marked ``H`` -- the finder is how a hidden entry is reached.
 * Escape clears the field.  Drawn only against a daemon that serves the
 * verb.
 */
export function FileFinder({ onView }: { onView: (path: string) => void }) {
  const serves = useJaato((s) => s.connection.phase === "connected" && servesFileSearch(s.connection.protocolVersion));
  const hidden = useJaato((s) => s.workspaceHidden);
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [busy, setBusy] = useState(false);
  const latest = useRef("");

  useEffect(() => {
    const q = query.trim();
    latest.current = q;
    if (!isSearchable(q)) { setAnswer(null); setBusy(false); return; }
    setBusy(true);
    const timer = setTimeout(() => {
      void (async () => {
        try {
          const event = await getClient().searchWorkspaceFiles(q);
          if (latest.current !== q) return;
          if (!event.ok) {
            setAnswer({ query: q, matches: [], summary: "", error: event.error || "search failed" });
          } else {
            const matches = matchesOf(event);
            setAnswer({ query: q, matches, summary: summaryText(event, matches.length) });
          }
        } catch (err) {
          if (latest.current !== q) return;
          setAnswer({ query: q, matches: [], summary: "", error: err instanceof Error ? err.message : String(err) });
        } finally {
          if (latest.current === q) setBusy(false);
        }
      })();
    }, SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [query]);

  if (!serves) return null;
  return (
    <div className="mb-2.5">
      <input
        type="search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Escape") setQuery(""); }}
        placeholder="Find any file…"
        aria-label="Find a file in the workspace"
        className="w-full font-mono text-[11.5px] px-2 py-1 border border-border bg-bg text-text focus:outline-none focus:border-steel"
      />
      {busy && !answer && <div className="mt-1 text-[11px] text-text-muted">Searching…</div>}
      {answer?.error && <div role="status" className="mt-1 text-[11px] text-error">{answer.error}</div>}
      {answer && !answer.error && (
        <div className="mt-1">
          <ul className="list-none m-0 p-0" aria-label="Matching files">
            {answer.matches.map((m) => {
              const hid = effectiveHidden(m.path, hidden);
              return (
                <li key={m.path} className="group flex items-center gap-1 font-mono text-[11.5px] leading-[1.8]">
                  {hid && <span className="text-text-muted" title="Hidden in the panel">H</span>}
                  <span className="truncate" title={m.path}>{m.path}</span>
                  <span className="text-text-muted text-[10px] shrink-0">{sizeText(m.size)}</span>
                  <span className="ml-auto shrink-0 flex gap-1">
                    {m.credential ? (
                      <span className="text-text-muted text-[10px]" title="Holds credentials; it cannot be viewed or downloaded here">credentials</span>
                    ) : (
                      <>
                        <button type="button" className={ACTION} onClick={() => onView(m.path)} aria-label={`View ${m.path}`}>view</button>
                        <button type="button" className={ACTION} onClick={() => { void downloadFromPanel(m.path); }} aria-label={`Download ${m.path}`}>download</button>
                      </>
                    )}
                  </span>
                </li>
              );
            })}
          </ul>
          <div className="text-[11px] text-text-muted">{answer.summary}</div>
        </div>
      )}
    </div>
  );
}
