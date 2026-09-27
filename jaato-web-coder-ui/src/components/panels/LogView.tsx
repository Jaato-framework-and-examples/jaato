/**
 * A log file in the Files panel's content viewer (``WorkspacePanel.tsx``),
 * shown as ENTRIES rather than wrapped text.  Loaded lazily, like the
 * markdown and image views.
 *
 * ``parseLog`` (``protocol/logParse``) splits the text: a header line plus
 * the lines after it that belong to it (a traceback, a dump).  Each entry
 * is one unwrapped row -- time, level, logger, message -- with a ``+N``
 * badge when it carries more lines; clicking the row expands it to the
 * whole message, the continuation and, for a JSONL record, the record
 * pretty-printed.
 *
 * The controls above the rows:
 *
 * - **level chips** hide or show a level, with its count.  An entry with
 *   no level (a trace line, the text before the first header) is never
 *   hidden by level.
 * - **logger** narrows to one logger (or trace component).
 * - **tag chips** hide a diagnostics family by the ``[TAG]`` its messages
 *   start with -- ``[RPC_DIAG]`` is most of a busy session log.
 * - **search** jumps between matching entries (Enter / Shift+Enter, or
 *   the arrows) and counts them; it does not filter, so the context around
 *   a match stays visible.
 * - **follow** keeps the view on the last entry while the parent re-fetches
 *   the file as it grows (``onReload``), and **reload** fetches it once.
 *
 * Rows are virtualised (``@tanstack/react-virtual``): a session log runs to
 * tens of thousands of lines and only the rows on screen are rendered.
 *
 * Text in none of the recognised formats is shown as it is.
 */
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  LOG_LEVELS, filterEntries, parseLog, searchEntries, shortLogger, shortTime, tally,
  type LogEntry, type LogLevel,
} from "@/protocol/logParse";

export interface LogViewProps {
  /** The file's text. */
  text: string;
  /** A Tailwind height class for the rows' scroll area (the viewer's expand state). */
  heightClass: string;
  /** Whether the parent re-fetches the file as it changes, and the view sticks to the end. */
  follow: boolean;
  onFollowChange: (follow: boolean) => void;
  /** Fetch the file again now. */
  onReload: () => void;
}

const LEVEL_CLASS: Record<LogLevel, string> = {
  DEBUG: "text-text-muted",
  INFO: "text-steel",
  WARNING: "text-warning",
  ERROR: "text-error",
  CRITICAL: "text-error font-semibold",
};

const LEVEL_SHORT: Record<LogLevel, string> = { DEBUG: "DBG", INFO: "INF", WARNING: "WRN", ERROR: "ERR", CRITICAL: "CRT" };

const CHIP = "px-1.5 py-0.5 border hairline text-[10.5px] font-mono leading-none";
const BTN = "link text-[11px]";

/** Tag chips beyond this many (by count) are not offered; the logger filter still reaches them. */
const MAX_TAG_CHIPS = 6;

function toggled<T>(set: ReadonlySet<T>, v: T): Set<T> {
  const next = new Set(set);
  if (next.has(v)) next.delete(v); else next.add(v);
  return next;
}

function EntryDetail({ entry }: { entry: LogEntry }) {
  const body = [entry.message, ...entry.continuation].join("\n");
  return (
    <pre className="m-0 mt-0.5 mb-1 ml-3 px-2 py-1 border-l-2 border-divider whitespace-pre-wrap break-words text-[11px]">
      {entry.record !== undefined ? JSON.stringify(entry.record, null, 2) : body}
    </pre>
  );
}

function EntryRow({ entry, expanded, current, onToggle }: { entry: LogEntry; expanded: boolean; current: boolean; onToggle: () => void }) {
  const more = entry.continuation.length;
  const expandable = more > 0 || entry.record !== undefined || entry.message.length > 120;
  const tone = entry.level === "ERROR" || entry.level === "CRITICAL" ? "bg-error/5" : entry.level === "WARNING" ? "bg-warning/5" : "";
  return (
    <div className={`${tone} ${current ? "outline outline-1 outline-steel -outline-offset-1" : ""}`} data-testid="log-entry" data-level={entry.level ?? ""}>
      <button
        type="button"
        className="w-full text-left flex items-baseline gap-2 px-2 py-px hover:bg-steel/5 disabled:cursor-default"
        onClick={onToggle}
        disabled={!expandable}
        aria-expanded={expandable ? expanded : undefined}
        title={`line ${entry.line}${entry.time ? ` · ${entry.time}` : ""}${entry.logger ? ` · ${entry.logger}` : ""}`}
      >
        {entry.time !== null && <span className="text-text-muted shrink-0">{shortTime(entry.time)}</span>}
        {entry.level && <span className={`shrink-0 ${LEVEL_CLASS[entry.level]}`}>{LEVEL_SHORT[entry.level]}</span>}
        {entry.logger && <span className="shrink-0 max-w-[12rem] truncate text-text-muted">{shortLogger(entry.logger)}</span>}
        <span className="flex-1 min-w-0 truncate whitespace-pre">{entry.message}</span>
        {more > 0 && <span className="shrink-0 text-[10px] text-text-muted border hairline px-1" aria-label={`${more} more lines`}>+{more}</span>}
      </button>
      {expanded && <EntryDetail entry={entry} />}
    </div>
  );
}

export default function LogView({ text, heightClass, follow, onFollowChange, onReload }: LogViewProps) {
  const parsed = useMemo(() => parseLog(text), [text]);
  const counts = useMemo(() => tally(parsed.entries), [parsed]);
  const [hiddenLevels, setHiddenLevels] = useState<ReadonlySet<LogLevel>>(new Set());
  const [hiddenTags, setHiddenTags] = useState<ReadonlySet<string>>(new Set());
  const [logger, setLogger] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<ReadonlySet<number>>(new Set());
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);

  const visible = useMemo(() => filterEntries(parsed.entries, { hiddenLevels, logger, hiddenTags }), [parsed, hiddenLevels, logger, hiddenTags]);
  const matches = useMemo(() => searchEntries(visible, query), [visible, query]);
  const matchSet = useMemo(() => new Set(matches), [matches]);
  const current = matches.length ? matches[Math.min(cursor, matches.length - 1)]! : -1;

  const parentRef = useRef<HTMLDivElement>(null);
  const virtualizer = useVirtualizer({
    count: visible.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 18,
    overscan: 12,
    getItemKey: (i) => visible[i]?.index ?? i,
  });

  // Open at the end, as ``tail`` does, and stay there while following.
  const opened = useRef(false);
  useLayoutEffect(() => {
    if (!visible.length) return;
    if (!opened.current || follow) {
      opened.current = true;
      // After layout: on the first render the scroll element has no size
      // yet, and a scroll asked for then is lost.
      const last = visible.length - 1;
      requestAnimationFrame(() => virtualizer.scrollToIndex(last, { align: "end" }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible.length, text, follow]);

  useEffect(() => { setCursor(0); }, [query]);
  const jump = (step: number) => {
    if (!matches.length) return;
    const next = (Math.min(cursor, matches.length - 1) + step + matches.length) % matches.length;
    setCursor(next);
    if (follow) onFollowChange(false);
    virtualizer.scrollToIndex(matches[next]!, { align: "center" });
  };
  const onSearchKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") { e.preventDefault(); jump(e.shiftKey ? -1 : 1); }
  };

  if (parsed.format === "plain") {
    return <pre className={`m-0 px-2 py-2 text-[11.5px] font-mono whitespace-pre-wrap break-words ${heightClass} overflow-auto`}>{text}</pre>;
  }

  const tags = [...counts.tags.entries()].sort((a, b) => b[1] - a[1]).slice(0, MAX_TAG_CHIPS);
  const loggers = [...counts.loggers.entries()].sort((a, b) => b[1] - a[1]);

  return (
    <div data-testid="log-view" data-format={parsed.format}>
      <div className="flex flex-wrap items-center gap-1.5 px-2 py-1.5 border-b hairline">
        {LOG_LEVELS.filter((l) => counts.levels.has(l)).map((l) => {
          const shown = !hiddenLevels.has(l);
          return (
            <button key={l} type="button" className={`${CHIP} ${shown ? LEVEL_CLASS[l] : "text-text-muted line-through opacity-60"}`} aria-pressed={shown} onClick={() => setHiddenLevels((s) => toggled(s, l))} title={`${shown ? "Hide" : "Show"} ${l} entries`}>
              {l} {counts.levels.get(l)}
            </button>
          );
        })}
        {tags.map(([t, n]) => {
          const shown = !hiddenTags.has(t);
          return (
            <button key={t} type="button" className={`${CHIP} ${shown ? "" : "text-text-muted line-through opacity-60"}`} aria-pressed={shown} onClick={() => setHiddenTags((s) => toggled(s, t))} title={`${shown ? "Hide" : "Show"} entries starting [${t}]`}>
              [{t}] {n}
            </button>
          );
        })}
        {loggers.length > 1 && (
          <select className="text-[11px] font-mono bg-transparent border hairline max-w-[14rem]" value={logger ?? ""} onChange={(e) => setLogger(e.target.value || null)} aria-label="Logger">
            <option value="">all {parsed.format === "trace" ? "components" : "loggers"}</option>
            {loggers.map(([name, n]) => <option key={name} value={name}>{name} ({n})</option>)}
          </select>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-2 px-2 py-1 border-b hairline">
        <input
          type="search"
          className="flex-1 min-w-[8rem] text-[11.5px] font-mono bg-transparent border hairline px-1.5 py-0.5"
          placeholder="Search the log"
          aria-label="Search the log"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onSearchKey}
        />
        {query.trim() && (
          <span className="text-[11px] font-mono text-text-muted" aria-live="polite" data-testid="log-match-count">
            {matches.length ? `${Math.min(cursor, matches.length - 1) + 1}/${matches.length}` : "0 matches"}
          </span>
        )}
        <button type="button" className={BTN} onClick={() => jump(-1)} disabled={!matches.length} aria-label="Previous match">↑</button>
        <button type="button" className={BTN} onClick={() => jump(1)} disabled={!matches.length} aria-label="Next match">↓</button>
        <button type="button" className={BTN} onClick={() => onFollowChange(!follow)} aria-pressed={follow} title="Re-fetch the file as it changes and stay on the last entry">
          {follow ? "following" : "follow"}
        </button>
        <button type="button" className={BTN} onClick={onReload} title="Fetch the file again">reload</button>
        <span className="text-[11px] text-text-muted" data-testid="log-shown">
          {visible.length === parsed.entries.length ? `${parsed.entries.length} entries` : `${visible.length} of ${parsed.entries.length} entries`}
        </span>
      </div>
      <div ref={parentRef} className={`${heightClass} overflow-auto font-mono text-[11.5px]`} data-testid="log-rows">
        {visible.length === 0 ? (
          <div className="px-2 py-2 text-text-muted italic">No entries match the filters.</div>
        ) : (
          <div style={{ height: virtualizer.getTotalSize(), position: "relative" }}>
            {virtualizer.getVirtualItems().map((v) => {
              const entry = visible[v.index]!;
              return (
                <div key={v.key} data-index={v.index} ref={virtualizer.measureElement} style={{ position: "absolute", top: 0, left: 0, right: 0, transform: `translateY(${v.start}px)` }} className={matchSet.has(v.index) && v.index !== current ? "border-l-2 border-steel/60" : ""}>
                  <EntryRow entry={entry} expanded={expanded.has(entry.index)} current={v.index === current} onToggle={() => setExpanded((s) => toggled(s, entry.index))} />
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
