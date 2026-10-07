/**
 * Log files for the Files panel's log view (``LogView``): which paths are
 * logs, how their text splits into ENTRIES, and how entries are filtered
 * and searched.  Pure functions, so the view is only layout.
 *
 * Three line formats are recognised, because three reach a workspace:
 *
 * - **python** -- the daemon's session and server logs,
 *   ``%(asctime)s [%(levelname)s] %(name)s: %(message)s``
 *   (``2026-09-27 07:38:28,512 [INFO] jaato_server.core: message``).
 * - **trace** -- ``jaato_sdk.trace``'s ``JAATO_TRACE_LOG`` /
 *   ``trace.session_log`` lines, ``[07:38:28.512] [COMPONENT] message``.
 *   They carry no level; the component plays the logger's part.
 * - **jsonl** -- one JSON object per line (the ledger, provider traces
 *   written as records).  Time, level and a label are read from the usual
 *   keys when present; the whole record is kept for the expanded view.
 *
 * **An entry is a header line plus everything after it until the next
 * header.**  A traceback, a pretty-printed dict, a multi-line model reply:
 * all of it belongs to the line that logged it, which is what makes the
 * log readable and what a level filter must hide or show TOGETHER.  Lines
 * before the first header are an entry of their own, with no level.
 *
 * The format is decided from the content, not the extension: a
 * ``.jsonl`` provider trace may well be written in the trace line format.
 * Text in none of them is ``plain``, and the viewer shows it as text.
 */

/** Extensions the viewer offers the log view for, including rotated copies (``x.log.1``). */
const LOG_EXT = /\.(log|jsonl|ndjson)(\.\d+)?$/i;

/** True for a path the Files panel viewer shows in the log view by default. */
export function isLogPath(path: string): boolean {
  return LOG_EXT.test(path);
}

/** Normalised severity; an entry with no level (trace lines, prologue text) has ``null``. */
export type LogLevel = "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL";

/** Levels in severity order, as the view lists them. */
export const LOG_LEVELS: readonly LogLevel[] = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"];

export type LogFormat = "python" | "trace" | "jsonl" | "plain";

export interface LogEntry {
  /** Position in the parsed list. */
  index: number;
  /** 1-based line of the entry's header in the file. */
  line: number;
  /** The timestamp as written, or ``null``. */
  time: string | null;
  level: LogLevel | null;
  /** The logger (python), component (trace) or label (jsonl), or ``null``. */
  logger: string | null;
  /** The header line's message. */
  message: string;
  /** Lines after the header that belong to it (a traceback, a dump). */
  continuation: string[];
  /**
   * A leading ``[TAG]`` in the message (``[RPC_DIAG]``, ``[PERMISSION]``),
   * without the brackets -- diagnostics families a reader wants to hide
   * wholesale.
   */
  tag: string | null;
  /** The parsed record, for ``jsonl``. */
  record?: unknown;
}

export interface ParsedLog {
  format: LogFormat;
  entries: LogEntry[];
}

const PYTHON_HEADER = /^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[,.]\d+)?)\s+\[([A-Za-z]+)\]\s+([^\s:]+):\s?(.*)$/;
const TRACE_HEADER = /^\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]\s+\[([^\]]+)\]\s?(.*)$/;
const LEADING_TAG = /^\[([A-Z][A-Z0-9_]{1,31})\]/;

/** ``WARN`` / ``FATAL`` and friends mapped onto the five levels; anything else is ``null``. */
export function normaliseLevel(raw: unknown): LogLevel | null {
  if (typeof raw !== "string") return null;
  switch (raw.trim().toUpperCase()) {
    case "DEBUG": case "TRACE": return "DEBUG";
    case "INFO": return "INFO";
    case "WARN": case "WARNING": return "WARNING";
    case "ERROR": case "ERR": return "ERROR";
    case "CRITICAL": case "FATAL": return "CRITICAL";
    default: return null;
  }
}

function tagOf(message: string): string | null {
  return LEADING_TAG.exec(message)?.[1] ?? null;
}

function isRecordLine(line: string): boolean {
  const t = line.trim();
  if (!t.startsWith("{") || !t.endsWith("}")) return false;
  try { const v: unknown = JSON.parse(t); return typeof v === "object" && v !== null && !Array.isArray(v); } catch { return false; }
}

/**
 * The format the most of the first 200 non-blank lines match, or
 * ``plain`` when none matches any.  Continuation lines match nothing, so a
 * log full of tracebacks is still recognised by its headers.
 */
export function detectLogFormat(text: string): LogFormat {
  const counts = { python: 0, trace: 0, jsonl: 0 };
  let seen = 0;
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    if (++seen > 200) break;
    if (PYTHON_HEADER.test(line)) counts.python++;
    else if (TRACE_HEADER.test(line)) counts.trace++;
    else if (isRecordLine(line)) counts.jsonl++;
  }
  const best = (Object.keys(counts) as (keyof typeof counts)[]).reduce((a, b) => (counts[b] > counts[a] ? b : a));
  return counts[best] > 0 ? best : "plain";
}

function firstString(rec: Record<string, unknown>, keys: string[]): string | null {
  for (const k of keys) {
    const v = rec[k];
    if (typeof v === "string" && v) return v;
    if (typeof v === "number" && Number.isFinite(v)) return String(v);
  }
  return null;
}

function recordEntry(rec: Record<string, unknown>, raw: string, index: number, line: number): LogEntry {
  const message = firstString(rec, ["message", "msg", "text"]) ?? raw.trim();
  return {
    index, line,
    time: firstString(rec, ["ts", "timestamp", "time", "t"]),
    level: normaliseLevel(rec.level ?? rec.severity ?? rec.levelname),
    logger: firstString(rec, ["event", "type", "kind", "component", "logger", "name"]),
    message,
    continuation: [],
    tag: tagOf(message),
    record: rec,
  };
}

/** Split ``text`` into entries in its detected format. */
export function parseLog(text: string): ParsedLog {
  const format = detectLogFormat(text);
  const lines = text.split("\n");
  if (lines.length && lines[lines.length - 1] === "") lines.pop();
  const entries: LogEntry[] = [];
  const push = (e: Omit<LogEntry, "index">) => entries.push({ ...e, index: entries.length });

  if (format === "plain") {
    return { format, entries: lines.map((l, i) => ({ index: i, line: i + 1, time: null, level: null, logger: null, message: l, continuation: [], tag: null })) };
  }

  lines.forEach((raw, i) => {
    const line = i + 1;
    if (format === "jsonl") {
      if (isRecordLine(raw)) {
        entries.push(recordEntry(JSON.parse(raw.trim()) as Record<string, unknown>, raw, entries.length, line));
        return;
      }
    } else {
      const m = (format === "python" ? PYTHON_HEADER : TRACE_HEADER).exec(raw);
      if (m) {
        if (format === "python") {
          const message = m[4] ?? "";
          push({ line, time: m[1]!, level: normaliseLevel(m[2]), logger: m[3]!, message, continuation: [], tag: tagOf(message) });
        } else {
          const message = m[3] ?? "";
          push({ line, time: m[1]!, level: null, logger: m[2]!, message, continuation: [], tag: tagOf(message) });
        }
        return;
      }
    }
    const last = entries[entries.length - 1];
    if (last) last.continuation.push(raw);
    else push({ line, time: null, level: null, logger: null, message: raw, continuation: [], tag: null });
  });
  // Trailing blank lines are the file's ending, not part of the last entry.
  const last = entries[entries.length - 1];
  while (last && last.continuation.length && !last.continuation[last.continuation.length - 1]!.trim()) last.continuation.pop();
  return { format, entries };
}

/**
 * The part of a timestamp worth a column: ``07:38:28,512`` from
 * ``2026-09-27 07:38:28,512``.  The full value stays in the row's title.
 */
export function shortTime(time: string | null): string {
  if (!time) return "";
  const m = /(\d{2}:\d{2}:\d{2}(?:[,.]\d{1,3})?)/.exec(time);
  return m ? m[1]! : time;
}

/**
 * A dotted logger name shortened to fit a column: its last two segments
 * (``jaato_server.server.session_manager`` → ``server.session_manager``).
 */
export function shortLogger(logger: string | null): string {
  if (!logger) return "";
  const parts = logger.split(".");
  return parts.length > 2 ? parts.slice(-2).join(".") : logger;
}

export interface LogFilter {
  /** Levels NOT shown.  An entry with no level is never hidden by level. */
  hiddenLevels: ReadonlySet<LogLevel>;
  /** Show only this logger; ``null`` for all. */
  logger: string | null;
  /** Tags NOT shown (``RPC_DIAG``). */
  hiddenTags: ReadonlySet<string>;
}

/** The entries ``filter`` keeps, in order. */
export function filterEntries(entries: readonly LogEntry[], filter: LogFilter): LogEntry[] {
  return entries.filter((e) =>
    !(e.level && filter.hiddenLevels.has(e.level)) &&
    !(filter.logger !== null && e.logger !== filter.logger) &&
    !(e.tag && filter.hiddenTags.has(e.tag)));
}

/** How many entries carry each level, logger and tag -- the counts the view's controls show. */
export function tally(entries: readonly LogEntry[]): { levels: Map<LogLevel, number>; loggers: Map<string, number>; tags: Map<string, number> } {
  const levels = new Map<LogLevel, number>();
  const loggers = new Map<string, number>();
  const tags = new Map<string, number>();
  for (const e of entries) {
    if (e.level) levels.set(e.level, (levels.get(e.level) ?? 0) + 1);
    if (e.logger) loggers.set(e.logger, (loggers.get(e.logger) ?? 0) + 1);
    if (e.tag) tags.set(e.tag, (tags.get(e.tag) ?? 0) + 1);
  }
  return { levels, loggers, tags };
}

/**
 * Positions (into ``entries``) of the entries containing ``query``,
 * case-insensitively, in the message, the continuation or the logger.
 * An empty query matches nothing: search is a way to jump, not a filter.
 */
export function searchEntries(entries: readonly LogEntry[], query: string): number[] {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  const out: number[] = [];
  entries.forEach((e, i) => {
    if (e.message.toLowerCase().includes(q) || (e.logger?.toLowerCase().includes(q) ?? false) || e.continuation.some((l) => l.toLowerCase().includes(q))) out.push(i);
  });
  return out;
}
