/**
 * The log view's parser: which format a file is in, where one entry ends
 * and the next begins, and what the filters and search keep.  The viewer
 * is only layout over these, so these are the cases that decide whether a
 * traceback is shown with the line that logged it.
 */
import { describe, expect, it } from "vitest";
import { detectLogFormat, filterEntries, isLogPath, normaliseLevel, parseLog, searchEntries, shortLogger, shortTime, tally } from "./logParse";

const PY = [
  "2026-09-27 07:38:28,512 [INFO] jaato_server.server.session_manager: Session created",
  "2026-09-27 07:38:29,001 [DEBUG] jaato_server.server.runner.rpc: [RPC_DIAG] daemon DISPATCHED id=3",
  "2026-09-27 07:38:30,100 [ERROR] jaato_server.server.core: turn failed: boom",
  "Traceback (most recent call last):",
  '  File "core.py", line 1, in f',
  "ValueError: boom",
  "2026-09-27 07:38:31,000 [WARNING] jaato_server.shared.gc: context at 85%",
  "",
].join("\n");

describe("isLogPath", () => {
  it("takes .log, .jsonl, .ndjson and rotated copies, and nothing else", () => {
    for (const p of ["a.log", ".jaato/logs/s.log", "x.jsonl", "x.ndjson", "server.log.1", "A.LOG"]) expect(isLogPath(p)).toBe(true);
    for (const p of ["log.txt", "a.json", "catalog", "a.log.bak"]) expect(isLogPath(p)).toBe(false);
  });
});

describe("parseLog", () => {
  it("reads the daemon's python format and folds a traceback into its entry", () => {
    const { format, entries } = parseLog(PY);
    expect(format).toBe("python");
    expect(entries).toHaveLength(4);
    expect(entries[2]).toMatchObject({ line: 3, level: "ERROR", logger: "jaato_server.server.core", message: "turn failed: boom" });
    expect(entries[2]!.continuation).toEqual(["Traceback (most recent call last):", '  File "core.py", line 1, in f', "ValueError: boom"]);
    expect(entries[1]!.tag).toBe("RPC_DIAG");
    expect(entries[3]!.continuation).toEqual([]);
  });

  it("keeps text before the first header as an entry with no level", () => {
    const { entries } = parseLog(`startup banner\n${PY}`);
    expect(entries[0]).toMatchObject({ level: null, message: "startup banner", line: 1 });
    expect(entries[1]!.line).toBe(2);
  });

  it("reads jaato_sdk.trace lines, with the component as the logger", () => {
    const { format, entries } = parseLog("[07:38:28.512] [PERMISSION] check_permission: tool=x\n[07:38:28.600] [TOOL_RUNNER] result: ok=True\n  detail\n");
    expect(format).toBe("trace");
    expect(entries.map((e) => e.logger)).toEqual(["PERMISSION", "TOOL_RUNNER"]);
    expect(entries[1]!.continuation).toEqual(["  detail"]);
    expect(entries[0]!.level).toBeNull();
  });

  it("reads JSON lines as records, taking time, level and label from the usual keys", () => {
    const { format, entries } = parseLog('{"ts":"07:00","level":"warn","event":"response","tokens":3}\n{"type":"permission-check","msg":"ok"}\n');
    expect(format).toBe("jsonl");
    expect(entries[0]).toMatchObject({ time: "07:00", level: "WARNING", logger: "response" });
    expect(entries[0]!.record).toEqual({ ts: "07:00", level: "warn", event: "response", tokens: 3 });
    expect(entries[1]!.message).toBe("ok");
  });

  it("decides by content: a .jsonl written in trace lines is trace, prose is plain", () => {
    expect(detectLogFormat("[07:00:00.000] [X] hi\n")).toBe("trace");
    expect(detectLogFormat("just some notes\nabout nothing\n")).toBe("plain");
    expect(parseLog("a\nb\n").entries.map((e) => e.message)).toEqual(["a", "b"]);
  });
});

describe("filters and search", () => {
  const { entries } = parseLog(`banner\n${PY}`);

  it("hides levels and tags, never an entry with no level", () => {
    const kept = filterEntries(entries, { hiddenLevels: new Set(["DEBUG", "INFO"]), logger: null, hiddenTags: new Set() });
    expect(kept.map((e) => e.level)).toEqual([null, "ERROR", "WARNING"]);
    const noDiag = filterEntries(entries, { hiddenLevels: new Set(), logger: null, hiddenTags: new Set(["RPC_DIAG"]) });
    expect(noDiag).toHaveLength(entries.length - 1);
  });

  it("narrows to one logger", () => {
    expect(filterEntries(entries, { hiddenLevels: new Set(), logger: "jaato_server.shared.gc", hiddenTags: new Set() }).map((e) => e.level)).toEqual(["WARNING"]);
  });

  it("searches message, continuation and logger, case-insensitively; empty matches nothing", () => {
    expect(searchEntries(entries, "valueerror")).toEqual([3]);
    expect(searchEntries(entries, "SHARED.GC")).toEqual([4]);
    expect(searchEntries(entries, "  ")).toEqual([]);
  });

  it("counts what the controls show", () => {
    const t = tally(entries);
    expect(t.levels.get("ERROR")).toBe(1);
    expect(t.tags.get("RPC_DIAG")).toBe(1);
    expect(t.loggers.size).toBe(4);
  });
});

describe("column helpers", () => {
  it("shortens times and loggers, keeping what is already short", () => {
    expect(shortTime("2026-09-27 07:38:28,512")).toBe("07:38:28,512");
    expect(shortTime(null)).toBe("");
    expect(shortLogger("jaato_server.server.session_manager")).toBe("server.session_manager");
    expect(shortLogger("root")).toBe("root");
  });

  it("maps level spellings onto five levels", () => {
    expect(["warn", "FATAL", "trace", "err", "info"].map(normaliseLevel)).toEqual(["WARNING", "CRITICAL", "DEBUG", "ERROR", "INFO"]);
    expect(normaliseLevel("notice")).toBeNull();
  });
});
