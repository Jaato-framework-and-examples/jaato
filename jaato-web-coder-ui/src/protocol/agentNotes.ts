/**
 * Output that is for the AGENT to read, shown collapsed in the chat.
 *
 * Two sources reach the transcript that a person does not need to read
 * line by line:
 *
 * - ``child``: a subagent reporting to this agent (its completion, an
 *   error, a request for permission or clarification, shared context).
 *   The server marks these (``jaato_server/shared/subagent_report.py``):
 *   ``source: "child"`` live, ``origin: "subagent"`` on a history unit.
 *   If the user needs to know something from one, the agent tells them.
 * - ``plan``: the todo reporter's lines (``Plan created: ...``, a step's
 *   progress).  The Plan section of the rail already shows the plan.
 *
 * ``store/transcript.ts`` folds consecutive blocks of one of these sources
 * into one ``collapsedNote`` item; this module writes its one-line summary.
 */

/** ``AgentOutputEvent.source`` values drawn collapsed. */
export const COLLAPSED_SOURCES: ReadonlySet<string> = new Set(["child", "plan"]);

/** ``HistoryPageEvent`` unit ``origin`` the server gives a subagent report. */
export const SUBAGENT_REPORT_ORIGIN = "subagent";

/** A subagent report's header, as the server writes it. */
export interface ReportHeader {
  agentId: string;
  /** ``COMPLETED``, ``ERROR``, ``CLARIFICATION_REQUESTED``, ... or ``""``. */
  event: string;
  /** For a forwarded output line (``source=model``), its source; else ``""``. */
  source: string;
}

const HEADER_RE = /^\[SUBAGENT agent_id=(\S+?)(?: event=([^\]\s]+))?(?: source=([^\]\s]+))?\]/gm;

/** Every report header in ``text``, in order.  A batch can hold several. */
export function reportHeaders(text: string): ReportHeader[] {
  const out: ReportHeader[] = [];
  for (const m of text.matchAll(HEADER_RE)) {
    out.push({ agentId: m[1]!, event: m[2] ?? "", source: m[3] ?? "" });
  }
  return out;
}

/** The first line of ``text`` that is not a report header, trimmed. */
function firstBodyLine(text: string): string {
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (t && !t.startsWith("[SUBAGENT ")) return t;
  }
  return "";
}

function clip(s: string, max: number): string {
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}

/**
 * The collapsed line for a run of ``texts`` from one ``source``.
 *
 * ``names`` maps agent ids to the display names the tabs show, so the
 * summary names a subagent the way the rest of the page does.
 */
export function collapsedSummary(source: string, texts: readonly string[], names: Readonly<Record<string, string>> = {}): string {
  if (source === "plan") {
    const last = texts[texts.length - 1] ?? "";
    const line = clip(firstBodyLine(last), 90);
    return texts.length > 1 ? `Plan · ${texts.length} updates · ${line}` : `Plan · ${line}`;
  }
  const headers = texts.flatMap(reportHeaders);
  const label = (h: ReportHeader) => `${names[h.agentId] ?? h.agentId}${h.event ? ` ${h.event.toLowerCase().replace(/_/g, " ")}` : ""}`;
  const distinct = [...new Set(headers.map(label))];
  const who = distinct.length ? distinct.slice(0, 3).join(", ") + (distinct.length > 3 ? ", …" : "") : "subagent";
  const count = headers.length > 1 ? `${headers.length} reports · ` : "";
  const body = clip(firstBodyLine(texts[texts.length - 1] ?? ""), 70);
  return `Subagent · ${count}${who}${body ? ` · ${body}` : ""}`;
}
