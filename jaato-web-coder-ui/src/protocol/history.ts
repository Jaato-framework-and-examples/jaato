/**
 * ``HistoryEvent.history`` — the daemon's serialized conversation — read
 * two ways:
 *
 * - **replay**: after ``session attach`` the conversation is rebuilt as
 *   output blocks, so switching sessions shows what was said there (the
 *   TUI's attach does the same through its buffer).  User text becomes a
 *   user block, model text a model block, a function call a collapsed
 *   tool block whose output is the matching response.
 * - **listing**: the explicit ``history`` command renders the TUI's
 *   compact summary instead — role headers, truncated text, call / response
 *   markers and the per-turn token line — so it does not duplicate the
 *   conversation already on screen.
 *
 * Parts are ``{type: text|function_call|function_response, ...}``; the
 * call carries ``id`` / ``name`` / ``args``, the response ``call_id`` /
 * ``name`` and ``result`` (``response`` is tolerated for older daemons).
 */
import type { OutputBlock, ToolBlock } from "@/store/types";
import { SUBAGENT_REPORT_ORIGIN } from "./agentNotes";

type Part = Record<string, unknown>;
type Msg = { role?: string; parts?: Part[] };

function partText(p: Part): string {
  return typeof p.text === "string" ? p.text : "";
}

function resultText(p: Part): string {
  const r = p.result ?? p.response;
  if (r == null) return "";
  if (typeof r === "string") return r;
  try { return JSON.stringify(r, null, 2); } catch { return String(r); }
}

/** Rebuild a conversation as output blocks (ids come from the store's generator). */
export function historyBlocks(history: unknown, agentId: string, nextId: () => string, toolsExpanded: boolean): OutputBlock[] {
  const out: OutputBlock[] = [];
  const openCalls = new Map<string, ToolBlock>();
  const msgs = Array.isArray(history) ? (history as Msg[]) : [];
  for (const m of msgs) {
    const role = String(m.role ?? "");
    let text = "";
    for (const p of m.parts ?? []) {
      const type = String(p.type ?? "");
      if (type === "text") {
        text += partText(p);
      } else if (type === "function_call") {
        if (text.trim()) { out.push(role === "user" ? { id: nextId(), kind: "user", agentId, text } : { id: nextId(), kind: "text", agentId, source: "model", text }); text = ""; }
        const callId = String(p.id ?? p.call_id ?? nextId());
        const block: ToolBlock = {
          id: nextId(), kind: "tool", agentId, callId, toolName: String(p.name ?? "tool"),
          args: (p.args as Record<string, unknown> | undefined) ?? {}, status: "success", startedAt: 0,
          output: "", media: [], expanded: toolsExpanded,
        };
        openCalls.set(callId, block);
        out.push(block);
      } else if (type === "function_response") {
        const callId = String(p.call_id ?? p.id ?? "");
        const block = openCalls.get(callId) ?? [...openCalls.values()].reverse().find((b) => b.toolName === String(p.name ?? "") && !b.output);
        if (block) {
          block.output = resultText(p);
          if (p.is_error === true) block.status = "error";
          openCalls.delete(block.callId);
        }
      }
    }
    if (text.trim()) out.push(role === "user" ? { id: nextId(), kind: "user", agentId, text } : { id: nextId(), kind: "text", agentId, source: "model", text });
  }
  return out;
}

/** The TUI's ``history`` command rendering. */
export function formatHistoryListing(history: unknown, turnAccounting: unknown): string {
  const msgs = Array.isArray(history) ? (history as Msg[]) : [];
  if (msgs.length === 0) return "No conversation history.";
  const acc = Array.isArray(turnAccounting) ? (turnAccounting as Record<string, unknown>[]) : [];
  const lines = [`Conversation History (${msgs.length} messages, ${acc.length} turns):`, ""];
  let turn = 0;
  msgs.forEach((m, i) => {
    const role = String(m.role ?? "unknown");
    lines.push(role === "user" ? "[User]" : role === "model" ? "[Model]" : `[${role}]`);
    for (const p of m.parts ?? []) {
      const type = String(p.type ?? "");
      if (type === "text") {
        const t = partText(p);
        lines.push(`  ${t.length > 500 ? t.slice(0, 500) + "..." : t}`);
      } else if (type === "function_call") lines.push(`  [Function Call: ${String(p.name ?? "unknown")}]`);
      else if (type === "function_response") lines.push(`  [Function Response: ${String(p.name ?? "unknown")}]`);
    }
    const last = i === msgs.length - 1;
    const nextIsUser = !last && String(msgs[i + 1]?.role) === "user";
    if (role === "model" && (last || nextIsUser)) {
      const a = acc[turn];
      if (a) {
        const prompt = Number(a.prompt ?? 0), output = Number(a.output ?? 0);
        const total = Number(a.total ?? prompt + output);
        lines.push(`  --- Turn ${turn + 1}: ${total.toLocaleString()} tokens (in: ${prompt.toLocaleString()}, out: ${output.toLocaleString()}) ---`);
      }
      turn += 1;
    }
  });
  return lines.join("\n");
}

/**
 * One unit of a ``HistoryPageEvent`` (protocol 1.28): the daemon's
 * rendered transcript, cut into pieces a page boundary never splits.
 * ``model`` text has already been through the daemon's output formatter
 * (``<j-code>``, ``<j-table>``...), exactly as the live stream is.
 */
export interface HistoryUnit {
  id?: string;
  kind?: string;
  group?: string;
  turn?: number;
  text?: string;
  tools?: Array<Record<string, unknown>>;
  /** ``"subagent"`` on a ``user`` unit that is a subagent's report, not the user's turn. */
  origin?: string;
}

/**
 * Rebuild one history PAGE as output blocks, oldest first.
 *
 * Consecutive ``model`` units sharing a ``group`` are segments of ONE text
 * part and are joined back into one text block, so a reply split across
 * units renders as the single answer it was.  A ``tools`` unit becomes one
 * finished tool block per call; ``success: null`` (no result recorded, e.g.
 * a call cancelled mid-batch) is drawn as success, as the live tree draws a
 * call that ended without an error.
 */
export function historyPageBlocks(units: unknown, agentId: string, nextId: () => string, toolsExpanded: boolean): OutputBlock[] {
  const out: OutputBlock[] = [];
  let lastGroup: string | null = null;
  for (const u of (Array.isArray(units) ? units : []) as HistoryUnit[]) {
    const kind = String(u.kind ?? "");
    const group = String(u.group ?? "");
    const text = typeof u.text === "string" ? u.text : "";
    const prev = out[out.length - 1];
    if (kind === "user" && u.origin === SUBAGENT_REPORT_ORIGIN) {
      // A subagent's report, not the user's turn: drawn as the live echo is.
      out.push({ id: nextId(), kind: "text", agentId, source: "child", text });
    } else if (kind === "user") {
      out.push({ id: nextId(), kind: "user", agentId, text, echoed: true });
    } else if (kind === "model" || kind === "thinking") {
      if (kind === "model" && prev && prev.kind === "text" && prev.source === "model" && group && group === lastGroup) {
        out[out.length - 1] = { ...prev, text: prev.text + text };
      } else {
        out.push({ id: nextId(), kind: "text", agentId, source: kind, text });
      }
    } else if (kind === "tools") {
      for (const t of u.tools ?? []) {
        const ok = t.success !== false;
        const block: ToolBlock = {
          id: nextId(), kind: "tool", agentId, callId: String(t.call_id ?? nextId()),
          toolName: String(t.tool_name ?? "tool"),
          args: (t.tool_args as Record<string, unknown> | undefined) ?? {},
          status: ok ? "success" : "error", startedAt: 0, output: "", media: [],
          expanded: toolsExpanded || !ok,
          toolClass: (t.tool_class as ToolBlock["toolClass"] | undefined) ?? null,
        };
        out.push(block);
      }
    }
    lastGroup = group;
  }
  return out;
}
