/**
 * The transcript's view model (jaato/#1304 §1): a pure function over
 * ``blocks[agentId]`` (plus the id→name map, ``toolIdNames``) that folds
 * the raw, one-event-per-row block list into the six shapes a reader
 * actually wants -- ``UserMessage``, ``Thinking``, ``AssistantText``,
 * ``ToolGroup``, ``Banner`` and ``SystemNote``.  The reducer and the wire
 * protocol are untouched: this reads ``OutputBlock[]`` exactly as
 * ``store.ts`` already produces it and invents nothing the daemon did not
 * say.
 *
 * Testable without React: no store, no DOM, just arrays and (for the one
 * exception below) a plain ``Map`` the caller owns.
 */
import type { OutputBlock, ToolBlock } from "./types";
import { classifyTool, type ToolClass } from "@/protocol/toolClass";

export interface UserMessageItem {
  kind: "userMessage";
  id: string;
  agentId: string;
  text: string;
  /** 1-based position among this agent's user turns (the pane's ``T<n>``). */
  turn: number;
}

/**
 * A model's reasoning, collapsed by default.  ``startedAt`` is a CLIENT
 * estimate -- the wire carries no timestamp on a text block (the reducer
 * is untouched, so this cannot change that) -- taken the first moment
 * this session observed the block; ``streaming`` is exact, not
 * estimated: true iff this is still the LAST block in the agent's list,
 * i.e. nothing has superseded it yet.
 */
export interface ThinkingItem {
  kind: "thinking";
  id: string;
  agentId: string;
  text: string;
  streaming: boolean;
  startedAt: number | null;
}

export interface AssistantTextItem {
  kind: "assistantText";
  id: string;
  agentId: string;
  text: string;
}

/**
 * One or more tool calls, folded per jaato/#1304 §1:
 *
 * - ``"row"``    -- one call, shown in full (``calls.length === 1``).
 * - ``"fold"``   -- a run's housekeeping calls, collapsed to one summary
 *                   line (``label``); ``calls`` is every call it covers.
 * - ``"recovered"`` -- a call that failed and a LATER call of the same
 *   tool, in the same contiguous run, that succeeded: shown as one muted
 *   line rather than a red row followed by a green one.  ``calls`` is
 *   ``[failed, recovered]``, in that order; only a failure that never
 *   recovers keeps its own red row.
 */
export interface ToolGroupItem {
  kind: "toolGroup";
  id: string;
  agentId: string;
  mode: "row" | "fold" | "recovered";
  toolClass: ToolClass;
  calls: ToolBlock[];
  /** the fold's summary line, or the recovered line's caption. */
  label?: string;
}

export interface BannerItem {
  kind: "banner";
  id: string;
  agentId: string;
  text: string;
  style: string;
}

export interface SystemNoteItem {
  kind: "systemNote";
  id: string;
  agentId: string;
  text: string;
  style: string;
  /** the plugin/source name, for a non-model ``TextBlock`` folded in here. */
  source?: string;
}

export type TranscriptItem =
  | UserMessageItem
  | ThinkingItem
  | AssistantTextItem
  | ToolGroupItem
  | BannerItem
  | SystemNoteItem;

/** ``SystemBlock`` styles severe enough to read as a banner rather than a quiet note. */
const BANNER_STYLES = new Set(["error", "warning"]);

/** block id -> first-observed ms.  The one piece of state this module keeps; see ``buildTranscript``. */
export type ThinkingClock = Map<string, number>;

/**
 * Pairs a failed call with a LATER call of the same display name and
 * status ``"success"`` within one contiguous run of tool calls.  Returns
 * the index pairs (failed, recovered); every index appears at most once
 * on either side, so two failures of the same tool each claim their own
 * (nearest, chronologically first) recovery rather than one recovery
 * covering both.
 */
function pairRecoveries(run: readonly ToolBlock[], names: readonly string[]): Array<[number, number]> {
  const claimed = new Set<number>();
  const pairs: Array<[number, number]> = [];
  for (let i = 0; i < run.length; i += 1) {
    if (claimed.has(i) || run[i]!.status !== "error") continue;
    let match = -1;
    for (let j = i + 1; j < run.length; j += 1) {
      if (claimed.has(j)) continue;
      if (run[j]!.status === "success" && names[j] === names[i]) { match = j; break; }
    }
    if (match === -1) continue;
    claimed.add(i);
    claimed.add(match);
    pairs.push([i, match]);
  }
  return pairs;
}

/** "6 internal calls · createPlan · list_tools ×3 …" */
function foldSummary(calls: readonly ToolBlock[], names: readonly string[]): string {
  const counts = new Map<string, number>();
  const order: string[] = [];
  names.forEach((n) => {
    if (!counts.has(n)) { counts.set(n, 0); order.push(n); }
    counts.set(n, counts.get(n)! + 1);
  });
  const MAX_NAMES = 3;
  const shown = order.slice(0, MAX_NAMES).map((n) => (counts.get(n)! > 1 ? `${n} ×${counts.get(n)}` : n));
  const more = order.length > MAX_NAMES;
  const noun = calls.length === 1 ? "internal call" : "internal calls";
  return [`${calls.length} ${noun}`, ...shown, ...(more ? ["…"] : [])].join(" · ");
}

/**
 * One contiguous run of ``ToolBlock``s (no other block kind between
 * them) folded per the rules above, in a stable order: a recovered pair
 * emits where its SUCCESSFUL call sits, an unclaimed housekeeping call
 * folds into one summary emitted where the FIRST such call in the run
 * sits, and everything else is its own full row.
 */
function flushToolRun(run: ToolBlock[], names: string[], agentId: string): ToolGroupItem[] {
  if (run.length === 0) return [];
  const classes = names.map(classifyTool);
  const recoveries = pairRecoveries(run, names);
  const failedIdx = new Set(recoveries.map(([f]) => f));
  const recoveredAt = new Map(recoveries.map(([f, r]) => [r, f]));

  const firstHousekeeping = run.findIndex((_c, i) => !failedIdx.has(i) && !recoveredAt.has(i) && classes[i] === "housekeeping");
  const housekeepingIdx = run
    .map((_c, i) => i)
    .filter((i) => !failedIdx.has(i) && !recoveredAt.has(i) && classes[i] === "housekeeping");

  const items: ToolGroupItem[] = [];
  for (let i = 0; i < run.length; i += 1) {
    if (failedIdx.has(i)) continue; // represented by its recovery, below
    const failedI = recoveredAt.get(i);
    if (failedI !== undefined) {
      const failed = run[failedI]!;
      const recovered = run[i]!;
      items.push({
        kind: "toolGroup",
        id: `group:${recovered.id}`,
        agentId,
        mode: "recovered",
        toolClass: classes[i]!,
        calls: [failed, recovered],
        label: `retried ${names[i]}`,
      });
      continue;
    }
    if (classes[i] === "housekeeping") {
      if (i === firstHousekeeping) {
        const calls = housekeepingIdx.map((k) => run[k]!);
        const foldedNames = housekeepingIdx.map((k) => names[k]!);
        items.push({
          kind: "toolGroup",
          id: `group:${calls[calls.length - 1]!.id}`,
          agentId,
          mode: "fold",
          toolClass: "housekeeping",
          calls,
          label: foldSummary(calls, foldedNames),
        });
      }
      continue;
    }
    items.push({ kind: "toolGroup", id: run[i]!.id, agentId, mode: "row", toolClass: classes[i]!, calls: [run[i]!] });
  }
  return items;
}

/** Text before the first ``.``/``!``/``?`` or newline, capped, for the collapsed thinking summary. */
export function firstSentence(text: string, max = 80): string {
  const trimmed = text.trim();
  const m = trimmed.match(/^[^.!?\n]*[.!?]?/);
  let s = (m?.[0] ?? trimmed).trim();
  if (!s) s = trimmed;
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}

/**
 * Fold ``blocks`` into the transcript's six item shapes.
 *
 * ``clock`` is the one piece of memory this needs and does not have from
 * the wire: how long ago a thinking block was first seen, so a completed
 * one can say "Thought for Ns".  Callers that care about that number own
 * a persistent ``Map`` (one per agent, surviving across renders); a test
 * -- or a caller that does not care -- can pass a fresh one and get
 * ``startedAt: null`` back for every thinking block, which the renderer
 * reads as "unknown duration" rather than "0s".
 */
export function buildTranscript(
  blocks: readonly OutputBlock[],
  toolIdNames: Readonly<Record<string, string>>,
  clock: ThinkingClock = new Map(),
  now: number = Date.now(),
): TranscriptItem[] {
  const items: TranscriptItem[] = [];
  let turn = 0;
  let run: ToolBlock[] = [];
  let runNames: string[] = [];

  const flush = () => {
    if (run.length) items.push(...flushToolRun(run, runNames, run[0]!.agentId));
    run = [];
    runNames = [];
  };

  blocks.forEach((b, i) => {
    if (b.kind === "tool") {
      run.push(b);
      runNames.push(toolIdNames[b.toolName] ?? b.toolName);
      return;
    }
    flush();
    if (b.kind === "user") {
      turn += 1;
      items.push({ kind: "userMessage", id: b.id, agentId: b.agentId, text: b.text, turn });
      return;
    }
    if (b.kind === "text") {
      if (b.source === "thinking") {
        if (!clock.has(b.id)) clock.set(b.id, now);
        const streaming = i === blocks.length - 1;
        items.push({ kind: "thinking", id: b.id, agentId: b.agentId, text: b.text, streaming, startedAt: clock.get(b.id) ?? null });
        return;
      }
      if (b.source === "model") {
        items.push({ kind: "assistantText", id: b.id, agentId: b.agentId, text: b.text });
        return;
      }
      items.push({ kind: "systemNote", id: b.id, agentId: b.agentId, text: b.text, style: "info", source: b.source });
      return;
    }
    // b.kind === "system"
    if (BANNER_STYLES.has(b.style)) items.push({ kind: "banner", id: b.id, agentId: b.agentId, text: b.text, style: b.style });
    else items.push({ kind: "systemNote", id: b.id, agentId: b.agentId, text: b.text, style: b.style });
  });
  flush();
  return items;
}

/** A thinking block's elapsed seconds right now -- ``null`` when the start was never observed. */
export function thinkingElapsedSeconds(item: ThinkingItem, now: number = Date.now()): number | null {
  if (item.startedAt == null) return null;
  return Math.max(0, (now - item.startedAt) / 1000);
}

