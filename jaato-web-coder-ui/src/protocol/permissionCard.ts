/**
 * Pure helpers for the regrouped permission card (jaato/#1304 §2, the
 * final rollout item from the closed issue #1304): sorting the daemon's
 * flat ``response_options`` into the groups the card draws --
 * **Allow** (primary), an **"Allow for…"** dropdown (turn / idle /
 * always / all, plus the destructive **never**), **Deny** (primary), a
 * **note** field (the ``c`` / ``yc`` comment variants), and a **hidden**
 * group (``once`` / edit) -- and the risk tag + plain-language question a
 * tool's coarse class implies.
 *
 * Grouping is BY KEY, against the exact vocabulary
 * ``DEFAULT_PERMISSION_OPTIONS`` declares
 * (``jaato_server/shared/plugins/permission/channels.py``) -- never by
 * matching a label's text, which a deployment could reword.  A key this
 * table has no opinion about is never dropped: it lands in ``other``, so
 * a minimal option set (a test's bare ``y``/``n``) degrades to just the
 * two primary buttons, and a daemon that adds a new option kind later is
 * still visible rather than silently missing an action a person might
 * need.
 */
import type { PermissionOption } from "@/store/types";
import type { ToolClass } from "./toolClass";

export interface GroupedPermissionOptions {
  /** "y" -- always present unless the daemon sent a genuinely custom set. */
  allow: PermissionOption | null;
  /** "n". */
  deny: PermissionOption | null;
  /** "t", "i", "a", "all", in that display order -- the "Allow for…" dropdown's scoped grants. */
  durations: PermissionOption[];
  /** "never" -- rendered in the SAME dropdown as the durations, marked destructive. */
  destructive: PermissionOption | null;
  /** "yc" -- allow, with feedback the model sees. */
  allowComment: PermissionOption | null;
  /** "c" -- deny, with feedback the model sees. */
  denyComment: PermissionOption | null;
  /** "once", "e" (edit) -- present but not shown until asked for. */
  hidden: PermissionOption[];
  /** Anything the vocabulary above has no opinion about -- never silently dropped. */
  other: PermissionOption[];
}

const DURATION_ORDER = ["t", "i", "a", "all"];
const HIDDEN_KEYS = new Set(["once", "e"]);

export function groupPermissionOptions(options: readonly PermissionOption[]): GroupedPermissionOptions {
  const byKey = new Map(options.map((o) => [o.key, o] as const));
  const durations = DURATION_ORDER.map((k) => byKey.get(k)).filter((o): o is PermissionOption => !!o);
  const known = new Set<string>(["y", "n", "never", "yc", "c", ...DURATION_ORDER, ...HIDDEN_KEYS]);
  return {
    allow: byKey.get("y") ?? null,
    deny: byKey.get("n") ?? null,
    durations,
    destructive: byKey.get("never") ?? null,
    allowComment: byKey.get("yc") ?? null,
    denyComment: byKey.get("c") ?? null,
    hidden: options.filter((o) => HIDDEN_KEYS.has(o.key)),
    other: options.filter((o) => !known.has(o.key)),
  };
}

/** A plain-language question, per the tool's coarse class -- what a
 * reader who does not know the tool's name is actually being asked. */
const QUESTIONS: Record<ToolClass, string> = {
  housekeeping: "Let the agent do this bookkeeping step?",
  write: "Let the agent change this file?",
  exec: "Let the agent run this?",
  read: "Let the agent read this?",
  agent: "Let the agent hand this off to another session?",
  other: "Allow this tool call?",
};

export function plainQuestion(cls: ToolClass): string {
  return QUESTIONS[cls];
}

export interface RiskTag {
  label: string;
  tone: "muted" | "warning" | "steel";
}

/** A one-word risk tag, from the same coarse class the transcript folds
 * housekeeping calls by -- no separate risk taxonomy invented here. */
const RISK_TAGS: Record<ToolClass, RiskTag> = {
  housekeeping: { label: "Routine", tone: "muted" },
  read: { label: "Read-only", tone: "muted" },
  write: { label: "Changes a file", tone: "warning" },
  exec: { label: "Runs a command", tone: "warning" },
  agent: { label: "Delegates work", tone: "steel" },
  other: { label: "Tool call", tone: "muted" },
};

export function riskTag(cls: ToolClass): RiskTag {
  return RISK_TAGS[cls];
}
