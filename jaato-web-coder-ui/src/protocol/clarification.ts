/**
 * One question shape for the two the daemon speaks.
 *
 * A clarification reaches a client on two wires and they do not spell a
 * question the same way:
 *
 * | wire | text | choices | required |
 * |---|---|---|---|
 * | ``ClarificationBatchEvent.questions[]`` (``question_payload`` in ``shared/plugins/clarification/channels.py``) | ``text`` | ``choices: [{text, default?, expects_attachment?}]`` | ``required`` |
 * | ``ClarificationQuestionEvent`` (per-question, daemon-local) | ``question_text`` | ``options: [{text}]`` or ``string[]`` | — |
 *
 * The card used to read the second vocabulary only, so on the batch wire
 * — the ONLY wire a runner-served session has — it rendered the request's
 * context and then "Please answer in the box below." with the question
 * itself missing, and offered no choice buttons.  The mock daemon had
 * been written to the card's vocabulary rather than the daemon's, which
 * is why the e2e suite never saw it.  Everything the store keeps is run
 * through :func:`normalizeClarificationQuestion`, so the card reads one
 * shape whichever wire produced it.
 */
import type { ClarificationQuestion } from "@/store/types";

type Raw = Record<string, unknown>;

function asText(v: unknown): string | undefined {
  return typeof v === "string" && v ? v : undefined;
}

/** ``choices[]`` / ``options[]`` entries are dicts on the daemon wires and bare strings in older fixtures. */
function choiceText(entry: unknown, position: number): string {
  if (typeof entry === "string") return entry;
  if (entry && typeof entry === "object") return asText((entry as Raw).text) ?? asText((entry as Raw).label) ?? `choice ${position}`;
  return `choice ${position}`;
}

export function normalizeClarificationQuestion(raw: unknown): ClarificationQuestion {
  const r = (raw && typeof raw === "object" ? raw : {}) as Raw;
  const rawChoices = Array.isArray(r.choices) ? r.choices : Array.isArray(r.options) ? r.options : [];
  const options = rawChoices.map((c, i) => choiceText(c, i + 1));
  // The batch wire marks the default ON the choice; older shapes carried a
  // 1-based ``default_choice`` (the TUI's reading) or a free-text ``default``.
  let def: string | number | null = null;
  const flagged = rawChoices.findIndex((c) => !!c && typeof c === "object" && (c as Raw).default === true);
  if (flagged >= 0) def = flagged + 1;
  else if (typeof r.default_choice === "number") def = r.default_choice;
  else if (typeof r.default === "number" || (typeof r.default === "string" && r.default)) def = r.default;
  const expectsAttachment = rawChoices.map((c) => !!c && typeof c === "object" && (c as Raw).expects_attachment === true);
  const optional = r.required === false || r.optional === true;
  return {
    ...r,
    question_text: asText(r.question_text) ?? asText(r.text),
    question_type: asText(r.question_type),
    options,
    default: def,
    optional,
    expects_attachment: expectsAttachment.some(Boolean) ? expectsAttachment : undefined,
  };
}

/** True for the two choice types the daemon's ``QuestionType`` names, plus the mock's older ``choice``. */
export function isChoiceQuestion(q: ClarificationQuestion): boolean {
  return (q.options?.length ?? 0) > 0;
}

export function isMultipleChoice(q: ClarificationQuestion): boolean {
  return q.question_type === "multiple_choice";
}
