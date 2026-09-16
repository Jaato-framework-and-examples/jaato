/**
 * Clarification card: one question at a time (with a progress marker),
 * choice questions as buttons, free-text answered from the composer.
 * See ``PendingClarification`` for the two wire shapes it serves; both
 * arrive here already normalised (``protocol/clarification.ts``), so the
 * card reads ``question_text`` / ``options`` whichever wire produced them.
 *
 * A choice button sends the choice's 1-based position, which is what the
 * daemon's answer parser reads.  A ``multiple_choice`` question takes a
 * comma-separated list of positions from the composer instead, so the
 * buttons stay single-pick and the hint says so.
 */
import type { PendingClarification } from "@/store/types";
import { isMultipleChoice } from "@/protocol/clarification";

export function ClarificationPrompt({ c, onAnswer, onCancel }: { c: PendingClarification; onAnswer: (answer: string) => void; onCancel: () => void }) {
  const q = c.questions[c.index];
  const total = c.questions.length || (c.index + 1);
  return (
    <div className="my-2 rounded-lg border border-primary/50 surface-1 overflow-hidden" role="group" aria-label="Clarification">
      <div className="px-3 py-1.5 flex items-center gap-2 border-b hairline text-sm">
        <span className="text-primary">?</span>
        <span>The agent needs clarification</span>
        <span className="text-xs text-text-muted">· {c.index + 1}/{total}{c.toolName ? ` · ${c.toolName}` : ""}</span>
      </div>
      {c.context && c.index === 0 && <div className="px-3 pt-2 text-xs text-text-muted whitespace-pre-wrap">{c.context}</div>}
      <div className="px-3 py-2 whitespace-pre-wrap">{q?.question_text ?? (c.inputMode ? "Please answer in the box below." : "Waiting for the question…")}</div>
      {q && isMultipleChoice(q) && <div className="px-3 pb-1 text-[11px] text-text-muted">Pick one or more: click a choice, or type the numbers comma-separated (e.g. <kbd>1,3</kbd>).</div>}
      {q?.options && q.options.length > 0 && (
        <div className="px-3 pb-2 flex flex-wrap gap-2">
          {q.options.map((o, i) => (
            <button key={i} type="button" onClick={() => onAnswer(String(i + 1))} className={`px-2.5 py-1 rounded-md text-xs border hover:bg-surface text-left ${q.default === i + 1 ? "border-primary/60" : "hairline"}`}>
              <span className="font-mono text-accent mr-1">{i + 1}</span>{o}
              {q.default === i + 1 && <span className="ml-1 text-text-muted">(default)</span>}
              {q.expects_attachment?.[i] && <span className="ml-1 text-text-muted" title="This choice expects you to attach a file">📎</span>}
            </button>
          ))}
        </div>
      )}
      <div className="px-3 pb-2 text-[11px] text-text-muted flex gap-3">
        {q && <span>{q.optional ? "optional (Enter on empty skips)" : "required"}</span>}
        {q?.default != null && <span>default: <kbd>{String(q.default)}</kbd> (Enter on empty)</span>}
        <button type="button" className="underline hover:text-text" onClick={onCancel}>cancel</button>
      </div>
    </div>
  );
}
