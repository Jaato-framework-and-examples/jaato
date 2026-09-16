/**
 * Clarification plate: one question at a time (with a progress marker),
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
import { Plate } from "@/components/layout/Plate";

export function ClarificationPrompt({ c, onAnswer, onCancel }: { c: PendingClarification; onAnswer: (answer: string) => void; onCancel: () => void }) {
  const q = c.questions[c.index];
  const total = c.questions.length || (c.index + 1);
  return (
    <Plate edge="steel" className="my-3" role="group" aria-label="Clarification">
      <div className="px-3.5 py-2.5 flex items-center gap-3 border-b hairline">
        <span className="text-steel">?</span>
        <span className="kicker">The agent needs clarification</span>
        <span className="flex-1" />
        <span className="font-mono text-[11px] text-text-muted">{c.index + 1}/{total}{c.toolName ? ` · ${c.toolName}` : ""}</span>
      </div>
      {c.context && c.index === 0 && <div className="px-3.5 pt-2.5 text-[13px] text-text-muted whitespace-pre-wrap">{c.context}</div>}
      <div className="px-3.5 py-2.5 text-[15px] whitespace-pre-wrap">{q?.question_text ?? (c.inputMode ? "Please answer in the box below." : "Waiting for the question…")}</div>
      {q && isMultipleChoice(q) && <div className="px-3.5 pb-1.5 text-[12px] text-text-muted">Pick one or more: click a choice, or type the numbers comma-separated (e.g. <kbd>1,3</kbd>).</div>}
      {q?.options && q.options.length > 0 && (
        <div className="px-3.5 pb-3 flex flex-wrap gap-2">
          {q.options.map((o, i) => (
            <button key={i} type="button" onClick={() => onAnswer(String(i + 1))} className={`btn btn-md normal-case tracking-normal font-sans font-normal text-left ${q.default === i + 1 ? "btn-steel" : ""}`}>
              <span className="font-mono text-[11px] text-steel">{i + 1}</span>{o}
              {q.default === i + 1 && <span className="text-text-muted text-xs">(default)</span>}
              {q.expects_attachment?.[i] && <span className="text-text-muted" title="This choice expects you to attach a file">📎</span>}
            </button>
          ))}
        </div>
      )}
      <div className="px-3.5 py-2 border-t hairline text-[12px] text-text-muted flex gap-4">
        {q && <span>{q.optional ? "optional (Enter on empty skips)" : "required"}</span>}
        {q?.default != null && <span>default: <kbd>{String(q.default)}</kbd> (Enter on empty)</span>}
        <span className="flex-1" />
        <button type="button" className="link" onClick={onCancel}>cancel</button>
      </div>
    </Plate>
  );
}
