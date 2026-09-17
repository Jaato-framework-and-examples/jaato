/** A reference-selection plate: the prompt, then each option as a monospace button. */
import type { PendingReferenceSelection } from "@/store/types";
import { Plate } from "@/components/layout/Plate";

export function ReferenceSelectionPrompt({ r, onRespond }: { r: PendingReferenceSelection; onRespond: (v: string) => void }) {
  return (
    <Plate edge="steel" className="my-3" role="group" aria-label="Reference selection">
      <div className="px-3.5 py-2.5 border-b hairline flex items-baseline gap-3">
        <span className="kicker">Reference</span>
        <span className="text-[15px]">{r.prompt}</span>
      </div>
      <div className="px-3.5 py-3 flex flex-wrap gap-2">
        {r.options.map((o, i) => (
          <button key={i} type="button" onClick={() => onRespond(o)} className="btn btn-md normal-case tracking-normal font-mono font-normal">{o}</button>
        ))}
      </div>
    </Plate>
  );
}
