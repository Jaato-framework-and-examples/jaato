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
      {/* A long reference name becomes a full-width, multi-line row rather
          than overflowing the plate; any long option tips the set into a
          vertical list, and ``whitespace-normal`` / ``max-w-full`` do the
          wrapping now that ``.btn`` no longer forces ``white-space: nowrap``
          from an unlayered rule (#1245). */}
      <div className={`px-3.5 py-3 gap-2 ${r.options.some((o) => o.length > 40) ? "flex flex-col items-stretch" : "flex flex-wrap"}`}>
        {r.options.map((o, i) => (
          <button key={i} type="button" onClick={() => onRespond(o)} className="btn btn-md normal-case tracking-normal font-mono font-normal text-left whitespace-normal items-start max-w-full break-all">{o}</button>
        ))}
      </div>
    </Plate>
  );
}
