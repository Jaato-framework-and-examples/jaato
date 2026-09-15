import type { PendingReferenceSelection } from "@/store/types";

export function ReferenceSelectionPrompt({ r, onRespond }: { r: PendingReferenceSelection; onRespond: (v: string) => void }) {
  return (
    <div className="my-2 rounded-lg border border-secondary/50 surface-1 overflow-hidden" role="group" aria-label="Reference selection">
      <div className="px-3 py-1.5 border-b hairline text-sm">{r.prompt}</div>
      <div className="px-3 py-2 flex flex-wrap gap-2">
        {r.options.map((o, i) => (
          <button key={i} type="button" onClick={() => onRespond(o)} className="px-2.5 py-1 rounded-md text-xs border hairline hover:bg-surface font-mono">{o}</button>
        ))}
      </div>
    </div>
  );
}
