/**
 * One ``ToolGroupItem`` (``store/transcript.ts``): a full row, or one of
 * the two folded shapes jaato/#1304 §1 asks for instead of a row per
 * call -- a run's housekeeping calls collapsed to one summary line, and
 * a failed call followed by a later success of the same tool collapsed
 * to one muted ``↻`` line.  Both folds are collapsed by default and
 * expand, on click, into the ordinary rows they cover -- so nothing this
 * view removes from sight is unrecoverable, only quieter until asked for.
 */
import { memo, useState } from "react";
import type { ToolGroupItem } from "@/store/transcript";
import { useJaato } from "@/store/store";
import { ToolBlockView } from "./ToolBlockView";

function FoldedRow({ group, glyph, glyphLabel, glyphClass = "" }: { group: ToolGroupItem; glyph: string; glyphLabel: string; glyphClass?: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="border-t hairline">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-3 py-1.5 pr-1 text-left hover:bg-tint/70 text-text-muted"
        aria-expanded={expanded}
      >
        <span className={`w-3 shrink-0 text-center ${glyphClass}`} aria-label={glyphLabel}>{glyph}</span>
        <span className="font-mono text-xs italic truncate flex-1">{group.label}</span>
        <span className="text-[10px] w-[52px] text-right">{expanded ? "▾" : "▸"}</span>
      </button>
      {expanded && (
        <div className="ml-6 mb-1.5 flex flex-col gap-0.5">
          {group.calls.map((c) => <ToolBlockView key={c.id} block={c} toolClass={group.toolClass} />)}
        </div>
      )}
    </div>
  );
}

export const ToolGroupView = memo(function ToolGroupView({ group }: { group: ToolGroupItem }) {
  // Hashed tool / category ids in the arguments, shown by name (protocol/toolIds.ts).
  const toolIdNames = useJaato((s) => s.toolIdNames);

  if (group.mode === "row") return <ToolBlockView block={group.calls[0]!} toolClass={group.toolClass} />;

  if (group.mode === "recovered") {
    const recovered = group.calls[1]!;
    const displayName = toolIdNames[recovered.toolName] ?? recovered.toolName;
    return <FoldedRow group={group} glyph="↻" glyphLabel={`${displayName} recovered after a retry`} />;
  }

  // "fold": a run of housekeeping calls, collapsed.
  return <FoldedRow group={group} glyph="⋯" glyphLabel="internal calls" />;
});
