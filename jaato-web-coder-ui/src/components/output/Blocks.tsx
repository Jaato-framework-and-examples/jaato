/**
 * The four block kinds of the conversation column, each drawn the way
 * design frame 04 draws it: a user turn is a tinted plate with a steel
 * rule on its left, right-aligned, with its turn number in the gutter;
 * the model's prose runs at 15px on a 74ch measure; a tool call is a row
 * (``ToolBlockView``); a system line is quiet text in the colour of what
 * it says.
 */
import { memo } from "react";
import type { OutputBlock, SystemBlock, TextBlock, UserBlock } from "@/store/types";
import { JMarkup } from "./JMarkup";
import { ToolBlockView } from "./ToolBlockView";

const TextBlockView = memo(function TextBlockView({ block }: { block: TextBlock }) {
  const isModel = block.source === "model";
  return (
    <div className={`py-1.5 ${isModel ? "" : "text-[13px]"}`}>
      {!isModel && block.source !== "system" && (
        <div className="kicker kicker-muted text-[10px]">{block.source}</div>
      )}
      <div className={isModel ? "" : "text-text-muted"}>
        <JMarkup text={block.text} />
      </div>
    </div>
  );
});

/**
 * ``turn`` is the 1-based index of this user message in the column (the
 * pane counts them), shown as ``T14`` in the gutter so a turn can be
 * named across the plan, the budget's "last turn" and a permission
 * request's "turn 15".
 */
const UserBlockView = memo(function UserBlockView({ block, turn }: { block: UserBlock; turn?: number }) {
  return (
    <div className="my-2.5 flex items-start gap-3">
      <span className="font-mono text-[10px] text-steel w-9 shrink-0 pt-2.5 select-none" aria-hidden="true">{turn ? `T${turn}` : ""}</span>
      <span className="flex-1" />
      <div className="max-w-[76%] tint border-l-2 border-steel px-3 py-2 whitespace-pre-wrap break-words text-[15px]">
        {block.text}
      </div>
    </div>
  );
});

const STYLE_CLASS: Record<string, string> = {
  error: "text-error",
  warning: "text-warning",
  success: "text-success",
  hint: "text-text-muted italic",
  help: "text-text-muted font-mono text-xs whitespace-pre",
  info: "text-text-muted",
};

const SystemBlockView = memo(function SystemBlockView({ block }: { block: SystemBlock }) {
  const cls = STYLE_CLASS[block.style] ?? STYLE_CLASS[block.style.split("_")[0] ?? ""] ?? "text-text-muted";
  return <div className={`py-0.5 text-[13px] whitespace-pre-wrap ${cls}`}>{block.text}</div>;
});

export function BlockView({ block, turn }: { block: OutputBlock; turn?: number }) {
  switch (block.kind) {
    case "text": return <TextBlockView block={block} />;
    case "user": return <UserBlockView block={block} turn={turn} />;
    case "system": return <SystemBlockView block={block} />;
    case "tool": return <ToolBlockView block={block} />;
  }
}
