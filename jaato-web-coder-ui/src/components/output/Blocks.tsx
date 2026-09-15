import { memo } from "react";
import type { OutputBlock, SystemBlock, TextBlock, UserBlock } from "@/store/types";
import { JMarkup } from "./JMarkup";
import { ToolBlockView } from "./ToolBlockView";

const TextBlockView = memo(function TextBlockView({ block }: { block: TextBlock }) {
  const isModel = block.source === "model";
  return (
    <div className={`py-1 ${isModel ? "" : "text-[13px]"}`}>
      {!isModel && block.source !== "system" && (
        <div className="text-[10px] uppercase tracking-wide text-text-muted font-mono">{block.source}</div>
      )}
      <div className={isModel ? "" : "text-text-muted"}>
        <JMarkup text={block.text} />
      </div>
    </div>
  );
});

const UserBlockView = memo(function UserBlockView({ block }: { block: UserBlock }) {
  return (
    <div className="my-2 flex justify-end">
      <div className="max-w-[85%] rounded-lg px-3 py-1.5 whitespace-pre-wrap break-words bg-surface border hairline">
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

export function BlockView({ block }: { block: OutputBlock }) {
  switch (block.kind) {
    case "text": return <TextBlockView block={block} />;
    case "user": return <UserBlockView block={block} />;
    case "system": return <SystemBlockView block={block} />;
    case "tool": return <ToolBlockView block={block} />;
  }
}
