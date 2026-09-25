/**
 * The transcript's six item kinds (``store/transcript.ts``, jaato/#1304
 * §1), each drawn the way design frame 04 draws it: a user turn is a
 * tinted plate with a steel rule on its left, right-aligned, with its
 * turn number in the gutter; a model's prose runs at 15px on a 74ch
 * measure; reasoning is one collapsed line by default; a tool call is a
 * row or one of the two folds (``ToolGroupView``); a banner is a
 * full-width notice; a system note is quiet text in the colour of what
 * it says.
 */
import { memo, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { useJaato } from "@/store/store";
import type { AssistantTextItem, BannerItem, SystemNoteItem, ThinkingItem, TranscriptItem, UserMessageItem } from "@/store/transcript";
import { firstSentence, thinkingElapsedSeconds } from "@/store/transcript";
import { agentNameMap, resolveAgentIdsInText } from "@/protocol/agentNames";
import { JMarkup } from "./JMarkup";
import { ToolGroupView } from "./ToolGroupView";

/**
 * "One name everywhere" (#1304 §4): a subagent's raw id, mentioned in the
 * PARENT's own prose ("Subagent spawned (id: subagent_1)"), is resolved to
 * the display name the tab already shows -- at render time, off the same
 * ``s.agents`` map the tab reads, so the two cannot disagree.
 */
const AssistantTextView = memo(function AssistantTextView({ item }: { item: AssistantTextItem }) {
  const names = useJaato(useShallow((s) => agentNameMap(s.agents)));
  return (
    <div className="py-1.5">
      <JMarkup text={resolveAgentIdsInText(item.text, names)} />
    </div>
  );
});

/**
 * Reasoning, collapsed by default: "Thought for Ns · '<first sentence>'".
 * While still streaming (this is the tail item, nothing has superseded
 * it yet) only its last line shows, dimmed, with no duration -- the
 * block is not over, so there is nothing to have taken Ns yet.
 * ``startedAt`` is a client-side estimate (see ``store/transcript.ts``'s
 * docstring on ``ThinkingItem`` for why), so a completed block whose
 * start this session never observed reads "a while" rather than "0s".
 */
const ThinkingBlockView = memo(function ThinkingBlockView({ item }: { item: ThinkingItem }) {
  const [expanded, setExpanded] = useState(false);

  if (item.streaming) {
    const lastLine = item.text.split("\n").filter((l) => l.trim()).at(-1) ?? item.text;
    return (
      <div className="py-1 text-[12px] text-text-muted italic truncate" aria-live="polite" aria-label="Thinking">
        {lastLine.trim() || "Thinking…"}
      </div>
    );
  }

  const elapsed = thinkingElapsedSeconds(item);
  const duration = elapsed != null ? `${Math.max(1, Math.round(elapsed))} s` : "a while";
  const summary = `Thought for ${duration} · '${firstSentence(item.text)}'`;
  return (
    <div className="py-1">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-1.5 text-[12px] text-text-muted italic hover:text-steel"
        aria-expanded={expanded}
      >
        <span aria-hidden="true">{expanded ? "▾" : "▸"}</span>
        <span className="truncate">{summary}</span>
      </button>
      {expanded && (
        <div className="mt-1 ml-4 pl-2.5 border-l-2 hairline text-[13px] text-text-muted whitespace-pre-wrap">
          {item.text}
        </div>
      )}
    </div>
  );
});

const UserMessageView = memo(function UserMessageView({ item }: { item: UserMessageItem }) {
  return (
    <div className="my-2.5 flex items-start gap-3">
      <span className="font-mono text-[10px] text-steel w-9 shrink-0 pt-2.5 select-none" aria-hidden="true">T{item.turn}</span>
      <span className="flex-1" />
      <div className="max-w-[76%] tint border-l-2 border-steel px-3 py-2 whitespace-pre-wrap break-words text-[15px]">
        {item.text}
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

function styleClass(style: string): string {
  return STYLE_CLASS[style] ?? STYLE_CLASS[style.split("_")[0] ?? ""] ?? "text-text-muted";
}

const SystemNoteView = memo(function SystemNoteView({ item }: { item: SystemNoteItem }) {
  const names = useJaato(useShallow((s) => agentNameMap(s.agents)));
  return (
    <div className={`py-0.5 text-[13px] whitespace-pre-wrap ${styleClass(item.style)}`}>
      {item.source && item.source !== "system" && (
        <div className="kicker kicker-muted text-[10px]">{item.source}</div>
      )}
      {resolveAgentIdsInText(item.text, names)}
    </div>
  );
});

const BannerView = memo(function BannerView({ item }: { item: BannerItem }) {
  return (
    <div className={`my-1 px-3 py-1.5 border hairline text-[13px] whitespace-pre-wrap ${styleClass(item.style)}`} role="alert">
      {item.text}
    </div>
  );
});

export function BlockView({ item }: { item: TranscriptItem }) {
  switch (item.kind) {
    case "userMessage": return <UserMessageView item={item} />;
    case "thinking": return <ThinkingBlockView item={item} />;
    case "assistantText": return <AssistantTextView item={item} />;
    case "toolGroup": return <ToolGroupView group={item} />;
    case "banner": return <BannerView item={item} />;
    case "systemNote": return <SystemNoteView item={item} />;
  }
}
