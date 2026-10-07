/**
 * A call to a tool that does not exist: the model named a tool id the
 * session never offered (a hallucinated ``t_<hex>`` id, a misspelt name,
 * a tool of a plugin the profile does not load).  The daemon answers it
 * without running anything, and the model almost always reads the error
 * and calls the right tool next, so for most readers the call is noise.
 * The transcript folds it to one muted line (``store/transcript.ts``,
 * mode ``"misfire"``); expanding the line shows the ordinary row.
 *
 * The one signal is the daemon's refusal text.  Every path that meets an
 * unknown name answers with the same words: ``ToolExecutor``
 * (``ai_tool_runner.py``), the runner's cli-only executor
 * (``runner/tool_executor.py``, which quotes the name) and the three
 * dispatch sites in ``jaato_session.py``, all ``No executor registered
 * for <name>``.  A tool that exists and fails says something else, and
 * keeps its red row.
 */
import type { ToolBlock } from "@/store/types";

const UNKNOWN_TOOL_RE = /^No executor registered for /;

/** Whether this finished call was refused because its tool does not exist. */
export function isUnknownToolCall(block: Pick<ToolBlock, "status" | "errorMessage" | "output">): boolean {
  if (block.status !== "error") return false;
  const text = (block.errorMessage ?? block.output ?? "").trim();
  return UNKNOWN_TOOL_RE.test(text);
}

/** The folded line's caption, naming what the model asked for (capped). */
export function misfireLabel(name: string): string {
  const shown = name.length > 32 ? name.slice(0, 31) + "…" : name;
  return `called a tool that does not exist: ${shown}`;
}
