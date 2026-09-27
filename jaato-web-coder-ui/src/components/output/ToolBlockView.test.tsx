/**
 * The trailing-lines preview under a collapsed ``exec`` row.  Two reports
 * from a live session: a finished cli call kept drawing its output under a
 * row that said it was collapsed, and a notebook cell's preview showed its
 * ``<nb-row>`` tags as text (a tail of markup is cut markup).
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ToolBlock } from "@/store/types";
import { ToolBlockView } from "./ToolBlockView";

function call(extra: Partial<ToolBlock>): ToolBlock {
  return {
    id: "b1", kind: "tool", agentId: "main", callId: "c1", toolName: "cli_based_tool",
    args: { command: "ls" }, status: "running", startedAt: 0, output: "one\ntwo\n", media: [],
    expanded: false, toolClass: "exec", ...extra,
  };
}

describe("the exec preview under a collapsed row", () => {
  it("shows the tail while the call runs", () => {
    render(<ToolBlockView block={call({})} />);
    expect(screen.getByText(/one\s+two/)).toBeTruthy();
  });

  it("shows nothing once the call has finished: collapsed means collapsed", () => {
    render(<ToolBlockView block={call({ status: "success", durationSeconds: 0.7 })} />);
    expect(screen.queryByText(/one\s+two/)).toBeNull();
  });

  it("never draws notebook markup as a raw tail", () => {
    const output = '<nb-row type="stdout" label="Out [1]:">\n42\n</nb-row>\n';
    const { container } = render(<ToolBlockView block={call({ toolName: "notebook_execute", args: { code: "print(42)" }, output })} />);
    expect(container.textContent).not.toContain("<nb-row");
  });
});
