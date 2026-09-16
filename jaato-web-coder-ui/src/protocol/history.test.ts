import { describe, expect, it } from "vitest";
import { formatHistoryListing, historyBlocks } from "./history";

const HISTORY = [
  { role: "user", parts: [{ type: "text", text: "list the files" }] },
  { role: "model", parts: [{ type: "text", text: "Sure." }, { type: "function_call", id: "c1", name: "cli_based_tool", args: { command: "ls" } }] },
  { role: "tool", parts: [{ type: "function_response", call_id: "c1", name: "cli_based_tool", result: { stdout: "a.py\n" } }] },
  { role: "model", parts: [{ type: "text", text: "One file: a.py" }] },
];

describe("history replay", () => {
  it("rebuilds user, model and tool blocks and pairs responses with their calls", () => {
    let n = 0;
    const blocks = historyBlocks(HISTORY, "main", () => `h${++n}`, false);
    expect(blocks.map((b) => b.kind)).toEqual(["user", "text", "tool", "text"]);
    const tool = blocks[2]!;
    expect(tool.kind === "tool" && tool.toolName).toBe("cli_based_tool");
    expect(tool.kind === "tool" && tool.output).toContain('"stdout": "a.py\\n"');
    expect(tool.kind === "tool" && tool.status).toBe("success");
    expect(tool.kind === "tool" && tool.expanded).toBe(false);
  });
  it("follows the tools-expanded setting for the rebuilt blocks", () => {
    const blocks = historyBlocks(HISTORY, "main", () => "x", true);
    expect(blocks.filter((b) => b.kind === "tool").every((b) => b.kind === "tool" && b.expanded)).toBe(true);
  });
});

describe("history listing", () => {
  it("renders the TUI's summary with the per-turn token line", () => {
    const text = formatHistoryListing(HISTORY, [{ prompt: 10, output: 5 }, { prompt: 20, output: 7, total: 27 }]);
    expect(text).toContain("Conversation History (4 messages, 2 turns):");
    expect(text).toContain("[Function Call: cli_based_tool]");
    expect(text).toContain("[Function Response: cli_based_tool]");
    expect(text).toContain("--- Turn 1: 15 tokens (in: 10, out: 5) ---");
    expect(formatHistoryListing([], [])).toBe("No conversation history.");
  });
});
