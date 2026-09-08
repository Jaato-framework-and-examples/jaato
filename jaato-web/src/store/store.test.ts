import { beforeEach, describe, expect, it } from "vitest";
import { useJaato, MAIN_AGENT } from "./store";
import type { JaatoEvent } from "@jaato/sdk";

const ev = (o: Record<string, unknown>) => o as unknown as JaatoEvent;

beforeEach(() => {
  useJaato.getState().resetSessionState();
});

describe("reduce — streaming output", () => {
  it("appends to the last block for mode=append and opens a new block for write", () => {
    const d = useJaato.getState().dispatch;
    d([ev({ type: "agent.output", agent_id: "main", source: "model", text: "Hel", mode: "write" }), ev({ type: "agent.output", agent_id: "main", source: "model", text: "lo", mode: "append" })]);
    d([ev({ type: "agent.output", agent_id: "main", source: "model", text: "Again", mode: "write" })]);
    const blocks = useJaato.getState().blocks[MAIN_AGENT]!;
    expect(blocks.map((b) => (b.kind === "text" ? b.text : b.kind))).toEqual(["Hello", "Again"]);
  });
  it("routes output with no agent_id to main and creates unknown agents on the fly", () => {
    useJaato.getState().dispatch([ev({ type: "agent.output", text: "x", mode: "write" }), ev({ type: "agent.output", agent_id: "sub-1", text: "y", mode: "write" })]);
    const s = useJaato.getState();
    expect(s.blocks[MAIN_AGENT]).toHaveLength(1);
    expect(s.agentOrder).toEqual([MAIN_AGENT, "sub-1"]);
  });
});

describe("reduce — tool lifecycle", () => {
  it("tracks a call from start through output to end, by call_id", () => {
    const d = useJaato.getState().dispatch;
    d([ev({ type: "tool.call_start", agent_id: "main", tool_name: "run", tool_args: { cmd: "ls" }, call_id: "c1" })]);
    d([ev({ type: "tool.output", agent_id: "main", call_id: "c1", chunk: "a\n" }), ev({ type: "tool.output", agent_id: "main", call_id: "c1", chunk: "b\n" })]);
    let t = useJaato.getState().blocks[MAIN_AGENT]![0]!;
    expect(t.kind).toBe("tool");
    if (t.kind !== "tool") throw new Error();
    expect(t.status).toBe("running");
    expect(t.output).toBe("a\nb\n");
    expect(useJaato.getState().ui.popupCallId).toBe("c1");
    d([ev({ type: "tool.call_end", agent_id: "main", call_id: "c1", tool_name: "run", success: true, duration_seconds: 0.5, show_output: true })]);
    t = useJaato.getState().blocks[MAIN_AGENT]![0]!;
    if (t.kind !== "tool") throw new Error();
    expect(t.status).toBe("success");
    expect(t.expanded).toBe(true);
    expect(useJaato.getState().ui.popupCallId).toBeNull();
  });
  it("marks is_error_result as failure and expands it", () => {
    const d = useJaato.getState().dispatch;
    d([ev({ type: "tool.call_start", agent_id: "main", tool_name: "run", call_id: "c2" }), ev({ type: "tool.call_end", agent_id: "main", call_id: "c2", success: true, is_error_result: true, error_message: "nope" })]);
    const t = useJaato.getState().blocks[MAIN_AGENT]![0]!;
    if (t.kind !== "tool") throw new Error();
    expect(t.status).toBe("error");
    expect(t.expanded).toBe(true);
    expect(t.errorMessage).toBe("nope");
  });
  it("collects binary chunks as media and model speech under the reserved call id", () => {
    const d = useJaato.getState().dispatch;
    d([ev({ type: "tool.call_start", agent_id: "main", tool_name: "snap", call_id: "c3" }), ev({ type: "tool.output", agent_id: "main", call_id: "c3", chunk: "", mime_type: "image/png", data_b64: "AAAA", stream_id: "s1", sequence: 0, final: true })]);
    const t = useJaato.getState().blocks[MAIN_AGENT]![0]!;
    if (t.kind !== "tool") throw new Error();
    expect(t.media).toHaveLength(1);
    d([ev({ type: "tool.output", agent_id: "main", call_id: "model-output", chunk: "", mime_type: "audio/pcm;rate=24000", data_b64: "AAAA", final: false })]);
    d([ev({ type: "tool.output", agent_id: "main", call_id: "model-output", chunk: "", mime_type: "audio/pcm;rate=24000", data_b64: "BBBB", final: true })]);
    const blocks = useJaato.getState().blocks[MAIN_AGENT]!;
    const speech = blocks[1]!;
    if (speech.kind !== "tool") throw new Error();
    expect(speech.callId).toBe("model-output");
    expect(speech.media).toHaveLength(2);
    expect(speech.status).toBe("success");
  });
  it("keeps the transcript that rides the final speech chunk, including a one-frame utterance", () => {
    const d = useJaato.getState().dispatch;
    d([ev({ type: "tool.output", agent_id: "main", call_id: "model-output", chunk: "", mime_type: "audio/pcm;rate=24000", data_b64: "AAAA", final: false })]);
    d([ev({ type: "tool.output", agent_id: "main", call_id: "model-output", chunk: "Hello there.", mime_type: "audio/pcm;rate=24000", data_b64: "BBBB", final: true })]);
    const first = useJaato.getState().blocks[MAIN_AGENT]![0]!;
    if (first.kind !== "tool") throw new Error();
    expect(first.output).toBe("Hello there.");
    d([ev({ type: "tool.output", agent_id: "main", call_id: "model-output", chunk: "One frame.", mime_type: "audio/pcm;rate=24000", data_b64: "CCCC", final: true })]);
    const second = useJaato.getState().blocks[MAIN_AGENT]![1]!;
    if (second.kind !== "tool") throw new Error();
    expect(second.output).toBe("One frame.");
    expect(second.status).toBe("success");
  });
});

describe("reduce — prompts", () => {
  it("merges permission.requested and permission.input_mode into one pending record and clears it on resolve", () => {
    const d = useJaato.getState().dispatch;
    d([ev({ type: "permission.requested", agent_id: "main", request_id: "r1", tool_name: "write", response_options: [{ key: "y", label: "yes" }], prompt_lines: ["+x"], format_hint: "diff" })]);
    d([ev({ type: "permission.input_mode", agent_id: "main", request_id: "r1", tool_name: "write", call_id: "c9" })]);
    const p = useJaato.getState().permissions;
    expect(p).toHaveLength(1);
    expect(p[0]).toMatchObject({ requestId: "r1", inputMode: true, callId: "c9", formatHint: "diff" });
    expect(p[0]!.options).toEqual([{ key: "y", label: "yes" }]);
    expect(useJaato.getState().agents[MAIN_AGENT]!.status).toBe("awaiting_permission");
    d([ev({ type: "permission.resolved", agent_id: "main", request_id: "r1", granted: true })]);
    expect(useJaato.getState().permissions).toHaveLength(0);
  });
  it("walks a batch_only clarification and reports completion", () => {
    const d = useJaato.getState().dispatch;
    d([ev({ type: "clarification.batch", agent_id: "main", request_id: "q1", batch_only: true, questions: [{ question_text: "A?" }, { question_text: "B?" }] })]);
    const st = useJaato.getState();
    expect(st.clarifications[0]).toMatchObject({ inputMode: true, index: 0, batchOnly: true });
    const n1 = st.answerClarification("q1", "one")!;
    expect(n1.index).toBe(1);
    const n2 = useJaato.getState().answerClarification("q1", "two")!;
    expect(n2.index).toBe(2);
    expect(n2.answers).toEqual(["one", "two"]);
  });
  it("assembles the per-question clarification path from its two events", () => {
    const d = useJaato.getState().dispatch;
    d([ev({ type: "clarification.question", agent_id: "main", request_id: "q2", question_index: 0, total_questions: 1, question_text: "Which?", options: ["a", "b"] })]);
    d([ev({ type: "clarification.input_mode", agent_id: "main", request_id: "q2", question_index: 0 })]);
    const c = useJaato.getState().clarifications[0]!;
    expect(c.batchOnly).toBe(false);
    expect(c.inputMode).toBe(true);
    expect(c.questions[0]?.options).toEqual(["a", "b"]);
  });
});

describe("reduce — session metadata", () => {
  it("captures plan, context, commands, workspace files and session info", () => {
    const d = useJaato.getState().dispatch;
    d([
      ev({ type: "plan.updated", agent_id: "main", plan_name: "P", steps: [{ step_id: "1", content: "a", status: "pending" }] }),
      ev({ type: "plan.step_updated", agent_id: "main", step_id: "1", status: "completed", result: "ok" }),
      ev({ type: "context.updated", agent_id: "main", usage: { total_tokens: 10 }, context_limit: 100, percent_used: 10 }),
      ev({ type: "command.list", commands: [{ name: "waypoint list", description: "w" }] }),
      ev({ type: "workspace.files_changed", changes: [{ path: "a/b.py", change: "created" }, { path: "c.py", change: "deleted" }] }),
      ev({ type: "session.info", session_id: "S1", model_provider: "anthropic", model_name: "m" }),
    ]);
    const s = useJaato.getState();
    expect(s.plan[MAIN_AGENT]!.steps[0]).toMatchObject({ status: "completed", result: "ok" });
    expect(s.context[MAIN_AGENT]!.percentUsed).toBe(10);
    expect(s.commands.some((c) => c.name === "waypoint list")).toBe(true);
    expect(s.commands.some((c) => c.name === "help")).toBe(true);
    expect(s.workspaceFiles).toEqual({ "a/b.py": "created" });
    expect(s.sessionId).toBe("S1");
    expect(s.session.provider).toBe("anthropic");
  });
});

describe("reduce — errors", () => {
  it("treats the workspace-mode probe reply as state, not as an error line", () => {
    useJaato.getState().dispatch([ev({ type: "error", error: "Workspace mode not enabled", error_type: "WorkspaceModeDisabled" })]);
    const s = useJaato.getState();
    expect(s.workspace.mode).toBe("disabled");
    expect(s.blocks[MAIN_AGENT]).toHaveLength(0);
    useJaato.getState().dispatch([ev({ type: "error", error: "boom", error_type: "ProviderError" })]);
    expect(useJaato.getState().blocks[MAIN_AGENT]![0]).toMatchObject({ kind: "system", style: "error", text: "[ProviderError] boom" });
  });
});
