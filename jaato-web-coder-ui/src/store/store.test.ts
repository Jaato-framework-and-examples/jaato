import { beforeEach, describe, expect, it } from "vitest";
import { useJaato, MAIN_AGENT, runningToolCallIds, uploadScope } from "./store";
import { agentPhase, anyBusy, isBusy } from "./phase";
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
    d([ev({ type: "permission.requested", agent_id: "main", request_id: "r1", tool_name: "write", response_options: [{ key: "y", label: "yes", description: "allow this call" }], prompt_lines: ["+x"], format_hint: "diff", warnings: "careful", warning_level: "warning" })]);
    d([ev({ type: "permission.input_mode", agent_id: "main", request_id: "r1", tool_name: "write", call_id: "c9" })]);
    const p = useJaato.getState().permissions;
    expect(p).toHaveLength(1);
    expect(p[0]).toMatchObject({ requestId: "r1", inputMode: true, callId: "c9", formatHint: "diff" });
    expect(p[0]).toMatchObject({ warnings: "careful", warningLevel: "warning", promptLines: ["+x"] });
    expect(p[0]!.options).toEqual([{ key: "y", label: "yes", description: "allow this call" }]);
    // The agent's status stays the daemon's word; that a person is being
    // waited on is the PHASE, derived from the pending record itself.
    expect(agentPhase(useJaato.getState(), MAIN_AGENT)).toMatchObject({ kind: "waiting", on: "permission", toolName: "write" });
    d([ev({ type: "permission.resolved", agent_id: "main", request_id: "r1", granted: true })]);
    expect(useJaato.getState().permissions).toHaveLength(0);
  });
  it("walks a batch_only clarification and reports completion", () => {
    const d = useJaato.getState().dispatch;
    // The payload is what question_payload() emits, not the per-question vocabulary.
    d([ev({ type: "clarification.batch", agent_id: "main", request_id: "q1", batch_only: true, context: "Before I start", questions: [
      { index: 1, text: "A?", question_type: "single_choice", required: true, choices: [{ text: "yes", default: true }, { text: "no" }] },
      { index: 2, text: "B?", question_type: "free_text", required: false },
    ] })]);
    const st = useJaato.getState();
    expect(st.clarifications[0]).toMatchObject({ inputMode: true, index: 0, batchOnly: true, context: "Before I start" });
    expect(st.clarifications[0]!.questions[0]).toMatchObject({ question_text: "A?", options: ["yes", "no"], default: 1, optional: false });
    expect(st.clarifications[0]!.questions[1]).toMatchObject({ question_text: "B?", options: [], optional: true });
    const n1 = st.answerClarification("q1", "one")!;
    expect(n1.index).toBe(1);
    const n2 = useJaato.getState().answerClarification("q1", "two")!;
    expect(n2.index).toBe(2);
    expect(n2.answers).toEqual(["one", "two"]);
  });
  it("assembles the per-question clarification path from its two events", () => {
    const d = useJaato.getState().dispatch;
    d([ev({ type: "clarification.question", agent_id: "main", request_id: "q2", question_index: 0, total_questions: 1, question_text: "Which?", options: [{ text: "a" }, { text: "b" }] })]);
    d([ev({ type: "clarification.input_mode", agent_id: "main", request_id: "q2", question_index: 0 })]);
    const c = useJaato.getState().clarifications[0]!;
    expect(c.batchOnly).toBe(false);
    expect(c.inputMode).toBe(true);
    expect(c.questions[0]?.options).toEqual(["a", "b"]);
  });
});

describe("reduce — turn accounting", () => {
  it("reads turn.completed's function_calls as the list of records it is, not a count", () => {
    const d = useJaato.getState().dispatch;
    d([ev({ type: "turn.completed", agent_id: "main", turn_number: 1, duration_seconds: 2.5, finish_reason: "stop",
      function_calls: [
        { name: "readFile", start_time: "t0", end_time: "t1", duration_seconds: 0.2 },
        { name: "readFile", start_time: "t1", end_time: "t2", duration_seconds: 0.1 },
        { name: "cli_based_tool", start_time: "t2", end_time: "t3", duration_seconds: 1.4 },
      ],
      usage: { prompt_tokens: 10, output_tokens: 5, total_tokens: 15 } })]);
    const last = useJaato.getState().context[MAIN_AGENT]!.lastTurn!;
    expect(last.toolCalls).toEqual({ count: 3, byName: [["readFile", 2], ["cli_based_tool", 1]] });
    expect(last.finishReason).toBe("stop");
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
      ev({ type: "workspace.files_changed", changes: [{ path: "a/b.py", status: "created" }, { path: "c.py", status: "deleted" }] }),
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

describe("reduce — workspace files", () => {
  it("reads the daemon's status key, so a created file is not shown as modified", () => {
    const d = useJaato.getState().dispatch;
    d([ev({ type: "workspace.files_snapshot", files: [{ path: "x.py", status: "created" }, { path: "gone.py", status: "deleted" }, { path: "y.py", status: "modified" }] })]);
    expect(useJaato.getState().workspaceFiles).toEqual({ "x.py": "created", "y.py": "modified" });
    d([ev({ type: "workspace.files_changed", changes: [{ path: "y.py", status: "deleted" }, { path: "z.py", change: "created" }] })]);
    expect(useJaato.getState().workspaceFiles).toEqual({ "x.py": "created", "z.py": "created" });
  });

  it("hide is a per-session client-side set; a directory id hides its subtree", () => {
    const st = useJaato.getState();
    st.toggleWorkspaceHidden("src/app.py");
    st.toggleWorkspaceHidden(".jaato/");
    expect(useJaato.getState().workspaceHidden).toEqual(["src/app.py", ".jaato/"]);
    useJaato.getState().toggleWorkspaceHidden("src/app.py");
    expect(useJaato.getState().workspaceHidden).toEqual([".jaato/"]);
    useJaato.getState().toggleWorkspaceShowHidden();
    expect(useJaato.getState().workspaceShowHidden).toBe(true);
    useJaato.getState().resetSessionState();
    expect(useJaato.getState().workspaceHidden).toEqual([]);
    expect(useJaato.getState().workspaceShowHidden).toBe(false);
  });

  it("a workspace.ignore.result records the entry's state and one notice; a refusal is an error notice", () => {
    const d = useJaato.getState().dispatch;
    d([ev({ type: "workspace.ignore.result", path: "build/", ok: true, ignored: true, gitignore_path: "/w/.gitignore" })]);
    expect(useJaato.getState().workspaceIgnored).toEqual({ "build/": true });
    expect(useJaato.getState().workspaceNotice).toEqual({ text: "build/ added to .gitignore" });
    d([ev({ type: "workspace.ignore.result", path: "build/", ok: true, ignored: false })]);
    expect(useJaato.getState().workspaceIgnored).toEqual({ "build/": false });
    expect(useJaato.getState().workspaceNotice?.text).toBe("build/ removed from .gitignore");
    d([ev({ type: "workspace.ignore.result", path: "/etc", ok: false, error: "workspace.ignore: absolute paths are not addressable via the workspace .gitignore" })]);
    expect(useJaato.getState().workspaceNotice).toMatchObject({ error: true });
    expect(useJaato.getState().workspaceIgnored).toEqual({ "build/": false });
  });
});

describe("reduce — permission status", () => {
  it("reads effective_default and suspension_scope, the daemon's keys", () => {
    const d = useJaato.getState().dispatch;
    d([ev({ type: "permission.status", effective_default: "deny", suspension_scope: null })]);
    expect(useJaato.getState().permissionStatus).toEqual({ effectiveDefault: "deny", suspensionScope: null });
    d([ev({ type: "permission.status", effective_default: "ask", suspension_scope: "turn" })]);
    expect(useJaato.getState().permissionStatus).toEqual({ effectiveDefault: "ask", suspensionScope: "turn" });
  });
});

describe("reduce — session list and history", () => {
  const SESSIONS = [{ id: "s-1", description: "first", is_loaded: true, workspace_path: "/w/a" }, { id: "s-2", is_loaded: false }];
  it("a session.list reply is kept for completion and printed as the TUI listing", () => {
    useJaato.getState().dispatch([ev({ type: "session.list", sessions: SESSIONS })]);
    const s = useJaato.getState();
    expect(s.sessions.map((x) => x.id)).toEqual(["s-1", "s-2"]);
    const last = s.blocks[MAIN_AGENT]!.at(-1)!;
    expect(last.kind === "system" && last.text).toContain("● s-1 - first");
  });
  it("a silent request keeps the list and prints nothing", () => {
    useJaato.getState().setSessionListSilent(1);
    useJaato.getState().dispatch([ev({ type: "session.list", sessions: SESSIONS })]);
    const s = useJaato.getState();
    expect(s.sessions).toHaveLength(2);
    expect(s.blocks[MAIN_AGENT]).toHaveLength(0);
    expect(s.sessionListSilent).toBe(0);
  });
  it("two silent requests swallow two replies; the third, typed, prints", () => {
    // The picker asks twice under React's development double-effect; a flag
    // cleared by the first reply let the second print a listing nobody typed.
    useJaato.getState().setSessionListSilent(1);
    useJaato.getState().setSessionListSilent(1);
    useJaato.getState().dispatch([ev({ type: "session.list", sessions: SESSIONS }), ev({ type: "session.list", sessions: SESSIONS })]);
    expect(useJaato.getState().blocks[MAIN_AGENT]).toHaveLength(0);
    useJaato.getState().dispatch([ev({ type: "session.list", sessions: SESSIONS })]);
    expect(useJaato.getState().blocks[MAIN_AGENT]).toHaveLength(1);
  });
  it("session.info's snapshot refreshes the listing without printing", () => {
    useJaato.getState().dispatch([ev({ type: "session.info", session_id: "s-1", sessions: SESSIONS })]);
    expect(useJaato.getState().sessions).toHaveLength(2);
    expect(useJaato.getState().blocks[MAIN_AGENT]).toHaveLength(0);
  });
  const HISTORY = [{ role: "user", parts: [{ type: "text", text: "hi" }] }, { role: "model", parts: [{ type: "text", text: "hello" }] }];
  it("history replays as blocks after an attach, and lists otherwise", () => {
    useJaato.getState().setHistoryMode("replay");
    useJaato.getState().dispatch([ev({ type: "history", agent_id: "main", history: HISTORY })]);
    let blocks = useJaato.getState().blocks[MAIN_AGENT]!;
    expect(blocks.map((b) => b.kind)).toEqual(["user", "text"]);
    expect(useJaato.getState().historyMode).toBe("listing");
    useJaato.getState().dispatch([ev({ type: "history", agent_id: "main", history: HISTORY, turn_accounting: [{ prompt: 1, output: 2 }] })]);
    blocks = useJaato.getState().blocks[MAIN_AGENT]!;
    expect(blocks).toHaveLength(3);
    expect(blocks[2]!.kind === "system" && blocks[2]!.text).toContain("Conversation History (2 messages, 1 turns)");
  });
});

describe("tools toggle", () => {
  it("expands or collapses every tool block, and new blocks follow the setting", () => {
    const st = useJaato.getState();
    st.dispatch([ev({ type: "tool.call_start", agent_id: "main", tool_name: "run", tool_args: {}, call_id: "c1" })]);
    expect(useJaato.getState().blocks[MAIN_AGENT]![0]!.kind === "tool" && (useJaato.getState().blocks[MAIN_AGENT]![0] as { expanded: boolean }).expanded).toBe(false);
    useJaato.getState().setToolsExpanded(true);
    expect((useJaato.getState().blocks[MAIN_AGENT]![0] as { expanded: boolean }).expanded).toBe(true);
    useJaato.getState().dispatch([ev({ type: "tool.call_start", agent_id: "main", tool_name: "run", tool_args: {}, call_id: "c2" })]);
    expect((useJaato.getState().blocks[MAIN_AGENT]![1] as { expanded: boolean }).expanded).toBe(true);
    useJaato.getState().setToolsExpanded(false);
    expect(useJaato.getState().blocks[MAIN_AGENT]!.every((b) => b.kind === "tool" && !b.expanded)).toBe(true);
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

  it("a recoverable error (the default) still reaches the transcript and leaves sessionFault unset", () => {
    useJaato.getState().dispatch([ev({ type: "error", error: "Session not found: x", error_type: "SessionError", recoverable: true })]);
    expect(useJaato.getState().sessionFault).toBeNull();
  });

  it("recoverable: false sets sessionFault -- RunnerBootstrapFailed's own shape", () => {
    useJaato.getState().dispatch([ev({ type: "error", error: "Runner bootstrap failed", error_type: "RunnerBootstrapFailed", recoverable: false })]);
    expect(useJaato.getState().sessionFault).toEqual({ errorType: "RunnerBootstrapFailed", message: "Runner bootstrap failed" });
  });

  it("a session that comes up afterward clears the fault", () => {
    useJaato.getState().dispatch([ev({ type: "error", error: "boom", error_type: "RunnerBootstrapFailed", recoverable: false })]);
    expect(useJaato.getState().sessionFault).not.toBeNull();
    useJaato.getState().dispatch([ev({ type: "session.info", session_id: "s-1" })]);
    expect(useJaato.getState().sessionFault).toBeNull();
  });

  it("resetSessionState clears a standing fault too", () => {
    useJaato.getState().dispatch([ev({ type: "error", error: "boom", error_type: "RunnerBootstrapFailed", recoverable: false })]);
    useJaato.getState().resetSessionState();
    expect(useJaato.getState().sessionFault).toBeNull();
  });
});

describe("the command palette flag", () => {
  it("setPaletteOpen toggles it, and resetSessionState closes it", () => {
    useJaato.getState().setPaletteOpen(true);
    expect(useJaato.getState().paletteOpen).toBe(true);
    useJaato.getState().resetSessionState();
    expect(useJaato.getState().paletteOpen).toBe(false);
  });
});

describe("reduce — post-auth setup offer", () => {
  it("keeps the daemon's auth.setup offer as a pending prompt, not as an output line", () => {
    useJaato.getState().dispatch([ev({
      type: "auth.setup", request_id: "r1", provider_name: "mock", provider_display_name: "Mock Provider",
      available_models: [{ name: "mock-1", description: "fast" }, { name: "mock-2" }, { description: "nameless — dropped" }],
      has_active_session: false, current_provider: "", current_model: "", workspace_path: "/srv/ws/p",
    })]);
    const s = useJaato.getState();
    expect(s.postAuth).toEqual({
      requestId: "r1", providerName: "mock", providerDisplayName: "Mock Provider",
      models: [{ name: "mock-1", description: "fast" }, { name: "mock-2", description: undefined }],
      hasActiveSession: false, currentProvider: undefined, currentModel: undefined, workspacePath: "/srv/ws/p",
    });
    expect(s.blocks[MAIN_AGENT] ?? []).toHaveLength(0);
  });

  it("dismissPostAuth clears the offer; a new offer replaces the old one", () => {
    useJaato.getState().dispatch([ev({ type: "auth.setup", request_id: "r1", provider_name: "a" })]);
    useJaato.getState().dispatch([ev({ type: "auth.setup", request_id: "r2", provider_name: "b", has_active_session: true, current_provider: "a", current_model: "m" })]);
    expect(useJaato.getState().postAuth?.requestId).toBe("r2");
    expect(useJaato.getState().postAuth?.hasActiveSession).toBe(true);
    useJaato.getState().dismissPostAuth();
    expect(useJaato.getState().postAuth).toBeNull();
  });
});

describe("reduce — workspace verbs", () => {
  const LIST = [{ name: "a", configured: false }, { name: "b", configured: true, provider: "anthropic", model: "m" }];
  it("workspace.created appends the row the daemon sent, and a reply naming nothing adds no row", () => {
    // The daemon repeats name/path beside the whole row; an older daemon
    // sent a dict its own model dropped, so the event arrived nameless and
    // the table gained an unnamed entry a click then selected as "".
    const d = useJaato.getState().dispatch;
    d([ev({ type: "workspace.list_response", root: "/srv/ws", workspaces: LIST })]);
    d([ev({ type: "workspace.created", name: "c", path: "/srv/ws/c", workspace: { name: "c", path: "/srv/ws/c", configured: false, owner: "app:me", last_accessed: "2026-09-16T10:00:00Z" } })]);
    expect(useJaato.getState().workspace.list.map((w) => w.name)).toEqual(["a", "b", "c"]);
    expect(useJaato.getState().workspace.list[2]).toMatchObject({ owner: "app:me", path: "/srv/ws/c", configured: false });
    d([ev({ type: "workspace.created", name: "", path: "" })]);
    expect(useJaato.getState().workspace.list.map((w) => w.name)).toEqual(["a", "b", "c"]);
    expect(useJaato.getState().workspace.root).toBe("/srv/ws");
  });
  it("config.updated is merged over the status held: the saved provider is configured, the provider list survives", () => {
    // ``config.updated`` carries what was written and no status field; read
    // as a status it emptied the dropdown and reported the provider it had
    // just saved as missing.
    const d = useJaato.getState().dispatch;
    d([ev({ type: "workspace.list_response", workspaces: LIST })]);
    d([ev({ type: "config.status", workspace: "a", configured: false, provider: null, model: null, available_providers: ["anthropic", "zhipuai"], missing_fields: ["provider", "model"] })]);
    d([ev({ type: "config.updated", workspace: "a", provider: "zhipuai", model: "glm-5.2", success: true })]);
    const s = useJaato.getState();
    expect(s.workspace.config).toEqual({ workspace: "a", configured: true, provider: "zhipuai", model: "glm-5.2", availableProviders: ["anthropic", "zhipuai"], missingFields: [] });
    expect(s.workspace.list[0]).toMatchObject({ name: "a", configured: true, provider: "zhipuai", model: "glm-5.2" });
    d([ev({ type: "config.updated", workspace: "a", provider: "zhipuai", model: null, success: true })]);
    expect(useJaato.getState().workspace.config?.missingFields).toEqual(["model"]);
  });
  it("a refused config.updated leaves the status alone and reports the reason", () => {
    const d = useJaato.getState().dispatch;
    d([ev({ type: "config.status", workspace: "a", configured: false, available_providers: ["anthropic"], missing_fields: ["provider"] })]);
    d([ev({ type: "config.updated", workspace: "a", provider: "nope", success: false, error: "Unknown provider 'nope'" })]);
    const s = useJaato.getState();
    expect(s.workspace.config?.configured).toBe(false);
    expect(s.workspace.config?.availableProviders).toEqual(["anthropic"]);
    expect(s.workspace.notice).toEqual({ text: "Unknown provider 'nope'", error: true });
  });
});

describe("reduce — the daemon's prompt echo", () => {
  it("confirms the bubble the composer drew instead of drawing the prompt again as agent text", () => {
    // The daemon echoes every prompt as agent.output with source "user";
    // rendered as a text block it showed each prompt twice, the second
    // time under a "USER" header.
    useJaato.getState().addUserBlock(MAIN_AGENT, "hola");
    useJaato.getState().dispatch([ev({ type: "agent.output", agent_id: MAIN_AGENT, source: "user", text: "hola", mode: "write" })]);
    const blocks = useJaato.getState().blocks[MAIN_AGENT]!;
    expect(blocks).toHaveLength(1);
    expect(blocks[0]).toMatchObject({ kind: "user", text: "hola", echoed: true });
    // A second echo of the same text is a new prompt (someone sent it again), not this one's.
    useJaato.getState().dispatch([ev({ type: "agent.output", agent_id: MAIN_AGENT, source: "user", text: "hola", mode: "write" })]);
    expect(useJaato.getState().blocks[MAIN_AGENT]!.filter((b) => b.kind === "user")).toHaveLength(2);
  });
  it("an echo with no local bubble -- a replay after attach -- becomes a user bubble, never a text block", () => {
    useJaato.getState().dispatch([
      ev({ type: "agent.output", agent_id: MAIN_AGENT, source: "user", text: "first", mode: "write" }),
      ev({ type: "agent.output", agent_id: MAIN_AGENT, source: "model", text: "answer", mode: "write" }),
      ev({ type: "agent.output", agent_id: MAIN_AGENT, source: "user", text: "second", mode: "write" }),
    ]);
    const blocks = useJaato.getState().blocks[MAIN_AGENT]!;
    expect(blocks.map((b) => b.kind)).toEqual(["user", "text", "user"]);
    expect(blocks.some((b) => b.kind === "text" && b.source === "user")).toBe(false);
  });
});

describe("rail width", () => {
  it("is clamped to the rail's bounds and starts at the default", () => {
    expect(useJaato.getState().ui.railWidth).toBe(300);
    useJaato.getState().setRailWidth(420);
    expect(useJaato.getState().ui.railWidth).toBe(420);
    useJaato.getState().setRailWidth(10);
    expect(useJaato.getState().ui.railWidth).toBe(220);
    useJaato.getState().setRailWidth(5000);
    expect(useJaato.getState().ui.railWidth).toBe(720);
    useJaato.getState().setRailWidth(Number.NaN);
    expect(useJaato.getState().ui.railWidth).toBe(300);
  });
});

describe("uploads: the attachment strip's state", () => {
  it("takeUploads hands back the staged paths and keeps what is still in flight", () => {
    const st = useJaato.getState();
    st.addUploads([
      { id: "u1", path: "a.pdf", size: 10, status: "queued", scope: "" },
      { id: "u2", path: "docs/b.md", size: 20, status: "queued", scope: "" },
      { id: "u3", path: "c.bin", size: 30, status: "queued", scope: "" },
    ]);
    st.updateUpload("u1", { status: "staged" });
    st.updateUpload("u2", { status: "failed", error: "unsafe_path" });
    st.updateUpload("u3", { status: "staging" });
    expect(useJaato.getState().takeUploads()).toEqual(["a.pdf"]);
    // the failed chip is dropped with the send; the one on the wire stays
    expect(useJaato.getState().uploads.map((u) => [u.id, u.status])).toEqual([["u3", "staging"]]);
    useJaato.getState().removeUpload("u3");
    expect(useJaato.getState().uploads).toEqual([]);
  });

  it("survives the reset an attach performs, so files queued on the picker reach the session", () => {
    const st = useJaato.getState();
    st.addUploads([{ id: "u9", path: "brief.txt", size: 5, status: "queued", scope: "" }]);
    st.resetSessionState();
    expect(useJaato.getState().uploads.map((u) => u.id)).toEqual(["u9"]);
    useJaato.getState().removeUpload("u9");
  });

  // #1250: uploads carry the session they belong to, so the strip (and
  // takeUploads) show one context's files, never another's.
  it("scopes an upload to a session, hides it from another, and shows it again on return", () => {
    const st = useJaato.getState();
    st.addUploads([{ id: "a1", path: "a.txt", size: 5, status: "staged", scope: "session:A" }]);
    // Active session A: its upload is visible.
    const visibleIn = (sid: string) =>
      useJaato.getState().uploads.filter((u) => u.scope === uploadScope({ sessionId: sid })).map((u) => u.id);
    expect(visibleIn("A")).toEqual(["a1"]);
    // Switch to B: A's upload is hidden, and a send in B names none of it.
    expect(visibleIn("B")).toEqual([]);
    // The store still holds it (nothing wiped on navigation).
    expect(useJaato.getState().uploads.map((u) => u.id)).toEqual(["a1"]);
    // Back to A: visible again.
    expect(visibleIn("A")).toEqual(["a1"]);
    useJaato.getState().removeUpload("a1");
  });

  it("takeUploads leaves another session's staged files untouched", () => {
    const st = useJaato.getState();
    st.addUploads([
      { id: "act", path: "here.txt", size: 5, status: "staged", scope: "session:A" },
      { id: "other", path: "there.txt", size: 5, status: "staged", scope: "session:B" },
    ]);
    // SESSION_INFO for A makes A the active scope.
    st.dispatch([{ type: "session.info", session_id: "A" } as never]);
    expect(useJaato.getState().takeUploads()).toEqual(["here.txt"]);
    // B's staged file is not consumed by A's send.
    expect(useJaato.getState().uploads.map((u) => u.id)).toEqual(["other"]);
    useJaato.getState().removeUpload("other");
  });

  // #1250: a file attached before any session exists (scope "") is adopted
  // by the session that opens, so the picker stages into the session it is
  // about to open.
  it("re-stamps an unscoped upload onto the session that opens", () => {
    const st = useJaato.getState();
    st.resetSessionState();
    st.addUploads([{ id: "pk", path: "picked.txt", size: 5, status: "queued", scope: "" }]);
    st.dispatch([{ type: "session.info", session_id: "S1" } as never]);
    expect(useJaato.getState().uploads.find((u) => u.id === "pk")?.scope).toBe("session:S1");
    // A later SESSION_INFO for the same session does not re-stamp again.
    st.dispatch([{ type: "session.info", session_id: "S1" } as never]);
    expect(useJaato.getState().uploads.find((u) => u.id === "pk")?.scope).toBe("session:S1");
    useJaato.getState().removeUpload("pk");
  });
});

/**
 * The indicator's own defect, and the two shapes around it.
 *
 * Every case here fails against the flag this replaced: ``processing`` was
 * written from ``status === "processing" || status === "running"`` and the
 * daemon emits neither word, so the real ``active`` read as "not busy".
 */
describe("phase — what the agent is doing", () => {
  const d = () => useJaato.getState().dispatch;

  it("reads the daemon's own status word: active means busy", () => {
    d()([ev({ type: "agent.status_changed", agent_id: "main", status: "active" })]);
    expect(isBusy(useJaato.getState(), MAIN_AGENT)).toBe(true);
    expect(agentPhase(useJaato.getState(), MAIN_AGENT).kind).toBe("thinking");
  });

  it("is busy for a turn this client did not start", () => {
    // An attach to a running session, a wake, an injected prompt: no
    // composer, no optimistic flag, only the daemon's replayed status.
    d()([ev({ type: "agent.status_changed", agent_id: "sub-1", status: "active" })]);
    expect(isBusy(useJaato.getState(), "sub-1")).toBe(true);
    expect(anyBusy(useJaato.getState())).toBe(true);
  });

  it("goes idle on every status the daemon can end a turn with", () => {
    for (const status of ["idle", "done", "error", "cancelled"]) {
      d()([ev({ type: "agent.status_changed", agent_id: "main", status: "active" })]);
      expect(isBusy(useJaato.getState(), MAIN_AGENT)).toBe(true);
      d()([ev({ type: "agent.status_changed", agent_id: "main", status })]);
      expect(agentPhase(useJaato.getState(), MAIN_AGENT)).toEqual({ kind: "idle" });
    }
  });

  it("names the open tool, counts the batch, and keeps the oldest clock", () => {
    d()([ev({ type: "agent.status_changed", agent_id: "main", status: "active" })]);
    d()([ev({ type: "tool.call_start", agent_id: "main", tool_name: "cli_based_tool", call_id: "c1" })]);
    d()([ev({ type: "tool.call_start", agent_id: "main", tool_name: "readFile", call_id: "c2" })]);
    const p = agentPhase(useJaato.getState(), MAIN_AGENT);
    expect(p).toMatchObject({ kind: "tool", toolName: "cli_based_tool", running: 2 });
    d()([ev({ type: "tool.call_end", agent_id: "main", call_id: "c1", success: true })]);
    expect(agentPhase(useJaato.getState(), MAIN_AGENT)).toMatchObject({ kind: "tool", toolName: "readFile", running: 1 });
    d()([ev({ type: "tool.call_end", agent_id: "main", call_id: "c2", success: true })]);
    expect(agentPhase(useJaato.getState(), MAIN_AGENT).kind).toBe("thinking");
  });

  it("a tool call whose end never arrived cannot pin the indicator", () => {
    // The reconnect case: the block stays ``running`` in the transcript,
    // but the phase is reachable only while the daemon says the agent is.
    d()([ev({ type: "agent.status_changed", agent_id: "main", status: "active" })]);
    d()([ev({ type: "tool.call_start", agent_id: "main", tool_name: "cli_based_tool", call_id: "c1" })]);
    d()([ev({ type: "turn.completed", agent_id: "main" })]);
    expect(agentPhase(useJaato.getState(), MAIN_AGENT)).toEqual({ kind: "idle" });
  });

  it("a prompt outranks the work it interrupted", () => {
    d()([ev({ type: "agent.status_changed", agent_id: "main", status: "active" })]);
    d()([ev({ type: "tool.call_start", agent_id: "main", tool_name: "writeNewFile", call_id: "c1" })]);
    d()([ev({ type: "permission.requested", agent_id: "main", request_id: "r1", tool_name: "writeNewFile" })]);
    expect(agentPhase(useJaato.getState(), MAIN_AGENT)).toMatchObject({ kind: "waiting", on: "permission" });
    d()([ev({ type: "permission.resolved", agent_id: "main", request_id: "r1", granted: true })]);
    // Answered, and the turn carries on: back to the tool, not to idle.
    expect(agentPhase(useJaato.getState(), MAIN_AGENT)).toMatchObject({ kind: "tool", toolName: "writeNewFile" });
  });

  it("a subagent's prompt gets a tab of its own", () => {
    d()([ev({ type: "permission.requested", agent_id: "sub-9", request_id: "r2", tool_name: "run" })]);
    expect(useJaato.getState().agentOrder).toContain("sub-9");
    expect(agentPhase(useJaato.getState(), "sub-9")).toMatchObject({ kind: "waiting" });
  });

  it("the composer's optimistic flag is given back by the daemon's first word", () => {
    useJaato.getState().markSending(MAIN_AGENT);
    expect(agentPhase(useJaato.getState(), MAIN_AGENT).kind).toBe("sending");
    d()([ev({ type: "agent.status_changed", agent_id: "main", status: "active" })]);
    expect(agentPhase(useJaato.getState(), MAIN_AGENT).kind).toBe("thinking");
    d()([ev({ type: "agent.status_changed", agent_id: "main", status: "idle" })]);
    expect(agentPhase(useJaato.getState(), MAIN_AGENT)).toEqual({ kind: "idle" });
  });

  it("a refused send does not leave the indicator claiming work", () => {
    useJaato.getState().markSending(MAIN_AGENT);
    useJaato.getState().clearSending(MAIN_AGENT);
    expect(agentPhase(useJaato.getState(), MAIN_AGENT)).toEqual({ kind: "idle" });
  });

  it("the busy clock does not restart while the turn runs", async () => {
    d()([ev({ type: "agent.status_changed", agent_id: "main", status: "active" })]);
    const first = useJaato.getState().busySince[MAIN_AGENT];
    await new Promise((r) => setTimeout(r, 5));
    d()([ev({ type: "agent.status_changed", agent_id: "main", status: "active" })]);
    expect(useJaato.getState().busySince[MAIN_AGENT]).toBe(first);
  });

  it("agent.completed ends the turn in the daemon's vocabulary", () => {
    d()([ev({ type: "agent.status_changed", agent_id: "sub-1", status: "active" })]);
    d()([ev({ type: "agent.completed", agent_id: "sub-1", success: true })]);
    expect(useJaato.getState().agents["sub-1"]!.status).toBe("done");
    expect(isBusy(useJaato.getState(), "sub-1")).toBe(false);
  });
});

describe("runningToolCallIds: what the live-output popup and the leader's O both read", () => {
  it("lists only running tools that have produced output, in block order", () => {
    useJaato.getState().dispatch([
      ev({ type: "tool.call_start", agent_id: "main", tool_name: "a", call_id: "c1" }),
      ev({ type: "tool.call_start", agent_id: "main", tool_name: "b", call_id: "c2" }),
      ev({ type: "tool.output", agent_id: "main", call_id: "c2", chunk: "x" }),
      ev({ type: "tool.call_start", agent_id: "main", tool_name: "c", call_id: "c3" }),
      ev({ type: "tool.output", agent_id: "main", call_id: "c3", chunk: "y" }),
    ]);
    // c1 has no output yet, and is left out; c2/c3 do, in the order started.
    expect(runningToolCallIds(useJaato.getState(), MAIN_AGENT)).toEqual(["c2", "c3"]);
  });
  it("drops a call once it ends", () => {
    useJaato.getState().dispatch([
      ev({ type: "tool.call_start", agent_id: "main", tool_name: "a", call_id: "c1" }),
      ev({ type: "tool.output", agent_id: "main", call_id: "c1", chunk: "x" }),
      ev({ type: "tool.call_end", agent_id: "main", call_id: "c1", tool_name: "a", success: true }),
    ]);
    expect(runningToolCallIds(useJaato.getState(), MAIN_AGENT)).toEqual([]);
  });
  it("an agent with no blocks answers empty, not an error", () => {
    expect(runningToolCallIds(useJaato.getState(), "nobody")).toEqual([]);
  });
});
