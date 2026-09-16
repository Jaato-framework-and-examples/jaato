/**
 * A scripted stand-in for ``python -m server --web-socket`` so the client
 * can be developed and end-to-end tested without a model provider.
 *
 * It speaks the real wire protocol (JSON frames, ``ConnectedEvent``
 * first, ``?token=`` auth when ``MOCK_TOKEN`` is set) and answers the
 * verbs the UI sends: ``workspace.list`` (workspace mode on/off via
 * ``MOCK_WORKSPACES``), ``session.new`` / ``session.profiles`` /
 * ``command.list_request`` / ``message.send`` / ``permission.response``
 * / ``clarification.*`` / ``session.stop`` / ``history.request``.
 * ``mock-auth login`` is a daemon-level auth command that works with no
 * session and is followed by the ``auth.setup`` offer, answered with
 * ``auth.setup_response`` -- the TUI's sign-in-first flow.
 *
 * Prompts drive a small scenario language so tests can request the
 * behaviour they need:
 *
 *   "code"      → a streamed answer with a <j-code> block and a <j-table>
 *   "tool"      → a tool call with streamed output, then success
 *   "permit"    → a tool call that asks permission (diff prompt_lines + a warning)
 *   "permit-bare" → the same ASK from a tool whose plugin renders no display
 *                 info: no prompt_lines, no warning -- the card falls back to
 *                 the tool arguments
 *   "ask"       → a batch_only clarification with two questions
 *   "fail"      → a failing tool call
 *   "subagent"  → spawns a subagent that streams in its own tab
 *   "model this is broken" (verbatim test) → echoes the text back
 *   anything else → a short streamed markdown reply
 *
 * Run: ``npm run mock-daemon`` (port 8090 by default, MOCK_PORT overrides).
 */
import { WebSocketServer, type WebSocket } from "ws";
import { randomUUID } from "node:crypto";

const PORT = Number(process.env.MOCK_PORT ?? 8090);
const HOST = process.env.MOCK_HOST ?? "127.0.0.1";
const TOKEN = process.env.MOCK_TOKEN ?? "";
const WORKSPACES = process.env.MOCK_WORKSPACES === "1";
const SPEED = Number(process.env.MOCK_SPEED ?? 1); // multiplier; 0 = no delays

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, SPEED ? ms * SPEED : 0));
const ts = () => new Date().toISOString();

interface Client { ws: WebSocket; sessionId: string | null; pending: Map<string, (v: unknown) => void>; ignored: Set<string>; }

/** What ``session.list`` answers: the daemon's free-form per-session dicts. */
function sessionListing(c: Client): Record<string, unknown>[] {
  return [
    { id: "20260916_090000", name: "", description: "fix the budget panel", model_provider: "anthropic", model_name: "claude-sonnet-4", is_loaded: true, is_current: c.sessionId === "20260916_090000", client_count: 1, turn_count: 3, workspace_path: "/srv/workspaces/project-a" },
    { id: "20260915_170000", name: "old notes", description: "", model_provider: "", model_name: "", is_loaded: false, is_current: false, client_count: 0, turn_count: 1, workspace_path: "/srv/workspaces/project-b" },
    ...(c.sessionId && !c.sessionId.startsWith("2026") ? [{ id: c.sessionId, name: "mock session", description: "", model_provider: "mock", model_name: "mock-1", is_loaded: true, is_current: true, client_count: 1, turn_count: 0, workspace_path: "/work" }] : []),
  ];
}

/** The conversation ``history.request`` replays for the sessions above. */
const HISTORIES: Record<string, Record<string, unknown>[]> = {
  "20260916_090000": [
    { role: "user", parts: [{ type: "text", text: "what are those [object Object] in the budget panel?" }] },
    { role: "model", parts: [{ type: "text", text: "Let me look." }, { type: "function_call", id: "h1", name: "readFile", args: { path: "src/components/panels/BudgetPanel.tsx" } }] },
    { role: "tool", parts: [{ type: "function_response", call_id: "h1", name: "readFile", result: { lines: 45 } }] },
    { role: "model", parts: [{ type: "text", text: "The panel reads function_calls as a number; it is a list of records." }] },
  ],
  "20260915_170000": [
    { role: "user", parts: [{ type: "text", text: "remember: notes live in docs/" }] },
    { role: "model", parts: [{ type: "text", text: "Noted." }] },
  ],
};

function send(c: Client, ev: Record<string, unknown>): void {
  if (c.ws.readyState !== c.ws.OPEN) return;
  c.ws.send(JSON.stringify({ timestamp: ts(), session_id: c.sessionId ?? "", ...ev }));
}

async function stream(c: Client, agentId: string, text: string, chunk = 6): Promise<void> {
  let first = true;
  for (let i = 0; i < text.length; i += chunk) {
    send(c, { type: "agent.output", agent_id: agentId, source: "model", text: text.slice(i, i + chunk), mode: first ? "write" : "append" });
    first = false;
    await sleep(12);
  }
}

function waitFor(c: Client, key: string): Promise<unknown> {
  return new Promise((resolve) => c.pending.set(key, resolve));
}

const CODE_REPLY = `Here is the function you asked for:

<j-code language="python">
<j-line n="1"><j-tok t="k">def</j-tok> <j-tok t="nf">greet</j-tok><j-tok t="p">(</j-tok><j-tok t="n">name</j-tok><j-tok t="p">):</j-tok></j-line>
<j-line n="2">    <j-tok t="k">return</j-tok> <j-tok t="s2">f"Hello, {name} &lt;3"</j-tok>  <j-tok t="c1"># &amp; that's it</j-tok></j-line>
</j-code>

And a comparison:

<j-table>
<j-thead><j-tr><j-th>Option</j-th><j-th>Latency</j-th><j-th>Notes</j-th></j-tr></j-thead>
<j-tr><j-td>React 19</j-td><j-td>fine</j-td><j-td>largest ecosystem</j-td></j-tr>
<j-tr><j-td>Svelte 5</j-td><j-td>best</j-td><j-td>signals, small bundle</j-td></j-tr>
</j-table>

- **bold** point with \`inline code\`
- a [link](https://example.com)
`;

async function turn(c: Client, text: string, agentId = "main"): Promise<void> {
  const lower = text.toLowerCase();
  send(c, { type: "agent.status_changed", agent_id: agentId, status: "processing" });
  await sleep(50);

  if (lower.includes("subagent")) {
    const subId = `sub-${randomUUID().slice(0, 6)}`;
    send(c, { type: "agent.created", agent_id: subId, agent_name: "researcher", agent_type: "subagent", parent_agent_id: agentId, profile_name: "researcher" });
    await stream(c, agentId, "Delegating to a researcher subagent…\n");
    await stream(c, subId, "# Research notes\n\nLooking into the question. Found **three** relevant sources.\n");
    send(c, { type: "agent.completed", agent_id: subId, summary: "done" });
    await stream(c, agentId, "\nThe subagent finished; see its tab.");
  } else if (lower.includes("permit")) {
    // The runner-tier wire (the default path): the daemon's PromptOperatorHandler
    // emits permission.requested from the runner's PromptPayload -- options are
    // {key, label, description} (no `action` on this path), the prompt content
    // rides prompt_lines / format_hint / warnings / warning_level -- then the
    // input_mode control event with the same options and call_id.  "permit-bare"
    // is the same ASK from a tool whose plugin renders no display info, so the
    // content fields are null and the card must fall back to tool_args.
    const bare = lower.includes("permit-bare");
    const callId = randomUUID();
    const reqId = randomUUID();
    const toolArgs = { path: "src/app.py", content: "print('hi')\n" };
    const options = [
      { key: "y", label: "yes", description: "allow this call" }, { key: "n", label: "no", description: "deny this call" },
      { key: "a", label: "always", description: "allow for the rest of the session" }, { key: "t", label: "this turn", description: "allow until the model finishes responding" },
    ];
    send(c, { type: "tool.call_start", agent_id: agentId, tool_name: "write_file", tool_args: toolArgs, call_id: callId });
    send(c, {
      type: "permission.requested", agent_id: agentId, request_id: reqId, tool_name: "write_file",
      tool_args: toolArgs, response_options: options,
      prompt_lines: bare ? null : ["Update file: src/app.py", "--- a/src/app.py", "+++ b/src/app.py", "@@ -1,2 +1,2 @@", "-print('hello')", "+print('hi')", " # end"],
      format_hint: bare ? null : "diff",
      warnings: bare ? null : "The path is outside the sandbox allowlist.", warning_level: bare ? null : "warning",
    });
    send(c, { type: "permission.input_mode", agent_id: agentId, request_id: reqId, tool_name: "write_file", call_id: callId, response_options: options, tool_args: null, editable_metadata: null });
    const answer = String(await waitFor(c, `perm:${reqId}`));
    const granted = ["y", "a", "t", "i", "once", "all", "yes"].includes(answer.toLowerCase());
    send(c, { type: "permission.resolved", agent_id: agentId, request_id: reqId, tool_name: "write_file", granted, method: "user" });
    send(c, { type: "tool.call_end", agent_id: agentId, tool_name: "write_file", call_id: callId, success: granted, duration_seconds: 0.21, error_message: granted ? null : "Permission denied by user", show_output: false });
    // WorkspaceFilesChangedEvent.changes carries {path, status} — the daemon's key.
    if (granted) send(c, { type: "workspace.files_changed", changes: [{ path: "src/app.py", status: "modified" }, { path: ".jaato/logs/session.log", status: "created" }] });
    await stream(c, agentId, granted ? `Written (you answered \`${answer}\`).` : "Understood, not writing the file.");
  } else if (lower.includes("ask")) {
    const reqId = randomUUID();
    send(c, {
      type: "clarification.batch", agent_id: agentId, request_id: reqId, tool_name: "request_clarification", batch_only: true,
      context: "Before I start:",
      // The shape question_payload() emits (shared/plugins/clarification/channels.py):
      // text / question_type / required / choices[{text, default?}] -- NOT the
      // per-question event's question_text / options.  The mock used to speak
      // the card's vocabulary, which is how a card that could not render the
      // daemon's passed every e2e test.
      questions: [
        { index: 1, text: "Which framework should the client use?", question_type: "single_choice", required: true, choices: [{ text: "React 19" }, { text: "Svelte 5", default: true }, { text: "Solid" }] },
        { index: 2, text: "Anything else I should know?", question_type: "free_text", required: false },
      ],
    });
    const answers = (await waitFor(c, `clar:${reqId}`)) as string[];
    // Like the server's ClarificationChannel._parse_answer: a bare number picks that option.
    const choices = ["React 19", "Svelte 5", "Solid"];
    const first = /^\d+$/.test(answers[0] ?? "") ? (choices[Number(answers[0]) - 1] ?? answers[0]) : answers[0];
    await stream(c, agentId, `Thanks — you chose **${first}** and said "${answers[1]}".`);
  } else if (lower.includes("fail")) {
    const callId = randomUUID();
    send(c, { type: "tool.call_start", agent_id: agentId, tool_name: "run_command", tool_args: { command: "false" }, call_id: callId });
    await sleep(80);
    send(c, { type: "tool.output", agent_id: agentId, call_id: callId, chunk: "boom\n" });
    send(c, { type: "tool.call_end", agent_id: agentId, tool_name: "run_command", call_id: callId, success: false, is_error_result: true, error_message: "exit status 1", duration_seconds: 0.08 });
    await stream(c, agentId, "The command failed; see the tool block.");
  } else if (lower.includes("tool")) {
    const callId = randomUUID();
    send(c, { type: "tool.call_start", agent_id: agentId, tool_name: "run_command", tool_args: { command: "ls -la", cwd: "/work" }, call_id: callId });
    for (const line of ["total 12", "drwxr-xr-x  3 u u 4096 .", "-rw-r--r--  1 u u  120 README.md", "-rw-r--r--  1 u u 2048 app.py"]) {
      send(c, { type: "tool.output", agent_id: agentId, call_id: callId, chunk: line + "\n" });
      await sleep(60);
    }
    send(c, { type: "tool.call_end", agent_id: agentId, tool_name: "run_command", call_id: callId, success: true, duration_seconds: 0.31, show_output: true });
    send(c, { type: "plan.updated", agent_id: agentId, plan_name: "Task plan", steps: [
      { step_id: "1", sequence: 1, content: "List the directory", status: "completed", result: "4 entries" },
      { step_id: "2", sequence: 2, content: "Read README.md", status: "in_progress" },
      { step_id: "3", sequence: 3, content: "Summarise", status: "pending" },
    ] });
    await stream(c, agentId, "Listed the directory; there are **4** entries.");
  } else if (lower.includes("code")) {
    await stream(c, agentId, CODE_REPLY, 10);
  } else {
    await stream(c, agentId, `You said: *${text.replace(/\*/g, "")}*\n\nThis is the **mock daemon**. Try \`code\`, \`tool\`, \`permit\`, \`ask\`, \`fail\` or \`subagent\`.`);
  }

  send(c, { type: "context.updated", agent_id: agentId, usage: { prompt_tokens: 1200, output_tokens: 340, total_tokens: 1540, cache_read_tokens: 800 }, context_limit: 200000, percent_used: 0.77, tokens_remaining: 198460, turns: 1 });
  send(c, { type: "turn.completed", agent_id: agentId, turn_number: 1, duration_seconds: 1.2, function_calls: lower.includes("tool") ? [{ name: "run_command", start_time: ts(), end_time: ts(), duration_seconds: 0.31 }] : [], finish_reason: "stop", usage: { prompt_tokens: 1200, output_tokens: 340, total_tokens: 1540 } });
  send(c, { type: "agent.status_changed", agent_id: agentId, status: "idle" });
}

const wss = new WebSocketServer({ host: HOST, port: PORT });
wss.on("connection", (ws, req) => {
  const url = new URL(req.url ?? "/", "http://x");
  const auth = req.headers.authorization ?? "";
  const presented = url.searchParams.get("token") ?? (auth.startsWith("Bearer ") ? auth.slice(7) : "");
  if (TOKEN && presented !== TOKEN) { ws.close(1008, "unauthorized"); return; }

  const c: Client = { ws, sessionId: null, pending: new Map(), ignored: new Set() };
  send(c, { type: "connected", protocol_version: "1.12", server_info: { server_version: "mock-0.0.1", client_id: randomUUID() } });

  ws.on("message", async (raw) => {
    let ev: Record<string, unknown>;
    try { ev = JSON.parse(String(raw)); } catch { return; }
    const type = String(ev.type);
    switch (type) {
      case "client.config": break;
      case "workspace.list":
        if (!WORKSPACES) send(c, { type: "error", error: "Workspace mode not enabled", error_type: "WorkspaceModeDisabled", recoverable: true });
        else send(c, { type: "workspace.list_response", root: "/srv/workspaces", workspaces: [
          { name: "project-a", path: "/srv/workspaces/project-a", owner: "mock:tester", configured: true, provider: "anthropic", model: "claude-sonnet-4", last_accessed: ts() },
          { name: "project-b", path: "/srv/workspaces/project-b", configured: false },
        ] });
        break;
      case "workspace.select":
        send(c, { type: "config.status", workspace: String(ev.name), configured: ev.name === "project-a", provider: ev.name === "project-a" ? "anthropic" : null, model: ev.name === "project-a" ? "claude-sonnet-4" : null, available_providers: ["anthropic", "google_genai", "openrouter"], missing_fields: ev.name === "project-a" ? [] : ["provider", "api_key"] });
        break;
      case "workspace.create":
        send(c, { type: "workspace.created", workspace: { name: String(ev.name), configured: false, owner: "mock:tester" } });
        break;
      case "workspace.delete":
        // The daemon refuses a workspace with loaded sessions; project-a has one.
        if (ev.name === "project-a") send(c, { type: "workspace.deleted", name: "project-a", ok: false, error: "Workspace 'project-a' has 1 loaded session(s): 20260916_090000 -- stop them first" });
        else send(c, { type: "workspace.deleted", name: String(ev.name), ok: true });
        break;
      case "config.update":
        send(c, { type: "config.updated", workspace: "project-b", configured: true, provider: ev.provider, model: ev.model, available_providers: [], missing_fields: [] });
        break;
      case "command.execute": {
        const cmd = String(ev.command);
        const args = (ev.args as string[] | undefined) ?? [];
        if (cmd === "session.new") {
          c.sessionId = `sess-${randomUUID().slice(0, 8)}`;
          send(c, { type: "init.progress", step: "plugins", status: "running", message: "Loading plugins", step_number: 1, total_steps: 2 });
          await sleep(120);
          send(c, { type: "init.progress", step: "provider", status: "complete", message: "Ready", step_number: 2, total_steps: 2 });
          send(c, { type: "agent.created", agent_id: "main", agent_name: "main", agent_type: "main", profile_name: args.includes("--profile") ? args[args.indexOf("--profile") + 1] : null });
          send(c, { type: "session.info", session_name: "mock session", model_provider: "mock", model_name: "mock-1", profile_name: args.includes("--profile") ? args[args.indexOf("--profile") + 1] : null, models: ["mock-1", "mock-2"], sessions: sessionListing(c) });
          // PermissionStatusEvent, emitted by the daemon at init: effective_default + suspension_scope.
          send(c, { type: "permission.status", effective_default: "ask", suspension_scope: null });
          send(c, { type: "system.message", message: "Connected to the mock daemon. Try: code, tool, permit, ask, fail, subagent.", style: "info" });
        } else if (cmd === "mock-auth") {
          // A daemon-level auth plugin command: works with NO session, like
          // ``anthropic-auth login`` on the real daemon.  A successful login
          // is followed by the daemon's ``auth.setup`` offer.
          if (args[0] === "login") {
            send(c, { type: "system.message", message: "Opening the browser for Mock Provider…\nAuthenticated as tester@example.com.", style: "info" });
            const reqId = randomUUID();
            c.pending.set(`auth:${reqId}`, () => undefined);
            send(c, {
              type: "auth.setup", request_id: reqId, provider_name: "mock", provider_display_name: "Mock Provider",
              available_models: [{ name: "mock-1", description: "fast" }, { name: "mock-2", description: "smart" }],
              has_active_session: c.sessionId !== null, current_provider: c.sessionId ? "mock" : "", current_model: c.sessionId ? "mock-1" : "",
              workspace_path: WORKSPACES ? "/srv/workspaces/project-b" : "",
            });
          } else {
            send(c, { type: "system.message", message: "mock-auth: login | logout | status", style: "info" });
          }
        } else if (cmd === "session.list") {
          send(c, { type: "session.list", sessions: sessionListing(c) });
        } else if (cmd === "session.attach") {
          const target = String(args[0] ?? "");
          if (!HISTORIES[target]) { send(c, { type: "error", error: `Session not found: ${target}`, error_type: "SessionError", recoverable: true }); break; }
          c.sessionId = target;
          send(c, { type: "agent.created", agent_id: "main", agent_name: "main", agent_type: "main", profile_name: null });
          send(c, { type: "session.info", session_id: target, session_name: target === "20260916_090000" ? "fix the budget panel" : "old notes", model_provider: target === "20260916_090000" ? "anthropic" : "mock", model_name: target === "20260916_090000" ? "claude-sonnet-4" : "mock-1", profile_name: null, models: ["mock-1"], sessions: sessionListing(c) });
          send(c, { type: "permission.status", effective_default: "allow", suspension_scope: null });
        } else if (cmd === "session.profiles") {
          send(c, { type: "session.profiles", profiles: [{ name: "researcher", description: "Deep research", provider: "anthropic", model: "claude-sonnet-4" }, { name: "coder", description: "Coding agent", provider: "openrouter", model: "openai/gpt-5" }] });
        } else if (cmd === "workspace.ignore") {
          // The daemon toggles one exact line in <workspace>/.gitignore and
          // answers with the entry's state AFTER the toggle (protocol 1.12).
          const p = args[0] ?? "";
          if (!p || p.startsWith("/")) {
            send(c, { type: "workspace.ignore.result", path: p, ok: false, error: `workspace.ignore: ${p ? "absolute paths are not addressable via the workspace .gitignore" : "empty pattern"}` });
          } else {
            const ignored = !c.ignored.has(p);
            if (ignored) c.ignored.add(p); else c.ignored.delete(p);
            send(c, { type: "workspace.ignore.result", path: p, ok: true, ignored, gitignore_path: "/work/.gitignore" });
          }
        } else if (cmd === "session.stop") {
          send(c, { type: "system.message", message: "Stopped.", style: "warning" });
        } else if (cmd === "model") {
          send(c, { type: "session.info", model_provider: "mock", model_name: args[0] ?? "mock-1" });
          send(c, { type: "system.message", message: `Model switched to ${args[0] ?? "mock-1"}`, style: "info" });
        } else if (cmd === "tools.list") {
          send(c, { type: "system.message", message: "Tools:\n  ✓ run_command\n  ✓ write_file\n  ✗ web_search (disabled)", style: "info" });
        } else if (cmd === "reset") {
          send(c, { type: "system.message", message: "History cleared.", style: "info" });
        } else {
          send(c, { type: "system.message", message: `mock: executed ${cmd} ${args.join(" ")}`.trim(), style: "info" });
        }
        break;
      }
      case "command.list_request":
        send(c, { type: "command.list", commands: [
          { name: "model", description: "Switch model (mock)" }, { name: "waypoint", description: "Manage waypoints" },
          { name: "mock-auth", description: "Mock Provider authentication" }, { name: "mock-auth login", description: "Sign in to Mock Provider" },
          { name: "waypoint list", description: "List waypoints" }, { name: "permissions status", description: "Show permission status" },
        ] });
        break;
      case "session.stop":
        send(c, { type: "system.message", message: "Stopped.", style: "warning" });
        break;
      case "history.request": {
        const history = (c.sessionId && HISTORIES[c.sessionId]) || [];
        send(c, { type: "history", agent_id: "main", history, turn_accounting: history.length ? [{ prompt: 120, output: 40, total: 160 }] : [] });
        break;
      }
      case "message.send":
        turn(c, String(ev.text ?? "")).catch(() => undefined);
        break;
      case "auth.setup_response": {
        if (!c.pending.has(`auth:${String(ev.request_id)}`)) break;
        c.pending.delete(`auth:${String(ev.request_id)}`);
        if (ev.connect !== true) { send(c, { type: "system.message", message: "No model selected, skipping session setup.", style: "dim" }); break; }
        const model = String(ev.model_name ?? "mock-1");
        if (ev.persist_env === true) send(c, { type: "system.message", message: `Saved JAATO_PROVIDER=mock and MODEL_NAME=${model} to .env`, style: "info" });
        c.sessionId = `sess-${randomUUID().slice(0, 8)}`;
        send(c, { type: "agent.created", agent_id: "main", agent_name: "main", agent_type: "main", profile_name: null });
        send(c, { type: "session.info", session_name: "mock session", model_provider: "mock", model_name: model, profile_name: null, models: ["mock-1", "mock-2"] });
        send(c, { type: "system.message", message: `Session created with mock / ${model}`, style: "info" });
        break;
      }
      case "permission.response":
        c.pending.get(`perm:${String(ev.request_id)}`)?.(ev.response);
        break;
      case "clarification.batch_response":
        c.pending.get(`clar:${String(ev.request_id)}`)?.(ev.answers ?? []);
        break;
      default:
        break;
    }
  });
});

console.log(`mock jaato daemon listening on ws://${HOST}:${PORT}${TOKEN ? " (token required)" : ""}${WORKSPACES ? " [workspace mode]" : ""}`);
