/**
 * What happens when the user submits the composer.
 *
 * Order of precedence, as in ``rich_client.py``'s input loop:
 *   1. a pending permission prompt for the selected agent takes the text
 *      as its answer (``y`` / ``a`` / an option key …);
 *   2. a pending clarification takes it as the current question's answer;
 *   3. a pending reference selection takes it as the choice;
 *   4. otherwise ``parseUserInput`` decides: a client-only command runs
 *      here, a server command becomes ``CommandRequest``, everything else
 *      becomes ``SendMessageRequest`` (echoed locally as a user block).
 *
 * The composer's ``verbatim`` flag (Escape on the proposal) is threaded
 * through to ``parseUserInput`` so a dismissed command word ships as text.
 */
import { EventTypeValue } from "@jaato/sdk";
import { parseUserInput } from "@/protocol/commands";
import { MAIN_AGENT, useJaato } from "@/store/store";
import type { PendingClarification } from "@/store/types";
import { THEME_NAMES, applyTheme, saveThemePreference } from "@/theme/themes";
import { disconnect, getClient } from "@/sdk/connection";

export const inputHistory: string[] = [];

function remember(text: string): void {
  if (!text.trim()) return;
  if (inputHistory[inputHistory.length - 1] === text) return;
  inputHistory.push(text);
  if (inputHistory.length > 200) inputHistory.shift();
}

export async function respondPermission(requestId: string, key: string): Promise<void> {
  const st = useJaato.getState();
  st.resolvePermission(requestId);
  await getClient().respondToPermission(requestId, key);
}

export async function answerClarification(c: PendingClarification, answer: string): Promise<void> {
  const st = useJaato.getState();
  const next = st.answerClarification(c.requestId, answer);
  if (!next) return;
  if (!c.batchOnly) {
    // Per-question path: one response per question; the daemon drives the
    // next one (a fresh ClarificationInputModeEvent re-arms the card).
    await getClient().respondToClarification(c.requestId, answer, c.index);
    st.dismissClarification(c.requestId);
    return;
  }
  if (next.index >= next.questions.length) {
    await getClient().respondToClarificationBatch(c.requestId, next.answers);
    st.dismissClarification(c.requestId);
  }
}

export async function cancelClarification(c: PendingClarification): Promise<void> {
  const st = useJaato.getState();
  st.dismissClarification(c.requestId);
  if (c.batchOnly) await getClient().respondToClarificationBatch(c.requestId, [], true);
  else await getClient().respondToClarification(c.requestId, "cancel", c.index);
}

export async function respondReference(requestId: string, value: string): Promise<void> {
  useJaato.getState().dismissReferenceSelection(requestId);
  await getClient().respondToReferenceSelection(requestId, value);
}

function helpText(): string {
  const st = useJaato.getState();
  const rows = st.commands.map((c) => `  ${c.name.padEnd(28)} ${c.description ?? ""}`.trimEnd());
  return [
    "Commands are typed as plain words (no / prefix). The composer proposes matches as you type;",
    "press Esc on the proposal to send the word verbatim instead.",
    "",
    ...rows,
    "",
    "References: @path (file)  @@path (sandbox)  %name (prompt library)  /name (.jaato/commands)",
    "Keys: Ctrl+P plan · Ctrl+B budget · Alt+W files · Ctrl+T tools · Ctrl+A next agent · Ctrl+O next running tool",
  ].join("\n");
}

function contextText(agentId: string): string {
  const ctx = useJaato.getState().context[agentId];
  if (!ctx) return "No usage reported yet.";
  const u = ctx.usage;
  return [
    `Context usage for ${agentId}`,
    `  prompt tokens:  ${u.prompt_tokens ?? "–"}`,
    `  output tokens:  ${u.output_tokens ?? "–"}`,
    `  total tokens:   ${u.total_tokens ?? "–"}`,
    `  limit:          ${ctx.contextLimit ?? "–"}`,
    `  percent used:   ${ctx.percentUsed != null ? ctx.percentUsed.toFixed(1) + "%" : "–"}`,
    `  turns:          ${ctx.turns ?? "–"}`,
  ].join("\n");
}

/** Handle a submitted line. Returns after the request is on the wire. */
export async function submitInput(text: string, verbatim: boolean): Promise<void> {
  const st = useJaato.getState();
  const agentId = st.selectedAgentId || MAIN_AGENT;
  const trimmed = text.trim();

  const perm = st.permissions.find((p) => p.agentId === agentId) ?? st.permissions[0];
  if (perm) {
    const key = trimmed || (perm.options[perm.focus]?.key ?? "");
    if (!key) return;
    await respondPermission(perm.requestId, key);
    return;
  }
  const clar = st.clarifications.find((c) => c.agentId === agentId && c.inputMode) ?? st.clarifications.find((c) => c.inputMode);
  if (clar) {
    if (trimmed.toLowerCase() === "cancel") return cancelClarification(clar);
    const q = clar.questions[clar.index];
    const answer = trimmed || (q?.default != null ? String(q.default) : "");
    await answerClarification(clar, answer);
    return;
  }
  const ref = st.referenceSelections.find((r) => r.agentId === agentId) ?? st.referenceSelections[0];
  if (ref && trimmed) {
    await respondReference(ref.requestId, trimmed);
    return;
  }

  if (!trimmed) return;
  remember(text);
  const parsed = parseUserInput(text, { serverCommands: st.commands, verbatim });
  const client = getClient();
  switch (parsed.action) {
    case "exit":
      await disconnect();
      useJaato.getState().setScreen("connect");
      return;
    case "stop":
      await client.stop();
      return;
    case "clear":
      st.clearOutput(agentId);
      return;
    case "help":
      st.addSystemBlock(agentId, helpText(), "help");
      return;
    case "context":
      st.addSystemBlock(agentId, contextText(agentId), "help");
      return;
    case "history":
      await client.requestHistory(agentId);
      return;
    case "server": {
      if (parsed.command === "theme") {
        const name = parsed.args?.[0];
        if (name && THEME_NAMES.includes(name)) {
          applyTheme(name); saveThemePreference(name); st.setTheme(name);
          st.addSystemBlock(agentId, `Theme: ${name}`, "info");
        } else st.addSystemBlock(agentId, `Themes: ${THEME_NAMES.join(", ")}`, "info");
        return;
      }
      st.addUserBlock(agentId, text);
      await client.executeCommand(parsed.command ?? "", parsed.args ?? []);
      return;
    }
    case "message":
      st.addUserBlock(agentId, parsed.text ?? text);
      st.dispatch([{ type: EventTypeValue.AGENT_STATUS_CHANGED, agent_id: agentId, status: "processing" } as never]);
      await client.sendMessage(parsed.text ?? text);
      return;
  }
}
