/**
 * The main view: agent tabs, the selected agent's output, pending
 * prompts, the composer, side panels and the status bar.  On mount it
 * asks the daemon for its command list and profiles, then creates (or
 * reattaches) a session.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { EventTypeValue } from "@jaato/sdk";
import { useJaato } from "@/store/store";
import { getClient } from "@/sdk/connection";
import { OutputPane } from "@/components/output/OutputPane";
import { ToolOutputPopup } from "@/components/output/ToolOutputPopup";
import { Composer } from "@/components/input/Composer";
import { PermissionPrompt } from "@/components/prompts/PermissionPrompt";
import { PostAuthSetupPrompt } from "@/components/prompts/PostAuthSetupPrompt";
import { ClarificationPrompt } from "@/components/prompts/ClarificationPrompt";
import { ReferenceSelectionPrompt } from "@/components/prompts/ReferenceSelectionPrompt";
import { PlanPanel } from "@/components/panels/PlanPanel";
import { BudgetPanel } from "@/components/panels/BudgetPanel";
import { WorkspacePanel } from "@/components/panels/WorkspacePanel";
import { AgentTabs } from "@/components/panels/AgentTabs";
import { StatusBar } from "@/components/layout/StatusBar";
import { answerClarification, cancelClarification, inputHistory, respondPermission, respondPostAuth, respondReference, submitInput } from "@/app/actions";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";

function SidePanel({ title, onClose, children }: { title: string; onClose: () => void; children: React.ReactNode }) {
  return (
    <section className="w-72 shrink-0 border-l hairline surface-1 flex flex-col min-h-0" aria-label={title}>
      <header className="flex items-center justify-between px-3 py-1.5 border-b hairline text-xs uppercase tracking-wide text-text-muted">
        <span>{title}</span>
        <button type="button" onClick={onClose} aria-label={`Close ${title}`} className="hover:text-text">✕</button>
      </header>
      <div className="flex-1 overflow-auto">{children}</div>
    </section>
  );
}

/**
 * A daemon-level auth command as the command list advertises it:
 * ``anthropic-auth``, ``openrouter-auth``, ``mock-auth`` -- the top-level
 * name only (the list also carries ``<name> login`` sub-entries).
 */
function authCommands(commands: { name: string; description?: string }[]): { name: string; description?: string }[] {
  return commands.filter((c) => /-auth$/.test(c.name) && !c.name.includes(" "));
}

/**
 * What the TUI lets you do before any session exists, in one card:
 * start with a profile or the workspace defaults, or sign in to a
 * provider first.  Nothing here is required -- ``skip`` drops to the
 * prompt, where any daemon command (``<provider>-auth login``,
 * ``help``) runs with no session, exactly as in the TUI.
 */
function ProfilePicker({ onPick, onAuth, onSkip }: {
  onPick: (profile: string | null) => void;
  onAuth: (command: string) => void;
  onSkip: () => void;
}) {
  const profiles = useJaato((s) => s.profiles);
  const commands = useJaato((s) => s.commands);
  const auth = authCommands(commands);
  return (
    <div className="h-full flex items-center justify-center p-6 overflow-auto">
      <div className="w-full max-w-lg rounded-xl border hairline surface-1 p-5 space-y-3">
        <div className="text-lg font-semibold">New session</div>
        <div className="text-sm text-text-muted">Pick an agent profile, or start with the workspace defaults.</div>
        <ul className="divide-y divide-[color-mix(in_srgb,var(--c-muted)_35%,transparent)] rounded-md border hairline max-h-80 overflow-auto">
          <li><button type="button" onClick={() => onPick(null)} className="w-full text-left px-3 py-2 hover:bg-surface/60"><span className="font-mono">default</span> <span className="text-xs text-text-muted">— workspace .env provider and model</span></button></li>
          {profiles.map((p) => (
            <li key={p.name}><button type="button" onClick={() => onPick(p.name)} className="w-full text-left px-3 py-2 hover:bg-surface/60">
              <span className="font-mono">{p.name}</span> <span className="text-xs text-text-muted">{[p.description, [p.provider, p.model].filter(Boolean).join("/")].filter(Boolean).join(" — ")}</span>
            </button></li>
          ))}
        </ul>
        {auth.length > 0 && (
          <div className="space-y-1.5" aria-label="Sign in to a provider">
            <div className="text-sm text-text-muted">No provider configured yet? Sign in first; the daemon then offers to open the session for you.</div>
            <div className="flex flex-wrap gap-2">
              {auth.map((c) => (
                <button key={c.name} type="button" onClick={() => onAuth(`${c.name} login`)} className="px-2.5 py-1 rounded-md text-xs border hairline hover:bg-surface font-mono" title={c.description}>{c.name} login</button>
              ))}
            </div>
          </div>
        )}
        <div className="text-right">
          <button type="button" onClick={onSkip} className="text-xs text-text-muted underline">Skip — go to the prompt without a session</button>
        </div>
      </div>
    </div>
  );
}

export function SessionScreen() {
  useKeyboardShortcuts();
  const sessionId = useJaato((s) => s.sessionId);
  const selected = useJaato((s) => s.selectedAgentId);
  const ui = useJaato((s) => s.ui);
  const toggle = useJaato((s) => s.toggleUi);
  const commands = useJaato((s) => s.commands);
  const initProgress = useJaato((s) => s.initProgress);
  const permissions = useJaato((s) => s.permissions);
  const clarifications = useJaato((s) => s.clarifications);
  const references = useJaato((s) => s.referenceSelections);
  const postAuth = useJaato((s) => s.postAuth);
  const processing = useJaato((s) => s.processing[selected] ?? false);
  const [picking, setPicking] = useState(true);
  const [creating, setCreating] = useState(false);
  const booted = useRef(false);

  useEffect(() => {
    if (booted.current) return;
    booted.current = true;
    const c = getClient();
    c.requestCommandList().catch(() => undefined);
    c.listProfiles().catch(() => undefined);
    const unsub = c.subscribe(EventTypeValue.COMMAND_LIST_REFRESH, () => { c.requestCommandList().catch(() => undefined); });
    return unsub;
  }, []);

  const startSession = async (profile: string | null) => {
    setPicking(false);
    setCreating(true);
    try {
      const c = getClient();
      await c.createSession(profile ? { profile } : {});
    } finally {
      setCreating(false);
    }
  };

  const agentPerms = permissions.filter((p) => p.agentId === selected);
  const agentClars = clarifications.filter((c) => c.agentId === selected);
  const agentRefs = references.filter((r) => r.agentId === selected);

  const captureMode = useMemo(() => {
    const p = agentPerms[0];
    if (p) return { kind: "permission" as const, placeholder: `Permission for ${p.toolName}: type an option key (${p.options.map((o) => o.key).join(", ") || "y/n"})`, suggestions: p.options.map((o) => o.key) };
    const cl = agentClars.find((c) => c.inputMode);
    if (cl) return { kind: "clarification" as const, placeholder: "Type your answer (or a choice number) and press Enter" };
    if (agentRefs[0]) return { kind: "reference" as const, placeholder: "Type the reference to use" };
    return null;
  }, [agentPerms, agentClars, agentRefs]);

  const runAuth = (command: string) => {
    // Leave the picker so the daemon's replies (the login URL, the
    // ``auth.setup`` offer) land on a visible prompt, then run the command
    // exactly as if it had been typed.
    setPicking(false);
    submitInput(command, false).catch((err) => useJaato.getState().addSystemBlock(selected, String(err), "error"));
  };

  const postAuthCard = postAuth ? (
    <div className="px-4"><PostAuthSetupPrompt p={postAuth} onRespond={(a) => { respondPostAuth(postAuth.requestId, a).catch((err) => useJaato.getState().addSystemBlock(selected, String(err), "error")); }} /></div>
  ) : null;

  if (picking && !sessionId) {
    return (
      <div className="h-full flex flex-col">
        {postAuthCard}
        <div className="flex-1 min-h-0"><ProfilePicker onPick={startSession} onAuth={runAuth} onSkip={() => setPicking(false)} /></div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <AgentTabs />
      <div className="flex-1 flex min-h-0">
        <main className="flex-1 flex flex-col min-w-0 relative">
          {(creating || initProgress) && (
            <div className="px-4 py-1 text-xs text-text-muted border-b hairline flex items-center gap-2">
              <span className="pulse">●</span>
              {initProgress ? `${initProgress.message ?? initProgress.step ?? "initialising"}${initProgress.stepNumber != null && initProgress.totalSteps ? ` (${initProgress.stepNumber}/${initProgress.totalSteps})` : ""}` : "Creating session…"}
            </div>
          )}
          <OutputPane agentId={selected} />
          <div className="px-4">
            {agentPerms.map((p) => <PermissionPrompt key={p.requestId} p={p} onRespond={(k) => respondPermission(p.requestId, k)} />)}
            {agentClars.map((c) => <ClarificationPrompt key={c.requestId} c={c} onAnswer={(a) => answerClarification(c, a)} onCancel={() => cancelClarification(c)} />)}
            {agentRefs.map((r) => <ReferenceSelectionPrompt key={r.requestId} r={r} onRespond={(v) => respondReference(r.requestId, v)} />)}
            {postAuth && selected === "main" && <PostAuthSetupPrompt p={postAuth} onRespond={(a) => { respondPostAuth(postAuth.requestId, a).catch((err) => useJaato.getState().addSystemBlock(selected, String(err), "error")); }} />}
          </div>
          <div className="px-4 pb-2 pt-1">
            {processing && !captureMode && (
              <div className="text-[11px] text-text-muted px-1 flex items-center gap-2"><span className="pulse text-primary">●</span> agent working — type to queue a follow-up, <kbd>stop</kbd> or <kbd>Ctrl</kbd>+<kbd>C</kbd> to interrupt</div>
            )}
            <Composer commands={commands} history={inputHistory} captureMode={captureMode} onSubmit={(t, v) => { submitInput(t, v).catch((err) => useJaato.getState().addSystemBlock(selected, String(err), "error")); }} />
          </div>
          <ToolOutputPopup agentId={selected} />
        </main>
        {ui.showPlan && <SidePanel title="Plan" onClose={() => toggle("showPlan")}><PlanPanel agentId={selected} /></SidePanel>}
        {ui.showBudget && <SidePanel title="Budget" onClose={() => toggle("showBudget")}><BudgetPanel agentId={selected} /></SidePanel>}
        {ui.showWorkspace && <SidePanel title="Files" onClose={() => toggle("showWorkspace")}><WorkspacePanel /></SidePanel>}
      </div>
      <StatusBar />
    </div>
  );
}
