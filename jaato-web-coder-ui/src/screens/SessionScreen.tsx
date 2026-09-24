/**
 * The main view (design frame 04): the session's identity in a 46px
 * header (brand, one tab per agent, workspace / model / context on the
 * right), the selected agent's output with pending prompts and the
 * composer under it, one persistent rail on the right whose Plan /
 * Budget / Files / Sessions / Memories sections open and close, and the
 * status bar.  On mount
 * it asks the daemon for its command list and profiles, then creates (or
 * reattaches) a session.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { EventTypeValue } from "@jaato/sdk";
import { useJaato } from "@/store/store";
import { getClient } from "@/sdk/connection";
import { OutputPane } from "@/components/output/OutputPane";
import { ToolOutputPopup } from "@/components/output/ToolOutputPopup";
import { Composer } from "@/components/input/Composer";
import { PhaseLine } from "@/components/panels/PhaseLine";
import { AttachStrip } from "@/components/input/AttachStrip";
import { PermissionPrompt } from "@/components/prompts/PermissionPrompt";
import { ExitPrompt } from "@/components/prompts/ExitPrompt";
import { PostAuthSetupPrompt } from "@/components/prompts/PostAuthSetupPrompt";
import { ClarificationPrompt } from "@/components/prompts/ClarificationPrompt";
import { ReferenceSelectionPrompt } from "@/components/prompts/ReferenceSelectionPrompt";
import { PlanPanel, planProgress } from "@/components/panels/PlanPanel";
import { BudgetPanel } from "@/components/panels/BudgetPanel";
import { WorkspacePanel, useVisibleWorkspaceFiles } from "@/components/panels/WorkspacePanel";
import { SessionRow, SessionsPanel, notedSummary } from "@/components/panels/SessionsPanel";
import { MemoriesPanel } from "@/components/panels/MemoriesPanel";
import { memoriesSummary } from "@/app/memories";
import { AgentTabs } from "@/components/panels/AgentTabs";
import { StatusBar } from "@/components/layout/StatusBar";
import { Plate } from "@/components/layout/Plate";
import { RailResizer } from "@/components/layout/RailResizer";
import { RailSectionResizer } from "@/components/layout/RailSectionResizer";
import { sharesFor, RAIL_SECTION_MIN_PX, type RailSectionId } from "@/store/railSplits";
import { answerClarification, attachSession, cancelClarification, createSession, ensureSessions, inputHistory, respondPermission, respondPostAuth, respondReference, submitInput } from "@/app/actions";
import { openSessionWithQueued } from "@/app/staging";
import { answerExit } from "@/app/exitChoice";
import { sessionsInWorkspace } from "@/protocol/sessions";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";

/**
 * The header: who this session is.  The brand cell, the agent tabs, and
 * on the right what the daemon reported -- the workspace, the model and
 * a context bar -- in monospace, since those are its words.
 */
function SessionHeader() {
  const session = useJaato((s) => s.session);
  const ws = useJaato((s) => s.workspace.selected);
  const selected = useJaato((s) => s.selectedAgentId);
  const ctx = useJaato((s) => s.context[selected]);
  const pct = ctx?.percentUsed;
  const model = [session.provider, session.model].filter(Boolean).join(" / ");
  return (
    <header className="flex items-stretch h-[46px] border-b hairline shrink-0">
      <div className="flex items-center px-4 border-r hairline"><span className="display text-[18px] tracking-[0.02em]">jaato</span></div>
      <AgentTabs />
      <span className="flex-1" />
      <div className="hidden md:flex items-center gap-[18px] px-[18px] font-mono text-xs">
        {ws && <span><span className="text-text-muted">ws </span>{ws}</span>}
        {model && <span><span className="text-text-muted">model </span>{model}</span>}
        {session.profile && <span><span className="text-text-muted">profile </span>{session.profile}</span>}
        {pct != null && (
          <span className="flex items-center gap-2" title="Context window used">
            <span className="text-text-muted">ctx</span>
            <span className="inline-block w-[84px] h-1.5 bg-[color-mix(in_srgb,var(--c-text)_16%,var(--c-bg))]"><span className={`block h-full ${pct > 80 ? "bg-error" : pct > 60 ? "bg-warning" : "bg-steel"}`} style={{ width: `${Math.min(100, pct)}%` }} /></span>
            <span className={pct > 80 ? "text-error" : pct > 60 ? "text-warning" : ""}>{pct.toFixed(0)}%</span>
          </span>
        )}
      </div>
    </header>
  );
}

/**
 * One section of the rail: a header that opens and closes it (the same
 * ``ui.show*`` flag the status bar and the shortcuts toggle) and, when
 * open, a labelled region with the panel.  The value on the header's
 * right is the one number the section is about, so a closed section
 * still says something.
 *
 * An OPEN section takes a share of the rail's height (``share``, a fraction
 * of the open sections' total) via ``flex-grow`` rather than a fixed cap, and
 * scrolls INSIDE its own ``min-h-0 overflow-auto`` box.  The rail no longer
 * scrolls as a whole: the open sections divide its height between them, and
 * the horizontal handles between them (``RailSectionResizer``) move that
 * division.  ``data-rail-section`` is the marker those handles measure.
 */
function RailSection({ id, title, value, open, share, onToggle, children }: { id: RailSectionId; title: string; value?: string | null; open: boolean; share: number; onToggle: () => void; children: React.ReactNode }) {
  return (
    <>
      <button type="button" onClick={onToggle} className="flex items-baseline justify-between px-3.5 py-2 border-b hairline w-full text-left shrink-0 hover:bg-tint/60" aria-expanded={open} aria-label={`${open ? "Close" : "Open"} ${title}`}>
        <span className="kicker">{title}</span>
        <span className="font-mono text-[11px] text-text-muted">{value ? `${value} ` : ""}{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <section
          aria-label={title}
          data-rail-section={id}
          className="border-b hairline overflow-auto min-h-0"
          style={{ flexGrow: share, flexShrink: 1, flexBasis: 0, minHeight: RAIL_SECTION_MIN_PX }}
        >
          {children}
        </section>
      )}
    </>
  );
}

function Rail({ agentId }: { agentId: string }) {
  const ui = useJaato((s) => s.ui);
  const toggle = useJaato((s) => s.toggleUi);
  const plan = useJaato((s) => s.plan[agentId]);
  const ctx = useJaato((s) => s.context[agentId]);
  const changed = Object.keys(useVisibleWorkspaceFiles()).length;
  const isReset = useJaato((s) => s.workspaceReset !== null);
  const sessions = useJaato((s) => s.sessions);
  const notes = useJaato((s) => s.notes);
  const memories = useJaato((s) => s.memories);
  const budget = ctx?.usage.cost_usd != null ? `$${Number(ctx.usage.cost_usd).toFixed(4)}` : ctx?.percentUsed != null ? `${ctx.percentUsed.toFixed(0)}%` : null;

  // Fixed order; each carries its ``ui.show*`` flag and its panel.  The open
  // subset shares the rail's height, and a handle sits on each boundary
  // between two OPEN sections — so a closed section (just its header) has
  // nothing to resize.
  const sections: { id: RailSectionId; title: string; value: string | null; open: boolean; toggle: () => void; panel: React.ReactNode }[] = [
    { id: "plan", title: "Plan", value: planProgress(plan), open: ui.showPlan, toggle: () => toggle("showPlan"), panel: <PlanPanel agentId={agentId} /> },
    { id: "budget", title: "Budget", value: budget, open: ui.showBudget, toggle: () => toggle("showBudget"), panel: <BudgetPanel agentId={agentId} /> },
    { id: "files", title: "Files", value: changed ? `${changed} ${isReset ? "since reset" : "changed"}` : isReset ? "reset" : null, open: ui.showWorkspace, toggle: () => toggle("showWorkspace"), panel: <WorkspacePanel /> },
    { id: "sessions", title: "Sessions", value: notedSummary(sessions, notes), open: ui.showSessions, toggle: () => toggle("showSessions"), panel: <SessionsPanel /> },
    { id: "memories", title: "Memories", value: memoriesSummary(memories), open: ui.showMemories, toggle: () => toggle("showMemories"), panel: <MemoriesPanel /> },
  ];
  const openIds = sections.filter((s) => s.open).map((s) => s.id);
  const shares = sharesFor(ui.railSplits, openIds);

  let prevOpen: { id: RailSectionId; title: string } | null = null;
  const rows: React.ReactNode[] = [];
  for (const s of sections) {
    if (s.open && prevOpen) {
      rows.push(<RailSectionResizer key={`${prevOpen.id}-${s.id}`} aboveId={prevOpen.id} belowId={s.id} aboveTitle={prevOpen.title} belowTitle={s.title} />);
    }
    rows.push(
      <RailSection key={s.id} id={s.id} title={s.title} value={s.value} open={s.open} share={shares[s.id] ?? 1} onToggle={s.toggle}>
        {s.panel}
      </RailSection>,
    );
    if (s.open) prevOpen = { id: s.id, title: s.title };
  }

  return (
    <aside data-rail className="hidden md:flex shrink-0 border-l hairline bg-surface flex-col min-h-0 overflow-hidden" style={{ width: ui.railWidth }} aria-label="Session rail">
      {rows}
    </aside>
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
 * What the TUI lets you do before any session exists, in one plate
 * (design frame 03): resume and start new, side by side.
 *
 * A workspace opened again is not a workspace being set up: when the
 * selected workspace's ``.env`` already binds a provider, the plate offers
 * its previous sessions to resume and a new session on that binding, and
 * asks nothing about providers or sign-in.  Only a workspace with no
 * provider gets the sign-in row, after which the daemon offers to open the
 * session itself.  Nothing here is required -- the link at the foot drops
 * to the prompt, where any daemon command runs with no session, as in the
 * TUI.
 */
function ProfilePicker({ onPick, onAttach, onAuth, onSkip }: {
  onPick: (profile: string | null) => void;
  onAttach: (sessionId: string) => void;
  onAuth: (command: string) => void;
  onSkip: () => void;
}) {
  const profiles = useJaato((s) => s.profiles);
  const commands = useJaato((s) => s.commands);
  const ws = useJaato((s) => s.workspace);
  const setScreen = useJaato((s) => s.setScreen);
  const sessions = useJaato((s) => s.sessions);
  useEffect(() => { ensureSessions().catch(() => undefined); }, []);
  const auth = authCommands(commands);
  const selected = ws.selected ? ws.list.find((w) => w.name === ws.selected) ?? { name: ws.selected } : undefined;
  const cfg = ws.config && ws.config.workspace === ws.selected ? ws.config : undefined;
  const configured = cfg?.configured === true;
  const binding = [cfg?.provider, cfg?.model].filter(Boolean).join(" / ");
  const resumable = selected ? sessionsInWorkspace(sessions, selected) : sessions;
  const row = "flex gap-3 py-2.5 border-t hairline w-full text-left hover:bg-tint/60";
  return (
    <div className="h-full flex items-center justify-center p-6 sm:p-12 overflow-auto">
      <Plate className="w-full max-w-[880px] flex flex-col" data-testid="session-picker">
        <div className="flex items-baseline justify-between gap-4 px-5 py-4 border-b hairline">
          <div className="flex items-baseline gap-3">
            {/* The way back.  `WorkspaceScreen` routes FORWARD here and the
                only routes back were ending a session or disconnecting, so a
                workspace opened by mistake could only be left by leaving the
                daemon.  Gated on the MODE rather than on `selected`: the
                list's own "server-provisioned workspace" link arrives here
                with nothing selected, and that is a state you may equally
                want to back out of.  A bordered button, not a `.link` — what
                was reported is that there is no button to find. */}
            {ws.mode === "enabled" && (
              <button
                type="button"
                onClick={() => setScreen("workspaces")}
                title="Back to the workspace list"
                className="btn btn-sm btn-quiet shrink-0"
              >
                <span aria-hidden="true">←</span> Workspaces
              </button>
            )}
            {selected ? <><span className="kicker tracking-[0.16em]">Workspace</span><span className="font-mono text-[16px]">{selected.name}</span></> : <span className="display text-[20px]">New session</span>}
          </div>
          {configured && <div className="font-mono text-xs text-text-muted">{binding} <span className="text-steel">from .env</span></div>}
        </div>
        <div className={`grid grid-cols-1 ${resumable.length > 0 ? "md:grid-cols-[1fr_1px_1fr]" : ""}`}>
          {resumable.length > 0 && (
            <>
              <div className="px-5 py-4 flex flex-col gap-2.5" aria-label="Resume a session">
                <div className="kicker kicker-muted text-[12px]">Resume</div>
                <div className="flex flex-col max-h-72 overflow-auto">
                  {/* Editable here too, and not read-only as first drawn: if
                      you forgot to write a note on the way out, the picker is
                      exactly where you notice, and attaching just to add one
                      costs a runner spawn -- the cost going BFF-side was
                      meant to avoid. */}
                  {resumable.map((sess) => <SessionRow key={sess.id} sess={sess} onAttach={onAttach} />)}
                </div>
              </div>
              <div className="hidden md:block bg-divider" aria-hidden="true" />
            </>
          )}
          <div className="px-5 py-4 flex flex-col gap-2.5">
            <div className="kicker kicker-muted text-[12px]">{resumable.length > 0 ? "Start new — agent profile" : "Agent profile"}</div>
            <div className="flex flex-col max-h-80 overflow-auto">
              <button type="button" onClick={() => onPick(null)} className={row}>
                <span className="font-mono text-[11px] text-steel w-[22px] shrink-0 pt-0.5">01</span>
                <span className="min-w-0"><span className="block font-mono text-[13px]">default</span><span className="block text-[13px] text-text-muted">{configured ? binding : "the workspace .env provider and model"}</span></span>
              </button>
              {profiles.map((p, i) => (
                <button key={p.name} type="button" onClick={() => onPick(p.name)} className={row}>
                  <span className="font-mono text-[11px] text-steel w-[22px] shrink-0 pt-0.5">{String(i + 2).padStart(2, "0")}</span>
                  <span className="min-w-0"><span className="block font-mono text-[13px]">{p.name}</span><span className="block text-[13px] text-text-muted">{[p.description, [p.provider, p.model].filter(Boolean).join("/")].filter(Boolean).join(" · ")}</span></span>
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="px-5 pt-3 border-t hairline" aria-label="Files for the session">
          <div className="kicker kicker-muted text-[12px] mb-1.5">Files for the session</div>
          <AttachStrip always hint={selected ? "Staged into this workspace before the session opens." : "Staged into the session's workspace as soon as it is provisioned."} />
        </div>
        <div className="flex flex-wrap items-center gap-3 px-5 py-3 border-t hairline">
          {auth.length > 0 && !configured && (
            <div className="flex flex-wrap items-center gap-3" aria-label="Sign in to a provider">
              <span className="kicker kicker-muted">Sign in first</span>
              {auth.map((c) => (
                <button key={c.name} type="button" onClick={() => onAuth(`${c.name} login`)} className="font-mono text-xs border hairline px-2 py-1 hover:border-steel hover:text-steel" title={c.description}>{c.name} login</button>
              ))}
              <span className="text-[12px] text-text-muted">No provider configured yet? Sign in first; the daemon then offers to open the session for you.</span>
            </div>
          )}
          <span className="flex-1" />
          <button type="button" onClick={onSkip} className="link text-[13px]">Go to the prompt without a session</button>
        </div>
      </Plate>
    </div>
  );
}

export function SessionScreen() {
  useKeyboardShortcuts();
  const sessionId = useJaato((s) => s.sessionId);
  const selected = useJaato((s) => s.selectedAgentId);
  const commands = useJaato((s) => s.commands);
  const initProgress = useJaato((s) => s.initProgress);
  const permissions = useJaato((s) => s.permissions);
  const exitChoice = useJaato((s) => s.exitChoice);
  const clarifications = useJaato((s) => s.clarifications);
  const references = useJaato((s) => s.referenceSelections);
  const postAuth = useJaato((s) => s.postAuth);
  const wsConfig = useJaato((s) => s.workspace.config);
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
      // ``createSession`` (app/actions) rather than the SDK call: a new
      // session starts on an empty pane, as an attach always has.
      await openSessionWithQueued(() => createSession(profile));
    } finally {
      setCreating(false);
    }
  };

  const agentPerms = permissions.filter((p) => p.agentId === selected);
  const agentClars = clarifications.filter((c) => c.agentId === selected);
  const agentRefs = references.filter((r) => r.agentId === selected);

  const captureMode = useMemo(() => {
    if (exitChoice) return { kind: "exit" as const, placeholder: `Exit: ${exitChoice.options.map((o) => o.key).join(" · ")}`, suggestions: exitChoice.options.map((o) => o.key) };
    const p = agentPerms[0];
    if (p) return { kind: "permission" as const, placeholder: `Answer ${p.toolName} — ${p.options.map((o) => o.key).join(" · ") || "y · n"}, or type a reply`, suggestions: p.options.map((o) => o.key) };
    const cl = agentClars.find((c) => c.inputMode);
    if (cl) return { kind: "clarification" as const, placeholder: "Type your answer (or a choice number) and press Enter" };
    if (agentRefs[0]) return { kind: "reference" as const, placeholder: "Type the reference to use" };
    return null;
  }, [exitChoice, agentPerms, agentClars, agentRefs]);

  const runAuth = (command: string) => {
    // Leave the picker so the daemon's replies (the login URL, the
    // ``auth.setup`` offer) land on a visible prompt, then run the command
    // exactly as if it had been typed.
    setPicking(false);
    submitInput(command, false).catch((err) => useJaato.getState().addSystemBlock(selected, String(err), "error"));
  };

  const resumeSession = (id: string) => {
    setPicking(false);
    openSessionWithQueued(() => attachSession(id)).catch((err) => useJaato.getState().addSystemBlock(selected, String(err), "error"));
  };

  // A workspace whose .env already names the provider just signed in to has
  // nothing to persist: the card does not ask.
  const alreadyConfigured = Boolean(postAuth && wsConfig?.configured && wsConfig.provider === postAuth.providerName);
  const postAuthCard = postAuth ? (
    <div className="px-5"><PostAuthSetupPrompt p={postAuth} alreadyConfigured={alreadyConfigured} onRespond={(a) => { respondPostAuth(postAuth.requestId, a).catch((err) => useJaato.getState().addSystemBlock(selected, String(err), "error")); }} /></div>
  ) : null;

  if (picking && !sessionId) {
    return (
      <div className="h-full flex flex-col">
        {postAuthCard}
        <div className="flex-1 min-h-0"><ProfilePicker onPick={startSession} onAttach={resumeSession} onAuth={runAuth} onSkip={() => setPicking(false)} /></div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <SessionHeader />
      <div className="flex-1 flex min-h-0">
        <main className="flex-1 flex flex-col min-w-0 relative">
          {(creating || initProgress) && (
            <div className="px-5 py-1 kicker kicker-muted border-b hairline flex items-center gap-2">
              <span className="pulse text-primary">●</span>
              {initProgress ? `${initProgress.message ?? initProgress.step ?? "initialising"}${initProgress.stepNumber != null && initProgress.totalSteps ? ` (${initProgress.stepNumber}/${initProgress.totalSteps})` : ""}` : "Creating session…"}
            </div>
          )}
          <OutputPane agentId={selected} />
          <div className="px-5">
            {exitChoice && <ExitPrompt x={exitChoice} onAnswer={(k) => { answerExit(k).catch((err) => useJaato.getState().addSystemBlock(selected, String(err), "error")); }} />}
            {agentPerms.map((p) => <PermissionPrompt key={p.requestId} p={p} onRespond={(k) => respondPermission(p.requestId, k)} />)}
            {agentClars.map((c) => <ClarificationPrompt key={c.requestId} c={c} onAnswer={(a) => answerClarification(c, a)} onCancel={() => cancelClarification(c)} />)}
            {agentRefs.map((r) => <ReferenceSelectionPrompt key={r.requestId} r={r} onRespond={(v) => respondReference(r.requestId, v)} />)}
            {postAuth && selected === "main" && <PostAuthSetupPrompt p={postAuth} alreadyConfigured={alreadyConfigured} onRespond={(a) => { respondPostAuth(postAuth.requestId, a).catch((err) => useJaato.getState().addSystemBlock(selected, String(err), "error")); }} />}
          </div>
          <div className="px-5 pb-3 pt-2.5 border-t hairline">
            {!captureMode && <PhaseLine agentId={selected} />}
            <Composer commands={commands} history={inputHistory} captureMode={captureMode} onSubmit={(t, v) => { submitInput(t, v).catch((err) => useJaato.getState().addSystemBlock(selected, String(err), "error")); }} />
          </div>
          <ToolOutputPopup agentId={selected} />
        </main>
        <RailResizer />
        <Rail agentId={selected} />
      </div>
      <StatusBar />
    </div>
  );
}
