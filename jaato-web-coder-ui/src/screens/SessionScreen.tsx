/**
 * The main view (design frame 04): the session's identity in a 52px
 * header (brand, one tab per agent, workspace / model / context on the
 * right), the selected agent's output with pending prompts and the
 * composer under it, one persistent rail on the right whose Plan /
 * Budget / Files / Sessions / Memories / Diagnostics sections open and
 * close, and the status bar.  On mount
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
import { DiagnosticsPanel } from "@/components/panels/DiagnosticsPanel";
import { diagnosticsSummary } from "@/app/diagnostics";
import { AgentTabs } from "@/components/panels/AgentTabs";
import { CommandPalette } from "@/components/prompts/CommandPalette";
import { AttentionBanners } from "@/components/panels/AttentionBanners";
import { StatusBar } from "@/components/layout/StatusBar";
import { Plate } from "@/components/layout/Plate";
import { RailResizer } from "@/components/layout/RailResizer";
import { RailSectionResizer } from "@/components/layout/RailSectionResizer";
import { sharesFor, RAIL_SECTION_MIN_PX, type RailPanelId } from "@/store/railSplits";
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
    <header className="flex items-stretch h-[52px] border-b hairline shrink-0">
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
 * The rail badges' icons: hand-drawn stroke SVGs (18x18 viewBox,
 * ``stroke="currentColor" strokeWidth={2}``, no fill) rather than the
 * Unicode glyphs the rail shipped with (``▤``/``⌗``/``$``/``☰``/``◆``/``⚙``
 * — mismatched with each other in visual weight, and with the #1304 board's
 * own custom icon set).  Kept together, one function per icon, so all six
 * share exactly one viewBox and stroke width and cannot drift apart.
 */
const RAIL_ICON_PROPS = {
  viewBox: "0 0 18 18",
  width: 15,
  height: 15,
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
} satisfies React.SVGProps<SVGSVGElement>;

/** Plan: a checklist. */
function PlanBadgeIcon() {
  return (
    <svg {...RAIL_ICON_PROPS}>
      <rect x="2" y="2.25" width="3" height="3" rx="0.5" />
      <path d="M8.5 3.75h7.5" />
      <rect x="2" y="7.5" width="3" height="3" rx="0.5" />
      <path d="M8.5 9h7.5" />
      <rect x="2" y="12.75" width="3" height="3" rx="0.5" />
      <path d="M8.5 14.25h7.5" />
    </svg>
  );
}

/** Files: a document with a folded corner. */
function FilesBadgeIcon() {
  return (
    <svg {...RAIL_ICON_PROPS}>
      <path d="M4.5 2h5l4 4v9.5a1 1 0 0 1-1 1h-8a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1Z" />
      <path d="M9.5 2v4h4" />
    </svg>
  );
}

/** Budget: a clock face. */
function BudgetBadgeIcon() {
  return (
    <svg {...RAIL_ICON_PROPS}>
      <circle cx="9" cy="9" r="7" />
      <path d="M9 5.25v3.75l2.75 1.75" />
    </svg>
  );
}

/** Sessions: two stacked windows. */
function SessionsBadgeIcon() {
  return (
    <svg {...RAIL_ICON_PROPS}>
      <rect x="2.25" y="2.25" width="9.5" height="7.5" rx="1" />
      <path d="M5.75 15.75h9.5a1 1 0 0 0 1-1v-6.5a1 1 0 0 0-1-1h-2.5" />
    </svg>
  );
}

/** Memories: a bookmark. */
function MemoriesBadgeIcon() {
  return (
    <svg {...RAIL_ICON_PROPS}>
      <path d="M4.75 2.5h8.5v13l-4.25-3.25L4.75 15.5Z" />
    </svg>
  );
}

/** Diagnostics: a shield with a checkmark. */
function DiagnosticsBadgeIcon() {
  return (
    <svg {...RAIL_ICON_PROPS}>
      <path d="M9 1.75 15 4v4.25c0 4-2.6 6.5-6 7.5-3.4-1-6-3.5-6-7.5V4Z" />
      <path d="M6.25 9 8.25 11 11.75 6.75" />
    </svg>
  );
}

/**
 * The 56px icon rail (#1304 §5), replacing the six-section accordion and
 * the StatusBar's duplicated toggle buttons with ONE panel open at a time
 * (``ui.activePanel``) plus an optional PINNED Plan (``ui.pinnedPlan``).
 *
 * Each badge button keeps the accordion's own ``aria-label`` convention
 * (``${open ? "Close" : "Open"} ${title}``) so existing e2e assertions on
 * "Open Plan" / "Open Sessions" / etc. keep working — only the trigger
 * moved, from a header row to an icon in the rail.  Clicking the badge for
 * the panel that is already open CLOSES it (``setActivePanel`` toggles),
 * the same behaviour the accordion's header click had.
 *
 * Plan is the one panel that may be shown ALONGSIDE another: pinning it
 * keeps it open no matter which other badge is selected, and the two share
 * height via the same pure ``railSplits`` math and ``RailSectionResizer``
 * drag handle the pre-#1304 accordion used for any two open sections —
 * there is exactly one such pair now (Plan + whichever panel is active),
 * so the reuse is a straight application of existing, tested code.
 */
const RAIL_BADGES: { id: RailPanelId; title: string; Icon: () => React.ReactElement }[] = [
  { id: "plan", title: "Plan", Icon: PlanBadgeIcon },
  { id: "files", title: "Files", Icon: FilesBadgeIcon },
  { id: "budget", title: "Budget", Icon: BudgetBadgeIcon },
  { id: "sessions", title: "Sessions", Icon: SessionsBadgeIcon },
  { id: "memories", title: "Memories", Icon: MemoriesBadgeIcon },
  { id: "diagnostics", title: "Diagnostics", Icon: DiagnosticsBadgeIcon },
];

function RailPanelSection({ id, title, share, showHeader, children }: { id: RailPanelId; title: string; share?: number; showHeader: boolean; children: React.ReactNode }) {
  const togglePinPlan = useJaato((s) => s.togglePinPlan);
  const pinned = useJaato((s) => s.ui.pinnedPlan);
  return (
    <section
      aria-label={title}
      data-rail-section={id}
      className="overflow-auto min-h-0 border-b hairline last:border-b-0"
      style={share != null ? { flexGrow: share, flexShrink: 1, flexBasis: 0, minHeight: RAIL_SECTION_MIN_PX } : { flex: 1 }}
    >
      {showHeader && (
        <div className="flex items-center justify-between px-3.5 py-2 border-b hairline sticky top-0 bg-surface">
          <span className="kicker">{title}</span>
          {id === "plan" && (
            <button
              type="button"
              onClick={togglePinPlan}
              aria-pressed={pinned}
              title={pinned ? "Unpin the plan panel" : "Pin the plan panel open beside another panel"}
              className={`chrome-sm ${pinned ? "text-steel" : "text-text-muted hover:text-steel"}`}
            >
              {pinned ? "pinned" : "pin"}
            </button>
          )}
        </div>
      )}
      {children}
    </section>
  );
}

function Rail({ agentId }: { agentId: string }) {
  const ui = useJaato((s) => s.ui);
  const setActivePanel = useJaato((s) => s.setActivePanel);
  const plan = useJaato((s) => s.plan[agentId]);
  const ctx = useJaato((s) => s.context[agentId]);
  const changed = Object.keys(useVisibleWorkspaceFiles()).length;
  const isReset = useJaato((s) => s.workspaceReset !== null);
  const sessions = useJaato((s) => s.sessions);
  const notes = useJaato((s) => s.notes);
  const memories = useJaato((s) => s.memories);
  const diagnostics = useJaato((s) => s.diagnostics);
  const budget = ctx?.usage.cost_usd != null ? `$${Number(ctx.usage.cost_usd).toFixed(4)}` : ctx?.percentUsed != null ? `${ctx.percentUsed.toFixed(0)}%` : null;

  const values: Record<RailPanelId, string | null> = {
    plan: planProgress(plan),
    files: changed ? String(changed) : isReset ? "0" : null,
    budget,
    sessions: notedSummary(sessions, notes),
    memories: memoriesSummary(memories),
    diagnostics: diagnosticsSummary(diagnostics),
  };
  const panels: Record<RailPanelId, React.ReactNode> = {
    plan: <PlanPanel agentId={agentId} />,
    files: <WorkspacePanel />,
    budget: <BudgetPanel agentId={agentId} />,
    sessions: <SessionsPanel />,
    memories: <MemoriesPanel />,
    diagnostics: <DiagnosticsPanel />,
  };

  const active = ui.activePanel;
  const showPlanPinned = ui.pinnedPlan && active !== null && active !== "plan";
  const openIds: RailPanelId[] = showPlanPinned && active ? ["plan", active] : active ? [active] : [];
  const shares = sharesFor(ui.railSplits, openIds);
  const activeTitle = active ? RAIL_BADGES.find((b) => b.id === active)?.title ?? active : "";

  return (
    <aside data-rail className="hidden md:flex shrink-0 border-l hairline bg-surface flex-row min-h-0 overflow-hidden" style={{ width: ui.railWidth }} aria-label="Session rail">
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
        {active ? (
          <>
            {showPlanPinned && (
              <>
                <RailPanelSection id="plan" title="Plan" share={shares.plan ?? 1} showHeader>
                  {panels.plan}
                </RailPanelSection>
                <RailSectionResizer aboveId="plan" belowId={active} aboveTitle="Plan" belowTitle={activeTitle} />
              </>
            )}
            <RailPanelSection id={active} title={activeTitle} share={showPlanPinned ? (shares[active] ?? 1) : undefined} showHeader>
              {panels[active]}
            </RailPanelSection>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-[13px] text-text-muted italic px-4 text-center">
            Pick a panel from the rail — Plan, Files, Budget, Sessions, Memories or Diagnostics.
          </div>
        )}
      </div>
      <div className="w-14 shrink-0 border-l hairline flex flex-col items-stretch py-1" role="toolbar" aria-label="Panels">
        {RAIL_BADGES.map((b) => {
          const open = active === b.id;
          const isPinnedPlan = b.id === "plan" && ui.pinnedPlan && !open;
          return (
            <button
              key={b.id}
              type="button"
              onClick={() => setActivePanel(b.id)}
              aria-expanded={open}
              aria-label={`${open ? "Close" : "Open"} ${b.title}`}
              title={b.title}
              className={`flex flex-col items-center justify-center gap-0.5 py-2 border-b hairline hover:bg-tint/60 ${open || isPinnedPlan ? "bg-tint text-steel" : "text-text-muted"}`}
            >
              <span className="leading-none" aria-hidden="true"><b.Icon /></span>
              <span className="chrome-xs uppercase tracking-[0.06em] leading-none">{b.title.slice(0, 4)}</span>
              {values[b.id] && <span className="font-mono text-[9px] leading-none text-text-muted">{values[b.id]}</span>}
            </button>
          );
        })}
      </div>
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
  const paletteOpen = useJaato((s) => s.paletteOpen);
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
        {paletteOpen && <CommandPalette />}
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
          <AttentionBanners />
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
      {paletteOpen && <CommandPalette />}
    </div>
  );
}
