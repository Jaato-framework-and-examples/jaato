/**
 * The one store.  ``reduce(state, event)`` folds every wire event into
 * client state; components subscribe to slices through ``useJaato``.
 *
 * Reduction is deliberately event-sourced and framework-free so it can
 * be unit-tested by replaying recorded event streams.  Streaming
 * chunks are batched by the adapter (``sdk/connection.ts``) and
 * applied in one ``set`` per animation frame, so a token-rate stream
 * costs one React commit per frame, not one per token.
 */
import { create } from "zustand";
import { EventTypeValue, type JaatoEvent } from "@jaato/sdk";
import { mergeCommandSpecs, type CommandSpec } from "@/protocol/commands";
import { normalizeClarificationQuestion } from "@/protocol/clarification";
import { summarizeToolCalls } from "@/protocol/turnStats";
import { formatSessionList, normalizeSessionList, type SessionSummary } from "@/protocol/sessions";
import { formatHistoryListing, historyBlocks } from "@/protocol/history";
import { clampRailWidth, loadRailWidth, saveRailWidth } from "@/store/railWidth";
import { loadRailSplits, sanitizeSplits, saveRailSplits, type RailSplits } from "@/store/railSplits";
import { applyChanged, applySnapshot, markReset, type WorkspaceReset } from "@/store/workspaceView";
import { toolIdMappings } from "@/protocol/toolIds";
import type { ToolClass } from "@/protocol/toolClass";
import { faultFromError, type SessionFault } from "@/protocol/sessionFault";
import { clampStallThreshold, DEFAULT_STALL_THRESHOLD_MS } from "@/store/phase";
import type { RailPanelId } from "@/store/railSplits";
import type { SessionNote } from "@/app/notes";

/**
 * What the note editor says about the last save, rendered in the section's
 * ``value`` slot so a failure is visible without a toast.
 *
 * ``signed-out`` is deliberately not folded into ``error``: BFF-side storage
 * means the cookie can expire mid-session, and "signed out" is a different
 * thing to tell somebody than "could not save".
 */
export type NoteStatus =
  | { state: "idle" }
  | { state: "saving" }
  | { state: "saved"; at: number }
  | { state: "error"; message: string }
  | { state: "signed-out"; loginUrl: string };
import { BUSY_STATUS } from "@/store/phase";
import type {
  StagedUpload,
  ExitChoice,
  ExitOption,
  UserBlock,
  Agent,
  ConfigStatus,
  ConnectionPhase,
  BudgetState,
  GcState,
  ContextState,
  MemoriesState,
  DiagnosticsState,
  InitProgress,
  OutputBlock,
  PendingClarification,
  PendingPermission,
  PendingReferenceSelection, PendingPostAuthSetup,
  PlanState,
  ProfileInfo,
  Screen,
  ToolBlock,
  ToolStatus,
  WorkspaceInfo,
} from "./types";

export const MAIN_AGENT = "main";

let seq = 0;
const nextId = () => `b${++seq}`;

// Any event; the generated union is discriminated on ``type`` but many
// payload fields are optional, so reducers read through a loose view.
type AnyEvent = JaatoEvent & Record<string, unknown>;

export interface JaatoState {
  connection: { phase: ConnectionPhase; detail?: string; attempt?: number; serverVersion?: string | null; protocolVersion?: string | null };
  url: string;
  screen: Screen;
  /**
   * The sign-in backend's per-user key store (``config.json``'s
   * ``credentialsUrl``, see ``app/credentials.ts``), or ``null`` when the
   * page was served without one.  Read by the workspace configure form to
   * offer previously used keys instead of asking for the key again.
   */
  credentialsUrl: string | null;
  /**
   * The sign-in backend's per-user session-note store (``config.json``'s
   * ``notesUrl``, see ``app/notes.ts``), or ``null`` when the page was
   * served without one -- in which case notes are kept in this browser
   * instead, which the UI says rather than pretending otherwise.
   */
  notesUrl: string | null;
  /**
   * The sign-in backend's per-user GitHub connection API (``config.json``'s
   * ``githubUrl``, see ``app/github.ts``), or ``null`` when the page was
   * served without a ``github:`` block.  Its presence is the whole gate on
   * the "Connect GitHub" settings entry and the workspace account dropdown.
   */
  githubUrl: string | null;
  /**
   * Where the server-side GitHub App connect starts (``config.json``'s
   * ``githubLoginUrl``), or ``null``.  The settings entry navigates here.
   */
  githubLoginUrl: string | null;
  /**
   * The notes THIS person has written, by session id.  Not part of
   * ``emptySessionState``: a note outlives the session being attached,
   * detached or swapped, and the listing it joins onto is cross-workspace.
   */
  notes: Record<string, SessionNote>;
  /** Per session: what the editor should say about the last save. */
  noteStatus: Record<string, NoteStatus>;
  /**
   * The sign-in backend behind this page, when ``config.json`` named a
   * ticket URL: where "Sign out" goes and who the backend says is signed
   * in (``null`` until answered, or when there is no backend).  Read by the
   * workspace screen's header; the connect screen sets it from the launcher
   * config (``app/backendSession.ts``).
   */
  backend: { logoutUrl: string; user: string | null } | null;

  workspace: {
    mode: "unknown" | "enabled" | "disabled";
    root?: string;
    list: WorkspaceInfo[];
    selected?: string;
    config?: ConfigStatus;
    /** The daemon's answer to the last ``workspace.delete``, for the workspace screen. */
    notice?: { text: string; error?: boolean } | null;
  };

  sessionId?: string | null;
  session: { name?: string | null; provider?: string | null; model?: string | null; profile?: string | null; models?: string[] };
  profiles: ProfileInfo[];
  /**
   * The daemon's session listing — every session it knows, whichever
   * workspace.  Refreshed by ``SessionListEvent`` (the answer to ``session
   * list``) and by the snapshot every ``SessionInfoEvent`` carries.  Feeds
   * the ``session attach <id>`` completion and the picker's resume list.
   */
  sessions: SessionSummary[];
  /**
   * How many ``SessionListEvent`` replies are owed to silent requests —
   * the completer's and the picker's, which want the listing above and not
   * a printed one.  Each such request adds one before it asks and each
   * reply consumes one, so a listing the user did not type ``session
   * list`` for is never written to the output.  A COUNT, not a flag: the
   * picker's request fires twice under React's development double-effect,
   * and a flag cleared by the first reply let the second print.
   */
  sessionListSilent: number;
  /**
   * How the next ``HistoryEvent`` renders: ``listing`` (the ``history``
   * command's summary) or ``replay`` (the conversation rebuilt as blocks,
   * after ``session attach``).
   */
  historyMode: "listing" | "replay";
  initProgress?: InitProgress | null;

  agents: Record<string, Agent>;
  agentOrder: string[];
  selectedAgentId: string;

  blocks: Record<string, OutputBlock[]>;
  /** call_id → agent that owns the tool block (tool events route by call_id). */
  toolOwner: Record<string, string>;

  permissions: PendingPermission[];
  clarifications: PendingClarification[];
  referenceSelections: PendingReferenceSelection[];
  /** The daemon's pending ``auth.setup`` offer, if any (see PendingPostAuthSetup). */
  postAuth: PendingPostAuthSetup | null;

  plan: Record<string, PlanState>;
  context: Record<string, ContextState>;
  /** What the window is spent ON, by instruction source (see BudgetState). */
  budget: Record<string, BudgetState>;
  /** GC policy and last pass, per agent (see GcState). */
  gc: Record<string, GcState>;
  /** Budget source layers whose children are showing — the TUI panel's drill-down. */
  budgetExpanded: string[];
  commands: CommandSpec[];
  /** Every file the session changed, path -> status: the FULL list, whatever the panel's reset point. */
  workspaceFiles: Record<string, string>;
  /**
   * Hashed tool / category id -> the name a person knows, from
   * ``tools.id_registry`` and ``session.info.tool_id_mappings``
   * (``protocol/toolIds.ts``).  Replaced wholesale on each receive -- the
   * daemon always sends the full current set.
   */
  toolIdNames: Record<string, string>;
  /** Path -> the daemon's number for its latest change (#1189, ``store/workspaceView.ts``). */
  workspaceSeqs: Record<string, number>;
  /** The workspace monitor those numbers belong to; ``null`` from a daemon that numbers nothing. */
  workspaceEpoch: string | null;
  /** The highest change number seen -- what a reset records. */
  workspaceSeq: number;
  /**
   * The Files panel's reset point (the TUI's ``workspace_clear``): when set,
   * the panel shows only files changed after it.  Per viewer and in memory,
   * like ``workspaceHidden``; voided -- with a notice -- when the daemon's
   * numbering says it can no longer be honoured.
   */
  workspaceReset: WorkspaceReset | null;
  /**
   * Entries hidden from the Files panel this session — the TUI panel's
   * ``h`` key.  A directory is stored with its trailing ``/`` and hides
   * everything under it.  Client-side only: nothing on the daemon changes,
   * and the set is dropped with the session.
   */
  workspaceHidden: string[];
  /** Show hidden entries (dimmed, with an ``H`` marker) so they can be unhidden. */
  workspaceShowHidden: boolean;
  /**
   * Directories collapsed in the Files panel -- the TUI panel's Left/Right
   * keys.  Entry ids, so each carries its trailing ``/``.  Every directory
   * starts expanded; a reset expands them all again, as the TUI's clear
   * does.  Client-side only.
   */
  workspaceCollapsed: string[];
  /**
   * What the daemon last said about an entry's ``.gitignore`` line, from
   * ``workspace.ignore.result`` — learned, not derived: the client never
   * reads the file, so an entry absent here has unknown state.
   */
  workspaceIgnored: Record<string, boolean>;
  /** One-line outcome of the last ``.gitignore`` toggle, shown in the panel. */
  workspaceNotice: { text: string; error?: boolean } | null;
  /** Files attached from the browser, in the order they were picked (see ``StagedUpload``). */
  uploads: StagedUpload[];
  /** ``PermissionStatusEvent``: the effective default policy and, when suspended, the scope. */
  permissionStatus?: { effectiveDefault: string; suspensionScope: string | null; autoAllowHousekeeping: boolean | null } | null;
  /**
   * When each agent became busy, for the phase indicator's elapsed clock.
   * Set the moment the daemon reports ``active`` (or the composer sends,
   * whichever is first) and dropped when the turn ends.  Absent = not busy;
   * ``store/phase.ts`` is what reads it.
   */
  busySince: Record<string, number>;
  /** The open exit confirmation, or ``null`` (``app/exitChoice.ts``). */
  exitChoice: ExitChoice | null;
  /** The command palette (#1304 §6): Ctrl/⌘+K, and where ``help`` now lands instead of the transcript. */
  paletteOpen: boolean;
  /**
   * A non-recoverable ``ErrorEvent`` (``recoverable: false`` --
   * ``RunnerBootstrapFailed`` among them, see ``protocol/sessionFault.ts``)
   * that means this session never came up.  The transport can be perfectly
   * connected while this is set -- it is a fact about the SESSION, not the
   * WebSocket -- which is what the status bar reads to show red "no
   * session" instead of a green "connected" that was never wrong about the
   * transport and always wrong about the session.  Cleared by a session
   * that DOES come up (a fresh ``SESSION_INFO``) or by ``resetSessionState``.
   */
  sessionFault: SessionFault | null;
  /** The rail's Memories section (#1232, ``app/memories.ts``). */
  memories: MemoriesState;
  /** The rail's Diagnostics section (#1294, ``app/diagnostics.ts``). */
  diagnostics: DiagnosticsState;
  /**
   * The last moment ANY event named this agent (#1304 §3): stamped
   * generically in ``reduce`` for every agent-scoped event type, not per
   * event handler, so a new one is covered without anyone remembering to
   * wire it.  Read by ``store/phase.ts``'s ``stalled`` -- an agent that is
   * ``thinking``/``sending`` with nothing heard from it past the threshold
   * reads as stalled; a running tool or a pending prompt is not silence
   * and never reads that way.
   */
  lastEventAt: Record<string, number>;
  /**
   * How long an agent may go silent while ``thinking``/``sending`` before
   * it reads as stalled (#1304 §3).  A global setting, not per-session --
   * it survives ``resetSessionState`` -- and deliberately not persisted to
   * ``localStorage`` the way ``ui.railWidth`` is: there is no settings UI
   * for it yet (the issue: "not necessarily UI-exposed unless cheap"), so
   * persisting a value nothing lets you set back would be a knob a test or
   * a stray ``setState`` could leave the wrong way for the next session.
   * ``setStallThreshold`` clamps to the documented 30s-300s range; a raw
   * ``useJaato.setState({stallThresholdMs: ...})`` (a test fixture) does
   * not, which is deliberate -- see ``store/phase.ts``'s own clamp for why
   * that is still safe.
   */
  stallThresholdMs: number;

  ui: {
    showTools: boolean;
    /**
     * Which ONE rail panel the 56px icon rail is showing (#1304 §5) --
     * ``null`` when none is.  Replaces the six independent ``show*``
     * booleans the accordion used: only one panel is on screen at a time,
     * selected by clicking its icon.
     */
    activePanel: RailPanelId | null;
    /**
     * Plan pinned open (#1304 §5): when true, Plan stays visible beside
     * whichever OTHER panel ``activePanel`` names, sharing the rail's
     * height with it via the same ``railSplits`` weights the old
     * multi-open accordion used.  Selecting Plan itself as the active
     * panel is unaffected by this flag either way.
     */
    pinnedPlan: boolean;
    theme: string;
    /** Tool call currently pinned in the live-output popup. */
    popupCallId?: string | null;
    /** Width of the session rail in px, dragged via the handle on its left edge; remembered per browser. */
    railWidth: number;
    /**
     * How the rail's open sections share its height, as a weight per section
     * id, dragged via the horizontal handle between two open sections;
     * remembered per browser.  Normalised to fractions over the open set at
     * render time — see ``@/store/railSplits``.
     */
    railSplits: RailSplits;
  };

  // ── actions ──
  dispatch: (events: JaatoEvent[]) => void;
  setConnection: (c: Partial<JaatoState["connection"]>) => void;
  setUrl: (url: string) => void;
  setScreen: (s: Screen) => void;
  setCredentialsUrl: (url: string | null) => void;
  setNotesUrl: (url: string | null) => void;
  setGithubUrls: (urls: { githubUrl: string | null; githubLoginUrl: string | null }) => void;
  /** Replace the whole set, from one listing. */
  setNotes: (notes: SessionNote[]) => void;
  /** Upsert one, or drop it when ``null`` (an emptied note is a forgotten one). */
  setNote: (sessionId: string, note: SessionNote | null) => void;
  setNoteStatus: (sessionId: string, status: NoteStatus) => void;
  setBackend: (b: { logoutUrl: string; user: string | null } | null) => void;
  setWorkspaceMode: (m: JaatoState["workspace"]["mode"]) => void;
  selectWorkspace: (name: string | undefined) => void;
  selectAgent: (id: string) => void;
  addUserBlock: (agentId: string, text: string) => void;
  /**
   * The composer sent; the agent counts as busy until the daemon's first
   * word about it arrives, so the indicator does not go dark over the
   * round trip.  This is the ONLY optimistic piece of the phase --
   * everything else is the daemon's own account.  ``clearSending`` gives
   * it back when the send failed, so a refused send does not leave the
   * indicator claiming work is under way.
   *
   * It replaces a fabricated ``AgentStatusChangedEvent`` the client used
   * to dispatch at itself carrying ``status: "processing"`` -- a word no
   * daemon emits, which the real ``active`` then read as "not busy" and
   * switched the indicator back OFF one round trip later.
   */
  markSending: (agentId: string) => void;
  clearSending: (agentId: string) => void;
  addSystemBlock: (agentId: string, text: string, style?: string) => void;
  clearOutput: (agentId: string) => void;
  toggleTool: (agentId: string, blockId: string) => void;
  setAllToolsExpanded: (agentId: string, expanded: boolean) => void;
  resolvePermission: (requestId: string) => void;
  openExitChoice: (running: boolean, options: ExitOption[]) => void;
  closeExitChoice: () => void;
  focusExitChoice: (focus: number) => void;
  focusPermission: (requestId: string, focus: number) => void;
  answerClarification: (requestId: string, answer: string) => PendingClarification | undefined;
  dismissClarification: (requestId: string) => void;
  dismissReferenceSelection: (requestId: string) => void;
  dismissPostAuth: () => void;
  toggleUi: (key: "showTools") => void;
  /** Open or close the command palette (Ctrl/⌘+K, or the ``help`` command). */
  setPaletteOpen: (open: boolean) => void;
  /** Select (or, given the panel already showing, close) the rail's one active panel. */
  setActivePanel: (id: RailPanelId | null) => void;
  /** Pin (or unpin) Plan open beside whichever panel is active. */
  togglePinPlan: () => void;
  /** Clamped to the documented 30s-300s range (#1304 §3). */
  setStallThreshold: (ms: number) => void;
  /** Merge into the Memories section's state (``app/memories.ts`` is its one writer). */
  patchMemories: (patch: Partial<MemoriesState> | ((m: MemoriesState) => Partial<MemoriesState>)) => void;
  /** Merge into the Diagnostics section's state (``app/diagnostics.ts`` is its one writer). */
  patchDiagnostics: (patch: Partial<DiagnosticsState> | ((d: DiagnosticsState) => Partial<DiagnosticsState>)) => void;
  /** The TUI's Ctrl+T: expand or collapse every tool block, and new ones follow. */
  setToolsExpanded: (expanded: boolean) => void;
  /** Add (``+1``, before a silent request) or give back (``-1``, when it failed to send) one silent reply. */
  setSessionListSilent: (delta: 1 | -1) => void;
  setHistoryMode: (mode: JaatoState["historyMode"]) => void;
  /** Show or hide one budget source's children. */
  toggleBudgetSource: (source: string) => void;
  toggleWorkspaceHidden: (entryId: string) => void;
  toggleWorkspaceShowHidden: () => void;
  toggleWorkspaceCollapsed: (entryId: string) => void;
  /** Reset the Files panel: from now on it shows only files that change after this moment. */
  resetWorkspaceView: () => void;
  /** Drop the reset point and show every file the session changed. */
  showAllWorkspace: () => void;
  setWorkspaceNotice: (n: JaatoState["workspaceNotice"]) => void;
  /** The workspace SCREEN's status line (``workspace.notice``), as opposed to the Files panel's above. */
  setWorkspaceListNotice: (n: JaatoState["workspace"]["notice"]) => void;
  setTheme: (t: string) => void;
  setPopup: (callId: string | null) => void;
  /** Clamped to the rail's bounds and persisted. */
  setRailWidth: (w: number) => void;
  /** Replace the rail section split weights (from a drag or a reset) and persist. */
  setRailSplits: (splits: RailSplits) => void;
  addUploads: (items: StagedUpload[]) => void;
  updateUpload: (id: string, patch: Partial<StagedUpload>) => void;
  removeUpload: (id: string) => void;
  /**
   * Consume the strip when a message is sent: returns the paths that were
   * staged (for the prompt's footer) and drops every settled entry.  A
   * file still ``queued`` or ``staging`` is left for the next send.
   */
  takeUploads: () => string[];
  resetSessionState: () => void;
}

const emptySessionState = () => ({
  sessionId: null,
  session: {},
  initProgress: null,
  agents: { [MAIN_AGENT]: { id: MAIN_AGENT, name: "main", type: "main", status: "idle" } } as Record<string, Agent>,
  agentOrder: [MAIN_AGENT],
  selectedAgentId: MAIN_AGENT,
  blocks: { [MAIN_AGENT]: [] } as Record<string, OutputBlock[]>,
  toolOwner: {} as Record<string, string>,
  permissions: [] as PendingPermission[],
  clarifications: [] as PendingClarification[],
  referenceSelections: [] as PendingReferenceSelection[],
  postAuth: null as PendingPostAuthSetup | null,
  plan: {} as Record<string, PlanState>,
  context: {} as Record<string, ContextState>,
  budget: {} as Record<string, BudgetState>,
  gc: {} as Record<string, GcState>,
  budgetExpanded: [] as string[],
  toolIdNames: {} as Record<string, string>,
  workspaceFiles: {} as Record<string, string>,
  workspaceSeqs: {} as Record<string, number>,
  workspaceEpoch: null as string | null,
  workspaceSeq: 0,
  workspaceReset: null as WorkspaceReset | null,
  workspaceHidden: [] as string[],
  workspaceShowHidden: false,
  workspaceCollapsed: [] as string[],
  workspaceIgnored: {} as Record<string, boolean>,
  workspaceNotice: null as { text: string; error?: boolean } | null,
  permissionStatus: null,
  busySince: {} as Record<string, number>,
  exitChoice: null as ExitChoice | null,
  paletteOpen: false,
  sessionFault: null as SessionFault | null,
  memories: emptyMemories(),
  diagnostics: emptyDiagnostics(),
  lastEventAt: {} as Record<string, number>,
});

/** The Memories section before anything was asked (and after a session change). */
export function emptyMemories(): MemoriesState {
  return {
    rows: [], status: "idle", error: null, mayCurate: null, thisSessionOnly: false,
    expanded: null, details: {}, busy: {}, editing: {}, notice: null,
  };
}

/** The Diagnostics section before anything was asked (and after a session change). */
export function emptyDiagnostics(): DiagnosticsState {
  return {
    status: "idle", error: null, category: null, runnerIdentity: null, confinementId: "",
    sandboxMode: null, consumption: null, notebookBoundaryKind: null, protocolVersion: "",
    serverVersion: "", probe: null, checkedAt: null,
  };
}

/** Drop one key, returning a new record (React identity) — or the same one. */
function without<T>(rec: Record<string, T>, key: string): Record<string, T> {
  if (!(key in rec)) return rec;
  const next = { ...rec };
  delete next[key];
  return next;
}

/** Stamp ``key`` with now, unless it is already stamped (the clock must not restart). */
function startedAt(rec: Record<string, number>, key: string): Record<string, number> {
  return key in rec ? rec : { ...rec, [key]: Date.now() };
}

/**
 * The context an upload belongs to (#1250): the active session once one
 * exists, else ``""`` (the picker, before a session opens).  Stamped onto
 * each upload at attach time and re-read to filter the attach strip, so a
 * file attached in one session is not shown in another.
 */
export function uploadScope(st: { sessionId?: string | null }): string {
  return st.sessionId ? `session:${st.sessionId}` : "";
}

function agentOf(ev: AnyEvent): string {
  const a = ev.agent_id;
  return typeof a === "string" && a ? a : MAIN_AGENT;
}

function ensureAgent(s: JaatoState, id: string): void {
  if (!s.agents[id]) {
    s.agents = { ...s.agents, [id]: { id, name: id, type: "subagent", status: "idle" } };
    s.agentOrder = [...s.agentOrder, id];
  }
  if (!s.blocks[id]) s.blocks = { ...s.blocks, [id]: [] };
}

/** Replace one agent's block list (new array identity → React sees the change). */
function setBlocks(s: JaatoState, agentId: string, blocks: OutputBlock[]): void {
  s.blocks = { ...s.blocks, [agentId]: blocks };
}

function findTool(s: JaatoState, callId: string | undefined | null, agentHint: string): { agentId: string; index: number } | null {
  if (!callId) return null;
  const agentId = s.toolOwner[callId] ?? agentHint;
  const list = s.blocks[agentId] ?? [];
  for (let i = list.length - 1; i >= 0; i--) {
    const b = list[i];
    if (b && b.kind === "tool" && b.callId === callId) return { agentId, index: i };
  }
  return null;
}

function updateTool(s: JaatoState, callId: string | undefined | null, agentHint: string, fn: (t: ToolBlock) => ToolBlock): boolean {
  const hit = findTool(s, callId, agentHint);
  if (!hit) return false;
  const list = [...(s.blocks[hit.agentId] ?? [])];
  list[hit.index] = fn(list[hit.index] as ToolBlock);
  setBlocks(s, hit.agentId, list);
  return true;
}

function upsertPermission(s: JaatoState, ev: AnyEvent, inputMode: boolean): void {
  const requestId = String(ev.request_id ?? "");
  if (!requestId) return;
  const existing = s.permissions.find((p) => p.requestId === requestId);
  const options = Array.isArray(ev.response_options) && ev.response_options.length
    ? (ev.response_options as PendingPermission["options"])
    : existing?.options ?? [];
  const merged: PendingPermission = {
    requestId,
    agentId: agentOf(ev),
    toolName: String(ev.tool_name ?? existing?.toolName ?? ""),
    callId: (ev.call_id as string | null | undefined) ?? existing?.callId ?? null,
    toolArgs: (ev.tool_args as Record<string, unknown> | null | undefined) ?? existing?.toolArgs ?? {},
    options,
    promptLines: (ev.prompt_lines as string[] | null | undefined) ?? existing?.promptLines ?? [],
    formatHint: (ev.format_hint as string | null | undefined) ?? existing?.formatHint ?? null,
    warnings: (ev.warnings as string | null | undefined) ?? existing?.warnings ?? null,
    warningLevel: (ev.warning_level as string | null | undefined) ?? existing?.warningLevel ?? null,
    editableMetadata: (ev.editable_metadata as Record<string, unknown> | null | undefined) ?? existing?.editableMetadata ?? null,
    focus: existing?.focus ?? 0,
    inputMode: inputMode || (existing?.inputMode ?? false),
    toolClass: (ev.tool_class as ToolClass | null | undefined) ?? existing?.toolClass ?? null,
  };
  s.permissions = existing
    ? s.permissions.map((p) => (p.requestId === requestId ? merged : p))
    : [...s.permissions, merged];
  // A prompt is the one thing that cannot wait behind an unknown agent: it
  // blocks that agent's turn until somebody answers, so give it a tab.
  ensureAgent(s, merged.agentId);
}

/**
 * Event types that name ONE agent's activity (#1304 §3): the set
 * ``lastEventAt`` stamps from, generically, rather than one stamp per
 * ``case`` in the switch below -- a stall predicate reads silence as
 * meaningful, so a handler added later that forgets to stamp would be a
 * false stall, not a missed feature.  Deliberately excludes session- and
 * workspace-scoped events (``SESSION_LIST``, ``WORKSPACE_*``,
 * ``COMMAND_LIST``, ``CONFIG_*``, ``CONNECTED``, ``SESSION_PROFILES``) and
 * the two the switch below routes to ``selectedAgentId`` as a fallback
 * (``SESSION_TERMINATED`` / ``DISCONNECTED``) rather than to a named
 * agent: none of those say anything about whether a PARTICULAR agent is
 * still there.
 */
const AGENT_ACTIVITY_EVENT_TYPES = new Set<string>([
  EventTypeValue.AGENT_OUTPUT,
  EventTypeValue.AGENT_STATUS_CHANGED,
  EventTypeValue.AGENT_COMPLETED,
  EventTypeValue.AGENT_ERROR,
  EventTypeValue.TOOL_CALL_START,
  EventTypeValue.TOOL_CALL_END,
  EventTypeValue.TOOL_OUTPUT,
  EventTypeValue.PERMISSION_REQUESTED,
  EventTypeValue.PERMISSION_INPUT_MODE,
  EventTypeValue.PERMISSION_RESOLVED,
  EventTypeValue.CLARIFICATION_BATCH,
  EventTypeValue.CLARIFICATION_QUESTION,
  EventTypeValue.CLARIFICATION_INPUT_MODE,
  EventTypeValue.CLARIFICATION_RESOLVED,
  EventTypeValue.REFERENCE_SELECTION_REQUESTED,
  EventTypeValue.REFERENCE_SELECTION_RESOLVED,
  EventTypeValue.PLAN_UPDATED,
  EventTypeValue.PLAN_STEP_UPDATED,
  EventTypeValue.PLAN_CLEARED,
  EventTypeValue.CONTEXT_UPDATED,
  EventTypeValue.INSTRUCTION_BUDGET_UPDATED,
  EventTypeValue.TURN_COMPLETED,
  EventTypeValue.GC_CONFIG,
  EventTypeValue.GC,
]);

/**
 * Fold one event into a shallow copy of the state.  Returns the same
 * object it was handed (mutated) — callers spread it into ``set``.
 */
export function reduce(s: JaatoState, raw: JaatoEvent): JaatoState {
  const ev = raw as AnyEvent;
  // ``ev.type`` is optional on a couple of generated request/response
  // shapes in the union (e.g. ``ReferenceSelectionResponseRequest``), so
  // this reads as ``string | undefined`` at the type level even though
  // every event actually delivered here carries one.
  if (ev.type && AGENT_ACTIVITY_EVENT_TYPES.has(ev.type)) {
    s.lastEventAt = { ...s.lastEventAt, [agentOf(ev)]: Date.now() };
  }
  switch (ev.type) {
    case EventTypeValue.CONNECTED: {
      const info = (ev.server_info as Record<string, unknown> | undefined) ?? {};
      s.connection = {
        ...s.connection,
        phase: "connected",
        serverVersion: (info.server_version as string | undefined) ?? s.connection.serverVersion ?? null,
        protocolVersion: (ev.protocol_version as string | undefined) ?? s.connection.protocolVersion ?? null,
      };
      break;
    }
    case EventTypeValue.AGENT_CREATED: {
      const id = agentOf(ev);
      const agent: Agent = {
        id,
        name: String(ev.agent_name ?? id),
        type: String(ev.agent_type ?? (id === MAIN_AGENT ? "main" : "subagent")),
        parentId: (ev.parent_agent_id as string | null | undefined) ?? null,
        profile: (ev.profile_name as string | null | undefined) ?? null,
        status: s.agents[id]?.status ?? "idle",
      };
      s.agents = { ...s.agents, [id]: agent };
      if (!s.agentOrder.includes(id)) s.agentOrder = [...s.agentOrder, id];
      if (!s.blocks[id]) s.blocks = { ...s.blocks, [id]: [] };
      break;
    }
    case EventTypeValue.AGENT_OUTPUT: {
      const agentId = agentOf(ev);
      ensureAgent(s, agentId);
      const text = String(ev.text ?? "");
      const source = String(ev.source ?? "model");
      const list = [...(s.blocks[agentId] ?? [])];
      const last = list[list.length - 1];
      if (source === "user" || source === "parent") {
        // The daemon echoes every prompt as output with source ``user`` --
        // on send, and again when a conversation is replayed to a client
        // that attached; ``parent`` is the same thing for a subagent, whose
        // prompt came from its parent.  It is the user's turn, not the agent's: it
        // confirms the bubble the composer already drew when the texts
        // match, and otherwise becomes a user bubble of its own.  Rendered
        // as agent text it showed each prompt twice, once under "USER".
        if (ev.mode === "append" && last && last.kind === "user" && last.echoed) {
          list[list.length - 1] = { ...last, text: last.text + text };
        } else {
          let i = list.length - 1;
          while (i >= 0 && list[i]!.kind !== "user") i -= 1;
          const pending = i >= 0 ? (list[i] as UserBlock) : undefined;
          if (pending && !pending.echoed && pending.text === text) list[i] = { ...pending, echoed: true };
          else list.push({ id: nextId(), kind: "user", agentId, text, echoed: true });
        }
        setBlocks(s, agentId, list);
        break;
      }
      if (ev.mode === "append" && last && last.kind === "text" && last.source === source) {
        list[list.length - 1] = { ...last, text: last.text + text };
      } else {
        list.push({ id: nextId(), kind: "text", agentId, source, text });
      }
      setBlocks(s, agentId, list);
      break;
    }
    case EventTypeValue.AGENT_STATUS_CHANGED: {
      const id = agentOf(ev);
      ensureAgent(s, id);
      const a = s.agents[id]!;
      // The daemon's word, verbatim: active | idle | done | error | cancelled.
      // The client invents no status of its own -- ``store/phase.ts`` derives
      // what to show from this plus the pending prompts and the open tools.
      const status = String(ev.status ?? a.status);
      s.agents = { ...s.agents, [id]: { ...a, status, error: (ev.error as string | null | undefined) ?? null } };
      // The daemon has spoken: its word replaces the composer's optimism,
      // whichever way the status went.
      s.busySince = status === BUSY_STATUS ? startedAt(s.busySince, id) : without(s.busySince, id);
      break;
    }
    case EventTypeValue.AGENT_COMPLETED:
    case EventTypeValue.AGENT_ERROR: {
      const id = agentOf(ev);
      ensureAgent(s, id);
      const a = s.agents[id]!;
      const isErr = ev.type === EventTypeValue.AGENT_ERROR;
      // ``done``, not a locally-invented ``finished``: the daemon emits a
      // ``done`` status beside this event and the two must not disagree.
      s.agents = { ...s.agents, [id]: { ...a, status: isErr ? "error" : "done", error: isErr ? String(ev.error ?? "") : null } };
      s.busySince = without(s.busySince, id);
      if (isErr) {
        const list = [...(s.blocks[id] ?? [])];
        list.push({ id: nextId(), kind: "system", agentId: id, text: String(ev.error ?? "agent error"), style: "error" });
        setBlocks(s, id, list);
      }
      break;
    }
    case EventTypeValue.TOOL_CALL_START: {
      const agentId = agentOf(ev);
      ensureAgent(s, agentId);
      const callId = String(ev.call_id ?? nextId());
      const block: ToolBlock = {
        id: nextId(),
        kind: "tool",
        agentId,
        callId,
        toolName: String(ev.tool_name ?? "tool"),
        args: (ev.tool_args as Record<string, unknown> | undefined) ?? {},
        status: "running",
        startedAt: Date.now(),
        output: "",
        media: [],
        expanded: s.ui.showTools,
        toolClass: (ev.tool_class as ToolClass | null | undefined) ?? null,
      };
      setBlocks(s, agentId, [...(s.blocks[agentId] ?? []), block]);
      s.toolOwner = { ...s.toolOwner, [callId]: agentId };
      break;
    }
    case EventTypeValue.TOOL_CALL_END: {
      const agentId = agentOf(ev);
      const ok = ev.success !== false && ev.is_error_result !== true;
      const found = updateTool(s, ev.call_id as string | undefined, agentId, (t) => ({
        ...t,
        status: ok ? "success" : "error",
        durationSeconds: (ev.duration_seconds as number | null | undefined) ?? null,
        errorMessage: (ev.error_message as string | null | undefined) ?? null,
        resultStatus: (ev.result_status as string | null | undefined) ?? null,
        isErrorResult: ev.is_error_result === true,
        backgrounded: ev.backgrounded === true,
        continuationId: (ev.continuation_id as string | null | undefined) ?? null,
        showOutput: ev.show_output === true,
        showPopup: ev.show_popup === true,
        expanded: t.expanded || ev.show_output === true || !ok,
        diff: (ev.diff as string | null | undefined) ?? null,
        diffTruncated: (ev.diff_truncated as boolean | null | undefined) ?? null,
        path: (ev.path as string | null | undefined) ?? null,
      }));
      if (!found) {
        // End without a start (reconnect mid-call): synthesise a finished block.
        ensureAgent(s, agentId);
        const callId = String(ev.call_id ?? nextId());
        setBlocks(s, agentId, [
          ...(s.blocks[agentId] ?? []),
          {
            id: nextId(), kind: "tool", agentId, callId, toolName: String(ev.tool_name ?? "tool"), args: {},
            status: ok ? "success" : "error", startedAt: Date.now(), output: "", media: [], expanded: !ok,
            errorMessage: (ev.error_message as string | null | undefined) ?? null,
          },
        ]);
      }
      if (s.ui.popupCallId && s.ui.popupCallId === ev.call_id && !ev.continuation_id) {
        s.ui = { ...s.ui, popupCallId: null };
      }
      break;
    }
    case EventTypeValue.TOOL_OUTPUT: {
      const agentId = agentOf(ev);
      const callId = ev.call_id as string | undefined;
      const mime = ev.mime_type as string | null | undefined;
      const data = ev.data_b64 as string | null | undefined;
      const chunk = String(ev.chunk ?? "");
      const mediaItem = mime && data
        ? { mimeType: mime, dataB64: data, streamId: ev.stream_id as string | undefined, sequence: ev.sequence as number | null | undefined, final: ev.final === true }
        : null;

      if (callId === "model-output") {
        // Model-emitted media (speech) rides the tool-output channel under a
        // reserved id and has no ToolCallEndEvent: its ``final`` chunk closes
        // the block, and the next utterance opens a new one.  ``chunk`` on a
        // speech frame is the utterance's transcript (#869), carried on the
        // final chunk, so it is kept whether the block is new or continuing.
        ensureAgent(s, agentId);
        const list = [...(s.blocks[agentId] ?? [])];
        const last = list[list.length - 1];
        const status: ToolStatus = ev.final === true ? "success" : "running";
        if (last && last.kind === "tool" && last.callId === "model-output" && last.status === "running") {
          list[list.length - 1] = { ...last, output: chunk ? last.output + chunk : last.output, media: mediaItem ? [...last.media, mediaItem] : last.media, status };
        } else if (mediaItem || chunk) {
          list.push({ id: nextId(), kind: "tool", agentId, callId: "model-output", toolName: "model output", args: {}, status, startedAt: Date.now(), output: chunk, media: mediaItem ? [mediaItem] : [], expanded: true });
        }
        setBlocks(s, agentId, list);
        break;
      }

      const applied = updateTool(s, callId, agentId, (t) => ({
        ...t,
        output: chunk ? t.output + chunk : t.output,
        media: mediaItem ? [...t.media, mediaItem] : t.media,
      }));
      if (applied && chunk && s.ui.popupCallId == null) {
        const hit = findTool(s, callId, agentId);
        const t = hit ? (s.blocks[hit.agentId]?.[hit.index] as ToolBlock | undefined) : undefined;
        if (t && t.status === "running") s.ui = { ...s.ui, popupCallId: t.callId };
      }
      break;
    }
    case EventTypeValue.PERMISSION_REQUESTED:
      upsertPermission(s, ev, false);
      break;
    case EventTypeValue.PERMISSION_INPUT_MODE:
      upsertPermission(s, ev, true);
      break;
    case EventTypeValue.PERMISSION_RESOLVED: {
      // Dropping the pending record is the whole of it: the agent's status
      // is still the ``active`` the daemon set before it asked, so the phase
      // goes straight back to whatever the turn is doing.
      const rid = String(ev.request_id ?? "");
      s.permissions = s.permissions.filter((p) => p.requestId !== rid);
      break;
    }
    case EventTypeValue.PERMISSION_STATUS:
      // PermissionStatusEvent carries effective_default ("allow" | "deny" |
      // "ask"), suspension_scope ("turn" | "idle" | "session" | null) and
      // auto_allow_housekeeping (jaato/#1304 phase 3) -- null there means
      // "not reported" (an older daemon), never "off"; a measured false
      // is what "off" looks like.
      s.permissionStatus = {
        effectiveDefault: String(ev.effective_default ?? "ask"),
        suspensionScope: (ev.suspension_scope as string | null | undefined) ?? null,
        autoAllowHousekeeping: (ev.auto_allow_housekeeping as boolean | null | undefined) ?? null,
      };
      break;
    case EventTypeValue.SESSION_LIST: {
      const list = normalizeSessionList(ev.sessions);
      s.sessions = list;
      if (s.sessionListSilent > 0) { s.sessionListSilent -= 1; break; }
      const id = s.selectedAgentId;
      setBlocks(s, id, [...(s.blocks[id] ?? []), { id: nextId(), kind: "system", agentId: id, text: formatSessionList(list, s.notes), style: "help" }]);
      break;
    }
    case EventTypeValue.HISTORY: {
      const id = String(ev.agent_id ?? s.selectedAgentId);
      ensureAgent(s, id);
      if (s.historyMode === "replay") {
        s.historyMode = "listing";
        setBlocks(s, id, [...(s.blocks[id] ?? []), ...historyBlocks(ev.history, id, nextId, s.ui.showTools)]);
      } else {
        setBlocks(s, id, [...(s.blocks[id] ?? []), { id: nextId(), kind: "system", agentId: id, text: formatHistoryListing(ev.history, ev.turn_accounting), style: "help" }]);
      }
      break;
    }
    case EventTypeValue.CLARIFICATION_BATCH: {
      const requestId = String(ev.request_id ?? "");
      if (!requestId) break;
      const existing = s.clarifications.find((c) => c.requestId === requestId);
      const rec: PendingClarification = {
        requestId,
        agentId: agentOf(ev),
        toolName: String(ev.tool_name ?? ""),
        context: (ev.context as string | null | undefined) ?? null,
        // The batch wire spells a question ``text``/``choices``/``required`` (question_payload); normalize.
        questions: (Array.isArray(ev.questions) ? ev.questions : []).map(normalizeClarificationQuestion),
        index: existing?.index ?? 0,
        answers: existing?.answers ?? [],
        batchOnly: ev.batch_only === true,
        inputMode: existing?.inputMode ?? ev.batch_only === true,
      };
      s.clarifications = existing ? s.clarifications.map((c) => (c.requestId === requestId ? rec : c)) : [...s.clarifications, rec];
      break;
    }
    case EventTypeValue.CLARIFICATION_QUESTION: {
      // Per-question path: make sure a record exists and carry the question text.
      const requestId = String(ev.request_id ?? "");
      if (!requestId) break;
      const idx = Number(ev.question_index ?? 0);
      const q = normalizeClarificationQuestion({ question_text: ev.question_text, question_type: ev.question_type, options: ev.options ?? [] });
      const existing = s.clarifications.find((c) => c.requestId === requestId);
      if (existing) {
        const questions = [...existing.questions];
        questions[idx] = { ...(questions[idx] ?? {}), ...q };
        s.clarifications = s.clarifications.map((c) => (c.requestId === requestId ? { ...c, questions, index: idx } : c));
      } else {
        const questions: PendingClarification["questions"] = [];
        questions[idx] = q;
        s.clarifications = [...s.clarifications, { requestId, agentId: agentOf(ev), toolName: String(ev.tool_name ?? ""), questions, index: idx, answers: [], batchOnly: false, inputMode: false }];
      }
      break;
    }
    case EventTypeValue.CLARIFICATION_INPUT_MODE: {
      const requestId = String(ev.request_id ?? "");
      const idx = Number(ev.question_index ?? 0);
      const existing = s.clarifications.find((c) => c.requestId === requestId);
      if (existing) {
        s.clarifications = s.clarifications.map((c) => (c.requestId === requestId ? { ...c, index: idx, inputMode: true } : c));
      } else {
        s.clarifications = [...s.clarifications, { requestId, agentId: agentOf(ev), toolName: String(ev.tool_name ?? ""), questions: [], index: idx, answers: [], batchOnly: false, inputMode: true }];
      }
      break;
    }
    case EventTypeValue.CLARIFICATION_RESOLVED: {
      const rid = String(ev.request_id ?? "");
      s.clarifications = s.clarifications.filter((c) => c.requestId !== rid);
      break;
    }
    case EventTypeValue.REFERENCE_SELECTION_REQUESTED: {
      const requestId = String(ev.request_id ?? "");
      if (!requestId) break;
      s.referenceSelections = [
        ...s.referenceSelections.filter((r) => r.requestId !== requestId),
        { requestId, agentId: agentOf(ev), prompt: String(ev.prompt ?? ev.message ?? "Select a reference"), options: (ev.options as string[] | undefined) ?? [] },
      ];
      break;
    }
    case EventTypeValue.REFERENCE_SELECTION_RESOLVED:
      s.referenceSelections = s.referenceSelections.filter((r) => r.requestId !== String(ev.request_id ?? ""));
      break;
    case EventTypeValue.POST_AUTH_SETUP: {
      // The daemon offers a session after a daemon-level auth command
      // succeeded (``<provider>-auth login`` with no session open).  Kept
      // out of the output stream: it is a question with a typed answer,
      // like a permission prompt, not a line the agent said.
      const models = ((ev.available_models as { name?: string; description?: string }[] | undefined) ?? [])
        .filter((m) => m && m.name)
        .map((m) => ({ name: String(m.name), description: m.description ? String(m.description) : undefined }));
      s.postAuth = {
        requestId: String(ev.request_id ?? ""),
        providerName: String(ev.provider_name ?? ""),
        providerDisplayName: String(ev.provider_display_name ?? ev.provider_name ?? ""),
        models,
        hasActiveSession: ev.has_active_session === true,
        currentProvider: (ev.current_provider as string | undefined) || undefined,
        currentModel: (ev.current_model as string | undefined) || undefined,
        workspacePath: (ev.workspace_path as string | undefined) || undefined,
      };
      break;
    }
    case EventTypeValue.PLAN_UPDATED: {
      const id = agentOf(ev);
      s.plan = { ...s.plan, [id]: { name: String(ev.plan_name ?? "Plan"), steps: (ev.steps as PlanState["steps"] | undefined) ?? [] } };
      break;
    }
    case EventTypeValue.PLAN_STEP_UPDATED: {
      const id = agentOf(ev);
      const cur = s.plan[id] ?? { name: "Plan", steps: [] };
      const stepId = ev.step_id as string | undefined;
      const patch = { step_id: stepId, sequence: ev.sequence as number | undefined, content: ev.content as string | undefined, status: ev.status as string | undefined, result: ev.result as string | null | undefined, error: ev.error as string | null | undefined, blocked_by: ev.blocked_by as string[] | undefined, depends_on: ev.depends_on as string[] | undefined };
      const i = cur.steps.findIndex((st) => st.step_id === stepId);
      const steps = [...cur.steps];
      if (i >= 0) steps[i] = { ...steps[i], ...patch };
      else steps.push(patch);
      s.plan = { ...s.plan, [id]: { ...cur, steps } };
      break;
    }
    case EventTypeValue.PLAN_CLEARED: {
      const id = agentOf(ev);
      const { [id]: _dropped, ...rest } = s.plan;
      void _dropped;
      s.plan = rest;
      break;
    }
    case EventTypeValue.CONTEXT_UPDATED: {
      const id = agentOf(ev);
      const prev = s.context[id];
      s.context = {
        ...s.context,
        [id]: {
          ...(prev ?? { usage: {} }),
          usage: (ev.usage as ContextState["usage"] | undefined) ?? prev?.usage ?? {},
          contextLimit: (ev.context_limit as number | null | undefined) ?? prev?.contextLimit ?? null,
          percentUsed: (ev.percent_used as number | null | undefined) ?? prev?.percentUsed ?? null,
          tokensRemaining: (ev.tokens_remaining as number | null | undefined) ?? prev?.tokensRemaining ?? null,
          turns: (ev.turns as number | null | undefined) ?? prev?.turns ?? null,
        },
      };
      break;
    }
    /**
     * ``InstructionBudgetEvent`` -- what the context window is spent ON.
     *
     * Read by the rail's Budget section.  The daemon's snapshot is taken
     * verbatim rather than reshaped: it is one dict, several clients read
     * it, and a client-side flattening is how two readers start disagreeing
     * about what a source layer costs.  A snapshot that carries no entries
     * is ignored, because replacing a populated breakdown with an empty one
     * would blank the panel mid-turn.
     */
    case EventTypeValue.INSTRUCTION_BUDGET_UPDATED: {
      const snap = ev.budget_snapshot as Record<string, unknown> | undefined;
      const entries = snap?.entries as BudgetState["entries"] | undefined;
      if (!snap || !entries || Object.keys(entries).length === 0) break;
      s.budget = {
        ...s.budget,
        [agentOf(ev)]: {
          contextLimit: (snap.context_limit as number | null | undefined) ?? null,
          totalTokens: (snap.total_tokens as number | null | undefined) ?? null,
          utilizationPercent: (snap.utilization_percent as number | null | undefined) ?? null,
          lockedTokens: (snap.locked_tokens as number | null | undefined) ?? null,
          gcEligibleTokens: (snap.gc_eligible_tokens as number | null | undefined) ?? null,
          entries,
        },
      };
      break;
    }
    case EventTypeValue.TURN_COMPLETED: {
      const id = agentOf(ev);
      const prev = s.context[id] ?? { usage: {} };
      s.context = {
        ...s.context,
        [id]: {
          ...prev,
          lastTurn: {
            turnNumber: ev.turn_number as number | null | undefined,
            durationSeconds: ev.duration_seconds as number | null | undefined,
            // A LIST of per-call records on the wire, never a count.
            toolCalls: summarizeToolCalls(ev.function_calls),
            finishReason: ev.finish_reason as string | null | undefined,
            usage: ev.usage as ContextState["usage"] | undefined,
          },
        },
      };
      s.busySince = without(s.busySince, id);
      break;
    }
    case EventTypeValue.SYSTEM_MESSAGE: {
      const id = agentOf(ev);
      ensureAgent(s, id);
      setBlocks(s, id, [...(s.blocks[id] ?? []), { id: nextId(), kind: "system", agentId: id, text: String(ev.message ?? ""), style: String(ev.style ?? "info") }]);
      break;
    }
    case EventTypeValue.HELP_TEXT: {
      const id = s.selectedAgentId;
      const lines = (ev.lines as unknown[] | undefined) ?? [];
      const text = lines.map((l) => (Array.isArray(l) ? String(l[0] ?? "") : String(l))).join("\n") || String(ev.text ?? "");
      setBlocks(s, id, [...(s.blocks[id] ?? []), { id: nextId(), kind: "system", agentId: id, text, style: "help" }]);
      break;
    }
    case EventTypeValue.ERROR: {
      // The answer to the workspace-mode probe (``workspace.list`` on a
      // single-workspace daemon) is the daemon's only way of saying "no
      // workspace mode"; it is a fact about the deployment, not a fault.
      if (String(ev.error ?? "").toLowerCase().includes("workspace mode not enabled")) {
        s.workspace = { ...s.workspace, mode: "disabled" };
        break;
      }
      const id = agentOf(ev);
      ensureAgent(s, id);
      setBlocks(s, id, [...(s.blocks[id] ?? []), { id: nextId(), kind: "system", agentId: id, text: `${ev.error_type ? `[${String(ev.error_type)}] ` : ""}${String(ev.error ?? "error")}`, style: "error" }]);
      // ``recoverable: false`` means this session never came up
      // (``RunnerBootstrapFailed`` among its causes) -- a fact about the
      // SESSION distinct from ``connection.phase``, which stays whatever
      // the transport reports.  ``protocol/sessionFault.ts`` is the one
      // predicate; the status bar reads this field, not the transcript.
      const fault = faultFromError(ev.error_type, ev.error, ev.recoverable);
      if (fault) s.sessionFault = fault;
      break;
    }
    case EventTypeValue.RETRY: {
      const id = agentOf(ev);
      ensureAgent(s, id);
      setBlocks(s, id, [...(s.blocks[id] ?? []), { id: nextId(), kind: "system", agentId: id, text: `Retrying (${String(ev.attempt ?? "?")}/${String(ev.max_attempts ?? "?")}): ${String(ev.reason ?? ev.error ?? "")}`, style: "warning" }]);
      break;
    }
    case EventTypeValue.INIT_PROGRESS:
      s.initProgress = { step: ev.step as string | undefined, status: ev.status as string | undefined, message: ev.message as string | undefined, stepNumber: ev.step_number as number | null | undefined, totalSteps: ev.total_steps as number | null | undefined };
      if (ev.status === "complete" || ev.status === "done" || ev.status === "ready") s.initProgress = null;
      break;
    case EventTypeValue.SESSION_INFO: {
      const sid = ev.session_id as string | undefined;
      // A session opening adopts the picker's uploads: files attached
      // before any session existed carry scope ``""``, and the one session
      // they were staged for is the one that now opens (#1250).  Re-stamp
      // only unscoped uploads, so another session's files are never
      // reassigned.  Only on the null -> session transition.
      if (sid && !s.sessionId) {
        const adopted = `session:${sid}`;
        if (s.uploads.some((u) => u.scope === "")) {
          s.uploads = s.uploads.map((u) => (u.scope === "" ? { ...u, scope: adopted } : u));
        }
      }
      // A session that comes up clears whatever fault a PRIOR attempt left
      // (this connection's, or a stale one this browser had rendered
      // before a reconnect) -- the bar's red "no session" is a claim about
      // the current attempt, never a sticky one.
      if (sid) { s.sessionId = sid; s.sessionFault = null; }
      if (Array.isArray(ev.sessions)) s.sessions = normalizeSessionList(ev.sessions);
      s.session = {
        ...s.session,
        name: (ev.session_name as string | null | undefined) ?? s.session.name ?? null,
        provider: (ev.model_provider as string | null | undefined) ?? s.session.provider ?? null,
        model: (ev.model_name as string | null | undefined) ?? s.session.model ?? null,
        profile: (ev.profile_name as string | null | undefined) ?? s.session.profile ?? null,
        models: (ev.models as string[] | undefined) ?? s.session.models,
      };
      {
        const names = toolIdMappings(ev.tool_id_mappings);
        if (names && Object.keys(names).length) s.toolIdNames = names;
      }
      break;
    }
    case EventTypeValue.GC_CONFIG: {
      const id = (ev.agent_id as string) || "main";
      const num = (v: unknown) => (typeof v === "number" ? v : null);
      s.gc = { ...s.gc, [id]: { ...s.gc[id], config: {
        strategy: typeof ev.strategy === "string" && ev.strategy ? ev.strategy : null,
        threshold: num(ev.threshold), targetPercent: num(ev.target_percent), continuous: ev.continuous_mode === true,
      } } };
      break;
    }
    case EventTypeValue.GC: {
      const id = (ev.agent_id as string) || "main";
      const cur = s.gc[id] ?? {};
      const num = (v: unknown) => (typeof v === "number" ? v : null);
      if (ev.phase === "started") s.gc = { ...s.gc, [id]: { ...cur, running: true } };
      else if (ev.phase === "completed") {
        const at = Date.parse(String(ev.timestamp ?? ""));
        s.gc = { ...s.gc, [id]: { ...cur, running: false, lastPass: {
          at: Number.isFinite(at) ? at : Date.now(),
          success: ev.success !== false,
          tokensFreed: num(ev.tokens_freed), tokensBefore: num(ev.tokens_before), tokensAfter: num(ev.tokens_after),
          trigger: typeof ev.trigger_reason === "string" ? ev.trigger_reason : null,
          strategy: typeof ev.strategy === "string" ? ev.strategy : null,
          error: typeof ev.error === "string" ? ev.error : null,
        } } };
      }
      break;
    }
    case EventTypeValue.TOOL_ID_REGISTRY: {
      const names = toolIdMappings(ev.mappings);
      if (names) s.toolIdNames = names;
      break;
    }
    case EventTypeValue.SESSION_PROFILES:
      s.profiles = ((ev.profiles as ProfileInfo[] | undefined) ?? []).map((p) => ({ ...p, name: String(p.name ?? "") }));
      break;
    case EventTypeValue.SESSION_TERMINATED:
    case EventTypeValue.DISCONNECTED: {
      const id = s.selectedAgentId;
      setBlocks(s, id, [...(s.blocks[id] ?? []), { id: nextId(), kind: "system", agentId: id, text: ev.type === EventTypeValue.SESSION_TERMINATED ? `Session ended${ev.reason ? `: ${String(ev.reason)}` : ""}` : "Disconnected", style: "warning" }]);
      break;
    }
    case EventTypeValue.COMMAND_LIST:
      s.commands = mergeCommandSpecs(((ev.commands as CommandSpec[] | undefined) ?? []).map((c) => ({ name: String(c.name ?? ""), description: c.description })));
      break;
    case EventTypeValue.WORKSPACE_LIST:
      s.workspace = { ...s.workspace, mode: "enabled", root: ev.root as string | undefined, list: ((ev.workspaces as WorkspaceInfo[] | undefined) ?? []).map((w) => ({ ...w, name: String(w.name ?? ""), configured: w.configured === true })) };
      break;
    case EventTypeValue.WORKSPACE_CREATED: {
      // The daemon answers with the row as ``workspace`` and repeats its
      // name/path beside it.  A reply naming nothing is not a row: an older
      // daemon dropped the dict, and appending ``{name: ""}`` put an unnamed
      // entry in the table that a click then turned into a select of "".
      const raw = (ev.workspace as Partial<WorkspaceInfo> | undefined) ?? {};
      const name = String(raw.name ?? ev.name ?? "");
      if (!name) break;
      const w: WorkspaceInfo = { ...raw, name, path: raw.path ?? (ev.path as string | undefined) ?? null, configured: raw.configured === true };
      s.workspace = { ...s.workspace, list: [...s.workspace.list.filter((x) => x.name !== name), w] };
      break;
    }
    case EventTypeValue.WORKSPACE_DELETED: {
      const name = String(ev.name ?? "");
      if (ev.ok === false) {
        s.workspace = { ...s.workspace, notice: { text: String(ev.error || `Could not delete workspace ${name}`), error: true } };
        break;
      }
      s.workspace = {
        ...s.workspace,
        list: s.workspace.list.filter((w) => w.name !== name),
        selected: s.workspace.selected === name ? undefined : s.workspace.selected,
        config: s.workspace.config?.workspace === name ? undefined : s.workspace.config,
        notice: { text: `Workspace ${name} deleted` },
      };
      break;
    }
    case EventTypeValue.CONFIG_UPDATED: {
      // ``config.updated`` says what was WRITTEN -- workspace, provider,
      // model, success -- and carries none of the status fields.  Read as a
      // ``config.status`` it emptied the provider list and reported the
      // provider it had just saved as missing.  So it is merged over the
      // status held, and the table row follows.
      const name = String(ev.workspace ?? s.workspace.selected ?? "");
      if (ev.success === false) {
        s.workspace = { ...s.workspace, notice: { text: String(ev.error || `Could not save the configuration of ${name}`), error: true } };
        break;
      }
      const provider = (ev.provider as string | null | undefined) || null;
      const model = (ev.model as string | null | undefined) || null;
      const prev = s.workspace.config;
      const cfg: ConfigStatus = {
        workspace: name,
        configured: provider !== null,
        provider,
        model,
        availableProviders: prev?.availableProviders ?? [],
        missingFields: [...(provider ? [] : ["provider"]), ...(model ? [] : ["model"])],
      };
      s.workspace = {
        ...s.workspace,
        config: cfg,
        selected: name || s.workspace.selected,
        list: s.workspace.list.map((w) => (w.name === name ? { ...w, configured: cfg.configured, provider, model } : w)),
      };
      break;
    }
    case EventTypeValue.CONFIG_STATUS: {
      const cfg: ConfigStatus = {
        workspace: String(ev.workspace ?? s.workspace.selected ?? ""),
        configured: ev.configured === true,
        provider: (ev.provider as string | null | undefined) ?? null,
        model: (ev.model as string | null | undefined) ?? null,
        availableProviders: (ev.available_providers as string[] | undefined) ?? [],
        missingFields: (ev.missing_fields as string[] | undefined) ?? [],
      };
      s.workspace = { ...s.workspace, config: cfg, selected: cfg.workspace || s.workspace.selected };
      break;
    }
    case EventTypeValue.WORKSPACE_FILES_CHANGED:
    case EventTypeValue.WORKSPACE_FILES_SNAPSHOT: {
      // #1189: both go through workspaceView.ts, which keeps the change
      // numbers the panel's reset point is measured against.
      const cur = { files: s.workspaceFiles, seqs: s.workspaceSeqs, epoch: s.workspaceEpoch, seq: s.workspaceSeq, reset: s.workspaceReset };
      const out = ev.type === EventTypeValue.WORKSPACE_FILES_SNAPSHOT ? applySnapshot(cur, ev) : applyChanged(cur, ev);
      s.workspaceFiles = out.files;
      s.workspaceSeqs = out.seqs;
      s.workspaceEpoch = out.epoch;
      s.workspaceSeq = out.seq;
      s.workspaceReset = out.reset;
      if (out.voided) s.workspaceNotice = { text: out.voided };
      break;
    }
    case EventTypeValue.WORKSPACE_IGNORE_RESULT: {
      const path = String(ev.path ?? "");
      if (ev.ok === false) {
        s.workspaceNotice = { text: String(ev.error || `Could not update .gitignore for ${path}`), error: true };
        break;
      }
      const ignored = ev.ignored === true;
      s.workspaceIgnored = { ...s.workspaceIgnored, [path]: ignored };
      s.workspaceNotice = { text: `${path} ${ignored ? "added to" : "removed from"} .gitignore` };
      break;
    }
    case EventTypeValue.SCAFFOLD_INTEGRATION_RESULT: {
      // The daemon ran ``jaato-scaffold integration --refresh`` into this
      // workspace (#1263).  A SKIPPED refresh — an ``edited`` / ``diverged``
      // / ``unstamped`` copy the daemon declined to overwrite — is reported
      // here rather than silently: the person edited the skill and should
      // know it was left as-is.  A clean apply of a current copy says
      // nothing (no notice), so a re-check on every session start is quiet.
      const name = String(ev.integration ?? "jaato-sdk");
      if (ev.ok === false) {
        s.workspaceNotice = { text: String(ev.error || `Could not install the ${name} skill`), error: true };
        break;
      }
      const state = String(ev.state_after ?? ev.state_before ?? "");
      // Report only a skip that LEFT LOCAL CONTENT alone (the person edited
      // the skill).  ``--refresh`` also declines an already-``current`` copy
      // with a skipped_reason, but that is the steady state on every session
      // start and is not worth a notice — reporting it would be constant noise.
      if (ev.skipped_reason && (state === "edited" || state === "diverged" || state === "unstamped")) {
        s.workspaceNotice = { text: `${name} skill left as-is (${state}): ${String(ev.skipped_reason)}` };
      } else if (ev.changed) {
        const ver = ev.server_version ? ` (jaato-server ${String(ev.server_version)})` : "";
        s.workspaceNotice = { text: `${name} skill installed${ver}` };
      }
      break;
    }
    default:
      break;
  }
  return s;
}

export const useJaato = create<JaatoState>()((set, get) => ({
  connection: { phase: "disconnected" },
  url: "",
  screen: "connect",
  credentialsUrl: null,
  notesUrl: null,
  githubUrl: null,
  githubLoginUrl: null,
  notes: {},
  noteStatus: {},
  backend: null,
  workspace: { mode: "unknown", list: [] },
  profiles: [],
  sessions: [],
  sessionListSilent: 0,
  historyMode: "listing",
  commands: mergeCommandSpecs([]),
  uploads: [],
  stallThresholdMs: DEFAULT_STALL_THRESHOLD_MS,
  ...emptySessionState(),
  ui: { showTools: false, activePanel: null, pinnedPlan: false, theme: "light", popupCallId: null, railWidth: loadRailWidth(), railSplits: loadRailSplits() },

  dispatch: (events) =>
    set((state) => {
      const draft = { ...state };
      for (const ev of events) reduce(draft, ev);
      return draft;
    }),
  setConnection: (c) => set((st) => ({ connection: { ...st.connection, ...c } })),
  setUrl: (url) => set({ url }),
  setScreen: (screen) => set({ screen }),
  setCredentialsUrl: (credentialsUrl) => set({ credentialsUrl }),
  setNotesUrl: (notesUrl) => set({ notesUrl }),
  setGithubUrls: ({ githubUrl, githubLoginUrl }) => set({ githubUrl, githubLoginUrl }),
  setNotes: (list) => set({ notes: Object.fromEntries(list.map((n) => [n.sessionId, n])) }),
  setNote: (sessionId, note) =>
    set((st) => {
      const notes = { ...st.notes };
      if (note) notes[sessionId] = note; else delete notes[sessionId];
      return { notes };
    }),
  setNoteStatus: (sessionId, status) => set((st) => ({ noteStatus: { ...st.noteStatus, [sessionId]: status } })),
  setBackend: (backend) => set({ backend }),
  setWorkspaceMode: (mode) => set((st) => ({ workspace: { ...st.workspace, mode } })),
  selectWorkspace: (name) => set((st) => ({ workspace: { ...st.workspace, selected: name } })),
  selectAgent: (id) => set({ selectedAgentId: id }),
  markSending: (agentId) => set((st) => ({ busySince: startedAt(st.busySince, agentId) })),
  clearSending: (agentId) =>
    // Only what the send itself claimed: a turn the daemon has since
    // confirmed is ``active`` is not this send's to cancel.
    set((st) => (st.agents[agentId]?.status === BUSY_STATUS ? {} : { busySince: without(st.busySince, agentId) })),
  addUserBlock: (agentId, text) =>
    set((st) => ({ blocks: { ...st.blocks, [agentId]: [...(st.blocks[agentId] ?? []), { id: nextId(), kind: "user", agentId, text }] } })),
  addSystemBlock: (agentId, text, style = "info") =>
    set((st) => ({ blocks: { ...st.blocks, [agentId]: [...(st.blocks[agentId] ?? []), { id: nextId(), kind: "system", agentId, text, style }] } })),
  clearOutput: (agentId) => set((st) => ({ blocks: { ...st.blocks, [agentId]: [] } })),
  toggleTool: (agentId, blockId) =>
    set((st) => ({ blocks: { ...st.blocks, [agentId]: (st.blocks[agentId] ?? []).map((b) => (b.id === blockId && b.kind === "tool" ? { ...b, expanded: !b.expanded } : b)) } })),
  setAllToolsExpanded: (agentId, expanded) =>
    set((st) => ({ blocks: { ...st.blocks, [agentId]: (st.blocks[agentId] ?? []).map((b) => (b.kind === "tool" ? { ...b, expanded } : b)) } })),
  resolvePermission: (requestId) => set((st) => ({ permissions: st.permissions.filter((p) => p.requestId !== requestId) })),
  openExitChoice: (running, options) => set({ exitChoice: { running, options, focus: 0 } }),
  closeExitChoice: () => set({ exitChoice: null }),
  focusExitChoice: (focus) => set((st) => (st.exitChoice ? { exitChoice: { ...st.exitChoice, focus } } : {})),
  focusPermission: (requestId, focus) => set((st) => ({ permissions: st.permissions.map((p) => (p.requestId === requestId ? { ...p, focus } : p)) })),
  answerClarification: (requestId, answer) => {
    const cur = get().clarifications.find((c) => c.requestId === requestId);
    if (!cur) return undefined;
    const answers = [...cur.answers];
    answers[cur.index] = answer;
    const next: PendingClarification = { ...cur, answers, index: cur.index + 1, inputMode: cur.batchOnly };
    set((st) => ({ clarifications: st.clarifications.map((c) => (c.requestId === requestId ? next : c)) }));
    return next;
  },
  dismissClarification: (requestId) => set((st) => ({ clarifications: st.clarifications.filter((c) => c.requestId !== requestId) })),
  dismissReferenceSelection: (requestId) => set((st) => ({ referenceSelections: st.referenceSelections.filter((r) => r.requestId !== requestId) })),
  dismissPostAuth: () => set({ postAuth: null }),
  toggleUi: (key) => set((st) => ({ ui: { ...st.ui, [key]: !st.ui[key] } })),
  setPaletteOpen: (open) => set({ paletteOpen: open }),
  setActivePanel: (id) => set((st) => ({ ui: { ...st.ui, activePanel: st.ui.activePanel === id ? null : id } })),
  togglePinPlan: () => set((st) => ({ ui: { ...st.ui, pinnedPlan: !st.ui.pinnedPlan } })),
  setStallThreshold: (ms) => set({ stallThresholdMs: clampStallThreshold(ms) }),
  setToolsExpanded: (expanded) => set((st) => ({
    ui: { ...st.ui, showTools: expanded },
    blocks: Object.fromEntries(Object.entries(st.blocks).map(([agentId, list]) => [agentId, list.map((b) => (b.kind === "tool" ? { ...b, expanded } : b))])),
  })),
  setSessionListSilent: (delta) => set((st) => ({ sessionListSilent: Math.max(0, st.sessionListSilent + delta) })),
  setHistoryMode: (mode) => set({ historyMode: mode }),
  toggleBudgetSource: (source) => set((st) => ({
    budgetExpanded: st.budgetExpanded.includes(source)
      ? st.budgetExpanded.filter((k) => k !== source)
      : [...st.budgetExpanded, source],
  })),
  toggleWorkspaceHidden: (entryId) => set((st) => ({
    workspaceHidden: st.workspaceHidden.includes(entryId)
      ? st.workspaceHidden.filter((h) => h !== entryId)
      : [...st.workspaceHidden, entryId],
  })),
  toggleWorkspaceShowHidden: () => set((st) => ({ workspaceShowHidden: !st.workspaceShowHidden })),
  toggleWorkspaceCollapsed: (entryId) => set((st) => ({
    workspaceCollapsed: st.workspaceCollapsed.includes(entryId)
      ? st.workspaceCollapsed.filter((c) => c !== entryId)
      : [...st.workspaceCollapsed, entryId],
  })),
  resetWorkspaceView: () => set((st) => ({ workspaceReset: markReset({ epoch: st.workspaceEpoch, seq: st.workspaceSeq }), workspaceCollapsed: [], workspaceNotice: null })),
  showAllWorkspace: () => set(() => ({ workspaceReset: null, workspaceNotice: null })),
  setWorkspaceNotice: (n) => set(() => ({ workspaceNotice: n })),
  setWorkspaceListNotice: (n) => set((st) => ({ workspace: { ...st.workspace, notice: n } })),
  setTheme: (theme) => set((st) => ({ ui: { ...st.ui, theme } })),
  setPopup: (callId) => set((st) => ({ ui: { ...st.ui, popupCallId: callId } })),
  patchMemories: (patch) => set((st) => ({
    memories: { ...st.memories, ...(typeof patch === "function" ? patch(st.memories) : patch) },
  })),
  patchDiagnostics: (patch) => set((st) => ({
    diagnostics: { ...st.diagnostics, ...(typeof patch === "function" ? patch(st.diagnostics) : patch) },
  })),
  setRailWidth: (w) => set((st) => { const railWidth = clampRailWidth(w); saveRailWidth(railWidth); return { ui: { ...st.ui, railWidth } }; }),
  setRailSplits: (splits) => set((st) => { const railSplits = sanitizeSplits(splits); saveRailSplits(railSplits); return { ui: { ...st.ui, railSplits } }; }),
  addUploads: (items) => set((st) => ({ uploads: [...st.uploads, ...items] })),
  updateUpload: (id, patch) => set((st) => ({ uploads: st.uploads.map((u) => (u.id === id ? { ...u, ...patch } : u)) })),
  removeUpload: (id) => set((st) => ({ uploads: st.uploads.filter((u) => u.id !== id) })),
  takeUploads: () => {
    // Only the ACTIVE context's staged files are named in the message and
    // its settled chips (staged + failed) cleared; another session's uploads
    // (a different scope) are left untouched (#1250).  A file still
    // queued/staging in this context is kept for the next send, as before.
    const active = uploadScope(get());
    const staged = get().uploads.filter((u) => u.scope === active && u.status === "staged").map((u) => u.path);
    set((st) => ({ uploads: st.uploads.filter((u) => u.scope !== active || u.status === "queued" || u.status === "staging") }));
    return staged;
  },
  resetSessionState: () => set(() => ({ ...emptySessionState() })),
}));

/** Selector helpers. */
export const selectBlocks = (agentId: string) => (s: JaatoState) => s.blocks[agentId] ?? EMPTY_BLOCKS;
const EMPTY_BLOCKS: OutputBlock[] = [];

/**
 * Running tool blocks with output for one agent, in block order -- the one
 * definition of what the live-output popup's tab strip shows and what the
 * leader's ``O`` steps through (``app/leaderKeys.ts``).  ``ToolOutputPopup``
 * used to compute this itself with a local listener owning its OWN Ctrl+O
 * binding; there is one leader now, not a leader plus a component keeping a
 * direct one alive beside it.
 */
export function runningToolCallIds(s: Pick<JaatoState, "blocks">, agentId: string): string[] {
  const out: string[] = [];
  for (const b of s.blocks[agentId] ?? []) {
    if (b.kind === "tool" && b.status === "running" && b.output.length > 0) out.push(b.callId);
  }
  return out;
}

// Test/debug hook: the store instance itself, so a Playwright test can
// drive state directly (a low ``stallThresholdMs`` for a stall to show up
// in seconds rather than the real 30s floor, since there is no settings UI
// yet to set it from) without a raw WS frame the daemon would never send.
// Harmless in production -- it exposes the same actions any component
// already calls, on an object nothing sensitive reaches through.
if (typeof window !== "undefined") {
  (window as unknown as { __jaatoStore?: typeof useJaato }).__jaatoStore = useJaato;
}
