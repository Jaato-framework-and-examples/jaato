/**
 * Client-side state model.  Everything here is derived from the wire
 * events by ``reduce()`` in ``store.ts``; components never touch raw
 * events.  The shapes mirror what ``jaato-tui/output_buffer.py`` keeps
 * (``OutputLine`` / ``ToolBlock`` / ``ActiveToolCall``) so the two
 * clients present the same session the same way.
 */
import type { ToolCallSummary } from "@/protocol/turnStats";
import type { ToolClass } from "@/protocol/toolClass";

export interface MediaItem {
  mimeType: string;
  dataB64: string;
  streamId?: string;
  sequence?: number | null;
  final?: boolean;
}

/** Streamed model / plugin text.  ``text`` carries ``<j-*>`` markup + markdown. */
export interface TextBlock {
  id: string;
  kind: "text";
  agentId: string;
  /** ``"model"``, ``"system"`` or a plugin name — the ``AgentOutputEvent.source``. */
  source: string;
  text: string;
}

/** What the user typed, echoed locally when the message is sent. */
export interface UserBlock {
  id: string;
  kind: "user";
  agentId: string;
  text: string;
  /**
   * The daemon has echoed this prompt back (``agent.output`` with
   * ``source: "user"``).  The composer adds the bubble locally when the
   * message is sent; the daemon then echoes it to every attached client,
   * and the echo matches the pending local bubble instead of being drawn
   * a second time.  A block created FROM an echo (a replay after attach,
   * another client's prompt) starts echoed.
   */
  echoed?: boolean;
}

/** Client-side notices (connection, command results, help). */
export interface SystemBlock {
  id: string;
  kind: "system";
  agentId: string;
  text: string;
  /** ``info`` | ``warning`` | ``error`` | ``hint`` — maps to a colour role. */
  style: string;
}

export type ToolStatus = "running" | "success" | "error";

/**
 * One tool invocation, from ``ToolCallStartEvent`` to ``ToolCallEndEvent``.
 *
 * ``output`` accumulates ``ToolOutputEvent`` text chunks (the live
 * "tail -f" view); ``media`` collects binary chunks a person can
 * play/see.  ``expanded`` is per-block UI state (default from
 * ``show_output`` on completion).  ``continuationId`` groups calls
 * that belong to one interactive-shell session so the output popup
 * stays put across them.
 */
export interface ToolBlock {
  id: string;
  kind: "tool";
  agentId: string;
  callId: string;
  toolName: string;
  args: Record<string, unknown>;
  status: ToolStatus;
  startedAt: number;
  durationSeconds?: number | null;
  errorMessage?: string | null;
  resultStatus?: string | null;
  isErrorResult?: boolean;
  backgrounded?: boolean;
  continuationId?: string | null;
  showOutput?: boolean;
  showPopup?: boolean;
  output: string;
  media: MediaItem[];
  expanded: boolean;
  /**
   * The daemon's own classification (``ToolCallStartEvent.tool_class``,
   * jaato/#1304 phase 3), when this daemon sends one.  ``null`` means
   * "not reported" -- an older daemon, or a call the table itself has no
   * opinion about -- and every reader falls back to
   * ``protocol/toolClass.classifyTool`` on the display name, never
   * treating absence as ``"other"`` (that would collapse "the daemon
   * said nothing" into "the daemon said unclassified").
   */
  toolClass?: ToolClass | null;
  /**
   * A capped unified diff for a file-writer call (``ToolCallEndEvent.diff``,
   * jaato/#1304 phase 3) -- the SAME text ``generate_unified_diff`` /
   * ``generate_new_file_diff`` compute server-side, so ``DiffLines`` can
   * render it directly.  ``null`` when the daemon sent none (an older
   * daemon, a tool that is not a file writer, or a best-effort read that
   * failed server-side) -- never fabricated from the call's own
   * arguments; that fallback lives in ``protocol/toolPreview.ts`` and is
   * used only when this is absent.
   */
  diff?: string | null;
  /** Whether ``diff`` was capped -- render a truncation note, never a fabricated line count. */
  diffTruncated?: boolean | null;
  /** The path ``diff`` is about, echoed from the tool's own result. */
  path?: string | null;
}

export type OutputBlock = TextBlock | UserBlock | SystemBlock | ToolBlock;

export interface PermissionOption {
  key: string;
  label: string;
  action?: string;
  description?: string;
}

/**
 * A permission prompt awaiting the user.  Arrives as a
 * ``PermissionRequestedEvent`` (content) followed by a
 * ``PermissionInputModeEvent`` (take input); either may come first on
 * a reconnect, so both upsert the same record keyed by ``requestId``.
 */
export interface PendingPermission {
  requestId: string;
  agentId: string;
  toolName: string;
  callId?: string | null;
  toolArgs: Record<string, unknown>;
  options: PermissionOption[];
  promptLines: string[];
  formatHint?: string | null;
  warnings?: string | null;
  warningLevel?: string | null;
  editableMetadata?: Record<string, unknown> | null;
  /** Keyboard focus among ``options`` (Tab cycles, like the TUI). */
  focus: number;
  inputMode: boolean;
  /**
   * The daemon's classification of the tool being asked about
   * (``PermissionRequestedEvent.tool_class``, jaato/#1304 phase 3) --
   * what the card's risk tag reads.  ``null`` falls back to
   * ``classifyTool(toolName)`` the same way ``ToolBlock.toolClass`` does.
   */
  toolClass?: ToolClass | null;
}

/**
 * One clarification question, as the card reads it.  The daemon spells a
 * question two ways (``text``/``choices``/``required`` on the batch wire,
 * ``question_text``/``options`` on the per-question wire); the store runs
 * both through ``protocol/clarification.normalizeClarificationQuestion``
 * so this shape is the only one rendered.
 */
export interface ClarificationQuestion {
  question_text?: string;
  /** ``single_choice`` | ``multiple_choice`` | ``free_text`` (the daemon's ``QuestionType``). */
  question_type?: string;
  /** Choice texts, in wire order; answers are their 1-based positions. */
  options?: string[];
  /** A 1-based choice position (choice questions) or a literal (free text); Enter on empty submits it. */
  default?: string | number | null;
  /** ``required: false`` on the wire — an empty answer skips it. */
  optional?: boolean;
  /** Per-choice: this branch expects the user to attach a file (#989). Absent when no choice does. */
  expects_attachment?: boolean[];
  [k: string]: unknown;
}

/**
 * A clarification request.  Two wire shapes exist (see
 * ``jaato-tui/clarification_batch.py``): daemon-local sessions stream
 * question by question (``ClarificationQuestionEvent`` +
 * ``ClarificationInputModeEvent``) and answers travel one
 * ``ClarificationResponseRequest`` at a time; runner-tier sessions send
 * ONE ``ClarificationBatchEvent`` with ``batch_only=true`` and expect
 * ONE ``ClarificationBatchResponseEvent``.  ``batchOnly`` selects the
 * reply shape; the UI walks ``questions`` identically in both.
 */
export interface PendingClarification {
  requestId: string;
  agentId: string;
  toolName: string;
  context?: string | null;
  questions: ClarificationQuestion[];
  index: number;
  answers: string[];
  batchOnly: boolean;
  /** Per-question mode: the daemon has asked for input on ``index``. */
  inputMode: boolean;
}

export interface PendingReferenceSelection {
  requestId: string;
  agentId: string;
  prompt: string;
  options: string[];
}

/**
 * The daemon's ``auth.setup`` offer: a daemon-level auth command
 * (``<provider>-auth login``) succeeded with no session open, so the
 * daemon asks whether to open one with that provider, which model, and
 * whether to persist ``JAATO_PROVIDER`` / ``MODEL_NAME`` to the workspace
 * ``.env``.  One pending offer at a time; answered with
 * ``respondToPostAuthSetup``.  The TUI walks the same three questions.
 */
export interface PendingPostAuthSetup {
  requestId: string;
  providerName: string;
  providerDisplayName: string;
  models: { name: string; description?: string }[];
  hasActiveSession: boolean;
  currentProvider?: string;
  currentModel?: string;
  workspacePath?: string;
}

export interface Agent {
  id: string;
  name: string;
  type: string;
  parentId?: string | null;
  profile?: string | null;
  /**
   * The daemon's own word, verbatim: ``active`` | ``idle`` | ``done`` |
   * ``error`` | ``cancelled`` (``AgentStatusChangedEvent``).  The client
   * invents none of its own -- what to SHOW is derived in
   * ``store/phase.ts``, which reads this beside the pending prompts and
   * the open tool calls.
   */
  status: string;
  error?: string | null;
}

export interface PlanStep {
  step_id?: string;
  sequence?: number;
  content?: string;
  status?: string;
  result?: string | null;
  error?: string | null;
  blocked_by?: string[];
  depends_on?: string[];
  [k: string]: unknown;
}

export interface PlanState {
  name: string;
  steps: PlanStep[];
}

export interface ContextState {
  usage: Record<string, number | null | undefined>;
  contextLimit?: number | null;
  percentUsed?: number | null;
  tokensRemaining?: number | null;
  turns?: number | null;
  lastTurn?: {
    turnNumber?: number | null;
    durationSeconds?: number | null;
    /**
     * The turn's tool calls, reduced from ``TurnCompletedEvent.function_calls``
     * -- a LIST of ``{name, start_time, end_time, duration_seconds}`` records,
     * not a count (``protocol/turnStats.ts``).
     */
    toolCalls?: ToolCallSummary | null;
    finishReason?: string | null;
    usage?: Record<string, number | null | undefined>;
  };
}

/**
 * One row of the instruction budget -- a source layer, or a child of one.
 *
 * The daemon's ``InstructionBudget.snapshot()`` serialises
 * ``SourceEntry.to_dict()`` verbatim, so this mirrors that shape rather
 * than reshaping it: ``tokens`` is the entry's OWN cost and
 * ``total_tokens`` includes its children, which is the number to show.
 */
export interface BudgetEntry {
  source?: string;
  /** This entry's own tokens, excluding children. */
  tokens?: number;
  /** This entry plus its children -- what the row displays. */
  total_tokens?: number;
  /** ``locked`` | ``preservable`` | ``partial`` | ``ephemeral`` | ``conditional``. */
  gc_policy?: string;
  gc_eligible_tokens?: number;
  /** The daemon's own glyph for the policy; the client does not invent one. */
  indicator?: string;
  label?: string | null;
  children?: Record<string, BudgetEntry>;
  [k: string]: unknown;
}

/**
 * ``InstructionBudgetEvent.budget_snapshot`` -- what the context window is
 * spent ON, by instruction source.
 *
 * Distinct from {@link ContextState}, which is how FULL the window is.  The
 * TUI keeps them apart too (Ctrl+B against the ``context`` command) and the
 * web rail showed only the second under a section labelled Budget.
 */
export interface BudgetState {
  contextLimit?: number | null;
  totalTokens?: number | null;
  utilizationPercent?: number | null;
  lockedTokens?: number | null;
  gcEligibleTokens?: number | null;
  /** Keyed by source name: ``system``, ``plugin``, ``enrichment``, ... */
  entries: Record<string, BudgetEntry>;
}

/**
 * What the Instructions panel says about garbage collection (#1190), per
 * agent: the policy in force (``gc.config``) and the last pass (``gc``).
 * Both are replayed by the daemon to a client that attaches late, so an
 * absent ``lastPass`` means "none reported", not "never ran" -- a daemon
 * that predates the replay says nothing about passes before the attach.
 */
export interface GcState {
  /** ``undefined`` until the daemon has said; ``strategy: null`` is "no GC configured". */
  config?: { strategy: string | null; threshold: number | null; targetPercent: number | null; continuous: boolean };
  lastPass?: {
    /** When the pass completed (the event's own timestamp, ms since epoch). */
    at: number;
    success: boolean;
    tokensFreed: number | null;
    tokensBefore: number | null;
    tokensAfter: number | null;
    trigger: string | null;
    strategy: string | null;
    error: string | null;
  };
  /** A pass is under way: between ``started`` and ``completed``. */
  running?: boolean;
}

export interface WorkspaceInfo {
  name: string;
  configured: boolean;
  provider?: string | null;
  model?: string | null;
  last_accessed?: string | null;
  /** Absolute path on the daemon host, when the daemon sends it. */
  path?: string | null;
  /** The authenticated user who created it; unset for a pre-existing or unowned workspace. */
  owner?: string | null;
  /** The git checkouts found in it (protocol 1.27), raw; read through ``normalizeSources``. */
  sources?: unknown;
}

export interface ConfigStatus {
  workspace: string;
  configured: boolean;
  provider?: string | null;
  model?: string | null;
  availableProviders: string[];
  missingFields: string[];
}

export interface ProfileInfo {
  name: string;
  description?: string;
  model?: string;
  provider?: string;
  [k: string]: unknown;
}

export interface InitProgress {
  step?: string;
  status?: string;
  message?: string;
  stepNumber?: number | null;
  totalSteps?: number | null;
}

export type ConnectionPhase = "disconnected" | "connecting" | "connected" | "reconnecting" | "closed";

export type Screen = "connect" | "workspaces" | "session";

/**
 * One answer the exit choice offers (``app/exitChoice.ts``): the key the
 * composer accepts for it, the label on its button, and what it does.
 */
export interface ExitOption {
  key: string;
  label: string;
  description: string;
}

/**
 * The TUI's exit confirmation, as a prompt in the session screen.  The
 * ``exit`` command and the status bar's Exit open it instead of leaving at
 * once; ``running`` records whether a turn was in flight when it opened,
 * which decides the option set (the TUI's ``[c/d/e/r]`` vs ``[d/e/r]``).
 * ``focus`` is the option Tab cycles to and Enter answers.  Closed
 * (``null``) by any answer, by Return, and by the session state reset.
 */
export interface ExitChoice {
  running: boolean;
  options: ExitOption[];
  focus: number;
}

/**
 * One file the user attached, on its way into the session's workspace
 * (``app/staging.ts``).  ``path`` is where it lands, workspace-relative.
 *
 * Lifecycle: ``queued`` (picked on the session picker, before there is a
 * workspace to stage into) → ``staging`` (bytes on the wire) → ``staged``
 * or ``failed`` (the daemon's ``StageFilesEvent`` said which, and why).
 * A ``staged`` entry stays in the composer's strip until the next message
 * is sent, which names it in its footer and drops it; a ``failed`` one is
 * dropped on send too.  The list lives outside the per-session state so
 * files queued on the picker survive the reset an attach performs.
 */
export interface StagedUpload {
  id: string;
  path: string;
  size: number;
  status: "queued" | "staging" | "staged" | "failed";
  /**
   * Which session (or picker) this upload belongs to, so the strip shows
   * only the active context's files rather than one flat global list an
   * upload in session A visibly followed into unrelated session B (#1250).
   * Stamped at attach time as ``uploadScope(state)`` — ``session:<id>`` once
   * a session exists, else ``""`` (the picker, before one opens).  A ``""``
   * upload is re-stamped to the session it opens, so a picker attachment
   * ends up scoped to the session it was staged for.
   */
  scope: string;
  /** The daemon's (or the client-side precheck's) reason, when ``failed``. */
  error?: string;
}

/**
 * One memory as the daemon's ``memory.list`` answer lists it (#1232,
 * protocol 1.22): every field but the content, which {@link MemoryDetail}
 * carries.  ``tier`` is which store it came from (``workspace`` /
 * ``global``) -- deliberately not ``scope``, which means how broadly the
 * memory APPLIES (``project`` / ``universal``).  ``curated_by`` is ``null``
 * on every raw memory; ``generated_by`` is ``null`` on a record older than
 * its stamp.  The two ``*_this_session`` flags are the daemon's, for the
 * session this client is attached to.
 */
export interface MemoryRow {
  id: string;
  description: string;
  tags: string[];
  /** ``raw`` | ``validated`` | ``escalated`` | ``dismissed``. */
  maturity: string;
  confidence?: number | null;
  scope?: string | null;
  tier: string;
  timestamp?: string | null;
  last_accessed?: string | null;
  usage_count?: number | null;
  generated_by?: Record<string, unknown> | null;
  curated_by?: Record<string, unknown> | null;
  source_agent?: string | null;
  source_session?: string | null;
  written_this_session?: boolean;
  retrieved_this_session?: boolean;
}

/** A row expanded: its content, or why it could not be fetched. */
export type MemoryDetail =
  | { state: "loading" }
  | { state: "loaded"; content: string; evidence?: string | null }
  | { state: "error"; message: string };

/** An open edit form's draft; ``tags`` is the comma-separated text being typed. */
export interface MemoryDraft {
  description: string;
  content: string;
  tags: string;
}

/**
 * The rail's Memories section (``app/memories.ts``).  Per session: dropped
 * with the rest of the session state, because the store it lists belongs
 * to the session's workspace.
 *
 * ``status`` is what the LAST ask established -- ``idle`` (never asked),
 * ``loading``, ``loaded``, ``error`` (the daemon answered and could not read
 * the store: its ``error`` is shown, and the rows are NOT replaced by an
 * empty list, which would read as "nothing remembered") or ``unsupported``
 * (a daemon below protocol 1.22, which serves no quiet list).
 */
export interface MemoriesState {
  rows: MemoryRow[];
  status: "idle" | "loading" | "loaded" | "error" | "unsupported";
  error: string | null;
  /** Whether THIS connection may change memories -- the daemon's owner gate, never guessed. */
  mayCurate: boolean | null;
  /** Show only memories written or retrieved in this session. */
  thisSessionOnly: boolean;
  /** The row whose content is showing. */
  expanded: string | null;
  /** Row id -> its content, fetched on expand. */
  details: Record<string, MemoryDetail>;
  /** Row id -> the action in flight on it, so its buttons disable. */
  busy: Record<string, string>;
  /** Row id -> an edit form's draft; present while the form is open. */
  editing: Record<string, MemoryDraft>;
  /** One-line outcome of the last action. */
  notice: { text: string; error?: boolean } | null;
}

/**
 * One thread the runner's live re-probe scanned (#1294): ``label`` is the
 * AppArmor profile that thread's kernel task reported, or ``(unreadable)``
 * when the scan could not read it.
 */
export interface DiagnosticsThread {
  tid: number;
  name: string | null;
  label?: string;
  reason?: string;
}

/**
 * The result of ONE on-demand re-probe of the runner's confinement
 * (``server.runner.bootstrap.probe_confinement_now``, #1294) — measured
 * fresh at the moment it was asked, never a cached value.
 *
 * ``ok === false`` means the probe itself could not determine an answer
 * (a ``/proc`` read failed, the thread walk raised) — rendered as exactly
 * that, never as a guessed ``enforced`` / ``confined``.  ``scan`` is
 * ``null`` in that same case; otherwise it is the thread-level breakdown
 * (#1023's ``ThreadProfileScan``) this probe found.
 */
export interface DiagnosticsProbe {
  ok: boolean;
  error: string;
  expected_profile: string;
  current_profile: string;
  current_mode: string | null;
  enforced: boolean;
  confined: boolean;
  scan: {
    scanned: number;
    matched: number;
    divergent: number;
    unreadable: number;
    gone: number;
    uniform: boolean;
    route: string;
    divergent_threads: DiagnosticsThread[];
    unreadable_threads: DiagnosticsThread[];
  } | null;
}

/**
 * What the session's AppArmor profile was provisioned with (#1326,
 * ``DiagnosticsResultEvent.apparmor_grants``, protocol 1.26).  Recorded
 * by the daemon when the profile was loaded, so it is a RECORD field:
 * rendered in the record tone, never as something just measured.
 * ``references`` is the one live part (listed from the boundary's refs
 * directory at the moment of the answer).
 */
export interface DiagnosticsGrants {
  /** ``false`` when the daemon has no record (the profile was loaded
   *  before it started, or on a path that does not record). */
  recorded: boolean;
  template_version?: number;
  profile_name?: string;
  exec_scope?: "scoped" | "unscoped";
  requested_fragments?: string[] | null;
  /** The profile in the ``inherits:`` chain that declared
   *  ``apparmor_fragments``: ``null`` when none did, ``""`` when unknown. */
  declared_by?: string | null;
  fragments?: { name: string; tier: string; path: string; rules: string[]; shadows?: string[] }[];
  missing_fragments?: string[];
  unreadable_fragments?: string[];
  plugin_rules?: { plugin: string; rules: string[] }[];
  references?: { ref_id: string; rules: string[] }[];
}

/**
 * The rail's Diagnostics section (#1294, ``app/diagnostics.ts``): the
 * caller's own attached session, self-diagnosed. Per session, and reset
 * with the rest of the session state.
 *
 * TWO KINDS OF FIELD, and the panel keeps them visually apart:
 *   - the RECORD fields (``runnerIdentity`` through ``serverVersion``) are
 *     the daemon's own ``Session`` bookkeeping -- CACHED, not re-measured
 *     by this call.  A cached ``sandboxMode`` reading "confined" when it is
 *     not is exactly what #1253 was filed about.
 *   - ``probe`` is measured FRESH on the runner at the moment of the last
 *     successful call (see ``checkedAt``) and is never merged into the
 *     record fields above.
 *
 * ``status`` is what the LAST ask established, the same vocabulary
 * {@link MemoriesState} uses: ``idle`` (never asked), ``loading``,
 * ``loaded``, ``error`` (the daemon answered and refused or could not
 * report — its ``error``/``category`` are shown, and the prior answer is
 * NOT replaced by a blank one) or ``unsupported`` (a daemon below protocol
 * 1.25, which serves no diagnostics verb).
 */
export interface DiagnosticsState {
  status: "idle" | "loading" | "loaded" | "error" | "unsupported";
  error: string | null;
  category: string | null;
  runnerIdentity: Record<string, unknown> | null;
  confinementId: string;
  sandboxMode: string | null;
  consumption: Record<string, unknown> | null;
  notebookBoundaryKind: string | null;
  protocolVersion: string;
  serverVersion: string;
  /** The last live re-probe -- ``null`` before any successful answer, or
   *  when that session carries no runner to probe. */
  probe: DiagnosticsProbe | null;
  /** What the AppArmor profile grants (#1326); ``null`` when the session
   *  names no profile, or the daemon predates protocol 1.26. */
  apparmorGrants: DiagnosticsGrants | null;
  /** ``Date.now()`` of the answer that populated the fields above --
   *  when the record was last read AND the probe was last measured, since
   *  one call answers both. */
  checkedAt: number | null;
}
