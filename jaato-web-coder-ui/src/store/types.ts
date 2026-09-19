/**
 * Client-side state model.  Everything here is derived from the wire
 * events by ``reduce()`` in ``store.ts``; components never touch raw
 * events.  The shapes mirror what ``jaato-tui/output_buffer.py`` keeps
 * (``OutputLine`` / ``ToolBlock`` / ``ActiveToolCall``) so the two
 * clients present the same session the same way.
 */
import type { ToolCallSummary } from "@/protocol/turnStats";

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
  /** The daemon's (or the client-side precheck's) reason, when ``failed``. */
  error?: string;
}
