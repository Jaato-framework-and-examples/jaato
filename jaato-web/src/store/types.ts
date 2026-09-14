/**
 * Client-side state model.  Everything here is derived from the wire
 * events by ``reduce()`` in ``store.ts``; components never touch raw
 * events.  The shapes mirror what ``jaato-tui/output_buffer.py`` keeps
 * (``OutputLine`` / ``ToolBlock`` / ``ActiveToolCall``) so the two
 * clients present the same session the same way.
 */

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

export interface ClarificationQuestion {
  question_text?: string;
  question_type?: string; // "choice" | "text" | "confirm" | ...
  options?: string[];
  default?: string | null;
  optional?: boolean;
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

export interface Agent {
  id: string;
  name: string;
  type: string;
  parentId?: string | null;
  profile?: string | null;
  status: string; // idle | processing | awaiting_permission | finished | error …
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
    functionCalls?: number | null;
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
