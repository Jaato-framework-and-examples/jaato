/**
 * Event types matching server/events.py protocol
 */

// Base event interface
export interface BaseEvent {
  type: string;
  timestamp: string;
}

// Connection events
export interface ConnectedEvent extends BaseEvent {
  type: 'connected';
  server_info: {
    version?: string;
    client_id: string;
    protocol_version?: string;
    workspace_mode?: boolean;
    model_provider?: string;
    model_name?: string;
  };
}

export interface ErrorEvent extends BaseEvent {
  type: 'error';
  message: string;
  error_type?: string;
}

export interface SystemMessageEvent extends BaseEvent {
  type: 'system_message';
  message: string;
  level: 'info' | 'warning' | 'success' | 'error';
}

// Agent events
export interface AgentCreatedEvent extends BaseEvent {
  type: 'agent.created';
  agent_id: string;
  agent_name: string;
  agent_type: 'main' | 'subagent';
  model?: string;
}

export interface AgentOutputEvent extends BaseEvent {
  type: 'agent.output';
  agent_id: string;
  source: 'model' | 'tool' | 'system' | string;
  text: string;
  mode: 'write' | 'append';
}

export interface AgentStatusChangedEvent extends BaseEvent {
  type: 'agent.status_changed';
  agent_id: string;
  status: 'active' | 'done' | 'error';
}

export interface AgentCompletedEvent extends BaseEvent {
  type: 'agent.completed';
  agent_id: string;
  total_tokens?: number;
  prompt_tokens?: number;
  output_tokens?: number;
}

// Tool events
export interface ToolCallStartEvent extends BaseEvent {
  type: 'tool.call_start';
  agent_id: string;
  tool_call_id: string;
  tool_name: string;
  tool_args: Record<string, unknown>;
}

export interface ToolCallEndEvent extends BaseEvent {
  type: 'tool.call_end';
  agent_id: string;
  tool_call_id: string;
  tool_name: string;
  duration_seconds: number;
  success: boolean;
  error?: string;
}

export interface ToolOutputEvent extends BaseEvent {
  type: 'tool.output';
  agent_id: string;
  tool_call_id: string;
  output: string;
}

// Permission events
export interface PermissionRequestedEvent extends BaseEvent {
  type: 'permission.requested';
  request_id: string;
  agent_id: string;
  tool_name: string;
  tool_args: Record<string, unknown>;
  prompt_lines: string[];
  response_options: Array<{
    key: string;
    label: string;
    action: string;
  }>;
  format_hint?: 'diff' | 'default';
  warnings?: string;
  warning_level?: 'info' | 'warning' | 'error';
}

export interface PermissionInputModeEvent extends BaseEvent {
  type: 'permission.input_mode';
  request_id: string;
  agent_id: string;
}

export interface PermissionResolvedEvent extends BaseEvent {
  type: 'permission.resolved';
  request_id: string;
  agent_id: string;
  tool_name: string;
  granted: boolean;
  method?: string;
}

// Clarification events
export interface ClarificationRequestedEvent extends BaseEvent {
  type: 'clarification.requested';
  request_id: string;
  agent_id: string;
  tool_name: string;
}

export interface ClarificationQuestionEvent extends BaseEvent {
  type: 'clarification.question';
  request_id: string;
  agent_id: string;
  question: string;
  question_type: 'text' | 'choice' | 'confirm';
  options?: string[];
  current_index: number;
  total_questions: number;
}

export interface ClarificationInputModeEvent extends BaseEvent {
  type: 'clarification.input_mode';
  request_id: string;
  agent_id: string;
}

export interface ClarificationResolvedEvent extends BaseEvent {
  type: 'clarification.resolved';
  request_id: string;
  agent_id: string;
  tool_name: string;
  qa_pairs: Array<[string, string]>;
}

// Context events
export interface ContextUpdatedEvent extends BaseEvent {
  type: 'context.updated';
  agent_id: string;
  total_tokens: number;
  prompt_tokens: number;
  output_tokens: number;
  context_limit: number;
  percent_used: number;
  tokens_remaining: number;
  gc_threshold: number;
  gc_strategy?: string;
}

// Plan events
export interface PlanStep {
  content: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'blocked';
  active_form?: string;
}

export interface PlanUpdatedEvent extends BaseEvent {
  type: 'plan.updated';
  agent_id: string;
  steps: PlanStep[];
}

export interface PlanClearedEvent extends BaseEvent {
  type: 'plan.cleared';
  agent_id: string;
}

// Turn events
export interface TurnCompletedEvent extends BaseEvent {
  type: 'turn.completed';
  agent_id: string;
  text?: string;
}

// Session events
export interface SessionInfoEvent extends BaseEvent {
  type: 'session.info';
  sessions: Array<{
    id: string;
    description?: string;
    created_at: string;
    agent_count: number;
  }>;
  active_session_id?: string;
  available_models: string[];
  current_model: string;
}

export interface SessionDescriptionUpdatedEvent extends BaseEvent {
  type: 'session.description_updated';
  description: string;
}

// Retry event
export interface RetryEvent extends BaseEvent {
  type: 'retry';
  attempt: number;
  max_attempts: number;
  delay_seconds: number;
  reason?: string;
}

// Init progress
export interface InitProgressEvent extends BaseEvent {
  type: 'init.progress';
  step: string;
  current: number;
  total: number;
}

// Mid-turn events
export interface MidTurnPromptQueuedEvent extends BaseEvent {
  type: 'mid_turn.prompt_queued';
  agent_id: string;
  text: string;
}

export interface MidTurnPromptInjectedEvent extends BaseEvent {
  type: 'mid_turn.prompt_injected';
  agent_id: string;
  text: string;
}

// Workspace events
export interface Workspace {
  name: string;
  path: string;
  configured: boolean;
  provider?: string;
  model?: string;
  last_accessed?: string;
}

export interface WorkspaceListEvent extends BaseEvent {
  type: 'workspace.list_response';
  workspaces: Workspace[];
}

export interface WorkspaceCreatedEvent extends BaseEvent {
  type: 'workspace.created';
  workspace: Workspace;
}

export interface ConfigStatusEvent extends BaseEvent {
  type: 'config.status';
  workspace: string;
  configured: boolean;
  provider?: string;
  model?: string;
  available_providers: string[];
  missing_fields: string[];
}

export interface ConfigUpdatedEvent extends BaseEvent {
  type: 'config.updated';
  workspace: string;
  provider: string;
  model?: string;
  success: boolean;
}

// Union type of all server events
export type ServerEvent =
  | ConnectedEvent
  | ErrorEvent
  | SystemMessageEvent
  | AgentCreatedEvent
  | AgentOutputEvent
  | AgentStatusChangedEvent
  | AgentCompletedEvent
  | ToolCallStartEvent
  | ToolCallEndEvent
  | ToolOutputEvent
  | PermissionRequestedEvent
  | PermissionInputModeEvent
  | PermissionResolvedEvent
  | ClarificationRequestedEvent
  | ClarificationQuestionEvent
  | ClarificationInputModeEvent
  | ClarificationResolvedEvent
  | ContextUpdatedEvent
  | PlanUpdatedEvent
  | PlanClearedEvent
  | TurnCompletedEvent
  | SessionInfoEvent
  | SessionDescriptionUpdatedEvent
  | RetryEvent
  | InitProgressEvent
  | MidTurnPromptQueuedEvent
  | MidTurnPromptInjectedEvent
  | WorkspaceListEvent
  | WorkspaceCreatedEvent
  | ConfigStatusEvent
  | ConfigUpdatedEvent;

// Client → Server request types
export interface SendMessageRequest {
  type: 'message.send';
  text: string;
  attachments: Array<{
    type: 'file' | 'image';
    path?: string;
    data?: string;
    mime_type?: string;
  }>;
  timestamp: string;
}

export interface PermissionResponseRequest {
  type: 'permission.response';
  request_id: string;
  response: string;
  timestamp: string;
}

export interface ClarificationResponseRequest {
  type: 'clarification.response';
  request_id: string;
  answer: string;
  timestamp: string;
}

export interface StopRequest {
  type: 'stop';
  agent_id?: string;
  timestamp: string;
}

export interface CommandRequest {
  type: 'command';
  name: string;
  args: string[];
  timestamp: string;
}

export interface ClientConfigRequest {
  type: 'client.config';
  cwd: string;
  env: Record<string, string>;
  terminal_width: number;
  timestamp: string;
}

export interface HistoryRequest {
  type: 'history';
  timestamp: string;
}

// Workspace requests
export interface WorkspaceListRequest {
  type: 'workspace.list';
  timestamp: string;
}

export interface WorkspaceCreateRequest {
  type: 'workspace.create';
  name: string;
  timestamp: string;
}

export interface WorkspaceSelectRequest {
  type: 'workspace.select';
  name: string;
  timestamp: string;
}

export interface ConfigUpdateRequest {
  type: 'config.update';
  provider: string;
  model?: string;
  api_key?: string;
  timestamp: string;
}

export type ClientRequest =
  | SendMessageRequest
  | PermissionResponseRequest
  | ClarificationResponseRequest
  | StopRequest
  | CommandRequest
  | ClientConfigRequest
  | HistoryRequest
  | WorkspaceListRequest
  | WorkspaceCreateRequest
  | WorkspaceSelectRequest
  | ConfigUpdateRequest;
