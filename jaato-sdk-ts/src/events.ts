/*
 * GENERATED FILE — DO NOT EDIT.
 *
 * Mirror of jaato-sdk/jaato_sdk/events.py (pydantic), regenerated
 * via scripts/codegen_ts_events.py.  CI fails if this file is stale
 * relative to events.py — re-run the codegen and commit the result.
 *
 * The wire protocol is the contract: every event below has a
 * matching pydantic model on the Python side and a baseline JSON
 * snapshot in jaato-sdk/jaato_sdk/tests/baselines/events_wire_format/.
 */

/* eslint-disable */

export type JaatoEvents =
  | ConnectedEvent
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
  | PermissionStatusEvent
  | ClarificationRequestedEvent
  | ClarificationInputModeEvent
  | ClarificationQuestionEvent
  | ClarificationResolvedEvent
  | ClarificationBatchEvent
  | ClarificationBatchResponseEvent
  | ReferenceSelectionRequestedEvent
  | ReferenceSelectionResolvedEvent
  | ReferenceSelectionResponseRequest
  | WorkspaceMismatchRequestedEvent
  | WorkspaceMismatchResolvedEvent
  | WorkspaceMismatchResponseRequest
  | PostAuthSetupEvent
  | PostAuthSetupResponse
  | PlanUpdatedEvent
  | PlanStepUpdatedEvent
  | PlanClearedEvent
  | ContextUpdatedEvent
  | InstructionBudgetEvent
  | TurnCompletedEvent
  | TurnProgressEvent
  | SystemMessageEvent
  | HelpTextEvent
  | InitProgressEvent
  | ErrorEvent
  | RetryEvent
  | SessionListEvent
  | SessionInfoEvent
  | MemoryListEvent
  | SandboxPathsEvent
  | ServiceListEvent
  | SessionDescriptionUpdatedEvent
  | SessionProfilesEvent
  | SendMessageRequest
  | PermissionResponseRequest
  | ClarificationResponseRequest
  | StopRequest
  | ExternalEventRequest
  | EventsSubscribedEvent
  | CommandRequest
  | GetInstructionBudgetRequest
  | CommandListRequest
  | CommandListEvent
  | CommandListRefreshEvent
  | ToolStatusEvent
  | ToolIdRegistryEvent
  | ToolDisableRequest
  | ToolsRegisterClientRequest
  | ToolExecuteRequestEvent
  | ToolExecuteResultEvent
  | HistoryRequest
  | HistoryEvent
  | ClientConfigRequest
  | MidTurnPromptQueuedEvent
  | MidTurnPromptInjectedEvent
  | MidTurnInterruptEvent
  | InterruptedTurnRecoveredEvent
  | WorkspaceListRequest
  | WorkspaceListEvent
  | WorkspaceCreateRequest
  | WorkspaceCreatedEvent
  | WorkspaceSelectRequest
  | ConfigStatusEvent
  | ConfigUpdateRequest
  | ConfigUpdatedEvent
  | WorkspaceFilesChangedEvent
  | WorkspaceFilesSnapshotEvent
  | StageFilesRequest
  | StageFilesEvent
  | PeerHeartbeatEvent
  | PeerSpawnRequestEvent
  | PeerSpawnAcceptedEvent
  | PeerSpawnRejectedEvent
  | PeerAgentOutputEvent
  | PeerAgentCompletedEvent
  | PeerStopRequestEvent
  | PeerStopAcknowledgedEvent
  | InjectPromptRequest
  | ReplayMessagesRequest
  | ReplayMessagesResultEvent
  | ResolveForkPointRequest
  | ResolveForkPointResultEvent
  | PermissionAddWhitelistRequest
  | PermissionAddBlacklistRequest
  | PermissionRemoveRequest
  | PermissionClearRequest
  | PermissionSetDefaultRequest
  | PermissionPolicySnapshotRequest
  | PermissionPolicySnapshotEvent;
/**
 * All event types in the protocol.
 */
export type EventType =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp = string;
export type ProtocolVersion = string;
/**
 * All event types in the protocol.
 */
export type EventType1 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp1 = string;
export type AgentId = string;
export type AgentName = string;
export type AgentType = string;
export type ProfileName = string | null;
export type ParentAgentId = string | null;
export type CreatedAt = string | null;
/**
 * All event types in the protocol.
 */
export type EventType2 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp2 = string;
export type AgentId1 = string;
export type Source = string;
export type Text = string;
export type Mode = string;
/**
 * All event types in the protocol.
 */
export type EventType3 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp3 = string;
export type AgentId2 = string;
export type Status = string;
export type Error = string | null;
/**
 * All event types in the protocol.
 */
export type EventType4 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp4 = string;
export type AgentId3 = string;
export type CompletedAt = string;
export type Success = boolean;
export type TokenUsage = {
  [k: string]: number;
} | null;
export type TurnsUsed = number | null;
export type Error1 = string;
export type Payload = {
  [k: string]: unknown;
} | null;
/**
 * All event types in the protocol.
 */
export type EventType5 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp5 = string;
export type AgentId4 = string;
export type ToolName = string;
export type CallId = string | null;
/**
 * All event types in the protocol.
 */
export type EventType6 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp6 = string;
export type AgentId5 = string;
export type ToolName1 = string;
export type CallId1 = string | null;
export type Success1 = boolean;
export type DurationSeconds = number;
export type ErrorMessage = string | null;
export type Backgrounded = boolean;
export type ContinuationId = string | null;
export type ShowOutput = boolean | null;
export type ShowPopup = boolean | null;
/**
 * All event types in the protocol.
 */
export type EventType7 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp7 = string;
export type AgentId6 = string;
export type CallId2 = string;
export type Chunk = string;
/**
 * All event types in the protocol.
 */
export type EventType8 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp8 = string;
export type AgentId7 = string;
export type RequestId = string;
export type ToolName2 = string;
export type ResponseOptions = {
  [k: string]: string;
}[];
export type PromptLines = string[] | null;
export type FormatHint = string | null;
export type Warnings = string | null;
export type WarningLevel = string | null;
/**
 * All event types in the protocol.
 */
export type EventType9 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp9 = string;
export type AgentId8 = string;
export type RequestId1 = string;
export type ToolName3 = string;
export type CallId3 = string | null;
export type ResponseOptions1 = {
  [k: string]: string;
}[];
export type ToolArgs2 = {
  [k: string]: unknown;
} | null;
export type EditableMetadata = {
  [k: string]: unknown;
} | null;
/**
 * All event types in the protocol.
 */
export type EventType10 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp10 = string;
export type AgentId9 = string;
export type RequestId2 = string;
export type ToolName4 = string;
export type Granted = boolean;
export type Method = string;
export type Comment = string;
/**
 * All event types in the protocol.
 */
export type EventType11 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp11 = string;
export type EffectiveDefault = string;
export type SuspensionScope = string | null;
/**
 * All event types in the protocol.
 */
export type EventType12 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp12 = string;
export type AgentId10 = string;
export type RequestId3 = string;
export type ToolName5 = string;
export type ContextLines = string[];
export type TotalQuestions = number;
/**
 * All event types in the protocol.
 */
export type EventType13 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp13 = string;
export type AgentId11 = string;
export type RequestId4 = string;
export type ToolName6 = string;
export type QuestionIndex = number;
export type TotalQuestions1 = number;
/**
 * All event types in the protocol.
 */
export type EventType14 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp14 = string;
export type AgentId12 = string;
export type RequestId5 = string;
export type QuestionIndex1 = number;
export type TotalQuestions2 = number;
export type QuestionType = string;
export type QuestionText = string;
export type Options =
  | {
      [k: string]: string;
    }[]
  | null;
/**
 * All event types in the protocol.
 */
export type EventType15 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp15 = string;
export type AgentId13 = string;
export type RequestId6 = string;
export type ToolName7 = string;
export type QaPairs = string[][];
/**
 * All event types in the protocol.
 */
export type EventType16 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp16 = string;
export type AgentId14 = string;
export type RequestId7 = string;
export type ToolName8 = string;
export type Context = string;
export type Questions = {
  [k: string]: unknown;
}[];
/**
 * All event types in the protocol.
 */
export type EventType17 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp17 = string;
export type RequestId8 = string;
export type Answers = string[];
/**
 * All event types in the protocol.
 */
export type EventType18 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp18 = string;
export type AgentId15 = string;
export type RequestId9 = string;
export type ToolName9 = string;
export type PromptLines1 = string[];
/**
 * All event types in the protocol.
 */
export type EventType19 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp19 = string;
export type AgentId16 = string;
export type RequestId10 = string;
export type ToolName10 = string;
export type SelectedIds = string[];
/**
 * All event types in the protocol.
 */
export type EventType20 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp20 = string;
export type RequestId11 = string;
export type Response = string;
/**
 * All event types in the protocol.
 */
export type EventType21 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp21 = string;
export type RequestId12 = string;
export type SessionId = string;
export type SessionWorkspace = string;
export type ClientWorkspace = string;
export type ResponseOptions2 = {
  [k: string]: string;
}[];
export type PromptLines2 = string[];
/**
 * All event types in the protocol.
 */
export type EventType22 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp22 = string;
export type RequestId13 = string;
export type SessionId1 = string;
export type Action = string;
export type NewSessionId = string | null;
/**
 * All event types in the protocol.
 */
export type EventType23 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp23 = string;
export type RequestId14 = string;
export type Response1 = string;
/**
 * All event types in the protocol.
 */
export type EventType24 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp24 = string;
export type RequestId15 = string;
export type ProviderName = string;
export type ProviderDisplayName = string;
export type AvailableModels = {
  [k: string]: string;
}[];
export type HasActiveSession = boolean;
export type CurrentProvider = string;
export type CurrentModel = string;
export type WorkspacePath = string;
/**
 * All event types in the protocol.
 */
export type EventType25 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp25 = string;
export type RequestId16 = string;
export type Connect = boolean;
export type ModelName = string;
export type PersistEnv = boolean;
/**
 * All event types in the protocol.
 */
export type EventType26 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp26 = string;
export type AgentId17 = string;
export type PlanName = string;
export type Steps = {
  [k: string]: unknown;
}[];
/**
 * All event types in the protocol.
 */
export type EventType27 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp27 = string;
export type AgentId18 = string;
export type StepId = string;
export type Sequence = number;
export type Content = string;
export type Status1 = string;
export type Result = string | null;
export type Error2 = string | null;
export type BlockedBy =
  | {
      [k: string]: unknown;
    }[]
  | null;
export type DependsOn =
  | {
      [k: string]: unknown;
    }[]
  | null;
export type ReceivedOutputs = {
  [k: string]: unknown;
} | null;
/**
 * All event types in the protocol.
 */
export type EventType28 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp28 = string;
export type AgentId19 = string;
/**
 * All event types in the protocol.
 */
export type EventType29 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp29 = string;
export type AgentId20 = string;
export type TotalTokens = number;
export type PromptTokens = number;
export type OutputTokens = number;
export type ContextLimit = number;
export type PercentUsed = number;
export type TokensRemaining = number;
export type Turns = number;
export type GcThreshold = number | null;
export type GcStrategy = string | null;
export type GcTargetPercent = number | null;
export type GcContinuousMode = boolean;
/**
 * All event types in the protocol.
 */
export type EventType30 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp30 = string;
export type AgentId21 = string;
/**
 * All event types in the protocol.
 */
export type EventType31 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp31 = string;
export type AgentId22 = string;
export type TurnNumber = number;
export type PromptTokens1 = number;
export type OutputTokens1 = number;
export type TotalTokens1 = number;
export type DurationSeconds1 = number;
export type FunctionCalls = {
  [k: string]: unknown;
}[];
export type FormattedText = string | null;
export type CacheReadTokens = number | null;
export type CacheCreationTokens = number | null;
/**
 * All event types in the protocol.
 */
export type EventType32 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp32 = string;
export type AgentId23 = string;
export type TotalTokens2 = number;
export type PromptTokens2 = number;
export type OutputTokens2 = number;
export type ContextLimit1 = number;
export type PercentUsed1 = number;
export type TokensRemaining1 = number;
export type PendingToolCalls = number;
export type CacheReadTokens1 = number | null;
export type CacheCreationTokens1 = number | null;
/**
 * All event types in the protocol.
 */
export type EventType33 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp33 = string;
export type Message = string;
export type Style = string;
/**
 * All event types in the protocol.
 */
export type EventType34 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp34 = string;
export type Lines = unknown[][];
/**
 * All event types in the protocol.
 */
export type EventType35 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp35 = string;
export type Step = string;
export type Status2 = string;
export type Message1 = string;
export type StepNumber = number;
export type TotalSteps = number;
/**
 * All event types in the protocol.
 */
export type EventType36 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp36 = string;
export type Error3 = string;
export type ErrorType = string;
export type Recoverable = boolean;
/**
 * All event types in the protocol.
 */
export type EventType37 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp37 = string;
export type Message2 = string;
export type Attempt = number;
export type MaxAttempts = number;
export type Delay = number;
export type ErrorType1 = string;
/**
 * All event types in the protocol.
 */
export type EventType38 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp38 = string;
export type Sessions = {
  [k: string]: unknown;
}[];
/**
 * All event types in the protocol.
 */
export type EventType39 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp39 = string;
export type SessionId2 = string;
export type SessionName = string;
export type ModelProvider = string;
export type ModelName1 = string;
export type ProfileName1 = string | null;
export type Sessions1 = {
  [k: string]: unknown;
}[];
export type Tools = {
  [k: string]: unknown;
}[];
export type Models = string[];
export type UserInputs = string[];
export type Memories = {
  [k: string]: unknown;
}[];
export type SandboxPaths = {
  [k: string]: string;
}[];
export type Services = {
  [k: string]: unknown;
}[];
/**
 * All event types in the protocol.
 */
export type EventType40 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp40 = string;
export type Memories1 = {
  [k: string]: unknown;
}[];
/**
 * All event types in the protocol.
 */
export type EventType41 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp41 = string;
export type Paths = {
  [k: string]: string;
}[];
/**
 * All event types in the protocol.
 */
export type EventType42 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp42 = string;
export type Services1 = {
  [k: string]: unknown;
}[];
/**
 * All event types in the protocol.
 */
export type EventType43 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp43 = string;
export type SessionId3 = string;
export type Description = string;
/**
 * All event types in the protocol.
 */
export type EventType44 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp44 = string;
export type Profiles = {
  [k: string]: unknown;
}[];
/**
 * All event types in the protocol.
 */
export type EventType45 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp45 = string;
export type Text1 = string;
export type Attachments = {
  [k: string]: unknown;
}[];
export type ParallelTools = boolean | null;
/**
 * All event types in the protocol.
 */
export type EventType46 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp46 = string;
export type RequestId17 = string;
export type Response2 = string;
export type EditedArguments = {
  [k: string]: unknown;
} | null;
/**
 * All event types in the protocol.
 */
export type EventType47 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp47 = string;
export type RequestId18 = string;
export type QuestionIndex2 = number;
export type Response3 = string;
/**
 * All event types in the protocol.
 */
export type EventType48 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp48 = string;
export type AgentId24 = string | null;
/**
 * All event types in the protocol.
 */
export type EventType49 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp49 = string;
export type Name = string;
/**
 * All event types in the protocol.
 */
export type EventType50 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp50 = string;
export type AgentId25 = string;
export type EventNames = string[];
/**
 * All event types in the protocol.
 */
export type EventType51 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp51 = string;
export type Command = string;
export type Args = string[];
/**
 * All event types in the protocol.
 */
export type EventType52 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp52 = string;
export type AgentId26 = string | null;
/**
 * All event types in the protocol.
 */
export type EventType53 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp53 = string;
/**
 * All event types in the protocol.
 */
export type EventType54 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp54 = string;
export type Commands = {
  [k: string]: string;
}[];
/**
 * All event types in the protocol.
 */
export type EventType55 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp55 = string;
/**
 * All event types in the protocol.
 */
export type EventType56 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp56 = string;
export type Tools1 = {
  [k: string]: unknown;
}[];
export type Message3 = string;
/**
 * All event types in the protocol.
 */
export type EventType57 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp57 = string;
/**
 * All event types in the protocol.
 */
export type EventType58 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp58 = string;
export type ToolName11 = string;
/**
 * All event types in the protocol.
 */
export type EventType59 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp59 = string;
export type Tools2 = {
  [k: string]: unknown;
}[];
/**
 * All event types in the protocol.
 */
export type EventType60 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp60 = string;
export type CallId4 = string;
export type AgentId27 = string;
export type ToolName12 = string;
/**
 * All event types in the protocol.
 */
export type EventType61 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp61 = string;
export type CallId5 = string;
export type Result1 = string;
export type Error4 = string;
/**
 * All event types in the protocol.
 */
export type EventType62 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp62 = string;
export type AgentId28 = string;
/**
 * All event types in the protocol.
 */
export type EventType63 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp63 = string;
export type AgentId29 = string;
export type History = {
  [k: string]: unknown;
}[];
export type TurnAccounting = {
  [k: string]: number;
}[];
/**
 * All event types in the protocol.
 */
export type EventType64 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp64 = string;
export type TraceLogPath = string | null;
export type ProviderTraceLog = string | null;
export type WorkingDir = string | null;
export type EnvFile = string | null;
export type Presentation = {
  [k: string]: unknown;
} | null;
export type PermissionTimeout = number | null;
/**
 * All event types in the protocol.
 */
export type EventType65 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp65 = string;
export type Text2 = string;
export type PositionInQueue = number;
/**
 * All event types in the protocol.
 */
export type EventType66 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp66 = string;
export type Text3 = string;
/**
 * All event types in the protocol.
 */
export type EventType67 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp67 = string;
export type PartialResponseChars = number;
export type UserPromptPreview = string;
/**
 * All event types in the protocol.
 */
export type EventType68 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp68 = string;
export type SessionId4 = string;
export type AgentId30 = string;
export type RecoveredCalls = number;
export type ActionTaken = string;
/**
 * All event types in the protocol.
 */
export type EventType69 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp69 = string;
/**
 * All event types in the protocol.
 */
export type EventType70 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp70 = string;
export type Root = string;
export type Workspaces = {
  [k: string]: unknown;
}[];
/**
 * All event types in the protocol.
 */
export type EventType71 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp71 = string;
export type Name1 = string;
/**
 * All event types in the protocol.
 */
export type EventType72 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp72 = string;
export type Name2 = string;
export type Path = string;
/**
 * All event types in the protocol.
 */
export type EventType73 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp73 = string;
export type Name3 = string;
/**
 * All event types in the protocol.
 */
export type EventType74 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp74 = string;
export type Workspace = string;
export type Configured = boolean;
export type Provider = string | null;
export type Model = string | null;
export type AvailableProviders = string[];
export type MissingFields = string[];
/**
 * All event types in the protocol.
 */
export type EventType75 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp75 = string;
export type Provider1 = string;
export type Model1 = string | null;
export type ApiKey = string | null;
/**
 * All event types in the protocol.
 */
export type EventType76 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp76 = string;
export type Workspace1 = string;
export type Provider2 = string;
export type Model2 = string | null;
export type Success2 = boolean;
export type Error5 = string | null;
/**
 * All event types in the protocol.
 */
export type EventType77 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp77 = string;
export type Changes = {
  [k: string]: string;
}[];
/**
 * All event types in the protocol.
 */
export type EventType78 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp78 = string;
export type Files = {
  [k: string]: string;
}[];
export type Total = number;
/**
 * All event types in the protocol.
 */
export type EventType79 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp79 = string;
export type WorkspaceId = string;
export type Name4 = string;
export type Size = number;
export type ContentType = string | null;
export type Mode1 = number | null;
export type Files1 = StagedFileSpec[];
/**
 * All event types in the protocol.
 */
export type EventType80 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp80 = string;
export type WorkspaceId1 = string;
export type Staged = string[];
export type Failed = {
  [k: string]: string;
}[];
/**
 * All event types in the protocol.
 */
export type EventType81 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp81 = string;
export type ServerId = string;
export type ServerName = string;
export type ServerVersion = string;
export type ActiveSessions = number;
export type ActiveAgents = number;
export type AvailableProviders1 = string[];
export type AvailableModels1 = string[];
export type Tags = string[];
export type CpuPercent = number;
export type MemoryPercent = number;
export type UptimeSeconds = number;
export type TrustState = string;
export type SuccessRate1H = number;
export type EscalatedTools = number;
/**
 * All event types in the protocol.
 */
export type EventType82 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp82 = string;
export type RequestId19 = string;
export type OriginServer = string;
export type AgentName1 = string;
export type Task = string;
export type Context1 = string;
export type ProfileJson = string;
export type InlineConfigJson = string;
export type WorkspaceGitUrl = string;
export type WorkspaceBranch = string;
export type WorkspaceCommit = string;
export type WorkspaceTempBranch = string;
/**
 * All event types in the protocol.
 */
export type EventType83 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp83 = string;
export type RequestId20 = string;
export type RemoteAgentId = string;
/**
 * All event types in the protocol.
 */
export type EventType84 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp84 = string;
export type RequestId21 = string;
export type Reason = string;
/**
 * All event types in the protocol.
 */
export type EventType85 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp85 = string;
export type RequestId22 = string;
export type RemoteAgentId1 = string;
export type Text4 = string;
export type Source1 = string;
/**
 * All event types in the protocol.
 */
export type EventType86 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp86 = string;
export type RequestId23 = string;
export type RemoteAgentId2 = string;
export type Success3 = boolean;
export type Summary = string;
export type Error6 = string;
export type WorkspaceModified = boolean;
/**
 * All event types in the protocol.
 */
export type EventType87 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp87 = string;
export type RequestId24 = string;
export type RemoteAgentId3 = string;
/**
 * All event types in the protocol.
 */
export type EventType88 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp88 = string;
export type RequestId25 = string;
export type RemoteAgentId4 = string;
/**
 * All event types in the protocol.
 */
export type EventType89 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp89 = string;
export type Text5 = string;
export type SourceType = string;
export type SourceId = string | null;
/**
 * All event types in the protocol.
 */
export type EventType90 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp90 = string;
export type RequestId26 = string;
export type Messages =
  | {
      [k: string]: unknown;
    }[]
  | null;
export type TimeoutSeconds = number;
/**
 * All event types in the protocol.
 */
export type EventType91 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp91 = string;
export type RequestId27 = string;
export type ResponseText = string;
export type Error7 = string;
/**
 * All event types in the protocol.
 */
export type EventType92 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp92 = string;
export type RequestId28 = string;
export type AfterMessage = number | null;
export type AfterToolCall = string | null;
export type AfterTimestamp = string | null;
/**
 * All event types in the protocol.
 */
export type EventType93 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp93 = string;
export type RequestId29 = string;
export type ForkIndex = number;
export type Error8 = string;
/**
 * All event types in the protocol.
 */
export type EventType94 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp94 = string;
export type Tools3 = string[];
export type Patterns = string[];
/**
 * All event types in the protocol.
 */
export type EventType95 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp95 = string;
export type Tools4 = string[];
export type Patterns1 = string[];
/**
 * All event types in the protocol.
 */
export type EventType96 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp96 = string;
export type Target = string;
export type Tools5 = string[];
export type Patterns2 = string[];
/**
 * All event types in the protocol.
 */
export type EventType97 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp97 = string;
export type Target1 = string;
/**
 * All event types in the protocol.
 */
export type EventType98 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp98 = string;
export type Policy = string;
/**
 * All event types in the protocol.
 */
export type EventType99 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp99 = string;
export type RequestId30 = string;
/**
 * All event types in the protocol.
 */
export type EventType100 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "tool.call_start"
  | "tool.call_end"
  | "tool.output"
  | "permission.requested"
  | "permission.input_mode"
  | "permission.resolved"
  | "permission.response"
  | "permission.status"
  | "clarification.requested"
  | "clarification.input_mode"
  | "clarification.question"
  | "clarification.resolved"
  | "clarification.response"
  | "clarification.batch"
  | "clarification.batch_response"
  | "reference_selection.requested"
  | "reference_selection.resolved"
  | "reference_selection.response"
  | "workspace_mismatch.requested"
  | "workspace_mismatch.resolved"
  | "workspace_mismatch.response"
  | "plan.updated"
  | "plan.step_updated"
  | "plan.cleared"
  | "context.updated"
  | "turn.completed"
  | "turn.progress"
  | "instruction_budget.updated"
  | "instruction_budget.request"
  | "system.message"
  | "help.text"
  | "error"
  | "init.progress"
  | "retry"
  | "session.list"
  | "session.info"
  | "session.description_updated"
  | "memory.list"
  | "sandbox.paths"
  | "service.list"
  | "message.send"
  | "session.stop"
  | "command.execute"
  | "command.list_request"
  | "command.list"
  | "command.list_refresh"
  | "tools.status"
  | "tools.id_registry"
  | "tools.disable"
  | "tools.register_client"
  | "tool.execute_request"
  | "tool.execute_result"
  | "history.request"
  | "history"
  | "client.config"
  | "mid_turn_prompt.queued"
  | "mid_turn_prompt.injected"
  | "mid_turn_prompt.interrupt"
  | "session.interrupted_turn_recovered"
  | "auth.setup"
  | "auth.setup_response"
  | "workspace.list"
  | "workspace.list_response"
  | "workspace.create"
  | "workspace.created"
  | "workspace.select"
  | "config.status"
  | "config.update"
  | "config.updated"
  | "workspace.files.stage_request"
  | "workspace.files.staged"
  | "session.profiles"
  | "workspace.files_changed"
  | "workspace.files_snapshot"
  | "event.external"
  | "inject_prompt.request"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "permission.add_whitelist"
  | "permission.add_blacklist"
  | "permission.remove"
  | "permission.clear"
  | "permission.set_default"
  | "permission.policy_snapshot.request"
  | "permission.policy_snapshot"
  | "events.subscribed"
  | "peer.heartbeat"
  | "peer.spawn_request"
  | "peer.spawn_accepted"
  | "peer.spawn_rejected"
  | "peer.agent_output"
  | "peer.agent_completed"
  | "peer.stop_request"
  | "peer.stop_acknowledged";
export type Timestamp100 = string;
export type RequestId31 = string;
export type DefaultPolicy = string;
export type SessionDefaultPolicy = string | null;
export type WhitelistTools = string[];
export type WhitelistPatterns = string[];
export type BlacklistTools = string[];
export type BlacklistPatterns = string[];
export type SessionWhitelist = string[];
export type SessionBlacklist = string[];

/**
 * Sent when client connects successfully.
 */
export interface ConnectedEvent {
  type?: EventType;
  timestamp?: Timestamp;
  protocol_version?: ProtocolVersion;
  server_info?: ServerInfo;
}
export interface ServerInfo {
  [k: string]: unknown;
}
/**
 * Sent when a new agent (main or subagent) is created.
 */
export interface AgentCreatedEvent {
  type?: EventType1;
  timestamp?: Timestamp1;
  agent_id?: AgentId;
  agent_name?: AgentName;
  agent_type?: AgentType;
  profile_name?: ProfileName;
  parent_agent_id?: ParentAgentId;
  created_at?: CreatedAt;
}
/**
 * Streaming text output from an agent.
 */
export interface AgentOutputEvent {
  type?: EventType2;
  timestamp?: Timestamp2;
  agent_id?: AgentId1;
  source?: Source;
  text?: Text;
  mode?: Mode;
}
/**
 * Agent status change (active, idle, done, error).
 */
export interface AgentStatusChangedEvent {
  type?: EventType3;
  timestamp?: Timestamp3;
  agent_id?: AgentId2;
  status?: Status;
  error?: Error;
}
/**
 * Agent has completed its task.
 *
 * The ``payload`` field carries the validated typed payload from
 * ``signal_completion`` when the agent's profile declared a
 * ``completion_payload_schema``. Reactor consumers should prefer
 * ``payload`` (structured fields) over ``summary`` (free text). When
 * the profile did not declare a schema, ``payload`` is ``None`` and
 * consumers fall back to reading the legacy ``summary`` field on the
 * associated tool result.
 */
export interface AgentCompletedEvent {
  type?: EventType4;
  timestamp?: Timestamp4;
  agent_id?: AgentId3;
  completed_at?: CompletedAt;
  success?: Success;
  token_usage?: TokenUsage;
  turns_used?: TurnsUsed;
  error?: Error1;
  payload?: Payload;
}
/**
 * Tool execution has started.
 */
export interface ToolCallStartEvent {
  type?: EventType5;
  timestamp?: Timestamp5;
  agent_id?: AgentId4;
  tool_name?: ToolName;
  tool_args?: ToolArgs;
  call_id?: CallId;
}
export interface ToolArgs {
  [k: string]: unknown;
}
/**
 * Tool execution has completed.
 */
export interface ToolCallEndEvent {
  type?: EventType6;
  timestamp?: Timestamp6;
  agent_id?: AgentId5;
  tool_name?: ToolName1;
  call_id?: CallId1;
  success?: Success1;
  duration_seconds?: DurationSeconds;
  error_message?: ErrorMessage;
  backgrounded?: Backgrounded;
  continuation_id?: ContinuationId;
  show_output?: ShowOutput;
  show_popup?: ShowPopup;
}
/**
 * Live output chunk from a running tool (tail -f style).
 */
export interface ToolOutputEvent {
  type?: EventType7;
  timestamp?: Timestamp7;
  agent_id?: AgentId6;
  call_id?: CallId2;
  chunk?: Chunk;
}
/**
 * Permission is requested for a tool execution.
 *
 * Includes pre-formatted prompt lines (with diff for file edits) when available.
 */
export interface PermissionRequestedEvent {
  type?: EventType8;
  timestamp?: Timestamp8;
  agent_id?: AgentId7;
  request_id?: RequestId;
  tool_name?: ToolName2;
  tool_args?: ToolArgs1;
  response_options?: ResponseOptions;
  prompt_lines?: PromptLines;
  format_hint?: FormatHint;
  warnings?: Warnings;
  warning_level?: WarningLevel;
}
export interface ToolArgs1 {
  [k: string]: unknown;
}
/**
 * Signal client to enter permission input mode.
 *
 * Sent AFTER permission content has been emitted via AgentOutputEvent.
 * This lightweight control event separates content delivery from input control.
 */
export interface PermissionInputModeEvent {
  type?: EventType9;
  timestamp?: Timestamp9;
  agent_id?: AgentId8;
  request_id?: RequestId1;
  tool_name?: ToolName3;
  call_id?: CallId3;
  response_options?: ResponseOptions1;
  tool_args?: ToolArgs2;
  editable_metadata?: EditableMetadata;
}
/**
 * Permission has been resolved (granted or denied).
 */
export interface PermissionResolvedEvent {
  type?: EventType10;
  timestamp?: Timestamp10;
  agent_id?: AgentId9;
  request_id?: RequestId2;
  tool_name?: ToolName4;
  granted?: Granted;
  method?: Method;
  comment?: Comment;
}
/**
 * Permission status update for client toolbar display.
 *
 * Emitted after permission commands (default/suspend/resume) and
 * permission resolutions that change the effective policy.
 */
export interface PermissionStatusEvent {
  type?: EventType11;
  timestamp?: Timestamp11;
  effective_default?: EffectiveDefault;
  suspension_scope?: SuspensionScope;
}
/**
 * Clarification session has started.
 */
export interface ClarificationRequestedEvent {
  type?: EventType12;
  timestamp?: Timestamp12;
  agent_id?: AgentId10;
  request_id?: RequestId3;
  tool_name?: ToolName5;
  context_lines?: ContextLines;
  total_questions?: TotalQuestions;
}
/**
 * Signal client to enter clarification input mode.
 *
 * Sent AFTER clarification content has been emitted via AgentOutputEvent.
 * This lightweight control event separates content delivery from input control.
 */
export interface ClarificationInputModeEvent {
  type?: EventType13;
  timestamp?: Timestamp13;
  agent_id?: AgentId11;
  request_id?: RequestId4;
  tool_name?: ToolName6;
  question_index?: QuestionIndex;
  total_questions?: TotalQuestions1;
}
/**
 * A single clarification question to answer.
 */
export interface ClarificationQuestionEvent {
  type?: EventType14;
  timestamp?: Timestamp14;
  agent_id?: AgentId12;
  request_id?: RequestId5;
  question_index?: QuestionIndex1;
  total_questions?: TotalQuestions2;
  question_type?: QuestionType;
  question_text?: QuestionText;
  options?: Options;
}
/**
 * All clarification questions have been answered.
 */
export interface ClarificationResolvedEvent {
  type?: EventType15;
  timestamp?: Timestamp15;
  agent_id?: AgentId13;
  request_id?: RequestId6;
  tool_name?: ToolName7;
  qa_pairs?: QaPairs;
}
/**
 * All clarification questions sent at once for batch answering (WS clients only).
 *
 * Emitted before the QueueChannel loop so WS clients can display all
 * questions simultaneously in a tabbed panel and let the user answer
 * in any order.
 */
export interface ClarificationBatchEvent {
  type?: EventType16;
  timestamp?: Timestamp16;
  agent_id?: AgentId14;
  request_id?: RequestId7;
  tool_name?: ToolName8;
  context?: Context;
  questions?: Questions;
}
/**
 * Client responds with all answers at once (WS batch mode).
 */
export interface ClarificationBatchResponseEvent {
  type?: EventType17;
  timestamp?: Timestamp17;
  request_id?: RequestId8;
  answers?: Answers;
}
/**
 * Reference selection has been requested.
 *
 * Sent when the model calls selectReferences and the user needs to choose
 * which references to include.
 */
export interface ReferenceSelectionRequestedEvent {
  type?: EventType18;
  timestamp?: Timestamp18;
  agent_id?: AgentId15;
  request_id?: RequestId9;
  tool_name?: ToolName9;
  prompt_lines?: PromptLines1;
}
/**
 * Reference selection has been completed.
 */
export interface ReferenceSelectionResolvedEvent {
  type?: EventType19;
  timestamp?: Timestamp19;
  agent_id?: AgentId16;
  request_id?: RequestId10;
  tool_name?: ToolName10;
  selected_ids?: SelectedIds;
}
/**
 * Respond to a reference selection request.
 */
export interface ReferenceSelectionResponseRequest {
  type?: EventType20;
  timestamp?: Timestamp20;
  request_id?: RequestId11;
  response?: Response;
}
/**
 * Workspace mismatch detected when attaching to a session.
 *
 * Sent when a client tries to attach to a session that was created
 * with a different workspace path. The client must choose to either
 * switch to the session's workspace or create a new session.
 */
export interface WorkspaceMismatchRequestedEvent {
  type?: EventType21;
  timestamp?: Timestamp21;
  request_id?: RequestId12;
  session_id?: SessionId;
  session_workspace?: SessionWorkspace;
  client_workspace?: ClientWorkspace;
  response_options?: ResponseOptions2;
  prompt_lines?: PromptLines2;
}
/**
 * Workspace mismatch has been resolved.
 */
export interface WorkspaceMismatchResolvedEvent {
  type?: EventType22;
  timestamp?: Timestamp22;
  request_id?: RequestId13;
  session_id?: SessionId1;
  action?: Action;
  new_session_id?: NewSessionId;
}
/**
 * Respond to a workspace mismatch request.
 */
export interface WorkspaceMismatchResponseRequest {
  type?: EventType23;
  timestamp?: Timestamp23;
  request_id?: RequestId14;
  response?: Response1;
}
/**
 * Offer session setup after successful authentication.
 *
 * Emitted by daemon after an auth command succeeds. The client renders a
 * multi-step wizard and sends back a single PostAuthSetupResponse.
 */
export interface PostAuthSetupEvent {
  type?: EventType24;
  timestamp?: Timestamp24;
  request_id?: RequestId15;
  provider_name?: ProviderName;
  provider_display_name?: ProviderDisplayName;
  available_models?: AvailableModels;
  has_active_session?: HasActiveSession;
  current_provider?: CurrentProvider;
  current_model?: CurrentModel;
  workspace_path?: WorkspacePath;
}
/**
 * User's response to post-auth session setup prompt.
 */
export interface PostAuthSetupResponse {
  type?: EventType25;
  timestamp?: Timestamp25;
  request_id?: RequestId16;
  connect?: Connect;
  model_name?: ModelName;
  persist_env?: PersistEnv;
}
/**
 * Plan has been created or updated.
 */
export interface PlanUpdatedEvent {
  type?: EventType26;
  timestamp?: Timestamp26;
  agent_id?: AgentId17;
  plan_name?: PlanName;
  steps?: Steps;
}
/**
 * Single step status change within a plan.
 *
 * Lean delta event — carries only the changed step's data, not the
 * full plan snapshot. The client maintains local plan state and applies
 * this delta to update the specific step.
 *
 * Sent for status-only changes (started, completed, failed, skipped,
 * blocked, unblocked). Structural changes (plan created, steps added,
 * plan completed) use ``PlanUpdatedEvent`` with the full snapshot.
 */
export interface PlanStepUpdatedEvent {
  type?: EventType27;
  timestamp?: Timestamp27;
  agent_id?: AgentId18;
  step_id?: StepId;
  sequence?: Sequence;
  content?: Content;
  status?: Status1;
  result?: Result;
  error?: Error2;
  blocked_by?: BlockedBy;
  depends_on?: DependsOn;
  received_outputs?: ReceivedOutputs;
}
/**
 * Plan has been cleared/completed.
 */
export interface PlanClearedEvent {
  type?: EventType28;
  timestamp?: Timestamp28;
  agent_id?: AgentId19;
}
/**
 * Context window usage has changed.
 */
export interface ContextUpdatedEvent {
  type?: EventType29;
  timestamp?: Timestamp29;
  agent_id?: AgentId20;
  total_tokens?: TotalTokens;
  prompt_tokens?: PromptTokens;
  output_tokens?: OutputTokens;
  context_limit?: ContextLimit;
  percent_used?: PercentUsed;
  tokens_remaining?: TokensRemaining;
  turns?: Turns;
  gc_threshold?: GcThreshold;
  gc_strategy?: GcStrategy;
  gc_target_percent?: GcTargetPercent;
  gc_continuous_mode?: GcContinuousMode;
}
/**
 * Instruction budget has been updated.
 *
 * Provides detailed breakdown of token usage by instruction source layer.
 * Sent after session configuration and when budget changes significantly.
 *
 * The budget_snapshot contains:
 * - session_id, agent_id, agent_type: Identity
 * - context_limit, total_tokens, utilization_percent: Overall usage
 * - gc_eligible_tokens, locked_tokens, preservable_tokens: GC info
 * - entries: Per-source breakdown (system, session, plugin, enrichment, conversation)
 */
export interface InstructionBudgetEvent {
  type?: EventType30;
  timestamp?: Timestamp30;
  agent_id?: AgentId21;
  budget_snapshot?: BudgetSnapshot;
}
export interface BudgetSnapshot {
  [k: string]: unknown;
}
/**
 * A conversation turn has completed.
 *
 * Includes optional prompt-cache token fields so clients (e.g. TUI) can
 * display cache hit rates.  ``cache_read_tokens`` is the number of prompt
 * tokens served from cache (90% cost saving on Anthropic);
 * ``cache_creation_tokens`` is the number written *into* the cache (1.25x
 * write premium on Anthropic).  Both are ``None`` when the provider does
 * not support prompt caching.
 */
export interface TurnCompletedEvent {
  type?: EventType31;
  timestamp?: Timestamp31;
  agent_id?: AgentId22;
  turn_number?: TurnNumber;
  prompt_tokens?: PromptTokens1;
  output_tokens?: OutputTokens1;
  total_tokens?: TotalTokens1;
  duration_seconds?: DurationSeconds1;
  function_calls?: FunctionCalls;
  formatted_text?: FormattedText;
  cache_read_tokens?: CacheReadTokens;
  cache_creation_tokens?: CacheCreationTokens;
}
/**
 * Incremental progress during turn execution.
 *
 * Emitted after each model response within a turn, enabling real-time
 * token tracking before the turn completes.  Includes optional cache
 * token fields mirroring ``TurnCompletedEvent``.
 */
export interface TurnProgressEvent {
  type?: EventType32;
  timestamp?: Timestamp32;
  agent_id?: AgentId23;
  total_tokens?: TotalTokens2;
  prompt_tokens?: PromptTokens2;
  output_tokens?: OutputTokens2;
  context_limit?: ContextLimit1;
  percent_used?: PercentUsed1;
  tokens_remaining?: TokensRemaining1;
  pending_tool_calls?: PendingToolCalls;
  cache_read_tokens?: CacheReadTokens1;
  cache_creation_tokens?: CacheCreationTokens1;
}
/**
 * System message (info, warning, status).
 */
export interface SystemMessageEvent {
  type?: EventType33;
  timestamp?: Timestamp33;
  message?: Message;
  style?: Style;
}
/**
 * Detailed help text for commands.
 *
 * Sent in response to 'help' subcommands to display formatted help
 * using the pager. Each line is a (text, style) tuple.
 */
export interface HelpTextEvent {
  type?: EventType34;
  timestamp?: Timestamp34;
  lines?: Lines;
}
/**
 * Initialization progress update.
 *
 * Sent during session initialization to show progress on each step.
 * Steps are shown in sequence with their status.
 */
export interface InitProgressEvent {
  type?: EventType35;
  timestamp?: Timestamp35;
  step?: Step;
  status?: Status2;
  message?: Message1;
  step_number?: StepNumber;
  total_steps?: TotalSteps;
}
/**
 * Error occurred.
 */
export interface ErrorEvent {
  type?: EventType36;
  timestamp?: Timestamp36;
  error?: Error3;
  error_type?: ErrorType;
  recoverable?: Recoverable;
}
/**
 * API retry notification with exponential backoff.
 *
 * Sent when a transient error (rate limit, server error) is encountered
 * and the system is retrying the request.
 */
export interface RetryEvent {
  type?: EventType37;
  timestamp?: Timestamp37;
  message?: Message2;
  attempt?: Attempt;
  max_attempts?: MaxAttempts;
  delay?: Delay;
  error_type?: ErrorType1;
}
/**
 * List of available sessions - for user display.
 */
export interface SessionListEvent {
  type?: EventType38;
  timestamp?: Timestamp38;
  sessions?: Sessions;
}
/**
 * Session state snapshot - sent on connect/attach with all data client needs.
 *
 * Includes current session info plus lists for completion/display:
 * - sessions: All available sessions (for session commands)
 * - tools: All available tools with enabled status (for tools commands)
 * - models: Available model names (for model command)
 *
 * Client stores this locally and uses it for both completion and display.
 * Server pushes updates when state changes.
 */
export interface SessionInfoEvent {
  type?: EventType39;
  timestamp?: Timestamp39;
  session_id?: SessionId2;
  session_name?: SessionName;
  model_provider?: ModelProvider;
  model_name?: ModelName1;
  profile_name?: ProfileName1;
  sessions?: Sessions1;
  tools?: Tools;
  models?: Models;
  user_inputs?: UserInputs;
  memories?: Memories;
  sandbox_paths?: SandboxPaths;
  services?: Services;
  tool_id_mappings?: ToolIdMappings;
}
export interface ToolIdMappings {
  [k: string]: string;
}
/**
 * List of available memories - for completion cache and pager display.
 */
export interface MemoryListEvent {
  type?: EventType40;
  timestamp?: Timestamp40;
  memories?: Memories1;
}
/**
 * List of sandbox-allowed paths - for @@ completion cache.
 *
 * Emitted after sandbox add/remove commands to refresh the client's
 * completion list for @@ (sandbox path) references.
 */
export interface SandboxPathsEvent {
  type?: EventType41;
  timestamp?: Timestamp41;
  paths?: Paths;
}
/**
 * List of discovered services - for completion cache.
 *
 * Emitted after services commands to refresh the client's
 * completion list for service names and HTTP methods.
 */
export interface ServiceListEvent {
  type?: EventType42;
  timestamp?: Timestamp42;
  services?: Services1;
}
/**
 * Session description was updated (by model calling session_describe).
 */
export interface SessionDescriptionUpdatedEvent {
  type?: EventType43;
  timestamp?: Timestamp43;
  session_id?: SessionId3;
  description?: Description;
}
/**
 * List of available agent profiles for session creation.
 *
 * Sent in response to a ``session.profiles`` command. Each profile
 * is a summary dict containing the profile's name, description,
 * model, provider, and list of plugins — enough for a client to
 * display a profile picker.
 */
export interface SessionProfilesEvent {
  type?: EventType44;
  timestamp?: Timestamp44;
  profiles?: Profiles;
}
/**
 * Send a message to the model.
 */
export interface SendMessageRequest {
  type?: EventType45;
  timestamp?: Timestamp45;
  text?: Text1;
  attachments?: Attachments;
  parallel_tools?: ParallelTools;
}
/**
 * Respond to a permission request.
 */
export interface PermissionResponseRequest {
  type?: EventType46;
  timestamp?: Timestamp46;
  request_id?: RequestId17;
  response?: Response2;
  edited_arguments?: EditedArguments;
}
/**
 * Respond to a clarification question.
 */
export interface ClarificationResponseRequest {
  type?: EventType47;
  timestamp?: Timestamp47;
  request_id?: RequestId18;
  question_index?: QuestionIndex2;
  response?: Response3;
}
/**
 * Stop current operation (cancel generation).
 */
export interface StopRequest {
  type?: EventType48;
  timestamp?: Timestamp48;
  agent_id?: AgentId24;
}
/**
 * External event injected by the host page via the web component.
 *
 * Published on the session's ``EventBus`` as an ``external_event``
 * so that agents subscribed via ``subscribeToEvents`` are notified.
 */
export interface ExternalEventRequest {
  type?: EventType49;
  timestamp?: Timestamp49;
  name?: Name;
  data?: Data;
}
export interface Data {
  [k: string]: unknown;
}
/**
 * Notification that an agent has subscribed to external events.
 *
 * Sent to WS clients so the host page knows which external event
 * names the agent is listening for.  ``["*"]`` means all external
 * events.
 */
export interface EventsSubscribedEvent {
  type?: EventType50;
  timestamp?: Timestamp50;
  agent_id?: AgentId25;
  event_names?: EventNames;
}
/**
 * Execute a command (like 'model', 'save', 'resume', etc.).
 */
export interface CommandRequest {
  type?: EventType51;
  timestamp?: Timestamp51;
  command?: Command;
  args?: Args;
}
/**
 * Request current instruction budget for an agent.
 *
 * Server responds with InstructionBudgetEvent containing the budget snapshot.
 * If agent_id is None or empty, returns budget for main agent.
 */
export interface GetInstructionBudgetRequest {
  type?: EventType52;
  timestamp?: Timestamp52;
  agent_id?: AgentId26;
}
/**
 * Request list of available commands from server.
 */
export interface CommandListRequest {
  type?: EventType53;
  timestamp?: Timestamp53;
}
/**
 * List of available commands from server/plugins.
 */
export interface CommandListEvent {
  type?: EventType54;
  timestamp?: Timestamp54;
  commands?: Commands;
}
/**
 * Signal that the command list should be refreshed.
 *
 * Emitted by core.py after commands that change completion state
 * (e.g., references select/unselect). The IPC client handles this
 * by re-requesting the full command list from the daemon.
 */
export interface CommandListRefreshEvent {
  type?: EventType55;
  timestamp?: Timestamp55;
}
/**
 * Tool status information for client display.
 */
export interface ToolStatusEvent {
  type?: EventType56;
  timestamp?: Timestamp56;
  tools?: Tools1;
  message?: Message3;
}
/**
 * Hash-derived ID → human-readable name mapping for client display.
 *
 * Sent after tool configuration and when deferred tools are activated.
 * Clients use this to resolve opaque tool/category IDs in tool arguments
 * and model output without pattern-matching or reverse engineering.
 *
 * The mapping is cumulative — each event carries the full current set,
 * not a delta. Clients should replace their local lookup on each receive.
 */
export interface ToolIdRegistryEvent {
  type?: EventType57;
  timestamp?: Timestamp57;
  mappings?: Mappings;
}
export interface Mappings {
  [k: string]: string;
}
/**
 * Client request to disable a tool.
 *
 * Directly calls registry.disable_tool() without generating response events.
 * Used by headless mode to disable tools before starting event handling.
 */
export interface ToolDisableRequest {
  type?: EventType58;
  timestamp?: Timestamp58;
  tool_name?: ToolName11;
}
/**
 * Register client-side tools that the browser/frontend can execute.
 *
 * The server creates proxy tools in the session's registry. When the model
 * calls one, the server routes execution to the WS client via
 * ``tool.execute_request`` and waits for ``tool.execute_result``.
 */
export interface ToolsRegisterClientRequest {
  type?: EventType59;
  timestamp?: Timestamp59;
  tools?: Tools2;
  categories?: Categories;
}
export interface Categories {
  [k: string]: string;
}
/**
 * Server requests the WS client to execute a client-registered tool.
 */
export interface ToolExecuteRequestEvent {
  type?: EventType60;
  timestamp?: Timestamp60;
  call_id?: CallId4;
  agent_id?: AgentId27;
  tool_name?: ToolName12;
  tool_args?: ToolArgs3;
}
export interface ToolArgs3 {
  [k: string]: unknown;
}
/**
 * Client returns the result of a client-side tool execution.
 */
export interface ToolExecuteResultEvent {
  type?: EventType61;
  timestamp?: Timestamp61;
  call_id?: CallId5;
  result?: Result1;
  error?: Error4;
}
/**
 * Client request for conversation history.
 */
export interface HistoryRequest {
  type?: EventType62;
  timestamp?: Timestamp62;
  agent_id?: AgentId28;
}
/**
 * Conversation history from server.
 */
export interface HistoryEvent {
  type?: EventType63;
  timestamp?: Timestamp63;
  agent_id?: AgentId29;
  history?: History;
  turn_accounting?: TurnAccounting;
}
/**
 * Client sends its configuration to the server.
 *
 * Sent after connection to apply client-specific settings like trace paths
 * and display capabilities.  The ``presentation`` dict is deserialized into
 * a ``PresentationContext`` on the server side.
 */
export interface ClientConfigRequest {
  type?: EventType64;
  timestamp?: Timestamp64;
  trace_log_path?: TraceLogPath;
  provider_trace_log?: ProviderTraceLog;
  working_dir?: WorkingDir;
  env_file?: EnvFile;
  presentation?: Presentation;
  permission_timeout?: PermissionTimeout;
}
/**
 * Sent when a user prompt is queued during model processing.
 *
 * Instead of returning an error when the user sends a message while the model
 * is running, the message is queued and will be injected at the next natural
 * pause point (between tool executions, after subagent completion, etc.).
 */
export interface MidTurnPromptQueuedEvent {
  type?: EventType65;
  timestamp?: Timestamp65;
  text?: Text2;
  position_in_queue?: PositionInQueue;
}
/**
 * Sent when a queued prompt is injected into the conversation.
 *
 * This notifies the client that the queued prompt is now being processed
 * by the model.
 */
export interface MidTurnPromptInjectedEvent {
  type?: EventType66;
  timestamp?: Timestamp66;
  text?: Text3;
}
/**
 * Sent when streaming is interrupted to process a mid-turn user prompt.
 *
 * This notifies the client that the model's current generation was interrupted
 * because a user prompt arrived and needs to be processed immediately.
 * The partial response is preserved and the user's prompt is being processed.
 */
export interface MidTurnInterruptEvent {
  type?: EventType67;
  timestamp?: Timestamp67;
  partial_response_chars?: PartialResponseChars;
  user_prompt_preview?: UserPromptPreview;
}
/**
 * Sent when the server recovers from an interrupted turn after reconnection.
 *
 * This event notifies the client that a turn was interrupted (e.g., by server
 * restart) and has been recovered with synthetic error responses injected
 * for any pending tool calls.
 */
export interface InterruptedTurnRecoveredEvent {
  type?: EventType68;
  timestamp?: Timestamp68;
  session_id?: SessionId4;
  agent_id?: AgentId30;
  recovered_calls?: RecoveredCalls;
  action_taken?: ActionTaken;
}
/**
 * Client requests list of available workspaces.
 */
export interface WorkspaceListRequest {
  type?: EventType69;
  timestamp?: Timestamp69;
}
/**
 * Response to workspace.list - list of available workspaces.
 */
export interface WorkspaceListEvent {
  type?: EventType70;
  timestamp?: Timestamp70;
  root?: Root;
  workspaces?: Workspaces;
}
/**
 * Client requests creation of a new workspace.
 */
export interface WorkspaceCreateRequest {
  type?: EventType71;
  timestamp?: Timestamp71;
  name?: Name1;
}
/**
 * Response to workspace.create - new workspace created.
 */
export interface WorkspaceCreatedEvent {
  type?: EventType72;
  timestamp?: Timestamp72;
  name?: Name2;
  path?: Path;
}
/**
 * Client selects a workspace to use for the session.
 */
export interface WorkspaceSelectRequest {
  type?: EventType73;
  timestamp?: Timestamp73;
  name?: Name3;
}
/**
 * Response to workspace.select - configuration status of selected workspace.
 */
export interface ConfigStatusEvent {
  type?: EventType74;
  timestamp?: Timestamp74;
  workspace?: Workspace;
  configured?: Configured;
  provider?: Provider;
  model?: Model;
  available_providers?: AvailableProviders;
  missing_fields?: MissingFields;
}
/**
 * Client updates workspace configuration (provider, model, API key).
 */
export interface ConfigUpdateRequest {
  type?: EventType75;
  timestamp?: Timestamp75;
  provider?: Provider1;
  model?: Model1;
  api_key?: ApiKey;
}
/**
 * Response to config.update - configuration was updated.
 */
export interface ConfigUpdatedEvent {
  type?: EventType76;
  timestamp?: Timestamp76;
  workspace?: Workspace1;
  provider?: Provider2;
  model?: Model2;
  success?: Success2;
  error?: Error5;
}
/**
 * Incremental workspace file change notification.
 *
 * Emitted in real-time (debounced) whenever files in the workspace are
 * created, modified, or deleted during the session.  Each entry carries
 * a ``status`` indicating the nature of the change relative to the
 * session baseline.
 *
 * Statuses:
 *     ``"created"``  – file did not exist at session start.
 *     ``"modified"`` – file existed at session start and was changed.
 *     ``"deleted"``  – file was previously tracked and is now gone.
 */
export interface WorkspaceFilesChangedEvent {
  type?: EventType77;
  timestamp?: Timestamp77;
  changes?: Changes;
}
/**
 * Complete workspace file state snapshot.
 *
 * Sent on client reconnect / initial attach so the client can rebuild
 * its local mirror of the session's file tracking state without
 * replaying individual deltas.
 */
export interface WorkspaceFilesSnapshotEvent {
  type?: EventType78;
  timestamp?: Timestamp78;
  files?: Files;
  total?: Total;
}
/**
 * Stage files into a workspace via a multi-frame WS protocol.
 *
 * **Wire protocol:**
 *
 * 1. Client sends *this* event as one TEXT WS frame.  ``files``
 *    declares the names, sizes, and (optional) content types of the
 *    payloads that will follow.
 * 2. Client immediately sends ``len(files)`` raw BINARY WS frames in
 *    the **same order** as ``files``.  Each frame's byte length must
 *    equal the corresponding ``files[i].size``.
 * 3. Server responds with a TEXT frame carrying
 *    :class:`StageFilesEvent` summarising what was written.
 *
 * The handler reads the binary frames inline (the per-connection
 * receive loop preserves frame order, so other event types cannot
 * interleave between the request and its blobs).
 *
 * ``workspace_id`` identifies the target workspace.  Clients learn
 * valid IDs from :class:`WorkspaceCreatedEvent` and
 * :class:`SessionInfoEvent`.  An empty value targets the connection's
 * currently-selected workspace (the WS server tracks one
 * selected workspace per client).
 *
 * Caps (server-enforced, configurable per deployment):
 *
 * - Per-file ``size`` cap (default 10 MB)
 * - Sum of ``size`` values cap (default 50 MB)
 *
 * Limits exceeded → server emits a :class:`StageFilesEvent` with the
 * failure recorded; no binary frames are read.  Clients should check
 * the response before considering files staged.
 *
 * Compared to the legacy ``staged_files`` field on the
 * ``session.new`` envelope (kept for premium back-compat), this is
 * the canonical SDK primitive: workspace-scoped (not session-scoped),
 * binary-framed (no base64 inflation), and supports staging into an
 * already-existing workspace mid-session.
 */
export interface StageFilesRequest {
  type?: EventType79;
  timestamp?: Timestamp79;
  workspace_id?: WorkspaceId;
  files?: Files1;
}
/**
 * Per-file metadata sent inside a :class:`StageFilesRequest`.
 *
 * The ``size`` is the exact byte length of the binary frame that will
 * follow for this file. The server validates the matching binary
 * frame's length against this value; a mismatch is fatal for the
 * whole staging operation.
 *
 * ``content_type`` is informational — the server uses it for logging
 * and may reflect it in the response. It does not gate writing.
 *
 * ``mode`` is reserved for future POSIX permission bits (e.g. 0o755
 * for executables in the workspace). Currently ignored by the server;
 * files are written with the daemon's umask. Add the field now so
 * clients don't need a protocol bump later.
 */
export interface StagedFileSpec {
  name?: Name4;
  size?: Size;
  content_type?: ContentType;
  mode?: Mode1;
}
/**
 * Server's response to :class:`StageFilesRequest`.
 *
 * Reports which files were written (by ``name``, in declared order)
 * and which failed.  Failures are surfaced per-file so clients can
 * retry just the ones that didn't make it.
 *
 * Possible per-file error categories:
 *
 * - ``"unsafe_path"`` — name is absolute, contains ``..``, or escapes
 *   the workspace root.
 * - ``"size_mismatch"`` — declared ``size`` did not match the binary
 *   frame length.
 * - ``"size_limit_per_file"`` — file exceeded the per-file cap.
 * - ``"size_limit_total"`` — sum of declared sizes exceeded the
 *   total payload cap.
 * - ``"workspace_not_found"`` — ``workspace_id`` doesn't match a
 *   known workspace for this client.
 * - ``"io_error"`` — write failed (disk full, permission denied,
 *   AppArmor refusal, ...).  ``error`` carries the OS message.
 */
export interface StageFilesEvent {
  type?: EventType80;
  timestamp?: Timestamp80;
  workspace_id?: WorkspaceId1;
  staged?: Staged;
  failed?: Failed;
}
/**
 * Heartbeat sent between peer servers at a configurable interval.
 *
 * Contains server identity, workload metrics, and health data used by the
 * PeerRegistry to track peer liveness and by the environment aspect (Phase 2)
 * to expose cluster state to the model.
 */
export interface PeerHeartbeatEvent {
  type?: EventType81;
  timestamp?: Timestamp81;
  server_id?: ServerId;
  server_name?: ServerName;
  server_version?: ServerVersion;
  active_sessions?: ActiveSessions;
  active_agents?: ActiveAgents;
  available_providers?: AvailableProviders1;
  available_models?: AvailableModels1;
  tags?: Tags;
  cpu_percent?: CpuPercent;
  memory_percent?: MemoryPercent;
  uptime_seconds?: UptimeSeconds;
  trust_state?: TrustState;
  success_rate_1h?: SuccessRate1H;
  escalated_tools?: EscalatedTools;
}
/**
 * Request to spawn a subagent on a remote peer server.
 *
 * Sent from the origin server (where the model called
 * ``spawn_subagent(server=...)``) to the remote peer that should execute
 * the subagent.  The ``request_id`` correlates all subsequent events in
 * this spawn lifecycle.
 */
export interface PeerSpawnRequestEvent {
  type?: EventType82;
  timestamp?: Timestamp82;
  request_id?: RequestId19;
  origin_server?: OriginServer;
  agent_name?: AgentName1;
  task?: Task;
  context?: Context1;
  profile_json?: ProfileJson;
  inline_config_json?: InlineConfigJson;
  workspace_git_url?: WorkspaceGitUrl;
  workspace_branch?: WorkspaceBranch;
  workspace_commit?: WorkspaceCommit;
  workspace_temp_branch?: WorkspaceTempBranch;
}
/**
 * Confirmation that a remote peer accepted the spawn request.
 *
 * Sent back to the origin server once the remote has created the
 * ephemeral session and is about to start processing.
 */
export interface PeerSpawnAcceptedEvent {
  type?: EventType83;
  timestamp?: Timestamp83;
  request_id?: RequestId20;
  remote_agent_id?: RemoteAgentId;
}
/**
 * Notification that a remote peer rejected the spawn request.
 *
 * The ``reason`` field contains a human-readable explanation (e.g.
 * capacity limits, missing provider, unknown profile).
 */
export interface PeerSpawnRejectedEvent {
  type?: EventType84;
  timestamp?: Timestamp84;
  request_id?: RequestId21;
  reason?: Reason;
}
/**
 * Streamed output chunk from a remote subagent.
 *
 * Sent from the remote server back to the origin as the subagent
 * produces output.  The origin's ``RemoteSpawnHandler`` forwards these
 * to the parent session via ``inject_prompt``.
 */
export interface PeerAgentOutputEvent {
  type?: EventType85;
  timestamp?: Timestamp85;
  request_id?: RequestId22;
  remote_agent_id?: RemoteAgentId1;
  text?: Text4;
  source?: Source1;
}
/**
 * Signal that a remote subagent has finished execution.
 *
 * ``success`` indicates whether the subagent completed normally.
 * ``summary`` contains a brief result description; ``error`` is
 * populated only when ``success`` is False.
 */
export interface PeerAgentCompletedEvent {
  type?: EventType86;
  timestamp?: Timestamp86;
  request_id?: RequestId23;
  remote_agent_id?: RemoteAgentId2;
  success?: Success3;
  summary?: Summary;
  error?: Error6;
  workspace_modified?: WorkspaceModified;
}
/**
 * Request to cancel a running remote subagent.
 *
 * Sent from the origin server when the parent session wants to stop
 * a previously spawned remote subagent.
 */
export interface PeerStopRequestEvent {
  type?: EventType87;
  timestamp?: Timestamp87;
  request_id?: RequestId24;
  remote_agent_id?: RemoteAgentId3;
}
/**
 * Confirmation that a remote peer received and processed the stop request.
 */
export interface PeerStopAcknowledgedEvent {
  type?: EventType88;
  timestamp?: Timestamp88;
  request_id?: RequestId25;
  remote_agent_id?: RemoteAgentId4;
}
/**
 * Inject a prompt into a session's message queue.
 *
 * Maps to :meth:`JaatoSession.inject_prompt`.  ``source_type``
 * selects the queue priority:
 *
 * * ``"user"`` — USER priority (mid-turn "steer", interrupts the
 *   model at the next safe point).
 * * ``"child"`` — CHILD priority (queued behind in-flight work; runs
 *   when the agent would otherwise stop, the "follow-up" pattern).
 * * ``"system"`` / ``"event"`` / ``"parent"`` — other priority
 *   tiers from :class:`SourceType` for reactor / hook callers.
 *
 * Single verb covers both pi-agent's ``steer`` and ``followUp``
 * patterns via the priority dimension.
 */
export interface InjectPromptRequest {
  type?: EventType89;
  timestamp?: Timestamp89;
  text?: Text5;
  source_type?: SourceType;
  source_id?: SourceId;
}
/**
 * Re-run the model loop against an explicit message list.
 *
 * Maps to :meth:`JaatoSession.replay_messages`.  When ``messages``
 * is omitted, replays the session's current ``get_history()`` —
 * semantically equivalent to "continue from the current state with
 * no new user input" (pi-agent's ``continue()`` shape).
 *
 * Acquires exclusive provider access internally so concurrent
 * in-flight turn calls are serialised.  Does NOT mutate session
 * history or turn accounting.  Use when you want a one-shot
 * completion against an arbitrary message list — fork/interrogate
 * flows compose this with ``resolve_fork_point``.
 */
export interface ReplayMessagesRequest {
  type?: EventType90;
  timestamp?: Timestamp90;
  request_id?: RequestId26;
  messages?: Messages;
  timeout_seconds?: TimeoutSeconds;
}
/**
 * Server's response to :class:`ReplayMessagesRequest`.
 */
export interface ReplayMessagesResultEvent {
  type?: EventType91;
  timestamp?: Timestamp91;
  request_id?: RequestId27;
  response_text?: ResponseText;
  error?: Error7;
}
/**
 * Resolve a fork point in the session's history to a message index.
 *
 * Maps to :meth:`JaatoSession.resolve_fork_point`.  Exactly one of
 * ``after_message`` / ``after_tool_call`` / ``after_timestamp``
 * should be supplied; if none are given, the server returns the
 * last message index (full-history fork).  The session's current
 * ``get_history()`` is used as the search space — clients don't
 * pass history over the wire.
 *
 * Composes with :class:`ReplayMessagesRequest` so a client can
 * resolve a fork point, snapshot history up to that point, edit,
 * and replay — the same shape premium's ``interrogate_session``
 * tool uses internally.
 */
export interface ResolveForkPointRequest {
  type?: EventType92;
  timestamp?: Timestamp92;
  request_id?: RequestId28;
  after_message?: AfterMessage;
  after_tool_call?: AfterToolCall;
  after_timestamp?: AfterTimestamp;
}
/**
 * Server's response to :class:`ResolveForkPointRequest`.
 */
export interface ResolveForkPointResultEvent {
  type?: EventType93;
  timestamp?: Timestamp93;
  request_id?: RequestId29;
  fork_index?: ForkIndex;
  error?: Error8;
}
/**
 * Add tools / patterns to the session's permission whitelist.
 *
 * Maps to :meth:`PermissionPlugin.add_whitelist_tools` for tools
 * and :meth:`PermissionPolicy.add_session_whitelist` for patterns.
 * Tools and patterns can be supplied together — both lists are
 * additive.
 */
export interface PermissionAddWhitelistRequest {
  type?: EventType94;
  timestamp?: Timestamp94;
  tools?: Tools3;
  patterns?: Patterns;
}
/**
 * Add tools / patterns to the session's permission blacklist.
 *
 * Maps to :meth:`PermissionPolicy.add_session_blacklist` for both
 * tools and patterns.  Tools and patterns supplied together —
 * both lists are additive.
 */
export interface PermissionAddBlacklistRequest {
  type?: EventType95;
  timestamp?: Timestamp95;
  tools?: Tools4;
  patterns?: Patterns1;
}
/**
 * Remove tools / patterns from a permission list.
 *
 * ``target`` selects which list: ``"whitelist"`` or
 * ``"blacklist"``.  Empty lists are no-ops.
 */
export interface PermissionRemoveRequest {
  type?: EventType96;
  timestamp?: Timestamp96;
  target?: Target;
  tools?: Tools5;
  patterns?: Patterns2;
}
/**
 * Clear the session-level permission lists.
 *
 * ``target`` selects which list to clear: ``"whitelist"``,
 * ``"blacklist"``, or ``"all"`` (clears both).  Does NOT affect
 * the base policy declared in ``permissions.json``; only the
 * session-level overrides.
 */
export interface PermissionClearRequest {
  type?: EventType97;
  timestamp?: Timestamp97;
  target?: Target1;
}
/**
 * Set the session-level default permission policy.
 *
 * ``policy`` is one of ``"allow"`` | ``"deny"`` | ``"ask"``.
 * Maps to ``PermissionPolicy.session_default_policy`` — overrides
 * the base default for this session only.
 */
export interface PermissionSetDefaultRequest {
  type?: EventType98;
  timestamp?: Timestamp98;
  policy?: Policy;
}
/**
 * Request a structured snapshot of the current permission policy.
 */
export interface PermissionPolicySnapshotRequest {
  type?: EventType99;
  timestamp?: Timestamp99;
  request_id?: RequestId30;
}
/**
 * Structured permission policy snapshot.
 *
 * Returned in response to :class:`PermissionPolicySnapshotRequest`.
 * Carries the full policy state — base policy + session overrides
 * — so clients can build introspection UIs without going through
 * the stringly-typed ``permissions check`` command.
 */
export interface PermissionPolicySnapshotEvent {
  type?: EventType100;
  timestamp?: Timestamp100;
  request_id?: RequestId31;
  default_policy?: DefaultPolicy;
  session_default_policy?: SessionDefaultPolicy;
  whitelist_tools?: WhitelistTools;
  whitelist_patterns?: WhitelistPatterns;
  blacklist_tools?: BlacklistTools;
  blacklist_patterns?: BlacklistPatterns;
  session_whitelist?: SessionWhitelist;
  session_blacklist?: SessionBlacklist;
}

/**
 * Discriminated union of every wire-protocol event.
 *
 * Use the ``type`` field (an ``EventType`` value) to narrow.
 * Generated from the union of every Event subclass declared in
 * ``jaato-sdk/jaato_sdk/events.py``; mirrors the Python-side
 * dispatch table in ``_EVENT_CLASSES``.
 */
export type JaatoEvent = JaatoEvents;

/**
 * Runtime const mirror of the Python ``EventType`` enum.
 *
 * Use these for constructing events:
 * ``{ type: EventTypeValue.SEND_MESSAGE, text: "hi" }``.
 * The ``EventType`` type alias above is the type-side mirror; this
 * const provides the runtime values.  Both are generated from
 * ``jaato-sdk/jaato_sdk/events.py`` to stay in lockstep.
 */
export const EventTypeValue = {
  CONNECTED: "connected",
  DISCONNECTED: "disconnected",
  AGENT_CREATED: "agent.created",
  AGENT_OUTPUT: "agent.output",
  AGENT_STATUS_CHANGED: "agent.status_changed",
  AGENT_COMPLETED: "agent.completed",
  TOOL_CALL_START: "tool.call_start",
  TOOL_CALL_END: "tool.call_end",
  TOOL_OUTPUT: "tool.output",
  PERMISSION_REQUESTED: "permission.requested",
  PERMISSION_INPUT_MODE: "permission.input_mode",
  PERMISSION_RESOLVED: "permission.resolved",
  PERMISSION_RESPONSE: "permission.response",
  PERMISSION_STATUS: "permission.status",
  CLARIFICATION_REQUESTED: "clarification.requested",
  CLARIFICATION_INPUT_MODE: "clarification.input_mode",
  CLARIFICATION_QUESTION: "clarification.question",
  CLARIFICATION_RESOLVED: "clarification.resolved",
  CLARIFICATION_RESPONSE: "clarification.response",
  CLARIFICATION_BATCH: "clarification.batch",
  CLARIFICATION_BATCH_RESPONSE: "clarification.batch_response",
  REFERENCE_SELECTION_REQUESTED: "reference_selection.requested",
  REFERENCE_SELECTION_RESOLVED: "reference_selection.resolved",
  REFERENCE_SELECTION_RESPONSE: "reference_selection.response",
  WORKSPACE_MISMATCH_REQUESTED: "workspace_mismatch.requested",
  WORKSPACE_MISMATCH_RESOLVED: "workspace_mismatch.resolved",
  WORKSPACE_MISMATCH_RESPONSE: "workspace_mismatch.response",
  PLAN_UPDATED: "plan.updated",
  PLAN_STEP_UPDATED: "plan.step_updated",
  PLAN_CLEARED: "plan.cleared",
  CONTEXT_UPDATED: "context.updated",
  TURN_COMPLETED: "turn.completed",
  TURN_PROGRESS: "turn.progress",
  INSTRUCTION_BUDGET_UPDATED: "instruction_budget.updated",
  INSTRUCTION_BUDGET_REQUEST: "instruction_budget.request",
  SYSTEM_MESSAGE: "system.message",
  HELP_TEXT: "help.text",
  ERROR: "error",
  INIT_PROGRESS: "init.progress",
  RETRY: "retry",
  SESSION_LIST: "session.list",
  SESSION_INFO: "session.info",
  SESSION_DESCRIPTION_UPDATED: "session.description_updated",
  MEMORY_LIST: "memory.list",
  SANDBOX_PATHS: "sandbox.paths",
  SERVICE_LIST: "service.list",
  SEND_MESSAGE: "message.send",
  STOP: "session.stop",
  COMMAND: "command.execute",
  COMMAND_LIST_REQUEST: "command.list_request",
  COMMAND_LIST: "command.list",
  COMMAND_LIST_REFRESH: "command.list_refresh",
  TOOL_STATUS: "tools.status",
  TOOL_ID_REGISTRY: "tools.id_registry",
  TOOL_DISABLE_REQUEST: "tools.disable",
  TOOLS_REGISTER_CLIENT: "tools.register_client",
  TOOL_EXECUTE_REQUEST: "tool.execute_request",
  TOOL_EXECUTE_RESULT: "tool.execute_result",
  HISTORY_REQUEST: "history.request",
  HISTORY: "history",
  CLIENT_CONFIG: "client.config",
  MID_TURN_PROMPT_QUEUED: "mid_turn_prompt.queued",
  MID_TURN_PROMPT_INJECTED: "mid_turn_prompt.injected",
  MID_TURN_INTERRUPT: "mid_turn_prompt.interrupt",
  INTERRUPTED_TURN_RECOVERED: "session.interrupted_turn_recovered",
  POST_AUTH_SETUP: "auth.setup",
  POST_AUTH_SETUP_RESPONSE: "auth.setup_response",
  WORKSPACE_LIST_REQUEST: "workspace.list",
  WORKSPACE_LIST: "workspace.list_response",
  WORKSPACE_CREATE_REQUEST: "workspace.create",
  WORKSPACE_CREATED: "workspace.created",
  WORKSPACE_SELECT_REQUEST: "workspace.select",
  CONFIG_STATUS: "config.status",
  CONFIG_UPDATE_REQUEST: "config.update",
  CONFIG_UPDATED: "config.updated",
  WORKSPACE_FILES_STAGE_REQUEST: "workspace.files.stage_request",
  WORKSPACE_FILES_STAGED: "workspace.files.staged",
  SESSION_PROFILES: "session.profiles",
  WORKSPACE_FILES_CHANGED: "workspace.files_changed",
  WORKSPACE_FILES_SNAPSHOT: "workspace.files_snapshot",
  EVENT_EXTERNAL: "event.external",
  INJECT_PROMPT_REQUEST: "inject_prompt.request",
  REPLAY_MESSAGES_REQUEST: "replay_messages.request",
  REPLAY_MESSAGES_RESULT: "replay_messages.result",
  RESOLVE_FORK_POINT_REQUEST: "resolve_fork_point.request",
  RESOLVE_FORK_POINT_RESULT: "resolve_fork_point.result",
  PERMISSION_ADD_WHITELIST_REQUEST: "permission.add_whitelist",
  PERMISSION_ADD_BLACKLIST_REQUEST: "permission.add_blacklist",
  PERMISSION_REMOVE_REQUEST: "permission.remove",
  PERMISSION_CLEAR_REQUEST: "permission.clear",
  PERMISSION_SET_DEFAULT_REQUEST: "permission.set_default",
  PERMISSION_POLICY_SNAPSHOT_REQUEST: "permission.policy_snapshot.request",
  PERMISSION_POLICY_SNAPSHOT: "permission.policy_snapshot",
  EVENTS_SUBSCRIBED: "events.subscribed",
  PEER_HEARTBEAT: "peer.heartbeat",
  PEER_SPAWN_REQUEST: "peer.spawn_request",
  PEER_SPAWN_ACCEPTED: "peer.spawn_accepted",
  PEER_SPAWN_REJECTED: "peer.spawn_rejected",
  PEER_AGENT_OUTPUT: "peer.agent_output",
  PEER_AGENT_COMPLETED: "peer.agent_completed",
  PEER_STOP_REQUEST: "peer.stop_request",
  PEER_STOP_ACKNOWLEDGED: "peer.stop_acknowledged",
} as const;
