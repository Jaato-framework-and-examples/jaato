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
  | AgentErrorEvent
  | SessionTerminatedEvent
  | SessionRestoredEvent
  | SlotSettledEvent
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
  | GCConfigEvent
  | InstructionBudgetEvent
  | TurnCompletedEvent
  | TurnProgressEvent
  | SystemMessageEvent
  | HelpTextEvent
  | InitProgressEvent
  | ErrorEvent
  | RetryEvent
  | SessionListEvent
  | GCEvent
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
  | GateAnnouncedEvent
  | GateReleasedEvent
  | GatesSnapshotEvent
  | InjectPromptRequest
  | InjectPromptResultEvent
  | ReplayMessagesRequest
  | ReplayMessagesResultEvent
  | ResolveForkPointRequest
  | ResolveForkPointResultEvent
  | WakeBindResultEvent
  | SessionWokenEvent
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp = string;
export type SessionId = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp1 = string;
export type SessionId1 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp2 = string;
export type SessionId2 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp3 = string;
export type SessionId3 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp4 = string;
export type SessionId4 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp5 = string;
export type SessionId5 = string;
export type AgentId4 = string;
export type ErrorType = string;
export type ErrorSummary = string;
export type RequestId = string | null;
export type Attempt = string;
export type Classification = string | null;
export type FrameworkRetriesExhausted = number | null;
export type OccurredAt = number | null;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp6 = string;
export type SessionId6 = string;
export type AgentId5 = string | null;
export type Reason = string;
export type ErrorSummary1 = string | null;
export type ErrorType1 = string | null;
export type Details = {
  [k: string]: unknown;
} | null;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp7 = string;
export type SessionId7 = string;
export type PendingToolCallCount = number;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp8 = string;
export type SessionId8 = string;
export type AgentId6 = string | null;
export type CascadeDriverId = string | null;
export type WasWarm = boolean;
export type PoolSlotPid = number;
export type TerminalReason = string | null;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp9 = string;
export type SessionId9 = string;
export type AgentId7 = string;
export type ToolName = string;
export type CallId = string | null;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp10 = string;
export type SessionId10 = string;
export type AgentId8 = string;
export type ToolName1 = string;
export type CallId1 = string | null;
export type Success1 = boolean;
export type IsErrorResult = boolean;
export type ResultStatus = string | null;
export type DurationSeconds = number;
export type ErrorMessage = string | null;
export type Backgrounded = boolean;
export type ContinuationId = string | null;
export type ShowOutput = boolean | null;
export type ShowPopup = boolean | null;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp11 = string;
export type SessionId11 = string;
export type AgentId9 = string;
export type CallId2 = string;
export type Chunk = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp12 = string;
export type SessionId12 = string;
export type AgentId10 = string;
export type RequestId1 = string;
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
export type EventType13 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp13 = string;
export type SessionId13 = string;
export type AgentId11 = string;
export type RequestId2 = string;
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
export type EventType14 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp14 = string;
export type SessionId14 = string;
export type AgentId12 = string;
export type RequestId3 = string;
export type ToolName4 = string;
export type Granted = boolean;
export type Method = string;
export type Comment = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp15 = string;
export type SessionId15 = string;
export type EffectiveDefault = string;
export type SuspensionScope = string | null;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp16 = string;
export type SessionId16 = string;
export type AgentId13 = string;
export type RequestId4 = string;
export type ToolName5 = string;
export type ContextLines = string[];
export type TotalQuestions = number;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp17 = string;
export type SessionId17 = string;
export type AgentId14 = string;
export type RequestId5 = string;
export type ToolName6 = string;
export type QuestionIndex = number;
export type TotalQuestions1 = number;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp18 = string;
export type SessionId18 = string;
export type AgentId15 = string;
export type RequestId6 = string;
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
export type EventType19 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp19 = string;
export type SessionId19 = string;
export type AgentId16 = string;
export type RequestId7 = string;
export type ToolName7 = string;
export type QaPairs = string[][];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp20 = string;
export type SessionId20 = string;
export type AgentId17 = string;
export type RequestId8 = string;
export type ToolName8 = string;
export type Context = string;
export type Questions = {
  [k: string]: unknown;
}[];
export type BatchOnly = boolean;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp21 = string;
export type SessionId21 = string;
export type RequestId9 = string;
export type Answers = string[];
export type Cancelled = boolean;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp22 = string;
export type SessionId22 = string;
export type AgentId18 = string;
export type RequestId10 = string;
export type ToolName9 = string;
export type PromptLines1 = string[];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp23 = string;
export type SessionId23 = string;
export type AgentId19 = string;
export type RequestId11 = string;
export type ToolName10 = string;
export type SelectedIds = string[];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp24 = string;
export type SessionId24 = string;
export type RequestId12 = string;
export type Response = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp25 = string;
export type SessionId25 = string;
export type RequestId13 = string;
export type SessionWorkspace = string;
export type ClientWorkspace = string;
export type ResponseOptions2 = {
  [k: string]: string;
}[];
export type PromptLines2 = string[];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp26 = string;
export type SessionId26 = string;
export type RequestId14 = string;
export type Action = string;
export type NewSessionId = string | null;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp27 = string;
export type SessionId27 = string;
export type RequestId15 = string;
export type Response1 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp28 = string;
export type SessionId28 = string;
export type RequestId16 = string;
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
export type EventType29 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp29 = string;
export type SessionId29 = string;
export type RequestId17 = string;
export type Connect = boolean;
export type ModelName = string;
export type PersistEnv = boolean;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp30 = string;
export type SessionId30 = string;
export type AgentId20 = string;
export type PlanName = string;
export type Steps = {
  [k: string]: unknown;
}[];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp31 = string;
export type SessionId31 = string;
export type AgentId21 = string;
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
export type EventType32 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp32 = string;
export type SessionId32 = string;
export type AgentId22 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp33 = string;
export type SessionId33 = string;
export type AgentId23 = string;
export type PromptTokens = number;
export type OutputTokens = number;
export type TotalTokens = number;
export type CacheReadTokens = number | null;
export type CacheCreationTokens = number | null;
export type ReasoningTokens = number | null;
export type ThinkingTokens = number | null;
export type CostUsd = number | null;
export type SpendTotalTokens = number | null;
export type SpendPromptTokens = number | null;
export type SpendOutputTokens = number | null;
export type SpendCacheReadTokens = number | null;
export type SpendCacheCreationTokens = number | null;
export type ContextLimit = number;
export type PercentUsed = number;
export type TokensRemaining = number;
export type Turns = number;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp34 = string;
export type SessionId34 = string;
export type AgentId24 = string;
export type Threshold = number | null;
export type Strategy = string | null;
export type TargetPercent = number | null;
export type ContinuousMode = boolean;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp35 = string;
export type SessionId35 = string;
export type AgentId25 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp36 = string;
export type SessionId36 = string;
export type AgentId26 = string;
export type TurnNumber = number;
export type CompletionGap = string | null;
export type DurationSeconds1 = number;
export type FunctionCalls = {
  [k: string]: unknown;
}[];
export type FormattedText = string | null;
export type FinishReason = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp37 = string;
export type SessionId37 = string;
export type AgentId27 = string;
export type ContextLimit1 = number;
export type PercentUsed1 = number;
export type TokensRemaining1 = number;
export type PendingToolCalls = number;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp38 = string;
export type SessionId38 = string;
export type Message = string;
export type Style = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp39 = string;
export type SessionId39 = string;
export type Lines = unknown[][];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp40 = string;
export type SessionId40 = string;
export type Step = string;
export type Status2 = string;
export type Message1 = string;
export type StepNumber = number;
export type TotalSteps = number;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp41 = string;
export type SessionId41 = string;
export type Error3 = string;
export type ErrorType2 = string;
export type Recoverable = boolean;
export type Details1 = {
  [k: string]: unknown;
} | null;
export type RequestId18 = string | null;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp42 = string;
export type SessionId42 = string;
export type Message2 = string;
export type Attempt1 = number;
export type MaxAttempts = number;
export type Delay = number;
export type ErrorType3 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp43 = string;
export type SessionId43 = string;
export type Sessions = {
  [k: string]: unknown;
}[];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp44 = string;
export type SessionId44 = string;
export type AgentId28 = string;
export type Phase = string;
export type TriggerReason = string | null;
export type Strategy1 = string | null;
export type PercentUsed2 = number | null;
export type Threshold1 = number | null;
export type ContextLimit2 = number | null;
export type Success2 = boolean | null;
export type ItemsCollected = number | null;
export type TokensBefore = number | null;
export type TokensAfter = number | null;
export type TokensFreed = number | null;
export type Error4 = string | null;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp45 = string;
export type SessionId45 = string;
export type RequestId19 = string | null;
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
export type EventType46 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp46 = string;
export type SessionId46 = string;
export type Memories1 = {
  [k: string]: unknown;
}[];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp47 = string;
export type SessionId47 = string;
export type Paths = {
  [k: string]: string;
}[];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp48 = string;
export type SessionId48 = string;
export type Services1 = {
  [k: string]: unknown;
}[];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp49 = string;
export type SessionId49 = string;
export type Description = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp50 = string;
export type SessionId50 = string;
export type Name = string;
export type Description1 = string;
export type Plugins = string[];
export type PreloadedPlugins = string[];
export type Model = string | null;
export type Provider = string | null;
export type MaxTurns = number;
export type BudgetControl = {
  [k: string]: unknown;
} | null;
export type Gc = {
  [k: string]: unknown;
} | null;
export type RuntimeLimits = {
  [k: string]: unknown;
} | null;
export type CompletionPayloadSchema =
  | string
  | {
      [k: string]: unknown;
    }
  | null;
export type EnvVarNames = string[];
export type Profiles = ProfileSummary[];
export type Name1 = string;
export type Error5 = string;
export type ParseErrors = ProfileParseError[];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp51 = string;
export type SessionId51 = string;
export type Text1 = string;
export type Attachments = {
  [k: string]: unknown;
}[];
export type ParallelTools = boolean | null;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp52 = string;
export type SessionId52 = string;
export type RequestId20 = string;
export type Response2 = string;
export type EditedArguments = {
  [k: string]: unknown;
} | null;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp53 = string;
export type SessionId53 = string;
export type RequestId21 = string;
export type QuestionIndex2 = number;
export type Response3 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp54 = string;
export type SessionId54 = string;
export type AgentId29 = string | null;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp55 = string;
export type SessionId55 = string;
export type Name2 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp56 = string;
export type SessionId56 = string;
export type AgentId30 = string;
export type EventNames = string[];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp57 = string;
export type SessionId57 = string;
export type Command = string;
export type Args = string[];
export type Payload1 = {
  [k: string]: unknown;
} | null;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp58 = string;
export type SessionId58 = string;
export type AgentId31 = string | null;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp59 = string;
export type SessionId59 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp60 = string;
export type SessionId60 = string;
export type Commands = {
  [k: string]: string;
}[];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp61 = string;
export type SessionId61 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp62 = string;
export type SessionId62 = string;
export type Tools1 = {
  [k: string]: unknown;
}[];
export type Message3 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp63 = string;
export type SessionId63 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp64 = string;
export type SessionId64 = string;
export type ToolName11 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp65 = string;
export type SessionId65 = string;
export type Tools2 = {
  [k: string]: unknown;
}[];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp66 = string;
export type SessionId66 = string;
export type CallId4 = string;
export type AgentId32 = string;
export type ToolName12 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp67 = string;
export type SessionId67 = string;
export type CallId5 = string;
export type Result1 = string;
export type Error6 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp68 = string;
export type SessionId68 = string;
export type AgentId33 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp69 = string;
export type SessionId69 = string;
export type AgentId34 = string;
export type History = {
  [k: string]: unknown;
}[];
export type TurnAccounting = {
  [k: string]: unknown;
}[];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp70 = string;
export type SessionId70 = string;
export type TraceLogPath = string | null;
export type ProviderTraceLog = string | null;
export type WorkingDir = string | null;
export type ConfigRoot = string | null;
export type EnvFile = string | null;
export type Presentation = {
  [k: string]: unknown;
} | null;
export type PermissionTimeout = number | null;
export type Apparmor = boolean;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp71 = string;
export type SessionId71 = string;
export type Text2 = string;
export type PositionInQueue = number;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp72 = string;
export type SessionId72 = string;
export type Text3 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp73 = string;
export type SessionId73 = string;
export type PartialResponseChars = number;
export type UserPromptPreview = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp74 = string;
export type SessionId74 = string;
export type AgentId35 = string;
export type RecoveredCalls = number;
export type ActionTaken = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp75 = string;
export type SessionId75 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp76 = string;
export type SessionId76 = string;
export type Root = string;
export type Workspaces = {
  [k: string]: unknown;
}[];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp77 = string;
export type SessionId77 = string;
export type Name3 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp78 = string;
export type SessionId78 = string;
export type Name4 = string;
export type Path = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp79 = string;
export type SessionId79 = string;
export type Name5 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp80 = string;
export type SessionId80 = string;
export type Workspace = string;
export type Configured = boolean;
export type Provider1 = string | null;
export type Model1 = string | null;
export type AvailableProviders = string[];
export type MissingFields = string[];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp81 = string;
export type SessionId81 = string;
export type Provider2 = string;
export type Model2 = string | null;
export type ApiKey = string | null;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp82 = string;
export type SessionId82 = string;
export type Workspace1 = string;
export type Provider3 = string;
export type Model3 = string | null;
export type Success3 = boolean;
export type Error7 = string | null;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp83 = string;
export type SessionId83 = string;
export type Changes = {
  [k: string]: string;
}[];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp84 = string;
export type SessionId84 = string;
export type Files = {
  [k: string]: string;
}[];
export type Total = number;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp85 = string;
export type SessionId85 = string;
export type WorkspaceId = string;
export type Name6 = string;
export type Size = number;
export type ContentType = string | null;
export type Mode1 = number | null;
export type Files1 = StagedFileSpec[];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp86 = string;
export type SessionId86 = string;
export type WorkspaceId1 = string;
export type Staged = string[];
export type Failed = {
  [k: string]: string;
}[];
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp87 = string;
export type SessionId87 = string;
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
export type EventType88 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp88 = string;
export type SessionId88 = string;
export type RequestId22 = string;
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
export type EventType89 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp89 = string;
export type SessionId89 = string;
export type RequestId23 = string;
export type RemoteAgentId = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp90 = string;
export type SessionId90 = string;
export type RequestId24 = string;
export type Reason1 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp91 = string;
export type SessionId91 = string;
export type RequestId25 = string;
export type RemoteAgentId1 = string;
export type Text4 = string;
export type Source1 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp92 = string;
export type SessionId92 = string;
export type RequestId26 = string;
export type RemoteAgentId2 = string;
export type Success4 = boolean;
export type Summary = string;
export type Error8 = string;
export type WorkspaceModified = boolean;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp93 = string;
export type SessionId93 = string;
export type RequestId27 = string;
export type RemoteAgentId3 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp94 = string;
export type SessionId94 = string;
export type RequestId28 = string;
export type RemoteAgentId4 = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp95 = string;
export type SessionId95 = string;
export type GateName = string;
export type TenantId = string;
export type Owner = string;
export type AnnouncedAt = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp96 = string;
export type SessionId96 = string;
export type GateName1 = string;
export type TenantId1 = string;
export type Owner1 = string;
export type Outcome = {
  [k: string]: unknown;
} | null;
export type ReleasedAt = string;
export type WasAnnounced = boolean;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp97 = string;
export type SessionId97 = string;
export type GateName2 = string;
export type TenantId2 = string;
export type State = string;
export type Owner2 = string | null;
export type Intent1 = {
  [k: string]: unknown;
} | null;
export type AcquiredAt = string | null;
export type ExpiresAt = string | null;
export type Gates = GateState[];
export type SnapshotAt = string;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp98 = string;
export type SessionId98 = string;
export type Text5 = string;
export type SourceType = string;
export type SourceId = string | null;
export type RequestId29 = string | null;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp99 = string;
export type SessionId99 = string;
export type RequestId30 = string;
export type Status3 = string;
export type Detail = string | null;
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
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp100 = string;
export type SessionId100 = string;
export type RequestId31 = string;
export type Messages =
  | {
      [k: string]: unknown;
    }[]
  | null;
export type TimeoutSeconds = number;
/**
 * All event types in the protocol.
 */
export type EventType101 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp101 = string;
export type SessionId101 = string;
export type RequestId32 = string;
export type ResponseText = string;
export type Error9 = string;
/**
 * All event types in the protocol.
 */
export type EventType102 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp102 = string;
export type SessionId102 = string;
export type RequestId33 = string;
export type AfterMessage = number | null;
export type AfterToolCall = string | null;
export type AfterTimestamp = string | null;
/**
 * All event types in the protocol.
 */
export type EventType103 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp103 = string;
export type SessionId103 = string;
export type RequestId34 = string;
export type ForkIndex = number;
export type Error10 = string;
/**
 * All event types in the protocol.
 */
export type EventType104 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp104 = string;
export type SessionId104 = string;
export type WakeRef = string;
export type Outcome1 = string;
export type Detail1 = string;
export type ExpiresAt1 = number;
export type Endpoint = string;
/**
 * All event types in the protocol.
 */
export type EventType105 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp105 = string;
export type SessionId105 = string;
export type WakeRef1 = string;
export type Source2 = string;
/**
 * All event types in the protocol.
 */
export type EventType106 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp106 = string;
export type SessionId106 = string;
export type Tools3 = string[];
export type Patterns = string[];
/**
 * All event types in the protocol.
 */
export type EventType107 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp107 = string;
export type SessionId107 = string;
export type Tools4 = string[];
export type Patterns1 = string[];
/**
 * All event types in the protocol.
 */
export type EventType108 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp108 = string;
export type SessionId108 = string;
export type Target = string;
export type Tools5 = string[];
export type Patterns2 = string[];
/**
 * All event types in the protocol.
 */
export type EventType109 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp109 = string;
export type SessionId109 = string;
export type Target1 = string;
/**
 * All event types in the protocol.
 */
export type EventType110 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp110 = string;
export type SessionId110 = string;
export type Policy = string;
/**
 * All event types in the protocol.
 */
export type EventType111 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp111 = string;
export type SessionId111 = string;
export type RequestId35 = string;
/**
 * All event types in the protocol.
 */
export type EventType112 =
  | "connected"
  | "disconnected"
  | "agent.created"
  | "agent.output"
  | "agent.status_changed"
  | "agent.completed"
  | "agent.error"
  | "session.terminated"
  | "slot.settled"
  | "session.restored"
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
  | "gc.config"
  | "gc"
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
  | "inject_prompt.result"
  | "replay_messages.request"
  | "replay_messages.result"
  | "resolve_fork_point.request"
  | "resolve_fork_point.result"
  | "session.wake_bind_result"
  | "session.woken"
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
  | "peer.stop_acknowledged"
  | "gate.announced"
  | "gate.released"
  | "gates.snapshot";
export type Timestamp112 = string;
export type SessionId112 = string;
export type RequestId36 = string;
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
  session_id?: SessionId;
  protocol_version?: ProtocolVersion;
  server_info?: ServerInfo;
}
export interface ServerInfo {
  [k: string]: unknown;
}
/**
 * Sent when a new agent (main or subagent) is created.
 *
 * Attributes:
 *     agent_id: Logical agent identifier (e.g. ``"main"`` or a
 *         subagent slot id).  This is the agent NAME slot, NOT the
 *         daemon's session_id.
 *     agent_name: Human-readable agent display name (typically the
 *         agent's persona name from ``.jaato/agents/<name>.md``).
 *     agent_type: ``"main"`` or ``"subagent"``.
 *     profile_name: Optional profile name resolved at spawn time.
 *     parent_agent_id: Optional logical id of the spawning agent
 *         (None for top-level / main agents).
 *     created_at: Optional ISO-8601 timestamp.
 *     session_id: Daemon-side session identifier (server 0.6.175+).
 *         Populated by every constructor site via the same parent-
 *         walk resolution used by ``RenderContext.session_id``
 *         (server 0.6.172+).  Subagent emit sites fall back to the
 *         parent's session_id when the immediate session has no
 *         ``_daemon_session_id`` of its own yet (subagent
 *         JaatoSession instances inherit the root agent's session
 *         via ``_parent_session``).  Empty string when no ancestor
 *         in the parent chain has a session_id set
 *         (e.g. ``main_agent_id`` emit during bootstrap before
 *         ``set_daemon_session_id`` fires).  Cascade observers
 *         use this for per-stage session_id correlation without
 *         having to maintain their own ``agent_id → session_id``
 *         map.
 */
export interface AgentCreatedEvent {
  type?: EventType1;
  timestamp?: Timestamp1;
  session_id?: SessionId1;
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
  session_id?: SessionId2;
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
  session_id?: SessionId3;
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
  session_id?: SessionId4;
  agent_id?: AgentId3;
  completed_at?: CompletedAt;
  success?: Success;
  token_usage?: TokenUsage;
  turns_used?: TurnsUsed;
  error?: Error1;
  payload?: Payload;
}
/**
 * An agent hit a terminal error that the framework could not self-resolve.
 *
 * This is the **recovery contract**: it fires when the framework's automatic
 * management (``with_retry`` for retryable provider errors, the completion
 * nudge loop) is **exhausted** or never applied — i.e. the framework is out of
 * moves. It gives a reactor *first refusal* to recover the failed stage
 * (re-spawn, reroute to another model/provider, escalate) via the existing
 * ``create_session`` path, BEFORE the session's terminal
 * ``SessionTerminatedEvent(reason="error")`` lands.
 *
 * Emit order on the wire is **always** ``AgentErrorEvent`` first, then
 * ``SessionTerminatedEvent(reason="error")``. A reactor that recovers should
 * mark the ``session_id`` handled so the (back-compat) terminated handler
 * no-ops; a cascade with no ``AGENT_ERROR`` handler ignores this event and the
 * terminated event drives the legacy abort — fully back-compatible.
 *
 * Recovery is **decoupled from transience**: a non-transient error is still
 * stage-recoverable (reroute/escalate). The framework offers the recovery
 * *point*; the reactor's policy decides what to do.
 *
 * Fields:
 *     agent_id: The failed agent / cascade stage.
 *     session_id: The failed session (dedupe / handled-marking key).
 *     error_type: Exception class name (``"APIError"``, ``"RunnerCallError"``,
 *         ``"NudgeExhausted"``, ...). Same value carried on the subsequent
 *         ``SessionTerminatedEvent.error_type``.
 *     error_summary: Human-readable cause.
 *     request_id: Provider request id (e.g. OpenAI ``req_…``) when the
 *         underlying exception carries one; ``None`` otherwise. For
 *         observability / support correlation.
 *     attempt: The **reactor-level** re-spawn count for this logical stage,
 *         echoed verbatim from the spawn's ``agent_params["attempt"]`` (a
 *         string on the wire). This is NOT ``with_retry``'s internal
 *         per-request attempt count (which is never surfaced). ``"0"`` /
 *         absent on the first spawn. The reactor owns the cap.
 *     classification: Optional COARSE shape hint — ``"transient_provider"`` /
 *         ``"fatal_contract"`` / ``"unknown"``. **Advisory only**: it never
 *         gates whether this event fires. ``None`` when unclassified.
 *     framework_retries_exhausted: Optional informational count of automatic
 *         retries the framework already burned before giving up. ``None`` when
 *         not applicable.
 *     occurred_at: Emit timestamp (epoch seconds).
 */
export interface AgentErrorEvent {
  type?: EventType5;
  timestamp?: Timestamp5;
  session_id?: SessionId5;
  agent_id?: AgentId4;
  error_type?: ErrorType;
  error_summary?: ErrorSummary;
  request_id?: RequestId;
  attempt?: Attempt;
  classification?: Classification;
  framework_retries_exhausted?: FrameworkRetriesExhausted;
  occurred_at?: OccurredAt;
}
/**
 * Session has fully wound down — safe to disconnect or
 * ``delete_session``.
 *
 * Fires in two scenarios:
 *
 * 1. **Natural completion**: emitted spontaneously after the
 *    agent's terminal completion (``AgentCompletedEvent``) AND
 *    the framework's post-completion wrap-up has drained
 *    (``_is_running`` returned False, plugin-on-end hooks ran,
 *    journal flushed).  Test harnesses can subscribe to this
 *    instead of the legacy "subscribe AGENT_COMPLETED + wait
 *    10s for TURN_COMPLETED" heuristic.
 *
 * 2. **Client-requested**: emitted in response to ``session.end``
 *    after the daemon has stopped any in-flight activity and run
 *    cleanup.  Replaces the legacy
 *    ``SystemMessageEvent("[SESSION_TERMINATED]")`` string-based
 *    marker.
 *
 * The ``reason`` field distinguishes the two paths so consumers
 * can handle them differently if needed.
 *
 * When ``reason="error"``, the framework populates
 * ``error_summary`` + ``error_type`` from the underlying
 * ``Exception`` at the emit site (server 0.6.159+ / SDK 0.14.1+).
 * Cascade observers can read these to surface the failure cause
 * without grepping the daemon log — e.g.
 * ``error_type="AnthropicAPIError"`` +
 * ``error_summary="402 Payment Required ..."``.  Both fields stay
 * ``None`` for the non-error reasons (``natural`` /
 * ``client_request`` / ``stopped``).
 *
 * When ``reason="budget_exhausted"``, the session hit a budget
 * ceiling and REFUSES all further turns -- exhaustion means "this
 * session is done", not "cancel this turn"
 * (:meth:`JaatoSession._refuse_if_budget_exhausted`).  ``details``
 * carries the refusal prose and the per-dimension usage.  Emitted
 * because the refusal short-circuits before any turn runs, so no
 * turn-completion notification fires and a wake-driven driver would
 * otherwise wait out its full timeout and report a generic failure --
 * a ceiling stop indistinguishable from a break.
 *
 * Canonical pattern (test harness):
 *
 *     client.subscribe_once(EventType.SESSION_TERMINATED, on_done)
 *     sid = await client.create_session(...)
 *     await client.send_message(...)
 *     await on_done.wait()
 *     # Session has fully wound down.  Optionally delete_session(sid).
 */
export interface SessionTerminatedEvent {
  type?: EventType6;
  timestamp?: Timestamp6;
  session_id?: SessionId6;
  agent_id?: AgentId5;
  reason?: Reason;
  error_summary?: ErrorSummary1;
  error_type?: ErrorType1;
  details?: Details;
}
/**
 * Session was loaded from disk and the first client just attached.
 *
 * Phase 3 §3.12 disk-restore + peer-review M5/N1: when a session
 * is restored from disk (daemon restart / cold attach), the
 * daemon may have held in-flight tool calls during the
 * no-client window using the defer-and-flush posture (vs
 * denying outright per the pre-§3.12 behaviour).  This event
 * fires on the first client-attach so the client can surface a
 * "this session was restored — N pending tool calls to review"
 * prompt; the operator drains the queue (each held ASK relays
 * through the now-attached ``client.prompt_operator`` channel
 * as if it had just landed) and the
 * ``Session.restored_pending_attach`` flag clears.
 *
 * ``pending_tool_call_count`` is 0 for clean restores with no
 * in-flight work; the event still fires in that case so clients
 * can distinguish a fresh-attach from a restored-attach for
 * telemetry / UX purposes.
 */
export interface SessionRestoredEvent {
  type?: EventType7;
  timestamp?: Timestamp7;
  session_id?: SessionId7;
  pending_tool_call_count?: PendingToolCallCount;
}
/**
 * A cascade stage's session has fully settled — its runner/slot has
 * returned to the pool (warm) or been torn down (cold) — and the next stage
 * is safe to spawn.
 *
 * Emitted by the daemon at the END of ``JaatoServer.shutdown`` for **every**
 * cascade session (``cascade_driver_id`` set), on ALL teardown paths:
 * pool-slot-returned, pool-slot-torn-down-on-error, and cold-spawned.  This
 * universality is the point — a cascade reactor can gate the next stage's
 * spawn on this single event with NO timeout and NO stall risk, because it
 * fires exactly once per stage regardless of how the stage's runner ended.
 *
 * ``was_warm`` reports whether a warm pre-warm-pool slot was returned (so the
 * next stage's spawn will reuse it, ≈30s→7s bootstrap) vs. a cold/torn-down
 * teardown (next stage cold-spawns).  It is observability for the reactor —
 * the spawn happens either way; ``was_warm`` just says whether it'll be fast.
 *
 * Replaces the earlier warm-only ``SlotReusableEvent`` (which did not fire for
 * cold-spawned stages — common for the early cascade stages — and so could
 * stall a pure-reactor handoff).  Correlate by ``cascade_driver_id``; route
 * per-stage by ``agent_id``.  Distinct from :class:`SessionTerminatedEvent`,
 * which fires EARLIER (before the slot returns) so spawning on it races the
 * slot and cold-spawns.
 */
export interface SlotSettledEvent {
  type?: EventType8;
  timestamp?: Timestamp8;
  session_id?: SessionId8;
  agent_id?: AgentId6;
  cascade_driver_id?: CascadeDriverId;
  was_warm?: WasWarm;
  pool_slot_pid?: PoolSlotPid;
  terminal_reason?: TerminalReason;
}
/**
 * Tool execution has started.
 */
export interface ToolCallStartEvent {
  type?: EventType9;
  timestamp?: Timestamp9;
  session_id?: SessionId9;
  agent_id?: AgentId7;
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
  type?: EventType10;
  timestamp?: Timestamp10;
  session_id?: SessionId10;
  agent_id?: AgentId8;
  tool_name?: ToolName1;
  call_id?: CallId1;
  success?: Success1;
  is_error_result?: IsErrorResult;
  result_status?: ResultStatus;
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
  type?: EventType11;
  timestamp?: Timestamp11;
  session_id?: SessionId11;
  agent_id?: AgentId9;
  call_id?: CallId2;
  chunk?: Chunk;
}
/**
 * Permission is requested for a tool execution.
 *
 * Includes pre-formatted prompt lines (with diff for file edits) when available.
 */
export interface PermissionRequestedEvent {
  type?: EventType12;
  timestamp?: Timestamp12;
  session_id?: SessionId12;
  agent_id?: AgentId10;
  request_id?: RequestId1;
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
  type?: EventType13;
  timestamp?: Timestamp13;
  session_id?: SessionId13;
  agent_id?: AgentId11;
  request_id?: RequestId2;
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
  type?: EventType14;
  timestamp?: Timestamp14;
  session_id?: SessionId14;
  agent_id?: AgentId12;
  request_id?: RequestId3;
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
  type?: EventType15;
  timestamp?: Timestamp15;
  session_id?: SessionId15;
  effective_default?: EffectiveDefault;
  suspension_scope?: SuspensionScope;
}
/**
 * Clarification session has started.
 */
export interface ClarificationRequestedEvent {
  type?: EventType16;
  timestamp?: Timestamp16;
  session_id?: SessionId16;
  agent_id?: AgentId13;
  request_id?: RequestId4;
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
  type?: EventType17;
  timestamp?: Timestamp17;
  session_id?: SessionId17;
  agent_id?: AgentId14;
  request_id?: RequestId5;
  tool_name?: ToolName6;
  question_index?: QuestionIndex;
  total_questions?: TotalQuestions1;
}
/**
 * A single clarification question to answer.
 */
export interface ClarificationQuestionEvent {
  type?: EventType18;
  timestamp?: Timestamp18;
  session_id?: SessionId18;
  agent_id?: AgentId15;
  request_id?: RequestId6;
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
  type?: EventType19;
  timestamp?: Timestamp19;
  session_id?: SessionId19;
  agent_id?: AgentId16;
  request_id?: RequestId7;
  tool_name?: ToolName7;
  qa_pairs?: QaPairs;
}
/**
 * All clarification questions sent at once for batch answering.
 *
 * Emitted on two distinct paths, told apart by ``batch_only``:
 *
 * * **Daemon-local sessions** (``batch_only=False``) — emitted before the
 *   QueueChannel loop so a client that can render every question at once
 *   (a tabbed panel, say) does not have to wait for them to trickle in.
 *   The per-question flow still follows: an ``AgentOutputEvent`` carrying
 *   the question text plus a ``ClarificationInputModeEvent`` for each
 *   question in turn.  A client that prefers the per-question flow may
 *   ignore this event entirely.
 * * **Runner-tier sessions** (``batch_only=True``) — emitted by
 *   ``server.runner_rpc_handlers.clarification_relay``, which relays the
 *   whole batch from the runner and awaits the whole answer set.  Nothing
 *   else follows: no ``AgentOutputEvent``, no
 *   ``ClarificationInputModeEvent``, no ``ClarificationResolvedEvent``.
 *   A client that ignores this event leaves the tool call — and with it
 *   the turn — blocked forever (#704), so handling it is mandatory.
 *
 * Either way the reply is a single :class:`ClarificationBatchResponseEvent`.
 */
export interface ClarificationBatchEvent {
  type?: EventType20;
  timestamp?: Timestamp20;
  session_id?: SessionId20;
  agent_id?: AgentId17;
  request_id?: RequestId8;
  tool_name?: ToolName8;
  context?: Context;
  questions?: Questions;
  batch_only?: BatchOnly;
}
/**
 * Client responds with all answers at once (batch mode).
 *
 * ``cancelled=True`` abandons the clarification instead of answering it:
 * the tool returns ``{"cancelled": True}`` to the model and the turn
 * continues.  It is the only way out of a ``batch_only`` clarification
 * the user cannot or will not answer — without it, an unanswerable
 * question blocks the turn indefinitely.  ``answers`` is ignored when
 * ``cancelled`` is set.
 */
export interface ClarificationBatchResponseEvent {
  type?: EventType21;
  timestamp?: Timestamp21;
  session_id?: SessionId21;
  request_id?: RequestId9;
  answers?: Answers;
  cancelled?: Cancelled;
}
/**
 * Reference selection has been requested.
 *
 * Sent when the model calls selectReferences and the user needs to choose
 * which references to include.
 */
export interface ReferenceSelectionRequestedEvent {
  type?: EventType22;
  timestamp?: Timestamp22;
  session_id?: SessionId22;
  agent_id?: AgentId18;
  request_id?: RequestId10;
  tool_name?: ToolName9;
  prompt_lines?: PromptLines1;
}
/**
 * Reference selection has been completed.
 */
export interface ReferenceSelectionResolvedEvent {
  type?: EventType23;
  timestamp?: Timestamp23;
  session_id?: SessionId23;
  agent_id?: AgentId19;
  request_id?: RequestId11;
  tool_name?: ToolName10;
  selected_ids?: SelectedIds;
}
/**
 * Respond to a reference selection request.
 */
export interface ReferenceSelectionResponseRequest {
  type?: EventType24;
  timestamp?: Timestamp24;
  session_id?: SessionId24;
  request_id?: RequestId12;
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
  type?: EventType25;
  timestamp?: Timestamp25;
  session_id?: SessionId25;
  request_id?: RequestId13;
  session_workspace?: SessionWorkspace;
  client_workspace?: ClientWorkspace;
  response_options?: ResponseOptions2;
  prompt_lines?: PromptLines2;
}
/**
 * Workspace mismatch has been resolved.
 */
export interface WorkspaceMismatchResolvedEvent {
  type?: EventType26;
  timestamp?: Timestamp26;
  session_id?: SessionId26;
  request_id?: RequestId14;
  action?: Action;
  new_session_id?: NewSessionId;
}
/**
 * Respond to a workspace mismatch request.
 */
export interface WorkspaceMismatchResponseRequest {
  type?: EventType27;
  timestamp?: Timestamp27;
  session_id?: SessionId27;
  request_id?: RequestId15;
  response?: Response1;
}
/**
 * Offer session setup after successful authentication.
 *
 * Emitted by daemon after an auth command succeeds. The client renders a
 * multi-step wizard and sends back a single PostAuthSetupResponse.
 */
export interface PostAuthSetupEvent {
  type?: EventType28;
  timestamp?: Timestamp28;
  session_id?: SessionId28;
  request_id?: RequestId16;
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
  type?: EventType29;
  timestamp?: Timestamp29;
  session_id?: SessionId29;
  request_id?: RequestId17;
  connect?: Connect;
  model_name?: ModelName;
  persist_env?: PersistEnv;
}
/**
 * Plan has been created or updated.
 */
export interface PlanUpdatedEvent {
  type?: EventType30;
  timestamp?: Timestamp30;
  session_id?: SessionId30;
  agent_id?: AgentId20;
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
  type?: EventType31;
  timestamp?: Timestamp31;
  session_id?: SessionId31;
  agent_id?: AgentId21;
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
  type?: EventType32;
  timestamp?: Timestamp32;
  session_id?: SessionId32;
  agent_id?: AgentId22;
}
/**
 * Context window usage has changed.
 *
 * Carries a typed ``usage`` (``UsageBreakdown``) shared with
 * ``TurnCompletedEvent`` and ``TurnProgressEvent`` so consumers can
 * treat the three events uniformly. Context-window framing fields
 * (``context_limit``, ``percent_used``, ``tokens_remaining``,
 * ``turns``) stay on this event because they describe the *window*,
 * not the most recent generation.
 *
 * GC configuration moved to ``GCConfigEvent`` in v1.0 — query that
 * event (or read it from session init) for status-bar display.
 */
export interface ContextUpdatedEvent {
  type?: EventType33;
  timestamp?: Timestamp33;
  session_id?: SessionId33;
  agent_id?: AgentId23;
  usage?: UsageBreakdown;
  context_limit?: ContextLimit;
  percent_used?: PercentUsed;
  tokens_remaining?: TokensRemaining;
  turns?: Turns;
}
/**
 * Provider-agnostic per-turn usage shape carried by Context/Turn events.
 *
 * Single source of truth for token + cost reporting on the wire.  All
 * optional fields default to ``None`` so a provider that doesn't
 * report a given dimension simply omits it instead of zero-padding
 * (which would make a ``0`` cache hit indistinguishable from "no
 * caching support").
 *
 * ``cost_usd`` is populated by the daemon when either:
 * - the provider reports a real cost (e.g. ``claude_cli`` exposes
 *   ``total_cost_usd`` from the underlying CLI), or
 * - the operator has loaded a pricing table at
 *   ``.jaato/pricing.json`` (Litellm-compatible) and the model name
 *   is found there.
 *
 * When neither source has a number, ``cost_usd`` is ``None`` —
 * consumers must not assume zero means free.
 */
export interface UsageBreakdown {
  prompt_tokens?: PromptTokens;
  output_tokens?: OutputTokens;
  total_tokens?: TotalTokens;
  cache_read_tokens?: CacheReadTokens;
  cache_creation_tokens?: CacheCreationTokens;
  reasoning_tokens?: ReasoningTokens;
  thinking_tokens?: ThinkingTokens;
  cost_usd?: CostUsd;
  spend_total_tokens?: SpendTotalTokens;
  spend_prompt_tokens?: SpendPromptTokens;
  spend_output_tokens?: SpendOutputTokens;
  spend_cache_read_tokens?: SpendCacheReadTokens;
  spend_cache_creation_tokens?: SpendCacheCreationTokens;
}
/**
 * GC configuration snapshot for the active session.
 *
 * Emitted on session init and whenever the GC plugin is reconfigured.
 * Carries only configuration — actual usage lives in
 * ``ContextUpdatedEvent``. Splitting these two concerns avoids the
 * pre-1.0 hack where ``ContextUpdatedEvent`` doubled as a status-bar
 * config carrier.
 */
export interface GCConfigEvent {
  type?: EventType34;
  timestamp?: Timestamp34;
  session_id?: SessionId34;
  agent_id?: AgentId24;
  threshold?: Threshold;
  strategy?: Strategy;
  target_percent?: TargetPercent;
  continuous_mode?: ContinuousMode;
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
  type?: EventType35;
  timestamp?: Timestamp35;
  session_id?: SessionId35;
  agent_id?: AgentId25;
  budget_snapshot?: BudgetSnapshot;
}
export interface BudgetSnapshot {
  [k: string]: unknown;
}
/**
 * A conversation turn has completed.
 *
 * The ``usage`` field carries the provider-agnostic
 * ``UsageBreakdown`` (token counts, cache hits, reasoning/thinking
 * tokens, cost when known).  Treat it as the canonical per-turn
 * usage record — it's the same shape ``TurnProgressEvent`` and
 * ``ContextUpdatedEvent`` use.
 */
export interface TurnCompletedEvent {
  type?: EventType36;
  timestamp?: Timestamp36;
  session_id?: SessionId36;
  agent_id?: AgentId26;
  turn_number?: TurnNumber;
  usage?: UsageBreakdown;
  completion_gap?: CompletionGap;
  duration_seconds?: DurationSeconds1;
  function_calls?: FunctionCalls;
  formatted_text?: FormattedText;
  finish_reason?: FinishReason;
}
/**
 * Incremental progress during turn execution.
 *
 * Emitted after each model response within a turn, enabling
 * real-time token tracking before the turn completes.  The
 * ``usage`` field is the same provider-agnostic shape used by
 * ``TurnCompletedEvent`` and ``ContextUpdatedEvent``.
 */
export interface TurnProgressEvent {
  type?: EventType37;
  timestamp?: Timestamp37;
  session_id?: SessionId37;
  agent_id?: AgentId27;
  usage?: UsageBreakdown;
  context_limit?: ContextLimit1;
  percent_used?: PercentUsed1;
  tokens_remaining?: TokensRemaining1;
  pending_tool_calls?: PendingToolCalls;
}
/**
 * System message (info, warning, status).
 */
export interface SystemMessageEvent {
  type?: EventType38;
  timestamp?: Timestamp38;
  session_id?: SessionId38;
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
  type?: EventType39;
  timestamp?: Timestamp39;
  session_id?: SessionId39;
  lines?: Lines;
}
/**
 * Initialization progress update.
 *
 * Sent during session initialization to show progress on each step.
 * Steps are shown in sequence with their status.
 */
export interface InitProgressEvent {
  type?: EventType40;
  timestamp?: Timestamp40;
  session_id?: SessionId40;
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
  type?: EventType41;
  timestamp?: Timestamp41;
  session_id?: SessionId41;
  error?: Error3;
  error_type?: ErrorType2;
  recoverable?: Recoverable;
  details?: Details1;
  request_id?: RequestId18;
}
/**
 * API retry notification with exponential backoff.
 *
 * Sent when a transient error (rate limit, server error) is encountered
 * and the system is retrying the request.
 */
export interface RetryEvent {
  type?: EventType42;
  timestamp?: Timestamp42;
  session_id?: SessionId42;
  message?: Message2;
  attempt?: Attempt1;
  max_attempts?: MaxAttempts;
  delay?: Delay;
  error_type?: ErrorType3;
}
/**
 * List of available sessions - for user display.
 */
export interface SessionListEvent {
  type?: EventType43;
  timestamp?: Timestamp43;
  session_id?: SessionId43;
  sessions?: Sessions;
}
/**
 * Garbage collection lifecycle — one event, switched on ``phase``.
 *
 * DISTINCT from :class:`GCConfigEvent`, which carries configuration only
 * (threshold / strategy / target) at init and reconfigure.  This is the
 * lifecycle: GC is about to run, is running, has finished.
 *
 * Before this existed there was NO lifecycle signal on the bus.  The
 * framework opened an OpenTelemetry span with the trigger reason and the
 * strategy, and clients got either prose -- a ``SystemMessageEvent`` reading
 * "Context usage (84.2%) exceeds threshold (80%). GC will run after this
 * turn." -- or nothing at all.  A client wanting to show "compacting..." had
 * to substring-match that sentence for the start and guess at the end, which
 * is the parse-the-log shape typed events exist to replace.
 *
 * Phases:
 *
 * ``about_to_run``
 *     The threshold was crossed; GC will run after this turn.  Carries
 *     ``percent_used`` / ``threshold``.  Announces a FUTURE pass -- the
 *     session keeps serving the current turn.
 * ``started``
 *     A pass is beginning now.  Carries ``trigger_reason`` / ``strategy``
 *     plus the "before" figures.
 * ``completed``
 *     The pass finished.  Carries ``success``, ``items_collected``,
 *     ``tokens_freed``, ``tokens_before`` / ``tokens_after``, and ``error``
 *     when it failed.
 *
 * "Ongoing" is the interval BETWEEN ``started`` and ``completed`` -- a
 * client renders its spinner there.  ``collect()`` is atomic, so there is no
 * sub-pass progress to report and none is invented.
 *
 * Every GC pass emits ``started`` + ``completed``, including a failed one:
 * the failure is the case an operator most needs.  ``about_to_run`` fires
 * only for threshold-triggered passes, since the other triggers (manual,
 * context-limit recovery) have no advance warning by nature.
 */
export interface GCEvent {
  type?: EventType44;
  timestamp?: Timestamp44;
  session_id?: SessionId44;
  agent_id?: AgentId28;
  phase?: Phase;
  trigger_reason?: TriggerReason;
  strategy?: Strategy1;
  percent_used?: PercentUsed2;
  threshold?: Threshold1;
  context_limit?: ContextLimit2;
  success?: Success2;
  items_collected?: ItemsCollected;
  tokens_before?: TokensBefore;
  tokens_after?: TokensAfter;
  tokens_freed?: TokensFreed;
  error?: Error4;
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
  type?: EventType45;
  timestamp?: Timestamp45;
  session_id?: SessionId45;
  request_id?: RequestId19;
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
  type?: EventType46;
  timestamp?: Timestamp46;
  session_id?: SessionId46;
  memories?: Memories1;
}
/**
 * List of sandbox-allowed paths - for @@ completion cache.
 *
 * Emitted after sandbox add/remove commands to refresh the client's
 * completion list for @@ (sandbox path) references.
 */
export interface SandboxPathsEvent {
  type?: EventType47;
  timestamp?: Timestamp47;
  session_id?: SessionId47;
  paths?: Paths;
}
/**
 * List of discovered services - for completion cache.
 *
 * Emitted after services commands to refresh the client's
 * completion list for service names and HTTP methods.
 */
export interface ServiceListEvent {
  type?: EventType48;
  timestamp?: Timestamp48;
  session_id?: SessionId48;
  services?: Services1;
}
/**
 * Session description was updated (by model calling session_describe).
 */
export interface SessionDescriptionUpdatedEvent {
  type?: EventType49;
  timestamp?: Timestamp49;
  session_id?: SessionId49;
  description?: Description;
}
/**
 * List of available agent profiles for session creation.
 *
 * Sent in response to a ``session.profiles`` command.  Each profile
 * is a typed ``ProfileSummary`` carrying enough metadata for a
 * profile picker UI without leaking secrets.
 *
 * Profiles that failed to parse during discovery are reported in
 * ``parse_errors`` rather than mixed into ``profiles`` — a picker
 * can surface them separately or hide them entirely.
 *
 * The shape of this event (and of nested ``ProfileSummary``) is
 * versioned by the global ``ConnectedEvent.protocol_version``.  Pre-
 * 1.0 versions of the SDK had a per-event ``schema_version`` field
 * here; that was promoted to the global protocol version in v1.0 and
 * removed from this event.
 */
export interface SessionProfilesEvent {
  type?: EventType50;
  timestamp?: Timestamp50;
  session_id?: SessionId50;
  profiles?: Profiles;
  parse_errors?: ParseErrors;
}
/**
 * Stable summary of a profile, safe to expose to external clients.
 *
 * Versioned by the global ``ConnectedEvent.protocol_version`` —
 * breaking changes to this shape bump the protocol's MAJOR; additive
 * optional fields bump the MINOR.  Sensitive material is intentionally
 * omitted: env *values* are summarised by name only;
 * ``system_instructions``, ``icon_name`` and ``inherits`` are not
 * exposed (deprecated or already resolved during discovery).
 * Structural config (``plugin_configs``, ``model_tiers``,
 * ``runtime_limits``, ``gc``) is exposed as-is — profile authors are
 * expected to use ``${VAR}`` indirection for secrets and put the
 * actual values in ``env`` (which is summarised by key).
 */
export interface ProfileSummary {
  name: Name;
  description?: Description1;
  plugins?: Plugins;
  preloaded_plugins?: PreloadedPlugins;
  plugin_configs?: PluginConfigs;
  model?: Model;
  provider?: Provider;
  max_turns?: MaxTurns;
  model_tiers?: ModelTiers;
  budget_control?: BudgetControl;
  gc?: Gc;
  runtime_limits?: RuntimeLimits;
  completion_payload_schema?: CompletionPayloadSchema;
  env_var_names?: EnvVarNames;
}
export interface PluginConfigs {
  [k: string]: {
    [k: string]: unknown;
  };
}
export interface ModelTiers {
  [k: string]: unknown;
}
/**
 * Profile file that failed to parse during discovery.
 *
 * Carried in ``SessionProfilesEvent.parse_errors`` (a separate field
 * from ``profiles``) so a picker can surface broken files distinctly
 * rather than treating them as unusable entries in the main list.
 */
export interface ProfileParseError {
  name: Name1;
  error: Error5;
}
/**
 * Send a message to the model.
 */
export interface SendMessageRequest {
  type?: EventType51;
  timestamp?: Timestamp51;
  session_id?: SessionId51;
  text?: Text1;
  attachments?: Attachments;
  parallel_tools?: ParallelTools;
}
/**
 * Respond to a permission request.
 */
export interface PermissionResponseRequest {
  type?: EventType52;
  timestamp?: Timestamp52;
  session_id?: SessionId52;
  request_id?: RequestId20;
  response?: Response2;
  edited_arguments?: EditedArguments;
}
/**
 * Respond to a clarification question.
 */
export interface ClarificationResponseRequest {
  type?: EventType53;
  timestamp?: Timestamp53;
  session_id?: SessionId53;
  request_id?: RequestId21;
  question_index?: QuestionIndex2;
  response?: Response3;
}
/**
 * Stop current operation (cancel generation).
 */
export interface StopRequest {
  type?: EventType54;
  timestamp?: Timestamp54;
  session_id?: SessionId54;
  agent_id?: AgentId29;
}
/**
 * External event injected by the host page via the web component.
 *
 * Published on the session's ``EventBus`` as an ``external_event``
 * so that agents subscribed via ``subscribeToEvents`` are notified.
 */
export interface ExternalEventRequest {
  type?: EventType55;
  timestamp?: Timestamp55;
  session_id?: SessionId55;
  name?: Name2;
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
  type?: EventType56;
  timestamp?: Timestamp56;
  session_id?: SessionId56;
  agent_id?: AgentId30;
  event_names?: EventNames;
}
/**
 * Execute a command (like 'model', 'save', 'resume', etc.).
 *
 * ``args`` carries CLI-style positional/flag arguments — the same
 * array a TUI user would type after the command name.
 *
 * ``payload`` is an opt-in escape hatch for SDK consumers that need
 * to pass structured data argv can't ergonomically carry (e.g.
 * ``session.new`` with an inline profile spec dict).  Commands that
 * accept ``payload`` document their expected keys in the relevant
 * server handler; the wire stays generic.
 *
 * The TUI never produces ``payload`` (its input is always argv-shaped),
 * so commands that *only* read from ``payload`` are SDK-only by
 * construction.
 */
export interface CommandRequest {
  type?: EventType57;
  timestamp?: Timestamp57;
  session_id?: SessionId57;
  command?: Command;
  args?: Args;
  payload?: Payload1;
}
/**
 * Request current instruction budget for an agent.
 *
 * Server responds with InstructionBudgetEvent containing the budget snapshot.
 * If agent_id is None or empty, returns budget for main agent.
 */
export interface GetInstructionBudgetRequest {
  type?: EventType58;
  timestamp?: Timestamp58;
  session_id?: SessionId58;
  agent_id?: AgentId31;
}
/**
 * Request list of available commands from server.
 */
export interface CommandListRequest {
  type?: EventType59;
  timestamp?: Timestamp59;
  session_id?: SessionId59;
}
/**
 * List of available commands from server/plugins.
 */
export interface CommandListEvent {
  type?: EventType60;
  timestamp?: Timestamp60;
  session_id?: SessionId60;
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
  type?: EventType61;
  timestamp?: Timestamp61;
  session_id?: SessionId61;
}
/**
 * Tool status information for client display.
 */
export interface ToolStatusEvent {
  type?: EventType62;
  timestamp?: Timestamp62;
  session_id?: SessionId62;
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
  type?: EventType63;
  timestamp?: Timestamp63;
  session_id?: SessionId63;
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
  type?: EventType64;
  timestamp?: Timestamp64;
  session_id?: SessionId64;
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
  type?: EventType65;
  timestamp?: Timestamp65;
  session_id?: SessionId65;
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
  type?: EventType66;
  timestamp?: Timestamp66;
  session_id?: SessionId66;
  call_id?: CallId4;
  agent_id?: AgentId32;
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
  type?: EventType67;
  timestamp?: Timestamp67;
  session_id?: SessionId67;
  call_id?: CallId5;
  result?: Result1;
  error?: Error6;
}
/**
 * Client request for conversation history.
 */
export interface HistoryRequest {
  type?: EventType68;
  timestamp?: Timestamp68;
  session_id?: SessionId68;
  agent_id?: AgentId33;
}
/**
 * Conversation history from server.
 */
export interface HistoryEvent {
  type?: EventType69;
  timestamp?: Timestamp69;
  session_id?: SessionId69;
  agent_id?: AgentId34;
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
  type?: EventType70;
  timestamp?: Timestamp70;
  session_id?: SessionId70;
  trace_log_path?: TraceLogPath;
  provider_trace_log?: ProviderTraceLog;
  working_dir?: WorkingDir;
  config_root?: ConfigRoot;
  env_file?: EnvFile;
  presentation?: Presentation;
  permission_timeout?: PermissionTimeout;
  apparmor?: Apparmor;
}
/**
 * Sent when a user prompt is queued during model processing.
 *
 * Instead of returning an error when the user sends a message while the model
 * is running, the message is queued and will be injected at the next natural
 * pause point (between tool executions, after subagent completion, etc.).
 */
export interface MidTurnPromptQueuedEvent {
  type?: EventType71;
  timestamp?: Timestamp71;
  session_id?: SessionId71;
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
  type?: EventType72;
  timestamp?: Timestamp72;
  session_id?: SessionId72;
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
  type?: EventType73;
  timestamp?: Timestamp73;
  session_id?: SessionId73;
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
  type?: EventType74;
  timestamp?: Timestamp74;
  session_id?: SessionId74;
  agent_id?: AgentId35;
  recovered_calls?: RecoveredCalls;
  action_taken?: ActionTaken;
}
/**
 * Client requests list of available workspaces.
 */
export interface WorkspaceListRequest {
  type?: EventType75;
  timestamp?: Timestamp75;
  session_id?: SessionId75;
}
/**
 * Response to workspace.list - list of available workspaces.
 */
export interface WorkspaceListEvent {
  type?: EventType76;
  timestamp?: Timestamp76;
  session_id?: SessionId76;
  root?: Root;
  workspaces?: Workspaces;
}
/**
 * Client requests creation of a new workspace.
 */
export interface WorkspaceCreateRequest {
  type?: EventType77;
  timestamp?: Timestamp77;
  session_id?: SessionId77;
  name?: Name3;
}
/**
 * Response to workspace.create - new workspace created.
 */
export interface WorkspaceCreatedEvent {
  type?: EventType78;
  timestamp?: Timestamp78;
  session_id?: SessionId78;
  name?: Name4;
  path?: Path;
}
/**
 * Client selects a workspace to use for the session.
 */
export interface WorkspaceSelectRequest {
  type?: EventType79;
  timestamp?: Timestamp79;
  session_id?: SessionId79;
  name?: Name5;
}
/**
 * Response to workspace.select - configuration status of selected workspace.
 */
export interface ConfigStatusEvent {
  type?: EventType80;
  timestamp?: Timestamp80;
  session_id?: SessionId80;
  workspace?: Workspace;
  configured?: Configured;
  provider?: Provider1;
  model?: Model1;
  available_providers?: AvailableProviders;
  missing_fields?: MissingFields;
}
/**
 * Client updates workspace configuration (provider, model, API key).
 */
export interface ConfigUpdateRequest {
  type?: EventType81;
  timestamp?: Timestamp81;
  session_id?: SessionId81;
  provider?: Provider2;
  model?: Model2;
  api_key?: ApiKey;
}
/**
 * Response to config.update - configuration was updated.
 */
export interface ConfigUpdatedEvent {
  type?: EventType82;
  timestamp?: Timestamp82;
  session_id?: SessionId82;
  workspace?: Workspace1;
  provider?: Provider3;
  model?: Model3;
  success?: Success3;
  error?: Error7;
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
  type?: EventType83;
  timestamp?: Timestamp83;
  session_id?: SessionId83;
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
  type?: EventType84;
  timestamp?: Timestamp84;
  session_id?: SessionId84;
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
  type?: EventType85;
  timestamp?: Timestamp85;
  session_id?: SessionId85;
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
  name?: Name6;
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
  type?: EventType86;
  timestamp?: Timestamp86;
  session_id?: SessionId86;
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
  type?: EventType87;
  timestamp?: Timestamp87;
  session_id?: SessionId87;
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
  type?: EventType88;
  timestamp?: Timestamp88;
  session_id?: SessionId88;
  request_id?: RequestId22;
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
  type?: EventType89;
  timestamp?: Timestamp89;
  session_id?: SessionId89;
  request_id?: RequestId23;
  remote_agent_id?: RemoteAgentId;
}
/**
 * Notification that a remote peer rejected the spawn request.
 *
 * The ``reason`` field contains a human-readable explanation (e.g.
 * capacity limits, missing provider, unknown profile).
 */
export interface PeerSpawnRejectedEvent {
  type?: EventType90;
  timestamp?: Timestamp90;
  session_id?: SessionId90;
  request_id?: RequestId24;
  reason?: Reason1;
}
/**
 * Streamed output chunk from a remote subagent.
 *
 * Sent from the remote server back to the origin as the subagent
 * produces output.  The origin's ``RemoteSpawnHandler`` forwards these
 * to the parent session via ``inject_prompt``.
 */
export interface PeerAgentOutputEvent {
  type?: EventType91;
  timestamp?: Timestamp91;
  session_id?: SessionId91;
  request_id?: RequestId25;
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
  type?: EventType92;
  timestamp?: Timestamp92;
  session_id?: SessionId92;
  request_id?: RequestId26;
  remote_agent_id?: RemoteAgentId2;
  success?: Success4;
  summary?: Summary;
  error?: Error8;
  workspace_modified?: WorkspaceModified;
}
/**
 * Request to cancel a running remote subagent.
 *
 * Sent from the origin server when the parent session wants to stop
 * a previously spawned remote subagent.
 */
export interface PeerStopRequestEvent {
  type?: EventType93;
  timestamp?: Timestamp93;
  session_id?: SessionId93;
  request_id?: RequestId27;
  remote_agent_id?: RemoteAgentId3;
}
/**
 * Confirmation that a remote peer received and processed the stop request.
 */
export interface PeerStopAcknowledgedEvent {
  type?: EventType94;
  timestamp?: Timestamp94;
  session_id?: SessionId94;
  request_id?: RequestId28;
  remote_agent_id?: RemoteAgentId4;
}
/**
 * A reactor producer announced its intent on a held HandoffGate.
 *
 * Fired after a producer reactor calls ``gate.try_acquire(...)`` and
 * then ``gate.announce(intent)``.  When ``intent.session_id`` is set,
 * subscribers can ``client.attach_session(intent['session_id'])`` to
 * observe the spawned session's events.
 */
export interface GateAnnouncedEvent {
  type?: EventType95;
  timestamp?: Timestamp95;
  session_id?: SessionId95;
  gate_name?: GateName;
  tenant_id?: TenantId;
  owner?: Owner;
  intent?: Intent;
  announced_at?: AnnouncedAt;
}
export interface Intent {
  [k: string]: unknown;
}
/**
 * A held HandoffGate was released (work completed, failed, or timed out).
 *
 * ``was_announced=False`` indicates the producer crashed or errored
 * between ``try_acquire`` and ``announce`` — subscribers that
 * auto-attached on the announce event simply have nothing to detach.
 * ``outcome.status='timeout'`` indicates the watchdog auto-released
 * on TTL expiry.
 */
export interface GateReleasedEvent {
  type?: EventType96;
  timestamp?: Timestamp96;
  session_id?: SessionId96;
  gate_name?: GateName1;
  tenant_id?: TenantId1;
  owner?: Owner1;
  outcome?: Outcome;
  released_at?: ReleasedAt;
  was_announced?: WasAnnounced;
}
/**
 * All currently-RED gates, sent on subscribe so late subscribers catch up.
 *
 * Mirrors ``SessionInfoEvent`` for sessions: rather than forcing
 * every subscriber to track gate state externally across reconnects,
 * the registry replays the live state once at subscription time.
 */
export interface GatesSnapshotEvent {
  type?: EventType97;
  timestamp?: Timestamp97;
  session_id?: SessionId97;
  gates?: Gates;
  snapshot_at?: SnapshotAt;
}
/**
 * Snapshot of a single gate's state.
 *
 * Used as the payload-level shape inside ``GatesSnapshotEvent`` and
 * accessible as a typed property on the live events.  Public/private
 * intent split is enforced server-side via ``public_intent_fields`` —
 * cross-tenant subscribers receive only the public keys; same-tenant
 * subscribers receive the full intent.
 *
 * See ``jaato-premium/docs/design/handoff-gate-api.md`` §3.4 for the canonical
 * intent shape.
 */
export interface GateState {
  gate_name?: GateName2;
  tenant_id?: TenantId2;
  state?: State;
  owner?: Owner2;
  intent?: Intent1;
  acquired_at?: AcquiredAt;
  expires_at?: ExpiresAt;
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
  type?: EventType98;
  timestamp?: Timestamp98;
  session_id?: SessionId98;
  text?: Text5;
  source_type?: SourceType;
  source_id?: SourceId;
  request_id?: RequestId29;
}
/**
 * Server's response to :class:`InjectPromptRequest`.
 *
 * Answers the only question an injecting caller actually has: **after
 * this call, will the target act on the message?**  The pre-1.3 verb
 * could not answer it — the runner's ``{"ok": True}`` was discarded by
 * the daemon and the SDK method returned ``None`` — so a driver got the
 * same silence whether its target was busy, idle, stranded, or dead.
 *
 * ``status`` is one of the constants in ``shared.message_delivery``:
 *
 * * ``"accepted"``    — the target was idle, so a turn was STARTED on it.
 * * ``"queued"``      — the target is mid-turn; its running turn will
 *   drain the message.
 * * ``"terminated"``  — the target is loaded but terminal and will run no
 *   further turns.  Reported from the target's own terminal stamp, never
 *   inferred from silence.
 * * ``"no_session"``  — no session with that id is loaded.
 * * ``"unreachable"`` — loaded and live, but NOTHING WAS SENT: no server
 *   attached, no runner channel, a runner too old to accept the offer verb,
 *   or a drive that failed.  A transport fault, not a decision by the
 *   target.  **Re-sending is safe** — nothing was enqueued, so it cannot
 *   duplicate — though it will keep failing until the path is restored.
 * * ``"not_confirmed"`` — an offer WAS made and its answer was lost (the
 *   call raised or timed out).  The message may be in the target's queue
 *   right now, or may never have arrived; from here those are
 *   indistinguishable.  **Re-sending may deliver it twice.**
 *
 * A consumer that only checks membership of the delivered set stays correct
 * when a word is added here -- which is how ``not_confirmed`` was split out
 * of ``unreachable`` without touching a caller.  Branch on the set, not on
 * the individual failure words, unless you are choosing whether to retry.
 *
 * Only ``accepted`` and ``queued`` mean the message will be acted on
 * (``shared.message_delivery.DELIVERED``).  The rest are failures and must
 * not be read as success: a caller that assumes delivery and is wrong gets
 * a silent stall it cannot attribute, which is the expensive direction to
 * be wrong in.
 *
 * ``detail`` carries a human-readable elaboration when one exists.  It is
 * **omitted rather than filled with a placeholder** when there is nothing
 * to say — a reader of ``"unknown"`` is back where they started, so
 * absence is left checkable instead of forgeable.
 */
export interface InjectPromptResultEvent {
  type?: EventType99;
  timestamp?: Timestamp99;
  session_id?: SessionId99;
  request_id?: RequestId30;
  status?: Status3;
  detail?: Detail;
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
  type?: EventType100;
  timestamp?: Timestamp100;
  session_id?: SessionId100;
  request_id?: RequestId31;
  messages?: Messages;
  timeout_seconds?: TimeoutSeconds;
}
/**
 * Server's response to :class:`ReplayMessagesRequest`.
 */
export interface ReplayMessagesResultEvent {
  type?: EventType101;
  timestamp?: Timestamp101;
  session_id?: SessionId101;
  request_id?: RequestId32;
  response_text?: ResponseText;
  error?: Error9;
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
  type?: EventType102;
  timestamp?: Timestamp102;
  session_id?: SessionId102;
  request_id?: RequestId33;
  after_message?: AfterMessage;
  after_tool_call?: AfterToolCall;
  after_timestamp?: AfterTimestamp;
}
/**
 * Server's response to :class:`ResolveForkPointRequest`.
 */
export interface ResolveForkPointResultEvent {
  type?: EventType103;
  timestamp?: Timestamp103;
  session_id?: SessionId103;
  request_id?: RequestId34;
  fork_index?: ForkIndex;
  error?: Error10;
}
/**
 * Server returns the result of ``session.bind_wake`` / ``session.unbind_wake``.
 *
 * ``outcome`` is the ``BindOutcome`` value (``ok`` / ``unauthorized`` /
 * ``malformed_key`` / ``too_many_keys`` / ``no_keys`` / ``no_session`` /
 * ``unknown``); route on it, not ``detail``.  On a successful ``bind_wake``,
 * ``wake_ref`` echoes the (session-supplied) ref and ``expires_at`` is the
 * binding's Unix expiry — the values the caller's waker keys on.
 */
export interface WakeBindResultEvent {
  type?: EventType104;
  timestamp?: Timestamp104;
  session_id?: SessionId104;
  wake_ref?: WakeRef;
  outcome?: Outcome1;
  detail?: Detail1;
  expires_at?: ExpiresAt1;
  endpoint?: Endpoint;
}
/**
 * A wake arrived for a session with NO attached client; the daemon revived
 * it and DEFERRED the turn until a client re-attaches.
 *
 * Routed to the session's cascade observers (a connected-but-detached client
 * that registered ``cascade.register(cid, "observer", ["SessionWokenEvent"])``
 * — the cascade filter matches on the event's CLASS NAME
 * (``type(event).__name__``), NOT the ``EventType`` value ``"session.woken"``;
 * registering the value string silently never matches), so a bot whose session
 * went cold can learn it must re-attach to serve the
 * woken turn's host tools + render.  Re-emitted whenever an observer
 * (re)registers for the cid while a wake is still pending, so a reconnecting
 * bot is re-nudged.
 *
 * Filter client-side by ``session_id`` (map it to your chat / attach target).
 * ``wake_ref`` names the matter (e.g. the PR); ``source`` is the provenance
 * tag.  The wake TEXT is NOT here — it stays inside the deferred turn (the
 * notification is a signal to attach, not the untrusted payload).
 */
export interface SessionWokenEvent {
  type?: EventType105;
  timestamp?: Timestamp105;
  session_id?: SessionId105;
  wake_ref?: WakeRef1;
  source?: Source2;
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
  type?: EventType106;
  timestamp?: Timestamp106;
  session_id?: SessionId106;
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
  type?: EventType107;
  timestamp?: Timestamp107;
  session_id?: SessionId107;
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
  type?: EventType108;
  timestamp?: Timestamp108;
  session_id?: SessionId108;
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
  type?: EventType109;
  timestamp?: Timestamp109;
  session_id?: SessionId109;
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
  type?: EventType110;
  timestamp?: Timestamp110;
  session_id?: SessionId110;
  policy?: Policy;
}
/**
 * Request a structured snapshot of the current permission policy.
 */
export interface PermissionPolicySnapshotRequest {
  type?: EventType111;
  timestamp?: Timestamp111;
  session_id?: SessionId111;
  request_id?: RequestId35;
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
  type?: EventType112;
  timestamp?: Timestamp112;
  session_id?: SessionId112;
  request_id?: RequestId36;
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
  AGENT_ERROR: "agent.error",
  SESSION_TERMINATED: "session.terminated",
  SLOT_SETTLED: "slot.settled",
  SESSION_RESTORED: "session.restored",
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
  GC_CONFIG: "gc.config",
  GC: "gc",
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
  INJECT_PROMPT_RESULT: "inject_prompt.result",
  REPLAY_MESSAGES_REQUEST: "replay_messages.request",
  REPLAY_MESSAGES_RESULT: "replay_messages.result",
  RESOLVE_FORK_POINT_REQUEST: "resolve_fork_point.request",
  RESOLVE_FORK_POINT_RESULT: "resolve_fork_point.result",
  WAKE_BIND_RESULT: "session.wake_bind_result",
  SESSION_WOKEN: "session.woken",
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
  GATE_ANNOUNCED: "gate.announced",
  GATE_RELEASED: "gate.released",
  GATES_SNAPSHOT: "gates.snapshot",
} as const;
