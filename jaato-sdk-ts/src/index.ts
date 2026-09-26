// Public surface of @jaato/sdk.
//
// Phase 2 ships only the codegen-generated event/request types
// (mirror of jaato-sdk's events.py, kept in lockstep via
// scripts/codegen_ts_events.py).  Phase 3 adds the JaatoClient
// class that wraps the WS protocol method-for-method with the
// Python jaato_sdk.client.IPCClient.
//
// See ../jaato-sdk-ts/README.md and the parity backlog at
// project_backlog_sdk_feature_parity.md for the full design.
export * from "./events.js";
export * from "./helpers.js";
export * from "./errors.js";
export * from "./state.js";
export type {
  CatchallEventHandler,
  EventByType,
  EventHandler,
  SubscribeManyMap,
  Unsubscribe,
} from "./event-typing.js";
export {
  JaatoClient,
  MIN_PROTOCOL_VERSION,
  MIN_ATTACHMENT_RESUME_PROTOCOL,
  MIN_WORKSPACE_IGNORE_PROTOCOL,
  MIN_FILE_FETCH_PROTOCOL,
  MIN_MEMORY_VERBS_PROTOCOL,
  MIN_DIAGNOSTICS_PROTOCOL,
  MIN_WORKSPACE_PICKER_PROTOCOL,
  STAGE_FILES_TIMEOUT_MS,
  LEGACY_SERVER_LIMITS,
  serverLimitsFrom,
  type ServerLimits,
  isProtocolCompatible,
  type WorkspaceFileFetchResult,
  type JaatoClientOptions,
  type TokenProvider,
} from "./client.js";
// app:// secret resolution, application side (#1226): the hook an application
// registers on its bind channel to answer the daemon's secret.resolve.
export {
  SecretResolveResponder,
  type SecretResolveHandler,
  type SecretResolveOutcome,
} from "./secretResolver.js";
// High-level convenience facade (mirror of jaato-sdk's convenience.py).
export {
  Session,
  openSession,
  ask,
  type SessionOpenOptions,
  type AskOptions,
  type CompleteOptions,
  type ClientToolSpec,
  type ClientToolHandler,
} from "./convenience.js";
