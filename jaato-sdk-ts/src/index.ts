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
