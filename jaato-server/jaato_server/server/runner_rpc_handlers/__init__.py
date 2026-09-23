"""Daemon-side runner→daemon RPC handlers (Phase 3 task 3.2).

Each module hosts one handler.  Handlers are async fns taking a
single ``args: dict`` and returning a result-shaped value (typically
a dict).  They register against
:class:`server.runner_rpc_server.RunnerRPCServer` at session-init.

Modules:

- :mod:`server.runner_rpc_handlers.prompt_operator` — relays ASK
  prompts to the connected client (§3.2.1).  Used by the runner-side
  permission plugin when it migrates in §3.7.
- :mod:`server.runner_rpc_handlers.apparmor_fragment` — kernel-side
  fragment load for the running session's profile (§3.2.2).  Used
  by the runner-side references plugin when it migrates in §3.8.
- ``telemetry`` — Phase 3 §3.15 cleanup target; runner publishes
  OTel sub-spans to the daemon's forwarder.  Not yet implemented.
"""
