# Design + Impact Analysis: notebook subprocess kernel (option 1c)

**Status:** Proposed (design under discussion). **Author:** advisor + Daniel.
**Date:** 2026-06-22. **Surfaced by:** enphase telegram self-extending bot —
notebook code's relative paths escape the workspace.

## Problem

`notebook` is `PLUGIN_TIER="runner"` and `LocalJupyterBackend` executes cells
**in-process** in the runner via `exec(compile(cell), namespace)`
(`backends/local.py:263-276`). So notebook code's `os.getcwd()` and relative
paths resolve against the **runner process CWD = the daemon launch dir**, not the
session `workspace_root`.

`file_edit`/`cli` are unaffected — they never use the process cwd (`file_edit`
resolves against the workspace ContextVar from #344; `cli` passes
`cwd=self._workspace_root`). But arbitrary notebook code uses the raw process
cwd, so `os.makedirs('tool_drafts'); open('tool_drafts/x','w')` lands at
`<launch-dir>/tool_drafts/` — **outside the workspace, silently.** (This is why
glm-5-turbo's `tool_drafts/<name>.py` writes never reached `file_edit`/
`register_tool` and it thrashed on "no write access.")

The obvious fix — `os.chdir(workspace_root)` — is **exactly what the framework
deliberately avoids** (`core.py:915`: *"Does NOT call os.chdir() — that is
process-global and not thread-safe"*; it races parallel tools). **1c** removes
that hazard by giving the notebook its own process whose CWD *is* the workspace.

## Design (1c)

Replace `LocalJupyterBackend`'s in-process `exec` with a **per-notebook
subprocess kernel** launched with `cwd=workspace_root`. State (the namespace)
lives in the kernel process; cells are dispatched to it over a pipe.

```
runner (jaato_session, confined)                  kernel subprocess (cwd=workspace, confined)
  notebook_execute(code)                              loop: recv cell → exec(cell, ns) → send result
    code_analyzer.gate(code)  ─ stays runner-side
    backend.execute(nb_id, code) ──cell──▶  [pipe]  ──▶  exec in persistent namespace
                                  ◀──result/stream──            os.getcwd()==workspace ✓
    tools.X(args)  ◀──tool-RPC──  [pipe]  ◀── the kernel's `tools` proxy
      executor_fn(name,args)  ─ runs runner-side  ──result──▶
```

Four pieces (the abstraction already exists — `NotebookBackend` protocol with
`LocalJupyterBackend`/`KaggleBackend`; 1c is a new local backend or a subprocess
mode on `LocalJupyterBackend`):

1. **Kernel process** — a small Python entrypoint: `os.chdir(workspace_root)`
   (safe — it's the kernel's *own* process), then a recv/exec/send loop over a
   pipe, holding the persistent namespace. PDEATHSIG=SIGKILL so it dies with the
   runner (per `feedback_runner_child_heavyweight_proc_needs_pdeathsig`).
2. **Cell channel** — dispatch a cell + stream stdout/stderr/results back. Mirror
   the existing `execute_streaming` shape over the pipe.
3. **Cross-process `tools` bridge** — the kernel's `tools.X()` can no longer hold
   the runner's `executor_fn` directly. It serialises `(name, args)` over the
   pipe; the runner runs `executor_fn` and returns the result. **This is the
   hardest part** (see impact §1).
4. **Permission propagation** — `tools.X()` from notebook code must inherit the
   `notebook_execute` approval (the `trusted_bridge_context` thread-local,
   `plugin.py:766`). The tool-RPC carries a "trusted-bridge" marker so the
   runner sets that context for the kernel-originated call.

## Impact analysis

1. **Cross-process tool bridge (largest).** Today `ToolBridge.__init__(executor_fn)`
   is an in-process callable; `tools.X()` → `executor_fn(name,args)` →
   `(ok, result)`. Cross-process needs a wire protocol (request `(name,args)` /
   response `(ok,result|error)`), JSON-serialisable args+results (tool results are
   generally JSON-able; **binary/large results — images, file bytes — need
   framing**), and `ToolExecutionError` propagation. It's a mini-RPC, comparable
   in spirit to (but smaller than) the daemon↔runner `RunnerRPC`. **Reuse:** model
   it on the existing `RunnerRPC` length-prefixed framing rather than inventing one.
2. **Permission context.** The runner-side tool call triggered *by the kernel* must
   run inside `trusted_bridge_context()` so inner tools are pre-approved (same as
   today's in-process call). The tool-RPC handler on the runner wraps the
   `executor_fn` call in that context. Low risk, but must not let a *non*-bridge
   path forge the marker (the channel is private kernel↔runner, so the marker is
   trusted by construction).
3. **Streaming.** `execute_streaming` (`plugin.py:1019`) must pump kernel
   stdout/stderr/result frames back over the pipe instead of capturing in-process.
   Moderate.
4. **State.** *Improves* — the namespace lives in the kernel and persists across
   cells by construction (today it's a dict the runner holds).
5. **`code_analyzer`.** Unchanged — risk-gating runs runner-side on the cell text
   *before* dispatch. ✓
6. **AppArmor / confinement.** Net neutral-to-positive. The kernel is a runner
   *child* → inherits the runner's session profile (confined to the workspace).
   The peer's "relative write reached the parent" was **not** a breach — it was
   CWD-in-an-allowed-dir (the launch dir is the runner's own dir). With 1c the CWD
   *is* the workspace, so relative writes stay in-workspace; the separate
   least-privilege question ("should the profile allow the launch dir at all")
   is orthogonal and unaffected by 1c.
7. **Lifecycle / cleanup.** New: spawn per notebook (or per session), reap on
   `notebook_reset`/session-end, PDEATHSIG, zombie reaping. Follows the
   jdtls/LSP runner-child precedent (PR-277/#285) — known territory, real work.
8. **Performance.** (a) kernel spawn ≈ Python interpreter startup per notebook —
   the runner pre-warm-pool idea could amortise if it matters. (b) **Every
   `tools.X()` becomes an RPC round-trip** vs an in-process call — adds latency to
   tool-heavy notebooks. Cell exec itself is unaffected.
9. **Dependencies.** Two sub-options: a **custom** subprocess kernel (a Python
   entrypoint + the pipe protocol — no new dep, ~all custom) vs a **real jupyter
   kernel** (`ipykernel`/`jupyter_client` — gains kernel infra + ZMQ messaging,
   but a heavy dep and the `tools` bridge is still custom on top). Recommend
   **custom** — the bridge is custom either way, and jupyter's weight buys little.
10. **Migration / blast radius.** `LocalJupyterBackend` is the default local
    backend; tests target it. Cleanest: add a `SubprocessKernelBackend` behind the
    `NotebookBackend` protocol, default to it, keep in-process `exec` as a
    fallback/feature-flag for one release. `KaggleBackend` (remote) is untouched.

## Effort

Multi-PR. Rough shape: (1) kernel entrypoint + pipe protocol + cell exec/stream;
(2) cross-process `tools` bridge + permission propagation; (3) lifecycle/reaping/
PDEATHSIG; (4) migration + backend selection + tests. The tool bridge (2) is the
risk concentrate.

## Lighter alternative discovered during analysis — 1d

**1d: in-process, namespace path-interception.** Keep in-process `exec`, but
inject workspace-aware `open` / `os.getcwd` / `os.makedirs` / `os.path.abspath`
into the notebook namespace that resolve relative paths against `workspace_root`.
No subprocess, no `os.chdir`, no bridge re-architecture.

- ✅ Tiny vs 1c; no process-global chdir; no tool-bridge rewrite.
- ⚠️ **Best-effort, not airtight** — catches the common path APIs but not deep
  library usage, dynamic path construction, or `subprocess(cwd=...)` defaults. A
  library that internally calls `os.getcwd()` still sees the launch dir unless we
  also patch `os.getcwd` in the namespace (which only affects direct lookups, not
  C-level `getcwd`).

## Recommendation

1c is the correct, complete fix and the only one that's both thread-safe and
airtight, but it's a genuine multi-PR re-architecture whose risk sits in the
cross-process tool bridge. If the bot's near-term need is the driver, **1d** (or
the scoped **1a** chdir, accepting its non-parallel-safety) unblocks it cheaply
while 1c is done properly.

**Decision (2026-06-22): build 1c — the correct architectural fix.** The bot
working is not urgent; do it properly.

## Build plan

Additive + opt-in until the last step, so the default (`LocalJupyterBackend`,
in-process) never regresses while 1c lands.

**Wire protocol** (kernel ↔ runner; length-prefixed JSON frames, modelled on
`RunnerRPC`'s framing). New `notebook/kernel_protocol.py`:

| Direction | Frame | Purpose |
|-----------|-------|---------|
| runner→kernel | `{type:"execute", cell_id, code}` | dispatch a cell |
| runner→kernel | `{type:"variables"\|"reset"\|"shutdown"}` | introspect / lifecycle |
| runner→kernel | `{type:"tool_result", call_id, ok, result\|error}` | answer a kernel tool call |
| kernel→runner | `{type:"stream", cell_id, name:"stdout"\|"stderr", text}` | streamed output |
| kernel→runner | `{type:"result", cell_id, status, value, execution_count}` | cell done |
| kernel→runner | `{type:"error", cell_id, ename, evalue, traceback}` | cell raised |
| kernel→runner | `{type:"tool_call", call_id, name, args}` | notebook `tools.X()` → bridge |

The **tool flow** interleaves with cell exec: the kernel, mid-`exec`, emits
`tool_call` and blocks reading for the matching `tool_result`; the runner runs
`executor_fn(name,args)` inside `trusted_bridge_context()` and replies. Binary/
large results are base64-framed (reuse the #353 attachment convention).

**PR sequence**

- **PR 1 — kernel + transport (no tool bridge).** New `notebook/kernel_main.py`
  (entrypoint: `os.chdir(workspace_root)`, then the recv/exec/stream loop over an
  fd pair, persistent namespace) + `kernel_protocol.py` + a new
  `backends/subprocess_kernel.py` (`SubprocessKernelBackend(NotebookBackend)`
  implementing the 12 protocol methods by talking to the kernel; spawns with
  `cwd=workspace_root`, `start_new_session`, PDEATHSIG). `tools` in the kernel
  namespace is a stub that raises `ToolExecutionError("tools bridge lands in
  PR 2")`. **Opt-in** via `plugin_configs.notebook.backend: "subprocess"` (default
  stays `"local"`). Validates the CWD fix (`os.getcwd()==workspace`, relative
  writes in-workspace), persistent state across cells, and streaming — without the
  bridge risk. Tests: a kernel-protocol round-trip + a backend cell-exec/CWD test.
- **PR 2 — cross-process tools bridge + permission.** The `tool_call`/`tool_result`
  frames; the kernel's `tools` proxy (reuse `tool_stubs.generate_tools_module`,
  swap the in-process `ToolBridge` for a frame-RPC one); the runner-side handler
  wraps `executor_fn` in `trusted_bridge_context()`. Now notebooks using `tools.X()`
  work in subprocess mode. Tests: bridged `tools.X()` round-trip + permission
  inheritance + error propagation.
- **PR 3 — lifecycle + cutover.** PDEATHSIG/reaping hardening (the jdtls/LSP
  precedent, `feedback_runner_child_heavyweight_proc_needs_pdeathsig`); optional
  pre-warm; make `"subprocess"` the **default** backend; migrate the
  `LocalJupyterBackend` tests; keep in-process `exec` behind
  `backend: "local-inprocess"` for one release as a fallback. Then the #344-class
  CWD escape is closed airtight.

**Acceptance:** the peer's 3-line repro — `os.makedirs('tool_drafts');
open('tool_drafts/x','w')` from notebook code under `apparmor:true` lands *inside*
the workspace, and `tools.X()` still works.
