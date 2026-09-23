# Jaato vs SwarmForge: A Comparison

Compared against [`unclebob/swarm-forge`](https://github.com/unclebob/swarm-forge)
at `main` commit `f4f5fbc` (2026-09-04), plus the `six-pack` product branch
for a concrete role configuration. File references below are to that
snapshot. SwarmForge changes quickly, so check any claim here that you
depend on against the branch you install.

## Executive Summary

The two projects work at **different layers**, so this is not a
like-for-like comparison.

- **SwarmForge** coordinates **existing agent CLIs** (`claude`, `codex`,
  `grok`, `copilot`) as black boxes. Each role runs in its own git worktree
  and tmux session. Roles pass work to each other as **git commits**,
  announced through a file-based handoff queue, and a local dashboard
  gives the operator a human control point. It owns the *process*: roles,
  order, audit, merge rules and the engineering constitution. It does not
  own the agent loop.
- **Jaato** is the **agent runtime**. It owns the model loop, the provider
  abstraction, tool execution, per-tool permissions, sandboxing, context
  GC, budgets, session persistence and the client protocol. Multi-agent
  work is built from in-process subagents and cascades of sessions that
  share one runtime.

In short, SwarmForge is a team of separate agent programs that talk
through git. Jaato is the machinery inside one agent program, with
primitives for building teams of them.

| Dimension | Jaato | SwarmForge |
|---|---|---|
| **Layer** | Agent runtime + orchestration framework | Workflow harness over third-party agent CLIs |
| **Agent loop** | Owned (`JaatoSession`, 20+ provider plugins) | Delegated to `claude` / `codex` / `grok` / `copilot` |
| **Unit of hand-off** | Tool result, subagent message, cascade stage payload | A git commit, named by a validated 10-char SHA |
| **Isolation between agents** | Separate sessions; runner subprocess per session; AppArmor / cgroups optional | A git worktree and a tmux session per role |
| **Per-tool permission gate** | Yes: `permission` plugin (ask / allow / deny, whitelists, evaluators) | No. Backends start with `--yolo` / `--permission-mode bypassPermissions` |
| **Human gates** | Permission ASK, `request_clarification`, completion gates | Dashboard approvals, clarifications, "Attention" |
| **Process discipline** | Opt-in (`completion_processors`, `budget_control`, payload schemas) | Built in: constitution articles, role prompts, audit re-submit, mandated quality tools |
| **Clients** | TUI, web client, SDK (Python/TS), IPC + WebSocket | Local web dashboard (`127.0.0.1`) + tmux panes |
| **Stack** | Python daemon | zsh, git, tmux, Babashka (Clojure) |
| **Licence** | BUSL-1.1 (Apache-2.0 from the change date) | No licence file in the repository at this commit |

---

## 1. Architecture

### SwarmForge: agent CLIs coordinated through git

`swarmforge.bb` reads `swarmforge/swarmforge.conf`, creates one git
worktree per role (`.worktrees/<name>`, with `master` meaning the main
checkout), starts a private tmux server, and launches the configured CLI
in each role's worktree with an appended system-prompt file. The
`six-pack` configuration shows what this looks like:

```text
window-invisible specifier codex master --yolo
window-invisible coder grok coder
window-invisible cleaner grok cleaner batch back-one
window-invisible architect grok architect batch back-all
window-invisible hardender codex hardender batch --yolo
window-invisible QA grok QA batch back-all
```

The file order is the pipeline. Each role commits its work and sends a
`git_handoff` naming that commit. The next role merges it
(`merge_and_process.sh`), does its own work, commits, and forwards. The
last role *broadcasts* to every other role, which marks the board card
done.

The transport is a small, durable filesystem protocol
(`swarmforge/handoff-protocol.md`):

```text
.swarmforge/handoffs/
  outbox/ outbox/tmp/ sent/ failed/
  inbox/new/ inbox/in_process/ inbox/completed/
```

Queue state is the file's location. A Babashka daemon (`handoffd.bb`)
copies outbox files into recipient inboxes and sends each recipient a
generic tmux wake-up ("You have new handoff mail…"). The wake-up
deliberately does not name the file, so the recipient always takes work
in queue order.

Above the packs sit two optional control layers. A **forge**
(`project-manager`, `lieutenant` branches) runs several projects under one
dashboard with a "lieutenant" agent. A **platoon** (`platoon-brainstorm.md`,
design stage) runs several squads, each an independently deployable
component, under one integrating lieutenant.

### Jaato: a runtime with multi-agent primitives

Jaato runs as a daemon (`python -m server`). Clients connect over IPC or
WebSocket. Each session runs in a runner subprocess, usually taken from a
pre-warmed pool, with its own `JaatoSession` over a shared
`JaatoRuntime` (providers, plugin registry, permissions, ledger).
Multi-agent work uses:

- **Subagents** (`spawn_subagent`): child sessions in the same runtime.
  They require a named profile, and their outcomes flow back to the
  parent as tool results.
- **Cascades**: a driver chains sessions stage by stage. Pool slots keep
  warm, slot-scoped plugin state between stages (`TRAIT_SLOT_SCOPED`, #890).
- **Completion gates**: `signal_completion` with a payload schema and
  `completion_processors`, so a stage can only end with a validated,
  machine-readable result.

Where SwarmForge's contract between agents is *"here is a commit, merge
it"*, jaato's is *"here is a structured payload that passed a schema and
a processor"*.

---

## 2. Who owns the agent loop

This is the most important difference.

**SwarmForge** treats the agent as a subprocess it cannot see into.
`known-agents` is fixed at `#{"claude" "codex" "copilot" "grok"}`
(`swarmforge.bb:147`), and the launcher builds each command line. It
cannot see token usage, tool calls, context size or model output except
as terminal text in a tmux pane. It controls agents **by instruction**
(the constitution and role prompts) and **by gating what leaves the
agent** (commit validation, audit, handoff schema, board state).

**Jaato** owns every step between the model and the side effect. Every
tool call goes through `ToolExecutor` → permission plugin → executor, and
is traced, metered and cancellable. This is what makes per-tool
containment, `budget_control` ceilings, `on_unmetered` policies, prompt
caching, GC, reasoning replay and provider fallbacks possible. SwarmForge
cannot offer these because it never sees a tool call.

In return, SwarmForge gets each vendor's newest CLI features for free, and
you can use your existing subscriptions with it (one of jaato's
providers, `claude_cli`, wraps Claude Code for the same reason).

---

## 3. Safety model

These are opposite designs, and both are deliberate.

| Concern | Jaato | SwarmForge |
|---|---|---|
| Tool-level approval | `permission` plugin; every call gets a decision, logged with `method=` and `asked=` (#951) | None. `swarmforge.bb:470-476` adds `--yolo` / `--permission-mode bypassPermissions` to every backend |
| Filesystem boundary | Workspace containment in `cli`, `interactive_shell`, `notebook`; AppArmor per-session profiles (free server) | A git worktree per role, enforced only by the constitution ("Work only in your assigned branch or worktree") |
| Secrets in child processes | `scrub_secret_env` on by default (#863) | Inherited from the launching shell |
| Resource caps | `runtime_limits` (cgroup memory/pids/cpu, tool timeouts, output caps, wall-clock bounds) | None of its own |
| Where the human intervenes | Per tool call (ASK), per question, per completion | Per card: approve / reject / retry the handoff, answer clarifications |

SwarmForge accepts full agent autonomy inside a worktree and puts its
control points at the **commit boundary**. Every forward goes through
`swarm_handoff.sh`, which checks the draft, confirms the commit is real
and unambiguous, and then, as a deliberate friction step, refuses the
first submission with `AUDIT_REQUIRED`. The agent must re-read its
inbound task, trace every requirement to evidence, and submit the
unchanged candidate a second time before anything is queued. The board
counts these audit challenges per card.

Jaato puts its control points on **individual tool calls** and makes the
commit boundary optional (a completion processor can run tests, but none
is required). It suits agents that are allowed to act outside a
disposable worktree, or that are driven by people who are not watching
them.

---

## 4. Process discipline

SwarmForge has strong opinions about process, which jaato mostly leaves
to the application:

- **A layered constitution** (`swarmforge/constitution/articles/`):
  `engineering.prompt`, `workflow.prompt` and `handoffs.prompt` come from
  `main` and cannot be overridden by a pack; packs only add
  `project.prompt` / `local-*.prompt`.
- **Mandated external quality tools**: CRAP, mutation and DRY analysis
  from `github.com/unclebob/...` for each language, plus the
  Acceptance-Pipeline-Specification Gherkin tooling. These are installed
  fresh at agent startup and must run one at a time.
- **Fixed role pipelines**: two-pack (`coder → cleaner`), four-pack
  (`specifier → coder → refactorer → architect`), six-pack (adds
  `hardender`, `QA`). The Lieutenant chooses a pack by component
  complexity.
- **Always forward**: an intermediate role must forward even if it changed
  nothing, so every role sees every card.
- **Back-propagation**: `back-one` / `back-all` send merge-only copies to
  earlier roles, so upstream worktrees follow downstream refactors.
- **Commit attribution**: a commit-msg hook appends `By <role>.`.

Jaato can express most of this, but the application has to build it:

| SwarmForge mechanism | Nearest jaato primitive |
|---|---|
| Role prompt + constitution | Persona (`.jaato/agents/<name>.md`) over `.jaato/instructions/` base layers |
| Pack pipeline | Cascade driver over profiles (`jaato-scaffold new cascade`) |
| `AUDIT_REQUIRED` double-submit | `completion_processors` with `max_refusals` (a scripted gate, not a prompt-level re-read) |
| Handoff schema validation | `completion_payload_schema` / `spawn_payload_schema` |
| Quality tools must pass | A completion processor that runs them. Nothing mandates it |
| Card approve / reject / retry | Driver-level decision on the completion payload. No built-in board |

Jaato has **no equivalent** of commit-as-contract or git-worktree-per-role
isolation. Its subagents share one workspace unless the driver chooses
otherwise. SwarmForge has **no equivalent** of a typed payload: a handoff
carries a commit, a task name and a priority, and the meaning lives in the
commit and the prompts.

---

## 5. Human interface

**SwarmForge**: one local web dashboard (`pack_web.bb`, bound to
`127.0.0.1`) with a work-queue board, card lanes, approvals,
clarifications (`pack_dashboard_request.sh clarify`), agent panes (tmux,
opened through iTerm2 / Terminal.app / Ghostty adapters) and swarm
controls. Agents are told never to ask questions in their pane; the
clarification queue is the only channel.

**Jaato**: a TUI and a web client, both built on the same event protocol
through the SDK (Python and TypeScript), over IPC or authenticated
WebSocket (bearer tokens, per-user connect tickets, #1074). Permission
prompts, clarifications (including audio/image answers, #989),
per-session notes, budgets and plans are all first-class events. Several
people can attach, and workspace/session visibility follows ownership.

SwarmForge is built for one operator at one machine. Jaato is built for a
daemon that several clients and users share.

---

## 6. Durability and recovery

Both are designed to survive restarts, in different ways:

- **SwarmForge**: all state is files and git. Agents recover with
  "On restart, run `ready_for_next.sh` and follow its output". Writes
  are atomic (tmp + rename), per-worktree sequence numbers are taken under
  a lock, and headers carry `created_at` / `enqueued_at` / `dequeued_at` /
  `completed_at` audit timestamps. The git history *is* the work record.
- **Jaato**: sessions persist to disk with their resolved profile, rendered
  prompt and creator (record 2.10). A revived session replays what it
  persisted, not what is on disk today (#787). A daemon-side watchdog
  bounds orphaned sessions (#812), and a dispatch-reconciliation protocol
  tells a lost RPC apart from a slow one (#856).

---

## 7. Cost, accounting and audit

SwarmForge has **no token or cost accounting**. Spend belongs to each CLI
vendor's own billing, and the dashboard cannot see it. Its audit trail
is the handoff headers, the audit-challenge count per card, and git.

Jaato records every response in a ledger, can hash-chain it
(`record_keeping.integrity: sha256-chain`, verified with
`jaato-doctor --audit-verify`), attributes spend per `(provider, model,
tier)` binding, enforces `budget_control` ceilings in the middle of a
turn, exports OpenInference spans, and maps each control to EU AI Act
obligations. That matters if several agents run unattended.

---

## 8. When to choose which

**SwarmForge fits when:**

- you want a software team made of agent roles (spec → code → clean →
  architect → harden → QA), with git as the only contract between them;
- you already pay for Claude Code, Codex, Grok or Copilot and want to use
  those CLIs as they are;
- you want a strict TDD / mutation / CRAP / Gherkin discipline built into
  the prompts;
- you have a single operator at a single workstation, and full agent
  autonomy inside a worktree is acceptable.

**Jaato fits when:**

- you are building an agent product, not just running one: custom tools,
  plugins, clients, providers;
- each tool call must be permissioned, contained, metered or audited;
- you need provider choice or local models (20+ providers, including
  Ollama, vLLM, LM Studio and Chrome's on-device model);
- sessions are long-lived, shared, headless or multi-tenant;
- stage results must be machine-readable and schema-checked rather than
  "whatever is in the commit".

**Using both together** is not supported today. SwarmForge's backend set is
closed (`known-agents`), and adding a jaato-based CLI would mean changing
`swarmforge.bb`. The design fit is reasonable, though: a jaato session
driven headlessly in a role worktree would give that role per-tool
permissions and budgets that the vendor CLIs, running with `--yolo`, do
not provide.

---

## 9. Ideas worth borrowing

These are observations from reading SwarmForge. Jaato has not committed
to any of them.

1. **Commit-as-handoff.** A cascade stage that works on code could end by
   naming a commit that the next stage must merge, which gives it an
   inspectable, revertable, diffable contract. Jaato's completion payloads
   could include a validated `commit` field in the same way.
2. **Audit-by-resubmission.** The first-submit refusal is a cheap,
   model-agnostic way to get a self-review pass. Jaato's
   `completion_processors` can do this already (refuse once, then
   accept), but no scaffold emits it as a pattern.
3. **Queue-order wake-ups that don't name the file.** This keeps an agent
   from cherry-picking work. It applies to jaato's `session.wake` /
   `inject_prompt` driven reactors.
4. **Back-propagation of refactors** to upstream roles in a pipeline, as
   an option for a cascade driver.
