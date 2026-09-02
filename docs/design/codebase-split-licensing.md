# Codebase Split & Licensing Design

## Distribution Model

| Package | License | Distribution | Install |
|---------|---------|-------------|---------|
| **jaato-sdk** | BSL 1.1 | Public PyPI | `pip install jaato-sdk` |
| **jaato-server** | BSL 1.1 | Public PyPI | `pip install jaato-server` |
| **jaato-tui** | BSL 1.1 | Public PyPI | `pip install jaato-tui` |
| **jaato-premium** | Commercial | Private GitHub repo | See [Premium Installation](#premium-installation) |

The public packages remain exactly where they are (same repo, same PyPI).
`jaato-premium` is a **separate private repo** that depends on `jaato-server`
and extends it via the existing plugin entry-point system.

### BSL 1.1 Parameters

```
Business Source License 1.1

Parameters

Licensor:             apanoia
Licensed Work:        jaato [version]
                      The Licensed Work is (c) 2024 apanoia.

Additional Use Grant: You may make production use of the Licensed Work,
                      provided that you may not use the Licensed Work
                      to offer a commercial AI agent orchestration service
                      or AI development tool that is provided to third
                      parties as a hosted, managed, or embedded product
                      and that includes substantial functionality of the
                      Licensed Work.

Change Date:          [4 years from release of each version]

Change License:       Apache License, Version 2.0

For information about alternative licensing arrangements for the Licensed Work,
please contact licensing@apanoia.dev
```

**What BSL 1.1 allows and prohibits:**

| Use Case | Allowed? |
|----------|----------|
| Internal use at a company of any size | Yes |
| Building a product that uses jaato internally | Yes |
| Academic research | Yes |
| Contributing to jaato | Yes |
| Running jaato for personal projects | Yes |
| Forking and modifying for internal use | Yes |
| Offering "jaato Cloud" as a hosted service | **No** |
| White-labeling jaato as a competing product | **No** |

---

## Installation

### Public Packages (BSL 1.1 — source-available)

Published on PyPI. Standard pip install:

```bash
pip install jaato-sdk jaato-server jaato-tui
```

Or with optional extras:

```bash
pip install "jaato-server[all]" "jaato-tui[all]"
```

These give you the full source-available framework: 8 model providers, 58 plugins,
TUI client, web client, GC strategies, telemetry — everything needed to build
and run single-server agentic AI applications.

### Premium Installation

The premium package lives in a **private GitHub repository**. Only users with
repository access (org members, collaborators) can install it.

**Via SSH (recommended — uses your SSH key):**

```bash
pip install git+ssh://git@github.com/Jaato-framework-and-examples/jaato-premium.git
```

**Via HTTPS (uses a GitHub PAT):**

```bash
pip install git+https://<GITHUB_TOKEN>@github.com/Jaato-framework-and-examples/jaato-premium.git
```

**Specific version/branch:**

```bash
# Install from a tag
pip install git+ssh://git@github.com/Jaato-framework-and-examples/jaato-premium.git@v1.0.0

# Install from a branch
pip install git+ssh://git@github.com/Jaato-framework-and-examples/jaato-premium.git@main
```

**In requirements.txt:**

```
jaato-server>=0.2.53
jaato-premium @ git+ssh://git@github.com/Jaato-framework-and-examples/jaato-premium.git@v1.0.0
```

**Editable / development install (from a local clone):**

```bash
git clone git@github.com:Jaato-framework-and-examples/jaato-premium.git
pip install -e jaato-premium/
```

### Access Control

- **Who can install:** Only GitHub users/teams granted access to the private repo
- **How access is managed:** Standard GitHub repository permissions (org teams, collaborators)
- **Source visibility:** The Python source is readable to anyone who installs it (Python packages are not compiled). Protection is access-based (who can reach the repo) and legal (commercial license terms), not code obfuscation.
- **No private PyPI needed:** GitHub itself is the distribution mechanism

---

## What Goes Where

### PUBLIC (BSL 1.1) — stays in this repo

Everything that is **framework plumbing** — the engine that makes tools run,
providers connect, sessions manage state. A fully functional single-server
agentic orchestrator, but without the opinionated "secret sauce" or
multi-server clustering.

#### jaato-sdk

- IPC/WebSocket client protocol
- Base plugin interfaces
- Event types (including peer event dataclasses — harmless without gossip)
- Model provider types, `CancelToken`, streaming types
- The top-level **`jaato` convenience facade** — `import jaato` /
  `jaato.session(mode="ipc"|"ws"|"in_process")`. Owned by jaato-sdk so a thin
  client (the daemon may be remote) needs only the SDK. `ipc`/`ws` are pure sdk;
  `in_process` lazily imports the server-side backend `jaato_embedded` and fails
  loud if jaato-server is absent. `JaatoClient`/`PluginRegistry`/`InProcessClient`
  are lazy PEP-562 re-exports (server-only). See
  [in-process-facade.md](in-process-facade.md#packaging--ownership-2026-07--sdk-only-facade).

#### jaato-server — core

- **`jaato_embedded/`** — the embedded (in-process) runtime backend behind the
  sdk-owned `jaato.session(mode="in_process")` facade (`client.py`,
  `permission.py`). jaato-server owns `jaato_embedded`; jaato-sdk owns `jaato`
  (two dists cannot both ship a regular `jaato` package).

- `shared/jaato_client.py`, `jaato_runtime.py`, `jaato_session.py`
- `shared/instruction_budget.py`, `shared/token_accounting.py`
- `shared/ai_tool_runner.py`, `shared/mcp_context_manager.py`
- `shared/plugins/base.py`, `shared/plugins/registry.py`
- `server/core.py`, `server/ipc.py`, `server/websocket.py`
- `server/session_manager.py` (single-server session management)
- `server/__main__.py` (daemon entry point with gossip hook points)

#### jaato-server — standard plugins (58 plugin directories)

All existing plugins stay in the public repo under BSL 1.1. They are the
framework's value as a source-available project. Specifically:

**Tool plugins:** cli, file_edit, filesystem_query, calculator, mcp,
interactive_shell, web_fetch, web_search, multimodal, notebook, todo,
environment, clarification, introspection, permission, ast_search,
artifact_tracker, lsp, sandbox_manager, service_connector, prompt_library,
waypoint, background, thinking, memory, vision_capture

**Formatter plugins:** code_block_formatter, code_validation_formatter,
diff_formatter, formatter_pipeline, hidden_content_filter,
inline_markdown_formatter, mermaid_formatter, notebook_output_formatter,
table_formatter

**GC plugins:** gc (base), gc_truncate, gc_summarize, gc_hybrid, gc_budget

**Cache plugins:** cache (base), cache_anthropic, cache_google_genai, cache_zhipuai

**Session/streaming:** session, streaming

**Telemetry:** telemetry

**Auth plugins:** anthropic_auth, antigravity_auth, github_auth, nim_auth, zhipuai_auth

**Model providers:** google_genai, anthropic, claude_cli, github_models,
antigravity, ollama, nim, zhipuai

**Coordination plugins:** subagent (plugin, config, serializer), references, template, reliability

#### jaato-server — agent profiles infrastructure

The **profile mechanism** stays public (framework plumbing):
- `SubagentProfile` dataclass, JSON schema, variable expansion
- Profile file discovery from `.jaato/profiles/`
- `SessionManager.create_session(profile_name=...)` / `list_profiles()`
- `SessionProfilesEvent` in SDK events
- Profile-authoritative plugin visibility (introspection filtering)
- `--profile` CLI flag

The **curated profile files** themselves (15 JSON files defining specific
agent types) are premium content — see below.

#### jaato-tui (unchanged)

- Rich terminal client, themes, renderers, keybindings
- `.jaato.example/` scaffold

#### Other public content

- `docs/` (architecture, design docs, etc.)
- `web-client/` (React web client)
- `out-of-tree-plugins/` (plugin development example)
- `gc-benchmark/`, `examples/`, `scripts/`
- `CLAUDE.md`, `README.md`

### PRIVATE (Commercial) — new `jaato-premium` repo

Two categories of premium content: **methodology** (opinionated knowledge and
behavioral tuning) and **infrastructure** (multi-server clustering).

#### Category 1: Methodology

Content that represents **opinionated methodology, curated knowledge, and
behavioral tuning** — the things that make jaato agents work *well* rather
than just *work*.

##### 1. System Instructions (`instructions/`)

**Source:** `.jaato/instructions/00-system-instructions.md`

The 19 behavioral principles (Transparency Mandate, Large Output Protocol,
Autonomous Decision Making, Anti-Fabrication, Delegation Authority, etc.)
are the single most valuable piece of IP. They encode months of iteration
on how to make LLM agents behave reliably.

These become the premium package's default instructions, loaded via
the same `.jaato/instructions/` mechanism the framework already supports.

##### 2. Knowledge Base (`knowledge/`)

**Source:** `knowledge/` (155 files)

- `ADRs/` — Architecture Decision Records (6 ADRs)
- `ERIs/` — Executable Reference Implementations (8 ERIs)
- `modules/` — Code generation modules with templates and validation
  (circuit-breaker, retry, timeout, rate-limiter, hexagonal-base,
  persistence-jpa, persistence-systemapi, api-integration, api-exposure,
  compensation) — 10 modules with Handlebars templates
- `model/` — Knowledge model definitions (domains, standards, authoring prompts)

##### 3. Subagent Profiles (`profiles/`)

**Source:** `.jaato/profiles/*.json` (15 files)

Curated subagent profile definitions — the JSON files, not the loading
infrastructure:
- `skill-code-*` / `skill-mod-code-*` — Coding specialist profiles (12)
- `validator-tier*` — Multi-tier validation profiles (3)
- `analyst-*` — Analysis profiles (1)

The framework discovers and loads these from `.jaato/profiles/`. Premium
supplies the actual definitions. Users can also write their own.

##### 4. Reference Catalog (`references/`)

**Source:** `.jaato/references/*.json`

Pre-built reference JSON files that link ADRs, ERIs, and modules
into a structured catalog with semantic embeddings and validation rules.

##### 5. Prompt Templates (`prompt_templates/`)

**Source:** `jaato-server/shared/prompt_templates/`

- COBOL analysis prompts (identify_code_changes, parse_mod_history)
- Confluence integration prompts (get_page, search, update_page — CLI & MCP)
- GitHub integration prompts (get_issue, list_issues, search_issues — CLI & MCP)

##### 6. Curated Prompts (`prompts/`)

**Source:** `.jaato/prompts/`

- `gen-references.md` — Prompt for scanning knowledge bases and generating
  reference catalogs, template indexes, and subagent profiles
- `execute-from-inputs.md` — Autonomous orchestrated execution prompt

##### 7. Framework Prompt Constants (PUBLIC — stays in jaato-server)

**Source:** `jaato-server/shared/jaato_runtime.py`

Three embedded prompt constants:
- `_TASK_COMPLETION_INSTRUCTION` — Anti-fabrication + relentless completion
- `_PARALLEL_TOOL_GUIDANCE` — Parallel tool batching guidance
- `_TURN_SUMMARY_INSTRUCTION` — Turn-end summarization guidance

These are **necessary for correct agent behavior** (safety, efficiency, GC)
and stay in the public repo as functional defaults. The premium package can
provide **enhanced versions** via the `jaato.premium` → `prompt_provider`
entry point, but the base agent works correctly without premium installed.

##### 8. Training Data & Specialized Tools

- `modlog-training-set-test/` — COBOL modification log training set generator
- `cli_vs_mcp/` — CLI vs MCP comparison harness
- `create_self_extractor.py` — Self-extracting archive builder

#### Category 2: Multi-Server Clustering

Server infrastructure enabling distributed jaato deployments — peer
discovery, remote subagent delegation, workspace replication, and
cluster management.

##### 9. Gossip Protocol & Peer Management (~5,500 LOC)

| Module | LOC | Description |
|--------|-----|-------------|
| `server/peers.py` | 551 | Gossip protocol, peer registry, heartbeats, liveness tracking |
| `server/remote_spawn.py` | 731 | Remote subagent delegation (origin + remote sides) |
| `server/workspace_sync.py` | 584 | Git-based workspace replication for remote subagents |
| `server/server_reliability.py` | 412 | Trust state, failure history, affinity scores for peers |
| `server/health.py` | 84 | Server health metrics collection (CPU, memory, sessions) |
| `server/health_http.py` | 301 | HTTP health endpoint + dashboard route dispatch |
| `server/dashboard/routes.py` | 560 | REST API for cluster config CRUD, Docker launch operations |
| `server/dashboard/docker_launcher.py` | 755 | Docker Compose generation + container lifecycle management |
| `server/dashboard/static/index.html` | 1,502 | Self-contained SPA for web-based cluster management |

Plus integration touchpoints in:
- `server/__main__.py` — `_init_gossip()`, `_load_servers_config()`, CLI args
  (`--health-port`, `--server-name`, `--servers-json`)
- `server/session_manager.py` — `set_gossip_context()`,
  `_configure_gossip_context()`
- `shared/plugins/environment/plugin.py` — `jaato_agentic_servers` aspect
- `shared/plugins/subagent/plugin.py` — `server` parameter on
  `spawn_subagent`, `_execute_remote_spawn()`
- `tests/e2e/gossip/`, `tests/e2e/workspace-sync/` — Docker-based E2E tests

---

## The Server-Level Split Challenge

The methodology-only premium items (categories 1-8) follow a clean boundary:
the framework provides a **loading mechanism**, the premium package provides
the **content**. No framework code changes needed.

The gossip/clustering modules (category 9) are fundamentally different:

1. **They modify core files** — `__main__.py` and `session_manager.py` both
   gain gossip-specific methods and constructor parameters
2. **They extend existing plugin APIs** — `subagent` plugin gains a `server`
   parameter, `environment` plugin gains a `jaato_agentic_servers` aspect
3. **They add SDK types** — `jaato-sdk/events.py` gets 8 new event types
   that must be present for deserialization even in non-gossip deployments
4. **They are conditionally activated** — `servers.json` gates everything,
   so a vanilla install never activates gossip, but the code is still present
5. **They include a full dashboard** — REST API, Docker orchestration, and
   a web SPA that are only useful with clustering

### Split approaches

#### Approach A: Conditional imports (keep code in jaato-server, gate on premium)

Keep the gossip modules in `jaato-server` but make them **import-gated on
jaato-premium**. The modules exist in the public PyPI package but refuse
to activate without the premium package installed.

```python
# server/__main__.py
def _init_gossip(self) -> None:
    try:
        from jaato_premium.gossip import verify_license
        verify_license()
    except ImportError:
        logger.info("Multi-server gossip requires jaato-premium")
        return
    # ... proceed with gossip setup
```

**Pros:** Simplest implementation, no code reorganization needed.
**Cons:** Premium code ships in the public PyPI package (visible to all).
Source is inspectable even though it won't run. Enforcement is trivial to
bypass (just comment out the check).

#### Approach B: Plugin-based gossip (extract to jaato-premium)

Extract all gossip modules into `jaato-premium` and have them register
via entry points. The public `jaato-server` only contains **hook points**
(the `set_gossip_context` methods, the `server` parameter schema stub).

```
jaato-premium/
├── jaato_premium/
│   ├── gossip/
│   │   ├── peers.py
│   │   ├── remote_spawn.py
│   │   ├── workspace_sync.py
│   │   ├── server_reliability.py
│   │   ├── health.py
│   │   ├── health_http.py
│   │   └── dashboard/
│   │       ├── routes.py
│   │       ├── docker_launcher.py
│   │       └── static/index.html
│   └── ...
```

The daemon's `__main__.py` would check for the entry point:

```python
def _init_gossip(self) -> None:
    eps = importlib.metadata.entry_points(group="jaato.gossip")
    if not eps:
        return  # No gossip provider installed
    gossip_init = eps["init"].load()
    gossip_init(self)  # Wire everything up
```

**Pros:** Premium code is truly separate — not in public PyPI. Clean boundary.
**Cons:** Requires careful interface design. The gossip init function needs
access to daemon internals (`session_manager`, transport info, CLI args).
The SDK event types (`PeerHeartbeat`, etc.) must still live in `jaato-sdk`
for deserialization, or be dynamically registered.

#### Approach C: Separate package `jaato-cluster` (premium, not in jaato-premium)

Create a dedicated `jaato-cluster` package (also private/commercial) rather
than bundling gossip into `jaato-premium`. This mirrors how many projects
separate their clustering/enterprise tier.

```
jaato-cluster/
├── jaato_cluster/
│   ├── peers.py
│   ├── remote_spawn.py
│   ├── workspace_sync.py
│   ├── server_reliability.py
│   ├── health.py
│   ├── health_http.py
│   ├── dashboard/
│   └── daemon_mixin.py      # Mixin/hook that wires into __main__
```

**Pros:** Clean conceptual boundary (single-server free, multi-server premium).
Separation of concerns between "methodology premium" and "infrastructure premium."
**Cons:** Two private repos to maintain. Users who want everything need
`pip install jaato-premium jaato-cluster`.

#### Approach D: Feature-flagged within jaato-server (free code, premium activation)

All gossip code stays in the public `jaato-server`. The code is BSL 1.1-licensed
and fully visible. But `servers.json` loading and gossip activation require
a valid **activation key** checked at runtime against the premium package.

This is the "open core" model used by GitLab, Minio, CockroachDB, etc.
The code is open, the right to run it commercially is gated by license.

**Pros:** No code split needed at all. Community can read, audit, and
contribute to gossip code. Only licensing changes.
**Cons:** Requires a license-key mechanism. Blurs the BSL/commercial boundary
(BSL code with additional commercial runtime restriction is confusing).

### Recommended approach

**Approach B (plugin-based extraction)** best fits the existing architecture:

1. jaato already has an entry-point plugin system
2. The gossip modules are naturally self-contained (9 files in `server/`,
   not scattered changes across `shared/`)
3. The integration points are narrow — `_init_gossip()` is one method,
   plugin wiring is two `set_*_context()` calls
4. The SDK event types can stay in `jaato-sdk` (they're just data classes,
   harmless without gossip)
5. The dashboard (routes + Docker launcher + SPA) is entirely self-contained
   and has zero coupling to existing framework code

The key design work is defining the `jaato.gossip` entry-point interface:
what the daemon passes to the gossip initializer, and what the initializer
wires back.

### Integration surface to keep public

Even under Approach B, certain **hook points** must remain in the public
jaato-server for the plugin to wire into:

- `SessionManager.set_gossip_context()` — stores references for plugin injection
- `SessionManager._configure_gossip_context()` — wires references into per-session plugins
- `EnvironmentPlugin.set_gossip_context()` — accepts gossip references
- `SubagentPlugin.set_peer_context()` — accepts peer registry + remote handler
- `SubagentPlugin._execute_remote_spawn()` — stub that delegates to the handler
- `spawn_subagent` tool schema's `server` parameter — conditionally added when handler is present
- SDK event types (`PeerHeartbeat`, etc.) — data classes, safe to keep public

These are lightweight (a few `Optional[Any]` fields and setter methods) and
don't expose premium logic.

---

## Implementation Approach

### Step 1: Create premium plugin entry points in jaato-server

Add `jaato.premium` and `jaato.gossip` entry-point groups to
`jaato-server/pyproject.toml` that premium plugins can register with.
The runtime checks for these at startup and loads them if present.

### Step 2: Make framework prompt constants pluggable

In `jaato_runtime.py`, replace the hardcoded `_TASK_COMPLETION_INSTRUCTION`,
`_PARALLEL_TOOL_GUIDANCE`, and `_TURN_SUMMARY_INSTRUCTION` with a lookup
that:
1. Checks if a premium prompt provider is registered (via entry point)
2. Falls back to generic defaults if not

### Step 3: Make gossip initialization pluggable

In `__main__.py`, replace the direct `_init_gossip()` implementation with
an entry-point lookup. The daemon passes a context object (session_manager,
transport info, CLI args) to the gossip initializer, which wires everything
up and returns the references the daemon needs.

### Step 4: Create jaato-premium repo structure

```
jaato-premium/
├── pyproject.toml              # Commercial license, depends on jaato-server
├── LICENSE                     # Commercial license (All Rights Reserved)
├── README.md
├── jaato_premium/
│   ├── __init__.py
│   ├── prompts.py              # Premium prompt constants (the 3 from jaato_runtime)
│   ├── instructions/           # 00-system-instructions.md (19 principles)
│   ├── knowledge/              # ADRs, ERIs, modules, model
│   ├── profiles/               # 15 curated subagent profile JSON files
│   ├── references/             # Reference catalog JSONs
│   ├── prompt_templates/       # COBOL, Confluence, GitHub prompts
│   ├── prompts/                # gen-references.md, execute-from-inputs.md
│   └── gossip/                 # Multi-server clustering
│       ├── __init__.py         # Entry point: init(daemon_context) -> GossipRefs
│       ├── peers.py
│       ├── remote_spawn.py
│       ├── workspace_sync.py
│       ├── server_reliability.py
│       ├── health.py
│       ├── health_http.py
│       └── dashboard/
│           ├── routes.py
│           ├── docker_launcher.py
│           └── static/index.html
├── tests/
│   ├── test_prompts.py
│   └── e2e/
│       ├── gossip/
│       └── workspace-sync/
└── docker/                     # Dockerfiles for cluster deployments
```

### Step 5: Wire premium content loading

The premium package registers itself via entry points:

```toml
# jaato-premium/pyproject.toml
[project.entry-points."jaato.premium"]
prompt_provider = "jaato_premium.prompts:get_prompts"
instructions = "jaato_premium:get_instructions_path"
knowledge = "jaato_premium:get_knowledge_path"

[project.entry-points."jaato.gossip"]
init = "jaato_premium.gossip:init_gossip"
```

### Step 6: Move content from public repo

Move (not copy) the premium content out of the public repo:
- `.jaato/instructions/00-system-instructions.md` → jaato-premium
- `.jaato/profiles/*.json` (14 of 15 files) → jaato-premium
  - **Keep** `github-resolver.json` in public repo as example profile
- `.jaato/references/*.json` → jaato-premium
- `.jaato/prompts/*.md` (premium prompts) → jaato-premium
  - **Keep** `gh_issue_fixer.md` in public repo as example prompt
- `knowledge/` → jaato-premium
- `shared/prompt_templates/` → jaato-premium
- `server/peers.py`, `remote_spawn.py`, `workspace_sync.py`,
  `server_reliability.py`, `health.py`, `health_http.py` → jaato-premium
- `server/dashboard/` → jaato-premium
- `tests/e2e/gossip/`, `tests/e2e/workspace-sync/` → jaato-premium
- `modlog-training-set-test/` → jaato-premium
- `cli_vs_mcp/` → jaato-premium

Replace the 3 prompt constants in `jaato_runtime.py` with generic fallbacks.
Replace `_init_gossip()` in `__main__.py` with entry-point lookup.

### Step 7: Update public repo

- Keep `.jaato/instructions/` as an empty directory with a README
  explaining that users can add their own instructions
- Keep `.jaato.example/profiles/` with `github-resolver.json` as a working
  example profile (demonstrates the schema and a real use case, not production
  methodology) plus a README explaining how to create custom profiles
- Keep `.jaato.example/prompts/` with `gh_issue_fixer.md` as a working
  example prompt (the operational prompt loaded by the github-resolver profile)
  plus a README explaining how to create custom prompts
- Keep `.jaato/references/` empty with README
- Update `CLAUDE.md` to remove references to moved content
- Update `README.md` to mention the premium package as optional

---

## What This Preserves

- **Zero breaking changes** — the public framework works exactly as before
- **Plugin architecture unchanged** — premium is just more plugins
- **Existing users unaffected** — `pip install jaato-server` still works
- **Clear value boundary** — framework (free) vs methodology + clustering (premium)
- **Simple upgrade path** — `pip install git+ssh://...` adds premium on top
- **Profile infrastructure stays open** — anyone can create their own profiles
- **SDK types stay public** — peer event dataclasses are harmless without gossip

## What Premium Users Get

1. Battle-tested system instructions (19 principles of agent behavior)
2. Knowledge base with ADRs, ERIs, and code generation modules
3. 15 pre-built subagent profiles for coding, validation, and analysis
4. Reference catalog with semantic matching
5. Domain-specific prompt templates (COBOL, Confluence, GitHub)
6. Curated orchestration prompts (gen-references, execute-from-inputs)
7. Optimized framework prompts (anti-fabrication, parallel batching, summarization)
8. Multi-server gossip clustering (peer discovery, remote subagent delegation,
   workspace sync, server reliability tracking, health monitoring)
9. Cluster management dashboard (web SPA, REST API, Docker orchestration)
