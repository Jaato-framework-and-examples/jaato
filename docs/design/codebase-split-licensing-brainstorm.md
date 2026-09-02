# Codebase Split & Licensing Brainstorm

**Date:** 2026-03-01
**Status:** Draft / Brainstorm
**Current License:** MIT (Copyright 2024 apanoia)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Assessment](#2-current-state-assessment)
3. [The Two-Repo Split Strategy](#3-the-two-repo-split-strategy)
4. [Repo A: Open Source under BSL 1.1](#4-repo-a-open-source-under-bsl-11)
5. [Repo B: Closed Source (Competitive Advantage)](#5-repo-b-closed-source-competitive-advantage)
6. [BSL 1.1 Parameter Configuration](#6-bsl-11-parameter-configuration)
7. [Technical Implementation Plan](#7-technical-implementation-plan)
8. [Dependency & Import Surgery](#8-dependency--import-surgery)
9. [Build & Distribution Architecture](#9-build--distribution-architecture)
10. [Git History & Migration Strategy](#10-git-history--migration-strategy)
11. [Contributor & Community Impact](#11-contributor--community-impact)
12. [Risks & Mitigations](#12-risks--mitigations)
13. [Alternative Approaches](#13-alternative-approaches)
14. [Recommended Path Forward](#14-recommended-path-forward)
15. [Decision Matrix](#15-decision-matrix)
16. [Open Questions](#16-open-questions)

---

## 1. Executive Summary

The proposal is to split the current monorepo (`jaato`) into two repositories:

| | **Repo A (Open)** | **Repo B (Closed)** |
|---|---|---|
| **License** | BSL 1.1 | Proprietary / All Rights Reserved |
| **Contains** | Framework core, plugin interfaces, server skeleton, SDK, TUI | System prompts, design docs, knowledge modules, prompt templates, subagent profiles, proprietary algorithms |
| **Purpose** | Community adoption, trust, contributions, ecosystem growth | Competitive moat, monetization, differentiation |
| **Visibility** | Public on GitHub | Private repository |

The key insight: jaato's competitive advantage lies not in the plumbing (IPC, event protocol, plugin discovery) but in the **intelligence layer** — the carefully engineered system prompts, the 19 operational principles, the knowledge modules, the GC strategies, and the design philosophy that makes the agent effective. The framework without the prompts is a capable but generic tool orchestrator. The prompts without the framework are just documents. Together they are the product.

---

## 2. Current State Assessment

### 2.1 What We Have Today

```
jaato (monorepo, MIT license)
├── jaato-sdk/      (v0.1.16) — 15 files, pure protocol & client library
├── jaato-server/   (v0.2.48) — 150+ files, all business logic & 55+ plugins
├── jaato-tui/      (v0.1.31) — 30+ files, terminal client
├── web-client/     — React web UI (early)
├── docs/           — 45MB+ of design documents, architecture, roadmaps
├── .jaato/         — System instructions, prompts, profiles
├── knowledge/      — Domain expertise modules (Java/Spring patterns)
├── examples/       — Example implementations
└── tests_enablement_2.0/ — Integration tests for knowledge system
```

**Total:** 571 Python files, ~241,680 lines, 806 files (py/md/json combined)

### 2.2 Existing Package Boundaries (Already Clean)

The codebase already has a well-architected layered dependency graph:

```
jaato-sdk (protocol only, zero business logic)
    ↑
jaato-server (all business logic, plugins, providers)
    ↑                    ↑
jaato-tui (thin client)  web-client (thin client)
```

- **No circular dependencies** — verified
- **SDK imports nothing** from server or TUI
- **TUI imports only one constant** (`PRERENDERED_LINE_PREFIX`) from SDK
- **Plugin discovery** via entry points — clean extension mechanism
- **Lazy loading** in `shared/__init__.py` — already supports minimal imports

This clean separation is a significant advantage for the split.

### 2.3 Asset Classification

After thorough analysis, here's how the codebase's assets classify by competitive value:

#### Tier 1: Crown Jewels (Highest Competitive Value)
- **System instructions** (`.jaato/instructions/00-system-instructions.md`) — 19 operational principles
- **Runtime instructions** (`jaato_runtime.py` embedded prompts) — anti-fabrication, parallel tool guidance, turn summaries
- **Subagent profiles** (`.jaato/profiles/`) — specialist definitions with tool/model/GC configs
- **Knowledge modules** (`knowledge/`, `tests_enablement_2.0/`) — Java/Spring patterns, ADRs, ERIs, validation
- **Prompt templates** (`.jaato/prompts/`, `shared/prompt_templates/`) — parameterized prompt library
- **Design philosophy** (`docs/design-philosophy.md`) — the 5 core principles
- **GC system design** — hybrid generational strategy, instruction budget algorithm

#### Tier 2: Significant Value (Hard to Replicate)
- **Full design documentation** (`docs/`) — 45MB of architecture, sequence diagrams, event protocol
- **Instruction budget system** (`instruction_budget.py`) — 5-tier source tracking with GC policies
- **Hybrid GC algorithm** (`gc_hybrid/plugin.py`) — Java-inspired three-tier generational collection
- **Token accounting** (`token_accounting.py`) — multi-stage tracking, error classification, SSL guidance
- **Reliability plugin** — nudges, patterns, persistence, policy-based error handling
- **CLAUDE.md** — the comprehensive project instruction file itself
- **Roadmap documents** (`docs/roadmap/`)

#### Tier 3: Valuable but Reproducible (Implementation Quality)
- **Plugin implementations** — 55+ plugins (cli, file_edit, interactive_shell, etc.)
- **Model provider adapters** — 8 providers with auth flows
- **Server daemon** — IPC, WebSocket, session management
- **TUI client** — rich terminal interface with theming, keybindings
- **MCP integration** — multi-server client management
- **Tool traits system** — semantic tool classification
- **Parallel tool execution** — thread pool with safe callbacks

#### Tier 4: Commodity (Standard Engineering)
- **SDK protocol definitions** — event types, serialization
- **IPC framing** — length-prefixed protocol
- **Plugin discovery** — entry points, `PLUGIN_KIND` convention
- **Configuration management** — env vars, JSON configs

---

## 3. The Two-Repo Split Strategy

### 3.1 Guiding Principles

1. **The open repo must be genuinely useful standalone** — a skeleton nobody can run is worse than no release at all; it breeds resentment, not adoption
2. **The closed repo must contain only what truly differentiates** — over-classifying assets as "secret" creates engineering friction and slows development
3. **The boundary must be a clean API** — the closed repo should plug into the open repo, not fork it
4. **The split must be maintainable** — two repos means two CI pipelines, two release cycles, merge conflicts across repos; minimize the surface area of the boundary

### 3.2 Proposed Boundary

```
┌─────────────────────────────────────────────────────┐
│                  REPO B (Closed)                     │
│                                                      │
│  .jaato/instructions/     System prompts (19 rules)  │
│  .jaato/profiles/         Subagent profiles           │
│  .jaato/prompts/          Prompt templates            │
│  knowledge/               Domain knowledge modules    │
│  tests_enablement_2.0/    Knowledge integration tests │
│  docs/design-philosophy   Design philosophy          │
│  docs/design/             Detailed design docs       │
│  docs/roadmap/            Roadmap & strategy         │
│  docs/reviews/            Code reviews               │
│  shared/prompt_templates/ Server prompt templates    │
│  Embedded prompts         Runtime instructions       │
│  gc_hybrid/               Hybrid GC algorithm        │
│  gc_budget/               Budget GC algorithm        │
│  instruction_budget.py    Instruction budget system  │
│  reliability/             Reliability plugin         │
│  CLAUDE.md                Project instructions       │
│                                                      │
└────────────────────┬────────────────────────────────┘
                     │ installs into / overlays
                     ▼
┌─────────────────────────────────────────────────────┐
│                  REPO A (BSL 1.1)                    │
│                                                      │
│  jaato-sdk/               Protocol & client library  │
│  jaato-server/server/     Server daemon              │
│  jaato-server/shared/     Core framework             │
│    ├── jaato_client.py    Facade                     │
│    ├── jaato_runtime.py   Runtime (minus prompts)    │
│    ├── jaato_session.py   Session management         │
│    ├── ai_tool_runner.py  Tool execution             │
│    ├── token_accounting.py Token ledger              │
│    ├── mcp_context_manager MCP integration           │
│    └── plugins/           Plugin framework + impls   │
│        ├── cli/           Shell commands             │
│        ├── file_edit/     File editing               │
│        ├── mcp/           MCP servers                │
│        ├── interactive_shell/ PTY sessions           │
│        ├── permission/    Permission control         │
│        ├── model_provider/ All 8 providers           │
│        ├── gc_truncate/   Basic GC                   │
│        ├── gc_summarize/  Summarization GC           │
│        └── ... (40+ tool plugins)                    │
│  jaato-tui/               Terminal client            │
│  web-client/              Web client                 │
│  examples/                Basic examples             │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 4. Repo A: Open Source under BSL 1.1

### 4.1 What Goes In

**Everything needed to run jaato as a functional agent framework**, minus the intelligence layer:

| Component | Package | Rationale |
|---|---|---|
| `jaato-sdk/` (complete) | jaato-sdk | Pure protocol; must be open for third-party client development |
| `jaato-server/server/` (complete) | jaato-server | Daemon, IPC, WebSocket — infrastructure plumbing |
| `jaato-server/shared/jaato_client.py` | jaato-server | Facade — public API |
| `jaato-server/shared/jaato_runtime.py` | jaato-server | Runtime — but with **stub/minimal system instructions** |
| `jaato-server/shared/jaato_session.py` | jaato-server | Session management — core loop |
| `jaato-server/shared/ai_tool_runner.py` | jaato-server | Tool execution engine |
| `jaato-server/shared/token_accounting.py` | jaato-server | Token ledger (the retry logic is standard) |
| `jaato-server/shared/mcp_context_manager.py` | jaato-server | MCP integration |
| `jaato-server/shared/plugins/registry.py` | jaato-server | Plugin discovery & lifecycle |
| `jaato-server/shared/plugins/base.py` | jaato-server | Plugin base classes |
| Tool plugins (40+) | jaato-server | cli, file_edit, mcp, interactive_shell, permission, todo, memory, web_search, etc. |
| Model providers (8) | jaato-server | google_genai, anthropic, claude_cli, github_models, ollama, nim, zhipuai, antigravity |
| Auth plugins (5) | jaato-server | anthropic_auth, github_auth, antigravity_auth, nim_auth, zhipuai_auth |
| `gc_truncate/` | jaato-server | Basic GC (commodity) |
| `gc_summarize/` | jaato-server | Summarization GC (commodity) |
| Cache plugins (3) | jaato-server | cache_anthropic, cache_google_genai, cache_zhipuai |
| `jaato-tui/` (complete) | jaato-tui | Terminal client |
| `web-client/` | web-client | Web UI |
| Tests for included components | — | Unit tests for the above |
| Basic `README.md` | — | Setup, usage, architecture overview (condensed) |
| Basic `.env.example` | — | Environment variable reference |
| `permissions.example.json` | — | Permission config template |

### 4.2 What Gets Stripped / Stubbed

These are the surgical points where the open repo diverges:

1. **`jaato_runtime.py`**: Remove embedded prompt constants (`_TASK_COMPLETION_INSTRUCTION`, `_PARALLEL_TOOL_GUIDANCE`, `_TURN_SUMMARY_INSTRUCTION`, `_SANDBOX_GUIDANCE`). Replace with a **hook/callback mechanism** or a **file-based instruction loader** that reads from a configurable path. The open repo ships with minimal/placeholder instructions.

2. **`.jaato/` directory**: Ship an empty `.jaato.example/` skeleton with directory structure but no content in `instructions/`, `prompts/`, `profiles/`.

3. **`instruction_budget.py`**: This is a borderline case. The *data structure* (5-tier source tracking) is useful for any GC plugin to work properly. Consider keeping the data model but removing the sophisticated GC policies. Or: keep it entirely — the algorithm is defensible but the real value is in how the instructions themselves are *written*, not how they're *managed*.

4. **`gc_hybrid/` and `gc_budget/`**: Move to closed repo. Open repo ships with `gc_truncate` and `gc_summarize` only.

5. **`reliability/` plugin**: Move to closed repo. The nudge patterns and anti-failure policies are part of the intelligence layer.

6. **`shared/prompt_templates/`**: Remove all prompt template files. Leave the template-loading mechanism in place.

7. **Design docs**: Ship `docs/architecture.md` (condensed) and `docs/sequence-diagram-architecture.md` (condensed) for contributor onboarding. Remove all `docs/design/`, `docs/roadmap/`, `docs/reviews/`, and detailed design documents.

### 4.3 What the Open Repo Can Do Without the Closed Repo

A user of the open repo alone would get:

- A fully functional multi-provider agent framework
- 40+ tool plugins (cli, file_edit, interactive_shell, mcp, memory, web_search, etc.)
- 8 model provider adapters with authentication
- Server daemon with IPC and WebSocket
- TUI client with theming and keybindings
- Basic GC (truncate + summarize)
- Plugin extension points to add their own tools, providers, GC strategies
- MCP server integration
- Parallel tool execution
- Token accounting

**What they would NOT get:**

- The system prompts that make the agent behave intelligently (the 19 principles)
- The subagent profiles that enable specialist delegation
- The knowledge modules for domain expertise
- The hybrid/budget GC strategies
- The reliability plugin (anti-failure nudges)
- The instruction budget system (if removed)
- The prompt templates
- The design philosophy and detailed architecture docs

In essence: they get a **powerful engine without a tuned brain**. They can write their own system prompts and it will work — but the out-of-box experience would be "generic LLM tool use" rather than "sophisticated agent orchestration."

---

## 5. Repo B: Closed Source (Competitive Advantage)

### 5.1 What Goes In

| Asset | Files/Dirs | Competitive Value |
|---|---|---|
| **System Instructions** | `.jaato/instructions/00-system-instructions.md` | The 19 operational principles — the core "personality" |
| **Runtime Prompts** | Extracted from `jaato_runtime.py` | Anti-fabrication, parallel tool guidance, turn summaries |
| **Subagent Profiles** | `.jaato/profiles/*.json` | Specialist definitions (skills, validators, analysts) |
| **Prompt Templates** | `.jaato/prompts/*.md`, `shared/prompt_templates/` | Parameterized prompt library |
| **Knowledge Modules** | `knowledge/` | ADRs, ERIs, templates, validation rules |
| **Enablement Tests** | `tests_enablement_2.0/` | Integration tests for knowledge system |
| **Hybrid GC Plugin** | `gc_hybrid/` | Three-tier generational GC algorithm |
| **Budget GC Plugin** | `gc_budget/` | Policy-aware instruction budget GC |
| **Instruction Budget** | `instruction_budget.py` (if extracted) | 5-tier source tracking with GC policies |
| **Reliability Plugin** | `reliability/` | Nudges, patterns, persistence, policies |
| **Design Documents** | `docs/design-philosophy.md`, `docs/design/`, `docs/roadmap/` | Architecture decisions, roadmap, strategy |
| **CLAUDE.md** | `CLAUDE.md` | Comprehensive project instructions |
| **Reviews** | `docs/reviews/` | Code review documents |

### 5.2 Structure of the Closed Repo

```
jaato-premium/  (or jaato-intelligence, jaato-pro, jaato-core)
├── LICENSE                          # Proprietary / All Rights Reserved
├── README.md                        # Internal documentation
├── pyproject.toml                   # Package definition
├── jaato_premium/
│   ├── __init__.py
│   ├── instructions/
│   │   └── 00-system-instructions.md
│   ├── runtime_prompts/
│   │   ├── task_completion.md
│   │   ├── parallel_tool_guidance.md
│   │   ├── turn_summary.md
│   │   └── sandbox_guidance.md
│   ├── profiles/
│   │   ├── skill-*.json
│   │   ├── validator-*.json
│   │   └── analyst-*.json
│   ├── prompts/
│   │   ├── code-review.md
│   │   ├── gen-references.md
│   │   └── ...
│   ├── knowledge/
│   │   ├── adr-*/
│   │   ├── eri-*/
│   │   ├── mod-*/
│   │   └── model-knowledge/
│   ├── plugins/
│   │   ├── gc_hybrid/              # Premium GC strategy
│   │   ├── gc_budget/              # Premium GC strategy
│   │   ├── reliability/            # Premium reliability plugin
│   │   └── instruction_budget/     # Premium instruction management
│   └── docs/
│       ├── design-philosophy.md
│       ├── design/
│       └── roadmap/
├── install.py                       # Installer that overlays onto jaato
└── tests/
```

### 5.3 Integration Mechanism

The closed repo integrates with the open repo through one of several possible patterns:

#### Option A: Overlay Package (Recommended)

```python
# jaato-premium is pip-installable and registers via entry points
# pyproject.toml
[project.entry-points."jaato.instructions"]
premium = "jaato_premium.instructions"

[project.entry-points."jaato.plugins"]
gc_hybrid = "jaato_premium.plugins.gc_hybrid"
gc_budget = "jaato_premium.plugins.gc_budget"
reliability = "jaato_premium.plugins.reliability"

[project.entry-points."jaato.profiles"]
premium = "jaato_premium.profiles"
```

The open repo's runtime would look for instruction sources in priority order:
1. Installed `jaato.instructions` entry points (premium, if installed)
2. `.jaato/instructions/` in the workspace
3. Built-in minimal fallback instructions

#### Option B: Directory Overlay

```bash
# Clone both repos
git clone https://github.com/apanoia/jaato.git
git clone https://private.repo/apanoia/jaato-premium.git

# Premium repo provides a script to install into jaato
cd jaato-premium && python install.py --target ../jaato
```

The install script copies/symlinks files into the expected locations (`.jaato/instructions/`, `.jaato/profiles/`, plugin directories).

#### Option C: Git Submodule (Not Recommended)

Using the closed repo as a git submodule of the open repo. This leaks the existence of the private repo in `.gitmodules` and creates authentication friction. Avoid.

#### Recommendation: Option A

The overlay package is the cleanest:
- Standard Python packaging conventions
- No filesystem assumptions
- Works with pip, supports versioning
- The open repo works standalone; `pip install jaato-premium` adds intelligence
- Clean separation of concerns

---

## 6. BSL 1.1 Parameter Configuration

### 6.1 The Three Parameters

#### Parameter 1: Additional Use Grant

**Proposed wording:**

> **Additional Use Grant:** You may make production use of the Licensed Work, provided that you may not use the Licensed Work to offer a commercial AI agent orchestration service, AI coding assistant, or AI development tool that is provided to third parties as a hosted, managed, or embedded product and that includes substantial functionality of the Licensed Work.

**Rationale:**
- Allows: internal enterprise use, integration into proprietary products (as a component), academic research, personal projects, non-commercial services
- Prohibits: building a competing hosted agent framework (e.g., a "jaato-as-a-service" offering), white-labeling jaato as your own product
- Modeled after HashiCorp's approach but scoped specifically to the AI agent tooling domain
- Does NOT prohibit using jaato as an internal tool to build other products (even commercial ones)

**Alternative (broader, more permissive):**

> **Additional Use Grant:** You may make production use of the Licensed Work, provided that you may not use the Licensed Work to offer to third parties a commercial product or service whose primary purpose is AI agent orchestration and that substantially replicates the functionality of the Licensed Work.

**Alternative (narrower, more restrictive):**

> **Additional Use Grant:** None.

(All production use requires commercial license. Maximizes monetization, minimizes adoption.)

#### Parameter 2: Change Date

**Proposed: 4 years** (the maximum BSL 1.1 allows)

Rationale:
- jaato is a new project (v0.2.x) — needs maximum runway to establish commercial viability
- 4 years per version, rolling: v0.3 released today → open source March 2030
- The rolling nature means only the latest 4 years of code is BSL-restricted at any time
- Consider reducing to 3 years later once the commercial model is proven

#### Parameter 3: Change License

**Proposed: Apache 2.0**

Rationale:
- Most widely adopted Change License (used by CockroachDB, Sentry, Couchbase, etc.)
- Maximum compatibility with downstream projects
- Well-understood by legal departments
- Compatible with GPL v3+ (satisfying the BSL requirement)

### 6.2 Concrete License Header

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

### 6.3 What BSL Allows and Prohibits (Examples)

| Use Case | Allowed? | Notes |
|---|---|---|
| Internal use at a company of any size | Yes | No restrictions on internal production use |
| Building a product that uses jaato internally | Yes | jaato is a component, not the product |
| Academic research | Yes | Non-production use always allowed |
| Contributing to jaato | Yes | Development/testing is non-production |
| Running jaato for personal projects | Yes | Non-commercial production use |
| Forking and modifying for internal use | Yes | Modification always allowed |
| Offering "jaato Cloud" as a hosted service | **No** | Directly competitive hosted service |
| White-labeling jaato as "MyAgentFramework" and selling it | **No** | Substantially replicates functionality |
| Embedding jaato in a commercial IDE with agent features | **Gray area** | Depends on whether the IDE's primary purpose is agent orchestration |
| Building a competing agent framework from scratch, inspired by jaato's architecture | Yes | BSL restricts the *code*, not ideas |

---

## 7. Technical Implementation Plan

### 7.1 Phase 1: Prepare the Boundary (In Current Repo)

Before splitting, make the open/closed boundary a clean seam *within* the existing codebase:

1. **Extract embedded prompts from `jaato_runtime.py`**
   - Move `_TASK_COMPLETION_INSTRUCTION`, `_PARALLEL_TOOL_GUIDANCE`, `_TURN_SUMMARY_INSTRUCTION`, `_SANDBOX_GUIDANCE` into separate `.md` files
   - Replace with a file-based loader: `_load_instruction(name: str) -> str`
   - The loader checks (in order): entry points → workspace `.jaato/instructions/` → built-in fallbacks
   - Built-in fallbacks are minimal/generic (shipped in the open repo)

2. **Create an instruction provider interface**
   ```python
   class InstructionProvider(Protocol):
       def get_system_instructions(self) -> str: ...
       def get_runtime_instructions(self) -> Dict[str, str]: ...
       def get_profiles(self) -> List[SubagentProfile]: ...
   ```

3. **Make GC plugins truly pluggable**
   - Verify that `gc_hybrid` and `gc_budget` can be removed without breaking anything
   - Ensure `gc_truncate` and `gc_summarize` work as standalone defaults
   - Test that the plugin entry point system correctly discovers GC plugins from external packages

4. **Extract knowledge modules**
   - Ensure `knowledge/` and `tests_enablement_2.0/` have no imports into `jaato-server/shared/`
   - They should be loaded purely through the template/reference system, not Python imports

5. **Mark `.jaato/` contents as external configuration**
   - The runtime should treat `.jaato/instructions/`, `.jaato/profiles/`, `.jaato/prompts/` as user-provided configuration, not baked-in code
   - If these directories are empty/missing, the framework still works with defaults

### 7.2 Phase 2: Create Repo A (Open, BSL 1.1)

1. **Create new repository** `apanoia/jaato` (or rename current to `jaato-oss`)
2. **Copy the codebase** minus the closed-source assets (see Section 3.2)
3. **Add BSL 1.1 `LICENSE` file** with configured parameters
4. **Add `NOTICE` file** listing all third-party licenses
5. **Update all file headers** with BSL 1.1 copyright notice
6. **Write new `README.md`** — focused on framework capabilities, setup, plugin development
7. **Provide minimal `.jaato.example/`** — directory structure with placeholder instructions
8. **Update `pyproject.toml`** files — add `license = "BUSL-1.1"` classifier
9. **Set up CI/CD** — test pipeline for open repo (must pass without premium)
10. **Verify standalone operation** — full test suite passes without closed-source components

### 7.3 Phase 3: Create Repo B (Closed, Proprietary)

1. **Create private repository** `apanoia/jaato-premium`
2. **Structure as pip-installable package** (see Section 5.2)
3. **Implement entry point registration** for instructions, profiles, plugins
4. **Extract and organize all Tier 1 & Tier 2 assets**
5. **Write integration tests** — premium features work when installed alongside open repo
6. **Add proprietary `LICENSE`** — All Rights Reserved
7. **Set up private CI/CD** — tests run against open repo + premium overlay
8. **Verify integration** — `pip install jaato-server jaato-premium` enables all features

### 7.4 Phase 4: Ongoing Maintenance

1. **Develop features primarily in Repo A** (the framework)
2. **Add intelligence/prompts in Repo B** (the brain)
3. **Version in lockstep** — Repo B declares compatible Repo A versions
4. **CI for both** — Repo B's CI installs from Repo A's latest release and tests integration
5. **Contribution model** — external contributions to Repo A only; Repo B is internal

---

## 8. Dependency & Import Surgery

### 8.1 Current Import Graph (Relevant Edges)

```
jaato_session.py
  → jaato_runtime.py (for _TASK_COMPLETION_INSTRUCTION, etc.)
  → instruction_budget.py (for InstructionSource, GCPolicy)
  → plugins/gc_hybrid/
  → plugins/gc_budget/
  → plugins/reliability/

jaato_runtime.py
  → Contains embedded prompt strings (direct Python constants)
  → References .jaato/instructions/ via file I/O

server/core.py
  → jaato_runtime.py
  → session_manager.py
```

### 8.2 Required Refactors for Clean Split

#### A. Extract Runtime Prompts

**Before:**
```python
# jaato_runtime.py (current)
_TASK_COMPLETION_INSTRUCTION = """
## Evidence-Based Completion
Never fabricate results. Every claim must be grounded...
"""
```

**After:**
```python
# jaato_runtime.py (open repo)
def _load_instruction(name: str, fallback: str = "") -> str:
    """Load instruction from premium package, workspace, or fallback."""
    # 1. Try entry point (premium package)
    for ep in entry_points(group="jaato.runtime_instructions"):
        if ep.name == name:
            return ep.load()()
    # 2. Try workspace file
    workspace_path = Path(".jaato/instructions") / f"{name}.md"
    if workspace_path.exists():
        return workspace_path.read_text()
    # 3. Fallback
    return fallback

_TASK_COMPLETION_INSTRUCTION = _load_instruction(
    "task_completion",
    fallback="Complete tasks thoroughly and verify results."
)
```

#### B. Make GC Strategy Selection Dynamic

**Before:**
```python
# Hard import
from shared.plugins.gc_hybrid import HybridGCPlugin
```

**After:**
```python
# Dynamic loading via registry
gc_plugin = registry.load_gc_plugin(config.gc_type)  # "hybrid" -> entry point lookup
```

#### C. Instruction Budget Boundary Decision

**Option 1: Keep in open repo** (recommended)
- The data model (`InstructionSource`, `GCPolicy`, turn types) is useful infrastructure
- The actual instructions and their classification are in the closed repo
- The budget system is the "how"; the instruction content is the "what"

**Option 2: Move to closed repo**
- Cleaner separation but forces open repo to use a simpler GC model
- Requires a more complex interface between repos

**Recommendation:** Keep `instruction_budget.py` in the open repo. It's a data structure, not secret sauce.

---

## 9. Build & Distribution Architecture

### 9.1 Package Names

| Package | PyPI Name | Install Command |
|---|---|---|
| SDK | `jaato-sdk` | `pip install jaato-sdk` |
| Server | `jaato-server` | `pip install jaato-server` |
| TUI | `jaato-tui` | `pip install jaato-tui` |
| Premium | `jaato-premium` | `pip install jaato-premium` (private PyPI / direct) |

### 9.2 Premium Distribution Options

Since `jaato-premium` is closed source, it cannot be on public PyPI. Options:

1. **Private PyPI server** (e.g., Artifactory, AWS CodeArtifact, GCP Artifact Registry)
   ```bash
   pip install jaato-premium --index-url https://pypi.apanoia.dev/simple/
   ```

2. **Direct Git install** (for authorized users)
   ```bash
   pip install git+https://github.com/apanoia/jaato-premium.git
   ```

3. **Bundled distribution** (self-extracting archive)
   - The existing `create_self_extractor.py` could package both repos together
   - Ship as a single archive: open framework + premium overlay

4. **License key activation** (deferred)
   - Premium features are in the open repo but encrypted/gated behind a license check
   - Most complex, highest distribution friction, but single repo

### 9.3 Recommended Approach

Use **Private PyPI** for automated installs and **direct Git** for development:

```bash
# Development setup
git clone https://github.com/apanoia/jaato.git
git clone git@github.com:apanoia/jaato-premium.git  # requires access
cd jaato
python3 -m venv .venv
.venv/bin/pip install -e jaato-sdk/. -e "jaato-server/.[all]" -e "jaato-tui/.[all]"
.venv/bin/pip install -e ../jaato-premium/.

# Production setup (for licensed users)
pip install jaato-server jaato-tui
pip install jaato-premium --index-url https://pypi.apanoia.dev/simple/
```

---

## 10. Git History & Migration Strategy

### 10.1 Options for History

#### Option A: Clean Start (Recommended for Open Repo)

- Open repo starts with a clean initial commit
- No historical MIT-licensed commits visible
- Avoids any argument about retroactive license changes
- Closed repo keeps full history (private, no concern)

**Downsides:** Lose contributor attribution, lose `git blame` context

#### Option B: Filtered History

- Use `git filter-repo` to create open repo history minus closed-source files
- Preserves contributor attribution and blame
- Existing MIT-licensed history remains MIT; BSL applies only to new commits

**Downsides:** More complex, potential for accidentally including sensitive content

#### Option C: Full History with License Change Notice

- Keep full history in open repo
- Add prominent notice: "Commits before [date] are MIT-licensed; commits after [date] are BSL 1.1"
- Simpler than filtering

**Downsides:** History contains files that are now in the closed repo (system prompts, docs)

### 10.2 Recommendation

**Option A (Clean Start)** for the open repo, for these reasons:

1. The current repo has never been public — there's no existing community to alienate
2. No external contributors to attribute
3. Clean history avoids any ambiguity about which license applies to which code
4. The closed repo retains full history for internal reference

### 10.3 Handling the Current MIT License

The current `LICENSE` file says MIT. Since (presumably) there are no external contributors and all copyright is held by apanoia:

- **apanoia can re-license** their own code under BSL at any time
- **Previously released MIT versions** (if any were published) remain MIT forever — you cannot retroactively change the license on already-distributed copies
- **If there are external contributors**, you need either: their consent, a CLA that grants relicensing rights, or to rewrite their contributions

---

## 11. Contributor & Community Impact

### 11.1 Current State

Based on analysis, this appears to be a primarily single-developer or small-team project. The contributor dynamics are different from HashiCorp/Redis scenarios where large communities were affected.

### 11.2 If/When External Contributors Join

**For the open repo (BSL 1.1):**

- Implement a CLA (Contributor License Agreement) **before accepting contributions**
- The CLA should grant apanoia the right to use contributions under any license (enabling the premium offering)
- Use a standard CLA like Apache's ICLA or create a project-specific one
- Consider the DCO (Developer Certificate of Origin) as a lighter alternative — but it does NOT grant relicensing rights

**For the closed repo:**

- No external contributions accepted
- Internal development only

### 11.3 Community Messaging

Key messages to prepare:

1. **"Source-available, not open source"** — Be honest about BSL. Don't claim it's open source.
2. **"We're building in the open"** — The framework is fully functional standalone. Premium adds intelligence.
3. **"4 years → Apache 2.0"** — Everything eventually becomes truly open source.
4. **"Your internal use is unrestricted"** — BSL only restricts competing hosted services.
5. **"We welcome contributions to the framework"** — Plugin development, provider adapters, bug fixes.

---

## 12. Risks & Mitigations

### 12.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Two-repo development slows velocity | High | Medium | Keep the boundary narrow; most features land in the open repo |
| Open repo tests break without premium | Medium | High | CI must test open repo in isolation; premium features must be genuinely optional |
| Version drift between repos | Medium | Medium | Lock-step versioning; Repo B CI tests against Repo A's latest |
| Accidental leak of closed-source content | Low | High | Pre-commit hooks, CI checks, `.gitignore` patterns |
| Plugin interface changes break premium | Medium | Medium | Treat plugin interface as stable API; semver discipline |

### 12.2 Business Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Community perceives BSL as hostile | Medium | Medium | You're starting fresh (no "rug pull"); be transparent about rationale |
| Someone forks and adds their own prompts | Certain | Low-Medium | Expected; the prompt engineering is ongoing R&D, not a one-time effort |
| BSL ambiguity creates FUD for enterprise adoption | Medium | Medium | Clear FAQ, Additional Use Grant examples, offer commercial licenses |
| Closed repo value erodes as AI agent patterns become commoditized | Medium | High | Continuously invest in prompt engineering, knowledge modules, new strategies |
| Someone reverse-engineers prompts by observing agent behavior | Medium | Low | Prompts evolve; observing behavior doesn't capture the full system |

### 12.3 Legal Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| BSL "production use" ambiguity leads to disputes | Low | High | Clear Additional Use Grant; public FAQ with examples |
| Third-party dependency license conflicts | Low | Medium | Audit all dependencies; ensure no GPL dependencies in BSL code |
| CLA not in place when contributions arrive | Medium | Medium | Set up CLA bot (e.g., CLA Assistant) before first external PR |

---

## 13. Alternative Approaches

### 13.1 Alternative: FSL (Functional Source License) Instead of BSL

**What:** Sentry's standardized source-available license with 2-year conversion to Apache 2.0/MIT.

**Pros:**
- Standardized (no custom Additional Use Grant — less ambiguity)
- Shorter conversion period (2 years)
- Growing adoption (Sentry, GitButler, etc.)

**Cons:**
- Less flexibility — can't customize the competition clause
- Newer, less legally tested
- 2-year conversion may be too short for jaato's stage

**Verdict:** Consider for a future version. BSL's configurability is an advantage at this stage.

### 13.2 Alternative: AGPL v3 Instead of BSL

**What:** Strong copyleft open-source license. Network use triggers source disclosure.

**Pros:**
- Actually OSI-approved open source
- Prevents proprietary hosted services (they'd need to disclose source)
- No conversion date complexity

**Cons:**
- Doesn't prevent competitors from offering the service with source disclosed (they just open their modifications)
- Scares enterprise legal departments more than BSL
- Doesn't allow a "premium" tier — everything must be AGPL

**Verdict:** Doesn't serve the two-repo model. AGPL would require the premium content to also be AGPL if distributed together.

### 13.3 Alternative: Open Core (MIT/Apache + Proprietary Add-ons)

**What:** Keep the open repo MIT or Apache 2.0. Premium features are proprietary plugins.

**Pros:**
- True open source for the core — maximum community goodwill
- Clear "free vs. paid" boundary
- No BSL ambiguity

**Cons:**
- MIT/Apache allows anyone to build a competing hosted service from the core
- Less protection for the framework itself
- Only the premium plugins are protected, not the core

**Verdict:** Viable if the competitive advantage is truly *only* in the premium plugins and you're comfortable with the core being freely used by competitors. The concern is that the framework itself (parallel tool execution, session management, plugin architecture) has significant value that MIT would expose.

### 13.4 Alternative: Single Repo with License Partitioning

**What:** One repo, different licenses for different directories.

```
jaato/
├── LICENSE-BSL         # BSL 1.1 for framework
├── LICENSE-PROPRIETARY # Proprietary for premium/
├── jaato-sdk/          # BSL 1.1
├── jaato-server/       # BSL 1.1
├── jaato-tui/          # BSL 1.1
└── premium/            # Proprietary (not on GitHub, or .gitignore'd)
```

**Pros:**
- Single repo — simpler development workflow
- No version drift
- `premium/` is simply not included in public distribution

**Cons:**
- Confusing for contributors (what license applies where?)
- Risk of accidentally committing proprietary content
- Harder to enforce access control within a single repo

**Verdict:** Viable for early stages before going public. Transition to two repos when the community grows.

### 13.5 Alternative: Delayed Open Source (Time-Gated Releases)

**What:** All code is proprietary initially. After N years, each version is released under Apache 2.0.

**Pros:**
- Simpler than BSL (no parameterized license)
- Full control during the proprietary window
- Same end state as BSL

**Cons:**
- No source available during the proprietary window (BSL at least lets people read and modify)
- Less community engagement
- Essentially the same as BSL but with worse optics

**Verdict:** BSL is strictly better — it provides the same time-gating but with source availability.

---

## 14. Recommended Path Forward

### Phase 0: Decisions (Now)

1. Confirm the two-repo strategy (vs. alternatives in Section 13)
2. Finalize BSL Additional Use Grant wording
3. Decide on instruction budget placement (open vs. closed)
4. Choose premium distribution mechanism (private PyPI recommended)
5. Determine if any external contributors exist who need consent

### Phase 1: Prepare the Seam (1-2 weeks)

1. Extract embedded prompts from `jaato_runtime.py` into file-based loading
2. Implement instruction provider interface with entry point discovery
3. Ensure GC plugins are dynamically loaded (no hard imports of hybrid/budget)
4. Verify framework operates correctly with minimal/stub instructions
5. Create comprehensive test suite for "open repo standalone" mode

### Phase 2: Split (1 week)

1. Create the open repo (clean start, BSL 1.1)
2. Create the private premium repo
3. Set up CI/CD for both
4. Verify: `pip install jaato-server` works standalone
5. Verify: `pip install jaato-server jaato-premium` enables all features

### Phase 3: Harden (1-2 weeks)

1. Set up CLA bot on open repo
2. Write public FAQ for BSL licensing
3. Audit third-party dependency licenses
4. Set up pre-commit hooks to prevent cross-contamination
5. Write contributor guidelines

### Phase 4: Launch

1. Open repo goes public on GitHub
2. Announce with transparent blog post about licensing rationale
3. Private repo available to licensed customers/partners

---

## 15. Decision Matrix

| Decision | Option A | Option B | Option C | Recommendation |
|---|---|---|---|---|
| **License for open repo** | BSL 1.1 | FSL 2.0 | AGPL v3 | **BSL 1.1** (configurability, precedent) |
| **Additional Use Grant** | Anti-competitive-service | Revenue threshold | None | **Anti-competitive-service** (broad adoption) |
| **Change Date** | 3 years | 4 years | — | **4 years** (maximum runway) |
| **Change License** | Apache 2.0 | MIT | MPL 2.0 | **Apache 2.0** (industry standard) |
| **instruction_budget.py** | Keep in open | Move to closed | — | **Keep in open** (it's infrastructure) |
| **Git history** | Clean start | Filtered | Full with notice | **Clean start** (no prior public history) |
| **Premium distribution** | Private PyPI | Git install | Bundled archive | **Private PyPI** (professional, automatable) |
| **Integration mechanism** | Entry point overlay | Directory overlay | Git submodule | **Entry point overlay** (cleanest) |
| **Mono vs. multi-repo** | Two repos | Single with partitioning | — | **Two repos** (clear boundary) |
| **GC in open repo** | truncate + summarize | All four strategies | — | **truncate + summarize** (hybrid/budget = premium) |

---

## 16. Open Questions

1. **Are there any external contributors?** — If so, do we have CLAs or need consent for relicensing?

2. **Is the framework commercially viable without the premium?** — If the open repo is "too good," does it undermine the premium offering? Conversely, if it's "too bare," does it fail to attract users?

3. **Should model provider adapters be premium?** — Currently proposed as open. But the 8-provider support is significant engineering. Counter-argument: providers are integration code, not proprietary logic, and keeping them open encourages provider ecosystem growth.

4. **What about the TUI?** — The TUI is proposed as open (BSL). The output formatting, agent rendering, and UX design in `output_buffer.py` (232KB!) and `pt_display.py` (129KB) represent significant effort. Is the TUI itself competitive advantage?

5. **How do we handle the `.jaato/` directory in practice?** — If a user's workspace has `.jaato/instructions/` from the premium package, and they push their repo to GitHub, they'd be redistributing proprietary content. Need clear guidance or `.gitignore` conventions.

6. **Should the server's embedded prompts in `core.py` and `session_manager.py` also be extracted?** — `core.py` is 125KB, `session_manager.py` is 93KB — there may be embedded strings with competitive value beyond what's in `jaato_runtime.py`.

7. **What's the pricing model for jaato-premium?** — Per-seat, per-organization, usage-based? This affects how the Additional Use Grant is worded.

8. **Do we need a dual-license option?** — Some BSL projects offer "contact us for a commercial license" for cases that fall outside the Additional Use Grant. Worth setting up the infrastructure even if not immediately needed.

9. **Should `CLAUDE.md` itself be open or closed?** — It contains the full architecture summary. It's incredibly useful for contributors but also reveals strategic thinking. Could ship a condensed version in the open repo.

10. **What about the web-client?** — If a web-based client is part of the product offering, should it be open (to encourage web integration) or closed (to differentiate the hosted experience)?
