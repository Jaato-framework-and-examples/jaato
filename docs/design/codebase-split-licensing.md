# Codebase Split & Licensing Design

## Distribution Model

| Package | License | Distribution | Install |
|---------|---------|-------------|---------|
| **jaato-sdk** | MIT | Public PyPI | `pip install jaato-sdk` |
| **jaato-server** | MIT | Public PyPI | `pip install jaato-server` |
| **jaato-tui** | MIT | Public PyPI | `pip install jaato-tui` |
| **jaato-premium** | Commercial | Private GitHub repo | See [Premium Installation](#premium-installation) |

The public packages remain exactly where they are (same repo, same PyPI).
`jaato-premium` is a **separate private repo** that depends on `jaato-server`
and extends it via the existing plugin entry-point system.

---

## Installation

### Public Packages (MIT — anyone)

Published on PyPI. Standard pip install:

```bash
pip install jaato-sdk jaato-server jaato-tui
```

Or with optional extras:

```bash
pip install "jaato-server[all]" "jaato-tui[all]"
```

These give you the full open-source framework: 8 model providers, 58 plugins,
TUI client, web client, GC strategies, telemetry — everything needed to build
and run agentic AI applications.

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
jaato-server>=0.2.48
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

### PUBLIC (MIT) — stays in this repo

Everything that is **framework plumbing** — the engine that makes tools run,
providers connect, sessions manage state. A fully functional agentic
orchestrator, but without the opinionated "secret sauce."

#### jaato-sdk (unchanged)
- IPC/WebSocket client protocol
- Base plugin interfaces
- Event types, model provider types

#### jaato-server — core
- `shared/jaato_client.py`, `jaato_runtime.py`, `jaato_session.py`
- `shared/instruction_budget.py`, `shared/token_accounting.py`
- `shared/ai_tool_runner.py`, `shared/mcp_context_manager.py`
- `shared/plugins/base.py`, `shared/plugins/registry.py`
- `server/` (core, ipc, websocket, session_manager, etc.)

#### jaato-server — standard plugins (58 plugin directories)
All existing plugins stay MIT. They are the framework's value as an
open-source project. Specifically:

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

Content that represents **opinionated methodology, curated knowledge, and
behavioral tuning** — the things that make jaato agents work *well* rather
than just *work*.

#### 1. System Instructions (`instructions/`)
**Source:** `.jaato/instructions/00-system-instructions.md`

The 19 behavioral principles (Transparency Mandate, Large Output Protocol,
Autonomous Decision Making, Anti-Fabrication, Delegation Authority, etc.)
are the single most valuable piece of IP. They encode months of iteration
on how to make LLM agents behave reliably.

These become the premium package's default instructions, loaded via
the same `.jaato/instructions/` mechanism the framework already supports.

#### 2. Knowledge Base (`knowledge/`)
**Source:** `knowledge/` (155 files)

- `ADRs/` — Architecture Decision Records (6 ADRs)
- `ERIs/` — Executable Reference Implementations (8 ERIs)
- `modules/` — Code generation modules with templates and validation
  (circuit-breaker, retry, timeout, rate-limiter, hexagonal-base,
  persistence-jpa, persistence-systemapi, api-integration, api-exposure,
  compensation) — 10 modules with Handlebars templates
- `model/` — Knowledge model definitions (domains, standards, authoring prompts)

#### 3. Subagent Profiles (`profiles/`)
**Source:** `.jaato/profiles/*.json`

Curated subagent profile definitions:
- `skill-code-*` / `skill-mod-code-*` — Coding specialist profiles
- `validator-tier*` — Multi-tier validation profiles
- `analyst-*` — Analysis profiles

#### 4. Reference Catalog (`references/`)
**Source:** `.jaato/references/*.json`

Pre-built reference JSON files that link ADRs, ERIs, and modules
into a structured catalog with semantic embeddings and validation rules.

#### 5. Prompt Templates (`prompt_templates/`)
**Source:** `jaato-server/shared/prompt_templates/`

- COBOL analysis prompts (identify_code_changes, parse_mod_history)
- Confluence integration prompts (get_page, search, update_page — CLI & MCP)
- GitHub integration prompts (get_issue, list_issues, search_issues — CLI & MCP)

#### 6. Curated Prompts (`prompts/`)
**Source:** `.jaato/prompts/gen-references.md`

The gen-references prompt — a sophisticated prompt for scanning knowledge
bases and generating reference catalogs, template indexes, and subagent profiles.

#### 7. Framework Prompt Constants
**Source:** `jaato-server/shared/jaato_runtime.py` (lines 26-49)

Three embedded prompt constants:
- `_TASK_COMPLETION_INSTRUCTION` — Anti-fabrication + relentless completion
- `_PARALLEL_TOOL_GUIDANCE` — Parallel tool batching guidance
- `_TURN_SUMMARY_INSTRUCTION` — Turn-end summarization guidance

These are injected into every system prompt. In the split:
- The public repo keeps **generic placeholders** (or empty strings)
- The premium repo provides these via a hook/override mechanism

#### 8. Training Data & Specialized Tools
- `modlog-training-set-test/` — COBOL modification log training set generator
- `cli_vs_mcp/` — CLI vs MCP comparison harness
- `create_self_extractor.py` — Self-extracting archive builder

---

## Implementation Approach

### Step 1: Create premium plugin entry point in jaato-server

Add a `jaato.premium` entry-point group to `jaato-server/pyproject.toml`
that premium plugins can register with. The runtime checks for these at
startup and loads them if present.

### Step 2: Make framework prompt constants pluggable

In `jaato_runtime.py`, replace the hardcoded `_TASK_COMPLETION_INSTRUCTION`,
`_PARALLEL_TOOL_GUIDANCE`, and `_TURN_SUMMARY_INSTRUCTION` with a lookup
that:
1. Checks if a premium prompt provider is registered (via entry point)
2. Falls back to generic defaults if not

### Step 3: Create jaato-premium repo structure

```
jaato-premium/
├── pyproject.toml              # Commercial license, depends on jaato-server
├── LICENSE                     # Commercial/BSL license
├── README.md
├── jaato_premium/
│   ├── __init__.py
│   ├── prompts.py              # Premium prompt constants (the 3 from jaato_runtime)
│   ├── instructions/           # 00-system-instructions.md (19 principles)
│   ├── knowledge/              # ADRs, ERIs, modules, model
│   ├── profiles/               # Subagent profiles
│   ├── references/             # Reference catalog JSONs
│   ├── prompt_templates/       # COBOL, Confluence, GitHub prompts
│   └── prompts/                # gen-references.md and others
├── setup.cfg                   # Entry point registration
└── tests/
```

### Step 4: Wire premium content loading

The premium package registers itself via entry points:

```toml
# jaato-premium/pyproject.toml
[project.entry-points."jaato.premium"]
prompt_provider = "jaato_premium.prompts:get_prompts"
instructions = "jaato_premium:get_instructions_path"
knowledge = "jaato_premium:get_knowledge_path"
```

### Step 5: Move content from public repo

Move (not copy) the premium content out of the public repo:
- `.jaato/instructions/00-system-instructions.md` → jaato-premium
- `.jaato/profiles/*.json` → jaato-premium
- `.jaato/references/*.json` → jaato-premium
- `.jaato/prompts/gen-references.md` → jaato-premium
- `knowledge/` → jaato-premium
- `shared/prompt_templates/` → jaato-premium
- `modlog-training-set-test/` → jaato-premium
- `cli_vs_mcp/` → jaato-premium

Replace the 3 prompt constants in `jaato_runtime.py` with generic fallbacks.

### Step 6: Update public repo

- Keep `.jaato/instructions/` as an empty directory with a README
  explaining that users can add their own instructions
- Keep `.jaato/profiles/` empty with README
- Keep `.jaato/references/` empty with README
- Update `CLAUDE.md` to remove references to moved content
- Update `README.md` to mention the premium package as optional

---

## What This Preserves

- **Zero breaking changes** — the public framework works exactly as before
- **Plugin architecture unchanged** — premium is just more plugins
- **Existing users unaffected** — `pip install jaato-server` still works
- **Clear value boundary** — framework (free) vs methodology (premium)
- **Simple upgrade path** — `pip install git+ssh://...` adds premium on top

## What Premium Users Get

1. Battle-tested system instructions (19 principles of agent behavior)
2. Knowledge base with ADRs, ERIs, and code generation modules
3. Pre-built subagent profiles for coding, validation, and analysis
4. Reference catalog with semantic matching
5. Domain-specific prompt templates (COBOL, Confluence, GitHub)
6. Optimized framework prompts (anti-fabrication, parallel batching, summarization)
