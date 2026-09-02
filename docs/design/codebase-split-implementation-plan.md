# Codebase Split — Implementation Plan

**Approach:** Incremental (public repo first, premium repo second)

**Rationale:** Each step is testable in isolation. The public repo stays working
at every commit. Stream A (methodology) is higher-value, lower-risk — do it
first. Gossip interface (riskiest part) benefits from lessons learned during
the simpler content extraction.

---

## Phase 1: Entry-point infrastructure for premium hooks

### Task 1.1 — Add `jaato.premium` entry-point group

Add a new entry-point group to `jaato-server/pyproject.toml` for premium
extensions. This is the hook that `jaato-premium` will register with. No
built-in entries — just the group definition and loader code.

### Task 1.2 — Make prompt constants pluggable in `jaato_runtime.py`

Replace the 3 hardcoded constants + `_get_sandbox_guidance()` with an
entry-point lookup:
- Check `jaato.premium` for a `prompt_provider` entry point
- If not found, use the current strings as-is (they stay in the public repo —
  they're useful but not the crown jewels; the 19 system instruction principles
  are the real IP)
- The `get_system_instructions()` method (line ~1066) calls the provider
  instead of referencing module constants

**Files:**
- `jaato-server/shared/jaato_runtime.py` — lines 26-68 (constants), lines
  1066-1076 (usage in `get_system_instructions()`)

### Task 1.3 — Make gossip initialization pluggable in `__main__.py`

Replace `_init_gossip()` body with entry-point lookup:
- Check `jaato.gossip` for an `init` entry point
- If not found, return (no gossip — single-server mode)
- If found, call it with a context dataclass containing: `session_manager`,
  `web_socket` address, `ipc_socket` path, `server_name`, and a callback to
  set references back
- The gossip module returns `(peer_registry, health_collector,
  remote_handler, server_reliability)` which the daemon stores

The current wiring flow stays the same — `set_gossip_context()`,
`_configure_gossip_context()`, plugin `set_*_context()` methods all remain
as public hook points.

**Files:**
- `jaato-server/server/__main__.py` — `_init_gossip()` (lines 915-1029),
  `set_gossip_context()` call (lines 1056-1062)
- `jaato-server/server/session_manager.py` — `set_gossip_context()` (lines
  222-245), `_configure_gossip_context()` (lines 247-277)

### Task 1.4 — Verify standalone mode

- Run full test suite without any premium package installed
- Verify `get_system_instructions()` still returns the prompt constants
- Verify `_init_gossip()` no-ops cleanly
- Verify no import errors from missing gossip modules

---

## Phase 2: Create `jaato-premium` repo

### Task 2.1 — Scaffold repo structure

```
jaato-premium/
├── pyproject.toml          # Commercial license, depends on jaato-server>=X
├── LICENSE                  # All Rights Reserved
├── jaato_premium/
│   ├── __init__.py
│   ├── prompts.py          # prompt_provider entry point
│   ├── instructions/       # (empty initially)
│   ├── knowledge/          # (empty initially)
│   ├── profiles/           # (empty initially)
│   ├── references/         # (empty initially)
│   ├── prompt_templates/   # (empty initially)
│   ├── prompts/            # (empty initially)
│   └── gossip/             # (empty initially)
└── tests/
```

### Task 2.2 — Wire entry points in `pyproject.toml`

```toml
[project.entry-points."jaato.premium"]
prompt_provider = "jaato_premium.prompts:get_prompts"

[project.entry-points."jaato.gossip"]
init = "jaato_premium.gossip:init_gossip"
```

### Task 2.3 — Verify integration

- `pip install -e jaato-premium/` alongside `jaato-server`
- Verify entry points are discovered
- Verify prompt provider is called
- Verify gossip init is called (even if gossip modules aren't moved yet)

---

## Phase 3: Move methodology content

### Task 3.1 — Prompt constants stay public

The 3 framework prompt constants (`_TASK_COMPLETION_INSTRUCTION`,
`_PARALLEL_TOOL_GUIDANCE`, `_TURN_SUMMARY_INSTRUCTION`) are necessary for
correct agent behavior (safety, efficiency, GC) and stay in `jaato_runtime.py`
as functional defaults.

The premium package can provide **enhanced versions** via the
`prompt_provider` entry point but the base agent works correctly without it.

### Task 3.2 — Move content files

- `.jaato/instructions/00-system-instructions.md` → `jaato_premium/instructions/`
- `.jaato/profiles/*.json` (14 files, not github-resolver) → `jaato_premium/profiles/`
- `.jaato/references/*.json` → `jaato_premium/references/`
- `.jaato/prompts/*.md` (premium prompts) → `jaato_premium/prompts/`
- `knowledge/` → `jaato_premium/knowledge/`
- `shared/prompt_templates/` → `jaato_premium/prompt_templates/`
- `modlog-training-set-test/`, `cli_vs_mcp/` → `jaato_premium/`

### Task 3.3 — Wire content loading

Premium needs to register paths so the framework discovers the content:
- Instructions: register via `jaato.premium` → `instructions` entry point
  returning a path
- Profiles: register via `jaato.premium` → `profiles` entry point returning
  a path
- The framework's existing file-discovery mechanisms (`.jaato/instructions/`,
  `.jaato/profiles/`) are extended to also check entry-point-provided paths

---

## Phase 4: Move gossip modules

### Task 4.1 — Move gossip code to premium

- `server/peers.py` → `jaato_premium/gossip/peers.py`
- `server/remote_spawn.py` → `jaato_premium/gossip/remote_spawn.py`
- `server/workspace_sync.py` → `jaato_premium/gossip/workspace_sync.py`
- `server/server_reliability.py` → `jaato_premium/gossip/server_reliability.py`
- `server/health.py` → `jaato_premium/gossip/health.py`
- `server/health_http.py` → `jaato_premium/gossip/health_http.py`
- `server/dashboard/` → `jaato_premium/gossip/dashboard/`

### Task 4.2 — Implement `init_gossip()` entry point

The `jaato_premium/gossip/__init__.py` implements the init function that:
- Receives the daemon context
- Creates PeerRegistry, HealthCollector, RemoteSpawnHandler,
  ServerReliabilityTracker
- Returns them to the daemon
- Essentially the body of today's `_init_gossip()` but living in premium

### Task 4.3 — Move E2E tests

- `tests/e2e/gossip/` → `jaato_premium/tests/e2e/gossip/`
- `tests/e2e/workspace-sync/` → `jaato_premium/tests/e2e/workspace-sync/`

### Task 4.4 — Final integration testing

- Public repo alone: all tests pass, gossip no-ops, prompts are generic
- Public + premium: all features work, gossip activates, premium prompts loaded

---

## Phase 5: Clean up public repo

### Task 5.1 — Update `.jaato.example/`

Already partially done — `github-resolver.json` and `gh_issue_fixer.md`
are in place as working examples.

### Task 5.2 — Update `CLAUDE.md`

Remove references to moved content (knowledge, premium profiles, etc.).

### Task 5.3 — Update `README.md`

Mention premium as optional. Add installation instructions for premium.

### Task 5.4 — Update `pyproject.toml` license classifiers

Change from MIT to `BUSL-1.1` in all three packages.

### Task 5.5 — Add BSL 1.1 `LICENSE` file

Replace the MIT LICENSE file with BSL 1.1 using the parameters from the
design doc (4-year change date, Apache 2.0 change license, anti-competitive
additional use grant).
