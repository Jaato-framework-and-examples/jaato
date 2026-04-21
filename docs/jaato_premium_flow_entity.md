# JAATO Flow Entity — Complete Reference

> Scope: How to define, author, validate, and execute multi-step Flow entities in jaato-premium, including the full YAML schema, data models, lifecycle hooks, context bag, conditional execution, groups, and the authoring tool chain.

## Table of Contents

1. [What Is a Flow Entity?](#1-what-is-a-flow-entity)
2. [Architecture Overview — How Flows Work](#2-architecture-overview--how-flows-work)
3. [Execution Models: Scripted vs. Agentic](#3-execution-models-scripted-vs-agentic)
4. [Flow YAML Schema](#4-flow-yaml-schema)
5. [States — The Core Execution Unit](#5-states--the-core-execution-unit)
6. [Data Bindings: Artifacts and Context](#6-data-bindings-artifacts-and-context)
7. [Conditional Execution with `when:`](#7-conditional-execution-with-when)
8. [Groups — Iterative Build-Validate Cycles](#8-groups--iterative-build-validate-cycles)
9. [Hooks — Lifecycle Boundary Scripts](#9-hooks--lifecycle-boundary-scripts)
10. [Parameters and Variable Expansion](#10-parameters-and-variable-expansion)
11. [Context Bag — Flow-Scoped Key-Value Store](#11-context-bag--flow-scoped-key-value-store)
12. [Dependency DAG — Compile-Time Verification](#12-dependency-dag--compile-time-verification)
13. [Completion Criteria](#13-completion-criteria)
14. [Rules — Agentic Flow Constraints](#14-rules--agentic-flow-constraints)
15. [Authoring Tools — begin/build/update/commit](#15-authoring-tools--beginbuildupdatecommit)
16. [Execution Engine — ScriptedFlowRunner](#16-execution-engine--scriptedflowrunner)
17. [Configuration / Schema / API Reference](#17-configuration--schema--api-reference)
18. [Runtime Internals — What the Source Code Reveals](#18-runtime-internals--what-the-source-code-reveals)
19. [Source Code Map](#19-source-code-map)

---

## 1. What Is a Flow Entity?

A **Flow entity** is a declarative YAML file (`kind: Flow`) that defines a deterministic or adaptive multi-step pipeline for code generation, validation, and orchestration. Flows are the primary mechanism in **jaato-premium** for coordinating complex, multi-agent workflows such as:

- End-to-end microservice generation (scaffold → persistence → API → resilience → validation)
- Adaptive feature implementation (analysis → selective activation → validation)
- Any pipeline requiring ordered state execution with data dependencies

Each Flow YAML file defines:
- **States** — individual execution steps (each spawns a subagent with a profile)
- **Groups** — iterative loops of states (e.g., build → validate → fix → re-validate)
- **Data bindings** — `from:` references that create a compile-time-verifiable dependency DAG
- **Context bag** — a flow-scoped key-value store that accumulates metadata across states
- **Hooks** — shell scripts that run at lifecycle boundaries (pre/post flow, pre/post state)
- **Conditions** — `when:` expressions that enable conditional state activation
- **Parameters** — user-supplied values that are expanded via `{{param}}` syntax
- **Completion criteria** — rules that determine whether a flow succeeded

Flows are authored using the tool chain: `begin_flow` → `build_step` → `update_step` → `commit_flow`, and executed with `start_flow`. The entire system lives in the `jaato_premium.flow_runner` package.

---

## 2. Architecture Overview — How Flows Work

```
┌─────────────────────────────────────────────────────────┐
│  Flow YAML (kind: Flow)                                 │
│  ├── params (user inputs + defaults)                    │
│  ├── hooks.pre_flow                                     │
│  ├── states[] or groups[]                               │
│  │   ├── State A → profile: "skill-xxx"                 │
│  │   ├── State B → profile: "validator-xxx"             │
│  │   └── Group C → states: [D, E, F], until: expr      │
│  ├── hooks.post_flow                                    │
│  ├── completion criteria                                │
│  └── rules (agentic only)                               │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────┐
│  loader.py      │───▶│  dag.py          │───▶│  runner.py   │
│  Parse YAML     │    │  Validate DAG    │    │  Execute     │
│  Resolve params │    │  Check ownership │    │  states      │
│  Expand {{}}    │    │  Topo sort       │    │  groups      │
└─────────────────┘    └──────────────────┘    │  hooks       │
                                                │  context bag │
┌─────────────────┐    ┌──────────────────┐    └──────┬───────┘
│  conditions.py  │    │  hooks.py        │           │
│  Evaluate when: │    │  Run shell       │◀──────────┘
│  ctx.key==val   │    │  Parse CTX_SET   │
└─────────────────┘    └──────────────────┘

┌─────────────────┐    ┌──────────────────┐    ┌──────────────┐
│  models.py      │    │  context.py      │    │  authoring.py│
│  Flow, State,   │    │  ContextBag      │    │  FlowDraft   │
│  StateGroup,    │    │  ownership rules │    │  validation  │
│  Hook, Rule...  │    │  persistence     │    │  incremental │
└─────────────────┘    └──────────────────┘    └──────────────┘
```

**Data flow through a flow execution:**

1. User provides **parameters** → `{{param}}` placeholders are expanded in the YAML
2. **pre_flow hooks** run (e.g., verify inputs exist, create directories)
3. For each state/group in declaration order:
   a. Evaluate `when:` condition against the **context bag**
   b. Run **pre_validate hooks** (e.g., idempotency guard)
   c. Spawn a **subagent** with the state's profile and prompt
   d. Run **post_validate hooks** (e.g., check outputs exist, extract context values)
   e. **Commit context values** to the context bag
4. **post_flow hooks** run (e.g., print reports)
5. **Completion criteria** are checked

---

## 3. Execution Models: Scripted vs. Agentic

| Aspect | Scripted | Agentic |
|--------|----------|---------|
| **State ordering** | Declaration order, strictly top-to-bottom | Orchestrator agent decides at runtime |
| **Conditional execution** | `when:` expressions evaluated by engine | `when:` expressions + orchestrator discretion |
| **Agent role** | Engine walks states; each state spawns a subagent | Orchestrator agent manages everything |
| **Rules** | Not applicable | Author-defined rules constrain orchestrator |
| **How to start** | `start_flow(flow_name, params)` | Executed as a regular jaato-task session |
| **Groups** | Engine manages iteration (`until:` + `max_iterations`) | Orchestrator manages iteration |
| **State catalogue** | Executed states are all states (minus skipped `when:`) | States are *available* — orchestrator selects which to activate |

> **Key insight:** In scripted mode, the YAML defines the execution plan. In agentic mode, the YAML defines the *capability catalogue* — the orchestrator agent reads the state list, dependency DAG, and rules, then decides what to do.

---
## 4. Flow YAML Schema

The top-level structure of a Flow YAML file:

```yaml
kind: Flow                    # Required: must be exactly "Flow"
version: "1"                  # Schema version
name: my-flow-name            # Required: kebab-case identifier
description: |                # Human-readable description
  What this flow does.
execution_model: scripted     # "scripted" or "agentic"
tags:                         # Discovery/filtering tags
  - generation
  - microservice

params:                       # Parameter definitions
  param_name:
    required: true|false
    type: string|integer
    default: "value"           # Used when not provided
    description: "What this param controls"

hooks:                        # Global lifecycle hooks
  pre_flow:
    - name: verify-inputs
      run: |
        test -f "{{inputs}}/prompt.md"
      on_failure: abort       # abort|warn
  post_flow:
    - name: print-context
      run: cat .jaato/.context-bag.json
      on_failure: warn

states:                       # Ordered list of states and groups
  - name: first-state
    profile: skill-xxx
    prompt: "Do X with {{inputs}}"
    when: "ctx.has_outbound == true"
    inputs:
      artifacts: [...]
      context: [...]
    outputs:
      artifacts: [...]
      context: [...]
    hooks:
      pre_validate: [...]
      post_validate: [...]
    max_retries: 1

  - group: build-validate-cycle
    max_iterations: 3
    until: "ctx.all_validations_passed == true"
    on_max_iterations: warn    # abort|warn
    states:
      - name: build-step
        ...

rules:                        # Agentic flows only
  - id: orchestrator-never-implements
    severity: must             # must|should|prefer
    scope: global              # global
    description: "..."
    rationale: "..."

completion:                    # Success criteria
  require_all_states: true
  required_outputs:
    - from: first-state.output-name
  summary_report: "{{workspace}}/.jaato/reports/report.md"
```

---

## 5. States — The Core Execution Unit

A **state** is a single execution step in a flow. Each state:
1. Optionally checks a `when:` condition
2. Runs pre_validate hooks
3. Spawns a **subagent** with the specified profile and prompt
4. Runs post_validate hooks
5. Commits context values to the context bag

### State fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| **name** | string | Yes | — | Kebab-case identifier. Must match `^[a-z][a-z0-9-]*$`. Must be unique within the flow (including inside groups). |
| **description** | string | No | `""` | Human-readable description of what this state does. |
| **profile** | string | No* | — | Subagent profile to use (e.g., `"skill-mod-code-015-hexagonal-base"`). Either `profile` or `prompt` must be present. |
| **agent** | string | No | `""` | Optional agent name (markdown instructions file). When provided, overrides the profile's system instructions. |
| **prompt** | string | No* | — | Task prompt for the subagent. Supports `{{param}}` and `{{ctx.key}}` placeholders. |
| **when** | string | No | `null` | Conditional expression. State is skipped if expression evaluates to `false`. See §7. |
| **inputs** | object | No | `{}` | Artifact and context inputs. See §6. |
| **outputs** | object | No | `{}` | Artifact and context outputs. See §6. |
| **hooks** | object | No | `{}` | Pre/post validate hooks. See §9. |
| **max_retries** | integer | No | `0` | Maximum retry attempts if hooks return `retry`. |
| **retry_delay_seconds** | integer | No | `0` | Delay between retries. |
| **transition** | object | No | `null` | Post-state routing (`next`, `on_failure`, `condition`). Currently not fully implemented in the runner. |

> **Validation rule:** A state must have at least `profile` or `prompt`. If neither is provided, the authoring engine emits a warning (not an error).

---

## 6. Data Bindings: Artifacts and Context

Data flows between states through two mechanisms: **artifacts** (files) and **context** (key-value metadata).

### Artifact Inputs

Artifact inputs declare which files a state needs to read. Two forms:

```yaml
# 1. Raw path (literal file glob)
inputs:
  artifacts:
    - path: "{{inputs}}/prompt.md"
    - path: "{{inputs}}/**"              # Glob expansion

# 2. From reference (consume another state's output)
inputs:
  artifacts:
    - from: generate-hexagonal-base.domain-model
      alias: domain-entities              # Optional alias
```

The `from:` syntax uses the pattern `state-name.artifact-name` and creates a **dependency edge** in the DAG. The referenced state must be declared earlier in the flow (or within the same group).

### Artifact Outputs

```yaml
outputs:
  artifacts:
    - name: pom                          # Logical name (used in from: references)
      path: "{{workspace}}/pom.xml"       # File path or glob
      format: xml                        # Format hint (xml, java, yaml, markdown, json, any)
      description: "Maven project descriptor."
      schema:                             # Optional JSON schema for structured outputs
        type: object
        required: [nodes, edges]
        properties:
          nodes:
            type: array
```

### Context Inputs

Context inputs declare which context keys a state reads:

```yaml
inputs:
  context:
    - key: base_package
      from: generate-hexagonal-base       # Producing state name
      required: true                      # Error if missing at runtime
    - key: has_outbound_ports
      from: generate-hexagonal-base
```

### Context Outputs

Context outputs declare which keys a state **produces** (writes to the context bag):

```yaml
outputs:
  context:
    - key: has_outbound_ports
      type: boolean
      description: "Whether outbound port interfaces exist."
    - key: entity_count
      type: integer
    - key: base_package
      type: string
```

> **Ownership rule:** Each context key must have exactly one producing state. If two states declare the same key as output, the DAG validator raises a `DAGError`. At runtime, `ContextBag.set()` enforces this — a state cannot overwrite another state's keys.

---

## 7. Conditional Execution with `when:`

The `when:` field on a state (or `until:` on a group) contains a minimal expression evaluated against the context bag.

### Supported syntax

```python
ctx.<key> == <value>          # Equality
ctx.<key> != <value>          # Inequality
ctx.<key> > <number>          # Numeric comparison (>, <, >=, <=)
ctx.<key> in [<a>, <b>, <c>]  # Membership test
<expr> and <expr>             # Logical AND
<expr> or <expr>              # Logical OR
not <expr>                    # Negation
(ctx.<expr>)                  # Parenthesized grouping
```

### Examples from real flows

```yaml
when: "ctx.has_outbound_ports == true"
when: "ctx.requires_persistence == true"
when: "ctx.has_public_api == true"
when: "ctx.requires_compensation == true"
when: "ctx.all_validations_passed == true"    # Group until:
when: "ctx.tier1_passed == true and ctx.tier2_passed == true"
```

### Evaluation semantics (from `conditions.py`)

- **Empty/null expression** → `true` (state always executes)
- **Unrecognized expression** → `false` (with a warning log)
- **Missing context key** → treated as `null`; equality with `true`/`false` fails, `!= true` succeeds
- **Type coercion**: `"true"` / `"false"` → `bool`, numeric strings → `int`/`float`, JSON arrays/objects parsed
- **Operator precedence**: `not` > `and` > `or` (standard)
- **Parentheses** supported for explicit grouping

---

## 8. Groups — Iterative Build-Validate Cycles

A **group** wraps multiple states into an iterative loop. This is the primary mechanism for build-validate-fix cycles.

```yaml
- group: build-validate-cycle
  description: "Build and validate; loop on failure."
  max_iterations: "{{max_fix_iterations}}"   # Param-expanded
  until: "ctx.all_validations_passed == true"
  on_max_iterations: warn                     # "abort" or "warn"
  states:
    - name: add-persistence-jpa
      ...
    - name: validate-tier1
      ...
    - name: validate-tier3
      hooks:
        post_validate:
          - name: aggregate-validation-results
            run: |
              FAILURES=$(grep -rl "FAIL" reports/ | wc -l)
              if [ "$FAILURES" -gt 0 ]; then
                echo "JAATO_CTX_SET all_validations_passed=false"
                exit 1
              else
                echo "JAATO_CTX_SET all_validations_passed=true"
              fi
            on_failure: warn
```

### Group execution rules (from `runner.py`)

1. For each iteration (1 to `max_iterations`):
   - Set `ctx.<group-name>.iteration` to the current iteration number
   - Execute all states in declaration order
   - If any state **fails** → group aborts immediately
   - After all states complete → evaluate `until:` condition
   - If `until:` is `true` → group succeeds
2. If max_iterations reached without satisfying `until:`:
   - `on_max_iterations: abort` → flow fails
   - `on_max_iterations: warn` → flow continues with a warning

### Group constraints

- **No nesting**: Groups cannot contain other groups (validated at authoring time)
- **Internal dependencies**: States within a group can reference each other via `from:`, but external dependencies must be declared before the group
- **Context relaxation**: On subsequent iterations, a state *can* overwrite its own context keys from prior iterations (the ownership check allows same-producer overwrites)

---

## 9. Hooks — Lifecycle Boundary Scripts

Hooks are **shell scripts** that run at lifecycle boundaries. They receive the context bag as environment variables and can produce new context values.

### Hook points

| Hook Point | When it runs | Valid on_failure modes |
|------------|-------------|----------------------|
| `pre_flow` | Before any state executes | `abort`, `warn` |
| `post_flow` | After all states complete (even on abort) | `abort`, `warn` |
| `pre_validate` | Before a state's agent is invoked | `abort`, `skip`, `retry`, `warn` |
| `post_validate` | After a state's agent completes | `abort`, `retry`, `warn` |

### Hook execution (from `hooks.py`)

1. Each hook receives the context bag as `JAATO_CTX_<KEY>` environment variables (keys uppercased)
2. The hook's `run:` field is executed via `subprocess.run(shell=True)` with a 300-second timeout
3. Hook stdout is scanned for `JAATO_CTX_SET key=value` lines → buffered in a `ContextBuffer`
4. If the hook succeeds (exit code 0) → continue to next hook
5. If the hook fails → apply `on_failure` policy:
   - `abort` → pipeline stops (state fails / flow aborts)
   - `skip` → state is skipped (pre_validate only)
   - `retry` → retry the entire state (up to `max_retries`)
   - `warn` → log warning and continue

### Context buffer and atomic commit

Hook-produced context values are **not written immediately** to the context bag. They are buffered in a `ContextBuffer` and only committed after the entire state pipeline completes (step 6 in the runner). This prevents partial state execution from polluting the context.

### Real hook examples

```yaml
# Verify inputs exist
- name: verify-inputs
  run: |
    test -f "{{inputs}}/prompt.md" \
      && echo "OK: prompt.md found" \
      || { echo "FATAL: prompt.md missing"; exit 1; }
  on_failure: abort

# Extract context from generated files
- name: extract-port-context
  run: |
    HAS_OUT=$(find "{{workspace}}/src" -path "*/port/out/*.java" | head -1)
    echo "JAATO_CTX_SET has_outbound_ports=$([ -n \"$HAS_OUT\" ] && echo true || echo false)"
  on_failure: abort

# Idempotency guard — skip if already done
- name: idempotency-guard
  run: |
    if [ -f "{{workspace}}/pom.xml" ]; then
      echo "WARN: project already scaffolded — skipping"
      exit 1
    fi
  on_failure: skip
```

---

## 10. Parameters and Variable Expansion

### Parameter definition

```yaml
params:
  inputs:
    required: true
    type: string
    description: "Path to the input folder."
  workspace:
    required: false
    type: string
    default: "."
    description: "Working directory."
  max_fix_iterations:
    required: false
    type: integer
    default: 3
```

### Resolution order (from `loader.py`)

1. If the user provides a value for the param → use it
2. Else if the param has a `default` → use the default
3. Else if the param is `required` → raise `FlowLoadError`
4. Else → the param is not included in the resolved set

### Expansion

All `{{param_name}}` placeholders in **every string field** in the YAML are expanded. This includes:
- `prompt:` text
- Hook `run:` scripts
- Artifact `path:` values
- Group `max_iterations:` values
- Completion `summary_report:` paths
- Description fields

The expansion uses a regex `\{\{(\w+)\}\}` that matches `{{word}}` patterns. Unresolved placeholders (params not in the resolved set) are left as-is.

### Late-binding context expansion

In addition to param expansion at load time, the runner performs **late-binding** expansion of `{{ctx.key}}` placeholders in `profile`, `agent`, and `prompt` fields at state entry time. This allows a state's prompt to reference context values produced by earlier states.

---
## 11. Context Bag — Flow-Scoped Key-Value Store

The **context bag** (`ContextBag` in `context.py`) is a flow-scoped key-value store that accumulates structured metadata across states. It enables downstream states to make decisions based on what earlier states produced.

### How values enter the context bag

1. **Resolved parameters** — seeded at flow start with producer `"__params__"`
2. **Hook `JAATO_CTX_SET` lines** — parsed from hook stdout
3. **Agent-produced context** — returned by the subagent's `exit_code`/`context` dict
4. **Group iteration counters** — e.g., `build-validate-cycle.iteration`

### Ownership enforcement

Each key tracks which state produced it. Rules:
- A state **cannot** overwrite another state's keys → raises `ContextConflictError`
- A state **can** overwrite its own keys (important for group iterations)
- Special producers (`__params__`, `__pre_flow__`, `__post_flow__`) are treated as system-level

### Environment variable access for hooks

When a hook executes, the entire context bag is available as `JAATO_CTX_<KEY>` env vars:

```bash
# Inside a hook script:
if [ "$JAATO_CTX_HAS_OUTBOUND_PORTS" = "true" ]; then
    echo "JAATO_CTX_SET needs_resilience=true"
fi
```

Value conversion rules:
- `bool` → `"true"` / `"false"`
- `list` / `dict` → JSON string
- Other → `str(value)`

### Persistence

The context bag is saved to `<workspace>/.jaato/.context-bag.json` after flow completion. It can be loaded additively with `ContextBag.load(workspace)`.

### Value coercion

When values enter via `JAATO_CTX_SET`, they are coerced:
- `"true"` / `"false"` → Python `bool`
- Integer strings → `int`
- Float strings → `float`
- JSON arrays/objects → parsed
- Everything else → raw string

---

## 12. Dependency DAG — Compile-Time Verification

The **DAG** (Directed Acyclic Graph) is built from `from:` references in artifact and context bindings. It is validated at two points:

### Load-time validation (`dag.py`)

The `build_and_validate(flow)` function:
1. Collects all states (flattening groups)
2. Extracts dependency edges from `from:` references
3. Verifies the declared state order is a **valid topological sort** — every dependency must appear earlier
4. Verifies **context key ownership** — no two states declare the same context output key

```python
# Example error:
# State 'add-system-api' depends on ['generate-hexagonal-base']
# which is declared after it.
raise DAGError("...")
```

### Authoring-time validation (`authoring.py`)

The `FlowDraft._validate_dag_order()` method performs the same check incrementally as each step is added. Additionally:
- External dependencies of group-internal states must be declared before the group
- `from:` references must point to known states
- `from:` artifact references should point to declared output artifacts (warning if not)

### What the DAG enforces

The DAG ensures **compile-time correctness**: if state B depends on state A's output, A must be declared before B. This prevents circular dependencies and ensures data availability. However, the DAG does **not** validate that files actually exist at runtime — that's the job of post_validate hooks.

---

## 13. Completion Criteria

Completion criteria determine whether a flow execution is considered successful.

```yaml
completion:
  require_all_states: true        # or false
  required_outputs:
    - from: generate-hexagonal-base.pom
    - from: validate-tier1.tier1-report
  summary_report: "{{workspace}}/.jaato/reports/orchestrator-report.md"
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| **require_all_states** | boolean | `true` | If `true`, any failed state causes flow failure. If `false`, only `required_outputs` are checked. |
| **required_outputs** | list | `[]` | List of `"state.artifact"` references. The producing state must have succeeded. |
| **summary_report** | string | `null` | Path where the orchestrator should write a summary report. |

### Completion checking (from `runner.py`)

1. If `require_all_states` is `true` → check no state has `FAILED` status
2. For each `required_outputs` entry:
   - Parse `"state.artifact"` → extract state name
   - Verify that state did not fail
3. Return `(success, error_message)` tuple

> **Note:** The runner currently checks that the *producing state succeeded*, not that the artifact file exists on disk. File existence validation is the responsibility of post_validate hooks.

### Scripted vs. Agentic differences

- **Scripted**: `require_all_states: true` is typical — every state must succeed
- **Agentic**: `require_all_states: false` — not all states need to run (orchestrator selects)

---

## 14. Rules — Agentic Flow Constraints

Rules are **author-defined constraints** that the orchestrator agent must follow. They exist only in agentic flows.

### Rule schema

```yaml
rules:
  - id: orchestrator-never-implements
    severity: must           # must | should | prefer
    scope: global            # Currently always "global"
    description: >
      The orchestrator MUST NOT write, modify, or generate any code.
    rationale: >
      Separation of concerns: the orchestrator coordinates,
      workers implement.
```

### Severity levels and enforcement

| Severity | Meaning | Enforcement |
|----------|---------|-------------|
| **must** | Hard constraint | Violation is a critical failure |
| **should** | Strong recommendation | Violation requires justification |
| **prefer** | Soft preference | Violation is acceptable if justified |

The `risk_tolerance` parameter controls how strictly `should` rules are enforced:
- `"strict"` → `should` treated as `must`
- `"normal"` → standard enforcement
- `"relaxed"` → `should` treated as `prefer`

### Real rule examples from `agentic-feature-implementation.yaml`

| ID | Severity | Purpose |
|----|----------|---------|
| `orchestrator-never-implements` | must | Orchestrator must not write code — only delegates |
| `analysis-before-execution` | must | Analyze before implementing |
| `validation-after-implementation` | must | All validators after all implementers |
| `all-validators-required` | must | Every validation tier must run |
| `max-inline-workers` | must | Max 3 unprofiled subagents per execution |
| `context-bag-integrity` | must | Only states and hooks set context values |
| `persistence-before-api` | should | Persistence before public API |
| `prefer-declared-profiles` | prefer | Use profiled workers over inline |
| `prefer-sequential-resilience` | prefer | Apply resilience patterns one at a time |
| `prefer-minimal-states` | prefer | Only activate needed states |

> **DAG-derived rules are implicit:** The dependency DAG auto-generates rules from `from:` references. For example, if `implement-system-api` has `from: scaffold-project.port-out`, the orchestrator must execute `scaffold-project` first. Authored rules add constraints *beyond* the DAG.

---

## 15. Authoring Tools — begin/build/update/commit

Flows are authored incrementally using four tools, backed by the `FlowDraft` class in `authoring.py`.

### 1. `begin_flow` — Create a draft

```
begin_flow(
    name="my-flow",               # Required: kebab-case
    execution_model="scripted",   # "scripted" or "agentic"
    description="...",
    tags=["generation", "java"],
    params={                      # Optional param definitions
        "inputs": {"required": true, "type": "string", ...},
        "workspace": {"required": false, "default": ".", ...}
    }
)
→ Returns: {draft_id: "abc12345", flow_name: "my-flow"}
```

### 2. `build_step` — Add a state or group

```
# Add a state
build_step(
    draft_id="abc12345",
    step={
        "name": "generate-base",
        "profile": "skill-mod-code-015-hexagonal-base",
        "prompt": "Generate the base project...",
        "when": "ctx.needs_base == true",
        "inputs": {
            "artifacts": [{"path": "{{inputs}}/prompt.md"}],
            "context": [{"key": "base_package", "from": "analyze-reqs", "required": true}]
        },
        "outputs": {
            "artifacts": [{"name": "pom", "path": "{{workspace}}/pom.xml", "format": "xml"}],
            "context": [{"key": "has_ports", "type": "boolean"}]
        },
        "hooks": {
            "post_validate": [{"name": "check-pom", "run": "test -f pom.xml", "on_failure": "retry"}]
        },
        "max_retries": 1
    },
    position={"after": "analyze-reqs"}  # Optional
)

# Add a group
build_step(
    draft_id="abc12345",
    step={
        "group": "build-validate",
        "max_iterations": 3,
        "until": "ctx.all_passed == true",
        "states": [
            {"name": "build", "profile": "...", ...},
            {"name": "validate", "profile": "...", ...}
        ]
    }
)
```

**Validation performed by `build_step`:**
- State name format: `^[a-z][a-z0-9-]*$`
- Duplicate name check (across all states and groups)
- `from:` references resolve to existing states/artifacts
- Context key ownership (no two states write same key)
- Hook `on_failure` modes valid for the hook point
- DAG order (dependencies declared before this step)
- Groups cannot be nested
- Group must have at least one state

### 3. `update_step` — Modify an existing state

```
update_step(
    draft_id="abc12345",
    state_name="generate-base",
    changes={
        "profile": "new-profile",              # Update profile
        "when": "ctx.always_true == true",     # Update condition
        "max_retries": 2,                       # Update retries
        "add_hooks": {                          # Add hooks
            "post_validate": [{"name": "new-check", "run": "...", "on_failure": "warn"}]
        },
        "remove_hooks": {                       # Remove hooks
            "pre_validate": ["old-check"]
        },
        "set_inputs": {                         # Replace inputs
            "artifacts": [...],
            "context": [...]
        },
        "set_outputs": {                        # Replace outputs
            "artifacts": [...],
            "context": [...]
        }
    }
)
```

### 4. `commit_flow` — Write to YAML

```
commit_flow(
    draft_id="abc12345",
    path="jaato_premium/flows/my-flow.yaml",
    completion={
        "require_all_states": true,
        "required_outputs": [{"from": "generate-base.pom"}],
        "summary_report": "{{workspace}}/.jaato/reports/report.md"
    },
    rules=[                                # Agentic only
        {"id": "rule-1", "severity": "must", "description": "..."}
    ],
    hooks={                                 # Global hooks
        "pre_flow": [...],
        "post_flow": [...]
    }
)
```

**Validation performed by `commit_flow`:**
- At least one state exists
- DAG topological order valid
- Completion criteria references resolve to known artifacts
- Agentic flows should have at least one rule (warning)

---

## 16. Execution Engine — ScriptedFlowRunner

The `ScriptedFlowRunner` (in `runner.py`) is the deterministic execution engine for scripted flows.

### Usage

```python
from jaato_premium.flow_runner import ScriptedFlowRunner

def my_spawner(profile, prompt, context, state_name, agent=""):
    # Bridge to the actual subagent infrastructure
    result = spawn_subagent(profile=profile, task=prompt, ...)
    return {"exit_code": 0, "context": {...}, "output": "..."}

runner = ScriptedFlowRunner(agent_spawner=my_spawner)
result = runner.run("path/to/flow.yaml", params={"inputs": "/data"})

assert result.status == FlowStatus.SUCCESS
for sr in result.state_results:
    print(f"  {sr.name}: {sr.status.value}")
```

### State entry pipeline (6 steps)

For each state, the runner executes:

1. **Resolve inputs** — check required context keys exist (error if missing)
2. **Evaluate `when:`** — skip if false
3. **Run pre_validate hooks** — abort/skip/retry on failure
4. **Invoke agent** — call the spawner callback with profile, prompt, context
5. **Run post_validate hooks** — abort/retry on failure
6. **Commit context** — drain the buffer into the context bag

### Progress events

The runner emits events via an optional `on_event` callback:

| Event | Data |
|-------|------|
| `flow.started` | `{flow_name, execution_model, steps, total_steps}` |
| `state.started` | `{state, retry, profile, agent}` |
| `state.completed` | `{state, status, retries, error}` |
| `group.iteration_started` | `{group, iteration, max_iterations}` |
| `group.completed` | `{group, status, iteration, until_satisfied}` |
| `flow.completed` | `{flow_name, status, error}` |

### AgentSpawner protocol

The runner calls `self._spawner(profile, prompt, context, state_name, agent)` for each state. The return value must be a dict with:
- `exit_code` (int): 0 for success
- `context` (dict): Context values produced by the agent
- `output` (str): Optional text output

If no spawner is provided, a no-op spawner is used (always succeeds, empty context).

---

## 17. Configuration / Schema / API Reference

### Flow data model (from `models.py`)

| Class | Fields | Description |
|-------|--------|-------------|
| `Flow` | name, description, version, execution_model, tags, params, resolved_params, pre_flow_hooks, post_flow_hooks, steps, rules, completion | Top-level flow definition |
| `State` | name, description, profile, agent, prompt, when, inputs, outputs, pre_validate_hooks, post_validate_hooks, max_retries, retry_delay_seconds, transition | Single execution step |
| `StateGroup` | name, description, max_iterations, until, on_max_iterations, states | Iterative group of states |
| `Hook` | name, run, description, on_failure | Shell script at a lifecycle boundary |
| `ArtifactInput` | from_ref, path, alias | Input file reference |
| `ContextInput` | key, from_state, required, default | Input context reference |
| `ArtifactOutput` | name, path, format, description, schema | Declared output file |
| `ContextOutput` | key, type, description | Declared output context key |
| `ParamDef` | name, required, default, type, description | Parameter definition |
| `Rule` | id, severity, scope, description, rationale, check | Agentic constraint |
| `CompletionCriteria` | require_all_states, required_outputs, summary_report | Success criteria |
| `Transition` | next, on_failure, condition | Post-state routing (partially implemented) |

### Enums

| Enum | Values |
|------|--------|
| `ExecutionModel` | `scripted`, `agentic` |
| `HookFailureMode` | `abort`, `warn`, `skip`, `retry` |
| `MaxIterationsPolicy` | `abort`, `warn` |
| `RuleSeverity` | `must`, `should`, `prefer` |

### Tool schemas (8 tools in `orchestration` category)

| Tool | Purpose | Auto-approved |
|------|---------|-------------|
| `list_flows` | Discover available flow YAML files | Yes |
| `describe_flow` | Inspect flow structure | Yes |
| `start_flow` | Execute a scripted flow | No |
| `get_flow_status` | Query running/completed flow status | Yes |
| `begin_flow` | Create a new flow draft | Yes |
| `build_step` | Add state/group to draft | Yes |
| `update_step` | Modify existing state in draft | Yes |
| `commit_flow` | Write draft to YAML | Yes |

### Plugin registration

The `FlowToolsPlugin` class registers as a `ToolPlugin` via the `jaato.premium` entry point. It:
- Registers the `orchestration` tool category
- Auto-discovers flow YAML directories from `get_flows_path()`
- Manages draft state in `_drafts` dict (in-memory, keyed by draft_id)
- Manages active flow executions in `_active_flows` dict
- Runs flows in background threads via `_run_flow_thread`

---

## 18. Runtime Internals — What the Source Code Reveals

### Parameter expansion is recursive and complete

The `_expand_all()` function in `loader.py` recursively walks every string in the parsed YAML dict, expanding `{{param}}` patterns. This means params are expanded in:
- Nested structures (e.g., artifact schema definitions)
- Hook scripts (multi-line strings)
- Description fields
- Group `max_iterations` (which is why it can be a param reference)

Unresolved placeholders are silently left as-is (not an error).

### Context bag values are seeded from params

At flow start, the runner seeds the context bag with all resolved params:

```python
for name, value in flow.resolved_params.items():
    ctx.set(name, value, producer="__params__")
```

This means `when:` expressions can reference param values via `ctx.inputs`, `ctx.workspace`, etc.

### Post-flow hooks always run

Even if the flow is aborted (pre_flow hook aborts, or a state fails), the runner always executes post_flow hooks before returning. This ensures cleanup and reporting always happen.

### Transition routing is partially implemented

The `State.transition` field (`next`, `on_failure`, `condition`) is parsed and stored but **not fully implemented** in the runner. Currently, if a state fails:
1. The runner logs the `on_failure` target
2. The flow aborts
3. The transition jump is not executed

This means the current execution model is strictly linear — states execute in declaration order, and any failure aborts the flow.

### Groups set iteration context

Each group iteration sets `ctx.<group-name>.iteration` to the iteration number. This is a dotted key that can be referenced in `when:` expressions using nested resolution (`_get_nested` in `conditions.py`).

### Agent invocation resolves `{{ctx.*}}` late

The `_resolve_ctx_placeholders()` method in `runner.py` expands `{{ctx.key}}` patterns in `profile`, `agent`, and `prompt` fields at state entry time. This is a separate expansion pass from the param expansion — it happens after the context bag has been populated by prior states.

### Hooks run with full context but write to a buffer

The design separates hook execution from context mutation:
1. Hooks read context via `JAATO_CTX_*` env vars (snapshot)
2. Hooks write via `JAATO_CTX_SET` stdout lines (buffered)
3. Only after the full state pipeline (pre → agent → post) does the runner commit buffered values

This atomicity prevents a failing hook from partially corrupting the context.

### The WebSocket extension bridges flows to the web UI

The `FlowRunnerExtension` in `extension.py` registers WebSocket message handlers (`flow.list`, `flow.describe`, `flow.start`, `flow.status`) so the web component `<jaato-flow>` can start and monitor flows. It forwards runner progress events to the WS client in real-time.

### Authoring validation is incremental

The `FlowDraft` class maintains tracking indices (`_state_names`, `_group_names`, `_artifact_registry`, `_context_owners`) for O(1) duplicate detection. Each `build_step` call validates against the current draft state, catching errors as early as possible.

---

## 19. Source Code Map

| File | Path | Purpose |
|------|------|---------|
| **models.py** | `jaato_premium/flow_runner/models.py` | All dataclasses: Flow, State, StateGroup, Hook, Rule, etc. No runtime behavior. |
| **loader.py** | `jaato_premium/flow_runner/loader.py` | YAML parsing, param resolution, `{{param}}` expansion. Entry points: `load_flow()`, `load_flow_from_dict()`. |
| **dag.py** | `jaato_premium/flow_runner/dag.py` | Dependency graph construction and validation. Entry point: `build_and_validate()`. |
| **conditions.py** | `jaato_premium/flow_runner/conditions.py` | `when:` expression evaluator. Entry point: `evaluate_when()`. |
| **context.py** | `jaato_premium/flow_runner/context.py` | `ContextBag` (key-value store with ownership), `ContextBuffer` (atomic commit), `parse_ctx_set_lines()`. |
| **hooks.py** | `jaato_premium/flow_runner/hooks.py` | Hook executor. Entry point: `run_hook_list()`. Returns `HookAction` (continue/abort/skip/retry). |
| **runner.py** | `jaato_premium/flow_runner/runner.py` | `ScriptedFlowRunner` — the main execution engine. Entry point: `runner.run(path, params)`. |
| **authoring.py** | `jaato_premium/flow_runner/authoring.py` | `FlowDraft` — incremental flow authoring with validation. Methods: `build_step()`, `update_step()`, `commit()`. |
| **tools.py** | `jaato_premium/flow_runner/tools.py` | `FlowToolsPlugin` — 8 tool schemas and executors (list/describe/start/status/begin/build/update/commit). |
| **extension.py** | `jaato_premium/flow_runner/extension.py` | `FlowRunnerExtension` — WebSocket handlers for web UI flow management. |
| **\_\_init\_\_.py** | `jaato_premium/flow_runner/__init__.py` | Package init, exports `ScriptedFlowRunner`. |

### Flow YAML files (examples)

| File | Path | Description |
|------|------|-------------|
| **scripted-generate-microservice.yaml** | `jaato_premium/flows/scripted-generate-microservice.yaml` | Full scripted flow: scaffold → build → validate with group iteration. 498 lines. |
| **agentic-feature-implementation.yaml** | `jaato_premium/flows/agentic-feature-implementation.yaml` | Full agentic flow: analyze → selective implementation → validation with rules. 690 lines. |
