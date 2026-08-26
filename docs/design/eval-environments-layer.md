# Eval Environments Layer — extracting a benchmark spine from what already exists

## Status

Design sketch. No implementation. Written 2026-08-26 after surveying
`jaato`, `jaato-cascade-based-prototype`, `jaato-cascade-coordination-example`
and `perpetual-monologue-cascade`.

## Origin

A LangChain writeup ("How we Build Agent Environments & Tasks", 2026-08-25)
describes a two-step pipeline for mass-producing agent eval tasks:
*spec generation* (traces + code → markdown task spec, human-reviewed) and
*Spec2Task* (spec → runnable task in their "Harbor" format), with a reusable
**world spec** holding everything not specific to one task. The question this
document answers is what that layer looks like on jaato.

## The finding

**jaato does not need a Harbor built. It needs one extracted.**

Three repositories in this organisation have each independently built most of
an environment/task/grader stack, locally and for their own purpose. They are
three copies of one latent framework:

| Repo | What it already is |
|---|---|
| `jaato-cascade-coordination-example/certify/` | a **rubric engine**: nine claims, three-valued verdicts, evidence attached per verdict, guards that are themselves reverted and watched failing (R1–R14) |
| `jaato-cascade-based-prototype` | a **full environment + grader + model-sweep matrix**: fixtures, six interchangeable profile sets, programmatic graders at the completion boundary, an LLM-judge stage, a curated-memory learning loop, a determinism hash |
| `perpetual-monologue-cascade` | **run analysis** (`analyze_run.py`, `observe.py`) over a long-running cascade |

The `certify/` README states the governing observation itself, about a
different subject:

> Any second copy of a fact rots unless something executes the comparison —
> and the copy that rots is the one that cannot fail.

Three private eval harnesses are that defect applied to the eval layer. The
work below is deduplication, not greenfield.

---

## Concept map

Article vocabulary → the jaato primitive that already implements it.

| Harbor concept | jaato primitive | Where | Status |
|---|---|---|---|
| Environment (filesystem) | `workspace_path` on the client + fixture tree | `jaato-sdk/jaato_sdk/client/ipc.py`; `jaato-cascade-based-prototype/fixtures/` | exists |
| Environment (read-only task definition) | `config_root` — `.jaato/` resolved separately from the workspace | `open_session(config_root=…)` | exists |
| Environment (isolation) | `apparmor=True`, `runtime_limits` (cgroup v2: memory/pids/cpu) | `shared/runtime_limits.py` | exists |
| Environment (mocked services) | `.mcp.json`, `.jaato/services/` | prototype `.jaato/services/maven_central/` | exists |
| Harness under test | profile: `plugins`, `tool_scopes`, `model`, `provider`, `plugin_configs`, `suppress_base_instructions` | `shared/plugins/subagent/config.py` | exists |
| Harness *variants* | profile sets selected by one env var | `JAATO_PROFILE_SET` → `.jaato/profiles/<set>/` | exists |
| Task input | prompt + `agent_params` (persona `{{param}}` substitution) | `open_session(agent_params=…)` | exists |
| Output contract | `completion_payload_schema` → typed `signal_completion` payload | profile field; `Session.complete()` returns it | exists |
| Grader — programmatic | completion processors: `validate(payload, context) -> list[str]`, with `context.tool_calls` ledger + `context.workspace_path` | prototype `.jaato/scripts/processors/` | exists |
| Grader — LLM judge | a session whose profile declares a rubric completion schema | prototype `build_judge` stage | exists |
| Grader — behavioural claim | `Verdict(claim_id, state, evidence, revert)` | `certify/verdict.py` | exists |
| Determinism / repeatability | canonical-JSON sha256 of the payload | `orchestrator/sdk_harness.py::hash_payload` | exists |
| Cost / token metrics | `UsageBreakdown.cost_usd`, cache + reasoning token splits, on `TurnCompletedEvent` | `jaato-sdk/jaato_sdk/events.py` | exists |
| Latency / trajectory | OTel spans `jaato.turn → jaato.tool → jaato.permission` | `docs/opentelemetry-design.md` | exists |
| Run grouping | `cascade_driver_id` + cascade-as-client subscription | `docs/design/cascade-as-client.md` | design locked |
| Per-run safety ceiling | `budget_control.limits` (usd / tokens / seconds / tool_calls / turns) | `shared/budget_control.py` | exists |
| World spec | agent persona + curated memory scope (raw→curated curator) | `docs/design/agent-continuity.md` | pattern documented |
| Post-training consumer | — | `kb-stage-agent-LoRA-training` (separate repo) | exists |
| **Task manifest** | — | — | **missing** |
| **Fixture materialisation** | — | — | **missing** |
| **Sweep driver** | — | — | **missing** |
| **Result store / report** | partial (`processors/_report.py` aggregates rejections only) | — | **missing** |

Fourteen of eighteen rows exist. The missing four are the spine.

---

## Why the pieces line up as well as they do

Three framework decisions, made for unrelated reasons, happen to be exactly
what a hermetic task runner needs.

**1. `config_root` is separate from `workspace_path`.** A task can ship its own
read-only `.jaato/` (profiles, agents, completion schemas, permissions) while
the workspace is a scratch copy the agent mutates and the grader inspects. This
is precisely Harbor's environment/task-definition split, and it is already
plumbed through both `IPCClient` and `IPCRecoveryClient` (since 2026-08-23).

**2. The harness is data, not code.** "What if I remove all the harness tools
and just use bash?" is `plugins: [cli]` against `plugins: [cli, file_edit, todo,
filesystem_query, …]` — two YAML files, profile inheritance sharing everything
else. The article treats harness tuning as an aspiration; here it is a diff.

**3. The completion boundary is already a grading boundary.** A completion
processor receives the schema-validated payload *plus* the full tool-call
ledger *plus* the workspace path, and returns a list of error strings. That is
a grader signature. `codegen_files_exist.py` uses it to catch an agent claiming
twenty rendered files when six of the underlying calls errored — a fabrication
check against filesystem ground truth. The only thing separating it from an
eval grader is that its return value feeds a retry instead of a scoreboard.

---

## The load-bearing idea to lift: three-valued verdicts

`certify/verdict.py` refuses to collapse "not exercised" into "passed":

```
PASS     exercised and held
FAIL     exercised and violated          → exit 1
BLOCKED  surface absent, nothing ran     → exit 2, NOT a pass
```

For a benchmark this matters more, not less. An eval run has many ways to
produce no signal: the fixture failed to materialise, the daemon was stale, the
provider returned 429, the budget ceiling tripped, the model hit `max_tokens`
mid-payload. Every one of those scored as `FAIL` corrupts the comparison you
are running — you conclude the cheap model is worse when in fact its provider
rate-limited you.

`TurnCompletedEvent.finish_reason` already exists to make this branchable
without inferring it from empty output. The eval layer must carry it through to
a distinct `BLOCKED` bucket and refuse to average over it.

**This is the single highest-value thing to lift, and it is twenty lines.**

---

## The missing spine

Five modules. Nothing else.

### 1. Task manifest — `task.yaml`

```yaml
id: customer-api/add-pagination
description: Add cursor pagination to an existing Spring REST controller.

environment:
  fixture: fixtures/customer-api        # copied per run
  config_root: .jaato                   # read-only task definition
  apparmor: true
  runtime_limits: {memory_max_mb: 2048, pids_max: 256, tool_timeout_seconds: 120}

input:
  agent: codegen
  agent_params: {capability: pagination, stack: java-spring}
  prompt: |
    Add cursor-based pagination to the customer listing endpoint.

harness:
  profile_set: ${JAATO_PROFILE_SET}     # the sweep axis
  profile: codegen

budget:
  usd: 0.50
  turns: 25

graders:
  - kind: processor                     # existing completion-processor contract
    script: .jaato/scripts/processors/codegen_files_exist.py
  - kind: script                        # runs against the mutated workspace
    run: mvn -q clean compile
  - kind: judge                         # a jaato session with a rubric schema
    profile: build_judge
    weight: 0.4
```

Three grader kinds, all of which already exist in some form. The manifest does
not invent a grading language — it names existing artefacts.

### 2. Fixture materialisation

Copy `fixture` → a fresh temp workspace per run; point `workspace_path` at the
copy and `config_root` at the (unmodified) task definition. Runs become
hermetic and repeatable, and N repeats of the same task stop contaminating each
other. The prototype's cascade boot path already does a version of this ("wipe
workspace, preserve memories, copy apparmor fragments"), and
`orchestrator/cascade_entry.py::preflight` guards the companion concern —
that the derived artefacts a run reads are in sync with their source before
the run starts. Generalise both.

### 3. Grader adapters

One `Verdict` out, three kinds in:

- **processor** — import the module, call `validate(payload, context)`; empty
  list → PASS.
- **script** — run in the mutated workspace; exit 0 → PASS. Non-zero from the
  *harness* (fixture missing, tool absent) → BLOCKED, not FAIL.
- **judge** — `open_session(profile=…)` with a rubric `completion_payload_schema`
  of `{score, criteria_met[], reasoning}`; `Session.complete()` returns it typed.

Reuse `certify/verdict.py` verbatim. Do not write a second one.

### 4. Sweep driver

The cartesian product of *(task × profile_set × repeat)*, run concurrently
against the pre-warm runner pool, each arm stamped with one
`cascade_driver_id` so `cascade_events(cid)` observes the whole matrix.
`JAATO_RUNNER_POOL_SIZE` must be raised to the fan-out width — sequential
stages reuse a slot, but a parallel sweep needs one warm slot per simultaneous
arm.

This is the piece that turns the article's three questions into commands:

| Question | The sweep |
|---|---|
| Cheaper model for a subset? | vary `profile_set`, hold task fixed, compare pass-rate against `usage.cost_usd` |
| Simplify the prompt? | vary `agent` (persona `.md`), hold profile fixed |
| Drop the harness, just bash? | vary `plugins` / `tool_scopes` in a profile variant |

### 5. Result store + report

JSONL, one row per arm: task id, profile set, repeat index, verdict state,
grader breakdown, `UsageBreakdown`, wall-clock, turn count, `finish_reason`,
payload hash. Then a pivot: pass-rate and cost per (task × profile set), with
BLOCKED counted separately and never averaged in.

`hash_payload` gives a determinism column for free — same task, same profile,
N repeats, how many distinct payload hashes? That is a flakiness metric the
article does not mention and this codebase already computes.

---

## Where it lives

A sibling package `jaato-eval/`, depending on **`jaato-sdk` only** — never on
`jaato-server/shared`. `certify/` already enforces this discipline with a
facade guard (R1: "plant a `from shared…` import → the facade guard must
catch it"); adopt the same guard here.

The constraint is not stylistic. If the eval harness can be built against the
SDK alone, that proves the SDK is sufficient for third-party drivers. If it
cannot, the gap it hits is a real SDK defect and should be fixed there rather
than worked around with a private import.

---

## The two-step pipeline on jaato

The article's spec/Spec2Task split maps onto primitives that exist:

**Spec generation.** An agent with `filesystem_query` + `memory`, reading
`.jaato/logs/` session logs and the OTel span stream, emitting task specs as
markdown for human review. The fine-tuner agent (`docs/design/finetuner-closed-loop.md`)
already does the analytical half of this — it consumes another session's
telemetry, forks its conversation read-only to interrogate decisions, and links
findings back to the profile and prompt that produced them. It stops at "here
is what went wrong"; emitting a task spec that *reproduces* what went wrong is
the natural next output.

**World spec.** `docs/design/agent-continuity.md` is the mechanism, already
documented as a pattern requiring no new framework code: a persona with a
`{{continuity_scope}}` placeholder, memory-plugin enrichment surfacing prior
runs' lessons, and — non-optional — a curator draining raw memories into
curated ones. The prototype ships this as `memory_curator`, closing its loop
when a `mem_id` is cited in a passing build's judgment. That is exactly the
article's "world spec accumulates across the first two or three tasks", with a
correctness caveat the article omits: **without the curator, nothing surfaces.**

**Spec2Task.** An agent whose `completion_payload_schema` emits a `task.yaml`
plus a fixture-generation script. The article's warning that "agents are bad at
knowing what method to use for generating different types of data" is handled
the same way the prototype handles it — prescribe the method in the persona
(LLM+rubric for free text, sqlite with declared schemas for tabular) rather
than leaving it to the model.

---

## Difficulty calibration

The article calibrates by running each task across model tiers. `JAATO_PROFILE_SET`
already spans six backends — OpenRouter Haiku, two vLLM builds, two TensorRT-LLM
builds, ZhipuAI GLM — with identical profile *names* in each set, so the swap is
one env var. A well-calibrated task separates them; a task that every set passes
or every set fails carries no information.

`shared/model_tiers.py` offers a second, finer axis (planner / dispatcher /
executor within one run), which measures something different: not "is this task
too easy" but "which cognitive step actually needs the expensive model".

---

## Deliberately not doing

- **No new grader DSL.** Three existing kinds, named by path.
- **No replacement for `certify/`.** It is the reference implementation of the
  verdict semantics; the eval layer imports it.
- **No core framework code.** SDK-only, as a proof of SDK sufficiency.
- **No new event types.** `TurnCompletedEvent.usage`, `AgentCompletedEvent.payload`
  and `finish_reason` carry everything the result store needs.
- **No averaging over BLOCKED.** Structural, not a policy knob.

---

## Open questions

1. **Does the fixture belong in git or is it generated?** The prototype ships
   `fixtures/customer-api` checked in. Generated fixtures scale better but
   introduce a generator whose own determinism must then be graded.
2. **Judge-grader cost.** An LLM judge on every arm of a six-backend × N-repeat
   sweep may cost more than the arms being measured. Gate judges behind
   programmatic graders passing first?
3. **Where does the result store live** so that the fine-tuner can read it and
   close the loop from `docs/design/finetuner-closed-loop.md` step 3 (fork
   replay against a patched profile)?
4. **Do the three repos converge on the extracted package, or does it ship
   alongside them?** Convergence is the point, but `certify/` deliberately
   predates the API it certifies — that inversion must survive extraction.

## Related

- [Cascade-as-Client](cascade-as-client.md) — run grouping and lifecycle
- [Fine-Tuner Closed Loop](finetuner-closed-loop.md) — the consumer of eval results
- [Agent Continuity](agent-continuity.md) — the world-spec mechanism
- [Payload-Schema Conventions](payload-schema-conventions.md) — authoring the output boundary
- [Budget Control & Degradation](budget-control-degradation.md) — per-run ceilings
- [OpenTelemetry Design](../opentelemetry-design.md) — trajectory + cost spans
