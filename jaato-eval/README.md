# jaato-eval

A benchmark spine over jaato's session and profile primitives.

The framework already carries most of what an agent benchmark needs. This
package is the part that was missing: **a task manifest, hermetic fixture
materialisation, three grader adapters over one verdict type, a sweep
driver, and a result store.**

Design rationale, and the survey that produced it, live in
[`docs/design/eval-environments-layer.md`](../docs/design/eval-environments-layer.md).

---

## What it turns into a command

| Question | The sweep |
|---|---|
| Can I use a cheaper model for a subset of tasks? What does it save? | vary `--profile-set`, hold the task fixed, read pass rate against `cost_usd` |
| Can I simplify the prompt? | vary the persona (`.jaato/agents/<name>.md`), hold the profile fixed |
| What if I drop the harness tools and just use bash? | a profile variant with `plugins: [cli]` |

All three are configuration edits, because the harness is data. The only
axis this driver iterates itself is the profile set — the others change
what an arm *is*, not which cell of the matrix it occupies.

## Install

```bash
pip install -e jaato-eval/                 # offline half only
pip install -e "jaato-eval/[run]"          # + jaato-sdk, to actually run arms
```

`jaato_sdk` is imported lazily, so the manifest parser, fixture
materialiser, ledger, script/processor graders, result store and report
all work — and are tested — without a daemon.

## Use

```bash
# run one task directory across two profile sets
python -m jaato_eval run tasks/ --profile-set openrouter_haiku,vllm_qwen3_14b \
    --out results.jsonl --concurrency 4

# re-pivot an existing results file
python -m jaato_eval report results.jsonl

# resume a sweep that was killed
python -m jaato_eval run tasks/ --out results.jsonl --resume
```

Exit codes carry the verdict semantics:

| code | meaning |
|---|---|
| 0 | every arm passed |
| 1 | at least one arm **failed** — a real defect in what was tested |
| 2 | at least one arm was **blocked**, or nothing ran |

A CI job that reads 2 as success is the vacuous pass this engine exists
to refuse. The code is distinct so that mistake has to be deliberate.

## A task

```yaml
id: customer-api/add-pagination
description: Add cursor pagination to an existing Spring REST controller.

environment:
  fixture: fixture            # copied fresh per arm; the agent mutates the copy
  config_root: config         # read-only: profiles, agents, schemas
                              # do NOT name this `.jaato` — the jaato repo's
                              # root .gitignore excludes that path, and a
                              # task's config root is committed data
  apparmor: true
  runtime_limits: {memory_max_mb: 2048, tool_timeout_seconds: 120}

input:
  agent: codegen
  agent_params: {capability: pagination, stack: java-spring}
  prompt: Add cursor-based pagination to the customer listing endpoint.

harness:
  profile: codegen
  profile_set: openrouter_haiku   # --profile-set overrides this per arm

budget: {usd: 0.50, turns: 25}

graders:
  - kind: script
    run: mvn -q clean compile
  - kind: processor
    script: scripts/processors/codegen_files_exist.py
  - kind: judge
    profile: build_judge
    threshold: 0.7
    gate_on: ["script:mvn -q clean compile"]   # skip the judge on an arm already known bad

repeats: 3
```

Nothing here invents a grading language. Every `kind` names something the
framework or the surrounding repos already execute.

## The three graders

| kind | is | PASS when |
|---|---|---|
| `script` | a command in the mutated workspace | exit 0 |
| `processor` | a framework completion processor, run post-hoc | `validate()` returns `[]` |
| `judge` | a jaato session with a rubric `completion_payload_schema` | `score >= threshold` |

The `judge` rubric is a completion schema, so the score comes back typed —
the provider enforces the shape at sampling time. There is no free-text
score to parse, and changing what "good" means is a schema edit.

## Why three verdict states

`PASS` / `FAIL` / `BLOCKED`, lifted from
`jaato-cascade-coordination-example/certify/verdict.py`.

An eval run has many ways to produce no signal: the fixture failed to
materialise, the daemon was stale, the provider returned 429, a budget
ceiling tripped, the model hit `max_tokens` mid-payload. Scoring any of
those as `FAIL` corrupts the comparison being run — you conclude the cheap
model is worse when its provider merely rate-limited you.

So `BLOCKED` is **never in a denominator** (pass rate is `PASS / (PASS +
FAIL)`) and **always visible** (its own column, plus a reasons digest). A
cell showing 100% over two arms with eight blocked is not a result; it is
a broken runner, and a report that hid the eight would read as success.

A cell that exercised nothing prints `—`, not `0%`. Zero would say "it
always failed"; the truth is "we never found out".

## The tool-call ledger

Completion processors that cross-reference `context.tool_calls` run here
exactly as they do in-band, because the ledger comes from the SDK:

```python
from jaato_sdk.completion_processors import build_ledger
```

That function is the single pairing rule (jaato #640); the server-side
`build_tool_call_ledger` is now a thin alias of it, so there is one
implementation rather than one per consumer. This package holds no copy —
`jaato_eval/ledger.py` is a thin wrapper, not a reimplementation.

**Pairing is by identifier, never by name and order.** A tool that errors
and is retried produces two calls and two responses sharing a name, and
positional pairing credits the retry's success to the call that failed —
reporting a fabricated artefact as verified. `tests/test_ledger.py`
exercises exactly that case against the real SDK builder loaded from the
checkout, not a stub.

**One guard remains**, and it is about deployment rather than data: a
daemon predating jaato #639 emits `function_call` Parts with no
identifier, so nothing can be paired. `history_carries_call_ids()`
witnesses the key rather than inferring it from an empty pairing, and the
processor grader returns BLOCKED on such a history instead of grading.
Upgrade the daemon rather than trusting the result.

## Concurrency and the runner pool

Each simultaneous arm needs its own pre-warm runner slot. Sequential
stages reuse one via the framework's `slot.settled` handoff; a parallel
sweep does not. Set `JAATO_RUNNER_POOL_SIZE >= --concurrency` or arms
cold-spawn (~30s) instead of claiming a warm slot (~7s). The driver prints
the number it wants rather than silently under-performing.

## Determinism

Every arm's completion payload is hashed with the same canonicalisation
`jaato-cascade-based-prototype`'s `hash_payload` uses. The `det` column is
the share of an arm group sharing the modal hash — 100% means byte-identical
output across repeats. That is a flakiness measurement the pass rate cannot
give you, and it costs nothing extra to collect.

## Tests

```bash
python3 -m unittest discover -s tests -t .      # no daemon, no SDK needed
```

69 tests. The runner is covered end-to-end against a stubbed SDK
(`tests/test_runner_integration.py`) — PASS, FAIL and BLOCKED arms, usage
accumulation, the `config_root`/workspace split, and `.env` profile-set
propagation.

## Constraint: SDK only

This package imports `jaato_sdk` and never `shared.*`. If something here
cannot be built on the SDK, that is an SDK gap to be fixed in the SDK.

That is not hypothetical: building this found the tool-call ledger
unreachable over the SDK, and the fix landed in the framework (#639, #640)
rather than as a private workaround here — after which the workaround was
deleted. The rule paid for itself once already.

`certify/` enforces the same discipline with a facade guard; adopting that
guard here is the obvious next hardening step.
