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
pip install -e jaato-eval/
```

`jaato-sdk` is a hard dependency now. It was an optional extra while the
tool-call ledger and the completeness rule lived here; both moved into the
SDK (jaato #640, #648) and this package deleted its copies rather than
keeping them in sync, so the grader layer imports the library directly.

Importing the SDK still does not require a **daemon**: the manifest parser,
fixture materialiser, ledger, script and processor graders, result store and
report all work — and are tested — with nothing running. Only `runner.py`
and the `judge` grader need a live session.

## Use

```bash
# run one task directory across two profile sets
python -m jaato_eval run tasks/ --profile-set openrouter_haiku,vllm_qwen3_14b \
    --out results.jsonl --concurrency 4

# re-pivot an existing results file
python -m jaato_eval report results.jsonl

# ... and write the per-arm report beside it (self-contained HTML; add
# --pdf for the same document rendered, via `pip install 'jaato-eval[report]'`)
python -m jaato_eval report results.jsonl --html report.html

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

environment:                  # ONLY what a profile cannot express
  fixture: fixture            # copied fresh per arm; the agent mutates the copy
  config_root: .jaato         # read-only: profiles, agents, schemas
                              # the repo's root .gitignore excludes `.jaato`,
                              # negated for jaato-eval/tasks/*/.jaato so a
                              # task's config root travels with the repository

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

### A script grader can see the task's inputs

`processor` and `judge` graders get a `GraderContext`, which carries
`agent_params`. A shell command cannot read a Python object, so the
`script` grader hands the same inputs over as environment variables:

| variable | is |
|---|---|
| `JAATO_EVAL` | `1` — this is a graded run |
| `JAATO_EVAL_PARAM_<KEY>` | one `agent_params` entry, key upper-cased, non-identifier characters replaced by `_` |
| `JAATO_EVAL_PARAMS` | the whole mapping as JSON, under the author's own key spellings |

So a check that depends on an input says so in the manifest:

```yaml
input:
  agent_params: {repo: Jaato-framework-and-examples/jaato, issue_id: "716"}
graders:
  - kind: script
    run: bash acceptance.sh compliant "$JAATO_EVAL_PARAM_ISSUE_ID"
```

Without this, an input-dependent check has to hardcode the input — and
then changing `issue_id` grades every arm against the *previous* issue's
acceptance criteria. No error, no warning: arms that did the work
correctly are reported `FAIL` against a claim the task never made. The
export makes the dependency real rather than something two files have to
remember about each other.

**Values.** Strings pass through verbatim. Everything else is JSON, so a
dict or list survives the trip (`{"a": 1}`), a bool is the `true` the
manifest author wrote rather than Python's `True`, and an explicit null
is `null` rather than the empty string that would make it
indistinguishable from an absent key.

**Absence.** A parameter the task does not declare leaves its variable
*unset*, not empty — so `set -u` in the grader is a working guard, and a
grader that depends on a parameter should use it rather than passing
vacuously on an empty expansion:

```bash
set -u   # $JAATO_EVAL_PARAM_ISSUE_ID being absent now fails the grader
```

For a finer distinction — declared-but-empty versus never declared —
parse `JAATO_EVAL_PARAMS`; it is the only place the two look different.

**Collisions** are BLOCKED, not arbitrated. `issue-id` and `issue_id`
both want `$JAATO_EVAL_PARAM_ISSUE_ID`; picking a winner would grade
against one input while the arm ran with the other, which is the very
disagreement this export exists to remove. Rename one key.

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

### The one error terminal that is still evidence

Both of those properties cut the other way when `BLOCKED` is applied to an
arm that *did* produce something, and one terminal does exactly that. When
the framework's completion-nudge budget runs out (`NudgeExhausted`), the
agent has run, worked and committed; the only thing missing is its
`signal_completion` call. Recording it as "nothing to grade" loses a
passing tree outright, and — because blocked arms leave the denominator —
lets a genuinely failing one *raise* the model's pass rate. The same two
arms and the same two trees, through `report.build_cells`:

```
recorded BLOCKED  ->  pass rate 100%   pass 1  fail 0  blocked 1
graded            ->  pass rate  50%   pass 1  fail 1  blocked 0
```

That is a measurement bias, not a missing row (jaato #773), and it is a
test rather than a claim — `tests/test_unsigned_arm_is_graded.py`.

So such an arm is graded, and grading is **per-grader**, not per-arm:

| grader | on an unsigned arm | why |
|---|---|---|
| `script` | runs, returns a verdict | it reads the workspace, and the workspace is real |
| `processor` | BLOCKED | its whole contract is `validate(payload, …)`, and there is no payload |
| `judge` | BLOCKED | it is handed the payload first and a listing second; scoring one arm on half the input it scores its siblings on is worse than a gap |

A judge- or processor-graded task therefore still rolls up BLOCKED when
its script graders all pass: the payload-reader cannot establish its
claim, and inventing a verdict for it would be the same error in the
other direction. So the "passing tree reported as unmeasured" half is
fully recovered only for workspace-only manifests; the "failing tree
leaves the denominator" half is recovered for every manifest, since a
FAIL outranks a BLOCKED in the roll-up.

The arm carries `error` (the terminal) with `blocked_reason` unset — the
record that it produced evidence *and* ended badly. Every other error
terminal keeps the conservative reading: a daemon that died mid-turn
leaves a tree nobody can vouch for. The rule lives in one place,
`jaato_eval/sign_off.py`, because the runner and the graders must not each
carry a copy of it.

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

## The per-arm report

The pivot answers *which configuration won*. It is the wrong artefact for
the other question a sweep raises constantly: **what happened to arm 3,
and can I go look at it upstream?**

```bash
python -m jaato_eval run tasks/ --out results.jsonl --html report.html
python -m jaato_eval report results.jsonl --html report.html --pdf report.pdf
```

`--html` writes a **self-contained** document — no dependency, no script,
no webfont, no network — carrying the pivot *and* one table per task, one
row per arm, shaped like a provider console's session list. It ships print
CSS, so any browser prints it to PDF. `--pdf` renders the same HTML through
the optional `report` extra (`pip install 'jaato-eval[report]'`) and fails
loudly with that install line rather than falling back silently to HTML —
a sweep run unattended asked for a PDF and must not quietly produce
something else.

### The session id is the point

| column | source |
|---|---|
| model, provider | the daemon's `SessionInfoEvent` — what it actually **bound**, not the profile-set name |
| **session id** | the runner already knew it and used to discard it |
| upstream provider, native finish reason | provider-reported; `—` until jaato #766 carries them off the wire |
| budget | which gate applied, and what the pool had left **on arrival** |
| nudges | `n/2`, counted from the session's own log |
| verdicts | one column per grader, not a blob |

`profile_set` (`openrouter_gemini25flash`) is a naming convention, not
data. The **session id** is the join key: OpenRouter's console groups its
Sessions view by exactly that string, so persisting it turns every row
into a link to the provider's own record of the arm — request count,
routed upstream, per-request cost, generation ids. Without it the two
views cannot be joined at all.

### Why budget is a column

Three arms of one sweep drew on a single `$6.00` pool and spent
`$3.81 + $0.17 + $2.03 = $6.0140`. The last was killed mid-work with
`SessionTerminatedEvent(reason=budget_exhausted)` and recorded BLOCKED.
From the results file that arm looks like a model failure. It was not — it
was billed for an earlier arm's appetite, and noticing took reading three
rows and adding them up.

```
$2.0300 / pool $6.0000 (66% consumed on arrival)
```

The pool reading is taken **per arm, immediately before it starts**
(`cascade.budget.get`): one taken up front would print the same number on
every row and answer nothing.

The two budget gates are shown as themselves. A session declaring its own
`budget_control` is on its own books and does not draw on the task pool —
so the column says `own $0.2500` or `pool $6.0000`, because a ceiling
shown without naming its pot reads as a pool that failed to bind.

### `—` is not zero

Every unestablished value renders as an em dash, and the document says in
prose what each one does *not* mean. `cost —` is not free (neither the
provider nor `.jaato/pricing.json` reported one). `nudges —` is not "none
fired" — it means the count could not be read, which is what a daemon
logging at INFO leaves behind, and reporting it as `0` would be a fact the
engine made up. The same discipline as `pass rate —` versus `0%`.

### Where each field comes from

Only one of them is read off a file. The model and provider come from the
daemon's own announcement, because a resolver that re-derived them would
be a second implementation of profile binding and would describe an arm
that never ran the moment it disagreed. The **budget ceiling** has no such
witness — the daemon enforces `budget_control` without announcing it — so
`jaato_eval/profile.py` reads that one field from the profile the arm
bound, following the framework's own two rules: set directory first, and
limits merge **min-wins** (a child may only ever *tighten* a ceiling).

Nudges come from the session's own log inside the arm's workspace, the
same move `_tracker_usage` already makes for the budget snapshot — the
count is announced with `logger.debug` and rides no event.

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

230 tests. The runner is covered end-to-end against a stubbed SDK
(`tests/test_runner_integration.py`) — PASS, FAIL and BLOCKED arms, usage
accumulation, the `config_root`/workspace split, and `.env` profile-set
propagation.

### The judge is the expensive grader, and its guards are the point

It opens a SECOND session per arm, so `gate_on` exists to skip it (BLOCKED,
with the reason recorded) when the cheap graders already failed. Two things
learned by running it for the first time, both now fixed:

- **It must score on the daemon its arm ran on.** It read `socket_path` from
  the manifest only, so a sweep pointed at a non-default daemon would have put
  arms on one and judges on another — silently, since the client default
  resolves to a real socket. The socket is a property of the **run**, so it
  travels on `GraderContext`, not in `task.yaml`.
- **A rubric profile is OFFERED `signal_completion`, not compelled to call
  it.** With a correct schema and a prompt that only said "score this", the
  judge answered in prose and every arm came back BLOCKED claiming the profile
  declared no schema — pointing at the one thing that was fine. The adapter
  now asks for the call explicitly, because every judge needs it and leaving
  it to each rubric author reproduces the failure with a misleading message.

And a rubric lesson that belongs to the profile, not the code: the first
working judge scored **0.0 while `answer.txt` held `READY`**, because it
reasoned from the agent's completion payload instead of opening the file. The
schema now requires reading the artefact and quoting the bytes. A rubric that
scores the claim measures the claim.

### The harness reads; the model judges

The judge used to be told to open the artefact with `filesystem_query`, and
roughly **1 run in 4 it simply did not** — saying so plainly: *"I did not open
answer.txt."* Not a tool failure; the call was never made. Four hypotheses
about the minority case that WAS a tool error were all refuted.

The fix was to stop asking. Reading a file is a **fact**, and a fact routed
through a model's discretion is unreliable by construction. So a MANDATORY
prefetch reads it and injects it before the first turn:

```
.jaato/agents/rubric.md              {{!py:scripts/prefetch_artefact.py answer.txt}}
.jaato/scripts/prefetch_artefact.py  def render(context, args) -> str
```

`{{!py:}}` without `?` raises and aborts session-prep, so a judge that cannot
get the artefact never starts. And `filesystem_query` is **removed** from the
rubric profile rather than kept as a fallback: with no tool there is nothing to
skip. Earlier attempts instructed the model more firmly, which competes with
the judgement that was failing; this removes the capability.

The `judge` grader takes `agent:` for this reason — the `{{!py:}}` placeholder
lives in the persona, so a judge given only `profile:` never expands it and
silently reverts to a bare file listing.

Measured after: 8/8 runs scored correctly, each quoting `"READY\n"` **including
the trailing newline** — a byte obtainable only from the injected content, so
the mechanism is evidenced rather than inferred from a pass count. And it still
discriminates: `NOT_READY` → 0.0, absent → 0.0, `ready` (wrong case) → 0.5, the
rubric's "present but wrong" band, unprompted.

**Report per arm, never averaged.** On a two-arm sample this engine appeared
to show one model outscoring another; at four arms both models were scoring
1.0 on one repeat and 0.0 on the other. An average over two repeats would
have read as a real difference in model quality.

### Validated against a live daemon

Both shipped tasks have been run against a real daemon (openrouter /
`openai/gpt-5-mini`), and each verdict state was reached deliberately
rather than merely observed:

| Checked | How |
|---|---|
| PASS is real | `answer.txt` on disk holding `READY` |
| FAIL is reachable | grader mutated to expect `STEADY`; arm went FAIL, exit 1 — **not** BLOCKED |
| the ledger is faithful | `ledger-probe`'s processor grader ran (the gate lets it) and its sabotaged run reported *"the ledger's write calls touched ['answer.txt']"* — a real `writeNewFile`, paired by identifier |
| BLOCKED is not a dumping ground | pass rate renders `—`, not `0%`, when every arm blocked |

That run is also what found the `tool_use` truncation defect below.

## Every result records which code it exercised

The branch does not determine what a sweep ran. An editable install resolves
`jaato_sdk` through a MetaPathFinder to wherever it was installed *from* — for
a git worktree that is the original checkout, so a branch can ship its own
`jaato-sdk/` and never run a line of it. The recorded version is no better: an
editable install stamps the version at install time.

So each result carries a `provenance` block read from the live process:

```json
"provenance": {
  "jaato_sdk_path": ".../jaato/jaato-sdk/jaato_sdk/__init__.py",
  "jaato_sdk_version": "0.15.0"
}
```

That example is real, and it is the point: the path is the main checkout while
the sweep ran from a worktree, and the version reads 0.15.0 while that
checkout's `pyproject.toml` says 0.16.0. A sweep's numbers are evidence about
the code that ran, and nothing in the repository state establishes which code
that was.

## Reproducing daemon behaviour

`tools/repro_cascade_event_routing.py` runs five scenarios against a live
daemon and prints the events each one actually received — with and without a
cascade id, with and without an observer registration, and through both the
facade and `send_message`. It exists because a description of an event-routing
bug is not checkable and a script is.

Two things in it are load-bearing and commented as such: it waits on events
rather than sleeping (a sleep short enough to keep it quick is also short
enough to make a slow turn look like a missing event), and its prompt drives
the profile to its **declared terminus**. A prompt the model can answer in
prose ends with `finish_reason="stop"` and never calls `signal_completion` —
and the routing difference under test only appears on the terminus path, so a
chatty prompt makes every scenario pass while testing nothing.

## The manifest holds only what a profile cannot express

`apparmor`, `runtime_limits` and per-arm `env` are all `SubagentProfile`
fields. A task declares them in its own profile, beside the model and provider
they belong with — not in `task.yaml`.

`runtime_limits` was the instructive one. It sat in the manifest with a
docstring saying it was "forwarded to the profile layer", and the shipped
example declared a `tool_timeout_seconds: 60`. It reached nothing:
`runner_spawn.py` reads `getattr(profile, "runtime_limits")` and there is no
session-kwarg vehicle, so no amount of plumbing could have delivered it. A
field that cannot work is worse than a missing one, because the docstring
recruits the reader into believing it does.

`apparmor` did work, through `ClientConfigRequest.apparmor`. It was removed
anyway: two writers for one setting requires a precedence rule, the framework
defines none, so the rule would have been this engine's invention — applied to
confinement.

What stays is what has no profile representation: `fixture`, `config_root`,
`prompt`, `agent` and `agent_params` (neither is a profile field), the profile
selection itself, `graders`, `repeats`, and `budget` — the cascade pool, which
is a runtime aggregate over one live cid rather than a property of a reusable
template.

## Two budget gates, and they are independent

jaato has two budget mechanisms. A sweep wants both, for different reasons,
and they do not compose on one session.

| | per-arm ceiling | task pool |
|---|---|---|
| declared in | the arm's profile, `budget_control:` | the manifest's `budget:` |
| scope | one session, its own books | every arm of the task (repeats × sets) |
| clamped by a pool? | never | n/a |
| depletes a pool? | never | yes |
| engine code | none — the daemon enforces it | `jaato_eval/pool.py` |

A session carrying its own `budget_control` **does not draw on a pool**, so
declaring both leaves the pool untouched. That is the framework's rule, not
this engine's. Verified live: two arms passed on a 5000-token pool that could
not fund one of them, because each ran on its profile's own 60000 ceiling.

An arm stopped by either gate is BLOCKED, never FAIL — it produced no signal
about the thing under test. A pool with no headroom refuses the spawn, and the
verdict names the pool.

### A pooled arm has no tool-call ledger

A session stamped with a cascade id is unloaded by the daemon's default
cascade policy on its own terminal event, so a history request from the
connection that created it finds nothing to serve. Since jaato #645 that is
answered with a typed `ERROR` rather than met with silence — the request no
longer hangs — but a pooled arm still has **no** ledger.

Absent is not empty, and the engine keeps them apart: `build_ledger_result`
returns unfaithful with a reason, and ledger-reading graders BLOCK. Collapsing
the two produced a real fabricated verdict — *"the agent reports writing
answer.txt but the ledger holds no call to writeNewFile"*, about an agent that
had written the file in a call the engine simply never saw.

So a task that grades against the ledger uses the per-arm ceiling, not a pool.
`tasks/example-echo` carries the pool; `tasks/ledger-probe` carries the ceiling.

Turn events, by contrast, now arrive for a pooled arm without any special
handling. They did not before jaato #643: `SessionTerminatedEvent` was emitted
before the final `TurnCompletedEvent`, so a policy detaching on the terminal
event stranded the turn event, and a pooled arm reported turns=0 with its file
written on disk. The engine briefly worked around that by registering as a
cascade observer; that call is gone, because it encoded an explanation which is
no longer true.

## Two budget gates, and they are independent

jaato has two budget mechanisms. A sweep wants both, for different reasons,
and they do not compose on one session.

| | per-arm ceiling | task pool |
|---|---|---|
| declared in | the arm's profile, `budget_control:` | the manifest's `budget:` |
| scope | one session, its own books | every arm of the task (repeats × sets) |
| clamped by a pool? | never | n/a |
| depletes a pool? | never | yes |
| engine code | none — the daemon enforces it | `jaato_eval/pool.py` |

A session carrying its own `budget_control` **does not draw on a pool**, so
declaring both leaves the pool untouched. That is the framework's rule, not
this engine's. Verified live: two arms passed on a 5000-token pool that could
not fund one of them, because each ran on its profile's own 60000 ceiling.

An arm stopped by either gate is BLOCKED, never FAIL — it produced no signal
about the thing under test. A pool with no headroom refuses the spawn, and the
verdict names the pool.

### A pooled arm has no tool-call ledger

A session stamped with a cascade id has its events fanned out to the cid's
registered **cascade-clients** rather than to the connection that created it.
The engine registers each pooled arm as an observer (`cascade_register`), which
restores the turn stream — but a history request from that connection still
goes unanswered, so a pooled arm's ledger is **absent**.

Absent is not empty, and the engine keeps them apart: `build_ledger_result`
returns unfaithful with a reason, and ledger-reading graders BLOCK. Collapsing
the two produced a real fabricated verdict — *"the agent reports writing
answer.txt but the ledger holds no call to writeNewFile"*, about an agent that
had written the file in a call the engine simply never saw.

So a task that grades against the ledger uses the per-arm ceiling, not a pool.
`tasks/example-echo` carries the pool; `tasks/ledger-probe` carries the ceiling.

### `finish_reason` is not a completeness signal

A profile with a `completion_payload_schema` ends by calling
`signal_completion`, which terminates the session on the spot — so the
terminal turn of a *complete* arm reports `finish_reason="tool_use"`,
the same value a genuinely truncated arm carries. Graders must read
`GraderContext.truncation_reason`, which settles it on the declared
terminus (a payload, or a `"stop"` turn) instead. Reading the raw field
blocks every schema-driven arm as truncated.

## Constraint: SDK only

This package imports `jaato_sdk` and never `shared.*`. If something here
cannot be built on the SDK, that is an SDK gap to be fixed in the SDK.

That is not hypothetical: building this found the tool-call ledger
unreachable over the SDK, and the fix landed in the framework (#639, #640)
rather than as a private workaround here — after which the workaround was
deleted. The rule paid for itself once already.

`certify/` enforces the same discipline with a facade guard; adopting that
guard here is the obvious next hardening step.
