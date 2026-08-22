# Design: a public example repo for goal cascades with suspend/resume

**Status:** design only — no repo created, no code written.
**Proposed repo:** `Jaato-framework-and-examples/goal-cascades-with-suspend-and-resume`
(topic-descriptive, following the existing public `budget-control-and-model-degradation`;
alternatives: `jaato-goal-cascade-example`, `suspend-resume-goal-cascade`).
**Scope decision:** public-SDK driver only — no jaato-premium dependency, no
framework changes required to run it.

---

## 1. What it exemplifies

A cascade that pursues a **goal** across an unbounded number of turns, where the
agent explicitly *suspends* itself whenever it is waiting on something outside
its control, and is resumed later to continue — until it reports the goal
accomplished or a budget ceiling stops it.

The teachable idea, in one line: **an agent that must wait should end its turn
and say when to come back, not block.** Everything else in the repo exists to
show how that is expressed safely — as a validated completion payload rather
than an ungoverned timer the model sets for itself.

Three things a reader should be able to lift directly:

1. A **dual completion schema** — `finished` vs `suspended` — as the agent's only
   exit, so pausing is a capability the operator grants in the schema rather
   than one the model discovers.
2. A **driver loop** that owns the clock, persists due-rows, and survives its own
   restart.
3. A **cascade budget ceiling** bounding the resume loop, so a goal that never
   completes still terminates.

## 2. Why it runs on shipped jaato today

The reason this is worth an example repo rather than a framework PR: every piece
already exists in the public SDK. Verified against this tree:

| Need | Shipped surface |
|---|---|
| Spawn a stage under a cascade identity | `IPCClient.create_session(profile=…)` stamped with a `cascade_driver_id` |
| Observe every session under that cid | `IPCClient.cascade_events(cid, event_types=…, role="observer")` — async iterator; sends `cascade.register` at start, `cascade.unregister` on close |
| Aggregate ceiling over the whole cascade | `IPCClient.cascade_budget_set(cid, limits={usd,tokens,seconds,tool_calls,turns}, degrade=[…])` — clamps each spawned session to `min(profile, cascade_remaining)` and **refuses a spawn with no headroom left** |
| Typed, validated agent exit | `completion_payload_schema` + `signal_completion`; validation failure returns the `validation_failed` self-correction shape and the turn loop continues |
| Delivery of the validated payload to the driver | `LifecycleTools._execute_signal_completion` → `hooks.on_agent_completed(payload=…)` → surfaces on the cascade event stream |
| Resume with history intact | the session persists after `signal_completion` (the turn loop terminates; teardown is the driver's call), so `attach_session` + `send_message` continues it |

**The `DEFERRED` wrinkle does not bite this design.** `wake_session` holds a turn
pending only when a session is revived cold with **no attached client** *and* a
cid is set — the predicate is `session.attached_clients`. A driver that owns the
timer is by definition running when the resume is due, so it attaches and drives
directly. `DEFERRED` is a blocker for the *unattended, framework-scheduled* path,
which this repo deliberately does not attempt (§7).

## 3. The pattern: a dual completion schema

```yaml
# .jaato/profiles/completion_schemas/goal_actor.yaml  (sketch)
type: object
required: [outcome, progress_note]
properties:
  outcome:
    enum: [finished, suspended]
  progress_note:      { type: string }   # what changed this turn
  # --- finished branch ---
  result:             { type: object }
  # --- suspended branch ---
  resume_at:          { type: string, format: date-time }
  resume_reason:      { type: string }   # what it is waiting for
  watch_handle:       { type: object }   # job id / path / URL to re-inspect
  warnings:           { type: array, items: { type: string } }
  errors:             { type: array, items: { type: string } }
```

A `completeness`-phase processor enforces the branch contract: a `suspended`
payload must carry `resume_at` and a `watch_handle`, so a half-set pause is
unrepresentable. Its `incomplete[]` entries surface to the model as neutral
"still needed" guidance with no retry penalty; `errors[]` reject.

**Deliberate constraint: the goal-actor profile declares no `finalization`
processors.** `phase` selects *when* a processor runs, not *conditional on which
branch validated* — so a finalization processor would write final artifacts on a
suspend too. The example sidesteps this rather than working around it: validation
lives in `completeness`, and **the driver** writes artifacts when it observes
`outcome: finished`. This keeps the repo inside shipped behaviour and isolates
the one genuine framework gap (§7).

## 4. The driver loop

```
register cascade budget (cid, limits, degrade)
open cascade_events(cid)
spawn stage-1 session from the goal-actor profile with the goal statement

loop:
  await agent.completed payload
    ├─ outcome == finished   → run finalization (write artifacts, report), exit 0
    └─ outcome == suspended  → persist due-row {session_id, resume_at,
                                  resume_reason, watch_handle, progress_note,
                                  attempt}
                               sleep until the nearest due row
                               attach_session + send_message(continuation)
                                 ^ carries watch_handle + progress_note VERBATIM,
                                   so resume never depends on history surviving GC
                               attempt += 1
  budget refuses / exhausted → report unfinished with the last progress_note
```

Three properties the example must actually demonstrate, not just describe:

- **Restart recovery.** Due-rows live in a small JSON file (atomic rename +
  fsync). On start the driver reloads them and resumes anything already due —
  killing and restarting the driver mid-wait must not lose the goal.
- **Idempotent resume.** A deterministic resume key per `(session_id, attempt)`
  so a double-fire cannot double-drive.
- **Termination is always reported.** Ending on a budget ceiling is a distinct,
  visible outcome from ending on `finished` — never a silent stop.

## 5. Proposed layout

```
README.md                     # the pattern, the caveat, how to run
.jaato/
  profiles/
    goal-actor.yaml           # provider/model first + swap block, plugins,
                              #   gc: budget (context grows across cycles),
                              #   completion schema ref
    completion_schemas/
      goal_actor.yaml
  policies/
    goal_actor_completeness.py    # branch-contract processor
  agents/
    goal-actor.md             # persona: pursue the goal; suspend rather than block
src/goal_cascade/
  driver.py                   # the loop in §4
  store.py                    # due-rows: atomic write, reload, claim
  __main__.py                 # CLI: --goal, --max-resumes, --budget-usd
fixtures/
  slow_job.py                 # the external thing being waited on (§6)
tests/
  test_store.py               # restart recovery, idempotent claim
  test_driver_transitions.py  # finished / suspended / budget-exhausted, faked events
docs/
  sequence.md                 # the mermaid walkthrough
```

## 6. The scenario

It must involve a genuine wait, need no credentials or network, and be
deterministic enough for CI. Proposal: `fixtures/slow_job.py` simulates a build
or evaluation — started by the agent, it writes `status: running` then
`status: passed|failed` to a file after a configurable delay (seconds in tests,
minutes in the demo).

The goal: *"get the job to pass, and write a short report of what it took."*
The agent starts the job, sees it running, suspends with a `resume_at` and the
status-file path as its `watch_handle`. The driver resumes it; it re-inspects,
and either fixes a seeded fault and restarts the job (another suspend) or
reports `finished`. Two or three suspend/resume cycles, fully self-contained.

## 7. Deliberate non-goals

Named in the README so readers do not mistake scope for limitation:

- **No daemon-side scheduler.** The clock lives in the driver. That is the point
  of the example — it needs no framework change — but it means **driver lifetime
  is the durability boundary**: if the driver is not running when a resume comes
  due, the resume happens when it next starts, not on time. State survives; the
  schedule does not advance unattended.
- **No unattended resume.** Would need the daemon-side due-row store, a sweeper,
  and the `DEFERRED` fix (drive immediately when the target profile declares no
  host tools). Tracked as framework work, not example work.
- **No conditional finalization.** Suppressing `finalization` processors on a
  suspend branch is not expressible today; §3 sidesteps it.
- **No premium dependency.** A reactor-routed variant would be a natural second
  example once the repo exists — routing on `outcome` with a `where` clause —
  but it cannot ship in a repo meant to run from public deps.

## 8. Resolved decisions

### 8.1 Resume in the same session

Each cycle continues the **same** `session_id`, so the agent keeps its reasoning
context across suspends. Two consequences follow, and the second is the one that
actually matters.

**Context grows monotonically, so GC is load-bearing.** The goal-actor profile
must declare a strategy — `gc_budget` is the right fit here, being the only one
that makes policy-aware removal decisions and supports continuous per-turn
collection rather than waiting for a threshold breach. A goal running for a dozen
cycles is exactly the workload it was built for.

**But correctness must not depend on GC keeping anything.** `GCPolicy`
(`LOCKED` / `PRESERVABLE` / `PARTIAL` / `EPHEMERAL` / `CONDITIONAL`) governs
*instruction sources*, not conversation messages — there is no per-message pin,
so a `progress_note` from cycle 3 can legitimately be summarised away by cycle 9.

The resolution is clean, and it is what makes "same session" safe: **the driver
already holds the payload** — it received the validated `progress_note` and
`watch_handle` through `on_agent_completed` — so it re-injects them verbatim in
the continuation message it sends on resume. That splits continuity in two:

| | carrier | guarantee |
|---|---|---|
| **State** — what I was waiting on, where to look, what I had achieved | the driver's continuation message | load-bearing, guaranteed, survives any GC |
| **Reasoning context** — how I got here, what I already ruled out | the session's own history | nice-to-have, degrades gracefully under GC |

So history GC can be as aggressive as it likes and the goal still advances. This
is worth calling out explicitly in the README: it is the non-obvious reason the
same-session choice does not become a slow context leak.

### 8.2 The ceiling is the cascade budget

`cascade_budget_set(cid, limits={...}, degrade=[...])` owns termination; the
driver does not carry its own `max_resumes`. Declared once, it clamps every
session under the cid and refuses a spawn with no headroom left.

**One precision the example must not blur: `turns` is a plain turn counter, not
a resume count.** A single resume cycle usually costs several assistant turns
(inspect → maybe act → signal), so `limits={"turns": 40}` is an outer bound on
work, not "40 resumes". The driver should therefore *report* its resume count as
observability while the budget *enforces* the ceiling — never present the two as
the same number.

This also gives the example something Prime Agent's autonomous mode cannot do at
all: a `degrade` ladder that rebinds model tiers as the goal consumes its budget,
so a long-running goal gets *cheaper* as it goes rather than simply dying at the
limit. A brownout ladder — cheaper executor at 60 %, cheaper planner at 80 %,
terminal action at 95 % — is a better demo of jaato's budget model than a bare
hard stop, and costs a few lines of profile YAML.

### 8.3 Provider is pinned but obviously swappable

Default to one cheap, fast, widely-available model needing a single environment
variable, so `pip install && export ONE_KEY && run` works. Structure the profile
so `provider:` and `model:` are the first two keys, with a commented swap block
immediately beneath listing two or three alternatives across different providers
(one hosted, one local via Ollama/LM Studio) — the point being that nothing else
in the repo changes when you swap them.

The README states the demo's approximate token cost per full run, since the
whole pattern is about work that spans many turns and a reader deserves to know
what they are starting.
