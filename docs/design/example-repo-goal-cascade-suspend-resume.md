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
                                  resume_reason, watch_handle, attempt}
                               sleep until the nearest due row
                               attach_session + send_message(continuation)
                               attempt += 1
  budget exhausted / max_resumes → report unfinished with the last progress_note
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
    goal-actor.yaml           # plugins, model, budget_control, completion schema ref
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

## 8. Open questions before implementation

1. **Resume granularity** — does the agent resume in its *same* session (history
   intact, context grows across every cycle and needs GC) or does each cycle
   spawn a fresh session seeded with the previous `progress_note` and
   `watch_handle`? The second is cheaper and more cascade-idiomatic; the first is
   simpler and demonstrates continuity. Worth showing the first and noting the
   second.
2. **Where `max_resumes` belongs** — the driver, or the cascade budget's `turns`
   dimension. Preference: the budget, so the ceiling is declared once and the
   degradation ladder can respond before the hard stop.
3. **Does the repo pin a provider?** A cheap, fast, widely-available model keeps
   the demo runnable; the profile should make substitution obvious.
