# jaato-eval — status report

**2026-08-28** · branch `claude/agent-optimization-environments-bhjp7w` · 16 commits
· 144 tests · exercised live against jaato `main` through `8bc3cee5`

## Verdict

**Shippable as a two-grader harness. Not shippable as a three-grader one.**

`script` and `processor` graders are usable for measurement today. The `judge`
grader works, is fully tested, and is **not a trustworthy instrument yet** —
see [Known limits](#known-limits). That distinction is now in the README so a
user meets it before a flaky benchmark does.

## What it is

A benchmark spine over jaato's session and profile primitives: a task manifest
becomes N runnable arms across a matrix of model configurations, each in a
hermetic workspace, graded by pluggable adapters, written to JSONL as it goes.

The design premise was **extract, don't build** — 14 of the 18 concepts such a
harness needs already existed in jaato, and three org repos had each privately
rebuilt a piece of it. One constraint was imposed as discipline: import
`jaato_sdk` only, never `shared.*`. Where the SDK could not do something, that
was treated as an SDK defect rather than something to work around.

That constraint paid for itself repeatedly — see [Upstream](#what-went-upstream).

## Verified

Everything below was run against a live daemon, not only unit-tested.

| capability | evidence |
|---|---|
| fixture → session → grade → record | both shipped tasks pass end to end |
| PASS is real | `answer.txt` on disk containing `READY` |
| FAIL is reachable | grader mutated to expect `STEADY` → FAIL, exit 1, **not** BLOCKED |
| BLOCKED is not a dumping ground | pass rate renders `—`, never `0%`, when every arm blocked |
| sweep matrix | 2 profile sets × 2 repeats = 4 arms, report pivots per set |
| per-arm budget ceiling | `budget_control` in the profile; `tokens: 2000` blocks the arm |
| task budget pool | `budget:` in the manifest; a 30 000-token pool funded one arm, then refused two at spawn in ~0.2 s each |
| the two gates are independent | 2 arms passed on a 5 000-token pool that could not fund one, because each ran on its profile's own 60 000 ceiling |
| tool-call ledger is faithful | processor grader cross-examined a claim and reported *"the ledger's write calls touched ['answer.txt']"* |
| arm wall-clock ceiling | `--arm-timeout`; removing the `wait_for` fails 2 tests and takes the suite from 1.6 s to 60 s |
| cost reporting | `cost_usd = 0.008238` on the event, matching the budget tracker's `usd: 0.008238` for the same arm |
| exit codes | 0 all-pass · 1 a FAIL · 2 a BLOCKED or nothing ran |

Every load-bearing guard was **watched to fail** under sabotage, not merely
observed to pass.

## A real sweep

```
| task                | profile set                  | pass rate | pass | fail | blocked | cost USD | tokens |
|---------------------|------------------------------|-----------|------|------|---------|----------|--------|
| example/echo-a-file | openrouter_gemini25flashlite |       50% |    1 |    1 |       0 |   0.0034 | 105930 |
| example/echo-a-file | openrouter_gpt5mini          |       50% |    1 |    1 |       0 |   0.0094 | 157542 |
```

Both models at 50% is **not** a difficulty result — it is the judge flipping.
Each model scored 1.0 on one repeat and 0.0 on the other. Which is the point of
the next section.

## Known limits

### The judge skips its own verification step, ~1 run in 4

Three of four captured failures read *"I did not open answer.txt"* — the tool
was never called. Only one was an actual tool error. The judge admits it
because the rubric requires that; the honesty is what makes it visible.

This is a property of LLM judges, not a framework defect, and nothing in this
package fixes it. What the package does is stop it being **attributed to the
thing under test**: a rubric reports any reason it could not assess in
`errors[]`, and a non-empty `errors[]` is BLOCKED, never FAIL. Previously the
admission sat in `reasoning` while `errors[]` was empty, and arms were recorded
as failures with correct artefacts on disk.

*Not verified:* six probes after that change came back clean, which is **not**
evidence it works. At a 1-in-4 base rate, six clean runs occur ~18% of the time
by luck, and the change was meant to make a failure BLOCK rather than prevent
one — so the new path never ran.

### Report per arm, never averaged

On a two-arm sample this engine appeared to show one model outscoring another.
At four arms both were flipping. An average over two repeats would have read as
a real difference in model quality. The JSONL keeps every arm; the pivot never
averages a BLOCKED row into a pass rate.

### A pooled arm has no tool-call ledger

A cascade-pooled session is unloaded on its terminal event, so a history
request finds nothing — a typed error since jaato #645, not a hang. Ledger-
grading tasks therefore use the per-arm ceiling rather than a pool.
`example-echo` carries the pool; `ledger-probe` carries the ceiling.

## What went upstream

Building against a live daemon surfaced defects that a stubbed harness could
not. Each was fixed **in the framework** rather than worked around here, and
the local copy deleted:

| upstream | what it was |
|---|---|
| #639 / #640 | the call side carried no identifier, so no consumer could rebuild the tool-call ledger |
| #642 | an exhausted pool handed back a working-looking session id, then a 30 s generic timeout |
| #643 | a terminal event was emitted *before* the final turn event it reported |
| #644 | provider-reported cost never reached `TurnCompletedEvent` — five missing links |
| #645 | a history request met with silence rather than an answer or a refusal |
| #646 | a live conformance suite, run against a real daemon in CI |
| #647 | a reused pool slot carried the previous session's environment — including decoded `pass://` secrets |
| #648–#651 | the completeness rule as a function; a `sweep` archetype; templates tracking their SDK contract; a doctor check |
| #652 | the secret-resolver registry was published empty before it was filled |
| #654 | a stage asked twice to signal completion, and never did, now says so |

Four copies of framework knowledge were retired from this package as the
framework absorbed them: the ledger pairing, the completeness rule, a cascade
observer workaround, and three manifest knobs that duplicated profile fields.

## Provenance

Every result records the resolved `jaato_sdk` path and version, read from the
live process. The branch does not determine what a run exercised: an editable
install resolves through a MetaPathFinder to wherever it was installed *from*,
so this worktree ships its own `jaato-sdk/` and never runs it.

## What is left

- The judge's skip rate is uncharacterised beyond "~1 in 4 on this model".
- The `errors[]` routing fix is unobserved in production (no failure has
  occurred since it landed).
- The branch is 14 commits behind `origin/main`; runs were against that newer
  code, which the provenance stamp records.
