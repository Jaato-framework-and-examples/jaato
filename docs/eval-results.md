# The `jaato-eval` results file, as a contract

A sweep writes one JSON object per line to its results file — one **arm**, one
record, appended as it lands. This page declares what a reader outside
`jaato-eval` may rely on, and is the only agreement between the two sides:
`jaato_eval` imports `jaato_sdk` and nothing else from this tree, so a consumer
that imported the engine would run that rule backwards. The producer's half is
`jaato-eval/jaato_eval/results_format.py`; the first consumer is
`jaato-server/jaato_server/shared/scaffold/eval_results.py`, which renders an arm's numbers
into the accuracy section of an Annex IV dossier (`jaato-scaffold new dossier
--eval-results <file>`).

## The version field

Every record carries `results_version`. Today's value is **`"1"`**.

A reader that does not recognise the value **refuses the whole file by name** —
naming the version it found and the versions it can read — rather than
rendering the fields it happens to understand. A dossier filled from a format
its reader guessed at is worse than an empty section, because it looks
complete; and the reader cannot know whether the field it read still means what
it meant. A record with **no** `results_version` is an unknown version, not a
version 1 record: the field has been written since the format was declared, so
its absence says the file predates the contract.

The version is bumped when a reader that knows only the previous value could be
**misled** by a file written under the new one — a field whose meaning changed,
a denominator that moved, a state that gained a member. It is *not* bumped for
a field added beside the existing ones, which a reader ignores.

## What version 1 guarantees

| Field | Type | Meaning |
|---|---|---|
| `results_version` | `str` | `"1"` |
| `caveats` | `list[str]` | The limits of the instruments that graded **this arm**, in the harness's own words. See below. |
| `arm_id` | `str` | `task@set#repeat` — stable identity |
| `task_id` | `str` | The manifest's id |
| `profile_set` | `str \| null` | The model/provider axis; `null` means the manifest's own binding |
| `repeat` | `int` | 0-based repeat index |
| `state` | `"PASS" \| "FAIL" \| "BLOCKED"` | The arm's roll-up |
| `verdicts` | `list[object]` | One per grader: `grader_id` (`kind:identifier`), `claim`, `state`, `detail`, `evidence`, `blocked_reason` |
| `blocked_reason` | `str \| null` | Set when the arm itself never ran |
| `usage` | `object` | Token and cost totals. `cost_usd` may be `null` — that is "nobody knew", never "free" |
| `provenance` | `object` | `jaato_sdk_path` and `jaato_sdk_version` of the process that ran the arm |

Three reader rules, each attached to a way the numbers mislead:

- **`BLOCKED` is excluded from a pass-rate denominator.** Nothing was
  exercised, so it is neither a pass nor a failure of the thing under test. A
  group in which every arm blocked has **no** pass rate — not `0%`.
- **A null is "we did not find out", never a zero.** `cost_usd: null` is not
  free; an absent `completion_nudges` is not "no nudges fired".
- **An arm is the unit; an average over arms is the consumer's invention.** The
  file keeps every repeat because repeats disagreeing *is* the measurement. Two
  repeats averaged into one number read as a difference in model quality when
  they were a coin flip.

## Caveats travel with the number

`caveats` carries what the graders that produced this arm's verdicts *cannot*
tell you, written by the harness that measured the limit. A consumer renders
each string **verbatim** beside the numbers it qualifies. It never summarises
one, and never writes its own.

The reason is the one this repository keeps meeting: a second copy of a fact
rots, and the copy that rots is the one nothing executes. A consumer's
hand-written warning about LLM judges outlives the limit it describes — it goes
on being quoted after the limit is fixed, or silently misses a limit added
later. The harness measured it, so the harness states it, once, and it rides
beside the number rather than in a document beside the document.

An empty `caveats` list asserts nothing. It means this harness has not measured
a limit worth stating for these graders — never that the instruments are
calibrated.

## Fields outside the contract

A record carries more than the table above: the per-arm provenance block
(`session_id`, `model`, `upstream_provider`, `budget_ceiling`,
`pool_on_arrival`, …) documented on `jaato_eval.arm.ArmResult`. Those are
written unconditionally, nulls included, and are readable — they are simply not
promised by this version, so a consumer that depends on one is depending on the
engine rather than on the contract.
